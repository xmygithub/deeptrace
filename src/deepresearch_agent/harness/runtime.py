"""Persistent Harness Runtime driving Context -> Plan -> Execute -> Report -> Verify."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from typing import Any, Callable

from deepresearch_agent.harness.budgets import BudgetExceeded, BudgetLimits, BudgetManager
from deepresearch_agent.harness.checkpoints import CheckpointManager
from deepresearch_agent.harness.contracts import ContractEvaluator, RunStatus, SourceMode, WorkflowMode
from deepresearch_agent.harness.evidence import EvidenceLedger
from deepresearch_agent.harness.event_bus import EventBus
from deepresearch_agent.harness.errors import RunCancelled, RunLeaseBusy, RunLeaseLost
from deepresearch_agent.harness.recovery import classify_exception, classify_verification_failures
from deepresearch_agent.harness.run_context import RunContext
from deepresearch_agent.harness.state_machine import StateMachine
from deepresearch_agent.harness.call_budget import CallBudget, current_call_budget, set_call_budget, reset_call_budget
from deepresearch_agent.harness.retrieval_runtime import RetrievalRuntime, set_retrieval_runtime, reset_retrieval_runtime
from deepresearch_agent.context.artifact_edit import ArtifactEditContextBuilder
from deepresearch_agent.config.settings import REPORT_RESERVED_TOKENS, VERIFICATION_RESERVED_TOKENS
from deepresearch_agent.models.prefix_cache import set_current_run
from deepresearch_agent.harness.report_safety import sanitize_report
from deepresearch_agent.persistence.artifact_store import ArtifactStore
from deepresearch_agent.persistence.repositories import (
    ArtifactRepository, CheckpointRepository, ContractRepository, EventRepository,
    EvidenceRepository, MessageRepository, PlanTaskToolRepository, RunRepository,
)

WorkflowFactory = Callable[[RunContext, EventBus], Any]


class HarnessRuntime:
    def __init__(
        self,
        *,
        run_repository: RunRepository,
        message_repository: MessageRepository,
        event_repository: EventRepository,
        checkpoint_repository: CheckpointRepository,
        evidence_repository: EvidenceRepository,
        contract_repository: ContractRepository,
        workflow_factory: WorkflowFactory,
        trajectory_repository: PlanTaskToolRepository | None = None,
        artifact_store: ArtifactStore | None = None,
        artifact_repository: ArtifactRepository | None = None,
        event_bus: EventBus | None = None,
        context_builder: Any | None = None,
        memory_extractor: Any | None = None,
        skill_distiller: Any | None = None,
        prefix_tracker: Any | None = None,
        delivery_service: Any | None = None,
        lease_seconds: int = 90,
    ):
        self.runs = run_repository
        self.messages = message_repository
        self.events = event_bus or EventBus(event_repository)
        self.checkpoints = CheckpointManager(checkpoint_repository)
        self.evidence_repository = evidence_repository
        self.evidence_ledger = EvidenceLedger(evidence_repository)
        self.contracts = ContractEvaluator(contract_repository)
        self.trajectory = trajectory_repository
        self.artifact_store = artifact_store
        self.artifact_repository = artifact_repository
        self.workflow_factory = workflow_factory
        self.context_builder = context_builder
        self.memory_extractor = memory_extractor
        self.skill_distiller = skill_distiller
        self.prefix_tracker = prefix_tracker
        self.delivery_service = delivery_service
        self.state_machine = StateMachine()
        self.lease_seconds = lease_seconds
        self._prefix_offsets: dict[str, dict[str, int]] = {}

    def _attach_prefix_usage(self, context: RunContext) -> None:
        """把当前 Run 的前缀缓存用量回填到 budget usage（幂等，可重复调用）。"""
        policy = current_call_budget()
        if policy is not None:
            tools, searches = policy.tool_usage()
            context.budget_usage.tool_calls = max(context.budget_usage.tool_calls, tools)
            context.budget_usage.tavily_calls = max(context.budget_usage.tavily_calls, searches)
        if self.prefix_tracker is None:
            return
        snapshot = self.prefix_tracker.run_snapshot(context.run_id)
        if not snapshot:
            return
        totals = snapshot.get("totals") or {}
        offsets = self._prefix_offsets.get(context.run_id, {})
        context.budget_usage.prefix_cache_requests = max(context.budget_usage.prefix_cache_requests, int(totals.get("requests", 0)) + offsets.get("requests", 0))
        context.budget_usage.prefix_cache_hit_tokens = max(context.budget_usage.prefix_cache_hit_tokens, int(totals.get("hit_tokens", 0)) + offsets.get("hit_tokens", 0))
        context.budget_usage.prefix_cache_miss_tokens = max(context.budget_usage.prefix_cache_miss_tokens, int(totals.get("miss_tokens", 0)) + offsets.get("miss_tokens", 0))
        reported_tokens = int(totals.get("input_tokens", 0)) + int(totals.get("output_tokens", 0))
        context.budget_usage.llm_tokens = max(context.budget_usage.llm_tokens, reported_tokens + offsets.get("tokens", 0))

    async def execute_run(self, run_id: str) -> RunContext:
        owner = f"harness-{uuid.uuid4().hex}"
        if not await self.runs.acquire_lease(run_id, owner, ttl_seconds=self.lease_seconds):
            raise RunLeaseBusy(f"Run {run_id} 已由其他 Runtime 执行")
        self._lease_owner = owner
        self._next_renewal = time.monotonic() + max(0.05, self.lease_seconds / 3)
        context: RunContext | None = None
        recovery_target: RunStatus | None = None
        set_current_run(run_id)
        call_budget_token = None
        retrieval_token = None
        driver = None
        try:
            context = await self._load_context(run_id)
            if self.prefix_tracker is not None and context.checkpoint_version:
                totals = (self.prefix_tracker.run_snapshot(run_id) or {}).get("totals", {})
                usage = context.budget_usage
                self._prefix_offsets[run_id] = {
                    "tokens": usage.llm_tokens - int(totals.get("input_tokens", 0)) - int(totals.get("output_tokens", 0)),
                    "requests": usage.prefix_cache_requests - int(totals.get("requests", 0)),
                    "hit_tokens": usage.prefix_cache_hit_tokens - int(totals.get("hit_tokens", 0)),
                    "miss_tokens": usage.prefix_cache_miss_tokens - int(totals.get("miss_tokens", 0)),
                }
            call_budget_token = set_call_budget(CallBudget(context.budget_limits, context.budget_usage))
            if context.status in {RunStatus.INTERRUPTED, RunStatus.PAUSED}:
                resume_target = context.resume_from_status or RunStatus.QUEUED
                context.resume_from_status = None
                await self._transition(context, resume_target, event_type="run.resumed")
                await self.runs.clear_pause_request(run_id)
                recovery_target = resume_target
            if context.status in {RunStatus.QUEUED, RunStatus.CONTEXT_BUILDING}:
                if context.status is RunStatus.QUEUED:
                    await self._transition(context, RunStatus.CONTEXT_BUILDING, event_type="run.started")
                if self.context_builder is not None:
                    context_budget = BudgetManager(context.budget_limits, context.budget_usage)
                    context = await self._run_with_heartbeat(context, context_budget, self.context_builder.build(context))
                    selected_skill = context.context_snapshot.get("selected_skill") or {}
                    await self.runs.update_model_snapshot(run_id, {
                        **context.model_snapshot,
                        "skill": None if not selected_skill else {
                            "name": selected_skill.get("name"), "version": selected_skill.get("version"),
                            "status": (selected_skill.get("selection") or {}).get("status", "active"),
                        },
                        "context_policy": {
                            "recent_turns": len(context.context_snapshot.get("recent_messages", [])),
                            "memory_snapshot_version": context.context_snapshot.get("memory_snapshot_version", 0),
                            "curated_memory_ids": [item.get("memory_id") for item in context.context_snapshot.get("curated_memory", [])],
                            "historical_recall_session_ids": context.context_snapshot.get("used_session_ids", []),
                            "stable_snapshot_id": context.context_snapshot.get("stable_snapshot_id"),
                            "input_tokens": context.context_snapshot.get("total_input_tokens", 0),
                        },
                    })
                else:
                    context.resolved_query = context.resolved_query or context.original_query
            async def save_research_step():
                self._attach_prefix_usage(context)
                if driver is not None:
                    context.workflow_state = driver.snapshot()
                # Unlike a stage-end checkpoint this must resume the SAME stage.
                context.context_snapshot["inflight_stage"] = context.status.value
                await self.checkpoints.save(context, context.status.value)

            retrieval_token = set_retrieval_runtime(RetrievalRuntime(self, context, save_research_step))
            driver = self.workflow_factory(context, self.events)
            if inspect.isawaitable(driver):
                driver = await driver
            if context.status is RunStatus.CONTEXT_BUILDING:
                context.workflow_state = driver.snapshot()
                await self.checkpoints.save(context, "context_building")
                await self.events.publish(run_id, "context.completed", stage="context_building", payload={
                    "query_resolved": True,
                    "used_message_ids": context.used_message_ids,
                    "curated_memory_count": len(context.context_snapshot.get("curated_memory", [])),
                    "memory_snapshot_version": context.context_snapshot.get("memory_snapshot_version", 0),
                    "historical_recall_count": len(context.context_snapshot.get("historical_recall", [])),
                    "historical_recall_searched": bool(context.context_snapshot.get("historical_recall_searched")),
                    "context_tokens": context.context_snapshot.get("total_input_tokens", 0),
                    "artifact_edit": bool(context.context_snapshot.get("artifact_edit")),
                })
                await self._transition(context, RunStatus.PLANNING)
            budget = BudgetManager(context.budget_limits, context.budget_usage)
            # A checkpoint and its relational side effects are normally written
            # in order, but a process can stop between those commits. Replaying
            # these idempotent persistence steps reconstructs missing rows from
            # the driver snapshot without repeating an external tool/LLM call.
            if recovery_target in {RunStatus.EXECUTING, RunStatus.REPORTING, RunStatus.VERIFYING}:
                await self._persist_plan(context, driver)
            if recovery_target in {RunStatus.REPORTING, RunStatus.VERIFYING}:
                await self._persist_execution(context, driver, budget)
                context.budget_usage = budget.usage
            if recovery_target is RunStatus.VERIFYING:
                await self._save_report_artifact(context)

            while not self.state_machine.is_terminal(context.status):
                if await self._pause_at_safe_boundary(context):
                    return context
                await self._assert_not_cancelled(context)
                budget.assert_available()

                if context.status in {RunStatus.PLANNING, RunStatus.REPLANNING}:
                    if self.delivery_service is not None and context.config_snapshot.get("delivery_checks"):
                        if not context.context_snapshot.get("delivery_requirements"):
                            context.context_snapshot["delivery_requirements"] = await self._run_with_heartbeat(
                                context, budget, self.delivery_service.extract(context.resolved_query or context.original_query))
                            context.context_snapshot["inflight_stage"] = context.status.value
                            await self.checkpoints.save(context, context.status.value)
                            await self.events.publish(run_id, "delivery.requirements.created", stage="planning",
                                payload={"requirements": context.context_snapshot["delivery_requirements"]})
                        driver.set_delivery_requirements(context.context_snapshot["delivery_requirements"])
                    failures = list(context.workflow_state.get("verification_failures", []))
                    await self._run_with_heartbeat(context, budget, driver.plan(failures or None))
                    context.plan_version += 1
                    context.workflow_state = driver.snapshot()
                    plan_extra: dict[str, Any] = {}
                    plan_record = driver.plan_record()
                    if plan_record is not None:
                        _, plan_tasks = plan_record
                        budget.observe_plan_tasks(len(plan_tasks))
                        context.budget_usage = budget.usage
                        plan_extra = {
                            "task_count": len(plan_tasks),
                            "tasks": [
                                {"task_id": str(t.get("task_id", "")), "task_type": str(t.get("task_type", "")), "description": str(t.get("description", ""))[:120]}
                                for t in plan_tasks
                            ],
                        }
                    await self._persist_plan(context, driver)
                    context.context_snapshot.pop("inflight_stage", None)
                    await self.checkpoints.save(context, context.status.value)
                    await self.events.publish(run_id, "plan.revised" if failures else "plan.created", stage=context.status.value, payload={"plan_version": context.plan_version, **plan_extra})
                    await self._transition(context, RunStatus.EXECUTING)

                elif context.status in {RunStatus.EXECUTING, RunStatus.RETRYING}:
                    await self._assert_not_cancelled(context)
                    try:
                        await self._run_with_heartbeat(context, budget, driver.execute())
                    except Exception as exc:
                        decision = classify_exception(exc)
                        if decision.action == "retry_same":
                            budget.consume_retry()
                            context.budget_usage = budget.usage
                            if context.status is not RunStatus.RETRYING:
                                await self._transition(context, RunStatus.RETRYING, event_type="run.retrying", error_code=decision.reason, error_message=str(exc))
                            await self._transition(context, RunStatus.EXECUTING, event_type="run.retry_resumed")
                            continue
                        raise
                    await self._assert_not_cancelled(context)
                    await self._persist_execution(context, driver, budget)
                    context.workflow_state = driver.snapshot()
                    context.context_snapshot.pop("inflight_stage", None)
                    context.budget_usage = budget.usage
                    persisted_evidence = await self.evidence_repository.list_for_run(run_id)
                    minimum_evidence = int(context.config_snapshot.get("min_evidence", 1))
                    if len(persisted_evidence) < minimum_evidence:
                        evidence_count = len(persisted_evidence)
                        coverage_reason = "no_source_evidence" if evidence_count == 0 else "insufficient_source_evidence"
                        failures = ["min_evidence", coverage_reason]
                        context.workflow_state["verification_failures"] = failures
                        await self.events.publish(
                            run_id,
                            "source.coverage_checked",
                            stage="executing",
                            payload={
                                "evidence_count": evidence_count,
                                "minimum_evidence": minimum_evidence,
                                "source_mode": context.source_mode.value,
                                "passed": False,
                            },
                        )
                        # One focused replan is useful for a poorly phrased first
                        # retrieval (and recovers known short-query cases). Repeating
                        # another full plan after two zero-evidence executions only
                        # burns time/tokens and cannot produce a supported report.
                        can_replan = not hasattr(driver, "can_replan") or driver.can_replan(failures)
                        if can_replan and budget.usage.replans == 0 and budget.limits.max_replans > 0:
                            budget.consume_replan()
                            context.budget_usage = budget.usage
                            await self._transition(
                                context,
                                RunStatus.REPLANNING,
                                event_type="plan.replanning",
                                payload={
                                    "recovery_action": "replan",
                                    "recovery_reason": coverage_reason,
                                    "attempt": 1,
                                    "max_attempts": 1,
                                    "failures": failures,
                                },
                            )
                            await self.checkpoints.save(context, "replanning")
                            continue

                        source_hint = (
                            "请导入与该主题相关的私域资料，或改用联网搜索。"
                            if context.source_mode is SourceMode.GRAPHRAG
                            else "请调整问题或检查联网检索配置。"
                        )
                        error_code = "NO_SOURCE_EVIDENCE" if evidence_count == 0 else "INSUFFICIENT_SOURCE_EVIDENCE"
                        evidence_summary = (
                            "未检索到可支持该问题的证据"
                            if evidence_count == 0
                            else f"仅检索到 {evidence_count} 条有效证据，低于本报告所需的 {minimum_evidence} 条"
                        )
                        await self._transition(
                            context,
                            RunStatus.FAILED,
                            event_type="run.failed",
                            error_code=error_code,
                            error_message=f"完成 {budget.usage.replans + 1} 轮研究后{evidence_summary}；{source_hint}",
                            payload={
                                "recovery_action": "fail",
                                "recovery_reason": coverage_reason,
                                "attempts_exhausted": True,
                                "failures": failures,
                            },
                        )
                        await self.checkpoints.save(context, "failed")
                        continue
                    await self.checkpoints.save(context, "executing")
                    remaining_tokens = max(
                        0,
                        budget.limits.max_llm_tokens - budget.usage.llm_tokens,
                    )
                    from .budgets import stage_token_reserves
                    report_reserve, verification_reserve = stage_token_reserves(
                        budget.limits.max_llm_tokens, REPORT_RESERVED_TOKENS, VERIFICATION_RESERVED_TOKENS)
                    await self.events.publish(
                        run_id,
                        "budget.stage_reserved",
                        stage="executing",
                        payload={
                            "remaining_tokens": remaining_tokens,
                            "report_reserved_tokens": report_reserve,
                            "verification_reserved_tokens": verification_reserve,
                            "reserved_budget_fallback": remaining_tokens < (
                                report_reserve + verification_reserve
                            ),
                        },
                    )
                    await self._transition(context, RunStatus.REPORTING)

                elif context.status is RunStatus.REPORTING:
                    await self._assert_not_cancelled(context)
                    generated_report = await self._run_with_heartbeat(context, budget, driver.report())
                    context.report = sanitize_report(self._apply_artifact_edit(context, generated_report))
                    await self._assert_not_cancelled(context)
                    context.workflow_state = driver.snapshot()
                    await self._save_report_artifact(context)
                    await self.checkpoints.save(context, "reporting")
                    report_metrics = (
                        driver.report_metrics() if hasattr(driver, "report_metrics") else {}
                    )
                    await self.events.publish(
                        run_id,
                        "report.completed",
                        stage="reporting",
                        payload={"characters": len(context.report or ""), **report_metrics,
                                 "research_stops": [{"task_id": task_id, "stop_reason": step.get("stop_reason"), "open_questions": step.get("open_questions", [])}
                                                    for task_id, step in context.context_snapshot.get("research_steps", {}).items()]},
                    )
                    await self._transition(context, RunStatus.VERIFYING)

                elif context.status is RunStatus.VERIFYING:
                    evidence = await self.evidence_repository.list_for_run(run_id)
                    delivery_review = await self._check_delivery(context, driver, budget, evidence)
                    # Delivery recovery can add persisted, source-isolated evidence.
                    evidence = await self.evidence_repository.list_for_run(run_id)
                    if hasattr(driver, "verify_report_consistency"):
                        await self._run_with_heartbeat(context, budget, driver.verify_report_consistency(context.report or ""))
                    verdict = await self.contracts.evaluate(
                        run_id=run_id, source_mode=context.source_mode, report=context.report or "",
                        evidence=evidence, min_evidence=int(context.config_snapshot.get("min_evidence", 1)),
                        required_sections=list(context.config_snapshot.get("required_sections", [])),
                        consistency_passed=driver.report_consistency(),
                        consistency_details=driver.report_consistency_details() if hasattr(driver, "report_consistency_details") else None,
                        research_coverage=(
                            {"open_questions": [question for step in context.context_snapshot.get("research_steps", {}).values()
                                                for question in step.get("open_questions", [])],
                             "stops": [{"task_id": task_id, "reason": step.get("stop_reason")} for task_id, step in context.context_snapshot.get("research_steps", {}).items()]}
                            if (context.config_snapshot.get("check_research_coverage") or context.config_snapshot.get("require_resolved_research")) and context.context_snapshot.get("research_steps") else None
                        ),
                        strict_research_coverage=bool(context.config_snapshot.get("require_resolved_research")),
                        delivery_coverage=delivery_review,
                        evidence_card_coverage=(
                            driver.evidence_card_coverage()
                            if hasattr(driver, "evidence_card_coverage")
                            else None
                        ),
                    )
                    context.workflow_state = driver.snapshot()
                    context.workflow_state["verification_failures"] = verdict.failures
                    await self.checkpoints.save(context, "verifying")
                    decision = classify_verification_failures(verdict.failures) if not verdict.passed else None
                    recovery_action = decision.action if verdict.recoverable and decision is not None else "complete" if verdict.passed else "fail"
                    attempt = None
                    max_attempts = None
                    attempts_exhausted = False
                    if recovery_action == "repair_report":
                        max_attempts = budget.limits.max_task_retries
                        attempts_exhausted = budget.usage.task_retries >= max_attempts
                        attempt = budget.usage.task_retries if attempts_exhausted else budget.usage.task_retries + 1
                    elif recovery_action == "replan":
                        max_attempts = budget.limits.max_replans
                        attempts_exhausted = budget.usage.replans >= max_attempts
                        attempt = budget.usage.replans if attempts_exhausted else budget.usage.replans + 1
                    await self.events.publish(
                        run_id, "verification.completed", stage="verifying",
                        payload={
                            "passed": verdict.passed,
                            "failures": verdict.failures,
                            "recovery_action": recovery_action,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "attempts_exhausted": attempts_exhausted,
                        },
                    )
                    if verdict.passed:
                        self.state_machine.validate(context.status, RunStatus.COMPLETED, contract_passed=True)
                        context.status = RunStatus.COMPLETED
                        context.budget_usage = budget.usage
                        self._attach_prefix_usage(context)
                        await self.runs.complete_verified(run_id, assistant_content=context.report or "", usage=budget.snapshot())
                        await self.checkpoints.save(context, "completed")
                        await self.events.publish(run_id, "run.completed", stage="completed", payload={"verified": True})
                        if self.memory_extractor is not None:
                            try:
                                candidates = await self.memory_extractor.extract_from_completed_run(run_id)
                                for candidate in candidates:
                                    await self.events.publish(run_id, "memory.candidate_created", stage="completed", payload={
                                        "memory_id": candidate.memory_id,
                                        "target": candidate.target or candidate.scope,
                                        "kind": candidate.kind,
                                        "confidence": candidate.confidence,
                                        "created_by": candidate.created_by,
                                    })
                            except Exception as exc:
                                await self.events.publish(run_id, "memory.extraction_failed", stage="completed", payload={"error": type(exc).__name__})
                        if self.skill_distiller is not None:
                            try:
                                candidate = await self.skill_distiller.distill(run_id)
                                if candidate is not None:
                                    await self.events.publish(run_id, "skill.candidate_created", stage="completed", payload={"candidate_id": candidate.candidate_id, "name": candidate.name, "version": candidate.proposed_version})
                            except Exception as exc:
                                await self.events.publish(run_id, "skill.distillation_failed", stage="completed", payload={"error": type(exc).__name__})
                        break

                    assert decision is not None
                    if (
                        verdict.recoverable
                        and decision.action == "repair_report"
                        and budget.usage.task_retries >= budget.limits.max_task_retries
                    ):
                        await self._transition(
                            context, RunStatus.FAILED, event_type="run.failed",
                            error_message=f"报告自动修复 {budget.usage.task_retries} 次后仍未通过: {', '.join(verdict.failures)}",
                            payload={"recovery_action": "repair_report", "attempts_exhausted": True, "failures": verdict.failures},
                        )
                    elif (
                        verdict.recoverable
                        and decision.action == "replan"
                        and budget.usage.replans >= budget.limits.max_replans
                    ):
                        await self._transition(
                            context, RunStatus.FAILED, event_type="run.failed",
                            error_message=f"重新规划 {budget.usage.replans} 次后仍未通过: {', '.join(verdict.failures)}",
                            payload={"recovery_action": "replan", "attempts_exhausted": True, "failures": verdict.failures},
                        )
                    elif verdict.recoverable and decision.action == "repair_report" and await self._run_with_heartbeat(context, budget, driver.repair_report(verdict.failures)):
                        budget.consume_retry()
                        context.budget_usage = budget.usage
                        context.workflow_state = driver.snapshot()
                        # This checkpoint admits a pending report commit, rather
                        # than recording a completed reporting stage.
                        context.context_snapshot["inflight_stage"] = "reporting"
                        await self._transition(
                            context, RunStatus.REPORTING, event_type="run.retrying",
                            payload={
                                "recovery_action": "repair_report", "attempt": budget.usage.task_retries,
                                "max_attempts": budget.limits.max_task_retries, "failures": verdict.failures,
                            },
                        )
                        await self.checkpoints.save(context, "reporting")
                    elif verdict.recoverable and decision.action == "replan":
                        budget.consume_replan()
                        context.budget_usage = budget.usage
                        await self._transition(
                            context, RunStatus.REPLANNING, event_type="plan.replanning",
                            payload={
                                "recovery_action": "replan", "attempt": budget.usage.replans,
                                "max_attempts": budget.limits.max_replans, "failures": verdict.failures,
                            },
                        )
                    else:
                        message = ("报告仍未完整覆盖用户交付要求，已保留草稿及逐项检查依据；请查看 delivery_coverage。"
                                   if "delivery_coverage" in verdict.failures else "研究仍有未解决的问题，已保留报告、证据和检查点；请查看 research_coverage 检查记录。"
                                   if "research_coverage" in verdict.failures else f"Completion Contract 未通过: {', '.join(verdict.failures)}")
                        await self._transition(context, RunStatus.FAILED, event_type="run.failed", error_message=message)

            return context
        except RunLeaseLost:
            raise
        except asyncio.CancelledError:
            run = await self.runs.get(run_id)
            if run is not None and run.cancellation_requested and context is not None:
                if context.status is not RunStatus.CANCELLING:
                    await self._transition(context, RunStatus.CANCELLING, event_type="run.cancelling")
                await self._transition(context, RunStatus.CANCELLED, event_type="run.cancelled")
                await self.checkpoints.save(context, "cancelled")
                return context
            if context is not None and not self.state_machine.is_terminal(context.status) and context.status not in {RunStatus.PAUSED, RunStatus.NEEDS_USER_INPUT, RunStatus.INTERRUPTED}:
                # Keep the last committed checkpoint. The operation being
                # cancelled may have only partially mutated its in-memory state.
                await self._transition(context, RunStatus.INTERRUPTED, event_type="run.interrupted",
                    payload={"reason": "executor_stopped"})
            raise
        except RunCancelled:
            assert context is not None
            if context.status is not RunStatus.CANCELLING:
                await self._transition(context, RunStatus.CANCELLING, event_type="run.cancelling")
            await self._transition(context, RunStatus.CANCELLED, event_type="run.cancelled")
            await self.checkpoints.save(context, "cancelled")
            return context
        except BudgetExceeded as exc:
            assert context is not None
            await self._transition(context, RunStatus.BUDGET_EXHAUSTED, event_type="run.budget_exhausted", error_code="BUDGET_EXHAUSTED", error_message=str(exc))
            await self.checkpoints.save(context, "budget_exhausted")
            return context
        except Exception as exc:
            if context is not None and not self.state_machine.is_terminal(context.status):
                if str(exc) == "needs_user_input":
                    question = context.config_snapshot.get("clarification_question") or str(exc)
                    await self._transition(context, RunStatus.NEEDS_USER_INPUT, event_type="run.needs_user_input",
                                           error_message=question, payload={"clarification_question": question})
                else:
                    await self._transition(context, RunStatus.FAILED, event_type="run.failed", error_code=type(exc).__name__, error_message=str(exc))
                await self.checkpoints.save(context, context.status.value)
            return context if context is not None else await self._load_context(run_id)
        finally:
            if retrieval_token is not None:
                reset_retrieval_runtime(retrieval_token)
            if call_budget_token is not None:
                current_call_budget().close()
                reset_call_budget(call_budget_token)
            set_current_run(None)
            await self.runs.release_lease(run_id, owner)

    async def _check_delivery(self, context, driver, budget, evidence):
        from .delivery import report_hash
        from .request_requirements import normalize_requirement, research_requirements
        from .delivery_evidence import coverage_progress, order_supplement_sources
        requirements = context.context_snapshot.get("delivery_requirements")
        if self.delivery_service is None or not context.config_snapshot.get("delivery_checks") or not requirements:
            return None
        requirements = [value for item in requirements if (value := normalize_requirement(item)) is not None]
        context.context_snapshot["delivery_requirements"] = requirements
        state = context.context_snapshot.setdefault("delivery", {"repair_attempts": 0, "history": []})

        async def checkpoint():
            context.context_snapshot["inflight_stage"] = "verifying"
            context.workflow_state = driver.snapshot()
            self._attach_prefix_usage(context)
            await self.checkpoints.save(context, "verifying")

        async def review():
            if state.get("review", {}).get("report_hash") != report_hash(context.report or ""):
                await checkpoint()
                state["review"] = await self._run_with_heartbeat(context, budget,
                    self.delivery_service.review(requirements, context.report or ""))
                from .report_reconciliation import reconcile_report
                report, reconciled = reconcile_report(context.report or "", state["review"])
                if report != context.report:
                    context.report = report
                    driver.apply_delivery_report(report)
                    state["review"] = reconciled
                    await self._save_report_artifact(context)
                state["history"].append(state["review"])
                await checkpoint()
                await self.events.publish(context.run_id, "delivery.review.completed", stage="verifying", payload=state["review"])
            return state["review"]

        result = await review()
        research_ids = {item["id"] for item in research_requirements(requirements)}
        # Two admitted supplementary generations per Run, persisted before each
        # call. A later citation repair can reuse the remaining attempt; no-op
        # or invalid patches stop immediately instead of looping unchanged text.
        while not result["passed"] and not state.get("stop_reason") and state["repair_attempts"] < 2 and any(
            item["status"] in {"missing", "partial", "insufficient_evidence"} and item["requirement_id"] in research_ids for item in result["items"]):
            gaps = [item for item in result["items"] if item["requirement_id"] in research_ids
                    and item["status"] != "covered" and (item.get("gap_type") == "evidence" or item["status"] == "insufficient_evidence")]
            if gaps:
                recovered = await self._retrieve_delivery_gaps(context, driver, budget, requirements, gaps, checkpoint)
                if not recovered:
                    state["stop_reason"] = "evidence_gap_unresolved"
                    state["repair_error"] = "缺少基础证据，停止重复补写：" + state.get("search_stop_reason", "定向补搜未获得新材料")
                    await checkpoint()
                    await self.events.publish(context.run_id, "delivery.supplement.failed", stage="verifying",
                        payload={"reason": state["repair_error"], "stop_reason": state["stop_reason"]})
                    break
                evidence = await self.evidence_repository.list_for_run(context.run_id)
            previous_progress = coverage_progress(result)
            state["repair_attempts"] += 1
            # Persist admission before the model call: a crash cannot cause an
            # unlimited sequence of supplementary generations on recovery.
            await checkpoint()
            try:
                from deepresearch_agent.retrieval.evidence_context import source_packet
                from deepresearch_agent.config.settings import REPORT_REPAIR_EVIDENCE_MAX_TOKENS
                from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
                known_ids = {item.evidence_id for item in evidence}
                raw_sources = [result for _, _, _, result in driver.evidence_results() if result.result_id in known_ids]
                supplied_ids = {item.result_id for item in raw_sources}
                # Provider identifiers are not RetrievalResult tool enums. Use
                # custom only for legacy ledger rows missing a full result.
                raw_sources.extend(RetrievalResult(result_id=item.evidence_id, source_mode=item.source_mode,
                    source="custom", granularity="DO", evidence=item.summary,
                    metadata=json.loads(item.metadata_json or "{}")) for item in evidence if item.evidence_id not in supplied_ids)
                new_ids = {row["result_id"] for row in state.get("retrieved_results", [])}
                raw_sources = order_supplement_sources(raw_sources, requirements, result, new_ids)
                sources = source_packet(raw_sources, max_tokens=REPORT_REPAIR_EVIDENCE_MAX_TOKENS,
                    max_sources=max(12, len(new_ids)))
                state["supplement_source_ids"] = [source["evidence_id"] for source in sources]
                await checkpoint()
                patch = await self._run_with_heartbeat(context, budget, self.delivery_service.supplement(
                    requirements, result, context.report or "",
                    sources))
                if patch:
                    state["processed_evidence_ids"] = sorted(set(state.get("processed_evidence_ids", [])) | new_ids.intersection(state["supplement_source_ids"]))
                    context.report = sanitize_report((context.report or "") + "\n\n" + patch)
                    driver.apply_delivery_report(context.report)
                    await self._save_report_artifact(context)
                    await checkpoint()
                    await self.events.publish(context.run_id, "delivery.supplement.completed", stage="verifying",
                        payload={"attempt": state["repair_attempts"], "characters": len(patch)})
                    result = await review()
                    progress = coverage_progress(result)
                    if not result["passed"] and not any(progress.get(rid, 0) > level for rid, level in previous_progress.items()):
                        state["stop_reason"] = "no_coverage_progress"
                        state["repair_error"] = "补写后交付缺口没有改善，停止重复补写并保留未完成项。"
                        await checkpoint()
                        await self.events.publish(context.run_id, "delivery.supplement.failed", stage="verifying",
                            payload={"reason": state["repair_error"], "stop_reason": state["stop_reason"], "attempt": state["repair_attempts"]})
                        break
                else:
                    state["repair_error"] = "交付补写未返回有效内容，保留报告和缺口"
                    await checkpoint()
                    await self.events.publish(context.run_id, "delivery.supplement.failed", stage="verifying",
                        payload={"reason": state["repair_error"], "attempt": state["repair_attempts"]})
                    break
            except ValueError as exc:
                state["repair_error"] = str(exc)[:1000]
                await checkpoint()
                await self.events.publish(context.run_id, "delivery.supplement.failed", stage="verifying",
                    payload={"reason": state["repair_error"]})
                break
        return result

    async def _retrieve_delivery_gaps(self, context, driver, budget, requirements, gaps, checkpoint):
        """At most two durable, read-only searches; never restart the full plan."""
        from .budgets import stage_token_reserves
        from .errors import AppError
        from .retrieval_runtime import RuntimeRetrievalProvider
        from deepresearch_agent.retrieval.base import SearchFilters, ToolCallContext, TimeoutBoundProvider
        state = context.context_snapshot["delivery"]
        provider = getattr(driver, "retrieval_provider", None)
        if provider is None:
            state["search_stop_reason"] = "当前工作流未提供可用检索接口"
            return False
        if provider.mode != context.source_mode:
            state["search_stop_reason"] = "检索来源与本次 Run 不一致，禁止跨来源补搜"
            return False
        if not isinstance(provider, RuntimeRetrievalProvider):
            provider = RuntimeRetrievalProvider(TimeoutBoundProvider(provider, timeout_seconds=budget.limits.tool_timeout_seconds))
        requests = state.setdefault("searches", [])
        requirements_by_id = {item["id"]: item for item in requirements}
        processed = set(state.get("processed_evidence_ids", []))
        recovered = any(entry.get("status") == "completed" and set(entry.get("evidence_ids", [])) - processed for entry in requests)
        for gap in gaps:
            rid = gap["requirement_id"]
            pending = next((entry for entry in requests if entry["requirement_id"] == rid and entry["status"] == "running"), None)
            if pending is None:
                if len(requests) >= 2 or any(entry["requirement_id"] == rid for entry in requests):
                    state["search_stop_reason"] = "已达到本 Run 两次定向补搜上限，或该缺口已补搜过"
                    continue
                policy = current_call_budget()
                remaining = min(policy.remaining_tokens(), budget.remaining_tokens()) if policy else budget.remaining_tokens()
                reserve = stage_token_reserves(budget.limits.max_llm_tokens, REPORT_RESERVED_TOKENS, VERIFICATION_RESERVED_TOKENS)[1]
                from deepresearch_agent.config.settings import REPORT_REPAIR_EVIDENCE_MAX_TOKENS, DELIVERY_MAX_OUTPUT_TOKENS
                tools, searches = policy.tool_usage() if policy else (budget.usage.tool_calls, budget.usage.tavily_calls)
                tools, searches = max(tools, budget.usage.tool_calls), max(searches, budget.usage.tavily_calls)
                if remaining < reserve + REPORT_REPAIR_EVIDENCE_MAX_TOKENS + DELIVERY_MAX_OUTPUT_TOKENS:
                    state["search_stop_reason"] = "剩余 Token 不足以同时保留补写和后续验证预算"
                    continue
                if tools >= budget.limits.max_tool_calls or context.source_mode is SourceMode.WEB and searches >= budget.limits.max_tavily_calls:
                    state["search_stop_reason"] = "本次 Run 的工具调用或网络搜索次数已达上限"
                    continue
                budget.assert_available()
                text = str(requirements_by_id[rid].get("text", ""))
                query = (text + " " + str(gap.get("search_query") or gap.get("reason", "")))[:500]
                pending = {"requirement_id": rid, "query": query,
                           "tool_call_id": f"call_delivery_{context.run_id}_{len(requests) + 1}", "status": "running",
                           "known_evidence_ids": [row.evidence_id for row in await self.evidence_repository.list_for_run(context.run_id)]}
                requests.append(pending)
                await checkpoint()
            try:
                rows = await self._run_with_heartbeat(context, budget, provider.search(pending["query"],
                    top_k=5, search_depth="advanced", filters=SearchFilters(),
                    call_context=ToolCallContext(run_id=context.run_id, source_mode=context.source_mode,
                                                 task_id=None, tool_call_id=pending["tool_call_id"])))
                existing = set(pending.get("known_evidence_ids", []))
                saved = {row["result_id"]: row for row in state.get("retrieved_results", [])}
                added = [row for row in rows if row.result_id not in existing]
                saved.update({row.result_id: row.to_dict() for row in added})
                state["retrieved_results"] = list(saved.values())
                pending.update(status="completed", evidence_ids=[row.result_id for row in added])
                if not added:
                    state["search_stop_reason"] = "定向补搜未返回新的证据，保留当前报告与缺口"
                recovered = recovered or bool(added)
            except (AppError, ValueError) as exc:
                pending.update(status="failed", error=type(exc).__name__)
                state["search_stop_reason"] = f"定向补搜失败（{type(exc).__name__}），保留当前报告与缺口"
            await checkpoint()
        return recovered

    async def _load_context(self, run_id: str) -> RunContext:
        run = await self.runs.get(run_id)
        if run is None:
            raise ValueError(f"Run 不存在: {run_id}")
        restored = await self.checkpoints.restore(run_id)
        if restored is not None:
            persisted_status = None if run.status == "pausing" else RunStatus(run.status)
            if persisted_status is RunStatus.INTERRUPTED:
                inflight = restored.context_snapshot.get("inflight_stage")
                restored.resume_from_status = RunStatus(inflight) if inflight else self._safe_stage_after(restored.status)
                restored.status = RunStatus.INTERRUPTED
            elif persisted_status is RunStatus.PAUSED:
                config = json.loads(run.config_snapshot_json or "{}")
                restored.resume_from_status = restored.resume_from_status or RunStatus(
                    config.get("pause_resume_status") or restored.context_snapshot.get("inflight_stage")
                    or self._safe_stage_after(restored.status).value
                )
                restored.status = RunStatus.PAUSED
            elif persisted_status is not None:
                restored.status = persisted_status
            restored.cancellation_requested = bool(run.cancellation_requested)
            restored.config_snapshot.update(json.loads(run.config_snapshot_json or "{}"))
            # Admission/heartbeat usage can be newer than the last checkpoint.
            persisted_usage = json.loads(run.usage_json or "{}").get("usage", {})
            for name in ("llm_tokens", "tool_calls", "tavily_calls", "elapsed_seconds",
                         "prefix_cache_requests", "prefix_cache_hit_tokens", "prefix_cache_miss_tokens"):
                if name in persisted_usage:
                    setattr(restored.budget_usage, name, max(getattr(restored.budget_usage, name), persisted_usage[name]))
            return restored
        message = await self.messages.get(run.trigger_message_id)
        if message is None:
            raise ValueError(f"Run {run_id} 的触发消息不存在")
        limits = json.loads(run.budget_json or "{}") or {}
        config_snapshot = json.loads(run.config_snapshot_json or "{}")
        return RunContext(
            run_id=run.run_id, session_id=run.session_id, trigger_message_id=run.trigger_message_id,
            source_mode=SourceMode(run.source_mode), workflow_mode=WorkflowMode(run.workflow_mode),
            status=(
                RunStatus(run.current_stage or "queued")
                if run.status == "pausing" else RunStatus(run.status)
            ), original_query=message.content,
            config_snapshot=config_snapshot,
            model_snapshot=json.loads(run.model_snapshot_json or "{}"),
            budget_limits=BudgetLimits.model_validate(limits),
            resume_from_status=(
                RunStatus(config_snapshot.get("pause_resume_status", "queued"))
                if run.status == "paused" else None
            ),
        )

    @staticmethod
    def _safe_stage_after(checkpoint_status: RunStatus) -> RunStatus:
        """Return the first operation not committed by a stage-end checkpoint."""
        return {
            RunStatus.CONTEXT_BUILDING: RunStatus.PLANNING,
            RunStatus.PLANNING: RunStatus.EXECUTING,
            RunStatus.REPLANNING: RunStatus.EXECUTING,
            RunStatus.EXECUTING: RunStatus.REPORTING,
            RunStatus.RETRYING: RunStatus.EXECUTING,
            RunStatus.REPORTING: RunStatus.VERIFYING,
            RunStatus.VERIFYING: RunStatus.VERIFYING,
        }.get(checkpoint_status, RunStatus.QUEUED)

    async def _transition(
        self,
        context: RunContext,
        target: RunStatus,
        *,
        event_type: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.state_machine.validate(context.status, target)
        context.status = target
        self._attach_prefix_usage(context)
        current_stage = (
            context.resume_from_status.value
            if target is RunStatus.PAUSED and context.resume_from_status is not None
            else target.value
        )
        await self.runs.update_status(context.run_id, status=target.value, current_stage=current_stage, error_code=error_code, error_message=error_message, usage={"limits": context.budget_limits.model_dump(), "usage": context.budget_usage.model_dump()})
        event_payload = {"status": target.value, "error_code": error_code, **(payload or {})}
        if target is RunStatus.PAUSED:
            event_payload["resume_from_status"] = current_stage
        await self.events.publish(context.run_id, event_type or "run.stage_changed", stage=target.value, payload=event_payload)

    async def _pause_at_safe_boundary(self, context: RunContext) -> bool:
        run = await self.runs.get(context.run_id)
        pause_requested = bool(
            run and json.loads(run.config_snapshot_json or "{}").get("pause_requested")
        )
        if run is None or (run.status != "pausing" and not pause_requested):
            return False
        context.resume_from_status = context.status
        await self._transition(context, RunStatus.PAUSED, event_type="run.paused")
        await self.checkpoints.save(context, "paused")
        await self.runs.clear_pause_request(context.run_id)
        return True

    async def _assert_not_cancelled(self, context: RunContext) -> None:
        run = await self.runs.get(context.run_id)
        if run is not None and run.cancellation_requested:
            context.cancellation_requested = True
            raise RunCancelled("Run 已请求取消")

    async def _run_with_heartbeat(self, context: RunContext, budget: BudgetManager, operation):
        """Run one long stage while persisting usage and enforcing cancel/budget limits."""
        task = asyncio.create_task(operation)
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=min(1.0, max(0.05, self.lease_seconds / 3)))
                if time.monotonic() >= self._next_renewal:
                    if not await self.runs.renew_lease(context.run_id, self._lease_owner, ttl_seconds=self.lease_seconds):
                        raise RunLeaseLost("运行租约已转移，原执行器停止")
                    self._next_renewal = time.monotonic() + max(0.05, self.lease_seconds / 3)
                self._attach_prefix_usage(context)
                context.budget_usage = budget.usage
                budget.assert_available()
                await self.runs.update_usage(context.run_id, budget.snapshot())
                if not task.done():
                    await self._assert_not_cancelled(context)
            return await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def _persist_plan(self, context: RunContext, driver: Any) -> None:
        if self.trajectory is None:
            return
        record = driver.plan_record()
        if record is None:
            return
        plan, tasks = record
        plan_id = str(plan.get("plan_id") or f"plan_{context.run_id}_{context.plan_version}")
        await self.trajectory.save_plan(run_id=context.run_id, plan_id=plan_id, version=context.plan_version, status=str(plan.get("status", "executing")), plan=plan, tasks=tasks, source_mode=context.source_mode.value)

    async def _persist_execution(self, context: RunContext, driver: Any, budget: BudgetManager) -> None:
        seen_calls: set[str] = set()
        live_events = await self.events.repository.list_after(context.run_id)
        live_task_ids = {
            str(json.loads(item.payload_json or "{}").get("task_id", ""))
            for item in live_events if item.event_type in {"task.completed", "task.failed"}
        }
        live_tool_ids = {
            str(json.loads(item.payload_json or "{}").get("tool_call_id", ""))
            for item in live_events if item.event_type in {"tool.completed", "tool.failed"}
        }
        published_evidence_ids = {item.evidence_id for item in await self.evidence_repository.list_for_run(context.run_id)}
        records = driver.execution_records()
        reported_tokens = sum(
            self._token_usage_total(record.metadata.token_usage)
            for record in records
        )
        if reported_tokens:
            budget.observe_tokens(reported_tokens)
        for record in records:
            if record.task_id not in live_task_ids:
                await self.events.publish(context.run_id, "task.completed", stage="executing", payload={"task_id": record.task_id, "evidence_count": len(record.evidence)})
            for call in record.tool_calls:
                if call.tool_call_id in seen_calls:
                    continue
                seen_calls.add(call.tool_call_id)
                existing = await self.trajectory.get_tool_call(call.tool_call_id) if self.trajectory else None
                if existing is not None and existing.status in {"completed", "failed"}:
                    continue
                budget.consume_tool(tavily=call.source_mode == "web")
                if self.trajectory:
                    await self.trajectory.prepare_tool_call(tool_call_id=call.tool_call_id, run_id=context.run_id, task_id=record.task_id, tool_name=call.tool_name, source_mode=context.source_mode.value, args=call.args)
                    await self.trajectory.complete_tool_call(call.tool_call_id, result=call.result, error_code="TOOL_FAILED" if call.status == "failed" else None)
                tool_payload: dict[str, Any] = {"task_id": record.task_id, "tool_call_id": call.tool_call_id, "tool_name": call.tool_name}
                if isinstance(call.args, dict) and call.args.get("query"):
                    tool_payload["query"] = str(call.args["query"])[:300]
                if isinstance(call.result, dict):
                    tool_payload["result_count"] = len(call.result.get("result_ids", []) or [])
                if not tool_payload.get("result_count") and record.evidence:
                    tool_payload["result_count"] = len(record.evidence)
                if call.tool_call_id not in live_tool_ids:
                    await self.events.publish(context.run_id, "tool.completed" if call.status != "failed" else "tool.failed", stage="executing", payload=tool_payload)

        for task_id, tool_call_id, provider, result in driver.evidence_results():
            saved = await self.evidence_ledger.record_results(run_id=context.run_id, task_id=task_id, tool_call_id=tool_call_id, provider=str(provider), results=[result])
            for item in saved:
                if item.evidence_id in published_evidence_ids:
                    continue
                published_evidence_ids.add(item.evidence_id)
                await self.events.publish(context.run_id, "evidence.added", stage="executing", payload={"task_id": task_id, "evidence_id": item.evidence_id, "source_mode": item.source_mode.value})

    @staticmethod
    def _token_usage_total(usage: dict[str, Any] | None) -> int:
        """Normalize provider usage without double-counting a reported total."""
        values = usage or {}
        for key in ("total_tokens", "total", "tokens"):
            if values.get(key) is not None:
                return max(0, int(values[key]))
        input_tokens = next(
            (max(0, int(values[key])) for key in ("input_tokens", "prompt_tokens", "prompt", "input") if values.get(key) is not None),
            0,
        )
        output_tokens = next(
            (max(0, int(values[key])) for key in ("output_tokens", "completion_tokens", "completion", "output") if values.get(key) is not None),
            0,
        )
        if input_tokens or output_tokens:
            return input_tokens + output_tokens
        return sum(max(0, int(value or 0)) for value in values.values())

    async def _save_report_artifact(self, context: RunContext) -> None:
        if self.artifact_store is None:
            return
        artifact = self.artifact_store.write_text(f"{context.run_id}/report/final.md", context.report or "", mime_type="text/markdown; charset=utf-8")
        if self.artifact_repository is not None:
            await self.artifact_repository.record(artifact, run_id=context.run_id)

    def _apply_artifact_edit(self, context: RunContext, generated_report: str) -> str:
        contract = context.context_snapshot.get("artifact_edit")
        if not contract:
            return generated_report
        if self.artifact_store is None:
            raise ValueError("Artifact edit requires an ArtifactStore")
        base_report = self.artifact_store.read_bytes(contract["relative_path"]).decode("utf-8")
        return ArtifactEditContextBuilder.apply_replacement(base_report, contract, generated_report)
