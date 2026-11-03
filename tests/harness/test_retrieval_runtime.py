"""Offline integration: real SQLite/checkpoints, deterministic provider/model."""
import json

import pytest
from langchain_core.messages import AIMessage

from backend.app.schemas import MessageCreate, RunCreate, SessionCreate
from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult, RetrievalMetadata
from deepresearch_agent.harness.budgets import BudgetLimits
from deepresearch_agent.harness.contracts import RunStatus, SourceMode, WorkflowMode
from deepresearch_agent.harness.retrieval_runtime import RuntimeRetrievalProvider
from deepresearch_agent.harness.runtime import HarnessRuntime
from deepresearch_agent.harness.workflow import DeepResearchDriver
from deepresearch_agent.persistence import Database
from deepresearch_agent.persistence.repositories import (
    CheckpointRepository, ContractRepository, EventRepository, EvidenceRepository,
    MessageRepository, PlanTaskToolRepository, RunRepository, SessionRepository,
)
from deepresearch_agent.retrieval.research_loop import BoundedResearchAgent


class InjectedStop(BaseException):
    """Simulates loss of the executing task without graceful stage completion."""


class Provider:
    mode, provider_name = SourceMode.WEB, "offline"
    calls = 0

    async def search(self, query, **kwargs):
        self.calls += 1
        return [RetrievalResult(granularity="DO", source="tavily_search", source_mode="web",
            evidence="Agent 能够通过工具调用来获取外部信息。",
            metadata=RetrievalMetadata(source_id="https://example.com/a", source_type="webpage"))]


class Model:
    model_name = "fake"

    def __init__(self, *, stop=False, stop_report=False, stale_gap=False):
        self.stop = stop
        self.stop_report = stop_report
        self.stale_gap = stale_gap

    async def ainvoke(self, messages, **kwargs):
        payload = json.loads(messages[-1].content)
        if "sources" in payload:
            source = payload["sources"][0]
            if self.stop:
                raise InjectedStop()
            return AIMessage(content=json.dumps({"findings": [{"claim": "Agent 能够通过工具调用获取外部信息。",
                "evidence_id": source["evidence_id"], "quote": source["text"]}], "next_query": None}))
        if self.stop_report:
            raise InjectedStop()
        item = payload["findings"][0]
        return AIMessage(content=json.dumps({"sections": [{"title": "工具调用", "paragraphs": [
            {"text": "Agent 能够通过工具调用来获取外部信息。", "evidence_ids": [item["evidence_id"]]}]}],
            "limitations": ["还无法说明Agent如何获取外部信息。", "资料来源有限，未做实际实验。"] if self.stale_gap else []}))


@pytest.mark.asyncio
@pytest.mark.parametrize("crash_stage", ["analysis", "step_checkpoint", "report", "delivery_review", "delivery_reconciliation"])
async def test_resume_reuses_committed_search_in_both_crash_windows(tmp_path, crash_stage):
    db = Database(f"sqlite+aiosqlite:///{(tmp_path / 'resume.db').as_posix()}")
    await db.create_schema()
    provider = Provider()
    runs, trajectory = RunRepository(db), PlanTaskToolRepository(db)
    session = await SessionRepository(db).create(SessionCreate(title="offline"))
    _, run, _ = await runs.create_for_user_message(
        MessageCreate(session_id=session.session_id, role="user", content="解释 Agent 的工具调用"),
        RunCreate(session_id=session.session_id, trigger_message_id="atomic", source_mode=SourceMode.WEB,
            config_snapshot={"delivery_checks": crash_stage in {"delivery_review", "delivery_reconciliation"}},
            workflow_mode=WorkflowMode.DEEP_RESEARCH, budget=BudgetLimits(max_tool_calls=1, max_tavily_calls=1).model_dump()),
    )
    stopping = True

    class Delivery:
        supplements = 0
        review_calls = 0

        async def extract(self, question):
            return [{"id": "req_1", "text": "给出验证方法", "request_quote": "工具调用", "kind": "analysis"}]

        async def review(self, requirements, report):
            from deepresearch_agent.harness.delivery import report_hash
            self.review_calls += 1
            if crash_stage == "delivery_reconciliation":
                from deepresearch_agent.harness.report_reconciliation import reconciliation_candidates, report_paragraphs
                paragraphs, eligible = report_paragraphs(report)
                pid = eligible[0]
                review = {"passed": True, "report_hash": report_hash(report), "requirements": requirements,
                          "items": [{"requirement_id": "req_1", "status": "covered", "paragraph_ids": [pid], "report_quote": paragraphs[pid], "reason": "已说明工具调用获取外部信息"}]}
                review["resolved_limitations"] = reconciliation_candidates(report, review,
                    [{"limitation_id": "lim_1", "requirement_id": "req_1", "paragraph_ids": [pid], "reason": "正文已回答此项"}])
                return review
            covered = "## 验证方法" in report
            if covered and stopping:
                raise InjectedStop()
            return {"passed": covered, "report_hash": report_hash(report), "requirements": requirements,
                    "items": [{"requirement_id": "req_1", "status": "covered" if covered else "missing", "report_quote": "", "reason": "离线测试"}]}

        async def supplement(self, requirements, review, report, evidence):
            self.supplements += 1
            return f"## 验证方法\n\n建议用固定问题集检查工具调用结果与完成情况。 [{evidence[0]['evidence_id']}]"

    delivery = Delivery()

    def factory(context, events):
        wrapped = RuntimeRetrievalProvider(provider)
        agent = BoundedResearchAgent(wrapped, run_id=context.run_id)
        agent.research_tool.llm = Model(stop=stopping and crash_stage in {"analysis", "step_checkpoint"}, stop_report=stopping and crash_stage == "report", stale_gap=crash_stage == "delivery_reconciliation")
        if stopping and crash_stage == "step_checkpoint":
            async def stop(*args):
                raise InjectedStop()
            wrapped.checkpoint = stop
        return DeepResearchDriver(context, agent, events)

    def runtime():
        result = HarnessRuntime(run_repository=runs, message_repository=MessageRepository(db),
            event_repository=EventRepository(db), checkpoint_repository=CheckpointRepository(db),
            evidence_repository=EvidenceRepository(db), contract_repository=ContractRepository(db),
            trajectory_repository=trajectory, workflow_factory=factory,
            delivery_service=delivery if crash_stage in {"delivery_review", "delivery_reconciliation"} else None)
        if stopping and crash_stage == "delivery_reconciliation":
            async def after_review(event):
                if event.event_type == "delivery.review.completed":
                    raise InjectedStop()
            result.events.subscribe(run.run_id, after_review)
        return result
    try:
        try:
            unexpected = await runtime().execute_run(run.run_id)
        except InjectedStop:
            pass
        else:
            row = await runs.get(run.run_id)
            pytest.fail(f"expected injected stop; got {unexpected.status}: {row.error_code}: {row.error_message}")
        assert provider.calls == 1
        evidence = await EvidenceRepository(db).list_for_run(run.run_id)
        assert len(evidence) == 1 and evidence[0].evidence_id.startswith("ev_")
        call = await trajectory.get_tool_call(evidence[0].tool_call_id)
        assert call.status == "completed"
        assert json.loads(call.result_json)["retrieval_results"][0]["evidence"] == "Agent 能够通过工具调用来获取外部信息。"
        tool_events = [json.loads(event.payload_json) for event in await EventRepository(db).list_after(run.run_id)
                       if event.event_type.startswith("tool.")]
        assert tool_events and all(item["tool_name"] == "tavily_search" for item in tool_events)
        assert all(item["query"] == "解释 Agent 的工具调用" for item in tool_events)
        await runs.update_status(run.run_id, status="interrupted", current_stage="executing")
        # Reopen SQLite and construct fresh repositories/runtime: no in-memory
        # checkpoint or driver can satisfy the recovery assertions.
        await db.close()
        db = Database(f"sqlite+aiosqlite:///{(tmp_path / 'resume.db').as_posix()}")
        runs, trajectory = RunRepository(db), PlanTaskToolRepository(db)
        stopping = False
        restored = await runtime().execute_run(run.run_id)
        row = await runs.get(run.run_id)
        assert restored.status == RunStatus.COMPLETED, f"{row.error_code}: {row.error_message}"
        assert provider.calls == 1
        assert restored.budget_usage.tool_calls == 1
        assert restored.budget_usage.tavily_calls == 1
        assert f"[{evidence[0].evidence_id}]" in restored.report
        if crash_stage == "delivery_review":
            assert delivery.supplements == 1
            assert restored.context_snapshot["delivery"]["repair_attempts"] == 1
            assert restored.context_snapshot["delivery_requirements"][0]["id"] == "req_1"
        if crash_stage == "delivery_reconciliation":
            from deepresearch_agent.harness.delivery import report_hash
            review = restored.context_snapshot["delivery"]["review"]
            assert delivery.review_calls == 1 and delivery.supplements == 0
            assert "还无法说明Agent" not in restored.report
            assert "资料来源有限，未做实际实验" in restored.report
            assert review["report_hash"] == report_hash(restored.report)
            assert review["reconciliation"]["removed_limitations"]
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("crash_stage", ["dependency_analysis", "dependency_review_checkpoint", "local_report_repair"])
async def test_per_dependency_handoff_recovers_with_sqlite_without_repeating_search(tmp_path, monkeypatch, crash_stage):
    from deepresearch_agent.agents.multi_agent.core.plan_spec import PlanSpec, ProblemStatement, TaskGraph, TaskNode
    from deepresearch_agent.agents.multi_agent.executor.retrieval_executor import RetrievalExecutor
    from deepresearch_agent.agents.multi_agent.executor.worker_coordinator import WorkerCoordinator
    from deepresearch_agent.agents.multi_agent.planner.base_planner import PlannerResult
    from deepresearch_agent.agents.multi_agent.planner.clarifier import ClarificationResult
    from deepresearch_agent.agents.multi_agent.reporter.base_reporter import ReportResult
    from deepresearch_agent.agents.multi_agent.reporter.outline_builder import ReportOutline
    from deepresearch_agent.harness.workflow import PlanExecuteReportDriver

    path = tmp_path / "per-dependency.db"
    db = Database(f"sqlite+aiosqlite:///{path.as_posix()}")
    await db.create_schema()
    provider, stopping = Provider(), True
    repair_calls, consistency_calls = [], []
    runs, trajectory = RunRepository(db), PlanTaskToolRepository(db)
    session = await SessionRepository(db).create(SessionCreate(title="dependency recovery"))
    _, run, _ = await runs.create_for_user_message(
        MessageCreate(session_id=session.session_id, role="user", content="解释 Agent 的工具调用"),
        RunCreate(session_id=session.session_id, trigger_message_id="atomic", source_mode=SourceMode.WEB,
                  workflow_mode=WorkflowMode.PLAN_EXECUTE_REPORT,
                  budget=BudgetLimits(max_tool_calls=1, max_tavily_calls=1).model_dump()))

    class Orchestrator:
        def __init__(self, wrapped):
            self.coordinator = WorkerCoordinator(executors=[RetrievalExecutor(provider=wrapped)], execution_mode="sequential")

        def plan(self, state, assumptions=None):
            graph = TaskGraph(nodes=[
                TaskNode(task_id="a", task_type="web_search", source_mode="web", description="Agent tool calling"),
                TaskNode(task_id="b", task_type="deep_research", source_mode="web", description="解释工具调用", depends_on=["a"])])
            spec = PlanSpec(source_mode="web", problem_statement=ProblemStatement(original_query=state.input), task_graph=graph)
            state.plan = spec
            return PlannerResult(plan_spec=spec, clarification=ClarificationResult(), executor_signal=spec.to_execution_signal())

        def execute(self, state, plan, **kwargs):
            return self.coordinator.execute_plan(state, plan.executor_signal, **kwargs)

        def report(self, state, report_type):
            item = state.execution_records[0].evidence[0]
            if isinstance(item, dict):
                item = RetrievalResult.from_dict(item)
            body = f"# 报告\n\nAgent 能够通过工具调用来获取外部信息。 [{item.result_id}]\n\n## 局限\n\n当前来源有限。"
            if crash_stage == "local_report_repair":
                body = "# 报告\n\n## 摘要\n\n工具调用能够保证任何异常后都不会重复执行。\n\n## 原理\n\n" + body.split("\n\n", 1)[1]
            return ReportResult(outline=ReportOutline(report_type="brief", title="报告", sections=[]), sections=[], final_report=body)

        def recheck_report_consistency(self, report, evidence):
            from deepresearch_agent.agents.multi_agent.reporter.consistency_checker import ConsistencyCheckResult
            from deepresearch_agent.harness.delivery import report_hash
            consistency_calls.append(report)
            return ConsistencyCheckResult(is_consistent=True, report_hash=report_hash(report))

    def factory(context, events):
        wrapped = RuntimeRetrievalProvider(provider)
        if stopping and crash_stage == "dependency_review_checkpoint":
            checkpoint = wrapped.checkpoint

            async def stop_after_checkpoint(task_id, state):
                await checkpoint(task_id, state)
                if state.get("analysis_origin") == "dependencies" and not state["pending_analysis"]:
                    raise InjectedStop()

            wrapped.checkpoint = stop_after_checkpoint
        return PlanExecuteReportDriver(context, Orchestrator(wrapped), events=events)

    monkeypatch.setattr("deepresearch_agent.retrieval.research_loop.get_llm_model",
                        lambda: Model(stop=stopping and crash_stage == "dependency_analysis"))
    if crash_stage == "local_report_repair":
        class PatchModel:
            async def ainvoke(self, messages, **options):
                data = json.loads(messages[-1].content)
                repair_calls.append(data)
                assert list(data["targets"].values()) == ["工具调用能够保证任何异常后都不会重复执行。"]
                return AIMessage(content=json.dumps({"patches": [{"paragraph_id": next(iter(data["targets"])),
                    "action": "remove", "reason": "删除没有实质结论的摘要"}]}))
        monkeypatch.setattr("deepresearch_agent.harness.delivery.get_llm_model", lambda: PatchModel())

    def runtime():
        checkpoints = CheckpointRepository(db)
        if stopping and crash_stage == "local_report_repair":
            save = checkpoints.save
            async def save_and_stop(run_id, stage, state):
                result = await save(run_id, stage, state)
                if stage == "reporting" and state.get("context_snapshot", {}).get("report_repairs"):
                    raise InjectedStop()
                return result
            checkpoints.save = save_and_stop
        return HarnessRuntime(run_repository=runs, message_repository=MessageRepository(db),
            event_repository=EventRepository(db), checkpoint_repository=checkpoints,
            evidence_repository=EvidenceRepository(db), contract_repository=ContractRepository(db),
            trajectory_repository=trajectory, workflow_factory=factory)

    try:
        with pytest.raises(InjectedStop):
            await runtime().execute_run(run.run_id)
        assert provider.calls == 1
        saved = json.loads((await CheckpointRepository(db).latest(run.run_id)).state_json)
        step = saved["context_snapshot"]["research_steps"]["b"]
        assert step["dependency_evidence_ids"] and step["calls"] == []
        assert saved["workflow_state"]["state"]["execution_context"]["completed_task_ids"] == (["a", "b"] if crash_stage == "local_report_repair" else ["a"])
        await runs.update_status(run.run_id, status="interrupted", current_stage="executing")
        await db.close()
        db = Database(f"sqlite+aiosqlite:///{path.as_posix()}")
        runs, trajectory = RunRepository(db), PlanTaskToolRepository(db)
        stopping = False
        restored = await runtime().execute_run(run.run_id)
        row = await runs.get(run.run_id)
        assert restored.status == RunStatus.COMPLETED, f"{row.error_code}: {row.error_message}"
        assert provider.calls == 1 and restored.budget_usage.tool_calls == restored.budget_usage.tavily_calls == 1
        evidence = await EvidenceRepository(db).list_for_run(run.run_id)
        assert len(evidence) == 1 and evidence[0].task_id == "a"
        call = await trajectory.get_tool_call(evidence[0].tool_call_id)
        assert call.task_id == "a" and call.status == "completed"
        assert restored.context_snapshot["research_steps"]["b"]["calls"] == []
        if crash_stage == "local_report_repair":
            from deepresearch_agent.harness.delivery import report_hash
            assert len(repair_calls) == 1
            # Initial structural prechecks cannot replace semantic review.
            # Check the original and edited versions once each, including recovery.
            assert len(consistency_calls) == 2
            assert "工具调用能够保证任何异常后" in consistency_calls[0]
            assert consistency_calls[1] == restored.report
            assert "工具调用能够保证任何异常后" not in restored.report
            assert "Agent 能够通过工具调用来获取外部信息" in restored.report
            assert restored.workflow_state["consistency_record"]["report_hash"] == report_hash(restored.report)
            assert restored.budget_usage.task_retries == 1
    finally:
        await db.close()
