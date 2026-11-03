"""Replay only a saved PER report in isolation; source database is read-only.

Uses configured LLM quota, but never calls retrieval or updates the source Run.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "data/app.db")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-limit", type=int, default=45000, help="Additional model admission allowance")
    args = parser.parse_args()
    if args.token_limit < 1:
        parser.error("token-limit must be positive")
    database, output = args.database.resolve(), args.output.resolve()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT version,state_hash,state_json FROM checkpoints WHERE run_id=? ORDER BY version DESC LIMIT 1",
            (args.run_id,),
        ).fetchone()
        if row is None:
            parser.error("Run has no checkpoint")
        payload = json.loads(row[2])
        if payload.get("workflow_mode") != "plan_execute_report":
            parser.error("Only PER report replay is supported")
        connection.row_factory = sqlite3.Row
        evidence = [dict(item) for item in connection.execute(
            "SELECT * FROM evidence WHERE run_id=? AND invalidated_at IS NULL", (args.run_id,),
        )]
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CACHE_DIR"] = str(output / "cache")
    os.environ["ARTIFACT_ROOT"] = str(output / "artifacts")
    os.chdir(output)
    asyncio.run(replay(args, row, payload, evidence, output))


async def replay(args, row, payload, evidence, output):
    from deepresearch_agent.agents.multi_agent.integration.multi_agent_factory import MultiAgentFactory
    from deepresearch_agent.harness.budgets import BudgetLimits, BudgetUsage
    from deepresearch_agent.harness.call_budget import CallBudget, set_call_budget, reset_call_budget
    from deepresearch_agent.harness.contracts import ContractEvaluator, SourceMode
    from deepresearch_agent.harness.run_context import RunContext
    from deepresearch_agent.harness.workflow import PlanExecuteReportDriver
    from deepresearch_agent.harness.delivery import DeliveryService
    from deepresearch_agent.models.prefix_cache import tracker, set_current_run

    class NoSearch:
        mode, provider_name = SourceMode(payload["source_mode"]), "replay-no-search"
        async def search(self, *args, **kwargs):
            raise AssertionError("Report replay must never perform retrieval")

    context = RunContext.model_validate(payload)
    bundle = MultiAgentFactory.create_default_bundle(retrieval_provider=NoSearch())
    driver = PlanExecuteReportDriver(context, bundle.orchestrator)
    policy = CallBudget(BudgetLimits(max_llm_tokens=args.token_limit), BudgetUsage())
    token = set_call_budget(policy)
    replay_id = "replay_" + output.name
    set_current_run(replay_id)
    started = time.monotonic()
    result = {
        "source_run_id": args.run_id, "checkpoint_version": row[0], "checkpoint_hash": row[1],
        "scope": "isolated report replay; original Run and evidence unchanged; retrieval forbidden",
        "admission_token_limit": args.token_limit,
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "src/deepresearch_agent/agents/multi_agent/reporter/section_writer.py",
            "src/deepresearch_agent/agents/multi_agent/reporter/citation_renderer.py",
        )},
    }
    try:
        report = await driver.report()
        delivery_review = None
        requirements = context.context_snapshot.get("delivery_requirements")
        if context.config_snapshot.get("delivery_checks") and requirements:
            delivery = DeliveryService()
            delivery_review = await delivery.review(requirements, report)
            if not delivery_review["passed"]:
                patch = await delivery.supplement(requirements, delivery_review, report,
                    [{"evidence_id": item["evidence_id"], "title": item["title"], "text": item["summary"][:1800]} for item in evidence])
                if patch:
                    report += "\n\n" + patch
                    driver.apply_delivery_report(report)
                    delivery_review = await delivery.review(requirements, report)
            result["delivery_review"] = delivery_review
        (output / "report.md").write_text(report, encoding="utf-8")
        await driver.verify_report_consistency(report)
        result["consistency_details"] = driver.report_consistency_details()
        verdict = await ContractEvaluator().evaluate(
            run_id=context.run_id, source_mode=context.source_mode, report=report, evidence=evidence,
            min_evidence=int(context.config_snapshot.get("min_evidence", 1)),
            required_sections=context.config_snapshot.get("required_sections", []),
            consistency_passed=driver.report_consistency(), consistency_details=driver.report_consistency_details(),
            evidence_card_coverage=driver.evidence_card_coverage(),
            research_coverage=(
                {"open_questions": [question for step in context.context_snapshot.get("research_steps", {}).values() for question in step.get("open_questions", [])]}
                if (context.config_snapshot.get("check_research_coverage") or context.config_snapshot.get("require_resolved_research")) and context.context_snapshot.get("research_steps") else None
            ),
            strict_research_coverage=bool(context.config_snapshot.get("require_resolved_research")),
            delivery_coverage=delivery_review,
        )
        result.update(status="completed" if verdict.passed else "verification_failed",
                      checks=[item.model_dump(mode="json") for item in verdict.checks], failures=verdict.failures,
                      report_chars=len(report), retrieval_calls=0)
    except Exception as exc:
        result.update(status="failed", error_code=type(exc).__name__, error_message=str(exc))
        import traceback
        (output / "error-trace.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["usage"] = tracker.run_snapshot(replay_id)
        policy.close()
        reset_call_budget(token)
        set_current_run(None)
        (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("status", "elapsed_seconds", "report_chars", "failures", "error_code", "error_message", "usage")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
