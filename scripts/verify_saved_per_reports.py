"""Strict report-only PER verification; source runs remain read-only, no search."""
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


async def run(trials, output):
    from deepresearch_agent.agents.multi_agent.integration.multi_agent_factory import MultiAgentFactory
    from deepresearch_agent.harness.budgets import BudgetLimits, BudgetUsage
    from deepresearch_agent.harness.call_budget import CallBudget, set_call_budget, reset_call_budget
    from deepresearch_agent.harness.contracts import SourceMode
    from deepresearch_agent.harness.run_context import RunContext
    from deepresearch_agent.harness.workflow import PlanExecuteReportDriver
    from deepresearch_agent.models.prefix_cache import tracker, set_current_run
    from deepresearch_agent.config import settings

    class NoSearch:
        mode, provider_name = SourceMode.WEB, "verification-no-search"
        async def search(self, *args, **kwargs):
            raise AssertionError("Report-only verification cannot search")

    results = []
    for folder in sorted(trials.iterdir()):
        path = folder / "run/app.db"
        if not path.exists():
            continue
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as con:
            row = con.execute("SELECT run_id,status FROM runs").fetchone()
            cp = con.execute("SELECT state_json FROM checkpoints ORDER BY version DESC LIMIT 1").fetchone()
            report = con.execute("SELECT content FROM messages WHERE role='assistant' ORDER BY created_at DESC LIMIT 1").fetchone()
        if not cp or not report:
            results.append({"case": folder.name, "status": "no_final_report", "source_status": row[1]})
            continue
        case_dir = output / folder.name
        case_dir.mkdir()
        context = RunContext.model_validate(json.loads(cp[0]))
        bundle = MultiAgentFactory.create_default_bundle(retrieval_provider=NoSearch())
        driver = PlanExecuteReportDriver(context, bundle.orchestrator)
        driver.apply_delivery_report(report[0])
        policy = CallBudget(BudgetLimits(max_llm_tokens=120000), BudgetUsage())
        token = set_call_budget(policy)
        check_id = "strict_" + row[0]
        set_current_run(check_id)
        started = time.monotonic()
        result = {"case": folder.name, "source_run_id": row[0], "source_status": row[1],
                  "scope": "read-only source; report-only verification and at most one local repair; no retrieval",
                  "model": settings.OPENAI_LLM_MODEL, "source_report_hash": hashlib.sha256(report[0].encode()).hexdigest()}
        try:
            await driver.verify_report_consistency(report[0])
            result["initial_check"] = driver.report_consistency_details()
            if result["initial_check"].get("method") == "deterministic":
                raise AssertionError("Structural precheck must not bypass strict semantic verification")
            if driver.report_consistency() is False:
                result["repair_attempted"] = True
                repaired = await driver.repair_report(["report_consistency"])
                result["repair_accepted"] = repaired
                result["repair_error"] = context.context_snapshot.get("report_repair_error")
                if repaired:
                    revised = await driver.report()
                    await driver.verify_report_consistency(revised)
            result["final_check"] = driver.report_consistency_details()
            result["status"] = result["final_check"]["status"]
            (case_dir / "report.md").write_text(driver.report_result.final_report, encoding="utf-8")
        except Exception as exc:
            result.update(status="error", error_type=type(exc).__name__)
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            result["usage"] = tracker.run_snapshot(check_id)
            policy.close()
            reset_call_budget(token)
            set_current_run(None)
        with (case_dir / "result.json").open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        results.append(result)
        print(folder.name + ": " + result["status"], flush=True)
    with (output / "summary.json").open("x", encoding="utf-8") as f:
        json.dump({"results": results, "passed": sum(r["status"] == "passed" for r in results),
                   "total": len(results), "search_calls": 0}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials, output = args.trials.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CACHE_DIR"] = str(output / "cache")
    os.environ["ARTIFACT_ROOT"] = str(output / "artifacts")
    os.chdir(output)
    asyncio.run(run(trials, output))
