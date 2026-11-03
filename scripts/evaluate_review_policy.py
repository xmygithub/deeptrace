"""Review fixed archived reports plus labelled policy probes; never search/write source DBs."""
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


def calibration_cases():
    def source(text, truncated=False):
        return [{"evidence_id": "ev_a", "title": "校准材料", "text": text,
                 "content_kind": "document_text", "truncated": truncated}]
    return [
        {"id": "supported_fact", "expected": "passed", "report": "# 报告\n\n该窗口保留最近八轮对话。 [ev_a]",
         "sources": source("该窗口保留最近八轮对话。")},
        {"id": "paragraph_citation", "expected": "passed", "report": "# 报告\n\n该窗口保留最近八轮对话。更早的对话不再进入模型输入。 [ev_a]",
         "sources": source("该窗口保留最近八轮对话，更早的对话不再进入模型输入。")},
        {"id": "labelled_advice_and_symbol", "expected": "passed", "report": "# 报告\n\n滑动窗口保留最近K轮对话。基于此机制，建议在预算有限时使用窗口，并另行保存关键约束；这是设计建议，不是实测收益。 [ev_a]",
         "sources": source("滑动窗口保留最近N轮对话，早期内容被移出上下文。")},
        {"id": "source_contradiction", "expected": "failed", "report": "# 报告\n\n配置A在断电时绝不会丢失消息。 [ev_a]",
         "sources": source("配置A在断电时可能丢失消息。")},
        {"id": "internal_contradiction", "expected": "failed", "report": "# 报告\n\n配置A在断电时绝不会丢失消息。 [ev_a]\n\n配置A在断电时仍然可能丢失消息。 [ev_a]",
         "sources": source("配置A在断电时可能丢失消息。")},
        {"id": "wrong_source_subject", "expected": "failed", "report": "# 报告\n\n该材料表明RabbitMQ能够原子扣减团购商品库存。 [ev_a]",
         "sources": source("滑动窗口保留最近八轮对话，早期内容被移出上下文。")},
        {"id": "truncated_uncertainty", "expected": "passed", "report": "# 报告\n\n当前片段仅说明配置A支持持久化，尚不足以确认断电时是否丢失消息。 [ev_a]",
         "sources": source("配置A支持持久化。", truncated=True)},
    ]


async def evaluate(args, output):
    from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
    from deepresearch_agent.agents.multi_agent.integration.multi_agent_factory import MultiAgentFactory
    from deepresearch_agent.agents.multi_agent.reporter.consistency_checker import ConsistencyChecker
    from deepresearch_agent.agents.multi_agent.reporter.review_policy import REVIEW_POLICY_VERSION
    from deepresearch_agent.harness.budgets import BudgetLimits, BudgetUsage
    from deepresearch_agent.harness.call_budget import CallBudget, set_call_budget, reset_call_budget
    from deepresearch_agent.harness.contracts import ContractEvaluator, SourceMode
    from deepresearch_agent.harness.run_context import RunContext
    from deepresearch_agent.harness.workflow import PlanExecuteReportDriver
    from deepresearch_agent.models.prefix_cache import tracker, set_current_run
    from scripts.evaluate_saved_writing import restore_provenance

    class NoSearch:
        mode, provider_name = SourceMode.WEB, "review-policy-no-search"
        async def search(self, *args, **kwargs):
            raise AssertionError("Policy review must not retrieve")

    results = []
    jobs = [] if args.skip_calibration else [("calibration", case["id"], case) for case in calibration_cases()]
    jobs += [("archived_report", path.name, path) for path in sorted(args.reports.iterdir()) if (path / "report.md").is_file()]
    for kind, name, data in jobs:
        case_dir = output / name
        case_dir.mkdir()
        result = {"kind": kind, "case": name, "policy_version": REVIEW_POLICY_VERSION}
        policy = CallBudget(BudgetLimits(max_llm_tokens=120000), BudgetUsage())
        token = set_call_budget(policy)
        run_key = "review_policy_" + name
        set_current_run(run_key)
        started = time.monotonic()
        class RecordingChecker(ConsistencyChecker):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def _invoke_llm(self, prompt, *, max_tokens):
                self.calls += 1
                response = super()._invoke_llm(prompt, max_tokens=max_tokens)
                (case_dir / f"review-output-{self.calls}.txt").write_text(response, encoding="utf-8")
                (case_dir / f"review-input-{self.calls}.txt").write_text(prompt, encoding="utf-8")
                return response
        try:
            if kind == "calibration":
                result["expected"] = data["expected"]
                verdict = await asyncio.to_thread(RecordingChecker().check, data["report"], json.dumps(data["sources"], ensure_ascii=False))
                details = verdict.model_dump(mode="json", exclude={"raw_response"})
                result.update(status="passed" if verdict.is_consistent else "failed", review=details)
                result["matched"] = result["status"] == result["expected"]
            else:
                report = (data / "report.md").read_text(encoding="utf-8")
                db = args.trials / name / "run/app.db"
                with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as con:
                    saved = con.execute("SELECT state_json FROM checkpoints ORDER BY version DESC LIMIT 1").fetchone()[0]
                    con.row_factory = sqlite3.Row
                    evidence = [dict(r) for r in con.execute("SELECT * FROM evidence WHERE invalidated_at IS NULL")]
                context = RunContext.model_validate(json.loads(saved))
                bundle = MultiAgentFactory.create_default_bundle(retrieval_provider=NoSearch())
                bundle.reporter._consistency_checker = RecordingChecker()
                driver = PlanExecuteReportDriver(context, bundle.orchestrator)
                driver.apply_delivery_report(report)
                driver._consistency_record = {}
                for record in driver.state.execution_records:
                    record.evidence = [r if isinstance(r, RetrievalResult) else RetrievalResult.from_dict(r) for r in record.evidence]
                    restore_provenance(record.evidence, db.parent / "artifacts")
                # Exact same report: reuse only a delivery verdict bound to its hash.
                prior = json.loads((data / "result.json").read_text(encoding="utf-8"))
                coverage = prior.get("delivery_review")
                report_hash = hashlib.sha256(report.encode()).hexdigest()
                if coverage is not None and coverage.get("report_hash") != report_hash:
                    raise ValueError("Archived delivery review belongs to another report")
                await driver.verify_report_consistency(report)
                details = driver.report_consistency_details()
                verdict = await ContractEvaluator().evaluate(run_id=context.run_id, source_mode=context.source_mode,
                    report=report, evidence=evidence, min_evidence=context.config_snapshot.get("min_evidence", 1),
                    required_sections=context.config_snapshot.get("required_sections", []),
                    consistency_passed=driver.report_consistency(), consistency_details=details,
                    evidence_card_coverage=driver.evidence_card_coverage(), delivery_coverage=coverage)
                result.update(status="passed" if verdict.passed else "failed", review=details,
                    failures=verdict.failures, source_report_hash=report_hash,
                    prior_status=prior["status"], coverage_reused=coverage is not None,
                    checks=[check.model_dump(mode="json") for check in verdict.checks])
                with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as con:
                    after = con.execute("SELECT state_json FROM checkpoints ORDER BY version DESC LIMIT 1").fetchone()[0]
                result["source_checkpoint_unchanged"] = saved == after
                result["source_report_unchanged"] = report == (data / "report.md").read_text(encoding="utf-8")
        except Exception as exc:
            result.update(status="error", error_type=type(exc).__name__)
            if kind == "calibration":
                result["matched"] = False
            if isinstance(exc, ValueError):
                result["error"] = str(exc)
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            result["usage"] = tracker.run_snapshot(run_key)
            policy.close()
            reset_call_budget(token)
            set_current_run(None)
        (case_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)
        print(kind + "/" + name + ": " + result["status"], flush=True)
    (output / "summary.json").write_text(json.dumps({"policy_version": REVIEW_POLICY_VERSION,
        "scope": "changed evaluator policy; fixed reports and labelled synthetic probes; not research quality improvement",
        "results": results, "retrieval_calls": 0, "report_generation_calls": 0}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--skip-calibration", action="store_true", help="Review only supplied reports without repeating policy probes")
    args = p.parse_args()
    args.trials, args.reports, args.output = args.trials.resolve(), args.reports.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "calibration-cases.json").write_text(json.dumps(calibration_cases(), ensure_ascii=False, indent=2), encoding="utf-8")
    source_names = ["scripts/evaluate_review_policy.py", "src/deepresearch_agent/agents/multi_agent/reporter/review_policy.py",
        "src/deepresearch_agent/agents/multi_agent/reporter/consistency_checker.py", "src/deepresearch_agent/config/prompts/reporter_prompts.py",
        "src/deepresearch_agent/harness/workflow.py", "src/deepresearch_agent/harness/report_repair.py",
        "src/deepresearch_agent/harness/verifiers/deterministic.py", "src/deepresearch_agent/retrieval/evidence_context.py"]
    (args.output / "source-hashes.json").write_text(json.dumps({n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names}, indent=2), encoding="utf-8")
    os.environ["CACHE_DIR"] = str(args.output / "cache")
    os.environ["ARTIFACT_ROOT"] = str(args.output / "artifacts")
    os.chdir(args.output)
    asyncio.run(evaluate(args, args.output))
