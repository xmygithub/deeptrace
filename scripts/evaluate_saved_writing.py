"""Regenerate archived PER reports with fixed evidence/outline, never search."""
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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def restore_provenance(rows, artifacts):
    """Annotate in-memory copies only when the archived response matches the body."""
    from deepresearch_agent.retrieval.web_utils import normalize_content, normalize_url
    annotations = {}
    for result in rows:
        relative = result.metadata.extra.get("artifact_path")
        if not relative:
            continue
        path = (artifacts / relative).resolve()
        if not path.is_relative_to(artifacts.resolve()) or not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        for entry in raw.get("results", []):
            try:
                url = normalize_url(str(entry.get("url") or ""))
            except ValueError:
                continue
            if url != result.metadata.url:
                continue
            page = entry.get("raw_content")
            has_page = isinstance(page, str) and bool(page.strip())
            body = normalize_content(page if has_page else str(entry.get("content") or ""))
            if body[:12000] != result.evidence:
                continue
            result.metadata.extra.update(content_kind="page_text" if has_page else "search_summary",
                raw_content_available=has_page, content_truncated=len(body) > 12000)
            annotations[result.result_id] = {"content_kind": result.metadata.extra["content_kind"],
                "artifact": relative, "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            break
    return annotations


async def evaluate(trials, output, case_ids, token_limit):
    from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
    from deepresearch_agent.agents.multi_agent.integration.multi_agent_factory import MultiAgentFactory
    from deepresearch_agent.agents.multi_agent.reporter.base_reporter import BaseReporter
    from deepresearch_agent.agents.multi_agent.reporter.outline_builder import ReportOutline
    from deepresearch_agent.agents.multi_agent.reporter.section_writer import SectionWriter
    from deepresearch_agent.config import settings
    from deepresearch_agent.harness.budgets import BudgetLimits, BudgetUsage
    from deepresearch_agent.harness.call_budget import CallBudget, set_call_budget, reset_call_budget
    from deepresearch_agent.harness.contracts import ContractEvaluator, SourceMode
    from deepresearch_agent.harness.delivery import DeliveryService
    from deepresearch_agent.harness.run_context import RunContext
    from deepresearch_agent.harness.workflow import PlanExecuteReportDriver
    from deepresearch_agent.models.prefix_cache import tracker, set_current_run

    class NoSearch:
        mode, provider_name = SourceMode.WEB, "writing-replay-no-search"
        async def search(self, *args, **kwargs):
            raise AssertionError("Writing replay must not search")

    results = []
    for folder in sorted(trials.iterdir()):
        if folder.name.split("--")[0] not in case_ids or "--plan_execute_report--" not in folder.name:
            continue
        database = folder / "run/app.db"
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as con:
            row = con.execute("SELECT state_json FROM checkpoints ORDER BY version DESC LIMIT 1").fetchone()
            payload = json.loads(row[0])
            con.row_factory = sqlite3.Row
            ledger = [dict(r) for r in con.execute("SELECT * FROM evidence WHERE invalidated_at IS NULL")]
        target = output / folder.name
        target.mkdir()
        original = payload["workflow_state"]["report_result"]
        (target / "original-report.md").write_text(original["final_report"], encoding="utf-8")
        outline = ReportOutline.model_validate(original["outline"])

        class FrozenOutline:
            def build_outline(self, **kwargs):
                return outline.model_copy(deep=True)

        class RecordingWriter(SectionWriter):
            def _call_model(self, messages, options):
                n = self.model_calls + 1
                request = [{"role": m.type, "content": m.content} for m in messages]
                (target / f"section-input-{n}.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
                response = super()._call_model(messages, options)
                (target / f"section-output-{n}.txt").write_text(str(response.content), encoding="utf-8")
                return response

        reporter = BaseReporter(outline_builder=FrozenOutline(), section_writer=RecordingWriter())
        bundle = MultiAgentFactory.create_default_bundle(reporter=reporter, retrieval_provider=NoSearch())
        context = RunContext.model_validate(payload)
        driver = PlanExecuteReportDriver(context, bundle.orchestrator)
        before = {r.result_id: r.evidence for _, _, _, r in driver.evidence_results()}
        annotations = {}
        for record in driver.state.execution_records:
            record.evidence = [r if isinstance(r, RetrievalResult) else RetrievalResult.from_dict(r) for r in record.evidence]
            annotations.update(restore_provenance(record.evidence, folder / "run/artifacts"))
        after = {r.result_id: r.evidence for _, _, _, r in driver.evidence_results()}
        if digest(before) != digest(after):
            raise AssertionError("Evidence IDs/text must remain fixed")
        policy = CallBudget(BudgetLimits(max_llm_tokens=token_limit), BudgetUsage())
        token = set_call_budget(policy)
        check_id = "writing_" + context.run_id
        set_current_run(check_id)
        started = time.monotonic()
        result = {"case": folder.name, "source_run_id": context.run_id,
            "scope": "fixed archived evidence and outline; in-memory provenance restoration; no new retrieval or automatic content repair",
            "model": settings.OPENAI_LLM_MODEL, "evidence_text_hash": digest(before),
            "checkpoint_sha256": hashlib.sha256(row[0].encode()).hexdigest(), "source_annotations": annotations,
            "frozen_outline": original["outline"], "token_limit": token_limit}
        try:
            generated = await asyncio.to_thread(reporter.generate_report, driver.state,
                report_type=context.config_snapshot.get("report_type", "brief"))
            driver.report_result = generated
            driver._consistency_record = {}
            report = generated.final_report
            result["routing_unchanged"] = generated.evidence_routing == original["evidence_routing"]
            result["routing_membership_unchanged"] = (
                {k: set(v) for k, v in generated.evidence_routing.items()}
                == {k: set(v) for k, v in original["evidence_routing"].items()})
            (target / "report.md").write_text(report, encoding="utf-8")
            (target / "report-result.json").write_text(generated.model_dump_json(indent=2), encoding="utf-8")
            await driver.verify_report_consistency(report)
            result["consistency"] = driver.report_consistency_details()
            requirements = context.context_snapshot.get("delivery_requirements")
            coverage = await DeliveryService().review(requirements, report) if requirements else None
            result["delivery_review"] = coverage
            verdict = await ContractEvaluator().evaluate(run_id=context.run_id, source_mode=context.source_mode,
                report=report, evidence=ledger, min_evidence=context.config_snapshot.get("min_evidence", 1),
                required_sections=context.config_snapshot.get("required_sections", []),
                consistency_passed=driver.report_consistency(), consistency_details=result["consistency"],
                evidence_card_coverage=driver.evidence_card_coverage(), delivery_coverage=coverage)
            result.update(status="passed" if verdict.passed else "verification_failed", failures=verdict.failures,
                checks=[check.model_dump(mode="json") for check in verdict.checks], report_chars=len(report))
        except Exception as exc:
            result.update(status="error", error_type=type(exc).__name__)
            if isinstance(exc, ValueError):
                result["error"] = str(exc)
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            result["usage"] = tracker.run_snapshot(check_id)
            policy.close()
            reset_call_budget(token)
            set_current_run(None)
        (target / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(result)
        print(folder.name + ": " + result["status"], flush=True)
    (output / "summary.json").write_text(json.dumps({"results": results, "retrieval_calls": 0,
        "passed": sum(r["status"] == "passed" for r in results), "total": len(results)}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--token-limit", type=int, default=120000)
    args = parser.parse_args()
    if args.token_limit < 1:
        parser.error("token-limit must be positive")
    trials, output = args.trials.resolve(), args.output.resolve()
    available = {p.name.split("--")[0] for p in trials.iterdir() if "--plan_execute_report--" in p.name}
    if set(args.case_id) - available:
        parser.error("Unknown case ID")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CACHE_DIR"] = str(output / "cache")
    os.environ["ARTIFACT_ROOT"] = str(output / "artifacts")
    source_files = ["scripts/evaluate_saved_writing.py", "src/deepresearch_agent/config/prompts/reporter_prompts.py",
        "src/deepresearch_agent/agents/multi_agent/reporter/section_writer.py", "src/deepresearch_agent/agents/multi_agent/reporter/evidence_cards.py",
        "src/deepresearch_agent/agents/multi_agent/reporter/base_reporter.py", "src/deepresearch_agent/agents/multi_agent/reporter/consistency_checker.py",
        "src/deepresearch_agent/retrieval/evidence_context.py", "src/deepresearch_agent/retrieval/tavily_provider.py"]
    (output / "source-hashes.json").write_text(json.dumps({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source_files}, indent=2), encoding="utf-8")
    os.chdir(output)
    asyncio.run(evaluate(trials, output, set(args.case_id), args.token_limit))
