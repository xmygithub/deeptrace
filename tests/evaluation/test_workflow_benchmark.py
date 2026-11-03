"""Offline checks for benchmark denominators, isolation planning and resume."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("workflow_benchmark", ROOT / "scripts/evaluate_workflows.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_suite_split_and_smoke_order():
    suite = bench.load_suite(ROOT / "evals/system/workflow-50-v1.json")
    assert len(suite["cases"]) == 50
    assert len(bench.make_plan(suite, "dev", False)) == 30
    assert len(bench.make_plan(suite, "test", False)) == 70
    smoke = bench.make_plan(suite, "dev", True)
    assert len(smoke) == 12
    assert smoke[0]["workflow"] != smoke[2]["workflow"]
    assert all(t["case_id"] in {c["case_id"] for c in suite["cases"] if c["split"] == "dev"} for t in smoke)


def row(**changes):
    evaluation = dict(verified_completion=True, contract_results={"delivery_coverage": True}, status="completed",
                      latency_ms=1000, valid_citation_count=2, citation_count=2, source_leakage_detected=False,
                      tool_call_count=1, failed_tool_call_count=0)
    evaluation.update(changes)
    return dict(attempt_state="recorded", tokens=100, evaluation=evaluation)


def test_missing_usage_failures_and_pending_are_not_success_or_zero_cost():
    good, failed = row(), row(verified_completion=False, status="failed", contract_results={})
    failed["tokens"] = None
    result = bench.aggregate([good, failed, {"attempt_state": "infrastructure_error"}, {"attempt_state": "pending"}])
    assert result["verified_completion_rate"] == 1 / 3
    assert result["pending"] == 1
    assert result["tokens"]["mean"] == 100
    assert result["tokens"]["missing"] == 1
    assert result["tokens"]["per_verified_delivery"] is None
    assert result["delivery_coverage"]["counts"]["not_run"] == 1
    assert result["latency_by_status"]["failed"]["n"] == 1


def test_citations_are_micro_averaged_and_no_citations_are_not_perfect():
    result = bench.aggregate([row(), row(citation_count=8, valid_citation_count=0), row(citation_count=0, valid_citation_count=0)])
    assert result["citation_validity"] == .2
    assert result["reports_without_citations"] == 1
    assert bench.aggregate([row(citation_count=0, valid_citation_count=0)])["citation_validity"] is None


def test_failure_token_cost_is_included():
    result = bench.aggregate([row(), row(verified_completion=False, status="failed")])
    assert result["tokens"]["per_verified_delivery"] == 200


def test_incomplete_attempt_is_not_silently_counted_as_completed(tmp_path):
    plan = {"trials": [{"trial_id": "x", "workflow": "deep_research", "case_id": "c", "category": "a"}]}
    bench.write_new(tmp_path / "plan.json", plan)
    result = bench.summarize(tmp_path)
    assert result["by_workflow"]["deep_research"]["pending"] == 1
    assert result["by_workflow"]["deep_research"]["verified_completion_rate"] is None
    (tmp_path / "trials/x").mkdir(parents=True)
    result = bench.summarize(tmp_path)
    assert result["by_workflow"]["deep_research"]["attempted"] == 1
    assert result["by_workflow"]["deep_research"]["infrastructure_errors"] == 1


def test_check_errors_are_separate_from_content_failure():
    trial = row(contract_results={"delivery_coverage": False})
    trial["check_errors"] = ["delivery_coverage"]
    result = bench.aggregate([trial])
    assert result["delivery_coverage"]["counts"] == {"error_or_unknown": 1}
    assert result["delivery_coverage"]["pass_rate"] == 0
