"""Quality labels must be auditable; runtime success cannot stand in for truth."""

import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from deepresearch_agent.evaluation.research import (
    claim_candidates, content_hash, evaluate_package, load_cases, make_bundle,
    read_run, render_summary, summarize_packages, validate_snapshot, write_package,
)


@pytest.fixture
def case():
    return {
        "case_id": "test_001", "question": "Explain retry safety",
        "criteria": [{"criterion_id": "idempotency", "description": "Explain duplicate effects"}],
    }


@pytest.fixture
def run():
    raw = "A retry can repeat an external operation unless it is idempotent."
    return {
        "run_id": "run_test", "question": "Explain retry safety", "status": "completed",
        "source_mode": "web", "workflow_mode": "plan_execute_report",
        "report": "# Answer\n\nA retry can repeat an external operation unless it is idempotent. [ev_test]",
        "model": {"llm_model": "test-model", "temperature": 0},
        "metrics": {"verified_completion": True, "llm_tokens": 100, "latency_seconds": 2.0},
        "evidence": [{"evidence_id": "ev_test", "source_id": "https://example.org/retry", "content_hash": content_hash(raw), "content": {"kind": "raw_snapshot", "text": raw}}],
    }


def reviewed(case, run):
    bundle, review = make_bundle(case, run)
    review["reviewer"] = "fixture reviewer"
    review["reviewed_at"] = "2026-10-23T10:00:00+08:00"
    review["coverage"][0].update(verdict="met", reason="Explains duplicate effects", report_quote=run["report"].split("\n\n")[1])
    review["claims"][0].update(verdict="supported", reason="Same constraint in source", evidence_quotes=[{"evidence_id": "ev_test", "quote": run["evidence"][0]["content"]["text"]}])
    for item in review["quality"].values():
        item.update(score=4, reason="Fixture assessment only", report_quote="A retry can repeat an external operation")
    return bundle, review


def test_pending_review_never_invents_quality_scores(case, run):
    bundle, review = make_bundle(case, run)
    result = evaluate_package(bundle, review)
    assert result["metrics"]["verified_completion"] is True
    assert result["citation_id_validity"] == 1.0
    assert result["coverage_rate"] is None
    assert result["claim_support_rate_resolved"] is None
    assert result["report_quality_score"] is None


def test_auditable_human_labels_produce_scores(case, run):
    result = evaluate_package(*reviewed(case, run))
    assert result["coverage_rate"] == 1
    assert result["claim_support_rate_resolved"] == 1
    assert result["claim_selection_scope"] == "sample"
    assert result["report_quality_score"] == 4


@pytest.mark.parametrize("verdict", ["unsupported", "contradicted", "unverifiable"])
def test_valid_citation_can_still_be_unsupported_or_unknown(case, run, verdict):
    bundle, review = reviewed(case, run)
    review["claims"][0]["verdict"] = verdict
    result = evaluate_package(bundle, review)
    assert result["citation_id_validity"] == 1
    assert result["claim_counts"][verdict] == 1
    assert result["claim_support_rate_resolved"] == (None if verdict == "unverifiable" else 0)
    assert result["claim_verifiability"] == (0 if verdict == "unverifiable" else 1)


@pytest.mark.parametrize("change", ["missing_source", "summary_only", "wrong_quote", "uncited_source"])
def test_support_requires_cited_raw_source_and_exact_quote(case, run, change):
    if change == "missing_source":
        run["evidence"] = []
    if change == "summary_only":
        run["evidence"][0]["content"] = {"kind": "summary_only", "text": ""}
    bundle, review = make_bundle(case, run)
    review.update(reviewer="reviewer", reviewed_at="2026-10-23T10:00:00Z")
    review["claims"][0].update(verdict="supported", reason="assertion", evidence_quotes=[{"evidence_id": "ev_test", "quote": "A retry"}])
    if change == "wrong_quote":
        review["claims"][0]["evidence_quotes"][0]["quote"] = "not in original page"
    if change == "uncited_source":
        review["claims"][0]["evidence_ids"] = ["ev_other"]
    with pytest.raises(ValueError):
        evaluate_package(bundle, review)


def test_modified_bundle_or_review_target_is_rejected(case, run):
    bundle, review = make_bundle(case, run)
    changed = copy.deepcopy(bundle)
    changed["run"]["report"] += " changed"
    with pytest.raises(ValueError, match="hash"):
        evaluate_package(changed, review)
    review["bundle_hash"] = "different"
    with pytest.raises(ValueError, match="hash"):
        evaluate_package(bundle, review)


@pytest.mark.parametrize("field", ["claim_text", "reviewer", "reviewed_at"])
def test_judged_claim_requires_atomic_claim_and_reviewer_provenance(case, run, field):
    bundle, review = reviewed(case, run)
    if field == "claim_text":
        review["claims"][0][field] = ""
    else:
        review[field] = ""
    with pytest.raises(ValueError):
        evaluate_package(bundle, review)


def test_failed_run_is_retained_without_report_grade(case, run, tmp_path):
    run.update(status="budget_exhausted", report="")
    run["metrics"]["verified_completion"] = False
    bundle, review = make_bundle(case, run)
    write_package(tmp_path / "failed", bundle, review)
    result = summarize_packages([tmp_path / "failed"])
    assert result["run_count"] == 1
    assert result["verified_completion_rate"] == 0
    assert result["reports_available"] == 0
    assert result["mean_report_quality_score"] is None
    assert result["failed_or_incomplete_runs"] == 1
    assert "budget_exhausted" in render_summary(result)
    review.update(reviewer="reviewer", reviewed_at="2026-10-23T10:00:00Z")
    review["coverage"][0].update(verdict="not_met", reason="No report")
    with pytest.raises(ValueError, match="missing final report"):
        evaluate_package(bundle, review)


def test_packages_are_never_overwritten_or_double_counted(case, run, tmp_path):
    bundle, review = make_bundle(case, run)
    write_package(tmp_path / "package", bundle, review)
    with pytest.raises(FileExistsError):
        write_package(tmp_path / "package", bundle, review)
    with pytest.raises(ValueError, match="Duplicate Run"):
        summarize_packages([tmp_path / "package", tmp_path / "package"])


def test_question_mismatch_keeps_quality_out_of_comparisons(case, run):
    case["question"] = "A different task"
    bundle, review = reviewed(case, run)
    result = evaluate_package(bundle, review)
    assert not result["question_matches_case"]
    assert result["coverage_rate"] is None
    assert result["report_quality_score"] is None


def test_sampling_spans_report_and_does_not_drop_small_samples():
    report = "\n\n".join(f"Paragraph {i} has a sufficiently long factual assertion. [ev_{i}]" for i in range(20))
    claims, total = claim_candidates(report, 3)
    assert total == 20 and len(claims) == 3
    assert "Paragraph 0" in claims[0]["report_quote"]
    assert "Paragraph 19" in claims[-1]["report_quote"]
    assert len(claim_candidates(report, 100)[0]) == 20


def test_candidate_quotes_remain_exact_with_headings_and_code():
    report = "First paragraph contains a long factual statement.\n# Heading\nSecond paragraph contains another long factual statement.\n```python\nprint('not a claim')\n```\nThird paragraph contains a final long factual statement."
    claims, _ = claim_candidates(report)
    assert len(claims) == 3
    assert all(item["report_quote"] in report for item in claims)


def test_missing_measurements_are_not_zero(case, run, tmp_path):
    run["metrics"] = {}
    write_package(tmp_path / "unknown", *make_bundle(case, run))
    summary = summarize_packages([tmp_path / "unknown"])
    assert summary["completion_measured_runs"] == 0
    assert summary["verified_completion_rate"] is None
    assert summary["runs"][0]["metrics"].get("llm_tokens") is None


def test_pair_delta_retains_confounds(case, run, tmp_path):
    write_package(tmp_path / "agent", *make_bundle(case, run))
    run["run_id"] = "run_baseline"
    run["metrics"]["llm_tokens"] = 60
    write_package(tmp_path / "baseline", *make_bundle(case, run, arm="baseline"))
    pair = summarize_packages([tmp_path / "agent", tmp_path / "baseline"])["comparisons"][0]
    assert pair["token_delta"] == 40
    assert not pair["controlled_comparison"]
    assert "Report quality review incomplete" in pair["limitations"]


def test_case_file_has_three_real_pilot_tasks():
    cases = load_cases(Path(__file__).resolve().parents[2] / "evals/research/cases.json")
    assert len(cases) == 3
    assert cases["rag-multihop-001"]["criteria"]


def test_bad_raw_hash_and_duplicate_evidence_rejected(run):
    bad = copy.deepcopy(run)
    bad["evidence"][0]["content"]["text"] = "changed snapshot"
    with pytest.raises(ValueError, match="hash"):
        validate_snapshot(bad)
    run["evidence"].append(copy.deepcopy(run["evidence"][0]))
    with pytest.raises(ValueError, match="Duplicate evidence"):
        validate_snapshot(run)


def test_redacted_snapshot_retains_provenance_and_verifiable_export_hash(case, run):
    secret = "sk-" + "example" * 4
    raw = "Original text contains credential " + secret
    run["evidence"][0]["content"]["text"] = raw
    run["evidence"][0]["content_hash"] = content_hash(raw)
    bundle, review = make_bundle(case, run)
    assert secret not in json.dumps(bundle)
    assert bundle["run"]["evidence"][0]["original_content_hash"] == content_hash(raw)
    validate_snapshot(bundle["run"])
    evaluate_package(bundle, review)


def test_readonly_export_matches_artifact_and_cannot_escape_run(tmp_path):
    db = tmp_path / "app.db"
    connection = sqlite3.connect(db)
    connection.executescript("""
        CREATE TABLE runs (run_id TEXT, trigger_message_id TEXT, status TEXT, source_mode TEXT, workflow_mode TEXT, usage_json TEXT, model_snapshot_json TEXT, budget_json TEXT, started_at TEXT, completed_at TEXT, error_code TEXT);
        CREATE TABLE messages (message_id TEXT, run_id TEXT, role TEXT, content TEXT, created_at TEXT);
        CREATE TABLE evidence (run_id TEXT, evidence_id TEXT, source_id TEXT, title TEXT, summary TEXT, source_mode TEXT, content_hash TEXT, metadata_json TEXT, invalidated_at TEXT);
        CREATE TABLE contract_checks (run_id TEXT, kind TEXT, required INTEGER, passed INTEGER);
        CREATE TABLE tool_calls (run_id TEXT, status TEXT);
    """)
    raw_text = "The original saved page explains retry semantics."
    root = tmp_path / "artifacts"
    saved = root / "run_saved/tavily/call.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"results": [{"url": "https://example.org/retry", "raw_content": raw_text}]}), encoding="utf-8")
    connection.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", ("run_saved", "msg_q", "completed", "web", "deep_research", '{"usage":{"llm_tokens":123}}', '{"llm_model":"test","api_key":"never-export"}', '{}', "2026-10-23T00:00:00Z", "2026-10-23T00:00:02Z", None))
    connection.executemany("INSERT INTO messages VALUES (?,?,?,?,?)", [("msg_q", None, "user", "Explain retry", "1"), ("msg_a", "run_saved", "assistant", "Retry explanation [ev_saved]", "2")])
    connection.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?)", ("run_saved", "ev_saved", "https://example.org/retry", "Retry", "Preview", "web", content_hash(raw_text), json.dumps({"extra": {"artifact_path": "run_saved/tavily/call.json"}}), None))
    connection.execute("INSERT INTO contract_checks VALUES (?,?,?,?)", ("run_saved", "citation_integrity", 1, 1))
    connection.commit()
    connection.close()
    before = db.read_bytes()
    exported = read_run(db, root, "run_saved")
    assert db.read_bytes() == before
    assert exported["evidence"][0]["content"]["kind"] == "raw_snapshot"
    assert exported["metrics"]["llm_tokens"] == 123
    assert exported["metrics"]["latency_seconds"] == 2
    assert "api_key" not in json.dumps(exported)
    connection = sqlite3.connect(db)
    connection.execute("UPDATE evidence SET metadata_json=?", (json.dumps({"extra": {"artifact_path": "../escape.json"}}),))
    connection.commit()
    connection.close()
    assert read_run(db, root, "run_saved")["evidence"][0]["content"]["kind"] == "summary_only"


def test_missing_database_does_not_create_file(tmp_path):
    db = tmp_path / "absent.db"
    with pytest.raises(sqlite3.OperationalError):
        read_run(db, tmp_path, "run_absent")
    assert not db.exists()


def test_cli_baseline_import_summary_and_no_overwrite(case, run, tmp_path):
    root = Path(__file__).resolve().parents[2]
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"schema_version": 1, "cases": [case]}), encoding="utf-8")
    snapshot = tmp_path / "baseline.json"
    snapshot.write_text(json.dumps(run), encoding="utf-8")
    package = tmp_path / "baseline"
    args = [sys.executable, str(root / "scripts/evaluate_research.py"), "import", "--input", str(snapshot), "--cases", str(cases), "--case-id", case["case_id"], "--arm", "baseline", "--output", str(package)]
    imported = subprocess.run(args, capture_output=True, text=True, timeout=15)
    assert imported.returncode == 0, imported.stderr
    assert (package / "worksheet.md").is_file()
    original = (package / "review.json").read_bytes()
    assert subprocess.run(args, capture_output=True, timeout=15).returncode == 2
    assert (package / "review.json").read_bytes() == original
    output = tmp_path / "summary"
    summarized = subprocess.run([sys.executable, str(root / "scripts/evaluate_research.py"), "summarize", "--package", str(package), "--output", str(output)], capture_output=True, text=True, timeout=15)
    assert summarized.returncode == 0, summarized.stderr
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["run_count"] == 1 and result["quality_graded_runs"] == 0


def test_assistant_review_is_separate_and_never_counted_as_human(case, run, tmp_path):
    bundle, human = reviewed(case, run)
    write_package(tmp_path / "package", bundle, human)
    assistant = copy.deepcopy(human)
    assistant["reviewer_kind"] = "assistant"
    (tmp_path / "package/review.assistant.json").write_text(json.dumps(assistant), encoding="utf-8")
    result = summarize_packages([tmp_path / "package"], review_filename="review.assistant.json")
    assert result["quality_by_reviewer_kind"]["human"]["graded_runs"] == 0
    assert result["quality_by_reviewer_kind"]["assistant"]["graded_runs"] == 1
    assert result["runs"][0]["reviewer_kind"] == "assistant"
    assert "AI" in render_summary(result)
    with pytest.raises(ValueError, match="basename"):
        summarize_packages([tmp_path / "package"], review_filename="../review.json")


def test_unsupported_quotes_are_validated_too(case, run):
    bundle, review = reviewed(case, run)
    review["claims"][0]["verdict"] = "unsupported"
    review["claims"][0]["evidence_quotes"][0]["quote"] = "fabricated quote"
    with pytest.raises(ValueError, match="saved source text"):
        evaluate_package(bundle, review)


def test_human_and_assistant_scores_are_not_pooled(case, run, tmp_path):
    bundle, human = reviewed(case, run)
    write_package(tmp_path / "human", bundle, human)
    run["run_id"] = "run_ai"
    bundle, assistant = reviewed(case, run)
    assistant["reviewer_kind"] = "assistant"
    for item in assistant["quality"].values():
        item["score"] = 2
    write_package(tmp_path / "assistant", bundle, assistant)
    summary = summarize_packages([tmp_path / "human", tmp_path / "assistant"])
    assert summary["mean_report_quality_score"] is None
    assert summary["quality_by_reviewer_kind"]["human"]["mean_report_quality_score"] == 4
    assert summary["quality_by_reviewer_kind"]["assistant"]["mean_report_quality_score"] == 2
