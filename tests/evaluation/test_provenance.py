"""Negative controls: a provenance evaluator must actually reject bad links."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from deepresearch_agent.evaluation.provenance import audit_run
from deepresearch_agent.retrieval.web_utils import content_hash


@pytest.fixture
def sample(tmp_path):
    db = tmp_path / "app.db"
    root = tmp_path / "artifacts"
    material = root / "run_a/source.json"
    material.parent.mkdir(parents=True)
    body = "可核验的完整来源正文 " * 2000  # Longer than a 12k display preview.
    source = "https://example.org/source"
    material.write_text(json.dumps({"results": [{"url": source, "raw_content": body}]}, ensure_ascii=False), encoding="utf-8")
    metadata = {"source_id": source, "content_hash": content_hash(body), "extra": {
        "run_id": "run_a", "task_id": "task_a", "tool_call_id": "call_a", "artifact_path": "run_a/source.json"}}
    with sqlite3.connect(db) as con:
        con.executescript("""
        CREATE TABLE runs (run_id TEXT, source_mode TEXT, status TEXT);
        CREATE TABLE messages (run_id TEXT, role TEXT, content TEXT, created_at TEXT);
        CREATE TABLE evidence (evidence_id TEXT, run_id TEXT, invalidated_at TEXT, task_id TEXT, tool_call_id TEXT, source_mode TEXT, source_id TEXT, content_hash TEXT, metadata_json TEXT, artifact_id TEXT);
        CREATE TABLE tasks (task_id TEXT, run_id TEXT, plan_id TEXT);
        CREATE TABLE plans (plan_id TEXT, run_id TEXT);
        CREATE TABLE tool_calls (tool_call_id TEXT, run_id TEXT, task_id TEXT, status TEXT, source_mode TEXT, result_json TEXT);
        CREATE TABLE artifacts (artifact_id TEXT, run_id TEXT, relative_path TEXT, mime_type TEXT, created_at TEXT, sha256 TEXT);
        INSERT INTO runs VALUES ('run_a','web','completed');
        INSERT INTO messages VALUES ('run_a','assistant','结论[ev_a]，补充[ev_a]','0');
        INSERT INTO tasks VALUES ('task_a','run_a','plan_a');
        INSERT INTO plans VALUES ('plan_a','run_a');
        """)
        con.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)", ("ev_a", "run_a", None, "task_a", "call_a", "web", source, content_hash(body), json.dumps(metadata), None))
        con.execute("INSERT INTO tool_calls VALUES (?,?,?,?,?,?)", ("call_a", "run_a", "task_a", "completed", "web", json.dumps({"result_ids": ["ev_a"], "retrieval_results": [{"result_id": "ev_a", "metadata": metadata}]})))
    return db, root


def test_complete_chain_and_full_raw_hash(sample):
    before = hashlib.sha256(sample[0].read_bytes()).hexdigest()
    result = audit_run(*sample, "run_a")
    assert result["traceability_rate"] == 1
    assert result["passed_occurrences"] == 2
    assert result["unique_citations"] == 1
    assert hashlib.sha256(sample[0].read_bytes()).hexdigest() == before


@pytest.mark.parametrize("mutation,reason", [
    ("UPDATE evidence SET run_id='other'", "missing_foreign_or_invalid_evidence"),
    ("UPDATE evidence SET invalidated_at='now'", "missing_foreign_or_invalid_evidence"),
    ("UPDATE tool_calls SET task_id='other'", "missing_mismatched_or_incomplete_call"),
    ("UPDATE plans SET run_id='other'", "missing_or_foreign_plan"),
    ("DELETE FROM tasks", "missing_or_foreign_task"),
    ("UPDATE tool_calls SET result_json='{}'", "evidence_not_in_recorded_tool_result"),
    ("UPDATE tool_calls SET source_mode='graphrag'", "source_mode_mismatch"),
])
def test_broken_links_fail(sample, mutation, reason):
    with sqlite3.connect(sample[0]) as con:
        con.execute(mutation)
    result = audit_run(*sample, "run_a")
    assert result["traceability_rate"] == 0
    assert reason in result["checks"][0]["reasons"]


def test_missing_and_corrupted_material_fail(sample):
    path = sample[1] / "run_a/source.json"
    path.write_text('{"results":[]}', encoding="utf-8")
    assert audit_run(*sample, "run_a")["traceability_rate"] == 0
    path.unlink()
    assert audit_run(*sample, "run_a")["traceability_rate"] == 0


def test_path_escape_fails(sample):
    with sqlite3.connect(sample[0]) as con:
        metadata = json.loads(con.execute("SELECT metadata_json FROM evidence").fetchone()[0])
        metadata["extra"]["artifact_path"] = "../outside.json"
        con.execute("UPDATE evidence SET metadata_json=?", (json.dumps(metadata),))
    result = audit_run(*sample, "run_a")
    assert result["traceability_rate"] == 0


def test_no_citation_not_perfect_and_unknown_id_fails(sample):
    with sqlite3.connect(sample[0]) as con:
        con.execute("UPDATE messages SET content='没有引用'")
    assert audit_run(*sample, "run_a")["traceability_rate"] is None
    with sqlite3.connect(sample[0]) as con:
        con.execute("UPDATE messages SET content='未知引用[ev_missing]'")
    assert audit_run(*sample, "run_a")["traceability_rate"] == 0


def test_missing_database_is_not_created(tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises(ValueError):
        audit_run(missing, tmp_path, "run_a")
    assert not missing.exists()
