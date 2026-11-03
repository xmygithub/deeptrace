"""Read-only, structural citation provenance audit. No semantic/model scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3

from deepresearch_agent.retrieval.web_utils import content_hash, normalize_url

CITATION = re.compile(r"\[\^?(ev_[A-Za-z0-9_-]+)\]")


def _json(value):
    try:
        return json.loads(value or "{}")
    except (ValueError, TypeError):
        return {}


def _file(root, relative, run_id):
    path = (root / relative).resolve()
    path.relative_to(root)
    # The material must also be owned by this run's artifact directory.
    path.relative_to((root / run_id).resolve())
    return path.read_bytes()


def audit_run(database: Path, artifact_root: Path, run_id: str) -> dict:
    database, artifact_root = Path(database).resolve(), Path(artifact_root).resolve()
    if not database.is_file():
        raise ValueError("Evaluation database does not exist")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        def one(table, column, value):
            # Table/column names are constants controlled by this module.
            row = con.execute(f"SELECT * FROM {table} WHERE {column}=?", (value,)).fetchone()
            return dict(row) if row else None
        run = one("runs", "run_id", run_id)
        if not run:
            raise ValueError("Run not found")
        report = con.execute("SELECT content FROM messages WHERE run_id=? AND role='assistant' ORDER BY created_at LIMIT 1", (run_id,)).fetchone()
        report_source = "assistant_message"
        if report is None:
            artifact = con.execute("SELECT * FROM artifacts WHERE run_id=? AND mime_type LIKE 'text/markdown%' ORDER BY created_at DESC LIMIT 1", (run_id,)).fetchone()
            report_source = "saved_draft"
            if artifact:
                raw = _file(artifact_root, artifact["relative_path"], run_id)
                if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
                    raise ValueError("Report artifact checksum mismatch")
                text = raw.decode("utf-8")
            else:
                text, report_source = "", "no_report"
        else:
            text = report["content"]
        citations = CITATION.findall(text)
        checks = []
        for eid in dict.fromkeys(citations):
            reasons = []
            evidence = one("evidence", "evidence_id", eid)
            if not evidence or evidence["run_id"] != run_id or evidence["invalidated_at"]:
                checks.append({"evidence_id": eid, "passed": False, "reasons": ["missing_foreign_or_invalid_evidence"]})
                continue
            task = one("tasks", "task_id", evidence["task_id"])
            call = one("tool_calls", "tool_call_id", evidence["tool_call_id"])
            if not task or task["run_id"] != run_id:
                reasons.append("missing_or_foreign_task")
            elif not (plan := one("plans", "plan_id", task["plan_id"])) or plan["run_id"] != run_id:
                reasons.append("missing_or_foreign_plan")
            if not call or call["run_id"] != run_id or call["task_id"] != evidence["task_id"] or call["status"] != "completed":
                reasons.append("missing_mismatched_or_incomplete_call")
            if evidence["source_mode"] != run["source_mode"] or (call and call["source_mode"] != run["source_mode"]):
                reasons.append("source_mode_mismatch")
            payload = _json(call["result_json"]) if call else {}
            result = next((r for r in payload.get("retrieval_results", []) if r.get("result_id") == eid), None)
            if eid not in payload.get("result_ids", []) or not result:
                reasons.append("evidence_not_in_recorded_tool_result")
            elif (result.get("metadata", {}).get("source_id") != evidence["source_id"]
                  or result.get("metadata", {}).get("content_hash") != evidence["content_hash"]):
                reasons.append("tool_result_identity_mismatch")
            metadata = _json(evidence["metadata_json"])
            extra = metadata.get("extra", {})
            if any(extra.get(key) != value for key, value in {
                "run_id": run_id, "task_id": evidence["task_id"], "tool_call_id": evidence["tool_call_id"]}.items()):
                reasons.append("evidence_metadata_identity_mismatch")
            artifact = one("artifacts", "artifact_id", evidence["artifact_id"]) if evidence["artifact_id"] else None
            relative = artifact["relative_path"] if artifact else extra.get("artifact_path")
            try:
                if not relative:
                    raise ValueError("No raw material")
                if artifact and artifact["run_id"] != run_id:
                    raise ValueError("Foreign artifact")
                raw = _file(artifact_root, relative, run_id)
                if artifact and hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
                    reasons.append("artifact_checksum_mismatch")
                # Tavily saves one complete JSON response per call, whose
                # checksum differs from each page's content_hash. Verify page
                # identity and the FULL normalized body, not the 12k preview.
                if evidence["source_mode"] == "web":
                    data = json.loads(raw)
                    matches = []
                    for row in data.get("results", []):
                        try:
                            if normalize_url(str(row.get("url", ""))) == evidence["source_id"]:
                                matches.append(row)
                        except ValueError:
                            continue
                    if not any(content_hash(str(r.get("raw_content") or r.get("content") or "")) == evidence["content_hash"] for r in matches):
                        reasons.append("source_content_checksum_mismatch")
                elif hashlib.sha256(raw).hexdigest() != evidence["content_hash"]:
                    reasons.append("unsupported_or_mismatched_private_material")
            except (OSError, ValueError, TypeError):
                reasons.append("source_material_missing_unreadable_or_unverifiable")
            checks.append({"evidence_id": eid, "task_id": evidence["task_id"], "tool_call_id": evidence["tool_call_id"],
                           "artifact_path": relative, "passed": not reasons, "reasons": reasons})
    passed = {c["evidence_id"] for c in checks if c["passed"]}
    successes = sum(eid in passed for eid in citations)
    return {"run_id": run_id, "status": run["status"], "report_source": report_source,
            "citation_occurrences": len(citations), "passed_occurrences": successes,
            "traceability_rate": successes / len(citations) if citations else None,
            "unique_citations": len(checks), "checks": checks,
            "scope": "structural ownership, tool membership and stored material integrity; not semantic entailment"}
