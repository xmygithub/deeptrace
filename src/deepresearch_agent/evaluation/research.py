"""Offline report review: immutable evidence bundles and explicit reviewer labels.

This module never invokes a model/search service or writes to the application DB.
An existing citation ID is deliberately not a semantic-support judgement.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from deepresearch_agent.retrieval.web_utils import content_hash, normalize_url


CITATION = re.compile(r"\[\^?(ev_[A-Za-z0-9_-]+)\]")
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")
QUALITY_DIMENSIONS = ("coverage", "analysis", "instruction_following", "readability")
CLAIM_VERDICTS = {"pending", "supported", "unsupported", "contradicted", "unverifiable"}
SECRET = re.compile(
    r"(?<![A-Za-z0-9_])(?:sk|tvly)-[A-Za-z0-9_-]{8,}"
    r"|\bBearer\s+[A-Za-z0-9._-]{8,}", re.I,
)


def digest(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return SECRET.sub("[REDACTED]", value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _redact(item) for key, item in value.items()
            if key.lower() not in {"api_key", "token", "authorization", "password", "secret"}
        }
    return value


def _export_copy(run: dict) -> dict:
    exported = _redact(run)
    for original, source in zip(run["evidence"], exported["evidence"]):
        if source["content"]["kind"] == "raw_snapshot" and source["content"]["text"] != original["content"]["text"]:
            source["original_content_hash"] = original["content_hash"]
            source["content_hash"] = content_hash(source["content"]["text"])
            source["content"]["redacted"] = True
    return exported


def load_cases(path: Path) -> dict[str, dict]:
    data = read_json(path)
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported research case schema")
    cases = {}
    for case in data["cases"]:
        case_id = case["case_id"]
        if not IDENTIFIER.fullmatch(case_id) or case_id in cases:
            raise ValueError(f"Invalid or duplicate case_id: {case_id}")
        if not case.get("question", "").strip() or not case.get("criteria"):
            raise ValueError(f"Question and criteria required: {case_id}")
        ids = [item["criterion_id"] for item in case["criteria"]]
        if len(ids) != len(set(ids)) or any(not IDENTIFIER.fullmatch(item) for item in ids):
            raise ValueError(f"Invalid/duplicate criterion ID: {case_id}")
        if any(not item.get("description", "").strip() for item in case["criteria"]):
            raise ValueError(f"Empty criterion: {case_id}")
        cases[case_id] = case
    if not cases:
        raise ValueError("Empty research suite")
    return cases


def _object(raw: str | None) -> dict:
    data = json.loads(raw or "{}")
    return data if isinstance(data, dict) else {}


def _artifact_content(root: Path, run_id: str, metadata: dict, source_id: str) -> dict:
    """Require matching URL and content hash; a preview is never called raw text."""
    result = {"kind": "summary_only", "text": "", "reason": "No saved raw snapshot"}
    relative = (metadata.get("extra") or {}).get("artifact_path")
    if not relative:
        return result
    path = (root / relative).resolve()
    run_root = (root / run_id).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_relative_to(run_root):
        return {**result, "reason": "Rejected artifact path outside this Run"}
    if not path.is_file():
        return {**result, "reason": "Saved artifact is missing"}
    try:
        raw = read_json(path)
        for item in raw.get("results", []):
            if normalize_url(str(item.get("url") or "")) != normalize_url(source_id):
                continue
            # Tavily content is a search snippet; only raw_content is a page snapshot.
            text = str(item.get("raw_content") or "")
            if not text:
                continue
            if metadata.get("content_hash") != content_hash(text):
                continue
            return {
                "kind": "raw_snapshot", "text": text,
                "artifact_path": str(relative), "content_hash_verified": True,
                "reason": "URL and normalized content hash match saved evidence",
            }
        return {**result, "reason": "No matching URL/hash with raw_content in artifact"}
    except (OSError, ValueError, TypeError, AttributeError):
        return {**result, "reason": "Unreadable or invalid artifact"}


def read_run(database: Path, artifact_root: Path, run_id: str) -> dict:
    """Take a consistent read-only SQLite snapshot. No schema creation/PRAGMA writes."""
    if not IDENTIFIER.fullmatch(run_id):
        raise ValueError("Invalid run_id")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        trigger = connection.execute(
            "SELECT content FROM messages WHERE message_id=?", (run["trigger_message_id"],),
        ).fetchone()
        report = connection.execute(
            "SELECT content FROM messages WHERE run_id=? AND role='assistant' ORDER BY created_at LIMIT 1",
            (run_id,),
        ).fetchone()
        evidence = connection.execute("SELECT * FROM evidence WHERE run_id=? AND invalidated_at IS NULL ORDER BY evidence_id", (run_id,)).fetchall()
        checks = connection.execute("SELECT kind,required,passed FROM contract_checks WHERE run_id=?", (run_id,)).fetchall()
        calls = connection.execute("SELECT status FROM tool_calls WHERE run_id=?", (run_id,)).fetchall()
    finally:
        connection.close()
    usage_container = _object(run["usage_json"])
    usage = usage_container.get("usage", usage_container)
    model = _object(run["model_snapshot_json"])
    required = [row for row in checks if row["required"]]
    items = []
    for row in evidence:
        metadata = _object(row["metadata_json"])
        # The DB column is authoritative, not a potentially stale metadata copy.
        metadata["content_hash"] = row["content_hash"]
        items.append({
            "evidence_id": row["evidence_id"], "source_id": row["source_id"],
            "title": row["title"], "summary": row["summary"],
            "source_mode": row["source_mode"], "content_hash": row["content_hash"],
            "content": _artifact_content(artifact_root, run_id, metadata, row["source_id"]),
        })
    latency = None
    if run["started_at"] and run["completed_at"]:
        latency = (datetime.fromisoformat(run["completed_at"].replace("Z", "+00:00")) - datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))).total_seconds()
    return _export_copy({
        "run_id": run_id, "question": trigger["content"] if trigger else "",
        "status": run["status"], "source_mode": run["source_mode"],
        "workflow_mode": run["workflow_mode"], "report": report["content"] if report else "",
        "started_at": run["started_at"], "completed_at": run["completed_at"],
        "error_code": run["error_code"],
        "model": {key: model[key] for key in ("llm_model", "temperature", "skill") if key in model},
        "budget": _object(run["budget_json"]),
        "metrics": {
            "verified_completion": run["status"] == "completed" and bool(required) and all(row["passed"] == 1 for row in required),
            "llm_tokens": usage.get("llm_tokens"), "elapsed_seconds": usage.get("elapsed_seconds"),
            "latency_seconds": latency, "reported_tool_calls": usage.get("tool_calls"),
            "persisted_tool_calls": len(calls), "reported_tavily_calls": usage.get("tavily_calls"),
        },
        "evidence": items,
        "saved_raw_search_files": len(list((artifact_root / run_id / "tavily").glob("*.json"))),
    })


def claim_candidates(report: str, limit: int = 12) -> tuple[list[dict], int]:
    """Paragraph candidates only: not an exhaustive semantic claim extractor."""
    body = re.split(r"(?m)^##\s+全量证据索引", report)[0]
    body = re.sub(r"<think>.*?</think>", "\n\n", body, flags=re.S)
    body = re.sub(r"```.*?```", "\n\n", body, flags=re.S)
    paragraphs = []
    for part in re.split(r"\n\s*\n|^\s*#{1,6}[ \t]+[^\n]*(?:\n|$)|^\[\^?ev_[^]]+\]:[^\n]*(?:\n|$)", body, flags=re.M):
        text = part.strip()
        if len(text) >= 20:
            paragraphs.append(text)
    # Spread a bounded sample over the whole report instead of only its opening.
    if len(paragraphs) <= limit:
        indices = list(range(len(paragraphs)))
    elif limit == 1:
        indices = [0]
    else:
        indices = [round(i * (len(paragraphs) - 1) / (limit - 1)) for i in range(limit)]
    return [
        {"claim_id": f"claim_{index + 1:03d}", "report_quote": paragraphs[position],
         "claim_text": paragraphs[position],
         "evidence_ids": list(dict.fromkeys(CITATION.findall(paragraphs[position]))),
         "verdict": "pending", "reason": "", "evidence_quotes": []}
        for index, position in enumerate(indices)
    ], len(paragraphs)


def validate_snapshot(run: dict) -> None:
    required = {"run_id", "question", "status", "source_mode", "workflow_mode", "report", "metrics", "evidence"}
    if not isinstance(run, dict) or not required.issubset(run):
        raise ValueError(f"Snapshot requires {sorted(required)}")
    for key in ("run_id", "question", "status", "source_mode", "workflow_mode", "report"):
        if not isinstance(run[key], str):
            raise ValueError(f"Snapshot {key} must be text")
    if not IDENTIFIER.fullmatch(run["run_id"]):
        raise ValueError("Invalid snapshot run_id")
    if not isinstance(run["metrics"], dict) or not isinstance(run["evidence"], list):
        raise ValueError("Invalid metrics/evidence structure")
    for key in ("llm_tokens", "elapsed_seconds", "latency_seconds"):
        value = run["metrics"].get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid metric: {key}")
    if run["metrics"].get("verified_completion") is not None and not isinstance(run["metrics"]["verified_completion"], bool):
        raise ValueError("Invalid completion measurement")
    ids = set()
    for source in run["evidence"]:
        if not {"evidence_id", "source_id", "content_hash", "content"}.issubset(source):
            raise ValueError("Evidence requires ID, source, content hash and content")
        if source["evidence_id"] in ids:
            raise ValueError("Duplicate evidence ID")
        ids.add(source["evidence_id"])
        content = source["content"]
        if content.get("kind") not in {"raw_snapshot", "summary_only"} or not isinstance(content.get("text"), str):
            raise ValueError("Evidence content must be raw_snapshot or summary_only with text")
        if content["kind"] == "raw_snapshot" and (not content["text"] or content_hash(content["text"]) != source["content_hash"]):
            raise ValueError("Raw snapshot content hash mismatch")


def make_bundle(case: dict, run: dict, *, arm: str = "agent", label: str = "", max_claims: int = 12) -> tuple[dict, dict]:
    if arm not in {"agent", "baseline"} or not 1 <= max_claims <= 100:
        raise ValueError("Invalid arm or max_claims")
    validate_snapshot(run)
    # Do not export obvious credentials from reports or imported baselines.
    run = _export_copy(run)
    candidates, total = claim_candidates(run["report"], max_claims)
    accepted = [case["question"], *case.get("accepted_questions", [])]
    question_match = run["question"].strip() in {item.strip() for item in accepted}
    warnings = []
    if not question_match:
        warnings.append("Question does not match case; review is exploratory and excluded from quality comparisons")
    if not run["report"]:
        warnings.append("No final report: retain execution failure, leave report quality ungraded")
    if any(item["content"]["kind"] != "raw_snapshot" for item in run["evidence"]):
        warnings.append("Some evidence has only summaries; missing page text cannot substantiate a claim")
    warnings.append("Pilot rubrics include evaluator requirements; linked reading material is not a unique gold answer")
    bundle = {
        "schema_version": 1, "exported_at": datetime.now(timezone.utc).isoformat(),
        "case": case, "question_matches_case": question_match,
        "arm": arm, "label": label or run["workflow_mode"], "run": run,
        "claim_selection": {"method": "spaced_paragraph_candidates", "scope": "sample", "candidate_paragraphs": total, "selected": len(candidates)},
        "warnings": warnings,
    }
    bundle["bundle_hash"] = digest(bundle)
    review = {
        "schema_version": 1, "bundle_hash": bundle["bundle_hash"],
        "reviewer": "", "reviewed_at": "", "reviewer_kind": "human",
        "coverage": [{"criterion_id": item["criterion_id"], "verdict": "pending", "reason": "", "report_quote": ""} for item in case["criteria"]],
        "claims": candidates,
        "quality": {key: {"score": None, "reason": "", "report_quote": ""} for key in QUALITY_DIMENSIONS},
        "evaluation_cost": {"llm_tokens": 0, "elapsed_seconds": None, "network_requests": 0},
    }
    return bundle, review


def write_package(output: Path, bundle: dict, review: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (("bundle.json", bundle), ("review.json", review)):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(bundle["run"]["report"] or "（该 Run 没有最终报告）\n", encoding="utf-8")
    lines = ["# 人工核查表", "", "以 bundle.json 为冻结依据；只在 review.json 中填写判定。report.md 是阅读副本。", "", "## 要点", ""]
    lines += [f"- `{item['criterion_id']}`：{item['description']}" for item in bundle["case"]["criteria"]]
    lines += ["", "## 候选主张", "", "以下是段落抽样，可能含多条事实。核查者应拆分为单一主张并保留原文片段，不能声称覆盖全文。", ""]
    for claim in review["claims"]:
        lines += [f"### {claim['claim_id']}", "", claim["report_quote"], "", f"引用：{', '.join(claim['evidence_ids']) or '无'}", ""]
    (output / "worksheet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_package(bundle: dict, review: dict) -> dict:
    payload = {key: value for key, value in bundle.items() if key != "bundle_hash"}
    if digest(payload) != bundle.get("bundle_hash") or review.get("bundle_hash") != bundle.get("bundle_hash"):
        raise ValueError("Bundle hash mismatch: export a new package rather than changing evidence")
    if review.get("schema_version") != 1 or review.get("reviewer_kind") not in {"human", "assistant"}:
        raise ValueError("Reviewer kind must be human or assistant; model judgements are not human labels")
    report = bundle["run"]["report"]
    criteria = {item["criterion_id"] for item in bundle["case"]["criteria"]}
    coverage = review.get("coverage", [])
    if len(coverage) != len(criteria) or {item["criterion_id"] for item in coverage} != criteria:
        raise ValueError("Coverage must contain every criterion exactly once")
    judged = False
    for item in coverage:
        if item["verdict"] not in {"pending", "met", "not_met"}:
            raise ValueError("Invalid coverage verdict")
        if item["verdict"] != "pending":
            judged = True
            if not item.get("reason", "").strip():
                raise ValueError("Coverage decisions require a reason")
            if item["verdict"] == "met" and (not item.get("report_quote") or item["report_quote"] not in report):
                raise ValueError("Met criterion requires an exact report quote")
    evidence = {item["evidence_id"]: item for item in bundle["run"]["evidence"]}
    counts: Counter = Counter()
    claim_ids = set()
    for claim in review.get("claims", []):
        cid = claim["claim_id"]
        if cid in claim_ids or not IDENTIFIER.fullmatch(cid):
            raise ValueError("Invalid/duplicate claim_id")
        claim_ids.add(cid)
        verdict = claim["verdict"]
        if verdict not in CLAIM_VERDICTS:
            raise ValueError("Invalid claim verdict")
        if not claim.get("report_quote") or claim["report_quote"] not in report:
            raise ValueError("Claim requires an exact report quote")
        counts[verdict] += 1
        if verdict == "pending":
            continue
        judged = True
        if not claim.get("claim_text", "").strip():
            raise ValueError("Reviewed claim requires a single claim_text; split multi-fact paragraphs")
        if not claim.get("reason", "").strip():
            raise ValueError("Claim decisions require a reason")
        quotes = claim.get("evidence_quotes", [])
        if verdict in {"supported", "contradicted"} and not quotes:
            raise ValueError("Supported/contradicted claim requires saved source quotes")
        if not quotes:
            continue
        citations = set(claim.get("evidence_ids", []))
        if not citations or not citations.issubset(set(CITATION.findall(claim["report_quote"]))):
            raise ValueError("Supported/contradicted claim must name citations in its report quote")
        for quote in quotes:
            source = evidence.get(quote["evidence_id"])
            if source is None or quote["evidence_id"] not in citations:
                raise ValueError("Evidence quote must refer to a cited, existing evidence ID")
            if source["content"]["kind"] != "raw_snapshot":
                raise ValueError("A summary cannot substitute for a saved page snapshot")
            if not quote.get("quote") or quote["quote"] not in source["content"]["text"]:
                raise ValueError("Evidence quote must occur in saved source text")
    scores = []
    quality = review.get("quality", {})
    if set(quality) != set(QUALITY_DIMENSIONS):
        raise ValueError("Four quality dimensions required")
    for item in quality.values():
        score = item.get("score")
        if score is None:
            continue
        judged = True
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 5:
            raise ValueError("Quality scores must be 1..5 or null")
        if not item.get("reason", "").strip() or not item.get("report_quote") or item["report_quote"] not in report:
            raise ValueError("Quality scores require a reason and an exact report quote")
        scores.append(score)
    if judged:
        if not report:
            raise ValueError("Cannot grade a missing final report")
        if not review.get("reviewer", "").strip() or not review.get("reviewed_at", "").strip():
            raise ValueError("Reviewed decisions require reviewer and reviewed_at")
        if datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00")).utcoffset() is None:
            raise ValueError("reviewed_at requires a timezone")
    coverage_done = sum(item["verdict"] != "pending" for item in coverage)
    resolved = sum(counts[key] for key in ("supported", "unsupported", "contradicted"))
    reviewed_claims = resolved + counts["unverifiable"]
    claims_finished = bool(claim_ids) and counts["pending"] == 0
    citations = CITATION.findall(report)
    valid = sum(cid in evidence for cid in citations)
    matches = bundle["question_matches_case"]
    return {
        "case_id": bundle["case"]["case_id"], "run_id": bundle["run"]["run_id"],
        "arm": bundle["arm"], "label": bundle["label"], "status": bundle["run"]["status"],
        "bundle_hash": bundle["bundle_hash"], "question_matches_case": matches,
        "report_available": bool(report), "metrics": bundle["run"]["metrics"],
        "reviewer": review.get("reviewer"), "reviewed_at": review.get("reviewed_at"),
        "reviewer_kind": review["reviewer_kind"],
        "coverage_reviewed": coverage_done, "coverage_total": len(criteria),
        "coverage_findings": [{key: item[key] for key in ("criterion_id", "verdict", "reason")} for item in coverage],
        "coverage_rate": sum(item["verdict"] == "met" for item in coverage) / len(criteria) if matches and coverage_done == len(criteria) and report else None,
        "claim_selection_scope": "sample", "claim_counts": dict(counts),
        "reviewed_claims": reviewed_claims, "resolved_claims": resolved,
        "claim_support_rate_resolved": counts["supported"] / resolved if claims_finished and resolved else None,
        "claim_verifiability": resolved / reviewed_claims if claims_finished and reviewed_claims else None,
        "report_quality_score": mean(scores) if matches and len(scores) == 4 and report else None,
        "quality_dimensions": quality, "citation_count": len(citations),
        "citation_id_validity": valid / len(citations) if citations else None,
        "evaluation_cost": review.get("evaluation_cost"), "warnings": bundle["warnings"],
        "review_notes": review.get("notes", []),
    }


def summarize_packages(paths: list[Path], *, review_filename: str = "review.json") -> dict:
    if Path(review_filename).name != review_filename or "\\" in review_filename:
        raise ValueError("Review filename must be a basename inside each package")
    rows = []
    bundles = []
    seen = set()
    for path in paths:
        bundle = read_json(path / "bundle.json")
        row = evaluate_package(bundle, read_json(path / review_filename))
        # One Run may not be counted twice under different labels or exports.
        if row["run_id"] in seen:
            raise ValueError(f"Duplicate Run in evaluation: {row['run_id']}")
        seen.add(row["run_id"])
        rows.append(row)
        bundles.append(bundle)
    if not rows:
        raise ValueError("No packages selected")
    pairs = []
    for case_id in dict.fromkeys(row["case_id"] for row in rows):
        groups = {arm: [(row, bundle) for row, bundle in zip(rows, bundles) if row["case_id"] == case_id and row["arm"] == arm] for arm in ("agent", "baseline")}
        if len(groups["agent"]) != 1 or len(groups["baseline"]) != 1:
            continue
        agent, a_bundle = groups["agent"][0]
        baseline, b_bundle = groups["baseline"][0]
        issues = []
        if not agent["question_matches_case"] or not baseline["question_matches_case"] or a_bundle["run"]["question"] != b_bundle["run"]["question"]:
            issues.append("Questions differ or do not match case")
        if digest(a_bundle["case"]) != digest(b_bundle["case"]):
            issues.append("Case/rubric versions differ")
        if agent["reviewer_kind"] != baseline["reviewer_kind"]:
            issues.append("Review provenance differs (human versus assistant)")
        a_model, b_model = a_bundle["run"].get("model", {}), b_bundle["run"].get("model", {})
        if not a_model.get("llm_model") or "temperature" not in a_model or "temperature" not in b_model or a_model != b_model:
            issues.append("Model/skill/parameter metadata missing or different")
        if not a_bundle["run"].get("budget") or a_bundle["run"].get("budget") != b_bundle["run"].get("budget"):
            issues.append("Execution budgets missing or different")
        sources = lambda b: sorted((e["source_id"], e["content_hash"]) for e in b["run"]["evidence"])
        if not sources(a_bundle) or sources(a_bundle) != sources(b_bundle):
            issues.append("Source snapshots differ or are missing; live-Web results may vary")
        if agent["report_quality_score"] is None or baseline["report_quality_score"] is None:
            issues.append("Report quality review incomplete")
        if agent["coverage_rate"] is None or baseline["coverage_rate"] is None:
            issues.append("Coverage review incomplete")
        delta = lambda key: agent["metrics"][key] - baseline["metrics"][key] if agent["metrics"].get(key) is not None and baseline["metrics"].get(key) is not None else None
        pairs.append({"case_id": case_id, "agent_run_id": agent["run_id"], "baseline_run_id": baseline["run_id"], "controlled_comparison": not issues, "limitations": issues, "token_delta": delta("llm_tokens"), "latency_delta_seconds": delta("latency_seconds")})
    quality = [row["report_quality_score"] for row in rows if row["report_quality_score"] is not None]
    quality_by_kind = {}
    for kind in ("human", "assistant"):
        values = [row["report_quality_score"] for row in rows if row["reviewer_kind"] == kind and row["report_quality_score"] is not None]
        quality_by_kind[kind] = {"graded_runs": len(values), "mean_report_quality_score": mean(values) if values else None}
    mixed = all(group["graded_runs"] for group in quality_by_kind.values())
    completion = [row["metrics"].get("verified_completion") for row in rows if isinstance(row["metrics"].get("verified_completion"), bool)]
    return {
        "schema_version": 1, "run_count": len(rows),
        "completion_measured_runs": len(completion),
        "verified_completion_rate": sum(completion) / len(completion) if completion else None,
        "reports_available": sum(row["report_available"] for row in rows),
        "quality_graded_runs": len(quality),
        "quality_by_reviewer_kind": quality_by_kind,
        "mean_report_quality_score": mean(quality) if quality and not mixed else None,
        "failed_or_incomplete_runs": sum(row["status"] != "completed" for row in rows),
        "comparisons": pairs, "runs": rows,
        "limitations": ["Pilot review, not an official benchmark score", "Assistant judgements have not been calibrated against independent human review", "Human and assistant quality scores are not pooled", "Claim support is sampled; unresolved/missing sources are reported separately", "Runtime completion does not establish report correctness"],
    }


def render_summary(summary: dict) -> str:
    def cell(value):
        return "待核查" if value is None else str(round(value, 4) if isinstance(value, float) else value).replace("|", "\\|")
    provenance = summary["quality_by_reviewer_kind"]
    lines = ["# 研究质量评测", "", f"运行样本：{summary['run_count']}；有最终报告：{summary['reports_available']}；四维评分完成：人工 {provenance['human']['graded_runs']}、AI {provenance['assistant']['graded_runs']}。", "", "AI 核查尚未经过独立人工校准，不作为人工真值。未填写的质量分数保持空值；失败 Run 保留在运行统计中。主张支持率仅针对已核查样本，不代表全文正确率。", "", "| 用例 | Run | 核查者 | 状态 | 要点覆盖 | 主张支持（可核验部分） | 待核查主张 | 已判无法核验 | Token | 耗时/秒 | 报告评分/5 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in summary["runs"]:
        lines.append("| " + " | ".join(cell(value) for value in (row["case_id"], row["run_id"], "AI" if row["reviewer_kind"] == "assistant" else "人工", row["status"], row["coverage_rate"], row["claim_support_rate_resolved"], row["claim_counts"].get("pending", 0), row["claim_counts"].get("unverifiable", 0), row["metrics"].get("llm_tokens"), row["metrics"].get("latency_seconds"), row["report_quality_score"])) + " |")
    lines += ["", "## 基线对照", ""]
    if not summary["comparisons"]:
        lines.append("尚无同一用例下唯一的 baseline/agent 配对，未计算优化收益。")
    for pair in summary["comparisons"]:
        lines += [f"- {pair['case_id']}：Token 差值 {cell(pair['token_delta'])}；耗时差值 {cell(pair['latency_delta_seconds'])}。差值为 Agent 减基线。"]
        lines += [f"  - 限制：{issue}" for issue in pair["limitations"]]
    lines += ["", "## 核查状态与限制", ""]
    for row in summary["runs"]:
        lines.append(f"- {row['run_id']}：要点已核查 {row['coverage_reviewed']}/{row['coverage_total']}；主张待核查 {row['claim_counts'].get('pending', 0)}；引用 ID 有效率 {cell(row['citation_id_validity'])}（仅结构检查）。")
        lines += [f"  - {warning}" for warning in row["warnings"]]
        lines += [f"  - 核查说明：{note}" for note in row["review_notes"]]
    lines += ["", "## 要点与报告评分依据", ""]
    for row in summary["runs"]:
        lines += [f"### {row['label']} · {row['run_id']}", ""]
        lines += [f"- {item['criterion_id']}（{item['verdict']}）：{item['reason'] or '待核查'}" for item in row["coverage_findings"]]
        lines += [f"- {key}：{cell(item['score'])}/5。{item['reason'] or '待核查'}" for key, item in row["quality_dimensions"].items()]
        lines.append("")
    return "\n".join(lines) + "\n"
