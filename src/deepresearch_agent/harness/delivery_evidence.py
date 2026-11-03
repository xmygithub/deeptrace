"""Saved delivery searches and lightweight source ordering; no vector index."""
import re

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult


def saved_delivery_evidence(context):
    rows = context.context_snapshot.get("delivery", {}).get("retrieved_results", [])
    return [RetrievalResult.from_dict(row) for row in rows]


def delivery_evidence_records(context):
    return [(None, row.metadata.extra.get("tool_call_id"), row.metadata.extra.get("provider", "unknown"), row)
            for row in saved_delivery_evidence(context)]


def coverage_progress(review):
    # Wording changes and more paragraphs alone are not evidence of progress.
    return {item["requirement_id"]: {"covered": 2, "partial": 1}.get(item["status"], 0)
            for item in review.get("items", [])}


def order_supplement_sources(rows, requirements, review, new_ids=()):
    missing = {item["requirement_id"] for item in review["items"] if item["status"] != "covered"}
    text = " ".join(str(item.get("text", "")) for item in requirements if item["id"] in missing)
    terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_]+", text.lower()))
    terms.update(text[i:i + 2] for i in range(len(text) - 1) if all('\u3400' <= ch <= '\u9fff' for ch in text[i:i + 2]))
    preferred = set(new_ids)

    def score(row):
        material = (str(row.metadata.title or "") + " " + str(row.evidence)).lower()
        return (row.result_id in preferred, sum(term in material for term in terms))

    return sorted(rows, key=score, reverse=True)
