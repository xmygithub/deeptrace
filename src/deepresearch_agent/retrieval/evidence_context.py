"""Source packets with explicit token budgets and stable evidence identities."""
import json

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from deepresearch_agent.context.tokens import count_tokens


def source_packet(results, *, max_tokens, max_sources=None, per_source_tokens=2200, preferred_ids=()):
    """Deduplicate and reserve space for every selected source before truncating.

    Preferred IDs select relevant evidence even when it appears late in the
    ledger. Text remains a contiguous original prefix, so quote validation still
    works. Titles/URLs and truncation flags are separate from quoted body text.
    """
    unique = {}
    for raw in results:
        item = raw if isinstance(raw, RetrievalResult) else RetrievalResult.from_dict(raw)
        unique.setdefault(item.result_id, item)
    preferred = set(preferred_ids)
    items = sorted(unique.values(), key=lambda item: item.result_id not in preferred)
    if max_sources is not None:
        items = items[:max_sources]
    packets = []
    remaining = max_tokens
    for index, item in enumerate(items):
        allowance = min(per_source_tokens, remaining // max(1, len(items) - index))
        raw = item.evidence if isinstance(item.evidence, str) else json.dumps(item.evidence, ensure_ascii=False)
        source = {"evidence_id": item.result_id, "title": (item.metadata.title or "")[:400],
                  "source": str(item.metadata.url or item.metadata.source_id)[:600],
                  "content_kind": item.metadata.extra.get("content_kind") or ("document_text" if item.source_mode != "web" else "unknown"),
                  "source_truncated": bool(item.metadata.extra.get("content_truncated")),
                  "origin_task_id": item.metadata.extra.get("task_id"), "text": raw, "truncated": False}
        cost = lambda: count_tokens(json.dumps(source, ensure_ascii=False))
        total_cost = lambda: count_tokens(json.dumps(packets + [source], ensure_ascii=False))
        if cost() > allowance or total_cost() > max_tokens:
            source["truncated"] = True
            lo, hi = 0, len(raw)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                source["text"] = raw[:mid]
                if cost() <= allowance and total_cost() <= max_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            source["text"] = raw[:lo]
        if not source["text"] or cost() > allowance:
            continue
        # Include list punctuation in the total, as it appears in actual JSON.
        total = count_tokens(json.dumps(packets + [source], ensure_ascii=False))
        if total > max_tokens:
            continue
        packets.append(source)
        remaining = max_tokens - total
    return packets
