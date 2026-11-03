"""User-facing report sanitization and identity-preserving citation helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_THINK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
_UNCLOSED_THINK = re.compile(r"<think\b[^>]*>.*$", re.I | re.S)
_FORBIDDEN = re.compile(
    r"SYSTEM\s+CONTRACT|CURRENT\s+TASK|SELECTED\s+SKILL|TOOL\s+ALLOWLIST|"
    r"created_from_runs|verification_failures|#\s*系统契约|#\s*当前任务",
    re.I,
)
_CITATION = re.compile(r"\[(\^?)(?:证据ID[:：]\s*)?((?:ev_|result_)[A-Za-z0-9_-]+)\]")
_SCOPE_LEAD = re.compile(r"^(?:本(?:篇|份)?(?:研究)?报告(?:将|旨在|主要|系统|围绕|聚焦|介绍|讨论|比较|对比|分析|梳理|整理|从|首先|最后)|报告(?:所有|各项|中的).{0,25}(?:引用|来源)|this report (?:compares|examines|discusses|reviews|summari[sz]es|focuses))", re.I)
_FACT_ASSERTION = re.compile(r"具有|采用|包括|实现|支持|能够|可以|优于|保证|证明|证实|显著|提升|降低|达到|结果显示|\d\s*(?:%|％|倍|秒|毫秒)|\b(?:guarantees?|outperforms?|proves?|improves?|reduces?)\b", re.I)


def is_scope_description(text: str) -> bool:
    """Exclude only pure descriptions of report scope, never whole abstracts.

    Every sentence must describe this report. A framework fact or quantitative
    claim mixed into the same paragraph keeps the paragraph subject to checks.
    """
    sentences = [part.strip() for part in re.split(r"[。！？!?]|\.(?:\s|$)", text) if part.strip()]
    return bool(sentences) and all(_SCOPE_LEAD.search(part) and not _FACT_ASSERTION.search(part) for part in sentences)


def sanitize_report(text: str | None) -> str:
    """Remove hidden reasoning and internal harness material from model output."""
    value = str(text or "").replace("\x00", "")
    value = _THINK.sub("", value)
    value = _UNCLOSED_THINK.sub("", value)
    lines = value.splitlines()
    cleaned: list[str] = []
    skipping_internal = False
    for line in lines:
        if _FORBIDDEN.search(line):
            skipping_internal = True
            continue
        if skipping_internal:
            # Resume only at an explicit, user-facing report section.
            if re.match(r"^#{1,3}\s+(研究报告|深度研究报告|结论|摘要|回答|要点|分析)", line.strip(), re.I):
                skipping_internal = False
            else:
                continue
        cleaned.append(line.rstrip())
    value = "\n".join(cleaned).strip()
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value


def has_internal_material(text: str | None) -> bool:
    value = str(text or "")
    return bool(re.search(r"<think\b", value, re.I) or _FORBIDDEN.search(value))


def _authority_key(item: Any) -> tuple[int, float]:
    metadata = getattr(item, "metadata", None)
    domain = str(getattr(metadata, "domain", "") or "").lower()
    extra = getattr(metadata, "extra", {}) or {}
    authority = int(extra.get("authority_rank") or 0)
    if not authority:
        authority = int(
            domain.startswith(("docs.", "help.", "developer."))
            or domain.endswith((".gov", ".gov.cn", ".edu", ".edu.cn"))
        )
    return authority, float(getattr(item, "score", 0.0) or 0.0)


def rank_evidence(items: Sequence[Any]) -> list[Any]:
    """Prefer official/documentation domains, then provider relevance score."""
    return sorted(items, key=_authority_key, reverse=True)


def citation_evidence_ids(items: Sequence[Any], *, limit: int = 5) -> list[str]:
    """Use the strongest authority tier for citations instead of mixing weaker pages."""
    ranked = rank_evidence(items)
    if not ranked:
        return []
    strongest = _authority_key(ranked[0])[0]
    selected = [item for item in ranked if _authority_key(item)[0] == strongest] if strongest else ranked
    return [str(getattr(item, "result_id", "")) for item in selected[:limit] if getattr(item, "result_id", None)]


def add_inline_citations(text: str, evidence_ids: Sequence[str]) -> str:
    """Normalize explicit citations only; never choose evidence for a claim.

    Older writer prompts produced ``result_<suffix>`` for an ``ev_<suffix>``
    input. Repair that prefix only when the exact target exists in this scope.
    Unknown markers and uncited claims remain visible to the verifier. The
    historical function name is retained for callers; no citations are added.
    """
    allowed = set(evidence_ids)

    def normalize(match: re.Match[str]) -> str:
        footnote, original = match.groups()
        target = original
        if target not in allowed and target.startswith("result_"):
            target = "ev_" + target[len("result_"):]
        if target not in allowed:
            return match.group(0)
        return f"[{footnote}{target}]"

    return _CITATION.sub(normalize, text).strip()


__all__ = ["add_inline_citations", "citation_evidence_ids", "has_internal_material", "is_scope_description", "rank_evidence", "sanitize_report"]
