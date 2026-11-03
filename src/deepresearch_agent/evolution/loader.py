"""Progressive disclosure: active catalog first, full text only on selection."""

from __future__ import annotations

import re
from deepresearch_agent.memory.retriever import lexical_terms

from .registry import SkillRegistry

_GENERIC = {"中的", "模型", "用户", "需要", "问题", "解释", "使用", "通过", "进行", "功能", "作用", "系统", "资料", "生成", "报告", "来源", "验证", "相关", "方法", "比较", "分析", "什么", "如何", "一个", "the", "and", "with", "for", "from", "this", "that"}


def match_skill(query, spec):
    """Conservative trigger matching; generic description overlap cannot activate."""
    query_terms = lexical_terms(query) - _GENERIC
    anchors = set(re.findall(r"[a-z][a-z0-9]+", spec.name.lower())) - {"explainer", "workflow", "research", "diagnosis", "skill", "evidence", "role", "agent", "rag", "web", "search", "retrieval"}
    matched = query_terms & (lexical_terms(" ".join(spec.triggers)) - _GENERIC)
    # An explicit trigger phrase is strong evidence even for short Chinese tasks.
    exact = [trigger for trigger in spec.triggers if len(trigger.strip()) >= 4 and trigger.lower() in query.lower()]
    anchor_hits = anchors & query_terms
    score = len(matched) + 2 * len(anchor_hits) + 4 * len(exact)
    accepted = bool(exact or anchor_hits or len(matched) >= 3)
    return score if accepted else 0, sorted(matched | anchor_hits)


class SkillLoader:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    async def resolve(self, query: str, *, source_mode: str, run_id: str | None = None) -> dict:
        catalog, best = [], None
        for version, spec in await self.registry.selectable_specs(run_id=run_id):
            catalog.append({"name": spec.name, "description": spec.description, "version": spec.version})
            if source_mode not in spec.source_modes:
                continue
            score, matched = match_skill(query, spec)
            if score and (best is None or score > best[0]):
                best = (score, version, spec, matched)
        if best is None:
            return {"catalog": catalog, "selected": None}
        _, version, spec, matched = best
        return {"catalog": catalog, "selected": {
            "name": spec.name, "version": spec.version,
            "content": self.registry.read_text(version.content_path),
            "allowed_tools": spec.allowed_tools, "machine_policy": spec.machine_policy,
            "selection": {"method": "source-filter+trigger-match", "score": best[0], "matched_terms": matched, "source_mode": source_mode, "status": version.status},
        }}

