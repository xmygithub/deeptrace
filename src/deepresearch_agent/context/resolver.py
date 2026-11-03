"""Resolve referential follow-ups using only bounded current-Session context."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedQuery:
    original_query: str
    resolved_query: str
    used_message_ids: list[str]
    assumptions: list[str]
    confidence: float = 1.0


class QueryResolver:
    # Match referential expressions, not characters embedded in words such as
    # 应该、其他、哪些、如何. A new standalone question needs no prior history.
    _reference = re.compile(r"(?:它们|他们|她们|它|这个|那个|上述|前面|该(?:方案|项目|报告|方法|问题|任务)|继续|再说|相比之下|[这那]一部分|[这那]部分)")
    _reference_prefix = re.compile(r"^(?:(?:请|再|帮我|那么|那|并|然后|接着|能否|可以|解释|分析|比较|讨论|说明|介绍|看看|讲讲|一下|说说|关于|对于|对|和|与|只|修改|把|将|给我|详细|仔细)\s*)*$")

    @classmethod
    def needs_history(cls, query: str) -> bool:
        match = cls._reference.search(query)
        if match is None:
            return False
        # A named subject before a pronoun can be its antecedent in this very
        # request ("比较工作流与自主规划，讨论它们..."). Do not treat every
        # occurrence of 它们/这个 as a reference to another conversation.
        prefix = query[:match.start()].strip(" \t\n，,。；;：:")
        return bool(cls._reference_prefix.fullmatch(prefix))

    def resolve(self, query: str, history) -> ResolvedQuery:
        previous = [item for item in history if getattr(item, "role", "") in {"user", "assistant"}]
        referential = self.needs_history(query) or (len(query.strip()) <= 24 and bool(previous))
        if referential and not previous:
            return ResolvedQuery(
                query, query, [], ["当前 Session 中没有足够信息消解指代"], 0.4,
            )
        if not referential:
            return ResolvedQuery(query, query, [], [], 1.0)
        selected = previous[-4:]
        context = "；".join(f"{item.role}: {item.content[:300]}" for item in selected)
        resolved = f"基于同一会话的已知上下文（仅用于消解指代）：{context}\n当前追问：{query}"
        return ResolvedQuery(query, resolved, [item.message_id for item in selected], [], 0.9)
