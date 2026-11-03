"""Bounded curated memory inspired by Hermes Agent's frozen stores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from deepresearch_agent.context.tokens import count_tokens
from deepresearch_agent.persistence.repositories import AuditRepository, MemoryRepository
from deepresearch_agent.persistence.repositories.utils import utc_now_iso

from .policies import MemoryPolicy


class MemoryRejected(ValueError):
    pass


@dataclass(frozen=True)
class MemoryLimits:
    max_tokens: int
    max_chars: int


class CuratedMemoryService:
    """The one persistent Memory system; user/project are targets, not layers."""

    VALID_STATUSES = {"candidate", "active", "rejected", "archived"}
    VALID_TARGETS = {"user", "project"}

    def __init__(
        self, repository: MemoryRepository, audit: AuditRepository, *,
        policy: MemoryPolicy | None = None, user_max_tokens: int = 500,
        project_max_tokens: int = 800, user_max_chars: int = 1375,
        project_max_chars: int = 2200,
    ):
        self.repository, self.audit = repository, audit
        self.policy = policy or MemoryPolicy()
        self.limits = {
            "user": MemoryLimits(max(1, user_max_tokens), max(1, user_max_chars)),
            "project": MemoryLimits(max(1, project_max_tokens), max(1, project_max_chars)),
        }

    @staticmethod
    def _target(item) -> str:
        return str(getattr(item, "target", None) or item.scope)

    def _validate_target(self, target: str) -> str:
        normalized = target.strip().lower()
        if normalized not in self.VALID_TARGETS:
            raise MemoryRejected("Curated Memory 只允许 user 或 project target")
        return normalized

    async def capacity(self, target: str, *, excluding_id: str | None = None) -> dict[str, int | float | str]:
        target = self._validate_target(target)
        items = await self.repository.list_curated(statuses=("active",), target=target, now=utc_now_iso())
        contents = [item.content for item in items if item.memory_id != excluding_id]
        rendered = ""
        if contents:
            heading = "# USER" if target == "user" else "# PROJECT MEMORY"
            rendered = heading + "\n" + "\n".join(f"- {value}" for value in contents)
        chars = len(rendered)
        tokens = count_tokens(rendered)
        limit = self.limits[target]
        ratio = max(chars / limit.max_chars, tokens / limit.max_tokens)
        return {
            "target": target, "chars": chars, "tokens": tokens,
            "max_chars": limit.max_chars, "max_tokens": limit.max_tokens,
            "usage_ratio": ratio, "warning": ratio >= 0.8,
        }

    async def capacities(self) -> dict[str, dict[str, int | float | str]]:
        return {target: await self.capacity(target) for target in ("user", "project")}

    async def _assert_capacity(self, target: str, content: str, *, excluding_id: str | None = None) -> None:
        usage = await self.capacity(target, excluding_id=excluding_id)
        existing = await self.repository.list_curated(statuses=("active",), target=target, now=utc_now_iso())
        contents = [item.content for item in existing if item.memory_id != excluding_id] + [content]
        heading = "# USER" if target == "user" else "# PROJECT MEMORY"
        rendered = heading + "\n" + "\n".join(f"- {value}" for value in contents)
        next_chars, next_tokens = len(rendered), count_tokens(rendered)
        if next_chars > int(usage["max_chars"]) or next_tokens > int(usage["max_tokens"]):
            raise MemoryRejected(
                f"{target} Memory 容量不足：写入后 {next_tokens}/{usage['max_tokens']} tokens，"
                f"{next_chars}/{usage['max_chars']} chars；请先合并或归档旧条目"
            )

    async def create_candidate(
        self, *, content: str, scope: str | None = None, target: str | None = None,
        kind: str = "note", provenance_refs: list[str] | None = None,
        confidence: float = 1.0, session_id: str | None = None,
        created_by: str = "agent", expires_at: str | None = None,
        activate: bool = False,
        extraction: dict[str, str] | None = None,
        supersedes_id: str | None = None,
    ):
        selected_target = self._validate_target(target or scope or "project")
        normalized = content.strip()
        extraction_record = None
        if extraction is not None:
            # Persist only bounded, policy-checked excerpts, never the entire
            # source message. Rationale is a model explanation, not evidence.
            extraction_record = {key: str(extraction.get(key, "")) for key in ("evidence_quote", "rationale")}
            for value in extraction_record.values():
                if value and not self.policy.validate(value).allowed:
                    raise MemoryRejected("提取依据不符合 Memory 保存规则")
        verdict = self.policy.validate(normalized)
        if not verdict.allowed:
            await self.audit.append(action="memory.rejected_by_policy", entity_type="memory", entity_id="pending", payload={"reason": verdict.reason})
            raise MemoryRejected(verdict.reason or "Memory policy rejected")
        refs = list(provenance_refs or [])
        if selected_target == "project" and kind in {"fact", "decision"} and not refs:
            raise MemoryRejected("项目事实或决策必须包含 provenance")

        existing_items = await self.repository.list_curated(statuses=("candidate", "active"), target=selected_target)
        for existing in existing_items:
            if existing.content.strip() == normalized:
                return existing

        supersedes = supersedes_id
        old = None
        if supersedes:
            old = await self.repository.get(supersedes)
            if old is None or old.status != "active" or self._target(old) != selected_target:
                raise MemoryRejected("待替代记忆必须是同一分类下已启用的条目")

        if activate:
            await self._assert_capacity(selected_target, normalized, excluding_id=supersedes)
        item = await self.repository.create_candidate(
            content=normalized, scope=selected_target, kind=kind, provenance_refs=refs,
            confidence=max(0.0, min(1.0, confidence)), session_id=None,
            created_by=created_by, supersedes=supersedes, expires_at=expires_at,
            superseded_content=old.content if old is not None else None,
        )
        if activate:
            try:
                limit = self.limits[selected_target]
                await self.repository.activate_curated(item.memory_id, max_tokens=limit.max_tokens, max_chars=limit.max_chars)
            except ValueError as exc:
                raise MemoryRejected(str(exc)) from exc
            item = await self.repository.get(item.memory_id)
        await self.audit.append(
            action="memory.activated" if activate else "memory.candidate_created",
            entity_type="memory", entity_id=item.memory_id,
            payload={"target": selected_target, "supersedes": supersedes, "provenance_refs": refs,
                     **({"extraction": extraction_record} if extraction_record else {})},
        )
        return item

    async def update(self, memory_id: str, *, content: str | None = None, status: str | None = None, expires_at: str | None = None):
        item = await self.repository.get(memory_id)
        if item is None:
            return None
        target = self._validate_target(self._target(item))
        values: dict[str, Any] = {}
        next_content = item.content
        if content is not None:
            next_content = content.strip()
            verdict = self.policy.validate(next_content)
            if not verdict.allowed:
                raise MemoryRejected(verdict.reason or "Memory policy rejected")
            values["content"] = next_content
        if status is not None:
            if status == "expired":
                status = "archived"
            if status not in self.VALID_STATUSES:
                raise ValueError("非法 Memory 状态")
            values["status"] = status
            if status == "archived":
                values["archived_at"] = utc_now_iso()
        if expires_at is not None:
            values["expires_at"] = expires_at
        next_status = str(values.get("status", item.status))
        if next_status == "active" and (item.status != "active" or content is not None):
            verdict = self.policy.validate(next_content)
            if not verdict.allowed:
                raise MemoryRejected(verdict.reason or "Memory policy rejected")
            excluding_id = memory_id if item.status == "active" else (item.supersedes or memory_id)
            await self._assert_capacity(target, next_content, excluding_id=excluding_id)
        if values:
            if next_status == "active":
                try:
                    limit = self.limits[target]
                    await self.repository.activate_curated(memory_id, max_tokens=limit.max_tokens, max_chars=limit.max_chars, **values)
                except ValueError as exc:
                    raise MemoryRejected(str(exc)) from exc
            else:
                await self.repository.update_lifecycle(memory_id, **values)
            await self.audit.append(action="memory.updated", entity_type="memory", entity_id=memory_id, payload={**values, "target": target})
        return await self.repository.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        archived = await self.repository.soft_delete(memory_id)
        if archived:
            await self.audit.append(action="memory.archived", entity_type="memory", entity_id=memory_id)
        return archived

    async def build_snapshot(self) -> dict[str, Any]:
        items = await self.repository.list_curated(statuses=("active",), now=utc_now_iso())
        groups: dict[str, list[dict[str, Any]]] = {"user": [], "project": []}
        for item in items:
            target = self._target(item)
            if target in groups:
                groups[target].append({
                    "memory_id": item.memory_id, "target": target, "kind": item.kind,
                    "status": item.status, "created_by": item.created_by,
                    "content": item.content,
                    "provenance_refs": json.loads(item.provenance_json or "[]"),
                    "updated_at": item.updated_at,
                })
        rendered_parts = []
        for target, heading in (("user", "USER"), ("project", "PROJECT MEMORY")):
            if groups[target]:
                rendered_target = f"# {heading}\n" + "\n".join(f"- {entry['content']}" for entry in groups[target])
                target_chars = len(rendered_target)
                target_tokens = count_tokens(rendered_target)
                limit = self.limits[target]
                if target_chars > limit.max_chars or target_tokens > limit.max_tokens:
                    raise MemoryRejected(
                        f"现有 {target} Memory 超出硬上限：{target_tokens}/{limit.max_tokens} tokens，"
                        f"{target_chars}/{limit.max_chars} chars；请先在 Memory 页面合并或归档条目"
                    )
                rendered_parts.append(rendered_target)
        rendered = "\n\n".join(rendered_parts)
        return {
            "version": await self.repository.snapshot_version(), "items": groups,
            "rendered": rendered, "tokens": count_tokens(rendered), "chars": len(rendered),
        }

    @staticmethod
    def serialize(item) -> dict[str, Any]:
        target = str(getattr(item, "target", None) or item.scope)
        return {
            "memory_id": item.memory_id, "target": target, "scope": target,
            "supersedes": item.supersedes,
            "kind": item.kind, "content": item.content,
            "provenance_refs": json.loads(item.provenance_json or "[]"),
            "confidence": item.confidence,
            "status": "archived" if item.status == "expired" else item.status,
            "created_by": item.created_by, "expires_at": item.expires_at,
            "created_at": item.created_at, "updated_at": item.updated_at,
        }


MemoryService = CuratedMemoryService
