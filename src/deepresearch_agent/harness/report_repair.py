"""Bounded paragraph edits: preserve every paragraph outside the repair targets."""
import hashlib
import re

from deepresearch_agent.harness.contracts import SourceMode
from deepresearch_agent.harness.delivery import DeliveryService
from deepresearch_agent.harness.report_reconciliation import report_paragraphs
from deepresearch_agent.harness.report_safety import has_internal_material
from deepresearch_agent.harness.verifiers.deterministic import DeterministicVerifiers
from deepresearch_agent.retrieval.research_loop import bounded_sources
from deepresearch_agent.config import settings

_CITATION = re.compile(r"\[\^?(ev_[A-Za-z0-9_-]+)\]")
_UNRESOLVED = re.compile(r"\[\^?(?:(?:result_|digest_)[A-Za-z0-9_-]+|证据ID[:：]\s*[^\]\n]+)\]")


def locate_correction(paragraphs, original):
    """Exact prose localization with whitespace/citation normalization only.

    No semantic similarity: ambiguous quotes never authorize an edit.
    """
    def clean(text):
        return re.sub(r"\s+", "", _CITATION.sub("", text))
    needle = clean(original)
    matches = {pid: text for pid, text in paragraphs.items() if len(needle) >= 12 and needle in clean(text)}
    return matches if len(matches) == 1 else {}


class ReportRepairService:
    def __init__(self, llm=None):
        self.json = DeliveryService(llm)

    async def repair(self, *, report, results, run_id, source_mode, failures, consistency=None):
        ledger = [{"evidence_id": item.result_id, "source_mode": item.source_mode} for item in results]
        verifier = DeterministicVerifiers(run_id=run_id, source_mode=SourceMode(source_mode), report=report, evidence=ledger)
        invalid_lines = {item["text"] for item in verifier.claim_support().observed["actual"]["claim_details"] if not item["passed"]}
        paragraphs, _ = report_paragraphs(report)
        known = {item.result_id for item in results}
        targets = {}
        if set(failures) & {"citation_integrity", "claim_support"}:
            for pid, text in paragraphs.items():
                if (any(line.strip()[:2000] in invalid_lines for line in text.splitlines())
                        or any(eid not in known for eid in _CITATION.findall(text)) or _UNRESOLVED.search(text)):
                    targets[pid] = text
        # Consistency issues must point to exact original text. An overall
        # failed verdict alone never authorizes replacing the entire report.
        if "report_consistency" in failures:
            for correction in (consistency or {}).get("corrections", []):
                original = correction.get("original")
                pid = correction.get("paragraph_id")
                if isinstance(pid, str) and pid in paragraphs and original == paragraphs[pid]:
                    targets[pid] = original
                elif isinstance(original, str) and len(original) >= 12:
                    targets.update(locate_correction(paragraphs, original))
        if not targets or len(targets) > 8:
            raise ValueError("无法定位最多8个待修复段落；保留当前报告")
        preferred = set(_CITATION.findall("\n".join(targets.values())))
        # Include the checker's proposed replacement citations as well as those
        # already in the broken paragraphs. Never fall back to the ledger head.
        for correction in (consistency or {}).get("corrections", []):
            preferred.update(_CITATION.findall(str(correction.get("corrected") or "")))
        sources = bounded_sources(results, max_tokens=settings.REPORT_REPAIR_EVIDENCE_MAX_TOKENS,
            max_sources=max(settings.REPORT_REPAIR_SOURCE_MAX_COUNT, len(preferred)), preferred_ids=preferred)
        allowed = {source["evidence_id"] for source in sources}
        data = await self.json._json(
            '只修复指定段落，返回 {"patches":[{"paragraph_id":"p1","action":"replace或remove",'
            '"text":"单个正文段落","evidence_ids":["ev_..."],"reason":"修改依据"}]}。'
            "不要重写报告，不改标题或其他段落，不重新检索，不新增无依据的事实。"
            "replace必须选择sources中实际支持该段的证据ID；Python负责生成引用，不在text内手写引用。"
            "仅说明报告范围的无引用摘要可remove；无法找到依据的陈述可remove并说明原因。"
            "remove会影响交付覆盖，不能为了通过引用检查删除核心回答。每段最多2000字，不含换行、标题或代码块。",
            {"targets": targets, "sources": sources, "failures": failures, "consistency": consistency or {}})
        patches = data.get("patches")
        if not isinstance(patches, list) or not 1 <= len(patches) <= len(targets):
            raise ValueError("局部修复缺少有效段落补丁")
        edits, seen, removed_size = [], set(), 0
        for patch in patches:
            if not isinstance(patch, dict):
                raise ValueError("段落补丁必须为对象")
            pid, action, reason = patch.get("paragraph_id"), patch.get("action"), patch.get("reason")
            if pid not in targets or pid in seen or action not in {"replace", "remove"} or not isinstance(reason, str) or not reason.strip():
                raise ValueError("段落补丁包含未知目标、重复目标或无效动作")
            seen.add(pid)
            replacement = ""
            if action == "replace":
                text, ids = patch.get("text"), patch.get("evidence_ids")
                problems = []
                if not isinstance(text, str):
                    problems.append("text必须为字符串")
                else:
                    if not 1 <= len(text.strip()) <= 2000:
                        problems.append("text为空或超过2000字")
                    if "\n" in text or "\r" in text:
                        problems.append("text包含换行")
                    if text.lstrip().startswith(("#", ">", "```", "~~~")):
                        problems.append("text包含标题、引用块或代码块")
                    if has_internal_material(text):
                        problems.append("text包含内部控制信息")
                    if _CITATION.search(text) or _UNRESOLVED.search(text):
                        problems.append("text手写引用标记，应通过evidence_ids提供")
                if not isinstance(ids, list) or not ids:
                    problems.append("evidence_ids必须为非空数组")
                elif any(not isinstance(eid, str) or eid not in allowed for eid in ids):
                    problems.append("evidence_ids包含未提供或非法来源编号")
                if problems:
                    # Diagnose fields, never persist raw model prose or relax the gate.
                    raise ValueError(f"段落{pid}替换被拒绝：" + "；".join(problems))
                replacement = text.strip() + " " + " ".join(f"[{eid}]" for eid in dict.fromkeys(ids))
            else:
                removed_size += len(targets[pid])
            edits.append({"paragraph_id": pid, "original": targets[pid], "replacement": replacement, "reason": reason[:1000]})
        if removed_size > len(report) / 4:
            raise ValueError("删除范围超过报告四分之一；保留当前报告")
        # Locate exact occurrences monotonically; duplicated paragraph text must
        # not cause edits to hit the wrong occurrence.
        cursor, spans = 0, {}
        for pid, text in paragraphs.items():
            start = report.index(text, cursor)
            spans[pid] = (start, start + len(text))
            cursor = start + len(text)
        revised = report
        for edit in sorted(edits, key=lambda item: spans[item["paragraph_id"]][0], reverse=True):
            start, end = spans[edit["paragraph_id"]]
            revised = revised[:start] + edit["replacement"] + revised[end:]
        check = DeterministicVerifiers(run_id=run_id, source_mode=SourceMode(source_mode), report=revised, evidence=ledger)
        if not check.citation_integrity().passed or not check.claim_support().passed or has_internal_material(revised):
            raise ValueError("局部修复后仍存在引用问题；保留原报告")
        return revised, {"before_hash": hashlib.sha256(report.encode()).hexdigest(),
                         "after_hash": hashlib.sha256(revised.encode()).hexdigest(), "edits": edits,
                         "method": "targeted-paragraph-edits"}
