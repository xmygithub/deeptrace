from __future__ import annotations

import json
import hashlib
import re
from typing import Any

from deepresearch_agent.harness.contracts import ContractCheckData, SourceMode
from deepresearch_agent.harness.report_safety import has_internal_material, is_scope_description

_CITATION = re.compile(r"\[\^?(ev_[A-Za-z0-9_-]+)\]")
_UNRESOLVED_CITATION = re.compile(r"\[\^?(?:(?:result_|digest_)[A-Za-z0-9_-]+|证据ID[:：]\s*[^\]\n]+)\]")


class DeterministicVerifiers:
    version = "1.4"

    def __init__(self, *, run_id: str, source_mode: SourceMode, report: str, evidence: list[Any]):
        self.run_id = run_id
        self.source_mode = source_mode
        self.report = report or ""
        self.evidence = evidence
        self._by_id = {str(self._get(item, "evidence_id", "")): item for item in evidence}

    @staticmethod
    def _get(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _check(self, kind: str, passed: bool, *, expected: Any, observed: Any, explanation: str, required: bool = True, threshold: float | None = None) -> ContractCheckData:
        return ContractCheckData(
            check_id=f"contract_{self.run_id}_{kind}", run_id=self.run_id, kind=kind,
            required=required, threshold=threshold, verifier="deterministic", verifier_version=self.version,
            passed=passed, observed={"expected": expected, "actual": observed}, explanation=explanation,
        )

    def source_match(self) -> ContractCheckData:
        raw_modes = [self._get(item, "source_mode", "") for item in self.evidence]
        modes = [mode.value if hasattr(mode, "value") else str(mode) for mode in raw_modes]
        passed = all(mode == self.source_mode.value for mode in modes)
        return self._check("source_match", passed, expected=self.source_mode.value, observed=modes, explanation="全部证据必须匹配冻结的 Run source_mode")

    def min_evidence(self, minimum: int) -> ContractCheckData:
        count = len(self.evidence)
        return self._check("min_evidence", count >= minimum, expected=minimum, observed=count, threshold=float(minimum), explanation="有效 Evidence Ledger 记录数量")

    def citation_integrity(self) -> ContractCheckData:
        citations = _CITATION.findall(self.report)
        missing = sorted({citation for citation in citations if citation not in self._by_id})
        unresolved = sorted(set(_UNRESOLVED_CITATION.findall(self.report)))
        claim_lines = self._claim_lines()
        uncited = [line[:160] for line in claim_lines if not _CITATION.search(line)]
        passed = bool(citations) and not missing and not unresolved and not uncited
        scope = [line.strip() for line in self.report.splitlines() if is_scope_description(line.strip())]
        return self._check("citation_integrity", passed, expected="每条实质性主张就近引用当前 Run 证据", observed={"citations": citations, "missing": missing, "unresolved_citations": unresolved, "uncited_claims": uncited, "scope_descriptions": scope}, explanation="检查引用 ID 存在及逐段引用覆盖；纯报告范围说明无需网页引用；不代表来源在语义上支持主张")

    def required_section(self, required_sections: list[str]) -> ContractCheckData:
        headings = [match.group(1).strip() for match in re.finditer(r"(?m)^#{1,4}\s+(.+?)\s*$", self.report)]
        missing = [name for name in required_sections if not any(name.lower() in heading.lower() for heading in headings)]
        passed = bool(headings) and not missing and not has_internal_material(self.report)
        return self._check("required_section", passed, expected=required_sections or "至少一个非空 Markdown 标题", observed={"headings": headings, "missing": missing}, explanation="报告必需章节存在且非空")

    def claim_support(self) -> ContractCheckData:
        claims = self._claim_lines()
        supported = [line for line in claims if any(citation in self._by_id for citation in _CITATION.findall(line))]
        passed = bool(claims) and len(supported) == len(claims)
        details = [{"text": line[:2000], "truncated": len(line) > 2000,
                    "evidence_ids": list(dict.fromkeys(_CITATION.findall(line))),
                    "passed": any(eid in self._by_id for eid in _CITATION.findall(line))} for line in claims]
        return self._check("claim_support", passed, required=False, expected="每条实质性主张都有当前 Run 证据引用", observed={"claims": len(claims), "supported": len(supported), "claim_details": details}, explanation="引用覆盖统计参考；引用完整性统一负责硬校验，本项不重复触发修复，也不验证语义支持")

    def report_consistency(self, consistency_passed: bool | None, *, details=None) -> ContractCheckData:
        if details and details.get("report_hash") != hashlib.sha256(self.report.encode("utf-8")).hexdigest():
            details = {**details, "status": "not_checked", "error": "一致性检查结果属于其他报告版本"}
        if details and details.get("status") in {"error", "not_checked"}:
            result = self._check("report_consistency", False, expected="完成当前报告版本的一致性检查",
                observed=None, explanation="一致性检查尚未完成或执行异常；不等同于报告内容不一致")
            result.observed["details"] = details
            return result
        passed = consistency_passed is True and bool(self.report.strip()) and not has_internal_material(self.report)
        result = self._check("report_consistency", passed, expected=True, observed=consistency_passed, explanation="仅阻断已定位的实质语义错误；建议与当前材料不足另列，不代表所有事实已核验。具体方法与策略版本见详情")
        if details:
            result.observed["details"] = details
        return result

    def source_diversity(self) -> ContractCheckData:
        sources = set()
        cited = set(_CITATION.findall(self.report))
        for item in self.evidence:
            if str(self._get(item, "evidence_id", "")) not in cited:
                continue
            source_id = str(self._get(item, "source_id", ""))
            metadata = self._get(item, "metadata_json", "{}")
            if self.source_mode is SourceMode.WEB:
                try:
                    payload = metadata if isinstance(metadata, dict) else json.loads(metadata or "{}")
                except (TypeError, ValueError):
                    payload = {}
                sources.add(str(payload.get("domain") or source_id))
            else:
                sources.add(source_id)
        sources.discard("")
        limitation = bool(re.search(r"局限|限制|single[- ]source|limited", self.report, re.I))
        passed = len(sources) >= 2 or (bool(sources) and limitation)
        return self._check("source_diversity", passed, expected="至少两个独立来源，或明确披露单一来源局限", observed={"sources": sorted(sources), "limitation_disclosed": limitation}, explanation="来源不足不得伪造多样性")

    def evidence_card_coverage(self, coverage: dict[str, Any]) -> ContractCheckData:
        ledger_ids = set(coverage.get("ledger_ids") or [])
        card_ids = set(coverage.get("card_ids") or [])
        routed_ids = set(coverage.get("routed_ids") or [])
        processed_ids = set(coverage.get("processed_ids") or [])
        annex_ids = set(coverage.get("annex_ids") or [])
        missing = {
            "cards": sorted(ledger_ids - card_ids),
            "routed": sorted(ledger_ids - routed_ids),
            "processed": sorted(ledger_ids - processed_ids),
            "annex": sorted(ledger_ids - annex_ids),
        }
        exact_sets = (
            ledger_ids == card_ids == routed_ids == processed_ids == annex_ids
        )
        reported_card_count = int(coverage.get("card_count", len(card_ids)) or 0)
        passed = (
            bool(ledger_ids)
            and exact_sets
            and reported_card_count == len(ledger_ids)
            and all(not values for values in missing.values())
        )
        return self._check(
            "evidence_card_coverage",
            passed,
            expected="Ledger/Card/路由/处理/结构化证据索引的 Evidence ID 集合完全一致",
            observed={
                "ledger_count": len(ledger_ids),
                "card_count": reported_card_count,
                "unique_card_count": len(card_ids),
                "routed_count": len(routed_ids),
                "processed_count": len(processed_ids),
                "annex_count": len(annex_ids),
                "missing_by_stage": missing,
            },
            explanation="全量 Evidence Card 保存在结构化证据台账中；正文只需引用支撑结论的证据",
        )

    def _claim_lines(self) -> list[str]:
        claims: list[str] = []
        excluded_section = False
        in_code = False
        for raw in self.report.splitlines():
            line = raw.strip()
            if line.startswith(("```", "~~~")):
                in_code = not in_code
                continue
            if in_code:
                continue
            if line.startswith("#"):
                title = re.sub(r"^#+\s*", "", line).strip()
                excluded_section = bool(re.fullmatch(r"局限(?:性|与尚未解决的问题)?|研究局限|限制|尚未解决的问题|未解决问题|资料范围与限制|证据引用|参考来源|references|limitations?", title, re.I))
                continue
            if line.startswith("> 运行说明："):
                continue
            if excluded_section or len(line) < 12 or line.startswith("|"):
                continue
            if re.fullmatch(r"基于证据[。.]?", line, re.I):
                continue
            if is_scope_description(line):
                continue
            claims.append(line)
        return claims
