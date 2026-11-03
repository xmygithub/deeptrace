from deepresearch_agent.harness.contracts import SourceMode
from types import SimpleNamespace

from deepresearch_agent.harness.report_safety import add_inline_citations, citation_evidence_ids, has_internal_material, sanitize_report
from deepresearch_agent.harness.verifiers.deterministic import DeterministicVerifiers


def evidence(evidence_id: str, source_id: str = "doc-1") -> dict:
    return {
        "evidence_id": evidence_id,
        "source_mode": "graphrag",
        "source_id": source_id,
        "metadata_json": "{}",
    }


def test_sanitize_report_removes_reasoning_and_internal_contract() -> None:
    raw = "<think># SYSTEM CONTRACT\nsecret\n</think>\n# 深度研究报告\n\n可见结论。"
    cleaned = sanitize_report(raw)
    assert cleaned == "# 深度研究报告\n\n可见结论。"
    assert not has_internal_material(cleaned)


def test_inline_citations_never_assign_sources_to_uncited_claims() -> None:
    original = "# 报告\n\n这是一个需要证据支持的完整结论。"
    report = add_inline_citations(original, ["ev_a", "ev_b"])
    assert report == original
    verifier = DeterministicVerifiers(
        run_id="run-1", source_mode=SourceMode.GRAPHRAG,
        report=report, evidence=[evidence("ev_a"), evidence("ev_b")],
    )
    assert verifier.citation_integrity().passed is False


def test_inline_citations_do_not_decorate_limitation_section() -> None:
    report = add_inline_citations("# 报告\n\n这是一个需要证据支持的完整结论。\n\n## 局限\n\n当前材料有限，不能覆盖全部细节。", ["ev_a"])
    assert "完整结论。 [ev_a]" not in report
    assert "覆盖全部细节。 [ev_a]" not in report


def test_normalize_only_same_identity_and_preserve_unknown_references() -> None:
    original = "已知[result_b]、脚注[^result_a]、中文[证据ID: ev_b]、未知[result_unknown]、原引用[ev_c]。"
    report = add_inline_citations(original, ["ev_a", "ev_b", "ev_c"])
    assert report == "已知[ev_b]、脚注[^ev_a]、中文[ev_b]、未知[result_unknown]、原引用[ev_c]。"
    assert add_inline_citations(report, ["ev_a", "ev_b", "ev_c"]) == report


def test_historical_agent_paragraph_does_not_get_an_arbitrary_linguistics_citation() -> None:
    linguistic = "ev_354d8bd9d94b64cd346791f5"
    business = "ev_d411176970e68b443626fc86"
    original = "商业中的 Agent 指受委托执行事务的代理人[result_d411176970e68b443626fc86]。\n\n化学中 Agent 的用法仍需要补充相关来源。"
    report = add_inline_citations(original, [linguistic, business])
    assert report == original.replace("result_d411176970e68b443626fc86", business)
    assert linguistic not in report
    assert report.endswith("化学中 Agent 的用法仍需要补充相关来源。")


def test_verifier_rejects_unresolved_citation_even_beside_valid_one() -> None:
    for marker in ("[result_unknown]", "[digest_l1_0]", "[证据ID: ev_unknown]"):
        verifier = DeterministicVerifiers(
            run_id="run-1", source_mode=SourceMode.GRAPHRAG,
            report=f"# 报告\n\n这是带有未解析来源的完整结论 [ev_a] {marker}",
            evidence=[evidence("ev_a")],
        )
        check = verifier.citation_integrity()
        assert check.passed is False
        assert check.observed["actual"]["unresolved_citations"] == [marker]


def test_citations_stay_with_strongest_authority_tier() -> None:
    official = SimpleNamespace(result_id="ev_official", score=0.7, metadata=SimpleNamespace(domain="docs.brand.com", extra={"authority_rank": 3}))
    secondary = SimpleNamespace(result_id="ev_secondary", score=0.9, metadata=SimpleNamespace(domain="docs.other.com", extra={"authority_rank": 2}))
    assert citation_evidence_ids([secondary, official]) == ["ev_official"]


def test_verifier_rejects_null_consistency_and_uncited_claims() -> None:
    report = "# 报告\n\n这是第一条有引用的完整结论 [ev_a]\n\n这是第二条没有引用的完整结论。\n\n## 局限\n\n来源有限。"
    verifier = DeterministicVerifiers(
        run_id="run-1", source_mode=SourceMode.GRAPHRAG,
        report=report, evidence=[evidence("ev_a")],
    )
    assert verifier.citation_integrity().passed is False
    assert verifier.claim_support().passed is False
    assert verifier.report_consistency(None).passed is False


def test_verifier_uses_only_cited_sources_for_diversity() -> None:
    report = "# 报告\n\n这是一个有来源支持的完整结论 [ev_a]"
    verifier = DeterministicVerifiers(
        run_id="run-1", source_mode=SourceMode.GRAPHRAG, report=report,
        evidence=[evidence("ev_a", "doc-1"), evidence("ev_b", "doc-2")],
    )
    assert verifier.source_diversity().passed is False
