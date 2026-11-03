"""Semantic gate ownership: genuine errors block; advice never causes rewriting."""
import json

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.agents.multi_agent.reporter.consistency_checker import ConsistencyChecker
from deepresearch_agent.agents.multi_agent.reporter.review_policy import REVIEW_POLICY_VERSION
from deepresearch_agent.harness.contracts import ContractEvaluator

REPORT = "# 报告\n\n该配置可以在任何故障下保证消息不丢失。 [ev_a]\n\n该配置在断电时仍然可能丢失消息。 [ev_a]"
SOURCES = [{"evidence_id": "ev_a", "title": "故障边界", "text": "该配置在断电时仍然可能丢失消息。", "content_kind": "page_text", "truncated": False}]


class Model:
    model_name = "DeepSeek-test"
    def __init__(self, payload):
        self.payload, self.calls = payload, 0
    def invoke(self, messages, **kwargs):
        self.calls += 1
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


def issue(kind="source_conflict", **extra):
    return {"kind": kind, "paragraph_id": "p1",
        "description": "对故障边界的表述与给定资料不同", "evidence_ids": ["ev_a"], **extra}


def review(issues, *, sources=None, model_bool=False, corrections=None):
    model = Model({"is_consistent": model_bool, "issues": issues, "corrections": corrections or []})
    result = ConsistencyChecker(model).check(REPORT, json.dumps(sources or SOURCES, ensure_ascii=False))
    return result, model


def test_validated_fact_conflict_blocks_even_when_model_boolean_says_pass():
    result, _ = review([issue()], model_bool=True)
    assert not result.is_consistent and result.issues[0]["blocking"]
    assert result.policy_version == REVIEW_POLICY_VERSION
    assert result.issues[0]["report_quote"] == REPORT.split("\n\n")[1]
    assert result.issues[0]["source_quotes"][0]["text"] == SOURCES[0]["text"]


def test_internal_contradiction_uses_two_real_report_quotes():
    result, _ = review([issue("report_conflict", evidence_ids=[], counter_paragraph_id="p2")])
    assert not result.is_consistent


def test_editorial_warning_does_not_trigger_rewrite_despite_negative_model_boolean():
    patch = {"paragraph_id": "p1", "corrected": "建议压缩措辞。", "reason": "文风修改"}
    result, _ = review([issue("editorial", evidence_ids=[], description="建议合并重复表达")], corrections=[patch])
    assert result.is_consistent and not result.issues
    assert len(result.warnings) == 1 and not result.corrections
    assert result.model_is_consistent is False


@pytest.mark.parametrize("source_change", [{"truncated": True}, {"content_kind": "search_summary"}, {"content_kind": "unknown"}])
def test_missing_context_is_not_relabelled_as_a_proven_fact_error(source_change):
    result, _ = review([issue("unsupported_claim")], sources=[{**SOURCES[0], **source_change}])
    assert result.is_consistent and result.warnings[0]["kind"] == "insufficient_context"
    assert not result.issues


def test_complete_source_unsupported_claim_remains_blocking():
    result, _ = review([issue("unsupported_claim")])
    assert not result.is_consistent


@pytest.mark.parametrize("change", [
    {"paragraph_id": "p999"},
    {"evidence_ids": ["ev_unknown"]},
    {"source_quotes": [{"evidence_id": "ev_a", "text": "没有出现在来源中的一段虚构内容。"}]},
    {"evidence_ids": []},
])
def test_unanchored_accusation_cannot_become_either_a_blocker_or_a_silent_pass(change):
    model = Model({"is_consistent": False, "issues": [issue(**change)], "corrections": []})
    with pytest.raises(ValueError, match="已重试一次"):
        ConsistencyChecker(model).check(REPORT, json.dumps(SOURCES, ensure_ascii=False))
    assert model.calls == 2


def test_duplicate_findings_are_not_counted_or_repaired_twice():
    result, _ = review([issue(), issue()])
    assert len(result.issues) == 1


def test_distinct_findings_in_one_paragraph_are_not_lost_as_duplicates():
    result, _ = review([issue(), issue(description="同一段落还存在另一处独立事实冲突")])
    assert len(result.issues) == 2


@pytest.mark.asyncio
async def test_invalid_citation_and_missing_delivery_still_block_independently():
    report = "# 报告\n\n这是一段具有足够长度的研究结论。 [ev_a] [ev_fake]"
    verdict = await ContractEvaluator().evaluate(run_id="r", source_mode="web", report=report,
        evidence=[{"evidence_id": "ev_a", "source_mode": "web", "source_id": "https://example.org/a"}],
        consistency_passed=True, delivery_coverage={"passed": False})
    assert "citation_integrity" in verdict.failures and "delivery_coverage" in verdict.failures
    assert "claim_support" not in verdict.failures
    assert not next(c for c in verdict.checks if c.kind == "claim_support").required
