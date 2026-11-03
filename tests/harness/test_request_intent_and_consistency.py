"""Regressions from the preference + web prompt-injection research Run."""
import hashlib
import json

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.harness.delivery import DeliveryService
from deepresearch_agent.harness.request_requirements import explicit_response_constraint, research_requirements
from deepresearch_agent.harness.report_reconciliation import reconciliation_candidates
from deepresearch_agent.agents.multi_agent.reporter.consistency_checker import ConsistencyChecker


QUESTION = "请记住：我长期偏好中文回答，技术问题先给结论，再解释原因。请调研研究型 Agent 如何防范检索网页中的提示词注入，比较提示词约束、内容隔离与工具权限控制的作用和局限，并引用来源。"
PREFERENCE = "我长期偏好中文回答，技术问题先给结论，再解释原因"
REPORT = "# 研究报告\n\n## 摘要\n\n结论：三种防护需要组合，单独依靠提示词不足以消除风险。[ev_a]\n\n## 原因\n\n提示词约束、内容隔离与工具权限控制分别作用于输入、数据边界和执行权限，需要互相配合。[ev_a]"


class Model:
    model_name = "DeepSeek-V4-Flash"

    def __init__(self, payload):
        self.payload, self.calls = payload, []

    async def ainvoke(self, messages, **options):
        self.calls.append((json.loads(messages[-1].content), options))
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


@pytest.mark.asyncio
async def test_exact_failed_run_preference_is_normalized_even_if_model_calls_it_a_fact():
    model = Model({"requirements": [
        {"kind": "fact", "text": PREFERENCE, "request_quote": PREFERENCE},
        {"kind": "analysis", "text": "比较三种防护手段", "request_quote": "比较提示词约束、内容隔离与工具权限控制的作用和局限"},
        {"kind": "fact", "text": "引用来源", "request_quote": "并引用来源"},
    ]})
    requirements = await DeliveryService(model).extract(QUESTION)
    assert [item["kind"] for item in requirements] == ["format", "analysis", "format"]
    assert [item["text"] for item in research_requirements(requirements)] == ["比较三种防护手段"]


@pytest.mark.parametrize("quote,expected", [
    (PREFERENCE, True), ("请用中文回答", True), ("先给结论，再解释原因", True),
    ("用表格呈现", True), ("并引用来源", True),
    ("比较中文回答和英文回答对检索质量的影响", False),
    ("请用中文分析网页提示词注入的风险", False),
    ("研究用户对先给结论再解释原因的偏好", False),
    ("分析 JSON 工具输出的安全问题", False),
])
def test_style_words_in_research_topics_are_not_filtered(quote, expected):
    assert explicit_response_constraint(quote) is expected


@pytest.mark.asyncio
async def test_memory_storage_request_does_not_become_a_report_obligation():
    question = "请记住：我的项目使用 SQLite。"
    model = Model({"requirements": [{"kind": "memory", "text": "保存项目事实", "request_quote": "我的项目使用 SQLite"}]})
    assert await DeliveryService(model).extract(question) == []
    model.payload["requirements"][0]["request_quote"] = "未提供的用户原话"
    with pytest.raises(ValueError, match="用户原文"):
        await DeliveryService(model).extract(question)


@pytest.mark.asyncio
async def test_actual_response_style_can_be_proven_by_its_opening_without_a_promise():
    requirements = [{"id": "req_1", "kind": "format", "text": PREFERENCE, "request_quote": PREFERENCE},
                    {"id": "req_2", "kind": "analysis", "text": "比较三类方法"}]
    model = Model({"items": [
        {"requirement_id": "req_1", "status": "covered", "paragraph_ids": ["p1"], "reason": "全文中文，开头先给结论，后续说明原因"},
        {"requirement_id": "req_2", "status": "covered", "paragraph_ids": ["p2"], "reason": "实质讨论三种方法"},
    ]})
    review = await DeliveryService(model).review(requirements, REPORT)
    assert review["passed"] and "结论：" in review["items"][0]["report_quote"]
    assert "承诺" not in REPORT
    # Research coverage still cannot be proven by an abstract alone.
    model.payload["items"][1]["paragraph_ids"] = ["p1"]
    with pytest.raises(ValueError, match="摘要"):
        await DeliveryService(model).review(requirements, REPORT)


@pytest.mark.asyncio
async def test_failed_style_is_not_waived_or_supplemented_with_web_facts():
    model = Model({"items": [{"requirement_id": "req_1", "status": "missing", "reason": "没有按语言要求输出"}]})
    service = DeliveryService(model)
    requirements = [{"id": "req_1", "kind": "format", "text": "中文回答"}]
    review = await service.review(requirements, "An English-only report.")
    assert review["passed"] is False
    assert await service.supplement(requirements, review, "An English-only report.", [{"evidence_id": "ev_a"}]) == ""
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_mixed_missing_requirements_only_supplement_research_content():
    model = Model({"sections": []})
    requirements = [{"id": "req_1", "kind": "format", "text": "先给结论"},
                    {"id": "req_2", "kind": "analysis", "text": "说明工具权限的局限"}]
    review = {"requirements": requirements, "items": [
        {"requirement_id": "req_1", "status": "partial"}, {"requirement_id": "req_2", "status": "missing"}]}
    await DeliveryService(model).supplement(requirements, review, REPORT, [])
    payload = model.calls[0][0]
    assert [item["id"] for item in payload["requirements"]] == ["req_2"]
    assert [item["requirement_id"] for item in payload["missing_items"]] == ["req_2"]
    assert payload["response_constraints"] == requirements[:1]


def test_format_pass_does_not_authorize_removing_research_limitations():
    report = REPORT + "\n\n## 局限\n\n- 尚缺真实攻击实验。"
    review = {"requirements": [{"id": "req_1", "kind": "format"}], "items": [
        {"requirement_id": "req_1", "status": "covered", "paragraph_ids": ["p2"]}]}
    assert reconciliation_candidates(report, review, [{"limitation_id": "lim_1", "requirement_id": "req_1",
        "paragraph_ids": ["p2"], "reason": "报告使用中文"}]) == []


class SyncModel:
    model_name = "DeepSeek-V4-Flash"

    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def invoke(self, messages, **options):
        self.calls.append((messages, options))
        return next(self.replies)


VERDICT = '{"is_consistent":true,"issues":[],"corrections":[]}'


def test_consistency_requests_json_and_disables_reasoning_then_retries_truncation():
    model = SyncModel([AIMessage(content=VERDICT, response_metadata={"finish_reason": "length"}), AIMessage(content=VERDICT)])
    result = ConsistencyChecker(model).check(REPORT, "[]")
    assert result.is_consistent and result.report_hash == hashlib.sha256(REPORT.encode()).hexdigest()
    assert len(model.calls) == 2
    from deepresearch_agent.config import settings
    assert [call[1]["max_tokens"] for call in model.calls] == [settings.CONSISTENCY_MAX_OUTPUT_TOKENS, settings.CONSISTENCY_RETRY_MAX_OUTPUT_TOKENS]
    assert all(options["response_format"] == {"type": "json_object"} for _, options in model.calls)
    assert all(options["extra_body"]["thinking"] == {"type": "disabled"} for _, options in model.calls)


@pytest.mark.parametrize("raw", ['{"is_consistent":"true","issues":[],"corrections":[]}', '{"issues":[]}',
    '{"is_consistent":true,"issues":{ },"corrections":[]}', '<think>example: '+VERDICT+'</think>', VERDICT+'\n'+VERDICT])
def test_invalid_or_ambiguous_consistency_output_never_becomes_a_pass(raw):
    model = SyncModel([AIMessage(content=raw), AIMessage(content=raw)])
    with pytest.raises(ValueError, match="已重试一次"):
        ConsistencyChecker(model).check(REPORT, "[]")
    assert len(model.calls) == 2


def test_checker_preserves_a_valid_negative_verdict_and_does_not_log_provider_text(caplog):
    bad = "sensitive provider text"
    payload = {"is_consistent": False, "issues": [{"kind": "source_conflict", "paragraph_id": "p1",
        "description": "报告与给定来源关于是否足够的表述冲突", "evidence_ids": ["ev_a"]}], "corrections": []}
    model = SyncModel([AIMessage(content=bad), AIMessage(content=json.dumps(payload, ensure_ascii=False))])
    sources = json.dumps([{"evidence_id": "ev_a", "text": "单独依靠提示词足以消除风险。"}], ensure_ascii=False)
    result = ConsistencyChecker(model).check(REPORT, sources)
    assert result.is_consistent is False and result.issues
    assert bad not in caplog.text
