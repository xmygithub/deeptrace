import json

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.harness.delivery import DeliveryService
from deepresearch_agent.agents.multi_agent.reporter.outline_builder import ReportOutline, SectionOutline, compact_outline


class JsonModel:
    model_name = "DeepSeek-V4-Flash"

    def __init__(self, payload):
        self.payload, self.calls = payload, []

    async def ainvoke(self, messages, **options):
        self.calls.append((json.loads(messages[-1].content), options))
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


@pytest.mark.asyncio
async def test_requirements_bound_to_original_request():
    model = JsonModel({"requirements": [{"text": "提出验证方案", "request_quote": "验证方案", "kind": "analysis"}]})
    requirements = await DeliveryService(model).extract("请比较方法并给出验证方案")
    assert requirements[0]["id"] == "req_1"
    assert model.calls[0][1]["extra_body"]["thinking"] == {"type": "disabled"}
    model.payload["requirements"][0]["request_quote"] = "用户没要求的内容"
    with pytest.raises(ValueError, match="用户原文"):
        await DeliveryService(model).extract("请比较方法并给出验证方案")


@pytest.mark.asyncio
@pytest.mark.parametrize("items", [[], [{"requirement_id": "req_1", "status": "covered", "report_quote": "虚构的报告依据", "reason": "声称已覆盖"}],
    [{"requirement_id": "req_1", "status": "covered", "report_quote": "## 验证方案和实施优先级的详细设计", "reason": "只有标题"}]])
async def test_review_rejects_missing_or_fabricated_body_evidence(items):
    requirements = [{"id": "req_1", "text": "给出验证方案", "request_quote": "验证方案", "kind": "analysis"}]
    with pytest.raises(ValueError):
        await DeliveryService(JsonModel({"items": items})).review(requirements, "## 验证方案和实施优先级的详细设计")


@pytest.mark.asyncio
async def test_review_uses_final_report_not_old_search_gaps():
    quote = "建议固定问题集，对比召回率、答案完整性及单次调用成本。"
    model = JsonModel({"items": [{"requirement_id": "req_1", "status": "covered", "report_quote": quote, "reason": "正文提供验证方法"}]})
    requirements = [{"id": "req_1", "text": "验证方法", "request_quote": "验证方法", "kind": "analysis"}]
    result = await DeliveryService(model).review(requirements, "# 报告\n\n" + quote)
    assert result["passed"] and result["requirements"] == requirements
    assert "open_questions" not in model.calls[0][0]


@pytest.mark.asyncio
async def test_supplement_cannot_invent_evidence_ids():
    model = JsonModel({"sections": [{"title": "验证方案", "paragraphs": [{"text": "建议用固定问题集验证。", "evidence_ids": ["ev_fake"]}]}]})
    with pytest.raises(ValueError):
        await DeliveryService(model).supplement([{"id": "req_1"}], {"items": [{"requirement_id": "req_1", "status": "missing"}]},
            "# 原报告", [{"evidence_id": "ev_real", "text": "可用证据"}])


@pytest.mark.asyncio
async def test_quote_is_copied_from_selected_body_paragraph():
    text = "建议固定问题集，对比召回率、答案完整性和调用成本。"
    model = JsonModel({"items": [{"requirement_id": "req_1", "status": "covered", "paragraph_ids": ["p1"], "reason": "提供了验证方法"}]})
    result = await DeliveryService(model).review([{"id": "req_1"}], "# 验证方案\n\n" + text)
    assert result["items"][0]["report_quote"] == text
    model.payload["items"][0]["paragraph_ids"] = ["p_fake"]
    with pytest.raises(ValueError, match="未知报告段落"):
        await DeliveryService(model).review([{"id": "req_1"}], text)


@pytest.mark.asyncio
async def test_operational_missing_section_note_cannot_prove_coverage():
    model = JsonModel({"items": [{"requirement_id": "req_1", "status": "covered", "paragraph_ids": ["p1"], "reason": "声称已完成"}]})
    with pytest.raises(ValueError, match="未知报告段落"):
        await DeliveryService(model).review([{"id": "req_1"}], "# 验证方案\n\n> 运行说明：本节未完成引用绑定。")


def test_outline_budget_merges_tail_without_dropping_verification():
    original = ReportOutline(report_type="long_document", title="研究", sections=[SectionOutline(
        section_id=f"s{i}", title=title, summary=f"交付要求{i}", evidence_ids=[f"ev_{i}"])
        for i, title in enumerate(["引言", "方法一", "方法二", "方法三", "比较", "优先级", "验证方案"])])
    compact = compact_outline(original, 6)
    assert len(compact.sections) == 6 and len(original.sections) == 7
    assert "验证方案" in compact.sections[-1].title
    assert set(compact.sections[-1].evidence_ids) == {"ev_5", "ev_6"}
    assert "交付要求6" in compact.sections[-1].summary


@pytest.mark.asyncio
async def test_short_comparison_checks_each_requested_scenario_and_exposes_missing_items():
    from deepresearch_agent.harness.delivery_policy import needs_delivery_check
    question = "比较检查点恢复与失败重试在工具超时、进程中断、重复调用下的区别。"
    assert len(question) < 100 and needs_delivery_check(question)
    model = JsonModel({"requirements": [
        {"text": f"比较{scenario}场景", "request_quote": scenario, "kind": "analysis"}
        for scenario in ("工具超时", "进程中断", "重复调用")]})
    service = DeliveryService(model)
    requirements = await service.extract(question)
    paragraph = "工具超时需要先检查调用结果，再决定是否重试，以避免重复执行。"
    model.payload = {"items": [
        {"requirement_id": "req_1", "status": "covered", "paragraph_ids": ["p1"], "reason": "正文比较了工具超时"},
        {"requirement_id": "req_2", "status": "missing", "reason": "没有讨论进程中断"},
        {"requirement_id": "req_3", "status": "missing", "reason": "没有讨论重复调用"}]}
    review = await service.review(requirements, "# 报告\n\n" + paragraph)
    assert not review["passed"]
    assert [i["requirement_id"] for i in review["items"] if i["status"] == "missing"] == ["req_2", "req_3"]


@pytest.mark.asyncio
async def test_disclosed_insufficient_evidence_is_not_misrepresented_as_covered():
    model = JsonModel({"items": [{"requirement_id": "req_1", "status": "insufficient_evidence",
        "paragraph_ids": ["p1"], "reason": "报告披露缺少项目案例，未实际完成该项比较"}]})
    review = await DeliveryService(model).review([{"id": "req_1", "text": "比较公开项目案例"}],
        "# 局限\n\n当前缺少公开项目的直接依据，无法完成具体案例比较。")
    assert not review["passed"] and review["items"][0]["status"] == "insufficient_evidence"
