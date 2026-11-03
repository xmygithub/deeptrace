"""Regressions extracted from the 2026-10-30 PER development trials."""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.agents.multi_agent.reporter.outline_builder import (
    OutlineBuilder, ReportOutline, SectionOutline, enforce_outline_policy, normalize_report_type,
)
from deepresearch_agent.agents.multi_agent.reporter.base_reporter import BaseReporter, SectionContent
from deepresearch_agent.agents.multi_agent.reporter.section_writer import SectionWriter
from deepresearch_agent.agents.multi_agent.reporter.citation_renderer import CitationBindingError
from deepresearch_agent.harness.report_repair import locate_correction


def outline():
    return ReportOutline(report_type="long_document", title="比较", abstract="未经证据绑定的摘要结论",
        sections=[SectionOutline(section_id=f"s{i}", title=f"主题{i}", summary=f"要求{i}", evidence_ids=[f"ev_{i}"], estimated_words=900) for i in range(8)])


def test_brief_cannot_silently_become_long_report_or_drop_requirements():
    result = enforce_outline_policy(outline(), "brief")
    assert result.report_type == "short_answer" and len(result.sections) == 3
    assert result.total_estimated_words <= 1200
    assert {eid for section in result.sections for eid in section.evidence_ids} == {f"ev_{i}" for i in range(8)}
    assert all(f"要求{i}" in result.sections[-1].summary for i in range(2, 8))


def test_outline_model_cannot_override_request():
    obj = OutlineBuilder(llm=object())
    prompts = []
    obj._invoke_llm = lambda prompt: prompts.append(prompt) or outline().model_dump_json()
    result = obj.build_outline(query="比较", plan_summary="要求", evidence_summary="证据", evidence_count=8, report_type="brief")
    assert result.report_type == "short_answer" and len(result.sections) == 3
    assert "short_answer" in prompts[0]
    assert len(enforce_outline_policy(outline(), "long_document").sections) == 8
    with pytest.raises(ValueError):
        normalize_report_type("unrecognized")


def test_uncited_planning_abstract_is_not_delivered():
    reporter = BaseReporter.__new__(BaseReporter)
    result = reporter._assemble_report(outline(), [SectionContent(section_id="s0", title="主题", content="已有正文与合法引用 [ev_0]", used_evidence_ids=["ev_0"])])
    assert "未经证据绑定" not in result
    assert "已有正文与合法引用 [ev_0]" in result


def test_two_invalid_section_responses_produce_bounded_diagnostics():
    writer = SectionWriter(llm=SimpleNamespace(model_name="deepseek-test"))
    calls = []
    writer._call_model = lambda messages, options: calls.append(messages) or AIMessage(content='{"paragraphs":[{"text":"缺少编号的正文"}]}')
    with pytest.raises(CitationBindingError) as exc:
        writer._invoke_llm("写章节", allowed_evidence_ids={"ev_a"})
    assert len(calls) == 2
    assert len(exc.value.protocol_diagnostics) == 2
    assert exc.value.protocol_diagnostics[0]["response_chars"] > 0
    assert "缺少编号的正文" not in json.dumps(exc.value.protocol_diagnostics, ensure_ascii=False)


def test_correction_localization_normalizes_only_citations_and_whitespace():
    text = "检查点保存已完成操作的状态，恢复后读取已有结果。 [ev_a]"
    quote = "检查点保存已完成操作的状态，\n恢复后读取已有结果。"
    assert locate_correction({"p1": text}, quote) == {"p1": text}
    assert not locate_correction({"p1": text, "p2": text}, quote)
    assert not locate_correction({"p1": text}, "语义相关但没有出现在报告中的另一句话")


@pytest.mark.asyncio
async def test_repair_rejection_emits_durable_diagnostic():
    from deepresearch_agent.harness.workflow import PlanExecuteReportDriver
    obj = PlanExecuteReportDriver.__new__(PlanExecuteReportDriver)
    obj.context = SimpleNamespace(run_id="run_test", context_snapshot={})
    obj.report_result = None
    calls = []
    async def publish(*args, **kwargs):
        calls.append((args, kwargs))
    obj.events = SimpleNamespace(publish=publish)
    assert not await obj.repair_report(["report_consistency"])
    assert calls[0][0] == ("run_test", "report.repair_failed")
    assert calls[0][1]["payload"]["reason"] == "缺少可修复的报告"
