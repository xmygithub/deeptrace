"""Citation regression tests; no API calls or application database writes."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from deepresearch_agent.agents.multi_agent.reporter.base_reporter import BaseReporter
from deepresearch_agent.agents.multi_agent.reporter.evidence_cards import EvidenceCardPipeline
from deepresearch_agent.agents.multi_agent.reporter.outline_builder import ReportOutline, SectionOutline
from deepresearch_agent.agents.multi_agent.reporter.section_writer import SectionWriter
from deepresearch_agent.harness.contracts import SourceMode
from deepresearch_agent.harness.verifiers.deterministic import DeterministicVerifiers
from deepresearch_agent.harness.workflow import DeepResearchDriver, PlanExecuteReportDriver


def result(evidence_id, title, content):
    return RetrievalResult(
        result_id=evidence_id, granularity="Chunk", evidence=content,
        metadata={"source_id": f"https://example.org/{evidence_id}",
                  "source_type": "webpage", "title": title,
                  "url": f"https://example.org/{evidence_id}"},
        source="tavily_search", source_mode="web",
    )


def test_protocol_failure_is_only_retained_as_draft_when_delivery_gate_is_enabled():
    from deepresearch_agent.agents.multi_agent.reporter.citation_renderer import CitationBindingError
    reporter = object.__new__(BaseReporter)
    reporter._cache_manager = None
    reporter._evidence_card_pipeline = EvidenceCardPipeline()
    def fail(**kwargs):
        raise CitationBindingError("没有合法引用")
    reporter._section_writer = SimpleNamespace(write_section=fail)
    item = result("ev_real", "证据", "一条可核验的来源文本。")
    cards, _ = reporter._evidence_card_pipeline.build_cards([item])
    outline = ReportOutline(title="报告", report_type="long_document", sections=[SectionOutline(
        section_id="s1", title="验证方案", summary="验证方法", evidence_ids=[item.result_id])])
    kwargs = dict(report_id="partial-test", outline=outline, cards=cards, section_cache_index={},
                  evidence_fingerprint=reporter._build_evidence_fingerprint({item.result_id: item}))
    with pytest.raises(CitationBindingError):
        reporter._generate_sections_from_cards(**kwargs)
    sections, _, _ = reporter._generate_sections_from_cards(**kwargs, allow_partial_sections=True)
    assert "未完成引用绑定" in sections[0].content
    assert "[ev_real]" not in sections[0].content


class RecordingLLM:
    def __init__(self, response):
        self.response = response
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return SimpleNamespace(content=self.response)


def test_writer_sees_id_title_url_and_content_beyond_old_200_char_cutoff():
    writer = SectionWriter(llm=RecordingLLM("test"))
    item = result("ev_business", "商业代理", "背景信息" * 800 + "关键事实：代表委托人处理事务。" + "x" * 12000)
    formatted = writer._format_evidence([item])
    record = json.loads(formatted)
    assert record["evidence_id"] == "ev_business"
    assert record["title"] == "商业代理"
    assert record["url"] == item.metadata.url
    assert "关键事实：代表委托人处理事务。" in record["content"]
    assert "已截断" in record["content"]
    assert len(record["content"]) < 8100


def test_digest_is_not_presented_as_first_source_and_retains_per_claim_identity():
    pipeline = EvidenceCardPipeline()
    cards, _ = pipeline.build_cards([
        result("ev_language", "施事", "语言学的 Agent 是动作执行者。"),
        result("ev_business", "商业代理", "商业的 Agent 代表委托人处理事务。"),
    ])
    digest = pipeline._cards_digest(cards, index=0, level=1)
    item = pipeline._digest_result(digest, cards)
    assert item.result_id == digest.digest_id
    assert item.metadata.source_id == digest.digest_id
    assert item.metadata.url is None
    formatted = SectionWriter(llm=RecordingLLM("test"))._format_evidence([item])
    record = json.loads(formatted.split("\n", 1)[1])
    assert "不可引用" in formatted
    assert {source["evidence_id"] for source in record["sources"]} == {"ev_language", "ev_business"}
    assert "[ev_language]" in record["content"]
    assert "[ev_business]" in record["content"]

    # An ID still listed as covered, but absent from the truncated summary,
    # cannot be presented as evidence actually shown to the writer.
    truncated = item.model_copy(update={"evidence": "[ev_language] 仅保留语言学信息。"})
    formatted = SectionWriter(llm=RecordingLLM("test"))._format_evidence([truncated])
    assert "ev_business" not in formatted


def test_card_report_path_normalizes_explicit_citation_but_keeps_uncited_failure():
    llm = RecordingLLM("商业代理代表委托人处理事务[result_business]。\n\n这是一个缺少材料支持的完整结论。")
    reporter = object.__new__(BaseReporter)
    reporter._cache_manager = None
    reporter._section_writer = SectionWriter(llm=llm)
    reporter._evidence_card_pipeline = EvidenceCardPipeline()
    item = result("ev_business", "商业代理", "商业的 Agent 代表委托人处理事务。")
    cards, _ = reporter._evidence_card_pipeline.build_cards([item])
    section = SectionOutline(section_id="s1", title="其他领域", summary="商业代理", evidence_ids=[item.result_id])
    outline = ReportOutline(title="Agent", report_type="short_answer", sections=[section])
    sections, processed, _ = reporter._generate_sections_from_cards(
        report_id="test", outline=outline, cards=cards, section_cache_index={},
        evidence_fingerprint=reporter._build_evidence_fingerprint({item.result_id: item}),
    )
    assert processed == [item.result_id]
    assert sections[0].content.count("[ev_business]") == 1
    assert "[result_business]" not in sections[0].content
    assert "title" in llm.messages[0][1].content
    assert "不得改成" in llm.messages[0][0].content
    verifier = DeterministicVerifiers(
        run_id="test", source_mode=SourceMode.WEB, report=sections[0].content,
        evidence=[{"evidence_id": item.result_id}],
    )
    assert verifier.citation_integrity().passed is False
    assert len(verifier.citation_integrity().observed["actual"]["uncited_claims"]) == 1


def test_cache_fingerprint_invalidates_old_policy_and_changed_evidence_text():
    reporter = object.__new__(BaseReporter)
    item = result("ev_business", "商业代理", "商业代理原始内容。")
    old_payload = {"granularity": item.granularity, "source": item.source,
                   "score": item.score, "metadata": item.metadata.model_dump(mode="json")}
    old = hashlib.sha1(json.dumps(old_payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    current = reporter._build_evidence_fingerprint({item.result_id: item})
    assert current[item.result_id] != old
    changed = item.model_copy(update={"evidence": "更正后的内容。"})
    assert current != reporter._build_evidence_fingerprint({changed.result_id: changed})
    section = SectionOutline(section_id="s1", title="商业", summary="代理", evidence_ids=[item.result_id])
    assert reporter._can_reuse_section(section, {
        "title": "商业", "summary": "代理", "used_evidence_ids": [item.result_id],
        "evidence_fingerprint": {item.result_id: old},
    }, current) is False


def test_structured_section_renders_only_model_selected_current_sources():
    class Model:
        model_name = "DeepSeek-V4-Flash"
        def invoke(self, messages, **kwargs):
            assert kwargs["response_format"]["type"] == "json_object"
            return SimpleNamespace(content=json.dumps({"paragraphs": [
                {"text": "商业代理代表委托人处理相关事务。", "evidence_ids": ["ev_business"]}
            ]}), response_metadata={})
    item = result("ev_business", "商业代理", "商业代理代表委托人处理事务。")
    section = SectionOutline(section_id="s1", title="商业代理", summary="定义", evidence_ids=[item.result_id])
    draft = SectionWriter(llm=Model()).write_section(
        ReportOutline(title="研究", report_type="brief", sections=[section]), section, {item.result_id: item})
    assert draft.content == "商业代理代表委托人处理相关事务。 [ev_business]"


def test_structured_renderer_rejects_missing_unknown_and_digest_citations():
    from deepresearch_agent.agents.multi_agent.reporter.citation_renderer import render_cited_paragraphs
    for ids in ([], ["ev_unknown"], ["digest_1"]):
        with pytest.raises(ValueError):
            render_cited_paragraphs([{"text": "完整结论。", "evidence_ids": ids}], {"ev_real"})


def test_renderer_normalizes_explicit_same_source_without_inventing_references():
    from deepresearch_agent.agents.multi_agent.reporter.citation_renderer import render_cited_paragraphs
    assert render_cited_paragraphs([{"text": "完整结论。", "evidence_ids": [" [ev_real] "]}], {"ev_real"}) == "完整结论。 [ev_real]"
    assert render_cited_paragraphs([{"text": "完整结论。 [ev_real]"}], {"ev_real"}) == "完整结论。 [ev_real]"
    with pytest.raises(ValueError):
        render_cited_paragraphs([{"text": "没有任何明确来源的完整结论。"}], {"ev_real"})
    with pytest.raises(ValueError):
        render_cited_paragraphs([{"text": "完整结论。 [ev_fake]", "evidence_ids": ["ev_real"]}], {"ev_real"})


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_section_retries_only_the_bad_protocol_once_using_same_evidence(repair_succeeds):
    class Model:
        model_name = "DeepSeek-V4-Flash"
        def __init__(self):
            self.calls = []
        def invoke(self, messages, **kwargs):
            self.calls.append(messages)
            if len(self.calls) == 2:
                assert messages[1].content == self.calls[0][1].content
                assert "不重新检索" in messages[-1].content
                assert "ev_real" in messages[-1].content
            ids = ["ev_real"] if len(self.calls) == 2 and repair_succeeds else ["ev_wrong"]
            return SimpleNamespace(content=json.dumps({"paragraphs": [{"text": "有依据的结论。", "evidence_ids": ids}]}), response_metadata={})
    model = Model()
    writer = SectionWriter(llm=model)
    if repair_succeeds:
        assert writer._invoke_llm("same evidence", allowed_evidence_ids={"ev_real"}) == "有依据的结论。 [ev_real]"
    else:
        with pytest.raises(ValueError, match="证据集合之外"):
            writer._invoke_llm("same evidence", allowed_evidence_ids={"ev_real"})
    assert len(model.calls) == 2


def test_runtime_note_and_code_are_not_facts_but_method_claims_are_checked():
    report = "# 报告\n\n> 运行说明：执行阶段已接近预留边界，以下按已保存证据生成摘录。\n\n## 方法\n\n这是一个缺少引用但包含实质主张的完整方法说明。\n\n```python\nprint('无需作为事实进行引用检查的示例代码')\n```\n\n这是有来源支撑的完整研究结论。 [ev_real]"
    verifier = DeterministicVerifiers(run_id="r", source_mode=SourceMode.WEB, report=report, evidence=[{"evidence_id": "ev_real"}])
    uncited = verifier.citation_integrity().observed["actual"]["uncited_claims"]
    assert len(uncited) == 1 and "方法说明" in uncited[0]


def test_checkpoint_roundtrip_restores_both_evidence_tracker_indexes():
    from deepresearch_agent.agents.multi_agent.core.state import PlanExecuteState
    from deepresearch_agent.agents.multi_agent.tools.evidence_tracker import get_evidence_tracker
    state = PlanExecuteState(input="研究", source_mode="web")
    item = result("ev_real", "来源", "已保存的原始证据内容。")
    get_evidence_tracker(state).register([item])
    restored = PlanExecuteState.model_validate_json(state.model_dump_json())
    tracker = get_evidence_tracker(restored)
    assert tracker.all_results()[0].result_id == "ev_real"
    assert tracker.lookup("ev_real").metadata.source_id == item.metadata.source_id
    assert tracker.register([item])[0].result_id == "ev_real"


@pytest.mark.asyncio
async def test_both_drivers_normalize_from_all_evidence_without_authority_or_top_k_remap():
    items = [result(f"ev_{index}", f"来源 {index}", "测试用来源内容。") for index in range(7)]
    items[0].metadata.extra = {"authority_rank": 3}
    original = "这是明确引用第七条证据的完整结论[result_6]。"
    deep = object.__new__(DeepResearchDriver)
    deep._repair_pending = False
    deep.results = items
    deep.answer = original
    assert "[ev_6]" in await deep.report()
    assert "[ev_0]" not in deep._report

    per = object.__new__(PlanExecuteReportDriver)
    per._repair_pending = True
    per.report_result = SimpleNamespace(final_report=original)
    per.state = SimpleNamespace(response=None)
    per.evidence_results = lambda: [(None, None, None, item) for item in items]
    assert await per.report() == original.replace("result_6", "ev_6")
