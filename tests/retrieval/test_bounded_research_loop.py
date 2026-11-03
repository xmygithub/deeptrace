import copy
import json

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult, RetrievalMetadata
from deepresearch_agent.context.resolver import QueryResolver
from deepresearch_agent.harness.contracts import RunStatus, SourceMode
from deepresearch_agent.harness.state_machine import StateMachine
from deepresearch_agent.retrieval.research_loop import BoundedResearchLoop


class Provider:
    mode = SourceMode.WEB
    provider_name = "fake_web"

    def __init__(self):
        self.calls, self.steps = [], {}

    async def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [RetrievalResult(granularity="DO", source="tavily_search", source_mode="web",
            evidence="工具调用让 Agent 获取外部信息。检索结果需要进行核验。",
            metadata=RetrievalMetadata(source_id=f"https://example.com/{query}", source_type="webpage"))]

    async def checkpoint(self, task_id, state):
        self.steps[task_id] = copy.deepcopy(state)

    async def load_step(self, task_id):
        return copy.deepcopy(self.steps.get(task_id, {}))


class Model:
    model_name = "DeepSeek-V4-Flash"

    def __init__(self, *, fail=False, repeat=False):
        self.calls = []
        self.fail, self.repeat = fail, repeat

    async def ainvoke(self, messages, **kwargs):
        payload = json.loads(messages[-1].content)
        self.calls.append((payload, kwargs))
        if self.fail:
            raise ConnectionError("offline injected analysis failure")
        source = payload["sources"][0]
        assert source["evidence_id"].startswith("ev_")
        return AIMessage(content=json.dumps({"findings": [
            {"claim": "Agent 通过工具获取外部信息。", "evidence_id": source["evidence_id"], "quote": "工具调用让 Agent 获取外部信息。"},
            {"claim": "虚构的提升", "evidence_id": source["evidence_id"], "quote": "召回率提升 90%"},
            {"claim": "虚构来源", "evidence_id": "ev_fake", "quote": source["text"]},
        ], "next_query": payload["searched"][-1] if self.repeat else "补查问题", "open_questions": [] if self.repeat else ["还有待补查"]}, ensure_ascii=False))


@pytest.mark.parametrize("query", ["对于RAG多跳问题召回率低应该怎么办", "其他方法有哪些", "如何提高召回率", "什么是 AI Agent？"])
def test_standalone_questions_do_not_require_prior_session(query):
    assert QueryResolver().resolve(query, []).confidence == 1.0


def test_genuine_reference_still_requests_clarification_and_transition_is_legal():
    assert QueryResolver().resolve("继续解释这个方案", []).confidence < 0.5
    StateMachine().validate(RunStatus.CONTEXT_BUILDING, RunStatus.NEEDS_USER_INPUT)


@pytest.mark.asyncio
async def test_loop_limits_searches_validates_quotes_and_exposes_stable_ids():
    provider, model = Provider(), Model()
    loop = BoundedResearchLoop(provider, run_id="run1", task_id="task1", llm=model, max_searches=2,
                               context_text="已冻结的 Memory 与任务约束")
    progress = []
    async def on_progress(message):
        progress.append(message)
    loop.progress_callback = on_progress
    text = await loop.run("原始问题")
    assert len(provider.calls) == 2
    assert len(loop.provider_calls) == 2
    assert len(loop.state["findings"]) == 1
    assert loop.state["rejected_findings"] == 4
    assert "90%" not in text and "ev_fake" not in text
    assert model.calls[0][1]["extra_body"]["thinking"] == {"type": "disabled"}
    assert model.calls[0][0]["run_context"] == "已冻结的 Memory 与任务约束"
    assert "Memory" not in provider.calls[0][0]
    assert [item["iteration_index"] for item in progress if item["kind"] == "iteration_done"] == [0, 1]
    assert len({item.result_id for item in loop.provider_results}) == 2
    # New instance consumes the persisted completed loop without calling model/provider.
    restored = BoundedResearchLoop(provider, run_id="run1", task_id="task1", llm=Model(fail=True))
    assert await restored.run("原始问题") == text
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_analysis_failure_resumes_saved_search_instead_of_searching_again():
    provider = Provider()
    loop = BoundedResearchLoop(provider, run_id="run1", task_id="task1", llm=Model(fail=True))
    with pytest.raises(ConnectionError):
        await loop.run("问题")
    assert provider.steps["task1"]["pending_analysis"]
    restored = BoundedResearchLoop(provider, run_id="run1", task_id="task1", llm=Model(repeat=True))
    await restored.run("问题")
    assert len(provider.calls) == 1
    assert len(restored.state["findings"]) == 1


@pytest.mark.asyncio
async def test_no_sources_does_not_call_model_or_invent_findings():
    provider = Provider()
    async def empty(*args, **kwargs):
        return []
    provider.search = empty
    model = Model(fail=True)
    loop = BoundedResearchLoop(provider, run_id="run1", task_id="task1", llm=model)
    assert await loop.run("问题") == ""
    assert not model.calls and not loop.state["findings"]


@pytest.mark.asyncio
async def test_report_protocol_repair_reuses_findings_without_retrieval():
    class ReportModel:
        model_name = "DeepSeek-V4-Flash"
        def __init__(self):
            self.calls = 0
        async def ainvoke(self, messages, **kwargs):
            self.calls += 1
            ids = [] if self.calls == 1 else ["ev_saved"]
            return AIMessage(content=json.dumps({"sections": [{"title": "发现", "paragraphs": [
                {"text": "工具调用能够获取外部信息。", "evidence_ids": ids}]}]}))
    provider, model = Provider(), ReportModel()
    loop = BoundedResearchLoop(provider, run_id="r1", task_id="t1", llm=model)
    loop.state = {"findings": [{"claim": "工具调用获取外部信息。", "quote": "原始文本。", "evidence_id": "ev_saved"}]}
    report = await loop.write_report("解释工具调用")
    assert "[ev_saved]" in report
    assert model.calls == 2 and provider.calls == []


@pytest.mark.asyncio
async def test_research_follows_gaps_beyond_old_two_search_limit():
    class GapProvider(Provider):
        async def search(self, query, **kwargs):
            result = (await super().search(query, **kwargs))[0]
            result.evidence = f"{query}的独立证据。"
            return [result]

    class GapModel:
        async def ainvoke(self, messages, **kwargs):
            payload = json.loads(messages[-1].content)
            source = payload["sources"][0]
            count = len(payload["searched"])
            return AIMessage(content=json.dumps({"findings": [{"claim": source["text"], "quote": source["text"], "evidence_id": source["evidence_id"]}],
                "open_questions": [f"子问题{count + 1}"] if count < 4 else [], "next_query": None}))

    provider = GapProvider()
    loop = BoundedResearchLoop(provider, run_id="r1", task_id="t1", llm=GapModel(), max_searches=6)
    await loop.run("子问题1")
    assert len(provider.calls) == 4
    assert loop.state["stop_reason"] == "resolved" and not loop.state["open_questions"]


@pytest.mark.asyncio
async def test_known_research_gaps_cannot_pass_detailed_completion():
    from deepresearch_agent.harness.contracts import ContractEvaluator
    verdict = await ContractEvaluator().evaluate(run_id="r1", source_mode="web", report="# 报告", evidence=[],
        strict_research_coverage=True,
        research_coverage={"open_questions": ["尚未验证成本约束"], "stops": [{"reason": "search_limit"}]})
    assert "research_coverage" in verdict.failures
    assert not verdict.recoverable
    assert next(check for check in verdict.checks if check.kind == "research_coverage").observed["open_questions"]


def test_detailed_report_renderer_does_not_silently_drop_sections():
    from deepresearch_agent.agents.multi_agent.reporter.citation_renderer import render_research_report
    payload = {"sections": [{"title": f"主题{i}", "paragraphs": [{"text": f"研究发现{i}", "evidence_ids": ["ev_a"]}]} for i in range(6)]}
    report = render_research_report(payload, {"ev_a"}, max_sections=6, max_paragraphs=3)
    assert report.count("## 主题") == 6 and "研究发现5" in report


def test_committed_evidence_ids_survive_same_source_deduplication():
    from deepresearch_agent.agents.multi_agent.tools.evidence_tracker import EvidenceTracker
    first = RetrievalResult(result_id="ev_first", granularity="DO", source="tavily_search", source_mode="web",
        evidence="第一条已提交观察", metadata=RetrievalMetadata(source_id="https://example.com/a", source_type="webpage"))
    second = first.model_copy(deep=True, update={"result_id": "ev_second", "evidence": "第二条已提交观察"})
    tracker = EvidenceTracker()
    assert [item.result_id for item in tracker.register([first, second])] == ["ev_first", "ev_second"]
    assert len(tracker.all_results()) == 2
    assert tracker.lookup("ev_first").evidence == "第一条已提交观察"
