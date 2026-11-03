"""Offline checks for evidence handoff, search admission and recovery."""
import copy
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.agents.multi_agent.core.plan_spec import (
    AcceptanceCriteria, PlanSpec, ProblemStatement, TaskGraph, TaskNode,
)
from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalMetadata, RetrievalResult
from deepresearch_agent.agents.multi_agent.core.state import PlanExecuteState
from deepresearch_agent.agents.multi_agent.executor.retrieval_executor import RetrievalExecutor
from deepresearch_agent.agents.multi_agent.executor.worker_coordinator import WorkerCoordinator
from deepresearch_agent.context.tokens import count_tokens
from deepresearch_agent.harness.contracts import SourceMode
from deepresearch_agent.harness.evidence import EvidenceLedger
from deepresearch_agent.harness.workflow import PlanExecuteReportDriver
from deepresearch_agent.retrieval.dependencies import dependency_evidence
from deepresearch_agent.retrieval.research_loop import BoundedResearchLoop, bounded_sources


def evidence(task="a", *, run="run1", mode="web", text="检查点保存中间状态，恢复可以保留已完成进度。"):
    item = RetrievalResult(granularity="DO", source="tavily_search" if mode == "web" else "hybrid_search",
        source_mode=mode, evidence=text,
        metadata=RetrievalMetadata(source_id=f"https://example.org/{task}", source_type="webpage"))
    EvidenceLedger().assign(run_id=run, task_id=task, tool_call_id=f"call_{task}", provider="offline", results=[item])
    return item


class Provider:
    mode, provider_name = SourceMode.WEB, "offline"

    def __init__(self):
        self.calls, self.steps = [], {}

    async def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        result = evidence(kwargs["call_context"].task_id, run=kwargs["call_context"].run_id,
                          text="重试需要幂等控制，避免重复副作用。")
        result.metadata.extra = {}  # The caller/Runtime assigns actual call identity.
        return [result]

    async def checkpoint(self, task_id, state):
        self.steps[task_id] = copy.deepcopy(state)

    async def load_step(self, task_id):
        return copy.deepcopy(self.steps.get(task_id, {}))


class Model:
    def __init__(self, *, followup=False, fail_phase=None):
        self.calls, self.followup, self.fail_phase = [], followup, fail_phase

    async def ainvoke(self, messages, **options):
        payload = json.loads(messages[-1].content)
        self.calls.append(payload)
        if payload["source_phase"] == self.fail_phase:
            raise ConnectionError("injected analysis failure")
        source = payload["sources"][0]
        more = self.followup and payload["source_phase"] == "dependencies"
        return AIMessage(content=json.dumps({"findings": [{"claim": source["text"],
            "evidence_id": source["evidence_id"], "quote": source["text"]}],
            "open_questions": ["缺少幂等控制说明"] if more else [],
            "next_query": "agent retry idempotency" if more else None}, ensure_ascii=False))


def test_only_completed_direct_dependency_evidence_from_same_run_is_selected():
    state = PlanExecuteState(input="研究", source_mode="web", run_id="run1")
    valid = evidence()
    state.execution_context.completed_task_ids = ["a", "other"]
    state.execution_context.retrieval_cache = {
        "a": [valid.to_dict(), valid.to_dict(), evidence(run="foreign").to_dict(), evidence(mode="graphrag").to_dict()],
        "pending": [evidence("pending").to_dict()], "other": [evidence("other").to_dict()]}
    task = TaskNode(task_id="child", task_type="deep_research", source_mode="web", description="比较", depends_on=["a", "pending"])
    selected = dependency_evidence(state, task)
    assert [item.result_id for item in selected] == [valid.result_id]
    selected[0].metadata.extra["task_id"] = "mutated"
    assert state.execution_context.retrieval_cache["a"][0]["metadata"]["extra"]["task_id"] == "a"


def test_multilevel_ancestry_shares_scenarios_without_unrelated_completed_tasks():
    nodes = [TaskNode(task_id=name, task_type="web_search", source_mode="web", description=name, depends_on=parents)
             for name, parents in [("a", []), ("b", ["a"]), ("c", ["a"]), ("d", ["a"]),
                                   ("e", ["b", "c", "d"]), ("f", ["e"]), ("unrelated", [])]]
    state = PlanExecuteState(input="比较", run_id="run1", source_mode="web",
        plan=PlanSpec(source_mode="web", problem_statement=ProblemStatement(original_query="比较"), task_graph=TaskGraph(nodes=nodes)))
    state.execution_context.completed_task_ids = ["a", "b", "c", "d", "e", "unrelated"]
    state.execution_context.retrieval_cache = {name: [evidence(name).to_dict()] for name in state.execution_context.completed_task_ids}
    for plan in [state.plan, state.plan.model_dump(mode="json")]:
        state.plan = plan
        selected = dependency_evidence(state, nodes[5])
        assert {item.metadata.extra["task_id"] for item in selected} == {"a", "b", "c", "d", "e"}
        assert len(dependency_evidence(state, nodes[5], limit=3)) == 3
    state.execution_context.completed_task_ids.remove("e")
    assert dependency_evidence(state, nodes[5]) == []


class ClassifiedModel(Model):
    def __init__(self, *, invalid_premise=False, needs_fact=False):
        super().__init__()
        self.invalid_premise, self.needs_fact = invalid_premise, needs_fact

    async def ainvoke(self, messages, **options):
        result = await super().ainvoke(messages, **options)
        data, payload = json.loads(result.content), self.calls[-1]
        source = payload["sources"][0]
        needs_fact = self.needs_fact and payload["source_phase"] == "dependencies"
        data["gaps"] = [{"kind": "missing_fact" if needs_fact else "synthesis", "question": "比较恢复与重试的适用条件",
            "query": "agent retry idempotency" if needs_fact else None,
            "evidence_ids": ["ev_not_provided" if self.invalid_premise else source["evidence_id"]]}]
        return AIMessage(content=json.dumps(data, ensure_ascii=False))


@pytest.mark.asyncio
@pytest.mark.parametrize("needs_fact", [False, True])
async def test_grounded_analysis_is_not_a_search_gap_and_real_missing_fact_is_targeted(needs_fact):
    provider, model = Provider(), ClassifiedModel(needs_fact=needs_fact)
    loop = BoundedResearchLoop(provider, run_id="run1", task_id="child", llm=model, seed_results=[evidence()])
    await loop.run("比较恢复与重试")
    assert [query for query, _ in provider.calls] == (["agent retry idempotency"] if needs_fact else [])
    assert loop.state["stop_reason"] == "ready_for_synthesis"
    assert loop.state["analysis_tasks"] and not loop.state["open_questions"]
    assert loop.state["fact_gaps"] == []


@pytest.mark.asyncio
async def test_synthesis_with_unknown_premises_remains_unresolved():
    provider = Provider()
    loop = BoundedResearchLoop(provider, run_id="run1", task_id="child", llm=ClassifiedModel(invalid_premise=True), seed_results=[evidence()])
    await loop.run("比较恢复与重试")
    assert loop.state["open_questions"] and not loop.state["analysis_tasks"]
    assert loop.state["stop_reason"] == "no_query" and not provider.calls


@pytest.mark.asyncio
async def test_report_does_not_append_historical_gap_after_final_analysis():
    from deepresearch_agent.retrieval.research_loop import classified_gaps
    with pytest.raises(ValueError):
        classified_gaps({"gaps": [{"kind": "invalid", "question": "未完成"}]}, [])
    model = Model()
    async def final_report(messages, **options):
        data = json.loads(messages[-1].content)
        assert data["open_questions"] == ["历史研究缺口"] and data["analysis_tasks"]
        return AIMessage(content=json.dumps({"sections": [{"title": "比较", "paragraphs": [
            {"text": "根据现有证据可比较恢复与重试的适用条件。", "evidence_ids": ["ev_a"]}]}],
            "limitations": ["尚未做实际故障注入实验。"]}))
    model.ainvoke = final_report
    loop = BoundedResearchLoop(Provider(), run_id="run1", task_id="child", llm=model)
    loop.state = {"findings": [{"evidence_id": "ev_a", "claim": "检查点保存状态", "quote": "保存状态"}],
                  "open_questions": ["历史研究缺口"], "analysis_tasks": [{"question": "比较"}]}
    report = await loop.write_report("比较恢复与重试")
    assert "历史研究缺口" not in report and "尚未做实际故障注入实验" in report


@pytest.mark.asyncio
async def test_sufficient_dependency_evidence_uses_no_search_and_preserves_provenance():
    provider, model, seed = Provider(), Model(), evidence()
    original = seed.model_dump(mode="json")
    loop = BoundedResearchLoop(provider, run_id="run1", task_id="child", llm=model, seed_results=[seed])
    answer = await loop.run("解释恢复机制")
    assert seed.result_id in answer and provider.calls == [] and loop.provider_calls == []
    assert model.calls[0]["source_phase"] == "dependencies"
    assert loop.state["stop_reason"] == "resolved" and not loop.state["open_questions"]
    assert loop.provider_results[0].model_dump(mode="json") == original
    # Completed reuse-only task survives reconstruction without any new calls.
    restored = BoundedResearchLoop(provider, run_id="run1", task_id="child", llm=Model(fail_phase="dependencies"))
    assert await restored.run("解释恢复机制") == answer
    assert provider.calls == []


@pytest.mark.asyncio
async def test_dependency_analysis_failure_recovers_frozen_input_before_any_search():
    provider, seed = Provider(), evidence()
    loop = BoundedResearchLoop(provider, run_id="run1", task_id="child",
        llm=Model(fail_phase="dependencies"), seed_results=[seed])
    with pytest.raises(ConnectionError):
        await loop.run("解释恢复机制")
    saved = provider.steps["child"]
    assert saved["pending_analysis"] and saved["analysis_origin"] == "dependencies" and not provider.calls
    model = Model()
    restored = BoundedResearchLoop(provider, run_id="run1", task_id="child", llm=model, seed_results=[evidence("changed")])
    await restored.run("解释恢复机制")
    assert model.calls[0]["sources"] == saved["dependency_sources"]
    assert restored.state["dependency_evidence_ids"] == [seed.result_id] and not provider.calls


@pytest.mark.asyncio
async def test_only_concrete_gap_is_searched_and_committed_search_is_reused_after_failure():
    provider = Provider()
    loop = BoundedResearchLoop(provider, run_id="run1", task_id="child", llm=Model(followup=True, fail_phase="search"),
                               seed_results=[evidence()])
    with pytest.raises(ConnectionError):
        await loop.run("解释恢复与重试")
    assert [query for query, _ in provider.calls] == ["agent retry idempotency"]
    saved = provider.steps["child"]
    assert saved["pending_analysis"] and saved["analysis_origin"] == "search"
    restored = BoundedResearchLoop(provider, run_id="run1", task_id="child", llm=Model())
    await restored.run("解释恢复与重试")
    assert len(provider.calls) == 1 and len(restored.state["findings"]) == 2
    assert {s["origin_task_id"] for s in bounded_sources(restored.provider_results)} == {"a", "child"}


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_coordinator_handoff_does_not_create_fake_tool_calls_or_rewrite_ledger(monkeypatch, mode):
    provider, model = Provider(), Model()
    monkeypatch.setattr("deepresearch_agent.retrieval.research_loop.get_llm_model", lambda: model)
    tasks = [TaskNode(task_id=name, task_type="web_search", source_mode="web", description=name) for name in ("a", "b")]
    tasks.append(TaskNode(task_id="child", task_type="deep_research", source_mode="web", description="比较恢复与重试", depends_on=["a", "b"]))
    plan = PlanSpec(source_mode="web", problem_statement=ProblemStatement(original_query="比较"),
                    task_graph=TaskGraph(nodes=tasks, execution_mode=mode), acceptance_criteria=AcceptanceCriteria())
    state = PlanExecuteState(input="比较", source_mode="web", run_id="run1", plan=plan)
    records = WorkerCoordinator(executors=[RetrievalExecutor(provider=provider)], execution_mode=mode,
                                max_parallel_workers=2).execute_plan(state, plan.to_execution_signal())
    child = next(record for record in records if record.task_id == "child")
    assert len(provider.calls) == 2 and child.tool_calls == [] and child.metadata.tool_calls_count == 0
    assert {item.metadata.extra["task_id"] for item in child.evidence} == {"a", "b"}
    assert state.execution_context.intermediate_results["child"]["raw_result"]["findings"]
    assert model.calls[0]["source_phase"] == "dependencies"
    driver = object.__new__(PlanExecuteReportDriver)
    driver.state = state
    driver.context = SimpleNamespace(context_snapshot={})
    origins = driver.evidence_results()
    assert len(origins) == 2 and {item[0] for item in origins} == {"a", "b"}
    assert all(item[1] == item[3].metadata.extra["tool_call_id"] for item in origins)


def test_inherited_prompt_packet_is_bounded_and_ids_remain_original():
    seeds = [evidence(str(i), text="长来源内容。" * 5000) for i in range(12)]
    sources = bounded_sources(seeds)
    from deepresearch_agent.config import settings
    assert sources and sum(count_tokens(json.dumps(s, ensure_ascii=False)) for s in sources) <= settings.RESEARCH_SOURCE_MAX_TOKENS
    assert count_tokens(json.dumps(sources, ensure_ascii=False)) <= settings.RESEARCH_SOURCE_MAX_TOKENS
    assert {s["evidence_id"] for s in sources} <= {item.result_id for item in seeds}
    assert all(s["text"] in seeds[int(s["origin_task_id"])].evidence for s in sources)
    assert count_tokens(json.dumps(bounded_sources(seeds, max_tokens=3500), ensure_ascii=False)) <= 3500


@pytest.mark.asyncio
async def test_diagnostic_gap_is_not_sent_as_a_search_query():
    class DiagnosticModel(Model):
        async def ainvoke(self, messages, **options):
            result = await super().ainvoke(messages, **options)
            data = json.loads(result.content)
            data.update(open_questions=["任务要求比较，但当前证据尚未覆盖这些具体场景。"], next_query=None)
            return AIMessage(content=json.dumps(data, ensure_ascii=False))
    provider = Provider()
    loop = BoundedResearchLoop(provider, run_id="run1", task_id="child", llm=DiagnosticModel(), seed_results=[evidence()])
    await loop.run("比较恢复与重试")
    assert not provider.calls and loop.state["stop_reason"] == "no_query"
    assert loop.state["open_questions"]
