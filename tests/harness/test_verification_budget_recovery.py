"""Budget admission and bounded recovery for the saved 40-source report."""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepresearch_agent.context.tokens import count_tokens
from deepresearch_agent.harness.budgets import BudgetLimits, BudgetUsage
from deepresearch_agent.harness.call_budget import CallBudget, estimate_model_reservation
from deepresearch_agent.harness.contracts import SourceMode
from deepresearch_agent.harness.delivery import DeliveryService, report_hash
from deepresearch_agent.harness.delivery_evidence import order_supplement_sources, delivery_evidence_records
from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult


def test_chinese_reservation_uses_token_estimate_and_includes_tool_schemas():
    messages = ([HumanMessage(content="状态管理与工具调用。" * 2000)],)
    options = {"tools": [{"type": "function", "function": {"name": "search", "description": "检索资料"}}]}
    estimate = estimate_model_reservation(messages, options, 8000)
    old = len(json.dumps([str(messages), options], ensure_ascii=False).encode("utf-8")) + 8000 + 128
    assert 8000 < estimate < old * 0.8
    assert estimate_model_reservation(messages, {"tools": [{"schema": "不同类型的参数。" * 1000}]}, 8000) > estimate
    policy = CallBudget(BudgetLimits(max_llm_tokens=400000), BudgetUsage(llm_tokens=252430))
    reservation = policy.reserve_model(messages, options, 8000)
    policy.settle_model(reservation, 9000)
    assert policy.charged == 261430 and policy.reserved == 0


class Model:
    def __init__(self, payload):
        self.payload, self.requests = payload, []

    async def ainvoke(self, messages, **kwargs):
        self.requests.append(json.loads(messages[-1].content))
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


@pytest.mark.asyncio
async def test_delivery_review_sends_body_once_and_supplement_only_target_paragraphs():
    target = "CrewAI 工具调用的流程需要进一步说明与验证。"
    unrelated = "LangGraph 的图状态管理与检查点机制已有相应分析。"
    report = f"# 比较报告\n\n{unrelated}\n\n## CrewAI\n\n{target}"
    model = Model({"items": [{"requirement_id": "req_1", "status": "partial", "paragraph_ids": ["p2"],
                              "reason": "缺少参数校验与结果回传机制的直接资料", "gap_type": "evidence",
                              "search_query": "CrewAI tools parameter validation result official docs"}]})
    requirements = [{"id": "req_1", "kind": "analysis", "text": "分析CrewAI工具调用机制"}]
    service = DeliveryService(model)
    review = await service.review(requirements, report)
    payload = model.requests[0]
    assert "report" not in payload and json.dumps(payload, ensure_ascii=False).count(target) == 1
    assert review["items"][0]["gap_type"] == "evidence"
    model.payload = {"sections": [{"title": "机制补充", "paragraphs": [{"text": "参数按工具定义校验。", "evidence_ids": ["ev_new"]}]}]}
    await service.supplement(requirements, review, report, [{"evidence_id": "ev_new", "text": "Parameter schema validation"}])
    request = model.requests[-1]
    assert request["existing_paragraphs"] == {"p2": target}
    assert unrelated not in json.dumps(request, ensure_ascii=False)
    assert "report_quotes" not in json.dumps(request)


def test_supplement_prioritizes_new_and_relevant_evidence_over_ledger_head():
    def row(eid, text):
        return RetrievalResult(result_id=eid, source="custom", source_mode="web", evidence=text, granularity="DO",
                               metadata={"source_id": eid, "source_type": "webpage"})
    rows = [row("ev_old", "LangGraph state"), row("ev_crewai", "CrewAI tool schema"), row("ev_new", "New tool details")]
    ordered = order_supplement_sources(rows, [{"id": "req_1", "text": "CrewAI工具机制"}],
        {"items": [{"requirement_id": "req_1", "status": "partial"}]}, ["ev_new"])
    assert [item.result_id for item in ordered] == ["ev_new", "ev_crewai", "ev_old"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["evidence", "empty", "no_progress", "no_tokens", "no_tools", "wrong_source", "crash_after_search"])
async def test_delivery_search_persists_reuses_and_stops_without_hiding_gaps(tmp_path, mode):
    from test_harness_phase3 import FakeDriver
    from backend.app.schemas import MessageCreate, RunCreate, SessionCreate
    from deepresearch_agent.harness.runtime import HarnessRuntime
    from deepresearch_agent.persistence import Database
    from deepresearch_agent.persistence.repositories import (
        RunRepository, SessionRepository, MessageRepository, EventRepository,
        CheckpointRepository, EvidenceRepository, ContractRepository, PlanTaskToolRepository,
    )

    class Provider:
        mode = SourceMode.WEB
        provider_name = "offline"
        calls = 0

        async def search(self, query, **kwargs):
            self.calls += 1
            assert "CrewAI" in query
            return [] if mode == "empty" else [RetrievalResult(result_id="raw_new", source="tavily_search",
                source_mode="web", granularity="DO", evidence="CrewAI validates tool arguments against its schema and returns results.",
                metadata={"source_id": "https://example.org/tools", "source_type": "webpage"})]

    provider = Provider()
    if mode == "wrong_source":
        provider.mode = SourceMode.GRAPHRAG

    class Driver(FakeDriver):
        retrieval_provider = provider

        def set_delivery_requirements(self, requirements):
            pass

        def apply_delivery_report(self, report):
            self._report = report

        def evidence_results(self):
            return super().evidence_results() + delivery_evidence_records(self.context)

    class Delivery:
        supplements = 0

        async def extract(self, query):
            return [{"id": "req_1", "kind": "analysis", "text": "分析CrewAI工具调用机制"}]

        async def review(self, requirements, report):
            passed = "## 机制补充" in report and mode != "no_progress"
            return {"passed": passed, "report_hash": report_hash(report), "requirements": requirements,
                "items": [{"requirement_id": "req_1", "status": "covered" if passed else "insufficient_evidence",
                           "gap_type": "evidence", "reason": "缺少具体调用流程资料", "search_query": "CrewAI tools official schema"}]}

        async def supplement(self, requirements, review, report, evidence):
            self.supplements += 1
            assert "CrewAI" in evidence[0]["text"]
            return f"## 机制补充\n\n工具按照定义校验参数，并将执行结果返回模型。 [{evidence[0]['evidence_id']}]"

    db = Database(f"sqlite+aiosqlite:///{(tmp_path / 'recovery.db').as_posix()}")
    await db.create_schema()
    try:
        sessions, runs = SessionRepository(db), RunRepository(db)
        session = await sessions.create(SessionCreate(title="budget delivery"))
        _, run, _ = await runs.create_for_user_message(
            MessageCreate(session_id=session.session_id, role="user", content="分析CrewAI工具调用机制"),
            RunCreate(session_id=session.session_id, trigger_message_id="atomic", source_mode="web", workflow_mode="plan_execute_report",
                config_snapshot={"delivery_checks": True}, budget=BudgetLimits(max_llm_tokens=400000,
                    max_tool_calls=1 if mode == "no_tools" else 30).model_dump()))
        delivery = Delivery()
        checkpoints = CheckpointRepository(db)
        kwargs = dict(run_repository=runs, message_repository=MessageRepository(db), event_repository=EventRepository(db),
            checkpoint_repository=checkpoints, evidence_repository=EvidenceRepository(db), contract_repository=ContractRepository(db),
            trajectory_repository=PlanTaskToolRepository(db),
            workflow_factory=lambda context, events: Driver(context, token_count=350000 if mode == "no_tokens" else 0),
            delivery_service=delivery)
        runtime = HarnessRuntime(**kwargs)
        if mode == "crash_after_search":
            original = runtime._retrieve_delivery_gaps

            async def crash(*args, **kwargs):
                await original(*args, **kwargs)
                import asyncio
                raise asyncio.CancelledError()

            runtime._retrieve_delivery_gaps = crash
            import asyncio
            with pytest.raises(asyncio.CancelledError):
                await runtime.execute_run(run.run_id)
            runtime = HarnessRuntime(**kwargs)
        result = await runtime.execute_run(run.run_id)
        expected_pass = mode in {"evidence", "crash_after_search"}
        assert result.status.value == ("completed" if expected_pass else "failed")
        assert provider.calls == (0 if mode in {"no_tokens", "no_tools", "wrong_source"} else 1)
        assert delivery.supplements == (1 if mode in {"evidence", "no_progress", "crash_after_search"} else 0)
        state = result.context_snapshot["delivery"]
        if not expected_pass:
            assert state["stop_reason"] in {"no_coverage_progress", "evidence_gap_unresolved"}
            assert not state["review"]["passed"]
        if provider.calls:
            events = await EventRepository(db).list_after(run.run_id)
            calls = [event for event in events if event.event_type == "tool.started"]
            assert len(calls) == 1 and calls[0].stage == "verifying"
            assert len(state["searches"]) == 1
        if expected_pass:
            assert len(state["retrieved_results"]) == 1
            tool = await PlanTaskToolRepository(db).get_tool_call(state["searches"][0]["tool_call_id"])
            assert tool.status == "completed" and tool.source_mode == "web"
            assert result.budget_limits.max_llm_tokens == 400000
    finally:
        await db.close()
