"""Regressions for the saved 35-source, empty-comparison research report."""
import json

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from deepresearch_agent.agents.multi_agent.reporter.evidence_cards import EvidenceCardPipeline
from deepresearch_agent.agents.multi_agent.reporter.outline_builder import ReportOutline, SectionOutline
from deepresearch_agent.harness.contracts import ContractEvaluator, SourceMode
from deepresearch_agent.harness.delivery import DeliveryService
from deepresearch_agent.harness.recovery import classify_verification_failures
from deepresearch_agent.harness.request_requirements import normalize_requirement
from deepresearch_agent.harness.verifiers.deterministic import DeterministicVerifiers


def source(eid, framework):
    return RetrievalResult(result_id=eid, source="tavily_search", source_mode="web", granularity="DO",
        evidence=f"{framework} architecture and state persistence mechanisms.",
        metadata={"source_id": f"https://example.org/{eid}", "source_type": "webpage", "title": framework})


def test_synthesis_reuses_balanced_evidence_and_preserves_explicit_secondary_routes():
    pipeline = EvidenceCardPipeline(max_cards_per_section_call=6)
    cards, _ = pipeline.build_cards([source(f"ev_{framework}_{i}", framework)
        for framework in ["LangGraph", "AutoGen", "CrewAI"] for i in range(4)])
    outline = ReportOutline(title="Comparison", report_type="long_document", sections=[
        SectionOutline(section_id=framework, title=framework, summary="Architecture and state",
                       evidence_ids=[card.evidence_id for card in cards if framework in card.evidence_id])
        for framework in ["LangGraph", "AutoGen", "CrewAI"]] + [
        SectionOutline(section_id="reuse", title="A specific tool", summary="Explicit reuse", evidence_ids=[cards[0].evidence_id]),
        SectionOutline(section_id="comparison", title="三维度横向对比", summary="Compare all three frameworks"),
        SectionOutline(section_id="conclusion", title="结论与选型建议", summary="Choose by requirements"),
        SectionOutline(section_id="unsupported", title="独立的财务统计", summary="No supporting material"),
    ])
    routes = pipeline.route_cards(cards, outline)
    for sid in ["comparison", "conclusion"]:
        assert len(routes[sid]) == 6
        assert all(any(framework in eid for eid in routes[sid]) for framework in ["LangGraph", "AutoGen", "CrewAI"])
    assert cards[0].primary_section == "LangGraph"
    assert "reuse" in cards[0].secondary_sections and cards[0].evidence_id in routes["reuse"]
    assert routes["unsupported"] == []
    assert {eid for ids in routes.values() for eid in ids} == {card.evidence_id for card in cards}
    # Repeated routing of a restored outline cannot accumulate duplicate IDs.
    again = pipeline.route_cards(cards, outline)
    assert all(len(ids) == len(set(ids)) for ids in again.values())


class Model:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append(json.loads(messages[-1].content))
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


@pytest.mark.asyncio
async def test_comparative_report_requirement_is_repaired_even_in_legacy_format_snapshot():
    requirement = {"id": "req_8", "kind": "format", "text": "生成带参考来源的对比报告",
                   "request_quote": "最后生成一份带有参考来源的对比报告"}
    assert normalize_requirement(requirement)["kind"] == "analysis"
    pure_style = {"kind": "format", "text": "引用来源", "request_quote": "并引用来源"}
    assert normalize_requirement(pure_style)["kind"] == "format"
    model = Model({"sections": [{"title": "横向比较", "paragraphs": [
        {"text": "方案一保存状态，方案二重试操作，应根据故障阶段选择。", "evidence_ids": ["ev_a"]}]}]})
    patch = await DeliveryService(model).supplement([requirement],
        {"items": [{"requirement_id": "req_8", "status": "partial"}]}, "# 报告",
        [{"evidence_id": "ev_a", "text": "状态恢复与操作重试具有不同边界。"}])
    assert "[ev_a]" in patch and model.calls[0]["requirements"][0]["kind"] == "analysis"


@pytest.mark.parametrize("paragraph,passed", [
    ("本报告围绕三种框架，从架构、工具调用与状态管理三个维度进行分析，最后给出选型建议。报告所有关键结论均标注参考来源，以支撑可追溯的技术判断。", True),
    ("本报告围绕三种框架进行分析。该框架保证所有外部操作恰好执行一次。", False),
    ("本报告分析方案优势，该方案能够将成本降低90%。", False),
    ("本报告证明该框架可以保证零故障运行。", False),
])
def test_pure_scope_is_exempt_but_facts_inside_abstract_still_require_citations(paragraph, passed):
    verifier = DeterministicVerifiers(run_id="r", source_mode=SourceMode.WEB,
        report="# 报告\n\n## 摘要\n\n" + paragraph + "\n\n## 正文\n\n保存状态有助于恢复已完成的研究步骤。 [ev_a]",
        evidence=[{"evidence_id": "ev_a"}])
    assert verifier.citation_integrity().passed is passed
    assert verifier.claim_support().passed is passed


@pytest.mark.asyncio
async def test_delivery_gap_does_not_block_repair_of_combined_citation_failure():
    verdict = await ContractEvaluator().evaluate(run_id="r", source_mode="web",
        report="# 报告\n\n这是缺少引用的实质性断言，需要单独修复。\n\n已保存的证据需要正确引用。 [ev_a]",
        evidence=[{"evidence_id": "ev_a", "source_mode": "web"}], consistency_passed=True,
        delivery_coverage={"passed": False, "items": []})
    assert {"citation_integrity", "delivery_coverage"} <= set(verdict.failures)
    assert "claim_support" not in verdict.failures  # Same coverage is no longer a second blocking gate.
    assert verdict.recoverable
    assert classify_verification_failures(verdict.failures).action == "repair_report"
    assert classify_verification_failures(["delivery_coverage"]).action == "fail"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["supplement", "empty", "still_partial", "citation_and_delivery"])
async def test_runtime_delivery_repair_is_bounded_and_combined_failures_can_complete(tmp_path, mode):
    from test_harness_phase3 import FakeDriver
    from backend.app.schemas import MessageCreate, RunCreate, SessionCreate
    from deepresearch_agent.harness.budgets import BudgetLimits
    from deepresearch_agent.harness.delivery import report_hash
    from deepresearch_agent.harness.runtime import HarnessRuntime
    from deepresearch_agent.persistence import Database
    from deepresearch_agent.persistence.repositories import (
        RunRepository, SessionRepository, MessageRepository, EventRepository,
        CheckpointRepository, EvidenceRepository, ContractRepository,
    )

    class Driver(FakeDriver):
        def set_delivery_requirements(self, requirements):
            self.context.context_snapshot["delivery_requirements"] = requirements

        def apply_delivery_report(self, report):
            self._report = report

        async def repair_report(self, failures):
            return await super().repair_report([kind for kind in failures if kind != "delivery_coverage"])

    class Delivery:
        calls = 0

        async def extract(self, query):
            return [{"id": "req_1", "kind": "format", "text": "生成带参考来源的对比报告", "request_quote": "对比报告"}]

        async def review(self, requirements, report):
            assert requirements[0]["kind"] == "analysis"
            passed = ("ev_missing" not in report if mode == "citation_and_delivery" else mode != "still_partial" and "## 对比补充" in report)
            return {"passed": passed, "report_hash": report_hash(report), "requirements": requirements,
                    "items": [{"requirement_id": "req_1", "status": "covered" if passed else "partial", "reason": "offline control"}]}

        async def supplement(self, requirements, review, report, evidence):
            self.calls += 1
            if mode in {"empty", "citation_and_delivery"}:
                return ""
            return f"## 对比补充\n\n不同操作的恢复方式应结合已保存状态进行比较。 [{evidence[0]['evidence_id']}]"

    db = Database(f"sqlite+aiosqlite:///{(tmp_path / 'delivery.db').as_posix()}")
    await db.create_schema()
    try:
        session = await SessionRepository(db).create(SessionCreate(title="delivery regression"))
        runs = RunRepository(db)
        _, run, _ = await runs.create_for_user_message(
            MessageCreate(session_id=session.session_id, role="user", content="生成对比报告"),
            RunCreate(session_id=session.session_id, trigger_message_id="atomic", source_mode="web", workflow_mode="plan_execute_report",
                config_snapshot={"delivery_checks": True}, budget=BudgetLimits(max_llm_tokens=400000).model_dump()))
        delivery = Delivery()
        runtime = HarnessRuntime(run_repository=runs, message_repository=MessageRepository(db),
            event_repository=EventRepository(db), checkpoint_repository=CheckpointRepository(db),
            evidence_repository=EvidenceRepository(db), contract_repository=ContractRepository(db),
            workflow_factory=lambda context, events: Driver(context, dangling=mode == "citation_and_delivery"), delivery_service=delivery)
        result = await runtime.execute_run(run.run_id)
        row = await runs.get(run.run_id)
        assert result.status.value == ("completed" if mode in {"supplement", "citation_and_delivery"} else "failed"), row.error_message
        assert delivery.calls == 1
        assert result.context_snapshot["delivery"]["repair_attempts"] == delivery.calls
        if mode == "citation_and_delivery":
            assert result.budget_usage.task_retries == 1
            assert "ev_missing" not in result.report
        if mode == "empty":
            events = await EventRepository(db).list_after(run.run_id)
            assert any(event.event_type == "delivery.supplement.failed" for event in events)
        if mode == "still_partial":
            assert result.context_snapshot["delivery"]["stop_reason"] == "no_coverage_progress"
    finally:
        await db.close()
