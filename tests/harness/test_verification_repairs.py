"""Regressions for real-world verification failures; all providers are offline."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.agents.multi_agent.core.execution_record import ExecutionRecord
from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult, RetrievalMetadata
from deepresearch_agent.agents.multi_agent.reporter.base_reporter import ReportResult
from deepresearch_agent.agents.multi_agent.reporter.consistency_checker import ConsistencyChecker, ConsistencyCheckResult
from deepresearch_agent.agents.multi_agent.reporter.outline_builder import ReportOutline
from deepresearch_agent.harness.contracts import ContractEvaluator, SourceMode, RunStatus, WorkflowMode
from deepresearch_agent.harness.delivery import DeliveryService, report_hash
from deepresearch_agent.harness.report_repair import ReportRepairService
from deepresearch_agent.harness.run_context import RunContext
from deepresearch_agent.harness.workflow import PlanExecuteReportDriver


SCOPE = "检查点恢复保证任何外部工具在进程崩溃后都恰好执行一次。"
BODY = "检查点恢复加载保存的状态，重试则重新尝试失败操作，两者解决不同层次的问题。 [ev_a]"
OTHER = "工具具有外部副作用时，仍需利用幂等键或状态确认来控制重复执行风险。 [ev_b]"
REPORT = f"# 报告\n\n## 摘要\n\n{SCOPE}\n\n## 机制比较\n\n{BODY}\n\n## 重复调用\n\n{OTHER}"


class Model:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    async def ainvoke(self, messages, **options):
        self.calls.append(json.loads(messages[-1].content))
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


def sources():
    return [RetrievalResult(result_id=f"ev_{name}", source="tavily_search", source_mode="web", granularity="DO", evidence=text,
        metadata=RetrievalMetadata(source_id=f"https://example.org/{name}", source_type="webpage"))
        for name, text in [("a", "检查点保存执行状态，重试重新尝试失败操作。"), ("b", "重复执行时需要幂等键或状态确认。")]]


def context():
    return RunContext(run_id="run_test", session_id="session_test", trigger_message_id="message_test", original_query="比较恢复与重试",
        source_mode=SourceMode.WEB, workflow_mode=WorkflowMode.PLAN_EXECUTE_REPORT, status=RunStatus.VERIFYING)


class Orchestrator:
    def __init__(self, *, error=False):
        self.calls, self.error = [], error

    def recheck_report_consistency(self, report, evidence):
        self.calls.append(report)
        if self.error:
            raise ValueError("一致性检查输出协议错误")
        return ConsistencyCheckResult(is_consistent=True, report_hash=report_hash(report))


def driver(orchestrator=None):
    result = PlanExecuteReportDriver(context(), orchestrator or Orchestrator())
    result.report_result = ReportResult(outline=ReportOutline(title="报告", report_type="brief", sections=[]), sections=[], final_report=REPORT)
    result.state.execution_records = [ExecutionRecord(task_id="a", session_id="session_test", worker_type="offline", evidence=sources())]
    result._bind_consistency(ConsistencyCheckResult(is_consistent=True), "deterministic")
    return result


@pytest.mark.asyncio
async def test_delivery_accepts_six_valid_paragraphs_without_truncating_proof():
    paragraphs = [f"第{i}个场景需要比较恢复状态、重试边界与外部副作用。" for i in range(6)]
    payload = {"items": [{"requirement_id": "req_1", "status": "covered", "paragraph_ids": [f"p{i+1}" for i in range(6)], "reason": "六个段落共同覆盖场景"}]}
    review = await DeliveryService(Model(payload)).review([{"id": "req_1"}], "\n\n".join(paragraphs))
    assert review["passed"] and len(review["items"][0]["report_quotes"]) == 6
    payload["items"][0]["paragraph_ids"][-1] = "p_unknown"
    with pytest.raises(ValueError, match="未知报告段落"):
        await DeliveryService(Model(payload)).review([{"id": "req_1"}], "\n\n".join(paragraphs))


@pytest.mark.asyncio
async def test_local_repair_keeps_all_cited_body_and_rechecks_new_revision(monkeypatch):
    model = Model({"patches": [{"paragraph_id": "p1", "action": "remove", "reason": "删除无实质结论的范围摘要"}]})
    service = ReportRepairService(model)
    monkeypatch.setattr("deepresearch_agent.harness.report_repair.ReportRepairService", lambda: service)
    obj = driver()
    assert await obj.repair_report(["citation_integrity", "claim_support"])
    revised = await obj.report()
    assert BODY in revised and OTHER in revised and SCOPE not in revised
    assert "证据支持的要点" not in revised and obj.report_consistency() is None
    await obj.verify_report_consistency(revised)
    assert obj.report_consistency() is True and obj.orchestrator.calls == [revised]
    assert obj.report_consistency_details()["report_hash"] == report_hash(revised)
    assert obj.context.context_snapshot["report_repairs"][0]["edits"][0]["original"] == SCOPE
    assert list(model.calls[0]["targets"]) == ["p1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("patch", [
    {"paragraph_id": "p2", "action": "remove", "reason": "不允许改无错误的正文"},
    {"paragraph_id": "p1", "action": "replace", "text": "伪造来源支持的结论，需要拒绝保存。", "evidence_ids": ["ev_unknown"], "reason": "伪造"},
    {"paragraph_id": "p1", "action": "replace", "text": "# 用新标题替换整段", "evidence_ids": ["ev_a"], "reason": "破坏结构"},
])
async def test_invalid_patch_preserves_original_report(monkeypatch, patch):
    service = ReportRepairService(Model({"patches": [patch]}))
    monkeypatch.setattr("deepresearch_agent.harness.report_repair.ReportRepairService", lambda: service)
    obj = driver()
    assert await obj.repair_report(["citation_integrity"]) is False
    assert obj.report_result.final_report == REPORT
    assert obj.context.context_snapshot["report_repair_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text,ids,reason", [
    ("第一行\n第二行", ["ev_a"], "text包含换行"),
    ("正文自带来源标记 [ev_a]", ["ev_a"], "text手写引用标记"),
    ("合法长度的正文，但没有提供来源。", [], "evidence_ids必须为非空数组"),
    ("合法长度的正文，但引用了未提供来源。", ["ev_unknown"], "evidence_ids包含未提供或非法来源编号"),
])
async def test_repair_rejection_identifies_field_without_saving_model_prose(monkeypatch, text, ids, reason):
    service = ReportRepairService(Model({"patches": [{"paragraph_id": "p1", "action": "replace",
        "text": text, "evidence_ids": ids, "reason": "修复摘要"}]}))
    monkeypatch.setattr("deepresearch_agent.harness.report_repair.ReportRepairService", lambda: service)
    obj = driver()
    assert not await obj.repair_report(["citation_integrity"])
    error = obj.context.context_snapshot["report_repair_error"]
    assert reason in error and "p1" in error
    assert text not in error
    assert obj.report_result.final_report == REPORT


@pytest.mark.asyncio
async def test_supplement_invalidates_old_failed_consistency_and_snapshot_keeps_new_result():
    obj = driver()
    obj._bind_consistency(ConsistencyCheckResult(is_consistent=False, issues=[{"description": "旧版本不完整"}]), "llm-assisted")
    revised = REPORT + "\n\n" + OTHER
    obj.apply_delivery_report(revised)
    assert obj.report_consistency() is None
    await obj.verify_report_consistency(revised)
    assert obj.report_consistency() is True and obj.orchestrator.calls == [revised]
    restored_context = context()
    restored_context.workflow_state = obj.snapshot()
    restored = PlanExecuteReportDriver(restored_context, Orchestrator(error=True))
    await restored.verify_report_consistency(revised)
    assert restored.report_consistency() is True and restored.orchestrator.calls == []


@pytest.mark.asyncio
async def test_checker_error_is_not_a_content_verdict_and_retry_does_not_rewrite_report():
    obj = driver(Orchestrator(error=True))
    obj._consistency_record = {}
    await obj.verify_report_consistency(REPORT)
    assert obj.report_consistency() is None and obj.report_consistency_details()["status"] == "error"
    verdict = await ContractEvaluator().evaluate(run_id="r", source_mode="web", report=REPORT, evidence=[{"evidence_id": "ev_a"}, {"evidence_id": "ev_b"}],
        consistency_passed=None, consistency_details=obj.report_consistency_details())
    check = next(item for item in verdict.checks if item.kind == "report_consistency")
    assert check.passed is False and check.observed["actual"] is None
    assert check.observed["details"]["status"] == "error"
    assert await obj.repair_report(["report_consistency"])
    assert await obj.report() == REPORT
    obj.orchestrator.error = False
    await obj.verify_report_consistency(REPORT)
    assert obj.report_consistency() is True and len(obj.orchestrator.calls) == 2


@pytest.mark.asyncio
async def test_contract_rejects_consistency_from_another_report_revision():
    verdict = await ContractEvaluator().evaluate(run_id="r", source_mode="web", report=BODY,
        evidence=[{"evidence_id": "ev_a"}], consistency_passed=True,
        consistency_details={"report_hash": report_hash(OTHER), "status": "passed"})
    check = next(item for item in verdict.checks if item.kind == "report_consistency")
    assert check.passed is False
    assert check.observed["details"]["status"] == "not_checked"


@pytest.mark.asyncio
async def test_unlocalized_consistency_failure_never_replaces_report_with_sources(monkeypatch):
    obj = driver()
    obj._bind_consistency(ConsistencyCheckResult(is_consistent=False, issues=[{"description": "整体不一致"}]), "llm-assisted")
    assert await obj.repair_report(["report_consistency"]) is False
    assert obj.report_result.final_report == REPORT


def test_consistency_json_retries_once_and_binds_exact_report_version():
    responses = iter(["not JSON", '{"is_consistent":true,"issues":[],"corrections":[]}'])
    checker = ConsistencyChecker(llm=SimpleNamespace(invoke=lambda _, **options: AIMessage(content=next(responses))))
    result = checker.check(BODY, "[]")
    assert result.is_consistent and result.report_hash == hashlib.sha256(BODY.encode()).hexdigest()
    checker = ConsistencyChecker(llm=SimpleNamespace(invoke=lambda _, **options: AIMessage(content='{"issues":[]}')))
    with pytest.raises(ValueError, match="已重试一次"):
        checker.check(BODY, "[]")


@pytest.mark.asyncio
async def test_structural_precheck_cannot_bypass_semantic_verification():
    obj = driver()
    assert obj.report_consistency_details()["method"] == "deterministic"
    await obj.verify_report_consistency(REPORT)
    assert obj.orchestrator.calls == [REPORT]
    assert obj.report_consistency_details()["method"] == "llm-assisted"
    await obj.verify_report_consistency(REPORT)
    assert obj.orchestrator.calls == [REPORT]  # Same-version actual verdict is reusable.


@pytest.mark.asyncio
async def test_legacy_policy_verdict_is_not_reused_as_current_review():
    obj = driver()
    obj._consistency_record = {"report_hash": report_hash(REPORT), "method": "llm-assisted", "status": "passed"}
    await obj.verify_report_consistency(REPORT)
    assert obj.orchestrator.calls == [REPORT]
    from deepresearch_agent.agents.multi_agent.reporter.review_policy import REVIEW_POLICY_VERSION
    assert obj.report_consistency_details()["policy_version"] == REVIEW_POLICY_VERSION


@pytest.mark.asyncio
async def test_repair_uses_reviewed_paragraph_id_when_prose_is_duplicated():
    report = "# 报告\n\n" + BODY + "\n\n" + BODY
    model = Model({"patches": [{"paragraph_id": "p2", "action": "replace", "text": "检查点保存执行状态，恢复时读取已经保存的进度。",
        "evidence_ids": ["ev_a"], "reason": "只修改已定位的第二段"}]})
    revised, audit = await ReportRepairService(model).repair(report=report, results=sources(), run_id="r", source_mode="web",
        failures=["report_consistency"], consistency={"corrections": [{"paragraph_id": "p2", "original": BODY,
            "corrected": "修改第二段", "reason": "已定位的错误"}]})
    assert revised.split("\n\n")[1] == BODY
    assert "恢复时读取已经保存的进度" in revised.split("\n\n")[2]
    assert [item["paragraph_id"] for item in audit["edits"]] == ["p2"]
