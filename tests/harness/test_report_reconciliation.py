import copy
import json

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.harness.delivery import DeliveryService, report_hash
from deepresearch_agent.harness.report_reconciliation import (
    limitation_entries, report_paragraphs, reconciliation_candidates, reconcile_report,
)
from deepresearch_agent.harness.verifiers.deterministic import DeterministicVerifiers
from deepresearch_agent.harness.contracts import SourceMode


class JsonModel:
    def __init__(self, payload):
        self.payload = payload

    async def ainvoke(self, messages, **options):
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


REPORT = ("# 研究报告\n\n## 局限\n\n- 还不能比较进程中断场景。\n- 当前资料有限，未进行实际故障注入。\n\n"
          "## 进程中断的作用与局限\n检查点恢复保留中间进度，单纯失败重试可能重新执行已完成步骤。 [ev_a]\n\n"
          "## 示例\n\n项目使用保存的任务状态恢复执行，并以幂等键约束重复调用。 [ev_b]")


def review_fixture():
    paragraphs, eligible = report_paragraphs(REPORT)
    pid = next(pid for pid in eligible if "检查点恢复" in paragraphs[pid])
    review = {"report_hash": report_hash(REPORT), "requirements": [{"id": "req_1", "text": "比较进程中断"}],
              "items": [{"requirement_id": "req_1", "status": "covered", "paragraph_ids": [pid], "report_quote": paragraphs[pid]}],
              "passed": True}
    proposals = [{"limitation_id": "lim_1", "requirement_id": "req_1", "paragraph_ids": [pid], "reason": "正文已经比较两种机制"}]
    review["resolved_limitations"] = reconciliation_candidates(REPORT, review, proposals)
    return review


def test_only_individually_reviewed_gap_removed_while_source_caveat_body_and_audit_remain():
    original = review_fixture()
    saved = copy.deepcopy(original)
    result, review = reconcile_report(REPORT, original)
    assert "还不能比较进程中断" not in result
    assert "当前资料有限，未进行实际故障注入" in result
    assert "## 进程中断的作用与局限" in result
    assert review["report_hash"] == report_hash(result) and review["reviewed_report_hash"] == report_hash(REPORT)
    paragraphs, _ = report_paragraphs(result)
    for item in review["items"]:
        assert item["report_quote"] in result
        assert item["report_quotes"][0]["text"] == paragraphs[item["paragraph_ids"][0]]
    assert review["reconciliation"]["removed_limitations"][0]["text"].startswith("- 还不能")
    assert original == saved
    assert reconcile_report(result, review) == (result, review)


@pytest.mark.parametrize("change", ["unknown_id", "foreign_requirement", "limitation_proof", "stale_hash", "altered_text", "not_covered"])
def test_invalid_cleanup_is_ignored(change):
    review = review_fixture()
    proposal = review["resolved_limitations"][0]
    if change == "unknown_id": proposal["limitation_id"] = "unknown"
    if change == "foreign_requirement": proposal["requirement_id"] = "req_other"
    if change == "limitation_proof": proposal["paragraph_ids"] = ["p1"]
    if change == "stale_hash": review["report_hash"] = "old"
    if change == "altered_text": proposal["text"] = "not the actual limitation"
    if change == "not_covered": review["items"][0]["status"] = "partial"
    assert reconcile_report(REPORT, review)[0] == REPORT


def test_legacy_reviews_and_reports_without_limitations_are_not_rewritten():
    assert len(limitation_entries(REPORT)) == 2
    review = review_fixture()
    review.pop("resolved_limitations")
    assert reconcile_report(REPORT, review) == (REPORT, review)
    assert limitation_entries("# 工具超时的作用与局限\n\n正文有局限分析。") == []


@pytest.mark.asyncio
async def test_coverage_cannot_be_proven_by_summary_or_limitations_and_all_quotes_are_kept():
    body = "# 摘要\n\n将比较检查点恢复与重试，并说明局限。\n\n# 比较\n\n检查点恢复保留中间进度，失败重试则重新执行失败操作。\n\n幂等性需要由工具调用层单独控制，两种机制都不能自动保证。\n\n# 局限\n\n缺少直接项目测量。"
    payload = {"items": [{"requirement_id": "req_1", "status": "covered", "paragraph_ids": ["p1"], "reason": "已覆盖"}]}
    model = JsonModel(payload)
    with pytest.raises(ValueError, match="摘要或局限"):
        await DeliveryService(model).review([{"id": "req_1"}], body)
    payload["items"][0]["paragraph_ids"] = ["p2", "p3"]
    review = await DeliveryService(model).review([{"id": "req_1"}], body)
    assert len(review["items"][0]["report_quotes"]) == 2
    assert all(item["text"] in body for item in review["items"][0]["report_quotes"])


def test_claim_details_persist_the_actual_checked_text_and_ids():
    report = "# 报告\n\n检查点恢复可以保留已完成进度，这一段有引用。 [ev_a]\n\n本段没有证据引用，需要明确显示未满足的内容。"
    result = DeterministicVerifiers(run_id="r", source_mode=SourceMode.WEB, report=report,
        evidence=[{"evidence_id": "ev_a", "source_mode": "web"}]).claim_support()
    details = result.observed["actual"]["claim_details"]
    assert result.passed is False and [item["passed"] for item in details] == [True, False]
    assert details[0]["evidence_ids"] == ["ev_a"] and details[1]["text"] in report


def test_scenario_limitation_headings_do_not_hide_uncited_body():
    report = "# 工具超时的作用与局限\n\n这里讨论超时局限，需要检查证据引用是否就近绑定。\n\n# 局限\n\n本研究未进行故障注入实验。"
    result = DeterministicVerifiers(run_id="r", source_mode=SourceMode.WEB, report=report, evidence=[]).claim_support()
    assert result.passed is False
    assert result.observed["actual"]["claims"] == 1
