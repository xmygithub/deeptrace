"""Evidence budget regressions based on the multi-framework report incident."""
import json

import pytest
from langchain_core.messages import AIMessage

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from deepresearch_agent.agents.multi_agent.reporter.base_reporter import BaseReporter
from deepresearch_agent.agents.multi_agent.reporter.evidence_cards import EvidenceCardPipeline
from deepresearch_agent.context.tokens import count_tokens
from deepresearch_agent.harness.report_repair import ReportRepairService
from deepresearch_agent.harness.budgets import stage_token_reserves
from deepresearch_agent.retrieval.evidence_context import source_packet


def evidence(eid, text, *, task="task_langgraph", title="Source"):
    return RetrievalResult(result_id=eid, source="tavily_search", source_mode="web", granularity="DO", evidence=text,
        metadata={"source_id": f"https://example.org/{eid}", "source_type": "webpage", "title": title,
                  "extra": {"task_id": task}})


def test_packet_admits_late_relevant_source_and_preserves_original_quotes_within_budget():
    rows = [evidence(f"ev_{i}", "Background facts. " * 300) for i in range(35)]
    late = evidence("ev_autogen", "AutoGen stores agent state. " * 300, task="task_autogen")
    packets = source_packet([*rows, late, late], max_tokens=6000, max_sources=12, preferred_ids=[late.result_id])
    assert packets[0]["evidence_id"] == late.result_id
    assert len(packets) == len({p["evidence_id"] for p in packets}) == 12
    assert count_tokens(json.dumps(packets, ensure_ascii=False)) <= 6000
    originals = {item.result_id: item.evidence for item in [*rows, late]}
    assert all(originals[p["evidence_id"]].startswith(p["text"]) for p in packets)


def test_checker_receives_title_and_body_beyond_old_200_character_prefix():
    item = evidence("ev_article", "Navigation links. " * 160 + "Body explains agent state persistence.",
                    title="LangGraph control vs AutoGen collaboration vs CrewAI convenience")
    reporter = object.__new__(BaseReporter)
    packet = json.loads(reporter._format_evidence_for_check([item], "Framework comparison [ev_article]"))[0]
    assert packet["title"] == item.metadata.title
    assert "Body explains agent state persistence." in packet["text"]
    assert packet["truncated"] is False


def test_checker_packet_stays_bounded_and_covers_all_57_evidence_ids():
    rows = [evidence(f"ev_{i}", "Long evidence body. " * 2000) for i in range(57)]
    packets = source_packet(rows, max_tokens=32000, per_source_tokens=3000)
    assert {p["evidence_id"] for p in packets} == {item.result_id for item in rows}
    assert count_tokens(json.dumps(packets, ensure_ascii=False)) <= 32000
    assert all(p["truncated"] for p in packets)
    assert all(len(p["text"]) > 200 for p in packets)


@pytest.mark.asyncio
async def test_repair_gets_target_and_replacement_sources_beyond_first_12_records():
    rows = [evidence(f"ev_{i}", "LangGraph source material.") for i in range(40)]
    rows.extend([evidence("ev_autogen", "AutoGen state is serializable.", task="task_autogen"),
                 evidence("ev_correct", "AutoGen supports saving agent state.", task="task_autogen")])
    original = "The paragraph incorrectly describes the AutoGen persistence mechanism. [ev_autogen]"

    class Model:
        payload = None

        async def ainvoke(self, messages, **kwargs):
            self.payload = json.loads(messages[-1].content)
            return AIMessage(content=json.dumps({"patches": [{"paragraph_id": "p1", "action": "replace",
                "text": "AutoGen supports saving agent state.", "evidence_ids": ["ev_correct"], "reason": "Correct the subject"}]}))

    model = Model()
    revised, _ = await ReportRepairService(model).repair(report="# Report\n\n" + original, results=rows,
        run_id="run_test", source_mode="web", failures=["report_consistency"],
        consistency={"corrections": [{"original": original, "corrected": "Supports state saving. [ev_correct]"}]})
    supplied = {item["evidence_id"] for item in model.payload["sources"]}
    assert {"ev_autogen", "ev_correct"} <= supplied
    assert "[ev_correct]" in revised and "incorrectly describes" not in revised


def test_larger_card_budget_does_not_reuse_a_previously_shortened_cache_entry():
    item = evidence("ev_long", "State persistence saves completed work. " * 150)
    old, _ = EvidenceCardPipeline(card_max_tokens=60).build_cards([item])
    cached = {old[0].content_hash: old[0].model_dump()}
    enlarged, hits = EvidenceCardPipeline(card_max_tokens=600).build_cards([item], cached_by_hash=cached)
    assert hits == 0 and len(enlarged[0].summary) > len(old[0].summary)


def test_expanded_reserves_leave_research_room_for_older_run_budget_snapshots():
    assert stage_token_reserves(400000, 100000, 100000) == (100000, 100000)
    assert stage_token_reserves(200000, 100000, 100000) == (50000, 50000)
    assert stage_token_reserves(200000, 60000, 15000) == (60000, 15000)
