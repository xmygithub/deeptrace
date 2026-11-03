"""Actual provenance reaches writing, including cached cards and digests."""
import json

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from deepresearch_agent.agents.multi_agent.reporter.evidence_cards import EvidenceCardPipeline
from deepresearch_agent.agents.multi_agent.reporter.section_writer import SectionWriter
from deepresearch_agent.retrieval.tavily_provider import TavilyProvider
from deepresearch_agent.context.tokens import count_tokens


def item(kind="search_summary"):
    return RetrievalResult(result_id="ev_a", source="tavily_search", source_mode="web", granularity="DO",
        evidence="支持恢复已有记录，但并非在任何故障下保证不丢失。" * 40,
        metadata={"source_id": "https://docs.example.org/a", "source_type": "webpage",
                  "extra": {"content_kind": kind, "content_truncated": True, "authority_rank": 2}})


def test_search_results_distinguish_raw_page_from_summary(tmp_path):
    provider = TavilyProvider(api_key="", client=object(), cache_dir=tmp_path)
    rows = provider._map_results({"results": [
        {"url": "https://example.org/a", "content": "搜索摘要", "raw_content": None},
        {"url": "https://example.org/b", "content": "摘要", "raw_content": "完整页面正文"},
        {"url": "https://example.org/c", "content": "回退摘要", "raw_content": "  "},
    ]}, artifact=None)
    by_url = {r.metadata.url: r for r in rows}
    assert by_url["https://example.org/a"].metadata.extra["content_kind"] == "search_summary"
    assert by_url["https://example.org/b"].metadata.extra["raw_content_available"] is True
    assert by_url["https://example.org/c"].evidence == "回退摘要"


def test_card_and_digest_keep_provenance_next_to_each_original_id():
    pipeline = EvidenceCardPipeline(card_max_tokens=60)
    cards, _ = pipeline.build_cards([item()])
    assert cards[0].card_truncated and cards[0].source_truncated
    writer = SectionWriter(llm=object())
    row = json.loads(writer._format_evidence([pipeline._card_result(cards[0])]))
    assert row["content_kind"] == "search_summary" and row["authority_rank"] == 2
    assert row["source_truncated"] and row["card_truncated"]
    digest = pipeline._cards_digest(cards, index=0, level=1)
    packed = pipeline._digest_result(digest, cards)
    row = json.loads(writer._format_evidence([packed]).split("\n", 1)[1])
    assert row["sources"][0]["evidence_id"] == "ev_a"
    assert row["sources"][0]["content_kind"] == "search_summary"
    assert row["sources"][0]["card_truncated"]


def test_cache_refreshes_provenance_without_guessing_missing_legacy_fields():
    pipeline = EvidenceCardPipeline()
    original = item()
    original.evidence = "一条相同的来源文本。"
    cards, _ = pipeline.build_cards([original])
    cache = {cards[0].content_hash: cards[0].model_dump(mode="json")}
    legacy = original.model_copy(deep=True)
    legacy.metadata.extra = {}
    rebuilt, hits = pipeline.build_cards([legacy], cached_by_hash=cache)
    assert hits == 1
    assert rebuilt[0].content_kind == "unknown" and rebuilt[0].authority_level == 0
    assert not rebuilt[0].source_truncated


def test_archived_provenance_requires_exact_body_match_and_never_changes_evidence(tmp_path):
    from scripts.evaluate_saved_writing import restore_provenance
    original = item()
    original.evidence = "只有这一条保存的搜索摘要。"
    original.metadata.url = original.metadata.source_id
    original.metadata.extra = {"artifact_path": "call.json"}
    artifact = tmp_path / "call.json"
    artifact.write_text(json.dumps({"results": [{"url": original.metadata.url,
        "content": original.evidence, "raw_content": None}]}, ensure_ascii=False), encoding="utf-8")
    before = artifact.read_bytes()
    annotations = restore_provenance([original], tmp_path)
    assert annotations["ev_a"]["content_kind"] == "search_summary"
    assert original.evidence == "只有这一条保存的搜索摘要。" and artifact.read_bytes() == before
    wrong = original.model_copy(deep=True)
    wrong.evidence = "与存档不匹配的正文。"
    wrong.metadata.extra = {"artifact_path": "call.json"}
    assert restore_provenance([wrong], tmp_path) == {}
    assert "content_kind" not in wrong.metadata.extra
    wrong.metadata.extra = {"artifact_path": "../call.json"}
    assert restore_provenance([wrong], tmp_path) == {}


def test_single_long_source_is_not_dropped_when_json_list_overhead_fills_budget():
    from deepresearch_agent.retrieval.evidence_context import source_packet
    original = item("page_text")
    original.evidence = "原文事实与限制。" * 1000
    packets = source_packet([original], max_tokens=500, per_source_tokens=500)
    assert len(packets) == 1 and packets[0]["evidence_id"] == original.result_id
    assert original.evidence.startswith(packets[0]["text"]) and packets[0]["truncated"]
    assert count_tokens(json.dumps(packets, ensure_ascii=False)) <= 500
