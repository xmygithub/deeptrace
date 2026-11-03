import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from backend.app.schemas import MessageCreate, SessionCreate
from deepresearch_agent.context import ContextBuilder, ContextCompactor
from deepresearch_agent.harness.run_context import RunContext
from deepresearch_agent.memory import MemoryService
from deepresearch_agent.persistence import Database
from deepresearch_agent.persistence.repositories import AuditRepository, MemoryRepository, MessageRepository, SessionRepository
from deepresearch_agent.sessions import SessionSearchService


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{(tmp_path / 'compact.db').as_posix()}")
    await database.create_schema()
    yield database
    await database.close()


class Summarizer:
    def __init__(self, *, bad=False, delay=0):
        self.inputs, self.bad, self.delay = [], bad, delay

    async def ainvoke(self, messages):
        value = json.loads(messages[-1].content)
        self.inputs.append(value)
        await asyncio.sleep(self.delay)
        if self.bad:
            return SimpleNamespace(content='{"constraints":[{"text":"invented","message_ids":["unknown"]}]}')
        previous = value["previous_summary"]
        refs = previous.get("constraints", [])
        text = "保留项目约束"
        if any("尾部长期约束" in m["content"] for m in value["messages"]):
            text = "尾部长期约束"
        return SimpleNamespace(content=json.dumps({"goal": "继续研究", "constraints": refs or [
            {"text": text, "message_ids": [value["messages"][0]["message_id"]]}]}, ensure_ascii=False))


async def history(db, count=24, size=80):
    session = await SessionRepository(db).create(SessionCreate(title="压缩回归"))
    repo, rows = MessageRepository(db), []
    for i in range(count):
        row, _ = await repo.append(MessageCreate(session_id=session.session_id, role="user" if i % 2 == 0 else "assistant",
            content=f"消息{i}：" + "历史研究资料" * size))
        rows.append(row)
    return session, rows


@pytest.mark.asyncio
async def test_compaction_reuses_summary_and_processes_only_uncovered_messages(db):
    session, rows = await history(db)
    model = Summarizer()
    engine = ContextCompactor(SessionRepository(db), MessageRepository(db), threshold_tokens=8000,
        protect_recent_messages=4, target_tokens=500, llm=model)
    first = await engine.prepare(session.session_id)
    assert first.compaction["status"] == "completed"
    assert first.summary["covered_through_message_id"] == rows[-5].message_id
    assert len(first.messages) == 4
    calls = len(model.inputs)
    second = await engine.prepare(session.session_id)
    assert second.compaction["status"] == "reused" and len(model.inputs) == calls
    for i in range(12):
        await MessageRepository(db).append(MessageCreate(session_id=session.session_id, role="user", content="新增资料" * 300))
    third = await engine.prepare(session.session_id)
    assert third.summary["version"] == 2
    assert model.inputs[calls]["previous_summary"]["constraints"] == first.summary["constraints"]
    old_ids = {r.message_id for r in rows[:-4]}
    assert not old_ids.intersection(m["message_id"] for p in model.inputs[calls:] for m in p["messages"])
    assert len(await MessageRepository(db).context_messages(session.session_id)) == 36


@pytest.mark.asyncio
async def test_failed_summary_preserves_sources_and_previous_cursor(db):
    session, rows = await history(db)
    engine = ContextCompactor(SessionRepository(db), MessageRepository(db), threshold_tokens=100,
        protect_recent_messages=2, llm=Summarizer(bad=True))
    result = await engine.prepare(session.session_id)
    assert result.compaction["reason"] == "summary_unknown_source"
    assert len(result.messages) == len(rows)
    assert (await SessionRepository(db).get(session.session_id)).summary_json is None


@pytest.mark.asyncio
async def test_same_session_parallel_builds_compact_once(db):
    session, _ = await history(db, size=30)
    model = Summarizer(delay=0.01)
    engine = ContextCompactor(SessionRepository(db), MessageRepository(db), threshold_tokens=1500,
        protect_recent_messages=2, target_tokens=500, llm=model)
    a, b = await asyncio.gather(engine.prepare(session.session_id), engine.prepare(session.session_id))
    assert {a.compaction["status"], b.compaction["status"]} == {"completed", "reused"}
    assert len(model.inputs) == 1


@pytest.mark.asyncio
async def test_latest_cursor_and_pagination_pass_500_without_future_leak(db):
    session, rows = await history(db, count=510, size=0)
    repo = MessageRepository(db)
    latest = await repo.context_messages(session.session_id, latest=True, limit=16)
    assert latest[-1].message_id == rows[-1].message_id
    bounded = await repo.context_messages(session.session_id, before_id=rows[505].message_id, latest=True, limit=16)
    assert bounded[-1].message_id == rows[504].message_id
    engine = ContextCompactor(SessionRepository(db), repo, threshold_tokens=100000)
    result = await engine.prepare(session.session_id, before_message_id=rows[-1].message_id)
    assert len(result.messages) == 509


@pytest.mark.asyncio
async def test_over_eight_turns_below_pressure_remain_in_model_input(db):
    session, rows = await history(db, count=20, size=0)
    repo, sessions = MessageRepository(db), SessionRepository(db)
    trigger, _ = await repo.append(MessageCreate(session_id=session.session_id, role="user", content="请完整梳理研究资料"))
    builder = ContextBuilder(messages=repo, sessions=sessions,
        memory_service=MemoryService(MemoryRepository(db), AuditRepository(db)),
        session_search=SessionSearchService(repo, sessions), compactor=ContextCompactor(sessions, repo), recent_turns=8)
    context = RunContext(run_id="run-test", session_id=session.session_id, trigger_message_id=trigger.message_id,
        original_query=trigger.content, source_mode="web", workflow_mode="deep_research", status="context_building")
    result = await builder.build(context)
    assert len(result.context_snapshot["recent_messages"]) == 20
    assert rows[0].content in result.model_input and rows[-1].content in result.model_input


@pytest.mark.asyncio
async def test_summary_timeout_does_not_advance_coverage(db):
    session, rows = await history(db)
    engine = ContextCompactor(SessionRepository(db), MessageRepository(db), threshold_tokens=100,
        llm=Summarizer(delay=0.1), timeout_seconds=0.01)
    result = await engine.prepare(session.session_id)
    assert result.compaction["status"] == "failed" and result.compaction["reason"] == "TimeoutError"
    assert len(result.messages) == len(rows)


@pytest.mark.asyncio
async def test_long_message_tail_is_sent_without_head_truncation(db):
    session, _ = await history(db, count=0)
    repo = MessageRepository(db)
    await repo.append(MessageCreate(session_id=session.session_id, role="user", content="背景资料" * 1800 + "尾部长期约束"))
    for _ in range(2):
        await repo.append(MessageCreate(session_id=session.session_id, role="assistant", content="近期消息"))
    model = Summarizer()
    result = await ContextCompactor(SessionRepository(db), repo, threshold_tokens=100, protect_recent_messages=2,
        llm=model, max_input_chars=4000).prepare(session.session_id)
    assert result.compaction["status"] == "completed"
    assert any("尾部长期约束" in m["content"] for p in model.inputs for m in p["messages"])


@pytest.mark.asyncio
async def test_older_run_never_uses_summary_from_future_messages(db):
    session, rows = await history(db, count=30)
    sessions, messages = SessionRepository(db), MessageRepository(db)
    model = Summarizer()
    engine = ContextCompactor(sessions, messages, threshold_tokens=100, protect_recent_messages=4, llm=model)
    await engine.prepare(session.session_id)
    saved = (await sessions.get(session.session_id)).summary_json
    model.inputs.clear()
    result = await engine.prepare(session.session_id, before_message_id=rows[10].message_id)
    future_ids = {row.message_id for row in rows[10:]}
    assert not future_ids.intersection(row.message_id for row in result.messages)
    assert not future_ids.intersection(engine._references(result.summary))
    assert not future_ids.intersection(m["message_id"] for p in model.inputs for m in p["messages"])
    assert (await sessions.get(session.session_id)).summary_json == saved


@pytest.mark.asyncio
async def test_separate_compactors_use_compare_and_swap_instead_of_overwriting(db):
    session, _ = await history(db)
    sessions, messages = SessionRepository(db), MessageRepository(db)
    a = ContextCompactor(sessions, messages, threshold_tokens=100, protect_recent_messages=4, llm=Summarizer(delay=.05))
    b = ContextCompactor(sessions, messages, threshold_tokens=100, protect_recent_messages=4, llm=Summarizer(delay=.05))
    results = await asyncio.gather(a.prepare(session.session_id), b.prepare(session.session_id))
    assert {r.compaction["status"] for r in results} == {"completed", "conflict"}
    assert json.loads((await sessions.get(session.session_id)).summary_json)["version"] == 1


@pytest.mark.asyncio
async def test_failed_increment_keeps_existing_summary_and_uncovered_history(db):
    session, _ = await history(db)
    sessions, messages = SessionRepository(db), MessageRepository(db)
    engine = ContextCompactor(sessions, messages, threshold_tokens=100, protect_recent_messages=4, llm=Summarizer())
    first = await engine.prepare(session.session_id)
    saved = (await sessions.get(session.session_id)).summary_json
    await messages.append(MessageCreate(session_id=session.session_id, role="user", content="新增长资料" * 300))
    engine.llm = Summarizer(bad=True)
    result = await engine.prepare(session.session_id)
    assert result.summary == first.summary and len(result.messages) == len(first.messages) + 1
    assert result.compaction["status"] == "failed" and result.compaction["model_calls"] == 1
    assert (await sessions.get(session.session_id)).summary_json == saved


@pytest.mark.asyncio
async def test_recent_window_shortens_under_pressure_before_discarding_history(db):
    session, rows = await history(db, count=18)
    result = await ContextCompactor(SessionRepository(db), MessageRepository(db), threshold_tokens=8000,
        protect_recent_messages=16, target_tokens=500, llm=Summarizer()).prepare(session.session_id)
    assert result.compaction["status"] == "completed"
    assert 2 <= result.compaction["recent_message_count"] < 16
    assert result.messages[-1].message_id == rows[-1].message_id
    assert len(result.messages) + result.summary["covered_message_count"] == len(rows)


@pytest.mark.asyncio
async def test_large_backlog_advances_in_bounded_batches_without_skipping_messages(db):
    session, rows = await history(db, count=30)
    model = Summarizer()
    sessions, messages = SessionRepository(db), MessageRepository(db)
    engine = ContextCompactor(sessions, messages, threshold_tokens=100, protect_recent_messages=2,
        target_tokens=500, llm=model, max_passes=2, max_input_chars=2000)
    first = await engine.prepare(session.session_id)
    assert first.compaction["status"] == "completed" and first.compaction["model_calls"] <= 2
    assert first.compaction["pending_message_count"] > 0
    covered = first.summary["covered_message_count"]
    assert first.summary["covered_through_message_id"] == rows[covered - 1].message_id
    calls = len(model.inputs)
    second = await engine.prepare(session.session_id)
    assert second.summary["version"] == 2
    assert second.summary["covered_message_count"] > covered
    assert {r.message_id for r in rows[:covered]}.isdisjoint(
        m["message_id"] for p in model.inputs[calls:] for m in p["messages"])
