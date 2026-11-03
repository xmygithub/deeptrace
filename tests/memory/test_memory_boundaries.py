"""Regression coverage for cross-session isolation and false Memory writes."""

import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from backend.app.api.v1.learning import list_memories
from backend.app.schemas import MessageCreate, RunCreate, SessionCreate
from deepresearch_agent.context import ContextBuilder, ContextCompactor
from deepresearch_agent.context.resolver import QueryResolver
from deepresearch_agent.harness import SourceMode, WorkflowMode
from deepresearch_agent.harness.contracts import ContractCheckData
from deepresearch_agent.harness.run_context import RunContext
from deepresearch_agent.memory import MemoryExtractor, MemoryRejected, MemoryService
from deepresearch_agent.memory.extraction_policy import eligible_memory_evidence
from deepresearch_agent.persistence import Database
from deepresearch_agent.persistence.repositories import (
    AuditRepository, ContractRepository, MemoryRepository, MessageRepository,
    RunRepository, SessionRepository,
)
from deepresearch_agent.sessions import SessionSearchService


@pytest_asyncio.fixture
async def database(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{(tmp_path / 'boundaries.db').as_posix()}")
    await db.create_schema()
    yield db
    await db.close()


def memories(db):
    return MemoryService(MemoryRepository(db), AuditRepository(db))


async def new_run(db, session_id, query, *, evaluation=False, mark_message=False):
    return await RunRepository(db).create_for_user_message(
        MessageCreate(session_id=session_id, role="user", content=query,
                      metadata={"evaluation": True} if mark_message else {}),
        RunCreate(session_id=session_id, trigger_message_id="atomic",
                  source_mode=SourceMode.WEB, workflow_mode=WorkflowMode.PLAN_EXECUTE_REPORT,
                  config_snapshot={"evaluation_run": evaluation}),
    )


async def build_context(db, session_id, query, *, evaluation=False):
    trigger, run, _ = await new_run(db, session_id, query, evaluation=evaluation)
    messages, sessions = MessageRepository(db), SessionRepository(db)
    builder = ContextBuilder(
        messages=messages, sessions=sessions, memory_service=memories(db),
        session_search=SessionSearchService(messages, sessions),
        compactor=ContextCompactor(sessions, messages),
    )
    context = RunContext(run_id=run.run_id, session_id=session_id,
                         trigger_message_id=trigger.message_id, source_mode=SourceMode.WEB,
                         workflow_mode=WorkflowMode.PLAN_EXECUTE_REPORT,
                         status="context_building", original_query=query,
                         config_snapshot={"evaluation_run": evaluation})
    return builder, context


@pytest.mark.parametrize("query", [
    "调研 AI Agent 中工作流与自主规划两种方式的区别，比较它们的适用场景、可靠性和成本，并结合公开案例说明如何选择。请引用来源，区分资料结论与你的建议。",
    "比较 Python 与 Java，并解释它们的差异。",
    "研究过去十年数据库的发展。",
    "比较 SQLite 之前版本和新版本的性能。",
    "分析之前十年的研究进展。",
])
def test_self_contained_questions_do_not_request_history(query):
    assert not SessionSearchService.should_search(query, unresolved_reference=True)
    assert QueryResolver().resolve(query, []).confidence == 1


@pytest.mark.asyncio
async def test_new_session_pronouns_do_not_import_other_conversations(database):
    sessions = SessionRepository(database)
    old = await sessions.create(SessionCreate(title="旧资料"))
    await new_run(database, old.session_id, "Agent 工作流的旧讨论，不应混入")
    current = await sessions.create(SessionCreate(title="独立问题"))
    builder, context = await build_context(database, current.session_id,
        "调研 AI Agent 中工作流与自主规划两种方式的区别，比较它们的适用场景、可靠性和成本。")
    result = await builder.build(context)
    assert not result.context_snapshot["historical_recall_searched"]
    assert result.context_snapshot["used_session_ids"] == []
    assert "不应混入" not in result.model_input


@pytest.mark.asyncio
async def test_unresolved_followup_asks_user_instead_of_searching_globally(database):
    sessions = SessionRepository(database)
    old = await sessions.create(SessionCreate(title="某方案"))
    await new_run(database, old.session_id, "这个方案是 SQLite WAL")
    current = await sessions.create(SessionCreate(title="新会话"))
    builder, context = await build_context(database, current.session_id, "继续解释这个方案")
    with pytest.raises(ValueError, match="needs_user_input"):
        await builder.build(context)
    assert "明确" in context.config_snapshot["clarification_question"]


@pytest.mark.asyncio
async def test_same_session_followup_wins_unless_other_session_is_explicit(database):
    sessions = SessionRepository(database)
    old = await sessions.create(SessionCreate(title="其他 SQLite"))
    await new_run(database, old.session_id, "SQLite 外部会话标记")
    current = await sessions.create(SessionCreate(title="当前会话"))
    await new_run(database, current.session_id, "SQLite 当前会话方案")
    builder, context = await build_context(database, current.session_id, "继续上次 SQLite 的讨论")
    result = await builder.build(context)
    assert not result.context_snapshot["historical_recall_searched"]
    assert "当前会话方案" in result.model_input
    assert "外部会话标记" not in result.model_input
    builder, context = await build_context(database, current.session_id, "继续另一个会话 SQLite 的讨论")
    result = await builder.build(context)
    assert result.context_snapshot["used_session_ids"] == [old.session_id]


@pytest.mark.asyncio
async def test_explicit_history_requires_unambiguous_session(database):
    sessions = SessionRepository(database)
    old = await sessions.create(SessionCreate(title="SQLite 甲"))
    await new_run(database, old.session_id, "SQLite 恢复方案甲")
    current = await sessions.create(SessionCreate(title="新会话"))
    builder, context = await build_context(database, current.session_id, "继续上次 SQLite 持久化的讨论")
    result = await builder.build(context)
    assert result.context_snapshot["used_session_ids"] == [old.session_id]
    other = await sessions.create(SessionCreate(title="SQLite 乙"))
    await new_run(database, other.session_id, "SQLite 恢复方案乙")
    fresh = await sessions.create(SessionCreate(title="再次新建"))
    builder, context = await build_context(database, fresh.session_id, "继续上次 SQLite 的讨论")
    with pytest.raises(ValueError, match="needs_user_input"):
        await builder.build(context)
    assert old.session_id in context.config_snapshot["clarification_question"]
    assert other.session_id in context.config_snapshot["clarification_question"]
    context.config_snapshot["clarifications"] = [{"content": other.session_id}]
    result = await builder.build(context)
    assert result.context_snapshot["used_session_ids"] == [other.session_id]
    assert "恢复方案乙" in result.model_input and "恢复方案甲" not in result.model_input


@pytest.mark.asyncio
async def test_search_excludes_eval_anchors_and_hydrated_windows_before_limit(database):
    sessions, messages = SessionRepository(database), MessageRepository(database)
    # Enough evaluation hits to starve legitimate results if filtered AFTER LIMIT.
    evaluation = await sessions.create(SessionCreate(title="已重命名的评测"))
    for index in range(22):
        _, run, _ = await new_run(database, evaluation.session_id, f"SQLite 评测题 {index}", evaluation=True)
        await messages.append(MessageCreate(session_id=evaluation.session_id, run_id=run.run_id,
                                            role="assistant", content="SQLite 评测答案"))
    legacy = await sessions.create(SessionCreate(title="[Skill Eval] legacy"))
    await new_run(database, legacy.session_id, "SQLite 旧评测题")
    normal = await sessions.create(SessionCreate(title="真实会话"))
    await new_run(database, normal.session_id, "SQLite 正常历史")
    _, mixed_eval, _ = await new_run(database, normal.session_id, "SQLite 混入评测题", evaluation=True)
    await messages.append(MessageCreate(session_id=normal.session_id, run_id=mixed_eval.run_id,
                                        role="assistant", content="混入评测答案"))
    await new_run(database, normal.session_id, "SQLite metadata评测题", mark_message=True)
    search = SessionSearchService(messages, sessions)
    results = await search.search("上次 SQLite 的讨论", limit=1)
    assert [item.session_id for item in results] == [normal.session_id]
    rendered = json.dumps(results[0].to_dict(), ensure_ascii=False)
    assert "正常历史" in rendered and "评测" not in rendered
    # Inspection still exposes persisted records; filtering does not delete them.
    assert "评测答案" in json.dumps(await search.read(evaluation.session_id), ensure_ascii=False)


@pytest.mark.parametrize("message,quote,target,expected", [
    ("假设一家企业知识库使用向量检索，无法重新训练模型，且需要控制调用成本。", "无法重新训练模型，且需要控制调用成本", "project", False),
    ("请比较方案，本次要求我们项目只使用 SQLite。", "我们项目只使用 SQLite", "project", False),
    ("假设我们的项目使用 SQLite。", "我们的项目使用 SQLite", "project", False),
    ("如果我们项目采用 SQLite，请分析利弊。", "我们项目采用 SQLite", "project", False),
    ("以企业为例，比如我们项目使用 SQLite。", "我们项目使用 SQLite", "project", False),
    ("我偏好简短报告。请调研知识库。", "我偏好简短报告", "user", True),
    ("这个表格太复杂了，我更容易阅读简洁的要点。", "我更容易阅读简洁的要点", "user", True),
    ("我们的项目使用 SQLite 保存 Run。", "我们的项目使用 SQLite 保存 Run", "project", True),
    ("我一直偏好中文。假设一家企业需要改进 RAG。", "我一直偏好中文", "user", True),
    ("假设一家企业使用 MySQL。现实中，我们项目采用 SQLite。", "我们项目采用 SQLite", "project", True),
    ("我们项目是否使用 SQLite？", "我们项目是否使用 SQLite", "project", False),
    ("我们项目使用 SQLite 吗？", "我们项目使用 SQLite", "project", False),
    ("我们项目使用 SQLite，怎么评估性能？", "我们项目使用 SQLite", "project", True),
])
def test_evidence_scope_is_checked_even_if_model_confidence_is_high(message, quote, target, expected):
    assert eligible_memory_evidence(message, quote, target) is expected


class FakeLlm:
    def __init__(self, candidates):
        self.candidates, self.calls = candidates, 0

    async def ainvoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=json.dumps({"candidates": self.candidates}, ensure_ascii=False))


async def finish_run(db, query, *, evaluation=False):
    session = await SessionRepository(db).create(SessionCreate(title="记忆提取"))
    trigger, run, _ = await new_run(db, session.session_id, query, evaluation=evaluation)
    await ContractRepository(db).upsert(ContractCheckData(
        check_id=f"check-{run.run_id}", run_id=run.run_id, kind="source_match",
        verifier="test", verifier_version="1", passed=True))
    await RunRepository(db).complete_verified(run.run_id, assistant_content="经过验证的回答", usage={})
    return trigger, run


@pytest.mark.asyncio
async def test_false_project_fact_is_not_persisted_and_existing_candidates_are_untouched(database):
    service = memories(database)
    legacy = await service.create_candidate(target="project", kind="fact", content="历史待审候选",
                                             provenance_refs=["message:legacy"], created_by="llm_review")
    _, run = await finish_run(database, "假设一家企业需要控制成本。我们项目无法重新训练模型，请调研改进方案。")
    llm = FakeLlm([dict(target="project", kind="fact", content="项目不能训练模型",
                        confidence=0.99, evidence_quote="我们项目无法重新训练模型", rationale="用户明确说明")])
    extractor = MemoryExtractor(service, RunRepository(database), MessageRepository(database), ContractRepository(database), llm=llm)
    assert await extractor.extract_from_completed_run(run.run_id) == []
    assert (await service.repository.get(legacy.memory_id)).status == "candidate"
    assert len(await service.repository.list_curated(statuses=("candidate",))) == 1


@pytest.mark.asyncio
async def test_durable_fact_saves_evidence_for_api_review_but_does_not_activate(database):
    service = memories(database)
    quote = "我们项目使用 SQLite 保存 Run"
    trigger, run = await finish_run(database, quote + "。请解释优缺点。")
    llm = FakeLlm([dict(target="project", kind="fact", content="项目用 SQLite 持久化 Run",
                        confidence=0.9, evidence_quote=quote, rationale="用户描述了自身项目的实现")])
    extractor = MemoryExtractor(service, RunRepository(database), MessageRepository(database), ContractRepository(database), llm=llm)
    created = await extractor.extract_from_completed_run(run.run_id)
    assert len(created) == 1 and created[0].status == "candidate"
    payload = await list_memories(q=None, status_filter=None, target=None, database=database)
    item = payload["items"][0]
    assert item["extraction"] == {"evidence_quote": quote, "rationale": "用户描述了自身项目的实现"}
    assert item["provenance_refs"] == [f"run:{run.run_id}", f"message:{trigger.message_id}"]
    assert not (await service.build_snapshot())["items"]["project"]
    # Legacy data has no invented extraction explanation.
    await service.create_candidate(target="user", content="旧候选", created_by="llm_review")
    payload = await list_memories(q=None, status_filter=None, target=None, database=database)
    assert next(i for i in payload["items"] if i["content"] == "旧候选")["extraction"] is None


@pytest.mark.asyncio
async def test_eval_runs_never_extract_memory_or_recall_live_history(database):
    _, run = await finish_run(database, "我偏好简短报告", evaluation=True)
    llm = FakeLlm([])
    extractor = MemoryExtractor(memories(database), RunRepository(database), MessageRepository(database), ContractRepository(database), llm=llm)
    assert await extractor.extract_from_completed_run(run.run_id) == [] and llm.calls == 0
    sessions = SessionRepository(database)
    old = await sessions.create(SessionCreate(title="真实 SQLite 历史"))
    await new_run(database, old.session_id, "SQLite 真实讨论")
    fresh = await sessions.create(SessionCreate(title="新的评测"))
    builder, context = await build_context(database, fresh.session_id, "请查找上次 SQLite 的讨论", evaluation=True)
    result = await builder.build(context)
    assert not result.context_snapshot["historical_recall_searched"]
    assert "真实讨论" not in result.model_input


@pytest.mark.asyncio
async def test_extraction_audit_does_not_store_secrets(database):
    service = memories(database)
    with pytest.raises(MemoryRejected):
        await service.create_candidate(target="user", content="用户偏好中文", created_by="llm_review",
            extraction={"evidence_quote": "我偏好中文", "rationale": "api_key=secret-example"})
    assert not await service.repository.list_curated(statuses=("candidate",))
