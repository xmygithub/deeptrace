"""User facts survive failed research; proposals never silently replace facts."""
import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from backend.app.api.v1.learning import memory_reviews
from backend.app.schemas import MessageCreate, RunCreate, SessionCreate
from backend.app.services.run_service import RunService
from deepresearch_agent.harness import SourceMode, WorkflowMode
from deepresearch_agent.memory import MemoryExtractor, MemoryRejected, MemoryService
from deepresearch_agent.persistence import Database
from deepresearch_agent.persistence.repositories import (
    AuditRepository, ContractRepository, MemoryRepository, MessageRepository, RunRepository, SessionRepository,
)


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{(tmp_path / 'durable.db').as_posix()}")
    await database.create_schema()
    yield database
    await database.close()


def service(db):
    return MemoryService(MemoryRepository(db), AuditRepository(db))


async def setup_run(db, text, *, evaluation=False):
    session = await SessionRepository(db).create(SessionCreate(title="用户信息"))
    return await RunRepository(db).create_for_user_message(
        MessageCreate(session_id=session.session_id, role="user", content=text),
        RunCreate(session_id=session.session_id, trigger_message_id="atomic", source_mode=SourceMode.WEB,
                  workflow_mode=WorkflowMode.PLAN_EXECUTE_REPORT, config_snapshot={"evaluation_run": evaluation}))


class Model:
    def __init__(self, quote="我一直偏好用中文交流", *, bad=False):
        self.quote, self.bad, self.inputs = quote, bad, []

    async def ainvoke(self, inputs):
        data = json.loads(inputs[-1].content.split("\n", 1)[1])
        self.inputs.append(data)
        if self.bad:
            raise RuntimeError("provider error containing sensitive data")
        message = data["user_message"]
        return SimpleNamespace(content=json.dumps({"candidates": [
            {"target": "user", "kind": "preference", "content": "用户长期偏好中文交流", "confidence": 0.9,
             "evidence_quote": self.quote, "source_message_id": message["message_id"]}
        ] if self.quote in message["content"] else []}, ensure_ascii=False))


def extractor(db, model, **kwargs):
    return MemoryExtractor(service(db), RunRepository(db), MessageRepository(db), ContractRepository(db), llm=model, **kwargs)


@pytest.mark.asyncio
async def test_user_facts_are_reviewed_even_when_research_failed(db):
    _, run, _ = await setup_run(db, "我一直偏好用中文交流。请研究一个问题。")
    await RunRepository(db).update_status(run.run_id, status="failed", error_code="MODEL_ERROR")
    outcome = await extractor(db, Model()).review_run(run.run_id)
    assert outcome["status"] == "candidates"
    item = outcome["candidates"][0]
    assert item.status == "candidate" and (await service(db).build_snapshot())["rendered"] == ""
    assert outcome["model_calls"] == 1


@pytest.mark.asyncio
async def test_long_input_tail_is_inspected_not_only_first_4000_chars(db):
    quote = "我一直偏好用中文交流"
    _, run, _ = await setup_run(db, "背景材料。" * 1700 + quote)
    model = Model(quote)
    outcome = await extractor(db, model).review_run(run.run_id)
    assert len(outcome["candidates"]) == 1
    assert len(model.inputs) > 1
    assert quote in model.inputs[1]["user_message"]["content"]  # tail checked early
    assert outcome["partial"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("text,evaluation,reason", [
    ("研究一下 RAG 的召回方法。", False, "no_durable_signal"),
    ("我一直偏好用中文交流", True, "evaluation"),
])
async def test_ordinary_questions_and_evaluation_do_not_call_memory_model(db, text, evaluation, reason):
    _, run, _ = await setup_run(db, text, evaluation=evaluation)
    model = Model()
    assert (await extractor(db, model).review_run(run.run_id))["reason"] == reason
    assert model.inputs == []


@pytest.mark.asyncio
async def test_complementary_facts_do_not_implicitly_supersede(db):
    store = service(db)
    first = await store.create_candidate(target="project", content="项目使用 SQLite 保存运行记录", provenance_refs=["message:1"], activate=True)
    second = await store.create_candidate(target="project", content="项目使用 Neo4j 保存知识图谱", provenance_refs=["message:2"])
    assert second.supersedes is None
    await store.update(second.memory_id, status="active")
    assert (await store.repository.get(first.memory_id)).status == "active"


@pytest.mark.asyncio
async def test_explicit_correction_replaces_only_on_approval_and_retains_history(db):
    store = service(db)
    first = await store.create_candidate(target="user", kind="preference", content="用户偏好英文", activate=True)
    second = await store.create_candidate(target="user", kind="preference", content="用户偏好中文", supersedes_id=first.memory_id)
    assert (await store.repository.get(first.memory_id)).status == "active"
    await store.update(second.memory_id, status="active")
    assert (await store.repository.get(first.memory_id)).status == "archived"
    assert (await store.repository.get(second.memory_id)).status == "active"
    assert (await store.repository.get(first.memory_id)).content == "用户偏好英文"


@pytest.mark.asyncio
async def test_stale_correction_cannot_archive_a_changed_predecessor(db):
    store = service(db)
    first = await store.create_candidate(target="user", kind="preference", content="用户偏好英文", activate=True)
    second = await store.create_candidate(target="user", kind="preference", content="用户偏好中文", supersedes_id=first.memory_id)
    await store.update(first.memory_id, content="用户偏好中英双语")
    with pytest.raises(MemoryRejected, match="已变化"):
        await store.update(second.memory_id, status="active")
    assert (await store.repository.get(first.memory_id)).status == "active"
    assert (await store.repository.get(second.memory_id)).status == "candidate"


@pytest.mark.asyncio
async def test_review_outcomes_are_persisted_once_and_provider_errors_are_not_exposed(db, tmp_path):
    _, run, _ = await setup_run(db, "我一直偏好用中文交流")
    scheduler = RunService(db, workflow_factory=lambda *_: None, artifact_root=tmp_path / "artifacts", skills_root=tmp_path / "skills")
    scheduler.memory_extractor = extractor(db, Model(bad=True))
    await scheduler._review_memory(run.run_id)
    await scheduler._review_memory(run.run_id)
    result = await memory_reviews(limit=20, database=db)
    assert len(result["items"]) == 1
    assert result["items"][0]["status"] == "failed"
    assert result["items"][0]["reason"] == "RuntimeError"
    assert "sensitive" not in json.dumps(result)


@pytest.mark.asyncio
async def test_review_reports_candidate_count_and_does_not_activate(db, tmp_path):
    _, run, _ = await setup_run(db, "我一直偏好用中文交流")
    scheduler = RunService(db, workflow_factory=lambda *_: None, artifact_root=tmp_path / "artifacts", skills_root=tmp_path / "skills")
    scheduler.memory_extractor = extractor(db, Model())
    await scheduler._review_memory(run.run_id)
    result = await memory_reviews(limit=20, database=db)
    assert result["items"][0]["candidate_count"] == 1
    assert result["items"][0]["status"] == "candidates"
    assert (await service(db).build_snapshot())["rendered"] == ""


@pytest.mark.asyncio
async def test_partial_failure_reports_candidates_already_persisted(db):
    _, run, _ = await setup_run(db, "我一直偏好用中文交流。" + "背景材料。" * 1700)

    class FailSecondPacket(Model):
        async def ainvoke(self, inputs):
            if self.inputs:
                raise TimeoutError("secret provider text")
            return await super().ainvoke(inputs)

    result = await extractor(db, FailSecondPacket()).review_run(run.run_id)
    assert result["status"] == "failed" and result["reason"] == "TimeoutError"
    assert result["partial"] and len(result["candidates"]) == 1
    assert result["model_calls"] == 2 and result["processed_packets"] == 1
    assert "secret" not in str({key: value for key, value in result.items() if key != "candidates"})


@pytest.mark.asyncio
async def test_concurrent_approvals_recheck_capacity_in_the_write_transaction(db):
    store = MemoryService(MemoryRepository(db), AuditRepository(db), user_max_chars=90)
    first = await store.create_candidate(target="user", content="用户的第一条长期偏好：" + "甲" * 40)
    second = await store.create_candidate(target="user", content="用户的第二条长期偏好：" + "乙" * 40)
    results = await asyncio.gather(store.update(first.memory_id, status="active"), store.update(second.memory_id, status="active"), return_exceptions=True)
    assert sum(isinstance(result, MemoryRejected) for result in results) == 1
    assert len(await store.repository.list_curated(statuses=("active",))) == 1
    assert (await store.capacity("user"))["chars"] <= 90
