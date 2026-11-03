"""Restart recovery against temporary SQLite and deterministic tools only."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import update

from backend.app.schemas import MessageCreate, RunCreate, SessionCreate
from backend.app.services.run_service import RunService
from deepresearch_agent.harness.budgets import BudgetLimits
from deepresearch_agent.harness.run_context import RunContext
from deepresearch_agent.harness.contracts import SourceMode, WorkflowMode, RunStatus
from deepresearch_agent.persistence import Database
from deepresearch_agent.persistence.models import RunModel
from deepresearch_agent.persistence.repositories import CheckpointRepository, EventRepository, MessageRepository, RunRepository, SessionRepository
from scripts.evaluate_recovery_e2e import DeterministicRecoveryDriver
from scripts.evaluate_recovery_e2e import _runtime


@pytest_asyncio.fixture
async def database(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{(tmp_path / 'restart.db').as_posix()}")
    await db.create_schema()
    yield db
    await db.close()


def service(db, root, driver=DeterministicRecoveryDriver, **kwargs):
    return RunService(db, workflow_factory=driver, artifact_root=root / "artifacts", skills_root=root / "skills",
                      recovery_interval_seconds=0.02, **kwargs)


async def until(predicate, timeout=5):
    async with asyncio.timeout(timeout):
        while True:
            value = await predicate()
            if value:
                return value
            await asyncio.sleep(0.02)


async def status_is(db, rid, status):
    row = await RunRepository(db).get(rid)
    return row if row.status == status else None


async def event_is(db, rid, kind):
    return await EventRepository(db).latest_of_type(rid, kind)


async def completed(db, app, rid):
    await until(lambda: status_is(db, rid, "completed"))
    # Status commits before the completion event and executor cleanup. Wait for
    # the whole scheduled task so assertions/shutdown cannot interrupt either.
    task = app._tasks.get(rid)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), 5)
    assert await event_is(db, rid, "run.completed")


async def seed(db, root, *, checkpoint=False):
    session = await SessionRepository(db).create(SessionCreate(title="Restart regression"))
    marker = root / "tool-side-effects.log"
    message, run, _ = await RunRepository(db).create_for_user_message(
        MessageCreate(session_id=session.session_id, role="user", content="offline recovery", client_message_id=session.session_id),
        RunCreate(session_id=session.session_id, trigger_message_id="atomic", source_mode=SourceMode.GRAPHRAG,
                  workflow_mode=WorkflowMode.PLAN_EXECUTE_REPORT,
                  config_snapshot={"evaluation_run": True, "side_effect_file": str(marker), "min_evidence": 1},
                  budget=BudgetLimits().model_dump()),
    )
    if checkpoint:
        context = RunContext(run_id=run.run_id, session_id=session.session_id, trigger_message_id=message.message_id,
                             source_mode=SourceMode.GRAPHRAG, workflow_mode=WorkflowMode.PLAN_EXECUTE_REPORT,
                             status=RunStatus.EXECUTING, original_query=message.content,
                             config_snapshot=json.loads(run.config_snapshot_json), budget_limits=BudgetLimits())
        driver = DeterministicRecoveryDriver(context)
        await driver.plan()
        await driver.execute()
        context.workflow_state = driver.snapshot()
        context.context_snapshot["inflight_stage"] = "executing"
        await CheckpointRepository(db).save(run.run_id, "executing", context.model_dump(mode="json"))
        await RunRepository(db).update_status(run.run_id, status="executing", current_stage="executing")
    return run.run_id, marker


async def expire(db, rid):
    async with db.transaction() as session:
        await session.execute(update(RunModel).where(RunModel.run_id == rid).values(
            lease_expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")))


@pytest.mark.asyncio
async def test_fast_restart_recovers_after_future_lease_expires_without_repeating_saved_tool(database, tmp_path):
    rid, marker = await seed(database, tmp_path, checkpoint=True)
    assert await RunRepository(database).acquire_lease(rid, "dead-process", ttl_seconds=90)
    app = service(database, tmp_path)
    try:
        assert await app.startup_recovery(auto_resume=True) == []
        assert await event_is(database, rid, "run.recovery_waiting")
        await asyncio.sleep(0.06)
        assert (await RunRepository(database).get(rid)).status == "executing"
        # Advance only the expired lease, after the startup scan already ran.
        await expire(database, rid)
        await completed(database, app, rid)
        assert marker.read_text().splitlines() == [rid]
        events = await EventRepository(database).list_after(rid)
        assert sum(e.event_type == "run.resumed" for e in events) == 1
        assert sum(e.event_type == "run.interrupted" for e in events) == 1
        assert sum(e.event_type == "run.completed" for e in events) == 1
        assert sum(e.event_type == "run.recovery_waiting" for e in events) == 1
        messages = await MessageRepository(database).list_for_session((await RunRepository(database).get(rid)).session_id)
        assert sum(m.role == "assistant" for m in messages) == 1
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_manual_resume_waits_for_old_lease_and_keeps_execution_checkpoint(database, tmp_path):
    rid, marker = await seed(database, tmp_path, checkpoint=True)
    repo = RunRepository(database)
    await repo.acquire_lease(rid, "dead-process", ttl_seconds=90)
    await repo.request_pause(rid)
    await repo.finalize_pause(rid)  # Same path as pausing an orphan after restart.
    app = service(database, tmp_path)
    try:
        assert await app.resume(rid)
        pending = app._tasks[rid]
        assert app.schedule(rid) is pending
        await until(lambda: event_is(database, rid, "run.recovery_waiting"))
        assert (await repo.get(rid)).status == "paused"
        assert marker.read_text().splitlines() == [rid]
        await expire(database, rid)
        await completed(database, app, rid)
        assert marker.read_text().splitlines() == [rid]
        events = await EventRepository(database).list_after(rid)
        # No planning restart: resume the saved executing stage.
        assert not any(e.event_type == "run.stage_changed" and e.stage == "planning" for e in events)
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_waiting_resume_can_be_cancelled_without_reviving_it(database, tmp_path):
    rid, _ = await seed(database, tmp_path)
    repo = RunRepository(database)
    await repo.update_status(rid, status="planning", current_stage="planning")
    await repo.acquire_lease(rid, "dead-process", ttl_seconds=90)
    await repo.request_pause(rid)
    await repo.finalize_pause(rid)
    app = service(database, tmp_path)
    try:
        await app.startup_recovery(auto_resume=True)
        assert await app.resume(rid)
        await until(lambda: event_is(database, rid, "run.recovery_waiting"))
        assert await app.cancel(rid)
        await until(lambda: status_is(database, rid, "cancelled"))
        await expire(database, rid)
        await asyncio.sleep(0.08)
        assert (await repo.get(rid)).status == "cancelled"
        assert not await event_is(database, rid, "run.resumed")
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_resume_wait_is_bounded_and_reports_timeout(database, tmp_path):
    rid, _ = await seed(database, tmp_path)
    repo = RunRepository(database)
    await repo.update_status(rid, status="planning", current_stage="planning")
    await repo.acquire_lease(rid, "still-owned", ttl_seconds=90)
    await repo.request_pause(rid)
    await repo.finalize_pause(rid)
    app = service(database, tmp_path, lease_wait_timeout_seconds=0.08)
    try:
        await app.resume(rid)
        await until(lambda: event_is(database, rid, "run.recovery_deferred"))
        row = await repo.get(rid)
        assert row.status == "paused" and row.lease_owner == "still-owned"
        assert not json.loads(row.config_snapshot_json).get("resume_requested")
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_disabled_auto_resume_marks_orphans_but_does_not_execute(database, tmp_path):
    rid, marker = await seed(database, tmp_path)
    app = service(database, tmp_path)
    try:
        await app.startup_recovery(auto_resume=False)
        await asyncio.sleep(0.06)
        assert (await RunRepository(database).get(rid)).status == "interrupted"
        assert rid not in app._tasks and not marker.exists()
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_pending_manual_resume_survives_another_restart_but_regular_pause_does_not_autostart(database, tmp_path):
    rid, marker = await seed(database, tmp_path, checkpoint=True)
    repo = RunRepository(database)
    await repo.acquire_lease(rid, "dead-process", ttl_seconds=90)
    await repo.request_pause(rid)
    await repo.finalize_pause(rid)
    first = service(database, tmp_path)
    second = service(database, tmp_path)
    try:
        await first.startup_recovery(auto_resume=True)
        await asyncio.sleep(0.06)
        assert rid not in first._tasks  # Pausing intentionally is respected.
        assert await first.resume(rid)
        await until(lambda: event_is(database, rid, "run.recovery_waiting"))
        await first.shutdown()
        row = await repo.get(rid)
        assert row.status == "paused" and json.loads(row.config_snapshot_json)["resume_requested"]
        await second.startup_recovery(auto_resume=True)
        await expire(database, rid)
        await completed(database, second, rid)
        assert marker.read_text().splitlines() == [rid]
    finally:
        await first.shutdown()
        await second.shutdown()


@pytest.mark.asyncio
async def test_shutdown_preserves_checkpoint_then_restart_completes_once(database, tmp_path):
    rid, marker = await seed(database, tmp_path)
    started = asyncio.Event()

    class StopAfterSave(DeterministicRecoveryDriver):
        async def execute(self):
            await super().execute()
            self.context.workflow_state = self.snapshot()
            self.context.context_snapshot["inflight_stage"] = "executing"
            await CheckpointRepository(database).save(rid, "executing", self.context.model_dump(mode="json"))
            started.set()
            await asyncio.Event().wait()

    first = service(database, tmp_path, StopAfterSave)
    second = service(database, tmp_path)
    try:
        first.schedule(rid)
        await asyncio.wait_for(started.wait(), 5)
        before = await CheckpointRepository(database).latest(rid)
        await first.shutdown()
        row = await RunRepository(database).get(rid)
        assert row.status == "interrupted" and not row.cancellation_requested
        assert row.lease_owner is None
        assert (await CheckpointRepository(database).latest(rid)).version == before.version
        await second.startup_recovery(auto_resume=True)
        await completed(database, second, rid)
        assert marker.read_text().splitlines() == [rid]
        assert not await event_is(database, rid, "run.cancelled")
    finally:
        await first.shutdown()
        await second.shutdown()


@pytest.mark.asyncio
async def test_periodic_scan_does_not_interrupt_a_locally_running_task(database, tmp_path):
    rid, _ = await seed(database, tmp_path)
    started = asyncio.Event()

    class Blocking(DeterministicRecoveryDriver):
        async def execute(self):
            started.set()
            await asyncio.Event().wait()

    app = service(database, tmp_path, Blocking)
    try:
        app.schedule(rid)
        await asyncio.wait_for(started.wait(), 5)
        await expire(database, rid)
        await app.startup_recovery(auto_resume=True)
        await asyncio.sleep(0.06)
        assert (await RunRepository(database).get(rid)).status == "executing"
        assert not await event_is(database, rid, "run.interrupted")
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_heartbeat_renews_lease_during_a_long_operation(database, tmp_path):
    rid, _ = await seed(database, tmp_path)
    started = asyncio.Event()

    class Blocking(DeterministicRecoveryDriver):
        async def execute(self):
            started.set()
            await asyncio.Event().wait()

    runtime = _runtime(database, tmp_path / "artifacts")
    runtime.workflow_factory = lambda context, events=None: Blocking(context, events)
    runtime.lease_seconds = 0.2
    task = asyncio.create_task(runtime.execute_run(rid))
    try:
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.sleep(0.35)  # Longer than the initial lease.
        row = await RunRepository(database).get(rid)
        assert datetime.fromisoformat(row.lease_expires_at.replace("Z", "+00:00")) > datetime.now(timezone.utc)
        assert not await RunRepository(database).acquire_lease(rid, "competitor", ttl_seconds=90)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
