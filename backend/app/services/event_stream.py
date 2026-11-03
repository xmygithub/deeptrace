"""Durable replay plus live notification for SSE clients."""

import asyncio
import json
import anyio
from collections.abc import AsyncIterator

from deepresearch_agent.harness.event_bus import EventBus
from deepresearch_agent.persistence.repositories import EventRepository, RunRepository, PlanTaskToolRepository


TERMINAL_STATUSES = {"completed", "failed", "budget_exhausted", "cancelled", "paused"}


class EventStreamService:
    def __init__(self, events: EventRepository, runs: RunRepository, event_bus: EventBus):
        self.events = events
        self.runs = runs
        self.event_bus = event_bus
        self.tools = PlanTaskToolRepository(runs.database)

    @staticmethod
    def serialize(event) -> dict[str, str]:
        payload = json.loads(event.payload_json or "{}")
        payload.update({"event_id": event.event_id, "run_id": event.run_id, "event_type": event.event_type, "stage": event.stage, "created_at": event.created_at})
        return {"id": str(event.event_id), "event": event.event_type, "data": json.dumps(payload, ensure_ascii=False)}

    async def serialize_with_tool(self, event):
        envelope = self.serialize(event)
        if event.event_type not in {"tool.started", "tool.completed", "tool.failed"}:
            return envelope
        payload = json.loads(envelope["data"])
        # Read-time enrichment repairs historical display without rewriting logs.
        call_id = payload.get("tool_call_id")
        if call_id and (not payload.get("tool_name") or not payload.get("query")):
            # SSE disconnect cancellation must not interrupt SQLAlchemy session
            # acquisition/close. Live complete events already contain metadata.
            with anyio.CancelScope(shield=True):
                call = await self.tools.get_tool_call(call_id)
            if call is not None and call.run_id == event.run_id:
                payload["tool_name"] = call.tool_name
                payload.setdefault("source_mode", call.source_mode)
                args = json.loads(call.args_json or "{}")
                if not payload.get("query") and args.get("query"):
                    payload["query"] = args["query"]
        envelope["data"] = json.dumps(payload, ensure_ascii=False)
        return envelope

    async def stream(self, run_id: str, *, after_event_id: int = 0) -> AsyncIterator[dict[str, str]]:
        cursor = max(0, after_event_id)
        for event in await self.events.list_after(run_id, after_event_id=cursor):
            cursor = event.event_id
            yield await self.serialize_with_tool(event)
        run = await self.runs.get(run_id)
        if run is None or run.status in TERMINAL_STATUSES:
            return
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        def on_event(event):
            if event.event_id > cursor:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

        unsubscribe = self.event_bus.subscribe(run_id, on_event)
        try:
            for event in await self.events.list_after(run_id, after_event_id=cursor):
                cursor = event.event_id
                yield await self.serialize_with_tool(event)
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20)
                    if event.event_id <= cursor:
                        continue
                    cursor = event.event_id
                    yield await self.serialize_with_tool(event)
                    refreshed = await self.runs.get(run_id)
                    if refreshed is None or refreshed.status in TERMINAL_STATUSES:
                        return
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": json.dumps({"run_id": run_id, "after_event_id": cursor})}
        finally:
            unsubscribe()
