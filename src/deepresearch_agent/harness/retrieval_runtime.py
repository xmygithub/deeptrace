"""Persist read-only searches before analysis; all DB work stays on the Run loop."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
import json

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult
from .call_budget import current_call_budget

_scope: ContextVar = ContextVar("retrieval_runtime", default=None)


def set_retrieval_runtime(value):
    return _scope.set(value)


def reset_retrieval_runtime(token):
    _scope.reset(token)


async def checkpoint_execution():
    scope = _scope.get()
    if scope is not None:
        async with scope.lock:
            await scope.save_step()


class RetrievalRuntime:
    def __init__(self, runtime, context, save_step):
        self.runtime, self.context, self.save_step = runtime, context, save_step
        self.loop = asyncio.get_running_loop()
        self.lock = asyncio.Lock()

    async def on_loop(self, factory):
        if asyncio.get_running_loop() is self.loop:
            return await factory()
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(factory(), self.loop))

    async def load_step(self, task_id):
        async def load():
            return json.loads(json.dumps(self.context.context_snapshot.get("research_steps", {}).get(task_id, {})))
        return await self.on_loop(load)

    async def checkpoint(self, task_id, state):
        async def save():
            async with self.lock:
                self.context.context_snapshot.setdefault("research_steps", {})[task_id] = json.loads(json.dumps(state))
                await self.save_step()
        await self.on_loop(save)

    async def search(self, provider, query, kwargs):
        call = kwargs["call_context"]
        repo = self.runtime.trajectory
        tool_name = "tavily_search" if provider.mode.value == "web" else (kwargs["filters"].strategy or "graphrag_search")
        identity = {"task_id": call.task_id, "tool_call_id": call.tool_call_id,
                    "tool_name": tool_name, "provider": provider.provider_name,
                    "source_mode": provider.mode.value, "query": query}
        stage = self.context.status.value

        async def prepare():
            await self.runtime._assert_not_cancelled(self.context)
            if call.run_id != self.context.run_id or provider.mode != self.context.source_mode:
                raise ValueError("检索请求与冻结的 Run 不一致")
            existing = await repo.get_tool_call(call.tool_call_id) if repo else None
            if existing and existing.status == "completed":
                payload = json.loads(existing.result_json or "{}")
                return [RetrievalResult.from_dict(item) for item in payload.get("retrieval_results", [])]
            policy = current_call_budget()
            if policy is not None:
                policy.admit_tool(web=provider.mode.value == "web")
            if repo:
                await repo.prepare_tool_call(
                    tool_call_id=call.tool_call_id, run_id=call.run_id, task_id=call.task_id,
                    tool_name=tool_name,
                    source_mode=provider.mode.value, args={"query": query, "top_k": kwargs["top_k"]},
                )
            await self.runtime.events.publish(call.run_id, "tool.started", stage=stage, payload={**identity, "status": "running"})
            self.runtime._attach_prefix_usage(self.context)
            await self.runtime.runs.update_usage(call.run_id, {
                "limits": self.context.budget_limits.model_dump(), "usage": self.context.budget_usage.model_dump(),
            })
            return None

        cached = await self.on_loop(prepare)
        if cached is not None:
            return cached
        try:
            results = await provider.search(query, **kwargs)
            if any(item.source_mode != provider.mode.value for item in results):
                raise ValueError("Provider 返回了跨来源证据")
        except (Exception, asyncio.CancelledError) as exc:
            async def fail():
                if repo:
                    await repo.complete_tool_call(call.tool_call_id, error_code=type(exc).__name__)
                await self.runtime.events.publish(call.run_id, "tool.failed", stage=stage, payload={
                    **identity, "status": "failed", "error_code": type(exc).__name__,
                })
            await self.on_loop(fail)
            raise

        async def complete():
            # Assign canonical IDs and persist evidence BEFORE any model sees it.
            evidence = await self.runtime.evidence_ledger.record_results(
                run_id=call.run_id, task_id=call.task_id, tool_call_id=call.tool_call_id,
                provider=provider.provider_name, results=results,
            )
            if repo:
                await repo.complete_tool_call(call.tool_call_id, result={
                    "result_ids": [item.result_id for item in results],
                    "retrieval_results": [item.to_dict() for item in results],
                })
            for item in evidence:
                await self.runtime.events.publish(call.run_id, "evidence.added", stage=stage, payload={
                    "task_id": call.task_id, "evidence_id": item.evidence_id, "source_mode": item.source_mode.value,
                })
            await self.runtime.events.publish(call.run_id, "tool.completed", stage=stage, payload={
                **identity, "status": "completed", "result_count": len(results),
            })
            return results
        return await self.on_loop(complete)


class RuntimeRetrievalProvider:
    def __init__(self, provider):
        self.provider = provider
        self.mode, self.provider_name = provider.mode, provider.provider_name
        self.scope = _scope.get()

    async def search(self, query, **kwargs):
        if self.scope is None:
            return await self.provider.search(query, **kwargs)
        return await self.scope.search(self.provider, query, kwargs)

    async def load_step(self, task_id):
        return await self.scope.load_step(task_id) if self.scope else {}

    async def checkpoint(self, task_id, state):
        if self.scope:
            await self.scope.checkpoint(task_id, state)
