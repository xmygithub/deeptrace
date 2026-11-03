from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from deepresearch_agent.harness.budgets import BudgetExceeded, BudgetLimits, BudgetUsage
from deepresearch_agent.harness.errors import RunCancelled
from deepresearch_agent.harness.call_budget import CallBudget, set_call_budget, reset_call_budget, current_call_budget
from deepresearch_agent.models.get_models import PrefixAwareChatOpenAI
from deepresearch_agent.retrieval.base import run_async_from_sync


def test_parallel_admission_reserves_inflight_calls_and_settles_usage():
    policy = CallBudget(BudgetLimits(max_llm_tokens=500), BudgetUsage())
    reserved = policy.reserve_model("hello", {}, 200)
    with pytest.raises(BudgetExceeded):
        policy.reserve_model("hello", {}, 200)
    policy.settle_model(reserved, 50)
    assert policy.charged == 50 and policy.reserved == 0
    policy.settle_model(policy.reserve_model("hello", {}, 100))
    assert policy.charged > 50 and policy.reserved == 0


def test_model_budget_blocks_before_transport_and_never_records_estimates_as_usage(monkeypatch):
    calls = []
    def fake(self, *args, **kwargs):
        calls.append(args)
        return AIMessage(content="ok", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
    monkeypatch.setattr(ChatOpenAI, "invoke", fake)
    model = PrefixAwareChatOpenAI(api_key="test", model="test", max_tokens=100)
    policy = CallBudget(BudgetLimits(max_llm_tokens=500), BudgetUsage())
    token = set_call_budget(policy)
    try:
        assert model.invoke("hello").content == "ok"
        assert policy.charged == 12 and policy.reserved == 0
        with pytest.raises(BudgetExceeded):
            model.invoke("独立的研究资料与证据校验。" * 1000)
        assert len(calls) == 1
    finally:
        reset_call_budget(token)


def test_stopped_run_rejects_new_model_and_tool_calls():
    policy = CallBudget(BudgetLimits(), BudgetUsage())
    policy.close()
    with pytest.raises(RunCancelled):
        policy.reserve_model("hello", {}, 100)
    with pytest.raises(RunCancelled):
        policy.admit_tool(web=True)


@pytest.mark.asyncio
async def test_sync_async_bridge_preserves_run_scoped_policy():
    policy = CallBudget(BudgetLimits(max_tool_calls=1), BudgetUsage())
    token = set_call_budget(policy)
    async def nested():
        return current_call_budget()
    try:
        assert run_async_from_sync(nested) is policy
        with ThreadPoolExecutor(max_workers=2) as pool:
            def admit():
                current_call_budget().admit_tool(web=True)
            pool.submit(copy_context().run, admit).result()
            with pytest.raises(BudgetExceeded):
                pool.submit(copy_context().run, admit).result()
        assert policy.tool_usage() == (1, 1)
    finally:
        reset_call_budget(token)
