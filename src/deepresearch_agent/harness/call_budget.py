"""Run-scoped, thread-safe admission control. Estimates are not billing usage."""
from contextvars import ContextVar
import json
import math
from threading import Lock

from deepresearch_agent.context.tokens import count_tokens
from .budgets import BudgetExceeded
from .errors import RunCancelled

_policy: ContextVar = ContextVar("harness_call_budget", default=None)


def _request_data(value):
    """Serialize message contents/tool schemas, without Python repr escaping."""
    if hasattr(value, "to_messages"):
        return _request_data(value.to_messages())
    if hasattr(value, "model_dump"):
        return _request_data(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return {str(key): _request_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_request_data(item) for item in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def estimate_model_reservation(value, options, max_output):
    # cl100k is an estimate for compatible models, not their billing tokenizer.
    # A 20% margin and message overhead replace the UTF-8-byte upper bound.
    payload = json.dumps(_request_data([value, options]), ensure_ascii=False)
    return math.ceil(count_tokens(payload) * 1.2) + max(1, max_output) + 128


class CallBudget:
    def __init__(self, limits, usage):
        self.limit = limits.max_llm_tokens
        self.charged = usage.llm_tokens
        self.reserved = 0
        self.tools = usage.tool_calls
        self.searches = usage.tavily_calls
        self.max_tools = limits.max_tool_calls
        self.max_searches = limits.max_tavily_calls
        self._lock = Lock()
        self.closed = False

    def reserve_model(self, value, options, max_output):
        # Reserve in-flight calls together so parallel workers share one limit.
        requested = estimate_model_reservation(value, options, max_output)
        with self._lock:
            if self.closed:
                raise RunCancelled("Run 已停止，不再发起新调用")
            total = self.charged + self.reserved + requested
            if total > self.limit:
                raise BudgetExceeded("llm_token_reservation", total, self.limit)
            self.reserved += requested
        return requested

    def settle_model(self, reservation, measured=None):
        # Unknown usage (including failed transport) keeps its conservative
        # charge, but is never reported as measured Token usage.
        with self._lock:
            self.reserved -= reservation
            self.charged += reservation if measured is None else max(0, measured)

    def admit_tool(self, *, web):
        with self._lock:
            if self.closed:
                raise RunCancelled("Run 已停止，不再发起新调用")
            if self.tools + 1 > self.max_tools:
                raise BudgetExceeded("tool_calls", self.tools + 1, self.max_tools)
            if web and self.searches + 1 > self.max_searches:
                raise BudgetExceeded("tavily_calls", self.searches + 1, self.max_searches)
            self.tools += 1
            self.searches += int(web)

    def tool_usage(self):
        with self._lock:
            return self.tools, self.searches

    def remaining_tokens(self):
        with self._lock:
            return max(0, self.limit - self.charged - self.reserved)

    def close(self):
        with self._lock:
            self.closed = True


def current_call_budget():
    return _policy.get()


def set_call_budget(policy):
    return _policy.set(policy)


def reset_call_budget(token):
    _policy.reset(token)
