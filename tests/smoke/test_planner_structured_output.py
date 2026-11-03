import json

import httpx
import pytest

from deepresearch_agent.agents.multi_agent.planner.clarifier import Clarifier
from deepresearch_agent.agents.multi_agent.planner.plan_reviewer import PlanReviewer
from deepresearch_agent.agents.multi_agent.planner.plan_reviewer import PlanValidationResult
from deepresearch_agent.agents.multi_agent.planner.task_decomposer import TaskDecomposer
from deepresearch_agent.models.get_models import PrefixAwareChatOpenAI


class RecordingLLM:
    def __init__(self, model_name="qwen"):
        self.bind_kwargs = None
        self.model_name = model_name

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self

    def invoke(self, _prompt):
        return type("Message", (), {"content": "{}"})()


@pytest.mark.parametrize("planner_type", [Clarifier, TaskDecomposer, PlanReviewer])
@pytest.mark.parametrize(
    "model_name, thinking_options",
    [
        ("qwen", {"enable_thinking": False}),
        (
            "DeepSeek-V4-Flash",
            {
                "thinking": {"type": "disabled"},
                "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
            },
        ),
    ],
)
def test_planner_structured_calls_disable_thinking(planner_type, model_name, thinking_options):
    llm = RecordingLLM(model_name)
    planner = object.__new__(planner_type)
    planner._llm = llm

    assert planner._invoke_llm("prompt") == "{}"
    assert llm.bind_kwargs == {
        "response_format": {"type": "json_object"},
        "extra_body": thinking_options,
    }


def test_plan_validation_normalizes_structured_findings():
    result = PlanValidationResult(
        issues=[{"task_id": "task_003", "reason": "缺少可验证来源"}],
        suggestions=[{"task_id": "task_004", "suggestion": "补充比较维度"}],
    )

    assert result.issues == ["task_id=task_003; reason=缺少可验证来源"]
    assert result.suggestions == ["task_id=task_004; suggestion=补充比较维度"]


def test_deepseek_planner_sends_template_controls_without_raising_token_limit():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "planner-test",
                "object": "chat.completion",
                "created": 0,
                "model": "DeepSeek-V4-Flash",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "{}"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = PrefixAwareChatOpenAI(
            model="DeepSeek-V4-Flash",
            api_key="test-key",
            base_url="https://test.invalid/v1",
            max_tokens=2000,
            http_client=client,
        )
        planner = Clarifier(llm=llm)
        assert planner._invoke_llm("Return a JSON object") == "{}"

    assert len(requests) == 1
    body = requests[0]
    assert body["thinking"] == {"type": "disabled"}
    assert body["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}
    assert body["response_format"] == {"type": "json_object"}
    assert body.get("max_completion_tokens", body.get("max_tokens")) == 2000
