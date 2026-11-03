"""Provider-specific options for bounded JSON planning calls."""

from typing import Any


def planner_json_options(llm: Any) -> dict[str, Any]:
    model_name = str(
        getattr(llm, "model_name", None) or getattr(llm, "model", "") or ""
    ).lower()
    # Cover both DeepSeek's API toggle and self-hosted chat-template controls.
    # Both template flags must be false: DeepSeek V4 templates may OR them.
    # Keep the existing option for other OpenAI-compatible model providers.
    thinking = (
        {
            "thinking": {"type": "disabled"},
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        }
        if "deepseek" in model_name
        else {"enable_thinking": False}
    )
    return {
        "response_format": {"type": "json_object"},
        "extra_body": thinking,
    }
