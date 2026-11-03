"""Separate research failures from post-run learning failures."""

import json


def research_events(events):
    return [e for e in events if not e.event_type.startswith(("learning.", "skill.")) and e.stage != "learning"]


def failure_details(events):
    result = []
    for event in events:
        if not (event.event_type.endswith((".failed", ".error")) or event.event_type == "run.budget_exhausted"):
            continue
        payload = json.loads(event.payload_json or "{}")
        result.append({"event_id": event.event_id, "event_type": event.event_type,
                       "stage": event.stage, "error_code": payload.get("error_code") or payload.get("error"),
                       "message": str(payload.get("message") or payload.get("error_message") or payload.get("reason") or payload.get("error") or "事件未保存具体原因，请查看关联原始记录。")[:1000],
                       "task_id": payload.get("task_id"), "tool_call_id": payload.get("tool_call_id")})
    return result


def review_error_summary(error):
    if "LengthFinishReasonError" in str(error):
        return "模型输出达到长度上限，未能生成完整的结构化复盘结果。"
    return "复盘处理失败，请展开错误详情查看原因。"
