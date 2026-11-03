"""Read-only recovery of historical critic results from review-scoped events."""
import json


def _object(raw):
    value = json.loads(raw or "{}")
    return value if isinstance(value, dict) else {}


def critic_display(review, events):
    critic = _object(review.critic_json)
    proposal = _object(review.proposal_json)
    checkpoint = _object(review.checkpoint_json)
    source = {"kind": "saved"} if critic else None
    latest = None
    for event in sorted(events, key=lambda item: item.event_id):
        payload = _object(event.payload_json)
        if payload.get("review_id") != review.review_id:
            continue
        if event.event_type in {"learning.review.started", "learning.review.retried"}:
            latest = None
        elif event.event_type == "skill.critic.completed" and payload.get("decision") in {"pass", "revise", "reject"}:
            latest = (event, payload)
    if latest and (not critic or (critic.get("decision") == "revise" and latest[1].get("revision") == 1)):
        event, payload = latest
        fields = {"decision", "scores", "blocking_issues", "revision_instructions", "unsupported_rule_refs", "conflict_refs"}
        critic = {key: value for key, value in payload.items() if key in fields}
        source = {"kind": "event", "event_id": event.event_id,
                  "partial": not fields.issubset(payload)}
    decision = critic.get("decision")
    if review.status == "criticizing":
        state = "running"
    elif review.status == "failed" and checkpoint.get("failed_stage") == "criticizing":
        state = "failed"
    elif proposal.get("decision") == "ignore":
        state = "reviewed_then_ignored" if decision else "skipped"
    elif decision:
        state = {"pass": "passed", "revise": "needs_revision", "reject": "rejected"}.get(decision, "unavailable")
    elif review.status in {"completed", "rejected"}:
        state = "unavailable"
    elif review.status == "failed":
        state = "not_reached"
    else:
        state = "not_started"
    return {"critic": critic, "critic_state": state, "critic_source": source}
