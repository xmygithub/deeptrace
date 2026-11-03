import json
from types import SimpleNamespace

from deepresearch_agent.evolution.critic_display import critic_display


def review(**kwargs):
    return SimpleNamespace(**({"review_id": "r1", "status": "completed", "critic_json": None,
        "proposal_json": '{"decision":"create"}', "checkpoint_json": "{}"} | kwargs))


def event(index, name, payload):
    return SimpleNamespace(event_id=index, event_type=name, payload_json=json.dumps(payload))


def test_historical_pass_is_read_from_matching_event_without_mutation():
    job = review()
    result = critic_display(job, [event(1, "skill.critic.completed", {"review_id": "other", "decision": "reject"}),
        event(2, "skill.critic.completed", {"review_id": "r1", "decision": "pass", "scores": {"safety": .9}, "blocking_issues": []})])
    assert result["critic_state"] == "passed"
    assert result["critic_source"] == {"kind": "event", "event_id": 2, "partial": True}
    assert result["critic"]["decision"] == "pass" and job.critic_json is None


def test_retry_boundary_prevents_reusing_older_pass():
    result = critic_display(review(status="proposing"), [
        event(1, "skill.critic.completed", {"review_id": "r1", "decision": "pass"}),
        event(2, "learning.review.retried", {"review_id": "r1"})])
    assert result["critic"] == {} and result["critic_state"] == "not_started"


def test_skip_and_missing_records_are_distinct():
    assert critic_display(review(proposal_json='{"decision":"ignore"}'), [])["critic_state"] == "skipped"
    assert critic_display(review(), [])["critic_state"] == "unavailable"
    assert critic_display(review(status="criticizing"), [])["critic_state"] == "running"


def test_revised_pass_replaces_stale_revision_in_read_view():
    result = critic_display(review(critic_json='{"decision":"revise"}'), [
        event(1, "skill.critic.completed", {"review_id": "r1", "decision": "pass", "revision": 1})])
    assert result["critic_state"] == "passed"
