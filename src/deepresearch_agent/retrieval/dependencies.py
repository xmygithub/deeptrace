"""Read bounded evidence from completed reachable ancestors, preserving provenance."""
from collections import deque
from itertools import zip_longest

from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalResult


def dependency_evidence(state, task, *, limit=12):
    context = state.execution_context
    if context is None:
        return []
    completed = set(context.completed_task_ids)
    plan = state.plan
    graph = plan.get("task_graph", {}) if isinstance(plan, dict) else getattr(plan, "task_graph", None)
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else getattr(graph, "nodes", [])
    dependencies = {
        (node.get("task_id") if isinstance(node, dict) else node.task_id):
        (node.get("depends_on", []) if isinstance(node, dict) else node.depends_on)
        for node in nodes
    }
    # Walk only the task's declared ancestry, never all completed Run tasks.
    # Round-robin below gives each upstream task a chance under the same cap.
    pending, seen_tasks, groups = deque(task.depends_on), {task.task_id}, []
    while pending and len(seen_tasks) <= 64:
        task_id = pending.popleft()
        if task_id in seen_tasks:
            continue
        seen_tasks.add(task_id)
        if task_id not in completed:
            continue
        pending.extend(dependencies.get(task_id, []))
        # Cache is persisted with the execution state, including parallel worker
        # merges. It contains evidence rather than ungrounded answer summaries.
        groups.append(context.retrieval_cache.get(task_id, []))
    selected, seen = [], set()
    for row in zip_longest(*groups):
        for raw in row:
            if raw is None:
                continue
            result = raw.model_copy(deep=True) if isinstance(raw, RetrievalResult) else RetrievalResult.from_dict(raw)
            if (result.source_mode != state.source_mode or
                result.metadata.extra.get("run_id") != state.run_id or
                not result.result_id.startswith("ev_") or
                not result.metadata.extra.get("tool_call_id")):
                continue
            if result.result_id in seen:
                continue
            selected.append(result)
            seen.add(result.result_id)
            if len(selected) >= limit:
                return selected
    return selected
