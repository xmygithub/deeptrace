"""Offline fault injection and read-only citation provenance acceptance suite."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_recovery_e2e import _runtime, _wilson_interval
from backend.app.schemas import MessageCreate, RunCreate, SessionCreate
from deepresearch_agent.agents.multi_agent.core.execution_record import ExecutionRecord, ToolCall
from deepresearch_agent.agents.multi_agent.core.retrieval_result import RetrievalMetadata, RetrievalResult
from deepresearch_agent.evaluation.provenance import audit_run
from deepresearch_agent.harness import SourceMode, WorkflowMode
from deepresearch_agent.harness.budgets import BudgetLimits
from deepresearch_agent.harness.retrieval_runtime import RuntimeRetrievalProvider, checkpoint_execution
from deepresearch_agent.persistence import ArtifactStore, Database
from deepresearch_agent.persistence.repositories import RunRepository, SessionRepository
from deepresearch_agent.retrieval.base import ToolCallContext, SearchFilters
from deepresearch_agent.retrieval.web_utils import content_hash

SCENARIOS = {"after_plan": ("executing", "search:t1"),
             "after_search": ("executing", "analyze:t1"),
             "after_task": ("executing", "search:t2"),
             "after_report": ("verifying", "verify"),
             "after_verification": ("verifying", "verify")}

# These are controlled workloads, not alternative production workflow drivers.
WORKLOADS = {
    "single_search": [{"key": "t1", "rounds": 1, "depends_on": []}],
    "iterative_search": [{"key": "t1", "rounds": 3, "depends_on": []}],
    "serial_tasks": [{"key": k, "rounds": 1, "depends_on": []} for k in ("t1", "t2", "t3")],
    "dependent_tasks": [{"key": "t1", "rounds": 1, "depends_on": []},
                        {"key": "t2", "rounds": 1, "depends_on": []},
                        {"key": "t3", "rounds": 1, "depends_on": ["t1", "t2"]}],
}


def steps(workload):
    return [f"{task['key']}-r{index}" for task in WORKLOADS[workload] for index in range(1, task["rounds"] + 1)]


def expected_resume(scenario, workload):
    target = SCENARIOS[scenario][0]
    if scenario == "after_plan":
        first = "search:" + steps(workload)[0]
    elif scenario == "after_search":
        first = "analyze:" + steps(workload)[0]
    elif scenario == "after_task":
        tasks = WORKLOADS[workload]
        first = "search:" + tasks[1]["key"] + "-r1" if len(tasks) > 1 else "report"
    else:
        first = "verify"
    return target, first


def workload_matrix(workloads, scenarios, repetitions):
    return [(w, s, r) for w in workloads for s in scenarios for r in range(1, repetitions + 1)]


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def journal(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()] if Path(path).exists() else []


def action(context, operation, **details):
    with Path(context.config_snapshot["probe_journal"]).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": context.run_id, "phase": os.environ.get("HARNESS_PROBE_PHASE", "unknown"),
                            "operation": operation, "status": context.status.value, **details}) + "\n")
        f.flush()
        os.fsync(f.fileno())


class ProbeProvider:
    mode = SourceMode.WEB
    provider_name = "offline-probe"

    def __init__(self, context):
        self.context = context

    async def search(self, query, **kwargs):
        action(self.context, "search:" + query)
        url = f"https://{query}.example.org/source"
        inherited = kwargs["filters"].extra.get("inherited_evidence_ids", [])
        body = f"固定资料 {query}：研究执行状态应持久化，恢复后复用已保存结果。前置证据：{','.join(inherited)}"
        raw = {"results": [{"url": url, "raw_content": body}]}
        artifact = ArtifactStore(self.context.config_snapshot["probe_artifacts"]).write_text(
            f"{self.context.run_id}/probe/{query}.json", json.dumps(raw, ensure_ascii=False))
        return [RetrievalResult(granularity="DO", evidence=body, source="tavily_search", source_mode="web",
                  metadata=RetrievalMetadata(source_id=url, url=url, domain=f"{query}.example.org", source_type="webpage",
                    content_hash=content_hash(body), extra={"artifact_path": artifact.relative_path, "artifact_id": artifact.artifact_id}), score=.9)]


class ProbeDriver:
    """Controlled workflow; production Runtime, retrieval persistence and recovery.

    The model is replaced, not the state machine or durable tool-result cache.
    """
    def __init__(self, context, events=None):
        self.context = context
        state = context.workflow_state
        self.planned = state.get("planned", False)
        self.workload = context.config_snapshot["probe_workload"]
        self.tasks = WORKLOADS[self.workload]
        self.done = list(state.get("done", []))
        self.analyzed = list(state.get("analyzed", []))
        self.lineage = dict(state.get("lineage", {}))
        self.saved = {key: [RetrievalResult.from_dict(r) for r in rows] for key, rows in state.get("saved", {}).items()}
        self._report = state.get("report")

    def tid(self, key):
        return f"{key}_{self.context.run_id}"

    def cid(self, key):
        return f"call_{key}_{self.context.run_id}"

    async def plan(self, failures=None):
        action(self.context, "plan")
        self.planned = True

    async def execute(self):
        provider = RuntimeRetrievalProvider(ProbeProvider(self.context))
        for task in self.tasks:
            key = task["key"]
            if key in self.done:
                continue
            if not set(task["depends_on"]).issubset(self.done):
                raise AssertionError("Dependent task started before its prerequisites")
            inherited = [r.result_id for step, rows in self.saved.items()
                         if step.split("-r")[0] in task["depends_on"] for r in rows]
            for index in range(1, task["rounds"] + 1):
                step = f"{key}-r{index}"
                if step in self.analyzed:
                    continue
                old_ids = [r.result_id for r in self.saved.get(step, [])]
                # Re-enter the real persisted tool cache after a search crash.
                rows = await provider.search(step, top_k=1, search_depth="basic",
                    filters=SearchFilters(extra={"inherited_evidence_ids": inherited}),
                    call_context=ToolCallContext(run_id=self.context.run_id, source_mode=SourceMode.WEB,
                                                task_id=self.tid(key), tool_call_id=self.cid(step)))
                if old_ids and old_ids != [r.result_id for r in rows]:
                    raise AssertionError("Persisted search results changed during recovery")
                self.saved[step] = rows
                self.context.context_snapshot["probe_point"] = "after_search" if step == steps(self.workload)[0] else "later_search"
                await checkpoint_execution()
                action(self.context, "analyze:" + step, inherited_evidence_ids=inherited)
                self.lineage[step] = inherited
                self.analyzed.append(step)
                self.context.context_snapshot["probe_point"] = "after_round"
                await checkpoint_execution()
            self.done.append(key)
            self.context.context_snapshot["probe_point"] = "after_task" if key == self.tasks[0]["key"] else "later_task"
            await checkpoint_execution()

    async def report(self):
        action(self.context, "report")
        ids = [r.result_id for rows in self.saved.values() for r in rows]
        self._report = "# 研究报告\n\n## 结果\n\n" + "\n\n".join(
            f"已保存的研究结果支持阶段恢复与证据回查。 [{eid}]" for eid in ids)
        if len(ids) == 1:
            self._report += "\n\n## 局限\n\n本实验仅提供单一来源，结论范围受限。"
        return self._report

    def report_consistency(self):
        action(self.context, "verify")
        return True

    async def repair_report(self, failures):
        return False

    def snapshot(self):
        return {"planned": self.planned, "done": self.done, "analyzed": self.analyzed, "lineage": self.lineage,
                "saved": {key: [r.to_dict() for r in rows] for key, rows in self.saved.items()}, "report": self._report}

    def plan_record(self):
        tasks = [{"task_id": self.tid(t["key"]), "task_type": "deep_research" if t["rounds"] > 1 else "web_search",
                  "description": t["key"], "depends_on": [self.tid(k) for k in t["depends_on"]],
                  "status": "completed" if t["key"] in self.done else "pending"} for t in self.tasks]
        return {"plan_id": "plan_" + self.context.run_id, "status": "executing", "tasks": tasks}, tasks

    def execution_records(self):
        records = []
        for key in self.done:
            selected = [step for step in self.analyzed if step.split("-r")[0] == key]
            records.append(ExecutionRecord(task_id=self.tid(key), session_id=self.context.session_id, worker_type="offline-probe",
                tool_calls=[ToolCall(tool_name="tavily_search", tool_call_id=self.cid(step), source_mode="web", args={"query": step},
                    result={"result_ids": [r.result_id for r in self.saved[step]]}) for step in selected],
                evidence=[r for step in selected for r in self.saved[step]]))
        return records

    def evidence_results(self):
        return [(self.tid(step.split("-r")[0]), self.cid(step), "offline-probe", r) for step, rows in self.saved.items() for r in rows]


async def create_run(url, folder, workload):
    db = Database(url)
    try:
        await db.create_schema()
        session = await SessionRepository(db).create(SessionCreate(title="Harness acceptance"))
        _, run, _ = await RunRepository(db).create_for_user_message(
            MessageCreate(session_id=session.session_id, role="user", content="验证研究状态保存与恢复，并引用来源", client_message_id="probe"),
            RunCreate(session_id=session.session_id, trigger_message_id="atomic", source_mode=SourceMode.WEB,
                workflow_mode=WorkflowMode.PLAN_EXECUTE_REPORT,
                config_snapshot={"min_evidence": 1, "probe_workload": workload, "probe_journal": str(folder / "operations.jsonl"),
                                 "probe_artifacts": str(folder / "artifacts")}, budget=BudgetLimits().model_dump()))
        return run.run_id
    finally:
        await db.close()


async def crash_async(url, folder, run_id, scenario):
    db = Database(url)
    runtime = _runtime(db, folder / "artifacts")
    runtime.workflow_factory = ProbeDriver
    original = runtime.checkpoints.save
    async def checkpoint(context, stage):
        result = await original(context, stage)
        point = context.context_snapshot.get("probe_point")
        hit = ((scenario == "after_plan" and stage == "planning") or
               (scenario in {"after_search", "after_task"} and stage == "executing" and point == scenario) or
               (scenario == "after_report" and stage == "reporting") or
               (scenario == "after_verification" and stage == "verifying"))
        if hit:
            save(folder / "crash-observation.json", {"run_id": run_id, "checkpoint_stage": stage,
                 "checkpoint_version": result.version, "saved_results": {k: [r["result_id"] for r in rows] for k, rows in context.workflow_state.get("saved", {}).items()},
                 "done": context.workflow_state.get("done", []), "analyzed": context.workflow_state.get("analyzed", []),
                 "report_hash": hashlib.sha256((context.report or "").encode()).hexdigest()})
            os._exit(86)
        return result
    runtime.checkpoints.save = checkpoint
    try:
        await runtime.execute_run(run_id)
    finally:
        await db.close()


async def resume_async(url, folder):
    from backend.app.services.run_service import RunService
    db = Database(url)
    service = RunService(db, workflow_factory=ProbeDriver, artifact_root=folder / "artifacts", skills_root=folder / "skills", recovery_interval_seconds=.1)
    try:
        await service.startup_recovery(auto_resume=True)
        while True:
            tasks = list(service._tasks.values())
            if tasks:
                await asyncio.gather(*tasks)
            if not await RunRepository(db).list_recoverable() and not service._tasks:
                break
            await asyncio.sleep(.05)
    finally:
        await service.shutdown()
        await db.close()


def worker(phase, url, folder, run_id, scenario):
    os.environ["HARNESS_PROBE_PHASE"] = phase
    os.environ["MEMORY_BACKGROUND_REVIEW_ENABLED"] = "false"
    os.environ["LEARNING_REVIEW_ENABLED"] = "false"
    asyncio.run(crash_async(url, Path(folder), run_id, scenario) if phase == "crash" else resume_async(url, Path(folder)))


def launch(ctx, args, timeout):
    proc = ctx.Process(target=worker, args=args)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
    return proc.exitcode


def inspect_case(folder, run_id, scenario, workload, crash_code, resume_code):
    operations = journal(folder / "operations.jsonl")
    before = [o for o in operations if o["phase"] == "crash"]
    after = [o for o in operations if o["phase"] == "resume"]
    saved_ops = {o["operation"] for o in before if o["operation"].startswith("search:")}
    duplicate_ops = len(saved_ops & {o["operation"] for o in after})
    repeated_attempts = sum(o["operation"] in saved_ops for o in after)
    db_path = folder / "app.db"
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as con:
        status = con.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
        messages = con.execute("SELECT count(*) FROM messages WHERE run_id=? AND role='assistant'", (run_id,)).fetchone()[0]
        events = [(k, json.loads(p)) for k, p in con.execute("SELECT event_type,payload_json FROM run_events WHERE run_id=? ORDER BY event_id", (run_id,))]
        raw = con.execute("SELECT state_json FROM checkpoints WHERE run_id=? ORDER BY version DESC LIMIT 1", (run_id,)).fetchone()
        final = json.loads(raw[0]) if raw else {}
        hashes_valid = all(hashlib.sha256(s.encode()).hexdigest() == h for s, h in con.execute("SELECT state_json,state_hash FROM checkpoints WHERE run_id=?", (run_id,)))
    observation = json.loads((folder / "crash-observation.json").read_text(encoding="utf-8")) if (folder / "crash-observation.json").exists() else {}
    final_saved = final.get("workflow_state", {}).get("saved", {})
    retained = all(ids == [r["result_id"] for r in final_saved.get(key, [])] for key, ids in observation.get("saved_results", {}).items())
    resumed = [p for k, p in events if k == "run.resumed"]
    target, first = expected_resume(scenario, workload)
    stage_ok = len(resumed) == 1 and resumed[0].get("status") == target
    first_ok = bool(after) and after[0]["operation"] == first
    counts = Counter(o["operation"] for o in operations)
    assertions = judge_recovery(crash_code=crash_code, resume_code=resume_code,
        operations=operations, observation=observation, run_id=run_id, stage_ok=stage_ok, first_ok=first_ok,
        retained=retained, hashes_valid=hashes_valid, counts=counts, duplicate_ops=duplicate_ops, status=status, messages=messages,
        expected_steps=steps(workload))
    final_state = final.get("workflow_state", {})
    expected_analyzed = steps(workload)
    assertions["round_order_and_progress"] = final_state.get("analyzed") == expected_analyzed
    assertions["all_tasks_completed_in_order"] = final_state.get("done") == [t["key"] for t in WORKLOADS[workload]]
    assertions["saved_progress_retained"] = set(observation.get("analyzed", [])).issubset(final_state.get("analyzed", [])) and set(observation.get("done", [])).issubset(final_state.get("done", []))
    dependency_ok = True
    for task in WORKLOADS[workload]:
        inherited = [r["result_id"] for step, rows in final_saved.items() if step.split("-r")[0] in task["depends_on"] for r in rows]
        for index in range(1, task["rounds"] + 1):
            step = f"{task['key']}-r{index}"
            actions = [o for o in operations if o["operation"] == "analyze:" + step]
            dependency_ok &= (final_state.get("lineage", {}).get(step) == inherited and len(actions) == 1
                              and actions[0].get("inherited_evidence_ids") == inherited)
    assertions["dependency_evidence_handoff"] = dependency_ok
    trace = audit_run(db_path, folder / "artifacts", run_id)
    return {"scenario": scenario, "workload": workload, "run_id": run_id, "status": status, "assertions": assertions,
        "passed": all(assertions.values()), "saved_operations": len(saved_ops), "repeated_saved_operations": duplicate_ops,
        "repeated_search_attempts": repeated_attempts,
        "expected_stage": target, "expected_first_action": first, "operation_counts": dict(counts), "trace": trace}


def judge_recovery(*, crash_code, resume_code, operations, observation, run_id, stage_ok, first_ok,
                   retained, hashes_valid, counts, duplicate_ops, status, messages, expected_steps=("t1", "t2")):
    return {"hard_crash": crash_code == 86, "resume_process_ok": resume_code == 0,
        "same_run": bool(operations) and all(o["run_id"] == run_id for o in operations) and observation.get("run_id") == run_id,
        "expected_resume_stage": stage_ok, "expected_first_action": first_ok,
        "saved_evidence_ids_retained": retained, "checkpoint_integrity": hashes_valid,
        "plan_once": counts["plan"] == 1,
        "analysis_once_per_task": all(counts["analyze:" + step] == 1 for step in expected_steps),
        "report_once": counts["report"] == 1,
        "search_once_per_task": all(counts["search:" + step] == 1 for step in expected_steps),
        "no_unexpected_search_or_analysis": all(op.split(":", 1)[1] in expected_steps for op in counts if op.startswith(("search:", "analyze:"))),
        "saved_search_not_repeated": duplicate_ops == 0, "completed": status == "completed", "one_final_answer": messages == 1}


def summarize_cases(cases):
    n = len(cases)
    saved = sum(c["saved_operations"] for c in cases)
    repeated = sum(c["repeated_saved_operations"] for c in cases)
    citations = sum(c["trace"]["citation_occurrences"] for c in cases)
    traced = sum(c["trace"]["passed_occurrences"] for c in cases)
    passed = sum(c["passed"] for c in cases)
    return {"attempts": n, "correct_recoveries": passed, "recovery_correctness_rate": passed / n if n else None,
        "recovery_ci95": _wilson_interval(passed, n) if n else None,
        "saved_operations": saved, "repeated_saved_operations": repeated,
        "saved_operation_repeat_rate": repeated / saved if saved else None,
        "citation_occurrences": citations, "traced_occurrences": traced,
        "traceability_rate": traced / citations if citations else None,
        "runs_without_citations": sum(c["trace"]["citation_occurrences"] == 0 for c in cases),
        "unique_final_answer_rate": sum(c["assertions"]["one_final_answer"] for c in cases) / n if n else None}


def run_case(ctx, output, workload, scenario, repetition, timeout):
    folder = output / f"{workload}--{scenario}-{repetition}"
    folder.mkdir()
    url = f"sqlite+aiosqlite:///{(folder / 'app.db').as_posix()}"
    run_id = asyncio.run(create_run(url, folder, workload))
    crash_code = launch(ctx, ("crash", url, str(folder), run_id, scenario), timeout)
    time.sleep(1.25)
    started = time.perf_counter()
    resume_code = launch(ctx, ("resume", url, str(folder), run_id, scenario), timeout)
    case = inspect_case(folder, run_id, scenario, workload, crash_code, resume_code)
    case["repetition"] = repetition
    case["resume_process_elapsed_seconds"] = time.perf_counter() - started
    save(folder / "result.json", case)
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--process-timeout", type=float, default=90)
    parser.add_argument("--jobs", type=int, default=1, help="Independent experiments in parallel (1-4)")
    parser.add_argument("--workload", action="append", choices=tuple(WORKLOADS), help="Default: all four workloads")
    parser.add_argument("--scenario", action="append", choices=tuple(SCENARIOS), help="Default: all five crash points")
    parser.add_argument("--trial-root", type=Path, help="Additionally audit existing workflow trial DBs read-only")
    args = parser.parse_args()
    if args.repetitions < 1 or args.process_timeout <= 0 or not 1 <= args.jobs <= 4:
        parser.error("Positive repetitions and timeout required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workloads = list(dict.fromkeys(args.workload or WORKLOADS))
    scenarios = list(dict.fromkeys(args.scenario or SCENARIOS))
    matrix = workload_matrix(workloads, scenarios, args.repetitions)
    manifest = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "repetitions": args.repetitions,
        "schema_version": 2, "workloads": {w: WORKLOADS[w] for w in workloads}, "jobs": args.jobs,
        "planned_experiments": len(matrix), "scenarios": {s: {w: expected_resume(s, w) for w in workloads} for s in scenarios},
        "driver": "deterministic offline ProbeDriver; real Runtime/RetrievalRuntime/RunService; controlled dependency scheduling",
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT / "src/deepresearch_agent/evaluation/provenance.py", ROOT / "src/deepresearch_agent/harness/runtime.py", ROOT / "src/deepresearch_agent/harness/retrieval_runtime.py"]}}
    save(output / "manifest.json", manifest)
    ctx = multiprocessing.get_context("spawn")
    cases = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_case, ctx, output, w, s, r, args.process_timeout): (w, s, r) for w, s, r in matrix}
        for future in as_completed(futures):
            w, s, repetition = futures[future]
            try:
                case = future.result()
            except Exception as exc:
                # Keep orchestration failures visible and in the experiment denominator.
                error = {"workload": w, "scenario": s, "repetition": repetition, "error": type(exc).__name__}
                errors.append(error)
                save(output / f"{w}--{s}-{repetition}" / "error.json", error)
                print(f"{len(cases)+len(errors)}/{len(matrix)} {w}/{s}: infrastructure_error", flush=True)
            else:
                cases.append(case)
                print(f"{len(cases)+len(errors)}/{len(matrix)} {w}/{s}: {case['passed']}", flush=True)
            (output / "progress.json").write_text(json.dumps({"planned": len(matrix), "completed": len(cases)+len(errors),
                "passed": sum(c["passed"] for c in cases), "errors": errors,
                "cases": [{"workload": c["workload"], "scenario": c["scenario"], "repetition": c["repetition"], "passed": c["passed"]} for c in cases]}, indent=2), encoding="utf-8")
    cases.sort(key=lambda c: (c["workload"], c["scenario"], c["repetition"]))
    audits = []
    if args.trial_root:
        for path in sorted(args.trial_root.resolve().glob("*/run/app.db")):
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as con:
                ids = [row[0] for row in con.execute("SELECT run_id FROM runs")]
            for run_id in ids:
                try:
                    audits.append({"trial": path.parent.parent.name, **audit_run(path, path.parent / "artifacts", run_id)})
                except (OSError, ValueError, sqlite3.Error) as exc:
                    audits.append({"trial": path.parent.parent.name, "run_id": run_id, "audit_error": type(exc).__name__})
    actual = [a for a in audits if "audit_error" not in a]
    citation_count = sum(a["citation_occurrences"] for a in actual)
    trace_count = sum(a["passed_occurrences"] for a in actual)
    real_summary = {"runs": len(audits), "audit_errors": len(audits)-len(actual),
        "runs_without_citations": sum(a["citation_occurrences"] == 0 for a in actual),
        "report_sources": dict(Counter(a["report_source"] for a in actual)),
        "citation_occurrences": citation_count, "passed_occurrences": trace_count,
        "unique_citations": sum(a["unique_citations"] for a in actual),
        "traceability_rate": trace_count / citation_count if citation_count else None,
        "failure_reasons": dict(Counter(reason for a in actual for check in a["checks"] for reason in check["reasons"]))}
    summary = summarize_cases(cases)
    summary.update(attempts=len(matrix), infrastructure_errors=len(errors),
                   recovery_correctness_rate=summary["correct_recoveries"] / len(matrix),
                   recovery_ci95=_wilson_interval(summary["correct_recoveries"], len(matrix)))
    if errors:
        summary["observed_saved_operation_repeat_rate"] = summary["saved_operation_repeat_rate"]
        summary["saved_operation_repeat_rate"] = None
        summary["unique_final_answer_rate"] = None
    by_combination = []
    for w in workloads:
        for s in scenarios:
            selected = [c for c in cases if c["workload"] == w and c["scenario"] == s]
            by_combination.append({"workload": w, "scenario": s, "attempts": args.repetitions,
                "correct": sum(c["passed"] for c in selected),
                "infrastructure_errors": sum(e["workload"] == w and e["scenario"] == s for e in errors),
                "expected_stage": expected_resume(s, w)[0], "expected_first_action": expected_resume(s, w)[1]})
    payload = {"manifest": manifest, "summary": summary, "by_combination": by_combination,
               "infrastructure_errors": errors, "cases": cases,
               "existing_run_summary": real_summary, "existing_run_audits": audits}
    save(output / "results.json", payload)
    repeat_text = "未知（存在基础设施异常）" if summary["saved_operation_repeat_rate"] is None else f"{summary['saved_operation_repeat_rate']:.2%}"
    lines = ["# Harness 恢复与追溯专项验收", "", "离线确定性 Driver，使用真实状态机、工具持久化缓存和独立进程恢复；不代表真实模型研究质量。", "",
        "| 指标 | 结果 |", "|---|---:|",
        f"| 恢复正确率 | {summary['recovery_correctness_rate']:.2%} ({summary['correct_recoveries']}/{summary['attempts']}) |",
        f"| 应复用的已保存操作 | {summary['saved_operations']} |",
        f"| 重复执行的已保存操作 | {summary['repeated_saved_operations']}（重复率 {repeat_text}） |",
        f"| 基础设施异常 | {len(errors)} |",
        f"| 引用追溯通过 | {summary['traced_occurrences']}/{summary['citation_occurrences']} |", "",
        "重复率分母为中断前已保存的不同搜索操作数；实际 Provider 调用由 fsync 日志计数。轨迹断言包括恢复阶段、首动作、证据ID保留、操作次数及回答唯一性。", "",
        "## 故障位置与执行负载", "", "| 负载 | 场景 | 恢复正确 | 恢复阶段 | 首动作 |", "|---|---|---:|---|---|"]
    for cell in by_combination:
        lines.append(f"| {cell['workload']} | {cell['scenario']} | {cell['correct']}/{cell['attempts']} | {cell['expected_stage']} | {cell['expected_first_action']} |")
    lines += ["", "## 既有真实运行追溯（只读）", "",
        f"审计 {len(audits)} 次运行；其中 {real_summary['runs_without_citations']} 次无引用，不计为满分；审计异常 {real_summary['audit_errors']} 次。",
        f"已检查引用出现次数 {trace_count}/{citation_count} 通过，对应不同证据 ID 合计 {real_summary['unique_citations']} 条。交付回答和失败草稿均保留，详细分组见 JSON。", "",
        "| 试次 | 状态 | 报告来源 | 通过/引用次数 |", "|---|---|---|---:|"]
    for a in audits:
        lines.append(f"| {a['trial']} | {a.get('status', 'audit_error')} | {a.get('report_source', '—')} | {a.get('passed_occurrences', '—')}/{a.get('citation_occurrences', '—')} |")
    lines += ["", "## 限制", "", "- 五类受控故障位置、四类可选负载，每种组合重复运行；重复不是不同真实研究题。",
        "- 执行负载由测试 Driver 定义；依赖分支按确定顺序执行。未证明生产 WorkerCoordinator 的并发依赖调度和所有 DR/PER 路径。",
        "- 未覆盖外部成功但尚未落库、磁盘损坏、多执行者竞争或代码版本迁移。",
        "- 校验来源材料身份与完整性，不检验材料是否在语义上支持陈述。无报告或无引用不计为100%。",
        "- 真实运行与合成场景分别报告；缺关联或材料无法核验时保留失败，不修改原始数据。",
        "- 恢复耗时包含恢复子进程启动与续跑；测试租约1秒，不推导生产等待时间。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Report: {output / 'REPORT.md'}", flush=True)
    if errors or any(not c["passed"] for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
