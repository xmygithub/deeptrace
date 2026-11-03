"""One real FastAPI -> Harness research trial in an isolated directory.

Explicit invocation spends configured LLM/Tavily quota. Existing output directories
are rejected. No production DB, .env, business code, or Skill is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "evals/research/pilot-v2.json")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--workflow", choices=("deep_research", "plan_execute_report"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.deadline_seconds < 1:
        parser.error("deadline must be positive")
    cases_raw = args.cases.resolve().read_bytes()
    selected = next(c for c in json.loads(cases_raw)["cases"] if c["case_id"] == args.case_id)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Set before importing settings. Router and ArtifactStore must agree on the
    # root, not just the app's explicit artifact_root argument.
    overrides = {
        "APP_DATABASE_URL": f"sqlite+aiosqlite:///{(output / 'app.db').as_posix()}",
        "ARTIFACT_ROOT": str(output / "artifacts"),
        "SKILLS_ROOT": str(output / "skills"),
        "CACHE_DIR": str(output / "cache"),
        "AUTO_RESUME_RUNS": "false", "LEARNING_REVIEW_ENABLED": "false",
        "MEMORY_BACKGROUND_REVIEW_ENABLED": "false",
    }
    os.environ.update(overrides)
    # BaseAgent also has relative ./cache and ./output defaults.
    os.chdir(output)
    from fastapi.testclient import TestClient
    from backend.app.main import create_app
    from deepresearch_agent.config import settings
    from deepresearch_agent.evaluation.research import make_bundle, read_run, write_package

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "case": selected,
        "cases_sha256": hashlib.sha256(cases_raw).hexdigest(),
        "workflow": args.workflow, "source_mode": "web", "report_type": "brief",
        "model": settings.OPENAI_LLM_MODEL, "temperature": settings.LLM_TEMPERATURE,
        "max_output_tokens": settings.LLM_MAX_TOKENS, "budgets": settings.HARNESS_BUDGETS,
        "search_depth": settings.TAVILY_SEARCH_DEPTH, "search_max_results": settings.TAVILY_MAX_RESULTS,
        "deadline_seconds": args.deadline_seconds, "overrides": overrides,
        "scope": "fresh session and isolated DB/cache; no Memory extraction or Skill learning",
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "backend/app/services/run_service.py", "src/deepresearch_agent/harness/workflow.py",
            "src/deepresearch_agent/harness/report_safety.py",
            "src/deepresearch_agent/harness/verifiers/deterministic.py",
            "src/deepresearch_agent/config/prompts/reporter_prompts.py",
            "src/deepresearch_agent/agents/multi_agent/reporter/section_writer.py",
            "src/deepresearch_agent/agents/multi_agent/reporter/base_reporter.py",
            "src/deepresearch_agent/agents/multi_agent/reporter/outline_builder.py",
            "src/deepresearch_agent/agents/multi_agent/reporter/consistency_checker.py",
            "src/deepresearch_agent/harness/report_repair.py",
            "src/deepresearch_agent/agents/multi_agent/reporter/evidence_cards.py",
            "src/deepresearch_agent/agents/multi_agent/reporter/citation_renderer.py",
            "src/deepresearch_agent/retrieval/research_loop.py",
            "src/deepresearch_agent/harness/retrieval_runtime.py",
            "src/deepresearch_agent/harness/call_budget.py",
            "src/deepresearch_agent/harness/runtime.py",
            "src/deepresearch_agent/context/resolver.py",
            "src/deepresearch_agent/models/get_models.py",
            "src/deepresearch_agent/agents/multi_agent/planner/task_decomposer.py",
        )},
    }
    save(output / "manifest.json", manifest)
    app = create_app(auto_resume=False)
    start = time.monotonic()
    terminal = {"completed", "failed", "cancelled", "budget_exhausted", "needs_user_input", "paused", "interrupted"}
    deadline_hit = False
    scheduler_stopped = False
    last_stage = None
    observations = []
    with TestClient(app) as client:
        session = client.post("/api/v1/sessions", json={"title": f"[Research Pilot] {args.case_id}"})
        session.raise_for_status()
        session_id = session.json()["session_id"]
        accepted = client.post(f"/api/v1/sessions/{session_id}/messages", json={
            "client_message_id": "pilot-once", "content": selected["question"],
            "source_mode": "web", "workflow_mode": args.workflow, "report_type": "brief",
        })
        accepted.raise_for_status()
        run_id = accepted.json()["run_id"]
        save(output / "accepted.json", {"session_id": session_id, **accepted.json()})
        print(json.dumps({"run_id": run_id, "case_id": args.case_id, "workflow": args.workflow}), flush=True)
        while True:
            response = client.get(f"/api/v1/runs/{run_id}")
            response.raise_for_status()
            run = response.json()
            stage = (run["status"], run["current_stage"])
            if stage != last_stage:
                observation = {"elapsed_seconds": round(time.monotonic() - start, 3),
                               "status": stage[0], "stage": stage[1], "usage": run["usage"]}
                observations.append(observation)
                print(json.dumps(observation, ensure_ascii=True), flush=True)
                last_stage = stage
            if run["status"] in terminal:
                break
            # A background exception can leave a nonterminal DB status behind.
            # Detect it in this in-process HTTP harness rather than waiting for
            # a timeout or rewriting its status to a fabricated application failure.
            active = client.portal.call(lambda: run_id in app.state.run_service._tasks)
            if not active:
                scheduler_stopped = True
                break
            if time.monotonic() - start > args.deadline_seconds:
                deadline_hit = True
                client.post(f"/api/v1/runs/{run_id}/cancel").raise_for_status()
                break
            time.sleep(1)
        for name in ("report", "evidence", "checkpoints"):
            response = client.get(f"/api/v1/runs/{run_id}/{name}")
            if response.is_success:
                save(output / f"api-{name}.json", response.json())
        save(output / "observations.json", observations)
        save(output / "execution.json", {
            "run_id": run_id, "run_snapshot": run, "deadline_hit": deadline_hit,
            "scheduler_stopped_before_terminal": scheduler_stopped,
            "observed_elapsed_seconds": time.monotonic() - start,
            "scope": "app completion status is distinct from semantic report quality",
        })
    snapshot = read_run(output / "app.db", output / "artifacts", run_id)
    # Export after lifespan shutdown so cancellation/interruption is committed.
    # Reuse the existing system evaluator; retain raw usage separately so a
    # missing measurement is not silently turned into the evaluator's zero.
    import asyncio
    import sqlite3
    from deepresearch_agent.evaluation import EvaluationService
    from deepresearch_agent.persistence import Database

    async def export_evaluation():
        database = Database(overrides["APP_DATABASE_URL"])
        try:
            return (await EvaluationService(database).evaluate_run(run_id)).model_dump(mode="json")
        finally:
            await database.close()

    with sqlite3.connect(f"file:{(output / 'app.db').as_posix()}?mode=ro", uri=True) as connection:
        raw = connection.execute("SELECT usage_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        check_rows = connection.execute("SELECT kind, passed, evidence_json FROM contract_checks WHERE run_id = ?", (run_id,)).fetchall()
    save(output / "system-evaluation.json", {
        "evaluation": asyncio.run(export_evaluation()),
        "persisted_usage": json.loads(raw[0] or "{}") if raw else {},
        "check_details": {kind: {"passed": None if passed is None else bool(passed),
                                  "evidence": json.loads(evidence or "{}")}
                          for kind, passed, evidence in check_rows},
        "deadline_hit": deadline_hit,
        "scheduler_stopped_before_terminal": scheduler_stopped,
    })
    bundle, review = make_bundle(selected, snapshot, label=args.workflow)
    write_package(output / "package", bundle, review)
    print(json.dumps({"final_status": snapshot["status"], "metrics": snapshot["metrics"],
                      "report_chars": len(snapshot["report"]), "deadline_hit": deadline_hit}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
