"""Plan, execute and summarize isolated workflow trials using existing metrics.

No service/model imports in planning or summary mode. --execute spends quota.
Each child uses run_research_pilot.py and its own database/cache/artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ("deep_research", "plan_execute_report")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest():
    files = sorted([*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("backend").rglob("*.py"),
                    ROOT / "scripts/run_research_pilot.py", Path(__file__).resolve()])
    return hashlib.sha256("\n".join(f"{p.relative_to(ROOT).as_posix()}:{digest(p)}" for p in files).encode()).hexdigest()


def load_suite(path):
    suite = read(path)
    cases = suite["cases"]
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)) or len({c["question"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case ID/question")
    for c in cases:
        if (not c["case_id"].replace("-", "").isalnum() or not c["question"].strip()
                or c["split"] not in {"dev", "test"} or not c.get("criteria")):
            raise ValueError("Invalid case")
    return suite


def make_plan(suite, split, smoke, repetitions=1, workflows=WORKFLOWS):
    selected = [c for c in suite["cases"] if c["split"] == split and (not smoke or c.get("smoke"))]
    if not selected:
        raise ValueError("No selected cases")
    trials = []
    for repetition in range(1, repetitions + 1):
        for index, case in enumerate(selected):
            order = workflows if (index + repetition) % 2 else workflows[::-1]
            for workflow in order:
                trials.append({"trial_id": f"{case['case_id']}--{workflow}--r{repetition}",
                               "case_id": case["case_id"], "category": case["category"],
                               "workflow": workflow, "repetition": repetition})
    return trials


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def aggregate(rows):
    evaluated = [r for r in rows if r.get("evaluation")]
    finished = [r for r in rows if r["attempt_state"] != "pending"]
    es = [r["evaluation"] for r in evaluated]
    verified = sum(e["verified_completion"] for e in es)
    tokens = [r["tokens"] for r in evaluated if r["tokens"] is not None]
    delivery = Counter()
    checks = {}
    kinds = set().union(*(set(e["contract_results"]) for e in es)) | {"delivery_coverage"}
    for row in evaluated:
        e = row["evaluation"]
        for kind in kinds:
            value = e["contract_results"].get(kind, "not_run")
            key = "not_run" if value == "not_run" else "error_or_unknown" if value is None else "passed" if value else "failed"
            if kind in row.get("check_errors", []):
                key = "error_or_unknown"
            checks.setdefault(kind, Counter())[key] += 1
        value = e["contract_results"].get("delivery_coverage", "not_run")
        key = "not_run" if value == "not_run" else "error_or_unknown" if value is None else "passed" if value else "failed"
        delivery["error_or_unknown" if "delivery_coverage" in row.get("check_errors", []) else key] += 1
    # Exceptions may be represented as failed checks. Raw details remain available
    # for diagnosis; never claim every False verdict is a content-quality failure.
    latency = {}
    for status in sorted({e["status"] for e in es}):
        values = [e["latency_ms"] / 1000 for e in es if e["status"] == status and e["latency_ms"] is not None]
        latency[status] = {"n": len(values), "p50_seconds": percentile(values, .5), "p95_seconds": percentile(values, .95)}
    return {
        "planned": len(rows), "attempted": len(finished), "pending": len(rows) - len(finished),
        "evaluated": len(es), "infrastructure_errors": sum(r["attempt_state"] in {"infrastructure_error", "incomplete_attempt"} for r in rows),
        "verified_count": verified, "verified_completion_rate": ratio(verified, len(finished)),
        "statuses": dict(Counter(e["status"] for e in es)), "checks": checks,
        "delivery_coverage": {"counts": dict(delivery), "pass_rate": ratio(delivery["passed"], delivery["passed"] + delivery["failed"] + delivery["error_or_unknown"])},
        "citation_validity": ratio(sum(e["valid_citation_count"] for e in es), sum(e["citation_count"] for e in es)),
        "reports_without_citations": sum(e["citation_count"] == 0 for e in es),
        "source_violation_count": sum(e["source_leakage_detected"] for e in es),
        "source_violation_rate_observed": ratio(sum(e["source_leakage_detected"] for e in es), len(es)),
        "tokens": {"measured": len(tokens), "missing": len(es) - len(tokens), "sum": sum(tokens),
                   "mean": mean(tokens) if tokens else None, "median": median(tokens) if tokens else None,
                   "per_verified_delivery": ratio(sum(tokens), verified) if len(tokens) == len(finished) else None},
        "tokens_by_status": {status: {"n": len(values), "mean": mean(values) if values else None}
                             for status in sorted({e["status"] for e in es})
                             for values in [[r["tokens"] for r in evaluated if r["evaluation"]["status"] == status and r["tokens"] is not None]]},
        "latency_by_status": latency,
        "mean_tool_calls": mean(e["tool_call_count"] for e in es) if es else None,
        "failed_tool_calls": sum(e["failed_tool_call_count"] for e in es),
    }


def summarize(output):
    plan = read(output / "plan.json")
    rows = []
    for trial in plan["trials"]:
        folder = output / "trials" / trial["trial_id"]
        outcome = read(folder / "attempt.json") if (folder / "attempt.json").exists() else None
        row = {**trial, "attempt_state": outcome["state"] if outcome else "incomplete_attempt" if folder.exists() else "pending", "tokens": None}
        if (folder / "run/manifest.json").exists():
            manifest = read(folder / "run/manifest.json")
            row["configuration"] = {key: manifest.get(key) for key in ("model", "temperature", "max_output_tokens", "budgets", "search_depth", "search_max_results", "report_type")}
        metrics = folder / "run/system-evaluation.json"
        if metrics.exists():
            data = read(metrics)
            row["evaluation"] = data["evaluation"]
            row["check_errors"] = []
            for kind, detail in data.get("check_details", {}).items():
                observed = detail.get("evidence", {}).get("observed") or {}
                if isinstance(observed, dict):
                    nested = observed.get("details") or {}
                    if observed.get("error") or observed.get("status") in {"error", "not_checked"} or (isinstance(nested, dict) and (nested.get("error") or nested.get("status") in {"error", "not_checked"})):
                        row["check_errors"].append(kind)
            row["deadline_hit"] = data["deadline_hit"]
            usage = data.get("persisted_usage", {})
            usage = usage.get("usage", usage)
            row["tokens"] = usage.get("llm_tokens")
            row["run_path"] = str((folder / "run").resolve())
        rows.append(row)
    groups = {w: aggregate([r for r in rows if r["workflow"] == w]) for w in WORKFLOWS}
    categories = {category: {w: aggregate([r for r in rows if r["workflow"] == w and r["category"] == category]) for w in WORKFLOWS}
                  for category in sorted({r["category"] for r in rows})}
    configurations = {json.dumps(r["configuration"], sort_keys=True) for r in rows if "configuration" in r}
    return {"plan": plan, "by_workflow": groups, "by_category": categories, "trials": rows,
            "configuration_variant_count": len(configurations),
            "configuration_consistent": len(configurations) <= 1,
            "limitations": ["Internal contract metrics, not factual accuracy", "Single trials do not establish stability",
                             "Some check exceptions are stored as False; inspect original details", "Token usage follows provider/runtime records; missing records stay unknown",
                             "Pending trials excluded from rate denominator; incomplete batches must not be reported as final"]}


def render(result):
    lines = ["# DeepTrace 工作流执行评测", "", "复用系统验证和运行记录；不是独立事实正确率。未完成批次为阶段性结果。", "",
             "| 工作流 | 已尝试/计划 | 完成校验通过 | 引用 ID 有效率 | 平均 Token | 基础设施异常 |", "|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"
    for w, g in result["by_workflow"].items():
        lines.append(f"| {w} | {g['attempted']}/{g['planned']} | {g['verified_count']} ({fmt(g['verified_completion_rate'])}) | {fmt(g['citation_validity'])} | {fmt(g['tokens']['mean'])} | {g['infrastructure_errors']} |")
    lines += ["", "## 逐次记录", "", "| 用例 | 工作流 | 执行结果 | Run ID |", "|---|---|---|---|"]
    for r in result["trials"]:
        e = r.get("evaluation", {})
        lines.append(f"| {r['case_id']} | {r['workflow']} | {e.get('status', r['attempt_state'])} | {e.get('run_id', '—')} |")
    lines += ["", "完整分类、各检查项、耗时分位数、成本分母见同目录 summary.json；原始检查详情见各 trial 的 run/api-report.json。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "evals/system/workflow-50-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--smoke", action="store_true", help="Six development questions, twelve trials")
    parser.add_argument("--case-id", action="append", help="Select specific cases within the split")
    parser.add_argument("--workflow", action="append", choices=WORKFLOWS, help="Default: both workflows")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=int, default=900)
    parser.add_argument("--execute", action="store_true", help="Run real LLM/Tavily calls")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize_only:
        if args.execute:
            parser.error("summary mode cannot execute")
    else:
        if args.repetitions < 1 or args.deadline_seconds < 1 or (args.smoke and args.split != "dev"):
            parser.error("Invalid repetition/deadline/smoke split")
        cases = args.cases.resolve()
        suite = load_suite(cases)
        selected_suite = suite
        if args.case_id:
            ids = set(args.case_id)
            if not ids.issubset({c['case_id'] for c in suite['cases']}):
                parser.error("Unknown case ID")
            selected_suite = {**suite, "cases": [c for c in suite['cases'] if c['case_id'] in ids]}
        trials = make_plan(selected_suite, args.split, args.smoke, args.repetitions, tuple(dict.fromkeys(args.workflow or WORKFLOWS)))
        plan = {"suite_id": suite["suite_id"], "cases_sha256": digest(cases), "source_sha256": source_digest(),
                "split": args.split, "deadline_seconds": args.deadline_seconds, "trials": trials}
        output.mkdir(parents=True, exist_ok=True)
        if (output / "plan.json").exists():
            if read(output / "plan.json") != plan:
                parser.error("Plan/code differs; use a new output directory")
        else:
            write_new(output / "plan.json", plan)
            write_new(output / "cases.json", suite)
        if args.execute:
            for trial in trials:
                folder = output / "trials" / trial["trial_id"]
                if (folder / "attempt.json").exists():
                    continue  # Never retry a failed research silently.
                if folder.exists():
                    parser.error(f"Incomplete prior attempt: {folder}; preserve it and use a new output directory")
                folder.mkdir(parents=True)
                print(f"Starting {trial['trial_id']}", flush=True)
                started = datetime.now(timezone.utc).isoformat()
                with (folder / "process.log").open("x", encoding="utf-8") as log:
                    try:
                        process = subprocess.run([sys.executable, str(ROOT / "scripts/run_research_pilot.py"),
                            "--cases", str(output / "cases.json"), "--case-id", trial["case_id"],
                            "--workflow", trial["workflow"], "--output", str(folder / "run"),
                            "--deadline-seconds", str(args.deadline_seconds)], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, timeout=args.deadline_seconds + 90)
                        code = process.returncode
                        state = "recorded" if code == 0 and (folder / "run/system-evaluation.json").exists() else "infrastructure_error"
                    except subprocess.TimeoutExpired:
                        code, state = None, "infrastructure_error"
                write_new(folder / "attempt.json", {"state": state, "returncode": code, "started_at": started,
                          "finished_at": datetime.now(timezone.utc).isoformat()})
                result = summarize(output)
                # Live summaries are derived views. Trial records are immutable.
                (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                (output / "REPORT.md").write_text(render(result), encoding="utf-8")
                print(f"Finished {trial['trial_id']}: {state}", flush=True)
    result = summarize(output)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "REPORT.md").write_text(render(result), encoding="utf-8")
    print(f"Report: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
