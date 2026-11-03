"""Export/review research reports offline; no model calls and no DB writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepresearch_agent.evaluation.research import (  # noqa: E402
    load_cases, make_bundle, read_json, read_run, render_summary,
    summarize_packages, write_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export an existing Run without rerunning it")
    export.add_argument("--database", type=Path, default=PROJECT_ROOT / "data/app.db")
    export.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "data/artifacts")
    export.add_argument("--run-id", required=True)
    imported = commands.add_parser("import", help="Import a separately generated baseline/report snapshot")
    imported.add_argument("--input", type=Path, required=True, help="JSON snapshot; see evals/research/README.md")
    for command in (export, imported):
        command.add_argument("--cases", type=Path, default=PROJECT_ROOT / "evals/research/cases.json")
        command.add_argument("--case-id", required=True)
        command.add_argument("--arm", choices=("agent", "baseline"), default="agent")
        command.add_argument("--label", default="")
        command.add_argument("--max-claims", type=int, default=12)
        command.add_argument("--allow-question-mismatch", action="store_true", help="Exploratory review only; excluded from quality comparison")
        command.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    summary = commands.add_parser("summarize", help="Validate review labels and produce JSON/Markdown results")
    summary.add_argument("--package", type=Path, action="append", required=True)
    summary.add_argument("--review-file", default="review.json", help="Review basename within each package, e.g. review.assistant.json")
    summary.add_argument("--output", type=Path, required=True, help="New directory")
    args = parser.parse_args()
    try:
        if args.command == "summarize":
            result = summarize_packages(args.package, review_filename=args.review_file)
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (args.output / "summary.md").write_text(render_summary(result), encoding="utf-8")
            print(f"Summary: {args.output / 'summary.md'}; reviewed reports: {result['quality_graded_runs']}/{result['run_count']}")
            return
        cases = load_cases(args.cases)
        if args.case_id not in cases:
            raise ValueError(f"Unknown case: {args.case_id}")
        if args.command == "export":
            run = read_run(args.database, args.artifact_root, args.run_id)
        else:
            run = read_json(args.input)
            # Missing measurements remain unknown; import never invents usage or contract checks.
            for key in ("verified_completion", "llm_tokens", "latency_seconds"):
                run["metrics"].setdefault(key, None)
            run["provenance"] = "external_import; generation and measurements supplied by evaluator"
        bundle, review = make_bundle(cases[args.case_id], run, arm=args.arm, label=args.label, max_claims=args.max_claims)
        if not bundle["question_matches_case"] and not args.allow_question_mismatch:
            raise ValueError("Run question does not match case; use the matching case or explicitly allow exploratory review")
        write_package(args.output, bundle, review)
        print(f"Package: {args.output}; report: {bool(run['report'])}; claims pending: {len(review['claims'])}")
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
