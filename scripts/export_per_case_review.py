"""Read-only export of PER failure diagnostics and retrospective review packets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepresearch_agent.evaluation.research import make_bundle, read_run, write_package


def export(trials, output):
    output.mkdir(parents=True, exist_ok=False)
    diagnostics = []
    selected = {"comparison-01--plan_execute_report--r1", "comparison-02--plan_execute_report--r1",
                "constraints-01--plan_execute_report--r1", "synthesis-01--plan_execute_report--r1",
                "comparison-01--deep_research--r1", "mechanism-01--plan_execute_report--r1"}
    for folder in sorted(trials.iterdir()):
        if folder.name not in selected or not (folder / "run/system-evaluation.json").exists():
            continue
        runroot = folder / "run"
        metric = json.loads((runroot / "system-evaluation.json").read_text(encoding="utf-8"))
        run_id = metric["evaluation"]["run_id"]
        snapshot = read_run(runroot / "app.db", runroot / "artifacts", run_id)
        origin = "final_answer" if snapshot["report"] else "no_report"
        with sqlite3.connect((runroot / "app.db").resolve().as_uri() + "?mode=ro", uri=True) as con:
            artifact = con.execute("SELECT relative_path,sha256 FROM artifacts WHERE run_id=? AND mime_type LIKE 'text/markdown%' ORDER BY created_at DESC LIMIT 1", (run_id,)).fetchone()
            if not snapshot["report"] and artifact:
                root = (runroot / "artifacts").resolve()
                path = (root / artifact[0]).resolve()
                path.relative_to(root / run_id)
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != artifact[1]:
                    raise ValueError("Draft checksum mismatch")
                snapshot["report"], origin = raw.decode("utf-8"), "saved_draft"
            cp = con.execute("SELECT state_json FROM checkpoints WHERE run_id=? ORDER BY version DESC LIMIT 1", (run_id,)).fetchone()
            state = json.loads(cp[0]) if cp else {}
            result = state.get("workflow_state", {}).get("report_result") or {}
            outline = result.get("outline") or {}
            events = [{"type": k, "data": json.loads(p)} for k, p in con.execute("SELECT event_type,payload_json FROM run_events WHERE run_id=? ORDER BY event_id", (run_id,))
                      if k in {"verification.completed", "run.retrying", "report.repair_failed", "task.failed"}]
        snapshot["report_origin"] = origin
        manifest = json.loads((runroot / "manifest.json").read_text(encoding="utf-8"))
        case = manifest["case"]
        bundle, review = make_bundle(case, snapshot, label=folder.name, max_claims=5)
        review.update(reviewer_kind="assistant", reviewer="Codex retrospective source review", reviewed_at=datetime.now(timezone.utc).isoformat())
        package = output / folder.name
        write_package(package, bundle, review)
        diagnostics.append({"case": folder.name, "run_id": run_id, "status": snapshot["status"], "report_origin": origin,
            "report_chars": len(snapshot["report"]), "requested_report_type": state.get("config_snapshot", {}).get("report_type"),
            "generated_report_type": outline.get("report_type"), "sections": len(outline.get("sections", [])),
            "uncited_outline_abstract": bool(outline.get("abstract")) and "[ev_" not in outline.get("abstract", ""),
            "error_code": snapshot["error_code"], "metrics": snapshot["metrics"], "checks": metric.get("check_details", {}),
            "events": events, "raw_section_response_available": False,
            "raw_section_response_note": "原运行未结构化保存失败章节的两次响应，无法由最终报错区分正文为空与编号缺失。" if snapshot["error_code"] == "CitationBindingError" else "not_applicable"})
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(diagnostics)} retrospective packets to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.trials.resolve(), args.output.resolve())
