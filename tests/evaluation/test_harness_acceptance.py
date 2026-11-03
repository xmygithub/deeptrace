"""Recovery oracle negative controls independent of any saved experiment output."""
from collections import Counter
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("acceptance_eval", ROOT / "scripts/evaluate_harness_acceptance.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def parameters():
    return dict(crash_code=86, resume_code=0, operations=[{"run_id": "r"}], observation={"run_id": "r"},
                run_id="r", stage_ok=True, first_ok=True, retained=True, hashes_valid=True,
                counts=Counter({"plan": 1, "analyze:t1": 1, "analyze:t2": 1, "report": 1, "search:t1": 1, "search:t2": 1}),
                duplicate_ops=0, status="completed", messages=1)


def test_expected_recovery_passes():
    assert all(module.judge_recovery(**parameters()).values())


@pytest.mark.parametrize("change", [
    {"crash_code": 0}, {"resume_code": 1}, {"stage_ok": False}, {"first_ok": False},
    {"retained": False}, {"hashes_valid": False}, {"duplicate_ops": 1},
    {"status": "failed"}, {"messages": 2}, {"operations": [{"run_id": "other"}]},
    {"observation": {"run_id": "other"}},
])
def test_wrong_recovery_does_not_pass(change):
    args = parameters()
    args.update(change)
    assert not all(module.judge_recovery(**args).values())


def test_repeated_analysis_is_detected():
    args = parameters()
    args["counts"]["analyze:t1"] = 2
    assert not module.judge_recovery(**args)["analysis_once_per_task"]


def test_no_saved_operation_has_no_repeat_rate():
    case = {"passed": True, "saved_operations": 0, "repeated_saved_operations": 0,
            "trace": {"citation_occurrences": 0, "passed_occurrences": 0}, "assertions": {"one_final_answer": True}}
    result = module.summarize_cases([case])
    assert result["saved_operation_repeat_rate"] is None
    assert result["traceability_rate"] is None


def test_sixty_trial_matrix_has_twenty_distinct_combinations():
    matrix = module.workload_matrix(module.WORKLOADS, module.SCENARIOS, 3)
    assert len(matrix) == len(set(matrix)) == 60
    counts = Counter((w, s) for w, s, _ in matrix)
    assert len(counts) == 20
    assert set(counts.values()) == {3}


def test_workloads_have_different_operations_and_dependency_graph():
    assert module.steps("single_search") == ["t1-r1"]
    assert module.steps("iterative_search") == ["t1-r1", "t1-r2", "t1-r3"]
    assert module.steps("serial_tasks") == ["t1-r1", "t2-r1", "t3-r1"]
    assert module.WORKLOADS["dependent_tasks"][2]["depends_on"] == ["t1", "t2"]


@pytest.mark.parametrize("workload", module.WORKLOADS)
def test_first_resumed_action_is_specific_to_workload(workload):
    assert module.expected_resume("after_plan", workload) == ("executing", "search:t1-r1")
    assert module.expected_resume("after_search", workload) == ("executing", "analyze:t1-r1")
    expected = "report" if len(module.WORKLOADS[workload]) == 1 else "search:t2-r1"
    assert module.expected_resume("after_task", workload) == ("executing", expected)
    assert module.expected_resume("after_report", workload) == ("verifying", "verify")


@pytest.mark.parametrize("workload", module.WORKLOADS)
def test_missing_later_round_or_task_does_not_pass(workload):
    args = parameters()
    expected = module.steps(workload)
    args["expected_steps"] = expected
    args["counts"] = Counter({"plan": 1, "report": 1})
    for step in expected:
        args["counts"]["analyze:" + step] = 1
        args["counts"]["search:" + step] = 1
    assert all(module.judge_recovery(**args).values())
    args["counts"]["search:" + expected[-1]] = 0
    assert not module.judge_recovery(**args)["search_once_per_task"]
