"""`compiled-merge` on arms whose records leave a block out: a treatment run with no Commit record, a Commit record without its
openings, coverage, value correspondence or compiled source, a process whose versions identify no generated kernels or list none
by name, and a round's verdict without run roots.  A block that is absent, null, empty or zero reads alike: the merge tests
each for its truth.  Torch-free: the arms' records are written here; the pairing and its refusals are `test_compiled_merge.py`."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.pipeline.test_compiled_merge import COMPILATION, KERNELS
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import compiled_merge as cm

ABSENT = "<absent>"
#: what stands in a block's place where a record has none: the block left out, null, and each empty or zero value
NOTHING: list[Any] = [ABSENT, None, {}, [], "", 0, False]
NOT_IDENTIFIED = "served path: generated kernels not identified in both arms (inductor_kernels.json)"
#: what the summary lists for each treatment run, and what it lists for the run of `process("served", "aa")`
PER_RUN: dict[str, Any] = {"openings_verified": True, "coverage": True, "value_correspondence": True, "compiled_source": {"pieces": 3},
                           "run_roots": "aa"}
#: a part of a Commit record, the field the summary takes from it, and the summary's list of that field
PARTS = [("openings", "openings_verified"), ("coverage", "coverage"), ("value_correspondence", "value_correspondence"),
         ("compiled_source", "compiled_source")]


def process(taps: str, root: str | None = None) -> dict[str, Any]:
    """One process's records as its output directory holds them: a control's, or a treatment's whose one run committed under
    `root`."""
    run: dict[str, Any] = {"wall_s": 1.0, "step_wall_decode_mean_s": 0.5, "step_wall_prefill_s": 0.25, "tokens_digest": "t" * 16}
    if root is not None:
        run["commit"] = {"commit_pass": True, "run_root": root, "openings": {"verified": True}, "coverage": {"ok": True},
                         "value_correspondence": {"result": True}, "compiled_source": {"stats": {"pieces": 3}}}
    return {"versions": {"compilation": {**COMPILATION, "compiled_taps": taps, "generated_kernels": copy.deepcopy(KERNELS)}},
            "runs": [run],
            "verdict": {"run_roots": [] if root is None else [root], "runs_instrumented": int(root is not None), "runs_failed": 0}}


def stand_in(block: dict[str, Any], key: str, value: Any) -> None:
    """Leaves `key` out of `block` (ABSENT), or puts `value` in its place."""
    if value is ABSENT:
        del block[key]
    else:
        block[key] = value


def merge(tmp_path: Path, control: list[dict[str, Any]], treatment: list[dict[str, Any]]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """The merge of the two arms' processes, one directory each: its exit code, its summary and its verdict."""
    argv: list[str] = []
    for arm, processes in (("control", control), ("treatment", treatment)):
        for i, records in enumerate(processes):
            d = tmp_path / f"{arm}{i}"
            d.mkdir()
            (d / "versions.json").write_text(json.dumps(records["versions"]))
            (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records["runs"]))
            (d / "verdict.json").write_text(json.dumps(records["verdict"]))
            argv += [f"--{arm}", str(d)]
    out = tmp_path / "merged"
    with pytest.raises(SystemExit) as e:
        cm.main(cli.parse("compiled-merge", [*argv, "--out", str(out)], {}))
    return e.value.code, json.loads((out / "summary.json").read_text()), json.loads((out / "verdict.json").read_text())


def test_a_run_of_each_arm_merges_and_the_summary_lists_the_treatment_runs_commit(tmp_path: Path) -> None:
    rc, summary, verdict = merge(tmp_path, [process("off")], [process("served", "aa")])
    assert rc == 0 and summary["problems"] == [] and summary["commit_pass"] is True and summary["commit_fail_reasons"] == []
    assert {k: summary[k] for k in PER_RUN} == {k: [v] for k, v in PER_RUN.items()}
    assert (summary["per_pair"][0]["commit_pass"], summary["per_pair"][0]["run_root"]) == (True, "aa")
    assert verdict["run_roots"] == ["aa"] and verdict["first_fail_reason"] is None


@pytest.mark.parametrize("commit", NOTHING)
def test_a_treatment_run_without_a_commit_record_fails_the_pair_and_gives_no_reason(tmp_path: Path, commit: Any) -> None:
    treatment = process("served", "aa")
    stand_in(treatment["runs"][0], "commit", commit)
    rc, summary, verdict = merge(tmp_path, [process("off")], [treatment])
    assert rc == 3 and summary["problems"] == [] and summary["commit_pass"] is False and verdict["commit_pass"] is False
    assert summary["commit_fail_reasons"] == [None] and verdict["first_fail_reason"] is None
    assert (summary["per_pair"][0]["commit_pass"], summary["per_pair"][0]["run_root"]) == (None, None)
    assert {k: summary[k] for k in PER_RUN} == dict.fromkeys(PER_RUN, [None])


def test_a_treatment_run_whose_commit_failed_fails_the_pair_by_its_reason(tmp_path: Path) -> None:
    treatment = process("served", "aa")
    treatment["runs"][0]["commit"] |= {"commit_pass": False, "commit_fail_reason": "openings"}
    rc, summary, verdict = merge(tmp_path, [process("off")], [treatment])
    assert rc == 3 and summary["problems"] == [] and summary["commit_pass"] is False
    assert summary["commit_fail_reasons"] == ["openings"] and verdict["first_fail_reason"] == "openings"
    assert (summary["per_pair"][0]["commit_pass"], summary["per_pair"][0]["run_root"]) == (False, "aa")


@pytest.mark.parametrize("part, listed", PARTS)
@pytest.mark.parametrize("value", NOTHING)
def test_a_commit_record_without_a_part_lists_nothing_for_it(tmp_path: Path, part: str, listed: str, value: Any) -> None:
    treatment = process("served", "aa")
    stand_in(treatment["runs"][0]["commit"], part, value)
    rc, summary, _ = merge(tmp_path, [process("off")], [treatment])
    assert rc == 0 and summary["commit_pass"] is True
    assert {k: summary[k] for k in PER_RUN} == {k: [None if k == listed else v] for k, v in PER_RUN.items()}


def test_two_rounds_join_their_verdicts_run_roots_in_round_order(tmp_path: Path) -> None:
    rc, summary, verdict = merge(tmp_path, [process("off"), process("off")], [process("served", "aa"), process("served", "bb")])
    assert rc == 0 and (summary["rounds"], summary["pairs"]) == (2, 2) and summary["run_roots"] == ["aa", "bb"]
    assert verdict["run_roots"] == ["aa", "bb"] == summary["treatment_verdict"]["run_roots"]
    assert (verdict["runs_instrumented"], verdict["runs_failed"]) == (2, 0)


@pytest.mark.parametrize("without, joined", [(0, ["bb"]), (1, ["aa"])])
@pytest.mark.parametrize("value", NOTHING)
def test_a_rounds_verdict_without_run_roots_adds_none(tmp_path: Path, without: int, joined: list[str], value: Any) -> None:
    treatment = [process("served", "aa"), process("served", "bb")]
    stand_in(treatment[without]["verdict"], "run_roots", value)
    rc, summary, verdict = merge(tmp_path, [process("off"), process("off")], treatment)
    assert rc == 0 and verdict["run_roots"] == joined and summary["run_roots"] == ["aa", "bb"]


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("value", NOTHING)
def test_an_arm_whose_versions_identify_no_generated_kernels_is_refused(tmp_path: Path, arm: int, value: Any) -> None:
    arms = [process("off"), process("served", "aa")]
    stand_in(arms[arm]["versions"]["compilation"], "generated_kernels", value)
    rc, summary, _ = merge(tmp_path, arms[:1], arms[1:])
    generated = summary["generated_kernels"]
    assert rc == 3 and summary["problems"] == [NOT_IDENTIFIED]
    assert (generated["identified"], generated["kernel_set_equal"], generated["kernel_diff"]) == (False, False, None)
    assert set(generated[("control", "treatment")[arm]].values()) == {None}
    assert generated[("treatment", "control")[arm]]["kernel_set_sha256"] == "aa"


@pytest.mark.parametrize("arm, only", [(0, "only_treatment"), (1, "only_control")])
@pytest.mark.parametrize("value", NOTHING)
def test_an_arm_that_lists_no_kernel_by_name_leaves_the_others_kernels_as_the_difference(tmp_path: Path, arm: int, only: str,
                                                                                          value: Any) -> None:
    arms = [process("off"), process("served", "aa")]
    stand_in(arms[arm]["versions"]["compilation"]["generated_kernels"], "kernels", value)
    rc, summary, _ = merge(tmp_path, arms[:1], arms[1:])
    generated = summary["generated_kernels"]
    assert rc == 0 and summary["problems"] == [] and (generated["identified"], generated["kernel_set_equal"]) == (True, True)
    names = sorted(KERNELS["kernels"])
    assert generated["kernel_diff"] == {"only_control": [], "only_treatment": [], "same_name_different_source": []} | {only: names}
