"""The Programs a tensor-parallel row hands its fold Match (`verity_vllm.pipeline.row_tp`: `TpRow.match`), on CPU.

For each rank, in rank order: its first request Program on disk is the rank's Build (``--build-rank DIR``), each further one is
``--derived-rank R=DIR``, and its workload Program, when it is on disk, is ``--gprog-rank R=FILE``. A request dir whose derive
left no ``result.json`` is passed over, as are the log and the exit code beside each dir; a rank with no request Program is
handed none.

The row runs its Match's CPU half for real over a row directory (`tests.pipeline.test_tp_row_stages`): the fold Match is a
stand-in that keeps its command."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.pipeline.test_tp_row_stages import _Match
from verity_vllm.pipeline.row_tp import TpRow


@pytest.fixture
def match(tmp_path, monkeypatch) -> _Match:
    m = _Match(tmp_path, monkeypatch)
    m.record("tp2_match.json", True)
    m.record("fold_match/tp2_fold_match.json", True)
    return m


def _request(t: TpRow, r: int, shape: str, *, result: bool = True) -> str:
    """Rank `r`'s request dir of `shape` with the log and the exit code its derive leaves beside it, and its result when it has one."""
    d = f"{t.rank(r)}/build_request_{shape}"
    os.makedirs(d)
    for beside in (".log", ".rc"):
        Path(d + beside).write_text("0\n")
    if result:
        Path(d, "result.json").write_text('{"ok": true}')
    return d


def _workload(t: TpRow, r: int) -> str:
    gp = f"{t.rank(r)}/build_workload/workload_program.json"
    os.makedirs(os.path.dirname(gp))
    Path(gp).write_text('{"workload_digest": "' + str(r) * 64 + '"}')
    return gp


def _handed(m: _Match) -> list[str]:
    """What the row hands the fold Match between its profile and its workload: the ranks' Programs, then how it runs."""
    m.row.match()
    assert m.verdict() == "PASS rc=0/0/0/0"
    cmd = m.ran["match.fold_match"]
    return cmd[cmd.index("--profile") + 2:cmd.index("--workload")]


def test_each_rank_hands_its_first_request_program_as_its_build_the_rest_as_derived_and_its_workload_program(match):
    t = match.row
    first, second, third = (_request(t, 0, shape) for shape in ("LP256_T31", "LP16_T3", "LP100_T4"))       # read in the order of their names
    derived, build = _request(t, 1, "LP256_T31"), _request(t, 1, "LP16_T3")
    programs = [_workload(t, r) for r in range(2)]
    for r in range(2):                                                                                      # the step Program is no request's
        os.makedirs(f"{t.rank(r)}/build_step")
        Path(t.rank(r), "build_step", "result.json").write_text('{"ok": true}')
    assert _handed(match) == ["--build-rank", third, "--derived-rank", f"0={second}", "--derived-rank", f"0={first}", "--gprog-rank", f"0={programs[0]}",
                              "--build-rank", build, "--derived-rank", f"1={derived}", "--gprog-rank", f"1={programs[1]}", "--shared-x09"]


def test_a_request_dir_with_no_result_is_passed_over_and_the_next_one_with_a_result_is_the_ranks_build(match):
    t = match.row
    _request(t, 0, "LP100_T4", result=False)
    build, derived = _request(t, 0, "LP16_T3"), _request(t, 0, "LP256_T31")
    _request(t, 0, "LP300_T2", result=False)
    os.makedirs(f"{t.rank(1)}/build_request_LP256_T31/result.json")                   # a directory of that name is no result
    assert _handed(match) == ["--build-rank", build, "--derived-rank", f"0={derived}", "--shared-x09"]


def test_a_ranks_workload_program_is_handed_only_when_it_is_a_file_on_disk(match):
    t = match.row
    builds = [_request(t, r, "LP256_T31") for r in range(2)]
    os.makedirs(f"{t.rank(0)}/build_workload/workload_program.json")
    program = _workload(t, 1)
    assert _handed(match) == ["--build-rank", builds[0], "--build-rank", builds[1], "--gprog-rank", f"1={program}", "--shared-x09"]


def test_a_row_with_no_program_on_disk_hands_the_fold_match_none(match):
    assert _handed(match) == ["--shared-x09"]
