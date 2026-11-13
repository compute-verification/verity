"""A single-GPU row's Match stage with every stage command stood in (`verity_vllm.pipeline.row_stages`): the commands it runs, in
their order, and the small files it writes beside them (``match_shape.json``, the log of a comparison that did not run, the row
log's line for a manifest that does not read).  CPU only: no stage process starts."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from tests.pipeline.test_row import _row, _timeline
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.row_stages import SingleRow

Ran = list[tuple[str, list[str]]]


def _match_row(tmp_path: Path, monkeypatch, *argv: str, B: int = 1, rcs: dict[str, int | list[int]] | None = None) -> tuple[SingleRow, Ran]:
    """A row at its Match (LP 8, T 3) whose stage commands are recorded and not run -> the row and each (stage or log name, command).
    ``rcs``: what a stage returns, 0 by default; a list gives one rc a call."""
    r = _row(tmp_path, *argv)
    r.B, r.I, r.O, r.LP, r.T, r.MML, r.ck, r.snap_def, r.snap_steps = B, 256, 32, 8, 3, 288, "/ck", "0,1", None
    ran: Ran = []
    calls: dict[str, int] = {}

    def rc_of(name: str) -> int:
        rc = (rcs or {}).get(name, 0)
        calls[name] = calls.get(name, 0) + 1
        return rc[calls[name] - 1] if isinstance(rc, list) else rc

    def run_tl(stage: str, to: int, lf: str, cmd: list[str], *extra: tuple) -> int:
        ran.append((stage, cmd))
        return rc_of(stage)

    def run_to(to: int, lf: str, cmd: list[str]) -> int:
        ran.append((os.path.basename(lf), cmd))
        return rc_of(os.path.basename(lf))

    monkeypatch.setattr(r, "run_tl", run_tl)
    monkeypatch.setattr(r, "run_to", run_to)
    return r, ran


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


# ---------------------------------------------------------------------------------------------------------------- the comparison

def test_a_single_request_is_compared_against_the_program_of_the_shape_it_was_served_at(tmp_path, monkeypatch):
    """B=1: the request Program is the Build's when the fold ran every decode step the row asks for, and the one of the served shape
    (derived when not on disk) when the request stopped early; ``match_shape.json`` holds, in this order, LP, the served T, the row's T
    and the Program dir either way, under the names its reader looks them up by."""
    r, ran = _match_row(tmp_path, monkeypatch, rcs={"match_compare.log": 7})
    d = r.d
    assert r.compare(60) == (7, None, False)
    shape = Path(d, "match_shape.json").read_text()
    assert shape.endswith("}\n") and list(json.loads(shape).values()) == [8, 3, 3, f"{d}/build_request"] and R.served_shape(d) == ""
    (name, cmd), = ran
    assert name == "match_compare.log" and cmd[3] == "program-compare"
    assert (_flag(cmd, "--a"), _flag(cmd, "--b"), _flag(cmd, "--out")) == (f"{d}/build_request", f"{d}/match", f"{d}/match_compare.json")

    ran.clear()
    os.makedirs(f"{d}/match")
    Path(d, "match", "fold_summary.json").write_text(json.dumps({"n_steps": 3}))      # two decode steps served of the row's three
    assert r.compare(60) == (7, None, False)
    early = f"{d}/build_request_LP8_T2"
    assert list(json.loads(Path(d, "match_shape.json").read_text()).values()) == [8, 2, 3, early]
    assert re.findall(r"\d+", R.served_shape(d)) == ["8", "2", "3"]
    (derive, dcmd), (name, cmd) = ran
    assert derive == "build_request_LP8_T2.log" and (_flag(dcmd, "--LP"), _flag(dcmd, "--T"), _flag(dcmd, "--out")) == ("8", "2", early)
    assert name == "match_compare.log" and _flag(cmd, "--a") == early
    assert Path(d, "stages.txt").read_text().startswith("eos-truncated T_r=2 ") and "early EOS: served T_r=2" in Path(d, "row.log").read_text()

    ran.clear()
    os.makedirs(early)
    Path(early, "result.json").write_text("{}")                                       # the served shape's Program is on disk
    assert r.compare(60) == (7, None, False) and [name for name, _ in ran] == ["match_compare.log"]


def test_a_batchs_comparison_runs_inside_the_global_match_and_derives_the_shapes_it_names_once(tmp_path, monkeypatch):
    """B>=2 under the shared pipeline: GM-01 runs first and writes the X-09 decomposition, so the compare log only says where to look;
    a first pass that names missing request shapes (rc 4) derives them and runs once more."""
    r, ran = _match_row(tmp_path, monkeypatch, B=2, rcs={"match.global_match": [4, 0]})
    d = r.d
    os.makedirs(f"{d}/build_workload")
    Path(d, "build_workload", "workload_program.json").write_text("{}")
    Path(d, "match_decomp.json").write_text(json.dumps({"verdict": "MISSING_DERIVED", "missing_derived": [[16, 2]]}))
    assert r.compare(60) == (4, 0, True)
    assert Path(d, "match_compare.log").read_text() == "$ (shared) X-09 document written by cb_a.global_match --decomp-out; see global_match.log\n"
    (gm1, c1), (derive, dcmd), (gm2, c2) = ran
    assert (gm1, gm2, derive) == ("match.global_match", "match.global_match", "build_request_LP16_T2.log")
    assert (_flag(dcmd, "--LP"), _flag(dcmd, "--T")) == ("16", "2") and _flag(c1, "--decomp-out") == f"{d}/match_decomp.json"
    assert c2[-2:] == ["--derived", f"{d}/build_request_LP16_T2"] and c1[-2:] == ["--derived", f"{d}/build_request"]
    assert [e["shared"] for e in _timeline(r) if e["stage"] == "match.compare"] == ["match.global_match"]

    ran.clear()
    Path(d, "match_decomp.json").write_text(json.dumps({"verdict": "PASS"}))
    os.remove(f"{d}/build_workload/workload_program.json")                            # no workload Program: X-09 on its own
    assert r.compare(60) == (0, None, False) and [name for name, _ in ran] == ["match_compare.log"]
    assert ran[0][1][3] == "batch-decomp"


# ---------------------------------------------------------------------------------------------------------------- the stage

def _summary(monkeypatch, verdict: str = "PASS") -> list[tuple]:
    """Stand in for the Match summary -> the (rcm, rcc, gm, rcs, resumed) it is asked to judge."""
    asked: list[tuple] = []

    def match_summary(d: str, rcm: int, rcc: int, gm: int, t0: int, rcs: int, resumed: bool) -> str:
        asked.append((rcm, rcc, gm, rcs, resumed))
        return verdict

    monkeypatch.setattr(R, "match_summary", match_summary)
    monkeypatch.setattr(R, "match_line", lambda d: "the summary's line")
    return asked


def test_a_match_whose_fold_left_no_program_is_not_compared_and_says_so(tmp_path, monkeypatch):
    """No ``match/program.json`` or ``match/instances.jsonl``: neither comparison runs, both count rc 99, and the compare log says why."""
    r, ran = _match_row(tmp_path, monkeypatch, rcs={"match.run_config": 5})
    asked = _summary(monkeypatch, "FAIL")
    d = r.d
    Path(d, "match_compare.log").write_text("$ an earlier attempt's comparison\n")
    with pytest.raises(Stop) as e:
        r.match()
    assert e.value.rc == 11 and [name for name, _ in ran] == ["match.run_config"] and asked == [(5, 99, 99, -1, False)]
    assert Path(d, "match_compare.log").read_text() == f"no program.json/instances.jsonl in {d}/match (fold did not run or failed)\n"
    last = Path(d, "stages.txt").read_text().splitlines()[-1].split()
    assert last[:2] == ["match", "FAIL"] and last[3] == "rc=5/99/99"
    assert [e["rc"] for e in _timeline(r) if e["stage"] == "match.compare" and e["ev"] == "end"] == [99]


def test_a_compared_match_without_the_builds_workload_program_has_no_global_match_and_says_so(tmp_path, monkeypatch):
    """The fold's Program is compared; the global Match needs the row's Build workload Program, runs over it when it is there, and
    counts rc 98 with the reason in its log when it is not."""
    r, ran = _match_row(tmp_path, monkeypatch)
    asked = _summary(monkeypatch)
    d = r.d
    os.makedirs(f"{d}/match")
    for name in ("program.json", "instances.jsonl"):
        Path(d, "match", name).write_text("")
    r.match()
    assert [name for name, _ in ran] == ["match.run_config", "match_compare.log"] and asked == [(0, 0, 98, -1, False)]
    assert Path(d, "global_match.log").read_text() == ("no build_workload/workload_program.json: the global Match needs this row's Build "
                                                       "workload Program (GP-01)\n")
    last = Path(d, "stages.txt").read_text().splitlines()[-1].split()
    assert last[:2] == ["match", "PASS"] and last[3] == "rc=0/0/98" and " ".join(last[5:]) == "the summary's line"

    ran.clear()
    asked.clear()
    os.remove(f"{d}/global_match.log")
    os.makedirs(f"{d}/build_workload")
    Path(d, "build_workload", "workload_program.json").write_text("{}")
    r.match()
    assert [name for name, _ in ran] == ["match.run_config", "match_compare.log", "match.global_match"] and asked == [(0, 0, 0, -1, False)]
    gm = ran[-1][1]
    assert (_flag(gm, "--gprog"), _flag(gm, "--within-step")) == (f"{d}/build_workload/workload_program.json", "dag")
    assert not os.path.exists(f"{d}/global_match.log")


# ---------------------------------------------------------------------------------------------------------------- form (B)

def _manifest(r: SingleRow, *specs: str) -> None:
    """The row's required-value manifest, one ``instance_outputs`` identity for each producer spec."""
    Path(r.d, "manifest.json").write_text(json.dumps({
        "schema": "verity-sweep/required-manifest/v1", "program_digest": "p" * 64, "manifest_digest": "m" * 64, "requests": ["r0"],
        "identities": [{"family": "instance_outputs", "spec": s} for s in specs]}))


def test_required_values_with_no_replay_evaluator_make_the_match_snapshot_every_step(tmp_path, monkeypatch):
    """The Commit checks such values by form (B) alone, complete only over every step: the row's default snapshot steps become ``all``
    for them, an explicit choice stands with a warning, and a manifest that names none changes nothing."""
    r, _ = _match_row(tmp_path, monkeypatch)
    _manifest(r, "no_such_family{a}", "no_such_family{b}", "another_unheard_of{c}")
    r.form_b()
    log = Path(r.d, "row.log").read_text()
    assert r.snap_def == "all" and "required values with no replay evaluator: another_unheard_ofx1 no_such_familyx2)" in log

    r.snap_def, r.snap_steps = "0,1", "0,1"
    r.form_b()
    assert r.snap_def == "0,1" and "WARN match: MATCH_SNAP_STEPS=0,1 explicit while form (B) must be complete" in Path(r.d, "row.log").read_text()
    r.snap_steps = "all"
    before = Path(r.d, "row.log").read_text()
    r.form_b()
    assert Path(r.d, "row.log").read_text() == before

    r.snap_steps = None
    _manifest(r)
    r.form_b()
    assert r.snap_def == "0,1" and Path(r.d, "row.log").read_text() == before


def test_a_manifest_that_does_not_read_leaves_the_snapshot_steps_and_is_named_in_the_row_log(tmp_path, monkeypatch):
    r, _ = _match_row(tmp_path, monkeypatch)
    Path(r.d, "manifest.json").write_text("{")
    r.form_b()
    first, warn = Path(r.d, "row.log").read_text().splitlines()
    assert first.startswith("form (B) families: JSONDecodeError: ") and r.snap_def == "0,1"
    assert warn.endswith("WARN match: form (B) families not derived (rc=1, see row.log): snapshot steps stay 0,1")
    Path(r.d, "manifest.json").write_text('{"identities": []}')                     # JSON, and no manifest of record
    r.form_b()
    assert Path(r.d, "row.log").read_text().splitlines()[2].startswith("form (B) families: MissingField: ") and r.snap_def == "0,1"
    os.remove(f"{r.d}/manifest.json")
    r.form_b()
    assert Path(r.d, "row.log").read_text().splitlines()[4].startswith("form (B) families: FileNotFoundError: ")
