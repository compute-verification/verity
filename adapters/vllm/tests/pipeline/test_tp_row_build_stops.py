"""A tensor-parallel row's Build where it leaves its usual path (`verity_vllm.pipeline.row_tp`: `TpRow.build` and its parts), on CPU.

* A switched row's AOT Build with no ``--aot-trace`` fetches its trace Build's derive dirs from the store, every rank's step and
  declared request dirs and nothing else, and derives over them. A fetch the store refuses fails the Build by name before a
  derive runs.
* A signal that arrives while the derives run stops the row once every derive of the pool has ended, with the signal's exit code
  and no Build verdict.
* The manifest is not built over a rank that has no workload digest.
* ``--build-relabel on`` hands the row's derives to `verity_vllm.pipeline.tp.relabel_build`; the AOT Build never does.
* A Build summary that cannot be read fails the Build with what the reader raised; anything else it raises is not caught.

The Build runs for real over a row directory with its children stood in (`tests.pipeline.test_tp_build_width`), and the store
is `store_data.data`, the one place the row's reader runs `research data`."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.pipeline import test_tp_build_width as BW
from tests.pipeline.test_tp_row_precheck import _events, _log, _stages
from verity_vllm.pipeline import row_driver, row_tp
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.row_tp import TpRow
from verity_vllm.pipeline.tp import relabel_build

AOT = {"comparison": "r20261018-001500-aaaa", "trace": "art:" + "5" * 64}
SAID = f"T [row {BW.ROW}] "
#: what the trace Build's tree holds: the derive dirs of three ranks and three shapes, and what lies beside them
TREE = [f"build/rank{r}/{name}/{f}" for r in range(3) for name in ("build_step", "build_request_LP256_T31", "build_request_LP16_T3",
                                                                    "build_request_LP256_T310", "build_workload")
        for f in ("result.json", "program/words.bin")] + \
       [f"build/rank{r}/{name}" for r in range(2) for name in ("build_step.log", "build_step.rc", "build_request_LP256_T31.log")] + \
       ["rank0/build_step/result.json", "manifest.json", "build_summary.json", "match/capture/build/rank0/build_step/result.json"]


def _row(tmp: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> tuple[TpRow, BW._FakeChildren]:
    """`tests.pipeline.test_tp_build_width`'s row with its children stood in, a fixed stamp, and no launch context to read."""
    t, children, _pools = BW._tp_row(tmp, monkeypatch, **kw)
    monkeypatch.setattr(row_driver, "utc", lambda: "T")
    monkeypatch.setattr(t, "launch_args", lambda lp, t_: [])
    return t, children


def _said(t: TpRow) -> list[str]:
    return [ln[len(SAID):] for ln in _log(t) if ln.startswith(SAID)]


def _stage(t: TpRow) -> list[tuple[str, Any]]:
    return [(e["ev"], e.get("ok")) for e in _events(t, "stage.build")]


def _derives(children: BW._FakeChildren) -> list[list[str]]:
    return [a for a in children.cmds if a[0] in ("build", "predict")]


class _Store:
    """`store_data.data`, `research data` as the row's reader runs it: a tree's listing and a fetch of named paths, each call
    recorded. `refuses` is the word of the call it refuses (``--list``, ``--to``), with what the command said."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, refuses: str = "") -> None:
        self.refuses, self.asked = refuses, list[tuple[str, ...]]()
        monkeypatch.setattr(row_tp.SD, "data", self)

    def __call__(self, *args: str) -> str:
        self.asked.append(args)
        if self.refuses and self.refuses in args:
            raise RuntimeError(f"research data fetch {args[1]}: rc=1: no such artifact")
        if "--list" in args:
            return json.dumps([{"path": p, "bytes": 1} for p in TREE])
        return f"fetched {len(args) - 4} file(s)\n{args[args.index('--to') + 1]}\n"


# ---------------------------------------------------------------------------------------------------------------- the AOT Build's trace

def test_an_aot_build_without_a_trace_dir_fetches_each_ranks_declared_derive_dirs_and_derives_over_them(tmp_path, monkeypatch):
    monkeypatch.setattr(row_tp.CR, "aot_of", lambda row: AOT if row == BW.ROW else None)
    store = _Store(monkeypatch)
    t, children = _row(tmp_path, monkeypatch, world=2, shapes=[(256, 31), (16, 3)])
    t.build()
    into = f"{t.d}/aot_trace"
    taken = [f"build/rank{r}/{name}/{f}" for r in range(2) for name in ("build_step", "build_request_LP256_T31", "build_request_LP16_T3")
             for f in ("result.json", "program/words.bin")]
    assert store.asked == [("fetch", AOT["trace"], "--list", "--json"), ("fetch", AOT["trace"], "--to", into, *(f"--path={p}" for p in taken))]
    assert t.aot == AOT and t.aot_dir == into
    assert sorted(a[a.index("--trace") + 1] for a in _derives(children)) == sorted(
        f"{into}/build/rank{r}/{name}".replace(str(tmp_path), "<sweep>") for r in range(2)
        for name in ("build_step", "build_request_LP256_T31", "build_request_LP16_T3"))
    (mark,) = _events(t, "build.construction")
    assert (mark["construction"], mark["comparison"], mark["trace"]) == ("aot", AOT["comparison"], AOT["trace"])
    assert _stage(t) == [("start", None), ("end", "PASS")] and _stages(t)[0].endswith(" shapes=2 construction=aot")


@pytest.mark.parametrize("environ, aot, construction", [
    ({"AOT_TRACE": "/given/trace"}, AOT, "aot"),             # the trace is on the host
    ({"BUILD_CONSTRUCTION": "trace"}, AOT, "trace"),         # a switched row, traced
    ({}, None, "trace"),                                     # a row with no AOT record
])
def test_a_build_with_its_trace_given_or_no_aot_asks_the_store_for_nothing(tmp_path, monkeypatch, environ, aot, construction):
    monkeypatch.setattr(row_tp.CR, "aot_of", lambda row: aot)
    store = _Store(monkeypatch)
    t, children = _row(tmp_path, monkeypatch, world=2, environ=environ)
    t.build()
    assert store.asked == [] and t.aot_dir == ("/given/trace" if construction == "aot" else None)
    assert (t.aot is None) is (construction == "trace") and len(_derives(children)) == 4
    assert [(e["ev"], e["construction"], "trace" in e) for e in _events(t, "build.construction")] == [("mark", construction, construction == "aot")]


@pytest.mark.parametrize("refuses, asked", [("--list", 1), ("--to", 2)])
def test_an_aot_trace_the_store_does_not_give_fails_the_build_by_name_before_any_derive(tmp_path, monkeypatch, refuses, asked):
    monkeypatch.setattr(row_tp.CR, "aot_of", lambda row: AOT)
    store = _Store(monkeypatch, refuses=refuses)
    t, children = _row(tmp_path, monkeypatch, world=2)
    with pytest.raises(Stop) as stop:
        t.build()
    assert stop.value.rc == 10 and stop.value.__cause__ is None and stop.value.__suppress_context__
    why = f"the AOT Build's trace {AOT['trace']} did not fetch (research data fetch {AOT['trace']}: rc=1: no such artifact); --aot-trace DIR gives it"
    assert len(store.asked) == asked and children.cmds == []
    assert _stages(t) == [f"build FAIL T rc=1 {why}"]
    assert _stage(t) == [("start", None)] and _events(t, "build.construction") == []
    assert [(e["ev"], e["ok"], e["stopped_at"]) for e in _events(t, "row")] == [("end", False, "build")]


# ---------------------------------------------------------------------------------------------------------------- a signal during the derives

@pytest.mark.parametrize("name, rc", [("TERM", 143), ("INT", 130), ("HUP", 129)])
def test_a_signal_while_the_derives_run_stops_the_row_when_every_derive_has_ended(tmp_path, monkeypatch, name, rc):
    """The row's handler notes the signal and forwards it to the running children; the pool's threads cannot stop the row, so the
    Build does, after the pool, with the code a signalled row ends by."""
    t, children = _row(tmp_path, monkeypatch, world=2, environ={"BUILD_JOBS": "1"})

    def run_to(limit: int, log: str, cmd: list[str]) -> int:
        if len(children.cmds) == 1:
            t.signalled = name
        return children(limit, log, cmd)

    monkeypatch.setattr(t, "run_to", run_to)
    with pytest.raises(Stop) as stop:
        t.build()
    assert stop.value.rc == rc and [a[0] for a in children.cmds] == ["build"] * 4
    assert _said(t)[-1] == f"row cancelled by SIG{name}; stage ended rc=1 -> stop" and not Path(t.d, "stages.txt").exists()
    assert [(e["ev"], e.get("signal"), e.get("rc"), e.get("stopped_at")) for e in _events(t, "row")] == [
        ("cancelled", name, 1, None), ("end", None, None, "cancel")]
    assert _stage(t) == [("start", None)]


def test_a_derive_that_raises_ends_the_build_with_what_it_raised(tmp_path, monkeypatch):
    t, children = _row(tmp_path, monkeypatch, world=2, environ={"BUILD_JOBS": "1"})

    def run_to(limit: int, log: str, cmd: list[str]) -> int:
        if len(children.cmds) == 2:
            raise OSError("no space left on device")
        return children(limit, log, cmd)

    monkeypatch.setattr(t, "run_to", run_to)
    with pytest.raises(OSError, match="no space left on device"):
        t.build()
    assert not Path(t.d, "stages.txt").exists() and _stage(t) == [("start", None)]


# ---------------------------------------------------------------------------------------------------------------- the manifest's ranks

def test_the_manifest_is_built_over_every_ranks_programs_and_workload_digest_in_rank_order(tmp_path, monkeypatch):
    t, children = _row(tmp_path, monkeypatch, world=3)
    t.build()
    (cmd,) = [a for a in children.cmds if a[:2] == ["manifest", "build-global"]]
    digests = [json.loads(Path(t.rank(r), "build_workload/workload_program.json").read_text())["workload_digest"] for r in range(3)]
    assert len(set(digests)) == 3
    assert cmd[cmd.index("--programs-root"):cmd.index("--ranks")] == [a for r in range(3) for a in (
        "--programs-root", t.rank(r).replace(str(tmp_path), "<sweep>"), "--rank", str(r))]
    assert cmd[cmd.index("--ranks"):cmd.index("--ranks") + 8] == ["--ranks", "3", *(a for d in digests for a in ("--workload-digest", d))]
    assert t.manifest_global("commit.required_manifest") == 0 and children.cmds[-1] == cmd       # as the Commit builds it again
    assert [(e["ev"], e.get("rc")) for e in _events(t, "commit.required_manifest")] == [("start", None), ("end", 0)]
    monkeypatch.setattr(t, "run_to", lambda limit, log, cmd: 4)
    assert t.manifest_global("commit.required_manifest") == 4                                    # the answer is the child's exit code


@pytest.mark.parametrize("left", [None, "{}", '{"workload_digest": ""}', '{"workload_digest": null}', "not JSON", "[]"])
def test_a_rank_without_a_workload_digest_gets_no_manifest_built(tmp_path, monkeypatch, left):
    t, children = _row(tmp_path, monkeypatch, world=3)
    t.build()
    ran, said = len(children.cmds), len(_said(t))
    program = Path(t.rank(1), "build_workload/workload_program.json")
    if left is None:
        program.unlink()
    else:
        program.write_text(left)
    assert t.manifest_global("commit.required_manifest") == 4
    assert len(children.cmds) == ran and _events(t, "commit.required_manifest") == []
    assert _said(t)[said:] == [f"manifest: 3 component builds (3 ranks x 1 requests), limit {R.manifest_timeout(3)}s",
                               "manifest: GP-01 workload digest absent for rank 1"]


# ---------------------------------------------------------------------------------------------------------------- the relabel Build

@pytest.mark.parametrize("environ, aot, handed", [({"BUILD_RELABEL": "on"}, None, True), ({"BUILD_RELABEL": "off"}, None, False), ({}, None, False),
                                                  ({"BUILD_RELABEL": "on", "AOT_TRACE": "/given/trace"}, AOT, False)])
def test_build_relabel_on_hands_the_rows_derives_to_the_relabel_build_and_an_aot_build_never_does(tmp_path, monkeypatch, environ, aot, handed):
    monkeypatch.setattr(row_tp.CR, "aot_of", lambda row: aot)
    t, children = _row(tmp_path, monkeypatch, world=2, shapes=[(256, 31), (16, 3)], environ=environ)
    got: list[tuple[Any, ...]] = []

    def derive(row: TpRow, todo: list[tuple[Any, ...]], jobs: int) -> None:
        got.append((row, [(job[0], os.path.basename(job[1])) for job in todo], jobs))
        row.derive_all(todo, jobs)

    monkeypatch.setattr(relabel_build, "derive", derive)
    t.build()
    (jobs,) = _events(t, "build.jobs")
    todo = [(r, name) for name in ("build_request_LP256_T31", "build_request_LP16_T3") for r in range(2)] + [(r, "build_step") for r in range(2)]
    assert got == ([(t, todo, jobs["jobs"])] if handed else []) and (jobs["jobs"], jobs["derives"]) == (6, 6)
    assert len(_derives(children)) == 6 and _stage(t) == [("start", None), ("end", "PASS")]


# ---------------------------------------------------------------------------------------------------------------- the Build's summary

def _a_directory(path: Path) -> None:
    path.unlink()
    path.mkdir()


@pytest.mark.parametrize("spoil, said", [
    (lambda p: p.write_text("not JSON"), "JSONDecodeError: Expecting value: line 1 column 1 (char 0)"),
    (lambda p: p.write_text("[]"), "AttributeError: 'list' object has no attribute 'get'"),
    (lambda p: p.write_text('{"ok": true, "message": 7}'), "TypeError: 'int' object is not subscriptable"),
    (_a_directory, "IsADirectoryError: [Errno 21] Is a directory: '<step>/result.json'"),
], ids=["not-json", "a-list", "a-number-for-its-message", "a-directory"])
def test_a_build_summary_that_cannot_be_read_fails_the_build_with_what_the_reader_raised(tmp_path, monkeypatch, spoil, said):
    """A derive's ``result.json`` that the summary cannot read, spoiled here after its rank's compose: the Build fails with no
    manifest, and the row says what was raised."""
    t, children = _row(tmp_path, monkeypatch, world=2)
    step = f"{t.rank(1)}/build_step"

    def run_to(limit: int, log: str, cmd: list[str]) -> int:
        rc = children(limit, log, cmd)
        if "global-program" in cmd and cmd[cmd.index("--out") + 1] == f"{t.rank(1)}/build_workload":
            spoil(Path(step, "result.json"))
        return rc

    monkeypatch.setattr(t, "run_to", run_to)
    with pytest.raises(Stop) as stop:
        t.build()
    assert stop.value.rc == 10 and not [a for a in children.cmds if a[0] == "manifest"]
    assert f"build summary failed: {said.replace('<step>', step)}" in _said(t) and _said(t)[-1].startswith("BUILD FAIL -> stop (")
    (verdict,) = _stages(t)
    assert re.sub(r" wall=\d+s ", " ", verdict).startswith("build FAIL T rc=1 manifest_rc=None digest=") and verdict.endswith(" shapes=1 construction=trace")
    assert _stage(t) == [("start", None), ("end", "FAIL")] and not Path(t.d, "manifest.json").exists()
    assert [(e["ev"], e["ok"], e["stopped_at"]) for e in _events(t, "row")] == [("end", False, "build")]


@pytest.mark.parametrize("raised", [OSError("gone"), ValueError("no"), KeyError("rank0"), TypeError("a list"), AttributeError("get")])
def test_the_summary_guard_answers_fail_for_each_error_a_malformed_build_raises(tmp_path, monkeypatch, raised):
    t, _children = _row(tmp_path, monkeypatch, world=3)

    def summary(b: str, world: int) -> str:
        assert (b, world) == (t.bdir, 3)
        raise raised

    monkeypatch.setattr(R, "tp_build_summary", summary)
    assert t.guarded_summary() == "FAIL" and _said(t) == [f"build summary failed: {type(raised).__name__}: {raised}"]


def test_the_summary_guard_passes_on_what_the_summary_says_and_catches_no_other_error(tmp_path, monkeypatch):
    t, _children = _row(tmp_path, monkeypatch, world=2)
    t.build()
    said = len(_said(t))
    assert t.guarded_summary() == "PASS"
    shutil.rmtree(f"{t.rank(1)}/build_step")
    assert t.guarded_summary() == "FAIL" and len(_said(t)) == said          # a rank with no step Program: the summary's answer, nothing raised
    monkeypatch.setattr(R, "tp_build_summary", lambda b, world: (_ for _ in ()).throw(RuntimeError("a bug")))
    with pytest.raises(RuntimeError, match="a bug"):
        t.guarded_summary()
