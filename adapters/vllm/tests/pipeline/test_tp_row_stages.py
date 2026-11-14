"""A TP row's stages where they decide something on a CPU (`verity_vllm.pipeline.row_tp`): what a Build's derive leaves and what
its launch context does, the Match's CPU half (``TpRow.match`` under ``--match-phase cpu``), and what the row reads of the
replay its Commit deferred (``TpRow.commit_pending``, ``TpRow.replay``).

A derive leaves its exit code beside its log and the Build says each; a shape's launch context is read once for every rank,
and one that does not read fails the Build.

The Match passes only when the capture, the control and the collective-level check each passed and, with a rank profile, the
fold Match did too. A request that ended before its token limit gets its served-shape Programs derived on every rank, the
longest shape first, and the workload Programs composed again, before the fold Match.

The row runs for real over a row directory: its GPU half first (``--match-phase gpu``), which leaves the two GPU stages' exit
codes for the CPU half. Its children (the capture, the control, the check, the fold Match, the derives) are stand-ins that
answer an exit code, and the two Match records are files the test writes.

The Commit's summary is a document another process wrote: for each shape of its ``replay_deferred`` the two readers give one
answer, or raise, and the table says which."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.pipeline import test_tp_build_width as BW
from tests.pipeline.test_row import TP_DENSE, _tp_row
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.row_tp import TpRow

PROFILE = "derived_LLAMA32_1B_tp2"


def _said(t: TpRow, name: str) -> str:
    """The row's file `name` (``stages.txt``, ``row.log``), empty when the row has not written it."""
    path = Path(t.d, name)
    return path.read_text() if path.exists() else ""


# ------------------------------------------------------------------------------------------------ the Build's derives

def test_a_derive_leaves_its_exit_code_beside_its_log_and_the_build_says_each(tmp_path, monkeypatch):
    t, children, _pools = BW._tp_row(tmp_path, monkeypatch, world=2)
    killed = f"{t.rank(1)}/build_request_LP256_T31"

    def run_to(limit: int, log: str, cmd: list[str]) -> int:
        out = cmd[cmd.index("--out") + 1]
        if out == killed:                        # before it wrote anything
            Path(log).write_text("out of memory\n")
            return 137
        if "global-program" in cmd and "--derived" not in cmd:               # as the compose refuses a request with no Program
            os.makedirs(out)
            Path(out, "workload_program.json").write_text(json.dumps({"ok": False, "reason": "missing_derived"}))
            return 4
        return children(limit, log, cmd)

    monkeypatch.setattr(t, "run_to", run_to)
    monkeypatch.setattr(t, "launch_args", lambda lp, t_: [])
    with pytest.raises(Stop) as stop:
        t.build()
    assert stop.value.rc == 10 and _said(t, "stages.txt").split()[:2] == ["build", "FAIL"]
    assert {str(p.relative_to(t.bdir)): p.read_text() for p in Path(t.bdir).rglob("*.rc")} == {
        "rank0/build_request_LP256_T31.rc": "0\n", "rank0/build_step.rc": "0\n", "rank1/build_request_LP256_T31.rc": "137\n", "rank1/build_step.rc": "0\n"}
    derives = [ln.split("] ", 1)[1] for ln in _said(t, "row.log").splitlines() if "] derive " in ln]     # each with its log's last bytes
    assert [" ".join(ln.split()[:4]) for ln in derives] == ["derive build_request_LP256_T31 rank0 rc=0", "derive build_step rank0 rc=0",
                                                            "derive build_request_LP256_T31 rank1 rc=137", "derive build_step rank1 rc=0"]
    assert derives[2] == "derive build_request_LP256_T31 rank1 rc=137 out of memory " and derives[3].endswith("--wrapper step --M 8 --pos0 8 ")


def test_a_shapes_launch_context_is_read_once_for_every_rank(tmp_path, monkeypatch):
    t, _children, _pools = BW._tp_row(tmp_path, monkeypatch, world=4, shapes=[(256, 31), (100, 4)])
    read: list[tuple[int, int]] = []

    def launch_args(lp: int, t_: int) -> list[str]:
        read.append((lp, t_))
        return ["--launch-context", f"of {lp},{t_}"]

    monkeypatch.setattr(t, "launch_args", launch_args)
    requests = [job for job in t.build_todo() if "request" in job]
    assert read == [(256, 31), (100, 4)] and len(requests) == 8
    assert [(job[0], *job[4:]) for job in requests] == [(r, "--LP", str(lp), "--T", str(t_), "--launch-context", f"of {lp},{t_}")
                                                        for lp, t_ in [(256, 31), (100, 4)] for r in range(4)]


def test_a_launch_context_that_does_not_read_fails_the_build(tmp_path, monkeypatch):
    t, children, _pools = BW._tp_row(tmp_path, monkeypatch, world=2)
    read: list[tuple[int, int]] = []

    def launch_args(lp: int, t_: int) -> None:
        read.append((lp, t_))

    monkeypatch.setattr(t, "launch_args", launch_args)
    for _ in range(2):                           # the second Build of the row object reads no context again, and fails the same
        with pytest.raises(Stop) as stop:
            t.build()
        assert stop.value.rc == 10 and read == [(256, 31)] and children.cmds == []
    assert [ln.split()[:2] for ln in _said(t, "stages.txt").splitlines()] == [["build", "FAIL"]] * 2
    assert "rc=1 the launch context of shape LP256 T31 did not read (row.log)" in _said(t, "stages.txt")
    assert [(mark["ev"], mark["ok"], mark["stopped_at"]) for mark in BW._marks(t, "row")] == [("end", False, "build")] * 2


# ------------------------------------------------------------------------------------------------ the Match's CPU half

def _row(tmp: Path, phase: str, profile: str) -> TpRow:
    t: TpRow = _tp_row(tmp, TP_DENSE, "--match-phase", phase)
    t.fold_profile, t.shapes = profile, [(256, 31)]
    return t


class _Match:
    """A two-rank row at its Match's CPU half, after its GPU half ran with the exit codes `capture` and `control`: each CPU
    child's exit code in ``rc``, and the commands the CPU half ran in ``ran``, by stage."""

    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch, *, capture: int = 0, control: int = 0, profile: str = PROFILE) -> None:
        gpu, t = _row(tmp, "gpu", profile), _row(tmp, "cpu", profile)
        os.makedirs(t.M)
        gpu_rc = {"match.capture": capture, "match.control": control}
        monkeypatch.setattr(gpu, "run_gpu", lambda stage, limit, log, cmd, **lease: gpu_rc[stage])
        with pytest.raises(Stop) as stop:
            gpu.match()
        assert stop.value.rc == (0 if capture == control == 0 else 11)
        self.row, self.rc, self.ran = t, {"match.check": 0, "match.fold_match": 0}, dict[str, list[str]]()
        self.underived: set[str] = set()
        monkeypatch.setattr(R, "derived_ok", lambda path: os.path.basename(path) not in self.underived)
        monkeypatch.setattr(t, "run_tl", self._child)

    def _child(self, stage: str, limit: int, log: str, cmd: list[str]) -> int:
        self.ran[stage] = cmd
        return self.rc[stage]

    def record(self, name: str, passed: bool) -> None:
        """The check's record (``tp2_match.json``) or the fold Match's (``fold_match/tp2_fold_match.json``), saying `passed`."""
        path = Path(self.row.M, name)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"pass": passed}))

    def verdict(self) -> str:
        """The row's last ``match`` line of ``stages.txt``, its verdict and the exit codes."""
        line = [ln.split() for ln in Path(self.row.d, "stages.txt").read_text().splitlines() if ln.startswith("match ")][-1]
        return f"{line[1]} {next(word for word in line if word.startswith('rc='))}"

    def summary(self) -> dict[str, Any]:
        return json.loads(Path(self.row.d, "match_summary.json").read_text())


@pytest.fixture
def match(tmp_path, monkeypatch) -> _Match:
    m = _Match(tmp_path, monkeypatch)
    m.record("tp2_match.json", True)
    m.record("fold_match/tp2_fold_match.json", True)
    return m


def test_a_tp_match_whose_every_stage_passed_passes(match):
    match.row.match()
    assert match.verdict() == "PASS rc=0/0/0/0"
    assert list(match.ran) == ["match.check", "match.fold_match"]
    summary = match.summary()
    assert (summary["pass"], summary["capture_rc"], summary["control_rc"], summary["check_rc"], summary["fold_rc"]) == (True, 0, 0, 0, 0)
    assert (summary["fold_profile"], summary["tp2_match"], summary["tp2_fold_match"]["pass"]) == (PROFILE, {"pass": True}, True)
    fold = match.ran["match.fold_match"]
    assert fold[fold.index("--profile") + 1] == PROFILE and fold[fold.index("--tp2-match") + 1] == f"{match.row.M}/tp2_match.json"


@pytest.mark.parametrize("stage,rcs", [("capture", "3/0/0/0"), ("control", "0/3/0/0"), ("match.check", "0/0/3/0"), ("match.fold_match", "0/0/0/3")])
def test_a_tp_match_with_one_stage_that_failed_fails_and_stops_the_row(tmp_path, monkeypatch, stage, rcs):
    m = _Match(tmp_path, monkeypatch, capture=3 if stage == "capture" else 0, control=3 if stage == "control" else 0)
    m.record("tp2_match.json", True)
    m.record("fold_match/tp2_fold_match.json", True)
    if stage in m.rc:
        m.rc[stage] = 3
    with pytest.raises(Stop) as stop:
        m.row.match()
    assert stop.value.rc == 11 and m.verdict() == f"FAIL rc={rcs}"
    summary = m.summary()
    assert (summary["pass"], [summary[k] for k in ("capture_rc", "control_rc", "check_rc", "fold_rc")]) == (False, [int(rc) for rc in rcs.split("/")])
    assert "MATCH FAIL -> stop" in Path(m.row.d, "row.log").read_text()


@pytest.mark.parametrize("name", ["tp2_match.json", "fold_match/tp2_fold_match.json"])
@pytest.mark.parametrize("written", [False, None], ids=["says-fail", "not-written"])
def test_a_tp_match_whose_record_does_not_say_pass_fails_whatever_its_stages_exit_codes(match, name, written):
    if written is None:
        Path(match.row.M, name).unlink()
    else:
        match.record(name, written)
    with pytest.raises(Stop) as stop:
        match.row.match()
    assert stop.value.rc == 11 and match.verdict() == "FAIL rc=0/0/0/0" and match.summary()["pass"] is False


def test_a_tp_match_with_no_rank_profile_is_the_collective_level_check_alone(tmp_path, monkeypatch):
    m = _Match(tmp_path, monkeypatch, profile="")
    m.record("tp2_match.json", True)
    m.rc["match.fold_match"] = 3                 # not run, so its exit code and its record are not read
    m.row.match()
    assert m.verdict() == "PASS rc=0/0/0/-1" and list(m.ran) == ["match.check"]
    summary = m.summary()
    assert (summary["pass"], summary["fold_rc"], summary["fold_profile"], summary["tp2_fold_match"]) == (True, None, None, None)
    m.record("tp2_match.json", False)
    with pytest.raises(Stop):
        m.row.match()
    assert m.verdict() == "FAIL rc=0/0/0/-1"


def test_a_tp_match_without_a_ranks_step_program_fails_before_any_stage(match):
    match.underived = {"build_step"}
    with pytest.raises(Stop) as stop:
        match.row.match()
    assert stop.value.rc == 11 and match.ran == {}
    assert f"no step Program for rank 0 ({match.row.rank(0)}/build_step): Build first" in Path(match.row.d, "stages.txt").read_text()


def test_a_request_that_ended_before_its_token_limit_gets_its_served_shapes_derived_longest_first_on_every_rank(match, monkeypatch):
    row = match.row
    os.makedirs(f"{row.M}/capture")
    served = [{"prompt_token_ids": [0] * lp, "generated_token_ids": [0] * (t + 1)} for lp, t in [(100, 4), (256, 31), (200, 9), (100, 4)]]
    Path(row.M, "capture", "tokens.json").write_text(json.dumps({"requests": served}))
    match.underived = {"build_request_LP100_T4", "build_request_LP200_T9"}
    did: list[Any] = []

    def jobs(shapes: list[tuple[int, int]], derives: int) -> int:
        did.append(("jobs", shapes, derives))
        return 2

    monkeypatch.setattr(row, "launch_args", lambda lp, t: ["--launch", f"{lp},{t}"])
    monkeypatch.setattr(row, "derive_jobs", jobs)
    monkeypatch.setattr(row, "derive_all", lambda todo, jobs: did.append(("derive", todo, jobs)))
    monkeypatch.setattr(row, "compose_all", lambda ranks: did.append(("compose", ranks)))
    monkeypatch.setattr(row, "guarded_summary", lambda: did.append(("summary",)))
    monkeypatch.setattr(R, "tp_rank_workload_digest", lambda bdir, r: f"{r}" * 20)
    row.match()
    todo = [(r, f"{row.rank(r)}/build_request_LP{lp}_T{t}", "--wrapper", "request", "--LP", str(lp), "--T", str(t), "--launch", f"{lp},{t}")
            for lp, t in [(200, 9), (100, 4)] for r in (0, 1)]
    assert did == [("jobs", [(200, 9), (100, 4)], 4), ("derive", todo, 2), ("compose", [0, 1]), ("summary",)]
    assert all(type(shape) is tuple and len(shape) == 2 for shape in did[0][1])
    said = next(ln for ln in Path(row.d, "stages.txt").read_text().splitlines() if ln.startswith("eos-truncated "))
    assert said.startswith("eos-truncated served=[100,4 256,31 200,9 100,4] ") and " declared=[256,31]: request(s) stopped before" in said
    assert "Program(s) [ 100,4 100,4 200,9 200,9 100,4 100,4] per rank" in said
    assert "served-shape derivation done: workload digests r0=0000000000000000 r1=1111111111111111 " in Path(row.d, "row.log").read_text()
    assert match.verdict() == "PASS rc=0/0/0/0" and list(match.ran) == ["match.check", "match.fold_match"]


# ------------------------------------------------------------------------------------------------ the replay the Commit deferred

BUNDLES = [{"path": "/row/commit/replay_bundle_p0_rank0"}, {"path": "/row/commit/replay_bundle_p0_rank1"}]
NO_GET = "object has no attribute 'get'"
#: a Commit summary, then what `commit_pending` answers for it and what `replay` does: "skips" (nothing deferred, so no replay
#: runs), "runs" (the CPU replay), or the exception either raises
SUMMARIES: list[tuple[Any, Any, Any]] = [
    (None, False, "runs"),                                                    # no summary.json
    ("not JSON", False, "runs"),
    ({}, False, "runs"),
    ([], False, "runs"),
    (0, False, "runs"),
    ([1], NO_GET, NO_GET),
    ("a string", NO_GET, NO_GET),
    ({"pass": None}, False, "runs"),
    ({"pass": None, "replay_deferred": None}, False, "runs"),
    ({"pass": None, "replay_deferred": {}}, False, "runs"),
    ({"pass": None, "replay_deferred": []}, False, "runs"),
    ({"pass": None, "replay_deferred": ""}, False, "runs"),
    ({"pass": None, "replay_deferred": 0}, False, "runs"),
    ({"pass": None, "replay_deferred": "full"}, NO_GET, NO_GET),
    ({"pass": None, "replay_deferred": [1]}, NO_GET, NO_GET),
    ({"pass": None, "replay_deferred": 1}, NO_GET, NO_GET),
    ({"pass": None, "replay_deferred": {"bundles": None}}, False, "runs"),
    ({"pass": None, "replay_deferred": {"bundles": []}}, False, "runs"),
    ({"pass": None, "replay_deferred": {"bundles": {}}}, False, "runs"),
    ({"pass": None, "replay_deferred": {"bundles": 0}}, False, "runs"),
    ({"pass": None, "replay_deferred": {"bundles": BUNDLES}}, True, "runs"),
    ({"replay_deferred": {"bundles": BUNDLES}}, True, "runs"),
    ({"pass": True, "replay_deferred": {"bundles": BUNDLES}}, False, "runs"),
    ({"pass": False, "replay_deferred": {"bundles": BUNDLES}}, False, "runs"),
    ({"pass": None, "replay_deferred": {"bundles": BUNDLES[:1], "skipped": ""}}, True, "runs"),
    ({"pass": None, "replay_deferred": {"bundles": [{"rank": 0}]}}, "'path'", "runs"),
    ({"pass": None, "replay_deferred": {"bundles": [1]}}, "'int' object is not subscriptable", "runs"),
    ({"pass": None, "replay_deferred": {"bundles": 3}}, "object of type 'int' has no len()", "runs"),
    ({"pass": True, "replay_deferred": {"bundles": 3}}, False, "runs"),
    ({"pass": True, "replay_deferred": {"skipped": "world 4"}}, False, "skips"),
    ({"pass": True, "replay_deferred": {"skipped": None}}, False, "runs"),
    ({"pass": True, "replay_deferred": {"skipped": ""}}, False, "runs"),
    ({"pass": True, "replay_deferred": {"skipped": 0}}, False, "runs"),
    ({"pass": True, "replay_deferred": {"skipped": ["world 4"]}}, False, "skips"),
    ({"pass": None, "replay_deferred": {"skipped": "world 4", "bundles": BUNDLES}}, True, "skips"),
]


def _committed(tmp: Path, summary: Any) -> TpRow:
    """A row that defers its replay, with `summary` as its Commit's ``summary.json`` (None: no file)."""
    t: TpRow = _tp_row(tmp, TP_DENSE, "--replay-deferred", "1")
    os.makedirs(f"{t.d}/commit")
    if summary is not None:
        Path(t.d, "commit", "summary.json").write_text(summary if summary == "not JSON" else json.dumps(summary))
    return t


@pytest.mark.parametrize("summary,pending,replays", SUMMARIES, ids=[json.dumps(s[0])[:60] for s in SUMMARIES])
def test_what_a_tp_row_reads_of_the_replay_its_commit_deferred(tmp_path, monkeypatch, summary, pending, replays):
    t = _committed(tmp_path, summary)
    if isinstance(pending, str):
        with pytest.raises((AttributeError, KeyError, TypeError), match=pending):
            t.commit_pending(0, 7)
    else:
        assert t.commit_pending(0, 7) is pending and t.replay_pending is pending
        stages = _said(t, "stages.txt")
        assert ("commit PENDING" in stages) is pending
        if pending:
            bundles = summary["replay_deferred"]["bundles"]
            assert f"{len(bundles)} rank replay bundles sealed ({', '.join(os.path.basename(b['path']) for b in bundles)})" in stages
    ran: list[str] = []

    def child(stage: str, limit: int, log: str, cmd: list[str]) -> int:
        ran.append(stage)
        return 0

    monkeypatch.setattr(t, "run_tl", child)
    Path(t.d, "commit", "replay_deferred.json").write_text("{}")
    t.ck = str(tmp_path)
    if replays in ("runs", "skips"):
        t.replay()
        assert ran == (["replay.tp_commit_replay"] if replays == "runs" else [])
        assert ("replay: nothing deferred" in _said(t, "row.log")) is (replays == "skips")
        assert ("replay PASS" in _said(t, "stages.txt")) is (replays == "runs")
    else:
        with pytest.raises(AttributeError, match=replays):
            t.replay()
        assert ran == [] and "replay" not in _said(t, "stages.txt")


@pytest.mark.parametrize("deferral,rc", [("1", 3), ("0", 0), ("slim", 0)])
def test_a_tp_commit_is_pending_only_when_the_row_deferred_its_replay_and_the_commit_exited_zero(tmp_path, deferral, rc):
    t: TpRow = _tp_row(tmp_path, TP_DENSE, "--replay-deferred", deferral, *(["--world", "2"] if deferral == "slim" else []))
    os.makedirs(f"{t.d}/commit")
    Path(t.d, "commit", "summary.json").write_text(json.dumps({"pass": None, "replay_deferred": {"bundles": BUNDLES}}))
    assert t.commit_pending(rc, 7) is False and not t.replay_pending and not os.path.exists(f"{t.d}/stages.txt")
