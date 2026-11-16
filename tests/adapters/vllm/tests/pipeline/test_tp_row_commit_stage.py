"""A tensor-parallel row's Commit stage where the row itself decides (`verity_vllm.pipeline.row_tp`: `TpRow.commit`), on CPU.

Before its one GPU process the stage refuses a row without the tap build its ranks run, without a rank's workload Program, without a
complete manifest, or without the build of a tap its manifest names; a manifest that is not of record is built again first.  After
the process, the verdict is its exit and its summary, with the last line by which its log decided.

The row runs over a row directory with real files: each rank's workload Program, the manifest, the Commit's log and summary.
`tp-commit` and the manifest's rebuild are stand-ins that write those files and answer an exit code; whether the manifest beside the
Build is of record is the answer of `row_records.tp_manifest_not_of_record`, a stand-in too.  The driver's clock is injected."""
from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from tests.pipeline.test_tp_row_precheck import DENSE, _events, _log, _stages, _tp_row
from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.pipeline import row_config as RC
from verity_vllm.pipeline import row_driver
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.row_tp import TpRow

MANIFEST = {"complete": True, "populations": {"identities": 12}, "tp_peer_binding": {"n_unbound": 0}, "unmodelled": [],
            "manifest_digest": "ab" * 32, "identities": [{"family": "instance_outputs"}]}
MANIFEST_LINE = "complete True identities 12 tp_peer_binding_n_unbound 0 unmodelled [] digest abababababababab"
SUMMARY = {"pass": True, "pairs": 2, "tokens_equal": True, "sampled_replay": {"result": True}}
SUMMARY_LINE = ("pass True pairs 2 delta None roots null tokens True replay True linkage None xrank None fold_binding None "
                "weights_pin None")
PASSED = "[tp2commit] PASS every check of 2 pairs"
SAID = f"T [row {DENSE}] "


class _Commit:
    """A row at its Commit stage, after a Build of `world` ranks left a complete manifest: `of_record` is why that manifest is not the
    Commit's ("" when it is), `rebuilt` what building it again leaves (None: nothing) with the exit `rebuild_rc`, and `rc`, `log` and
    `summary` what ``tp-commit`` exits with and writes.  `ran` holds each child the stage started, with the stages' environment at
    that moment."""

    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch, *argv: str, world: int = 2, of_record: str = "", rc: int = 0,
                 log: tuple[str, ...] = (PASSED,), summary: Any = SUMMARY, rebuilt: Any = None, rebuild_rc: int = 0) -> None:
        self.row = t = _tp_row(tmp, *argv, world=world)
        self.ran: list[tuple[Any, ...]] = []
        self.asked: list[tuple[Any, ...]] = []
        os.makedirs(t.d)
        t.bdir, t.M, t.B, t.engine, t.ck, t.fold_profile, t.child_env = f"{t.d}/build", f"{t.d}/match", 1, [], str(tmp / "ck"), "", {}
        t.hidden_so = str(tmp / "verity_fa2_matReqG.so")
        Path(t.hidden_so).touch()
        for r in range(world):
            os.makedirs(f"{t.rank(r)}/build_workload")
            Path(t.rank(r), "build_workload", "workload_program.json").write_text(json.dumps({"workload_digest": f"{r}" * 64}))
        self.manifest(MANIFEST)

        def not_of_record(d: str, ck: Any) -> str:
            self.asked.append((d, ck))
            return of_record

        def manifest_global(stage: str) -> int:
            self.ran.append((stage,))
            if rebuilt is not None:
                self.manifest(rebuilt)
            return rebuild_rc

        def run_tl(stage: str, limit: int, lf: str, cmd: list[str]) -> int:
            self.ran.append((stage, limit, lf, cmd, dict(t.child_env)))
            Path(lf).write_text("".join(ln + "\n" for ln in log))
            if summary is not None:
                os.makedirs(f"{t.d}/commit", exist_ok=True)
                Path(t.d, "commit", "summary.json").write_text(json.dumps(summary))
            return rc

        monkeypatch.setattr(R, "tp_manifest_not_of_record", not_of_record)
        monkeypatch.setattr(t, "manifest_global", manifest_global)
        monkeypatch.setattr(t, "run_tl", run_tl)
        monkeypatch.setattr(row_driver, "utc", lambda: "T")

    def manifest(self, doc: Any) -> None:
        Path(self.row.d, "manifest.json").write_text(json.dumps(doc))

    def stops(self) -> int:
        with pytest.raises(Stop) as stop:
            self.row.commit()
        return stop.value.rc

    def said(self) -> list[str]:
        """The lines of the row's log that the row said, without the stamp and the row's name."""
        return [ln[len(SAID):] for ln in _log(self.row) if ln.startswith(SAID)]


def _stage(t: TpRow) -> list[tuple[str, Any]]:
    return [(e["ev"], e.get("ok")) for e in _events(t, "stage.commit")]


# ---------------------------------------------------------------------------------------------------------------- before the GPU process

def test_a_commit_without_the_tap_build_its_ranks_run_stops_before_anything(tmp_path, monkeypatch):
    c = _Commit(tmp_path, monkeypatch)
    os.remove(c.row.hidden_so)
    assert c.stops() == 12 and c.ran == [] and c.asked == []
    assert _stages(c.row) == [f"commit FAIL T hidden .so missing: {c.row.hidden_so}"] and _stage(c.row) == [("start", None)]


@pytest.mark.parametrize("world, rank, left", [(2, 0, None), (2, 1, None), (4, 3, None), (2, 1, {}), (2, 1, {"workload_digest": ""}),
                                               (2, 1, {"workload_digest": None}), (2, 1, ["a list"])])
def test_a_commit_without_a_ranks_workload_program_digest_stops_and_names_the_first_such_rank(tmp_path, monkeypatch, world, rank, left):
    """A rank's workload Program that is not there, or states no digest: the Commit names that rank and starts nothing."""
    c = _Commit(tmp_path, monkeypatch, world=world)
    for r in range(rank, world):
        program = Path(c.row.rank(r), "build_workload", "workload_program.json")
        program.unlink() if left is None else program.write_text(json.dumps(left))
    assert c.stops() == 12 and c.ran == [] and c.asked == []
    assert _stages(c.row) == [f"commit FAIL T GP-01 workload digest absent for rank {rank}: Build first"]


def test_the_manifest_beside_the_build_is_the_commits_when_it_is_of_record(tmp_path, monkeypatch):
    c = _Commit(tmp_path, monkeypatch)
    c.row.commit()
    assert c.asked == [(c.row.d, c.row.ck_)] and [ran[0] for ran in c.ran] == ["commit.tp_commit"]
    assert c.said()[:2] == ["required manifest of record: the one beside the Build", f"manifest rc=0: {MANIFEST_LINE}"]


def test_a_config_runs_complete_manifest_is_its_own_and_one_that_is_not_complete_is_asked_about(tmp_path, monkeypatch):
    """A config run word-checked the manifest of its own Build, so a complete one is not asked about; one that is not complete is."""
    c = _Commit(tmp_path, monkeypatch, "--config-run", "1")
    monkeypatch.setattr(RC, "commit_verdict", lambda row, rc, wall: None)
    c.row.commit()
    assert c.asked == [] and c.said()[0] == "required manifest of record: the one beside the Build (config run)"

    c = _Commit(tmp_path / "incomplete", monkeypatch, "--config-run", "1", of_record="absent or not complete", rebuilt=MANIFEST)
    monkeypatch.setattr(RC, "commit_verdict", lambda row, rc, wall: None)
    c.manifest({**MANIFEST, "complete": False})
    c.row.commit()
    assert c.asked == [(c.row.d, c.row.ck_)] and [ran[0] for ran in c.ran] == ["commit.required_manifest", "commit.tp_commit"]


def test_a_manifest_that_is_not_of_record_is_built_again_before_the_commit(tmp_path, monkeypatch):
    why = "disagrees with this Commit's taps on the norm_scales policy"
    c = _Commit(tmp_path, monkeypatch, of_record=why, rebuilt={**MANIFEST, "manifest_digest": "cd" * 32})
    c.row.commit()
    assert [ran[0] for ran in c.ran] == ["commit.required_manifest", "commit.tp_commit"]
    assert c.said()[:2] == [f"manifest.json beside the Build is not of record ({why}): rebuilding it",
                            f"manifest rc=0: {MANIFEST_LINE.replace('ab' * 8, 'cd' * 8)}"]
    assert _stages(c.row)[-1].startswith("commit PASS T rc=0 ")


@pytest.mark.parametrize("of_record, rebuild_rc, left, said", [
    ("absent or not complete", 4, "absent", "rc=4): no manifest.json"),                          # the rebuild failed and left none
    ("absent or not complete", 4, MANIFEST, f"rc=4): {MANIFEST_LINE}"),                          # it failed over the one that was there
    ("absent or not complete", 0, {**MANIFEST, "complete": False}, f"rc=0): {MANIFEST_LINE.replace('True', 'False')}"),
    ("", 0, {**MANIFEST, "complete": False}, f"rc=0): {MANIFEST_LINE.replace('True', 'False')}"),   # of record by the reader, and not complete
    ("", 0, {**MANIFEST, "complete": "yes"}, f"rc=0): {MANIFEST_LINE.replace('True', 'yes')}"),     # complete is the word true, nothing else
    ("", 0, "not JSON", "rc=0): no manifest.json"),
])
def test_a_manifest_that_does_not_come_out_complete_stops_the_commit_and_the_row(tmp_path, monkeypatch, of_record, rebuild_rc, left, said):
    c = _Commit(tmp_path, monkeypatch, of_record=of_record, rebuild_rc=rebuild_rc)
    manifest = Path(c.row.d, "manifest.json")
    if left == "absent":
        manifest.unlink()
    else:
        manifest.write_text(left if left == "not JSON" else json.dumps(left))
    assert c.stops() == 13 and [ran[0] for ran in c.ran] == (["commit.required_manifest"] if of_record else [])
    assert c.said()[-1] == f"manifest {said.replace('):', ':')}"
    assert _stages(c.row) == [f"commit FAIL T required manifest not complete ({said} (manifest.log)"]
    assert _stage(c.row) == [("start", None), ("end", False)]
    assert [(e["ev"], e["ok"], e["stopped_at"]) for e in _events(c.row, "row")] == [("end", False, "commit")]


@pytest.mark.parametrize("family, tap, script", [("norm_scales", "norm_tap", "pod_norm_tap.sh"), ("router_softmax", "router_tap", "pod_router_tap.sh")])
def test_a_tap_the_manifest_names_whose_build_is_not_on_the_host_stops_the_commit(tmp_path, monkeypatch, family, tap, script):
    """The manifest names a tap's family and the row runs the tap, so its ranks need the tap's build: without it the Commit stops by
    name; with the tap off (for an A/B) the build is not asked for."""
    named = {**MANIFEST, "identities": [{"family": "instance_outputs"}, {"family": family}]}
    c = _Commit(tmp_path, monkeypatch)
    c.manifest(named)
    so = str(tmp_path / "absent" / f"verity_{tap}.so")
    c.row.env = dataclasses.replace(c.row.env, **{f"{tap}_so": so})
    why = f"{tap.upper()}=1 and the manifest names its family, but the tap .so is missing: {so} (ops/{script}, run by ops/pod_bootstrap.sh)"
    assert c.stops() == 12 and c.ran == []
    assert _stages(c.row) == [f"commit FAIL T {why}"]
    assert _stage(c.row) == [("start", None)] and not _events(c.row, "row")

    off = _Commit(tmp_path / "off", monkeypatch, f"--{tap.replace('_', '-')}", "0")
    off.manifest(named)
    off.row.env = dataclasses.replace(off.row.env, **{f"{tap}_so": so})
    off.row.commit()
    assert [ran[0] for ran in off.ran] == ["commit.tp_commit"] and "taps: none (on, and the manifest names their families); guarded max on" in off.said()


# ---------------------------------------------------------------------------------------------------------------- the GPU process

@pytest.mark.parametrize("argv, staged, limit, said", [
    ((), ("256", "8", "host", "0"), 7200, "window_mb=256 slots=8 retain=host; weights hash device"),
    (("--window-mb", "64", "--window-slots", "2", "--retain", "device", "--commit-timeout", "90"), ("64", "2", "device", "0"), 90,
     "window_mb=64 slots=2 retain=device; weights hash device"),
    (("--config-run", "1"), ("256", "8", "host", "1"), 7200, "window_mb=256 slots=8 retain=host bounded; weights hash device"),
])
def test_tp_commit_runs_under_the_rows_staging_and_the_row_says_what_it_staged(tmp_path, monkeypatch, argv, staged, limit, said):
    c = _Commit(tmp_path, monkeypatch, *argv)
    monkeypatch.setattr(RC, "commit_verdict", lambda row, rc, wall: None)
    c.row.commit()
    ((stage, to, lf, cmd, env),) = c.ran
    assert (stage, to, lf, cmd) == ("commit.tp_commit", limit, f"{c.row.d}/commit.log", c.row.commit_cmd([]))
    assert tuple(env[k] for k in ("VERITY_WINDOW_MB", "VERITY_WINDOW_SLOTS", "VERITY_RETAIN", "VERITY_STAGING_BOUNDED")) == staged
    assert env[GL.WEIGHTS_HASH] == "device" and f"commit staging: {said}" in c.said()


def test_a_commit_names_the_taps_it_runs_and_hands_their_sources_to_tp_commit(tmp_path, monkeypatch):
    c = _Commit(tmp_path, monkeypatch, "--guarded-max-tap", "0")
    c.manifest({**MANIFEST, "identities": [{"family": "norm_scales"}, {"family": "vocab_range"}]})
    so = tmp_path / "verity_norm_tap.so"
    so.touch()
    c.row.env = dataclasses.replace(c.row.env, norm_tap_so=str(so))
    c.row.commit()
    cmd = c.ran[0][3]
    assert "taps: norm_tap, vocab_tap (on, and the manifest names their families); guarded max off" in c.said()
    assert cmd == c.row.commit_cmd(["norm_tap", "vocab_tap"]) and cmd[-3:] == ["--norm-tap-so", str(so), "--vocab-range"]


# ---------------------------------------------------------------------------------------------------------------- the verdict

def _verdict(t: TpRow) -> tuple[str, str]:
    """The row's ``commit`` line of ``stages.txt`` without its wall time, which is the driver's clock's."""
    (line,) = [ln for ln in _stages(t) if ln.startswith("commit ")]
    assert re.search(r" wall=\d+s ", line), line
    status, rest = line.split(" ", 2)[1:]
    return status, re.sub(r" wall=\d+s ", " ", rest)


def test_a_commit_that_exits_zero_with_a_passing_summary_passes_with_the_line_its_log_decided_by(tmp_path, monkeypatch):
    log = ("[tp2commit] FAIL weights pin of rank 1 (first pair)", "[tp2commit] PASS pair 0", PASSED, "[tp2commit] delta wall 1.2%")
    c = _Commit(tmp_path, monkeypatch, log=log)
    c.row.commit()
    assert _verdict(c.row) == ("PASS", f"T rc=0 {SUMMARY_LINE} {PASSED}")
    assert _stage(c.row) == [("start", None), ("end", "PASS")] and not _events(c.row, "row")
    assert _log(c.row)[-4:] == [SAID + "commit staging: window_mb=256 slots=8 retain=host; weights hash device", *log[:3]]


@pytest.mark.parametrize("rc, summary, said", [
    (1, SUMMARY, SUMMARY_LINE),                                               # a passing summary of a process that did not exit zero
    (0, {**SUMMARY, "pass": False}, SUMMARY_LINE.replace("pass True", "pass False")),
    (0, {**SUMMARY, "pass": None}, SUMMARY_LINE.replace("pass True", "pass None")),
    (0, {**SUMMARY, "pass": "true"}, SUMMARY_LINE.replace("pass True", "pass true")),
    (0, None, "no commit/summary.json"),
    (137, None, "no commit/summary.json"),
])
def test_a_commit_that_did_not_exit_zero_or_whose_summary_does_not_say_pass_fails_and_stops_the_row(tmp_path, monkeypatch, rc, summary, said):
    last = "[tp2commit] FAIL sampled replay: 3 of 64 units differ"
    c = _Commit(tmp_path, monkeypatch, rc=rc, summary=summary, log=(PASSED, last))
    assert c.stops() == 12
    assert _verdict(c.row) == ("FAIL", f"T rc={rc} {said} {last}")
    assert c.said()[-1] == "COMMIT FAIL -> stop" and _stage(c.row) == [("start", None), ("end", "FAIL")]
    assert [(e["ev"], e["ok"], e["stopped_at"]) for e in _events(c.row, "row")] == [("end", False, "commit")]


def test_a_commit_whose_log_decided_nothing_has_a_verdict_of_its_exit_and_summary_alone(tmp_path, monkeypatch):
    c = _Commit(tmp_path, monkeypatch, log=("Traceback (most recent call last):", "[tp2commit] coverage 12 of 12"), summary=None, rc=1)
    assert c.stops() == 12
    (line,) = [ln for ln in _stages(c.row) if ln.startswith("commit ")]
    assert re.fullmatch(r"commit FAIL T rc=1 wall=\d+s no commit/summary\.json ", line)


def test_the_row_log_keeps_the_last_sixteen_lines_tp_commit_tagged_each_cut_to_400_characters(tmp_path, monkeypatch):
    """Of ``commit.log`` the row keeps the lines ``tp-commit`` tagged with a word the row reads a decision from, the last sixteen, each
    cut to 400 characters; the verdict quotes the last PASS or FAIL among all of them, cut to 200."""
    kept = [f"[tp2commit] {key} {i}" for i, key in enumerate(
        ("PASS", "FAIL", "sampled replay", "match oracle", "weights pin", "cross-rank collectives", "fold-Match", "C2 population", "coverage"))]
    decided = "[tp2commit] PASS " + "y" * 300
    long = "[tp2commit]    sampled replay " + "x" * 500                      # spaces after the tag are passed over
    other = ["[tp2commit] delta wall 1.2%", "[tp2] PASS rank 0", " [tp2commit] PASS indented", "[tp2commit]PASS no space", "PASS", ""]
    log = [*other, *kept, *other, *kept, decided, long, *other]
    c = _Commit(tmp_path, monkeypatch, log=tuple(log))
    c.row.commit()
    tagged = [*kept, *kept, decided, long]
    wrote = _log(c.row)
    at = wrote.index(tagged[-16])
    assert wrote[at:at + 16] == [ln[:400] for ln in tagged[-16:]] and len(wrote[at + 15]) == 400
    assert wrote[at - 1].endswith("weights hash device") and len(wrote) == at + 16
    status, rest = _verdict(c.row)
    assert status == "PASS" and rest == f"T rc=0 {SUMMARY_LINE} {decided[:200]}"


def test_the_rows_name_and_stamp_are_on_the_lines_the_row_says_and_not_on_the_ones_it_copies(tmp_path, monkeypatch):
    c = _Commit(tmp_path, monkeypatch)
    c.row.commit()
    assert _log(c.row)[-1] == PASSED and all(ln.startswith(SAID) for ln in _log(c.row)[:-1])
