"""A single-GPU row's Build and Commit stages with every stage command stood in (`verity_vllm.pipeline.row_stages`): which derives
run and what their exits make of the Build, the Build's ends that are not a PASS, the Commit's command line and what it is armed
with, its refusals by name, the weights of record, the manifest check and the verdict that decides the stage.  CPU only: no stage
process starts."""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.pipeline.test_row import (
    CURRENT,
    _derive_row,
    _match_record,
    _row,
    _tap_manifest,
    _timeline,
)
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline import row_stages as S
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.row_stages import SingleRow


def _stages(r: SingleRow) -> list[tuple[str, str, str]]:
    """The row's stage lines: (stage, status, detail)."""
    return [(ln.split(" ", 3)[0], ln.split(" ", 3)[1], ln.split(" ", 3)[3]) for ln in Path(r.d, "stages.txt").read_text().splitlines()]


def _events(r: SingleRow, *keys: str) -> list[tuple]:
    """The row's timeline: (stage, event, then the value of each of `keys`, None where the event has none)."""
    return [(e["stage"], e["ev"], *(e.get(k) for k in keys)) for e in _timeline(r)]


def _log(r: SingleRow) -> list[str]:
    """The row log's lines without their stamp and row id."""
    return [ln.split("] ", 1)[-1] for ln in Path(r.d, "row.log").read_text().splitlines()]


def _children(monkeypatch, rc: list[int], said: str = "", leaves: str = "") -> list[tuple[list[str], dict[str, Any]]]:
    """Stand in for the commands a stage runs to their end: each exits with the next of `rc` (its last from then on), writes `said`
    and, when it exits 0, leaves the file `leaves` -> each one's command line and how it was started (a stream that is a file: its
    name)."""
    ran: list[tuple[list[str], dict[str, Any]]] = []

    def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
        ran.append((cmd, {k: getattr(v, "name", v) for k, v in kw.items()}))
        if hasattr(kw["stdout"], "write"):
            kw["stdout"].write(said)
        code = rc[min(len(ran), len(rc)) - 1]
        if leaves and code == 0:
            Path(leaves).write_text("{}")
        return subprocess.CompletedProcess(cmd, code, stdout=said)
    monkeypatch.setattr(S.subprocess, "run", run)
    return ran


# ---------------------------------------------------------------------------------------------------------------- Build: derives

def test_a_summary_that_cannot_be_written_fails_its_stage_and_is_named_in_the_row_log(tmp_path):
    r = _row(tmp_path)

    def summary(d: str, rc: int) -> str:
        raise KeyError("request")
    assert r.guarded("build summary", summary, r.d, 1) == "FAIL" and _log(r) == ["build summary failed: KeyError: 'request'"]
    assert r.guarded("build summary", lambda d: "PASS", r.d) == "PASS" and len(_log(r)) == 1


@pytest.mark.parametrize("argv, wide", [(("--build-jobs", "1"), False), ((), True)])
def test_a_shape_already_derived_is_not_derived_again_and_one_whose_launch_context_does_not_read_fails_the_request(tmp_path, monkeypatch,
                                                                                                                 argv, wide):
    """One derive after another and several at once alike: a request shape whose dir holds a derive that succeeded is kept, one whose
    derive failed runs again, and a shape with no launch context is not derived and makes the request derives' exit 1."""
    r, ran = _derive_row(tmp_path, monkeypatch, *argv, cpus=4)
    d = r.d
    kept, again, unread, *rest = R.extra_shapes(r.path(r.wl), r.LP, r.T)
    for (lp, t), ok in ((kept, True), (again, False)):
        os.makedirs(f"{d}/build_request_LP{lp}_T{t}")
        Path(f"{d}/build_request_LP{lp}_T{t}/result.json").write_text(json.dumps({"ok": ok}))
    monkeypatch.setattr(r, "launch_args", lambda lp, t: None if (lp, t) == unread else [])
    Path(f"{d}/build_step.log.rc").write_text("7\n")                               # an earlier attempt's exit
    assert r.derive_all(60) == (0, 1)
    assert sorted(ran) == sorted(["build.derive_step", "build.derive_request", *(f"build.derive_request_LP{lp}_T{t}" for lp, t in (again, *rest))])
    if wide:
        assert Path(f"{d}/build_step.log.rc").read_text() == "0\n"
        assert re.fullmatch(rf"Build derives ran 4-wide: step rc=0 request rc=1 shapes={len(rest) + 2} wall=\d+s", _log(r)[-1])


def test_a_row_signalled_while_its_derives_ran_side_by_side_stops_once_they_have_ended(tmp_path, monkeypatch):
    """A derive a worker thread waited for leaves the stop to the row (`run_to`): it is taken here, with the step derive's exit."""
    r, ran = _derive_row(tmp_path, monkeypatch, cpus=4)
    r.signalled = "INT"
    with pytest.raises(Stop) as e:
        r.derive_all(60)
    assert e.value.rc == 130 and len(ran) == 2 + len(R.extra_shapes(r.path(r.wl), r.LP, r.T))
    assert _log(r) == ["row cancelled by SIGINT; stage ended rc=0 -> stop"]
    assert _events(r, "signal", "stopped_at")[-2:] == [("row", "cancelled", "INT", None), ("row", "end", None, "cancel")]


def test_the_compose_is_given_every_request_program_dir_of_the_build_and_no_file_named_like_one(tmp_path):
    r = _row(tmp_path)
    d = r.d
    for name in ("build_request_LP9_T1", "build_request_LP10_T2", "build_request_LP9_T1_C4"):
        os.makedirs(f"{d}/{name}")
    Path(d, "build_request_LP9_T1.log").write_text("$ a derive's log\n")
    assert r.derived_dirs() == [x for name in ("build_request", "build_request_LP10_T2", "build_request_LP9_T1", "build_request_LP9_T1_C4")
                                for x in ("--derived", f"{d}/{name}")]


# ---------------------------------------------------------------------------------------------------------------- Build: its ends

def _build_row(tmp_path: Path, monkeypatch, *, family: tuple[int, str] = (0, "TARGET-FAMILY-OK: 8.9"), manifest: int | None = 0) -> tuple[SingleRow, dict]:
    """A B=1 row at its Build whose derives and summary pass and whose compose writes the workload Program; the Build's target-family
    check answers `family` and its manifest build exits `manifest` -> the row and what each stand-in was asked."""
    r = _row(tmp_path)
    r.B, r.LP, r.T, r.MML, r.ck = 1, 8, 3, 288, "/ck"
    seen: dict[str, list] = {"compose": [], "family": [], "manifest": [], "boundaries": []}

    def derive_all(to: int) -> tuple[int, int]:
        os.makedirs(f"{r.d}/build_request", exist_ok=True)
        Path(f"{r.d}/build_request/instances.json.gz").write_bytes(b"")
        return 0, 0

    def run_tl(stage: str, to: int, lf: str, cmd: list[str], *extra: tuple) -> int:
        seen["compose"].append((stage, cmd))
        os.makedirs(f"{r.d}/build_workload", exist_ok=True)
        Path(f"{r.d}/build_workload/workload_program.json").write_text(json.dumps({"workload_digest": "w" * 64}))
        return 0

    def target_family(*extra: str, out: str) -> tuple[int, str]:
        seen["family"].append((extra, out))
        return family

    def required_manifest(stage: str, digest: Any = None) -> int | None:
        seen["manifest"].append(stage)
        return manifest
    for name, fn in (("derive_all", derive_all), ("run_tl", run_tl), ("target_family", target_family), ("required_manifest", required_manifest),
                     ("guarded", lambda what, fn, *a: "PASS"), ("call_boundaries", lambda to, cmd=None: seen["boundaries"].append(cmd))):
        monkeypatch.setattr(r, name, fn)
    monkeypatch.setattr(R, "build_line", lambda d: ("d" * 16, "m" * 16))
    return r, seen


def test_a_build_whose_programs_are_of_another_family_than_the_rows_fails_and_a_waiver_is_recorded_as_one(tmp_path, monkeypatch):
    """After the compose the Build's summary is held to the row's family with no device read; a mismatch fails the Build (exit 10)
    before any manifest is built, and a waived one passes with the waiver beside the Build's line."""
    refusal = "PRECHECK_FAIL_TARGET_FAMILY: Build target 9.0 != row family 8.9"
    r, seen = _build_row(tmp_path, monkeypatch, family=(3, refusal))
    d = r.d
    with pytest.raises(Stop) as e:
        r.build()
    assert e.value.rc == 10 and seen["manifest"] == [] and seen["boundaries"] == []
    assert seen["compose"] == [("build.global_program", r.cli("global-program", "--workload", r.wl, "--derived", f"{d}/build_request", "--out",
                                                              f"{d}/build_workload", "--checkpoint-dir", "/ck", py=r.py312))]
    assert seen["family"] == [(("--no-device", "--build-summary", f"{d}/build_summary.json"), f"{d}/target_family.build.json")]
    (target, t_status, t_detail), (build, b_status, b_detail) = _stages(r)
    assert (target, t_status, t_detail) == ("build-target", "FAIL", refusal) and (build, b_status) == ("build", "FAIL")
    assert re.fullmatch(rf"rc=0/0 wall=\d+s digest={'d' * 16} workload_digest={'w' * 16} manifest={'m' * 16}", b_detail)
    assert _log(r)[-1] == "BUILD FAIL -> stop"
    assert _events(r, "ok", "stopped_at")[-2:] == [("stage.build", "end", "FAIL", None), ("row", "end", False, "build")]

    waiver = "TARGET-FAMILY-WAIVED (canary r16): Build target 9.0 != row family 8.9"
    r, seen = _build_row(tmp_path / "waived", monkeypatch, family=(0, waiver))
    r.build()
    assert [(stage, status) for stage, status, _ in _stages(r)] == [("build-target", "WAIVED"), ("build", "PASS")] and _stages(r)[0][2] == waiver
    assert seen["manifest"] == ["build"] and seen["boundaries"] == [None]
    r, seen = _build_row(tmp_path / "held", monkeypatch)
    r.build()
    assert [(stage, status) for stage, status, _ in _stages(r)] == [("build", "PASS")]


@pytest.mark.parametrize("mrc, rc, lines", [
    (3, 3, [("build", "FAIL", "PRECHECK_FAIL_REGISTRY_UNAVAILABLE (environment: core verity package not importable; see {d}/manifest.log)")]),
    (1, 13, [("build-manifest", "FAIL", "required-value manifest not built (manifest.log)"), ("build", "FAIL", None)]),
    (None, 13, [("build-manifest", "FAIL", "required-value manifest not built (manifest.log)"), ("build", "FAIL", None)]),
])
def test_a_build_whose_manifest_is_not_built_stops_at_13_and_one_without_the_core_registry_at_3_by_name(tmp_path, monkeypatch, mrc, rc, lines):
    r, seen = _build_row(tmp_path, monkeypatch, manifest=mrc)
    with pytest.raises(Stop) as e:
        r.build()
    assert e.value.rc == rc and seen["manifest"] == ["build"] and seen["boundaries"] == []
    got = _stages(r)
    assert [(stage, status) for stage, status, _ in got] == [(stage, status) for stage, status, _ in lines]
    assert all(detail is None or g[2] == detail.format(d=r.d) for g, (_, _, detail) in zip(got, lines))
    assert _events(r, "ok", "stopped_at")[-2:] == [("stage.build", "end", False if mrc == 3 else "FAIL", None), ("row", "end", False, "build")]


# ---------------------------------------------------------------------------------------------------------------- Commit: its command

def _commit_row(tmp_path: Path, *argv: str, B: int = 1) -> SingleRow:
    """A row at its Commit whose tap build is on disk."""
    r = _row(tmp_path, *argv)
    Path(r.d, "tap.so").write_bytes(b"")
    r.hidden_so, r.B, r.I, r.O, r.ck, r.snap_def, r.snap_steps = f"{r.d}/tap.so", B, 256, 32, "/ck", "0,1", None
    return r


def _flags(cmd: list[str], name: str) -> list[str]:
    return [cmd[i + 1] for i, x in enumerate(cmd) if x == name]


def test_the_commit_takes_the_callers_engine_args_and_classes_and_the_declared_target_only_with_the_rows_protocols(tmp_path):
    r = _commit_row(tmp_path, "--extra-engine-args", "enforce_eager=True max_num_batched_tokens=4096", "--required-classes", "weights,activations")
    r.tgt_args = ["--target", '{"protocols": ["pouw"]}']
    cmd = r.commit_cmd([])
    assert _flags(cmd, "--engine-arg") == [f"gpu_memory_utilization={r.gpu_util}", "enforce_eager=True", "max_num_batched_tokens=4096"]
    assert _flags(cmd, "--required-classes") == ["weights,activations"] and "--target" not in cmd
    r.protocols = ("pouw",)
    assert _flags(r.commit_cmd([]), "--target") == ['{"protocols": ["pouw"]}']
    cmd = _commit_row(tmp_path / "plain").commit_cmd([])
    assert _flags(cmd, "--engine-arg") == [f"gpu_memory_utilization={r.gpu_util}"] and "--required-classes" not in cmd


def test_a_commit_without_its_tap_build_is_refused_at_12_by_name_before_it_prepares(tmp_path, monkeypatch):
    """The hidden-state tap's build is looked for first; then, of the taps the manifest names, each one's own."""
    r = _commit_row(tmp_path)
    r.hidden_so = f"{r.d}/absent.so"
    with pytest.raises(Stop) as e:
        r.commit_prepare()
    assert e.value.rc == 12 and _stages(r) == [("commit", "FAIL", f"hidden .so missing: {r.d}/absent.so")]
    assert _events(r, "ok", "stopped_at") == [("stage.commit", "start", None, None), ("stage.commit", "end", False, None), ("row", "end", False, "commit")]

    r = _commit_row(tmp_path / "norm")
    r.env = dataclasses.replace(r.env, norm_tap_so=str(tmp_path / "absent" / "verity_norm_tap.so"))
    man = _tap_manifest(tmp_path / "manifest.json", "norm_scales")
    monkeypatch.setattr(r, "manifest_of_record", lambda stage="commit": man)
    with pytest.raises(Stop) as e:
        r.commit_prepare()
    (stage, status, detail), = _stages(r)
    assert e.value.rc == 12 and (stage, status) == ("commit", "FAIL")
    assert detail == (f"NORM_TAP=1 and the manifest names its family, but the tap .so is missing: {r.env.norm_tap_so} (ops/pod_norm_tap.sh, "
                      "run by ops/pod_bootstrap.sh)")
    assert _events(r, "ok", "stopped_at")[-2:] == [("stage.commit", "end", False, None), ("row", "end", False, "commit")]


@pytest.mark.parametrize("mrc, rc, detail", [
    (3, 3, "PRECHECK_FAIL_REGISTRY_UNAVAILABLE (environment: core verity package not importable; see {d}/manifest.log)"),
    (1, 13, "required-value manifest not built (B=1; manifest.log)"),
    (None, 13, "required-value manifest not built (B=1; manifest.log)"),
])
@pytest.mark.parametrize("stage", ["commit", "plan"])
def test_a_manifest_of_record_that_is_not_rebuilt_stops_the_stage_that_asked_for_it_by_name(tmp_path, monkeypatch, mrc, rc, detail, stage):
    r = _commit_row(tmp_path)
    os.makedirs(f"{r.d}/build_request")
    Path(r.d, "build_request", "instances.json.gz").write_bytes(b"")
    asked: list[str] = []

    def required_manifest(st: str) -> int | None:
        asked.append(st)
        return mrc
    monkeypatch.setattr(r, "required_manifest", required_manifest)
    with pytest.raises(Stop) as e:
        r.manifest_of_record(stage)
    assert e.value.rc == rc and asked == [stage] and _stages(r) == [(stage, "FAIL", detail.format(d=r.d))]
    assert _events(r, "ok", "stopped_at") == [(f"stage.{stage}", "end", False, None), ("row", "end", False, stage)]


def test_the_commit_is_armed_with_the_match_record_beside_it_and_says_so_or_what_it_lacks(tmp_path, monkeypatch):
    """A Match record of this attempt arms the live oracle compare, named with the steps it snapshot (the record's, else the row's); a
    batch without one is warned of, and a single request whose record names a served shape short of the row's is said to keep the
    row's Program at the Commit, which no batch is asked."""
    served, asked = [""], []

    def served_shape(d: str) -> str:
        asked.append(d)
        return served[0]
    monkeypatch.setattr(R, "served_shape", served_shape)

    def prepared(name: str, B: int = 1) -> tuple[SingleRow, int, list[str], str]:
        r = _commit_row(tmp_path / name, B=B)
        monkeypatch.setattr(r, "manifest_of_record", lambda stage="commit": "")
        return (r, *r.commit_prepare())
    r, to, cmd, man = prepared("bare")
    assert (to, man) == (2400, "") and "--match-dir" not in cmd and "--required-manifest" not in cmd and "--program-digest" not in cmd
    assert asked == [r.d] and not any("oracle-compare" in ln or "EOS-stopped" in ln for ln in _log(r))
    assert _events(r) == [("stage.commit", "start"), ("commit.prepare", "start"), ("commit.prepare", "end")]

    r = _commit_row(tmp_path / "early")
    monkeypatch.setattr(r, "manifest_of_record", lambda stage="commit": "")
    served[0] = "LP=8 T_r=2 of 3"
    _match_record(Path(r.d))
    _to, cmd, _man = r.commit_prepare()
    assert _flags(cmd, "--match-dir") == [f"{r.d}/match"] and asked[1:] == [r.d]
    assert any(ln.startswith("B=1 EOS-stopped request (LP=8 T_r=2 of 3, match_shape.json): Program of record at Commit = the ") for ln in _log(r))
    armed = f"C2 live oracle-compare armed: --match-dir {r.d}/match (Match record of this attempt; snapshot steps "
    assert armed + "0,1 (env; run.json unreadable))" in _log(r)
    Path(r.d, "match", "run.json").write_text(json.dumps({"snapshot_steps": [0, 1, 2]}))
    r.commit_prepare()
    assert armed + "0,1,2)" in _log(r)

    r, to, cmd, _man = prepared("batch", B=8)
    assert to == 4800 and "--match-dir" not in cmd and r.d not in asked and not any("EOS-stopped" in ln for ln in _log(r))
    assert (f"WARNING B=8: no {r.d}/match/instances.jsonl: live C2 oracle-compare NOT armed -- request attribution of decode-step identities "
            "unverified") in _log(r)


# ---------------------------------------------------------------------------------------------------------------- Commit: weights of record

def test_the_weights_of_record_are_derived_beside_the_program_and_a_record_of_it_is_reused(tmp_path, monkeypatch):
    """No record beside the request Program and a checkpoint dir on disk: `weights-of-record` derives one, and the Commit takes it;
    a record that is this Program's is taken as it is, with no derive."""
    r = _commit_row(tmp_path)
    d, ck = r.d, tmp_path / "ck"
    ck.mkdir()
    r.ck = str(ck)
    os.makedirs(f"{d}/build_request")
    wor = f"{d}/build_request/weights_of_record.json"
    ran = _children(monkeypatch, [0], "$ weights-of-record\n163 of 163 fields pinned\n", leaves=wor)
    assert r.weights_of_record() == ["--weights-of-record", wor]
    assert ran == [(r.cli("weights-of-record", "--program-dir", f"{d}/build_request", "--checkpoint-dir", str(ck), "--out", wor),
                    {"stdout": f"{d}/weights_of_record.log", "stderr": subprocess.STDOUT, "cwd": r.tree, "env": r.child_env})]
    assert _log(r) == ["weights of record: 163 of 163 fields pinned"]
    assert _events(r, "rc") == [("commit.weights_of_record", "start", None), ("commit.weights_of_record", "end", 0)]

    Path(wor).write_text(json.dumps(CURRENT))
    assert r.weights_of_record("plan") == ["--weights-of-record", wor] and len(ran) == 1 and len(_log(r)) == 1
    assert _events(r, "reused")[-1] == ("plan.weights_of_record", "mark", True)


def test_a_batchs_weights_of_record_are_derived_from_the_manifests_first_component(tmp_path, monkeypatch):
    """B>=2 with a manifest: the record is derived beside the component Program the manifest names first, which the row log names
    unless it is the Build's request Program itself."""
    r = _commit_row(tmp_path, B=2)
    d, ck = r.d, tmp_path / "ck"
    ck.mkdir()
    r.ck = str(ck)
    ran = _children(monkeypatch, [1], "$ weights-of-record\nno safetensors under the checkpoint dir\n")
    for name, lp, t in (("build_request_LP9_T2", 9, 2), ("build_request", 9, 3)):
        os.makedirs(f"{d}/{name}")
        Path(d, name, "result.json").write_text(json.dumps({"ok": True, "LP": lp, "T": t}))
    Path(d, "manifest.json").write_text(json.dumps({"components": {"r0": {"LP": 9, "T": 2}}}))
    assert r.weights_of_record() == ["--checkpoint-dir", str(ck)]
    assert _flags(ran[0][0], "--program-dir") == [f"{d}/build_request_LP9_T2"]
    assert _log(r) == ["weights of record: B=2 -> derived from the component Program build_request_LP9_T2 (of-record set)",
                       "WARNING weights of record not derived (see weights_of_record.log): no safetensors under the checkpoint dir"]
    assert _events(r, "rc")[-1] == ("commit.weights_of_record", "end", 1)

    Path(d, "manifest.json").write_text(json.dumps({"components": {"r0": {"LP": 9, "T": 3}}}))
    r.weights_of_record()
    assert _flags(ran[1][0], "--program-dir") == [f"{d}/build_request"] and len(_log(r)) == 3


# ---------------------------------------------------------------------------------------------------------------- Commit: its verdict

def test_the_manifest_check_rebuilds_from_the_request_program_or_the_batchs_programs_and_says_how_it_went(tmp_path, monkeypatch):
    r = _commit_row(tmp_path)
    d = r.d
    ran = _children(monkeypatch, [0, 2], "$ manifest-verify\n3 identities differ\n")
    how = {"stdout": f"{d}/manifest_verify.log", "stderr": subprocess.STDOUT, "cwd": r.tree, "env": r.child_env}
    verify = r.cli("manifest-verify", "--manifest", "/m/manifest.json", "--out", f"{d}/commit/manifest_verify.json")
    policies = ["--norm-scales", "--router-softmax", "--vocab-range"]
    r.manifest_verify("/m/manifest.json")
    assert ran == [([*verify, "--program", f"{d}/build_request", "--workload", r.wl, *policies], how)]
    assert _log(r)[-1].startswith("manifest of record VERIFIED (manifest_verify: rebuilt by the query path")
    assert _events(r, "rc") == [("validate.manifest_verify", "start", None), ("validate.manifest_verify", "end", 0)]

    r.B = 2
    r.manifest_verify("/m/manifest.json")
    assert ran[1] == ([*verify, "--fixture", r.wl, "--programs-root", d, *policies], how)
    assert _log(r)[-1] == "manifest of record VERIFY FAILED: 3 identities differ" and _events(r, "rc")[-1] == ("validate.manifest_verify", "end", 1)
    os.makedirs(f"{d}/build_workload")
    Path(d, "build_workload", "workload_program.json").write_text(json.dumps({"workload_digest": "w" * 64}))
    r.manifest_verify("/m/manifest.json")
    assert ran[2][0] == [*verify, "--fixture", r.wl, "--programs-root", d, "--workload-digest", "w" * 64, *policies]


@pytest.mark.parametrize("said, outcome, status", [("PASS verdict.json: every check held\n", "PASS", "PASS"),
                                                   ("FAIL the instrumented run's tokens differ\n", "FAIL", "FAIL"),
                                                   ("INCOMPLETE the control arm did not run\n", "INCOMPLETE", "FAIL"), ("", "ERROR", "FAIL")])
def test_the_commit_stage_is_what_the_verdict_record_says_of_its_run(tmp_path, monkeypatch, said, outcome, status):
    """`verdict from-record` is given the Commit's exit and prints the outcome first on its line: PASS passes the stage, any other
    fails it, and a verdict that printed nothing is an ERROR.  The manifest the Commit ran over is checked after a Commit that exited
    0, and not after one that failed."""
    r = _commit_row(tmp_path)
    d = r.d
    gpu: list[tuple] = []
    verified: list[str] = []
    rck = 0 if status == "PASS" else 5

    def run_gpu(*a: Any) -> int:
        gpu.append(a)
        return rck
    monkeypatch.setattr(r, "commit_prepare", lambda: (60, ["verity-vllm", "commit"], "/m/manifest.json"))
    monkeypatch.setattr(r, "run_gpu", run_gpu)
    monkeypatch.setattr(r, "manifest_verify", verified.append)
    ran = _children(monkeypatch, [0], said)
    if status == "PASS":
        r.commit()
    else:
        with pytest.raises(Stop) as e:
            r.commit()
        assert e.value.rc == 12 and _log(r)[-1] == "COMMIT FAIL -> stop"
    assert gpu == [("commit.commit_delta", 60, f"{d}/commit.log", ["verity-vllm", "commit"])]
    assert verified == (["/m/manifest.json"] if status == "PASS" else [])
    assert ran == [(r.cli("verdict", "from-record", d, "--rc", str(rck)),
                    {"stdout": subprocess.PIPE, "stderr": f"{d}/row.log", "text": True, "cwd": r.tree, "env": r.child_env})]
    (stage, got, detail), = _stages(r)
    assert (stage, got) == ("commit", status)
    assert re.fullmatch(rf"rc={rck} wall=\d+s outcome={outcome}  manifest_verify=not-run", detail)
    assert f"verdict.json (vllm-verdict/v1): {outcome} -> Commit stage {status}" in Path(d, "row.log").read_text()
    events = _events(r, "outcome", "ok", "stopped_at")
    assert events[:3] == [("closure.verdict", "start", None, None, None), ("closure.verdict", "end", outcome, None, None),
                          ("stage.commit", "end", None, status, None)]
    assert events[3:] == ([] if status == "PASS" else [("row", "end", None, False, "commit")])
