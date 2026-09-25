"""`verity-vllm row`: the options a row runs with, the driver's stage processes and the records it decides from.

* a cancelled row takes its running stage (and the stage's forked workers) with it: GNU ``timeout`` moves the stage into
  its own process group, which never sees the runner's killpg, so the driver forwards TERM/INT/HUP to timeout's pid,
  waits for the stage to be gone, records the cancel and exits 128+signal;
* a Commit over a Match record whose account files are gone refuses to start (PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE,
  exit 14); the Match summary accepts an explicit ``--match-phase cpu`` resume over the preserved capture;
* the Match of record (``fast`` inside ``shared``) is the default and the caller can choose the audit checker;
* the weights of record beside a component are reused only when they are this Program's, of this tree's schema and
  complete; an unresolved checkpoint dir is named in row.log;
* the admission plan is advisory: it writes admission.json and one row.log line and never stops the row.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from verity_vllm.check.weights_of_record import SCHEMA, SCHEMA_V1
from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.row_stages import SingleRow

TREE = Path(__file__).resolve().parents[2]
WL = TREE / "workloads"
ROW = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"
_GNU_TIMEOUT = shutil.which("timeout") is not None and b"GNU" in subprocess.run(["timeout", "--version"], capture_output=True).stdout


def _opts(tmp: Path, *argv: str, environ: dict[str, str] | None = None, row: str = ROW):
    env = {k: v for k, v in os.environ.items()
           if k not in ("MATCH_IMPL", "MATCH_PIPELINE", "MATCH_PHASE", "VERITOR_REPO", "RESEARCH_RUN_DIR", "RESEARCH_RUN_ID")}
    env.update(environ or {})
    return cli.parse("row", ["run", row, "LLAMA32_1B", "unsloth/Llama-3.2-1B", "9535bd9b", "--sweep-dir", str(tmp), "--py", sys.executable,
                             *argv], env)


def _row(tmp: Path, *argv: str, **kw) -> SingleRow:
    o = _opts(tmp, *argv, **kw)
    r = SingleRow(row=o.row, role=o.role, repo=o.repo, rev=o.rev, stages=o.stages, env=o.env, gpu_util="0.5", build=o.build, match=o.match,
                  commit=o.commit)
    os.makedirs(r.d, exist_ok=True)
    r.child_env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TREE), os.environ.get("PYTHONPATH", "")])}
    return r


def _timeline(r: SingleRow) -> list[dict]:
    return [json.loads(ln) for ln in open(r.timeline)]


# ---------------------------------------------------------------------------------------------------------------- options

def test_the_match_of_record_is_the_default_and_the_caller_can_choose_the_audit(tmp_path):
    o = _opts(tmp_path)
    assert (o.match.match_impl, o.match.match_pipeline, o.match.match_phase) == ("fast", "shared", "all")
    o = _opts(tmp_path, environ={"MATCH_IMPL": "record", "MATCH_PIPELINE": "standalone"})
    assert (o.match.match_impl, o.match.match_pipeline) == ("record", "standalone")
    assert _opts(tmp_path, "--match-impl", "record").match.match_impl == "record"
    with pytest.raises(SystemExit):
        _opts(tmp_path, "--match-phase", "gpu")


def test_machine_paths_are_defaults_that_the_environment_and_flags_override(tmp_path):
    o = _opts(tmp_path, environ={"HF_HOME": "/elsewhere/hf"})
    assert o.env.hf_home == "/elsewhere/hf" and o.env.hot_root == cli.MACHINE["RunEnv.hot_root"] and o.env.sweep_dir == str(tmp_path)
    assert _opts(tmp_path, "--hf-home", "/flag/hf", environ={"HF_HOME": "/elsewhere/hf"}).env.hf_home == "/flag/hf"
    assert o.env.environ["HF_HOME"] == "/elsewhere/hf"


def test_the_stage_flags_of_every_row_subcommand_are_one_set(tmp_path):
    env = {"PY": sys.executable}
    st = cli.parse("row", ["stage", "commit", ROW, "R", "o/r", "rev", "--pairs", "1", "--commit-args", "--x 1", "--retain", "device"], env)
    assert (st.stage, st.row, st.commit.pairs, st.commit.extra_commit_args, st.commit.retain) == ("commit", ROW, 1, "--x 1", "device")
    ch = cli.parse("row", ["chain", "--stages", "build,match", ROW, "R", "o/r", "rev", "--pairs", "1"], env)
    assert ch.stages == "build,match" and ch.flags == ["--pairs", "1"]
    with pytest.raises(SystemExit):
        cli.parse("row", ["stage", "commit", ROW, "R", "o/r", "rev", "--no-such-flag", "1"], env)


def test_chain_refuses_its_own_flags_among_the_stage_flags(tmp_path):
    from verity_vllm.pipeline import row
    ch = cli.parse("row", ["chain", ROW, "R", "o/r", "rev", "--stages", "build"], {"PY": sys.executable})
    with pytest.raises(UsageError, match="--stages"):
        row.main(ch)


def test_the_commit_flags_the_row_arms_exist():
    from verity_vllm.check import weights_of_record
    from verity_vllm.pipeline import commit
    flags = {fl for f in commit.Options.__dataclass_fields__.values() for fl in f.metadata.get("flags", ())}
    assert {"--match-dir", "--program-dir", "--programs-root", "--weights-of-record", "--checkpoint-dir", "--required-manifest",
            "--program-digest"} <= flags
    assert "--supersedes" in {fl for f in weights_of_record.Options.__dataclass_fields__.values() for fl in f.metadata.get("flags", ())}


def test_a_foreign_veritor_repo_is_refused_by_name(tmp_path, capsys):
    r = _row(tmp_path, environ={"VERITOR_REPO": str(tmp_path)})
    with pytest.raises(Stop) as e:
        r.setup({})
    assert e.value.rc == 3 and "PRECHECK_FAIL_SOURCE_IDENTITY" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------------- cancel

# the stage: a python parent that forks a grandchild; both write their pids and sleep.  A row's fold / commit look like this.
STAGE = r'''
import os, sys, time
d = sys.argv[1]
open(os.path.join(d, "stage.pid"), "w").write(str(os.getpid()))
pid = os.fork()
if pid == 0:
    open(os.path.join(d, "grandchild.pid"), "w").write(str(os.getpid()))
    time.sleep(600)
    sys.exit(0)
time.sleep(600)
'''

HARNESS = r'''
import os, sys
from verity_vllm.pipeline import cli
from verity_vllm.pipeline.row_driver import Row, Stop
tmp, to, stage = sys.argv[1:4]
o = cli.parse("row", ["run", "r", "R", "o/r", "x", "--sweep-dir", tmp, "--py", sys.executable], os.environ)
row = Row(row="r", role="R", repo="o/r", rev="x", stages="build", env=o.env, proc="row_pod")
os.makedirs(row.d, exist_ok=True)
row.child_env = dict(os.environ)
row.install_signal_forwarding()
try:
    rc = row.run_to(int(to), row.d + "/stage.log", [sys.executable, stage, row.d])
except Stop as s:
    sys.exit(s.rc)
open(row.d + "/row.log", "a").write(f"run_to rc={rc}\n")
sys.exit(rc)
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def _wait_for(pred, seconds: float) -> bool:
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def _start(tmp: Path, timeout_s: int, stage_exit: str | None = None) -> tuple[subprocess.Popen, Path]:
    stage = tmp / "stage.py"
    stage.write_text(STAGE if stage_exit is None else f"import sys; sys.exit({stage_exit})")
    d = tmp / "r"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TREE), os.environ.get("PYTHONPATH", "")])}
    env.pop("VERITY_TIMELINE", None)
    # the research runner starts the row as a session leader and later killpg()s that group -- same here
    p = subprocess.Popen([sys.executable, "-c", HARNESS, str(tmp), str(timeout_s), str(stage)], start_new_session=True, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert _wait_for(lambda: (d / "grandchild.pid").exists() or (d / "row.log").exists() and "run_to rc" in (d / "row.log").read_text(), 20)
    return p, d


@pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")
@pytest.mark.parametrize("sig,rc", [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_killpg_on_the_row_takes_the_stage_and_its_grandchild(tmp_path, sig, rc):
    p, d = _start(tmp_path, 600)
    stage, grand = int((d / "stage.pid").read_text()), int((d / "grandchild.pid").read_text())
    assert _alive(stage) and _alive(grand)
    assert os.getpgid(stage) != p.pid, "GNU timeout puts the stage in its own process group -- the premise of the forwarding"
    os.killpg(p.pid, sig)
    assert p.wait(timeout=15) == rc
    assert _wait_for(lambda: not _alive(stage) and not _alive(grand), 10), "stage or its forked worker survived the cancel"
    name = signal.Signals(sig).name
    assert f"CANCELLED by {name}" in (d / "stage.log").read_text()
    row_log = (d / "row.log").read_text()
    assert f"{name} received: forwarding to the running stage" in row_log and f"row cancelled by {name}" in row_log
    tl = [json.loads(ln) for ln in open(d / "timeline.jsonl")]
    assert {"stage": "row", "ev": "cancelled", "signal": name[3:]}.items() <= tl[0].items()
    assert {"stage": "row", "ev": "end", "ok": False, "stopped_at": "cancel"}.items() <= tl[1].items()


@pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")
def test_the_expiry_path_still_kills_the_forked_worker(tmp_path):
    p, d = _start(tmp_path, 1)
    stage, grand = int((d / "stage.pid").read_text()), int((d / "grandchild.pid").read_text())
    assert p.wait(timeout=15) == 124
    assert _wait_for(lambda: not _alive(stage) and not _alive(grand), 10), "timeout expiry left the forked worker alive"
    log = (d / "stage.log").read_text()
    assert "TIMEOUT after 1s" in log and "CANCELLED" not in log


@pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")
def test_a_normal_exit_returns_the_stage_rc_unchanged(tmp_path):
    p, d = _start(tmp_path, 600, stage_exit="7")
    assert p.wait(timeout=15) == 7
    assert "run_to rc=7" in (d / "row.log").read_text() and "cancelled" not in (d / "row.log").read_text()
    assert not (d / "timeline.jsonl").exists()
    assert (d / "stage.log").read_text().startswith(f"$ {sys.executable} ")


# ---------------------------------------------------------------------------------------------------------------- Commit precheck

def _match_record(d: Path, missing: tuple[str, ...] = ()) -> None:
    (d / "match" / "capture").mkdir(parents=True)
    (d / "match" / "control").mkdir(parents=True)
    (d / "match" / "instances.jsonl").write_text("")
    for f in ("match/capture/tokens.json", "match/control/tokens.json", "global_match.json"):
        if f not in missing:
            (d / f).write_text("{}")


def test_a_complete_record_and_no_record_fall_through(tmp_path):
    r = _row(tmp_path / "a")
    _match_record(Path(r.d))
    r.match_record_precheck()
    _row(tmp_path / "b").match_record_precheck()


@pytest.mark.parametrize("missing", [("match/capture/tokens.json",),
                                     ("match/capture/tokens.json", "match/control/tokens.json", "global_match.json")])
def test_a_moved_capture_refuses_the_commit_by_name(tmp_path, missing):
    r = _row(tmp_path)
    d = Path(r.d)
    _match_record(d, missing)
    with pytest.raises(Stop) as e:
        r.match_record_precheck()
    assert e.value.rc == 14
    lacks = " ".join(missing)
    assert f"PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE: the Match record beside this attempt lacks {lacks}" in (d / "row.log").read_text()
    stages = (d / "stages.txt").read_text().split(" ", 3)
    assert stages[:2] == ["commit", "FAIL"] and stages[3].startswith("PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE") and "NOT run" in stages[3]
    end, row_end = _timeline(r)
    assert {"stage": "stage.commit", "ev": "end", "rc": 3, "fail_class": "PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE", "missing": lacks}.items() <= end.items()
    assert {"stage": "row", "ev": "end", "ok": False, "stopped_at": "commit"}.items() <= row_end.items()


# ---------------------------------------------------------------------------------------------------------------- Match summary

def _summary_row(tmp_path: Path, *, old: bool, stage_record: bool) -> Path:
    d = tmp_path / "row"
    (d / "match" / "capture").mkdir(parents=True)
    (d / "match" / "control").mkdir(parents=True)
    toks = {"requests": [{"request_id": "r0", "prompt_token_ids": [1, 2], "generated_token_ids": [3]}]}
    for p in ("capture", "control"):
        (d / "match" / p / "tokens.json").write_text(json.dumps(toks))
    (d / "match" / "capture" / "log.jsonl.gz").write_bytes(b"x")
    (d / "match" / "program.json").write_text("{}")
    (d / "match" / "instances.jsonl").write_text("")
    if old:
        t = time.time() - 3600
        os.utime(d / "match" / "capture" / "log.jsonl.gz", (t, t))
    if stage_record:
        (d / "match" / "stages.json").write_text(json.dumps({"stages": [{"stage": "capture", "returncode": 0, "status": "ok"},
                                                                        {"stage": "control", "returncode": 0, "status": "ok"},
                                                                        {"stage": "fold", "returncode": 0, "status": "ok"}]}))
    (d / "match_decomp.json").write_text(json.dumps({"verdict": "PASS", "reasons": [], "requests": [{"steps": 3}], "attribution": {}}))
    (d / "global_match.json").write_text(json.dumps({"verdict": "PASS", "comparator": "cb_a.global_match/v1", "checks": {}, "failed_checks": []}))
    return d


def _summary(d: Path, resumed: bool) -> dict:
    R.match_summary(str(d), 0, 0, 0, int(time.time()) - 60, -1, resumed)
    return json.load(open(d / "match_summary.json"))


def test_without_the_resume_an_old_record_is_stale(tmp_path):
    s = _summary(_summary_row(tmp_path, old=True, stage_record=True), False)
    assert s["record"]["fresh"] is False and s["record"]["resumed"] is False
    assert "predates" in s["record"]["reason"] and s["pass"] is False


def test_a_fresh_record_passes_unchanged(tmp_path):
    s = _summary(_summary_row(tmp_path, old=False, stage_record=True), False)
    assert s["record"]["fresh"] is True and s["record"]["reason"] is None


def test_the_cpu_resume_accepts_a_complete_old_record_and_labels_it(tmp_path):
    s = _summary(_summary_row(tmp_path, old=True, stage_record=True), True)
    assert s["record"]["fresh"] is True and s["record"]["resumed"] is True and s["record"]["resumed_record_complete"] is True
    assert s["record"]["stage_record_absent"] is False and s["record"]["capture_rc"] == 0


def test_the_cpu_resume_without_a_stage_record_is_accepted_on_the_completion_markers(tmp_path):
    s = _summary(_summary_row(tmp_path, old=True, stage_record=False), True)
    assert s["record"]["fresh"] is True and s["record"]["stage_record_absent"] is True and s["record"]["capture_rc"] is None


def test_the_cpu_resume_refuses_an_incomplete_record(tmp_path):
    d = _summary_row(tmp_path, old=True, stage_record=False)
    (d / "match" / "control" / "tokens.json").unlink()
    s = _summary(d, True)
    assert s["record"]["fresh"] is False and s["record"]["resumed_record_complete"] is False


# ---------------------------------------------------------------------------------------------------------------- weights of record

D = "38f7b397917d7951016a28d9a38df848f66557e0da4d1dd7d4155ad64ed9dbdf"
D2 = "9699b09e" + "0" * 56
CURRENT = {"schema": SCHEMA, "program_digest": D, "complete": True, "n_pinned": 163, "n_fields": 163,
           "unpinned": [], "unresolved": [], "engine_assumptions": {"kv_cache_dtype": "auto"}, "root_of_record": "2303f39a" + "0" * 56}
LEGACY_INT6 = {"schema": SCHEMA_V1, "program_digest": D, "complete": True, "n_pinned": 98, "n_fields": 163,
               "unpinned": [f"model.model.layers.{i}.self_attn.attn._k_scale" for i in range(65)], "unresolved": [], "root_of_record": "2303f39a" + "0" * 56}


def _stale(tmp_path: Path, record, result, schema: str = SCHEMA) -> str:
    comp = tmp_path / "build_request_LP9_T12"
    comp.mkdir(exist_ok=True)
    rec = comp / "weights_of_record.json"
    rec.write_text(json.dumps(record) if isinstance(record, dict) else record)
    if result is not None:
        (comp / "result.json").write_text(json.dumps(result))
    elif (comp / "result.json").exists():
        (comp / "result.json").unlink()
    return R.wor_stale(str(rec), str(comp / "result.json"), schema)


def test_a_record_of_an_older_schema_is_named_and_re_derived(tmp_path):
    v1 = {**CURRENT, "schema": SCHEMA_V1}
    ok = {"ok": True, "program_digest": D, "LP": 9, "T": 12}
    why = _stale(tmp_path, v1, ok)
    assert why.startswith(f"schema {SCHEMA_V1}, this tree derives {SCHEMA}") and "root formula differs" in why, why
    assert _stale(tmp_path, CURRENT, ok) == ""
    assert _stale(tmp_path, v1, ok, schema="") == ""
    why2 = _stale(tmp_path, v1, {**ok, "program_digest": D2})
    assert "stale" in why2 and "schema" not in why2


def test_the_component_digest_is_read_where_a_real_result_json_carries_it(tmp_path):
    real_other = {"schema": "x", "ok": True, "LP": 245, "T": 31, "program": {"digest": D2, "n": 1}, "artifact": {"program_digest": D2}}
    assert "stale" in _stale(tmp_path, CURRENT, real_other) and D2[:16] in _stale(tmp_path, CURRENT, real_other)
    assert _stale(tmp_path, CURRENT, {**real_other, "program": {"digest": D}, "artifact": {"program_digest": D}}) == ""
    assert "stale" in _stale(tmp_path, CURRENT, {"ok": True, "program": {"digest": D2}})
    assert "stale" in _stale(tmp_path, CURRENT, {"ok": True, "artifact": {"program_digest": D2}})
    comp = tmp_path / "build_request_LP9_T12"
    with gzip.open(comp / "instances.json.gz", "wt") as f:
        json.dump({"digest": D2, "rows": []}, f)
    assert "stale" in _stale(tmp_path, CURRENT, {"ok": True})
    with gzip.open(comp / "instances.json.gz", "wt") as f:
        json.dump({"digest": D, "rows": []}, f)
    assert _stale(tmp_path, CURRENT, {"ok": True}) == ""
    (comp / "instances.json.gz").unlink()
    assert _stale(tmp_path, CURRENT, {"ok": True}) == ""


def test_stale_legacy_incomplete_and_unreadable_records_are_named(tmp_path):
    ok = {"ok": True, "program_digest": D, "LP": 9, "T": 12}
    assert _stale(tmp_path, CURRENT, {**ok, "program_digest": D2}) == f"built for Program {D[:16]}, this component is {D2[:16]} (stale)"
    assert _stale(tmp_path, LEGACY_INT6, ok) == "predates MAN-09c (no engine_assumptions; 65 engine constant(s) unpinned by name)"
    rec = {**CURRENT, "complete": False, "unresolved": ["model.model.layers.0.mlp.experts.w13_weight"]}
    assert _stale(tmp_path, rec, ok) == "incomplete (1 unresolved field(s))"
    assert _stale(tmp_path, "{not json", ok).startswith("unreadable record (")
    assert _stale(tmp_path, CURRENT, None) == "" and _stale(tmp_path, LEGACY_INT6, None).startswith("predates MAN-09c")


@pytest.mark.parametrize("ck", ["None", ""])
def test_an_unresolved_checkpoint_dir_is_named_in_row_log_not_silently_unpinned(tmp_path, ck):
    r = _row(tmp_path)
    r.B, r.ck = 1, ck
    pdir = Path(r.d) / "build_request"
    pdir.mkdir()
    flags = r.weights_of_record()
    log = (Path(r.d) / "row.log").read_text()
    assert "WARNING weights of record NOT derived: no local checkpoint dir" in log and "weights unpinned (MAN-09)" in log
    assert not (pdir / "weights_of_record.json").exists() and not any(pdir.glob("weights_of_record.stale-*"))
    assert flags == (["--checkpoint-dir", "None"] if ck == "None" else [])


def test_a_stale_record_is_moved_aside_and_its_rederivation_names_what_it_supersedes(tmp_path):
    r = _row(tmp_path)
    ck = tmp_path / "ck"
    ck.mkdir()
    r.B, r.ck, r.py = 1, str(ck), "/bin/false"
    pdir = Path(r.d) / "build_request"
    pdir.mkdir()
    (pdir / "result.json").write_text(json.dumps({"ok": True, "program_digest": D2}))
    (pdir / "weights_of_record.json").write_text(json.dumps(CURRENT))
    flags = r.weights_of_record()
    log = (Path(r.d) / "row.log").read_text()
    assert "not reused -- built for Program" in log and "WARNING weights of record not derived (see weights_of_record.log)" in log
    assert len(list(pdir.glob("weights_of_record.stale-*.json"))) == 1
    assert flags == ["--checkpoint-dir", str(ck)]


def test_the_pin_at_b_ge_2_is_derived_from_the_first_manifest_component(tmp_path):
    d = tmp_path
    (d / "manifest.json").write_text(json.dumps({"components": {"r1": {"LP": 5, "T": 3}, "r0": {"LP": 9, "T": 2}}}))
    for name, lp, t in (("build_request_LP9_T2", 9, 2), ("build_request_LP5_T3", 5, 3), ("build_request", 9, 3)):
        (d / name).mkdir()
        (d / name / "result.json").write_text(json.dumps({"ok": True, "LP": lp, "T": t}))
    assert R.wor_component(str(d)) == str(d / "build_request_LP9_T2")
    (d / "build_request_LP9_T2" / "result.json").write_text(json.dumps({"ok": False, "LP": 9, "T": 2}))
    assert R.wor_component(str(d)) == str(d / "build_request_LP5_T3")


# ---------------------------------------------------------------------------------------------------------------- admission

@pytest.mark.skipif(not WL.exists(), reason="sweep workloads not in this checkout")
def test_the_admission_plan_is_written_logged_once_and_never_stops_the_row(tmp_path):
    row = "mistral-7b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager"
    r = _row(tmp_path, "--snapshot-max-bytes", "33554432", row=row)
    r.snap_steps, r.snap_def = None, "0,1"
    r.admission()
    doc = json.loads((Path(r.d) / "admission.json").read_text())
    assert doc["schema"] == "verity-sweep/admission/v1" and doc["advisory"] is True and doc["model"] == "mistral-7b"
    assert doc["config"]["snapshot_max_bytes"] == 33554432 and doc["config"]["snapshot_steps"] == "0,1" and doc["config"]["pairs"] == 3
    assert set(doc["stages"]) == {"build", "match", "commit"}
    assert (Path(r.d) / "row.log").read_text().count("ADMISSION advisory") == 1
    evs = [(e["stage"], e["ev"]) for e in _timeline(r)]
    assert evs == [("admission", "start"), ("admission", "end")] and _timeline(r)[1]["advisory"] is True


def test_the_admission_plan_survives_a_broken_planner_input(tmp_path):
    r = _row(tmp_path, row="nosuch-row__x")
    r.snap_steps, r.snap_def = None, "0,1"
    r.admission()
    log = (Path(r.d) / "row.log").read_text()
    assert "ADMISSION advisory unavailable" in log and "WARN" not in log
