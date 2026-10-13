"""`verity-vllm row`: the options a row runs with, the driver's stage processes and the records it decides from.

* a cancelled row takes its running stage (and the stage's forked workers) with it: GNU ``timeout`` moves the stage into
  its own process group, which never sees the runner's killpg, so the driver forwards TERM/INT/HUP to timeout's pid,
  waits for the stage to be gone, records the cancel and exits 128+signal;
* a row's derives run in parallel by default, as many as the memory headroom holds at the largest shape's planned peak (a fit
  that bounds the epoch's measured derives) and at most one per CPU, the longest shapes first;
* a batch's workload Program compose waits only on the shapes its requests run, and its manifest is built beside the compose and the
  envelope's derive, taking the workload digest once it is composed (none when the compose fails), kept only when the Build passes;
* after a PASS Build the row stops (exit 21, no Match) when a call-boundary identity of its manifest has no source the
  Commit attaches, the first one named in stages.txt, and goes on to Match when each has one;
* a Commit over a Match record whose account files are gone refuses to start (PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE,
  exit 14); the Match summary accepts an explicit ``--match-phase cpu`` resume over the preserved capture;
* the Match of record (``fast`` inside ``shared``) is the default and the caller can choose the audit checker;
* the weights of record beside a component are reused only when they are this Program's, of this tree's schema and
  complete; an unresolved checkpoint dir is named in row.log;
* the admission plan is advisory: it writes admission.json and one row.log line and never stops the row;
* the taps run by default (NORM_TAP=0 etc. for A/B only), each attached only where the row's manifest names its family, and a TP
  row runs the guarded-max build and every tap that applies to it.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import os
import select
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.acquire.call_boundary_programs import LP, PF, T, build_dir, dense_program
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
           if k not in ("MATCH_IMPL", "MATCH_PIPELINE", "MATCH_PHASE", "VERITY_TREE", "RESEARCH_RUN_DIR", "RESEARCH_RUN_ID")}
    env.update(environ or {})
    # `prepare` creates the collector's build dir, whose default is a pod path (`/workspace/cp/...`) a non-root check can't create
    return cli.parse("row", ["run", row, "LLAMA32_1B", "unsloth/Llama-3.2-1B", "9535bd9b", "--sweep-dir", str(tmp), "--py", sys.executable,
                             "--native-collect-default", str(tmp / "nc_build"), *argv], env)


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


def test_a_foreign_verity_tree_is_refused_by_name(tmp_path, capsys):
    r = _row(tmp_path, environ={"VERITY_TREE": str(tmp_path)})
    with pytest.raises(Stop) as e:
        r.setup({})
    assert e.value.rc == 3 and "PRECHECK_FAIL_SOURCE_IDENTITY" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------------------------- cancel

# the stage: a python parent that forks a grandchild; both write their pids (and the stage its `timeout`'s), close their end of the
# test's FIFO and block on a pipe nobody writes to, until a signal ends them.  A row's fold / commit look like this.  SIGINT is the
# kernel's default here: Python's handler only flags the signal, so one that lands between closing the FIFO and entering read()
# leaves that process blocked for good, and the test's wait with it.
STAGE = r'''
import os, signal, sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
d = sys.argv[1]
for name, pid in (("stage", os.getpid()), ("timeout", os.getppid())):
    open(os.path.join(d, name + ".pid"), "w").write(str(pid))
ready = open(READY, "wb")
block, _ = os.pipe()
if os.fork() == 0:
    open(os.path.join(d, "grandchild.pid"), "w").write(str(os.getpid()))
ready.close()
os.read(block, 1)
'''

# the row; one byte on fd `registered` once the stage's `timeout` pid is where the signal forwarding finds it
HARNESS = r'''
import os, sys
from verity_vllm.pipeline import cli
from verity_vllm.pipeline.row_driver import Row, Stop
tmp, to, stage, registered = sys.argv[1:5]
o = cli.parse("row", ["run", "r", "R", "o/r", "x", "--sweep-dir", tmp, "--py", sys.executable], os.environ)
row = Row(row="r", role="R", repo="o/r", rev="x", stages="build", env=o.env, proc="row_pod")
os.makedirs(row.d, exist_ok=True)
row.child_env = dict(os.environ)
row.install_signal_forwarding()
class Announced(set):
    def add(self, pid):
        super().add(pid)
        os.write(int(registered), b".")
row._running = Announced()
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


def _wait_exited(fds: list[int]) -> None:
    for fd in fds:                                                       # a pidfd turns readable once its process has exited
        select.select([fd], [], [])
        os.close(fd)


def _start(tmp: Path, stage_exit: str | None = None) -> tuple[subprocess.Popen, Path]:
    """The row, running `stage.py` under `timeout 0` (no timer: the expiry test raises it); returns once the row forwards signals to the
    stage and, for the blocking stage, once the stage and its grandchild have written their pids."""
    stage, ready = tmp / "stage.py", tmp / "ready"
    os.mkfifo(ready)
    stage.write_text(f"READY = {str(ready)!r}\n" + STAGE if stage_exit is None else f"import sys; sys.exit({stage_exit})")
    d = tmp / "r"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TREE), os.environ.get("PYTHONPATH", "")])}
    env.pop("VERITY_TIMELINE", None)
    registered_r, registered_w = os.pipe()
    # the research runner starts the row as a session leader and later killpg()s that group -- same here
    p = subprocess.Popen([sys.executable, "-c", HARNESS, str(tmp), "0", str(stage), str(registered_w)], start_new_session=True, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(registered_w,))
    os.close(registered_w)
    assert os.read(registered_r, 1) == b".", "the row exited before it ran the stage"
    os.close(registered_r)
    if stage_exit is None:
        with open(ready, "rb") as f:
            assert f.read() == b""                                       # EOF: the stage and its grandchild both closed their ends
    return p, d


@pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")
@pytest.mark.parametrize("sig,rc", [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_killpg_on_the_row_takes_the_stage_and_its_grandchild(tmp_path, sig, rc):
    p, d = _start(tmp_path)
    stage, grand = int((d / "stage.pid").read_text()), int((d / "grandchild.pid").read_text())
    assert _alive(stage) and _alive(grand)
    assert os.getpgid(stage) != p.pid, "GNU timeout puts the stage in its own process group -- the premise of the forwarding"
    fds = [os.pidfd_open(stage), os.pidfd_open(grand)]
    os.killpg(p.pid, sig)
    assert p.wait() == rc
    _wait_exited(fds)                                                    # the stage or its forked worker surviving the cancel hangs here
    name = signal.Signals(sig).name
    assert f"CANCELLED by {name}" in (d / "stage.log").read_text()
    row_log = (d / "row.log").read_text()
    assert f"{name} received: forwarding to the running stage" in row_log and f"row cancelled by {name}" in row_log
    tl = [json.loads(ln) for ln in open(d / "timeline.jsonl")]
    assert {"stage": "row", "ev": "cancelled", "signal": name[3:]}.items() <= tl[0].items()
    assert {"stage": "row", "ev": "end", "ok": False, "stopped_at": "cancel"}.items() <= tl[1].items()


@pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")
def test_the_expiry_path_still_kills_the_forked_worker(tmp_path):
    p, d = _start(tmp_path)
    stage, grand, to = (int((d / f"{n}.pid").read_text()) for n in ("stage", "grandchild", "timeout"))
    fds = [os.pidfd_open(stage), os.pidfd_open(grand)]
    os.kill(to, signal.SIGALRM)                                          # what GNU timeout's own timer raises on expiry
    assert p.wait() == 124
    _wait_exited(fds)                                                    # timeout expiry leaving the forked worker alive hangs here
    log = (d / "stage.log").read_text()
    assert "TIMEOUT after 0s" in log and "CANCELLED" not in log


@pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")
def test_a_normal_exit_returns_the_stage_rc_unchanged(tmp_path):
    p, d = _start(tmp_path, stage_exit="7")
    assert p.wait() == 7
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


# ---------------------------------------------------------------------------------------------------------------- Build: derives

#: request derives of the 2026-09-29 epoch's stored Builds: (row, LP, T, decoder layers, measured peak RSS in MiB)
EPOCH_DERIVES = (("#11", 4096, 511, 16, 124529), ("#75", 1024, 127, 48, 43974), ("#74", 1024, 127, 36, 34107), ("#74", 817, 120, 36, 26893),
                 ("#57", 1024, 127, 26, 21118), ("#67", 1024, 127, 16, 14071), ("#67", 1024, 28, 16, 11067))


def test_the_planned_derive_peak_bounds_the_epochs_measured_ones_without_doubling_them():
    for row, lp, t, layers, mib in EPOCH_DERIVES:
        planned = R.derive_bytes(lp, t, layers) / (1 << 20)
        assert mib <= planned <= 2 * mib, (row, lp, t, round(planned), mib)


def test_auto_derives_as_many_as_the_memory_and_cpus_hold_at_the_largest_shape():
    shapes = [(1024, 127), *[(lp, 64) for lp in range(100, 1000, 100)]]
    per = R.derive_bytes(1024, 127, 16)
    assert R.build_jobs_auto(shapes, 16, headroom=10 * per, cpus=32) == 8
    assert R.build_jobs_auto(shapes, 16, headroom=10 * per, cpus=4) == 4
    assert R.build_jobs_auto(shapes[:3], 16, headroom=10 * per, cpus=32) == 4     # the step's derive and the three shapes'
    assert R.build_jobs_auto(shapes[:1], 16, headroom=10 * per, cpus=32) == 2     # B=1: the step's beside the request's
    assert R.build_jobs_auto(shapes, 16, headroom=per, cpus=32) == 1
    assert R.build_jobs_auto(shapes, None, headroom=100 << 30, cpus=32) == 5       # the model's layers unknown: 20 GiB per derive
    # a sweep's per-config share of a packed host: the budget, not the host's headroom, sizes auto
    assert R.build_jobs_auto(shapes, 16, headroom=10 * per, cpus=32, budget_gb=5 * per / (1 << 30)) == 4
    assert R.build_jobs_auto(shapes, 16, headroom=2 * per, cpus=32, budget_gb=100 * per / (1 << 30)) == 1


def test_the_model_layers_come_from_the_checkpoints_config(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "config.json").write_text(json.dumps({"num_hidden_layers": 16}))
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 26}}))
    assert (R.model_layers(str(tmp_path / "a")), R.model_layers(str(tmp_path / "b")), R.model_layers(str(tmp_path / "c"))) == (16, 26, None)


def _derive_row(tmp_path, monkeypatch, *argv: str, cpus: int, environ: dict[str, str] | None = None):
    row = "llama32-1b__bf16__l40s__tp1__b8__i256__o32__mixed__greedy__bi-eager"
    r = _row(tmp_path, *argv, row=row, environ=environ)
    s = R.row_shape(r.path(r.wl), r.path("manifests/checkpoints.json"), str(tmp_path))
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "config.json").write_text(json.dumps({"num_hidden_layers": 16}))
    r.B, r.LP, r.T, r.MML, r.ck, r.tgt_args = s["B"], s["LP"], s["T"], s["I"] + s["O"], str(ck), []
    ran: list[str] = []
    monkeypatch.setattr(r, "launch_args", lambda lp, t: [])
    monkeypatch.setattr(r, "run_tl", lambda st, to, lf, cmd, *extra: ran.append(st) or 0)
    monkeypatch.setattr(R, "_memory_headroom", lambda: 1 << 40)
    monkeypatch.setattr(R, "_cpus", lambda: cpus)
    return r, ran


def test_a_rows_derives_run_in_parallel_by_default(tmp_path, monkeypatch):
    r, ran = _derive_row(tmp_path, monkeypatch, cpus=4)
    assert r.derive_all(60) == (0, 0)
    mark = next(e for e in _timeline(r) if e["stage"] == "build.jobs")
    assert (mark["jobs"], mark["requested"]) == (4, "auto") and len(ran) == 2 + len(R.extra_shapes(r.path(r.wl), r.LP, r.T))
    r, _ran = _derive_row(tmp_path / "one", monkeypatch, "--build-jobs", "1", cpus=4)
    r.derive_all(60)
    assert next(e for e in _timeline(r) if e["stage"] == "build.jobs")["jobs"] == 1
    budget = 2.6 * R.derive_bytes(r.LP, r.T, 16) / (1 << 30)
    r, _ran = _derive_row(tmp_path / "budget", monkeypatch, cpus=4, environ={"BUILD_RAM_BUDGET_GB": str(budget)})
    r.derive_all(60)
    assert next(e for e in _timeline(r) if e["stage"] == "build.jobs")["jobs"] == 2


def test_the_extra_shapes_derive_longest_first(tmp_path, monkeypatch):
    r, ran = _derive_row(tmp_path, monkeypatch, cpus=1)
    r.derive_all(60)
    extra = [tuple(int(x) for x in st[len("build.derive_request_LP"):].split("_T")) for st in ran[2:]]
    assert ran[:2] == ["build.derive_step", "build.derive_request"] and len(extra) > 1
    assert [lp + t for lp, t in extra] == sorted((lp + t for lp, t in extra), reverse=True)


def _batch_build(tmp_path, monkeypatch, summary: str = "PASS", compose_rc: int = 0, **shape) -> tuple[SingleRow, dict]:
    """A B=8 Build whose children are stubbed: the compose records the dirs it was given, the manifest build takes the workload digest
    and writes manifest.json, and the derives end only once the compose and the manifest beside them have, every request's shape
    derived first (so nothing either one waits on is the envelope's derive)."""
    r, _ran = _derive_row(tmp_path, monkeypatch, cpus=4)
    for k, v in shape.items():
        setattr(r, k, v)
    seen: dict = {"derived": None, "manifests": [], "follow": None}

    def run_tl(stage, to, lf, cmd, *extra):
        assert stage == "build.global_program"
        seen["derived"] = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--derived"]
        if compose_rc == 0:
            os.makedirs(f"{r.d}/build_workload", exist_ok=True)
            Path(f"{r.d}/build_workload/workload_program.json").write_text(json.dumps({"workload_digest": "w" * 64}))
        return compose_rc

    def required_manifest(stage, digest=None):
        seen["manifests"].append(wd := digest() if digest else None)
        if wd == "":
            return None
        Path(f"{r.d}/manifest.json").write_text("{}")
        return 0

    follow = r.follow_compose

    def follow_compose(to):
        seen["follow"] = follow(to)
        return seen["follow"]

    def derive_all(to):
        wl = r.path(r.wl)
        for d in [*([f"{r.d}/build_request"] if (r.LP, r.T) in R.request_shapes(wl) else []),
                  *(f"{r.d}/build_request_LP{lp}_T{t}" for lp, t in R.extra_shapes(wl, r.LP, r.T))]:
            os.makedirs(d, exist_ok=True)
            Path(f"{d}/result.json").write_text(json.dumps({"ok": True}))
        seen["before_derives_end"] = (seen["follow"][0].result(), os.path.exists(f"{r.d}/manifest.json"))
        os.makedirs(f"{r.d}/build_request", exist_ok=True)
        Path(f"{r.d}/build_request/instances.json.gz").write_bytes(b"")
        return 0, 0

    for name, fn in (("gpu_exclusive", lambda why: None), ("follow_compose", follow_compose), ("derive_all", derive_all),
                     ("guarded", lambda what, fn, *a: summary), ("target_family", lambda *a, out: (0, "TARGET-FAMILY-OK")),
                     ("required_manifest", required_manifest), ("run_tl", run_tl), ("call_boundaries", lambda to, cmd=None: None)):
        monkeypatch.setattr(r, name, fn)
    monkeypatch.setattr(R, "build_line", lambda d: ("d" * 16, "m" * 16))
    return r, seen


def test_a_batch_composes_and_builds_its_manifest_without_waiting_for_the_envelope(tmp_path, monkeypatch):
    r, seen = _batch_build(tmp_path, monkeypatch)
    assert (r.LP, r.T) not in R.request_shapes(r.path(r.wl))
    r.build()
    assert seen["before_derives_end"] == ((0, 0), True) and seen["manifests"] == ["w" * 64]
    assert seen["derived"] == [f"{r.d}/build_request_LP{lp}_T{t}" for lp, t in R.extra_shapes(r.path(r.wl), r.LP, r.T)]
    assert (Path(r.d) / "manifest.json").exists() and (Path(r.d) / "stages.txt").read_text().startswith("build PASS ")
    # an envelope a request runs is a component: the compose lists it
    lp, t = sorted(R.request_shapes(r.path(r.wl)))[-1]
    r, seen = _batch_build(tmp_path / "envelope", monkeypatch, LP=lp, T=t)
    r.build()
    assert seen["derived"][0] == f"{r.d}/build_request" and len(seen["derived"]) == len(R.request_shapes(r.path(r.wl)))


def test_a_failed_batch_build_keeps_no_manifest(tmp_path, monkeypatch):
    r, seen = _batch_build(tmp_path, monkeypatch, summary="FAIL")
    with pytest.raises(Stop) as e:
        r.build()
    assert e.value.rc == 10 and seen["before_derives_end"][1] and not (Path(r.d) / "manifest.json").exists()
    # a compose that fails gives the manifest no workload digest: none is built
    r, seen = _batch_build(tmp_path / "compose", monkeypatch, compose_rc=1)
    with pytest.raises(Stop) as e:
        r.build()
    assert e.value.rc == 10 and seen["before_derives_end"] == ((1, None), False) and seen["manifests"] == [""]
    assert not (Path(r.d) / "manifest.json").exists()


#: `manifest build-global` as far as the row sees it: the workload digest it was given (`-`: the next stdin line, an empty one fails)
#: becomes manifest.json
_MANIFEST_CHILD = """
import json, sys
a = sys.argv[1:]
wd = a[a.index("--workload-digest") + 1]
if wd == "-":
    wd = sys.stdin.readline().strip()
    if not wd:
        sys.exit("no workload digest on stdin")
json.dump({"workload_digest": wd}, open(a[a.index("--out") + 1], "w"))
"""


def test_a_batch_manifest_takes_the_workload_digest_once_it_is_composed(tmp_path, monkeypatch):
    r, _ran = _derive_row(tmp_path, monkeypatch, cpus=4)
    os.makedirs(f"{r.d}/build_request_LP9_T1")
    monkeypatch.setattr(r, "cli", lambda *a, py=None: [sys.executable, "-c", _MANIFEST_CHILD, *a])
    assert r.required_manifest("build", lambda: "w" * 64) == 0
    assert json.loads((Path(r.d) / "manifest.json").read_text()) == {"workload_digest": "w" * 64}
    assert r.required_manifest("build", lambda: "") is None and not (Path(r.d) / "manifest.json").exists()
    assert [e.get("rc") for e in _timeline(r) if e["stage"] == "build.required_manifest" and e["ev"] == "end"] == [0, "none"]


# ---------------------------------------------------------------------------------------------------------------- Build: call boundaries

def _built(r: SingleRow, monkeypatch, **kw) -> dict:
    """A PASS Build of the call-boundary source's dense stand-in (`dense_program(**kw)`: its Build dir and manifest.json in the row dir),
    the Build's children stubbed; the call-boundary gate after it runs for real.  Returns the manifest."""
    b, _at = dense_program(**kw)
    _d, man = build_dir(Path(r.d), b.doc(LP), [PF[len("model."):]])
    (Path(r.d) / "manifest.json").write_text(json.dumps(man))
    r.B, r.LP, r.T, r.ck = 1, LP, T, "/ck"
    run_tl = r.run_tl

    def child(stage, to, lf, cmd, *extra):
        if stage != "build.global_program":
            return run_tl(stage, to, lf, cmd, *extra)
        os.makedirs(f"{r.d}/build_workload")
        Path(f"{r.d}/build_workload/workload_program.json").write_text(json.dumps({"workload_digest": "w" * 64}))
        return 0

    for name, fn in (("prepare", lambda: None), ("gpu_exclusive", lambda why: None), ("derive_all", lambda to: (0, 0)),
                     ("guarded", lambda what, fn, *a: "PASS"), ("target_family", lambda *a, out: (0, "TARGET-FAMILY-OK")),
                     ("required_manifest", lambda stage: 0), ("run_tl", child), ("timeline_report", lambda: None)):
        monkeypatch.setattr(r, name, fn)
    monkeypatch.setattr(R, "build_line", lambda d: ("d" * 16, "m" * 16))
    return man


@pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")
def test_a_call_boundary_without_a_source_stops_the_row_after_its_build_by_name(tmp_path, monkeypatch):
    r = _row(tmp_path, "--stages", "build,match")
    _built(r, monkeypatch, embed=PF)
    matched: list[int] = []
    monkeypatch.setattr(r, "match", lambda: matched.append(1))
    assert r.run() == 21 and not matched
    build, gate = (ln.split(" ", 3) for ln in (Path(r.d) / "stages.txt").read_text().splitlines())
    assert build[:2] == ["build", "PASS"] and gate[:2] == ["build-call-boundaries", "FAIL"]
    rec = json.loads((Path(r.d) / "call_boundaries.json").read_text())
    assert rec["pass"] is False and rec["uncovered"] > 0
    assert gate[3].startswith(f"rc=1 identities={rec['identities']} claimed=0 covered={rec['covered']} uncovered={rec['uncovered']} ")
    assert gate[3].endswith(f"first: {rec['named']['uncovered'][0]}") and "reads Program input 'prompt'" in gate[3]
    assert "call boundaries without a source the Commit attaches -> stop" in (Path(r.d) / "row.log").read_text()
    assert {"stage": "row", "ev": "end", "ok": False, "stopped_at": "call_boundaries"}.items() <= _timeline(r)[-1].items()


@pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")
def test_a_row_whose_every_call_boundary_has_a_source_goes_on_to_match(tmp_path, monkeypatch):
    r = _row(tmp_path, "--stages", "build,match")
    man = _built(r, monkeypatch)
    n = sum(1 for x in man["identities"] if x["family"] == "call_boundaries")
    matched: list[int] = []
    monkeypatch.setattr(r, "match", lambda: matched.append(1))
    assert r.run() == 0 and matched and n > 0
    build, gate = (ln.split(" ", 3) for ln in (Path(r.d) / "stages.txt").read_text().splitlines())
    assert build[:2] == ["build", "PASS"] and gate[:2] == ["build-call-boundaries", "PASS"]
    assert gate[3].startswith(f"identities={n} claimed=0 covered={n} uncovered=0 unattached=0 sources=call_boundary wall=")


@pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")
def test_a_tp_row_stops_after_its_build_on_any_call_boundary(tmp_path, monkeypatch):
    t = _tp_row(tmp_path, TP_DENSE)
    os.makedirs(t.d, exist_ok=True)
    t.child_env, t.shapes = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TREE), os.environ.get("PYTHONPATH", "")])}, []
    b, _at = dense_program()
    _d, man = build_dir(tmp_path, b.doc(LP), [PF[len("model."):]])

    def manifest_global(stage, to):
        (Path(t.d) / "manifest.json").write_text(json.dumps({**man, "ranks": 2}))
        return 0

    for name, fn in (("derive_jobs", lambda shapes, derives: 1), ("derive_all", lambda todo, jobs: None), ("compose_all", lambda ranks: None),
                     ("guarded_summary", lambda: "PASS"), ("manifest_global", manifest_global)):
        monkeypatch.setattr(t, name, fn)
    monkeypatch.setattr(R, "tp_build_digests", lambda d: ("d" * 16, []))
    monkeypatch.setattr(R, "tp_rank_workload_digest", lambda bdir, r: "w" * 16)
    with pytest.raises(Stop) as e:
        t.build()
    assert e.value.rc == 21
    build, gate = (ln.split(" ", 3) for ln in (Path(t.d) / "stages.txt").read_text().splitlines())
    assert build[:2] == ["build", "PASS"] and gate[:2] == ["build-call-boundaries", "FAIL"]
    assert gate[3].endswith(": a manifest of 2 ranks (the source serves one rank)") and " sources=none " in gate[3]


# ---------------------------------------------------------------------------------------------------------------- taps

TP_MOE = "qwen3-30b-a3b__bf16__l40s__tp2__b2__i1024__o128__mixed__greedy__bi-eager"
TP_DENSE = "llama32-1b__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager"


def _tap_manifest(path: Path, *families: str) -> str:
    """A manifest built under every tap policy whose identities name `families` (a policy with no site writes its header only)."""
    query = {"engine": "v2-query", **{k: {} for k in ("norm_scales", "router_softmax", "vocab_range", "guarded_max")}}
    path.write_text(json.dumps({"query": query, "identities": [{"family": "instance_outputs"}] + [{"family": f} for f in families]}))
    return str(path)


def test_the_taps_are_on_by_default_and_0_turns_one_off_for_ab_only(tmp_path):
    ck = _opts(tmp_path).commit
    assert (ck.norm_tap, ck.guarded_max_tap, ck.router_tap, ck.vocab_tap) == ("1", "1", "1", "1")
    off = _opts(tmp_path, "--router-tap", "0", environ={"NORM_TAP": "0", "GUARDED_MAX_TAP": "0"}).commit
    assert (off.norm_tap, off.guarded_max_tap, off.router_tap, off.vocab_tap) == ("0", "0", "0", "1")
    assert R.tap_policy_flags(ck) == ["--norm-scales", "--router-softmax", "--vocab-range", "--guarded-max"]
    assert R.tap_policy_flags(off) == ["--no-norm-scales", "--no-router-softmax", "--vocab-range", "--no-guarded-max"]
    for argv in (["manifest", "build", "--program", "p"], ["manifest", "build-global", "--fixture", "f", "--programs-root", "r"],
                 ["manifest-verify", "--manifest", "m"]):
        on, ab = cli.parse(argv[0], [*argv[1:], "--out", "o"], {}), cli.parse(argv[0], [*argv[1:], "--out", "o", *R.tap_policy_flags(off)], {})
        assert (on.norm_scales, on.router_softmax, on.vocab_range, on.guarded_max) == (True, True, True, True)
        assert (ab.norm_scales, ab.router_softmax, ab.vocab_range, ab.guarded_max) == (False, False, True, False)


def test_a_manifest_of_other_policies_is_rebuilt_either_way(tmp_path):
    man = _tap_manifest(tmp_path / "manifest.json", "norm_scales")
    assert R.manifest_policy_mismatch(man, _opts(tmp_path).commit) == []
    assert R.manifest_policy_mismatch(man, _opts(tmp_path, environ={"NORM_TAP": "0"}).commit) == ["norm_scales"]
    (tmp_path / "old.json").write_text(json.dumps({"query": {"engine": "v2-query"}, "identities": []}))
    assert R.manifest_policy_mismatch(str(tmp_path / "old.json"), _opts(tmp_path).commit) == ["norm_scales", "router_softmax", "vocab_range",
                                                                                                "guarded_max"]


PD = "p" * 64


def _record_manifest(path: Path, query_id: str, *, program_digest: str = PD, partition: bool = True, word_check: bool = True) -> None:
    query = {"query_id": query_id, **{k: {} for k in ("norm_scales", "router_softmax", "vocab_range", "guarded_max")}}
    if partition:
        query["partition"] = {"object": {"format": "verity/partition/v1"}, "digest": "d" * 128}
    if word_check:
        query["word_check"] = {"query": query_id, "strict": True, "ok": True, "violations": 0}
    path.write_text(json.dumps({"query": query, "program_digest": program_digest, "identities": [{"family": "instance_outputs"}]}))


def _row_with_build(tmp_path: Path, monkeypatch) -> tuple[SingleRow, list[str]]:
    """A B=1 row whose Build recorded Program `PD`, and whose manifest rebuild (recorded in the list) writes the manifest of record."""
    r = _row(tmp_path)
    r.B = 1
    (Path(r.d) / "build_request").mkdir()
    (Path(r.d) / "build_request" / "instances.json.gz").write_bytes(b"")
    (Path(r.d) / "build_summary.json").write_text(json.dumps({"request": {"program_digest": PD}}))
    built: list[str] = []
    monkeypatch.setattr(r, "required_manifest", lambda stage: built.append(stage) or _record_manifest(Path(r.d) / "manifest.json", R.QUERY_OF_RECORD) or 0)
    return r, built


def test_the_rows_query_of_record_is_the_one_its_manifest_build_writes():
    from verity_vllm.query import required
    assert R.QUERY_OF_RECORD == required.QUERY_OF_RECORD == required.query_header("ampere", "fa2_hidden_m1_stream", return_slots=True)["query_id"]


def test_the_builds_q_word_manifest_with_a_partition_of_its_program_is_the_manifest_of_record(tmp_path, monkeypatch):
    """#4's Commit (15:31Z) moved the Build's Q_word manifest aside as "not the query path's" and rebuilt it: a third full build per row."""
    r, built = _row_with_build(tmp_path, monkeypatch)
    assert r.program_digest() == PD
    _record_manifest(Path(r.d) / "manifest.json", R.QUERY_OF_RECORD)
    assert r.manifest_of_record() == f"{r.d}/manifest.json" and built == [] and list(Path(r.d).glob("manifest.*-*.json")) == []
    assert "required-value manifest of record:" in (Path(r.d) / "row.log").read_text()


@pytest.mark.parametrize("kw, slug, why", [
    ({"query_id": "Q_module_body_v1"}, "other-query", "is not built under the query of record"),      # a pre-S1 Build's
    ({"program_digest": "q" * 64}, "other-program", "is of Program qqqqqqqqqqqqqqqq, not the Build's pppppppppppppppp"),
    ({"partition": False}, "no-partition", "names no partition"),
    # the Commit refuses a Q_word manifest without a passing word check (commit.partition.word_check_refusal): a Build of before the
    # header recorded it, or one built with --word-check off, is rebuilt instead of starting a Commit that refuses
    ({"word_check": False}, "unchecked", "does not record a passing word check (no word check recorded"),
])
def test_a_manifest_of_an_older_query_another_program_or_no_partition_is_moved_aside_and_rebuilt(tmp_path, monkeypatch, kw, slug, why):
    r, built = _row_with_build(tmp_path, monkeypatch)
    _record_manifest(Path(r.d) / "manifest.json", kw.pop("query_id", R.QUERY_OF_RECORD), **kw)
    assert r.manifest_of_record() == f"{r.d}/manifest.json" and built == ["commit"]
    aside = list(Path(r.d).glob(f"manifest.{slug}-*.json"))
    assert len(aside) == 1 and why in (Path(r.d) / "row.log").read_text()
    assert R.manifest_not_of_record(R.load_json(f"{r.d}/manifest.json"), PD, r.ck_) == ("", "")


def test_a_dense_tp1_row_attaches_the_norm_tap_only(tmp_path):
    r = _row(tmp_path)
    r.hidden_so, r.B = "/x/verity_fa2_matReqG.so", 1
    r.env = dataclasses.replace(r.env, norm_tap_so=str(tmp_path / "absent" / "verity_norm_tap.so"))   # a host with the default build (/workspace/cp/...) has the .so
    man = _tap_manifest(tmp_path / "manifest.json", "norm_scales")
    taps = R.applicable_taps(man, r.ck_)
    assert taps == ["norm_tap"]
    cmd = r.commit_cmd(taps)
    assert cmd[cmd.index("--norm-tap-so") + 1] == r.env.norm_tap_so and "--router-tap-so" not in cmd and "--vocab-range" not in cmd
    assert R.missing_tap_build(taps, r.env).startswith("NORM_TAP=1 and the manifest names its family, but the tap .so is missing")
    assert R.applicable_taps(_tap_manifest(tmp_path / "bare.json"), r.ck_) == [] and R.missing_tap_build([], r.env) == ""
    assert R.applicable_taps(man, _opts(tmp_path, environ={"NORM_TAP": "0"}).commit) == []


def _tp_row(tmp_path: Path, row: str, *argv: str, **kw):
    from verity_vllm.pipeline.row_tp import TpRow
    o = _opts(tmp_path, *argv, row=row, **kw)
    t = TpRow(row=o.row, role=o.role, repo=o.repo, rev=o.rev, stages=o.stages, env=o.env, world=2, gpu_util=None,
              build=o.build, match=o.match, commit=o.commit)
    t.hidden_so, t.B, t.engine, t.ck, t.fold_profile = t.hidden_build(), 2, [], "/ck", ""
    t.bdir, t.M = f"{t.d}/build", f"{t.d}/match"
    return t


def test_a_tp_row_runs_the_guarded_build_and_every_tap_its_manifest_names(tmp_path, monkeypatch):
    t = _tp_row(tmp_path, TP_MOE)
    assert t.hidden_so == t.env.hidden_so_fa2_guarded
    assert _tp_row(tmp_path, TP_MOE, environ={"GUARDED_MAX_TAP": "0"}).hidden_so == t.env.hidden_so_fa2
    seen: list[list[str]] = []
    monkeypatch.setattr(R, "tp_rank_workload_digest", lambda bdir, r: f"wd{r}")
    monkeypatch.setattr(t, "run_tl", lambda stage, to, log, cmd: seen.append(cmd) or 0)
    t.manifest_global("commit.required_manifest", 1)
    assert {"--norm-scales", "--router-softmax", "--vocab-range", "--guarded-max"} <= set(seen[0])
    taps = R.applicable_taps(_tap_manifest(tmp_path / "moe.json", "norm_scales", "router_softmax", "vocab_range"), t.ck_)
    cmd = t.commit_cmd(taps)
    assert cmd[cmd.index("--hidden-so") + 1] == t.env.hidden_so_fa2_guarded and "--vocab-range" in cmd
    assert cmd[cmd.index("--norm-tap-so") + 1] == t.env.norm_tap_so and cmd[cmd.index("--router-tap-so") + 1] == t.env.router_tap_so
    dense = _tp_row(tmp_path, TP_DENSE)
    cmd = dense.commit_cmd(R.applicable_taps(_tap_manifest(tmp_path / "dense.json", "norm_scales", "vocab_range"), dense.ck_))
    assert "--norm-tap-so" in cmd and "--vocab-range" in cmd and "--router-tap-so" not in cmd


@pytest.mark.parametrize("row, p2p", [
    ("glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager", None),   # an H box: its env_required names none
    (TP_DENSE, "1"),                                                          # an L40S row's env_required names it
    (TP_MOE, "1"),                  # an L40S workload written before env_required: the generator's rule for its target
])
def test_a_tp_rows_stages_have_nccl_p2p_disable_only_when_its_workload_requires_it(tmp_path, monkeypatch, row, p2p):
    """The caller's NCCL_P2P_DISABLE=1 (an L40S operator's shell) reaches the stages only when the row's workload requires it."""
    t = _tp_row(tmp_path, row, environ={"NCCL_P2P_DISABLE": "1"})

    def precheck_reached() -> str:
        raise Stop(99)
    monkeypatch.setattr(t, "hidden_build", precheck_reached)
    with pytest.raises(Stop) as e:
        t.prepare()
    assert e.value.rc == 99 and t.nccl == p2p
    os.makedirs(t.d, exist_ok=True)
    stage = [sys.executable, "-c", "import os; print('NCCL_P2P_DISABLE=' + os.environ.get('NCCL_P2P_DISABLE', 'unset'))"]
    assert t.run_to(0, f"{t.d}/stage.log", stage) == 0
    assert open(f"{t.d}/stage.log").read().splitlines()[-1] == f"NCCL_P2P_DISABLE={p2p or 'unset'}"


# ---------------------------------------------------------------------------------------------------------------- Match summary

STAGE_START = 1_600_000_000


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
    t = STAGE_START - 3600 if old else STAGE_START
    os.utime(d / "match" / "capture" / "log.jsonl.gz", (t, t))
    if stage_record:
        (d / "match" / "stages.json").write_text(json.dumps({"stages": [{"stage": "capture", "returncode": 0, "status": "ok"},
                                                                        {"stage": "control", "returncode": 0, "status": "ok"},
                                                                        {"stage": "fold", "returncode": 0, "status": "ok"}]}))
    (d / "match_decomp.json").write_text(json.dumps({"verdict": "PASS", "reasons": [], "requests": [{"steps": 3}], "attribution": {}}))
    (d / "global_match.json").write_text(json.dumps({"verdict": "PASS", "comparator": "cb_a.global_match/v1", "checks": {}, "failed_checks": []}))
    return d


def _summary(d: Path, resumed: bool) -> dict:
    R.match_summary(str(d), 0, 0, 0, STAGE_START, -1, resumed)
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
