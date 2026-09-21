"""Reference harness: `tele run --attempt-dir D --stage S [--timeout T] -- CMD...` (design §8).

Writes job.json (unless present), status.json{running}, starts the sidecar in its own session (setsid) with --root-pid = the workload,
emits span_start, runs CMD in its own process group with stdout/stderr tee'd to D/stdout.log / D/stderr.log (or inherited with --inherit-io),
on timeout writes cancel_intent{actor: timeout} then SIGTERM -> grace -> SIGKILL with cancel_results, records rc/signal/pid/pgid in
status.json, stops the sidecar (SIGTERM, waits <= linger + 5 s), emits span_end{outcome}, runs the classifier, exits with CMD's status
(128+signal when killed).  The sampler is never the reason the workload stops: a sampler start failure is recorded and CMD still runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import CANCEL_ACTORS
from .cancel import add_args as cancel_args, cancel as cancel_run
from .classify import classify, write as write_failure
from .events import Emitter, append, events_path, utc_now_ms
from .job import build_job, input_ref, set_status, sha256_file, write_json_atomic
from .procs import stat_fields


def _tele_argv() -> list[str]:
    return [sys.executable, "-m", "research.telemetry"]


def _marker(attempt_dir: Path, actor: str, target_pid: int | None, target_pgid: int | None, sig: int, reason: str, starttime: int | None,
            deliver: bool, job: str | None) -> int:
    ns = argparse.Namespace(actor=actor, reason=reason, signal=sig, target_pid=target_pid, target_pgid=target_pgid, target_pod=None,
                            attempt_dir=str(attempt_dir), fallback=None, expect_starttime=starttime, expect_comm=None, expect_nonempty=False,
                            deliver=deliver, confirm_exit_s=0.0, project=None, job=job, attempt=None, controller="tele-run")
    return cancel_run(ns)


def run(a: argparse.Namespace, cmd: list[str]) -> int:
    d = Path(a.attempt_dir)
    d.mkdir(parents=True, exist_ok=True)
    em = Emitter(d, proc="tele-run")
    policy_sha = sha256_file(a.policy)[0] if a.policy and Path(a.policy).exists() else None
    policy_version = a.policy_version
    if a.policy and Path(a.policy).exists() and not policy_version:
        try:
            import tomllib
            policy_version = tomllib.loads(Path(a.policy).read_text()).get("policy", {}).get("version")
        except Exception:  # noqa: BLE001
            policy_version = None
    if not (d / "job.json").exists() or a.rewrite_job:
        job = build_job(project=a.project, task=a.task, job=a.job, attempt=a.attempt, lane=a.lane, generation=a.generation, tree_sha=a.tree_sha,
                        release_tag=a.release_tag, workload_digest=a.workload_digest, stage=a.stage, inputs=[input_ref(x) for x in a.input],
                        profile={k: os.environ.get(k) for k in a.profile_key if k in os.environ}, command=cmd, declared_outputs=a.declared_output,
                        timeout_s=a.timeout, reservation={"gpu": a.reservation_gpu, "host_memory_bytes": a.reservation_host_memory_bytes},
                        policy_version=policy_version, policy_sha256=policy_sha, instruments=["cgroup", "process", "pss", "gpu", "storage"] if not a.no_sampler else [],
                        cadence_s=a.cadence, pss_every_s=a.pss_every)
        write_json_atomic(d / "job.json", job)
    em.queue("started", stage=a.stage, job=a.job, attempt=a.attempt)
    # ---- workload first (so the sampler gets its pid), suspended until the sampler is up? no: start it, then the sampler; the first sample is <1 s later
    out_f = err_f = None
    if a.inherit_io:
        stdout, stderr = None, None
    else:
        out_f = open(d / "stdout.log", "ab")
        err_f = open(d / "stderr.log", "ab")
        stdout, stderr = out_f, err_f
    t_start = time.monotonic()
    try:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, start_new_session=True, cwd=a.cwd or None)
    except OSError as e:
        set_status(d, "failed", rc=127, signal=None, error=f"spawn failed: {e}", stage=a.stage)
        em.emit("mark", stage=a.stage, name="spawn_failed", error=str(e))
        return 127
    st = stat_fields(proc.pid)
    starttime = st["starttime_ticks"] if st else None
    set_status(d, "running", pid=proc.pid, pgid=proc.pid, starttime_ticks=starttime, stage=a.stage, timeout_s=a.timeout, started_mono=t_start,
               started_utc=utc_now_ms(), command=cmd[:1] + ["..."] if a.hide_argv else cmd)
    # ---- sampler in its own session
    sampler = None
    if not a.no_sampler:
        sargs = _tele_argv() + ["sample", "--attempt-dir", str(d), "--cadence", str(a.cadence), "--root-pid", str(proc.pid), "--pss-every", str(a.pss_every),
                                "--linger", str(a.linger)] + sum((["--mount", m] for m in a.mount), [])
        try:
            sampler = subprocess.Popen(sargs, start_new_session=True, stdout=open(d / "sampler.log", "ab"), stderr=subprocess.STDOUT)
        except OSError as e:
            em.emit("mark", stage=a.stage, name="sampler_start_failed", error=str(e))
            sampler = None
    sid = em.span_start(a.stage, pid=proc.pid, pgid=proc.pid, cmd0=Path(cmd[0]).name)
    timed_out = False
    rc: int | None = None
    try:
        while True:
            try:
                rc = proc.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                pass
            if a.timeout and time.monotonic() - t_start >= a.timeout and not timed_out:
                timed_out = True
                _marker(d, "timeout", None, proc.pid, signal.SIGTERM, f"stage {a.stage} exceeded timeout_s={a.timeout}", None, True, a.job)
                try:
                    rc = proc.wait(timeout=a.grace)
                    break
                except subprocess.TimeoutExpired:
                    _marker(d, "timeout", None, proc.pid, signal.SIGKILL, f"stage {a.stage} still alive {a.grace}s after SIGTERM", None, True, a.job)
                    rc = proc.wait()
                    break
    except KeyboardInterrupt:
        _marker(d, "manual", None, proc.pid, signal.SIGTERM, "tele run interrupted (SIGINT)", None, True, a.job)
        rc = proc.wait()
    t_end = time.monotonic()
    sig = -rc if rc is not None and rc < 0 else None
    status_rc = 128 + sig if sig else rc
    outcome = "ok" if rc == 0 else "fail"
    em.span_end(a.stage, sid, outcome=outcome, rc=status_rc, signal=sig, timed_out=timed_out, refs=list(a.declared_output))
    set_status(d, "cancelled" if timed_out else ("done" if rc == 0 else "failed"), rc=status_rc, signal=sig, timed_out=timed_out, ended_mono=t_end,
               ended_utc=utc_now_ms(), wall_s=t_end - t_start, rc_is_shell_status=True)
    # ---- stop the sampler (it lingers by itself once the root pid is gone; SIGTERM makes it take final readings now)
    if sampler is not None:
        try:
            deadline = time.monotonic() + a.linger + 5.0
            while sampler.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if sampler.poll() is None:
                sampler.terminate()
                try:
                    sampler.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    sampler.kill()
                    em.emit("mark", stage=a.stage, name="sampler_killed_by_runner", note="sampler did not stop within 10 s of SIGTERM")
        except OSError:
            pass
    for f in (out_f, err_f):
        if f:
            f.close()
    if not a.no_classify:
        logs = [] if a.inherit_io else [str(d / "stdout.log"), str(d / "stderr.log")]
        doc = classify(d, logs=logs + list(a.log), check_record=a.check_record)
        write_failure(d, doc)
        print(f"[tele run] {a.stage}: rc={status_rc} class={doc['class']} ({doc['certainty']}) -- {doc['why']}", file=sys.stderr)
    return int(status_rc if status_rc is not None else 1)


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--timeout", type=float, default=None, help="seconds; SIGTERM with a timeout marker, then SIGKILL after --grace")
    ap.add_argument("--grace", type=float, default=10.0)
    ap.add_argument("--cadence", type=float, default=1.0)
    ap.add_argument("--pss-every", type=float, default=5.0)
    ap.add_argument("--linger", type=float, default=3.0)
    ap.add_argument("--mount", action="append", default=[])
    ap.add_argument("--no-sampler", action="store_true")
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--inherit-io", action="store_true", help="do not tee stdout/stderr into the attempt dir")
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--log", action="append", default=[], help="additional log files for the classifier")
    ap.add_argument("--check-record", default=None)
    ap.add_argument("--hide-argv", action="store_true")
    # job identity
    ap.add_argument("--project", default=os.environ.get("TELEMETRY_PROJECT", "unspecified"))
    ap.add_argument("--task", default=None)
    ap.add_argument("--job", default=None)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--lane", default=os.environ.get("TELEMETRY_LANE"))
    ap.add_argument("--generation", type=int, default=1)
    ap.add_argument("--tree-sha", default=None)
    ap.add_argument("--release-tag", default=None)
    ap.add_argument("--workload-digest", default=None)
    ap.add_argument("--input", action="append", default=[], help="name=path input artifact reference")
    ap.add_argument("--profile-key", action="append", default=[], help="environment variable NAMES to record (allowlist); values are read from the environment")
    ap.add_argument("--declared-output", action="append", default=[])
    ap.add_argument("--reservation-gpu", default=None)
    ap.add_argument("--reservation-host-memory-bytes", type=int, default=None)
    ap.add_argument("--policy", default=None, help="policy toml; its version + sha256 are recorded")
    ap.add_argument("--policy-version", default=None)
    ap.add_argument("--rewrite-job", action="store_true")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        i = argv.index("--")
        own, cmd = argv[:i], argv[i + 1:]
    else:
        own, cmd = argv, []
    ap = argparse.ArgumentParser(prog="tele run", description=__doc__.splitlines()[0])
    add_args(ap)
    a = ap.parse_args(own)
    if not cmd:
        ap.error("command after -- is required")
    return run(a, cmd)


if __name__ == "__main__":
    sys.exit(main())
