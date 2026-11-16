"""Reference harness: `tele run --attempt-dir D --stage S [--timeout T] -- CMD...`."""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import CANCEL_ACTORS, end_state, pyspy
from .cancel import cancel as cancel_run
from .cgroup import Cgroup, session_limit, uncapped
from .classify import classify, write as write_failure
from .events import Emitter, utc_now_ms
from .job import build_job, input_ref, sampler_instruments, set_status, sha256_file, write_json_atomic
from .procs import live_group, stat_fields
from .scope import CONTROL_CANCEL, Sweep, cancel_request, claim


def _tele_argv() -> list[str]:
    return [sys.executable, "-m", "research.telemetry"]


def _sampler_argv(a: argparse.Namespace, d: Path, root_pid: int) -> list[str]:
    return _tele_argv() + ["sample", "--attempt-dir", str(d), "--cadence", str(a.cadence), "--root-pid", str(root_pid), "--pss-every", str(a.pss_every),
                           "--pss-max-rss-gb", str(a.pss_max_rss_gb), "--linger", str(a.linger)] + sum((["--mount", m] for m in a.mount), [])


def _tele_env() -> dict[str, str]:
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not pp else (root + os.pathsep + pp if root not in pp.split(os.pathsep) else pp)
    return env


def _marker(attempt_dir: Path, actor: str, target_pid: int | None, target_pgid: int | None, sig: int, reason: str, starttime: int | None,
            deliver: bool, job: str | None) -> int:
    ns = argparse.Namespace(actor=actor, reason=reason, signal=sig, target_pid=target_pid, target_pgid=target_pgid, target_pod=None,
                            attempt_dir=str(attempt_dir), fallback=None, expect_starttime=starttime, expect_comm=None, expect_nonempty=False,
                            expect_run_dir=None, deliver=deliver, confirm_exit_s=0.0, project=None, job=job, attempt=None, controller="tele-run")
    return cancel_run(ns)


CONTROL_TIMEOUT = Path("control") / "timeout_s"
OUT_DIR = "out"
EX_NO_MEMORY_LIMIT = 75                               # EX_TEMPFAIL: the workload never started; it can start once its login gets its scope
clock = time.monotonic


class _TimeoutControl:
    def __init__(self, attempt_dir: Path):
        self.path = attempt_dir / CONTROL_TIMEOUT
        self._sig: tuple[int, int] | None = None
        self._bad: str | None = None

    def poll(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            st = self.path.stat()
        except OSError as e:
            print(f"[tele run] {self.path.name} unread, so no timeout change is taken ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            return None
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._sig:
            return None
        self._sig = sig
        raw = ""
        try:
            raw = self.path.read_text().strip()
            doc = json.loads(raw)
            if not isinstance(doc, dict):
                doc = {"timeout_s": doc}
            v = doc.get("timeout_s")
            if v is not None:
                v = float(v)
                if v <= 0:
                    raise ValueError("timeout_s must be > 0 or null")
            doc["timeout_s"] = v
            return doc
        except (OSError, ValueError, TypeError) as e:
            if raw != self._bad:
                print(f"[tele run] ignoring unparseable {self.path}: {e}", file=sys.stderr)
                self._bad = raw
            return None


def _close_cancel(attempt_dir: Path) -> dict[str, Any] | None:
    try:
        if claim(attempt_dir / CONTROL_CANCEL, {"closed_utc": utc_now_ms(), "by": "runner"}):
            return None
    except OSError as e:
        print(f"[tele run] {CONTROL_CANCEL} not claimed as closed ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        return None
    return cancel_request(attempt_dir)


STALL_CHECK_S = 30.0
RECORDER_FILES = ("status.json", "job.json", "launcher.log", "sampler.json", "sampler.log", "events*.jsonl", "resources*.jsonl", "control/*",
                  "logs/stall-*.txt")


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _progress(d: Path, logs: list[str]) -> tuple[int, int, int]:
    paths = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs]
    paths = [p for p in paths if not any(fnmatch.fnmatchcase(os.path.relpath(p, d), g) for g in RECORDER_FILES)] + list(logs)
    newest = total = n = 0
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            continue
        newest, total, n = max(newest, st.st_mtime_ns), total + st.st_size, n + 1
    return newest, total, n


class _StallWatch:
    def __init__(self, d: Path, em: Emitter, stage: str, after_s: float, logs: list[str], procs: Callable[[], list[tuple[int, str]]], *,
                 run: Callable[..., Any] = subprocess.run, which: Callable[[str], str | None] = shutil.which, stamp: Callable[[], str] = _utc_stamp):
        self.d, self.em, self.stage, self.after_s, self.logs, self.procs = d, em, stage, after_s, logs, procs
        self.run, self.which, self.stamp = run, which, stamp
        self.sig = _progress(d, logs)
        self.since = self.checked = clock()
        self.armed, self.said_missing = True, False

    def poll(self) -> None:
        now = clock()
        if now - self.checked < STALL_CHECK_S:
            return
        self.checked = now
        sig = _progress(self.d, self.logs)
        if sig != self.sig:
            self.sig, self.since, self.armed = sig, now, True
            return
        if not self.armed or now - self.since < self.after_s:
            return
        self.armed = False
        idle_s = round(now - self.since, 1)
        results = pyspy.dump(self.procs(), run=self.run, which=self.which)
        if results is None:
            if not self.said_missing:
                self.em.mark(self.stage, "stall_dump_skipped", idle_s=idle_s, reason=pyspy.MISSING)
                self.said_missing = True
            return
        rel = f"logs/stall-{self.stamp()}.txt"
        (self.d / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.d / rel).write_text(f"# no progress in {self.d} for {idle_s:g} s\n" + pyspy.render(results))
        self.em.mark(self.stage, "stall_dump", idle_s=idle_s, path=rel, pids=[r["pid"] for r in results],
                     failed=[r["pid"] for r in results if r["rc"] != 0])


def set_timeout(attempt_dir: str | Path, timeout_s: float | None, actor: str, reason: str) -> Path:
    p = Path(attempt_dir) / CONTROL_TIMEOUT
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(p, {"timeout_s": timeout_s, "actor": actor, "reason": reason, "set_utc": utc_now_ms()})
    return p


def set_timeout_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="tele set-timeout", description="raise/lower/clear a running attempt's stage timeout (recorded as a mark by its tele run)")
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("--timeout-s", required=True, help="seconds, or 'none' to clear the timeout")
    ap.add_argument("--actor", default="manual", choices=sorted(CANCEL_ACTORS))
    ap.add_argument("--reason", required=True)
    a = ap.parse_args(argv)
    v = None if a.timeout_s.lower() in ("none", "null", "0") else float(a.timeout_s)
    p = set_timeout(a.attempt_dir, v, a.actor, a.reason)
    print(f"[tele set-timeout] {p}: timeout_s={v} actor={a.actor}", file=sys.stderr)
    return 0


def _end_leftovers(em: Emitter, stage: str, pgid: int) -> None:
    if not (left := live_group(pgid)):
        return
    em.mark(stage, "group_outlived_leader", n=len(left), pids=[f"{st['pid']} {st['comm']}" for st in left[:10]])
    print(f"[tele run] {stage}: {len(left)} process(es) of the workload's group {pgid} outlived it; SIGKILL to the group",
          file=sys.stderr)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _end_record(ended: Callable[[], dict[str, Any]] | None, em: Emitter, stage: str) -> dict[str, Any]:
    if ended is None:
        return {}
    try:
        return dict(ended())
    except Exception as e:  # noqa: BLE001  what the caller records is never the reason a run has no terminal status
        em.mark(stage, "ended_failed", error=f"{e.__class__.__name__}: {e}"[:300])
        return {}


def run(a: argparse.Namespace, cmd: list[str], poll: Callable[[], None] | None = None,
        ended: Callable[[], dict[str, Any]] | None = None) -> int:
    d = Path(a.attempt_dir)
    d.mkdir(parents=True, exist_ok=True)
    em = Emitter(d, proc="tele-run")
    policy_sha = sha256_file(a.policy)[0] if a.policy and Path(a.policy).exists() else None
    policy_version = a.policy_version
    if a.policy and Path(a.policy).exists() and not policy_version:
        try:
            import tomllib
            policy_version = tomllib.loads(Path(a.policy).read_text()).get("policy", {}).get("version")
        except (OSError, ValueError, AttributeError) as e:
            print(f"[tele run] {a.policy} unread, so the run records no policy version ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            policy_version = None
    if not (d / "job.json").exists() or a.rewrite_job:
        job = build_job(project=a.project, task=a.task, job=a.job, attempt=a.attempt, lane=a.lane, generation=a.generation, tree_sha=a.tree_sha,
                        release_tag=a.release_tag, workload_digest=a.workload_digest, stage=a.stage, inputs=[input_ref(x) for x in a.input],
                        profile={k: os.environ.get(k) for k in a.profile_key if k in os.environ}, command=cmd, declared_outputs=a.declared_output,
                        timeout_s=a.timeout, reservation={"gpu": a.reservation_gpu, "host_memory_bytes": a.reservation_host_memory_bytes},
                        policy_version=policy_version, policy_sha256=policy_sha, instruments=sampler_instruments(a.pss_every) if not a.no_sampler else [],
                        cadence_s=a.cadence, pss_every_s=a.pss_every)
        write_json_atomic(d / "job.json", job)
    em.queue("started", stage=a.stage, job=a.job, attempt=a.attempt)
    if (why := uncapped(Cgroup().limits(), session_limit())) is not None:
        set_status(d, "failed", rc=EX_NO_MEMORY_LIMIT, signal=None, error=f"no memory limit: {why}", stage=a.stage,
                   **_end_record(ended, em, a.stage))
        em.emit("mark", stage=a.stage, name="no_memory_limit", error=why)
        print(f"[tele run] refused to start the workload: {why}", file=sys.stderr)
        return EX_NO_MEMORY_LIMIT
    out_f = err_f = None
    if a.inherit_io:
        stdout, stderr = None, None
    else:
        out_f = open(d / "stdout.log", "ab")
        err_f = open(d / "stderr.log", "ab")
        stdout, stderr = out_f, err_f
    (d / OUT_DIR).mkdir(exist_ok=True)
    (d / CONTROL_CANCEL).unlink(missing_ok=True)
    t_start, armed = time.monotonic(), clock()
    try:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, start_new_session=True, cwd=a.cwd or None,
                                env={**os.environ, "RESEARCH_RUN_DIR": str(d.resolve()), "OUT": str((d / OUT_DIR).resolve())})
    except OSError as e:
        set_status(d, "failed", rc=127, signal=None, error=f"spawn failed: {e}", stage=a.stage, **_end_record(ended, em, a.stage))
        em.emit("mark", stage=a.stage, name="spawn_failed", error=str(e))
        return 127
    st = stat_fields(proc.pid)
    starttime = st["starttime_ticks"] if st else None
    set_status(d, "running", pid=proc.pid, pgid=proc.pid, starttime_ticks=starttime, stage=a.stage, timeout_s=a.timeout, started_mono=t_start,
               started_utc=utc_now_ms(), command=cmd[:1] + ["..."] if a.hide_argv else cmd)
    sampler = None
    if not a.no_sampler:
        try:
            sampler = subprocess.Popen(_sampler_argv(a, d, proc.pid), start_new_session=True, stdout=open(d / "sampler.log", "ab"), stderr=subprocess.STDOUT, env=_tele_env())
        except OSError as e:
            em.emit("mark", stage=a.stage, name="sampler_start_failed", error=str(e))
            sampler = None
    sid = em.span_start(a.stage, pid=proc.pid, pgid=proc.pid, cmd0=Path(cmd[0]).name)
    timed_out = False
    rc: int | None = None
    timeout, limit = a.timeout, "the run's --timeout"
    ctl = _TimeoutControl(d)
    sweep: Sweep | None = None
    deadline, stop_actor, stop_grace = 0.0, "timeout", a.grace
    stall = (_StallWatch(d, em, a.stage, a.stall_dump_s, list(a.log), lambda: pyspy.python_tree(proc.pid, starttime))
             if a.stall_dump_s > 0 else None)
    try:
        while True:
            try:
                rc = proc.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                pass
            new = ctl.poll()
            if new is not None and new["timeout_s"] != timeout:
                em.mark(a.stage, "timeout_changed", old_timeout_s=timeout, new_timeout_s=new["timeout_s"], actor=new.get("actor"), reason=new.get("reason"),
                        elapsed_s=round(clock() - armed, 3), control=str(ctl.path))
                timeout, limit = new["timeout_s"], new.get("reason") or f"{ctl.path}, set by {new.get('actor') or 'someone'}"
                set_status(d, "running", timeout_s=timeout, timeout_changed_utc=utc_now_ms())
            if stall is not None:
                try:
                    stall.poll()
                except Exception as e:  # noqa: BLE001 (the watch is never the reason the workload stops)
                    em.mark(a.stage, "stall_watch_failed", error=f"{e.__class__.__name__}: {e}"[:300])
                    stall = None
            if poll is not None:
                try:
                    poll()
                except Exception as e:  # noqa: BLE001  nor is what the caller watches for
                    em.mark(a.stage, "poll_failed", error=f"{e.__class__.__name__}: {e}"[:300])
                    poll = None
            if sweep is None and (req := cancel_request(d)) is not None:
                sweep = Sweep(proc.pid, str(d.resolve()), leader_start=starttime, scopes=req.get("scopes") or [])
                stop_actor, stop_grace = (req.get("actor") if req.get("actor") in CANCEL_ACTORS else "manual"), float(req["grace_s"])
                deadline = clock() + stop_grace
                em.mark(a.stage, "cancel_seen", actor=stop_actor, grace_s=stop_grace, reason=req.get("reason"))
            if timeout and clock() - armed >= timeout and not timed_out:
                timed_out = True
                _marker(d, "timeout", None, proc.pid, signal.SIGTERM, f"stage {a.stage} exceeded timeout_s={timeout} ({limit})", None, True, a.job)
                if sweep is None:
                    sweep = Sweep(proc.pid, str(d.resolve()), leader_start=starttime)
                    stop_actor, stop_grace, deadline = "timeout", a.grace, clock() + a.grace
                    sweep.signal(signal.SIGTERM, skip=[st["pid"] for st in live_group(proc.pid)])
            if sweep is not None and clock() >= deadline:
                _marker(d, stop_actor, None, proc.pid, signal.SIGKILL, f"stage {a.stage} still alive {stop_grace:g}s after SIGTERM", None,
                        True, a.job)
                rc = proc.wait()
                break
    except KeyboardInterrupt:
        _marker(d, "manual", None, proc.pid, signal.SIGTERM, "tele run interrupted (SIGINT)", None, True, a.job)
        rc = proc.wait()
    t_end = time.monotonic()
    if sweep is None and (req := _close_cancel(d)) is not None:
        sweep = Sweep(proc.pid, str(d.resolve()), leader_start=starttime, scopes=req.get("scopes") or [])
        stop_grace, deadline = float(req["grace_s"]), clock() + float(req["grace_s"])
    scope_end = None
    if sweep is not None:
        scope_end = sweep.end(max(0.0, deadline - clock()), termed=True, scopes=lambda: (cancel_request(d) or {}).get("scopes") or [])
        em.mark(a.stage, "scope_ended", **scope_end)
        if not scope_end["empty"]:
            print(f"[tele run] {a.stage}: {len(scope_end['left'])} of the run's process(es) outlived SIGKILL: {', '.join(scope_end['left'])}",
                  file=sys.stderr)
    _end_leftovers(em, a.stage, proc.pid)
    sig = -rc if rc is not None and rc < 0 else None
    status_rc = 128 + sig if sig else rc
    outcome = "ok" if rc == 0 else "fail"
    em.span_end(a.stage, sid, outcome=outcome, rc=status_rc, signal=sig, timed_out=timed_out, refs=list(a.declared_output))
    set_status(d, end_state(rc, timed_out), rc=status_rc, signal=sig, timed_out=timed_out, ended_mono=t_end,
               ended_utc=utc_now_ms(), wall_s=t_end - t_start, rc_is_shell_status=True, **({"scope_end": scope_end} if scope_end else {}),
               **_end_record(ended, em, a.stage))
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
        except OSError as e:
            print(f"[tele run] the sampler was not stopped ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
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
    ap.add_argument("--stall-dump-s", type=float, default=900.0,
                    help="seconds without a change in the attempt dir (its logs included) before a py-spy dump of the workload's Python "
                         "processes goes to logs/stall-<UTC stamp>.txt, once per stall (default 900); 0 turns it off")
    ap.add_argument("--cadence", type=float, default=1.0)
    ap.add_argument("--pss-every", type=float, default=5.0, help="seconds between the sampler's PSS rounds (default 5); 0 turns PSS off")
    ap.add_argument("--pss-max-rss-gb", type=float, default=8.0,
                    help="the sampler never reads smaps_rollup of a process with more RSS than this, in GB (default 8; tele sample --help)")
    ap.add_argument("--linger", type=float, default=3.0)
    ap.add_argument("--mount", action="append", default=[])
    ap.add_argument("--no-sampler", action="store_true")
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--inherit-io", action="store_true", help="do not tee stdout/stderr into the attempt dir")
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--log", action="append", default=[], help="additional log files for the classifier")
    ap.add_argument("--check-record", default=None)
    ap.add_argument("--hide-argv", action="store_true")
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
    ap.add_argument("--declared-output", action="append", default=[], metavar="GLOB",
                    help="run-dir-relative glob of files custody uploads beside all of out/ ($OUT) (repeatable; `*` crosses `/`); other "
                         "files are listed by size and hash, not uploaded, past 256 MiB. See .agents/skills/writing-runs/SKILL.md")
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
