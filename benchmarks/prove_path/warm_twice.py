#!/usr/bin/env python3
"""A window's composes timed twice: `records_audit.py warm` into an empty compose cache, then again over the cache it filled.

A job for `research run`, with the tree whose `warm` it times as its working directory (``--cwd clone``: it syncs that tree's
venv). The first pass composes every template of the verifier's window, ``--jobs`` at a time; the second should find each one
kept. The cache holds every composed circuit at once, so it is this job's own, on a shared disk, for as long as the job runs:

- it is a fresh directory under ``--cache-parent``, on the window's file system (a later audit could take it by link), outside
  the window and outside the run directory, and it is removed on every way out: a normal end, a failed pass, the deadline, a
  signal. A guardian outside the job's process group and cgroup removes it when the job dies without running its own removal
  (SIGKILL, an OOM kill), after ending what is left of the pass. The last line of output says the cache is gone and gives the
  free space after, or says it is not gone and where it is;
- before the first pass it refuses unless ``--min-free-gb`` are free on that file system (exit 70);
- while a pass runs it reads the free space every ``--watch-every`` seconds, and under ``--floor-gb`` it ends the pass, removes
  the cache and exits 71;
- at ``--deadline-min`` after its start it ends the pass, removes the cache and exits 72. Give `research run` a longer
  ``--timeout``: its SIGTERM is followed by a SIGKILL seconds later, which a large removal can outlast.

GB is 10^9 bytes throughout; the lines it prints give GiB beside it.

Under ``$OUT`` (``$RESEARCH_RUN_DIR/out``), all small: ``result.json`` (rewritten after every step, so a killed job leaves
what it had: the commit, the command lines, each pass's wall time and its templates' seconds and whether the cache held them,
the cache's size, peak memory, the free space before pass 1, after each pass and after the removal), each pass's output
(``passN.log``, ``passN.err``) and ``sync.log``. Nothing it writes names the machine.

Peak memory is read three ways, since a pass forks one process a template and they share the window's pages: the sum of
resident sets (counts a shared page once a process: an upper bound), the sum of proportional sets (counts it once; read once a
minute, since the kernel walks every page to answer), and the anonymous memory of the cgroup the job runs in.

Exit: 0 both passes ran (``result.json`` says whether pass 2 found every template kept); 64 a refused argument; 66 the venv
didn't sync; 70 too little free space to start; 71 free space under the floor; 72 the deadline; 73 or 74 pass 1 or 2 failed;
76 the cache is still there (whatever else happened); 77 ended by a signal.

``--stop-at-gb`` and ``--stop-at-templates`` end a pass early and count it as run: a tryout of the command line on one small
circuit, with ``--jobs 1 --passes 1``. ``--stand-in`` and ``--free-from`` are the tests' (``tests/test_warm_twice.py``), which
start no unit (``--guardian detached``).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GB, GIB = 10**9, 2**30
EXIT_ARG, EXIT_SYNC, EXIT_START, EXIT_FLOOR, EXIT_DEADLINE, EXIT_PASS, EXIT_LEFT, EXIT_SIGNAL = 64, 66, 70, 71, 72, 73, 76, 77
RSS_EVERY_S, PSS_EVERY_S, END_WAIT_S = 5.0, 60.0, 60.0
SYNC = "uv sync -q --all-packages --extra torch-cpu"

# The guardian: started in a transient user unit (else detached), with no environment of the job's, so nothing that ends the
# job's group or cgroup ends it. It waits for the job's process to go; if the job's directory is still there then, the job
# never ran its removal: it kills the pass's session, which would write on, and removes the directory.
GUARDIAN = r"""
import os, shutil, signal, sys, time
pid, start, root = int(sys.argv[1]), sys.argv[2], sys.argv[3]
def stat(p):
    try:
        return open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()
    except OSError:
        return None
def alive():
    s = stat(pid)
    return s is not None and s[19] == start and s[0] != "Z"
for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
    signal.signal(s, signal.SIG_IGN)
while alive() and os.path.isdir(root):
    time.sleep(1)
for _ in range(600):
    if not os.path.isdir(root):
        break
    try:
        sid = open(os.path.join(root, "pass.sid")).read().strip()
    except OSError:
        sid = ""
    for p in [int(n) for n in os.listdir("/proc") if n.isdigit()] if sid else []:
        s = stat(p)
        if s is not None and s[3] == sid:
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
    shutil.rmtree(root, ignore_errors=True)
    time.sleep(1)
"""


class Stop(Exception):
    """A way out other than the passes' own end: its exit code and its line."""

    def __init__(self, code: int, why: str):
        super().__init__(why)
        self.code, self.why = code, why


def say(text: str) -> None:
    print(text, flush=True)


def size(n: float) -> str:
    return f"{n / GB:.2f} GB ({n / GIB:.2f} GiB)"


def proc_stat(pid: int) -> list[str] | None:
    """/proc/PID/stat's fields after the command's name (state first), or None once the process is gone."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return None


def session(sid: int) -> list[int]:
    """The live processes of session ``sid``."""
    out = []
    for name in os.listdir("/proc"):
        if name.isdigit() and (st := proc_stat(int(name))) is not None and st[0] != "Z" and int(st[3]) == sid:
            out.append(int(name))
    return out


def status_kb(pid: int, fields: tuple[str, ...], file: str = "status") -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/{file}").read_text().splitlines():
            k, _, v = line.partition(":")
            if k in fields:
                out[k] = int(v.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return out


def cgroup_dir() -> Path | None:
    try:
        rel = next(ln.split("::", 1)[1] for ln in Path("/proc/self/cgroup").read_text().splitlines() if ln.startswith("0::"))
    except (OSError, StopIteration):
        return None
    d = Path("/sys/fs/cgroup") / rel.strip().lstrip("/")
    return d if (d / "memory.stat").is_file() else None


def cgroup_anon(d: Path | None) -> int | None:
    if d is None:
        return None
    try:
        st = dict(ln.split() for ln in (d / "memory.stat").read_text().splitlines())
        return int(st["anon"]) + int(st.get("shmem", 0))
    except (OSError, ValueError, KeyError):
        return None


def tree_size(root: Path) -> dict:
    """What is under ``root``: its files' bytes, the bytes the disk holds for them, and the composed circuits (`made.json`)."""
    size_b = disk_b = files = made = 0
    stack = [str(root)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                            continue
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    size_b, disk_b, files = size_b + st.st_size, disk_b + st.st_blocks * 512, files + 1
                    made += e.name == "made.json" and ".tmp" not in os.path.basename(os.path.dirname(e.path))
        except OSError:
            continue
    return {"bytes": size_b, "disk_bytes": disk_b, "files": files, "circuits": made}


class Job:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.out = Path(os.environ.get("OUT") or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out").resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()
        self.root: Path | None = None       # this job's directory under --cache-parent: cache/ and pass.sid
        self.child: subprocess.Popen | None = None
        self.signalled: int | None = None
        self.cg = cgroup_dir()
        self.r: dict = {"schema": "warm-twice/v1", "state": "starting", "stand_in": bool(a.stand_in),
                        "settings": {"jobs": a.jobs, "passes": a.passes, "min_free_gb": a.min_free_gb, "floor_gb": a.floor_gb,
                                     "watch_every_s": a.watch_every, "deadline_min": a.deadline_min, "stop_at_gb": a.stop_at_gb,
                                     "stop_at_templates": a.stop_at_templates, "free_from_file": bool(a.free_from)},
                        "free_bytes": {}, "passes": [], "cache": {}}

    # ---- records

    def write(self) -> None:
        tmp = self.out / "result.json.tmp"
        tmp.write_text(json.dumps(self.r, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, self.out / "result.json")

    def free(self) -> int:
        """Free bytes on the cache's file system as a writer without privilege has them."""
        if self.a.free_from:
            return int(Path(self.a.free_from).read_text().split()[0])
        st = os.statvfs(self.root or self.a.cache_parent)
        return st.f_bavail * st.f_frsize

    def note_free(self, at: str) -> int:
        n = self.r["free_bytes"][at] = self.free()
        self.write()
        return n

    # ---- signals: the handler only notes the signal, and the job's loops, none of which waits over a second, read the note
    # (`check`): no step is cut between starting a process and recording it, and nothing interrupts the removal

    def on_signal(self, signum: int, _frame: object) -> None:
        if self.signalled is None:
            self.signalled = signum

    def check(self) -> None:
        if self.signalled is not None:
            raise Stop(EXIT_SIGNAL, f"ended by {signal.Signals(self.signalled).name}")
        if time.monotonic() - self.t0 >= self.a.deadline_min * 60:
            raise Stop(EXIT_DEADLINE, f"DEADLINE in {self.r['state']}: {self.a.deadline_min:g} min after the job's start")

    # ---- the pass

    def end_child(self) -> None:
        """End the running pass and every process of its session, with SIGKILL: a pass has nothing to put in order that the
        removal doesn't remove, and no writer is left when the removal starts."""
        p, self.child = self.child, None
        if p is None:
            return
        t_end = time.monotonic() + END_WAIT_S
        while True:
            running, left = p.poll() is None, session(p.pid)
            if not running and not left:
                break
            for target, kill in ((p.pid, os.killpg), *((pid, os.kill) for pid in left)):
                try:
                    kill(target, signal.SIGKILL)
                except OSError:
                    pass
            if time.monotonic() >= t_end:
                say(f"the pass's session still has processes {END_WAIT_S:g} s after SIGKILL")
                break
            time.sleep(0.2)

    def memory(self, sid: int, peak: dict, pss: bool) -> None:
        pids = session(sid)
        rss = sum(sum(status_kb(p, ("VmRSS",)).values()) for p in pids) * 1024
        peak["rss_sum_bytes"] = max(peak.get("rss_sum_bytes", 0), rss)
        peak["processes"] = max(peak.get("processes", 0), len(pids))
        if (anon := cgroup_anon(self.cg)) is not None:
            peak["cgroup_anon_bytes"] = max(peak.get("cgroup_anon_bytes", 0), anon)
        if pss:
            got = [status_kb(p, ("Pss", "Pss_Anon"), "smaps_rollup") for p in pids]
            for k, name in (("Pss", "pss_sum_bytes"), ("Pss_Anon", "pss_anon_sum_bytes")):
                peak[name] = max(peak.get(name, 0), sum(g.get(k, 0) for g in got) * 1024)
            peak["pss_samples"] = peak.get("pss_samples", 0) + 1

    def run_pass(self, n: int, argv: list[str], env: dict[str, str]) -> None:
        a, cache = self.a, self.root / "cache"  # type: ignore[operator]
        rec: dict = {"pass": n, "command": shlex.join(argv), "state": "running", "peak": {}}
        self.r["passes"].append(rec)
        self.r["state"] = f"pass {n}"
        say(f"PASS {n}: {rec['command']}")
        t0 = time.monotonic()
        with open(self.out / f"pass{n}.log", "wb") as so, open(self.out / f"pass{n}.err", "wb") as se:
            self.child = p = subprocess.Popen(argv, stdout=so, stderr=se, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        (self.root / "pass.sid").write_text(f"{p.pid}\n")  # type: ignore[operator]
        self.write()
        next_free = next_rss = next_pss = 0.0
        tick = min(1.0, a.watch_every)
        stopped = None
        try:
            while True:
                try:
                    p.wait(timeout=tick)
                    self.check()
                    break
                except subprocess.TimeoutExpired:
                    pass
                self.check()
                now = time.monotonic()
                if now >= next_free:
                    next_free = now + a.watch_every
                    free = self.free()
                    rec["free_min_bytes"] = min(rec.get("free_min_bytes", free), free)
                    if free < a.floor_gb * GB:
                        raise Stop(EXIT_FLOOR, f"LOW DISK in pass {n}: {size(free)} free, under the floor of {a.floor_gb:g} GB")
                    if a.stop_at_gb is not None or a.stop_at_templates is not None:
                        ts = tree_size(cache)
                        if a.stop_at_gb is not None and ts["disk_bytes"] >= a.stop_at_gb * GB:
                            stopped = f"the cache reached {size(ts['disk_bytes'])} (--stop-at-gb {a.stop_at_gb:g})"
                        elif a.stop_at_templates is not None and ts["circuits"] >= a.stop_at_templates:
                            stopped = f"{ts['circuits']} template(s) composed (--stop-at-templates {a.stop_at_templates})"
                        if stopped:
                            break
                if now >= next_rss:
                    pss, next_rss = now >= next_pss, now + RSS_EVERY_S
                    if pss:
                        next_pss = now + PSS_EVERY_S
                    self.memory(p.pid, rec["peak"], pss)
        finally:
            rec["wall_s"] = round(time.monotonic() - t0, 1)
            rec["exit"] = p.poll()
            self.end_child()
            (self.root / "pass.sid").unlink(missing_ok=True)  # type: ignore[operator]
            if rec["exit"] is None:
                rec["exit"] = "ended by this job"
            rec["cache"] = tree_size(cache)
            self.read_log(n, rec)
            rec["state"] = "stopped early" if stopped else "done" if rec["exit"] == 0 else "ended"
            self.write()
        say(f"PASS {n}: {rec['wall_s']} s, exit {rec['exit']}" + (f", stopped: {stopped}" if stopped else "")
            + f"; {rec.get('templates', 0)} template line(s), {rec.get('kept', 0)} kept; cache {size(rec['cache']['disk_bytes'])}"
            f" in {rec['cache']['circuits']} circuit(s)")
        if not stopped and rec["exit"] != 0:
            raise Stop(EXIT_PASS + n - 1, f"PASS {n} FAILED: exit {rec['exit']} (pass{n}.err)")

    def read_log(self, n: int, rec: dict) -> None:
        """`warm`'s lines: a template's {"template", "seconds", "kept", "k_log"}, then ``WARM`` with the totals."""
        rows: list[dict] = []
        try:
            lines = (self.out / f"pass{n}.log").read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        for ln in lines:
            try:
                if ln.startswith("WARM "):
                    rec["warm"] = json.loads(ln[5:])
                elif ln.startswith("{") and "template" in (d := json.loads(ln)):
                    rows.append({k: d.get(k) for k in ("template", "seconds", "kept", "k_log")})
            except ValueError:
                continue
        rec["rows"], rec["templates"], rec["kept"] = rows, len(rows), sum(bool(r["kept"]) for r in rows)

    # ---- the job

    def guardian(self) -> str:
        st = proc_stat(os.getpid())
        if st is None:
            return "none"
        cmd = [sys.executable, "-c", GUARDIAN, str(os.getpid()), st[19], str(self.root)]
        quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if self.a.guardian == "unit" and shutil.which("systemd-run"):
            unit = ["systemd-run", "--user", "--collect", "--quiet", f"--unit=warm-twice-guard-{os.getpid()}", "--"]
            try:
                if subprocess.run([*unit, *cmd], timeout=30, **quiet).returncode == 0:  # type: ignore[call-overload]
                    return "unit"
            except (OSError, subprocess.TimeoutExpired):
                pass
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
        subprocess.Popen(cmd, env=env, start_new_session=True, cwd="/", **quiet)  # type: ignore[call-overload]
        return "detached"

    def refuse_start(self, at: str) -> None:
        free = self.note_free(at)
        if free < self.a.min_free_gb * GB:
            raise Stop(EXIT_START, f"REFUSED: {size(free)} free on the cache's file system, and the first pass starts only "
                                   f"with {self.a.min_free_gb:g} GB ({self.a.min_free_gb * GB / GIB:.1f} GiB) or more")
        say(f"free before pass 1 ({at}): {size(free)}; needed {self.a.min_free_gb:g} GB")

    def main(self) -> int:
        a = self.a
        tree = Path.cwd()
        window, parent = Path(a.window).resolve(), Path(a.cache_parent).resolve()
        run_dir = Path(os.environ["RESEARCH_RUN_DIR"]).resolve() if os.environ.get("RESEARCH_RUN_DIR") else None
        if not a.stand_in:
            head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            self.r["commit"] = head
            if not head or (a.commit and not head.startswith(a.commit)):
                raise Stop(EXIT_ARG, f"REFUSED: the tree is at {head or '(no commit)'}, not --commit {a.commit}")
            if not window.is_dir():
                raise Stop(EXIT_ARG, f"REFUSED: no window at {window}")
            if os.stat(window).st_dev != os.stat(parent).st_dev:
                raise Stop(EXIT_ARG, "REFUSED: --cache-parent is on another file system than the window")
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            raise Stop(EXIT_ARG, f"REFUSED: --cache-parent {parent} is not a directory this job can write in")
        for inside in (window, run_dir, tree):
            if inside is not None and (parent == inside or inside in parent.parents):
                raise Stop(EXIT_ARG, f"REFUSED: --cache-parent {parent} is inside {inside}")
        self.refuse_start("at_start")

        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        if not a.stand_in:
            env["PATH"] = f"{Path.home()}/.local/bin:{Path.home()}/.cargo/bin:{env.get('PATH', '')}"
            env |= {"RAYON_NUM_THREADS": str(len(os.sched_getaffinity(0))), "RUST_MIN_STACK": str(64 << 20)}
            self.r["environment"] = {"unset": ["PYTHONPATH"], "RAYON_NUM_THREADS": env["RAYON_NUM_THREADS"],
                                     "RUST_MIN_STACK": env["RUST_MIN_STACK"], "sync": SYNC,
                                     "FLOCK": sorted(k for k in env if k.startswith("FLOCK_"))}
            self.r["state"] = "sync"
            self.write()
            t0 = time.monotonic()
            with open(self.out / "sync.log", "wb") as log:
                self.child = subprocess.Popen(shlex.split(SYNC), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                              env=env, start_new_session=True)
                while self.child.poll() is None:
                    self.check()
                    time.sleep(0.5)
                rc, self.child = self.child.returncode, None
            self.r["sync_s"] = round(time.monotonic() - t0, 1)
            if rc != 0:
                raise Stop(EXIT_SYNC, f"SYNC FAILED: `{SYNC}` exit {rc} (sync.log)")
            say(f"venv synced in {self.r['sync_s']} s")
        self.refuse_start("before_pass1")

        self.root = Path(tempfile.mkdtemp(prefix="warm-twice.", dir=parent))
        cache = self.root / "cache"
        cache.mkdir()
        self.r["cache"] = {"dir": str(cache), "guardian": self.guardian()}
        say(f"cache: {cache} (guardian: {self.r['cache']['guardian']})")
        if a.stand_in:
            argv = ["bash", "-c", a.stand_in, "stand-in", str(cache)]
        else:
            argv = [str(tree / ".venv/bin/python"), "benchmarks/prove_path/records_audit.py", "warm", "--window", str(window),
                    "--law", a.law, "--compose-cache", str(cache), "--jobs", str(a.jobs)]
        for n in range(1, a.passes + 1):
            self.check()
            self.run_pass(n, argv, env)
            self.note_free(f"after_pass{n}")
        last = self.r["passes"][-1]
        if a.passes > 1 and last["state"] == "done":
            self.r["pass2_all_kept"] = bool(last["templates"]) and last["kept"] == last["templates"]
            say(f"PASS 2: {last['kept']} of {last['templates']} template(s) were in the cache")
        return 0

    def run(self) -> int:
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, self.on_signal)
        code, why = 0, "both passes ran" if self.a.passes > 1 else "the pass ran"
        try:
            code = self.main()
        except Stop as e:
            code, why = e.code, e.why
        except Exception as e:  # noqa: BLE001  (every way out removes the cache)
            code, why = 1, f"FAILED: {type(e).__name__}: {e}"
        say(why)
        self.r["state"], self.r["why"] = "removing", why
        root, removal, free = self.root, "none was made", -1
        try:
            self.end_child()
            if root is not None:
                self.r["cache"]["at_removal"] = tree_size(root / "cache")
                self.write()
                t0 = time.monotonic()
                for _ in range(5):
                    shutil.rmtree(root, ignore_errors=True)
                    if not root.exists():
                        break
                    time.sleep(1)
                self.r["cache"]["removal_s"] = round(time.monotonic() - t0, 1)
                removal = f"in {self.r['cache']['removal_s']} s"
        except Exception as e:  # noqa: BLE001  (the last line still says whether the cache is there)
            say(f"removal: {type(e).__name__}: {e}")
        gone = root is None or not root.exists()
        if not gone:
            code = EXIT_LEFT
        try:
            self.root = None if gone else root
            free = self.r["free_bytes"]["after_removal"] = self.free()
            if self.cg is not None and (self.cg / "memory.peak").is_file():  # the cgroup's whole charge, file pages included
                self.r["cgroup_memory_peak_bytes"] = int((self.cg / "memory.peak").read_text())
            self.r["cache"]["removed"], self.r["state"], self.r["exit"] = gone, "ended", code
            self.r["wall_s"] = round(time.monotonic() - self.t0, 1)
            self.write()
        except Exception as e:  # noqa: BLE001
            say(f"result.json: {type(e).__name__}: {e}")
        if gone:
            say(f"CACHE REMOVED ({removal}); free disk after: {size(free)}; exit {code}")
        else:
            say(f"CACHE NOT REMOVED: {root} is still there; free disk: {size(free)}; exit {code}")
        return code


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    p.add_argument("--window", required=True, help="the window's directory, as `records_audit.py warm` takes it")
    p.add_argument("--cache-parent", required=True, help="where this job makes its cache: a directory on the window's file "
                   "system, outside the window, the tree and the run directory")
    p.add_argument("--law", default="subset:1", help="the window's law, as its audit takes it")
    p.add_argument("--jobs", type=int, default=24, help="templates composed at once (`warm --jobs`)")
    p.add_argument("--passes", type=int, default=2, choices=(1, 2))
    p.add_argument("--commit", help="refuse unless the tree's commit starts with this")
    p.add_argument("--min-free-gb", type=float, default=547.0, help="free GB (10^9 bytes) the first pass needs to start")
    p.add_argument("--floor-gb", type=float, default=15.0, help="free GB under which a running pass is ended")
    p.add_argument("--watch-every", type=float, default=30.0, help="seconds between two reads of the free space")
    p.add_argument("--deadline-min", type=float, default=85.0, help="minutes after its start at which the job ends itself")
    p.add_argument("--stop-at-gb", type=float, help="tryout: end a pass once its cache holds this much")
    p.add_argument("--stop-at-templates", type=int, help="tryout: end a pass once this many templates are composed")
    p.add_argument("--guardian", default="unit", choices=("unit", "detached"), help="how the guardian is started: in a transient "
                   "user unit, outside the job's cgroup (a machine without one falls back to detached), or only detached")
    p.add_argument("--stand-in", help="tests: a shell command run in `warm`'s place, the cache's directory its $1")
    p.add_argument("--free-from", help="tests: a file holding the free bytes to read in the file system's place")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(Job(parse()).run())
