#!/usr/bin/env python3
"""Node operations for a shared GPU server: an OOM guard, one status page, and alerts (stdlib only; runs as the job user).

    node_ops.py            the loop: the guard every GUARD_S (5 s); the page and the alerts every PAGE_S (60 s)
    node_ops.py --page     write the page once and print it

OOM guard.  The kernel's OOM killer picks by size alone and can take the lease loop (which stops the VM on time), sshd sessions or
the samplers.  When MemAvailable falls under ACT_PCT of MemTotal, the guard stops the largest process group of this user by
resident memory, outside PROTECT: SIGTERM, then SIGKILL after 10 s.  Under WARN_PCT it only alerts.  Every OOM kill anywhere
(/proc/vmstat oom_kill, including a gpu-lease job killed at its own cgroup cap) is an alert, with the user journal's line naming
the unit.

Page: OUT/status.md, with who holds which GPU (gpu-lease status), the GPUs' latest sample (gpu_util_sampler's JSONL, so no NVML
query here and none during a timed window), waiters, the fill queue, research runs live and failed in the last 2 h, CPU, RAM,
disks, and open alerts.

Alerts: OUT/alerts.jsonl, one JSON line per alert, repeated at most once an hour per key: disk at or over DISK_PCT, memory low,
OOM kills, the guard's kills, a run marked running whose process is gone, a failed fill job, a sampler or fill runner gone quiet.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(os.environ.get("OPS_DIR", "/workspace/pouw/infra"))
UTIL = Path(os.environ.get("UTIL_OUT_DIR", "/workspace/pouw/infra/util"))
FILL = Path(os.environ.get("FILL_DIR", "/workspace/pouw/fill"))
RUNS = Path(os.environ.get("RUNS_DIR", "/workspace/research/runs"))
LEASE_FILE = Path(os.environ.get("LEASE_FILE", "/home/research/.research/lease"))
DEADLINE_FILE = Path(os.environ.get("LEASE_DEADLINE_FILE", "/etc/research/deadline"))
GUARD_S, PAGE_S = 5.0, 60.0
WARN_PCT, ACT_PCT, DISK_PCT = 15.0, 5.0, 60.0
PROTECT = re.compile(r"lease\.sh|vy-lease|gpu_util_sampler|fill_runner|node_ops|tmux|sshd|systemd|dbus|\(sd-pam\)|bash -l$|-bash$")
ME = os.getpid()
_last_alert: dict[str, float] = {}


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime | None = None) -> str:
    return (t or now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def alert(kind: str, key: str, msg: str, **kw) -> None:
    k = f"{kind}:{key}"
    if time.monotonic() - _last_alert.get(k, -1e9) < 3600:
        return
    _last_alert[k] = time.monotonic()
    with open(OUT / "alerts.jsonl", "a") as f:
        f.write(json.dumps({"t": iso(), "kind": kind, "key": key, "msg": msg, **kw}) + "\n")


def meminfo() -> dict[str, int]:
    return {l.split(":")[0]: int(l.split()[1]) * 1024 for l in Path("/proc/meminfo").read_text().splitlines() if l.split()[1:]}


def in_verity_scope(pid: int) -> bool:
    """Verity's guest CPU jobs run in `fill-verity-*` scopes (fill_runner.py); the OOM guard stops them before any pous work."""
    try:
        return "fill-verity-" in Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return False


TIMING_SLICE = os.environ.get("FILL_TIMING_SLICE", "vy-timing.slice")


def in_timing_slice(pid: int) -> bool:
    """Fill's timing jobs run in scopes of the top-level FILL_TIMING_SLICE (fill_runner.py, through root's systemd-run), on cores
    of their own.  A slice of that name under a user's manager (`systemd-run --user --slice=...`) is anyone's to make."""
    try:
        return any(l.startswith(f"0::/{TIMING_SLICE}/") for l in Path(f"/proc/{pid}/cgroup").read_text().splitlines())
    except OSError:
        return False


def in_lease_scope(pid: int) -> bool:
    """A GPU lease's job runs in a `gpu-lease-<pid>.scope` (gpu_lease.sh): Quiet leaves it, and every group its builds start (ninja
    gives each compile a group of its own), running, since a paused build can hold a lock a timed lease's job then waits on."""
    try:
        return "/gpu-lease-" in Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return False


def oom_kills() -> int:
    for line in Path("/proc/vmstat").read_text().splitlines():
        if line.startswith("oom_kill "):
            return int(line.split()[1])
    return 0


def groups() -> dict[int, dict]:
    """This user's process groups: resident bytes and the leader's command line."""
    uid, page = os.getuid(), os.sysconf("SC_PAGE_SIZE")
    out: dict[int, dict] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if d.stat().st_uid != uid:
                continue
            stat = (d / "stat").read_text()
            pgid = int(stat.rsplit(")", 1)[1].split()[2])
            rss = int((d / "statm").read_text().split()[1]) * page
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (OSError, ValueError, IndexError):
            continue
        g = out.setdefault(pgid, {"rss": 0, "cmds": [], "pids": []})
        g["rss"] += rss
        g["pids"].append(int(d.name))
        if int(d.name) == pgid or not g["cmds"]:
            g["cmds"].insert(0, cmd)
    return out


class Guard:
    def __init__(self) -> None:
        self.ooms = oom_kills()
        self.pending: dict[int, float] = {}

    def tick(self) -> None:
        m = meminfo()
        total, avail = m["MemTotal"], m["MemAvailable"]
        pct = 100 * avail / total
        n = oom_kills()
        if n > self.ooms:
            j = subprocess.run(["journalctl", "--user", "--since", "-3min", "--no-pager", "-o", "cat", "-g", "oom"],
                               capture_output=True, text=True).stdout.strip().splitlines()[-3:]
            alert("oom-kill", str(n), f"{n - self.ooms} OOM kill(s) (vmstat oom_kill {self.ooms} -> {n}); journal: {' | '.join(j) or '-'}")
            self.ooms = n
        if pct < WARN_PCT:
            alert("mem-low", "node", f"MemAvailable {avail / 2**30:.0f} GiB ({pct:.1f}%) of {total / 2**30:.0f} GiB")
        for pgid, t0 in list(self.pending.items()):
            if time.monotonic() - t0 > 10:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                del self.pending[pgid]
        if pct >= ACT_PCT or self.pending:
            return
        gs = {k: v for k, v in groups().items() if k != os.getpgid(ME) and not PROTECT.search(" ".join(v["cmds"][:1]))}
        if not gs:
            return
        guests = {k: v for k, v in gs.items() if in_verity_scope(k)}
        pgid, g = max((guests or gs).items(), key=lambda kv: kv[1]["rss"])
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        self.pending[pgid] = time.monotonic()
        rec = {"t": iso(), "pgid": pgid, "rss_gib": round(g["rss"] / 2**30, 1), "cmd": g["cmds"][0][:300],
               "mem_available_gib": round(avail / 2**30, 1)}
        with open(OUT / "logs" / "oom-guard.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        alert("oom-guard", str(pgid), f"stopped process group {pgid} ({rec['rss_gib']} GiB): {rec['cmd'][:160]}; "
                                      f"MemAvailable was {rec['mem_available_gib']} GiB", **rec)


LEASE_DIR = Path(os.environ.get("GPU_LEASE_DIR", "/run/gpu-lease"))
BUILD = {"lake", "lean", "leanc", "cargo", "rustc", "cc1", "cc1plus", "ld", "ld.lld", "make", "ninja", "nvcc", "ptxas", "cicc", "xz"}
QUIET_MATCH = re.compile(os.environ["QUIET_MATCH"]) if os.environ.get("QUIET_MATCH") else None
# a train check (slot.py's group, check.py under it) runs straight through a timed window (top 24 Oct 04:44Z, landing first): on
# n2 at 05:16Z one --timed lease froze train eda3's check and c567's post-landing check for 28 min
QUIET_EXEMPT = re.compile(r"-m research |research launch-request|/workspace/pouw/fill/running/|timeout |tools/verity/check/(check|slot)\.py")
# of those, the ones whose descendants in groups of their own (ninja's compiles, a check's subprocesses) are spared too; a research
# runner's are not, since every run's workload descends from one
QUIET_EXEMPT_TREE = re.compile(r"/workspace/pouw/fill/running/|timeout |tools/verity/check/(check|slot)\.py")
# `research run`'s per-run sampler polls nvidia-smi every 5 s for the whole life of a run, the timed run's own included,
# so every one is paused for a window whatever its CPU use (the pous root, 17:19Z); its run records the gap.
SAMPLER = re.compile(r"-m research\.telemetry sample ")


class Quiet:
    """While a timed window holds GPUs (a lease tagged timed=1, or every GPU held by one holder), stop this user's CPU-heavy
    process groups other than the window's own, and continue them once it ends.  CPU-heavy: at least one core busy over the last
    tick, or a build, Lean or cargo process in the group.  Left running: the window's process tree, research runners, fill jobs
    (their runner pauses them), fill's timing jobs (a pause voids their chunk, and nothing stops them before their max_min),
    anything under GNU `timeout` (it would lose the paused time), train checks (slot.py, check.py), every process a GPU lease runs
    (`in_lease_scope`), the descendants of those in groups of their own, and PROTECT's daemons.  Measured
    reason: CPU load on the other NUMA node slows decode-shape GEMM timing by 0.3-1.4% (run r20260930-075259-37cc)."""

    STATE = OUT / "logs" / "quiet-paused.json"

    def __init__(self) -> None:
        self.paused: dict[int, dict] = {}
        self.ticks: dict[int, int] = {}
        self.t = time.monotonic()
        self.hz = os.sysconf("SC_CLK_TCK")
        try:                                   # groups an earlier instance stopped: kept stopped while the window lasts, else continued
            earlier = [int(p) for p in json.loads(self.STATE.read_text())]
        except (OSError, ValueError):
            earlier = []
        open_window = self.window() is not None
        for pgid in earlier:
            try:
                if open_window:
                    os.killpg(pgid, 0)
                    self.paused[pgid] = {"t0": time.monotonic(), "cmd": "(adopted after an ops restart)"}
                else:
                    os.killpg(pgid, signal.SIGCONT)
                    self.log(event="resume", pgid=pgid, why="ops restart")
            except ProcessLookupError:
                pass
        self.save()

    def save(self) -> None:
        self.STATE.write_text(json.dumps(sorted(self.paused)))

    def window(self) -> int | None:
        held = []
        for lock in sorted(LEASE_DIR.glob("*.lock")):
            if subprocess.run(["flock", "-n", "-s", str(lock), "true"], capture_output=True).returncode == 0:
                held.append(None)
                continue
            try:
                held.append((LEASE_DIR / (lock.stem + ".owner")).read_text().strip())
            except OSError:
                held.append("")
        pids = [re.match(r"pid=(\d+)", o or "") for o in held]
        for o, m in zip(held, pids):
            if o and m and " timed=1 " in f" {o} ":
                return int(m.group(1))
        if held and all(held) and len({m.group(1) for m in pids if m}) == 1 and all(pids):
            return int(pids[0].group(1))
        return None

    def log(self, **kw) -> None:
        with open(OUT / "logs" / "quiet.jsonl", "a") as f:
            f.write(json.dumps({"t": iso(), **kw}) + "\n")

    def resume(self, why: str) -> None:
        for pgid, info in list(self.paused.items()):
            try:
                os.killpg(pgid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            self.log(event="resume", pgid=pgid, why=why, cmd=info["cmd"][:200], paused_s=round(time.monotonic() - info["t0"]))
            del self.paused[pgid]
        self.save()

    def tick(self) -> None:
        now_t = time.monotonic()
        dt, self.t = max(now_t - self.t, 1e-3), now_t
        uid, procs = os.getuid(), {}
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                if d.stat().st_uid != uid:
                    continue
                st = (d / "stat").read_text().rsplit(")", 1)
                comm = st[0].split("(", 1)[1]
                f = st[1].split()
                procs[int(d.name)] = {"ppid": int(f[1]), "pgid": int(f[2]), "cpu": int(f[11]) + int(f[12]), "comm": comm,
                                      "cmd": (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()}
            except (OSError, ValueError, IndexError):
                continue
        holder = self.window()
        if holder is None:
            if self.paused:
                self.resume("window over")
            self.ticks = {p: v["cpu"] for p, v in procs.items()}
            return
        mine, p = set(), ME                                      # this process's own ancestors seed nothing (a test under `timeout`)
        while p in procs and p not in mine:
            mine.add(p)
            p = procs[p]["ppid"]
        tree = {holder} | {p for p, v in procs.items() if p not in mine and (QUIET_EXEMPT_TREE.search(v["cmd"]) or in_lease_scope(p))}
        grew = True
        while grew:
            grew = False
            for p, v in procs.items():
                if v["ppid"] in tree and p not in tree:
                    tree.add(p)
                    grew = True
        groups: dict[int, dict] = {}
        for p, v in procs.items():
            g = groups.setdefault(v["pgid"], {"rate": 0.0, "comms": set(), "cmds": [], "pids": set()})
            g["rate"] += max(0, v["cpu"] - self.ticks.get(p, v["cpu"])) / self.hz / dt
            g["comms"].add(v["comm"])
            g["pids"].add(p)
            g["cmds"].append(v["cmd"])
        self.ticks = {p: v["cpu"] for p, v in procs.items()}
        me = os.getpgid(ME)
        for pgid, g in groups.items():
            if pgid in self.paused or pgid == me or g["pids"] & tree:
                continue
            text = " | ".join(g["cmds"])
            if PROTECT.search(text) or QUIET_EXEMPT.search(text) or (QUIET_MATCH and not QUIET_MATCH.search(text)):
                continue
            if (g["rate"] >= 1.0 or g["comms"] & BUILD or SAMPLER.search(text)) and not any(map(in_timing_slice, g["pids"])):
                try:
                    os.killpg(pgid, signal.SIGSTOP)
                except (ProcessLookupError, PermissionError):
                    continue
                cmd = next((c for c in g["cmds"] if c), "?")
                self.paused[pgid] = {"t0": time.monotonic(), "cmd": cmd}
                self.save()
                self.log(event="pause", pgid=pgid, holder=holder, cores=round(g["rate"], 2), cmd=cmd[:200])


QUIET: "Quiet | None" = None


def busy_last_hour() -> str:
    """Busy share of GPU time over the last 60 min of samples: a kernel resident, DCGM SM activity >= 1%, or a timed window."""
    cutoff = (now() - __import__("datetime").timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    busy = total = 0
    for f in sorted(UTIL.glob("*.jsonl"))[-2:]:
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["t"] < cutoff:
                continue
            for g in r.get("gpus", []):
                total += 1
                busy += bool(r.get("timed") or (g.get("util") or 0) >= 1 or (g.get("smact") or 0) >= 0.01)
    return f"{100 * busy / total:.0f}% of GPU time busy over the last hour (target 80%)" if total else "no samples in the last hour"


def last_sample() -> dict | None:
    files = sorted(UTIL.glob("*.jsonl"))
    if not files:
        return None
    with open(files[-1], "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 65536))
        lines = f.read().decode(errors="replace").strip().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def alive(pid: int | None, ticks: int | None) -> bool:
    if not pid:
        return False
    try:
        st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return False
    return ticks is None or int(st[19]) == int(ticks)


def runs(window_s: float = 7200) -> tuple[list[dict], list[dict]]:
    live, failed = [], []
    cutoff = time.time() - window_s
    for d in sorted(RUNS.glob("r2026*")):
        try:
            if d.stat().st_mtime < cutoff:
                continue
            st = json.loads((d / "status.json").read_text())
        except (OSError, ValueError):
            continue
        # the runner's transition, the last naming a pid: `tele set-timeout` appends a `running` one without
        trs = [t for t in st.get("transitions") or [] if isinstance(t, dict)]
        tr = next((t for t in reversed(trs) if t.get("pid") is not None), trs[-1] if trs else {})
        cmd = " ".join(tr.get("command") or [])[:120]
        if st.get("state") == "running":
            ok = alive(tr.get("pid"), tr.get("starttime_ticks"))
            live.append({"run": d.name, "since": (tr.get("started_utc") or "")[:19], "cmd": cmd, "alive": ok})
            if not ok:
                alert("dead-run", d.name, f"run {d.name} is marked running but its process {tr.get('pid')} is gone: {cmd}")
        else:
            try:
                cls = json.loads((d / "failure.json").read_text()).get("class")
            except (OSError, ValueError):
                cls = None
            if cls and cls != "SUCCESS":
                failed.append({"run": d.name, "class": cls, "cmd": cmd})
    return live, failed


def disks() -> list[tuple[str, float, float]]:
    out = []
    for p in ("/", "/workspace"):
        s = os.statvfs(p)
        used = 100 * (1 - s.f_bavail / s.f_blocks)
        out.append((p, used, s.f_bavail * s.f_frsize / 2**40))
        if used >= DISK_PCT:
            alert("disk", p, f"{p} is {used:.0f}% full ({s.f_bavail * s.f_frsize / 2**30:.0f} GiB free)")
    return out


def stop_at() -> datetime | None:
    """When lease.sh stops the VM: the lease's expiry, never later than 5 min before root's deadline (both `EPOCH ...`)."""
    ends = []
    for f, margin in ((LEASE_FILE, 0), (DEADLINE_FILE, 300)):
        try:
            ends.append(int(f.read_text().split()[0]) - margin)
        except (OSError, ValueError, IndexError):
            pass
    return datetime.fromtimestamp(min(ends), timezone.utc) if ends else None


def fill_alerts() -> None:
    """The fill runner's last 50 events that need a person: a failed job, and a timing job's scope that root's systemctl couldn't
    kill (its processes keep the GPU's lease until someone does)."""
    for line in (FILL / "events.jsonl").read_text().splitlines()[-50:] if (FILL / "events.jsonl").exists() else []:
        e = json.loads(line)
        if e.get("event") == "failed":
            alert("fill-failed", e["job"], f"fill job {e['job']} (owner {e.get('owner')}) failed rc={e.get('rc')}")
        elif e.get("event") == "scope-kill-failed":
            alert("fill-scope-kill-failed", e["job"], f"fill couldn't {e.get('signal')} {e.get('unit')}.scope (job {e['job']}): "
                  f"what is left in it holds its GPU's lease; `sudo systemctl kill --signal=SIGKILL {e.get('unit')}.scope`")


def page() -> str:
    t = now()
    stop = stop_at()
    left = (stop - t).total_seconds() / 3600 if stop else float("nan")
    lease = subprocess.run(["gpu-lease", "status"], capture_output=True, text=True).stdout.strip().splitlines()
    s = last_sample() or {}
    age = (t - datetime.strptime(s["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds() if s else 1e9
    if age > 180:
        alert("stale", "sampler", f"gpu_util_sampler's last sample is {age / 60:.0f} min old")
    fill = (FILL / "status.txt").read_text().strip().splitlines() if (FILL / "status.txt").exists() else ["(no fill runner)"]
    if (FILL / "status.txt").exists() and time.time() - (FILL / "status.txt").stat().st_mtime > 180:
        alert("stale", "fill", "the fill runner's status is over 3 min old")
    fill_alerts()
    live, failed = runs()
    m = meminfo()
    L = [f"# vy-nebius-2 at {iso(t)}", "",
         (f"Stops itself at {stop:%Y-%m-%dT%H:%MZ} (lease and deadline): {left:.1f} h, {8 * left:.0f} GPU-h left. " if stop else
          "Stop time unknown: the lease and deadline files are unreadable. ") +
         f"Clocks locked at 2,100 MHz. Refreshed every minute by node_ops.py; `cat /workspace/pouw/infra/status.md`.", ""]
    alerts = [json.loads(l) for l in (OUT / "alerts.jsonl").read_text().splitlines()[-20:]] if (OUT / "alerts.jsonl").exists() else []
    recent = [a for a in alerts if (t - datetime.strptime(a["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds() < 7200]
    paused = [f"- pgid {k} for {int((time.monotonic() - v['t0']) / 60)} min: `{v['cmd'][:120]}`" for k, v in (QUIET.paused.items() if QUIET else [])]
    hourly = (OUT / "hourly.txt").read_text().strip().splitlines()[-8:] if (OUT / "hourly.txt").exists() else []
    L += [f"**Utilisation:** {busy_last_hour()}.", "", "By hour (the infra lane's hourly record):", ""] + \
         ([f"- {h}" for h in hourly] or ["- none yet"]) + ["",
          "## Paused for a timed window", ""] + (paused or ["- none"]) + [""]
    L += ["## Alerts (last 2 h)", ""] + ([f"- {a['t'][11:16]}Z {a['kind']}: {a['msg']}" for a in recent] or ["- none"]) + [""]
    L += ["## GPUs", "", f"Sample at {s.get('t', '-')}{' (timed window: no NVML queries)' if s.get('timed') else ''}.", "",
          "| GPU | UUID | util % | SM % | mem GiB | W | MHz | lease |", "|---|---|---|---|---|---|---|---|"]
    held = {int(l.split()[0]): " ".join(l.split()[1:]) for l in lease if l[:1].isdigit()}
    for g in s.get("gpus", []):
        mem = f"{g['mem_mib'] / 1024:.0f}" if g.get("mem_mib") is not None else "-"
        sm = f"{100 * g['smact']:.0f}" if g.get("smact") is not None else "-"
        L.append(f"| {g['i']} | {str(g.get('uuid', '-'))[4:12]} | {g.get('util', '-')} | {sm} | {mem} | {g.get('power_w', '-')} | "
                 f"{g.get('sm_mhz', '-')} | {held.get(g['i'], '-')[:110]} |")
    waits = [l for l in lease if l.startswith("waiting")]
    L += ["", "## Waiting for GPUs", ""] + ([f"- {w}" for w in waits] or ["- none"])
    L += ["", "## Fill queue", "", "~~~text", *fill, "~~~"]
    L += ["", f"## Research runs live ({len(live)})", ""] + [
        f"- {r['run']} since {r['since'][11:16]}Z{'' if r['alive'] else ' **DEAD**'}: `{r['cmd']}`" for r in live] + [""]
    L += [f"## Runs failed in the last 2 h ({len(failed)})", ""] + [f"- {r['run']} {r['class']}: `{r['cmd']}`" for r in failed[-15:]] + [""]
    L += ["## Host", "",
          f"- CPU busy {100 * (s.get('cpu_busy') or 0):.0f}% of 192 vCPU, load {s.get('load1', '-')}",
          f"- RAM {(m['MemTotal'] - m['MemAvailable']) / 2**30:.0f} of {m['MemTotal'] / 2**30:.0f} GiB used; OOM kills since boot {oom_kills()}",
          *[f"- disk {p}: {u:.0f}% used, {free:.2f} TiB free" for p, u, free in disks()], ""]
    text = "\n".join(L) + "\n"
    tmp = OUT / ".status.md.tmp"
    tmp.write_text(text)
    tmp.replace(OUT / "status.md")
    return text


def main() -> None:
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    if "--page" in sys.argv:
        print(page())
        return
    guard, quiet, last_page = Guard(), Quiet(), 0.0
    global QUIET
    QUIET = quiet
    while True:
        try:
            guard.tick()
        except Exception as e:  # noqa: BLE001
            print(f"guard: {e}", file=sys.stderr, flush=True)
        try:
            quiet.tick()
        except Exception as e:  # noqa: BLE001
            print(f"quiet: {e}", file=sys.stderr, flush=True)
        if time.monotonic() - last_page > PAGE_S:
            last_page = time.monotonic()
            try:
                page()
            except Exception as e:  # noqa: BLE001
                print(f"page: {e}", file=sys.stderr, flush=True)
        time.sleep(GUARD_S)


if __name__ == "__main__":
    main()
