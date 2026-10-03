"""Monitors: invariants in code, each with an owner, a severity, a threshold and an action (the infra vision's Signals surface).
One registry (MONITORS) and one runner. Node 1's service (`vy-monitors.timer`, pods/nebius/monitoring/) runs the node-side
monitors and writes each incident once when it opens and once when it clears; the relay on infra's VM runs the relay-side ones
(they need GitHub, the evidence store or Slack) and posts both sets to #agent-alerts.

    research monitors list [--where node|relay] [--json]
    research monitors check [ID ...] [--where node|relay] [--json]     evaluate now and write nothing
    research monitors run --by WHO [--where node|relay] [--state PATH] [--alerts DIR] [--only ID ...] [--json]
    research monitors relay --by WHO [--node1 HOST] [--node-alerts DIR] [--every 60] [--once] [--dry-run]

**A monitor** has an id, an owner handle, a severity (`warn`, or `page` when work stops), where it runs, how often, a check that
returns findings, and the action its owner takes. A finding's `key` is stable while the incident lasts: the runner opens an
incident once a key has been found for `for_s`, and clears it once the key has been gone for `clear_s`. An `events` monitor's
findings are posted once each and never cleared; its first pass only remembers what is already there. A check that raises
`Skip` (node 2 in a timed window) changes nothing. One that fails otherwise keeps its incidents open, and once it has failed
for `unread_s` it opens `<id>|unreadable` for infra.

**Sinks.** Each open and clear is a line of `<alerts>/monitors.jsonl` (`research/monitors-event/v1`), and each open also a note
in `<alerts>/outbox/` in alert_sink.py's format, so one relay posts these and Grafana's alike. A write there that fails opens or
clears nothing, so the next pass tries again. With `$VY_MONITORS_SLACK_WEBHOOK` set (an incoming webhook; its URL is never
logged), the runner also posts each event there, retrying on later passes until Slack takes it, and the relay leaves those
events alone.

**Node 2** is read from node 1 over ssh (`$VY_N2`, default `vy-n2`: the vy-cluster key), each command behind a guard in the same
ssh call: anything but a fill status under 5 minutes old saying `timed False` is a Skip, and the command doesn't run.

`relay` runs the relay-side monitors into a local alert dir, then posts every new event of node 1's alert dir (over ssh) and the
local one to #agent-alerts as `vy-monitors`, tagging the owners; each clear is a reply in its alert's thread. An owner other
than infra also gets a pointer in #agent-coordination, where its doorbell rings (#agent-alerts rings only infra). Grafana's
notes in node 1's outbox are posted too. On its first pass over an alert dir it posts only the incidents still open.

Exit codes: 0 nothing open (`check`: nothing found), 1 incidents open (`check`: a finding or a failed check), 2 usage, 3 refused
(another runner holds the state's lock).
"""
from __future__ import annotations

import argparse
import calendar
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SCHEMA = "research/monitors-{}/v1"
STAMP = "%Y-%m-%dT%H:%M:%SZ"
ALERTS = "/workspace/monitoring/alerts"
STATE = "/workspace/monitoring/monitors/state.json"
LOCAL = Path.home() / ".research" / "monitors"
WEBHOOK_ENV = "VY_MONITORS_SLACK_WEBHOOK"
POSTER = "vy-monitors"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
NICE = ["nice", "-n", "19", "ionice", "-c3"]
SKIPPED = "vy-monitors: skipped:"
FILL_STATUS = "/workspace/pouw/fill/status.txt"
FILL_FRESH_S = 300
N2_GUARD = (f's=$(head -n 1 {FILL_STATUS} 2>/dev/null); a=$(( $(date +%s) - $(date -d "${{s%% *}}" +%s 2>/dev/null || echo 0) )); '
            f'case "$s" in *"timed False"*) [ "$a" -lt {FILL_FRESH_S} ] || {{ echo "{SKIPPED} fill status ${{a}} s old"; exit 0; }};; '
            f'*) echo "{SKIPPED} ${{s:-no fill status}}"; exit 0;; esac; ')
HANDLES = ("infra", "proofs", "circuits", "compute-accounting", "memory-accounting", "network-accounting", "console")
EVENTS_KEEP_S = 14 * 86400
RETRY_KEEP_S = 86400


class Skip(Exception):
    """The check can't look now, by design (node 2 in a timed window): nothing opens or clears."""


@dataclass(frozen=True)
class Finding:
    key: str
    text: str
    owners: tuple[str, ...] = ()        # default: the monitor's owner
    severity: str | None = None         # default: the monitor's
    cleared: str | None = None          # what its clear says (default: "Cleared:" and the alert's first line)
    for_s: float | None = None          # default: the monitor's


@dataclass(frozen=True)
class Monitor:
    id: str
    owner: str
    severity: str
    where: str
    check: Callable[["Ctx"], list[Finding]]
    action: str
    every_s: float = 60
    for_s: float = 0
    clear_s: float = 0
    events: bool = False
    unread_s: float = 1800

    @property
    def doc(self) -> str:
        return " ".join((self.check.__doc__ or "").split())


@dataclass
class Ctx:
    """What a check reads through: the clock, its own memo (kept in the state between passes), commands, and files under `root`."""
    now: float
    memo: dict = field(default_factory=dict)
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    root: Path = Path("/")
    n2_host: str = "vy-n2"
    node1: str = "n1"
    node_state: str = STATE
    machines: tuple[str, ...] = ("vy-nebius-1", "vy-nebius-2")
    research: tuple[str, ...] = (sys.executable, "-m", "research")

    def path(self, p: str) -> Path:
        return self.root / p.lstrip("/")

    def sh(self, argv: list[str], *, input: str | None = None, timeout: float = 60, ok: tuple[int, ...] = (0,)) -> str:
        r = self.run(argv, input=input, capture_output=True, text=True, timeout=timeout)
        if r.returncode not in ok:
            what = argv[-1] if argv[0] == "ssh" else " ".join(argv[:4])
            raise RuntimeError(f"`{what[-120:]}` exited {r.returncode}: {(r.stderr or r.stdout or '').strip()[-300:]}")
        return r.stdout

    def n2(self, script: str, *, input: str | None = None, timeout: float = 120) -> str:
        out = self.sh([*SSH, self.n2_host, N2_GUARD + script], input=input, timeout=timeout)
        if out.startswith(SKIPPED):
            raise Skip(out.strip())
        return out


def iso(t: float) -> str:
    return time.strftime(STAMP, time.gmtime(t))


def epoch(s: str) -> float:
    return float(calendar.timegm(time.strptime(s, STAMP)))


def fields(line: str) -> dict[str, str]:
    """gpu-lease's owner and wait lines: `k=v ...`, then `cmd=` to the end."""
    head = line.split(" cmd=", 1)
    f = dict(kv.split("=", 1) for kv in head[0].split() if "=" in kv)
    if len(head) == 2:
        f["cmd"] = head[1]
    return f


def tail(path: Path, nbytes: int = 1 << 18) -> list[str]:
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        lines = f.read().decode(errors="replace").splitlines()
    return lines[1:] if size > nbytes else lines


def gpu_list(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        if a.isdigit():
            out.update(range(int(a), int(b) + 1) if b.isdigit() else {int(a)})
    return out


# ---- the watermarks (the resource steward's probe, pods/nebius/resource_probe.py) ----------------------------------------------

def _probe():
    from .pods.nebius import resource_probe

    return resource_probe


def norm(line: str) -> str:
    """The resource steward's dedupe key for a breach line (its tick.sh): numbers become `#`, a GPU's index stays."""
    return re.sub(r"(^|[^<0-9])[0-9][0-9.,]*", r"\1#", re.sub(r"GPU ([0-9]+)", r"GPU<\1>", line))


def watermarks(node: str) -> Callable[[Ctx], list[Finding]]:
    name = {"n1": "Node 1", "n2": "Node 2"}[node]

    def check(ctx: Ctx) -> list[Finding]:
        rp = _probe()
        if node == "n1":
            out = ctx.sh([*NICE, sys.executable, rp.__file__, "--node", "n1", "--reading"], timeout=150)
        else:
            out = ctx.n2(f"{' '.join(NICE)} python3 - --node n2 --reading", input=Path(rp.__file__).read_text(), timeout=180)
        r = json.loads(out)
        lim = {**rp.COMMON, **rp.LIMITS[node]}
        base = ctx.memo.get("rate_base")
        lines = rp.evaluate(r, base, lim)
        ctx.memo["rate_base"] = rp.next_base(r, base, lim)
        found = [Finding(norm(ln), f"*{name}: {ln}*", severity="page" if ln.startswith("HARD") else None) for ln in lines]
        return found + [Finding("error:" + e.split(":")[0], f"*{name}'s resource probe can't read {e.split(':')[0]}*: {e}",
                                for_s=1200) for e in r.get("errors") or []]

    check.__doc__ = (f"{name}'s disk, inode, RAM, replay-RAM, idle-GPU-memory and custody-backlog watermarks (resource_probe.py "
                     "--reading; the inode rate's base reading kept between passes). A HARD line pages.")
    return check


# ---- node 1's lease pools and dispatcher (pods/nebius/n1_alerts.py) -------------------------------------------------------------

def stalls_n1(ctx: Ctx) -> list[Finding]:
    """Node 1's lease pool blocked, or its dispatcher failing its last 3 ticks or not ticking for 10 minutes (n1_alerts.py's
    incidents, each to its owner)."""
    from .pods.nebius import n1_alerts

    snap = json.loads(ctx.sh([sys.executable, "-"], input=n1_alerts.REMOTE, timeout=60))
    return [Finding(k, text, owners=tuple(who), cleared=cleared) for k, (who, text, cleared) in n1_alerts.incidents(snap).items()]


# ---- the held-idle panel (node 1's held-idle-hourly.jsonl, which verity-console.service writes) ---------------------------------

HELD_IDLE = "/workspace/usage/held-idle-hourly.jsonl"
HELD_IDLE_RECENT_S = 86400      # well inside EVENTS_KEEP_S, so an hour the state has forgotten is never posted again


def _hour_end(r: dict) -> float | None:
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:00Z", str(r.get("hour"))):
        return None
    return calendar.timegm(time.strptime(r["hour"], "%Y-%m-%dT%H:00Z")) + 3600


def _held_rows(ctx: Ctx) -> list[dict]:
    try:
        lines = tail(ctx.path(HELD_IDLE))[-400:]
    except FileNotFoundError:
        return []
    rows = []
    for ln in lines:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("hour"):
            rows.append(r)
    return rows


def held_idle(ctx: Ctx) -> list[Finding]:
    """A lane above its held-idle target whose share fell by less than 5 points from the hour before: once per hour completed
    in the last day, node and lane, to that lane."""
    out = []
    for r in _held_rows(ctx):
        t, pct, ch, end = r.get("target_pct"), r.get("held_idle_pct"), r.get("change_pts"), _hour_end(r)
        if end is None or ctx.now - end > HELD_IDLE_RECENT_S:
            continue
        if t is None or pct is None or pct <= t or (ch is not None and ch <= -5):
            continue
        lane = r.get("lane") or "?"
        moved = "no reading for the hour before" if ch is None else f"{'+' if ch > 0 else ''}{ch} points from the hour before"
        p = r.get("phases_gpu_h")
        phases = (f"\nWhere the held time went, in GPU-hours: {p.get('leases')} Commit leases spent {p.get('engine.build')} in "
                  f"engine.build, {p.get('prep.warmup_instrumented')} in prep.warmup_instrumented, "
                  f"{p.get('validate.weights_of_record_digests')} in validate.weights_of_record_digests, {p.get('work')} in work "
                  f"and {p.get('other')} elsewhere in the lease. {p.get('outside_leases')} was held outside any Commit lease."
                  ) if isinstance(p, dict) else ""
        out.append(Finding(f"{r['hour']}|{r.get('node')}|{lane}",
                           f"*Held-idle alert, {r.get('node')}, {lane}, the {r['hour'][11:16]}Z hour:* {pct}% of "
                           f"{r.get('held_gpu_h')} held GPU-hours were idle, against a target of {t}%. That's {moved}; above "
                           f"the target, the rule is a drop of at least 5 points an hour.{phases}",
                           owners=(lane,) if lane in HANDLES else ("infra",)))
    return out


def held_idle_stale(ctx: Ctx) -> list[Finding]:
    """No completed hour in node 1's held-idle-hourly.jsonl for 70 minutes: the held-idle panel has stopped."""
    ends = [e for e in map(_hour_end, _held_rows(ctx)) if e is not None]
    late = ctx.now - max(ends, default=3600)
    if late <= 70 * 60:
        return []
    last = f"the newest completed hour in it is {iso(max(ends) - 3600)[11:16]}Z" if ends else "it has no hour at all"
    return [Finding("stale", f"*Held-idle panel: no reading for {late / 60:.0f} minutes*: {last} (node 1's `{HELD_IDLE}`).",
                    cleared="*Cleared:* the held-idle panel has a new hour.")]


# ---- a GPU idle while a waiter waits ----------------------------------------------------------------------------------------------

POOLS = {"/run/gpu-lease": "proofs", "/run/gpu-lease-circuits": "circuits"}     # n1_alerts.POOLS
POOL_DOC = "/workspace/usage/infra-pool-n1.json"
LEASE_FRESH_S = 120
KUEUE_HOLD = ((12, 10), (13, 30))                                               # dispatch.PACK_HOLD
MIN_WAIT_S = 60


def held_back(waiters: list[dict], usable: Callable[[dict], int], now: float) -> tuple[dict, float] | None:
    """The oldest waiter, waiting at least MIN_WAIT_S, that the idle GPUs could serve now: (its wait record, seconds waited)."""
    for w in waiters:
        try:
            waited = now - epoch(w["since"])
        except (KeyError, ValueError):
            continue
        if waited >= MIN_WAIT_S and int(w.get("n") or 1) <= usable(w):
            return w, waited
    return None


def idle_while_waiting_n1(ctx: Ctx) -> list[Finding]:
    """A lease waiter on node 1 waits while GPUs it could take sit idle: free in its pool (a holder, no lease), or held by
    nobody (class `free` in infra-pool-n1.json, so the pool could grow onto them). Opens after 5 minutes of that; not in the
    daily Kueue hold (12:10-13:30Z), nor for a blocked pool (stalls-n1 has that)."""
    g = time.gmtime(ctx.now)
    if KUEUE_HOLD[0] <= (g.tm_hour, g.tm_min) < KUEUE_HOLD[1]:
        raise Skip("node 1's daily Kueue hold")
    unheld: list[int] = []
    try:
        doc = json.loads(ctx.path(POOL_DOC).read_text())
        if ctx.now - epoch(doc["generated_at"]) <= 900:
            unheld = sorted(x["index"] for x in doc["nodes"]["n1"]["gpus"] if x.get("class") == "free" and not x.get("holder"))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    out = []
    for d in POOLS:
        p = ctx.path(d) / "n1-lease.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        name = s.get("pool_name") or d
        age = ctx.now - epoch(s["t"])
        if age > LEASE_FRESH_S:
            out.append(Finding(f"{name}:stale", f"*Node 1's lease pool `{name}` (`{d}`) hasn't ticked for {age / 60:.0f} minutes*: "
                               f"`{d}/n1-lease.json` is from {s['t']}, so its waiters get no new GPU.", for_s=300))
            continue
        if s.get("blocked"):
            continue
        pool_free = sorted(x["index"] for x in s.get("gpus") or [] if x.get("holder") and not x.get("lease") and not x.get("fence_pids"))
        idle = sorted({*pool_free, *unheld})

        def usable(w: dict) -> int:
            return len(idle) if not w.get("on") else len(gpu_list(w["on"]) & set(idle))

        hit = held_back(s.get("waiting") or [], usable, ctx.now)
        if hit:
            w, waited = hit
            out.append(Finding(name, f"*Node 1: GPU{'s' if len(idle) > 1 else ''} {', '.join(map(str, idle))} idle while "
                               f"{len(s.get('waiting') or [])} lease waiter(s) wait in pool `{name}`*: the oldest it could serve, "
                               f"{w.get('who')} ({w.get('run') or 'no run'}, {w.get('n')} GPU), has waited {waited / 60:.0f} min. "
                               f"Free in the pool: {pool_free or 'none'}; held by nobody: {unheld or 'none'}; pool of "
                               f"{s.get('pool')}, mode {s.get('mode')}.", for_s=300))
    return out


N2_WAITERS = r'''
import fcntl, glob, json, os
out = {"status": open("/workspace/pouw/fill/status.txt", errors="replace").readline().strip(), "waiters": []}
for f in sorted(glob.glob("/run/gpu-lease/wait.*")):
    try:
        fd = os.open(f, os.O_RDONLY)
    except OSError:
        continue
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            continue
        except BlockingIOError:
            pass
        out["waiters"].append(os.read(fd, 4096).decode(errors="replace").strip())
    finally:
        os.close(fd)
print(json.dumps(out))
'''


def idle_while_waiting_n2(ctx: Ctx) -> list[Finding]:
    """A gpu-lease waiter on node 2 waits while at least as many GPUs as it asks for are free (the fill runner's free count;
    a dead waiter's file, which nobody holds locked, isn't a waiter). Opens after 5 minutes of that; never while a timed window
    runs or waits. A waiter pinned to some GPUs (`on=`) counts only when every GPU is free: the status says how many are free,
    not which."""
    snap = json.loads(ctx.n2("python3 -", input=N2_WAITERS))
    st = snap["status"]
    m = re.search(r"free GPUs (\d+)/(\d+)", st)
    if not m:
        raise RuntimeError(f"node 2's fill status has no free count: {st[:160]}")
    if "window waiting True" in st:
        return []
    free, n = int(m[1]), int(m[2])
    waiters = [fields(w) for w in snap.get("waiters") or []]
    hit = held_back(waiters, lambda w: free if not w.get("on") or free == n else 0, ctx.now)
    if not hit:
        return []
    w, waited = hit
    return [Finding("idle", f"*Node 2: {free} of {n} GPUs free while {len(waiters)} gpu-lease waiter(s) wait*: the oldest they "
                    f"could serve, {w.get('who')} (pid {w.get('pid')}, {w.get('n')} GPU), has waited {waited / 60:.0f} min.",
                    for_s=300)]


def fill_status_n2(ctx: Ctx) -> list[Finding]:
    """Node 2's fill status is over 5 minutes old: its fill runner has stopped, and no other monitor reads node 2 meanwhile (their
    guard needs a fresh `timed False`)."""
    line = ctx.sh([*SSH, ctx.n2_host, f"head -n 1 {FILL_STATUS}"]).strip()
    try:
        age = ctx.now - epoch(line.split()[0])
    except (IndexError, ValueError):
        raise RuntimeError(f"node 2's fill status doesn't start with its time: {line[:120]!r}") from None
    if age <= FILL_FRESH_S:
        return []
    return [Finding("stale", f"*Node 2's fill status is {age / 60:.0f} minutes old* ({FILL_STATUS}: `{line[:160]}`), so the fill "
                    "runner isn't ticking, and the node-2 monitors are skipped until it does.", for_s=300)]


# ---- a lease preempted within 60 s of its grant (gpu-lease's preempted.log; the cluster agent's evictions log the same) ----------

PREEMPT_RE = re.compile(r"^(\S+Z) (SIG\w+) GPU (\d+) for pid (\d+) \(n=(\d+), who=([^)]*)\): (.*)$")
PREEMPT_WINDOW_S = 1800
PREEMPT_WITHIN_S = 60


def fast_preemptions(lines: list[str], now: float) -> list[dict]:
    """Each lease the log shows preempted within PREEMPT_WITHIN_S of its grant, in the last PREEMPT_WINDOW_S (once per lease)."""
    out, seen = [], set()
    for ln in lines:
        m = PREEMPT_RE.match(ln.strip())
        if not m:
            continue
        v = fields(m[7])
        try:
            t, since = epoch(m[1]), epoch(v["since"])
        except (KeyError, ValueError):
            continue
        if now - t > PREEMPT_WINDOW_S or not 0 <= t - since <= PREEMPT_WITHIN_S or (v.get("pid"), m[3]) in seen:
            continue
        seen.add((v.get("pid"), m[3]))
        out.append({"t": m[1], "gpu": int(m[3]), "by_pid": m[4], "by": m[6], "pid": v.get("pid"), "who": v.get("who"),
                    "run": v.get("run"), "after_s": int(t - since)})
    return out


def _fast_finding(key: str, where: str, fast: list[dict]) -> Finding:
    x = fast[-1]
    return Finding(key, f"*{where}: {len(fast)} lease{'s' if len(fast) > 1 else ''} preempted within {PREEMPT_WITHIN_S} s of "
                   f"{'their' if len(fast) > 1 else 'its'} grant in the last {PREEMPT_WINDOW_S // 60} minutes*. The newest: GPU "
                   f"{x['gpu']}, pid {x['pid']} ({x['who']}, run {x['run']}), {x['after_s']} s after its grant, for pid "
                   f"{x['by_pid']} ({x['by']}) at {x['t']}.")


def fast_preempt_n1(ctx: Ctx) -> list[Finding]:
    """A lease on node 1 preempted within 60 s of its grant, in the last 30 minutes (each pool's preempted.log): a waiter that
    preempts the leases its own queue was just granted restarts that queue forever."""
    out = []
    for d in POOLS:
        p = ctx.path(d) / "preempted.log"
        if p.exists() and (fast := fast_preemptions(tail(p, 1 << 16), ctx.now)):
            out.append(_fast_finding(d, f"Node 1 (`{d}`)", fast))
    return out


def fast_preempt_n2(ctx: Ctx) -> list[Finding]:
    """A lease on node 2 preempted within 60 s of its grant, in the last 30 minutes (/run/gpu-lease/preempted.log)."""
    fast = fast_preemptions(ctx.n2("tail -c 65536 /run/gpu-lease/preempted.log 2>/dev/null || true").splitlines()[1:], ctx.now)
    return [_fast_finding("/run/gpu-lease", "Node 2", fast)] if fast else []


# ---- failed systemd units --------------------------------------------------------------------------------------------------------

EXPECTED_FAILED = {
    "nvidia-fabricmanager.service": "both nodes' GPUs are PCIe, with no NVSwitch to manage, so it fails at every boot",
}
UNITS = "systemctl --failed --no-legend --plain; echo ---; systemctl --user --failed --no-legend --plain"


def failed_units(text: str) -> list[tuple[str, str]]:
    """(scope, unit) for each failed unit in UNITS' output (`system` before the `---`, `user` after), less EXPECTED_FAILED."""
    out, scope = [], "system"
    for ln in text.splitlines():
        if ln.strip() == "---":
            scope = "user"
        elif ln.split() and ln.split()[0].lstrip("●* ") not in EXPECTED_FAILED:
            out.append((scope, ln.split()[0].lstrip("●* ")))
    return out


def units(node: str) -> Callable[[Ctx], list[Finding]]:
    name = {"n1": "Node 1", "n2": "Node 2"}[node]

    def check(ctx: Ctx) -> list[Finding]:
        if node == "n2":
            text = ctx.n2(UNITS)
        else:
            text = ctx.sh(["systemctl", "--failed", "--no-legend", "--plain"])
            env = {**os.environ, "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"}
            r = ctx.run(["systemctl", "--user", "--failed", "--no-legend", "--plain"], capture_output=True, text=True, timeout=30,
                        env=env)
            if r.returncode and "connect to bus" not in (r.stderr or ""):
                raise RuntimeError(f"systemctl --user exited {r.returncode}: {(r.stderr or '').strip()[-200:]}")
            text += "---\n" + (r.stdout if r.returncode == 0 else "")
        return [Finding(f"{scope}:{u}", f"*{name}: {'the user unit' if scope == 'user' else 'unit'} `{u}` has failed*")
                for scope, u in failed_units(text)]

    check.__doc__ = (f"A failed systemd unit on {name.lower()}, system or research's user manager (`systemctl --failed`), less "
                     "EXPECTED_FAILED: units that fail on this hardware by design, each with its reason.")
    return check


# ---- relay-side: GitHub, the evidence store, Slack, and node 1 itself -------------------------------------------------------------

def deploy_drift(ctx: Ctx) -> list[Finding]:
    """Files on the nodes that aren't main's, still run old code, or stray (`research deploy drift --json`; it fetches origin,
    which the nodes can't): one incident per node and kind."""
    from .pods.nebius.preflight_probe import findings

    reps = json.loads(ctx.sh([*ctx.research, "deploy", "drift", "--json"], ok=(0, 1), timeout=900))
    if bad := sorted(n for n, r in reps.items() if "error" in r):
        raise RuntimeError(f"deploy drift can't read {', '.join(bad)}: {str(reps[bad[0]]['error'])[:200]}")
    groups: dict[tuple[str, str], list[str]] = {}
    for line in findings(reps):
        n, kind, path = line.split(" ", 2)
        groups.setdefault((n, kind), []).append(path)
    return [Finding(f"{n}:{kind}", f"*Deploy drift on {n}: {len(ps)} {kind}*: {', '.join(f'`{p}`' for p in ps[:5])}"
                    + (f" and {len(ps) - 5} more" if len(ps) > 5 else "") + ".") for (n, kind), ps in sorted(groups.items())]


def unpreserved(ctx: Ctx) -> list[Finding]:
    """Runs that ended over 30 minutes ago and aren't preserved on the remote, per machine (`research pods unpreserved MACHINE
    --older-min 30 --json`)."""
    out = []
    for mach in ctx.machines:
        r = ctx.run([*ctx.research, "pods", "unpreserved", mach, "--older-min", "30", "--json"], capture_output=True, text=True,
                    timeout=900)
        if r.returncode == 2 and "unrecognized arguments" in (r.stderr or ""):
            raise RuntimeError("`research pods unpreserved` here has no --older-min (the custody lane's "
                               "cursor/node-custody-unit-558b adds it)")
        if r.returncode not in (0, 1):
            raise RuntimeError(f"research pods unpreserved {mach} exited {r.returncode}: {(r.stderr or '').strip()[-200:]}")
        ids = [x if isinstance(x, str) else str(x.get("id") or x.get("attempt") or x) for x in json.loads(r.stdout).get("unpreserved") or []]
        if ids:
            out.append(Finding(mach, f"*{mach}: {len(ids)} run{'s' if len(ids) > 1 else ''} ended over 30 minutes ago and "
                               f"{'are' if len(ids) > 1 else 'is'} not preserved on the remote*: {', '.join(ids[:3])}"
                               + (f" and {len(ids) - 3} more" if len(ids) > 3 else "") + "."))
    return out


def asks(ctx: Ctx) -> list[Finding]:
    """An ask in #agent-coordination with no 👀, reply, ✅ or ❌ after 2 hours (`research slack traffic --json`), to its owner."""
    rep = json.loads(ctx.sh([*ctx.research, "slack", "traffic", "--json", "--hours", "24", "--stale-h", "2"], timeout=300))
    out = []
    for a in rep.get("unanswered") or []:
        to = tuple(a.get("to") or ())
        out.append(Finding(str(a["ts"]), f"*An ask by @{a.get('by') or '?'} to {' '.join('@' + t for t in to) or 'nobody'} has "
                           f"had no 👀, reply, ✅ or ❌ for over 2 hours* (ts {a['ts']} in #agent-coordination, from "
                           f"{iso(float(a['ts']))}).", owners=to or ("infra",),
                           cleared="*Cleared:* that ask was answered, or is now over 24 hours old."))
    return out


def node_heartbeat(ctx: Ctx) -> list[Finding]:
    """Node 1's monitor service hasn't finished a pass for 10 minutes (its state's `last_pass`), so no node-side monitor runs."""
    out = ctx.sh([*SSH, ctx.node1, f"python3 -c \"import json, os; p = '{ctx.node_state}'; "
                  "print(json.load(open(p)).get('last_pass', 0) if os.path.exists(p) else 'missing')\""]).strip()
    age = None if out == "missing" else ctx.now - float(out or 0)
    if age is not None and age <= 600:
        return []
    when = "have never finished a pass" if age is None else f"haven't run for {age / 60:.0f} minutes"
    return [Finding("stale", f"*Node 1's monitors {when}* (`{ctx.node_state}`), so nothing watches node 1's or node 2's "
                    "invariants.", cleared="*Cleared:* node 1's monitors run again.")]


# ---- the registry ---------------------------------------------------------------------------------------------------------------

RETENTION = ("Free space by the retention policy: `research retention report` on the node lists what is recorded and what nobody "
             "owns; `research retention gc --apply` deletes only what records allow; ask the owner about anything else. A HARD "
             "line means no new check starts there.")
FAST = ("Read the lines in preempted.log: a preemptible waiter that preempts leases its own queue was just granted (the same "
        "`who`) restarts that queue forever. Hold its submitter, or run its jobs without --preemptible.")
UNITS_ACTION = ("`systemctl status UNIT` and `journalctl -u UNIT -n 50` on the node (`--user` for a user unit); fix it and "
                "`systemctl reset-failed UNIT`, or, if it fails on this hardware by design, add it to EXPECTED_FAILED in "
                "research/monitors.py with the reason.")

MONITORS: list[Monitor] = [
    Monitor("disk-n1", "infra", "warn", "node", watermarks("n1"), RETENTION, every_s=600, clear_s=1800),
    Monitor("disk-n2", "infra", "warn", "node", watermarks("n2"), RETENTION, every_s=600, clear_s=1800),
    Monitor("stalls-n1", "infra", "page", "node", stalls_n1,
            "A blocked pool: stop the process its `blocked` file names (or have its owner stop it), then remove the file. A "
            "dispatcher that fails or doesn't tick: read its pane and loop.log, fix the cause, restart it in tmux "
            "`node1-dispatch`.", unread_s=600),
    Monitor("idle-while-waiting-n1", "infra", "warn", "node", idle_while_waiting_n1,
            "Why isn't the waiter granted? `cat /run/gpu-lease*/n1-lease.json` (mode, pool, waiting), the pool's cap "
            "(VY_POOL_MAX) and its queue's quota (n1_lease.py's grow rule), and the cluster agent's grants when the brain is on. "
            "A GPU nobody holds while lease waiters wait means the pool can't grow onto it: raise the cap or quota, or move the "
            "waiters."),
    Monitor("idle-while-waiting-n2", "infra", "warn", "node", idle_while_waiting_n2,
            "Read node 2's fill status and the wait files in /run/gpu-lease; while the cluster agent holds agent.lock, it "
            "grants: see its log. A waiter pinned (`on=`) to a busy GPU waits by design."),
    Monitor("fill-status-n2", "infra", "warn", "node", fill_status_n2,
            "Restart node 2's fill runner (tmux `pouw-infra-fill`) after reading why it stopped. Until its status is fresh, no "
            "monitor reads node 2."),
    Monitor("fast-preempt-n1", "infra", "warn", "node", fast_preempt_n1, FAST),
    Monitor("fast-preempt-n2", "infra", "warn", "node", fast_preempt_n2, FAST),
    Monitor("units-n1", "infra", "warn", "node", units("n1"), UNITS_ACTION, every_s=300),
    Monitor("units-n2", "infra", "warn", "node", units("n2"), UNITS_ACTION, every_s=300),
    Monitor("held-idle", "infra", "warn", "node", held_idle,
            "Your held GPUs sat idle above your target: release GPUs between phases, or lease for less. The `held-idle-hourly` "
            "panel on the live console says where the time went.", every_s=300, events=True),
    Monitor("held-idle-stale", "infra", "warn", "node", held_idle_stale,
            "Check `verity-console.service` and its timer on node 1 (`journalctl -u verity-console -n 50`).", every_s=300),
    Monitor("drift", "infra", "warn", "relay", deploy_drift,
            "`research deploy drift --node N` for the detail; install main's version (`research deploy install N --by WHO "
            "PATH`), restart what runs old code, or move strays to the attic (`research deploy attic N PATH`).",
            every_s=3600, unread_s=4 * 3600),
    Monitor("unpreserved", "infra", "warn", "relay", unpreserved,
            "`research pods unpreserved MACHINE` lists them and `research data preserved RUN` says why each isn't preserved: "
            "push them, or fix the custody service that should have.", every_s=1800, unread_s=4 * 3600),
    Monitor("asks", "infra", "warn", "relay", asks,
            "Pick the ask up (👀), answer it, or decline it with a reason: `research slack pickup|done|decline --thread TS`.",
            every_s=900, unread_s=3 * 3600),
    Monitor("node-heartbeat", "infra", "page", "relay", node_heartbeat,
            "`ssh n1 systemctl status vy-monitors.timer vy-monitors.service` and `ssh n1 journalctl -u vy-monitors -n 50`.",
            every_s=120, unread_s=600),
]
BY_ID = {m.id: m for m in MONITORS}


# ---- sinks ------------------------------------------------------------------------------------------------------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "alert"


def title(text: str) -> str:
    return re.sub(r"[*_`]", "", text.splitlines()[0] if text else "")[:160]


def note(ev: dict, now: float) -> tuple[str, str]:
    """An open as alert_sink.py's note: (file name, markdown), the lane contract's front matter plus the monitor's own fields."""
    name = f"{time.strftime('%Y%m%dT%H%MZ', time.gmtime(now))}-alert-{_slug(ev['monitor'])}-{hashlib.sha256(ev['key'].encode()).hexdigest()[:8]}.md"
    lines = ["---", f"id: {name[:-3]}", "campaign: nebius-monitoring", "lane: infra", "kind: finding", "status: open",
             f"tags: [{ev['severity']}]", "repo: danielreuter/verity", f"origin: {ev['by']} (research monitors, research/monitors.py)",
             f"monitor: {ev['monitor']}", f"key: {json.dumps(ev['key'])}", f"owners: [{', '.join(ev['owners'])}]",
             f"severity: {ev['severity']}", "---", "", f"# {title(ev['text'])}", "", ev["text"], "", f"What to do: {ev['action']}", ""]
    return name, "\n".join(lines)


class AlertDir:
    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, ev: dict, now: float) -> None:
        if ev["event"] != "clear":
            name, text = note(ev, now)
            out = self.path / "outbox" / name
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            tmp.write_text(text)
            tmp.replace(out)
            ev["note"] = name
        with open(self.path / "monitors.jsonl", "a") as f:
            f.write(json.dumps(ev) + "\n")


def message(ev: dict, tag: Callable[[str], str] = lambda h: "@" + h) -> str:
    """An event as one Slack post: the owners tagged, the alert, what to do."""
    if ev["event"] == "clear":
        return ev["text"]
    sev = "[page] " if ev["severity"] == "page" else ""
    return (f"{' '.join(tag(o) for o in ev['owners'])} {sev}{ev['text']}\n_What to do:_ {ev['action']}\n"
            f"_monitor `{ev['monitor']}`, {ev['by']}_")


def webhook_post(url: str, text: str) -> None:
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()


def _why(e: Exception) -> str:
    """A webhook failure, without the URL a message might carry."""
    return f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else type(e).__name__


# ---- the runner -----------------------------------------------------------------------------------------------------------------

def new_state() -> dict:
    return {"schema": SCHEMA.format("state"), "open": {}, "pending": {}, "events": {}, "monitors": {}, "retry": []}


def step(state: dict, monitors: list[Monitor], ctx_for: Callable[[dict], Ctx], now: float, sink: AlertDir, *, by: str,
         webhook: str | None = None, post: Callable[[str, str], None] = webhook_post, force: bool = False) -> list[dict]:
    """One pass: each due monitor's check, its incidents opened and cleared through `sink` (and `webhook`). Returns the events."""
    for k, v in new_state().items():
        state.setdefault(k, v)
    opened, pending, seen = state["open"], state["pending"], state["events"]
    emitted: list[dict] = []

    def emit(m: Monitor, kind: str, key: str, f: Finding | None, o: dict | None = None) -> bool:
        ev = {"schema": SCHEMA.format("event"), "event": kind, "t": iso(now), "monitor": m.id, "key": key, "by": by,
              "owners": list(f.owners if f and f.owners else (o or {}).get("owners") or (m.owner,)),
              "severity": (f.severity if f else None) or (o or {}).get("severity") or m.severity,
              "text": f.text if f else o["cleared"], "action": m.action, "webhook": bool(webhook)}
        if o:
            ev["since"] = o["since"]
        try:
            sink.write(ev, now)
        except OSError as e:
            state["sink_error"] = f"{iso(now)} {type(e).__name__}: {e}"[:300]
            return False
        state.pop("sink_error", None)
        if webhook:
            state["retry"].append({"t": now, "text": message(ev)})
        emitted.append(ev)
        return True

    for m in monitors:
        st = state["monitors"].setdefault(m.id, {"since": now})
        if not force and now - st.get("last_run", 0) < m.every_s - 5:
            continue
        st["last_run"] = now
        try:
            found: dict[str, Finding | None] = {f.key: f for f in m.check(ctx_for(st.setdefault("memo", {})))}
            st.update(last_ok=now, error=None, skipped=None)
            ok = True
        except Skip as e:
            st.update(last_ok=now, skipped=str(e)[:300])
            continue
        except Exception as e:  # noqa: BLE001  (an input that can't be read is the monitor's own finding, in time)
            st["error"] = f"{type(e).__name__}: {e}"[:500]
            late = now - st.get("last_ok", st["since"])
            if late < m.unread_s:
                continue
            ok = False
            found = {k.split("|", 1)[1]: None for k, o in opened.items() if o["monitor"] == m.id}
            found["unreadable"] = Finding("unreadable", f"*Monitor `{m.id}` can't read its input, for {late / 60:.0f} minutes "
                                          f"now*: {st['error']}", owners=("infra",), severity="warn", for_s=0,
                                          cleared=f"*Cleared:* monitor `{m.id}` reads its input again.")
        if m.events and ok:
            new = [f for f in found.values() if f and f"{m.id}|{f.key}" not in seen]
            if st.get("primed"):
                for f in new:
                    if emit(m, "event", f"{m.id}|{f.key}", f):
                        seen[f"{m.id}|{f.key}"] = now
            else:
                seen.update({f"{m.id}|{f.key}": now for f in new})
                st["primed"] = True
            found = {}
        for k, f in found.items():
            key = f"{m.id}|{k}"
            if key in opened:
                opened[key].pop("absent_since", None)
                continue
            if f is None:
                continue
            first = pending.setdefault(key, now)
            if now - first >= (f.for_s if f.for_s is not None else m.for_s) and emit(m, "open", key, f):
                opened[key] = {"monitor": m.id, "since": iso(now), "owners": list(f.owners or (m.owner,)),
                               "severity": f.severity or m.severity, "text": f.text,
                               "cleared": f.cleared or f"*Cleared:* {title(f.text)}"}
                del pending[key]
        for key in [k for k in pending if k.split("|", 1)[0] == m.id and k.split("|", 1)[1] not in found]:
            del pending[key]
        for key in [k for k, o in opened.items() if o["monitor"] == m.id and k.split("|", 1)[1] not in found]:
            o = opened[key]
            o.setdefault("absent_since", now)
            if now - o["absent_since"] >= (0 if key.endswith("|unreadable") else m.clear_s) and emit(m, "clear", key, None, o):
                del opened[key]
    for k in [k for k, t in seen.items() if now - t > EVENTS_KEEP_S]:
        del seen[k]
    if webhook:
        left = []
        for i, r in enumerate(state["retry"]):
            try:
                post(webhook, r["text"])
            except Exception as e:  # noqa: BLE001
                state["webhook_error"] = f"{iso(now)} {_why(e)}"
                left = [x for x in state["retry"][i:] if now - x["t"] < RETRY_KEEP_S]
                break
        else:
            state.pop("webhook_error", None)
        state["retry"] = left
    state["last_pass"] = now
    return emitted


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return new_state()


def save(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True))
    tmp.replace(path)


def lock(path: Path) -> int | None:
    """An exclusive lock beside `path` (its fd), or None when another runner holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path.with_name(path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def select(where: str | None, only: list[str] | None) -> list[Monitor]:
    if unknown := [i for i in only or [] if i not in BY_ID]:
        raise ValueError(f"unknown monitor {', '.join(unknown)}; `research monitors list` names them")
    return [m for m in MONITORS if (not where or m.where == where) and (not only or m.id in only)]


# ---- the relay (infra's VM: Slack) --------------------------------------------------------------------------------------------

def still_open(events: list[dict]) -> list[dict]:
    """The opens among `events` that no later clear ends, oldest first."""
    live: dict[str, dict] = {}
    for ev in events:
        if ev.get("event") == "open":
            live[ev["key"]] = ev
        elif ev.get("event") == "clear":
            live.pop(ev.get("key"), None)
    return list(live.values())


def front_matter(text: str) -> dict[str, str]:
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    return dict(ln.split(": ", 1) for ln in m[1].splitlines() if ": " in ln) if m else {}


class Source:
    """An alert dir the relay reads: node 1's over ssh, or a local one. `lines(after)` -> (total, the lines after `after`);
    `notes()` -> the outbox's names; `head(name)` -> a note's start."""

    def __init__(self, name: str, path: str, host: str | None = None, run: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.name, self.path, self.host, self.run = name, path.rstrip("/"), host, run

    def _sh(self, script: str) -> str:
        argv = [*SSH, self.host, script] if self.host else ["sh", "-c", script]
        r = self.run(argv, capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise RuntimeError(f"{self.name}: exited {r.returncode}: {(r.stderr or '').strip()[-200:]}")
        return r.stdout

    def lines(self, after: int) -> tuple[int, list[str]] | None:
        f = f"{self.path}/monitors.jsonl"
        out = self._sh(f"[ -f {f} ] || {{ echo missing; exit 0; }}; wc -l < {f}; tail -n +{after + 1} {f}").splitlines()
        if not out or out[0].strip() == "missing":
            return None
        return int(out[0].strip() or 0), out[1:]

    def notes(self) -> list[str]:
        return sorted(n for n in self._sh(f"ls -1 {self.path}/outbox 2>/dev/null || true").split() if n.endswith(".md"))

    def head(self, name: str) -> str:
        return self._sh(f"head -c 4000 {self.path}/outbox/{name}")


def _event(line: str) -> dict:
    try:
        ev = json.loads(line)
    except ValueError:
        return {}
    return ev if isinstance(ev, dict) else {}


def relay_pass(rstate: dict, sources: list[Source], post: Callable[[dict, dict | None], dict], now: float) -> int:
    """Post each source's new events and Grafana's new notes; returns how many. The cursor moves past each event once it is
    posted, so a post that raises is the next pass's first."""
    posted = 0
    threads = rstate.setdefault("threads", {})

    def one(src: Source, ev: dict) -> None:
        nonlocal posted
        tkey = f"{src.name}|{ev.get('key')}"
        if not ev.get("event") or ev.get("webhook") or (ev["event"] == "open" and tkey in threads) or (
                ev["event"] == "clear" and tkey not in threads):
            return
        got = post(ev, threads.get(tkey))
        posted += 1
        if ev["event"] == "open":
            threads[tkey] = {**got, "t": now}
        elif ev["event"] == "clear":
            del threads[tkey]

    for src in sources:
        s = rstate.setdefault("sources", {}).setdefault(src.name, {})
        first = "lines" not in s
        got = src.lines(0 if first else s["lines"])
        if got is not None:
            total, lines = got
            if total < s.get("lines", 0):
                s["lines"] = 0                          # a new file: read from its start next pass
            elif first:
                for ev in still_open([_event(ln) for ln in lines[:total]]):
                    one(src, ev)
                s["lines"] = total
            else:
                for ln in lines[:total - s["lines"]]:
                    one(src, _event(ln))
                    s["lines"] += 1
        names = src.notes()
        if "notes" not in s:
            s["notes"] = names
            continue
        seen = set(s["notes"]) & set(names)
        for n in names:
            if n in seen:
                continue
            text = src.head(n)
            fm = front_matter(text)
            if "monitor" not in fm:
                post({"event": "note", "monitor": "grafana", "key": n, "owners": [fm.get("lane", "infra")], "severity": "warn",
                      "text": text.split("\n---\n", 1)[-1].strip()[:2500], "action": f"read `{n}` in node 1's alert outbox",
                      "by": fm.get("origin", "grafana")}, None)
                posted += 1
            seen.add(n)
            s["notes"] = sorted(seen)
        s["notes"] = sorted(seen)
    for k in [k for k, v in threads.items() if now - v.get("t", now) > EVENTS_KEEP_S]:
        del threads[k]
    return posted


def slack_post(dry: bool) -> Callable[[dict, dict | None], dict]:
    from . import slack

    reg = slack.load_registry(slack.registry_path())
    api = slack.Slack(os.environ.get("SLACK_BOT_TOKEN"), dry_run=dry)
    alerts, front, routes = slack.channel_id_for(reg, "agent-alerts"), slack.channel_id_for(reg, slack.FRONT_DOOR), slack.Routes()

    def tag(h: str) -> str:
        try:
            return slack.tag(slack.entry(reg, h))
        except slack.SlackError:
            return "@" + h

    def post(ev: dict, thread: dict | None) -> dict:
        if ev["event"] == "clear":
            for k, ch in (("ts", alerts), ("pointer", front)):
                if (thread or {}).get(k):
                    slack._post(api, ch, POSTER, message(ev), thread[k])
            return {}
        ts = slack._post(api, alerts, POSTER, message(ev, tag))
        out = {"ts": ts}
        if others := [o for o in ev["owners"] if o != "infra"]:
            link = api.call("chat.getPermalink", channel=alerts, message_ts=ts).get("permalink") or "#agent-alerts"
            out["pointer"] = slack._post(api, front, POSTER, f"{' '.join(tag(o) for o in others)} {title(ev['text'])} (alert: {link})",
                                         route=(reg, routes), tags=("alert",))
        return out
    return post


# ---- the command line -----------------------------------------------------------------------------------------------------------

def _ctx_for(ns: argparse.Namespace, now: float) -> Callable[[dict], Ctx]:
    return lambda memo: Ctx(now=now, memo=memo, n2_host=os.environ.get("VY_N2", "vy-n2"), node1=getattr(ns, "node1", "n1"),
                            node_state=getattr(ns, "node_state", STATE))


def cmd_list(ns: argparse.Namespace) -> int:
    ms = select(ns.where, None)
    if ns.json:
        print(json.dumps({"schema": SCHEMA.format("list"), "monitors": [
            {"id": m.id, "owner": m.owner, "severity": m.severity, "where": m.where, "every_s": m.every_s, "for_s": m.for_s,
             "clear_s": m.clear_s, "events": m.events, "unread_s": m.unread_s, "doc": m.doc, "action": m.action} for m in ms]},
            indent=1))
        return 0
    for m in ms:
        print(f"{m.id:<22} {m.where:<5} @{m.owner:<6} {m.severity:<4} every {m.every_s:.0f} s"
              f"{f', for {m.for_s:.0f} s' if m.for_s else ''}{', events' if m.events else ''}\n    {m.doc}\n    action: {m.action}")
    return 0


def cmd_check(ns: argparse.Namespace) -> int:
    now = time.time()
    rows, bad = [], False
    for m in select(ns.where, ns.ids):
        row: dict[str, Any] = {"id": m.id, "findings": [], "error": None, "skipped": None}
        try:
            row["findings"] = [{"key": f.key, "text": f.text, "owners": list(f.owners or (m.owner,)),
                                "severity": f.severity or m.severity} for f in m.check(_ctx_for(ns, now)({}))]
        except Skip as e:
            row["skipped"] = str(e)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        bad = bad or bool(row["findings"] or row["error"])
        rows.append(row)
    if ns.json:
        print(json.dumps({"schema": SCHEMA.format("check"), "t": iso(now), "monitors": rows}, indent=1))
    else:
        for r in rows:
            state = "error: " + r["error"] if r["error"] else "skipped: " + r["skipped"] if r["skipped"] else f"{len(r['findings'])} found"
            print(f"{r['id']}: {state}")
            for f in r["findings"]:
                print(f"  [{f['severity']}] {f['key']}: {title(f['text'])}")
    return 1 if bad else 0


def run_once(ns: argparse.Namespace, where: str, state_path: Path, alerts: Path, now: float) -> tuple[dict, list[dict]] | None:
    fd = lock(state_path)
    if fd is None:
        return None
    try:
        state = load(state_path)
        evs = step(state, select(where, ns.only), _ctx_for(ns, now), now, AlertDir(alerts), by=ns.by,
                   webhook=os.environ.get(WEBHOOK_ENV) or None)
        save(state_path, state)
        return state, evs
    finally:
        os.close(fd)


def cmd_run(ns: argparse.Namespace) -> int:
    now = time.time()
    got = run_once(ns, ns.where, Path(ns.state), Path(ns.alerts), now)
    if got is None:
        print(f"monitors: refused: another runner holds {ns.state}.lock", file=sys.stderr)
        return 3
    state, evs = got
    if ns.json:
        print(json.dumps({"schema": SCHEMA.format("run"), "t": iso(now), "by": ns.by, "events": evs, "open": sorted(state["open"]),
                          "errors": {k: v["error"] for k, v in state["monitors"].items() if v.get("error")}}, indent=1))
    else:
        for ev in evs:
            print(f"{ev['event']} {ev['key']}: {title(ev['text'])}")
        print(f"{iso(now)} open: {', '.join(sorted(state['open'])) or 'none'}")
    return 1 if state["open"] else 0


def cmd_relay(ns: argparse.Namespace) -> int:
    posts = Path(ns.posts)
    fd = lock(posts)
    if fd is None:
        print(f"monitors: refused: another relay holds {posts}.lock", file=sys.stderr)
        return 3
    post = slack_post(ns.dry_run)
    sources = [Source("node1", ns.node_alerts, host=ns.node1), Source("relay", ns.alerts)]
    rstate = json.loads(posts.read_text()) if posts.exists() else {"schema": SCHEMA.format("relay"), "by": ns.by}
    while True:
        now = time.time()
        stamp = iso(now)[11:]
        if run_once(ns, "relay", Path(ns.state), Path(ns.alerts), now) is None:
            print(f"{stamp} relay-side monitors skipped: another runner holds {ns.state}.lock", flush=True)
        try:
            n = relay_pass(rstate, sources, post, now)
            print(f"{stamp} posted {n}", flush=True)
        except Exception as e:  # noqa: BLE001  (a source or post that failed is read again next pass)
            print(f"{stamp} relay failed: {type(e).__name__}: {e}", flush=True)
        if not ns.dry_run:
            save(posts, rstate)
        if ns.once:
            return 0
        time.sleep(ns.every)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research monitors", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="every monitor: owner, severity, where it runs, threshold, action")
    p.add_argument("--where", choices=("node", "relay"))
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("check", help="evaluate now and write nothing: exit 1 on a finding or a failed check")
    p.add_argument("ids", nargs="*", metavar="ID")
    p.add_argument("--where", choices=("node", "relay"))
    p.add_argument("--node1", default="n1", help="node 1's ssh host, for relay-side checks")
    p.add_argument("--node-state", default=STATE)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("run", help="one pass: open and clear incidents into the alert dir (node 1's service)")
    p.add_argument("--by", required=True, help="who runs it, recorded in every event (vy-monitors@vy-nebius-1)")
    p.add_argument("--where", choices=("node", "relay"), default="node")
    p.add_argument("--state", default=STATE)
    p.add_argument("--alerts", default=ALERTS)
    p.add_argument("--only", nargs="+", metavar="ID")
    p.add_argument("--node1", default="n1")
    p.add_argument("--node-state", default=STATE)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("relay", help="where Slack works: run the relay-side monitors, post every new event to #agent-alerts")
    p.add_argument("--by", required=True, help="who runs it, recorded in the relay-side events")
    p.add_argument("--node1", default="n1", help="node 1's ssh host")
    p.add_argument("--node-alerts", default=ALERTS, help="node 1's alert dir")
    p.add_argument("--node-state", default=STATE, help="node 1's monitor state (its heartbeat)")
    p.add_argument("--alerts", default=str(LOCAL / "alerts"), help="the relay-side monitors' alert dir")
    p.add_argument("--state", default=str(LOCAL / "relay-state.json"))
    p.add_argument("--posts", default=str(LOCAL / "relay-posts.json"), help="what has been posted (one writer: the relay)")
    p.add_argument("--only", nargs="+", metavar="ID")
    p.add_argument("--every", type=float, default=60.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the posts instead, and keep what was posted only in memory")
    ns = ap.parse_args(argv)
    try:
        select(None, getattr(ns, "ids", None) or getattr(ns, "only", None))
    except ValueError as e:
        print(f"monitors: {e}", file=sys.stderr)
        return 2
    return {"list": cmd_list, "check": cmd_check, "run": cmd_run, "relay": cmd_relay}[ns.cmd](ns)


if __name__ == "__main__":
    sys.exit(main())
