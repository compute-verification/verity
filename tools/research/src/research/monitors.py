"""Monitors: invariants in code, each with an owner, a severity, a threshold and an action (the infra vision's Signals surface).
One registry (MONITORS) and one runner. Node 1's service (`vy-monitors.timer`, pods/nebius/monitoring/) runs the node-side
monitors and writes each incident once when it opens and once when it clears; the relay on infra's VM runs the relay-side ones
(they need GitHub, the evidence store or Slack) and posts both sets to #agent-alerts. Every Nebius machine runs the `machine`
monitors (`vy-watchdog.timer`: `run --where machine --machine NAME --peer HOST ...`): memory pressure that holds, each process
the OOM killer took (named with its run and the run's sender), a peer that stops answering, /workspace filling within hours on its
recent trend (named with the biggest live runs), a unit deploy.toml runs there that was never installed or is off, and
yesterday's count of full checks lost to infrastructure. The machines watch each other, so one that hangs is reported by one that doesn't.

    research monitors list [--where node|relay] [--json]
    research monitors check [ID ...] [--where node|relay] [--json]     evaluate now and write nothing
    research monitors run --by WHO [--where node|relay] [--state PATH] [--alerts DIR] [--only ID ...] [--json]
    research monitors relay --by WHO [--node1 HOST] [--node-alerts DIR] [--every 60] [--once] [--dry-run]

**A monitor** has an id, an owner handle, a severity (`warn`, or `page` when work stops), where it runs, how often, a check that
returns findings, and the action its owner takes. A finding's `key` is stable while the incident lasts: the runner opens an
incident once a key has been found for `for_s`, and clears it once the key has been gone for `clear_s`. An `events` monitor's
findings are posted once each and never cleared; its first pass only remembers what is already there. A check that raises
`Skip` (node 2 in a timed window) changes nothing; one refused by node 2 is unread meanwhile, so its next failure counts from
its last read. One that fails otherwise keeps its incidents open, and once it has failed for `unread_s` it opens
`<id>|unreadable` for infra.

**Sinks.** Each open and clear is a line of `<alerts>/monitors.jsonl` (`research/monitors-event/v1`), and each open also a note
in `<alerts>/outbox/` in alert_sink.py's format, so one relay posts these and Grafana's alike. A write there that fails opens or
clears nothing, so the next pass tries again. A node pass ends by PASS_S, inside the unit's timeout: a check still due then is
Late and runs next pass. The state is saved after each monitor, and an event monitors.jsonl already has (a pass killed before
it saved) isn't written again. With `$VY_MONITORS_SLACK_WEBHOOK` set (an incoming webhook; its URL is never
logged), the runner also posts each event there, retrying on later passes until Slack takes it, and the relay leaves those
events alone.

**Node 2** is read from node 1 over ssh (`$VY_N2`, default `vy-n2`: the vy-cluster key), in one login per pass, and the guard
runs first in it: unless node 2's fill status says `timed False, window waiting False` and is at most FILL_GO_S old on node 2's
clock (not from the future), it prints the status and nothing else runs, and each node-2 check is a Skip. Past the guard, one
python, bounded by `timeout` and niced there, reads every part the pass's due checks need, checking the status again before each.
While a window booked in node 2's fill/windows runs (as the last session read it), a pass doesn't log in at all. Refused for
UNWATCHED_S past the end of any booked window, fill-status-n2 opens `unwatched`: a timed window that long is the incident.

`relay` runs the relay-side monitors into a local alert dir, then posts every new event of node 1's alert dir (over ssh) and the
local one to #agent-alerts as `vy-monitors`, tagging the owners; each clear is a reply in its alert's thread (a post of its
own when that thread is gone). An owner other than infra also gets a pointer in #agent-coordination, where its doorbell rings
(#agent-alerts rings only infra). A finding with a `thread` (failed-checks': the merge thread) is instead a reply there,
ringing its owners. Grafana's notes in node 1's outbox are posted too. On its first pass over an alert dir it
posts only the incidents still open. A post that fails stops only its own source, until the next pass; one Slack refuses for
good, or that has failed for RELAY_GIVE_UP_S, is dropped with a log line.

Exit codes: 0 `check` found nothing, or `run` finished its pass (what is open is in the state and the alert dir); 1 `check` found
something or a check failed, or `run` couldn't write the alert dir (a crash is 1 too: vy-monitors.service fails on both); 2
usage; 3 refused (another runner holds the state's lock).
"""
from __future__ import annotations

import argparse
import calendar
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
WINDOWS = "/workspace/pouw/fill/windows"
QUIET = "timed False, window waiting False"      # what every node-2 guard on main requires (dispatch.n2_room, write_probe)
FILL_GO_S = 60                                   # the guard's limit; fill_runner rewrites its status every 10 s (POLL_S)
FILL_FRESH_S = 300                               # fill-status-n2's: older, the fill runner has stopped
UNWATCHED_S = 2 * 3600                           # node 2 refusing that long past any booked window: fill-status-n2's `unwatched`
# The status's first field must be a time, at most FILL_GO_S old on node 2's clock; a refusal prints `SKIPPED <node 2's now>
# <age, or - without a time> <status>` and exits.
N2_GUARD = (f'ok=0; s=$(head -n 1 {FILL_STATUS} 2>/dev/null); n=$(date +%s); t=${{s%% *}}; a=-; '
            'case "$t" in 20[0-9][0-9]-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]Z) '
            'a=$(( n - $(date -d "$t" +%s 2>/dev/null || echo 0) ));; esac; '
            f'case "$a" in -*) ;; *) case "$s" in *"{QUIET}"*) [ "$a" -gt {FILL_GO_S} ] || ok=1;; esac;; esac; '
            f'[ "$ok" = 1 ] || {{ printf "%s %s %s %s\\n" "{SKIPPED}" "$n" "$a" "$s"; exit 0; }}; ')
N2_S = 90                                        # node 2's one session per pass, local; node 2's `timeout` ends it 10 s sooner
PASS_S = 180                                     # a node pass's deadline, well inside vy-monitors.service's TimeoutStartSec
HANDLES = ("infra", "proofs", "circuits", "compute-accounting", "memory-accounting", "network-accounting")
EVENTS_KEEP_S = 14 * 86400
RETRY_KEEP_S = 86400


class Skip(Exception):
    """The check can't look now, by design (node 2 in a timed window): nothing opens or clears. `unread`: the skip isn't a read
    (node 2 refused it), so the monitor's input stays as late as its last read."""

    def __init__(self, why: str = "", *, unread: bool = False):
        super().__init__(why)
        self.unread = unread


class Late(Skip):
    """The pass's deadline came first: the check is still due, and runs next pass."""


class Refused(Skip):
    """Node 2's guard refused: its fill status `status`, `age` s old on node 2's clock `now` (None: it has no time)."""

    def __init__(self, status: str, now: float, age: float | None):
        super().__init__(refusal(status, age), unread=True)
        self.status, self.now, self.age = status, now, age


def refusal(status: str, age: float | None) -> str:
    if not status:
        return "node 2 has no fill status"
    if QUIET not in status:
        return f"node 2 isn't quiet: {status[:160]}"
    if age is None:
        return f"node 2's fill status has no time: {status[:160]}"
    return f"node 2's fill status is {-age:.0f} s ahead of its clock" if age < 0 else f"node 2's fill status is {age:.0f} s old"


@dataclass(frozen=True)
class Finding:
    key: str
    text: str
    owners: tuple[str, ...] = ()        # default: the monitor's owner
    severity: str | None = None         # default: the monitor's
    cleared: str | None = None          # what its clear says (default: "Cleared:" and the alert's first line)
    for_s: float | None = None          # default: the monitor's
    thread: str | None = None           # a thread of #agent-coordination the relay posts it in, instead of #agent-alerts


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
    n2: str | None = None               # the part of node 2's session it reads (n2_parts)

    @property
    def doc(self) -> str:
        return " ".join((self.check.__doc__ or "").split())


@dataclass
class N2Read:
    """A pass's session on node 2: the fill status it found, node 2's clock, and each part's result (`rc`, `out`, `err`, or
    `error`). `refused`: the guard refused, and nothing ran; `stopped`: the status a part found instead of a quiet one."""
    status: str
    now: float
    parts: dict = field(default_factory=dict)
    refused: Refused | None = None
    stopped: str | None = None


@dataclass
class Pass:
    """What one pass's checks share: the node-2 parts its due checks read, node 2's state kept between passes (`read_t`, the
    booked `windows`), its one session on node 2 (an N2Read, or what opening it raised), and its deadline on `clock`."""
    n2_parts: frozenset[str] = frozenset()
    n2_state: dict = field(default_factory=dict)
    n2: N2Read | Exception | None = None
    deadline: float = float("inf")
    clock: Callable[[], float] = time.monotonic

    def left(self) -> float:
        return self.deadline - self.clock()


@dataclass
class Ctx:
    """What a check reads through: the clock, its own memo (kept in the state between passes), what its pass shares, commands,
    and files under `root`."""
    now: float
    memo: dict = field(default_factory=dict)
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    root: Path = Path("/")
    n2_host: str = "vy-n2"
    node1: str = "n1"
    node_state: str = STATE
    machines: tuple[str, ...] = ("vy-nebius-1", "vy-nebius-2")
    machine: str = ""                   # where a `machine` monitor runs (its runner's --machine)
    peers: tuple[str, ...] = ()         # the ssh hosts it watches (--peer)
    research: tuple[str, ...] = (sys.executable, "-m", "research")
    act: bool = False                   # a runner's pass, which may act (failed-checks' cancels), not `check`'s
    pass_: Pass = field(default_factory=Pass)

    def path(self, p: str) -> Path:
        return self.root / p.lstrip("/")

    def bound(self, timeout: float) -> float:
        """`timeout`, cut to what is left of the pass; Late when that is under 5 s."""
        left = self.pass_.left()
        if left < 5:
            raise Late("the pass's deadline came first")
        return min(timeout, left)

    def sh(self, argv: list[str], *, input: str | None = None, timeout: float = 60, ok: tuple[int, ...] = (0,)) -> str:
        r = self.run(argv, input=input, capture_output=True, text=True, timeout=self.bound(timeout))
        if r.returncode not in ok:
            what = argv[-1] if argv[0] == "ssh" else " ".join(argv[:4])
            raise RuntimeError(f"`{what[-120:]}` exited {r.returncode}: {(r.stderr or r.stdout or '').strip()[-300:]}")
        return r.stdout

    def n2(self, script: str, *, input: str | None = None, timeout: float = N2_S) -> str:
        """`script` on node 2 behind N2_GUARD in the same ssh call, under `timeout` and niced there too; Refused when the guard
        refuses."""
        timeout = self.bound(timeout)
        bound = f"timeout -k 5 {max(5, int(timeout) - 10)} {' '.join(NICE)} sh -c {shlex.quote(script)}"
        out = self.sh([*SSH, self.n2_host, N2_GUARD + bound], input=input, timeout=timeout)
        if out.startswith(SKIPPED):
            raise refused(out, self.now)
        return out

    def node2(self, *want: str) -> N2Read:
        """This pass's one session on node 2. The first check that asks opens it, reading every part the pass's due checks
        need (and `want`); the others reuse it. A Skip, and no login, while a window booked on node 2 runs."""
        p = self.pass_
        if p.n2 is None:
            try:
                p.n2 = self._session(p.n2_parts | set(want))
            except Exception as e:  # noqa: BLE001  (each node-2 check gets it)
                p.n2 = e
        if isinstance(p.n2, Exception):
            raise p.n2
        return p.n2

    def _session(self, names: set[str]) -> N2Read:
        st = self.pass_.n2_state
        for start, end, who in st.get("windows") or []:
            if start <= self.now < end:
                raise Skip(f"node 2 is in a booked window until {iso(end)[11:16]}Z ({who or 'no name'}): no login until it ends")
        t = self.bound(N2_S)
        if t < 30:
            raise Late(f"{t:.0f} s of the pass is left, too little for node 2's session")
        try:
            out = self.n2("python3 -", input=n2_driver(n2_parts(names), t - 25), timeout=t)
        except Refused as e:
            return N2Read(e.status, e.now, refused=e)
        try:
            r = json.loads(out)
        except ValueError:
            raise RuntimeError(f"node 2's session printed no reading: {out.strip()[-200:]!r}") from None
        st.update(read_t=self.now, windows=[w for w in booked(r.get("windows") or "") if w[1] > self.now])
        return N2Read(r.get("status") or "", float(r.get("now") or self.now), r.get("parts") or {}, stopped=r.get("stopped"))

    def n2_part(self, name: str) -> str:
        """The output of part `name` of this pass's node-2 session: a Skip when the session didn't get to run it."""
        r = self.node2(name)
        if r.refused:
            raise r.refused
        got = r.parts.get(name)
        if got is None and r.stopped is not None:
            raise Skip(f"node 2 stopped being quiet during the session: {r.stopped[:160]}", unread=True)
        if got is None:
            raise RuntimeError(f"node 2's session didn't read `{name}`: its monitor doesn't name it (Monitor.n2)")
        if "error" in got:
            raise RuntimeError(f"node 2's {name}: {got['error']}")
        if got.get("rc"):
            raise RuntimeError(f"node 2's {name} exited {got['rc']}: {(got.get('err') or got.get('out') or '').strip()[-300:]}")
        return got.get("out") or ""


def refused(out: str, now: float) -> Refused:
    """N2_GUARD's refusal line (`now` and the age from the status's own time, when the line doesn't carry node 2's)."""
    line = out.strip().splitlines()[0]
    if m := re.match(rf"{re.escape(SKIPPED)} (\d+) (-?\d+|-) ?(.*)", line):
        return Refused(m[3], float(m[1]), None if m[2] == "-" else float(m[2]))
    status = line[len(SKIPPED):].strip()
    try:
        return Refused(status, now, now - epoch(status.split()[0]))
    except (IndexError, ValueError):
        return Refused(status, now, None)


def booked(text: str) -> list[list]:
    """The windows booked in node 2's fill/windows (`2026-10-01T10:00Z 30 # who`, read as dispatch.booked does): [start, end,
    who], epoch seconds."""
    out = []
    for line in text.splitlines():
        body, _, who = line.partition("#")
        f = body.split()
        try:
            s = datetime.fromisoformat(f[0])
            s = (s if s.tzinfo else s.replace(tzinfo=timezone.utc)).timestamp()
            out.append([s, s + 60 * float(f[1] if len(f) > 1 else 30), who.strip()[:80]])
        except (IndexError, ValueError):
            continue
    return out


# Node 2's session, after the guard: `python3 -` runs this, with __ARGS__ the parts to read ({name: [argv, stdin]}) and its limits.
N2_DRIVER = r'''
import calendar, json, subprocess, time
A = json.loads(__ARGS__)
END = time.time() + A["budget"]


def quiet():
    """The fill status, and whether it still lets a read run (the guard's rule, again before each part)."""
    try:
        with open(A["status"], errors="replace") as f:
            s = f.readline().strip()
    except OSError:
        return "", False
    try:
        age = time.time() - calendar.timegm(time.strptime(s.split()[0], "%Y-%m-%dT%H:%M:%SZ"))
    except (IndexError, ValueError):
        return s, False
    return s, A["quiet"] in s and 0 <= age <= A["go_s"]


out = {"status": quiet()[0], "now": time.time(), "parts": {}}
try:
    with open(A["windows"], errors="replace") as f:
        out["windows"] = f.read()[-16384:]
except OSError:
    out["windows"] = None
for name, (argv, stdin) in A["parts"].items():
    s, ok = quiet()
    if not ok:
        out["stopped"] = s
        break
    left = END - time.time()
    if left < 1:
        out["parts"][name] = {"error": "the session ran out of time first"}
        continue
    try:
        r = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=left)
        out["parts"][name] = {"rc": r.returncode, "out": r.stdout, "err": r.stderr[-300:]}
    except subprocess.TimeoutExpired:
        out["parts"][name] = {"error": "timed out after %.0f s" % left}
print(json.dumps(out))
'''


def n2_driver(parts: dict, budget: float, *, status: str = FILL_STATUS, windows: str = WINDOWS, go_s: float = FILL_GO_S) -> str:
    args = {"parts": parts, "budget": budget, "status": status, "windows": windows, "quiet": QUIET, "go_s": go_s}
    return N2_DRIVER.replace("__ARGS__", repr(json.dumps(args)))


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
    """The resource steward's dedupe key for a breach line (its tick.sh): numbers become `#`, a GPU's index stays, and the
    custody backlog's `(oldest <run ids>)` and the inode line's `(growing <dirs>)` go, so a breach keeps its key while the runs
    or directories it names change."""
    line = re.sub(r"\s*\((oldest|growing) [^)]*\)$", "", line)
    return re.sub(r"(^|[^<0-9])[0-9][0-9.,]*", r"\1#", re.sub(r"GPU ([0-9]+)", r"GPU<\1>", line))


def watermarks(node: str) -> Callable[[Ctx], list[Finding]]:
    name = {"n1": "Node 1", "n2": "Node 2"}[node]

    def check(ctx: Ctx) -> list[Finding]:
        rp = _probe()
        if node == "n1":
            out = ctx.sh([*NICE, sys.executable, rp.__file__, "--node", "n1", "--reading"], timeout=150)
        else:
            out = ctx.n2_part("probe")
        r = json.loads(out)
        lim = {**rp.COMMON, **rp.LIMITS[node]}
        base = ctx.memo.get("rate_base")
        lines = rp.evaluate(r, base, lim)
        ctx.memo["rate_base"] = rp.next_base(r, base, lim)
        found = [Finding(norm(ln), f"*{name}: {ln}*", severity="page" if ln.startswith("HARD") else None) for ln in lines]
        return found + [Finding("error:" + e.split(":")[0], f"*{name}'s resource probe can't read {e.split(':')[0]}*: {e}",
                                for_s=1200) for e in r.get("errors") or []]

    check.__doc__ = (f"{name}'s disk, inode, RAM, replay-RAM, idle-GPU-memory and custody-backlog watermarks (resource_probe.py "
                     "--reading; the inode rate's base readings, the last hour's, kept between passes). A HARD line pages; an "
                     "inode rate's only over a base at least 45 min old.")
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
out = []
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
        out.append(os.read(fd, 4096).decode(errors="replace").strip())
    finally:
        os.close(fd)
print(json.dumps(out))
'''


def idle_while_waiting_n2(ctx: Ctx) -> list[Finding]:
    """A gpu-lease waiter on node 2 waits while at least as many GPUs as it asks for are free (the fill runner's free count, in
    the quiet status the session ran under; a dead waiter's file, which nobody holds locked, isn't a waiter). Opens after 5
    minutes of that. A waiter pinned to some GPUs (`on=`) counts only when every GPU is free: the status says how many are free,
    not which."""
    waiters = [fields(w) for w in json.loads(ctx.n2_part("waiters"))]
    st = ctx.node2().status
    m = re.search(r"free GPUs (\d+)/(\d+)", st)
    if not m:
        raise RuntimeError(f"node 2's fill status has no free count: {st[:160]}")
    free, n = int(m[1]), int(m[2])
    hit = held_back(waiters, lambda w: free if not w.get("on") or free == n else 0, ctx.now)
    if not hit:
        return []
    w, waited = hit
    return [Finding("idle", f"*Node 2: {free} of {n} GPUs free while {len(waiters)} gpu-lease waiter(s) wait*: the oldest they "
                    f"could serve, {w.get('who')} (pid {w.get('pid')}, {w.get('n')} GPU), has waited {waited / 60:.0f} min.",
                    for_s=300)]


def fill_status_n2(ctx: Ctx) -> list[Finding]:
    """Node 2's fill status (as this pass's session found it, refused or not, on node 2's clock) is over 5 minutes old: its fill
    runner has stopped, and no other monitor reads node 2 meanwhile (the guard needs a quiet status at most a minute old). Or
    node 2 has refused every read for over 2 hours past the end of any window booked on it: a timed lease never released."""
    r = ctx.node2()
    if r.refused is None:
        ctx.memo.pop("refused_since", None)
    since = ctx.memo.setdefault("refused_since", ctx.now) if r.refused else None
    line = r.status
    try:
        age = r.now - epoch(line.split()[0])
    except (IndexError, ValueError):
        raise RuntimeError(f"node 2's fill status doesn't start with its time: {line[:120]!r}") from None
    if age > FILL_FRESH_S:
        return [Finding("stale", f"*Node 2's fill status is {age / 60:.0f} minutes old* ({FILL_STATUS}: `{line[:160]}`), so the "
                        "fill runner isn't ticking, and the node-2 monitors are skipped until it does.", for_s=300)]
    ended = [end for _, end, _ in ctx.pass_.n2_state.get("windows") or [] if end <= ctx.now]
    if since is None or ctx.now - max([since, *ended]) <= UNWATCHED_S:
        return []
    blind = ", ".join(m.id for m in MONITORS if m.n2)
    return [Finding("unwatched", f"*Node 2 has refused every read since {iso(since)[11:16]}Z*, "
                    f"{(ctx.now - since) / 3600:.1f} hours: {r.refused}. No window booked in {WINDOWS} covers the last "
                    f"{UNWATCHED_S / 3600:.0f}, so {blind} see nothing; a timed window that long is itself the incident.",
                    cleared="*Cleared:* node 2 lets the monitors read it again.")]


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
    fast = fast_preemptions(ctx.n2_part("preempted").splitlines()[1:], ctx.now)
    return [_fast_finding("/run/gpu-lease", "Node 2", fast)] if fast else []


# ---- failed systemd units --------------------------------------------------------------------------------------------------------

EXPECTED_FAILED = {
    "nvidia-fabricmanager.service": "both nodes' GPUs are PCIe, with no NVSwitch to manage, so it fails at every boot",
}
UNITS = "systemctl --failed --no-legend --plain; echo ---; systemctl --user --failed --no-legend --plain"
SESSION_RE = re.compile(r"session-(\d+)\.scope$")


#: A timer's oneshot that fails every run is `failed` only between its runs, so its incident stays open until the unit has
#: been out of `systemctl --failed` this long (vy-custody.service, every 5 minutes, opened and cleared 12 times in an hour on
#: node 1, 8 Oct).
UNITS_CLEAR_S = 1800


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
            text = ctx.n2_part("units")
        else:
            text = ctx.sh(["systemctl", "--failed", "--no-legend", "--plain"])
            env = {**os.environ, "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"}
            r = ctx.run(["systemctl", "--user", "--failed", "--no-legend", "--plain"], capture_output=True, text=True,
                        timeout=ctx.bound(30), env=env)
            if r.returncode and "connect to bus" not in (r.stderr or ""):
                raise RuntimeError(f"systemctl --user exited {r.returncode}: {(r.stderr or '').strip()[-200:]}")
            text += "---\n" + (r.stdout if r.returncode == 0 else "")
        failed = failed_units(text)
        sessions = sorted((u for scope, u in failed if scope == "system" and SESSION_RE.match(u)),
                          key=lambda u: int(SESSION_RE.match(u)[1]))
        out = [Finding(f"{scope}:{u}", f"*{name}: {'the user unit' if scope == 'user' else 'unit'} `{u}` has failed*")
               for scope, u in failed if not (scope == "system" and SESSION_RE.match(u))]
        if sessions:
            span = f"`{sessions[0]}`" + (f" to `{sessions[-1]}`" if len(sessions) > 1 else "")
            out.append(Finding("system:session-*.scope", f"*{name}: {len(sessions)} login session scope"
                               f"{'s have' if len(sessions) > 1 else ' has'} failed* ({span}): those logins (ssh, and the runs sent "
                               "over it) got no scope of their own, so no per-run memory cap. `journalctl -u UNIT` names the cause; once it's fixed, "
                               "`systemctl reset-failed 'session-*.scope'`.",
                               cleared=f"*Cleared:* no failed login session scope on {name.lower()}."))
        return out

    check.__doc__ = (f"A failed systemd unit on {name.lower()}, system or research's user manager (`systemctl --failed`), less "
                     "EXPECTED_FAILED: units that fail on this hardware by design, each with its reason. Failed login "
                     "session scopes are one incident.")
    return check


def n2_parts(names: set[str] | frozenset[str]) -> dict[str, list]:
    """What node 2's session runs for each part a node-2 check reads (Monitor.n2): {name: [argv, stdin]}."""
    every: dict[str, Callable[[], list]] = {
        "probe": lambda: [["python3", "-", "--node", "n2", "--reading"], Path(_probe().__file__).read_text()],
        "waiters": lambda: [["python3", "-"], N2_WAITERS],
        "preempted": lambda: [["sh", "-c", "tail -c 65536 /run/gpu-lease/preempted.log 2>/dev/null || true"], None],
        "units": lambda: [["sh", "-c", UNITS], None]}
    return {n: every[n]() for n in sorted(names)}


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
                    timeout=ctx.bound(900))
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
    """An ask in #agent-coordination with no 👀, reply, ✅ or ❌ after 2 hours (`research msg traffic --json`), to its owner."""
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
    """Node 1's monitor service hasn't finished a pass for 10 minutes (its state's `last_pass`), so no node-side monitor runs; or
    it can't write its alert dir (the state's `sink_error`), so nothing it finds reaches the relay."""
    out = ctx.sh([*SSH, ctx.node1, f"python3 -c \"import json, os; p = '{ctx.node_state}'; "
                  "s = json.load(open(p)) if os.path.exists(p) else None; print('missing' if s is None else s.get('last_pass', 0)); "
                  "print(json.dumps((s or {}).get('sink_error')))\""]).splitlines()
    last, sink = (out[0].strip() if out else ""), (json.loads(out[1]) if len(out) > 1 else None)
    found = []
    age = None if last == "missing" else ctx.now - float(last or 0)
    if age is None or age > 600:
        when = "have never finished a pass" if age is None else f"haven't run for {age / 60:.0f} minutes"
        found.append(Finding("stale", f"*Node 1's monitors {when}* (`{ctx.node_state}`), so nothing watches node 1's or node 2's "
                             "invariants.", cleared="*Cleared:* node 1's monitors run again."))
    if sink:
        found.append(Finding("sink", f"*Node 1's monitors can't write their alert dir*, so nothing they find reaches Slack: {sink}",
                             cleared="*Cleared:* node 1's monitors write their alert dir again."))
    return found


# ---- every machine: its memory, the runs infrastructure killed, the peers it watches, the checks lost -----------------------------

USER_SLICE = "/sys/fs/cgroup/user.slice/user-1001.slice"      # each research run is a session scope here (one ssh login each)
RUNS = "/workspace/research/runs"
REQUESTS = "/workspace/research/requests"
PSI_FULL_PCT = 10.0         # every non-idle task stalled on memory for 10% of the last minute: the machine is thrashing
AVAIL_PCT = 3.0
RUN_ID_RE = re.compile(r"\br\d{8}-\d{6}-[0-9a-f]{4}\b")
OOM_LINE_RE = re.compile(r"^(\d+)(?:\.\d+)?\s.*\boom-kill:(.*)$")
PEER_S = 30
TERMINAL = ("done", "failed", "cancelled")       # remote.TERMINAL
# a full check ended by the machine rather than by its own result: memory, disk, a lost runner or a signal nobody sent
LOST_CLASSES = ("CGROUP_OOM", "CGROUP_OOM_ATTRIBUTED", "OOM_UNRESOLVED_ORIGIN", "STORAGE_EXHAUSTED", "CANCELLED_DEADMAN",
                "CANCELLED_GUARDIAN", "UNKNOWN_SIGNAL")
LOST_TARGET_PCT = 2.0


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def run_scopes(ctx: Ctx) -> dict[str, dict]:
    """Each session scope under USER_SLICE: its memory now, and the run whose `research run --id` it holds (None: not a run)."""
    out = {}
    for d in sorted(ctx.path(USER_SLICE).glob("session-*.scope")):
        rid = None
        for pid in _read(d / "cgroup.procs").split()[:200]:
            argv = _read(ctx.path(f"/proc/{pid}/cmdline")).split("\0")
            if "research" in argv and "run" in argv and "--id" in argv[:-1]:
                rid = argv[argv.index("--id") + 1]
                break
        cur = _read(d / "memory.current").strip()
        out[d.name] = {"run": rid, "bytes": int(cur) if cur.isdigit() else 0}
    return out


def submitter(ctx: Ctx, rid: str) -> str | None:
    """Who sent run `rid`: its request's `submitted_by`, else the `--by` its command passes (check's slot.py)."""
    try:
        req = json.loads(_read(ctx.path(REQUESTS) / rid / "request.json") or "{}")
    except ValueError:
        return None
    if req.get("submitted_by"):
        return str(req["submitted_by"])
    cmd = req.get("command") or []
    return str(cmd[cmd.index("--by") + 1]) if isinstance(cmd, list) and "--by" in cmd[:-1] else None


def _biggest(scopes: dict[str, dict], n: int = 3) -> str:
    top = sorted(scopes.items(), key=lambda kv: -kv[1]["bytes"])[:n]
    return ", ".join(f"{v['run'] or name} {v['bytes'] / 2**30:.0f}G" for name, v in top) or "no session scopes"


def memory_pressure(ctx: Ctx) -> list[Finding]:
    """The machine thrashing: every non-idle task stalled on memory for PSI_FULL_PCT of the last minute (/proc/pressure/memory),
    or under AVAIL_PCT of its memory available, held for 5 minutes. Names the three biggest runs."""
    psi = re.search(r"^full avg10=[\d.]+ avg60=([\d.]+)", _read(ctx.path("/proc/pressure/memory")), re.M)
    mem = dict(re.findall(r"^(\w+):\s+(\d+) kB", _read(ctx.path("/proc/meminfo")), re.M))
    if psi is None or "MemTotal" not in mem:
        raise RuntimeError("no /proc/pressure/memory or /proc/meminfo to read")
    full, avail = float(psi[1]), 100 * int(mem.get("MemAvailable", 0)) / int(mem["MemTotal"])
    if full < PSI_FULL_PCT and avail >= AVAIL_PCT:
        return []
    return [Finding("memory", f"*{ctx.machine or 'This machine'} is short of memory*: every task stalled on memory {full:.0f}% of "
                    f"the last minute, {avail:.1f}% of its {int(mem['MemTotal']) / 2**20:.0f}G available. Biggest: "
                    f"{_biggest(run_scopes(ctx))}.", cleared=f"*Cleared:* {ctx.machine}'s memory pressure is back under "
                    f"{PSI_FULL_PCT:.0f}%.")]


def oom_kills(ctx: Ctx) -> list[Finding]:
    """A process the kernel's OOM killer took in the last 2 hours (`journalctl -k`'s oom-kill lines), each once: the run whose
    scope it was in (the scopes are read every pass, before a kill removes one), who sent it, and which limit it hit."""
    seen = ctx.memo.setdefault("scopes", {})
    out = []
    r = ctx.run(["journalctl", "-k", "-o", "short-unix", "--no-pager", "--since", "-2h"], capture_output=True, text=True,
                timeout=ctx.bound(60))
    if r.returncode or "not seeing messages" in (r.stderr or ""):
        raise RuntimeError(f"journalctl -k can't read the kernel's log (exit {r.returncode}): {(r.stderr or '').strip()[-200:]}")
    for ln in r.stdout.splitlines():
        if not (m := OOM_LINE_RE.match(ln.strip())):
            continue
        f = dict(re.findall(r"(?:^|,)(oom_memcg|task_memcg|task|pid)=([^,]*)", m[2]))
        scope = (f.get("task_memcg") or "").rsplit("/", 1)[-1]
        rid = (seen.get(scope) or {}).get("run")
        who = submitter(ctx, rid) if rid else None
        limit = f"the limit of `{f['oom_memcg']}`" if f.get("oom_memcg") else "the whole machine's memory"
        run = f"run {rid}" + (f", sent by {who}" if who else "") if rid else f"`{f.get('task_memcg') or '?'}`, no research run"
        out.append(Finding(f"{m[1]}:{f.get('pid', '?')}", f"*{ctx.machine}: the OOM killer took {f.get('task', '?')} (pid "
                           f"{f.get('pid', '?')}) of {run}* at {iso(float(m[1]))}: it hit {limit}, so its whole scope was killed "
                           "and the machine kept running."))
    for name, v in run_scopes(ctx).items():
        seen[name] = {**v, "t": ctx.now}
    for name in [n for n, v in seen.items() if ctx.now - v.get("t", 0) > 3 * 3600]:
        del seen[name]
    return out


def answering(ctx: Ctx) -> list[Finding]:
    """A peer machine that doesn't answer `ssh PEER true` within PEER_S, held for 3 minutes: hung, stopped or off the network."""
    out = []
    for peer in ctx.peers:
        try:
            ctx.sh([*SSH, peer, "true"], timeout=PEER_S)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            why = "no answer in %d s" % PEER_S if isinstance(e, subprocess.TimeoutExpired) else str(e)[-200:]
            out.append(Finding(peer, f"*{peer} isn't answering {ctx.machine}*: {why}.",
                               cleared=f"*Cleared:* {peer} answers {ctx.machine} again."))
    return out


WORKSPACE = "/workspace"
DISK_FULL_PCT = 85.0        # resource_probe's HARD /workspace line on node 1: no new check starts past it
INODES_FULL_PCT = 80.0      # resource_probe's inode_pct
FORECAST_S = 6 * 3600       # page while the line reaches FULL within this
FIT_S = 3 * 3600            # the readings the line is fitted to
FIT_MIN_S, FIT_MIN_N = 1800, 4
DU_S = 60


def disk_use(path: Path) -> dict[str, float]:
    s = os.statvfs(path)
    used, free = (s.f_blocks - s.f_bfree) * s.f_frsize, s.f_bavail * s.f_frsize
    return {"pct": 100.0 * used / (used + free) if used + free else 0.0, "size_b": float(used + free),
            "ipct": 100.0 * (s.f_files - s.f_ffree) / s.f_files if s.f_files else 0.0}


def _slope_per_h(pts: list[tuple[float, float]]) -> float:
    n = len(pts)
    mt, mv = sum(t for t, _ in pts) / n, sum(v for _, v in pts) / n
    den = sum((t - mt) ** 2 for t, _ in pts)
    return 3600 * sum((t - mt) * (v - mv) for t, v in pts) / den if den else 0.0


def _run_disk(ctx: Ctx) -> str:
    """The live runs' directories by size, biggest first (`du`, cut off after DU_S: what it measured by then)."""
    runs = sorted({v["run"] for v in run_scopes(ctx).values() if v["run"]})
    dirs = [str(ctx.path(RUNS) / r) for r in runs if (ctx.path(RUNS) / r).is_dir()]
    if not dirs:
        return "no live run has a run directory"
    bound = int(ctx.bound(DU_S))
    out = ctx.sh(["timeout", str(max(1, bound - 5)), *NICE, "du", "-xsB1G", *dirs], timeout=bound, ok=(0, 1, 124))
    sizes = sorted(((int(g), Path(p).name) for g, p in (ln.split("\t", 1) for ln in out.splitlines() if "\t" in ln)
                    if g.isdigit()), reverse=True)[:3]
    return ", ".join(f"{rid} {g}G" + (f" (sent by {who})" if (who := submitter(ctx, rid)) else "") for g, rid in sizes) or \
        f"`du` measured none of {len(dirs)} in {bound} s"


def disk_forecast(ctx: Ctx) -> list[Finding]:
    """/workspace's disk or inodes reaching DISK_FULL_PCT / INODES_FULL_PCT within 6 hours on the line fitted to the last 3
    hours of readings (kept in the memo; at least 4 over 30 minutes), or already there. Names the biggest live runs on disk."""
    now = disk_use(ctx.path(WORKSPACE))
    pts = [p for p in ctx.memo.get("readings", []) if ctx.now - FIT_S <= p[0] < ctx.now] + [[ctx.now, now["pct"], now["ipct"]]]
    ctx.memo["readings"] = pts
    found, biggest, span = [], None, pts[-1][0] - pts[0][0]
    for i, (what, full) in enumerate((("disk use", DISK_FULL_PCT), ("inode use", INODES_FULL_PCT)), start=1):
        pct = pts[-1][i]
        rate = _slope_per_h([(p[0], p[i]) for p in pts]) if len(pts) >= FIT_MIN_N and span >= FIT_MIN_S else 0.0
        if pct >= full:
            when = f"is past {full:.0f}%"
        elif rate > 0 and (full - pct) / rate * 3600 <= FORECAST_S:
            when = f"reaches {full:.0f}% in about {(full - pct) / rate:.1f} h"
        else:
            continue
        biggest = biggest or _run_disk(ctx)
        gain = f"{rate:+.2f} points/h" + (f" ({rate * now['size_b'] / 100 / 2**30:+,.0f} GiB/h)" if i == 1 else "")
        found.append(Finding(what.split()[0], f"*{ctx.machine or 'This machine'}: /workspace's {what} {when}*: {pct:.1f}% now, "
                             f"{gain} over the last {span / 3600:.1f} h. Biggest live runs on disk: {biggest}.",
                             cleared=f"*Cleared:* {ctx.machine}'s /workspace {what} no longer reaches {full:.0f}% within "
                                     f"{FORECAST_S // 3600} h."))
    return found


MANIFEST = Path(__file__).resolve().parent / "pods/nebius/deploy.toml"     # installed beside this file on every machine
UNIT_SUFFIX_RE = re.compile(r"\.(service|timer|socket|path|target|mount)$")
ENABLED = ("enabled", "enabled-runtime", "static", "indirect", "generated", "alias")
EXPECTED_OFF: dict[tuple[str, str], str] = {     # (machine, `systemd:UNIT`): why a unit deploy.toml runs is off there by design
    ("vy-nebius-1", "systemd:vy-n2-offload.service"): "node 2 runs checks, not its fill queue, since its restart on 8 Oct, so "
                                                      "nothing on node 2 would take an offloaded build",
    ("vy-nebius-1", "systemd:vy-n2-commit-offload.service"): "as vy-n2-offload: no fill runner on node 2 to take a Commit",
}


def expected_units(machine: str, manifest: str) -> list[str]:
    """`systemd:UNIT` / `user-systemd:UNIT` for each unit a `run` of deploy.toml names on `machine`, less held entries' and
    EXPECTED_OFF; a unit without a suffix is a service (ops deploy's unit_name)."""
    import tomllib

    m, out = tomllib.loads(manifest), set()
    for e in m.get("file", []) + m.get("tree", []):
        if machine not in e.get("nodes", []) or e.get("hold"):
            continue
        for r in e.get("run", []):
            kind, _, name = r.partition(":")
            if kind in ("systemd", "user-systemd") and name:
                out.add(f"{kind}:{name if UNIT_SUFFIX_RE.search(name) else name + '.service'}")
    return sorted(out - {u for (mm, u) in EXPECTED_OFF if mm == machine})


def unit_states(ctx: Ctx, scope: str, names: list[str]) -> dict[str, dict[str, str]]:
    """`systemctl show`'s load, unit-file and active state of each of `names` in `scope` (`systemd`, or research's user manager)."""
    argv = ["systemctl", *(["--user"] if scope == "user-systemd" else []), "show", "--no-pager",
            "-p", "LoadState", "-p", "UnitFileState", "-p", "ActiveState", "-p", "SubState", *names]
    env = {**os.environ, "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"}
    r = ctx.run(argv, capture_output=True, text=True, timeout=ctx.bound(30), env=env)
    if r.returncode:
        raise RuntimeError(f"`{' '.join(argv[:3])}` exited {r.returncode}: {(r.stderr or '').strip()[-200:]}")
    blocks = [dict(ln.split("=", 1) for ln in b.splitlines() if "=" in ln) for b in r.stdout.strip().split("\n\n")]
    if len(blocks) != len(names):
        raise RuntimeError(f"`systemctl show` gave {len(blocks)} units for {len(names)}")
    return dict(zip(names, blocks))


def units_expected(ctx: Ctx) -> list[Finding]:
    """A unit deploy.toml runs on this machine (its `run`s, beside this file) that was never installed, isn't enabled or isn't
    active, held for 5 minutes; less EXPECTED_OFF."""
    want = expected_units(ctx.machine, MANIFEST.read_text())
    out = []
    for scope in ("systemd", "user-systemd"):
        names = [u.split(":", 1)[1] for u in want if u.startswith(scope + ":")]
        for name, s in (unit_states(ctx, scope, names) if names else {}).items():
            off = []
            if s.get("LoadState") == "not-found":
                off.append("was never installed (systemd doesn't know it)")
            else:
                if s.get("UnitFileState") not in ENABLED:
                    off.append(f"isn't enabled ({s.get('UnitFileState') or '?'}), so it won't start at boot")
                if s.get("ActiveState") not in ("active", "reloading"):
                    off.append(f"isn't running ({s.get('ActiveState') or '?'}/{s.get('SubState') or '?'})")
            if off:
                what = "the user unit" if scope == "user-systemd" else "unit"
                out.append(Finding(f"{scope}:{name}", f"*{ctx.machine}: {what} `{name}` {' and '.join(off)}*, though deploy.toml "
                                   "runs it there.", cleared=f"*Cleared:* {ctx.machine}'s `{name}` is enabled and running."))
    return out


def is_full_check(req: dict) -> bool:
    args, cmd = [str(a) for a in req.get("harness_args") or []], " ".join(str(a) for a in req.get("command") or [])
    return ("--tool" in args[:-1] and args[args.index("--tool") + 1] == "check") or "check/slot.py" in cmd or "check/check.py" in cmd


def lost_checks(ctx: Ctx) -> list[Finding]:
    """Yesterday's (UTC) line: of the full checks that started on this machine, how many infrastructure ended (LOST_CLASSES, or
    a run that never ended), against the CI page's target of under LOST_TARGET_PCT%."""
    day = time.strftime("%Y%m%d", time.gmtime(ctx.now - 86400))
    total, lost, live = 0, [], {v["run"] for v in run_scopes(ctx).values()}
    for d in sorted(ctx.path(RUNS).glob(f"r{day}-*")):
        try:
            req = json.loads(_read(ctx.path(REQUESTS) / d.name / "request.json") or "{}")
        except ValueError:
            continue
        if not is_full_check(req):
            continue
        total += 1
        try:
            cls = json.loads(_read(d / "failure.json") or "{}").get("class")
            st = (json.loads(_read(d / "status.json") or "{}").get("transitions") or [{}])[-1].get("state")
        except ValueError:
            cls, st = None, None
        if cls in LOST_CLASSES or (cls is None and st not in TERMINAL and d.name not in live):
            lost.append(f"{d.name} ({cls or 'never ended'})")
    pct = 100 * len(lost) / total if total else 0.0
    date = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    return [Finding(date, f"*{ctx.machine}, {date} UTC: {len(lost)} of {total} full checks lost to infrastructure ({pct:.1f}%; the "
                    f"target is under {LOST_TARGET_PCT:.0f}%)*" + (f": {', '.join(lost[:8])}" if lost else "."),
                    severity="page" if pct >= LOST_TARGET_PCT and lost else None)]


# ---- failed checks (relay-side: GitHub and the Slack registry name each PR's lane) ---------------------------------------------

MERGE_THREAD = "1790957906.278529"      # #agent-coordination's merge thread, where the lander and the lanes follow the trains
LANDER = "old-circuits-and-proofs"
REPO_SLUG = "danielreuter/verity"
# Node 2 too without N2_GUARD: its check slots run whole checks at any time, so one niced read of their run files a minute is
# nothing they don't already bear (and its guard refuses everything while its fill runner is stopped)
CHECK_MACHINES = (("vy-nebius-1", "n1"), ("vy-nebius-2", "vy-n2"), ("vy-nebius-cpu-1", "vy-nebius-cpu-1"))
SLOT_WRAPPER = "check/slot.py"
CHECK_LINE = "/workspace/research/locks/check-line"     # slot.py's LINE beside its SLOTS: a waiter's ticket, locked while it waits
NOT_STARTED = 75        # slot.py's exit when CMD never ran: a waiter past --wait-max, or a twin whose twin took a slot first
FAILED_RECENT_S = 6 * 3600
#: how long a run that ended before any step failed may lack its failure.json: `research run` classifies the end a few seconds
#: after status.json says so, and a post read in between said "unclassified: no reason recorded" of a hand cancel (dc04, 12 Oct)
CLASSIFY_S = 120
TRAIN_RE = re.compile(r"^Merge #(\d+) \(([0-9a-f]+)\)")
TRAIN_MAIN_RE = re.compile(r"^Merge main \([0-9a-f]+\)")
BC_ID_RE = re.compile(r"bc-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
LANE_RE = re.compile(r"-([0-9a-f]{4})$")
# a log's first failing line: the first line matching the first of these that any line matches, else its last line
FAILING = (r"\b(?:FAILED|ERROR) \S+::\S+", r"(?i)\berror(?:\[\w+\])?:", r"\bFAIL(?:ED)?\b|\bfailed\b|Error\b|Traceback")

# A machine's failed checks: `python3 -` runs this there, with __ARGS__ its paths and limits. It prints {"rows": one a run,
# "waiting": each live ticket in slot.py's line (one another process holds locked: slot.py's `waiting`)}.
FAILED_READER = r'''
import fcntl, json, os, re, subprocess, time
A = json.loads(__ARGS__)
PICK = [re.compile(p) for p in A["pick"]]
SUITE = re.compile(r"^(passed|failed|reused|skipped)\s+(\S+)\s")


def load(p):
    try:
        with open(p, errors="replace") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def first_failing(p):
    try:
        with open(p, errors="replace") as f:
            lines = f.read(A["log_bytes"]).splitlines()
    except OSError:
        return None
    for rx in PICK:
        for ln in lines:
            if rx.search(ln):
                return ln.strip()[:300]
    return next((ln.strip()[:300] for ln in reversed(lines) if ln.strip()), None)


def failed_suite(p):
    """The first suite suites.py reports failed in the run's stdout (`failed  NAME  ...`, followed by its output up to the next
    suite's line), and that output's first failing line: a suites.py run writes no steps.jsonl."""
    try:
        with open(p, errors="replace") as f:
            lines = f.read(A["log_bytes"]).splitlines()
    except OSError:
        return None
    for i, ln in enumerate(lines):
        if m := SUITE.match(ln):
            if m[1] != "failed":
                continue
            out = []
            for later in lines[i + 1:]:
                if SUITE.match(later):
                    break
                out.append(later)
            line = next((o.strip()[:300] for rx in PICK for o in out if rx.search(o)), None)
            return m[2], line or ln.strip()[:300]
    return None


def steps(p):
    out = []
    try:
        with open(p, errors="replace") as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


now, rows = time.time(), []
try:
    names = sorted(n for n in os.listdir(A["runs"]) if any(n.startswith("r" + d + "-") for d in A["days"]))
except OSError:
    names = []
for n in names:
    d = os.path.join(A["runs"], n)
    ts = [os.stat(p).st_mtime for p in (os.path.join(d, f) for f in ("status.json", "steps.jsonl", "stdout.log")) if os.path.exists(p)]
    if not ts or now - max(ts) > A["recent_s"]:
        continue
    req = load(os.path.join(A["requests"], n, "request.json"))
    if A["wrapper"] not in " ".join(str(a) for a in req.get("command") or []):
        continue
    st = load(os.path.join(d, "status.json"))
    last = (st.get("transitions") or [{}])[-1]
    bad = next((s for s in steps(os.path.join(d, "steps.jsonl")) if s.get("status") not in ("passed", "skipped")), None)
    suite = failed_suite(os.path.join(d, "stdout.log")) if bad is None else None
    if suite:
        bad = {"step": "suite " + suite[0], "ended_utc": last.get("ended_utc") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
               "rc": None, "line": suite[1], "suite": True}
    state = st.get("state") or last.get("state")
    if bad is None and (state not in ("failed", "cancelled") or last.get("rc") == A["not_started"]):
        continue
    fail = load(os.path.join(d, "failure.json"))
    if bad is None and not fail and now - max(ts) < A["classify_s"]:
        continue
    sha, cwd = (req.get("remote") or {}).get("source_sha"), req.get("cwd") or ""
    row = {"run": n, "state": state, "rc": last.get("rc"), "ended": last.get("ended_utc"), "class": fail.get("class"),
           "why": (fail.get("why") or "")[:300], "by": req.get("submitted_by") or "", "sha": sha, "subjects": []}
    if bad is not None:
        row.update(step=bad.get("step"), step_ended=bad.get("ended_utc"), rc=bad.get("rc"), suite=bad.get("suite", False),
                   line=bad.get("line") or (first_failing(os.path.join(d, bad["log"])) if bad.get("log") else None))
    if sha and os.path.isdir(cwd):
        try:
            r = subprocess.run(["git", "-C", cwd, "log", "--first-parent", "--format=%s", "-n", "40", sha],
                               capture_output=True, text=True, timeout=10)
            row["subjects"] = r.stdout.splitlines() if r.returncode == 0 else []
        except (OSError, subprocess.TimeoutExpired):
            pass
    rows.append(row)
waiting = []
try:
    tickets = sorted(t for t in os.listdir(A["line"]) if not t.startswith("."))
except OSError:
    tickets = []
for t in tickets:
    try:
        fd = os.open(os.path.join(A["line"], t), os.O_RDONLY)
    except OSError:
        continue
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        continue
    except BlockingIOError:
        pass
    finally:
        os.close(fd)
    who = load(os.path.join(A["line"], t))
    waiting.append({"run": who.get("run"), "by": who.get("by")})
print(json.dumps({"rows": rows, "waiting": waiting}))
'''


def failed_reader(now: float, *, runs: str = RUNS, requests: str = REQUESTS, line: str = CHECK_LINE) -> str:
    days = sorted({time.strftime("%Y%m%d", time.gmtime(now - s)) for s in (0, FAILED_RECENT_S)})
    args = {"runs": runs, "requests": requests, "line": line, "days": days, "recent_s": FAILED_RECENT_S, "wrapper": SLOT_WRAPPER,
            "pick": list(FAILING), "log_bytes": 8 << 20, "not_started": NOT_STARTED, "classify_s": CLASSIFY_S}
    return FAILED_READER.replace("__ARGS__", repr(json.dumps(args)))


def _registry() -> dict:
    from . import slack

    return slack.load_registry(slack.registry_path())


def agent_ids(reg: dict, handle: str) -> list[str]:
    return next((BC_ID_RE.findall(h.get("holder") or "") for h in reg["handles"] if h["handle"] == handle), [])


def lane_of(reg: dict, branch: str) -> str | None:
    """The handle whose holder's agent id ends in `branch`'s suffix (`cursor/x-558b`: the agent bc-…558b)."""
    m = LANE_RE.search(branch)
    return next((h["handle"] for h in reg["handles"] if m and any(i.endswith(m[1]) for i in BC_ID_RE.findall(h.get("holder") or ""))),
                None)


def train_prs(subjects: list[str]) -> list[int]:
    """The PRs of a train: its first-parent subjects from the head, while each is the lander's `Merge #N (sha)` (or a `Merge
    main (sha)` that brings the train up to main)."""
    out = []
    for s in subjects:
        if m := TRAIN_RE.match(s):
            out.append(int(m[1]))
        elif not TRAIN_MAIN_RE.match(s):
            break
    return out


def _open_prs(ctx: Ctx, row: dict) -> list[tuple[int, str]]:
    """(number, branch) of each open PR the run checks: a train's, else the PR whose head it is."""
    gh = ["gh", "-R", REPO_SLUG, "pr"]
    if nums := list(dict.fromkeys(train_prs(row.get("subjects") or []))):
        out = []
        for n in nums:
            pr = json.loads(ctx.sh([*gh, "view", str(n), "--json", "headRefName,state"], timeout=30))
            if pr.get("state") == "OPEN":
                out.append((n, pr.get("headRefName") or ""))
        return out
    if not row.get("sha"):
        return []
    got = json.loads(ctx.sh([*gh, "list", "--state", "open", "--search", row["sha"], "--json", "number,headRefName"], timeout=30))
    return [(p["number"], p.get("headRefName") or "") for p in got]


def _hm(utc: str | None) -> str:
    return f"{utc[11:16]}Z" if utc and len(utc) >= 16 else "?"


def failed_check(ctx: Ctx, reg: dict, machine: str, row: dict) -> Finding | None:
    """The post for one failed check run, or None for one nobody needs telling about: cancelled by hand before any step failed
    (whoever cancelled it knows), or neither a train nor sent by the lander (a lane's own check: it reads its own result)."""
    if row.get("step") is None and row.get("class") == "CANCELLED_MANUAL":
        return None
    if not train_prs(row.get("subjects") or []) and not any(i in row.get("by", "") for i in agent_ids(reg, LANDER)):
        return None
    try:
        prs, unread = _open_prs(ctx, row), ""
    except Exception as e:  # noqa: BLE001  (the lander still hears of it)
        prs, unread = [], f" Its PRs are unread ({type(e).__name__}: {str(e)[:120]})."
    run, clean = row["run"], lambda s: (s or "").replace("`", "'")
    if row.get("step"):
        going = " (the run goes on)" if row.get("state") == "running" else ""
        what = (f"*Check `{run}` failed step `{row['step']}` on {machine}* at {_hm(row.get('step_ended'))}{going}: its log's "
                f"first failing line is `{clean(row.get('line')) or 'none'}`.")
    else:
        how = "was killed" if row.get("state") == "cancelled" or (row.get("rc") or 0) > 128 else f"ended rc {row.get('rc')}"
        what = (f"*Check `{run}` {how} on {machine}* at {_hm(row.get('ended'))}, before any step failed "
                f"({row.get('class') or 'unclassified'}): {clean(row.get('why')) or 'no reason recorded'}.")
    lanes = [lane_of(reg, b) for _, b in prs]
    if prs:
        what += f" {'Train' if train_prs(row.get('subjects') or []) else 'PR'} at `{(row.get('sha') or '?')[:9]}`: " + \
            ", ".join(f"#{n} `{b}`" + ("" if h else " (no lane in the registry)") for (n, b), h in zip(prs, lanes)) + "."
    infra = () if row.get("step") else ("infra",)
    return Finding(f"{machine}:{run}", what + unread, owners=tuple(dict.fromkeys([*(h for h in lanes if h), LANDER, *infra])),
                   thread=MERGE_THREAD)


def cancel_failed(ctx: Ctx, machine: str, row: dict, waiting: list[dict]) -> str | None:
    """Cancel a check whose step failed while a ticket waits for a check slot on its machine (top, 8 Oct: running on to find more
    failures is worth it only while nothing waits for its slot): what the post says, or None when it isn't cancelled. A suites.py
    run's failed suite only rings: the lander runs those to land a PR on, and decides whether a suite that failed is its change's
    (top, 9 Oct: 25ea's verity-ops failure was cpu-1's, and its circuit_check worth keeping)."""
    if not (ctx.act and waiting and row.get("step") and not row.get("suite") and row.get("state") == "running"):
        return None
    who = ", ".join(w.get("run") or "a check" for w in waiting[:3]) + (f" and {len(waiting) - 3} more" if len(waiting) > 3 else "")
    ctx.sh([*ctx.research, "cancel", row["run"], "--on", machine, "--by", "infra", "--reason",
            f"failed-checks: step {row['step']} failed while {len(waiting)} waited for a check slot on {machine} ({who})"],
           timeout=120)
    return f"Cancelled at {iso(ctx.now)[11:16]}Z, since {who} waited for a check slot on {machine}."


def failed_checks(ctx: Ctx) -> list[Finding]:
    """Each check run through slot.py on a Nebius machine with a failed step (while it still runs, under --keep-going), or that
    ended non-zero or was killed, in the last FAILED_RECENT_S: one post in the merge thread, leading with the lane of each open
    PR it checks (the handle its branch suffix names in the registry) and the lander. One still running whose step failed is
    cancelled once a ticket waits for a check slot on its machine, and the post says so (or a second one, when it comes later)."""
    reg, memo, errors, out = _registry(), ctx.memo.setdefault("runs", {}), [], []
    for machine, host in CHECK_MACHINES:
        script = failed_reader(ctx.now)
        try:
            got = json.loads(ctx.sh([*SSH, host, f"timeout -k 5 50 {' '.join(NICE)} python3 -"], input=script, timeout=60))
            rows, waiting = got["rows"], got["waiting"]
        except Exception as e:  # noqa: BLE001  (the other machines still post)
            errors.append(f"{machine}: {type(e).__name__}: {e}"[:300])
            continue
        for row in rows:
            key, new = f"{machine}:{row['run']}", f"{machine}:{row['run']}" not in memo
            if new:
                f = failed_check(ctx, reg, machine, row)
                memo[key] = {"t": ctx.now, "f": None if f is None else {"text": f.text, "owners": list(f.owners)}}
            if not (got := memo[key]["f"]):
                continue
            if "cancelled" not in got:
                try:
                    if said := cancel_failed(ctx, machine, row, waiting):
                        got["cancelled"] = said
                        if new:
                            got["text"] += " " + said
                except Exception as e:  # noqa: BLE001  (tried again next pass, while it still runs)
                    got["cancel_error"] = f"{type(e).__name__}: {e}"[:300]
            out.append(Finding(key, got["text"], owners=tuple(got["owners"]), thread=MERGE_THREAD))
            if got.get("cancelled") and not got["text"].endswith(got["cancelled"]):
                out.append(Finding(f"{key}:cancelled", f"*Check `{row['run']}` on {machine}*: {got['cancelled']}",
                                   owners=tuple(got["owners"]), thread=MERGE_THREAD))
    for k in [k for k, v in memo.items() if ctx.now - v["t"] > 2 * FAILED_RECENT_S]:
        del memo[k]
    if errors and len(errors) == len(CHECK_MACHINES):
        raise RuntimeError("; ".join(errors))
    return out


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
    Monitor("disk-n2", "infra", "warn", "node", watermarks("n2"), RETENTION, every_s=600, clear_s=1800, n2="probe"),
    Monitor("stalls-n1", "infra", "page", "node", stalls_n1,
            "A blocked pool: stop the process its `blocked` file names (or have its owner stop it), then remove the file. A "
            "dispatcher that fails or doesn't tick: read loop.log and `systemctl status vy-node1-dispatch`, fix the cause, "
            "`sudo systemctl restart vy-node1-dispatch`.", unread_s=600),
    Monitor("idle-while-waiting-n1", "infra", "warn", "node", idle_while_waiting_n1,
            "Why isn't the waiter granted? `cat /run/gpu-lease*/n1-lease.json` (mode, pool, waiting), the pool's cap "
            "(VY_POOL_MAX) and its queue's quota (n1_lease.py's grow rule), and the cluster agent's grants when the brain is on. "
            "A GPU nobody holds while lease waiters wait means the pool can't grow onto it: raise the cap or quota, or move the "
            "waiters."),
    Monitor("idle-while-waiting-n2", "infra", "warn", "node", idle_while_waiting_n2,
            "Read node 2's fill status and the wait files in /run/gpu-lease; while the cluster agent holds agent.lock, it "
            "grants: see its log. A waiter pinned (`on=`) to a busy GPU waits by design.", n2="waiters"),
    Monitor("fill-status-n2", "infra", "warn", "node", fill_status_n2,
            "Stale: restart node 2's fill runner (tmux `pouw-infra-fill`) after reading why it stopped; until its status is "
            "fresh, no monitor reads node 2. Unwatched: find the lease holding `timed` (`/run/gpu-lease` on node 2) and the "
            "window it belongs to; ask its owner to end it, or release it once it is dead."),
    Monitor("fast-preempt-n1", "infra", "warn", "node", fast_preempt_n1, FAST),
    Monitor("fast-preempt-n2", "infra", "warn", "node", fast_preempt_n2, FAST, n2="preempted"),
    Monitor("units-n1", "infra", "warn", "node", units("n1"), UNITS_ACTION, every_s=300, clear_s=UNITS_CLEAR_S),
    Monitor("units-n2", "infra", "warn", "node", units("n2"), UNITS_ACTION, every_s=300, clear_s=UNITS_CLEAR_S, n2="units"),
    Monitor("held-idle", "infra", "warn", "node", held_idle,
            "Your held GPUs sat idle above your target: release GPUs between phases, or lease for less. The `held-idle-hourly` "
            "panel on the live console says where the time went.", every_s=300, events=True),
    Monitor("held-idle-stale", "infra", "warn", "node", held_idle_stale,
            "Check `verity-console.service` and its timer on node 1 (`journalctl -u verity-console -n 50`).", every_s=300),
    Monitor("memory-pressure", "infra", "page", "machine", memory_pressure,
            "Find what is growing (`systemd-cgtop -m` on the machine; the alert names the biggest runs) and tell its sender; a "
            "run past its scope's cap is OOM-killed whole, so pressure that holds means the caps or the slice's headroom are too "
            "loose for what runs there.", for_s=300, clear_s=300),
    Monitor("oom-kills", "infra", "warn", "machine", oom_kills,
            "Tell the run's sender why it died (the alert names both) and what its scope's cap is; if a check, size its "
            "parallelism from memory (check.py's memory_gb) rather than raising the cap.", events=True),
    Monitor("answering", "infra", "page", "machine", answering,
            "`nebius compute instance get` for its state; preserve what it can (its runs' custody), then restart it "
            "(`nebius compute instance stop` and `start`), and tell the owners of the runs it had.", for_s=180, clear_s=120),
    Monitor("disk-forecast", "infra", "page", "machine", disk_forecast,
            "Tell the senders of the runs it names what they are writing; then free space by the retention policy (`research "
            "retention report` on the machine, `research retention gc --apply`).",
            every_s=600, clear_s=1800),
    Monitor("units-expected", "infra", "page", "machine", units_expected,
            "Never installed: `research deploy install MACHINE --by infra --apply PATH` from deploy.toml, then `systemctl "
            "daemon-reload`. Off: `systemctl status UNIT` and `journalctl -u UNIT -n 50` on the machine (`--user` for a user "
            "unit); `systemctl enable --now UNIT` once its cause is fixed. A unit off there by design goes in EXPECTED_OFF in research/monitors.py with "
            "the reason, or out of that machine's `nodes` in deploy.toml.", every_s=300, for_s=300, clear_s=300),
    Monitor("lost-checks", "infra", "warn", "machine", lost_checks,
            "Read each lost check's failure.json; fix the cause on the machine (memory caps, disk, the runner) and tell its "
            "sender to resend it.", every_s=3600, events=True),
    Monitor("drift", "infra", "warn", "relay", deploy_drift,
            "`research deploy drift --node N` for the detail; install main's version (`research deploy install N --by WHO "
            "PATH`), restart what runs old code, or move strays to the attic (`research deploy attic N PATH`).",
            every_s=3600, unread_s=4 * 3600),
    Monitor("unpreserved", "infra", "warn", "relay", unpreserved,
            "`research pods unpreserved MACHINE` lists them and `research data preserved RUN` says why each isn't preserved: "
            "push them, or fix the custody service that should have.", every_s=1800, unread_s=4 * 3600),
    Monitor("asks", "infra", "warn", "relay", asks,
            "Answer the ask, or close it: `research msg close --as YOU --thread TS` (✅), or with `--declined WHY` (❌).",
            every_s=900, unread_s=3 * 3600),
    Monitor("failed-checks", "infra", "warn", "relay", failed_checks,
            "Read the failed step's log in the run's dir on that machine (or `research inspect RUN`): the PR it names fixes it "
            "and says so in the thread. A run that ended before any step failed (killed, out of memory or disk) rings infra too.",
            every_s=60, events=True, unread_s=1800),
    Monitor("node-heartbeat", "infra", "page", "relay", node_heartbeat,
            "`ssh n1 systemctl status vy-monitors.timer vy-monitors.service` and `ssh n1 journalctl -u vy-monitors -n 50`; for "
            "an alert dir that takes no writes, its disk and owner (`df` and `ls -ld` on node 1).",
            every_s=120, unread_s=600),
]
BY_ID = {m.id: m for m in MONITORS}


# ---- sinks ------------------------------------------------------------------------------------------------------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "alert"


def title(text: str) -> str:
    return re.sub(r"[*_`<>]", "", text.splitlines()[0] if text else "")[:160]


def esc(text: str) -> str:
    """`text` for Slack: what a finding quotes (a lease's `who=<!channel>`) can't mention anyone or make a link."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def note(ev: dict, now: float) -> tuple[str, str]:
    """An open as alert_sink.py's note: (file name, markdown), the lane contract's front matter plus the monitor's own fields."""
    name = f"{time.strftime('%Y%m%dT%H%MZ', time.gmtime(now))}-alert-{_slug(ev['monitor'])}-{hashlib.sha256(ev['key'].encode()).hexdigest()[:8]}.md"
    lines = ["---", f"id: {name[:-3]}", "campaign: nebius-monitoring", "lane: infra", "kind: finding", "status: open",
             f"tags: [{ev['severity']}]", "repo: danielreuter/verity", f"origin: {ev['by']} (research monitors, research/monitors.py)",
             f"monitor: {ev['monitor']}", f"key: {json.dumps(ev['key'])}", f"owners: [{', '.join(ev['owners'])}]",
             f"severity: {ev['severity']}", "---", "", f"# {title(ev['text'])}", "", ev["text"], "", f"What to do: {ev['action']}", ""]
    return name, "\n".join(lines)


class AlertDir:
    """An alert dir: monitors.jsonl, and the outbox's notes. A pass killed after writing an event but before saving its state
    writes it again next pass, so `write` leaves out what monitors.jsonl's tail already has."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def check(self) -> None:
        """OSError unless the alert dir takes writes."""
        (self.path / "outbox").mkdir(parents=True, exist_ok=True)
        open(self.path / "monitors.jsonl", "a").close()

    def written(self, ev: dict) -> bool:
        """Whether monitors.jsonl's tail has `ev` already: an open or clear as its key's last, or the event itself."""
        try:
            lines = tail(self.path / "monitors.jsonl")
        except FileNotFoundError:
            return False
        mark, last = json.dumps(ev["key"]), None
        for ln in lines:
            if mark in ln and (e := _event(ln)).get("key") == ev["key"]:
                if ev["event"] == e.get("event") == "event":
                    return True
                last = e.get("event")
        return ev["event"] in ("open", "clear") and last == ev["event"]

    def write(self, ev: dict, now: float) -> bool:
        """Append `ev` (an open with its note); False when monitors.jsonl has it already."""
        if self.written(ev):
            return False
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
        return True


def message(ev: dict, tag: Callable[[str], str] = lambda h: "@" + h) -> str:
    """An event as one Slack post: the owners tagged, the alert, what to do (all but the tags escaped)."""
    if ev["event"] == "clear":
        return esc(ev["text"])
    sev = "[page] " if ev["severity"] == "page" else ""
    return (f"{' '.join(tag(o) for o in ev['owners'])} {sev}{esc(ev['text'])}\n_What to do:_ {esc(ev['action'])}\n"
            f"_monitor `{esc(ev['monitor'])}`, {esc(ev['by'])}_")


def webhook_post(url: str, text: str) -> None:
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()


def _why(e: Exception) -> str:
    """A webhook failure, without the URL a message might carry."""
    return f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else type(e).__name__


# ---- the runner -----------------------------------------------------------------------------------------------------------------

def new_state() -> dict:
    return {"schema": SCHEMA.format("state"), "open": {}, "pending": {}, "events": {}, "monitors": {}, "retry": [], "node2": {}}


def step(state: dict, monitors: list[Monitor], ctx_for: Callable[[dict], Ctx], now: float, sink: AlertDir, *, by: str,
         webhook: str | None = None, post: Callable[[str, str], None] = webhook_post, force: bool = False,
         checkpoint: Callable[[], None] | None = None, pass_s: float | None = PASS_S,
         clock: Callable[[], float] = time.monotonic) -> list[dict]:
    """One pass: each due monitor's check, its incidents opened and cleared through `sink` (and `webhook`), `checkpoint()` after
    each. A check still due at the pass's deadline (`pass_s` on `clock`) is Late and runs next pass. Returns the events."""
    for k, v in new_state().items():
        state.setdefault(k, v)
    seen = state["events"]
    emitted: list[dict] = []

    def emit(m: Monitor, kind: str, key: str, f: Finding | None, o: dict | None = None) -> bool:
        ev = {"schema": SCHEMA.format("event"), "event": kind, "t": iso(now), "monitor": m.id, "key": key, "by": by,
              "owners": list(f.owners if f and f.owners else (o or {}).get("owners") or (m.owner,)),
              "severity": (f.severity if f else None) or (o or {}).get("severity") or m.severity,
              "text": f.text if f else o["cleared"], "action": m.action, "webhook": bool(webhook)}
        if o:
            ev["since"] = o["since"]
        if f and f.thread:
            ev["thread"] = f.thread
        try:
            fresh = sink.write(ev, now)
        except OSError as e:
            state["sink_error"] = f"{iso(now)} {type(e).__name__}: {e}"[:300]
            return False
        state.pop("sink_error", None)
        if fresh and webhook:
            state["retry"].append({"t": now, "text": message(ev)})
        if fresh:
            emitted.append(ev)
        return True

    try:
        sink.check()
        state.pop("sink_error", None)
    except OSError as e:
        state["sink_error"] = f"{iso(now)} {type(e).__name__}: {e}"[:300]
    due = [m for m in monitors if force or now - state["monitors"].get(m.id, {}).get("last_run", 0) >= m.every_s - 5]
    shared = Pass(n2_parts=frozenset(m.n2 for m in due if m.n2), n2_state=state["node2"], clock=clock,
                  deadline=clock() + (pass_s if pass_s is not None else float("inf")))
    for m in due:
        judge(m, state, shared, ctx_for, now, emit)
        if checkpoint:
            checkpoint()
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


def judge(m: Monitor, state: dict, shared: Pass, ctx_for: Callable[[dict], Ctx], now: float,
          emit: Callable[..., bool]) -> None:
    """One monitor's part of a pass: run its check, then open, clear or post what it found."""
    opened, pending, seen = state["open"], state["pending"], state["events"]
    st = state["monitors"].setdefault(m.id, {"since": now})
    ran = st.get("last_run")
    st["last_run"] = now
    try:
        if shared.left() < 5:
            raise Late("the pass's deadline came first")
        ctx = ctx_for(st.setdefault("memo", {}))
        ctx.pass_ = shared
        found: dict[str, Finding | None] = {f.key: f for f in m.check(ctx)}
        st.update(last_ok=now, error=None, skipped=None)
        ok = True
    except Late as e:
        st["skipped"] = str(e)[:300]
        if ran is None:
            del st["last_run"]
        else:
            st["last_run"] = ran
        return
    except Skip as e:
        st["skipped"] = str(e)[:300]
        if not e.unread:
            st["last_ok"] = now
        return
    except Exception as e:  # noqa: BLE001  (an input that can't be read is the monitor's own finding, in time)
        st["error"] = f"{type(e).__name__}: {e}"[:500]
        late = now - st.get("last_ok", st["since"])
        if late < m.unread_s:
            return
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
        f = f"{shlex.quote(self.path)}/monitors.jsonl"
        out = self._sh(f"[ -f {f} ] || {{ echo missing; exit 0; }}; wc -l < {f}; tail -n +{int(after) + 1} {f}").splitlines()
        if not out or out[0].strip() == "missing":
            return None
        return int(out[0].strip() or 0), out[1:]

    def notes(self) -> list[str]:
        return sorted(n for n in self._sh(f"ls -1 {shlex.quote(self.path)}/outbox 2>/dev/null || true").split() if n.endswith(".md"))

    def head(self, name: str) -> str:
        return self._sh(f"head -c 4000 {shlex.quote(self.path)}/outbox/{shlex.quote(name)}")


def _event(line: str) -> dict:
    try:
        ev = json.loads(line)
    except ValueError:
        return {}
    return ev if isinstance(ev, dict) else {}


RELAY_GIVE_UP_S = 3600
# Slack's errors that no retry fixes: the relay drops the post (and logs it)
FINAL = re.compile(r"thread_not_found|message_not_found|channel_not_found|is_archived|msg_too_long|not_in_channel|no_text|"
                   r"invalid_auth|account_inactive")


def relay_pass(rstate: dict, sources: list[Source], post: Callable[[dict, dict | None], dict], now: float, *,
               log: Callable[[str], None] = print, checkpoint: Callable[[], None] | None = None) -> int:
    """Post each source's new events and Grafana's new notes; returns how many. An open's thread (a dict `post` fills in as it
    goes) is kept from its first attempt, so a pointer that fails doesn't post its alert again. A post that fails is logged:
    Slack refusing it for good (FINAL), or its failing for RELAY_GIVE_UP_S, drops it; otherwise its source stops there until
    the next pass, which tries it first. A source that can't be read is logged too, and the others go on. `checkpoint()`
    after each post and each source."""
    posted = 0
    threads, failing = rstate.setdefault("threads", {}), rstate.setdefault("failing", {})

    def one(src: Source, ev: dict) -> bool:
        """Post `ev` once; False when it failed and is its source's first next pass."""
        nonlocal posted
        kind, tkey = ev.get("event"), f"{src.name}|{ev.get('key')}"
        th = threads.get(tkey)
        if not kind or ev.get("webhook") or (kind == "open" and (th or {}).get("done")) or (kind == "clear" and th is None):
            return True
        if kind == "open":
            th = threads.setdefault(tkey, {"t": now})
        ident = f"{src.name}|{kind}|{ev.get('key')}|{ev.get('t', '')}"
        try:
            got = post(ev, th if kind in ("open", "clear") else None)
        except Exception as e:  # noqa: BLE001  (each post fails alone)
            since = failing.setdefault(ident, now)
            why = f"{type(e).__name__}: {e}"[:300]
            if not FINAL.search(str(e)) and now - since < RELAY_GIVE_UP_S:
                log(f"{ident} not posted, {src.name} stops here until the next pass: {why}")
                return False
            log(f"dropped {ident}: {why}")
            got = None
        else:
            posted += 1
        failing.pop(ident, None)
        if kind == "open" and (got is not None or th.get("ts")):
            th.update(got or {}, done=True)
        elif kind in ("open", "clear"):
            del threads[tkey]
        if checkpoint:
            checkpoint()
        return True

    def read(src: Source, s: dict) -> None:
        first = "lines" not in s
        got = src.lines(0 if first else s["lines"])
        if got is not None:
            total, lines = got
            if total < s.get("lines", 0):
                s["lines"] = 0                          # a new file: read from its start next pass
            elif first:
                if all(one(src, ev) for ev in still_open([_event(ln) for ln in lines[:total]])):
                    s["lines"] = total
            else:
                for ln in lines[:total - s["lines"]]:
                    if not one(src, _event(ln)):
                        break
                    s["lines"] += 1
        names = src.notes()
        if "notes" not in s:
            s["notes"] = names
            return
        seen = set(s["notes"]) & set(names)
        for n in names:
            if n in seen:
                continue
            text = src.head(n)
            fm = front_matter(text)
            if "monitor" not in fm and not one(src, {
                    "event": "note", "monitor": "grafana", "key": n, "owners": [fm.get("lane", "infra")], "severity": "warn",
                    "text": text.split("\n---\n", 1)[-1].strip()[:2500], "action": f"read `{n}` in node 1's alert outbox",
                    "by": fm.get("origin", "grafana")}):
                break
            seen.add(n)
            s["notes"] = sorted(seen)
        s["notes"] = sorted(seen)

    for src in sources:
        try:
            read(src, rstate.setdefault("sources", {}).setdefault(src.name, {}))
        except Exception as e:  # noqa: BLE001  (read again next pass)
            log(f"{src.name} unread: {type(e).__name__}: {e}"[:400])
        if checkpoint:
            checkpoint()
    for k in [k for k, v in threads.items() if now - v.get("t", now) > EVENTS_KEEP_S]:
        del threads[k]
    for k in [k for k, t in failing.items() if now - t > EVENTS_KEEP_S]:
        del failing[k]
    return posted


SENT_BY_RE = re.compile(r"sent by https://cursor\.com/agents/bc-([0-9a-f]{8})")


def sender_handles(reg: dict, text: str) -> list[str]:
    """The handles whose holder is the agent an alert says sent the run (holders name agents by at least `bc-` + 8 hex)."""
    ids = set(SENT_BY_RE.findall(text))
    return [h["handle"] for h in reg["handles"] if ids & set(re.findall(r"bc-([0-9a-f]{8})", h["holder"]))]


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

    def reply(ch: str, text: str, ts: str) -> None:
        try:
            slack._post(api, ch, POSTER, text, ts)
        except slack.SlackError as e:
            if not re.search(r"thread_not_found|message_not_found", str(e)):
                raise
            print(f"{ch} {ts}: its thread is gone ({e}), so this reply is a post of its own", flush=True)
            slack._post(api, ch, POSTER, text)

    def post(ev: dict, thread: dict | None) -> dict:
        """Post `ev`, recording each message in `thread` as soon as Slack has it, so a retry posts only what is still missing."""
        th = {} if thread is None else thread
        ev = {**ev, "owners": list(dict.fromkeys([*ev["owners"], *sender_handles(reg, ev["text"])]))}
        if ev["event"] == "clear":
            done = th.setdefault("cleared", [])
            for k, ch in (("ts", alerts), ("pointer", front)):
                if th.get(k) and k not in done:
                    reply(ch, message(ev), th[k])
                    done.append(k)
            return th
        if ev.get("thread"):
            if not th.get("pointer"):
                th["pointer"] = slack._post(api, front, POSTER, message(ev, tag), ev["thread"], route=(reg, routes), tags=("alert",))
            return th
        if not th.get("ts"):
            th["ts"] = slack._post(api, alerts, POSTER, message(ev, tag))
        if (others := [o for o in ev["owners"] if o != "infra"]) and not th.get("pointer"):
            try:
                link = api.call("chat.getPermalink", channel=alerts, message_ts=th["ts"]).get("permalink") or "#agent-alerts"
            except slack.SlackError:
                link = "#agent-alerts"
            th["pointer"] = slack._post(api, front, POSTER, f"{' '.join(tag(o) for o in others)} {esc(title(ev['text']))} "
                                        f"(alert: {link})", route=(reg, routes), tags=("alert",))
        return th
    return post


# ---- the command line -----------------------------------------------------------------------------------------------------------

def _ctx_for(ns: argparse.Namespace, now: float, act: bool = False) -> Callable[[dict], Ctx]:
    return lambda memo: Ctx(now=now, memo=memo, act=act, n2_host=os.environ.get("VY_N2", "vy-n2"), node1=getattr(ns, "node1", "n1"),
                            node_state=getattr(ns, "node_state", STATE), machine=getattr(ns, "machine", None) or os.uname()[1],
                            peers=tuple(getattr(ns, "peer", None) or ()))


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
    ms = select(ns.where, ns.ids)
    shared = Pass(n2_parts=frozenset(m.n2 for m in ms if m.n2))
    for m in ms:
        row: dict[str, Any] = {"id": m.id, "findings": [], "error": None, "skipped": None}
        try:
            ctx = _ctx_for(ns, now)({})
            ctx.pass_ = shared
            row["findings"] = [{"key": f.key, "text": f.text, "owners": list(f.owners or (m.owner,)),
                                "severity": f.severity or m.severity} for f in m.check(ctx)]
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
        evs = step(state, select(where, ns.only), _ctx_for(ns, now, act=True), now, AlertDir(alerts), by=ns.by,
                   webhook=os.environ.get(WEBHOOK_ENV) or None, checkpoint=lambda: save(state_path, state),
                   pass_s=PASS_S if where in ("node", "machine") else None)
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
                          "errors": {k: v["error"] for k, v in state["monitors"].items() if v.get("error")},
                          "sink_error": state.get("sink_error")}, indent=1))
    else:
        for ev in evs:
            print(f"{ev['event']} {ev['key']}: {title(ev['text'])}")
        print(f"{iso(now)} open: {', '.join(sorted(state['open'])) or 'none'}")
    if state.get("sink_error"):
        print(f"monitors: the alert dir {ns.alerts} takes no writes: {state['sink_error']}", file=sys.stderr)
        return 1
    return 0


def cmd_relay(ns: argparse.Namespace) -> int:
    posts = Path(ns.posts)
    fd = lock(posts)
    if fd is None:
        print(f"monitors: refused: another relay holds {posts}.lock", file=sys.stderr)
        return 3
    post = slack_post(ns.dry_run)
    sources = [Source("node1", ns.node_alerts, host=ns.node1), Source("relay", ns.alerts)]
    for spec in ns.machine_alerts or []:
        name, _, where = spec.partition("=")
        host, _, path = where.partition(":")
        if not (name and host and path):
            print(f"monitors: --machine-alerts {spec!r}: want NAME=HOST:DIR", file=sys.stderr)
            return 2
        sources.append(Source(name, path, host=host))
    rstate = json.loads(posts.read_text()) if posts.exists() else {"schema": SCHEMA.format("relay"), "by": ns.by}
    while True:
        now = time.time()
        stamp = iso(now)[11:]
        if run_once(ns, "relay", Path(ns.state), Path(ns.alerts), now) is None:
            print(f"{stamp} relay-side monitors skipped: another runner holds {ns.state}.lock", flush=True)
        checkpoint = None if ns.dry_run else (lambda: save(posts, rstate))
        try:
            n = relay_pass(rstate, sources, post, now, log=lambda s: print(f"{stamp} {s}", flush=True), checkpoint=checkpoint)
            print(f"{stamp} posted {n}", flush=True)
        except Exception as e:  # noqa: BLE001  (relay_pass catches each source and post; this is a bug, logged, not a stop)
            print(f"{stamp} relay failed: {type(e).__name__}: {e}", flush=True)
        if checkpoint:
            checkpoint()
        if ns.once:
            return 0
        time.sleep(ns.every)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research monitors", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="every monitor: owner, severity, where it runs, threshold, action")
    p.add_argument("--where", choices=("node", "relay", "machine"))
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("check", help="evaluate now and write nothing: exit 1 on a finding or a failed check")
    p.add_argument("ids", nargs="*", metavar="ID")
    p.add_argument("--where", choices=("node", "relay", "machine"))
    p.add_argument("--node1", default="n1", help="node 1's ssh host, for relay-side checks")
    p.add_argument("--node-state", default=STATE)
    p.add_argument("--machine", help="this machine's name, for `machine` monitors (default: its hostname)")
    p.add_argument("--peer", action="append", metavar="HOST", help="an ssh host `answering` watches (repeatable)")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("run", help="one pass: open and clear incidents into the alert dir (node 1's service)")
    p.add_argument("--by", required=True, help="who runs it, recorded in every event (vy-monitors@vy-nebius-1)")
    p.add_argument("--where", choices=("node", "relay", "machine"), default="node")
    p.add_argument("--state", default=STATE)
    p.add_argument("--alerts", default=ALERTS)
    p.add_argument("--only", nargs="+", metavar="ID")
    p.add_argument("--node1", default="n1")
    p.add_argument("--node-state", default=STATE)
    p.add_argument("--machine", help="this machine's name, for `machine` monitors (default: its hostname)")
    p.add_argument("--peer", action="append", metavar="HOST", help="an ssh host `answering` watches (repeatable)")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("relay", help="where Slack works: run the relay-side monitors, post every new event to #agent-alerts")
    p.add_argument("--by", required=True, help="who runs it, recorded in the relay-side events")
    p.add_argument("--node1", default="n1", help="node 1's ssh host")
    p.add_argument("--node-alerts", default=ALERTS, help="node 1's alert dir")
    p.add_argument("--node-state", default=STATE, help="node 1's monitor state (its heartbeat)")
    p.add_argument("--machine-alerts", action="append", metavar="NAME=HOST:DIR",
                   help="a machine watchdog's alert dir to post too (repeatable)")
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
