"""GPU usage per node, owner, kind and UTC hour: the GPU-seconds held, busy, idle, unmeasured and useful. The one
computation usage readers take their numbers from: the console computes the rows (`hourly`) and records them on vy-nebius-1
(`RECORD`), its held-idle file is derived from them (`held_idle`), and `cluster usage` (`research usage`) reads the record.

    cluster usage hourly [--node nebius-1|nebius-2|all] [--hours H] [--json] [--file PATH]
    cluster usage wasters --day YYYY-MM-DD [--top N] [--json] [--file PATH]   # held-but-idle GPU-hours by holder, most first

**Inputs** are per-minute readings of each GPU and, on a node with lease pools, the leases they granted (`Lease`, from
node 1's `n1_lease.py` logs). A reading is `(minute, GPU, held busy, held idle, free busy, free idle, holder[, unmeasured])`,
each a share of the minute: node 1's from DCGM's one reading a minute, node 2's from the fraction of POUS's 10 s samples.
Busy is utilization above 0; held is a lease, a pod or process, or 1 GiB of memory in use; unmeasured is time the GPU was
read without utilization (node 2's sampler reads no NVML in a timed window, so a timed sample is unmeasured).

**Rules.**
- A lease holds its GPU for exactly its span, for its owner: its part of each minute is busy, idle and unmeasured in the
  proportions of that GPU's reading (whichever holder the reading names), and unmeasured when there is no reading. The
  rest of the minute counts by the reading, where a lease pool's own pod is `POOL_UNLEASED`: the pool held the GPU and
  leased it to nobody. Leases that overlap on a GPU share its minute in proportion to their parts, so a GPU-minute is
  never held for more than 60 s.
- Unmeasured time is held but neither busy nor idle, as `ledger.gpu_usage` counts unsampled time, so a held-idle share is
  of measured held time.
- A row's `owner` is the lane its holder's GPU-time counts to and its `kind` the job family the holder's name gives (a
  Kueue job's, a SkyPilot cluster's, the `who=` prefix of a node-2 lease, `gpu-lease` for a pool's lease), as the caller's
  `attribute` maps them; not yet a kind registered in `kinds/<owner>.toml`. Time nobody holds has owner and kind None.
- A GPU-minute nobody read (the sampler down, a DCGM gap) is in no row unless a lease held it, so every row of an hour carries
  `minutes_read`, the minutes of that hour with a reading of the node. An hour with fewer than `WHOLE_MIN` has no busy or
  useful share, and `hourly` names it, as it names an hour missing from the record.
- The console records an hour once, `SETTLE_S` after its end, when its last minute's readings are in.

**Useful, today** (`USEFUL_TODAY`): busy time inside a holding the node's records name, a lease or a pod. Busy time on a
GPU that only memory or a process holds (`NO_POD`, `NO_LEASE`), or a lease pool outside its leases (`POOL_UNLEASED`),
isn't. No filler rule applies: neither node marks filler work (no fill job on node 2 has a `filler=` header, no Job on
node 1 a `verity.dev/filler` label; 3 Oct). The vision's useful is narrower, the busy time of a run whose kind's declared
outputs landed in the store. That needs each holding's run, which the readings don't carry yet (node 1: the pod DCGM
names, which the store's `attempts.host` records, as `pool_n1.py`'s delivered rule joins them; node 2: a lease's `run=`,
which `gpu-lease/usage/v1` records and the ledger's `end` records carry), and the store's verdict that the run's declared
outputs were preserved.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

from . import nebius1, nebius2

ROW = "cluster/usage-hour/v1"
HOURLY = "cluster/usage-hourly/v1"
WASTERS = "cluster/usage-wasters/v1"
RECORD = "/workspace/usage/usage-hourly.jsonl"  # on vy-nebius-1: both nodes' rows, each (hour, node) once, as its hour ends
ALL = "(all holders)"
NO_POD = "(no pod: direct or leaked)"
NO_LEASE = "(process, no lease)"
POOL_UNLEASED = "lease pool, not leased"
NOT_USEFUL = (NO_POD, NO_LEASE, POOL_UNLEASED)
USEFUL_TODAY = ("busy inside a lease or a pod the node's records name; not on a GPU only memory or a process holds, nor in a "
                "lease pool's pod outside its leases")
TARGET_BUSY_PCT = 95.0    # Daniel: per node and hour, >= 95% of the GPU time busy
TARGET_USEFUL_PCT = 90.0  # and >= 90% of the busy time useful
STALE_S = 2 * 3600        # the console records an hour within minutes of its end
SETTLE_S = 180            # after an hour ends, its last minute's DCGM scrape and sampler lines are in by then
WHOLE_MIN = 50            # of an hour's 60 minutes, read: fewer, and its shares say little about the hour
TAIL_BYTES = 8 << 20      # of a record, more than a day of rows: what `record` reads for the keys it has
NODES = {"nebius-1": nebius1.NODE, "nebius-2": nebius2.NODE}


class Lease(NamedTuple):
    gpu: int
    t0: float
    t1: float
    owner: str
    kind: str


def hour(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:00Z")


def hour_t(h: str) -> float:
    return datetime.strptime(h, "%Y-%m-%dT%H:00Z").replace(tzinfo=timezone.utc).timestamp()


def _unmeasured(r: tuple) -> float:
    return r[7] if len(r) > 7 else 0.0


def hourly(node: str, minutes: Iterable[tuple], leases: Iterable[Lease] | None = None, *,
           attribute: Callable[[str], tuple[str, str]], pooled: Callable[[str], bool] = lambda holder: False) -> list[dict]:
    """The node's rows, one per UTC hour, owner and kind, by the module's rules: `{schema, hour, node, owner, kind, held_s,
    busy_s, idle_s, unmeasured_s, useful_s, minutes_read}`, held hours first by owner and kind, then the hour's time nobody
    held. `attribute(holder)` is a reading's holder's (owner, kind); `leases` is None on a node without lease pools, and
    `pooled(holder)` says a holder is a pool's own pod."""
    acc: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])  # busy, idle, unmeasured seconds
    minutes = list(minutes)
    by_key = {(r[0], r[1]): r for r in minutes}
    covered: dict[tuple, float] = defaultdict(float)
    parts = []
    for g, a, b, owner, kind in leases or ():
        m = int(a) // 60 * 60
        while m < b:
            f = min(b, m + 60) - max(a, m)
            covered[(m, g)] += f
            parts.append((m, g, f, owner, kind))
            m += 60
    for m, g, f, owner, kind in parts:
        f *= min(1.0, 60 / covered[(m, g)])
        r = by_key.get((m, g))
        x = acc[(hour(m), owner, kind)]
        if r is None:
            x[2] += f
        else:
            x[0] += f * (r[2] + r[4])
            x[1] += f * (r[3] + r[5])
            x[2] += f * _unmeasured(r)
    for r in minutes:
        rest = max(0.0, 60 - covered.get((r[0], r[1]), 0.0))
        if not rest:
            continue
        h, um = hour(r[0]), _unmeasured(r)
        if r[6] and r[2] + r[3] + um:
            owner, kind = attribute(r[6])
            x = acc[(h, owner, POOL_UNLEASED if leases is not None and pooled(r[6]) else kind)]
            x[0] += rest * r[2]
            x[1] += rest * r[3]
            x[2] += rest * um
            um = 0.0
        if r[4] + r[5] + um:
            x = acc[(h, None, None)]
            x[0] += rest * r[4]
            x[1] += rest * r[5]
            x[2] += rest * um
    read: dict[str, set] = defaultdict(set)
    for r in minutes:
        read[hour(r[0])].add(r[0])
    out = []
    for (h, owner, kind), (busy, idle, um) in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1] is None,
                                                                                  kv[0][1] or "", kv[0][2] or "")):
        held = owner is not None
        out.append({"schema": ROW, "hour": h, "node": node, "owner": owner, "kind": kind,
                    "held_s": round(busy + idle + um, 3) if held else 0.0, "busy_s": round(busy, 3), "idle_s": round(idle, 3),
                    "unmeasured_s": round(um, 3), "useful_s": round(busy, 3) if held and kind not in NOT_USEFUL else 0.0,
                    "minutes_read": len(read[h])})
    return out


def held_idle(rows: Iterable[dict], now: float, targets: dict | None = None, names: dict | None = None) -> list[dict]:
    """The console's held-idle lines (`/workspace/usage/held-idle-hourly.jsonl`) from `hourly`'s rows: per UTC hour, node and
    lane (a row's owner; a lease pool outside its leases is `POOL_UNLEASED`'s own lane), and `ALL` for the node's every
    holder, the measured held GPU-hours, their idle part and share, the unmeasured GPU-hours, the lane's target and the
    change in points from its hour before. `names` maps a row's node to the line's; `targets` (node, lane) to a target %."""
    acc: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in rows:
        if r["owner"] is None:
            continue
        node = (names or {}).get(r["node"], r["node"])
        for lane in (POOL_UNLEASED if r["kind"] == POOL_UNLEASED else r["owner"], ALL):
            x = acc[(r["hour"], node, lane)]
            x[0] += r["busy_s"] + r["idle_s"]
            x[1] += r["idle_s"]
            x[2] += r["unmeasured_s"]
    out, cur = [], hour(now)
    for (h, node, lane), (held, idle, um) in sorted(acc.items()):
        prev = next((o for o in reversed(out) if o["node"] == node and o["lane"] == lane), None)
        pct = round(100 * idle / held, 1) if held else None
        out.append({"hour": h, "node": node, "lane": lane, "complete": h != cur, "held_gpu_h": round(held / 3600, 2),
                    "held_idle_gpu_h": round(idle / 3600, 2), "held_idle_pct": pct, "unmeasured_gpu_h": round(um / 3600, 2),
                    "target_pct": (targets or {}).get((node, lane)),
                    "change_pts": round(pct - prev["held_idle_pct"], 1) if pct is not None and prev
                    and prev["held_idle_pct"] is not None and prev["hour"] == hour(hour_t(h) - 3600) else None})
    return out


def record(lines: Iterable[dict], path: str | Path, keys: tuple[str, ...], now: float, since: float = 0) -> int:
    """Append each line of a UTC hour that ended SETTLE_S or more before `now` and started at or after `since` (so the
    readings covered it whole) whose `keys` no line in the file has yet, without its `complete`; returns how many it wrote.
    It reads the file's last TAIL_BYTES for the keys it has, which hold every hour since `since`."""
    path, have = Path(path), set()
    if path.exists():
        with open(path, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - TAIL_BYTES))
            tail = f.read().splitlines()[1 if size > TAIL_BYTES else 0:]
        for ln in tail:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            have.add(tuple(d.get(k) for k in keys))
    new = [ln for ln in lines if since <= hour_t(ln["hour"]) and hour_t(ln["hour"]) + 3600 + SETTLE_S <= now
           and tuple(ln.get(k) for k in keys) not in have]
    if new:
        with open(path, "a") as f:
            f.writelines(json.dumps({k: v for k, v in ln.items() if k != "complete"}) + "\n" for ln in new)
    return len(new)


# ------------------------------------------------------------------------------------------------- reading the record

def node_hours(rows: Iterable[dict]) -> list[dict]:
    """Each (hour, node)'s totals, newest first: GPU-seconds seen, held, busy, idle, unmeasured, useful and held idle, the
    minutes read, busy as a share of the measured time and useful as a share of the busy time (None for an hour read for
    fewer than WHOLE_MIN minutes), and its holdings, most held idle first."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["hour"], r["node"])].append(r)
    out = []
    for (h, node), rs in sorted(by.items(), reverse=True):
        s = {k: sum(r[k] for r in rs) for k in ("held_s", "busy_s", "idle_s", "unmeasured_s", "useful_s")}
        read = max(r.get("minutes_read", 0) for r in rs)
        whole = read >= WHOLE_MIN
        held = sorted((r for r in rs if r["owner"] is not None), key=lambda r: (-r["idle_s"], r["owner"], r["kind"] or ""))
        out.append({"hour": h, "node": node, "gpu_s": round(s["busy_s"] + s["idle_s"] + s["unmeasured_s"], 1),
                    **{k: round(v, 1) for k, v in s.items()}, "held_idle_s": round(sum(r["idle_s"] for r in held), 1),
                    "minutes_read": read,
                    "busy_pct": round(100 * s["busy_s"] / (s["busy_s"] + s["idle_s"]), 1)
                    if whole and s["busy_s"] + s["idle_s"] else None,
                    "useful_pct": round(100 * s["useful_s"] / s["busy_s"], 1) if whole and s["busy_s"] else None,
                    "holdings": [{k: r[k] for k in ("owner", "kind", "held_s", "busy_s", "idle_s", "unmeasured_s", "useful_s")}
                                 for r in held]})
    return out


def wasters(rows: Iterable[dict], top: int) -> list[dict]:
    """The `top` holdings (node, owner, kind) with the most held-but-idle GPU-seconds over `rows`, with their held,
    busy, unmeasured and useful seconds and the idle share of their measured held time."""
    acc: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if r["owner"] is not None:
            for k in ("held_s", "busy_s", "idle_s", "unmeasured_s", "useful_s"):
                acc[(r["node"], r["owner"], r["kind"])][k] += r[k]
    out = [{"node": n, "owner": o, "kind": k, **{f: round(v, 1) for f, v in a.items()},
            "idle_pct": round(100 * a["idle_s"] / (a["busy_s"] + a["idle_s"]), 1) if a["busy_s"] + a["idle_s"] else None}
           for (n, o, k), a in acc.items() if a["idle_s"] > 0]
    return sorted(out, key=lambda w: (-w["idle_s"], w["node"], w["owner"], w["kind"] or ""))[:top]


def parse(text: str) -> list[dict]:
    """A record's rows (`ROW`), skipping any other line: a partial first line from `tail -c`, or a torn last one."""
    out = []
    for ln in text.splitlines():
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("schema") == ROW:
            out.append(d)
    return out


def hourly_command(hours: int) -> str:
    return f"tail -c {(hours + 3) * 256 * 1024} {RECORD}"


def day_command(day: str) -> str:
    """The day's rows: `grep -F` for the hour prefix as `json.dumps` writes it; grep's exit 1 (no line) is no rows, not an error."""
    return f"grep -F '\"hour\": \"{day}T' {RECORD} || test $? -eq 1"


def _day(s: str) -> str:
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a UTC day (YYYY-MM-DD): {s!r}") from None


def _positive(s: str) -> int:
    if not s.isdigit() or int(s) < 1:
        raise argparse.ArgumentTypeError(f"not a positive integer: {s!r}")
    return int(s)


def add_parser(sub: Any) -> None:
    """`cluster usage`'s subcommands, on the cli's subparsers."""
    u = sub.add_parser("usage", help="GPU-hours held, busy and useful per node and hour, from the console's record on vy-nebius-1")
    us = u.add_subparsers(dest="ucmd", required=True)
    h = us.add_parser("hourly", help="per node and UTC hour: busy and useful shares against the targets, held idle, unmeasured")
    h.add_argument("--node", choices=(*NODES, "all"), default="all")
    h.add_argument("--hours", type=_positive, default=24, help="the last H complete UTC hours")
    w = us.add_parser("wasters", help="the holdings with the most held-but-idle GPU-hours on one UTC day")
    w.add_argument("--day", type=_day, required=True)
    w.add_argument("--top", type=_positive, default=3)
    for p in (h, w):
        p.add_argument("--json", action="store_true")
        p.add_argument("--file", help=f"read this record instead of vy-nebius-1's {RECORD}")


def _h(s: float | None) -> str:
    return "-" if s is None else f"{s / 3600:.2f}"


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def _cols(cells: list[tuple]) -> list[str]:
    w = [max(len(c[i]) for c in cells) for i in range(len(cells[0]))]
    return ["  ".join(c[i].ljust(w[i]) for i in range(len(c))).rstrip() for c in cells]


def render_hourly(rep: dict) -> str:
    out = [f"GPU usage per UTC hour, the last {rep['hours']} h ({rep['source']})",
           f"targets: busy >= {TARGET_BUSY_PCT:g}% of measured GPU time, useful >= {TARGET_USEFUL_PCT:g}% of busy; "
           f"useful today: {USEFUL_TODAY}"]
    cells = [("HOUR", "NODE", "MIN READ", "GPU-H", "BUSY %", "USEFUL %", "HELD IDLE GPU-H", "UNMEASURED GPU-H", "MOST HELD IDLE")]
    for r in rep["rows"]:
        w = r["holdings"][0] if r["holdings"] and r["holdings"][0]["idle_s"] > 0 else None
        cells.append((r["hour"], r["node"], str(r["minutes_read"]), _h(r["gpu_s"]), _pct(r["busy_pct"]), _pct(r["useful_pct"]),
                      _h(r["held_idle_s"]), _h(r["unmeasured_s"]), f"{w['owner']}/{w['kind']} {_h(w['idle_s'])}" if w else "-"))
    out += _cols(cells) if len(cells) > 1 else ["no hours recorded"]
    out += [f"finding: {f}" for f in rep["findings"]]
    return "\n".join(out) + "\n"


def render_wasters(rep: dict) -> str:
    out = [f"held but idle GPU-hours on {rep['day']} (UTC), the top {rep['top']} holdings ({rep['source']})",
           "hours covered: " + ", ".join(f"{n} {k} of 24" for n, k in rep["hours_covered"].items())]
    cells = [("NODE", "OWNER", "KIND", "HELD IDLE GPU-H", "HELD GPU-H", "IDLE % OF MEASURED", "UNMEASURED GPU-H")]
    cells += [(w["node"], w["owner"], w["kind"] or "-", _h(w["idle_s"]), _h(w["held_s"]), _pct(w["idle_pct"]),
               _h(w["unmeasured_s"])) for w in rep["wasters"]]
    out += _cols(cells) if len(cells) > 1 else ["no holding held a GPU idle that day"]
    out += [f"finding: {f}" for f in rep["findings"]]
    return "\n".join(out) + "\n"


clock = time.time


def main(a: argparse.Namespace, runner: Callable[[str], str] | None = None) -> int:
    """`cluster usage hourly|wasters`: 0 with the report, 1 when the record can't be read or has no rows, an hour of the
    window (the day) that ended over STALE_S ago isn't recorded, or (hourly) a node's newest recorded hour ended that long ago
    or an hour was read for fewer than WHOLE_MIN minutes; 2 on bad arguments (argparse's). Wasters report the hours each
    node's rows cover. Reads only: one `tail` or `grep` of RECORD on vy-nebius-1 over SSH, or `--file`."""
    from . import status

    now = clock()
    if a.file:
        src = a.file
        cmd = None
    else:
        src = f"{nebius1.NODE}:{RECORD}"
        cmd = hourly_command(a.hours) if a.ucmd == "hourly" else day_command(a.day)
        runner = runner or status.ssh(status.HOSTS[nebius1.NODE])
    findings: list[str] = []
    try:
        rows = parse(Path(a.file).read_text() if a.file else runner(cmd))
    except (OSError, status.ReadError, subprocess.TimeoutExpired) as e:
        rows = []
        findings.append(f"could not read {src}: {type(e).__name__}: {e}")
    if a.ucmd == "hourly":
        nodes = list(NODES.values()) if a.node == "all" else [NODES[a.node]]
        first = hour_t(hour(now)) - 3600 * a.hours
        mine = [r for r in rows if r["node"] in nodes]
        hours = node_hours(r for r in mine if hour_t(r["hour"]) >= first)
        due = [hour(first + 3600 * k) for k in range(a.hours) if first + 3600 * (k + 1) + STALE_S <= now]
        for n in nodes if not findings else ():
            newest = max((hour_t(r["hour"]) for r in mine if r["node"] == n), default=None)
            if newest is None:
                findings.append(f"{n}: no hours in {src}")
                continue
            if newest + 3600 < now - STALE_S:
                findings.append(f"{n}: its newest recorded hour is {hour(newest)}, over {STALE_S // 3600} h before now: "
                                "is the console recording (verity-console.timer on vy-nebius-1)?")
            have = {h["hour"] for h in hours if h["node"] == n}
            if missing := [h for h in due if h not in have]:
                findings.append(f"{n}: {len(missing)} of the last {a.hours} hours ended over {STALE_S // 3600} h ago and aren't "
                                f"recorded, from {missing[0]}")
            findings += [f"{n} {h['hour']}: {h['minutes_read']} of its 60 minutes read, so it has no busy or useful share"
                         for h in sorted(hours, key=lambda h: h["hour"]) if h["node"] == n and h["minutes_read"] < WHOLE_MIN]
        rep = {"schema": HOURLY, "t": int(now), "source": src, "hours": a.hours, "nodes": nodes, "useful_rule": USEFUL_TODAY,
               "targets": {"busy_pct": TARGET_BUSY_PCT, "useful_pct": TARGET_USEFUL_PCT}, "rows": hours, "findings": findings}
        render = render_hourly
    else:
        day = [r for r in rows if r["hour"].startswith(f"{a.day}T")]
        covered = {n: sorted({r["hour"] for r in day if r["node"] == n}) for n in NODES.values()}
        d0 = hour_t(f"{a.day}T00:00Z")
        due = [hour(d0 + 3600 * k) for k in range(24) if d0 + 3600 * (k + 1) + STALE_S <= now]
        if not findings and not day:
            findings.append(f"no hours of {a.day} in {src}")
        for n, have in covered.items() if not findings else ():
            if missing := [h for h in due if h not in have]:
                findings.append(f"{n}: {len(missing)} of {a.day}'s hours ended over {STALE_S // 3600} h ago and aren't "
                                f"recorded, from {missing[0]}: the wasters cover {len(have)} of its 24 hours")
        rep = {"schema": WASTERS, "t": int(now), "source": src, "day": a.day, "top": a.top, "useful_rule": USEFUL_TODAY,
               "hours_covered": {n: len(h) for n, h in covered.items()}, "wasters": wasters(day, a.top), "findings": findings}
        render = render_wasters
    print(json.dumps(rep, indent=1, sort_keys=True) if a.json else render(rep), end="\n" if a.json else "")
    return 1 if findings else 0


__all__ = ["ROW", "HOURLY", "WASTERS", "RECORD", "ALL", "NO_POD", "NO_LEASE", "POOL_UNLEASED", "USEFUL_TODAY",
           "TARGET_BUSY_PCT", "TARGET_USEFUL_PCT", "Lease", "hour", "hourly", "held_idle", "record", "node_hours",
           "wasters", "parse", "add_parser", "main"]
