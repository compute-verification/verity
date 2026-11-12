"""`research queue stranded`: the jobs waiting in one line while a place that could run them stands free in another, and the
watch `stranded` (research/monitors.py) that rings when one has waited so for STRANDED_S.

    research queue stranded [--json] [--moves] [--pool P] [--min MINUTES]

A job is put in one line when it is launched (a machine's pool, a check line) and nothing looks again. Each CPU machine of the
registry (`auto.spill`, none named here) is read at once, one `python3 -` a machine (`program`): `auto.PROBE`'s own functions,
cut at its `out = {}` line, with this tree's slot.py and lean_slot.py inlined as `auto.payload` inlines them, so the watch and
the placer count memory, floors and Lean slots with one code. Its ending reads every pool the machine has (`locks/slots` is pool
`check`, `locks/NAME/slots` pool NAME), each in POOL_S of its own, the Lean slots, the disk floor, the memory available, the
names in /etc/vy/peers, and of each waiter's request.json the `auto` key alone (why `research run --on auto` put it there).
Every `cmd` key is dropped on the machine before the document leaves it. A machine or a pool that can't be read is listed with
why (`unread`), never dropped, so the counts beside it read as a floor.

A place (`analyse`) is a slot the launcher's own rule would start the job in now, on a machine where the job has no line:

  - the check pool as check.py's `placement` decides: an idle slot slot.py doesn't block, more of them than the machine's own
    waiters ahead, and a Lean `check` slot free there too. A priority waiter (a candidate's shard) may take an only=trains slot
    of mem= TRAIN_GB or its ticket's ask if more; a lane's own check takes no only=trains slot. A held check slot that shares a
    core closes it; lane runs on its cores do not, since they run niced and yield (`lane_cores` says how many they hold);
  - any other pool as `auto._pool` decides for the ask on the job's ticket, against the memory not reserved (`free_gb`), with
    each slot that shares a core with any held slot on the machine first marked blocked;
  - two idle slots that overlap are one place, and one place takes one job: places are dealt oldest first (priority first in
    the check pool), and a job left with none is in line but not stranded.

A slot a window closes, a line stalled on memory (`auto.Stalled`), a wait for a Lean slot and a pool shut by a disk floor are
printed apart and never counted as stranded. `moves` says, for each stranded job, the lines it should be in; nothing here
launches, moves or cancels a job. Exit 0 with no stranded wait, 1 with one."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from . import auto, remote

SCHEMA = "research/stranded/v1"
#: the pool slot.py's default slots file (`<root>/locks/slots`) is, beside the lane pools under `<root>/locks/NAME/slots`
CHECK = "check"
#: how long a job waits beside a place before the watch rings
STRANDED_S = 600
#: a line stalled this long rings infra (`stalled|POOL`), a pool shut this long on a machine that holds a run (`shut|...`)
STALLED_S = SHUT_S = 1800
#: a wait for a Lean slot this long rings infra (`lean|MACHINE`)
LEAN_WAIT_MIN = 10
#: each pool's read on the machine, and the whole machine's, ssh included
POOL_S, READ_S = 20, 60
#: check.py's TRAIN_GB: the least mem= of a slot a candidate's shard takes
TRAIN_GB = 200
#: where auto.PROBE's functions end and its own ending begins
MARK = "\n\nout = {}\n"
PEERS = "/etc/vy/peers"
SLOT_PY, LEAN_SLOT_PY = auto.SLOT_PY, auto.LEAN_SLOT_PY
#: the tree this module runs from: its slot.py and lean_slot.py are the ones sent
TREE = Path(__file__).resolve().parents[4]
#: how many stranded jobs of a lane pool its incident names
OLDEST = 5
ENDING = r'''

import glob, re, signal, time


class Late(BaseException):
    pass


def _late(*_):
    raise Late()


def limited(f, s):
    signal.signal(signal.SIGALRM, _late)
    signal.alarm(s)
    t = time.monotonic()
    try:
        d = f()
    except Late:
        d = {"error": f"not read in {s} s"}
    except Exception as e:
        d = {"error": f"{type(e).__name__}: {e}"}
    finally:
        signal.alarm(0)
    if isinstance(d, dict):
        d["read_s"] = round(time.monotonic() - t, 1)
    return d


def scrub(x):
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if k != "cmd"}
    return [scrub(v) for v in x] if isinstance(x, list) else x


def peers():
    try:
        lines = open(PEERS).read().splitlines()
    except OSError:
        return None
    return sorted({f[0] for f in (ln.split("#")[0].split() for ln in lines) if len(f) >= 2})


def available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / 2 ** 20, 1)


def placed(run):
    if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", run):
        return None
    try:
        with open(f"{REQUESTS}/{run}/request.json") as f:
            said = json.load(f).get("auto")
    except (OSError, ValueError, AttributeError):
        return None
    return said if isinstance(said, dict) else None


def with_placed(d):
    for w in d.get("waiters") or [] if isinstance(d, dict) else []:
        if isinstance(w, dict) and (said := placed(w.get("run"))) is not None:
            w["auto"] = said
    return d


files = {"check": LOCKS + "/slots"} if os.path.exists(LOCKS + "/slots") else {}
for f in sorted(glob.glob(LOCKS + "/*/slots")):
    files[f.split("/")[-2]] = f
out = {"pools": {p: with_placed(limited(lambda p=p, f=f: pool_doc(p, f), POOL_S)) for p, f in files.items()}}
out["lean"] = limited(lean_slots, POOL_S)
out["floor"] = disk_short()
out["no_start"] = no_start()
out["available_gb"] = available_gb()
out["peers"] = peers()
print(json.dumps(scrub(out)))
'''


def program(tree: Path | None = None, root: str = remote.DEFAULT_ROOT, peers: str = PEERS, disk: str = auto.DISK_PATH) -> bytes:
    """What each machine runs as `python3 -`: auto.PROBE's functions with ENDING, and ``tree``'s (TREE's) slot.py and lean_slot.py."""
    tree = TREE if tree is None else tree
    defs, mark, _ = auto.PROBE.partition(MARK)
    if not mark:
        raise RuntimeError(f"auto.PROBE has no `{MARK.strip()}` line, where stranded.program cuts it from its own ending: cut "
                           "it where auto.PROBE's functions end and change MARK in research/stranded.py to that line")
    src = {}
    for key, rel in (("SLOT_SRC", SLOT_PY), ("LEAN_SRC", LEAN_SLOT_PY)):
        try:
            src[key] = (Path(tree) / rel).read_text()
        except OSError:
            raise RuntimeError(f"`{rel}` is not in the reader's tree ({tree}), which sends it to every machine: on the control "
                               f"pod, add `{SLOT_PY}` and `{LEAN_SLOT_PY}` to the deploy's paths (the `git archive` line of "
                               "infra/control/README.md) and deploy again") from None
    head = {**src, "CORES_SRC": "", "JOB_CPUS": 1, "SHARE": None, "CLAIMS": "", "CPU_ROOT": "", "HOLD": auto.HOLD_FILE,
            "NOW": None, "POOLS": {}, "DISK": disk, "MIN_INODES": auto.MIN_FREE_INODES, "MIN_GB": auto.MIN_FREE_GB,
            "VIA_SLOT": remote.RUNS_VIA_SLOT_FILE, "LEAN_SLOTS": f"{root}/locks/lean-slots", "NULL": auto.NULL_DEVICE,
            "GIT": auto.GIT, "NO_START": auto.NO_START, "POOL_S": POOL_S, "LOCKS": f"{root}/locks",
            "REQUESTS": f"{root}/requests", "PEERS": peers}
    return ("".join(f"{k} = {v!r}\n" for k, v in head.items()) + defs + ENDING).encode()


def machines() -> list[str]:
    """Every CPU machine of the registry, by number (`auto.spill`)."""
    return [m for m, _ in auto.spill()]


def _ssh(argv: list[str], data: bytes) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=data, capture_output=True, timeout=READ_S)


def read(names: list[str], *, tree: Path | None = None, reach: Callable[[str], list[str]] | None = None,
         run: Callable[[list[str], bytes], Any] | None = None) -> dict[str, dict | str]:
    """Each machine's document (`program`), read at once as `auto.read` reads its probe, or why it couldn't be."""
    reach, run, data = reach or auto._reach, run or _ssh, program(tree)

    def one(name: str) -> dict | str:
        try:
            r = run(reach(name), data)
        except subprocess.TimeoutExpired:
            return f"no answer in {READ_S} s"
        except Exception as e:  # noqa: BLE001  (not listed, not this project's: listed unread)
            return str(e) or type(e).__name__
        if r.returncode:
            err = (r.stderr or b"").decode(errors="replace").strip().splitlines()
            if r.returncode == 255 and err and err[-1].startswith("ssh: connect to host "):
                return f"{auto.NO_SSH} ({err[-1].rsplit(': ', 1)[-1]})"
            return f"exit {r.returncode}: {err[-1] if err else 'no output'}"
        try:
            doc = json.loads(r.stdout)
        except ValueError:
            return "its answer is not JSON"
        return doc if isinstance(doc, dict) else "its answer is not a JSON object"

    with ThreadPoolExecutor(max_workers=max(1, len(names))) as ex:
        return dict(zip(names, ex.map(one, names)))


# ---- a place ------------------------------------------------------------------------------------------------------------------

def _cores(spec: Any) -> set[int]:
    out: set[int] = set()
    for part in str(spec or "").split(","):
        lo, _, hi = part.partition("-")
        if lo.strip().isdigit():
            out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def _held(docs: dict[str, dict]) -> list[tuple[str, str, set[int]]]:
    """(pool, slot, cores) of every held slot of a machine's pools."""
    return [(p, s["name"], _cores(s.get("cpus"))) for p, d in docs.items() for s in d.get("slots") or [] if s.get("holder")]


def _idle(s: dict) -> bool:
    return s.get("holder") is None and not s.get("blocked")


def _disjoint(slots: list[dict]) -> list[dict]:
    """Two idle slots that overlap are one place: ``slots`` without each one that shares a core with one kept before it."""
    kept: list[dict] = []
    for s in slots:
        if not any(_cores(k.get("cpus")) & _cores(s.get("cpus")) for k in kept):
            kept.append(s)
    return kept


def _working(pools: dict[str, dict]) -> dict[str, dict]:
    """A machine's pool documents as places are dealt from: a copy, each idle slot that a held slot closes marked blocked with
    why (a lane pool's by any held slot on the machine, a check slot's by a held check slot), a lane pool's overlapping idle
    slots made one, and each check slot's `lane_cores`, its cores that lane runs hold. The check pool's overlapping idle slots
    are made one among those that fit a job (`_check_free`), since a lane's check takes no only=trains slot."""
    held = _held(pools)
    out = {}
    for p, d in pools.items():
        w = {**d, "slots": [dict(s) for s in d.get("slots") or []], "waiters": [dict(x) for x in d.get("waiters") or []]}
        for s in w["slots"]:
            mine = _cores(s.get("cpus"))
            if p == CHECK:
                s["lane_cores"] = len(mine & set().union(*(c for q, _, c in held if q != CHECK)))
            if not _idle(s):
                continue
            by = next(((q, n) for q, n, c in held if c & mine and (p != CHECK or q == CHECK) and not (q == p and n == s["name"])),
                      None)
            if by:
                s["blocked"] = f"shares cores with {by[0]}'s slot {by[1]}"
        if p != CHECK:
            kept = _disjoint([s for s in w["slots"] if _idle(s) and not s.get("trains")])
            for s in w["slots"]:
                if _idle(s) and not s.get("trains") and s not in kept:
                    o = next(k for k in kept if _cores(k.get("cpus")) & _cores(s.get("cpus")))
                    s["blocked"] = f"shares cores with {p}'s idle slot {o['name']}"
        out[p] = w
    return out


def _gib(gb: float) -> int:
    return -(-int(gb * 1e9) // 2 ** 30) if gb and gb > 0 else 0


def _check_free(check: dict, priority: bool, ask: float) -> list[dict]:
    """The check slots a check from elsewhere could take now, as check.py's `placement` counts them: the fitting idle slots
    after the machine's own waiters ahead have taken theirs."""
    least = max(TRAIN_GB, _gib(ask)) if priority else None
    mine = [s for s in check.get("slots") or [] if priority or not s.get("trains")]
    if least:
        mine = [s for s in mine if s.get("mem") is None or s["mem"] >= least]
    ahead = len([w for w in check.get("waiters") or [] if w.get("priority") or not priority])
    return _disjoint([s for s in mine if _idle(s)])[ahead:]


def _lean_room(lean: Any) -> float:
    """How many more checks can take a Lean `check` slot on the machine now (placement's): none when they can't be read."""
    if not isinstance(lean, dict) or "error" in lean:
        return 0
    if lean.get("pools") is None:
        return float("inf")
    if (pool := next((p for p in ("check", "any") if p in lean["pools"]), None)) is None:
        return 0
    p = lean["pools"][pool]
    return len([s for s in p.get("slots") or [] if s.get("holder") is None and not s.get("past_n")]) - len(p.get("waiters") or [])


def _lane_free(doc: dict, ask: float) -> tuple[int, dict | None]:
    """The places a lane pool has for a job asking ``ask`` GB, as `auto._pool` counts them, and the first idle slot of them."""
    try:
        _, n, _, _ = auto._pool(doc, 1, ask)
    except (ValueError, KeyError, TypeError, AttributeError):
        return 0, None
    return n, next((s for s in doc["slots"] if not s.get("trains") and _idle(s)), None) if n else None


def _reason(why: str) -> str:
    if why.startswith("window "):
        return "window"
    if auto.MEMORY_BLOCK in why:
        return "memory"
    if why.startswith("shares cores"):
        return "cores"
    if " inodes free" in why or re.search(r" GB free, under \d+$", why):
        return "disk"
    return "other"


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%MZ")


def _end(start: str, minutes: int) -> str | None:
    try:
        return f"{datetime.fromisoformat(start.replace('Z', '+00:00')) + timedelta(minutes=minutes):%Y-%m-%dT%H:%MZ}"
    except ValueError:
        return None


def _job(machine: str, pool: str, w: dict) -> tuple[str, str]:
    """A waiter's job and its home: a twin's pair and the machine its pair names (or this one), else its run and this machine."""
    if twin := w.get("twin"):
        pair, _, home = str(twin).partition("@")
        return pair, home or machine
    return str(w.get("run") or f"{machine}/{pool}/pid {w.get('pid')}"), machine


def analyse(docs: dict[str, dict | str], now: float, cpu_machines: list[str] | None = None, min_minutes: float = 0) -> dict:
    """The stranded document (`research/stranded/v1`) of the machines' documents ``docs`` (`read`) at ``now``."""
    cpu_machines = list(docs) if cpu_machines is None else cpu_machines
    out: dict[str, Any] = {"schema": SCHEMA, "read_at": _iso(now), "waits": [], "pools": {}, "stalled": [], "windows": [],
                           "shut": [], "peers": [], "lean_waits": [], "unread": []}
    work: dict[str, dict[str, dict]] = {}
    shut: set[tuple[str, str]] = set()
    jobs: dict[tuple[str, str], dict] = {}
    for m, d in docs.items():
        if not isinstance(d, dict):
            out["unread"].append({"machine": m, "pool": None, "why": str(d)})
            continue
        pools = {}
        for p, pd in sorted((d.get("pools") or {}).items()):
            if not isinstance(pd, dict) or "error" in pd:
                out["unread"].append({"machine": m, "pool": p, "why": pd.get("error") if isinstance(pd, dict) else str(pd)})
            elif pd.get("slots"):
                pools[p] = pd
        closed = f"{auto.NO_START}: {d['no_start']}" if d.get("no_start") else f"disk: {d['floor']}" if d.get("floor") else None
        for p in pools:
            if closed and p != CHECK:
                shut.add((m, p))
                out["shut"].append({"machine": m, "pool": p, "why": closed})
        work[m] = _working(pools)
        for p, pd in pools.items():
            for s in pd["slots"]:
                if str(s.get("blocked") or "").startswith("window "):
                    out["windows"] += [{"machine": m, "pool": p, "slot": s["name"], "start": w.get("start"),
                                        "end": _end(str(w.get("start")), int(w.get("minutes") or 0)), "note": w.get("note") or ""}
                                       for w in s.get("upcoming") or [] if (w.get("starts_in_min") or 0) <= s.get("lead", 30)]
            for w in pd.get("waiters") or []:
                job, home = _job(m, p, w)
                j = jobs.setdefault((p, job), {"job": job, "pool": p, "runs": [], "lines": [], "by": w.get("by"),
                                                "priority": bool(w.get("priority")), "mem_gb": w.get("mem_gb"),
                                                "twin": str(w.get("twin") or "").partition("@")[0] or None, "home": home,
                                                "placed": {"by": "auto", "said": w["auto"].get("said")} if isinstance(
                                                    w.get("auto"), dict) else {"by": "its lane", "said": None}})
                if w.get("run") and w["run"] not in j["runs"]:
                    j["runs"].append(w["run"])
                j["lines"].append({"machine": m, "since": w.get("since"), "minutes": w.get("minutes") or 0})
            head = max((w.get("minutes") or 0 for w in pd.get("waiters") or []), default=0)
            if p != CHECK and (m, p) not in shut and head >= auto.STALL_MIN:
                try:
                    auto._pool(pd, 1, auto.BACKFILL_MAX_GB + 1)
                except auto.Stalled:
                    free = pd.get("free_gb")
                    avail = d.get("available_gb")
                    out["stalled"].append({"machine": m, "pool": p, "in_line": len(pd.get("waiters") or []), "head_minutes": head,
                                           "available_gb": avail, "free_gb": free,
                                           "unreached_gb": round(avail - free, 1) if isinstance(avail, (int, float)) and
                                           isinstance(free, (int, float)) else None})
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass
        lean = d.get("lean")
        for p, lp in sorted(((lean or {}).get("pools") or {}).items()) if isinstance(lean, dict) and "error" not in lean else ():
            out["lean_waits"] += [{"machine": m, "pool": p, "job": w.get("run") or f"pid {w.get('pid')}",
                                   "minutes": w.get("minutes") or 0} for w in lp.get("waiters") or []]
        if isinstance(lean, dict) and "error" in lean:
            out["unread"].append({"machine": m, "pool": "lean", "why": lean["error"]})
        names = d.get("peers")
        others = [x for x in cpu_machines if x != m]
        if names is None or set(others) - set(names):
            out["peers"].append({"machine": m, "file": names is not None,
                                 "missing": [x for x in others if names is None or x not in names]})
    peers = {m: d.get("peers") for m, d in docs.items() if isinstance(d, dict)}

    def places(j: dict) -> list[dict]:
        mine, found = {ln["machine"] for ln in j["lines"]}, []
        for m, pools in work.items():
            if m in mine or j["pool"] not in pools or (m, j["pool"]) in shut:
                continue
            doc, ask = pools[j["pool"]], float(j["mem_gb"] or 0)
            if j["pool"] == CHECK:
                free = _check_free(doc, j["priority"], ask)
                n, first = int(min(len(free), _lean_room(docs[m].get("lean")))), (free[0] if free else None)
            else:
                n, first = _lane_free(doc, ask)
            if n > 0 and first is not None:
                found.append({"machine": m, "slot": first["name"], "cores": first.get("cpus"), "mem": first.get("mem"),
                              "lane_cores": first.get("lane_cores", 0), "count": n, "reaches_home": j["home"] in (peers.get(m) or ())})
        return sorted(found, key=lambda f: -f["count"])

    def deal(j: dict, at: dict) -> None:
        doc = work[at["machine"]][j["pool"]]
        if j["pool"] != CHECK:
            doc["waiters"].append({"pid": 0, "minutes": 0, "mem_gb": j["mem_gb"], "dealt": j["job"]})
            return
        s = next(s for s in doc["slots"] if s["name"] == at["slot"])
        s["holder"] = {"pid": 0, "dealt": j["job"]}
        for o in doc["slots"]:
            if _idle(o) and _cores(o.get("cpus")) & _cores(s.get("cpus")):
                o["blocked"] = f"shares cores with check's slot {s['name']}"
        lean = docs[at["machine"]].get("lean")
        if isinstance(lean, dict) and lean.get("pools"):
            pool = next((p for p in ("check", "any") if p in lean["pools"]), None)
            if pool:
                lean["pools"][pool].setdefault("waiters", []).append({"pid": 0, "dealt": j["job"]})

    summary = {p: {"places": 0, "blocked": {}} for ms in work.values() for p in ms}
    for m, pools in work.items():
        for p, doc in pools.items():
            for s in doc["slots"]:
                if s.get("holder") is None and s.get("blocked"):
                    r = _reason(str(s["blocked"]))
                    summary[p]["blocked"][r] = summary[p]["blocked"].get(r, 0) + 1
            if (m, p) in shut:
                continue
            if p == CHECK:
                summary[p]["places"] += int(min(len(_check_free(doc, True, 0)), max(_lean_room(docs[m].get("lean")), 0)))
            else:
                summary[p]["places"] += _lane_free(doc, 0)[0]
    for j in jobs.values():
        j["minutes"] = max(ln["minutes"] for ln in j["lines"])
    order = sorted(jobs.values(), key=lambda j: (j["pool"], not (j["pool"] == CHECK and j["priority"]), -j["minutes"], j["job"]))
    docs = json.loads(json.dumps(docs))                                     # deal from a copy: the Lean waiters grow
    for j in order:
        j["places"] = places(j) if j["minutes"] >= min_minutes else []
        j["stranded"] = bool(j["places"])
        if j["stranded"]:
            deal(j, j["places"][0])
        j["move"] = _move(j)
        out["waits"].append({k: j[k] for k in ("job", "pool", "runs", "lines", "by", "priority", "mem_gb", "twin", "placed",
                                               "minutes", "stranded", "places", "move")})
    out["waits"].sort(key=lambda w: (not w["stranded"], -w["minutes"], w["pool"], w["job"]))
    for p, s in sorted(summary.items()):
        line = [w for w in out["waits"] if w["pool"] == p]
        idle = {}
        for m, pools in work.items():
            for q, doc in pools.items():
                if q != p and (n := sum(1 for x in doc["slots"] if _idle(x) and not x.get("lane_cores"))):
                    idle[q] = idle.get(q, 0) + n
        out["pools"][p] = {"in_line": len(line), "stranded": sum(w["stranded"] for w in line), "places": s["places"],
                           "blocked": dict(sorted(s["blocked"].items())), "idle_in_other_pools": dict(sorted(idle.items())),
                           "unread": sorted(u["machine"] for u in out["unread"] if u["pool"] in (p, None))}
    return out


def _move(j: dict) -> dict:
    """Where a stranded job should wait: its own line, then the machines with a place for it now, most places first, three
    lines at most, never one that can't claim on the job's home (its /etc/vy/peers). `how`: `late twin` for a waiter that
    carries a pair (slot.py's claim lets one of them run), `at launch` for one that carries none (only its next launch can be
    placed better), `none` when nothing should move."""
    if not j.get("stranded"):
        return {"how": "none", "to": [], "why_not": "it has no place elsewhere now"}
    own = sorted({ln["machine"] for ln in j["lines"]})
    reach = [p["machine"] for p in j["places"] if p["reaches_home"]]
    if not reach:
        return {"how": "none", "to": own, "why_not": f"no machine with a place names {j['home']} in its {PEERS}, so a twin there "
                                                      "can't claim on its home"}
    return {"how": "late twin" if j.get("twin") else "at launch", "to": (own + reach)[:max(3, len(own))], "why_not": None}


def moves(doc: dict) -> list[dict]:
    """Each stranded wait's move (`_move`): {job, pool, how, to, why_not}."""
    return [{"job": w["job"], "pool": w["pool"], **w["move"]} for w in doc["waits"] if w["stranded"]]


def only(doc: dict, pool: str) -> dict:
    """``doc`` with only what concerns ``pool`` (the peers are no pool's and stay)."""
    keep = {k: [x for x in doc[k] if x.get("pool") in (pool, None)] for k in ("waits", "stalled", "windows", "shut", "unread")}
    return {**doc, **keep, "pools": {p: v for p, v in doc["pools"].items() if p == pool},
            "lean_waits": doc["lean_waits"]}


# ---- what it prints -----------------------------------------------------------------------------------------------------------

def _gb(v: Any) -> str:
    return f"{v:g} GB" if isinstance(v, (int, float)) else "? GB"


def text(doc: dict, with_moves: bool = False) -> str:
    """The document as `research queue stranded` prints it."""
    stranded = [w for w in doc["waits"] if w["stranded"]]
    out = [f"read {doc['read_at']}: {len(stranded)} stranded of {len(doc['waits'])} in line"
           + (f"; {len(doc['unread'])} not read, so every count is a floor" if doc["unread"] else "")]
    for w in stranded:
        lines = ", ".join(f"{ln['machine']}'s for {ln['minutes']} min" for ln in w["lines"])
        out += ["", f"{w['pool']}  {w['job']}" + (f" (twin pair, runs {', '.join(w['runs'])})" if w["twin"] else "")
                + f": in {lines}; by {w['by'] or 'no one named'}" + ("; priority" if w["priority"] else "")
                + (f"; asks {_gb(w['mem_gb'])}" if w["mem_gb"] else ""),
                "  placed by its lane" if w["placed"]["by"] != "auto" else f"  placed by --on auto: {str(w['placed']['said'])[:300]}"]
        out += [f"  could run now on {p['machine']}: slot {p['slot']}, cores {p['cores']}, mem {_gb(p['mem'])}"
                + (f", {p['count']} places" if p["count"] > 1 else "")
                + (f"; lane runs hold {p['lane_cores']} of its cores" if p["lane_cores"] else "") for p in w["places"]]
        if with_moves:
            mv = w["move"]
            out.append(f"  move: {mv['how']}" + (f" to {', '.join(mv['to'])}" if mv["how"] != "none" else "")
                       + (f" ({mv['why_not']})" if mv["why_not"] else ""))
    out += ["", "pools:"]
    for p, s in doc["pools"].items():
        blocked = ", ".join(f"{r} {n}" for r, n in s["blocked"].items()) or "none"
        idle = ", ".join(f"{q} {n}" for q, n in s["idle_in_other_pools"].items()) or "none"
        out.append(f"  {p}: {s['in_line']} in line, {s['stranded']} stranded, {s['places']} place{'' if s['places'] == 1 else 's'}; idle slots blocked: {blocked}; "
                   f"idle in other pools: {idle}" + (f"; unread on {', '.join(s['unread'])}" if s["unread"] else ""))
    sections = (
        ("stalled lines (a waiter beside an idle slot, for more memory than is left unreserved)", "stalled",
         lambda x: f"{x['machine']} {x['pool']}: {x['in_line']} in line, head {x['head_minutes']} min; available "
                   f"{_gb(x['available_gb'])}, not reserved {_gb(x['free_gb'])}, reserved and not reached {_gb(x['unreached_gb'])}"),
        ("windows", "windows", lambda x: f"{x['machine']}: {x['start']} to {x['end']}" + (f": {x['note']}" if x["note"] else "")
                                         + f"; closes {x['slots']}"),
        ("waits for a Lean slot", "lean_waits", lambda x: f"{x['machine']} {x['pool']}: {x['job']}, {x['minutes']} min"),
        ("shut", "shut", lambda x: f"{x['machine']} {x['pool']}: {x['why']}"),
        ("not read", "unread", lambda x: f"{x['machine']}" + (f" {x['pool']}" if x["pool"] else "") + f": {x['why']}"),
        (f"{PEERS} short", "peers", lambda x: f"{x['machine']}: " + ("names none of " if not x["file"] else "lacks ")
                                              + (", ".join(x["missing"]) if x["file"] else f"{len(x['missing'])} machines (no file)"))
    )
    windows: dict[tuple, list[str]] = {}
    for x in doc["windows"]:
        windows.setdefault((x["machine"], x["start"], x["end"], x["note"]), []).append(f"{x['pool']} {x['slot']}")
    rows = {**doc, "windows": [{"machine": m, "start": s, "end": e, "note": n, "slots": ", ".join(v)}
                               for (m, s, e, n), v in windows.items()]}
    for title, key, fmt in sections:
        if rows[key]:
            out += ["", f"{title}:"] + [f"  {fmt(x)}" for x in rows[key]]
    return "\n".join(out)


# ---- the watch ----------------------------------------------------------------------------------------------------------------

def findings(ctx: Any) -> list:
    """The monitor `stranded`'s findings from one read of every CPU machine through ``ctx.run``. A check-pool job stranded for
    STRANDED_S rings ci, one incident a job (the runner times it: `for_s`); any other pool rings infra once a pool while one of
    its jobs has been stranded STRANDED_S (``ctx.memo`` keeps when each was first seen so). Then infra's: a pool not read or
    shut by a floor (`shut`: at once on a machine that holds no run while that pool's jobs wait elsewhere, else after SHUT_S),
    a line stalled on memory for STALLED_S, a wait for a Lean slot of LEAN_WAIT_MIN, a machine whose /etc/vy/peers is missing
    or short. Raises only when no machine could be read at all."""
    from .monitors import Finding

    names = machines()
    if not names:
        raise RuntimeError(f"the registry names no CPU machine of project {auto.AUTO_PROJECT}: give the relay the registry "
                           "(RESEARCH_MACHINES_D in infra/control/watchers.toml's [env])")
    docs = read(names, run=lambda argv, data: ctx.run(argv, input=data, capture_output=True, timeout=READ_S))
    if not any(isinstance(d, dict) for d in docs.values()):
        raise RuntimeError("no CPU machine could be read: " + "; ".join(f"{m}: {d}" for m, d in docs.items())[:400])
    doc = analyse(docs, ctx.now, names)
    out: list = []
    first = ctx.memo.setdefault("first", {})
    now_stranded = {f"{w['pool']}|{w['job']}": w for w in doc["waits"] if w["stranded"] and w["pool"] != CHECK}
    for k in [k for k in first if k not in now_stranded]:
        del first[k]
    for k in now_stranded:
        first.setdefault(k, ctx.now)
    for w in doc["waits"]:
        if w["stranded"] and w["pool"] == CHECK:
            where = ", ".join(f"{p['machine']} (slot {p['slot']})" for p in w["places"])
            lines = ", ".join(f"{ln['machine']}'s" for ln in w["lines"])
            out.append(Finding(f"check|{w['job']}", f"*Check job `{w['job']}` waits in {lines} check line while a check slot "
                                                   f"stands free on {where}* ({w['minutes']} min in line; by {w['by'] or 'no one named'}"
                                                   + (", priority" if w["priority"] else "") + f"). Its move: {w['move']['how']}"
                                                   + (f" to {', '.join(w['move']['to'])}" if w["move"]["how"] != "none" else
                                                      f" ({w['move']['why_not']})"),
                               owners=("ci",), for_s=STRANDED_S))
    for pool in sorted({w["pool"] for w in now_stranded.values()}):
        late = sorted((w for k, w in now_stranded.items() if w["pool"] == pool and ctx.now - first[k] >= STRANDED_S),
                      key=lambda w: -w["minutes"])
        if not late:
            continue
        machines_ = sorted({ln["machine"] for w in late for ln in w["lines"]})
        places = sorted({p["machine"] for w in late for p in w["places"]})
        out.append(Finding(f"pool|{pool}", f"*{len(late)} job(s) of pool `{pool}` have waited {STRANDED_S // 60} min or more in "
                                           f"{', '.join(machines_)} while places stand free on {', '.join(places)}*. Oldest: "
                                           + "; ".join(f"`{w['job']}` {w['minutes']} min" for w in late[:OLDEST]),
                           owners=("infra",), for_s=0))
    held = {m for m, d in docs.items() if isinstance(d, dict) and any(
        s.get("holder") for pd in (d.get("pools") or {}).values() if isinstance(pd, dict) for s in pd.get("slots") or [])}
    waiting = {w["pool"] for w in doc["waits"]}
    for x in doc["shut"] + [u for u in doc["unread"] if u["pool"] != "lean"]:
        pool = x["pool"] or "*"
        idle = isinstance(docs.get(x["machine"]), dict) and x["machine"] not in held
        now_ = idle and (pool in waiting or pool == "*")
        out.append(Finding(f"shut|{x['machine']}|{pool}", f"*{x['machine']}'s " + (f"pool `{pool}`" if x["pool"] else "pools")
                                                          + f" can't take a job*: {x['why']}", owners=("infra",),
                           for_s=0 if now_ else SHUT_S))
    for pool in sorted({x["pool"] for x in doc["stalled"]}):
        lines = [x for x in doc["stalled"] if x["pool"] == pool]
        out.append(Finding(f"stalled|{pool}", f"*Pool `{pool}` is stalled on memory on {len(lines)} machine(s)*: "
                                              + "; ".join(f"{x['machine']} head {x['head_minutes']} min, reserved and not reached "
                                                          f"{_gb(x['unreached_gb'])}" for x in lines),
                           owners=("infra",), for_s=STALLED_S))
    for m in sorted({x["machine"] for x in doc["lean_waits"] if x["minutes"] >= LEAN_WAIT_MIN}):
        ws = [x for x in doc["lean_waits"] if x["machine"] == m and x["minutes"] >= LEAN_WAIT_MIN]
        out.append(Finding(f"lean|{m}", f"*{len(ws)} run(s) on {m} have waited {LEAN_WAIT_MIN} min or more for a Lean slot*, "
                                        "holding their CPU slot: " + "; ".join(f"{x['job']} ({x['pool']}, {x['minutes']} min)"
                                                                                for x in ws[:OLDEST]),
                           owners=("infra",), for_s=0))
    for x in doc["peers"]:
        out.append(Finding(f"peers|{x['machine']}", f"*{x['machine']}'s {PEERS} " + (
            f"lacks {', '.join(x['missing'])}*" if x["file"] else "is missing*") + ": a check's twin there can't claim on "
                                                                                  "those machines, so it gives its slot back",
                           owners=("infra",), for_s=0))
    return out


def main(argv: list[str], now: Callable[[], float] = time.time) -> int:
    ap = argparse.ArgumentParser(prog="research queue stranded", description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help=f"the document ({SCHEMA}) instead of text")
    ap.add_argument("--moves", action="store_true", help="print each stranded job's move: the lines it should be in")
    ap.add_argument("--pool", metavar="P", help="only pool P")
    ap.add_argument("--min", type=float, default=0, metavar="MINUTES", help="only jobs that have waited this long (default 0)")
    a = ap.parse_args(argv)
    names = machines()
    if not names:
        raise SystemExit("research queue stranded: the registry names no CPU machine of project "
                         f"{auto.AUTO_PROJECT} (`research pods list`)")
    try:
        docs = read(names)
    except RuntimeError as e:
        raise SystemExit(f"research queue stranded: {e}") from None
    doc = analyse(docs, now(), names, a.min)
    if a.pool:
        doc = only(doc, a.pool)
    print(json.dumps(doc, indent=1) if a.json else text(doc, a.moves))
    return 1 if any(w["stranded"] for w in doc["waits"]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
