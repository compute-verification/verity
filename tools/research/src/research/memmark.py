"""A run's memory mark: what it asked, the limit that would stop it, the most it held, and whether it passed that limit."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import checkview, scopemem
from .runs import SLOT_FILE, _read

GB = 10 ** 9
GIB = 2 ** 30
HEADROOM_BYTES = 128 * GB
KEY = "memory"
SAMPLES = "resources.jsonl"
RUN_MEM_ENV = "CHECK_RUN_MEM_GB"
FROM_ASK = f"ask+{HEADROOM_BYTES // GB}GB"
OLD_SLOT = ("the slot.py that took the run's slot predates the mark: its record of the slot says neither what the slot reserved "
            "for the run nor whether the workload left the sampled cgroup (`research deploy install` places the current slot.py "
            "on a machine that keeps a copy of its own)")


def ask(request: dict[str, Any], slot: dict[str, Any]) -> tuple[int | None, str]:
    booked = slot.get("reserves_gib")
    if slot.get("state") == "held" and isinstance(booked, (int, float)) and booked > 0:
        return int(booked * GIB), "slot"
    job = (request.get("queue") or {}).get("job")
    if isinstance(job, dict):
        return int(job.get("memory_gib", 1)) * GIB, "queue"
    gb = float(request.get("mem_gb") or (request.get("env") or {}).get(RUN_MEM_ENV) or 0)
    return (int(gb * GB), "--mem-gb") if gb > 0 else (None, "none")


def bound(cgroup: dict[str, Any]) -> int | None:
    for key in ("effective_memory_max_bytes", "memory_max_bytes"):
        if isinstance(v := cgroup.get(key), int) and v > 0:
            return v
    return None


def limit(ask_bytes: int | None, stated: dict[str, Any] | None, bound_bytes: int | None) -> dict[str, Any]:
    if stated:
        want, how = int(float(stated["gb"]) * GB), "stated"
    elif ask_bytes is None:
        want, how = HEADROOM_BYTES, "default"
    else:
        want, how = ask_bytes + HEADROOM_BYTES, FROM_ASK
    if bound_bytes is not None and want > bound_bytes:
        want, how = bound_bytes, "bound"
    return {"limit_bytes": want, "limit_from": how, **({"limit_by": stated.get("by")} if stated else {}), "bound_bytes": bound_bytes}


def held(cg: dict[str, Any]) -> int | None:
    """memory.current less `file - shmem`; None when the record lacks one of the three."""
    stat = cg.get("memory_stat") or {}
    cur, file, shmem = cg.get("memory_current_bytes"), stat.get("file"), stat.get("shmem")
    if not all(isinstance(v, int) for v in (cur, file, shmem)):
        return None
    return cur - (file - shmem)


def peak(records: Iterable[dict[str, Any]], limit_bytes: int, since: str | None = None) -> dict[str, Any]:
    top = top_utc = over_utc = path = None
    n = 0
    for rec in records:
        t = rec.get("t_utc")
        if since and str(t)[:19] < since[:19]:
            continue
        h = held(rec["cgroup"])
        if h is None:
            continue
        n += 1
        if top is None or h > top:
            top, top_utc, path = h, t, rec["cgroup"].get("path")
        if over_utc is None and h > limit_bytes:
            over_utc = t
    return {"peak_held_bytes": top, "peak_utc": top_utc, "samples": n, "over_limit": over_utc is not None if n else None,
            "over_limit_utc": over_utc, "cgroup": path}


def mark(request: dict[str, Any], slot: dict[str, Any], cgroup: dict[str, Any], records: Iterable[dict[str, Any]] | None, *,
         stated: dict[str, Any] | None = None, elsewhere: str | None = None, scope_peak: int | None = None) -> dict[str, Any]:
    if slot.get("state") == "held" and "reserves_gib" not in slot:
        return {"reason": OLD_SLOT}
    asked, ask_from = ask(request, slot)
    out = {"ask_bytes": asked, "ask_from": ask_from, **limit(asked, stated, bound(cgroup))}
    took = slot.get("took") if slot.get("state") == "held" else None
    if slot and slot.get("state") != "held":
        why = f"the run took no check slot (its slot record says {slot.get('state')})"
    elif slot.get("scope"):
        why = f"the workload ran in its own scope {slot['scope']}, and the samples are of the runner's cgroup"
    elif elsewhere:
        why = f"the workload ran in {elsewhere}, and the samples are of the runner's cgroup"
    elif records is None:
        why = f"the run has no {SAMPLES}: its sampler wrote no sample"
    else:
        got = peak(records, out["limit_bytes"], since=took)
        if got["samples"]:
            return {**out, **got, "reason": None}
        why = (f"no cgroup sample in {SAMPLES} holds memory.current and memory.stat's file and shmem"
               + (f" since the run took its slot at {took}" if took else ""))
    out.update(peak_held_bytes=None, peak_utc=None, samples=0, over_limit=None, over_limit_utc=None, cgroup=None, reason=why)
    if scope_peak is not None:
        out["scope_peak_bytes"] = scope_peak
        if scope_peak <= out["limit_bytes"]:
            out["over_limit"] = False
    return out


def _scope_peak(run_dir: Path) -> int | None:
    gbs = [gb for s in checkview.spans(run_dir) if isinstance(gb := (s.get("attributes") or {}).get("mem_peak_gb"), (int, float))]
    return round(max(gbs) * GB) if gbs else None


def of_run(run_dir: Path, request: dict[str, Any], cgroup: dict[str, Any], *, stated: dict[str, Any] | None = None,
           elsewhere: str | None = None) -> dict[str, Any]:
    """With ``elsewhere`` nothing in the run dir is read: a `--job-user` job can leave anything there."""
    try:
        if elsewhere:
            return {KEY: mark(request, {}, cgroup, None, stated=stated, elsewhere=elsewhere)}
        slot = _read(run_dir / SLOT_FILE) or {}
        scope_peak = _scope_peak(run_dir) if slot.get("scope") else None
        if not (run_dir / SAMPLES).is_file():
            return {KEY: mark(request, slot, cgroup, None, stated=stated, scope_peak=scope_peak)}
        with open(run_dir / SAMPLES, encoding="utf-8", errors="replace") as f:
            return {KEY: mark(request, slot, cgroup, scopemem.cgroup_records(f), stated=stated, scope_peak=scope_peak)}
    except Exception as e:  # noqa: BLE001
        return {KEY: {"reason": f"the mark was not computed: {type(e).__name__}: {e}"[:300]}}


def _gb(b: int) -> str:
    return f"{b / GB:.1f} GB"


def _when(utc: object) -> str:
    return f"{str(utc)[:19]}Z"


def line(m: object) -> str:
    if not isinstance(m, dict):
        return ("peak not recorded: the run's record holds no memory mark (its runner predates the mark, or never wrote the "
                "run's end)")
    if m.get("limit_bytes") is None:
        return f"peak not recorded: {m.get('reason')}"
    asked = f"asked {_gb(m['ask_bytes'])} ({m['ask_from']})" if m.get("ask_bytes") is not None else "asked none"
    how = {"stated": f"stated by {m.get('limit_by')}", "default": f"no ask: {_gb(HEADROOM_BYTES)}",
           "bound": "the machine's bound" + (f", under what {m['limit_by']} stated" if m.get("limit_by") else "")}
    lim = f"limit {_gb(m['limit_bytes'])} ({how.get(m.get('limit_from'), m.get('limit_from'))})"
    if m.get("peak_held_bytes") is None:
        top = f"peak held not recorded ({m.get('reason')})"
        if m.get("scope_peak_bytes") is not None:
            top += f"; its own scope's memory.peak, page cache included, {_gb(m['scope_peak_bytes'])}"
    else:
        top = (f"peak held {_gb(m['peak_held_bytes'])} at {_when(m.get('peak_utc'))} ({m.get('samples')} "
               f"sample{'' if m.get('samples') == 1 else 's'})")
    out = f"{top}; {asked}; {lim}"
    if m.get("over_limit"):
        return (f"{out}: would have been stopped at its memory limit of {_gb(m['limit_bytes'])} (asked "
                f"{_gb(m['ask_bytes']) if m.get('ask_bytes') is not None else 'none'}) at {_when(m.get('over_limit_utc'))}")
    if m.get("over_limit") is None and m.get("scope_peak_bytes") is not None:
        return f"{out}: that peak is over the limit, and whether what it held was is not recorded"
    return out
