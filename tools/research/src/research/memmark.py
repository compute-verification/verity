"""A run's memory mark: what it asked, the limit that would stop it, the most it held, and whether it passed that limit.

A mark stops nothing and sets no limit anywhere. It is the `memory` object of the run's status.json, written with the
run's end (`of_run`, which `research run` hands the harness), and one line of `research status RUN` (`line`): the figure a
lane sizes `--mem-gb` from.

    ask      what the run asked, in bytes: the booking of the check slot it holds (its slot.json's `reserves_gib`, what
             slot.py reserved for it), else, in no slot or one that reserves nothing, its queue job's memory, else its
             `--mem-gb`, else none
    limit    the ask and HEADROOM_BYTES more; HEADROOM_BYTES for a run with no ask; or what its launch stated
             (`--mem-limit-gb`, and who stated it: `limit_by`). Never more than `bound`, the least memory.max over the
             cgroup the runner started in and its ancestors, as job.json recorded it
    held     memory.current less the page cache, `file - shmem` of memory.stat, in one of the sampler's cgroup records
             (resources.jsonl, about one a second): what is left once the kernel has reclaimed what it can. The kernel counts
             shared memory in `file`, and without swap it can't be reclaimed, so it stays in
    peak     the most a sample held from when the run took its check slot (every sample, for a run in no slot), with that
             sample's time and the number of samples it is the most of
    over     whether a sample held more than the limit, and the first such sample's time

The samples are of the cgroup the runner and its sampler run in. That is the workload's too, unless the run moved its
workload out: a check slot with `scope=1` (its slot.json's `scope`), `--job-user`'s unit, a GPU lease. Then no sample is
of the workload, and the mark holds no peak and says where the workload ran (`reason`). For a check in its own scope it
holds the scope's memory.peak as the check's spans recorded it (`scope_peak_bytes`, to 0.1 GB): that counts the page
cache, so it bounds what the check held from above and is not the held figure. A figure that can't be computed is null
with its reason, never zero. A slot.py that writes no `reserves_gib` took the slot before it wrote either key: its run's
mark is the reason alone (OLD_SLOT), since its ask and where its workload ran are both unknown.

A GB is 10^9 bytes; a slot's booking and a queue job's memory are GiB, 2^30 bytes. The mark holds bytes.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import checkview, scopemem
from .runs import SLOT_FILE, _read

GB = 10 ** 9
GIB = 2 ** 30
HEADROOM_BYTES = 128 * GB
#: the mark's key in status.json
KEY = "memory"
SAMPLES = "resources.jsonl"
#: what a pool run's `--mem-gb` reaches its slot as (tools/verity/check/slot.py's RUN_MEM_ENV)
RUN_MEM_ENV = "CHECK_RUN_MEM_GB"
FROM_ASK = f"ask+{HEADROOM_BYTES // GB}GB"
#: a held slot record without `reserves_gib`: neither the ask nor where the workload ran is known, so nothing is marked
OLD_SLOT = ("the slot.py that took the run's slot predates the mark: its record of the slot says neither what the slot reserved "
            "for the run nor whether the workload left the sampled cgroup (`research deploy install` places the current slot.py "
            "on a machine that keeps a copy of its own)")


def ask(request: dict[str, Any], slot: dict[str, Any]) -> tuple[int | None, str]:
    """What the run asked, in bytes, and where that is from: `slot`, `queue`, `--mem-gb` or `none`."""
    booked = slot.get("reserves_gib")
    if slot.get("state") == "held" and isinstance(booked, (int, float)) and booked > 0:
        return int(booked * GIB), "slot"
    job = (request.get("queue") or {}).get("job")
    if isinstance(job, dict):
        return int(job.get("memory_gib", 1)) * GIB, "queue"
    gb = float(request.get("mem_gb") or (request.get("env") or {}).get(RUN_MEM_ENV) or 0)
    return (int(gb * GB), "--mem-gb") if gb > 0 else (None, "none")


def bound(cgroup: dict[str, Any]) -> int | None:
    """The least memory.max the runner's cgroup started under, from job.json's `environment.cgroup`; None when none is finite."""
    for key in ("effective_memory_max_bytes", "memory_max_bytes"):
        if isinstance(v := cgroup.get(key), int) and v > 0:
            return v
    return None


def limit(ask_bytes: int | None, stated: dict[str, Any] | None, bound_bytes: int | None) -> dict[str, Any]:
    """`limit_bytes` and `limit_from`: FROM_ASK, `default` (no ask), `stated` (with `limit_by`), or `bound` where the machine's
    bound is under what the rule or the statement gives."""
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
    """What one cgroup record held: memory.current less `file - shmem`; None when the record lacks one of the three."""
    stat = cg.get("memory_stat") or {}
    cur, file, shmem = cg.get("memory_current_bytes"), stat.get("file"), stat.get("shmem")
    if not all(isinstance(v, int) for v in (cur, file, shmem)):
        return None
    return cur - (file - shmem)


def peak(records: Iterable[dict[str, Any]], limit_bytes: int, since: str | None = None) -> dict[str, Any]:
    """The most the cgroup records held, when, over how many, in which cgroup, and whether and when first one held more than
    `limit_bytes`. `since` (a UTC time to the second) leaves out the records before it."""
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
    """The run's mark. `request` is its launch request ({} for a local run), `slot` its slot.json ({} without one), `cgroup`
    job.json's `environment.cgroup`, `records` its cgroup records in order (None without a resources.jsonl), `stated` a
    launch's `{"gb", "by"}`, `elsewhere` where the workload ran when not in the runner's cgroup, and `scope_peak` the bytes a
    check's own scope peaked at."""
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
    """The most memory.peak a check in its own scope recorded in its spans (`mem_peak_gb`), in bytes; None without one."""
    gbs = [gb for s in checkview.spans(run_dir) if isinstance(gb := (s.get("attributes") or {}).get("mem_peak_gb"), (int, float))]
    return round(max(gbs) * GB) if gbs else None


def of_run(run_dir: Path, request: dict[str, Any], cgroup: dict[str, Any], *, stated: dict[str, Any] | None = None,
           elsewhere: str | None = None) -> dict[str, Any]:
    """`{KEY: mark}` for the run's terminal status, from its run dir. With `elsewhere` nothing in the run dir is read: a
    `--job-user` job can leave anything there, and no sample is of the workload. A mark that can't be made is its reason:
    this never raises, since it runs between a workload's exit and the record of it."""
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
    """The mark as `research status RUN` prints it; for a record without one, that the peak was not recorded."""
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
