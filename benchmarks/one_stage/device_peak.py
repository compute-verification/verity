"""A process's device memory peak from an nsys trace taken with ``--cuda-memory-usage=true``: the largest sum of live device
allocations (cudaMalloc and the pools', as nsys records them), with what was live at that moment by size, in MiB.

    python3 benchmarks/one_stage/device_peak.py TRACE.sqlite [--top N] [--json OUT]

Unlike sampling ``nvidia-smi``, the trace sees an allocation however briefly it lives; the CUDA context's own memory (a few
hundred MiB, which ``nvidia-smi`` counts) is not an allocation and is not in it."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter

MIB = 1 << 20


def enum(con: sqlite3.Connection, table: str) -> dict[int, str]:
    """``table``'s ids and names, or nothing when this nsys version's export has no such table."""
    if con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone() is None:
        return {}
    return {i: n for i, n in con.execute(f"SELECT id, name FROM {table}")}


def peak(path: str, top: int = 8) -> dict:
    con = sqlite3.connect(path)
    kinds, ops = enum(con, "ENUM_CUDA_MEM_KIND"), enum(con, "ENUM_CUDA_DEV_MEM_EVENT_OPER")
    rows = con.execute("SELECT start, globalPid, address, bytes, memKind, memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS "
                       "ORDER BY start, memoryOperationType DESC").fetchall()
    device = {k for k, n in kinds.items() if "DEVICE" in n.upper() and "STATIC" not in n.upper()} or {2}
    alloc = {k for k, n in ops.items() if n.upper() == "ALLOCATION" or n.upper().endswith("_ALLOCATION")} or {0}
    out: dict[int, dict] = {}
    for start, pid, addr, size, kind, op in rows:
        if kind not in device:
            continue
        p = out.setdefault(pid, {"live": {}, "now": 0, "peak": 0, "at": 0, "at_peak": {}})
        if op in alloc:
            p["live"][addr] = size
            p["now"] += size
            if p["now"] > p["peak"]:
                p["peak"], p["at"], p["at_peak"] = p["now"], start, dict(p["live"])
        else:
            p["now"] -= p["live"].pop(addr, 0)
    res = []
    for pid, p in out.items():
        sizes = Counter(round(s / MIB, 1) for s in p["at_peak"].values())
        res.append({"pid": pid >> 24 & 0xFFFFFF, "peak_mib": round(p["peak"] / MIB, 1), "live_at_peak": len(p["at_peak"]),
                    "largest_mib": sorted(sizes.items(), key=lambda kv: -kv[0])[:top]})
    return {"trace": path, "processes": sorted(res, key=lambda r: -r["peak_mib"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json")
    a = ap.parse_args()
    r = peak(a.sqlite, a.top)
    for p in r["processes"]:
        big = ", ".join(f"{m:,.1f}" + (f" x{n}" if n > 1 else "") for m, n in p["largest_mib"])
        print(f"pid {p['pid']}: device peak {p['peak_mib']:,.1f} MiB allocated, {p['live_at_peak']} allocations live; largest {big}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
