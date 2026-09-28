"""Process tree, PSS, GPU and storage readings for the Level-0 sidecar (brief §4).  Fixed-size /proc reads, no recursive scans."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _read(path: str, n: int = 65536) -> tuple[str | None, str | None]:
    try:
        with open(path, "rb") as f:
            return f.read(n).decode("ascii", "replace"), None
    except FileNotFoundError:
        return None, "gone"
    except PermissionError:
        return None, "permission denied"
    except OSError as e:
        return None, f"read error: {e.strerror or e}"


def list_pids() -> list[int]:
    try:
        return sorted(int(x) for x in os.listdir("/proc") if x.isdigit())
    except OSError:
        return []


def stat_fields(pid: int) -> dict[str, Any] | None:
    raw, why = _read(f"/proc/{pid}/stat", 4096)
    if raw is None:
        return None
    try:
        lp, rp = raw.index("("), raw.rindex(")")
        comm = raw[lp + 1:rp]
        rest = raw[rp + 2:].split()
        return {"pid": pid, "comm": comm[:32], "state": rest[0], "ppid": int(rest[1]), "pgrp": int(rest[2]), "session": int(rest[3]),
                "utime_ticks": int(rest[11]), "stime_ticks": int(rest[12]), "num_threads": int(rest[17]), "starttime_ticks": int(rest[19]),
                "vsize_bytes": int(rest[20]), "rss_pages": int(rest[21])}
    except (ValueError, IndexError):
        return None


def status_fields(pid: int) -> dict[str, Any]:
    raw, why = _read(f"/proc/{pid}/status", 8192)
    out: dict[str, Any] = {}
    if raw is None:
        out["status_reason"] = why
        return out
    want = {"VmRSS": "rss_bytes", "RssAnon": "rss_anon_bytes", "RssFile": "rss_file_bytes", "RssShmem": "rss_shmem_bytes", "VmSwap": "swap_bytes",
            "VmHWM": "rss_hwm_bytes", "Threads": "threads", "voluntary_ctxt_switches": "ctxt_vol", "nonvoluntary_ctxt_switches": "ctxt_nonvol"}
    for line in raw.splitlines():
        k, _, v = line.partition(":")
        if k in want:
            v = v.strip()
            try:
                n = int(v.split()[0])
            except (ValueError, IndexError):
                continue
            out[want[k]] = n * 1024 if v.endswith("kB") else n
    return out


def io_fields(pid: int) -> dict[str, Any]:
    raw, why = _read(f"/proc/{pid}/io", 2048)
    if raw is None:
        return {"io": None, "io_reason": why}
    kv = {}
    for line in raw.splitlines():
        k, _, v = line.partition(":")
        try:
            kv[k.strip()] = int(v)
        except ValueError:
            pass
    return {"io": {"rchar": kv.get("rchar"), "wchar": kv.get("wchar"), "read_bytes": kv.get("read_bytes"), "write_bytes": kv.get("write_bytes"),
                   "cancelled_write_bytes": kv.get("cancelled_write_bytes")},
            "io_note": "wchar/write_bytes are process write accounting, not completed physical writeback"}


def oom_score_adj(pid: int) -> int | None:
    raw, _ = _read(f"/proc/{pid}/oom_score_adj", 64)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def oom_score(pid: int) -> int | None:
    """/proc/<pid>/oom_score: the kernel's badness (RSS+swap+pagetables scaled to the allowed memory, oom_score_adj folded in).  The OOM
    killer's victim is the process with the HIGHEST badness at kill time; a sampled value is evidence about the victim, not the kernel's
    decision itself (it may have moved between the sample and the kill)."""
    raw, _ = _read(f"/proc/{pid}/oom_score", 64)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def running(pid: int) -> bool:
    """`pid` exists and has not exited (a zombie has)."""
    st = stat_fields(pid)
    return st is not None and st["state"] not in ("Z", "X")


def live_group(pgid: int) -> list[dict[str, Any]]:
    """The members of process group `pgid` that have not exited."""
    return [st for pid in list_pids() if (st := stat_fields(pid)) and st["pgrp"] == pgid and st["state"] not in ("Z", "X")]


def descendants(root_pid: int, all_stats: dict[int, dict[str, Any]]) -> set[int]:
    kids: dict[int, list[int]] = {}
    for pid, st in all_stats.items():
        kids.setdefault(st["ppid"], []).append(pid)
    out = {root_pid}
    stack = [root_pid]
    while stack:
        p = stack.pop()
        for c in kids.get(p, []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def snapshot_tree(root_pid: int | None, pgid: int | None, max_pids: int, with_outsiders: bool = True) -> tuple[list[dict[str, Any]], bool, int, list[dict[str, Any]]]:
    """All processes selected by root (descendants incl. root), pgid, or every visible pid.  -> (rows, truncated, total_selected, outsiders).
    `outsiders` = every other visible process (same pod cgroup on RunPod) with pid/comm/starttime/anon+shmem -- what the shared-scope OOM
    attribution needs (who else was there, who else vanished, who was largest); bounded to max_pids by anon+shmem descending."""
    stats: dict[int, dict[str, Any]] = {}
    for pid in list_pids():
        st = stat_fields(pid)
        if st is not None:
            stats[pid] = st
    if root_pid is not None:
        sel = descendants(root_pid, stats) if root_pid in stats else set()
    elif pgid is not None:
        sel = {p for p, st in stats.items() if st["pgrp"] == pgid}
    else:
        sel = set(stats)
    sel.discard(os.getpid())   # the sampler itself is never part of the workload tree (it is a sibling/child of the runner)
    rows = []
    truncated = False
    for pid in sorted(sel):
        if len(rows) >= max_pids:
            truncated = True
            break
        st = stats.get(pid)
        if st is None:
            continue
        row = dict(st)
        row["cpu_s"] = (st["utime_ticks"] + st["stime_ticks"]) / CLK_TCK
        row["rss_bytes_stat"] = row.pop("rss_pages") * PAGE
        row.update(status_fields(pid))
        row.update(io_fields(pid))
        row["oom_score_adj"] = oom_score_adj(pid)
        row["oom_score"] = oom_score(pid)
        rows.append(row)
    outsiders: list[dict[str, Any]] = []
    if with_outsiders and (root_pid is not None or pgid is not None):
        me = os.getpid()
        for pid, st in stats.items():
            if pid in sel or pid == me:
                continue
            sf = status_fields(pid)
            outsiders.append({"pid": pid, "comm": st["comm"], "ppid": st["ppid"], "starttime_ticks": st["starttime_ticks"],
                              "anon_shmem_bytes": (sf.get("rss_anon_bytes") or 0) + (sf.get("rss_shmem_bytes") or 0), "rss_bytes": sf.get("rss_bytes"),
                              "oom_score": oom_score(pid), "oom_score_adj": oom_score_adj(pid)})
        outsiders.sort(key=lambda r: (-(r["oom_score"] or 0), -(r["anon_shmem_bytes"] or 0)))
        outsiders = outsiders[:max_pids]
    return rows, truncated, len(sel), outsiders


def pss(pid: int) -> dict[str, Any]:
    raw, why = _read(f"/proc/{pid}/smaps_rollup", 4096)
    if raw is None:
        return {"pid": pid, "pss_bytes": None, "reason": why}
    out: dict[str, Any] = {"pid": pid}
    want = {"Rss": "rss_bytes", "Pss": "pss_bytes", "Pss_Anon": "pss_anon_bytes", "Pss_File": "pss_file_bytes", "Pss_Shmem": "pss_shmem_bytes",
            "Shared_Clean": "shared_clean_bytes", "Shared_Dirty": "shared_dirty_bytes", "Private_Clean": "private_clean_bytes",
            "Private_Dirty": "private_dirty_bytes", "Anonymous": "anonymous_bytes", "Swap": "swap_bytes", "SwapPss": "swap_pss_bytes"}
    for line in raw.splitlines():
        k, _, v = line.partition(":")
        if k in want:
            try:
                out[want[k]] = int(v.split()[0]) * 1024
            except (ValueError, IndexError):
                pass
    if "pss_bytes" not in out:
        out["pss_bytes"], out["reason"] = None, "smaps_rollup without Pss field"
    return out


# ---- GPU ------------------------------------------------------------------------------------------------------------------------------

GPU_QUERY = "uuid,index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,ecc.errors.uncorrected.volatile.total,clocks_throttle_reasons.active"
APP_QUERY = "gpu_uuid,pid,used_gpu_memory"


def gpu_sample(timeout: float = 3.0) -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    out: dict[str, Any] = {"available": True, "devices": [], "apps": []}
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"available": True, "reason": f"nvidia-smi --query-gpu timed out after {timeout}s", "devices": None, "apps": None}
    if r.returncode != 0:
        return {"available": True, "reason": f"nvidia-smi rc={r.returncode}: {(r.stderr or '').strip()[:160]}", "devices": None, "apps": None}
    keys = ["uuid", "index", "memory_used_mib", "memory_total_mib", "gpu_util_pct", "memory_controller_util_pct", "temperature_c", "ecc_uncorrected_volatile", "throttle_reasons"]
    for line in r.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        d: dict[str, Any] = {}
        for k, v in zip(keys, p):
            if v in ("[N/A]", "[Not Supported]", "N/A", ""):
                d[k], d[k + "_reason"] = None, "not supported / N/A"
            else:
                try:
                    d[k] = int(v) if k not in ("uuid", "throttle_reasons") else v
                except ValueError:
                    d[k] = v
        for k in ("memory_used_mib", "memory_total_mib"):
            if isinstance(d.get(k), int):
                d[k.replace("_mib", "_bytes")] = d[k] * 1024 * 1024
        d["note"] = "memory_controller_util_pct is the memory-controller busy fraction (nvidia-smi utilization.memory), not copy throughput"
        out["devices"].append(d)
    try:
        r2 = subprocess.run(["nvidia-smi", f"--query-compute-apps={APP_QUERY}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout)
        if r2.returncode == 0:
            for line in r2.stdout.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 3:
                    try:
                        out["apps"].append({"gpu_uuid": p[0], "pid": int(p[1]), "used_gpu_memory_bytes": int(p[2]) * 1024 * 1024})
                    except ValueError:
                        out["apps"].append({"gpu_uuid": p[0], "pid": p[1], "used_gpu_memory_bytes": None, "reason": f"unparsable {p[2]!r}"})
        else:
            out["apps"], out["apps_reason"] = None, f"nvidia-smi --query-compute-apps rc={r2.returncode}"
    except subprocess.TimeoutExpired:
        out["apps"], out["apps_reason"] = None, f"--query-compute-apps timed out after {timeout}s"
    return out


# ---- storage ----------------------------------------------------------------------------------------------------------------------

def statvfs(path: str) -> dict[str, Any]:
    try:
        s = os.statvfs(path)
    except OSError as e:
        return {"path": path, "free_bytes": None, "reason": e.strerror}
    return {"path": path, "free_bytes": s.f_bavail * s.f_frsize, "total_bytes": s.f_blocks * s.f_frsize, "free_inodes": s.f_favail,
            "total_inodes": s.f_files, "inodes_note": None if s.f_files else "filesystem reports no inode table (network fs)"}


def diskstats() -> dict[str, Any] | None:
    raw, why = _read("/proc/diskstats", 65536)
    if raw is None:
        return None
    out = {}
    for line in raw.splitlines():
        p = line.split()
        if len(p) < 14:
            continue
        name = p[2]
        if name.startswith(("loop", "ram")):
            continue
        try:
            out[name] = {"reads": int(p[3]), "sectors_read": int(p[5]), "writes": int(p[7]), "sectors_written": int(p[9]), "io_ms": int(p[12])}
        except ValueError:
            pass
    return out
