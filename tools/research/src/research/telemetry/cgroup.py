"""cgroup discovery and readings, v2 and v1 (brief §3 environment, §4 sampling).

Discover fields from the files present; never assume a kernel version.  `max` / 2^63-ish values are UNLIMITED (recorded as null +
kind "unlimited"), an unreadable file is UNKNOWN (null + kind "unknown" + reason).  Ancestors are walked upward inside the visible mount;
what the cgroup namespace hides is recorded as a visibility limitation, not silently treated as unlimited.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

V1_UNLIMITED_MIN = 2 ** 62   # v1 limit_in_bytes reads as PAGE_COUNTER_MAX-ish when unlimited


def _read(path: str | Path) -> tuple[str | None, str | None]:
    try:
        with open(path, "rb") as f:
            return f.read(65536).decode("ascii", "replace").strip(), None
    except FileNotFoundError:
        return None, "absent"
    except PermissionError:
        return None, "permission denied"
    except OSError as e:
        return None, f"read error: {e.strerror or e}"


def _int_or_kind(raw: str | None, reason: str | None) -> tuple[int | None, str]:
    """-> (value_bytes, kind) with kind in finite | unlimited | unknown."""
    if raw is None:
        return None, "unknown"
    if raw == "max":
        return None, "unlimited"
    try:
        v = int(raw.split()[0])
    except ValueError:
        return None, "unknown"
    if v >= V1_UNLIMITED_MIN or v < 0:
        return None, "unlimited"
    return v, "finite"


def _kv_file(raw: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    if not raw:
        return out
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


def self_cgroup_path(pid: int | str = "self") -> tuple[str | None, int | None, str | None]:
    """-> (v2 path, v1 memory path, reason).  /proc/<pid>/cgroup lines: `0::/path` (v2) or `N:memory:/path` (v1)."""
    raw, reason = _read(f"/proc/{pid}/cgroup")
    if raw is None:
        return None, None, reason
    v2 = v1 = None
    for line in raw.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and parts[1] == "":
            v2 = parts[2]
        elif "memory" in parts[1].split(","):
            v1 = parts[2]
    return v2, v1, None


def mounts() -> dict[str, str | None]:
    """{'v2': mountpoint or None, 'v1_memory': mountpoint or None, 'v2_ro': bool}"""
    out: dict[str, Any] = {"v2": None, "v1_memory": None, "v2_ro": None}
    raw, _ = _read("/proc/self/mounts")
    for line in (raw or "").splitlines():
        p = line.split()
        if len(p) < 4:
            continue
        if p[2] == "cgroup2" and out["v2"] is None:
            out["v2"] = p[1]
            out["v2_ro"] = "ro" in p[3].split(",")
        elif p[2] == "cgroup" and "memory" in p[3].split(",") and out["v1_memory"] is None:
            out["v1_memory"] = p[1]
    return out


class Cgroup:
    """The memory-controlling cgroup of a pid, as visible from this mount namespace."""

    def __init__(self, pid: int | str = "self"):
        m = mounts()
        v2p, v1p, reason = self_cgroup_path(pid)
        self.reason = reason
        self.version: int | None = None
        self.dir: Path | None = None
        self.path: str | None = None
        self.mount_ro = m.get("v2_ro")
        self.v1_dir: Path | None = None
        if m["v2"] and v2p is not None:
            d = Path(m["v2"]) / v2p.lstrip("/")
            if (d / "memory.max").exists() or (d / "memory.current").exists():
                self.version, self.dir, self.path = 2, d, v2p
        if m["v1_memory"] and v1p is not None:
            d1 = Path(m["v1_memory"]) / v1p.lstrip("/")
            if (d1 / "memory.limit_in_bytes").exists():
                self.v1_dir = d1
                if self.version is None:
                    self.version, self.dir, self.path = 1, d1, v1p
        # a v2 root reading unlimited beside an enforcing v1 controller (cp-dev2/cp-moe pattern): both are recorded; `effective` picks the finite one

    # ---- limits -------------------------------------------------------------------------------------------------------------------
    def limits(self) -> dict[str, Any]:
        out: dict[str, Any] = {"version": self.version, "path": self.path, "mount_read_only": self.mount_ro, "reason": self.reason,
                               "ancestors": [], "ancestors_visible": None, "ancestors_reason": None}
        if self.dir is None:
            out.update(memory_max_bytes=None, memory_max_kind="unknown", memory_max_reason=self.reason or "no memory cgroup found")
            return out
        if self.version == 2:
            raw, why = _read(self.dir / "memory.max")
            v, kind = _int_or_kind(raw, why)
            out.update(memory_max_bytes=v, memory_max_kind=kind, memory_max_source=str(self.dir / "memory.max"), memory_max_reason=why)
            raw, why = _read(self.dir / "memory.high")
            v, kind = _int_or_kind(raw, why)
            out.update(memory_high_bytes=v, memory_high_kind=kind)
            raw, why = _read(self.dir / "memory.swap.max")
            v, kind = _int_or_kind(raw, why)
            out.update(swap_max_bytes=v, swap_max_kind=kind)
            raw, _ = _read(self.dir / "cpu.max")
            out["cpu_max"] = raw
            raw, _ = _read(self.dir / "pids.max")
            out["pids_max"] = raw
            raw, _ = _read(self.dir / "cpuset.cpus.effective")
            out["cpuset_cpus_effective"] = raw
            raw, _ = _read(self.dir / "cgroup.controllers")
            out["controllers"] = (raw or "").split()
        else:
            raw, why = _read(self.dir / "memory.limit_in_bytes")
            v, kind = _int_or_kind(raw, why)
            out.update(memory_max_bytes=v, memory_max_kind=kind, memory_max_source=str(self.dir / "memory.limit_in_bytes"), memory_max_reason=why)
            raw, why = _read(self.dir / "memory.memsw.limit_in_bytes")
            v, kind = _int_or_kind(raw, why)
            out.update(memsw_limit_bytes=v, memsw_limit_kind=kind)
        if self.v1_dir is not None and self.version == 2:
            raw, why = _read(self.v1_dir / "memory.limit_in_bytes")
            v, kind = _int_or_kind(raw, why)
            out["v1_memory_limit_bytes"], out["v1_memory_limit_kind"] = v, kind
        # ancestors inside the visible mount
        anc = []
        finite = [x for x in (out.get("memory_max_bytes"), out.get("v1_memory_limit_bytes")) if x is not None]
        try:
            root = Path(mounts()["v2"] if self.version == 2 else mounts()["v1_memory"] or "/")
            d = self.dir.parent
            while d != root.parent and str(d).startswith(str(root)) and d != d.parent:
                f = d / ("memory.max" if self.version == 2 else "memory.limit_in_bytes")
                raw, why = _read(f)
                v, kind = _int_or_kind(raw, why)
                anc.append({"path": str(d), "memory_max_bytes": v, "kind": kind})
                if v is not None:
                    finite.append(v)
                if d == root:
                    break
                d = d.parent
        except OSError as e:
            out["ancestors_reason"] = f"walk error: {e}"
        out["ancestors"] = anc
        if self.path in ("/", None):
            out["ancestors_visible"] = False
            out["ancestors_reason"] = "cgroup namespace root: ancestors outside the namespace are not readable from here"
        else:
            out["ancestors_visible"] = True
        out["effective_memory_max_bytes"] = min(finite) if finite else None
        out["effective_memory_max_kind"] = ("visible_minimum" if finite else ("unlimited_as_far_as_visible" if out.get("memory_max_kind") == "unlimited" else "unknown"))
        return out

    # ---- readings (sampler) -------------------------------------------------------------------------------------------------------
    def sample(self) -> dict[str, Any]:
        out: dict[str, Any] = {"version": self.version, "path": self.path}
        if self.dir is None:
            out["reason"] = self.reason or "no memory cgroup found"
            return out
        if self.version == 2:
            for name, key in (("memory.current", "memory_current_bytes"), ("memory.peak", "memory_peak_bytes"), ("memory.swap.current", "swap_current_bytes"),
                              ("pids.current", "pids_current")):
                raw, why = _read(self.dir / name)
                if raw is None:
                    out[key], out[key + "_reason"] = None, why
                else:
                    try:
                        out[key] = int(raw)
                    except ValueError:
                        out[key], out[key + "_reason"] = None, f"unparsable {raw[:20]!r}"
            for name, key in (("memory.events", "memory_events"), ("memory.events.local", "memory_events_local"), ("cpu.stat", "cpu_stat")):
                raw, why = _read(self.dir / name)
                out[key] = _kv_file(raw) if raw is not None else None
                if raw is None:
                    out[key + "_reason"] = why
            raw, why = _read(self.dir / "memory.stat")
            if raw is None:
                out["memory_stat"], out["memory_stat_reason"] = None, why
            else:
                kv = _kv_file(raw)
                want = ("anon", "file", "file_dirty", "file_writeback", "shmem", "kernel", "kernel_stack", "slab", "sock", "pagetables",
                        "file_mapped", "anon_thp", "unevictable", "pgfault", "pgmajfault")
                out["memory_stat"] = {k: kv[k] for k in want if k in kv}
                out["memory_stat_fields_present"] = sorted(kv)[:64]
            raw, why = _read(self.dir / "io.stat")
            if raw is None:
                out["io_stat"], out["io_stat_reason"] = None, why
            else:
                dev: dict[str, dict[str, int]] = {}
                for line in raw.splitlines():
                    p = line.split()
                    if not p:
                        continue
                    dev[p[0]] = {}
                    for kv_ in p[1:]:
                        k, _, v = kv_.partition("=")
                        try:
                            dev[p[0]][k] = int(v)
                        except ValueError:
                            pass
                out["io_stat"] = dev
        else:
            for name, key in (("memory.usage_in_bytes", "memory_current_bytes"), ("memory.max_usage_in_bytes", "memory_peak_bytes"),
                              ("memory.failcnt", "memory_failcnt")):
                raw, why = _read(self.dir / name)
                out[key] = int(raw) if raw is not None and raw.isdigit() else None
                if raw is None:
                    out[key + "_reason"] = why
            raw, why = _read(self.dir / "memory.oom_control")
            oc = _kv_file(raw) if raw else {}
            # oom_kill appeared in memory.oom_control in Linux 4.13: absent is None (no counter), never 0 (no kill)
            out["memory_events"] = {"oom_kill": oc.get("oom_kill"), "under_oom": oc.get("under_oom", 0)} if raw else None
            if raw and "oom_kill" not in oc:
                out["memory_events_reason"] = "memory.oom_control has no oom_kill field (kernel < 4.13): OOM kills are not counted"
            raw, why = _read(self.dir / "memory.stat")
            kv = _kv_file(raw)
            out["memory_stat"] = ({"anon": kv.get("total_rss"), "file": kv.get("total_cache"), "file_dirty": kv.get("total_dirty"),
                                   "file_writeback": kv.get("total_writeback"), "shmem": kv.get("total_shmem"), "kernel": None,
                                   "kernel_reason": "v1 has no kernel memory total in memory.stat"} if kv else None)
            out["memory_stat_fields_present"] = sorted(kv)[:64]
            cpuacct = self.dir.parent if False else None   # v1 cpuacct lives in another hierarchy; recorded absent
            out["cpu_stat"], out["cpu_stat_reason"] = None, "v1: cpuacct is a separate hierarchy (not sampled in v0.1)"
            out["io_stat"], out["io_stat_reason"] = None, "v1: blkio is a separate hierarchy (not sampled in v0.1)"
        if self.v1_dir is not None and self.version == 2:
            raw, _ = _read(self.v1_dir / "memory.usage_in_bytes")
            out["v1_memory_usage_bytes"] = int(raw) if raw and raw.isdigit() else None
            raw, _ = _read(self.v1_dir / "memory.oom_control")
            out["v1_memory_events"] = {"oom_kill": _kv_file(raw).get("oom_kill")} if raw else None
        return out


#: Where a machine gives each login session (an ssh login, and the run it starts) a memory cap of its own: the Nebius nodes'
#: 50-vy-job-memory.conf, which pods/nebius/deploy.toml installs.
SESSION_DROPINS = Path("/etc/systemd/system/session-.scope.d")


def session_cap(dropins: Path | None = None) -> str | None:
    """The MemoryMax= this machine's systemd gives every login session scope (the last one set, as systemd reads them), or None."""
    cap = None
    try:
        files = sorted((dropins or SESSION_DROPINS).glob("*.conf"))
    except OSError:
        return None
    for f in files:
        raw, _ = _read(f)
        for line in (raw or "").splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == "MemoryMax":
                cap = v.strip() or None
    return None if cap in (None, "infinity") else cap


def uncapped(limits: dict[str, Any], cap: str | None) -> str | None:
    """Why a run whose own cgroup reads `limits` (Cgroup.limits()) would go without the per-session cap `cap` (session_cap()), or None.
    When user-1001.slice is at its MemoryMax, logind can't create the login's session-N.scope and the login (and what it starts)
    stays in ssh.service: its memory.max is the service's, shared with every login that failed the same way, or none."""
    if cap is None:
        return None
    path = limits.get("path") or "?"
    if limits.get("memory_max_kind") != "finite":
        return (f"this machine caps each login session at MemoryMax={cap} ({SESSION_DROPINS}), but this run's cgroup {path} has memory.max "
                f"{limits.get('memory_max_kind')}, so its workload would run with no cap of its own. Launch it from an ssh login, or "
                f"with --queue and --mem-gb")
    if path.endswith(".service"):
        return (f"this machine caps each login session at MemoryMax={cap} ({SESSION_DROPINS}), but this run is in {path}, whose memory.max "
                f"is shared by everything that service runs: its login got no session scope (logind fails to create one while "
                f"user-1001.slice is at its MemoryMax; `journalctl -u systemd-logind` says why). Launch it again once the slice has room")
    return None


def scope_for(sampler_cg: Cgroup, workload_cg: Cgroup | None) -> tuple[str, str]:
    """(scope, reason).  job_cgroup only when the workload sits in a cgroup of its own that is NOT the sampler's; RunPod pods put every
    process in the namespace root -> shared_pod_cgroup."""
    if workload_cg is None or workload_cg.path is None:
        return "shared_pod_cgroup", "workload cgroup unknown; the pod cgroup is shared by every process on the pod"
    if workload_cg.path in ("/", None) or workload_cg.path == sampler_cg.path:
        return "shared_pod_cgroup", f"workload cgroup {workload_cg.path!r} is the pod's (shared with the sampler and every other job)"
    return "job_cgroup", f"workload has its own cgroup {workload_cg.path!r}"
