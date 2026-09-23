"""job.json / status.json -- identity of the attempt and its state machine (brief §3).

Integer bytes, explicit units in field names (`_bytes`, `_s`), `null` + `<field>_reason` for anything unavailable, unlimited != unknown
(cgroup.py).  No environment dump: only the profile keys the caller names.  Command lines pass through `redact_argv`.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from . import COLLECTOR_VERSION, CLASSIFIER_VERSION, JOB_SCHEMA, SCHEMA, STATES, STATUS_SCHEMA
from .cgroup import Cgroup
from .events import stream_id, utc_now_ms

SECRET_KEY = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|[-_]KEY\b|APIKEY|API[-_]KEY)", re.I)
SECRET_VALUE = re.compile(r"^(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)$")


def redact_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """KEY=VALUE tokens with a secret-looking KEY, `--flag VALUE` / `--flag=VALUE` with a secret-looking flag, and bare secret-shaped
    values are replaced.  Returns (redacted argv, names of what was redacted)."""
    out: list[str] = []
    redacted: list[str] = []
    skip_next = False
    for i, tok in enumerate(argv):
        if skip_next:
            out.append("<redacted>")
            skip_next = False
            continue
        if "=" in tok and not tok.startswith("-"):
            k, _, v = tok.partition("=")
            if SECRET_KEY.search(k) or SECRET_VALUE.match(v):
                out.append(f"{k}=<redacted>")
                redacted.append(k)
                continue
        if tok.startswith("--") and "=" in tok:
            k, _, v = tok.partition("=")
            if SECRET_KEY.search(k) or SECRET_VALUE.match(v):
                out.append(f"{k}=<redacted>")
                redacted.append(k)
                continue
        if tok.startswith("--") and SECRET_KEY.search(tok) and i + 1 < len(argv):
            out.append(tok)
            redacted.append(tok)
            skip_next = True
            continue
        if SECRET_VALUE.match(tok):
            out.append("<redacted>")
            redacted.append(f"argv[{i}]")
            continue
        out.append(tok)
    return out, redacted


def sha256_file(path: str | Path, max_bytes: int = 256 * 1024 * 1024) -> tuple[str | None, str | None]:
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError as e:
        return None, f"stat failed: {e.strerror}"
    if size > max_bytes:
        return None, f"not hashed at launch ({size} bytes > {max_bytes})"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), None


def _run(cmd: list[str], timeout: float = 3.0) -> tuple[str | None, str | None]:
    if shutil.which(cmd[0]) is None:
        return None, f"{cmd[0]} not found"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"{cmd[0]} timed out after {timeout}s"
    except OSError as e:
        return None, f"{cmd[0]}: {e.strerror}"
    if r.returncode != 0:
        return None, f"{cmd[0]} rc={r.returncode}: {(r.stderr or r.stdout).strip()[:200]}"
    return r.stdout, None


def gpu_inventory(timeout: float = 3.0) -> dict[str, Any]:
    out, why = _run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"], timeout)
    if out is None:
        return {"available": False, "reason": why, "devices": []}
    devs = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            try:
                mem = int(p[3]) * 1024 * 1024
            except ValueError:
                mem = None
            devs.append({"index": p[0], "uuid": p[1], "name": p[2], "memory_total_bytes": mem, "driver": p[4] if len(p) > 4 else None})
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {"available": True, "reason": None, "devices": devs, "cuda_visible_devices": vis}


def environment() -> dict[str, Any]:
    cg = Cgroup()
    boot_id, _ = _read_text("/proc/sys/kernel/random/boot_id")
    mem_total = None
    raw, _ = _read_text("/proc/meminfo")
    if raw:
        for line in raw.splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) * 1024
                break
    gpu = gpu_inventory()
    limits = cg.limits()
    vis = []
    if limits.get("ancestors_visible") is False:
        vis.append("cgroup ancestors hidden by the cgroup namespace")
    if not os.access("/dev/kmsg", os.R_OK):
        vis.append("kernel log (dmesg) not readable: OOM victim identity unavailable from the kernel")
    return {"pod_id": os.environ.get("RUNPOD_POD_ID") or os.environ.get("POD_ID"), "hostname": socket.gethostname(), "boot_id": boot_id,
            "kernel": platform.release(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()),
            "cpu_count_note": "CPUs this process may run on (affinity/cpuset), not the host's", "cpu_count_host": os.cpu_count(),
            "host_memory_total_bytes": mem_total, "host_memory_note": "host total, not the job's bound; the bound is cgroup.effective_memory_max_bytes",
            "host_class": "gpu" if gpu.get("available") else "cpu", "gpu": gpu, "cgroup": limits, "limits_visibility": vis}


def _read_text(path: str) -> tuple[str | None, str | None]:
    try:
        return Path(path).read_text().strip(), None
    except OSError as e:
        return None, e.strerror


def build_job(*, project: str, task: str, job: str, attempt: int, lane: str, generation: int, tree_sha: str | None, release_tag: str | None,
              workload_digest: str | None, stage: str, inputs: list[dict[str, Any]], profile: dict[str, str], command: list[str],
              declared_outputs: list[str], timeout_s: float | None, reservation: dict[str, Any], policy_version: str | None,
              policy_sha256: str | None, instruments: list[str], cadence_s: float, pss_every_s: float, level: int = 0,
              source_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    env = environment()
    argv, redacted = redact_argv(command)
    unavailable = {}
    if not env["gpu"]["available"]:
        unavailable["gpu"] = env["gpu"]["reason"]
    return {"schema": JOB_SCHEMA, "written_utc": utc_now_ms(), "writer_stream": stream_id(),
            "ownership": {"project": project, "task": task, "job": job, "attempt": attempt, "lane": lane, "generation": generation},
            "computation": {"tree_sha": tree_sha, "tree_sha_reason": None if tree_sha else "not supplied", "release_tag": release_tag,
                            "workload_digest": workload_digest, "workload_digest_reason": None if workload_digest else "not supplied",
                            "stage": stage, "inputs": inputs, "profile": profile,
                            # [R18 infra, F-r17-tp-06] resolved_path / repo_root / git_sha per imported package + sys.path[0], as the
                            # runner's precheck recorded them (research-infra/source-identity/v1); None = "not supplied" (pre-r18-int-1 runner)
                            "source_identity": source_identity, "source_identity_reason": None if source_identity else "not supplied"},
            "environment": env,
            "execution": {"command": argv, "command_redacted": redacted, "declared_outputs": declared_outputs, "timeout_s": timeout_s,
                          "reservation": reservation, "policy_version": policy_version, "policy_sha256": policy_sha256},
            "telemetry": {"schema": SCHEMA, "collector_version": COLLECTOR_VERSION, "classifier_version": CLASSIFIER_VERSION,
                          "instruments": [i for i in instruments if i not in unavailable], "instruments_unavailable": unavailable,
                          "cadence_s": cadence_s, "pss_every_s": pss_every_s, "level": level}}


def write_json_atomic(path: str | Path, doc: dict[str, Any]) -> None:
    p = Path(path)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    os.replace(tmp, p)


def input_ref(spec: str) -> dict[str, Any]:
    """`name=path` or `path`; hashes files <= 256 MiB, else null + reason."""
    name, _, path = spec.partition("=") if "=" in spec else (Path(spec).name, "=", spec)
    ref: dict[str, Any] = {"name": name, "path": path}
    if path.startswith("art:"):
        ref["artifact"] = path
        ref["sha256"], ref["sha256_reason"] = None, "artifact reference (a store id, not a file)"
    elif Path(path).is_file():
        st = Path(path).stat()
        ref["sha256"], ref["sha256_reason"] = sha256_file(path)
        ref["bytes"] = st.st_size
        # identity at launch even when the hash is deferred (rev d6 / F-r17-19 shape): size + mtime_ns + inode + device
        ref["stat"] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ino": st.st_ino, "dev": st.st_dev}
        if ref["sha256"] is None:
            ref["sha256_deferred_to"] = "preserve.verify"   # the preserving step hashes it (hash_inputs below), so every preserved bundle ends fully hashed
    elif Path(path).is_dir():
        ref["sha256"], ref["sha256_reason"] = None, "directory (hash its manifest, not the tree)"
    else:
        ref["sha256"], ref["sha256_reason"] = None, "absent at launch"
    return ref


def hash_inputs(attempt_dir: str | Path, max_bytes: int | None = None) -> list[dict[str, Any]]:
    """Complete deferred input hashes in job.json (`tele init --hash-inputs` / preserve.verify).  Re-stats first: a changed size/mtime/inode
    since launch is recorded as `stat_changed_since_launch` and the hash then describes the CURRENT file, not the launched one."""
    d = Path(attempt_dir)
    p = d / "job.json"
    doc = json.loads(p.read_text())
    done = []
    for ref in (doc.get("computation") or {}).get("inputs") or []:
        if ref.get("sha256") is not None or not Path(ref.get("path", "")).is_file():
            continue
        st = Path(ref["path"]).stat()
        launch = ref.get("stat") or {}
        changed = bool(launch) and (launch.get("size") != st.st_size or launch.get("mtime_ns") != st.st_mtime_ns or launch.get("ino") != st.st_ino)
        sha, reason = sha256_file(ref["path"], max_bytes if max_bytes is not None else 1 << 40)
        ref["sha256"], ref["sha256_reason"] = sha, reason
        ref["sha256_hashed_at"] = "preserve.verify"
        ref["stat_changed_since_launch"] = changed
        ref.pop("sha256_deferred_to", None)
        done.append(ref)
    if done:
        write_json_atomic(p, doc)
    return done


# ---- status.json ------------------------------------------------------------------------------------------------------------------------

def read_status(attempt_dir: str | Path) -> dict[str, Any]:
    p = Path(attempt_dir) / "status.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except ValueError:
            return {"schema": STATUS_SCHEMA, "corrupt_previous": True, "transitions": []}
    return {"schema": STATUS_SCHEMA, "transitions": []}


def set_status(attempt_dir: str | Path, state: str, **kv: Any) -> dict[str, Any]:
    if state not in STATES:
        raise ValueError(f"state {state!r} not in {STATES}")
    st = read_status(attempt_dir)
    st["schema"] = STATUS_SCHEMA
    st["state"] = state
    st["updated_utc"] = utc_now_ms()
    st["updated_mono"] = time.monotonic()
    st["updated_stream"] = stream_id()
    st.setdefault("transitions", []).append({"state": state, "t_utc": st["updated_utc"], "t_mono": st["updated_mono"], "stream": st["updated_stream"], **kv})
    for k, v in kv.items():
        st[k] = v
    write_json_atomic(Path(attempt_dir) / "status.json", st)
    return st
