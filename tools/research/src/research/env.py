"""`research.env.conditions()`: the one canonical baseline of an attempt's conditions (store README §6.3).

Dotted keys, scalar values, `None` when unknown -- never omitted, never guessed.  Collected by the runner for every attempt;
lanes never hand-write it.  `hw.family` comes from the ONE SKU->family table below (`SKU_FAMILY`); nobody else normalises.
Every probe is a subprocess with a timeout or a stdlib call, and none of them raises: an unavailable fact is `None`.

    source.commit  source.tree  source.dirty  source.dirty_digest  source.lockfiles.<name>   (git in `source_root`, default cwd)
    rt.os  rt.kernel  rt.python  rt.rust  rt.cuda  rt.driver  rt.torch  rt.triton  rt.nvcc
    hw.family  hw.sku  hw.count  hw.vram_mb  hw.cpu  hw.hostname
    exec.exclusive  exec.load_start  exec.load_end  exec.project  exec.campaign  exec.runs_root

`source.lockfiles.<name>` is one key per lockfile found at the source root (`uv.lock`, `Cargo.lock`); the nested form used by
the attempt manifest and the reuse key is `source_identity(conds)`.  A shipped `git archive` has no `.git`: the control side
passes the identity of the commit it shipped (`source_hint`, remote.shipped_identity), which is the whole identity where git
knows no commit here and is not read where it does.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

KEYS = (
    "source.commit", "source.tree", "source.dirty", "source.dirty_digest",
    "rt.os", "rt.kernel", "rt.python", "rt.rust", "rt.cuda", "rt.driver", "rt.torch", "rt.triton", "rt.nvcc",
    "hw.family", "hw.sku", "hw.count", "hw.vram_mb", "hw.cpu", "hw.hostname",
    "exec.exclusive", "exec.load_start", "exec.load_end", "exec.project", "exec.campaign", "exec.runs_root", "exec.machine",
)
# exec.machine: the machines.toml name a `research run --on <machine>` ran on (job.json run.remote.machine), indexed so that
# `research pods unpreserved|drain <machine>` can find every Attempt recorded on a machine; hw.hostname is the container's hostname.
LOCKFILES = ("uv.lock", "Cargo.lock")
UNTRACKED_CAP_BYTES = 64 << 20
#: The runner's capture of a dirty source in the run dir (`source_conditions(dirty_capture=...)`), which custody always uploads
#: (store.custody.RUN_FILES): the tree on the machine is never a run's only copy of its source.  Inside: the diff against HEAD, and
#: each untracked, unignored file under DIRTY_UNTRACKED/.
DIRTY_CAPTURE = "source-dirty.tar"
DIRTY_PATCH = "diff-HEAD.patch"
DIRTY_UNTRACKED = "untracked"

# The ONE SKU -> family table.  First match wins; `\b` keeps L40 off L40S and A10 off A100.  Strings as nvidia-smi prints them:
# "NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA L40S", "NVIDIA GeForce RTX 4090".
SKU_FAMILY: list[tuple[str, str]] = [
    (r"\bH200\b", "h200"),
    (r"\bH100\b", "h100"),
    (r"\bB200\b", "b200"),
    (r"\bA100\b", "a100"),
    (r"\bL40S\b", "l40s"),
    (r"\bL40\b", "l40"),
    (r"\bRTX\s*4090\b", "rtx4090"),
    (r"\bRTX\s*5090\b", "rtx5090"),
    (r"\bRTX\s*PRO\s*6000\b", "rtxpro6000"),
    (r"\bA10G?\b", "a10"),
    (r"\bT4\b", "t4"),
]


def sku_family(sku: str | None, count: int | None) -> str | None:
    """`cpu` when there is no GPU, `unknown` for a GPU the table does not know, None when the probe itself failed."""
    if count == 0:
        return "cpu"
    if not sku:
        return None
    for pat, fam in SKU_FAMILY:
        if re.search(pat, sku, re.I):
            return fam
    return "unknown"


# ---- probes (never raise) -----------------------------------------------------------------------------------------------------------------

def _run(cmd: list[str], *, cwd: Path | None = None, timeout: float = 10.0, binary: bool = False) -> bytes | str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout if binary else r.stdout.decode("utf-8", errors="replace")


def _git(args: list[str], root: Path, *, timeout: float = 10.0, binary: bool = False) -> bytes | str | None:
    out = _run(["git", *args], cwd=root, timeout=timeout, binary=binary)
    return out if binary or out is None else out.strip()


class _Hashing:
    """A file read through `h`: what a tar member copies is exactly what the digest hashed."""

    def __init__(self, f: Any, h: Any) -> None:
        self.f, self.h = f, h

    def read(self, n: int = -1) -> bytes:
        b = self.f.read(n)
        self.h.update(b)
        return b


def _dirty_digest(root: Path, commit: str = "HEAD", capture: Path | None = None) -> str | None:
    """sha256 over `git diff <commit> --binary`, then each untracked, unignored path as `rel NUL bytes NUL` in path order (a path
    that is not a regular file, such as a nested repository's `sub/`, as `rel NUL NUL`).  With `capture`, the same bytes go there as
    a tar in the same pass (DIRTY_PATCH, then DIRTY_UNTRACKED/<rel> per file), so the tar reproduces the digest; it is left only
    when it holds every byte of the dirty source (removed for None, `too-large`, a path that is not a regular file, or any error).
    Capturing never changes the digest.  EMPTY_DIGEST means nothing was hashed: the working tree is the commit's."""
    import tarfile

    diff = _git(["diff", commit, "--binary"], root, timeout=60, binary=True)
    if diff is None:
        return None
    others = _git(["ls-files", "-z", "--others", "--exclude-standard"], root, timeout=60)
    if others is None:
        return None
    tar = None
    if capture is not None:
        try:
            tar = tarfile.open(capture, "w", format=tarfile.PAX_FORMAT)
            info = tarfile.TarInfo(DIRTY_PATCH)
            info.size = len(diff)
            tar.addfile(info, io.BytesIO(diff))
        except (OSError, tarfile.TarError):
            tar = _drop(tar, capture)
    h = hashlib.sha256()
    h.update(diff)
    total = 0
    for rel in sorted(p for p in others.split("\0") if p):
        p = root / rel
        try:
            if not p.is_file():
                h.update(rel.encode("utf-8") + b"\0\0")
                tar = _drop(tar, capture)
                continue
            total += p.stat().st_size
            if total > UNTRACKED_CAP_BYTES:
                _drop(tar, capture)
                return "too-large"
            h.update(rel.encode("utf-8") + b"\0")
            with open(p, "rb") as f:
                if tar is not None:
                    try:
                        info = tarfile.TarInfo(f"{DIRTY_UNTRACKED}/{rel}")
                        info.size = os.fstat(f.fileno()).st_size
                        tar.addfile(info, _Hashing(f, h))
                    except (OSError, tarfile.TarError):
                        tar = _drop(tar, capture)
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            h.update(b"\0")
        except OSError:
            tar = _drop(tar, capture)
            continue
    if tar is not None:
        try:
            tar.close()
        except (OSError, tarfile.TarError):
            _drop(tar, capture)
    return h.hexdigest()


def _drop(tar: Any, capture: Path | None) -> None:
    """Abandon a capture: close it and remove the file, so a partial tar is never mistaken for the dirty state."""
    if tar is not None:
        with contextlib.suppress(OSError, ValueError):
            tar.close()
    if capture is not None:
        with contextlib.suppress(OSError):
            capture.unlink()
    return None


def _sha256_file(p: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()
SOURCE_READS = 3


def _git_source(root: Path, capture: Path | None) -> dict[str, Any] | None:
    """One read of `source.*` against the commit HEAD named at its start; None when HEAD moved during it.  Dirty means the working
    tree's bytes differ from that commit's: a status entry that changes no byte (an index-only change) is clean."""
    out: dict[str, Any] = {"source.commit": None, "source.tree": None, "source.dirty": None, "source.dirty_digest": None}
    commit = _git(["rev-parse", "HEAD"], root)
    if not commit:
        return out
    out["source.commit"] = commit
    out["source.tree"] = _git(["rev-parse", f"{commit}^{{tree}}"], root) or None
    porcelain = _git(["status", "--porcelain", "--untracked-files=all"], root, timeout=60)
    if porcelain is not None:
        digest = _dirty_digest(root, commit, capture) if porcelain else None
        if digest == EMPTY_DIGEST:
            porcelain, digest = "", _drop(None, capture)
        out["source.dirty"], out["source.dirty_digest"] = bool(porcelain), digest
    if _git(["rev-parse", "HEAD"], root) != commit:
        _drop(None, capture)
        return None
    return out


def source_conditions(source_root: Path | None = None, source_hint: dict[str, Any] | None = None, *,
                      dirty_capture: Path | None = None) -> dict[str, Any]:
    """`source.*` from git in `source_root`, read again when a commit lands during the read (SOURCE_READS times, then unknown).  With
    `dirty_capture` (a path in the run dir), a dirty source's bytes are written there (`_dirty_digest`); nothing is written for a
    clean one.  Without git (a shipped `git archive`) the hint is the whole identity; with git it is not read."""
    root = Path(source_root or os.getcwd())
    out: dict[str, Any] = {"source.commit": None, "source.tree": None, "source.dirty": None, "source.dirty_digest": None}
    for _ in range(SOURCE_READS):
        if (read := _git_source(root, dirty_capture)) is not None:
            out = read
            break
    hint = source_hint or {}
    if out["source.commit"] is None and hint.get("commit"):
        for k in ("commit", "tree", "dirty", "dirty_digest"):
            out[f"source.{k}"] = hint.get(k)
    for name in LOCKFILES:
        p = root / name
        if p.is_file():
            out[f"source.lockfiles.{name}"] = _sha256_file(p)
    return out


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except Exception:  # noqa: BLE001  (PackageNotFoundError and anything a broken dist-info can throw)
        return None


def _nvidia_smi() -> dict[str, Any]:
    out = {"hw.sku": None, "hw.count": None, "hw.vram_mb": None, "rt.driver": None, "rt.cuda": None}
    if shutil.which("nvidia-smi") is None:
        out["hw.count"] = 0
        return out
    q = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"], timeout=15)
    if q is not None:
        rows = [[c.strip() for c in line.split(",")] for line in q.strip().splitlines() if line.strip()]
        out["hw.count"] = len(rows)
        if rows and len(rows[0]) >= 3:
            out["hw.sku"] = rows[0][0] or None
            out["rt.driver"] = rows[0][1] or None
            try:
                out["hw.vram_mb"] = int(float(rows[0][2]))
            except ValueError:
                pass
    full = _run(["nvidia-smi", "-q"], timeout=15)
    if full:
        m = re.search(r"CUDA Version\s*:\s*(\S+)", full)
        if m:
            out["rt.cuda"] = m.group(1)
    return out


def _cpu() -> str | None:
    model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if not model and platform.system() == "Darwin":
        model = (_run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5) or "").strip() or None
    model = model or platform.processor() or platform.machine() or None
    n = os.cpu_count()
    return f"{model} x{n}" if model and n else model


def _rustc() -> str | None:
    out = _run(["rustc", "--version"], timeout=10)
    if not out:
        return None
    parts = out.split()
    return parts[1] if len(parts) > 1 else out.strip()


def _nvcc() -> str | None:
    out = _run(["nvcc", "--version"], timeout=10)
    if not out:
        return None
    m = re.search(r"V(\d+\.\d+(?:\.\d+)?)", out)
    return m.group(1) if m else out.strip().splitlines()[-1]


def load_now() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        return None


def conditions(*, run_dir: Path | str | None = None, project: str | None = None, campaign: str | None = None, exclusive: bool | None = None,
               source_root: Path | str | None = None, source_hint: dict[str, Any] | None = None,
               dirty_capture: Path | None = None) -> dict[str, object]:
    """Every key in KEYS present (plus `source.lockfiles.<name>` per lockfile found); None when unknown."""
    out: dict[str, Any] = {k: None for k in KEYS}
    out.update(source_conditions(Path(source_root) if source_root else None, source_hint, dirty_capture=dirty_capture))
    out["rt.os"] = platform.system() or None
    out["rt.kernel"] = platform.release() or None
    out["rt.python"] = platform.python_version()
    out["rt.rust"] = _rustc()
    out["rt.nvcc"] = _nvcc()
    out["rt.torch"] = _version("torch")
    out["rt.triton"] = _version("triton")
    out.update(_nvidia_smi())
    out["hw.family"] = sku_family(out["hw.sku"], out["hw.count"])
    out["hw.cpu"] = _cpu()
    try:
        out["hw.hostname"] = socket.gethostname()
    except OSError:
        pass
    out["exec.exclusive"] = exclusive
    out["exec.load_start"] = load_now()
    out["exec.project"] = project
    out["exec.campaign"] = campaign if campaign is not None else os.environ.get("RESEARCH_CAMPAIGN")
    out["exec.runs_root"] = str(Path(run_dir).parent) if run_dir else None
    return out


def conditions_from_job(job: dict[str, Any] | None) -> dict[str, object]:
    """Conditions of an attempt reconstructed from what its job.json recorded (after-the-fact publish of a run that predates
    `conditions()`).  Only recorded facts are copied; everything else stays None."""
    out: dict[str, Any] = {k: None for k in KEYS}
    if not job:
        return out
    env = job.get("environment") or {}
    comp = job.get("computation") or {}
    own = job.get("ownership") or {}
    run = job.get("run") or {}
    store = job.get("store") or {}
    out["source.commit"] = comp.get("tree_sha")
    out["rt.kernel"] = env.get("kernel")
    out["rt.python"] = env.get("python")
    out["hw.hostname"] = env.get("hostname")
    gpu = env.get("gpu") or {}
    devs = gpu.get("devices") or []
    if gpu.get("available") is False and gpu.get("reason", "").startswith("nvidia-smi not found"):
        out["hw.count"] = 0
    elif gpu.get("available"):
        out["hw.count"] = len(devs)
    if devs:
        out["hw.sku"] = devs[0].get("name")
        out["rt.driver"] = devs[0].get("driver")
        mem = devs[0].get("memory_total_bytes")
        out["hw.vram_mb"] = int(mem // (1024 * 1024)) if isinstance(mem, int) else None
    out["hw.family"] = sku_family(out["hw.sku"], out["hw.count"])
    if env.get("cpu_count_host"):
        out["hw.cpu"] = f"{env.get('machine') or 'cpu'} x{env['cpu_count_host']}"
    out["exec.project"] = own.get("project")
    out["exec.campaign"] = store.get("campaign")
    out["exec.runs_root"] = run.get("runs_root")
    out["exec.machine"] = (run.get("remote") or {}).get("machine")
    return out


def source_identity(conds: dict[str, Any]) -> dict[str, Any]:
    """The nested `source` block of the attempt manifest / reuse key from the flat conditions."""
    prefix = "source.lockfiles."
    return {"commit": conds.get("source.commit"), "tree": conds.get("source.tree"), "dirty": conds.get("source.dirty"),
            "dirty_digest": conds.get("source.dirty_digest"),
            "lockfiles": {k[len(prefix):]: v for k, v in sorted(conds.items()) if k.startswith(prefix)}}
