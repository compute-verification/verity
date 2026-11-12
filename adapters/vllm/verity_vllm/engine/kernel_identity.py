"""Generated-kernel identity (CONTRACT §16.1, rung (a)).

Inductor writes every compiled graph piece as a Python *wrapper module* (``<cache>/<xx>/<hash>.py``)
that embeds the Triton source of each kernel it launches (``async_compile.triton('<name>', '''...''')``)
and a ``call()`` body that is the launch program of the fused region. The *cache key* of such a
module depends on the whole Inductor config (so it differs between production and a replay whose
``CompilationConfig`` differs); the *sources* are what actually ran. This module collects, per
process, the set of generated kernel sources and wrapper programs by content hash. Equality of two
processes' kernel-source sets (``summary.kernel_set_sha256``, which a run's record pins) is the
evidence that a replay executed production's generated kernels.

Also collected: Triton's compiled artifacts in ``TRITON_CACHE_DIR`` when it is set: per kernel the
``*.cubin`` hash and the hash of the *code* part of its PTX. Triton embeds the path of the Python
source file into the binary (DWARF ``.file``/``.loc`` line info and the ``assertFile_*`` strings of
``tl.device_assert``); Inductor's generated kernels live under the Inductor cache root, so two
processes with different cache roots get byte-different cubins for byte-identical sources. Verified
on the pinned torch/Triton: ttir, ttgir, llir and ptx of such a pair differ only in those path
strings. ``ptx_code_sha256`` hashes the PTX with ``.file``/``.loc`` directives, ``.debug_*``
sections and ``assertFile_*`` globals removed; equality of the ``ptx_code`` sets is the binary-level
identity, the raw cubin hashes are reported as well. Wrappers get a ``norm_sha256`` likewise (their
``# kernel path:`` comments name the cache root; ``# AOT ID`` differs per compilation).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

__all__ = ["collect", "inductor_cache_dir", "triton_cache_dir"]

_TRITON_SRC = re.compile(r"async_compile\.triton\('([^']+)', '''(.*?)''', device_str=", re.S)
_AOT_ID = re.compile(r"^# AOT ID: \[(.*?)\]", re.M)
_KERNEL_HEADER = re.compile(r"^@triton_heuristics\.\w+\(|^@triton\.jit", re.M)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def inductor_cache_dir() -> str:
    """Where this process's Inductor wrappers live (env override or torch's default); raises when torch can't say."""
    d = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if d:
        return d
    from torch._inductor.runtime.runtime_utils import cache_dir

    return cache_dir()


def triton_cache_dir() -> str | None:
    return os.environ.get("TRITON_CACHE_DIR")


_WRAPPER_PATH_LINE = re.compile(r"^# (kernel path|AOT ID|Original path): .*$", re.M)


def normalize_wrapper(src: str) -> str:
    """Wrapper text minus the lines that name the cache root / this compilation (comments only)."""
    return _WRAPPER_PATH_LINE.sub("", src)


def ptx_code(ptx: str) -> str:
    """PTX minus debug line info (``.file``/``.loc``, ``.debug_*`` sections) and the ``assertFile_*``
    source-path strings of ``tl.device_assert``: what ptxas compiles into instructions."""
    out: list[str] = []
    skip = False
    for line in ptx.splitlines():
        s = line.strip()
        if s.startswith(".section"):
            parts = s.split()
            skip = len(parts) > 1 and parts[1].startswith(".debug_")
            continue
        if skip or s.startswith((".file", ".loc")):
            continue
        if s.startswith(".global") and "assertFile_" in s:
            continue
        out.append(line)
    return "\n".join(out)


def _call_body(src: str) -> str:
    """The launch program: from the last ``class Runner``/``def call`` to the end of the module,
    minus the benchmark harness."""
    idx = src.rfind("\nclass Runner:")
    if idx < 0:
        idx = src.rfind("\ndef call(")
    if idx < 0:
        return ""
    body = src[idx:]
    cut = body.find("\ndef benchmark_compiled_module")
    return body if cut < 0 else body[:cut]


def _read_text(p: Path, root: Path, unreadable: dict[str, str]) -> str | None:
    """``p``'s text, or None with the failure recorded under its path relative to ``root``."""
    try:
        return p.read_text()
    except Exception as e:  # noqa: BLE001
        unreadable[str(p.relative_to(root))] = repr(e)[:200]
        return None


def collect(inductor_dir: str | None, triton_dir: str | None = None, *, vllm_cache_dir: str | None = None) -> dict[str, Any]:
    """Scan the Inductor cache directory: wrapper modules (graph pieces) and kernel modules.

    A file that can't be read is named in ``unreadable`` and in ``summary.error``, both present only then: the sets and digests
    leave it out, so they are partial, and a reader claims no identity over them."""
    wrappers: list[dict[str, Any]] = []
    kernel_files: list[dict[str, Any]] = []
    kernels: dict[str, str] = {}  # kernel name -> source sha256 (collisions listed separately)
    collisions: list[dict[str, Any]] = []
    unreadable: dict[str, str] = {}
    root = Path(inductor_dir) if inductor_dir else None
    if root is not None and root.is_dir():
        for p in sorted(root.rglob("*.py")):
            src = _read_text(p, root, unreadable)
            if src is None:
                continue
            rel = str(p.relative_to(root))
            if "async_compile.triton(" in src or "\ndef call(" in src:
                ks = []
                for m in _TRITON_SRC.finditer(src):
                    name, body = m.group(1), m.group(2)
                    h = sha256_bytes(body.encode())
                    ks.append({"name": name, "source_sha256": h})
                    prev = kernels.get(name)
                    if prev is not None and prev != h:
                        collisions.append({"name": name, "a": prev, "b": h, "wrapper": rel})
                    kernels.setdefault(name, h)
                aot = _AOT_ID.findall(src)
                call = _call_body(src)
                wrappers.append({
                    "path": rel, "sha256": sha256_bytes(src.encode()), "norm_sha256": sha256_bytes(normalize_wrapper(src).encode()),
                    "bytes": len(src), "aot_id": aot[-1] if aot else None,
                    "call_sha256": sha256_bytes(call.encode()) if call else None, "kernels": ks,
                    "launches": len(re.findall(r"^\s+\w+\.run\(", call, re.M)) if call else None,
                    "extern_calls": sorted(set(re.findall(r"(extern_kernels\.\w+|torch\.ops\.[\w.]+)\(", call))) if call else [],
                })
            elif _KERNEL_HEADER.search(src):
                kernel_files.append({"path": rel, "sha256": sha256_bytes(src.encode()), "bytes": len(src)})
    autotune = autotune_configs(root, unreadable) if root is not None and root.is_dir() else []
    cubins: list[dict[str, Any]] = []
    troot = Path(triton_dir) if triton_dir else None
    if troot is not None and troot.is_dir():
        for p in sorted(troot.rglob("*.cubin")):
            entry = {"path": str(p.relative_to(troot)), "name": p.stem, "sha256": sha256_file(p), "bytes": p.stat().st_size, "ptx_code_sha256": None}
            ptx = p.with_suffix(".ptx")
            if ptx.exists():
                ptx_src = _read_text(ptx, troot, unreadable)
                if ptx_src is not None:
                    entry["ptx_code_sha256"] = sha256_bytes(ptx_code(ptx_src).encode())
            cubins.append(entry)
    kernel_pairs = sorted({(k["name"], k["source_sha256"]) for w in wrappers for k in w["kernels"]})
    doc: dict[str, Any] = {
        "schema": "verity-vllm/kernel-identity/v1",
        "inductor_cache_dir": inductor_dir,
        "triton_cache_dir": triton_dir,
        "vllm_compile_cache_dir": vllm_cache_dir,
        "wrappers": wrappers,
        "kernel_files": kernel_files,
        "kernels": {k: h for k, h in kernel_pairs},
        "kernel_name_collisions": collisions,
        "autotune_configs": autotune,
        "cubins": cubins,
        "summary": {
            "wrappers": len(wrappers),
            "distinct_kernel_sources": len({h for _, h in kernel_pairs}),
            "kernel_set_sha256": sha256_bytes(json.dumps(kernel_pairs).encode()),
            "wrapper_set_sha256": sha256_bytes(json.dumps(sorted(w["sha256"] for w in wrappers)).encode()),
            "wrapper_norm_set_sha256": sha256_bytes(json.dumps(sorted(w["norm_sha256"] for w in wrappers)).encode()),
            "call_set_sha256": sha256_bytes(json.dumps(sorted(w["call_sha256"] or "" for w in wrappers)).encode()),
            "autotuned_kernels": len(autotune),
            "cubins": len(cubins),
            "cubin_set_sha256": sha256_bytes(json.dumps(sorted(c["sha256"] for c in cubins)).encode()) if cubins else None,
            "ptx_code_set_sha256": sha256_bytes(json.dumps(sorted(c["ptx_code_sha256"] or "" for c in cubins)).encode()) if cubins else None,
        },
    }
    if unreadable:
        doc["unreadable"] = [{"path": k, "error": v} for k, v in sorted(unreadable.items())]
        doc["summary"]["error"] = (f"{len(unreadable)} file(s) unreadable, so these sets are partial: "
                                   + "; ".join(f"{k}: {v}" for k, v in sorted(unreadable.items())[:4]))
    return doc


_CONFIG_KEYS = ("XBLOCK", "YBLOCK", "R0_BLOCK", "RBLOCK", "num_warps", "num_stages")


def autotune_configs(root: Path, unreadable: dict[str, str]) -> list[dict[str, Any]]:
    """Inductor's runtime autotuner writes ``<kernel hash>.best_config`` next to a kernel module when it
    benchmarked more than one launch configuration (``autotune_pointwise``, pointwise kernels) — the chosen
    ``XBLOCK`` / ``num_warps`` / ``num_stages``. Those are *launch parameters* selected by timing in each
    process, not part of the generated source; a pointwise kernel's per-element arithmetic does not depend on
    them, but they are the one thing production and replay may legitimately differ in, so they are recorded.
    Each entry: kernel name (from the sibling ``.py``), the kernel module's sha256, the config.
    A file that can't be read or parsed is recorded in ``unreadable`` (path relative to ``root``)."""
    out: list[dict[str, Any]] = []
    for bc in sorted(root.rglob("*.best_config")):
        text = _read_text(bc, root, unreadable)
        if text is None:
            continue
        try:
            cfg = json.loads(text)
        except ValueError as e:
            unreadable[str(bc.relative_to(root))] = repr(e)[:200]
            continue
        name, ksha = None, None
        for py in sorted(bc.parent.glob("*.py")):
            src = _read_text(py, root, unreadable)
            if src is None:
                continue
            m = re.search(r"^def (\w+)\(", src, re.M)
            if m:
                name, ksha = m.group(1), sha256_bytes(src.encode())
                break
        out.append({
            "path": str(bc.relative_to(root)), "kernel": name, "kernel_module_sha256": ksha,
            "config": {k: cfg[k] for k in _CONFIG_KEYS if k in cfg},
            "configs_hash": cfg.get("configs_hash"), "found_by_coordesc": cfg.get("found_by_coordesc"),
        })
    return out
