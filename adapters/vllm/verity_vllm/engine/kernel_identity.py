"""Generated-kernel identity (CONTRACT §16.1, rung (a)).

Inductor writes every compiled graph piece as a Python *wrapper module* (``<cache>/<xx>/<hash>.py``)
that embeds the Triton source of each kernel it launches (``async_compile.triton('<name>', '''...''')``)
and a ``call()`` body that is the launch program of the fused region. The *cache key* of such a
module depends on the whole Inductor config (so it differs between production and a replay whose
``CompilationConfig`` differs); the *sources* are what actually ran. This module collects, per
process, the set of generated kernel sources and wrapper programs by content hash, and compares
two such collections. Equality of the kernel-source set is the evidence that a replay executed
production's generated kernels; a difference is reported kernel by kernel.

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

CLI::

    verity-vllm kernel-identity --inductor-dir D [--triton-dir T] --out kernels.json
    verity-vllm kernel-identity --compare a.json b.json --out cmp.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import option

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
    leave it out, so they are partial, and ``compare`` claims no identity over them."""
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
    doc = {
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


def _whole(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Neither collection left a file out (``collect``'s ``unreadable``)."""
    return not (a.get("unreadable") or b.get("unreadable"))


_CONFIG_KEYS = ("XBLOCK", "YBLOCK", "R0_BLOCK", "RBLOCK", "num_warps", "num_stages")


def autotune_configs(root: Path, unreadable: dict[str, str]) -> list[dict[str, Any]]:
    """Inductor's runtime autotuner writes ``<kernel hash>.best_config`` next to a kernel module when it
    benchmarked more than one launch configuration (``autotune_pointwise``, pointwise kernels) — the chosen
    ``XBLOCK`` / ``num_warps`` / ``num_stages``. Those are *launch parameters* selected by timing in each
    process, not part of the generated source; a pointwise kernel's per-element arithmetic does not depend on
    them, but they are the one thing production and replay may legitimately differ in, so they are recorded
    and compared. Each entry: kernel name (from the sibling ``.py``), the kernel module's sha256, the config.
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


def compare_autotune(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Per kernel module (keyed by the module's content sha256 — Inductor names standalone kernel modules
    ``triton_`` generically, so the name alone is ambiguous): the autotuner's chosen launch config in a vs b
    (``differing`` lists the kernels whose chosen config is not the same; ``candidates_identical`` says whether
    both saw the same candidate set). Neither is True when either collection left a file out."""
    whole = _whole(a, b)
    ca = {e["kernel_module_sha256"]: e for e in a.get("autotune_configs", []) if e.get("kernel_module_sha256")}
    cb = {e["kernel_module_sha256"]: e for e in b.get("autotune_configs", []) if e.get("kernel_module_sha256")}
    differing = []
    for k in sorted(set(ca) | set(cb), key=lambda h: ((ca.get(h) or cb.get(h))["kernel"] or "", h)):
        ea, eb = ca.get(k), cb.get(k)
        if ea is None or eb is None or ea["config"] != eb["config"]:
            differing.append({"kernel": (ea or eb)["kernel"], "kernel_module_sha256": k,
                              "a": ea["config"] if ea else None, "b": eb["config"] if eb else None})
    return {
        "a": len(ca), "b": len(cb),
        "chosen_configs_identical": (not differing and whole) if (ca or cb) else None,
        "candidates_identical": (whole and all(ca[k]["configs_hash"] == cb[k]["configs_hash"] for k in set(ca) & set(cb)))
                                if (set(ca) & set(cb)) else None,
        "differing": differing,
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Set comparison of generated kernel sources (by content), wrappers and call programs.

    When either collection left a file out (``collect``'s ``unreadable``), every identity below is False and ``unreadable`` names
    the files by side: sets that are equal without them show nothing."""
    whole = _whole(a, b)
    ka = {(k["name"], k["source_sha256"]) for w in a["wrappers"] for k in w["kernels"]}
    kb = {(k["name"], k["source_sha256"]) for w in b["wrappers"] for k in w["kernels"]}
    ha = {h for _, h in ka}
    hb = {h for _, h in kb}
    wa = {w["sha256"]: w for w in a["wrappers"]}
    wb = {w["sha256"]: w for w in b["wrappers"]}
    ca = {w["call_sha256"] for w in a["wrappers"] if w["call_sha256"]}
    cb = {w["call_sha256"] for w in b["wrappers"] if w["call_sha256"]}
    cub_a = {c["sha256"] for c in a.get("cubins", [])}
    cub_b = {c["sha256"] for c in b.get("cubins", [])}
    na = {w.get("norm_sha256") for w in a["wrappers"] if w.get("norm_sha256")}
    nb = {w.get("norm_sha256") for w in b["wrappers"] if w.get("norm_sha256")}
    pa = {c["ptx_code_sha256"]: c["name"] for c in a.get("cubins", []) if c.get("ptx_code_sha256")}
    pb = {c["ptx_code_sha256"]: c["name"] for c in b.get("cubins", []) if c.get("ptx_code_sha256")}
    out = {
        "kernel_sources_identical": ha == hb and bool(ha) and whole,
        "kernel_sources": {"a": len(ha), "b": len(hb), "common": len(ha & hb), "only_a": sorted(n for n, h in ka if h not in hb), "only_b": sorted(n for n, h in kb if h not in ha)},
        "wrappers_identical": set(wa) == set(wb) and bool(wa) and whole,
        "wrappers_identical_modulo_cache_path": na == nb and bool(na) and whole,
        "wrappers_norm": {"a": len(na), "b": len(nb), "common": len(na & nb)},
        "wrappers": {"a": len(wa), "b": len(wb), "common": len(set(wa) & set(wb)), "only_a": sorted(w["path"] for h, w in wa.items() if h not in wb), "only_b": sorted(w["path"] for h, w in wb.items() if h not in wa)},
        "call_programs_identical": ca == cb and bool(ca) and whole,
        "call_programs": {"a": len(ca), "b": len(cb), "common": len(ca & cb)},
        "cubins_identical": (cub_a == cub_b and bool(cub_a) and whole) if (cub_a or cub_b) else None,
        "cubins": {"a": len(cub_a), "b": len(cub_b), "common": len(cub_a & cub_b)},
        "ptx_code_identical": (set(pa) == set(pb) and bool(pa) and whole) if (pa or pb) else None,
        "ptx_code": {"a": len(pa), "b": len(pb), "common": len(set(pa) & set(pb)),
                     "only_a": sorted(n for h, n in pa.items() if h not in pb), "only_b": sorted(n for h, n in pb.items() if h not in pa)},
        "b_kernel_sources_subset_of_a": hb <= ha and bool(hb) and whole,
        "autotune": compare_autotune(a, b),
    }
    if not whole:
        out["unreadable"] = {side: d["unreadable"] for side, d in (("a", a), ("b", b)) if d.get("unreadable")}
    return out


@dataclass(frozen=True, kw_only=True)
class Options:
    inductor_dir: str | None = option("--inductor-dir", default=None)
    triton_dir: str | None = option("--triton-dir", default=None)
    compare: list[str] | None = option("--compare", default=None, nargs=2, metavar=("A_JSON", "B_JSON"))
    out: str = option("--out")


def main(args: Options) -> None:
    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        doc = compare(a, b)
    else:
        doc = collect(args.inductor_dir or inductor_cache_dir(), args.triton_dir or triton_cache_dir())
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc.get("summary", doc), indent=1))
