"""Kernel-identity evidence: compare the Triton kernel sources Inductor generated for production's
compile (no pass) and replay's compile (with ``verity::record`` inserted).

Inductor writes every generated Triton kernel as ``<TORCHINDUCTOR_CACHE_DIR>/<xx>/<hash>.py``
containing ``async_compile.triton('<name>', '''<source>''')`` (vLLM's standalone_compile path still
goes through the same codegen and cache directory) and every wrapper as a ``.py`` with ``def call(args)``.
We hash each kernel's *normalised* source: the kernel name (``triton_red_fused_..._N``; the trailing
index depends on ordering within a wrapper) and the ``kernel_name`` inductor_meta are replaced by a
placeholder, everything else - the Triton program, the heuristics decorator with its size hints,
reduction hints, `tl.constexpr` block choices - is kept byte-for-byte. Two kernels with equal
normalised hashes are the same program (same numerics); production's set must be a subset of
replay's, and replay must add *no* kernel that production does not have (an added kernel means a
record op split a fusion group / materialised a value that was register-resident).

Also parses the wrappers: kernel launches, extern kernel calls and ``torch.ops.verity.record.default(bufN, id)``
lines, so each record id can be attributed to the buffer it observes and to the kernel that produced it.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_TRITON_DEF = re.compile(r"async_compile\.triton\(\s*'([A-Za-z0-9_]+)'\s*,\s*'''(.*?)'''", re.S)
_KNAME = re.compile(r"\btriton_(?:red|per|poi|tem|for|unk|foreach)_[A-Za-z0-9_]+")
_META_NAME = re.compile(r"'kernel_name':\s*'[^']*'")
_BACKEND_HASH = re.compile(r"'backend_hash':\s*'[^']*'")
# [COMP, CX1] under tensor parallelism rank r compiles on cuda:r and Inductor bakes `DeviceProperties(type='cuda', index=r, ...)` into
# every kernel's triton_meta: the ranks' kernels differ in that one integer and nothing else (cx1 ref1: 8 vs 8 kernels, 0 common by
# norm_hash, every pair identical after this normalisation). A second, device-agnostic hash for the comparison; norm_hash is unchanged.
_DEVICE_INDEX = re.compile(r"(DeviceProperties\(type='cuda', index=)\d+")
# Inductor's *combo* kernels (horizontal fusion, `num_xblocks_0/1`) carry a benchmark harness inside the triton source string
# (`rand_strided(..., device='cuda:r')`, `_DeviceGuard(r)`, `set_device(r)`, `get_raw_stream(r)`, `raw_streamr`): the same device
# residue in four more spellings (cx1 commit: the k-rope+qkv-bias combo kernel counted as "only in production" until these were
# normalised; rank 0's production and replay bodies were byte-identical)
_DEVICE_RESIDUE = re.compile(r"(device='cuda:)\d+|(_DeviceGuard\()\d+|(set_device\()\d+|(get_raw_stream\()\d+|(raw_stream)\d+")
_RECORD = re.compile(r"torch\.ops\.verity\.record\.default\(([A-Za-z0-9_]+),\s*(\d+)\)")
_LAUNCH = re.compile(r"^\s*([A-Za-z0-9_]+)\.run\((.*)$", re.M)
_EXTERN = re.compile(r"^\s*(extern_kernels\.[a-z_0-9]+|torch\.ops\.[A-Za-z0-9_.]+)\((.*)$", re.M)


@dataclass
class KernelSrc:
    name: str
    path: str
    norm_hash: str
    raw_hash: str
    lines: int
    dev_hash: str = ""  # norm_hash with the CUDA device index normalised (equal across TP ranks)


@dataclass
class WrapperInfo:
    path: str
    kernels: list[str]
    externs: list[str]
    records: list[tuple[str, int]]  # (buffer, id)
    record_after: dict[int, str] = field(default_factory=dict)  # id -> preceding launch/extern line
    graph_hash: str = ""


def normalise_kernel(src: str) -> str:
    s = _KNAME.sub("KERNEL", src)
    s = _META_NAME.sub("'kernel_name': 'KERNEL'", s)
    s = _BACKEND_HASH.sub("'backend_hash': 'BACKEND'", s)
    return s.strip()


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def device_agnostic(norm_src: str) -> str:
    s = _DEVICE_INDEX.sub(r"\1N", norm_src)
    return _DEVICE_RESIDUE.sub(lambda m: next(g for g in m.groups() if g) + "N", s)


def collect(cache_dir: Path) -> tuple[list[KernelSrc], list[WrapperInfo]]:
    kernels: list[KernelSrc] = []
    wrappers: list[WrapperInfo] = []
    for p in sorted(cache_dir.rglob("*.py")):
        try:
            txt = p.read_text()
        except Exception:  # noqa: BLE001
            continue
        found = _TRITON_DEF.findall(txt)
        # kernel definitions live both in the wrapper (inline async_compile.triton(...) blocks) and,
        # for kernels Inductor compiled asynchronously, in their own <hash>.py; dedupe by content
        seen_here: set[str] = set()
        for name, src in found:
            h = _sha(normalise_kernel(src))
            if h in seen_here:
                continue
            seen_here.add(h)
            kernels.append(KernelSrc(name=name, path=str(p), norm_hash=h, raw_hash=_sha(src.strip()), lines=src.count("\n"),
                                     dev_hash=_sha(device_agnostic(normalise_kernel(src)))))
        if "def call(" in txt:
            wrappers.append(parse_wrapper(p, txt))
    return kernels, wrappers


def parse_wrapper(p: Path, txt: str) -> WrapperInfo:
    parts = re.split(r"\n\s*def call\((?:self, )?args\):", txt)
    bodies = parts[1:] or [txt]
    body = next((b for b in bodies if re.search(r"^\s*s\d+ = arg\d+_1", b, re.M)), bodies[-1])
    body = body.split("\ndef benchmark_compiled_module", 1)[0].split("\nrunner = Runner", 1)[0]
    kernels = [m.group(1) for m in _LAUNCH.finditer(body) if m.group(1).startswith("triton_")]
    externs = [m.group(1) for m in _EXTERN.finditer(body) if "verity.record" not in m.group(1)]
    records: list[tuple[str, int]] = []
    record_after: dict[int, str] = {}
    last = ""
    for line in body.splitlines():
        m = _RECORD.search(line)
        if m:
            records.append((m.group(1), int(m.group(2))))
            record_after[int(m.group(2))] = last.strip()
            continue
        if _LAUNCH.match(line) or _EXTERN.match(line) or "triton_kernel_wrapper" in line:
            last = line
    # the wrapper's own identity: body with record lines removed (so production and replay wrappers
    # of the same piece can be compared), buffer names kept
    stripped = "\n".join(l for l in body.splitlines() if "verity.record" not in l)
    return WrapperInfo(path=str(p), kernels=kernels, externs=externs, records=records, record_after=record_after, graph_hash=_sha(stripped))


def compare(prod_dir: Path, replay_dir: Path) -> dict[str, Any]:
    pk, pw = collect(prod_dir)
    rk, rw = collect(replay_dir)
    p_by = {}
    for k in pk:
        p_by.setdefault(k.norm_hash, []).append(k)
    r_by = {}
    for k in rk:
        r_by.setdefault(k.norm_hash, []).append(k)
    only_prod = sorted(set(p_by) - set(r_by))
    only_replay = sorted(set(r_by) - set(p_by))
    common = sorted(set(p_by) & set(r_by))

    def describe(h: str, by: dict[str, list[KernelSrc]]) -> dict[str, Any]:
        ks = by[h]
        names = sorted({_KNAME.sub(lambda m: re.sub(r"_\d+$", "", m.group(0)), k.name) for k in ks})
        return {"norm_hash": h, "kernel_families": names, "copies": len(ks), "example_path": ks[0].path, "lines": ks[0].lines}

    p_dev = {k.dev_hash for k in pk}
    r_dev = {k.dev_hash for k in rk}
    return {
        "production_cache": str(prod_dir),
        "replay_cache": str(replay_dir),
        "production": {"kernel_files": len(pk), "distinct_kernels": len(p_by), "wrappers": len(pw), "distinct_kernels_modulo_device_index": len(p_dev)},
        "replay": {"kernel_files": len(rk), "distinct_kernels": len(r_by), "wrappers": len(rw), "distinct_kernels_modulo_device_index": len(r_dev)},
        "common_distinct_kernels": len(common),
        "only_in_production": [describe(h, p_by) for h in only_prod],
        "only_in_replay": [describe(h, r_by) for h in only_replay],
        "kernels_identical": not only_prod and not only_replay,
        # TP: the production cache holds every rank's kernels (one flat TORCHINDUCTOR_CACHE_DIR), this replay dir one rank's; the
        # per-rank statement is "the rank's kernels are production kernels" (modulo the device index), and the production-only set
        # modulo the device index is what production compiled that no rank of this replay did
        "kernels_identical_modulo_device_index": p_dev == r_dev,
        "replay_subset_of_production_modulo_device_index": r_dev <= p_dev,
        "only_in_production_modulo_device_index": len(p_dev - r_dev),
        "only_in_replay_modulo_device_index": len(r_dev - p_dev),
        "replay_record_ops_in_wrappers": sum(len(w.records) for w in rw),
        "production_record_ops_in_wrappers": sum(len(w.records) for w in pw),
        "replay_wrapper_graph_hashes": sorted({w.graph_hash for w in rw}),
        "production_wrapper_graph_hashes": sorted({w.graph_hash for w in pw}),
        "common_kernels": [describe(h, r_by) for h in common],
    }


def record_attribution(replay_dir: Path) -> dict[int, dict[str, Any]]:
    """id -> {buffer, after: the wrapper line that produced the buffer (kernel launch / extern call)}."""
    _k, ws = collect(replay_dir)
    out: dict[int, dict[str, Any]] = {}
    for w in ws:
        for buf, rid in w.records:
            out[rid] = {"buffer": buf, "after": w.record_after.get(rid, ""), "wrapper": w.path}
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--production", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = compare(Path(a.production), Path(a.replay))
    rep["record_attribution"] = {str(k): v for k, v in sorted(record_attribution(Path(a.replay)).items())}
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("production", "replay", "common_distinct_kernels", "kernels_identical", "replay_record_ops_in_wrappers")}, indent=1))
    for k in rep["only_in_production"]:
        print("ONLY-PROD", k["kernel_families"], k["example_path"], file=sys.stderr)
    for k in rep["only_in_replay"]:
        print("ONLY-REPLAY", k["kernel_families"], k["example_path"], file=sys.stderr)
    return 0 if rep["kernels_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
