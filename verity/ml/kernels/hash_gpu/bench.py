"""Measured GPU Merkle-commitment rates (hash-gpu lane).  Run on the pod:

    python -m hash_gpu.bench --reps 5 --out result.json [--shapes ligero,generic] [--hashes sha256,blake3,...]

Timing: CUDA events on the default stream, one warm-up build, then ``reps`` builds; the median is reported.
Matrices are filled with a deterministic arithmetic pattern (hash cost is data-independent).  Shapes:

* ``ligero``: the campaign's Ligero layouts at B = 4096, K = 1536, BabyBear, 2^-128 (``explore.tensor``): N = 16384
  codeword columns; B-Ligero commits 310,602 rows (20.36 GB, 1.24 MB per leaf), A-GKR 48,873 rows (3.20 GB).
  Poseidon2 runs as the fused CUDA kernel on the full shape; the torch version (memory-bound elementwise) runs on
  a 1024-column slice of the same rows and is scaled by 16 (labelled ``scaled_from_columns``).
* ``generic``: 32-byte leaves (8 BabyBear elements per column) at 2^20, 2^23, 2^25, 2^27 leaves -- the shape of
  TensorZKP's Merkle number (2^25 x 32 B in 8.59 ms on an H100).

Writes ``result.json`` (also to ``$RESEARCH_RUN_DIR/result.json`` when set) with the device, driver, library
versions, every median and the device memory in use after each build.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp
import torch

from hash_gpu import merkle
from hash_gpu.poseidon2_ref import P, instance
from hash_gpu.poseidon2_torch import Poseidon2Torch

N_CODE = 16384
LIGERO_ROWS = {"B": 310_602, "A": 48_873}
GENERIC_LOG2 = (20, 23, 25, 27)
P2_SLICE_COLS = 1024


def device_info() -> dict:
    props = cp.cuda.runtime.getDeviceProperties(0)
    _free, total = cp.cuda.runtime.memGetInfo()
    drv = cp.cuda.runtime.driverGetVersion()
    return {"gpu": props["name"].decode(), "sms": props["multiProcessorCount"], "device_memory_bytes": total,
            "cuda_driver_version": drv, "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cupy": cp.__version__,
            "numpy": np.__version__, "nvidia_smi_driver": _smi_driver()}


def _smi_driver() -> str | None:
    try:
        import subprocess

        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def fill_matrix(R: int, N: int) -> cp.ndarray:
    """(R, N) uint32 with values < P, filled in row blocks: v = (row * 2654435761 + col * 40503 + 12345) mod P."""
    m = cp.empty((R, N), dtype=cp.uint32)
    cols = cp.arange(N, dtype=cp.uint64) * 40503
    step = max(1, (1 << 24) // N)          # 128 MB uint64 temporaries beside a matrix that may fill the card
    for r0 in range(0, R, step):
        rows = cp.arange(r0, min(R, r0 + step), dtype=cp.uint64) * 2654435761
        m[r0:r0 + step] = ((rows[:, None] + cols[None, :] + 12345) % P).astype(cp.uint32)
    return m


def timed(fn, reps: int) -> tuple[float, list[float]]:
    """Median of ``reps`` CUDA-event timings (ms) after one warm-up."""
    fn()
    cp.cuda.Device().synchronize()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        fn()
        e.record()
        e.synchronize()
        out.append(cp.cuda.get_elapsed_time(s, e))
    return statistics.median(out), out


def mem_in_use() -> int:
    free, total = cp.cuda.runtime.memGetInfo()
    return total - free


def bench_tree(m, hash: str, reps: int, engine=None) -> dict:
    R, N = m.shape
    kw = {"engine": engine} if engine is not None else {}
    if hash == "blake3":
        nchunks = max(1, -(-R // merkle.CHUNK_ROWS))
        kw["workspace"] = cp.empty((nchunks, N, 8), dtype=cp.uint32)
    leaf_ms, leaf_all = timed(lambda: merkle.leaves(m, hash, **kw), reps)
    tree_ms, tree_all = timed(lambda: merkle.commit(m, hash, **kw), reps)
    nbytes = R * N * 4
    return {"hash": hash, "rows": R, "cols": N, "bytes": nbytes, "leaf_ms_median": leaf_ms, "leaf_ms_all": leaf_all,
            "tree_ms_median": tree_ms, "tree_ms_all": tree_all,
            "leaf_bytes_per_s": nbytes / (leaf_ms / 1e3), "tree_bytes_per_s": nbytes / (tree_ms / 1e3),
            "leaves_per_s": N / (tree_ms / 1e3), "device_memory_in_use_bytes": mem_in_use(), "reps": reps}


def bench_poseidon2_permutation(reps: int, S: int = 1 << 22, S_torch: int = 1 << 20) -> list[dict]:
    """Batched permutation throughput: the fused CUDA kernel, then torch by MDS route (with the isolated
    external-layer op), widths 16 and 24."""
    from hash_gpu.poseidon2_cuda import Poseidon2Cuda

    out = []
    for width in (16, 24):
        states = torch.randint(0, P, (S, width), dtype=torch.int64, device="cuda")
        eng_c = Poseidon2Cuda(instance(width))
        st_u32 = cp.asarray(states.to(torch.int32)).view(cp.uint32)
        perm_ms, _ = timed(lambda e=eng_c, st=st_u32: e.permute(st), reps)
        out.append({"width": width, "mds": "cuda-fused", "states": S, "permute_ms_median": perm_ms,
                    "permutations_per_s": S / (perm_ms / 1e3), "compression_bytes_per_s": S * 64 / (perm_ms / 1e3),
                    "note": "fused CUDA kernel, one thread per permutation, Montgomery arithmetic"})
        del st_u32
        states = states[:S_torch].contiguous()
        S = S_torch
        for mds in ("blocks", "fp64", "int8"):
            eng = Poseidon2Torch(instance(width), mds=mds, chunk_states=S)
            perm_ms, _ = timed(lambda e=eng, st=states: e.permute(st), reps)
            mds_ms, _ = timed(lambda e=eng, st=states: e._mds(st), reps)
            out.append({"width": width, "mds": mds, "states": S, "permute_ms_median": perm_ms,
                        "permutations_per_s": S / (perm_ms / 1e3), "mds_only_ms_median": mds_ms,
                        "compression_bytes_per_s": S * 64 / (perm_ms / 1e3),
                        "note": "bytes = 64 B of input per 2:1 compression (two 8-element digests)"})
        del states
        torch.cuda.empty_cache()
        S = 1 << 22
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--shapes", default="ligero,generic,permutation")
    ap.add_argument("--hashes", default="sha256,sha256-chunked,blake3,poseidon2-16,poseidon2-24,poseidon2-16-torch")
    ap.add_argument("--generic-log2", default=",".join(map(str, GENERIC_LOG2)))
    ap.add_argument("--p2-mds", default="blocks,int8")
    ap.add_argument("--out", default="result.json")
    a = ap.parse_args(argv)
    shapes = a.shapes.split(",")
    hashes = a.hashes.split(",")
    res = {"device": device_info(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ligero": [],
           "generic": [], "permutation": [], "scope": "component: proof-internal Merkle commitment hashing only"}
    print(json.dumps(res["device"]), flush=True)

    def p2_engines(hash):
        w = int(hash.split("-")[1])
        return [(mds, Poseidon2Torch(instance(w), mds=mds, chunk_states=1 << 20)) for mds in a.p2_mds.split(",")]

    def p2_cuda(m, hash, cand_or_lg, key):
        from hash_gpu.poseidon2_cuda import Poseidon2Cuda

        eng = Poseidon2Cuda(instance(int(hash.split("-")[1])))
        r = bench_tree(m, hash, a.reps, engine=eng)
        r.update({key: cand_or_lg, "backend": "cuda"})
        res["ligero" if key == "candidate" else "generic"].append(r)
        print(f"{key}={cand_or_lg} {hash}/cuda: leaves {r['leaf_ms_median']:.1f} ms, tree {r['tree_ms_median']:.1f} ms, "
              f"{r['tree_bytes_per_s'] / 1e9:.1f} GB/s, {r['leaves_per_s'] / 1e6:.1f} M leaves/s, mem {r['device_memory_in_use_bytes'] / 2**30:.1f} GiB", flush=True)

    if "ligero" in shapes:
        for cand, R in LIGERO_ROWS.items():
            for hash in hashes:
                if hash.startswith("poseidon2") and hash.endswith("-torch"):
                    hash = hash[: -len("-torch")]
                    m = torch.as_tensor(fill_matrix(R, P2_SLICE_COLS).view(cp.int32), device="cuda")
                    for mds, eng in p2_engines(hash):
                        r = bench_tree(m, hash, a.reps, engine=eng)
                        r.update({"candidate": cand, "mds": mds, "scaled_from_columns": P2_SLICE_COLS,
                                  "full_shape_bytes": R * N_CODE * 4, "full_shape_tree_ms_scaled": r["tree_ms_median"] * N_CODE / P2_SLICE_COLS})
                        res["ligero"].append(r)
                        print(f"ligero {cand} {hash}/{mds} slice {P2_SLICE_COLS} cols: tree {r['tree_ms_median']:.1f} ms  "
                              f"{r['tree_bytes_per_s'] / 1e9:.2f} GB/s", flush=True)
                    del m
                elif hash.startswith("poseidon2"):
                    m = fill_matrix(R, N_CODE)
                    p2_cuda(m, hash, cand, "candidate")
                    del m
                else:
                    m = fill_matrix(R, N_CODE)
                    r = bench_tree(m, hash, a.reps)
                    r["candidate"] = cand
                    res["ligero"].append(r)
                    print(f"ligero {cand} {hash}: leaves {r['leaf_ms_median']:.1f} ms, tree {r['tree_ms_median']:.1f} ms, "
                          f"{r['tree_bytes_per_s'] / 1e9:.1f} GB/s, mem {r['device_memory_in_use_bytes'] / 2**30:.1f} GiB", flush=True)
                    del m
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()
    if "generic" in shapes:
        for lg in map(int, a.generic_log2.split(",")):
            N = 1 << lg
            for hash in hashes:
                if hash == "sha256-chunked":
                    continue                                   # identical to sha256 for 8-row columns
                if hash.startswith("poseidon2") and hash.endswith("-torch"):
                    hash = hash[: -len("-torch")]
                    if lg > 25:
                        continue                               # int64 states would exceed the 24 GB card
                    m = torch.as_tensor(fill_matrix(8, N).view(cp.int32), device="cuda")
                    for mds, eng in p2_engines(hash):
                        r = bench_tree(m, hash, a.reps, engine=eng)
                        r.update({"log2_leaves": lg, "mds": mds})
                        res["generic"].append(r)
                        print(f"generic 2^{lg} {hash}/{mds}: tree {r['tree_ms_median']:.2f} ms  {r['tree_bytes_per_s'] / 1e9:.2f} GB/s  "
                              f"{r['leaves_per_s'] / 1e6:.1f} M leaves/s", flush=True)
                    del m
                elif hash.startswith("poseidon2"):
                    m = fill_matrix(8, N)
                    p2_cuda(m, hash, lg, "log2_leaves")
                    del m
                else:
                    m = fill_matrix(8, N)
                    r = bench_tree(m, hash, a.reps)
                    r["log2_leaves"] = lg
                    res["generic"].append(r)
                    print(f"generic 2^{lg} {hash}: leaves {r['leaf_ms_median']:.2f} ms, tree {r['tree_ms_median']:.2f} ms, "
                          f"{r['tree_bytes_per_s'] / 1e9:.1f} GB/s, {r['leaves_per_s'] / 1e6:.0f} M leaves/s", flush=True)
                    del m
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()
    if "permutation" in shapes:
        res["permutation"] = bench_poseidon2_permutation(a.reps)
        for r in res["permutation"]:
            print(f"poseidon2 w{r['width']} mds={r['mds']}: {r['permutations_per_s'] / 1e6:.2f} M perm/s  "
                  f"({r['compression_bytes_per_s'] / 1e9:.3f} GB/s of compression input; mds-only {r.get('mds_only_ms_median', float('nan')):.2f} ms / {r['states']} states)", flush=True)
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        Path(rd, "result.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
