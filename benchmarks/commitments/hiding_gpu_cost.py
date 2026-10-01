#!/usr/bin/env python3
"""The GPU chunk tree's cost per leaf under each leaf rule, on one GPU: today's ``vllm-v1`` chunk leaves and tree
(``hidden_gpu_tree.cu``), the ``hm96-sha256/v1`` pass over them (``hm96_leaves``: two ChaCha20 blocks, 128 table rows, five
SHA-256 compressions per leaf), and, for the SHA-512 alternative, the same tree on SHA-512 with ``hm96-sha512/v1`` leaves
(C-Flock's ``sha512.cuh`` kernels: a 256-byte chunk leaf, ``hm96_finish_leaves``, the node rule; three ChaCha20 blocks).

    python benchmarks/commitments/hiding_gpu_cost.py --out J [--leaves N ...] [--reps 5]

Before any time is reported, the hm96-sha256 leaves of a random step are compared with core's reference
(``hm96.SHA256.tree_leaf``) over salts from the host's copy of the stream (``hiding.StreamSalts``).  Times are CUDA events
around each kernel phase, the median of ``--reps`` after one warm-up, on device-resident streams of ``N`` 256-byte chunks
(the defaults: the leaf counts of a SmolLM2-135M B1 256/32 and a B8 1024/128 Commit's prefill and decode steps).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "integrations" / "vllm" / "verity_vllm" / "commit" / "committer" / "hidden_gpu_src"))

import hidden_gpu as HG  # noqa: E402
from verity.commitments import hm96  # noqa: E402
from verity_vllm.commit import scheme  # noqa: E402
from verity_vllm.commit.hiding import StreamSalts  # noqa: E402

CHUNK = 256
DEFAULT_LEAVES = (7834, 1478217, 9484, 52305488)


def sha512_ext():
    from torch.utils.cpp_extension import load

    return load("hiding_gpu_cost_sha512", [str(HERE / "hiding_gpu_cost_sha512.cu")], extra_include_paths=[str(REPO / "backends" / "flock" / "cuda")],
                extra_cuda_cflags=["-O3"], verbose=False)


def check(e, salts: StreamSalts, n: int = 4096, step: int = 3) -> dict:
    rng = np.random.default_rng(7)
    buf = rng.integers(0, 256, n * CHUNK, dtype=np.uint8).tobytes()
    st = torch.frombuffer(bytearray(buf), dtype=torch.int32).cuda()
    leaves = e.chunk_leaves(st, CHUNK // 4, step, 1, step, n * CHUNK // 4, 0)
    inner = [bytes(r) for r in leaves.cpu().numpy().astype("<u4").view(np.uint8).reshape(n, 32)]
    tables, mid = salts.device_args(e, st.device)
    e.hm96_leaves(leaves, 0, salts.key_words(step), tables, mid)
    got = [bytes(r) for r in leaves.cpu().numpy().astype("<u4").view(np.uint8).reshape(n, 32)]
    ys = salts.salts(step, 0, n)
    k = salts.leaves.key
    ref = [hm96.SHA256.tree_leaf(k, hm96.SHA256.commit_string(k, inner[i], ys[128 * i:128 * (i + 1)])) for i in range(0, n, 97)]
    fast = salts.leaves_of(step, inner)
    ok = got[::97] == ref and got == fast and scheme.fold(got) != scheme.fold(inner)
    if not ok:
        raise SystemExit("hm96-sha256 GPU leaves differ from core's reference: no times reported")
    return {"leaves_checked": n, "reference_sampled": len(ref), "equal": True}


def timed(fn, reps: int) -> float:
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        ev[0].record()
        fn()
        ev[1].record()
        torch.cuda.synchronize()
        out.append(ev[0].elapsed_time(ev[1]))
    return statistics.median(out)


def bench(e, e512, salts: StreamSalts, n: int, reps: int) -> dict:
    st = torch.randint(-(1 << 31), (1 << 31) - 1, (n * CHUNK // 4,), dtype=torch.int32, device="cuda")
    tables, mid = salts.device_args(e, st.device)
    kw = salts.key_words(0)
    leaves = e.chunk_leaves(st, CHUNK // 4, 0, 1, 0, n * CHUNK // 4, 0)
    r = {"leaves": n, "stream_bytes": n * CHUNK}
    r["v1_leaf_ms"] = timed(lambda: e.chunk_leaves(st, CHUNK // 4, 0, 1, 0, n * CHUNK // 4, 0), reps)
    r["v1_tree_ms"] = timed(lambda: e.tree_levels(leaves), reps)
    r["hm96_sha256_ms"] = timed(lambda: e.hm96_leaves(leaves, 0, kw, tables, mid), reps)
    r["v1_ms"] = r["v1_leaf_ms"] + r["v1_tree_ms"]
    r["hm96_sha256_total_ms"] = r["v1_ms"] + r["hm96_sha256_ms"]
    if e512 is not None:
        inst = hm96.SHA512
        k = inst.default_key
        flat = b"".join(t.to_bytes(64, "little") for table in inst.byte_tables(k) for t in table)
        t512 = torch.frombuffer(bytearray(flat), dtype=torch.uint8).cuda()
        pre = torch.frombuffer(bytearray(inst.salt_prefix + inst.leaf_prefix(k)), dtype=torch.uint8).cuda()
        u8 = st.view(torch.uint8)
        tree = torch.empty((2 * n) * 64, dtype=torch.uint8, device="cuda")
        key = torch.frombuffer(bytearray(os.urandom(32)), dtype=torch.uint8).cuda()
        ms = e512.bench(u8, tree, n, CHUNK, t512, pre, key, reps)
        r.update({f"sha512_{a}_ms": float(b) for a, b in zip(("leaf", "hm96", "tree"), ms.tolist())})
        r["sha512_ms"] = r["sha512_leaf_ms"] + r["sha512_tree_ms"]
        r["hm96_sha512_total_ms"] = r["sha512_ms"] + r["sha512_hm96_ms"]
    for k_ in [k_ for k_ in r if k_.endswith("_ms")]:
        r[k_.replace("_ms", "_ns_per_leaf")] = round(1e6 * r[k_] / n, 3)
    del st, leaves
    torch.cuda.empty_cache()
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaves", type=int, nargs="*", default=list(DEFAULT_LEAVES))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-sha512", action="store_true")
    a = ap.parse_args(argv)
    e = HG.ext()
    salts = StreamSalts()
    doc = {"schema": "verity-vllm/hiding-gpu-cost/v1", "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "cuda": torch.version.cuda, "chunk_bytes": CHUNK, "reps": a.reps, "check": check(e, salts)}
    e512 = None if a.no_sha512 else sha512_ext()
    doc["rows"] = [bench(e, e512, salts, n, a.reps) for n in a.leaves]
    doc["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (HERE / "hiding_gpu_cost.py", HERE / "hiding_gpu_cost_sha512.cu", Path(HG.HERE) / "hidden_gpu_tree.cu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["rows"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
