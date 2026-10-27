#!/usr/bin/env python3
"""Serving's per-step row-table hashing: what ``verity_vllm.commit.row_tables.step_roots`` costs at the serving loop's step
boundary, for one decode step of Llama-3.1-8B int8 at batch ``--batch``.

    research run --on auto --cwd source --cpus 16 --mem-gb 32 -- python benchmarks/commitments/step_root_cost.py --workers 1,4,8,16

A decode step has one row table per layer and linear (qkv, o, gate_up with K = 4096; down with K = 14336), each of ``--batch``
token rows (the s8 codes, then the f32 scale): 128 tables of 32 rows at B32, 27.3 MB.  Per call, from the host copies of the
quantize outputs: each row's salt (ChaCha20 under the run's row key), its ``sha512/row-seg/v1`` digest, ``b || c`` and
``frame-v3-sha512`` leaf, each table's tree, and the step's tree over the table roots.  The first call is ``cold``; ``seconds`` is
the median of the next ``--reps``.  ``workers`` > 1 forks a pool per call (``serving_rows.commit_ports``), so its time includes the
pool's start.  Beside each wall time, ``cpu_s`` is this process's CPU time, which other load on a shared node does not inflate (at
``workers`` > 1 only the parent's share: the pool's processes are its children, so read their wall time).  Apart: the salts alone and the row digests alone, one thread; and the floor, one SHA-512 pass over every row's
bytes on each worker count's threads of a pool that outlives the calls (hashlib releases the GIL on rows this long), what any
SHA-512 commitment of these rows costs at least on this CPU.  Validation: the root equals
``verity.ml.commitments.row_table.commit_step``'s over the same rows and salts (the pure-Python reference).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.commitments import row_table as RT
from verity.ml.commitments.rowleaf import sha512_row_seg_digest
from verity_vllm.commit import row_tables as RTB
from verity.ml.windows import rows as SR

RESULT_SCHEMA = "research/result/v0.1"
#: Llama-3.1-8B's int8 linears and their K
LINEARS = {"self_attn.qkv_proj": 4096, "self_attn.o_proj": 4096, "mlp.gate_up_proj": 4096, "mlp.down_proj": 14336}


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=int, default=32, help="decode batch: each table's rows")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--k-scale", type=int, default=1, help="divide every K by this (tests)")
    ap.add_argument("--workers", default="1", help="comma-separated worker counts")
    ap.add_argument("--reps", type=int, default=5, help="warm calls after the cold one, per worker count")
    ap.add_argument("--out", default=None, help="directory for result.json (default $RESEARCH_RUN_DIR)")
    ns = ap.parse_args(argv)
    ns.workers = [int(w) for w in ns.workers.split(",")]
    if min(ns.batch, ns.layers, ns.k_scale, ns.reps, *ns.workers) <= 0:
        ap.error("--batch, --layers, --k-scale, --reps and --workers must be positive")
    return ns


def outputs(batch: int, layers: int, k_scale: int, seed: int = 7) -> dict[str, tuple[list[tuple[str, int]], np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    ids = [(f"r{b:03d}", 0) for b in range(batch)]
    return {f"model.layers.{layer}.{lin}": (ids, rng.integers(-128, 128, (batch, k // k_scale), dtype=np.int8),
                                            (np.abs(rng.standard_normal(batch)) / 127).astype(np.float32))
            for layer in range(layers) for lin, k in LINEARS.items()}


def reference(ctx: bytes, step: int, outs: dict[str, Any], key: bytes) -> bytes:
    tables = []
    ys, at = SR.segment_salts(key, step, sum(len(v[0]) for v in outs.values())), 0
    for op in sorted(outs, key=RTB.table_key):
        ids, x, s = outs[op]
        order = sorted(range(len(ids)), key=lambda b: ids[b])
        rows = [RT.s8_row(x[b].tobytes(), int(s[b:b + 1].view("<u4")[0])) for b in order]
        salts = [ys[at + j * SR.SALT_BYTES:at + (j + 1) * SR.SALT_BYTES] for j in range(len(rows))]
        tables.append((rows, salts))
        at += len(rows) * SR.SALT_BYTES
    return RT.commit_step(ctx, step, tables).root


def sha512_pass(pool: ThreadPoolExecutor, parts: list[list[bytes]]) -> None:
    list(pool.map(lambda p: [hashlib.sha512(r).digest() for r in p], parts))


def timed(f: Any) -> tuple[float, float, Any]:
    """(wall seconds, this process's CPU seconds, ``f()``)."""
    t0, c0 = time.perf_counter(), time.process_time()
    out = f()
    return time.perf_counter() - t0, time.process_time() - c0, out


def medians(f: Any, reps: int) -> tuple[float, float]:
    ts = [timed(f)[:2] for _ in range(reps)]
    return statistics.median(t for t, _ in ts), statistics.median(c for _, c in ts)


def main(argv: list[str] | None = None) -> int:
    ns = parse(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    ctx, key, step = hashlib.sha512(b"step-root-cost/context").digest(), hashlib.sha256(b"step-root-cost/row-key").digest(), 1
    outs = outputs(ns.batch, ns.layers, ns.k_scale)
    tables, rows = len(outs), sum(len(v[0]) for v in outs.values())
    hashed = sum(x.nbytes + s.nbytes for _ids, x, s in outs.values())
    by_workers, root = {}, None
    for w in ns.workers:
        calls, cpus = [], []
        for _ in range(1 + ns.reps):
            dt, cpu, got = timed(lambda w=w: RTB.step_roots(ctx, step, outs, key, workers=w))
            calls.append(dt)
            cpus.append(cpu)
            if root not in (None, got.root):
                raise SystemExit(f"workers={w}: another root")
            root = got.root
        by_workers[w] = {"cold_s": calls[0], "seconds": statistics.median(calls[1:]), "warm_s": calls[1:],
                         "cpu_s": statistics.median(cpus[1:])}
    salts_s, salts_cpu = medians(lambda: SR.segment_salts(key, step, rows), ns.reps)
    flat = [(x[b].tobytes() + s[b:b + 1].tobytes()) for _ids, x, s in outs.values() for b in range(len(x))]
    digest_s, digest_cpu = medians(lambda: [sha512_row_seg_digest(r, 8 * len(r)) for r in flat], ns.reps)
    floor, floor_cpu = {}, {}
    for w in ns.workers:
        parts = [flat[i::w] for i in range(w)]
        with ThreadPoolExecutor(w) as pool:
            sha512_pass(pool, parts)
            floor[w], floor_cpu[w] = medians(lambda: sha512_pass(pool, parts), ns.reps)
    ref_s, _, ref = timed(lambda: reference(ctx, step, outs, key))
    ok = ref == root
    summary = {"batch": ns.batch, "layers": ns.layers, "k_scale": ns.k_scale, "tables": tables, "rows": rows, "bytes_hashed": hashed,
               "root": root.hex(), "reference_root_equal": ok, "reference_s": ref_s, "salts_s": salts_s, "salts_cpu_s": salts_cpu,
               "row_digests_s": digest_s, "row_digests_cpu_s": digest_cpu, "sha512_floor_s": floor, "sha512_floor_cpu_s": floor_cpu,
               "by_workers": by_workers}
    print(json.dumps({k: v for k, v in summary.items() if k != "root"}, indent=1))
    out = ns.out or os.environ.get("RESEARCH_RUN_DIR")
    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "step_root_cost.json").write_text(json.dumps(summary, indent=1) + "\n")
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
            "kind": "bench-result/v1",
            "workload_fingerprint": {"workload": "step_root_cost", "model": "Llama-3.1-8B int8", "linears": LINEARS, "batch": ns.batch,
                                     "layers": ns.layers, "k_scale": ns.k_scale, "row_schema": RT.SCHEMA, "roots_schema": RT.ROOTS_SCHEMA,
                                     "implementation": "verity_vllm.commit.row_tables.step_roots (hashlib SHA-512, numpy ChaCha20, fork pool)",
                                     "workers": ns.workers, "reps": ns.reps, "cpu": platform.processor() or platform.machine(),
                                     "nproc": os.cpu_count(), "python": sys.version.split()[0]},
            "validation": {"status": "passed" if ok else "failed",
                           "checker": "the step root equals verity.ml.commitments.row_table.commit_step's over the same rows and salts",
                           "detail": "ok" if ok else f"reference root {ref.hex()[:16]}", "evidence": []},
            "measurements": [{"name": f"step_roots.w{w}.seconds", "value": v["seconds"], "unit": "s"} for w, v in by_workers.items()]
            + [{"name": f"step_roots.w{w}.cold_seconds", "value": v["cold_s"], "unit": "s"} for w, v in by_workers.items()]
            + ([{"name": "step_roots.w1.cpu_seconds", "value": by_workers[1]["cpu_s"], "unit": "s"}] if 1 in by_workers else [])
            + [{"name": f"sha512_floor.w{w}.seconds", "value": v, "unit": "s"} for w, v in floor.items()]
            + [{"name": f"sha512_floor.w{w}.cpu_seconds", "value": v, "unit": "s"} for w, v in floor_cpu.items()]
            + [{"name": "salts.seconds", "value": salts_s, "unit": "s"}, {"name": "row_digests.seconds", "value": digest_s, "unit": "s"},
               {"name": "salts.cpu_seconds", "value": salts_cpu, "unit": "s"},
               {"name": "row_digests.cpu_seconds", "value": digest_cpu, "unit": "s"},
               {"name": "reference.seconds", "value": ref_s, "unit": "s"}, {"name": "rows", "value": rows, "unit": "count"},
               {"name": "bytes_hashed", "value": hashed, "unit": "bytes"},
               {"name": "workload_wall", "value": time.perf_counter() - started, "unit": "s"}],
            "measurement_files": ["step_root_cost.json"],
            "artifacts": [],
        }
        tmp = d / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        os.replace(tmp, d / "result.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
