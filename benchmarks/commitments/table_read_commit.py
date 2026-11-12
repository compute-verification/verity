#!/usr/bin/env python3
r"""Serve-time hashing of DSA long rows' table reads: what serving's row-table committer (`row_tables_device.step_roots`, the
``hm96-sha512/row-seg/v1`` rows a step commits on a side stream) costs for one decode step's reads at batch ``--batch``, on one
GPU, and its CPU twin's rate on one thread.

    research run --on NODE --gpus 1 --cwd source -- bash -lc 'cd adapters/vllm && export PATH=/usr/local/cuda/bin:$PATH \
        PYTHONPATH=.:../..:../../tools/research/src && /workspace/jobs/venv312/bin/python3 \
        ../../benchmarks/commitments/table_read_commit.py --batch 1,16'

A long row reads ``--reads`` history rows a layer from its request's table for the layer (``EmbeddingRead_v1{V=2^17,H=576}``),
and each read commits four fresh rows (`ROWS`): the history row it opens, the row's salt, its path (17 siblings of 64 bytes) and
its output row. A step's reads at batch B are, per layer and per kind, one row table of ``B * reads`` rows of that kind's bytes,
laid out as the committer takes rows (int8 codes of K = bytes - 4, then a 4-byte scale), so a table's hashing is the reads':
each row's segment digests and row digest, its salt from the step's ChaCha20 stream, its hm96 string and leaf, its frame
leaf, the table's tree and the step's tree over the tables' roots. The rows are random and every kind's table shares one
tensor across layers: the kernels' work depends on the rows' count and widths, not their values.

Arms, per batch: ``alone``, the kernels' time on an otherwise idle GPU (CUDA events, median of ``--reps``), from which
``gb_s`` is the rows' bytes over it; ``beside``, a decode-shaped bf16 GEMM loop (``x (B, H) @ w (H, N)``, memory-bound, sized
to last about as long as the hash alone) on the serving stream with the step's hash on the side stream, against the loop
alone: ``busy_x`` the loop's time as a multiple of its time alone, ``hash_x`` the hash's, and ``added_ms`` how much later both
end than the loop alone. ``cpu``: the CPU twin (`host_step`, ``row_table_host.cpp``) on one thread at batch 1 over
``--cpu-layers`` layers. ``tables``: the committer's history tables (`commit.history_tables`, the tables the reads open),
as `history.HistoryTable` (``python``), the CPU twin of the device's (``twin``, one thread) and on the GPU (``device``,
`commit.history_tables_device`): a table's creation (its 2^17-leaf tree), ``--table-rows`` rows appended at once (a
prefill's), one row at a time (a decode step's) and 2048 reads' paths, each call through to the host. Validation: a small
step's device roots and CPU twin's equal `row_tables.step_roots`, the Python reference; each table's root is
`history.check`'s over its strings, and the device table's is the twin's. JSON to ``--out`` (default
``$RESEARCH_RUN_DIR``): ``table_read_commit.json`` and ``result.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from verity_config.settings import SHARED
from verity_vllm.commit import row_tables as RTB
from verity_vllm.commit import row_tables_device as D

RESULT_SCHEMA = "research/result/v0.1"
#: a read's fresh rows and their bytes: the opened history row (576 u16 words), its salt, its path (17 x 64) and its output row
ROWS = {"row": 1152, "salt": 192, "path": 17 * 64, "out": 1152}
#: the busy loop's GEMM: GLM-5.3's hidden size by a wide projection, bf16
BUSY_H, BUSY_N = 6144, 16384


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", default="1,16", help="comma-separated batches: long-row requests a decode step")
    ap.add_argument("--reads", type=int, default=2048, help="reads a long row a layer (INDEX_TOPK)")
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--cpu-layers", type=int, default=78, help="layers of the CPU twin's batch-1 step (0: none)")
    ap.add_argument("--table-rows", type=int, default=4096, help="rows a history table appends at once (0: no tables arm)")
    ap.add_argument("--out", default=None)
    ns = ap.parse_args(argv)
    ns.batches = [int(b) for b in ns.batch.split(",")]
    if min(ns.batches + [ns.reads, ns.layers, ns.reps]) <= 0 or ns.cpu_layers < 0:
        ap.error("positive batches, reads, layers and reps")
    return ns


def step(batch: int, reads: int, layers: int, device: str, seed: int) -> dict[str, tuple[list, Any, Any]]:
    """One decode step's read tables: ``outputs[op] = (row ids, codes, scales)`` as `row_tables.step_roots` takes them."""
    g = torch.Generator(device=device).manual_seed(seed)
    ids = [(f"r{b:03d}", j) for b in range(batch) for j in range(reads)]
    kinds = {k: (torch.randint(-128, 128, (batch * reads, n - 4), dtype=torch.int8, device=device, generator=g),
                 torch.rand(batch * reads, dtype=torch.float32, device=device, generator=g)) for k, n in ROWS.items()}
    return {f"model.layers.{l}.self_attn.indexer.read.{k}": (ids, *kinds[k]) for l in range(layers) for k in ROWS}


def nbytes(outputs: dict) -> int:
    return sum(c.numel() + 4 * s.numel() for _ids, c, s in outputs.values())


def host(outputs: dict) -> dict:
    return {op: (ids, c.cpu().numpy(), s.cpu().numpy()) for op, (ids, c, s) in outputs.items()}


def alone(outputs: dict, ctx: bytes, key: bytes, reps: int) -> dict[str, Any]:
    kernel, launch = [], []
    for i in range(1 + reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = D.step_roots(ctx, 100 + i, outputs, key, timing=True)
        t1 = time.perf_counter()
        p.result()
        if i:
            kernel.append(p.kernel_ms())
            launch.append(1e3 * (t1 - t0))
    return {"kernel_ms": statistics.median(kernel), "launch_ms": statistics.median(launch), "kernel_ms_all": kernel}


def beside(outputs: dict, ctx: bytes, key: bytes, batch: int, hash_ms: float, launch_ms: float, reps: int) -> dict[str, Any]:
    """The busy loop alone and with the step's hash beside it; ``n`` GEMMs, sized from one GEMM's time to last ~``hash_ms``.
    Both streams wait on a device sleep that outlasts the host's enqueueing (the step's plan, the loop's launches), so the
    loop and the hash start together."""
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(batch, BUSY_H, dtype=torch.bfloat16, device="cuda", generator=g)
    w = torch.randn(BUSY_H, BUSY_N, dtype=torch.bfloat16, device="cuda", generator=g)
    serving, side, gs = torch.cuda.current_stream(), D.side_stream(), torch.cuda.Stream()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record(gs)
    torch.cuda._sleep(10_000_000)
    b.record(gs)
    torch.cuda.synchronize()
    cycles_ms = 10_000_000 / a.elapsed_time(b)

    def loop(n: int, with_hash: bool, i: int) -> tuple[float, float, float]:
        a, b, end, gate = (torch.cuda.Event(enable_timing=True) for _ in range(4))
        torch.cuda.synchronize()
        with torch.cuda.stream(gs):
            torch.cuda._sleep(int(cycles_ms * (2 * launch_ms + 0.05 * n + 20)))
            gate.record(gs)
        serving.wait_event(gate)
        side.wait_event(gate)
        a.record(serving)
        p = None
        if with_hash:
            with torch.cuda.stream(side):
                p = D.step_roots(ctx, 500 + i, outputs, key, stream=side, timing=True)
        for _ in range(n):
            torch.mm(x, w)
        b.record(serving)
        end.record(side)
        torch.cuda.synchronize()
        busy = a.elapsed_time(b)
        return busy, (p.kernel_ms() if p else 0.0), max(busy, a.elapsed_time(end))

    one = statistics.median(loop(50, False, 0)[0] / 50 for _ in range(3))
    n = max(1, round(hash_ms / one))
    solo, both = [], []
    for i in range(reps):
        solo.append(loop(n, False, 2 * i))
        both.append(loop(n, True, 2 * i + 1))
    busy0 = statistics.median(s[0] for s in solo)
    busy1, hash1, end1 = (statistics.median(b[k] for b in both) for k in range(3))
    return {"gemms": n, "gemm_ms": one, "busy_ms": busy0, "busy_beside_ms": busy1, "hash_beside_ms": hash1,
            "busy_x": busy1 / busy0, "hash_x": hash1 / hash_ms, "added_ms": end1 - busy0}


def cpu_twin(layers: int, reads: int, ctx: bytes, key: bytes, reps: int) -> dict[str, Any]:
    outs = host(step(1, reads, layers, "cpu", 4))
    secs = []
    for i in range(1 + reps):
        t0 = time.perf_counter()
        D.host_step(ctx, 200 + i, outs, key)
        if i:
            secs.append(time.perf_counter() - t0)
    b = sum(c.nbytes + s.nbytes for _ids, c, s in outs.values())
    return {"layers": layers, "bytes": b, "s": statistics.median(secs), "gb_s": b / statistics.median(secs) / 1e9}


def _table(make: Any, rows: int, reps: int) -> tuple[dict[str, Any], Any]:
    from verity_vllm.commit import history_tables as H

    hs = H.Histories("ab" * 64, make=make)
    t0 = time.perf_counter()
    t = hs.table("r0", 0)
    init = time.perf_counter() - t0
    block = torch.randint(0, 1 << 16, (rows, H.WORDS), generator=torch.Generator().manual_seed(5), dtype=torch.int32)
    block = block.numpy().astype("uint16")
    t0 = time.perf_counter()
    hs.append("r0", 0, 0, block)
    bulk = time.perf_counter() - t0
    one = []
    for i in range(8 * reps):
        t0 = time.perf_counter()
        hs.append("r0", 0, hs.edge("r0", 0), block[i % rows:i % rows + 1])
        one.append(time.perf_counter() - t0)
    reads = [(97 * e) % len(t.rows) for e in range(2048)]
    t0 = time.perf_counter()
    if hasattr(t, "paths"):
        t.paths(reads)
    else:
        for e in reads:
            t.path(e)
    path = (time.perf_counter() - t0) / len(reads)
    return {"init_s": init, "append_us": bulk / rows * 1e6, "append_one_us": statistics.median(one) * 1e6, "path_us": path * 1e6,
            "check": H.check(t, t.bcs)}, t


def tables(rows: int, reps: int, gpu: bool, key: bytes) -> dict[str, Any]:
    from verity.commitments import history as HT
    from verity_vllm.commit import history_tables as H
    from verity_vllm.commit import history_tables_device as HTD

    makes = {"python": HT.HistoryTable, "twin": lambda p, v, w, c: HTD.Table(p, v, w, c, key=key)}
    if gpu:
        makes["device"] = lambda p, v, w, c: HTD.Table(p, v, w, c, key=key, device=torch.cuda.current_device())
    got = {name: _table(make, rows, reps) for name, make in makes.items()}
    out: dict[str, Any] = {"rows": rows, "row_bytes": H.ROW_BYTES, **{name: r for name, (r, _t) in got.items()}}
    if gpu:
        out["device_is_twin"] = got["device"][1].root == got["twin"][1].root
    return out


def cpu_name() -> str:
    info = Path("/proc/cpuinfo")
    names = [ln.split(":", 1)[1].strip() for ln in info.read_text().splitlines() if ln.startswith("model name")] \
        if info.exists() else []
    return names[0] if names else platform.machine()


def main(argv: list[str] | None = None) -> int:
    ns = parse(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    ctx, key = hashlib.sha512(b"table-read-commit/context").digest(), hashlib.sha256(b"table-read-commit/row-key").digest()
    gpu = torch.cuda.is_available()
    small = step(1, 8, 2, "cuda" if gpu else "cpu", 1)
    want = RTB.step_roots(ctx, 7, host(small), key)
    gate = {"twin": D.host_step(ctx, 7, host(small), key) == want}
    if gpu:
        gate["device"] = D.step_roots(ctx, 7, small, key).result() == want
    per_batch: dict[str, Any] = {}
    for b in ns.batches if gpu else ():
        outs = step(b, ns.reads, ns.layers, "cuda", 2)
        n = nbytes(outs)
        solo = alone(outs, ctx, key, ns.reps)
        per_batch[str(b)] = {"bytes": n, "tables": len(outs), "rows": len(outs) * b * ns.reads, **solo,
                             "gb_s": n / solo["kernel_ms"] / 1e6,
                             "beside": beside(outs, ctx, key, b, solo["kernel_ms"], solo["launch_ms"], max(3, ns.reps // 2))}
        del outs
        torch.cuda.empty_cache()
    cpu = cpu_twin(ns.cpu_layers, ns.reads, ctx, key, 2) if ns.cpu_layers else None
    tab = tables(ns.table_rows, ns.reps, gpu, key) if ns.table_rows else None
    if tab is not None:
        gate["tables"] = all(tab[k]["check"] for k in ("python", "twin", "device") if k in tab) and tab.get("device_is_twin", True)
    ok = all(gate.values())
    nvcc_v = subprocess.run([D.nvcc(), "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:] \
        if D.nvcc() else None
    summary = {"gpu": torch.cuda.get_device_name() if gpu else None,
               "capability": list(torch.cuda.get_device_capability()) if gpu else None, "cpu": cpu_name(),
               "cpus": os.cpu_count(), "torch": torch.__version__, "cuda": torch.version.cuda, "nvcc": nvcc_v, "reads": ns.reads,
               "layers": ns.layers, "row_bytes": ROWS, "bytes_per_request_token": ns.layers * ns.reads * sum(ROWS.values()),
               "gate": gate, "batches": per_batch, "cpu_twin": cpu, "tables": tab}
    print(json.dumps({**summary, "batches": {b: {k: v for k, v in r.items() if k != "kernel_ms_all"}
                                            for b, r in per_batch.items()}}, indent=1))
    out = ns.out or SHARED.RESEARCH_RUN_DIR
    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "table_read_commit.json").write_text(json.dumps(summary, indent=1) + "\n")
        result = {
            "schema": RESULT_SCHEMA, "run_id": SHARED.RESEARCH_RUN_ID or "local", "kind": "bench-result/v1",
            "workload_fingerprint": {"workload": "table_read_commit", "reads": ns.reads, "layers": ns.layers, "row_bytes": ROWS,
                                     "batches": ns.batches, "reps": ns.reps,
                                     "implementation": "verity_vllm.commit.row_tables_device (pouw_cuda/row_table.cu)",
                                     "gpu": summary["gpu"], "capability": summary["capability"], "torch": torch.__version__,
                                     "cuda": torch.version.cuda, "nvcc": nvcc_v},
            "validation": {"status": "passed" if ok else "failed",
                           "checker": "a small step's device and CPU-twin roots equal row_tables.step_roots; each history "
                                      "table's root is history.check's over its strings, the device table's the twin's",
                           "detail": json.dumps(gate), "evidence": []},
            "measurements": [{"name": f"b{b}.{k}", "value": r[k], "unit": u} for b, r in per_batch.items()
                             for k, u in (("kernel_ms", "ms"), ("gb_s", "GB/s"), ("bytes", "bytes"))]
            + [{"name": f"b{b}.beside.{k}", "value": r["beside"][k], "unit": u} for b, r in per_batch.items()
               for k, u in (("busy_x", "x"), ("hash_x", "x"), ("added_ms", "ms"))]
            + ([{"name": "cpu_twin.gb_s", "value": cpu["gb_s"], "unit": "GB/s"}] if cpu else [])
            + [{"name": f"tables.{m}.{k}", "value": tab[m][k], "unit": u} for m in ("python", "twin", "device")
               if tab is not None and m in tab
               for k, u in (("init_s", "s"), ("append_us", "us"), ("append_one_us", "us"), ("path_us", "us"))]
            + [{"name": "workload_wall", "value": time.perf_counter() - started, "unit": "s"}],
            "measurement_files": ["table_read_commit.json"], "artifacts": [],
        }
        tmp = d / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        os.replace(tmp, d / "result.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
