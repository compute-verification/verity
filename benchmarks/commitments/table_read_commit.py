#!/usr/bin/env python3
r"""Serve-time hashing of DSA long rows' table reads: what serving's row-table committer (`row_tables_device.step_roots`, the
``hm96-sha512/row-seg/v1`` rows a step commits on a side stream) costs for one decode step's reads at batch ``--batch``, on one
GPU, and its CPU twin's rate on one thread.

    research run --on NODE --gpus 1 --cwd source -- bash -lc 'cd adapters/vllm && export PATH=/usr/local/cuda/bin:$PATH \
        PYTHONPATH=.:../..:../../tools/research/src:../../tools/config && /workspace/jobs/venv312/bin/python3 \
        ../../benchmarks/commitments/table_read_commit.py --batch 1,16'

A long row reads ``--reads`` history rows a layer from its request's table for the layer (``EmbeddingRead_v1{V=2^17,H=576}``),
and each read commits four fresh rows (`ROWS`): the history row it opens, the row's salt, its path (17 siblings of 64 bytes) and
its output row. A step's reads at batch B are, per layer and per kind, one row table of ``B * reads`` rows of that kind's bytes,
laid out as the committer takes rows (int8 codes of K = bytes - 4, then a 4-byte scale), so a table's hashing is the reads':
each row's segment digests and row digest, its salt from the step's ChaCha20 stream, its hm96 string and leaf, its frame
leaf, the table's tree and the step's tree over the tables' roots. The rows are random and every kind's table shares one
tensor across layers: the kernels' work depends on the rows' count and widths, not their values.

Arms (`ARMS`), run in that order, the cheapest first, so a run that is stopped has measured the most it could; ``--arms``
runs a subset. ``gate``: a small step's device roots and CPU twin's equal `row_tables.step_roots`, the Python reference.
``tables.python``, ``tables.twin``, ``tables.device``: the committer's history tables (`commit.history_tables`, the tables
the reads open), as `history.HistoryTable`, the CPU twin of the device's (one thread) and on the GPU
(`commit.history_tables_device`): a table's creation (its 2^17-leaf tree), ``--table-rows`` rows appended at once (a
prefill's), one row at a time (a decode step's) and 2048 reads' paths, each call through to the host; each table's root is
`history.check`'s over its strings, and the device table's is the twin's over its rows under its salts. Then, per batch: ``plan``, the host's layout of a
step (`row_tables_device.plan`: domains and rank order, on the launching thread, no GPU); ``alone``, the kernels' time on an
otherwise idle GPU (CUDA events, median of ``--reps``) with ``gb_s`` the rows' bytes over it and ``launch_ms`` the host's
time to enqueue the step; ``beside``, a decode-shaped bf16 GEMM loop (``x (B, H) @ w (H, N)``, memory-bound, sized to last
about as long as the hash alone) on the serving stream with the step's hash on the side stream, against the loop alone:
``busy_x`` the loop's time as a multiple of its time alone, ``hash_x`` the hash's, ``added_ms`` how much later both end than
the loop alone, and ``hold_ms`` the device sleep both streams waited on. ``cpu``, before ``beside``: the CPU twin
(`host_step`, ``row_table_host.cpp``) on one thread at batch 1 over ``--cpu-layers`` layers.

Each arm's start and end are logged to stderr with a UTC time. ``table_read_commit.json`` under ``--out`` (default
``$RESEARCH_RUN_DIR``) is rewritten whole, atomically, before the first arm and after every arm: ``arms`` holds each ended
arm's status (``done``, or ``failed`` with its error: the arms after it still run), start, end, seconds and result, and
``pending`` names the arms still to run, the first of them the one under way. ``result.json`` is written once, when every arm
has ended.
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
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity_config.settings import SHARED
from verity_vllm.commit import row_tables as RTB
from verity_vllm.commit import row_tables_device as D

RESULT_SCHEMA = "research/result/v0.1"
#: a read's fresh rows and their bytes: the opened history row (576 u16 words), its salt, its path (17 x 64) and its output row
ROWS = {"row": 1152, "salt": 192, "path": 17 * 64, "out": 1152}
#: the busy loop's GEMM: GLM-5.3's hidden size by a wide projection, bf16
BUSY_H, BUSY_N = 6144, 16384
#: the arms in the order they run, the cheapest first; `PER_BATCH` run once a batch, `NEEDS_GPU` only on a CUDA device
ARMS = ("gate", "tables.python", "tables.twin", "tables.device", "plan", "alone", "cpu", "beside")
PER_BATCH = ("plan", "alone", "beside")
NEEDS_GPU = ("tables.device", "alone", "beside")
#: each arm's measurements in result.json, with their units
MEASURES = {"tables": (("init_s", "s"), ("append_us", "us"), ("append_one_us", "us"), ("path_us", "us")),
            "plan": (("plan_ms", "ms"),), "alone": (("kernel_ms", "ms"), ("launch_ms", "ms"), ("gb_s", "GB/s"), ("bytes", "bytes")),
            "cpu": (("gb_s", "GB/s"),), "beside": (("busy_x", "x"), ("hash_x", "x"), ("added_ms", "ms"))}
#: what an arm raises that fails it alone: the arms after it still run
ARM_ERRORS = (RuntimeError, ValueError, ArithmeticError, LookupError, AssertionError, OSError, subprocess.SubprocessError)


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", default="1,16", help="comma-separated batches: long-row requests a decode step")
    ap.add_argument("--reads", type=int, default=2048, help="reads a long row a layer (INDEX_TOPK)")
    ap.add_argument("--layers", type=int, default=78)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--cpu-layers", type=int, default=78, help="layers of the CPU twin's batch-1 step")
    ap.add_argument("--table-rows", type=int, default=4096, help="rows a history table appends at once")
    ap.add_argument("--arms", default=None, help=f"comma-separated arms to run, of {', '.join(ARMS)}; they run in that order. "
                                                 f"Default: every arm this machine can run")
    ap.add_argument("--out", default=None)
    ns = ap.parse_args(argv)
    ns.batches = [int(b) for b in ns.batch.split(",")]
    if min(ns.batches + [ns.reads, ns.layers, ns.reps, ns.cpu_layers, ns.table_rows]) <= 0:
        ap.error("positive batches, reads, layers, reps, cpu layers and table rows")
    gpu = torch.cuda.is_available()
    if ns.arms is None:
        ns.arms = [a for a in ARMS if gpu or a not in NEEDS_GPU]
        return ns
    asked = ns.arms.split(",")
    if unknown := sorted(set(asked) - set(ARMS)):
        ap.error(f"--arms {','.join(unknown)}: the arms are {', '.join(ARMS)}")
    ns.arms = [a for a in ARMS if a in asked]
    if not gpu and (need := [a for a in ns.arms if a in NEEDS_GPU]):
        ap.error(f"--arms {','.join(need)}: no CUDA device here, and these arms run on one")
    if "beside" in ns.arms and "alone" not in ns.arms:
        ap.error("--arms beside sizes its loop from alone's time: name alone too")
    return ns


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(text: str) -> None:
    print(f"{utc()} table_read_commit: {text}", file=sys.stderr, flush=True)


def write(path: Path, doc: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, path)


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


def gate(ctx: bytes, key: bytes, gpu: bool) -> dict[str, bool]:
    small = step(1, 8, 2, "cuda" if gpu else "cpu", 1)
    want = RTB.step_roots(ctx, 7, host(small), key)
    got = {"twin": D.host_step(ctx, 7, host(small), key) == want}
    if gpu:
        got["device"] = D.step_roots(ctx, 7, small, key).result() == want
    return got


def host_plan(batch: int, reads: int, layers: int, ctx: bytes, key: bytes, reps: int) -> dict[str, Any]:
    shapes = {op: (ids, *c.shape) for op, (ids, c, _s) in step(batch, reads, layers, "cpu", 6).items()}
    ms = []
    for i in range(1 + reps):
        t0 = time.perf_counter()
        D.plan(ctx, 300 + i, shapes, key)
        if i:
            ms.append(1e3 * (time.perf_counter() - t0))
    return {"tables": len(shapes), "rows": len(shapes) * batch * reads, "plan_ms": statistics.median(ms)}


def alone(batch: int, reads: int, layers: int, ctx: bytes, key: bytes, reps: int) -> dict[str, Any]:
    outputs = step(batch, reads, layers, "cuda", 2)
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
    n, ms = nbytes(outputs), statistics.median(kernel)
    return {"bytes": n, "tables": len(outputs), "rows": len(outputs) * batch * reads, "kernel_ms": ms,
            "launch_ms": statistics.median(launch), "kernel_ms_all": kernel, "gb_s": n / ms / 1e6}


def beside(batch: int, reads: int, layers: int, ctx: bytes, key: bytes, solo: dict[str, Any], reps: int) -> dict[str, Any]:
    """The busy loop alone and with the step's hash beside it; ``n`` GEMMs, sized from one GEMM's time to last about as long
    as the hash alone (``solo``, `alone`'s result at this batch). Both streams wait on a device sleep that outlasts the
    host's enqueueing (the step's plan, the loop's launches), so the loop and the hash start together. The sleep counts
    clock cycles, calibrated by a sleep on its own stream and refused unless a 20 ms sleep then lasts 10 to 40 ms."""
    outputs = step(batch, reads, layers, "cuda", 2)
    hash_ms, launch_ms = solo["kernel_ms"], solo["launch_ms"]
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(batch, BUSY_H, dtype=torch.bfloat16, device="cuda", generator=g)
    w = torch.randn(BUSY_H, BUSY_N, dtype=torch.bfloat16, device="cuda", generator=g)
    serving, side, gs = torch.cuda.current_stream(), D.side_stream(), torch.cuda.Stream()

    def sleep(cycles: int) -> tuple[Any, Any]:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(gs):
            a.record(gs)
            torch.cuda._sleep(cycles)
            b.record(gs)
        return a, b

    sleep(1)
    torch.cuda.synchronize()
    a, b = sleep(10_000_000)
    torch.cuda.synchronize()
    cycles_ms = 10_000_000 / a.elapsed_time(b)
    a, b = sleep(int(20 * cycles_ms))
    torch.cuda.synchronize()
    if not 10 <= (held := a.elapsed_time(b)) <= 40:
        raise RuntimeError(f"the device sleep is miscalibrated: {cycles_ms:.0f} cycles a ms held a 20 ms sleep {held:.2f} ms")

    def loop(n: int, with_hash: bool, i: int) -> tuple[float, float, float, float]:
        a, b, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        torch.cuda.synchronize()
        h0, go = sleep(int(cycles_ms * (2 * launch_ms + 0.05 * n + 20)))
        serving.wait_event(go)
        side.wait_event(go)
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
        return busy, (p.kernel_ms() if p else 0.0), max(busy, a.elapsed_time(end)), h0.elapsed_time(go)

    one = statistics.median(loop(50, False, 0)[0] / 50 for _ in range(3))
    n = max(1, round(hash_ms / one))
    solo_runs, both = [], []
    for i in range(reps):
        solo_runs.append(loop(n, False, 2 * i))
        both.append(loop(n, True, 2 * i + 1))
    busy0 = statistics.median(s[0] for s in solo_runs)
    busy1, hash1, end1, hold = (statistics.median(r[k] for r in both) for k in range(4))
    return {"gemms": n, "gemm_ms": one, "busy_ms": busy0, "busy_beside_ms": busy1, "hash_beside_ms": hash1,
            "busy_x": busy1 / busy0, "hash_x": hash1 / hash_ms, "added_ms": end1 - busy0, "hold_ms": hold}


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


def table(kind: str, rows: int, reps: int) -> dict[str, Any]:
    """One history table as ``kind`` (``python``, ``twin`` or ``device``): its creation, ``rows`` rows appended at once, one
    row at a time and 2048 reads' paths, with its root and whether `history.check` finds it over the table's strings; the
    device table's also whether the CPU twin over its rows under its salts has its root (``twin``)."""
    from verity.commitments import history as HT
    from verity_vllm.commit import history_tables as H
    from verity_vllm.commit import history_tables_device as HTD

    make = {"python": HT.HistoryTable, "twin": HTD.Table,
            "device": lambda p, v, w, c: HTD.Table(p, v, w, c, device=torch.cuda.current_device())}[kind]
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
    same = {}
    if kind == "device":
        twin = HTD.Table(t.program, t.value, t.words, t.leaves)
        twin.append(np.stack(t.rows), t.salts)
        same = {"twin": twin.root == t.root and twin.bcs == t.bcs}
    return {"rows": rows, "row_bytes": H.ROW_BYTES, "init_s": init, "append_us": bulk / rows * 1e6,
            "append_one_us": statistics.median(one) * 1e6, "path_us": path * 1e6, "check": H.check(t, t.bcs), "root": t.root.hex(),
            **same}


def units(ns: argparse.Namespace, ctx: bytes, key: bytes) -> list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]]:
    """The run's arms in order, each ``(name, run)``: ``run(done)`` takes the results of the arms done so far."""
    def unit(arm: str, b: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
        return {"gate": lambda done: gate(ctx, key, torch.cuda.is_available()),
                "plan": lambda done: host_plan(b, ns.reads, ns.layers, ctx, key, ns.reps),
                "alone": lambda done: alone(b, ns.reads, ns.layers, ctx, key, ns.reps),
                "cpu": lambda done: cpu_twin(ns.cpu_layers, ns.reads, ctx, key, 2),
                "beside": lambda done: beside(b, ns.reads, ns.layers, ctx, key, done[f"alone.{b}"], max(3, ns.reps // 2)),
                }.get(arm) or (lambda done: table(arm.split(".")[1], ns.table_rows, ns.reps))

    return [(f"{arm}.{b}" if arm in PER_BATCH else arm, unit(arm, b))
            for arm in ns.arms for b in (ns.batches if arm in PER_BATCH else [0])]


def gates(done: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """What the arms done so far validate: the gate's roots, and each history table's check, the device table's with its
    twin's root."""
    got = dict(done.get("gate", {}))
    if tabs := [r for n, r in done.items() if n.startswith("tables.")]:
        got["tables"] = all(r["check"] and r.get("twin", True) for r in tabs)
    return got


def checker(done: dict[str, dict[str, Any]]) -> str:
    """The checks a passed validation rests on: `gates`' for the arms in ``done``, and no check of an arm left out."""
    said = ["every arm ended done"]
    if roots := [name for k, name in (("device", "device"), ("twin", "CPU-twin")) if k in done.get("gate", {})]:
        said.append(f"a small step's {' and '.join(roots)} roots equal row_tables.step_roots")
    if tabs := [a.split(".")[1] for a in ARMS if a.startswith("tables.") and a in done]:
        said.append(f"each history table's root ({', '.join(tabs)}) is history.check's over its strings")
        if "device" in tabs:
            said.append("the device table's root is its CPU twin's over its rows under its salts")
    return "; ".join(said)


def run(arms: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]], summary: dict[str, Any],
        path: Path | None) -> dict[str, dict[str, Any]]:
    """Run ``arms`` in order, ``summary`` rewritten to ``path`` before the first and after each: ``summary["arms"]`` each ended
    arm's status, start, end, seconds and result or error, ``summary["pending"]`` the arms still to run. An arm that raises
    one of `ARM_ERRORS` is recorded as failed and the next runs. -> the results of the arms done."""
    state, done = summary["arms"], {}
    for i, (name, fn) in enumerate(arms):
        summary["pending"], summary["gate"] = [n for n, _fn in arms[i:]], gates(done)
        if path is not None:
            write(path, summary)
        log(f"arm {name} start")
        start, t0 = utc(), time.perf_counter()
        try:
            done[name] = fn(done)
            state[name] = {"status": "done", "result": done[name]}
        except ARM_ERRORS as e:
            state[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        ended = state[name]
        ended.update(start=start, end=utc(), s=time.perf_counter() - t0)
        log(f"arm {name} {ended['status']} in {ended['s']:.1f} s" + (f": {ended['error']}" if "error" in ended else ""))
    summary["pending"], summary["gate"] = [], gates(done)
    if path is not None:
        write(path, summary)
    return done


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
    nvcc_v = subprocess.run([D.nvcc(), "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:] \
        if D.nvcc() else None
    summary: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name() if gpu else None,
        "capability": list(torch.cuda.get_device_capability()) if gpu else None, "cpu": cpu_name(), "cpus": os.cpu_count(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "nvcc": nvcc_v, "reads": ns.reads, "layers": ns.layers,
        "batches": ns.batches, "row_bytes": ROWS, "bytes_per_request_token": ns.layers * ns.reads * sum(ROWS.values()),
        "arms": {}, "pending": [], "gate": {}}
    out = ns.out or SHARED.RESEARCH_RUN_DIR
    d = Path(out) if out else None
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
    done = run(units(ns, ctx, key), summary, d / "table_read_commit.json" if d is not None else None)
    failed = sorted(n for n, a in summary["arms"].items() if a["status"] != "done")
    ok = not failed and all(summary["gate"].values())
    print(json.dumps(summary, indent=1))
    if d is not None:
        write(d / "result.json", {
            "schema": RESULT_SCHEMA, "run_id": SHARED.RESEARCH_RUN_ID or "local", "kind": "bench-result/v1",
            "workload_fingerprint": {"workload": "table_read_commit", "reads": ns.reads, "layers": ns.layers, "row_bytes": ROWS,
                                     "batches": ns.batches, "reps": ns.reps, "arms": ns.arms,
                                     "implementation": "verity_vllm.commit.row_tables_device (pouw_cuda/row_table.cu)",
                                     "gpu": summary["gpu"], "capability": summary["capability"], "torch": torch.__version__,
                                     "cuda": torch.version.cuda, "nvcc": nvcc_v},
            "validation": {"status": "passed" if ok else "failed",
                           "checker": checker(done),
                           "detail": json.dumps({"gate": summary["gate"], "failed": failed}), "evidence": []},
            "measurements": [{"name": f"{name}.{k}", "value": r[k], "unit": u} for name, r in done.items()
                             for k, u in MEASURES.get(name.split(".")[0], ())]
            + [{"name": "workload_wall", "value": time.perf_counter() - started, "unit": "s"}],
            "measurement_files": ["table_read_commit.json"], "artifacts": [],
        })
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
