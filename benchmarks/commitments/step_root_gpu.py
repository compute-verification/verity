#!/usr/bin/env python3
r"""Serving's per-step row-table hashing on the GPU, overlapped: what ``row_tables_device.step_roots`` adds to a graphed int8
decode step of Llama-3.1-8B at batch ``--batch``, on one GPU.

    research run --on auto --cpus 8 --mem-gb 32 --cwd source -- bash -lc 'cd integrations/vllm && export PATH=/usr/local/cuda/bin:$PATH \
        PYTHONPATH=.:../..:../../tools/research/src && GPU_LEASE_WHO=<you> gpu-lease 1 --wait --max-min 20 -- \
        /workspace/jobs/venv312/bin/python3 ../../benchmarks/commitments/step_root_gpu.py'

The step is a proxy of the ``int8_exact`` profile's decode linears: per layer qkv, o, gate_up and down (K = 4096, 4096, 4096,
14336; N = 6144, 4096, 28672, 4096), each ``int8_exact.quantize_rows``, ``int_mm`` and ``dequantize`` on random int8 weights, in
one CUDA graph.  Attention, the norms and the LM head are left out, so each ratio is against the linears alone (an upper bound
on the hashing's share of a served step).  Arms, interleaved rep by rep:

  plain       the graph alone;
  copied      the graph with each quantize output also copied into a step arena (two, by step parity): 256 copy kernels;
  kept        the graph keeping each quantize output where it wrote it (two graphs, by step parity, each holding its outputs
              in its own memory pool): the arenas without a copy;
  overlapped  the kept graph, then ``row_tables_device.step_roots`` of the step's tables on the side stream; before replaying a
              graph again the serving stream waits, on the device, for the hash of the step that graph last wrote.

Each step ends with a host sync on the serving stream (a served step's sampled tokens), and the overlapped loop then hands every
ready step root to its recorder (``record_ready``).  ``step_ms``: per arm, the median over reps of a rep's median step.  Apart,
on an otherwise idle GPU: ``kernel_ms`` (``rt_step``'s four kernels, CUDA events) and ``host_ms`` (``step_roots``' host time, of
which ``plan_ms`` is the layout: domains and rank order).  In the overlapped arm, ``latency_ms``: from the end of a step's graph
to its root on the host side (CUDA events).  Validation: the device's roots of a random step (and of the last overlapped step,
read back from its graph's outputs) are ``row_tables.step_roots``' on the CPU, and the CPU twin's are too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity_vllm.commit import row_tables as RTB
from verity_vllm.commit import row_tables_device as D
from verity_vllm.engine import int8_exact as E

RESULT_SCHEMA = "research/result/v0.1"
#: Llama-3.1-8B's int8 linears: (K, N)
LINEARS = {"self_attn.qkv_proj": (4096, 6144), "self_attn.o_proj": (4096, 4096), "mlp.gate_up_proj": (4096, 28672),
           "mlp.down_proj": (14336, 4096)}
ARMS = ("plain", "copied", "kept", "overlapped")


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--steps", type=int, default=40, help="steps per rep and arm")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default=None, help="directory for result.json (default $RESEARCH_RUN_DIR)")
    ns = ap.parse_args(argv)
    if min(ns.batch, ns.layers, ns.steps, ns.reps) <= 0 or ns.batch < 17:
        ap.error("positive counts, and --batch >= 17 (int_mm's M without padding)")
    return ns


class Model:
    """The decode linears, their inputs and the two copy arenas; ``forward(arena, keep)`` is one step's work (captured), copying
    its quantize outputs into ``arena`` or keeping them in ``keep``."""

    def __init__(self, ns: argparse.Namespace) -> None:
        g = torch.Generator(device="cuda").manual_seed(1)
        self.layers, self.batch = ns.layers, ns.batch
        self.w = {(l, lin): (torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda", generator=g),
                             torch.full((n,), 1.0 / (73.0 * k ** 0.5), dtype=torch.float32, device="cuda"))
                  for l in range(ns.layers) for lin, (k, n) in LINEARS.items()}
        self.x0 = torch.randn(ns.batch, 4096, dtype=torch.bfloat16, device="cuda", generator=g)
        self.out = torch.empty_like(self.x0)
        self.ops = [self.op(l, lin) for l in range(ns.layers) for lin in LINEARS]
        self.arenas = [{self.op(l, lin): (torch.zeros(ns.batch, k, dtype=torch.int8, device="cuda"),
                                          torch.zeros(ns.batch, dtype=torch.float32, device="cuda"))
                        for l in range(ns.layers) for lin, (k, _n) in LINEARS.items()} for _ in range(2)]
        self.kept: list[dict] = [{}, {}]
        self.ids = [(f"r{b:03d}", 0) for b in range(ns.batch)]

    @staticmethod
    def op(layer: int, lin: str) -> str:
        return f"model.layers.{layer}.{lin}"

    def linear(self, x: torch.Tensor, layer: int, lin: str, arena: dict | None, keep: dict | None) -> torch.Tensor:
        xq, s = E.quantize_rows(x)
        if arena is not None:
            c, sc = arena[self.op(layer, lin)]
            c.copy_(xq)
            sc.copy_(s)
        if keep is not None:
            keep[self.op(layer, lin)] = (xq, s)
        w, s_w = self.w[(layer, lin)]
        return E.dequantize(E.int_mm(xq, w), s, s_w)

    def forward(self, arena: dict | None, keep: dict | None) -> None:
        h = self.x0
        for l in range(self.layers):
            q = self.linear(h, l, "self_attn.qkv_proj", arena, keep)
            h = h + self.linear(q[:, :4096].contiguous(), l, "self_attn.o_proj", arena, keep)
            gu = self.linear(h, l, "mlp.gate_up_proj", arena, keep)
            h = h + self.linear((torch.nn.functional.silu(gu[:, :14336]) * gu[:, 14336:]).contiguous(), l, "mlp.down_proj", arena,
                                keep)
        self.out.copy_(h)

    def outputs(self, arena: dict) -> dict:
        return {op: (self.ids, c, s) for op, (c, s) in arena.items()}

    def host_outputs(self, arena: dict) -> dict:
        return {op: (self.ids, c.cpu().numpy(), s.cpu().numpy()) for op, (c, s) in arena.items()}


def capture(model: Model, arena: dict | None = None, keep: dict | None = None) -> torch.cuda.CUDAGraph:
    """The step's graph; ``keep`` ends up holding the graph's own quantize outputs, which every replay rewrites in place."""
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            model.forward(arena, keep)
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model.forward(arena, keep)
    return graph


def run(arm: str, model: Model, graphs: dict, ctx: bytes, key: bytes, first: int, steps: int) -> dict[str, Any]:
    """``steps`` steps of ``arm`` from engine step ``first``: each step's wall time (host sync to host sync) and, overlapped, each
    step's recorded roots and its graph-end-to-root latency."""
    serving = torch.cuda.current_stream()
    tok = torch.cuda.Event()
    pending: deque[D.Pending] = deque()
    recorded: list[RTB.StepRoots] = []
    hashed: dict[int, Any] = {}
    marks: list[tuple[Any, Any]] = []
    walls = []
    torch.cuda.synchronize()
    last = time.perf_counter()
    for s in range(first, first + steps):
        if arm == "plain":
            graphs["plain"].replay()
        else:
            if s - 2 in hashed:
                serving.wait_event(hashed.pop(s - 2))
            graphs[("copied" if arm == "copied" else "kept", s % 2)].replay()
            if arm == "overlapped":
                end = torch.cuda.Event(enable_timing=True)
                end.record(serving)
                p = D.step_roots(ctx, s, model.outputs(model.kept[s % 2]), key)
                root = torch.cuda.Event(enable_timing=True)
                root.record(D.side_stream())
                marks.append((end, root))
                hashed[s] = p.done
                pending.append(p)
        tok.record(serving)
        tok.synchronize()
        if arm == "overlapped":
            D.record_ready(pending, recorded.append)
        now = time.perf_counter()
        walls.append(now - last)
        last = now
    torch.cuda.synchronize()
    D.record_ready(pending, recorded.append)
    return {"walls": walls, "recorded": recorded, "latency_ms": [a.elapsed_time(b) for a, b in marks]}


def alone(model: Model, ctx: bytes, key: bytes, reps: int) -> dict[str, float]:
    """``step_roots`` of kept graph 0's step on an idle GPU: the kernels' time (events), and the host's, with the plan's part."""
    kernel, host, plan = [], [], []
    shapes = {op: (model.ids, int(c.shape[0]), int(c.shape[1])) for op, (c, _s) in model.kept[0].items()}
    for i in range(1 + reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        D.plan(ctx, 1000 + i, shapes, key)
        t1 = time.perf_counter()
        p = D.step_roots(ctx, 1000 + i, model.outputs(model.kept[0]), key, timing=True)
        t2 = time.perf_counter()
        p.result()
        if i:
            kernel.append(p.kernel_ms())
            host.append(1e3 * (t2 - t1))
            plan.append(1e3 * (t1 - t0))
    return {"kernel_ms": statistics.median(kernel), "host_ms": statistics.median(host), "plan_ms": statistics.median(plan),
            "kernel_ms_all": kernel}


def main(argv: list[str] | None = None) -> int:
    ns = parse(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    ctx, key = hashlib.sha512(b"step-root-gpu/context").digest(), hashlib.sha256(b"step-root-gpu/row-key").digest()
    model = Model(ns)
    graphs = {"plain": capture(model), **{("copied", i): capture(model, arena=model.arenas[i]) for i in range(2)},
              **{("kept", i): capture(model, keep=model.kept[i]) for i in range(2)}}
    if any(len(k) != len(model.ops) for k in model.kept) or {id(c) for c, _s in model.kept[0].values()} & \
            {id(c) for c, _s in model.kept[1].values()}:
        raise SystemExit("each kept graph holds its own quantize output per op")

    # Gate: a step of random rows, on the device, on the CPU twin and on the CPU reference.
    g = torch.Generator(device="cuda").manual_seed(2)
    for c, s in model.kept[0].values():
        c.copy_(torch.randint(-128, 128, c.shape, dtype=torch.int8, device="cuda", generator=g))
        s.copy_(torch.rand(s.shape, device="cuda", generator=g))
    host = model.host_outputs(model.kept[0])
    want = RTB.step_roots(ctx, 7, host, key)
    gate = {"device": D.step_roots(ctx, 7, model.outputs(model.kept[0]), key).result() == want,
            "twin": D.host_step(ctx, 7, host, key) == want}
    cold = alone(model, ctx, key, 3)
    del cold

    first = 1
    for arm in ARMS:
        run(arm, model, graphs, ctx, key, first, 5)
        first += 5
    per_arm: dict[str, list[float]] = {a: [] for a in ARMS}
    walls_all: dict[str, list[float]] = {a: [] for a in ARMS}
    latency: list[float] = []
    last_check = None
    for _ in range(ns.reps):
        for arm in ARMS:
            r = run(arm, model, graphs, ctx, key, first, ns.steps)
            per_arm[arm].append(statistics.median(r["walls"]))
            walls_all[arm] += r["walls"]
            if arm == "overlapped":
                latency += r["latency_ms"]
                if [x.engine_step for x in r["recorded"]] != list(range(first, first + ns.steps)):
                    raise SystemExit("the overlapped loop did not record every step in order")
                s_last = first + ns.steps - 1
                last_check = (s_last, r["recorded"][-1], model.host_outputs(model.kept[s_last % 2]))
            first += ns.steps
    s_last, got, outs = last_check
    gate["overlapped_last_step"] = got == RTB.step_roots(ctx, s_last, outs, key)
    solo = alone(model, ctx, key, 2 * ns.reps)
    step_ms = {a: 1e3 * statistics.median(v) for a, v in per_arm.items()}
    ok = all(gate.values())
    nbytes = sum(c.numel() + 4 * s.numel() for c, s in model.arenas[0].values())
    nvcc_v = subprocess.run([D.nvcc() or "nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    summary = {
        "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__,
        "cuda": torch.version.cuda, "nvcc": nvcc_v, "batch": ns.batch, "layers": ns.layers, "tables": len(model.ops),
        "rows": len(model.ops) * ns.batch, "bytes_hashed": nbytes, "steps": ns.steps, "reps": ns.reps, "gate": gate,
        "step_ms": step_ms, "step_ms_reps": {a: [1e3 * v for v in vs] for a, vs in per_arm.items()},
        "copied_pct": 100 * (step_ms["copied"] / step_ms["plain"] - 1),
        "kept_pct": 100 * (step_ms["kept"] / step_ms["plain"] - 1),
        "overlapped_pct": 100 * (step_ms["overlapped"] / step_ms["plain"] - 1),
        "overlapped_vs_kept_pct": 100 * (step_ms["overlapped"] / step_ms["kept"] - 1),
        "latency_ms": {"median": statistics.median(latency), "max": max(latency)},
        **{k: v for k, v in solo.items() if k != "kernel_ms_all"},
    }
    print(json.dumps(summary, indent=1))
    out = ns.out or os.environ.get("RESEARCH_RUN_DIR")
    if out:
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "step_root_gpu.json").write_text(json.dumps({**summary, "walls_s": walls_all, "kernel_ms_all": solo["kernel_ms_all"],
                                                          "latency_ms_all": latency}, indent=1) + "\n")
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
            "kind": "bench-result/v1",
            "workload_fingerprint": {"workload": "step_root_gpu", "model": "Llama-3.1-8B int8 decode linears (graphed proxy)",
                                     "linears": LINEARS, "batch": ns.batch, "layers": ns.layers, "steps": ns.steps, "reps": ns.reps,
                                     "implementation": "verity_vllm.commit.row_tables_device (pouw_cuda/row_table.cu, ncp2 SHA-512)",
                                     "gpu": summary["gpu"], "capability": summary["capability"], "torch": torch.__version__,
                                     "cuda": torch.version.cuda, "nvcc": nvcc_v},
            "validation": {"status": "passed" if ok else "failed",
                           "checker": "device and CPU-twin step roots equal row_tables.step_roots on the CPU (a random step, the "
                                      "last overlapped step from its graph's outputs)",
                           "detail": json.dumps(gate), "evidence": []},
            "measurements": [{"name": f"step_ms.{a}", "value": v, "unit": "ms"} for a, v in step_ms.items()]
            + [{"name": n, "value": summary[n], "unit": "%"} for n in ("copied_pct", "kept_pct", "overlapped_pct",
                                                                        "overlapped_vs_kept_pct")]
            + [{"name": n, "value": summary[n], "unit": "ms"} for n in ("kernel_ms", "host_ms", "plan_ms")]
            + [{"name": "latency_ms.median", "value": summary["latency_ms"]["median"], "unit": "ms"},
               {"name": "latency_ms.max", "value": summary["latency_ms"]["max"], "unit": "ms"},
               {"name": "bytes_hashed", "value": nbytes, "unit": "bytes"},
               {"name": "workload_wall", "value": time.perf_counter() - started, "unit": "s"}],
            "measurement_files": ["step_root_gpu.json"],
            "artifacts": [],
        }
        tmp = d / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        os.replace(tmp, d / "result.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
