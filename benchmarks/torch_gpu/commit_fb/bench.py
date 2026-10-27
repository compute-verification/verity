#!/usr/bin/env python3
r"""On one GPU: the fallback commit call's throughput at shape H's unit outputs (`commit_fb.shape_h`: d 16,384, 52 layers,
256 query heads of 64, one KV head, MLP 65,536, context 32) and its share of the step at the planned batch.

    bash benchmarks/torch_gpu/commit_fb/run.sh bench        # under gpu-lease, on the H200 (run.sh says how)

One layer's unit outputs (16 tables, 672,064 bytes a sequence) and the head's (4 tables) are made once at ``--batch``
sequences (random values; the hash's work doesn't depend on them) and committed through `commit_fb.commit_step_async`, so
`row_tables_device.step_roots` as it is.  Arms:

  layer    one layer as one segment on an idle GPU, ``--reps`` times: ``kernel_ms`` (``rt_step``'s four kernels, CUDA events),
           ``host_ms`` (the call's host time: layout, domains, rank order, staging) and ``GB/s`` = the layer's bytes over
           ``kernel_ms``; also at each ``--sweep`` batch (the first rows of the same tensors)
  step     a whole step as the loop commits it at this batch: 52 layer segments and the head's (`commit_fb.segment_of`), at
           most two in flight, the host recording each root as it lands; wall from the first launch to the last root.  Every
           op's tables, then without compute-accounting's fused cuts (`commit_fb.FUSED_OUT`)
  beside   the step's hashing beside a stream of int8 GEMMs of shape H's attention projection (M = batch, K = N = d,
           ``torch._int_mm``) on another stream: each GEMM's time alone and while the hashing runs, and the hashing's wall

``share``: the step arm's wall over compute-accounting's shape-H step at M = 20,480 (ALLP-GPU, 25 Oct 22:50Z, H2: 7.5 s with
P2 decode staged serially at 40 GB/s, 3.7 s with the decode hidden under the GEMMs).  Writes ``bench.json`` into ``--out``
(default ``$OUT``) and ``result.json`` into the run dir, whose validation is ``check_roots.py``'s when its ``roots.json`` is
in ``--out``.
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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import commit_fb as F  # noqa: E402
from check_roots import RESULT_SCHEMA, write_result  # noqa: E402
from verity_vllm.commit import row_tables_device as RTD  # noqa: E402

#: compute-accounting's shape-H step at M = 20,480 (s): P2 decode staged serially at 40 GB/s, and hidden under the GEMMs
STEP_S = {"staged": 7.5, "overlapped": 3.7}
#: compute-accounting's need from the RTX plan (GB/s): every op, and with the fused cuts
NEED_GBS = {"every_op": 134.0, "fused": 93.0}


def make(part: dict, prefix: str, batch: int, gen: torch.Generator) -> dict:
    out = {}
    for name, (kind, n) in part.items():
        if kind == "s8":
            out[prefix + name] = (torch.randint(-128, 128, (batch, n), dtype=torch.int8, device="cuda", generator=gen),
                                  torch.rand(batch, device="cuda", generator=gen))
        elif kind == "bf16":
            out[prefix + name] = torch.randint(-32768, 32768, (batch, n), dtype=torch.int16, device="cuda",
                                               generator=gen).view(torch.bfloat16)
        else:
            out[prefix + name] = torch.randint(0, 256, (batch,), dtype=torch.int32, device="cuda", generator=gen)
    return out


def nbytes(outs: dict) -> int:
    """Committed bytes: every row's codes and its four scale bytes."""
    return sum((v[0].numel() if isinstance(v, tuple) else v.numel() * v.element_size()) + 4 * (v[0] if isinstance(v, tuple)
               else v).shape[0] for v in outs.values())


def head_rows(outs: dict, b: int) -> dict:
    return {op: ((v[0][:b], v[1][:b]) if isinstance(v, tuple) else v[:b]) for op, v in outs.items()}


class Salts:
    def __init__(self) -> None:
        self.key = hashlib.sha256(b"commit-fb/bench/row-key").digest()
        self.context = hashlib.sha512(b"commit-fb/bench/context").digest()

    def __call__(self, segment: int) -> F.SaltSource:
        return F.SaltSource(self.key, segment, self.context)


def layer_arm(outs: dict, salts: Salts, reps: int, first: int) -> dict:
    rows = F.slot_rows(_batch(outs))
    kernel, host, wall = [], [], []
    for i in range(1 + reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        p, _ids, _ranks = F.commit_step_async(outs, salts(first + i), rows=rows, timing=True)
        t1 = time.perf_counter()
        p.result()
        t2 = time.perf_counter()
        if i:
            kernel.append(p.kernel_ms())
            host.append(1e3 * (t1 - t0))
            wall.append(1e3 * (t2 - t0))
    b = nbytes(outs)
    k = statistics.median(kernel)
    return {"bytes": b, "rows": len(rows) * len(outs), "kernel_ms": k, "host_ms": statistics.median(host),
            "wall_ms": statistics.median(wall), "GBps": b / k / 1e6, "kernel_ms_all": kernel}


def step_arm(parts: list[dict], salts: Salts, engine_step: int, inflight: int = 2) -> dict:
    """One step's segments as the loop commits them: at most ``inflight`` pending, each root recorded as it lands."""
    rows = F.slot_rows(_batch(parts[0]))
    pending: deque = deque()
    roots = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for part, outs in enumerate(parts):
        p, _ids, _ranks = F.commit_step_async(outs, salts(F.segment_of(engine_step, part, len(parts))), rows=rows, timing=True)
        pending.append(p)
        while len(pending) >= inflight or pending[0].ready():
            q = pending.popleft()
            roots.append((q.result().root, q.kernel_ms()))
            if not pending:
                break
    while pending:
        q = pending.popleft()
        roots.append((q.result().root, q.kernel_ms()))
    wall = time.perf_counter() - t0
    b = sum(nbytes(o) for o in parts)
    return {"segments": len(parts), "bytes": b, "wall_s": wall, "GBps": b / wall / 1e9,
            "kernel_s": sum(k for _r, k in roots) / 1e3, "roots": len({r for r, _k in roots})}


def _batch(outs: dict) -> int:
    v = next(iter(outs.values()))
    return int((v[0] if isinstance(v, tuple) else v).shape[0])


def beside_arm(parts: list[dict], salts: Salts, engine_step: int, batch: int, d: int, gen: torch.Generator) -> dict:
    """The step's hashing beside int8 GEMMs (M = batch, K = N = d) on their own stream: each GEMM's time alone and while the
    hashing runs (CUDA events between GEMMs), and the hashing's wall beside them."""
    a = torch.randint(-127, 128, (batch, d), dtype=torch.int8, device="cuda", generator=gen)
    w = torch.randint(-127, 128, (d, d), dtype=torch.int8, device="cuda", generator=gen).t()
    c = torch.empty(batch, d, dtype=torch.int32, device="cuda")
    gs = torch.cuda.Stream()

    def gemms(n: int) -> list:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
        with torch.cuda.stream(gs):
            ev[0].record(gs)
            for i in range(n):
                torch._int_mm(a, w, out=c)
                ev[i + 1].record(gs)
        return ev

    gemms(3)
    torch.cuda.synchronize()
    ev = gemms(20)
    torch.cuda.synchronize()
    alone = statistics.median(ev[i].elapsed_time(ev[i + 1]) for i in range(20))
    solo = step_arm(parts, salts, engine_step)
    n = int(solo["wall_s"] * 1e3 / alone * 1.6) + 8
    torch.cuda.synchronize()
    ev = gemms(n)
    s = step_arm(parts, salts, engine_step + 1)
    torch.cuda.synchronize()
    times = [ev[i].elapsed_time(ev[i + 1]) for i in range(n)]
    start = [ev[0].elapsed_time(ev[i]) for i in range(n)]
    during = [t for t, t0 in zip(times, start, strict=True) if t0 + t <= s["wall_s"] * 1e3] or times[:1]
    ops = 2 * batch * d * d
    return {"gemm": [batch, d, d], "gemm_ms_alone": alone, "gemm_tops_alone": ops / alone / 1e9,
            "gemm_ms_beside": statistics.median(during), "gemm_slowdown_pct": 100 * (statistics.median(during) / alone - 1),
            "gemms_beside": len(during), "hash_wall_s_alone": solo["wall_s"], "hash_wall_s_beside": s["wall_s"],
            "hash_slowdown_pct": 100 * (s["wall_s"] / solo["wall_s"] - 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=int, default=20480)
    ap.add_argument("--layers", type=int, default=52)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--sweep", default="660,4096", help="other batches for the layer arm (comma list; '' for none)")
    ap.add_argument("--no-beside", action="store_true")
    ap.add_argument("--out", default=os.environ.get("OUT") or ".")
    ns = ap.parse_args(argv)
    started = time.perf_counter()
    spec = F.shape_h()
    gen = torch.Generator(device="cuda").manual_seed(1)
    layer = make(spec["layer"], "model.layers.0.", ns.batch, gen)
    head = make(spec["head"], "model.head.", ns.batch, gen)
    fused = {op: v for op, v in layer.items() if op.removeprefix("model.layers.0.") not in F.FUSED_OUT}
    salts = Salts()
    torch.cuda.synchronize()
    print(f"[commit_fb] {torch.cuda.get_device_name()}: one layer {nbytes(layer) / 1e9:.2f} GB, head {nbytes(head) / 1e9:.3f} GB "
          f"at batch {ns.batch}", flush=True)

    res: dict = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                 "torch": torch.__version__, "cuda": torch.version.cuda, "batch": ns.batch, "layers": ns.layers,
                 "shape": {"d": 16384, "layers": 52, "heads": 256, "head_dim": 64, "kv_heads": 1, "mlp": 65536, "ctx": 32},
                 "nvcc": (subprocess.run([RTD.nvcc() or "nvcc", "--version"], capture_output=True, text=True).stdout
                          .strip().splitlines() or [""])[-1]}
    res["layer"] = layer_arm(layer, salts, ns.reps, 1_000_000)
    res["layer_fused"] = layer_arm(fused, salts, ns.reps, 2_000_000)
    res["head"] = layer_arm(head, salts, ns.reps, 3_000_000)
    print(json.dumps({k: {kk: vv for kk, vv in res[k].items() if kk != "kernel_ms_all"} for k in ("layer", "layer_fused", "head")}),
          flush=True)
    res["sweep"] = {b: {k: v for k, v in layer_arm(head_rows(layer, b), salts, ns.reps, 4_000_000 + 100 * b).items()
                        if k != "kernel_ms_all"} for b in [int(x) for x in ns.sweep.split(",") if x]}
    print(json.dumps({"sweep": res["sweep"]}), flush=True)
    every = [layer] * ns.layers + [head]
    res["step"] = step_arm(every, salts, 10)
    res["step_fused"] = step_arm([fused] * ns.layers + [head], salts, 11)
    print(json.dumps({"step": res["step"], "step_fused": res["step_fused"]}), flush=True)
    if not ns.no_beside:
        try:
            res["beside"] = beside_arm(every, salts, 20, ns.batch, 16384, gen)
        except (RuntimeError, TypeError) as e:
            res["beside_error"] = repr(e)
        print(json.dumps({"beside": res.get("beside", res.get("beside_error"))}), flush=True)
    res["share"] = {f"{arm}/{name}": res[arm]["wall_s"] / s for arm in ("step", "step_fused") for name, s in STEP_S.items()}
    res["need_GBps"] = {**{f"rtx_plan/{k}": v for k, v in NEED_GBS.items()},
                        **{f"shape_h_{arm}/{name}": res[arm]["bytes"] / s / 1e9 for arm in ("step", "step_fused")
                           for name, s in STEP_S.items()}}
    res["bytes_per_step_GB"] = {"every_op": res["step"]["bytes"] / 1e9, "fused": res["step_fused"]["bytes"] / 1e9}
    print(json.dumps({"share": res["share"], "need_GBps": res["need_GBps"]}), flush=True)

    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(res, indent=1) + "\n")
    roots = out / "roots.json"
    check = json.loads(roots.read_text()) if roots.exists() else None
    result = {
        "schema": RESULT_SCHEMA, "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "kind": "bench-result/v1",
        "workload_fingerprint": {"workload": "commit_fb/bench", "shape": res["shape"], "batch": ns.batch, "reps": ns.reps,
                                 "implementation": "commit_fb.commit_step_async -> verity_vllm.commit.row_tables_device.step_roots "
                                                   "(protocol_options/pouw_cuda/row_table.cu)",
                                 "gpu": res["gpu"], "capability": res["capability"], "torch": res["torch"], "nvcc": res["nvcc"]},
        "validation": {"status": ("passed" if check["ok"] else "failed") if check else "not_validated",
                       "checker": "check_roots.py: device roots equal the CPU references" if check else "no roots.json beside",
                       "detail": json.dumps({c["case"]: c["equal"] for c in check["cases"]}) if check else "", "evidence": []},
        "measurements": [{"name": "layer.GBps", "value": res["layer"]["GBps"], "unit": "GB/s"},
                         {"name": "layer.kernel_ms", "value": res["layer"]["kernel_ms"], "unit": "ms"},
                         {"name": "layer.host_ms", "value": res["layer"]["host_ms"], "unit": "ms"},
                         {"name": "step.wall_s", "value": res["step"]["wall_s"], "unit": "s"},
                         {"name": "step.GBps", "value": res["step"]["GBps"], "unit": "GB/s"},
                         {"name": "step_fused.wall_s", "value": res["step_fused"]["wall_s"], "unit": "s"},
                         {"name": "step_fused.GBps", "value": res["step_fused"]["GBps"], "unit": "GB/s"}]
        + [{"name": f"share.{k}", "value": v, "unit": "fraction"} for k, v in res["share"].items()]
        + ([{"name": f"beside.{k}", "value": res["beside"][k], "unit": u} for k, u in
            (("gemm_slowdown_pct", "%"), ("hash_slowdown_pct", "%"), ("gemm_tops_alone", "TOPS"))] if "beside" in res else [])
        + [{"name": "workload_wall", "value": time.perf_counter() - started, "unit": "s"}],
        "measurement_files": ["out/bench.json"] + (["out/roots.json"] if check else []), "artifacts": []}
    write_result(result, out)
    return 0 if check is None or check["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
