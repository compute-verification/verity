"""On a CUDA device: the GPU call's throughput at the frozen H200 shape (README.md), as JSON to ``--out``/bench.json.

1. **Alone.** A whole step, the 52 layers' outputs (resident, random) committed at each of ``--slots``: seconds a step, GB/s
   of committed bytes, the split (the engine's rows and leaves, the units' slot-tree leaves, the slot trees, the step
   tree), and its share of a serving step whose other work (GEMMs, attention, elementwise) takes each of ``--rest-s``
   seconds at ``--step-slots`` slots, scaled with the slot count: commit / (rest + commit). At each of ``--group-slots``
   also with ``--group-layers`` layers a group (``slots_gpu.regroup``: one launch for several layers, the same roots).
2. **Beside the GEMMs.** At the first slot count, a layer's four int8 GEMMs at M = slots (cuBLASLt through
   ``torch._int_mm``: QKV, o_proj, up, down at the frozen widths) alone, the layer's commit alone, and both at once (the
   commit on the call's side stream): the GEMMs' slowdown, and a step estimate from the measured GEMMs.

Times are CUDA events on the streams that ran the work; clocks are read, never set.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import slots as S  # noqa: E402
import slots_gpu as G  # noqa: E402
from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402

CTX = identity_digest_sha512("verity/commit-slots/bench/context/v1", {"bench": 1})
KEY = bytes(range(64, 96))
D, F, QKV = 16384, 65536, 16384 + 2 * 64
GEMMS = (("qkv", D, QKV), ("o_proj", D, D), ("up", D, F), ("down", F, D))   # (name, K, N)


def _ms(a: Any, b: Any) -> float:
    return a.elapsed_time(b)


def data(layout: S.CommitShape, slots: int) -> list[torch.Tensor]:
    return [torch.empty((slots, n), dtype=torch.uint8, device="cuda").random_() for n in layout.groups[0].sources]


def step(plan: G.Plan, bufs: list[torch.Tensor], slots: int, layers: int) -> dict[str, Any]:
    """One step of ``layers`` adds of ``bufs``: seconds and the kernels' split, from events on the side stream."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    start.record(plan.stream)
    marks: list[tuple[str, Any]] = []
    sc = G.StepCommitter(plan, 3, slots, KEY, marks=marks)
    for _ in range(layers):
        sc.add(bufs)
    sc.finish().result()
    split: dict[str, float] = {}
    prev = start
    for name, ev in marks:
        split[name] = split.get(name, 0.0) + _ms(prev, ev) / 1e3
        prev = ev
    return {"s": _ms(start, marks[-1][1]) / 1e3, "split_s": split}


def alone(layout: S.CommitShape, slots: int, reps: int, rest_s: list[float], step_slots: int, per_group: int = 1) -> dict[str, Any]:
    layers = len(layout.groups)
    layout = G.regroup(layout, per_group)
    plan = G.Plan(CTX, layout)
    bufs = data(layout, slots)
    step(plan, bufs, slots, len(layout.groups))
    runs = [step(plan, bufs, slots, len(layout.groups)) for _ in range(reps)]
    best = min(runs, key=lambda r: r["s"])
    nbytes = slots * layout.slot_bytes
    blocks = slots * sum((r.nbytes + 17 + 127) // 128 for u in layout.units for r in u)
    del bufs
    torch.cuda.empty_cache()
    return {"slots": slots, "layers": layers, "layers_per_group": per_group, "committed_bytes": nbytes,
            "rows": slots * layout.row_count,
            "units": slots * layout.unit_count, "s": [r["s"] for r in runs], "best_s": best["s"], "split_s": best["split_s"],
            "GB_per_s": nbytes / best["s"] / 1e9, "row_compressions_per_s": blocks / best["s"],
            "share_of_step": {str(t): best["s"] / (t * slots / step_slots + best["s"]) for t in rest_s}}


def beside(layout: S.CommitShape, slots: int, reps: int) -> dict[str, Any]:
    """A layer's GEMMs alone, its commit alone, and both at once (GEMMs on the current stream, the commit on the side)."""
    plan = G.Plan(CTX, layout)
    bufs = data(layout, slots)
    x = torch.randint(-128, 128, (slots, D), dtype=torch.int8, device="cuda")
    h = torch.randint(-128, 128, (slots, F), dtype=torch.int8, device="cuda")
    ws = {name: torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda").t() for name, k, n in GEMMS}

    def gemms() -> None:
        for name, k, _ in GEMMS:
            torch._int_mm(h if k == F else x, ws[name])

    main = torch.cuda.current_stream()

    def timed(commit: bool, gemm: bool) -> dict[str, float]:
        torch.cuda.synchronize()
        e0, eg, ec = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        e0.record(main)
        if commit:
            sc = G.StepCommitter(plan, 5, slots, KEY)
            sc.add(bufs)
            ec.record(plan.stream)
        if gemm:
            gemms()
            eg.record(main)
        torch.cuda.synchronize()
        r = {}
        if gemm:
            r["gemm_s"] = _ms(e0, eg) / 1e3
        if commit:
            r["commit_s"] = _ms(e0, ec) / 1e3
        r["both_s"] = max(r.values())
        return r

    for c, g in ((True, False), (False, True), (True, True)):
        timed(c, g)
    res = {}
    for label, c, g in (("gemm_alone", False, True), ("commit_alone", True, False), ("together", True, True)):
        runs = [timed(c, g) for _ in range(reps)]
        res[label] = min(runs, key=lambda r: r["both_s"])
    macs = slots * sum(k * n for _, k, n in GEMMS)
    g0, c0, both = res["gemm_alone"]["gemm_s"], res["commit_alone"]["commit_s"], res["together"]
    layers = 52
    return {"slots": slots, "gemms": [list(g) for g in GEMMS], "macs_per_layer": macs, **res,
            "gemm_int8_tops": 2 * macs / g0 / 1e12, "gemm_slowdown": both["gemm_s"] / g0 - 1,
            "layer_overlapped_s": both["both_s"], "layer_serial_s": g0 + c0,
            "step_gemm_s": layers * g0, "step_commit_s": layers * c0, "step_overlapped_s": layers * both["both_s"],
            "commit_share_of_gemm_step": c0 / g0,
            "overlap_cost_share": (both["both_s"] - g0) / both["both_s"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slots", type=int, nargs="+", default=[16384, 4096, 12288, 20480, 1024])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rest-s", type=float, nargs="+", default=[5.33],
                    help="a serving step's time without the commit at --step-slots, scaled with the slots (a step's tokens); "
                         "5.33 s is compute-accounting's block at 16,384 (GEMMs 5.08 s, attention and elementwise 0.25 s)")
    ap.add_argument("--step-slots", type=int, default=16384)
    ap.add_argument("--group-slots", type=int, nargs="*", default=[4096], help="slot counts to time again with layers grouped")
    ap.add_argument("--group-layers", type=int, default=4)
    ap.add_argument("--no-beside", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    layout = S.h200_layout()
    props = torch.cuda.get_device_properties(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,power.limit",
                          "--format=csv,noheader", "-i", "0"], capture_output=True, text=True).stdout.strip()
    doc: dict[str, Any] = {"device": props.name, "capability": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
                           "torch": torch.__version__, "nvidia_smi": smi, "layout": layout.name,
                           "slot_bytes": layout.slot_bytes, "alone": []}
    runs = [(slots, 1) for slots in a.slots] + [(slots, a.group_layers) for slots in a.group_slots]
    for slots, per_group in runs:
        r = alone(layout, slots, a.reps, a.rest_s, a.step_slots, per_group)
        doc["alone"].append(r)
        print(json.dumps(r), flush=True)
    if not a.no_beside:
        doc["beside"] = beside(layout, a.slots[0], a.reps)
        print(json.dumps(doc["beside"]), flush=True)
    (a.out / "bench.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
