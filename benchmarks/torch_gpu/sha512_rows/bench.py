"""On a CUDA device: the engine's throughput at the frozen H200 shape's per-step rows, as JSON to ``--out``/bench.json.

1. **Variants.** One layer's rows (``h200_rows``: 25 rows, 663,824 bytes a slot) at ``--sweep-slots``, the row kernel in
   every compression variant: GB/s of row bytes and row compressions a second, the SM clock and power after each; then
   that layer's leaves in both leaf kernels (1 and 2 CTAs an SM), leaves a second.
2. **Steps.** A whole step at each of ``--slots`` and ``--layers-per-call`` k (52 / k calls, each the rows and leaves of k
   layers, their tensors reused by every call): every row's digest and hm96 leaf (salts from a key), each slot's tree over
   its 1,300 one-row units' roots (frame leaves, pads, 11 levels), and the step's tree over the slots' roots. Seconds a step,
   GB/s of committed bytes, SHA-512 compressions a second (a leaf's 5; its 3 ChaCha20 chunks aren't counted) and the split.

The layout is commit_slots' ``h200_layout`` (benchmarks/torch_gpu/commit_slots/slots.py), derived again here from
the frozen shape: every op's per-token output, one-row units of at most 32 KiB. Times are CUDA events; clocks are read, never
set.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine as E

from verity.ml.commitments.identity import identity_digest_sha512

CAP = 32768
LAYERS = 52
KEY = bytes(range(64, 96))


def _cut(src: int, n: int) -> list[list[tuple[int, int, int]]]:
    return [[(src, off, min(CAP, n - off))] for off in range(0, n, CAP)]


def h200_rows(d: int = 16384, heads: int = 256, head_dim: int = 64, kv_heads: int = 1, mlp: int = 65536,
              ctx_rows: int = 16) -> tuple[list[int], list[list[tuple[int, int, int]]]]:
    """One layer's tensors (bytes a slot's row) and rows (parts), commit_slots' h200_layout: attention RMSNorm, its s8
    quantizer (codes, then the f32 scale), QKV, scores (f32 heads x ctx_rows), softmax, PV, o_proj's quantizer and output,
    the residual, the MLP's RMSNorm and quantizer, up, ReLU, down's quantizer, down and the residual."""
    qkv = 2 * (heads * head_dim + 2 * kv_heads * head_dim)
    ops = [("bf16", 2 * d), ("quant", d), ("bf16", qkv), ("f32", 4 * heads * ctx_rows), ("bf16", 2 * heads * ctx_rows),
           ("bf16", 2 * d), ("quant", d), ("bf16", 2 * d), ("bf16", 2 * d), ("bf16", 2 * d), ("quant", d), ("bf16", 2 * mlp),
           ("bf16", 2 * mlp), ("quant", mlp), ("bf16", 2 * d), ("bf16", 2 * d)]
    sources: list[int] = []
    rows: list[list[tuple[int, int, int]]] = []
    for kind, n in ops:
        src = len(sources)
        if kind == "quant":
            sources += [n, 4]
            rows += [[(src, 0, n), (src + 1, 0, 4)]] if n + 4 <= CAP else _cut(src, n) + [[(src + 1, 0, 4)]]
        else:
            sources.append(n)
            rows += _cut(src, n)
    return sources, rows


def blocks(nbytes: int) -> int:
    return (nbytes + 17 + 127) // 128


class Clock:
    def __init__(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        except Exception as e:  # noqa: BLE001
            self.nv, self.missing = None, repr(e)

    def read(self) -> dict[str, Any]:
        if self.nv is None:
            return {"missing": self.missing}
        nv = self.nv
        return {"sm_mhz": nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM), "power_w": nv.nvmlDeviceGetPowerUsage(self.h) / 1e3,
                "temp_c": nv.nvmlDeviceGetTemperature(self.h, nv.NVML_TEMPERATURE_GPU)}


def _ms(a: Any, b: Any) -> float:
    return a.elapsed_time(b)


def layer(slots: int, layers: int = 1) -> tuple[list[torch.Tensor], list[list[tuple[int, int, int]]]]:
    """``layers`` layers' tensors (each its own) and their rows, layer by layer."""
    sources, rows = h200_rows()
    tensors: list[torch.Tensor] = []
    out: list[list[tuple[int, int, int]]] = []
    for _ in range(layers):
        base = len(tensors)
        tensors += [torch.empty((slots, n), dtype=torch.uint8, device="cuda").random_() for n in sources]
        out += [[(base + src, off, n) for src, off, n in parts] for parts in rows]
    return tensors, out


def sweep(engine: E.Engine, slots: int, reps: int, clock: Clock, variants: list[str]) -> list[dict[str, Any]]:
    tensors, rows = layer(slots)
    table = engine.rows(rows, tensors)
    nbytes = slots * sum(table.nbytes)
    comps = slots * sum(blocks(n) for n in table.nbytes)
    out = engine.row_digests(table, variants[0])
    res = []
    for v in variants:
        engine.row_digests(table, v, out)
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            engine.row_digests(table, v, out)
            b.record()
            b.synchronize()
            ts.append(_ms(a, b) / 1e3)
        best = min(ts)
        r = {"variant": v, "slots": slots, "row_bytes": nbytes, "s": ts, "GB_per_s": nbytes / best / 1e9,
             "Gcomp_per_s": comps / best / 1e9, **clock.read()}
        print(json.dumps(r), flush=True)
        res.append(r)
    G = len(rows)
    dids = [identity_digest_sha512("verity/sha512-rows/bench/unit/v1", {"unit": g}) for g in range(G)]
    lt = engine.leaf_table(dids, list(range(G)), [0] * G, None, list(range(G)))
    leaves = torch.empty_like(out)
    for minb in (1, 2):
        engine.leaves(lt, out, E.SaltKey(KEY), leaves, minb)
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            engine.leaves(lt, out, E.SaltKey(KEY), leaves, minb)
            b.record()
            b.synchronize()
            ts.append(_ms(a, b) / 1e3)
        r = {"leaf_minb": minb, "slots": slots, "leaves": G * slots, "s": ts, "Gleaves_per_s": G * slots / min(ts) / 1e9,
             **clock.read()}
        print(json.dumps(r), flush=True)
        res.append(r)
    del tensors
    torch.cuda.empty_cache()
    return res


def step(engine: E.Engine, slots: int, reps: int, variant: str, clock: Clock, per_call: int = 1, minb: int = 0) -> dict[str, Any]:
    """One step: LAYERS // per_call calls, each the rows and leaves of ``per_call`` layers (their own tensors, reused by
    every call), then the slot trees and the step tree."""
    tensors, rows = layer(slots, per_call)
    table = engine.rows(rows, tensors)
    G = len(rows)
    calls = LAYERS // per_call
    U = G * calls
    unit_dids = [identity_digest_sha512("verity/sha512-rows/bench/unit/v1", {"unit": u}) for u in range(U)]
    slot_did = identity_digest_sha512("verity/sha512-rows/bench/slot/v1", {"units": U})
    step_did = identity_digest_sha512("verity/sha512-rows/bench/step/v1", {"slots": slots})
    lts = [engine.leaf_table(unit_dids, list(range(G * i, G * (i + 1))), [0] * G, None, list(range(G * i, G * (i + 1))))
           for i in range(calls)]
    units = torch.empty((U, slots, 8), dtype=torch.int64, device="cuda")
    x = torch.empty((G, slots, 8), dtype=torch.int64, device="cuda")
    salts = E.SaltKey(KEY)

    def once() -> dict[str, Any]:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 * calls + 4)]
        ev[0].record()
        for i in range(calls):
            engine.row_digests(table, variant, x)
            ev[1 + 2 * i].record()
            engine.leaves(lts[i], x, salts, units[G * i:G * (i + 1)], minb)
            ev[2 + 2 * i].record()
        per_slot = units.permute(1, 0, 2).contiguous()
        roots = engine.root(engine.frame_leaves(per_slot, [slot_did]), [slot_did])
        ev[-3].record()
        top = engine.root(engine.frame_leaves(roots.reshape(1, slots, 8), [step_did]), [step_did])
        ev[-2].record()
        ev[-1].record()
        ev[-1].synchronize()
        rows_s = sum(_ms(ev[2 * i], ev[1 + 2 * i]) for i in range(calls)) / 1e3
        leaves_s = sum(_ms(ev[1 + 2 * i], ev[2 + 2 * i]) for i in range(calls)) / 1e3
        return {"s": _ms(ev[0], ev[-2]) / 1e3, "split_s": {"rows": rows_s, "leaves": leaves_s,
                "slot_trees": _ms(ev[2 * calls], ev[-3]) / 1e3, "step_tree": _ms(ev[-3], ev[-2]) / 1e3},
                "root": E.digests(top)[0].hex()}

    once()
    runs = [once() for _ in range(reps)]
    best = min(runs, key=lambda r: r["s"])
    nbytes = slots * sum(table.nbytes) * calls
    row_comps = slots * sum(blocks(n) for n in table.nbytes) * calls
    leaf_comps = slots * U * 5
    width = 1 << (U - 1).bit_length()
    tree_comps = slots * (2 * U + 2 * (width - 1)) + 4 * slots
    r = {"slots": slots, "variant": variant, "leaf_minb": minb, "layers": LAYERS, "layers_per_call": per_call, "rows_a_slot": U,
         "committed_bytes": nbytes,
         "s": [q["s"] for q in runs], "best_s": best["s"], "split_s": best["split_s"], "GB_per_s": nbytes / best["s"] / 1e9,
         "row_GB_per_s": nbytes / best["split_s"]["rows"] / 1e9,
         "compressions": {"rows": row_comps, "leaves": leaf_comps, "trees": tree_comps},
         "Gcomp_per_s": (row_comps + leaf_comps + tree_comps) / best["s"] / 1e9, "root": best["root"], **clock.read()}
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slots", type=int, nargs="*", default=[4096, 16384])
    ap.add_argument("--sweep-slots", type=int, default=4096)
    ap.add_argument("--variants", nargs="*", default=None, help="the sweep's variants (default: every built one)")
    ap.add_argument("--step-variant", default=None, help="the steps' row kernel (default: the sweep's fastest)")
    ap.add_argument("--layers-per-call", type=int, nargs="*", default=[1, 4], help="layers a row and leaf call (divides 52)")
    ap.add_argument("--leaf-minb", type=int, choices=(1, 2), default=None, help="the steps' leaf kernel (default: the faster)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--check", type=Path, default=None, help="check.json: time steps only on a variant it found bit-exact")
    a = ap.parse_args(argv)
    if any(k < 1 or LAYERS % k for k in a.layers_per_call):
        ap.error(f"--layers-per-call divides {LAYERS}")
    a.out.mkdir(parents=True, exist_ok=True)
    engine = E.Engine()
    exact = None
    if a.check is not None:
        c = json.loads(a.check.read_text()) if a.check.exists() else {}
        exact = {v for v, bad in c.get("rows", {}).get("mismatches", {}).items() if bad == 0}
        trees = c.get("trees", {}).get("mismatches")
        if c.get("leaves", {}).get("mismatches") != {"salt_key": 0, "salts": 0} or not trees or any(trees.values()):
            exact = set()
    clock = Clock()
    props = torch.cuda.get_device_properties(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,power.limit",
                          "--format=csv,noheader", "-i", "0"], capture_output=True, text=True, check=False).stdout.strip()
    doc: dict[str, Any] = {"device": props.name, "capability": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
                           "torch": torch.__version__, "nvidia_smi": smi, "default": engine.default, "lib": str(engine.lib.path),
                           "layer_bytes_a_slot": sum(n for p in h200_rows()[1] for _, _, n in p)}
    t0 = time.time()
    swept = sweep(engine, a.sweep_slots, a.reps, clock, a.variants or engine.variants) if a.sweep_slots else []
    doc["sweep"] = [r for r in swept if "variant" in r]
    doc["leaf_sweep"] = [r for r in swept if "leaf_minb" in r]
    for r in doc["sweep"]:
        r["bit_exact"] = None if exact is None else r["variant"] in exact
    ok = [r for r in doc["sweep"] if r["bit_exact"] is not False]
    fastest = max(ok, key=lambda r: r["GB_per_s"])["variant"] if ok else engine.default
    doc["step_variant"] = a.step_variant or fastest
    doc["step_leaf_minb"] = a.leaf_minb or (max(doc["leaf_sweep"], key=lambda r: r["Gleaves_per_s"])["leaf_minb"]
                                            if doc["leaf_sweep"] else 0)
    if exact is not None and doc["step_variant"] not in exact:
        doc["steps_skipped"] = f"no bit-exact variant to time (check: {sorted(exact)})"
        a.slots = []
    doc["steps"] = []
    for slots in a.slots:
        for k in a.layers_per_call:
            r = step(engine, slots, a.reps, doc["step_variant"], clock, k, doc["step_leaf_minb"])
            torch.cuda.empty_cache()
            print(json.dumps(r), flush=True)
            doc["steps"].append(r)
    doc["seconds"] = time.time() - t0
    (a.out / "bench.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
