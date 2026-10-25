"""Serve-time commitment at the large model's batch: one layer's unit outputs, each token row one hm96-sha512 hiding leaf in a
frame-v3-sha512 row table, hashed by the vLLM commit path's GPU row tables (`verity_vllm.commit.row_tables_device.step_roots`,
on its side stream), against the layer's exact linears at the same M on one GPU.

The unit sets, per token row of one layer (d the model width; the kernels hash a row's bytes and a 4-byte word after them):
    quantize   the served commit's tables (`commit.row_tables`): the four W8A8 linears' s8 inputs, d, d, d and 4d codes, each
               with its f32 token scale
    every      every unit's output (`UNITS`): the two norms, the four quantizers (codes and scale), the four linears' bf16
               outputs, attention's output, the ReLU and the two residual adds, 43 d bytes; a bf16 row is hashed as its bytes
               with a zero word, so this measures the throughput of those bytes, not a unit's row format
Attention's internals (scores and probabilities, heads x context bf16 a row each) are left out of every unless --contexts
names a context length; the final norm and the head are left out of the layer.

Per d, set, context and M:
    check      at M in --check-ms, the device's roots equal the CPU twin's (`host_step`) for one layer, and at the smallest
               for the whole model's step; at d = 64 the device, the twin and the Python reference (`row_tables.step_roots`)
               agree
    commit     the kernels' time on the side stream (`Pending.kernel_ms`), and the host's time in `step_roots` (plan, domain
               ids, staging); bytes/s of the rows' values (B (K + 4) summed over tables); the roots equal rep to rep
    linears    the layer's four exact linears (`int8_fused.scaled_mm` at `run.cfg_for`, gated against `run.reference` on the
               first call) as one CUDA graph
    overlap    the serving order: a layer's tables hash while the next layer's linears run. One replay of the linears (more
               where the host's launch is longer) hides `step_roots`' host work; from that replay's end, the commit and a
               second replay run together: their joint time against the replay alone, and the gap before the replay starts
The three items alternate rep by rep (order rotated), the SM clock read around each.

share = commit / linears in one layer. Its denominator is only the linears (norms, quantizers and attention are left out), so
it overstates the commit's share of a whole step. The whole model: L = round(80e9 / 12 d^2) layers of int8 weights, and
    whole_extrapolated   L x one layer's commit and linears (EXTRAPOLATED from one layer)
    whole_measured       one `step_roots` call over all L layers' tables (the step the serving loop launches), the layers'
                         buffers cycled over as many distinct copies as --whole-gb holds (`copies`)

    python commit.py --out OUT [--ds 8192,12288] [--ms 64,128,256,512] [--sets quantize,every] [--contexts 0] [--reps 15]
Writes commit.jsonl (a row per d, set, context and M) and commit.json; exits 4 when a check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "hash" / "exactlin"))
import fixture as FX  # noqa: E402
import kbench  # noqa: E402
import run as R  # noqa: E402
from verity_vllm.commit import row_tables as RTB  # noqa: E402
from verity_vllm.commit import row_tables_device as D  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.engine import int8_fused as X  # noqa: E402

CTX, KEY = bytes(range(64)), bytes(range(100, 132))
MODEL_BYTES = 80e9
SETS = ("quantize", "every")
#: one layer's units: (op under ``model.layers.L.``, elements a token row in units of d, bytes an element, a scale word)
UNITS = (
    ("input_layernorm", 1, 2, False),
    ("self_attn.qkv_proj.quant", 1, 1, True),
    ("self_attn.qkv_proj", 3, 2, False),
    ("self_attn.attn", 1, 2, False),
    ("self_attn.o_proj.quant", 1, 1, True),
    ("self_attn.o_proj", 1, 2, False),
    ("self_attn.residual", 1, 2, False),
    ("post_attention_layernorm", 1, 2, False),
    ("mlp.up_proj.quant", 1, 1, True),
    ("mlp.up_proj", 4, 2, False),
    ("mlp.act", 4, 2, False),
    ("mlp.down_proj.quant", 4, 1, True),
    ("mlp.down_proj", 1, 2, False),
    ("mlp.residual", 1, 2, False),
)
#: the linears' inputs: each linear of `fixture.linears` reads its quantizer's codes and scales
LINEAR_INPUT = {"qkv": "self_attn.qkv_proj.quant", "o": "self_attn.o_proj.quant", "up": "mlp.up_proj.quant",
                "down": "mlp.down_proj.quant"}


def layers_of(d: int) -> int:
    return round(MODEL_BYTES / (12 * d * d))


def unit_rows(d: int, unit_set: str, context: int, head_dim: int) -> list[tuple[str, int, bool]]:
    """(op, bytes a token row, a scale word) of one layer's tables."""
    rows = [(op, e * d * b, s) for op, e, b, s in UNITS if unit_set == "every" or op in LINEAR_INPUT.values()]
    if context and unit_set == "every":
        heads = max(1, d // head_dim)
        rows += [("self_attn.scores", 2 * heads * context, False), ("self_attn.probs", 2 * heads * context, False)]
    return rows


class Layer:
    """One layer's tables at M token rows (codes random, a quantizer's scales random, other words zero), on the GPU."""

    def __init__(self, units: list[tuple[str, int, bool]], m: int, seed: int):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.ids = [(f"r{i}", 0) for i in range(m)]
        self.tabs = {}
        for op, k, scaled in units:
            c = torch.randint(-128, 128, (m, k), generator=g, dtype=torch.int16, device="cuda").to(torch.int8)
            s = (torch.rand(m, generator=g, device="cuda") * 2.0**-7 if scaled else torch.zeros(m, device="cuda"))
            self.tabs[op] = (c, s.to(torch.float32).contiguous())
        self.value_bytes = sum(c.numel() + 4 * m for c, _ in self.tabs.values())

    def outputs(self, layer: int) -> dict:
        return {f"model.layers.{layer}.{op}": (self.ids, c, s) for op, (c, s) in self.tabs.items()}


def host_copy(outputs: dict) -> dict:
    return {op: (ids, c.cpu().numpy(), s.cpu().numpy()) for op, (ids, c, s) in outputs.items()}


def tiny_check() -> dict:
    """At d = 64, M = 13 (rows out of rank order: r10 sorts before r2) and a context of 256: device == twin == reference."""
    doc = {}
    for unit_set in SETS:
        lay = Layer(unit_rows(64, unit_set, 256, 128), 13, seed=5)
        outs = {**lay.outputs(0), **lay.outputs(1)}
        dev = D.step_roots(CTX, 2**33 + 7, outs, KEY).result()
        host = host_copy(outs)
        twin = D.host_step(CTX, 2**33 + 7, host, KEY)
        ref = RTB.step_roots(CTX, 2**33 + 7, host, KEY)
        doc[unit_set] = {"tables": len(outs), "device_is_twin": dev == twin, "twin_is_reference": twin == ref}
    return doc


class Linears:
    """The layer's four exact linears at M (`fixture.linears(d)`, weights without special scales), one CUDA graph."""

    def __init__(self, d: int, m: int):
        self.work = X.Workspace("cuda")
        self.calls, self.gate = [], {}
        for i, (name, (n, k)) in enumerate(FX.linears(d).items()):
            wt = R.Weights(n, k, seed=3000 + i, special=False)
            x = R.codes(m, k, 41 + i, {})
            sa = torch.from_numpy(FX.act_scales(m, 41 + i, special=False)).cuda()
            y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            cfg = R.cfg_for(m, n, k)
            R.poison(y)
            X.scaled_mm(x, sa, wt.w, wt.sw, y, cfg, self.work)
            torch.cuda.synchronize()
            self.gate[name] = {"cfg": str(cfg), "mismatches": R.diff(y, R.reference(E.int_mm(x, wt.w), sa, wt.sw))}
            self.calls.append((x, sa, wt, y, cfg))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._f()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._f()
        for _ in range(3):
            self.graph.replay()
        torch.cuda.synchronize()

    def _f(self):
        for x, sa, wt, y, cfg in self.calls:
            X.scaled_mm(x, sa, wt.w, wt.sw, y, cfg, self.work)

    @property
    def ok(self) -> bool:
        return all(g["mismatches"] == 0 for g in self.gate.values())


def _ev() -> torch.cuda.Event:
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def commit_once(outputs: dict, step: int) -> tuple[float, float, RTB.StepRoots]:
    t0 = time.perf_counter()
    p = D.step_roots(CTX, step, outputs, KEY, timing=True)
    host_us = (time.perf_counter() - t0) * 1e6
    roots = p.result()
    return p.kernel_ms() * 1e3, host_us, roots


def linears_once(lin: Linears) -> float:
    e0 = _ev()
    lin.graph.replay()
    e1 = _ev()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3


def overlap_once(lin: Linears, outputs: dict, step: int, pre: int) -> dict:
    for _ in range(pre):
        lin.graph.replay()
    e0 = _ev()
    p = D.step_roots(CTX, step, outputs, KEY, timing=True)
    eb = _ev()
    lin.graph.replay()
    e2 = _ev()
    torch.cuda.current_stream().wait_event(p.done)
    e1 = _ev()
    torch.cuda.synchronize()
    return {"us": e0.elapsed_time(e1) * 1e3, "gap_us": e0.elapsed_time(eb) * 1e3, "linears_us": eb.elapsed_time(e2) * 1e3,
            "commit_us": p.kernel_ms() * 1e3, "roots": p.result()}


def measure(d: int, unit_set: str, context: int, m: int, a, clock, step: int) -> dict:
    units = unit_rows(d, unit_set, context, a.head_dim)
    lay = Layer(units, m, seed=d + m)
    outs = lay.outputs(0)
    lin = Linears(d, m)
    row = {"d": d, "set": unit_set, "context": context, "m": m, "tables": len(outs), "value_bytes": lay.value_bytes,
           "row_bytes": {op: k + 4 for op, k, _ in units}, "linear_gate": lin.gate}
    if not lin.ok:
        row["failed"] = "a linear's words differ from the reference"
        return row
    if m in a.check_ms:
        row["check_layer"] = D.step_roots(CTX, step, outs, KEY).result() == D.host_step(CTX, step, host_copy(outs), KEY)
    _, host_us, first = commit_once(outs, step)
    t_lin = linears_once(lin)
    pre = max(1, math.ceil(1.5 * host_us / max(t_lin, 1.0)))
    got = {"commit": [], "host": [], "linears": [], "overlap": [], "gap": [], "overlap_linears": [], "overlap_commit": []}
    clocks, stable = {"commit": [], "linears": [], "overlap": []}, True
    items = ("commit", "linears", "overlap")
    for r in range(a.reps):
        for j in range(len(items)):
            item = items[(j + r) % len(items)]
            c0 = clock()
            if item == "commit":
                us, h, roots = commit_once(outs, step)
                got["commit"].append(us)
                got["host"].append(h)
                stable &= roots == first
            elif item == "linears":
                got["linears"].append(linears_once(lin))
            else:
                o = overlap_once(lin, outs, step, pre)
                for key, src in (("overlap", "us"), ("gap", "gap_us"), ("overlap_linears", "linears_us"),
                                 ("overlap_commit", "commit_us")):
                    got[key].append(o[src])
                stable &= o["roots"] == first
            clocks[item].append([c0, clock()])
    med = {k: statistics.median(v) for k, v in got.items()}
    n_layers = layers_of(d)
    row.update({
        "roots_stable": stable, "pre_replays": pre, "us": med, "us_all": {k: [round(v, 2) for v in vs] for k, vs in got.items()},
        "sm_clock_mhz": clocks, "commit_gbs": lay.value_bytes / med["commit"] / 1e3, "share": med["commit"] / med["linears"],
        "overlap_slowdown": med["overlap"] / med["linears"] - 1, "layers": n_layers,
        "whole_extrapolated": {"label": "EXTRAPOLATED: L x one layer", "commit_ms": n_layers * med["commit"] / 1e3,
                               "linears_ms": n_layers * med["linears"] / 1e3, "overlap_ms": n_layers * med["overlap"] / 1e3,
                               "value_bytes": n_layers * lay.value_bytes}})
    del lin
    torch.cuda.empty_cache()
    if a.whole:
        row["whole_measured"] = whole(lay, units, m, n_layers, a, step + 1)
    return row


def whole(lay: Layer, units, m: int, n_layers: int, a, step: int) -> dict:
    copies = max(1, min(n_layers, int(a.whole_gb * 2**30 // lay.value_bytes)))
    bufs = [lay] + [Layer(units, m, seed=7919 * (i + 1) + m) for i in range(copies - 1)]
    outs = {}
    for layer in range(n_layers):
        outs.update(bufs[layer % copies].outputs(layer))
    doc = {"layers": n_layers, "copies": copies, "tables": len(outs), "value_bytes": n_layers * lay.value_bytes}
    if m == min(a.check_ms):
        doc["check"] = D.step_roots(CTX, step, outs, KEY).result() == D.host_step(CTX, step, host_copy(outs), KEY)
    commit_once(outs, step)
    kern, host = [], []
    for _ in range(a.whole_reps):
        us, h, _ = commit_once(outs, step)
        kern.append(us)
        host.append(h)
    doc.update({"commit_ms": statistics.median(kern) / 1e3, "host_ms": statistics.median(host) / 1e3,
                "commit_ms_all": [round(v / 1e3, 3) for v in kern], "host_ms_all": [round(v / 1e3, 3) for v in host]})
    doc["commit_gbs"] = doc["value_bytes"] / doc["commit_ms"] / 1e6
    del bufs, outs
    torch.cuda.empty_cache()
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds", default="8192,12288")
    ap.add_argument("--ms", default="64,128,256,512")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--contexts", default="0")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--check-ms", default="1,64")
    ap.add_argument("--whole", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--whole-gb", type=float, default=16.0)
    ap.add_argument("--whole-reps", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ints = lambda s: tuple(int(v) for v in s.split(",") if v)  # noqa: E731
    ds, ms, contexts, a.check_ms = ints(a.ds), ints(a.ms), ints(a.contexts), ints(a.check_ms)
    ms = tuple(sorted(set(ms) | set(a.check_ms)))
    clock = kbench.nvml()
    meta = {"device": torch.cuda.get_device_name(), "arch": X.arch(), "torch": torch.__version__,
            "configs": {f"{n},{k}": v for (n, k), v in R.CONFIGS.items()}, "model_bytes": MODEL_BYTES,
            "units": [list(u) for u in UNITS], "tiny": tiny_check()}
    print(json.dumps({"tiny": meta["tiny"]}), flush=True)
    failed = [s for s, v in meta["tiny"].items() if not (v["device_is_twin"] and v["twin_is_reference"])]
    rows, step = [], 1
    for d in ds:
        for unit_set in a.sets.split(","):
            for context in (contexts if unit_set == "every" else (0,)):
                for m in ms:
                    t0 = time.time()
                    row = measure(d, unit_set, context, m, a, clock, step)
                    step += 2
                    row["s"] = round(time.time() - t0, 1)
                    rows.append(row)
                    with open(out / "commit.jsonl", "a") as f:
                        f.write(json.dumps(row, default=str) + "\n")
                    bad = ("failed" in row or row.get("check_layer") is False or not row.get("roots_stable", True)
                           or row.get("whole_measured", {}).get("check") is False)
                    if bad:
                        failed.append(f"d{d}:{unit_set}:c{context}:m{m}")
                    print(json.dumps({k: row.get(k) for k in ("d", "set", "context", "m", "value_bytes", "commit_gbs", "share",
                                                              "overlap_slowdown", "s")}
                                     | {"us": {k: round(v, 1) for k, v in row.get("us", {}).items()},
                                        "whole_ms": row.get("whole_measured", {}).get("commit_ms"), "bad": bad}), flush=True)
    doc = {**meta, "failed": failed, "ok": not failed,
           "rows": [{k: r.get(k) for k in ("d", "set", "context", "m", "tables", "value_bytes", "commit_gbs", "share",
                                           "overlap_slowdown", "us", "layers", "whole_extrapolated", "whole_measured")}
                    for r in rows]}
    (out / "commit.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0 if doc["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
