"""What a tensor-core reduction of a linear's s32 output costs on the card, alone and beside the linear: O = X W^T [M, N]
(s32) times a public int8 matrix on the s8 tensor cores, so that only the [M, r] result would be hashed, priced against
the linear itself. A cost, not a layout: whether a reduction binds O is not this script's question.

Per linear (N, K) of `--models` at M and each r of `--rs`, the arms, timed by `kbench.timed` (each a CUDA graph of the
same number of calls over buffer copies past 3x the L2, so every operand streams from DRAM; items of about 10 ms
alternating arm by arm, order rotated each rep; the SM clock and board power read around each item):

    served_wg:CFG      the exact linear, `wgmma.scaled_mm` (bf16 y), at its `configs.CONFIGS` pick or default and at
                       SERVED_CFGS, each config's y held to `run.reference` of torch._int_mm's sums before it is timed;
                       served_wg is the fastest, and served_wgc the same config on BESIDE_CTAS of the SMs
    served_lt          torch._int_mm (cuBLASLt's s8 GEMM): O itself
    read               one streaming read of O (`bench.read_kernel`): the floor of a reduction that reads O as stored
    split, cut         O into its four byte planes [4, M, N], and into its low byte [M, N] (torch copies): what the forms
                       that read derived bytes add, unless the linear's epilogue writes those bytes itself
    bytes4_{lt,wg}:r   (i) O's four bytes a cell, O viewed as int8 [M, 4N] (no copy), times R [r, 4N]: one GEMM, K = 4N
    planes_{lt,wg}:r   (i) the four byte planes, each [M, N] times its own R_b [r, N]: four GEMMs
    trunc_{lt,wg}:r    (ii) the low byte [M, N] times R [r, N]: one GEMM
    beside_{wg,wgc,lt}+{bytes4,trunc}:r
                       that served linear with that reduction (lt) on a second stream in the same graph, call by call,
                       the reduction reading the previous call's O: beside minus served is the reduction's marginal cost

The lt reductions are torch._int_mm into s32 [M, r] (exact while K < 2^17, else the sums wrap mod 2^32); the wg ones are
`wgmma.scaled_mm` with unit scales (bf16 out, held to the lt sums' `run.reference`), its narrowest tile 64 columns, so an
r under 64 pads. R is seeded random s8 and O's bytes are read as s8 codes.

Hashing only the reduction commits r s32 words a row (4r for the four planes) instead of O's N: SHA-512 compressions per
T MAC of the linear, blocks / (M N K) x 1e12, with blocks packed (M x 4r / 128 bytes) or row by row with SHA-512's padding
(M x ceil((4r + 17) / 128)), beside O's own (M x 4N / 128).

    python reduce.py --out OUT [--models llama-3.1-8b,mqa16k52] [--m 16384] [--rs 8,16,32,64,128] [--reps 7] [--only M:NAME,...]

Writes OUT/rows.jsonl (one line a linear), OUT/summary.json (per model, variant and r: sums over the layer's linears) and
OUT/table.md.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import triton

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "hash" / "exactlin"))
import bench as B  # noqa: E402
import fixture as FX  # noqa: E402
import kbench  # noqa: E402
import kgate  # noqa: E402
import run as R  # noqa: E402
import wgmma as W  # noqa: E402

RS = (8, 16, 32, 64, 128)
SERVED_CFGS = ("128x256:c0:g8", "128x256:c0:g16", "128x128:c0:g8", "128x256:c0:g8:m2")
BESIDE_CTAS = 116
FORMS = {"bytes4": "(i) bytes, one GEMM, K = 4N", "planes": "(i) byte planes, 4 GEMMs", "trunc": "(ii) low byte"}
BESIDE_FORMS = ("bytes4", "trunc")
BLOCK, PAD = 128, 17


def linears(models: tuple[str, ...]) -> list[tuple[str, str, int, int]]:
    """(model, linear, N, K) of each model's layer: `kgate.MODELS`' (llama-3.1-8b, qwen3-8b), else `fixture.model`'s."""
    out = []
    for model in models:
        if model in kgate.MODELS:
            out += [(model, name, n, k) for name, (n, k) in kgate.MODELS[model]["linears"].items()]
        else:
            out += [(model, name, n, k) for name, n, k, _ in FX.model(model).linears]
    return out


def probe():
    """[SM clock MHz, board power W] of torch's GPU (NVML, by the device's UUID: NVML's indices ignore
    CUDA_VISIBLE_DEVICES), or [None, None]."""
    try:
        import pynvml
        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        h = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
        return lambda: [pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM), round(pynvml.nvmlDeviceGetPowerUsage(h) / 1e3)]
    except Exception:  # noqa: BLE001
        return lambda: [None, None]


def copies(nbytes: int) -> int:
    return max(2, math.ceil(3 * torch.cuda.get_device_properties(0).L2_cache_size / nbytes))


def mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """a [M, K] s8 times b [N, K] s8 transposed, into out [M, N] s32 (cuBLASLt)."""
    torch._int_mm(a, b.t(), out=out)


def split(o: torch.Tensor, planes: torch.Tensor) -> None:
    m, n = o.shape
    planes.copy_(o.view(torch.int8).view(m, n, 4).permute(2, 0, 1))


class Linear:
    """One (N, K) at M: X, weight copies, O copies and the bytes derived from them, outputs, and the arms."""

    def __init__(self, n: int, k: int, m: int, seed: int):
        self.n, self.k, self.m = n, k, m
        cw, co = copies(n * k), copies(4 * m * n)
        self.calls = max(cw, co)
        self.ws = [R.Weights(n, k, seed=seed + i, special=False) for i in range(cw)]
        self.x = R.codes(m, k, seed, FX.STRUCT_ROWS)
        self.sa = torch.from_numpy(FX.act_scales(m, seed, special=False)).cuda()
        self.os = [torch.empty((m, n), dtype=torch.int32, device="cuda") for _ in range(co)]
        for i, o in enumerate(self.os):
            mm(self.x, self.ws[i % cw].w, o)
        self.planes = [torch.empty((4, m, n), dtype=torch.int8, device="cuda") for _ in self.os]
        self.lows = [torch.empty((m, n), dtype=torch.int8, device="cuda") for _ in self.os]
        for o, p, lo in zip(self.os, self.planes, self.lows):
            split(o, p)
            lo.copy_(o)
        self.y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        self.z = torch.empty((m, n), dtype=torch.int32, device="cuda")
        self.gates: dict = {}

    def loop(self, one):
        def f():
            for i in range(self.calls):
                one(i)
        return f

    def wg_one(self, cfg: W.Cfg):
        def one(i):
            wt = self.ws[i % len(self.ws)]
            W.scaled_mm(self.x, self.sa, wt.w, wt.sw, self.y, cfg)
        return one

    def lt_one(self, i):
        mm(self.x, self.ws[i % len(self.ws)].w, self.z)

    def gate_wg(self, cfgs: list[W.Cfg]) -> list[W.Cfg]:
        """The configs whose y equals `run.reference` of the first weight copy's sums on every cell."""
        ref = R.reference(self.os[0], self.sa, self.ws[0].sw)
        ok = []
        for cfg in cfgs:
            try:
                R.poison(self.y)
                h = W.scaled_mm(self.x, self.sa, self.ws[0].w, self.ws[0].sw, self.y, cfg)
                torch.cuda.synchronize()
                d = R.diff(self.y, ref)
            except Exception as ex:  # noqa: BLE001 - a config that fails to launch is a record
                torch.cuda.synchronize()
                self.gates[f"served_wg:{cfg}"] = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
                continue
            self.gates[f"served_wg:{cfg}"] = {"mismatches": d, **h}
            if d == 0:
                ok.append(cfg)
        del ref
        torch.cuda.empty_cache()
        return ok

    def read_one(self):
        n = self.m * self.n
        tile = 4096
        iters = max(1, min(16, n // (tile * 4 * 132)))
        grid = triton.cdiv(n, tile * iters)
        out = torch.empty(grid, dtype=torch.int32, device="cuda")

        def one(i):
            B.read_kernel[(grid,)](self.os[i % len(self.os)], out, n, BLOCK=tile, ITERS=iters, num_warps=8)
        return one

    def split_one(self, i):
        j = i % len(self.os)
        split(self.os[j], self.planes[j])

    def cut_one(self, i):
        j = i % len(self.os)
        self.lows[j].copy_(self.os[j])


class Reduction:
    """The public matrices and outputs of one r, and its arms' single calls."""

    def __init__(self, lin: Linear, r: int, seed: int):
        self.lin, self.r = lin, r
        g = torch.Generator(device="cuda").manual_seed(seed)

        def codes(*shape):
            return torch.randint(-128, 128, shape, generator=g, dtype=torch.int16, device="cuda").to(torch.int8)
        m, n = lin.m, lin.n
        self.mats = {"bytes4": [codes(r, 4 * n)], "planes": list(codes(4, r, n).unbind(0)), "trunc": [codes(r, n)]}
        self.z = torch.empty((4, m, r), dtype=torch.int32, device="cuda")
        self.y = torch.empty((4, m, r), dtype=torch.bfloat16, device="cuda")
        self.one_m = torch.ones(m, dtype=torch.float32, device="cuda")
        self.one_r = torch.ones(r, dtype=torch.float32, device="cuda")

    def operands(self, form: str, j: int) -> list[torch.Tensor]:
        lin = self.lin
        return {"bytes4": lambda: [lin.os[j].view(torch.int8)], "planes": lambda: list(lin.planes[j].unbind(0)),
                "trunc": lambda: [lin.lows[j]]}[form]()

    def one(self, form: str, backend: str, shift: int = 0):
        lin, mats = self.lin, self.mats[form]
        cfg = W.default_cfg(lin.m, self.r, mats[0].shape[1])

        def call(i):
            for b, (a, rm) in enumerate(zip(self.operands(form, (i - shift) % len(lin.os)), mats)):
                if backend == "lt":
                    mm(a, rm, self.z[b])
                else:
                    W.scaled_mm(a, self.one_m, rm, self.one_r, self.y[b], cfg)
        return call

    def check_wg(self, form: str) -> int:
        """wg's bf16 sums against `run.reference` of lt's on the first O copy, every cell."""
        self.one(form, "lt")(0)
        self.one(form, "wg")(0)
        torch.cuda.synchronize()
        nb = len(self.mats[form])
        return sum(R.diff(self.y[b], R.reference(self.z[b], self.one_m, self.one_r)) for b in range(nb))


def beside(lin: Linear, gemm_one, red_one):
    """Call by call: fork, the reduction on a side stream, the GEMM on the caller's, join."""
    side = torch.cuda.Stream()
    forks = [torch.cuda.Event() for _ in range(lin.calls)]
    joins = [torch.cuda.Event() for _ in range(lin.calls)]

    def f():
        main = torch.cuda.current_stream()
        for i in range(lin.calls):
            forks[i].record(main)
            side.wait_event(forks[i])
            with torch.cuda.stream(side):
                red_one(i)
            gemm_one(i)
            joins[i].record(side)
            main.wait_event(joins[i])
    return f


def tried(lin: Linear, name: str, f) -> bool:
    try:
        f()
        torch.cuda.synchronize()
        return True
    except Exception as ex:  # noqa: BLE001 - an arm that fails to launch is a record
        torch.cuda.synchronize()
        lin.gates[name] = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
        return False


def clock_med(items: list) -> list:
    vals = [c for pair in items for c in pair if c and c[0] is not None]
    return [statistics.median(v[0] for v in vals), statistics.median(v[1] for v in vals)] if vals else [None, None]


def measure(model: str, name: str, n: int, k: int, m: int, rs: tuple[int, ...], reps: int, clock, read_gbs: float,
            kern) -> dict:
    lin = Linear(n, k, m, seed=1000 + n + k)
    cfgs = lin.gate_wg(list(dict.fromkeys([kern.pick(m, n, k)] + [W.Cfg.parse(t) for t in SERVED_CFGS])))
    if not cfgs:
        raise RuntimeError(f"{model}:{name}: no served config passed its gate: {lin.gates}")
    pre, _, _ = kbench.timed({str(c): lin.loop(lin.wg_one(c)) for c in cfgs}, 3, clock, lin.calls)
    best = min(cfgs, key=lambda c: pre[str(c)])
    narrow = W.Cfg(best[0], best[1], BESIDE_CTAS, *best[3:])
    fns = {f"served_wg:{c}": lin.loop(lin.wg_one(c)) for c in cfgs}
    fns["served_wgc"] = lin.loop(lin.wg_one(narrow))
    fns["served_lt"] = lin.loop(lin.lt_one)
    fns["read"] = lin.loop(lin.read_one())
    fns["split"] = lin.loop(lin.split_one)
    fns["cut"] = lin.loop(lin.cut_one)
    reds = {r: Reduction(lin, r, seed=7 + r) for r in rs}
    served_one = {"wg": lin.wg_one(best), "wgc": lin.wg_one(narrow), "lt": lin.lt_one}
    for r, red in reds.items():
        for form in FORMS:
            for backend in ("lt", "wg"):
                arm = f"{form}_{backend}:{r}"
                if tried(lin, arm, lambda: red.one(form, backend)(0)):
                    fns[arm] = lin.loop(red.one(form, backend))
            if f"{form}_wg:{r}" in fns and f"{form}_lt:{r}" in fns:
                lin.gates[f"{form}_wg:{r}"] = {"mismatches_vs_lt": red.check_wg(form)}
        for form in BESIDE_FORMS:
            for s, g1 in served_one.items():
                arm = f"beside_{s}+{form}:{r}"
                f = beside(lin, g1, red.one(form, "lt", shift=1))
                if tried(lin, arm, f):
                    fns[arm] = f
    med, us, meta = kbench.timed(fns, reps, clock, lin.calls)
    served_wg = min(med[f"served_wg:{c}"] for c in cfgs)
    row = {"model": model, "name": name, "n": n, "k": k, "m": m, "calls": lin.calls,
           "copies": {"w": len(lin.ws), "o": len(lin.os)}, "macs": m * n * k, "o_bytes": 4 * m * n,
           "served_cfg": str(best), "served_cfg_narrow": str(narrow), "served_wg": served_wg, "us": med,
           "read_floor_us": 4 * m * n / read_gbs / 1e3,
           "clock": {arm: clock_med(v) for arm, v in meta["clocks"].items()}, "replays_per_item": meta["replays_per_item"],
           "us_all": {arm: [round(v, 3) for v in us[arm]] for arm in us}, "gates": lin.gates}
    del lin, reds, fns
    torch.cuda.empty_cache()
    return row


def blocks(m: int, r: int, form: str) -> tuple[int, int]:
    """SHA-512 blocks of the reduction's [M, r] s32 (four of them for the planes): packed, and row by row padded."""
    per_row = 4 * r * (4 if form == "planes" else 1)
    return math.ceil(m * per_row / BLOCK), m * math.ceil((per_row + PAD) / BLOCK)


def derived(row: dict, form: str, r: int) -> dict | None:
    """The reduction's numbers for one linear, form and r (the faster backend's time), or None where neither launched.
    beside_S: the served linear S with the reduction; diff_S: that minus the linear alone on every SM (wgmma at the
    config beside ran, for wgc too, or cuBLASLt)."""
    us = row["us"]
    lt, wg = us.get(f"{form}_lt:{r}"), us.get(f"{form}_wg:{r}")
    if lt is None and wg is None:
        return None
    red = min(v for v in (lt, wg) if v is not None)
    packed, by_row = blocks(row["m"], r, form)
    alone = {"wg": us[f"served_wg:{row['served_cfg']}"], "lt": us["served_lt"]}
    out = {"red_us": red, "red_lt_us": lt, "red_wg_us": wg,
           "derive_us": {"bytes4": 0.0, "planes": us["split"], "trunc": us["cut"]}[form],
           "served_wg_us": row["served_wg"], "served_lt_us": us["served_lt"], "alone_wg_us": alone["wg"],
           "alone_lt_us": alone["lt"], "blocks_packed": packed, "blocks_rows": by_row,
           "clock": row["clock"].get(f"{form}_{'lt' if red == lt else 'wg'}:{r}"),
           "clock_served": row["clock"].get(f"served_wg:{row['served_cfg']}")}
    for s, base in (("wg", alone["wg"]), ("wgc", alone["wg"]), ("lt", alone["lt"])):
        b = us.get(f"beside_{s}+{form}:{r}")
        out[f"beside_{s}_us"], out[f"diff_{s}_us"] = b, (None if b is None else b - base)
    return out


def summarize(rows: list[dict], rs: tuple[int, ...]) -> dict:
    out = {}
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        macs = sum(r["macs"] for r in mine)
        doc = {"linears": [r["name"] for r in mine], "macs": macs,
               "o_per_tmac": sum(r["o_bytes"] / BLOCK for r in mine) / macs * 1e12,
               "served_wg_us": sum(r["served_wg"] for r in mine), "served_lt_us": sum(r["us"]["served_lt"] for r in mine),
               "served_wgc_us": sum(r["us"]["served_wgc"] for r in mine), "read_us": sum(r["us"]["read"] for r in mine),
               "read_floor_us": sum(r["read_floor_us"] for r in mine), "variants": {}}
        for form in FORMS:
            for r in rs:
                ds = [derived(row, form, r) for row in mine]
                if None in ds:
                    continue
                v = {key: sum(d[key] for d in ds) for key in ("red_us", "derive_us", "served_wg_us", "alone_wg_us",
                                                             "alone_lt_us", "blocks_packed", "blocks_rows")}
                v["per_tmac_packed"] = v.pop("blocks_packed") / macs * 1e12
                v["per_tmac_rows"] = v.pop("blocks_rows") / macs * 1e12
                v["ratio_wg"] = v["red_us"] / v["served_wg_us"]
                v["share_wg"] = v["red_us"] / (v["served_wg_us"] + v["red_us"])
                for key in (f"{kind}_{s}_us" for s in ("wg", "wgc", "lt") for kind in ("beside", "diff")):
                    if all(d[key] is not None for d in ds):
                        v[key] = sum(d[key] for d in ds)
                doc["variants"][f"{form}:{r}"] = v
        out[model] = doc
    return out


def per_tmac(x: float) -> str:
    return f"{x / 1e6:.2f} M" if x >= 1e5 else f"{x:,.0f}"


def table(rows: list[dict], summary: dict, rs: tuple[int, ...], meta: dict) -> str:
    def us(v, key, digits=0):
        return "-" if v is None or v.get(key) is None else f"{v[key]:,.{digits}f}"

    lines = [f"M = {rows[0]['m']:,}; microseconds a call, medians; DRAM read {meta['read_gbs']:,.0f} GB/s measured.", ""]
    for model, doc in summary.items():
        lines += [f"### {model}: one layer's linears ({', '.join(doc['linears'])}) summed", "",
                  f"The linear: wgmma {doc['served_wg_us']:,.0f}, cuBLASLt {doc['served_lt_us']:,.0f}, wgmma on "
                  f"{BESIDE_CTAS} SMs {doc['served_wgc_us']:,.0f}. Reading O once: {doc['read_us']:,.0f} (at the "
                  f"measured rate {doc['read_floor_us']:,.0f}). Hashing O itself: {per_tmac(doc['o_per_tmac'])} "
                  "compressions per T MAC.", "",
                  "| r | variant | reduction | + derive | reduction / linear (wgmma) | share of step | compressions per T MAC "
                  "(packed; by row) |", "|---|---|---|---|---|---|---|"]
        for r in rs:
            for form, label in FORMS.items():
                v = doc["variants"].get(f"{form}:{r}")
                lines.append(f"| {r} | {label} | not launched | | | | |" if v is None else
                             f"| {r} | {label} | {v['red_us']:,.0f} | {v['derive_us']:,.0f} | {v['ratio_wg']:.3f} | "
                             f"{v['share_wg']:.3f} | {per_tmac(v['per_tmac_packed'])}; {per_tmac(v['per_tmac_rows'])} |")
        lines += ["", "Beside the linear (the reduction by cuBLASLt on a second stream, reading the previous call's O): the "
                  "linear alone, with the reduction, and the difference.", "",
                  f"| r | variant | wgmma alone | + reduction | difference | wgmma on {BESIDE_CTAS} SMs + reduction | "
                  "difference | cuBLASLt alone | + reduction | difference |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            for form in BESIDE_FORMS:
                v = doc["variants"].get(f"{form}:{r}")
                lines.append(f"| {r} | {FORMS[form]} | " + " | ".join(us(v, key) for key in (
                    "alone_wg_us", "beside_wg_us", "diff_wg_us", "beside_wgc_us", "diff_wgc_us", "alone_lt_us",
                    "beside_lt_us", "diff_lt_us")) + " |")
        lines.append("")
    lines += ["### per linear", "",
              "| linear | N x K | r | variant | reduction (cuBLASLt; wgmma) | linear (wgmma; cuBLASLt) | ratio | difference "
              f"beside wgmma (all SMs; {BESIDE_CTAS}) | compressions per T MAC | SM MHz, W (reduction; linear) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        for r in rs:
            for form, label in FORMS.items():
                d = derived(row, form, r)
                head = f"| {row['model']}:{row['name']} | {row['n']} x {row['k']} | {r} | {label} | "
                if d is None:
                    lines.append(head + "not launched | | | | | |")
                    continue
                clocks = "; ".join("-" if not c or c[0] is None else f"{c[0]:.0f}, {c[1]:.0f}"
                                   for c in (d["clock"], d["clock_served"]))
                lines.append(head + f"{us(d, 'red_lt_us', 1)}; {us(d, 'red_wg_us', 1)} | {d['served_wg_us']:,.1f}; "
                             f"{d['served_lt_us']:,.1f} | {d['red_us'] / d['served_wg_us']:.3f} | {us(d, 'diff_wg_us', 1)}; "
                             f"{us(d, 'diff_wgc_us', 1)} | {per_tmac(d['blocks_packed'] / row['macs'] * 1e12)} | {clocks} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="llama-3.1-8b,mqa16k52")
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--rs", default=",".join(map(str, RS)))
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--only", default="", help="model:linear,... (default: every linear of --models)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rs = tuple(int(v) for v in a.rs.split(","))
    only = set(filter(None, a.only.split(",")))
    clock = probe()
    meta = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
            "cubin": W.lib().identity(), "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size, "m": a.m, "rs": rs,
            "reps": a.reps, "item_ms": kbench.ITEM_MS, "bw_before": B.bandwidth(clock, a.reps)}
    meta["read_gbs"] = meta["bw_before"]["read_gbs"]
    print(json.dumps({"bw_before": {k: meta["bw_before"][k] for k in ("read_gbs", "copy_gbs")}}), flush=True)
    kern = R.kernel("wgmma")
    rows = []
    for model, name, n, k in linears(tuple(a.models.split(","))):
        if only and f"{model}:{name}" not in only:
            continue
        row = measure(model, name, n, k, a.m, rs, a.reps, clock, meta["read_gbs"], kern)
        rows.append(row)
        with open(out / "rows.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        shown = ("served_wg", "served_wgc", "served_lt", "read", "split", "cut", "bytes4_lt:64", "trunc_lt:64",
                 "beside_wg+bytes4:64", "beside_wgc+bytes4:64")
        print(json.dumps({"linear": f"{model}:{name}", "served_cfg": row["served_cfg"], "served_wg": round(row["served_wg"], 1),
                          "us": {arm: round(row["us"][arm], 1) for arm in shown if arm in row["us"]},
                          "clock_served": row["clock"].get(f"served_wg:{row['served_cfg']}")}), flush=True)
    meta["bw_after"] = B.bandwidth(clock, a.reps)
    summary = summarize(rows, rs)
    (out / "summary.json").write_text(json.dumps({**meta, "models": summary}, indent=1) + "\n")
    (out / "table.md").write_text(table(rows, summary, rs, meta))
    print((out / "table.md").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
