"""What the four-byte reduction of a linear's s32 output costs fused into the GEMM's epilogue (`wgmma.reduce_mm`,
s8_wgmma.cu's s8mr kernels) against the same GEMM alone (`wgmma.scaled_mm` at the same config): z [M, r] = O viewed as int8
[M, 4N] (each cell's four little-endian bytes) times R^T, R [r, 4N] a public seeded int8 matrix, computed from the
accumulators while the tile is on chip, so that only z would be hashed. A cost, not a layout: whether a reduction binds O is
not this script's question.

X and W are uniform random int8 codes (a few structured rows and columns: `run.codes`), the served case's power.

Per linear (N, K) of `--models` at M, first the gate, in this process and before any timing (a failed gate times nothing,
exit 4): scaled_mm's y at each config written into 0xA5 equals `run.reference` of torch._int_mm's sums O on every cell; for
each r, flush and fused config, with y and z first filled with 0xA5, reduce_mm's y equals scaled_mm's at that config on every
cell and its z equals the separate pass's, torch._int_mm(O.view(int8), R^T) (cuBLASLt, s32, wrapping mod 2^32 as integer
addition does), on every word. Then the arms, timed by `kbench.timed` (each a CUDA graph of the same calls over weight copies
past 3x the L2, so W streams from DRAM; items of about 10 ms alternating arm by arm, order rotated each rep). Around each item,
NVML's SM clock and board power, and the SM clock the item's last GEMM launch counted itself (its CTAs' clock64 cycles over
their %globaltimer nanoseconds, `wgmma.sm_mhz`: NVML's SM clock reads above it under load). NVML's power is an average over
its last second, so an item's reading mixes arms: each arm at config A also runs alone back to back for HOLD_S ("held"), its
power and clocks read at the end. theta is the linear's MACs a second over PEAK, the s8 tensor cores' rate at 1,830 MHz.

    alone:CFG           scaled_mm at CFG: the linear as served. CFG A is the fastest of CANDIDATES alone (a short timing
                        first); B is A's other tile width at A's raster group
    fused:CFG:r:FLUSH   reduce_mm at CFG (z's zeroing in the graph), FLUSH "red" (red.global.add of each s32) or "bulk" (one
                        bulk reduce-add a warpgroup through shared memory)
    sep:r               the separate pass alone, torch._int_mm(O.view(int8), R^T) over O copies (the four-byte form, one
                        cuBLASLt GEMM at depth 4N), for scale

and after the timing, each fused arm's z (its last call's) equals the separate pass's on that call's weight copy. Hashing z
instead of O commits 4r bytes a row instead of 4N: SHA-512 compressions per T MAC of the linear, blocks / (M N K) x 1e12,
blocks packed (M x 4r / 128 bytes) or row by row with SHA-512's padding (M x ceil((4r + 17) / 128)), beside O's (M x 4N / 128).

    python fuse.py --out OUT [--models llama-3.1-8b,mqa16k52] [--m 16384] [--rs 8,16,32,64,128] [--reps 7] [--only M:NAME,...]

Writes OUT/gate.json, OUT/rows.jsonl (one line a linear, its held readings with it), OUT/summary.json (per model, config, r
and flush: sums over the layer's linears, theta, the held power and clocks, and each kernel's registers from the build's
sass.json) and OUT/table.md.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "hash" / "exactlin"))
import fixture as FX  # noqa: E402
import kbench  # noqa: E402
import kgate  # noqa: E402
import run as R  # noqa: E402
import wgmma as W  # noqa: E402

CANDIDATES = ("128x256:c0:g8", "128x256:c0:g16", "128x256:c0:g32", "128x128:c0:g8", "128x128:c0:g1")
BLOCK, PAD = 128, 17
GATE_FAILED = 4
#: s8 wgmma's MACs a second on the H100 and H200 at 1,830 MHz (132 SMs, 4,096 MACs a cycle each)
PEAK = 989.4e12
HOLD_S = 2.0


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


def reader(nvml, clk: torch.Tensor):
    """[NVML SM MHz, NVML board W, clock64 MHz of the last launch that wrote clk since the last read, or None]."""
    def read():
        torch.cuda.synchronize()
        mhz = W.sm_mhz(clk)
        clk.zero_()
        return nvml() + [mhz]
    return read


def copies(nbytes: int) -> int:
    return max(2, math.ceil(3 * torch.cuda.get_device_properties(0).L2_cache_size / nbytes))


def mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """a [M, K] s8 times b [N, K] s8 transposed, s32 (cuBLASLt)."""
    return torch._int_mm(a, b.t()) if out is None else torch._int_mm(a, b.t(), out=out)


def med(v: list):
    return statistics.median(v) if v else None


def clock_med(items: list) -> dict:
    """An arm's items' readings: NVML's around each (before and after), clock64's of the item's own launches (after)."""
    nv = [c for pair in items for c in pair if c and c[0] is not None]
    return {"nvml_mhz": med([c[0] for c in nv]), "nvml_w": med([c[1] for c in nv]),
            "clock64_mhz": med([pair[1][2] for pair in items if pair[1][2] is not None])}


def graph(f) -> torch.cuda.CUDAGraph:
    """f captured as `kbench.timed` captures it: two warm calls on a side stream, then three replays."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            f()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    return g


def held(fns: dict, read, calls: int) -> dict:
    """Each fn's graph replayed back to back for about HOLD_S, alone, then read: {arm: {us, nvml_mhz, nvml_w, clock64_mhz}}."""
    out = {}
    for arm, f in fns.items():
        g = graph(f)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        n = max(1, math.ceil(HOLD_S * 1e3 / max(e0.elapsed_time(e1), 1e-3)))
        read()
        e0.record()
        for _ in range(n):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        mhz, watts, c64 = read()
        out[arm] = {"us": 1e3 * e0.elapsed_time(e1) / (n * calls), "nvml_mhz": mhz, "nvml_w": watts, "clock64_mhz": c64,
                    "seconds": e0.elapsed_time(e1) / 1e3}
        del g
    return out


def other_width(cfg: W.Cfg, n: int) -> W.Cfg | None:
    """cfg's other fused tile width at its group, or None where N is narrower than a 256-column tile."""
    tm, bn, ctas, group, cl, splits = cfg
    bn2 = 128 if bn == 256 else 256
    return None if bn2 > n else W.Cfg(tm, bn2, ctas, group, cl, splits)


class Linear:
    """One (N, K) at M: X, weight copies, O copies (torch._int_mm's sums), the outputs."""

    def __init__(self, n: int, k: int, m: int, seed: int):
        self.n, self.k, self.m = n, k, m
        cw, co = copies(n * k), copies(4 * m * n)
        self.calls = max(cw, co)
        self.ws = [R.Weights(n, k, seed=seed + i, special=False) for i in range(cw)]
        self.x = R.codes(m, k, seed, FX.STRUCT_ROWS)
        self.sa = torch.from_numpy(FX.act_scales(m, seed, special=False)).cuda()
        self.os = [mm(self.x, self.ws[i % cw].w) for i in range(co)]
        self.y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        self.yf = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        self.clk = torch.zeros(4 * W.lib().sms, dtype=torch.int64, device="cuda")

    def w(self, i: int) -> R.Weights:
        return self.ws[i % len(self.ws)]

    def loop(self, one):
        def f():
            for i in range(self.calls):
                one(i)
        return f

    def alone(self, cfg: W.Cfg):
        return self.loop(lambda i: W.scaled_mm(self.x, self.sa, self.w(i).w, self.w(i).sw, self.y, cfg, self.clk))


class Reduction:
    """R [r, 4N] (seeded s8 codes), its kernel order, one z a fused arm, and the separate pass's arm."""

    def __init__(self, lin: Linear, r: int, seed: int):
        self.lin, self.r = lin, r
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.mat = torch.randint(-128, 128, (r, 4 * lin.n), generator=g, dtype=torch.int16, device="cuda").to(torch.int8)
        self.rk = W.kernel_order(self.mat)
        self.zs: dict[str, torch.Tensor] = {}
        self.zsep = torch.empty((lin.m, r), dtype=torch.int32, device="cuda")

    def reference(self, o: torch.Tensor) -> torch.Tensor:
        return mm(o.view(torch.int8), self.mat)

    def z(self, arm: str) -> torch.Tensor:
        if arm not in self.zs:
            self.zs[arm] = torch.empty((self.lin.m, self.r), dtype=torch.int32, device="cuda")
        return self.zs[arm]

    def fused(self, cfg: W.Cfg, flush: str, arm: str):
        lin, z = self.lin, self.z(arm)
        return lin.loop(lambda i: W.reduce_mm(lin.x, lin.sa, lin.w(i).w, lin.w(i).sw, lin.yf, self.rk, z, cfg, flush, lin.clk))

    def sep(self):
        lin = self.lin
        return lin.loop(lambda i: mm(lin.os[i % len(lin.os)].view(torch.int8), self.mat, self.zsep))


def flushes(cfg: W.Cfg, r: int) -> list[str]:
    """The flushes with a reduction kernel at cfg's tile and width r."""
    return [f for f in W.FLUSHES if W.has_reduction(cfg, r, f)]


def gate(lin: Linear, cfgs: list[W.Cfg], fused_cfgs: list[W.Cfg], reds: dict[int, Reduction]) -> dict:
    """Mismatched cells or words of every alone config and every fused (config, r, flush) on weight copy 0."""
    w0, o0 = lin.ws[0], lin.os[0]
    ref = R.reference(o0, lin.sa, w0.sw)
    doc = {"alone": {}, "fused": {}}
    for cfg in dict.fromkeys(cfgs + fused_cfgs):
        R.poison(lin.y)
        doc["alone"][str(cfg)] = {**W.scaled_mm(lin.x, lin.sa, w0.w, w0.sw, lin.y, cfg)}
        torch.cuda.synchronize()
        doc["alone"][str(cfg)]["y_vs_reference"] = R.diff(lin.y, ref)
    for r, red in reds.items():
        zref = red.reference(o0)
        for cfg in fused_cfgs:
            W.scaled_mm(lin.x, lin.sa, w0.w, w0.sw, lin.y, cfg)
            for flush in flushes(cfg, r):
                arm = f"fused:{cfg}:{r}:{flush}"
                z = red.z(arm)
                R.poison(lin.yf, z)
                h = W.reduce_mm(lin.x, lin.sa, w0.w, w0.sw, lin.yf, red.rk, z, cfg, flush)
                torch.cuda.synchronize()
                doc["fused"][arm] = {**h, "y_vs_alone": int((R.bits16(lin.yf) != R.bits16(lin.y)).sum()),
                                     "y_vs_reference": R.diff(lin.yf, ref), "z_vs_separate": int((z != zref).sum()),
                                     "z_nonzero": int((zref != 0).sum())}
        del zref
    del ref
    torch.cuda.empty_cache()
    doc["ok"] = (all(v["y_vs_reference"] == 0 for v in doc["alone"].values())
                 and all(v["y_vs_alone"] == 0 and v["y_vs_reference"] == 0 and v["z_vs_separate"] == 0
                         for v in doc["fused"].values()))
    return doc


def after(lin: Linear, reds: dict[int, Reduction]) -> dict:
    """Each fused arm's z after its timing: its last call's, on weight copy (calls - 1)."""
    o = mm(lin.x, lin.w(lin.calls - 1).w)
    out = {}
    for red in reds.values():
        zref = red.reference(o)
        for arm, z in red.zs.items():
            out[arm] = int((z != zref).sum())
        del zref
    del o
    torch.cuda.empty_cache()
    return out


def measure(model: str, name: str, n: int, k: int, m: int, rs: tuple[int, ...], reps: int, nvml, kern) -> dict:
    lin = Linear(n, k, m, seed=1000 + n + k)
    clock = reader(nvml, lin.clk)
    pick = kern.pick(m, n, k)
    cands = [W.Cfg.parse(t) for t in CANDIDATES]
    if (pick[0], pick[1]) in W.FUSED_TILES and pick[4] == 1 and pick[5] == 1:
        cands.insert(0, pick)
    cands = [c for c in dict.fromkeys(cands) if c[1] <= max(n, 128)]
    pre, _, _ = kbench.timed({str(c): lin.alone(c) for c in cands}, 3, clock, lin.calls)
    a = min(cands, key=lambda c: pre[str(c)])
    b = other_width(a, n)
    fused_cfgs = [a] + ([b] if b else [])
    reds = {r: Reduction(lin, r, seed=7 + r) for r in rs}
    g = gate(lin, cands, fused_cfgs, reds)
    row = {"model": model, "name": name, "n": n, "k": k, "m": m, "calls": lin.calls, "macs": m * n * k, "o_bytes": 4 * m * n,
           "copies": {"w": len(lin.ws), "o": len(lin.os)}, "cfg": str(a), "cfg_b": str(b) if b else None,
           "pre_us": pre, "gate": g}
    if not g["ok"]:
        return row
    fns = {f"alone:{c}": lin.alone(c) for c in fused_cfgs}
    for r, red in reds.items():
        for cfg in fused_cfgs:
            for flush in flushes(cfg, r):
                arm = f"fused:{cfg}:{r}:{flush}"
                fns[arm] = red.fused(cfg, flush, arm)
        fns[f"sep:{r}"] = red.sep()
    meds, us, meta = kbench.timed(fns, reps, clock, lin.calls)
    row.update({"us": meds, "us_all": {arm: [round(v, 3) for v in us[arm]] for arm in us},
                "clock": {arm: clock_med(v) for arm, v in meta["clocks"].items()},
                "replays_per_item": meta["replays_per_item"], "after": after(lin, reds)})
    row["ok"] = all(v == 0 for v in row["after"].values())
    row["held"] = held({arm: f for arm, f in fns.items() if arm == f"alone:{a}" or arm.startswith(f"fused:{a}:")}, clock,
                       lin.calls)
    del lin, reds, fns
    torch.cuda.empty_cache()
    return row


def blocks(m: int, r: int) -> tuple[int, int]:
    """SHA-512 blocks of z [M, r] s32: packed, and row by row padded."""
    return math.ceil(m * 4 * r / BLOCK), m * math.ceil((4 * r + PAD) / BLOCK)


def weighted(mine: list[dict], arm_of, key: str):
    """The held readings' `key` over the linears, weighted by each one's time at that arm; None where any lacks one."""
    vals = [(row["us"][arm_of(row)], row["held"].get(arm_of(row), {}).get(key)) for row in mine]
    if any(v is None for _, v in vals):
        return None
    return sum(t * v for t, v in vals) / sum(t for t, _ in vals)


def summarize(rows: list[dict], rs: tuple[int, ...]) -> dict:
    """Per model: the layer's linears summed at each one's config A (and B, where every linear has one), per r and flush;
    theta of the sums, and at A the held power and clock64 MHz weighted by each linear's time."""
    out = {}
    for model in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == model]
        macs = sum(r["macs"] for r in mine)
        doc = {"linears": [r["name"] for r in mine], "macs": macs, "cfgs": {r["name"]: [r["cfg"], r["cfg_b"]] for r in mine},
               "o_per_tmac": sum(r["o_bytes"] / BLOCK for r in mine) / macs * 1e12, "variants": {}}
        for which in ("cfg", "cfg_b"):
            if any(row[which] is None for row in mine):
                continue
            alone = sum(row["us"][f"alone:{row[which]}"] for row in mine)
            doc[f"alone_{which}_us"] = alone
            for r in rs:
                packed = sum(blocks(row["m"], r)[0] for row in mine)
                by_row = sum(blocks(row["m"], r)[1] for row in mine)
                sep = sum(row["us"][f"sep:{r}"] for row in mine)
                for flush in W.FLUSHES:
                    if any(f"fused:{row[which]}:{r}:{flush}" not in row["us"] for row in mine):
                        continue
                    fused = sum(row["us"][f"fused:{row[which]}:{r}:{flush}"] for row in mine)
                    v = {"alone_us": alone, "fused_us": fused, "added_us": fused - alone, "added_over_alone": (fused - alone) / alone,
                         "share": (fused - alone) / fused, "sep_us": sep, "sep_share": sep / (alone + sep),
                         "theta_alone": macs / (alone * 1e-6) / PEAK, "theta_fused": macs / (fused * 1e-6) / PEAK,
                         "per_tmac_packed": packed / macs * 1e12, "per_tmac_rows": by_row / macs * 1e12}
                    if which == "cfg":
                        for key in ("nvml_w", "clock64_mhz", "nvml_mhz"):
                            v[f"held_alone_{key}"] = weighted(mine, lambda row: f"alone:{row['cfg']}", key)
                            v[f"held_fused_{key}"] = weighted(mine, lambda row, r=r, flush=flush: f"fused:{row['cfg']}:{r}:{flush}",
                                                              key)
                    doc["variants"][f"{which}:{r}:{flush}"] = v
        out[model] = doc
    return out


def num(v, fmt: str = ".0f") -> str:
    return "-" if v is None else format(v, fmt)


def registers(sass: Path | None) -> dict:
    """Each kernel's ptxas registers and highest register in its SASS (sass.json), where the build left it."""
    if sass is None or not sass.exists():
        return {}
    kernels = json.loads(sass.read_text()).get("kernels", {})
    return {name: {"regs": r.get("regs"), "max_reg": r.get("max_reg"), "forms": r.get("forms")}
            for name, r in kernels.items() if name.startswith(("s8mm_128x", "s8mr"))}


def per_tmac(x: float) -> str:
    return f"{x / 1e6:.2f} M" if x >= 1e5 else f"{x:,.0f}"


def table(rows: list[dict], summary: dict, rs: tuple[int, ...], meta: dict) -> str:
    lines = [f"M = {rows[0]['m']:,}; microseconds a call, medians of interleaved items; added = fused - alone at the same "
             "config; share = added / fused; the separate pass (cuBLASLt, K = 4N) for scale, its share = sep / (alone + sep); "
             f"theta = MACs a second / {PEAK / 1e12:.1f} T; held: each arm alone for {HOLD_S:.0f} s at config A, NVML's board "
             "power and the SM clock its launch counted (clock64 over %globaltimer), weighted by each linear's time.", ""]
    for model, doc in summary.items():
        cfgs = ", ".join(f"{n} {a}" + (f" (B {b})" if b else "") for n, (a, b) in doc["cfgs"].items())
        lines += [f"### {model}: one layer's linears ({', '.join(doc['linears'])}) summed", "",
                  f"Configs: {cfgs}. Hashing O itself: {per_tmac(doc['o_per_tmac'])} compressions per T MAC.", "",
                  "| r | flush | config | alone | fused | added | added / alone | share | theta alone; fused | "
                  "held W (alone; fused) | held clock64 MHz (alone; fused) | separate pass, share | "
                  "compressions per T MAC (packed; by row) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            for which in ("cfg", "cfg_b"):
                for flush in W.FLUSHES:
                    v = doc["variants"].get(f"{which}:{r}:{flush}")
                    if v is None:
                        continue
                    lines.append(f"| {r} | {flush} | {'A' if which == 'cfg' else 'B'} | {v['alone_us']:,.0f} | {v['fused_us']:,.0f} | "
                                 f"{v['added_us']:,.0f} | {v['added_over_alone']:.3f} | {v['share']:.3f} | "
                                 f"{v['theta_alone']:.3f}; {v['theta_fused']:.3f} | "
                                 f"{num(v.get('held_alone_nvml_w'))}; {num(v.get('held_fused_nvml_w'))} | "
                                 f"{num(v.get('held_alone_clock64_mhz'))}; {num(v.get('held_fused_clock64_mhz'))} | "
                                 f"{v['sep_us']:,.0f}, {v['sep_share']:.3f} | "
                                 f"{per_tmac(v['per_tmac_packed'])}; {per_tmac(v['per_tmac_rows'])} |")
        lines.append("")
    lines += ["### per linear", "",
              "| linear | N x K | config | r | flush | alone | fused | added / alone | separate | clock64 MHz (alone; fused) | "
              "NVML MHz (alone; fused) | held W (alone; fused) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if "us" not in row:
            lines.append(f"| {row['model']}:{row['name']} | {row['n']} x {row['k']} | gate failed | | | | | | | | | |")
            continue
        for r in rs:
            for cfg in (row["cfg"], row["cfg_b"]):
                if cfg is None:
                    continue
                for flush in W.FLUSHES:
                    arms = (f"alone:{cfg}", f"fused:{cfg}:{r}:{flush}")
                    if arms[1] not in row["us"]:
                        continue
                    al, fu = (row["us"][arm] for arm in arms)
                    c64 = "; ".join(num(row["clock"][arm]["clock64_mhz"]) for arm in arms)
                    nv = "; ".join(num(row["clock"][arm]["nvml_mhz"]) for arm in arms)
                    hw = "; ".join(num(row["held"].get(arm, {}).get("nvml_w")) for arm in arms)
                    lines.append(f"| {row['model']}:{row['name']} | {row['n']} x {row['k']} | {cfg} | {r} | {flush} | {al:,.1f} | "
                                 f"{fu:,.1f} | {(fu - al) / al:.3f} | {row['us'][f'sep:{r}']:,.1f} | {c64} | {nv} | {hw} |")
    regs = meta.get("registers", {})
    if regs:
        lines += ["", "### registers (ptxas at launch; highest register in the SASS, a consumer's after setmaxnreg)", "",
                  "| kernel | ptxas | highest |", "|---|---|---|"]
        lines += [f"| {k} | {v['regs']} | {v['max_reg']} |" for k, v in sorted(regs.items())]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="llama-3.1-8b,mqa16k52")
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--rs", default=",".join(map(str, W.RS)))
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--only", default="", help="model:linear,... (default: every linear of --models)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rs = tuple(int(v) for v in a.rs.split(","))
    only = set(filter(None, a.only.split(",")))
    nvml = probe()
    lib = W.lib()
    meta = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cubin": lib.identity(), "layout": lib.layout,
            "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size, "m": a.m, "rs": rs, "reps": a.reps,
            "item_ms": kbench.ITEM_MS, "registers": registers(lib.path.with_name("sass.json"))}
    kern = R.kernel("wgmma")
    todo = [lin for lin in linears(tuple(a.models.split(","))) if not only or f"{lin[0]}:{lin[1]}" in only]
    rows, ok = [], True
    for model, name, n, k in todo:
        row = measure(model, name, n, k, a.m, rs, a.reps, nvml, kern)
        rows.append(row)
        with open(out / "rows.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        ok = ok and row["gate"]["ok"] and row.get("ok", False)
        shown = {arm: round(v, 1) for arm, v in row.get("us", {}).items()
                 if arm.startswith("alone") or arm.endswith((":8:red", ":128:red", ":8:bulk", ":128:bulk"))}
        print(json.dumps({"linear": f"{model}:{name}", "cfg": row["cfg"], "gate_ok": row["gate"]["ok"], "after_ok": row.get("ok"),
                          "us": shown}), flush=True)
        if not row["gate"]["ok"]:
            bad = {k: v for k, v in row["gate"]["fused"].items() if v["z_vs_separate"] or v["y_vs_alone"]}
            print(json.dumps({"gate_failed": f"{model}:{name}", "alone": row["gate"]["alone"], "fused_bad": bad}), flush=True)
            break
    (out / "gate.json").write_text(json.dumps({r["model"] + ":" + r["name"]: r["gate"] for r in rows}, indent=1) + "\n")
    timed = [r for r in rows if "us" in r]
    # a model's sums only where every linear of it asked for was timed
    count = {model: sum(1 for lin in todo if lin[0] == model) for model, *_ in todo}
    summary = summarize([r for r in timed if sum(1 for t in timed if t["model"] == r["model"]) == count[r["model"]]], rs)
    (out / "summary.json").write_text(json.dumps({**meta, "models": summary}, indent=1) + "\n")
    if timed:
        (out / "table.md").write_text(table(timed, summary, rs, meta))
        print((out / "table.md").read_text(), flush=True)
    return 0 if ok else GATE_FAILED


if __name__ == "__main__":
    sys.exit(main())
