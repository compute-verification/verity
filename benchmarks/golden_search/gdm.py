"""GD-MODEL's J (golden/model.py of cursor/compute-research-gd-model-6eb4) as a model for search.py.

    python search.py run --model gdm --grid coarse --shard 0/16 --scenario nominal --out DIR

The module is vendored at `GD_MODEL_COMMIT` under vendor/golden, so a `research run` checkout of this branch carries it;
`GD_MODEL=PATH` imports another copy (a directory holding golden/model.py).

What this adds to GD-MODEL's J:
- the knobs as a flat grid in GD-MODEL's vocabulary (`KNOBS`, `GRIDS`), each design mapped to its design dict (`to_design`);
- proof-block shapes. Values: a tile of r tokens x c output columns of a linear a block (r = c = 1 is GD-MODEL's
  coordinate). PoUW: a block of r x c cells over one segment (Dither) or all of K (JR2), proved as one block (`single`)
  or as TOP-138's leaf form (`lanes`): r lane blocks of 1 x c cells and a combine, drawn as a group (delta counts
  leaves);
- feasibility: every proof block at most 2^27 ANDs (C-Flock compose's K_MAX), its lowering within TOP-138's memory and
  15 minutes, every credited call's credit floor at least (1 - eps) of its G, and GD-MODEL's own refusals (m_max,
  lowering RAM, the bytes a window retains for its draw);
- a low and a high for every uncertain input (`INPUTS`), for the sensitivity sweeps. Each range is this harness's
  estimate unless its source says otherwise.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

GD_MODEL_COMMIT = "e0df0e9e76bc986422681d26ac34f0e12466ef71"
sys.path.insert(0, os.environ.get("GD_MODEL") or str(Path(__file__).resolve().parent / "vendor"))
import golden.model as gm  # noqa: E402

BLOCK_CAP = 2 ** 27
LOWER_WALL_S = 15 * 60
DENOISE = "note:compute-research/20261028T1930Z-report-gd-denoise (run r20261028-190744-7676)"
#: GD-DENOISE's served periodic-noise call at L 1,024, model sums over the plain GEMM, by m: V's coins at L' 32, 128 and
#: 512, X, and O's recompute less the plain GEMM. Its fourth part, the noised GEMM over (S-1)L, is priced by L here.
DENOISE_PARTS = {64: ((1.483, 1.479, 1.499), 0.161, 0.333), 512: ((0.759, 0.749, 0.782), 0.100, 0.258),
                 8192: ((0.155, 0.205, 0.470), 0.075, 0.202)}
#: GD-DENOISE's fresh-noise served call (O recomputed), over the plain GEMM, by m
FRESH_CALL = {64: 3.2, 512: 3.3, 8192: 4.0}


@dataclass(frozen=True)
class Input:
    value: float
    low: float
    high: float
    source: str
    uncertain: bool = True


def _gm(key: str, low: float, high: float, why: str) -> Input:
    v, status, src = gm.INPUTS[key]
    return Input(v, low, high, f"GD-MODEL {status}: {src}; range (estimate): {why}")


INPUTS: dict[str, Input] = {
    "o_per_and": _gm("o_per_and", 0.54e-8, 2.17e-8, "x1/2 to x2: blocks far from the fit's 3-6 M ANDs"),
    "h200_vs_sm120": _gm("h200_vs_sm120", 0.5, 2.0, "C-Flock's prover on the H200 against sm_120, unmeasured"),
    "stmt_fixed": _gm("stmt_fixed", 1.5, 6.0, "x1/2 to x2"),
    "stmt_per_and": _gm("stmt_per_and", 0.0, 7.0e-6, "0 (preprocessing kept across windows) to x2"),
    "staged_per_and": _gm("staged_per_and", 0.5e-10, 2.0e-10, "x1/2 to x2"),
    "b1_rate_vs_s8": _gm("b1_rate_vs_s8", 4.0, 16.0, "x1/2 to x2: unmeasured on Hopper"),
    "wrap_g_frac": _gm("wrap_g_frac", 0.05, 1.0, "pending GD-DENOISE"),
    "noise_fused_keep": _gm("noise_fused_keep", 0.05, 1.0, "fused at half the estimate, to a separate pass"),
    "corun_share": _gm("corun_share", 0.083, 0.43, "x1/2 to the 4-warp share (r20261028-012220-309e's 0.43)"),
    "seal_GBs": _gm("seal_GBs", 0.18, 4.0, "x1/2 to a host tree at 4 GB/s"),
    "seal_rest_ms": _gm("seal_rest_ms", 0.0, 16.5, "0 to x2: nobody has split the seal's host time"),
    "ew_ands_per_elem": _gm("ew_ands_per_elem", 500.0, 8000.0, "x1/4 to x4, pending GD-PARTITION"),
    "attn_ands_per_mac": _gm("attn_ands_per_mac", 200.0, 2000.0, "x1/3 to x3, pending GD-PARTITION"),
    "gates_per_and": _gm("gates_per_and", 2.1, 5.0, "the stub's 2.1 to 5"),
    "lower_s_per_gate": _gm("lower_s_per_gate", 0.45e-6, 10e-6, "x1/2 the target to 10 us"),
    "lower_bytes_per_gate": _gm("lower_bytes_per_gate", 8.6, 600.0, "x1/2 the target to today's 600 B (TOP-138)"),
    "launch_us": _gm("launch_us", 2.0, 8.0, "x1/2 to x2"),
    "cpu_s_in_gpu_s": _gm("cpu_s_in_gpu_s", 1 / 110, 1 / 20, "x1/2 to x2.75"),
    "gemm_corun_penalty_per_hash_cycle": _gm("gemm_corun_penalty_per_hash_cycle", 0.0, 0.7, "0 to x2"),
    "hash_overlap_today": _gm("hash_overlap_today", 0.1, 0.5, "x1/2 to x2"),
    "h1_ands_per_mac": Input(350.0, 230.0, 500.0,
                             "inbox GD-SEARCH 19:16Z: the K 2,048 32 x 32 H1 tile is ~736 M ANDs, ~350 a MAC with its "
                             "noise and openings (a calibration until GD-PROVER); used for JR2 blocks"),
    "combine_ands": Input(1.0e6, 0.2e6, 10e6, "estimate: a lanes leaf's combine block (PouwH1Combine_v1), unmeasured"),
    "dither_ands_scale": Input(1.0, 0.7, 2.0, "estimate: Dither blocks' ANDs over GD-MODEL's count (work, noise "
                                              "expansion, row openings)"),
    "retain_GB": Input(100.0, 50.0, 120.0, "GD-MODEL sampling.retain_GB (estimate): GB a window may hold for its draw"),
    "denoise_coins_scale": Input(1.0, 0.5, 2.0, f"x the served Dither call's coins and X, {DENOISE} (measured arms)"),
    "denoise_main": Input(1.0, 0.5, 2.5,
                          f"the noised GEMM over (S-1)L as a share of its plain mainloop: 1 at the protocol's in-kernel "
                          f"T_s (estimate), 2.5 at {DENOISE}'s chunked cuBLASLt with T_s through DRAM (an upper bound)"),
    "rep_share": Input(0.3, 0.1, 0.6, "estimate: share of C-Flock's prove and verify time in its two reps' query "
                                      "phases (GD-PROVER to measure); prices prover.reps 1"),
    "fresh_call_g": Input(1.0, 0.75, 1.5, f"x the served fresh-noise call, {DENOISE}: +3.2 (m 64) to +4.0 (m 8,192) G, "
                                          f"measured without T_s and the weight side"),
}


def nominal() -> dict[str, float]:
    return {k: i.value for k, i in INPUTS.items()}


_applied: dict | None = None


def _apply(v: dict[str, float]) -> None:
    global _applied
    if _applied is v:
        return
    for k, x in v.items():
        if k in gm.INPUTS:
            gm.INPUTS[k] = (x,) + tuple(gm.INPUTS[k][1:])
    _applied = v


# ---------------------------------------------------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------------------------------------------------

KNOBS = ("values", "bits", "row", "seal", "host", "kernels", "corun", "padded", "scheme", "L", "t", "noise", "drow",
         "block", "law", "steps", "vtile", "period", "hiding", "reps")
#: knobs a grid may leave out, at these values (a design's key names them only when they differ)
DEFAULTS = {"period": 1, "hiding": "hm96", "reps": 2}

GRIDS: dict[str, dict[str, tuple]] = {
    "coarse": {
        "values": ("every_op", "dedup", "fused_mlp"),
        "bits": (16, 12, 8),
        "row": (1024, 4096, 16384),
        "seal": ("tree", "roots"),
        "host": ("today", "fixed"),
        "kernels": ("served", "bar"),
        "corun": (False, True),
        "padded": (True, False),
        "scheme": ("dither", "jr2"),
        "L": (32, 64, 128, 256, 1024),
        "t": (1, 2, 3, 4),
        "noise": ("permuted", "fresh"),
        "drow": (1024, 4096, 16384),
        "block": ((1, 1, "single"), (1, 32, "single"), (8, 8, "single"), (8, 8, "lanes"), (32, 32, "single"),
                  (32, 32, "lanes")),
        "law": ("achievability", "uniform"),
        "steps": (1, 4, 16),
        "vtile": ((1, 1), (1, 8), (8, 8)),
    },
}
GRIDS["smoke"] = {k: v[:2] for k, v in GRIDS["coarse"].items()}
#: the coarse grid under Achievability's law alone (the kd-subset is refused wherever units' work differs, and its search
#: is slow on the demo's small populations)
GRIDS["coarse-ach"] = {**GRIDS["coarse"], "law": ("achievability",)}
#: the coarse grid at the engineering levers every coarse optimum takes, for the sensitivity sweeps
GRIDS["oat"] = {**GRIDS["coarse"], "seal": ("roots",), "host": ("fixed",), "kernels": ("bar",), "corun": (True,),
                "law": ("achievability",), "vtile": ((1, 1), (1, 8))}
#: the oat grid less the knobs the coarse optimum never moves on (vtile 1 x 1; bits 16 or 8), for a sweep on a few cores
GRIDS["oat-lite"] = {**GRIDS["oat"], "bits": (16, 8), "vtile": ((1, 1),)}
GRIDS["fine"] = {**GRIDS["coarse"],
                 "bits": (16, 12, 10, 8),
                 "row": (1024, 2048, 4096, 8192, 16384),
                 "L": (16, 32, 48, 64, 96, 128, 256, 512, 1024),
                 "t": (1, 2, 3, 4, 6),
                 "drow": (1024, 4096, 16384),
                 "block": ((1, 1, "single"), (1, 2, "single"), (2, 2, "single"), (1, 4, "single"), (1, 8, "single"),
                           (1, 32, "single"), (4, 4, "single"), (8, 8, "single"), (8, 8, "lanes"),
                           (16, 16, "lanes"), (32, 32, "single"), (32, 32, "lanes")),
                 "steps": (1, 2, 4, 8, 16, 32),
                 "vtile": ((1, 1), (1, 2), (1, 4), (1, 8), (2, 2), (4, 4), (8, 8))}
#: the accounting period: oat-lite at each period n (windows certified together), each hiding form and C-Flock's reps
GRIDS["period"] = {**GRIDS["oat-lite"], "period": (1, 10, 100, 1000, 10_000, 100_000), "hiding": ("hm96", "prf"),
                   "reps": (2, 1)}

_DITHER_ONLY = ("L", "t", "noise", "drow")


def size(name: str) -> int:
    return sum(1 for _ in grid(name))


def grid(name: str):
    """The product of GRIDS[name], less JR2's copies over Dither's knobs (each JR2 design once, at their first values)."""
    g = GRIDS[name]
    first = {k: g[k][0] for k in _DITHER_ONLY}
    j = 0
    for values in product(*(g.get(k, (DEFAULTS.get(k),)) for k in KNOBS)):
        d = dict(zip(KNOBS, values))
        if d["scheme"] == "jr2" and any(d[k] != first[k] for k in _DITHER_ONLY):
            continue
        yield j, d
        j += 1


def shard(name: str, i: int, n: int):
    for j, d in grid(name):
        if j % n == i:
            yield j, d


def key(d: dict) -> str:
    b = d["block"]
    blk = f"{b[0]}x{b[1]}{'L' if b[2] == 'lanes' else ''}"
    pouw = (f"dither/L{d['L']}/t{d['t']}/{d['noise']}/drow{d['drow']}" if d["scheme"] == "dither" else "jr2")
    return (f"{d['values']}/b{d['bits']}/row{d['row']}/{d['seal']}/{d['host']}/{d['kernels']}/"
            f"{'corun' if d['corun'] else 'nocorun'}/{'pad' if d['padded'] else 'real'}/{pouw}/blk{blk}/{d['law']}/"
            f"S{d['steps']}/vt{'x'.join(map(str, d['vtile']))}"
            + "".join(f"/{k}{d[k]}" for k in DEFAULTS if d.get(k, DEFAULTS[k]) != DEFAULTS[k]))


POINTS = {"t8192": "throughput", "t512": "demo"}


def served_call_g(gd: dict, v: dict[str, float]) -> float:
    """The served noised call's cost over the plain GEMM (all four linears), from GD-DENOISE's parts at the step's m,
    as GD-MODEL's `denoise_g_frac`: less the noise pass and the PRF, which GD-MODEL prices apart, plus writing the T_s
    words (GD-MODEL hashes them, from HBM)."""
    w, p = gd["pouw"], gd["point"]
    tok = p["real_tokens"] or p["T"]
    m = min(DENOISE_PARTS, key=lambda x: abs(math.log(x / tok)))
    G_ms = gm.INT8_MACS_PER_TOKEN * tok / (gm.theta(gm.I("gemm_mhz")) * gm.I("gemm_eff_per_clock"))
    act = gm.LAYERS * tok * (3 * gm.HIDDEN + gm.INTER)
    noise_ms = 2 * (act + gm.WEIGHT_BYTES) / (gm.I("hbm_GBs") * 1e6) * (gm.I("noise_fused_keep") if w["noise_fused"] else 1)
    prf_ms = gm.hash_ms((act + gm.WEIGHT_BYTES) / 8 / 64, 1.0, gm.I("hash_mhz")) * (1 - gm.I("corun_share"))
    if w["scheme"] == "jr2":
        return v["fresh_call_g"] * FRESH_CALL[m]
    L = w["L"]
    words = sum(gm.LAYERS * N * tok * (2 * (math.ceil(K / L) - 1) + 4) for _, K, N in gm.LINEARS if L < K)
    ts_ms = words / (gm.I("hbm_GBs") * 1e6)
    if w["noise"] == "fresh":
        call = v["fresh_call_g"] * FRESH_CALL[m]
    else:
        coins, x, recompute = DENOISE_PARTS[m]
        lp = min(max(gm.l_prime(w), 32), 512)
        c = coins[1]
        if lp < 128:
            c = coins[0] + (coins[1] - coins[0]) * math.log(lp / 32) / math.log(4)
        elif lp > 128:
            c = coins[1] + (coins[2] - coins[1]) * math.log(lp / 128) / math.log(4)
        macs = sum(K * N for _, K, N in gm.LINEARS)
        main = sum(K * N * (K - L) / K for _, K, N in gm.LINEARS if L < K) / macs
        call = v["denoise_coins_scale"] * (c + x) + v["denoise_main"] * main + recompute
    return max(0.0, call + (ts_ms - noise_ms - prf_ms) / G_ms)


def to_design(d: dict, point: str, v: dict[str, float]) -> dict:
    t = max(1, d["t"])
    pouw = {"scheme": d["scheme"]}
    if d["scheme"] == "dither":
        pouw.update(L=d["L"], L_prime=max(1, d["L"] // t), noise=d["noise"], value_row_bytes=d["drow"])
    else:
        pouw.update(h1_host="stack", kdhash=True, hash_at_bar=True, coin_off_loop=True)
    bar = d["kernels"] == "bar"
    gd = gm.design(
        POINTS[point],
        point={"steps_per_window": d["steps"]},
        commit={"values": d["values"], "bits_per_word": d["bits"], "row_bytes": d["row"], "seal": d["seal"],
                "host": d["host"], "hash_eff": 1.0 if bar else None, "gather": not bar, "corun": d["corun"],
                "padded": d["padded"]},
        pouw=pouw,
        sampling={"law": d["law"], "retain_GB": v["retain_GB"]},
    )
    gd["pouw"]["denoise_g_frac"] = served_call_g(gd, v)
    return gd


# ---------------------------------------------------------------------------------------------------------------------
# Proof blocks
# ---------------------------------------------------------------------------------------------------------------------

def _rows(nbytes: float, row: int) -> list[float]:
    """The rows of `row` bytes that hold nbytes contiguous: one row's size each (GD-MODEL opens min(K, R) for a K-byte
    operand; a K-byte operand at R < K spans ceil(K / R) rows, so it's counted here)."""
    n = max(1, math.ceil(nbytes / row))
    return [min(nbytes, row)] * n


def value_units(gd: dict, d: dict) -> list:
    """GD-MODEL's per-op units with each linear's coordinate replaced by a tile of r tokens x c columns: its r A rows
    and c B rows of K bytes, and its row scales, column scales and outputs, each contiguous."""
    r, c = d["vtile"]
    R = gd["commit"]["row_bytes"]
    units = [u for u in gm.preset_units(gd) if "coordinate" not in u["name"]]
    for name, K, N in gm.LINEARS:
        units.append({"name": f"{name} tile {r}x{c} (K {K})", "per_token": gm.LAYERS * N / (r * c),
                      "work": r * c * gm.s8_coord_work(K), "extra_ands": 0.0,
                      "open_rows": _rows(K, R) * (r + c) + _rows(4 * r, R) + _rows(4 * c, R) + _rows(2 * r * c, R),
                      "out_bits": 16})
    return units


def pouw_units(gd: dict, d: dict, v: dict[str, float]) -> list:
    """PoUW's units: a block of r x c cells over a Dither segment of L (or JR2's whole K), one block or a lanes leaf.
    A unit's `blocks` lists (ANDs, blocks a draw) of each proof block a draw proves."""
    w = gd["pouw"]
    r, c, form = d["block"]
    units = []
    for name, K, N in gm.LINEARS:
        if w["scheme"] == "dither":
            L, Rv = w["L"], w["value_row_bytes"]
            if L >= K:
                continue
            S_seg = math.ceil(K / L)

            def ands(rr: int) -> float:
                work = rr * c * L / 32 * gm.I("s8_step_ands")
                noise = (rr + c) * L * gm.I("sha512_ands") / 64
                # T_s and T_{s-1} of the block's cells: whole value rows, as GD-MODEL's [Rv, Rv] for one cell
                rows = _rows(L, Rv) * (rr + c) + [Rv] * (2 * math.ceil(2 * rr * c / Rv))
                return v["dither_ands_scale"] * (work + noise + sum(gm.row_ands(x) for x in rows)), work

            n_per_token = gm.LAYERS * N * S_seg / (r * c)
        elif w["scheme"] == "jr2":
            if K != w["K"]:
                continue

            R = gd["commit"]["row_bytes"]

            def opens(rr: int, row: int) -> float:
                rows = _rows(K, row) * (rr + c) + _rows(w["record_bytes_per_cell"] * rr * c, row)
                return sum(gm.row_ands(x) for x in rows)

            # the inbox's calibration: a 32 x 32 tile at K 2,048 (row 4 KiB) is h1_ands_per_mac a MAC with its noise
            # and openings; the in-block rate a MAC is that less the tile's openings, and a block's openings are its own
            core = v["h1_ands_per_mac"] - opens(32, 4096) / (1024 * 2048)

            def ands(rr: int) -> float:
                work = rr * c * gm.s8_coord_work(K)
                return core * rr * c * K + opens(rr, R), work

            n_per_token = gm.LAYERS * N / (r * c)
        else:
            return []
        leaf, work = ands(r)
        if form == "lanes" and r > 1:
            lane, _ = ands(1)
            blocks = [(lane, r), (v["combine_ands"], 1)]
        else:
            blocks = [(leaf, 1)]
        units.append({"name": f"{w['scheme']} {name} {r}x{c}{' lanes' if form == 'lanes' else ''}",
                      "per_token": n_per_token, "work": work, "blocks": blocks})
    return units


def proving(gd: dict, units: list, label: str, n_win: int = 1, prover: float = 1.0) -> dict:
    """GD-MODEL's `proving` (achievability or the uniform kd-subset), with a unit's draw proving each of its `blocks`
    (ANDs, count): statements a block template, k_log and lowering a block, the 2^27 cap a block."""
    s, pv, lo = gd["sampling"], gd["prover"], gd["lowering"]
    T = gm.window_tokens(gd)
    lam = math.log(1 / s["delta"])
    o, so, f0, f1 = gm.I("o_per_and"), gm.I("staged_per_and"), gm.I("stmt_fixed"), gm.I("stmt_per_and")
    speed = gm.I("h200_vs_sm120")
    gpa, spg, bpg = gm.I("gates_per_and"), gm.I("lower_s_per_gate"), gm.I("lower_bytes_per_gate")
    vf, va = gm.I("verify_fixed"), gm.I("verify_per_and")
    for u in units:
        if "blocks" not in u:
            u["blocks"] = [(u["work"] + u["extra_ands"] + sum(gm.row_ands(x) for x in u["open_rows"]), 1)]
    F = sum(u["per_token"] * T * u["work"] for u in units)
    C = s["eps"] * F * n_win
    out = {"population": label, "draws": 0.0, "gpu_s": 0.0, "statements": 0, "lower_cpu_s": 0.0, "stage_cpu_s": 0.0,
           "verify_cpu_s": 0.0, "lower_peak_GiB": 0.0, "lower_peak_GiB_today": 0.0, "max_block_ands": 0.0,
           "lower_wall_s": 0.0, "classes": [], "infeasible": []}
    if F <= 0:
        return out
    if s["law"] == "uniform":
        n_all = sum(u["per_token"] * T for u in units)
        kd = gm.uniform_kd(n_all, s["eps"], s["delta"])
    for u in units:
        n = u["per_token"] * T
        q = kd / n_all if s["law"] == "uniform" else min(1.0, lam * u["work"] / C)
        draws = n * q
        cls = {"name": u["name"], "units": n, "q": q, "draws": draws, "gpu_s": 0.0, "blocks": []}
        for ands, count in u["blocks"]:
            nb = draws * count
            k_log = math.ceil(math.log2(max(ands, 2))) + 1
            cap = 1 << max(0, pv["m_max"] - k_log)
            stmts = (math.ceil(nb * n_win / cap) if nb > 0 else 0) / n_win
            gpu = prover * (nb * o * ands + nb * so * ands + stmts * (f0 + f1 * ands)) / speed
            gates = ands * gpa
            lower = (1 if nb > 0 else 0) * gates * spg / n_win
            peak = gates * bpg / 2 ** 30
            if ands > BLOCK_CAP:
                out["infeasible"].append("block>2^27")
            if peak > lo["ram_gib"] or gates * spg > LOWER_WALL_S:
                out["infeasible"].append("lowering")
            stage = stmts * (gm.I("stage_fixed") + gm.I("stage_per_and") * ands) + nb * gm.I("stage_per_and_staged") * ands
            cls["gpu_s"] += gpu
            cls["blocks"].append({"ands": ands, "per_draw": count, "statements": stmts, "lower_peak_GiB": peak})
            out["gpu_s"] += gpu
            out["statements"] += stmts
            out["lower_cpu_s"] += lower
            out["stage_cpu_s"] += stage
            out["verify_cpu_s"] += prover * stmts * (vf + va * ands)
            out["lower_peak_GiB"] = max(out["lower_peak_GiB"], peak)
            out["lower_peak_GiB_today"] = max(out["lower_peak_GiB_today"],
                                              gates * gm.I("lower_bytes_per_gate_today") / 2 ** 30)
            out["max_block_ands"] = max(out["max_block_ands"], ands)
            out["lower_wall_s"] = max(out["lower_wall_s"], gates * spg)
        out["draws"] += draws
        out["classes"].append(cls)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# J under the guarantees
# ---------------------------------------------------------------------------------------------------------------------

_gm_proving, _gm_pouw_units = gm.proving, gm.pouw_units
_gm_row_ands, _gm_compressions_per_row = gm.row_ands, gm.compressions_per_row
#: GD-KNOBS (inbox 19:46Z): one C-Flock rep with each level's queries raised to clear 100 bits, about +3% queries
REP1_EXTRA_QUERIES = 0.03


def evaluate(d: dict, v: dict[str, float] | None = None, point: str = "t8192") -> dict:
    v = v or nominal()
    _apply(v)
    if point == "t512" and d["steps"] != 1:
        return {"J": math.inf, "serve": math.inf, "prove_ratio": math.inf, "feasible": False,
                "reasons": ["window class"]}
    n_win = d.get("period", 1)
    prover = 1.0 if d.get("reps", 2) == 2 else 1 - v["rep_share"] / 2 * (1 - REP1_EXTRA_QUERIES)
    if d.get("hiding", "hm96") == "prf":
        # a leaf H(salt || row) under SHA-512 as a PRF: an opening drops hm96's two compressions (116,240 ANDs), a row
        # on the serve side its 7 of 42 at 4 KiB
        gm.row_ands = lambda nbytes: gm.I("sha512_ands") * math.ceil((nbytes + 17) / 128)
        gm.compressions_per_row = lambda row: math.ceil((row + 17) / 128) + 2
    try:
        gd = to_design(d, point, v)
        vunits = value_units(gd, d)
        punits = pouw_units(gd, d, v)
        gd["partition"]["units"] = vunits
        # J's own proving is replaced by the block-aware one; its serve side and retention are GD-MODEL's
        pops = iter((proving(gd, vunits, "values", n_win, prover), proving(gd, punits, "PoUW", n_win, prover)))
        gm.proving = lambda *_: next(pops)
        gm.pouw_units = lambda _d: punits
        r = gm.J(gd)
    finally:
        gm.proving, gm.pouw_units = _gm_proving, _gm_pouw_units
        gm.row_ands, gm.compressions_per_row = _gm_row_ands, _gm_compressions_per_row
    vals, pouw = r["proving"]["values"], r["proving"]["pouw"]
    floor = gm.pouw_forced_frac(gd)
    credit = gm.pouw_credit(gd)
    reasons = []
    if d["law"] == "uniform" and any(len({u["work"] for u in us}) > 1 for us in (vunits, punits)):
        # the kd-subset's eps counts units; with units of unequal work it bounds no share of the work
        reasons.append("uniform")
    if not credit["earns_credit"]:
        reasons.append("no credit")
    elif floor < 1 - gd["sampling"]["eps"]:
        reasons.append("floor")
    for x in vals["infeasible"] + pouw["infeasible"]:
        if x not in reasons:
            reasons.append(x)
    if r["retained_GB_per_window"] > gd["sampling"]["retain_GB"]:
        reasons.append("retain")
    for x in r["infeasible"]:
        if "m_max" in x and "m_max" not in reasons:
            reasons.append("m_max")
    return {
        "J": r["J"], "serve": r["serve_term"], "prove_ratio": r["prove_term"],
        "prove_gpu_s": r["proving"]["total_gpu_s"], "verify_cpu_s": r["verification"]["cpu_s"],
        "t_remote_ms": r["t_remote_ms"], "t_stock_ms": r["t_stock_ms"],
        "t_window_ms": [r["t_stock_window_ms"], r["t_remote_window_ms"]],
        "parts_ms": {k: round(x, 3) for k, x in r["serve_overhead_ms"].items()},
        "prove_parts": {"values": vals["gpu_s"], "pouw": pouw["gpu_s"],
                        "cpu_in_gpu_s": r["proving"]["cpu_in_gpu_s"],
                        "top_classes": sorted(((c["name"], round(c["gpu_s"], 2)) for c in vals["classes"] + pouw["classes"]),
                                              key=lambda x: -x[1])[:4]},
        "commit_gb": r["commit"].get("step_bytes", 0.0) / 1e9,
        "retained_gb": r["retained_GB_per_window"],
        "floor_min": floor, "credited": credit["credited_share_of_linear_macs"],
        "max_ands": max(vals["max_block_ands"], pouw["max_block_ands"]),
        "lower_gib": max(vals["lower_peak_GiB"], pouw["lower_peak_GiB"]),
        "lower_gib_today": max(vals["lower_peak_GiB_today"], pouw["lower_peak_GiB_today"]),
        "lower_min": max(vals["lower_wall_s"], pouw["lower_wall_s"]) / 60,
        "draws_values": vals["draws"], "draws_compute": pouw["draws"],
        "feasible": not reasons, "reasons": reasons,
    }
