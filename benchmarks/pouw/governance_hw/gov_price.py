"""Roll-up for always-on logging on the H200 (sm_90) and the RTX PRO 6000 (sm_120): what the counted-only classes leak
as a share of tensor capacity, what the logging blocks cost in area and power, and the transcript's rates.

    python3 gov_price.py --synth SYNTH_JSON --energy ENERGY_JSON --out OUT

SYNTH_JSON is gov_synth.py --fold's output (counters, GF(2^64) multiplier, FT's round, the plane-limited unit),
ENERGY_JSON gov_energy.py's (both Keccak units against the 4-wide s8 MAC). Every input below carries its tag: R read (source named), C recalled, not re-read,
M measured or computed by a script, E estimated, A argued. Outputs are E unless every input is M or R.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GE_DFF = 17 / 3          # GE a flop bit, SI-ALG/SI-FOLD's library [R]
UM2_PER_GE = {"low": 0.035, "central": 0.05, "high": 0.06}   # N4-class NAND2-equivalent, placed [E]
S8_MAC_GE, FP8_MAC_GE = 817.9, 2285.5   # SI-BASELINE's s8 and FP8/FP32 MAC a lane [R of M]
POUW_FT_UNIT = 22722.0                  # SI-FOLD's Keccak-f[1600] unit, a round a clock [R of M]

DEVICES = {
    "h200": {
        "sms": 132, "tensor_hz": 1830e6, "vector_hz": 1980e6,   # sm_90 sheet.toml (main) [R]
        "s8_per_sm_clk": 4096, "bf16_per_sm_clk": 2048, "tf32_per_sm_clk": 1024,   # sheet.toml [R]
        "cores_per_sm": 4, "smsp_per_sm": 4, "warps_per_smsp": 16,   # Hopper SM [C]
        "words_per_core_clk": 32,          # SI-FOLD Wiring: 1,024 MACs/core-clk over 32 MACs a word [R]
        "issue_lanes_per_sm_clk": 128,     # census DS-90 'issue' row, 3.998 warp-instr/SM-clk [M, main rows.csv]
        "die_mm2": 814,                    # Hopper whitepaper via GOV-INSPECT [R]
        "board_w": 700,                    # H200 SXM TDP [C]
    },
    "sm120": {
        "sms": 188, "tensor_hz": 2430e6, "vector_hz": 2430e6,    # OCC-RATE §5 [R]
        "s8_per_sm_clk": 1024, "fp4_over_s8": 1.951,              # OCC-RATE §5 (FP-EXACT) [R of M]
        "cores_per_sm": 4, "smsp_per_sm": 4, "warps_per_smsp": 12,   # Blackwell client SM, 48 warps/SM [C]
        "words_per_core_clk": 8,           # SI-FOLD Wiring [R]
        "issue_lanes_per_sm_clk": 128,     # 4 SMSPs x 1 warp-instr [C]
        "die_mm2": 750,                    # GB202 [C]
        "board_w": 600,                    # RTX PRO 6000 Server Edition [C]
    },
}

# Counted-only routes to undeclared matmul, in s8-MAC-equivalents a lane-op. `cap` is lanes an SM-clock, `pipe` the
# pipe it shares, `exact` whether int8 x int8 products come out exact. Clock: vector pipes run at vector_hz.
ROUTES = {
    "h200": [
        # name, MACs a lane-op, lane cap (lo, hi), pipe, exact, tag, basis
        ("IDP4A", 4, (32, 64), "imad", True, "E",
         "no IDP4A row on sm_90 (model.toml); half or quarter issue; beside saturated wgmma 3.13% of R* [M, OCC-RATE]"),
        ("DFMA packed", 2, (64, 64), "fp64", True, "C/A",
         "64 FP64 FMA/SM-clk (CUDA Guide CC 9.0) [C]; two int8 products a 53-bit mantissa for K <= 2^11 [A]; "
         "shares the tensor pipe's time (OCC-RATE §3.5 t = 0.941) [M], so not additive beside tensor work"),
        ("HFMA2", 2, (128, 128), "fma", False, "C",
         "256 FP16 FMA/SM-clk (CUDA Guide CC 9.0) [C]; inexact for int8, useful at fp16"),
        ("FFMA", 1, (128, 128), "fma", True, "M",
         "issues at the 128-lane ceiling (census DS-90 'issue' row) [M]; one int8 product a 24-bit mantissa, K <= 2^10"),
        ("IMAD", 1, (64, 64), "imad", True, "M", "IMAD 1.9998 warp-instr/SM-clk (census row I) [M]"),
        ("LDS product table + shared atomic add", 1, (16, 32), "lsu", False, "E",
         "4-bit x 4-bit product table in shared memory, one LDS lane a product, accumulate by red.shared [E]"),
        ("MUFU exp/log", 1, (16, 16), "sfu", False, "C/E",
         "16 MUFU/SM-clk (CUDA Guide CC 9.0) [C]; a product as ex2(lg2 a + lg2 b), logs amortized [E]"),
        ("TEX bilinear", 4, (4, 4), "tex", False, "E",
         "4 texture units an SM, a bilinear fetch is 4 weighted MACs at ~8-bit weights [E]; issue ~1/4 lane a MAC"),
    ],
    "sm120": [
        ("IDP4A", 4, (64, 64), "imad", True, "M",
         "256 MACs/SM-clk beside IMMA, additive (OCC-RATE §5, r20261027-001958-0482) [M]"),
        ("FFMA", 1, (128, 128), "fma", True, "C", "128 FP32 lanes an SM [C]"),
        ("HFMA2", 2, (64, 64), "fma", False, "C/E", "FP16 non-tensor at the FP32 FLOP rate [C]"),
        ("RT core products", 58 / 4, (4, 4), "rt", True, "M",
         "58 integer products/SM-clk (OCC-RATE §5, RT-MODEL-MIN RM2) [M]; issue nearly free"),
        ("TEX", 61.4 / 4, (4, 4), "tex", False, "M", "about 6% of int8 (OCC-RATE §5, RT-MODEL-MIN RM4) [M]"),
        ("DFMA packed", 2, (2, 2), "fp64", True, "C", "FP64 at 1/64 of FP32 [C]"),
    ],
}


def leakage(dev: str, ends: str) -> dict:
    """Each route alone, and the best mix under the issue ceiling (greedy by MACs a lane, shared pipe caps)."""
    d = DEVICES[dev]
    tensor = d["s8_per_sm_clk"] * d["tensor_hz"]
    clk = d["vector_hz"]
    alone = {}
    for name, mpl, cap, pipe, exact, tag, basis in ROUTES[dev]:
        lanes = cap[0] if ends == "lo" else cap[1]
        alone[name] = {"macs_per_sm_clk": mpl * lanes, "share_of_tensor": mpl * lanes * clk / tensor,
                       "exact_int8": exact, "tag": tag, "basis": basis}
    issue = d["issue_lanes_per_sm_clk"]
    pipe_cap = {"imad": 64, "fma": 128, "fp64": 64, "lsu": 32, "sfu": 16, "tex": 1e9, "rt": 1e9}
    free_issue = {"tex", "rt"}            # one instruction launches many operations
    used_pipe: dict[str, float] = {}
    mix = []
    total = 0.0
    for name, mpl, cap, pipe, exact, tag, basis in sorted(ROUTES[dev], key=lambda r: -r[1]):
        if not exact and ends == "lo":
            continue                      # the low end counts exact int8 routes only
        lanes = min(cap[0] if ends == "lo" else cap[1], pipe_cap[pipe] - used_pipe.get(pipe, 0))
        if pipe not in free_issue:
            lanes = min(lanes, issue)
            issue -= lanes
        if lanes <= 0:
            continue
        used_pipe[pipe] = used_pipe.get(pipe, 0) + lanes
        mix.append((name, lanes, mpl * lanes))
        total += mpl * lanes
    return {"alone": alone, "mix": mix, "mix_macs_per_sm_clk": total, "mix_share": total * clk / tensor}


def costs(dev: str, synth: dict) -> dict:
    """GE a block, lo/central/hi, from named components."""
    d = DEVICES[dev]
    rows = synth["rows"]
    plane4 = rows["ft_plane4"]["ge"]      # four plane-limited rounds a clock, 32 words [M]
    round1 = rows["keccak1600_rnd"]["ge"] # one round a clock [M]; plane-limited at 8 words a round on sm_120
    unit = plane4 if d["words_per_core_clk"] == 32 else round1
    wiring_per_mac = (0.0, 11.8, 23.6)    # SI-FOLD Wiring [E]
    macs_per_core = d["s8_per_sm_clk"] // d["cores_per_sm"]
    state = 1600 * GE_DFF
    inline_reg = d["words_per_core_clk"] * 32 * GE_DFF
    counters = synth["rows"]["cnt_step"]["ge"]          # [M]
    gf = synth["rows"]["gf64_mul"]["ge"]                # [M]
    ft_core = {
        k: unit + wiring_per_mac[i] * macs_per_core + slots * state + ctl + inl * inline_reg
        for i, (k, slots, ctl, inl) in enumerate(
            [("low", 0, 1000, 0), ("central", 1, 2000, 1), ("high", 3, 3000, 1)])
    }
    # RT, a tap on every register-file write port of an SMSP: one round a clock, per-warp chains, an in-order buffer
    w = d["warps_per_smsp"]
    rt_smsp = {
        k: plane4 + w * 1600 * gb + depth * 1024 * gb + counters + 3000
        for k, gb, depth in [("low", 1.0, 8), ("central", 2.5, 16), ("high", GE_DFF, 32)]
    }
    rt_orderfree_smsp = 17 * gf + w * 64 * GE_DFF + counters      # GF(2^64) power-table variant [M parts, E count]
    per_sm = {k: d["cores_per_sm"] * ft_core[k] + d["smsp_per_sm"] * rt_smsp[k] for k in ft_core}
    ft_sm = {k: d["cores_per_sm"] * ft_core[k] for k in ft_core}
    counters_sm = d["smsp_per_sm"] * counters
    keyed_per_mac = unit / macs_per_core
    return {"keyed_unit": unit, "keyed_ge_per_mac": keyed_per_mac, "keyed_share_s8": keyed_per_mac / S8_MAC_GE,
            "keyed_share_fp8": keyed_per_mac / FP8_MAC_GE, "pouw_ft_share_s8": POUW_FT_UNIT / 1024 / S8_MAC_GE,
            "ft_core": ft_core, "ft_sm": ft_sm, "rt_smsp": rt_smsp, "rt_orderfree_smsp": rt_orderfree_smsp,
            "counters_sm": counters_sm, "per_sm": per_sm,
            "pouw_ft_sm": d["cores_per_sm"] * POUW_FT_UNIT}


def rates(dev: str, seg_absorbs: int = 1 << 14) -> dict:
    d = DEVICES[dev]
    ft_rounds = d["sms"] * d["cores_per_sm"] * d["tensor_hz"] * d["words_per_core_clk"] / 32
    words = ft_rounds * 32
    rt_rounds = d["sms"] * d["smsp_per_sm"] * d["vector_hz"]          # full issue, a warp result a round
    segs = (ft_rounds + rt_rounds) / seg_absorbs
    sha_per_seg = 4                       # chain id, sequence, 1600-bit state, counts: ~2 blocks + chain start + tree
    sha_hz = 1.9e9 / 80                   # one round a clock at ~1.9 GHz [E]
    sha_engines = segs * sha_per_seg / sha_hz
    return {"tensor_words_per_s": words, "tensor_bytes_per_s": words * 4, "ft_rounds_per_s": ft_rounds,
            "rt_rounds_per_s": rt_rounds, "segments_per_s": segs, "digest_bytes_per_s": segs * 64,
            "sha512_engines": sha_engines, "port_bytes_per_window": 64 + 16 * 8 + 32 + 64,
            "seed_bytes_per_window": 64}


def energy(dev: str, en: dict) -> dict:
    """Upper bounds: all of 0.634 pJ/MAC above 100 W (OCC-RATE, H200 uniform s8) charged to MAC-datapath toggles."""
    d = DEVICES[dev]
    rand = en["streams"]["rand"]
    toggles_per_mac = rand["mac4"]["total"] / 4
    fj_per_toggle = 0.634e3 / toggles_per_mac
    # H200: ft_plane4 per 32 words; sm_120: one plane-limited round per 8 words, i.e. a round per 64 mac4-cycles
    ft_ratio = rand["ft4_over_mac_total"] if d["words_per_core_clk"] == 32 else 4 * rand["ft_over_mac_total"]
    ft_pj_per_mac = ft_ratio * 0.634
    rt_pj_per_warp = rand["ft_plane4"]["total"] * fj_per_toggle / 1e3
    rate = d["s8_per_sm_clk"] * d["sms"] * d["tensor_hz"]
    words_ratio = d["words_per_core_clk"] * 32 * d["cores_per_sm"] / d["s8_per_sm_clk"]   # 1 on H200
    ft_w = ft_pj_per_mac * words_ratio * rate * 1e-12
    rt_w = rt_pj_per_warp * d["sms"] * d["smsp_per_sm"] * d["vector_hz"] * 1e-12
    return {"fj_per_ge_toggle": fj_per_toggle, "ft_pj_per_mac": ft_pj_per_mac, "ft_w_full_tensor": ft_w,
            "ft_share_of_board": ft_w / d["board_w"], "rt_pj_per_warp_result": rt_pj_per_warp,
            "rt_w_full_issue": rt_w, "rt_share_of_board": rt_w / d["board_w"],
            "keyed_ft_over_mac": ft_ratio, "pouw_ft_over_mac": rand["ft_over_mac_total"],
            "rt_over_idp4a_warp": rand["rt4_over_idp4a_warp_total"],
            "llm_like": {"ft4_over_mac": en["streams"]["llm_like"]["ft4_over_mac_total"],
                         "rt4_over_idp4a_warp": en["streams"]["llm_like"]["rt4_over_idp4a_warp_total"]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=Path, required=True)
    ap.add_argument("--energy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    synth = json.loads(a.synth.read_text())
    en = json.loads(a.energy.read_text())
    out: dict = {}
    md: list[str] = []
    for dev, d in DEVICES.items():
        c = costs(dev, synth)
        r = rates(dev)
        e = energy(dev, en)
        lk = {ends: leakage(dev, ends) for ends in ("lo", "hi")}
        die_ge = {k: d["sms"] * c["per_sm"][k] + r["sha512_engines"] * 31000 + 50000 for k in c["per_sm"]}
        die_mm2 = {k: die_ge[k] * UM2_PER_GE[k] / 1e6 for k in die_ge}
        out[dev] = {"costs": c, "rates": r, "energy": e, "leakage": lk, "die_ge": die_ge, "die_mm2": die_mm2,
                    "die_share": {k: die_mm2[k] / d["die_mm2"] for k in die_mm2},
                    "pouw_ft_die_share": d["sms"] * c["pouw_ft_sm"] * UM2_PER_GE["central"] / 1e6 / d["die_mm2"]}
        md.append(f"## {dev}\n")
        md.append("| route (counted only) | s8-MAC-eq/SM-clk | share of s8 tensor | exact int8 | tag |")
        md.append("|---|---:|---:|---|---|")
        for name in lk["hi"]["alone"]:
            lo, hi = lk["lo"]["alone"][name], lk["hi"]["alone"][name]
            md.append(f"| {name} | {lo['macs_per_sm_clk']:.0f}–{hi['macs_per_sm_clk']:.0f} | "
                      f"{lo['share_of_tensor']:.2%}–{hi['share_of_tensor']:.2%} | {hi['exact_int8']} | {hi['tag']} |")
        md.append(f"| **best mix under issue** (exact only / all) | {lk['lo']['mix_macs_per_sm_clk']:.0f} / "
                  f"{lk['hi']['mix_macs_per_sm_clk']:.0f} | {lk['lo']['mix_share']:.1%} / {lk['hi']['mix_share']:.1%}"
                  f" | | E |\n")
        md.append(f"mix (all): {lk['hi']['mix']}\n")
        md.append("| block | low | central | high |")
        md.append("|---|---:|---:|---:|")
        for key, label in [("ft_core", "FT for governance, a tensor core (GE)"),
                           ("rt_smsp", "RT, a sub-partition (GE)"), ("per_sm", "all logging, an SM (GE)")]:
            v = c[key]
            md.append(f"| {label} | {v['low']:,.0f} | {v['central']:,.0f} | {v['high']:,.0f} |")
        md.append(f"| die (GE) | {die_ge['low']/1e6:.1f}M | {die_ge['central']/1e6:.1f}M | {die_ge['high']/1e6:.1f}M |")
        md.append(f"| die (mm², at {UM2_PER_GE['low']}/{UM2_PER_GE['central']}/{UM2_PER_GE['high']} µm²/GE) | "
                  f"{die_mm2['low']:.2f} | {die_mm2['central']:.2f} | {die_mm2['high']:.2f} |")
        md.append(f"| share of {d['die_mm2']} mm² | {die_mm2['low']/d['die_mm2']:.2%} | "
                  f"{die_mm2['central']/d['die_mm2']:.2%} | {die_mm2['high']/d['die_mm2']:.2%} |\n")
        md.append(f"keyed absorb unit {c['keyed_unit']:,.0f} GE = {c['keyed_ge_per_mac']:.1f} GE/MAC = "
                  f"{c['keyed_share_s8']:.2%} of an s8 MAC ({c['keyed_share_fp8']:.2%} of FP8); PoUW's FT "
                  f"{c['pouw_ft_share_s8']:.2%}\n")
        md.append(f"order-free RT a sub-partition: {c['rt_orderfree_smsp']:,.0f} GE; counters an SM "
                  f"{c['counters_sm']:,.0f} GE; PoUW's FT alone {out[dev]['pouw_ft_die_share']:.2%} of the die\n")
        md.append(f"rates: {json.dumps({k: float(f'{v:.3g}') for k, v in r.items()})}\n")
        md.append(f"energy: {json.dumps({k: (float(f'{v:.3g}') if isinstance(v, float) else v) for k, v in e.items()})}\n")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "gov_price.json").write_text(json.dumps(out, indent=1, default=float))
    (a.out / "gov_price.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
