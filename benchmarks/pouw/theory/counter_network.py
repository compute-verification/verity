"""COUNTER (LEAN program): NETWORK's stubs at ebe1fcd968 (`Proofs/Pouw/Lead/Network/Term.lean`), checked exactly.

- `lead_cells_rot`: `K ≤ 21 + 28 (S − 1)` for every K, and the step's lanes, coins and the rest of its coin term as
  the stub states them, on the port (`counter_pad`'s `_word_scalar`) at random coins.
- `networkGap_le`, `networkGap_le320`: `nMax · wMac · callBits / 32` against 0.036% (0.36%) of stock over a scan of
  calls, with `wMac = honest / stock` bounded in closed form.
- `forward_sum`: Llama-3.2-1B's calls at `T` rows sum to `cap · T` bits, and what a column count off a multiple of 8
  does to the hypothesis.
- `llama_forward`: the four linears at every `m` up to the bound given, against 0.0362% (0.362%).
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as Q

import numpy as np

import counter_pad as cp
import counter_statement as cs

LLAMA = [(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)]


def call_bits(lay: dict) -> Q:
    return Q(lay["cap"] * lay["rows"] * lay["n"] * lay["K"], cp.FWD_MACS)


def cells_rot(k_max: int, seed: int) -> dict:
    bad_k = [k for k in range(k_max) if k > 21 + 28 * (cs.steps_for(k) - 1)]
    rng = np.random.default_rng(seed)
    lay = cp.layout(1, 130, 12, 21 + 28 * 3)
    om = cs.draw(lay, rng)
    mism = 0
    for _ in range(300):
        i, t, j = int(rng.integers(lay["m"])), int(rng.integers(lay["S"])), int(rng.integers(lay["n"]))
        k = i // 64
        s = [int(om["s"][t, k, ell, j]) for ell in range(3)]
        lane = 127 * s[0] + s[1] + s[2] if i % 2 == 0 else s[0] + 127 * s[1] + s[2]
        step = cp._word_scalar(lay, om, i, t, j) - (cp._word_scalar(lay, om, i, t - 1, j) if t else -cs.bias(lay["S"], j))
        rest = (sum(cs.row_tag(i)[q] * int(om["G"][q, j]) + int(om["a1"][i, q]) * cs.col_code(j)[q] for q in range(4))
                if t == 0 else 1 + int(om["b"][i, t]))
        odd = all(abs(x) <= 127 and x % 2 == 1 for x in s)
        mism += int(step != rest + lane or not odd or k >= cs.tiles(lay))
    return {"K_checked": k_max, "K_failing": bad_k[:10], "entries": 300, "mismatches": mism,
            "holds": not bad_k and mism == 0}


def wmac_closed(m: int, rows: int, n_min: int = 4096, k_min: int = 4096) -> Q:
    sk = Q(1, 28) + Q(34, 28 * k_min)
    return (32 * sk * (1 + Q(351538, 3200)) + Q(351538, 100) * 32 * (sk / 65536 + Q(1, m * n_min * k_min))
            + Q(344905, 100) * Q(rows, 4 * n_min * m))


def network_le(ms: list[int], ns: list[int], ks: list[int]) -> dict:
    out = {}
    for cap, lim in ((32, Q(36, 100000)), (320, Q(36, 10000))):
        worst, arg, wmax = Q(0), None, Q(0)
        for m in ms:
            for n in ns:
                for k in ks:
                    lay = cp.layout(1, m, n, k, cap)
                    g = cp.gap(lay)
                    wmac = g["honest"] / g["stock"]
                    net = cp.N_MAX * wmac * call_bits(lay) / 32
                    assert net == g["terms"]["network"]
                    sh = net / g["stock"]
                    if sh > worst:
                        worst, arg = sh, (m, n, k)
                    wmax = max(wmax, wmac)
        closed_w = wmac_closed(16, 16)
        closed = cp.N_MAX * cap * closed_w / cp.FWD_MACS
        out[f"cap{cap}"] = {"claimed": float(lim), "scan_max": float(worst), "at_m_n_K": arg,
                            "wMac_scan_max": float(wmax), "wMac_closed": float(closed_w),
                            "closed": float(closed), "holds": bool(worst <= lim and closed <= lim)}
    return out


def forward_sum(ts: list[int]) -> dict:
    out = {}
    for t in ts:
        calls = [(t, n, k) for n, k in LLAMA] * 16
        macs = sum(r * cp.pad_cols(n) * k for r, n, k in calls)
        b32 = sum(call_bits(cp.layout(t, r, n, k, 32)) for r, n, k in calls)
        b320 = sum(call_bits(cp.layout(t, r, n, k, 320)) for r, n, k in calls)
        out[f"T{t}"] = {"hypothesis": macs == t * cp.FWD_MACS, "bits32": str(b32), "bits320": str(b320),
                        "holds": macs != t * cp.FWD_MACS or (b32 == 32 * t and b320 == 320 * t)}
    odd = [(1, 3073, 2048), (1, 2047, 2048), (1, 16384, 2048), (1, 2048, 8192)] * 16
    real = sum(r * n * k for r, n, k in odd)
    padded = sum(r * cp.pad_cols(n) * k for r, n, k in odd)
    out["off_multiple_of_8"] = {"calls": odd, "macs_unpadded": real, "macs_padded": padded,
                                "bits32_over_unpadded_cap": float(Q(32 * padded, cp.FWD_MACS) / Q(32 * real, cp.FWD_MACS))}
    return out


def llama_forward(m_max: int) -> dict:
    out = {}
    for cap, lim in ((32, Q(362, 1000000)), (320, Q(362, 100000))):
        worst, arg, worst_call = Q(0), None, Q(0)
        for m in range(1, m_max + 1):
            gs = [cp.gap(cp.layout(1, m, n, k, cap)) for n, k in LLAMA]
            sh = sum(g["terms"]["network"] for g in gs) / sum(g["stock"] for g in gs)
            worst_call = max(worst_call, *(g["terms"]["network"] / g["stock"] for g in gs))
            if sh > worst:
                worst, arg = sh, m
        out[f"cap{cap}"] = {"claimed": float(lim), "m_range": [1, m_max], "forward_max": float(worst),
                            "at_m": arg, "worst_single_call": float(worst_call), "holds": bool(worst <= lim)}
    return out


def llama_gap() -> dict:
    """`gapShare` of Llama-3.2-1B's own calls (K = 2,048 on three of them, below `GapAt`'s range) at the points' rows
    and at decode, against the points' claims."""
    out = {}
    for name, (t, m, lim32, lim320) in [*cp.GAP_POINTS.items(), ("decode", (1, 1, Q(232, 100000), Q(252, 100000)))]:
        for cap, lim in ((32, lim32), (320, lim320)):
            gs = {f"{n}x{k}": cp.gap(cp.layout(t, m, n, k, cap)) for n, k in LLAMA}
            fwd = sum(g["gap"] for g in gs.values()) / sum(g["stock"] for g in gs.values())
            out[f"{name}/cap{cap}"] = {"claimed_at_n_K_4096": float(lim), "forward": float(fwd),
                                       "calls": {c: float(g["gapShare"]) for c, g in gs.items()},
                                       "within_claim": bool(fwd <= lim)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k-max", type=int, default=200000)
    ap.add_argument("--m-max", type=int, default=8192)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cp.use_head("ebe1fcd968")
    rep = {"port": cs.check_port()}
    assert rep["port"]["mismatches"] == 0
    rep["lead_cells_rot"] = cells_rot(args.k_max, 7)
    print("lead_cells_rot:", rep["lead_cells_rot"], flush=True)
    ms = [*range(1, 34), 48, 63, 64, 65, 512, 4096, 8192, 8193]
    ns = [4096, 4097, 4103, 4104, 8192, 16384]
    ks = [*range(4096, 4096 + 29), 4110, 8192, 65536]
    rep["networkGap_le"] = network_le(ms, ns, ks)
    print("networkGap_le:", rep["networkGap_le"], flush=True)
    rep["forward_sum"] = forward_sum([1, 7, 16, 512, 8192])
    print("forward_sum:", rep["forward_sum"], flush=True)
    rep["llama_forward"] = llama_forward(args.m_max)
    print("llama_forward:", rep["llama_forward"], flush=True)
    rep["llama_gap"] = llama_gap()
    for key, v in rep["llama_gap"].items():
        print(f"llama gapShare {key}:", v, flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1, default=str)


if __name__ == "__main__":
    main()
