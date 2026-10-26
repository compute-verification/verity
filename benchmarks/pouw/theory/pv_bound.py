"""ACCEL ALGO-COST-2 (CMP-114): PV with O bound (R-ACCEL-8), priced at A = B = 0 on both cards.

Under R-ACCEL-8 attention is credited only as PV with O bound to the words, so PV is a v2 round of its own: K the
attended keys, N = d_v, m the query rows sharing V (GQA group × prefill chunk), words every L keys, O the last word.
`nofold.v2_round` is general in (K, N, m), so this prices PV through it. The routes the prover at A = B = 0 has (PD2-1's
O-route bcᵀ in place of the peels, the zero high digits of Y_W, the one-segment skip, the tiles' cross-term route, and
Four-Russians lookups on the bit terms, A15) all save tensor time; ε_alg is the saving that fits the tensor pipe's
slack over the ALU without wraps (`eps_ovl`). PV is ALU-bound at every shape (hashing a cell's words and O over only K
MACs, and P's dither at 1/d_v bits a MAC), so the slack is 0.

Needs `nofold.py` from cursor/compute-research-accel-proto-design-ea24 (070e970980):

    git show origin/cursor/compute-research-accel-proto-design-ea24:benchmarks/pouw/accel_score/nofold.py > /tmp/nofold.py
    uv run python benchmarks/pouw/theory/pv_bound.py /tmp/nofold.py
"""

from __future__ import annotations

import importlib.util
import math
import sys

CONTEXTS = (1024, 4096, 8192, 16384, 32768, 65536, 131072)


def load(path: str):
    spec = importlib.util.spec_from_file_location("nofold", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pv_round(n, card: str, K: int, d_v: int, group: int, budget: float = 0.0) -> dict:
    return n.v2_pick(card, K, d_v, 16, budget, tile_set=(n.TILES[card],), overlap=True, m=n.M_PREFILL * group)


def world(n) -> None:
    print("\nPV-bound words at the world model (COVER-ATTN's 4,500-token request), honest picks at ε_alg budget 0")
    for card in n.V2_CARDS:
        t = n.TILES[card]
        for mdl in n.WORLD:
            hist, st, qk, pv, ix = n.world_macs(mdl)
            bins: dict[int, float] = {}
            for keys, rows in hist.items():
                k = max(64 * t, 64 * math.ceil(keys / 64))
                bins[k] = bins.get(k, 0.0) + rows * keys
            W = G = T = A = S = 0.0
            least = (math.inf, 0)
            for k, w in bins.items():
                r = pv_round(n, card, k, mdl["dv"], mdl["hq"] // mdl["hkv"])
                W, G, T, A, S = W + w, G + w * r["gamma_max"], T + w * r["T"], A + w * r["A_nw"], S + w * r["saved"]
                least = min(least, (r["A_nw"] / r["T"], k))
            print(f"  {card:13} {mdl['name']:16} PV {pv / (st + qk + pv + ix):6.2%} of MACs: γ {G / W:5.2f} (tensor "
                  f"{T / W:.2f}, ALU without wraps {A / W:.2f}); routes' tensor saving {S / W:.2f}; least lane load "
                  f"{least[0]:.2f} at K {least[1]}; ε_alg 0")


def contexts(n) -> None:
    print("\nPV-bound words by context: honest pick at budget 0 (lane load, γ) and the leak at the least-γ pick")
    for card in n.V2_CARDS:
        for name, d_v, group in (("Llama d_v 128 GQA 4", 128, 4), ("Llama d_v 128 GQA 8", 128, 8),
                                 ("V4 d_v 512 MQA 128", 512, 128)):
            cells = []
            for K in CONTEXTS:
                r, f = pv_round(n, card, K, d_v, group), pv_round(n, card, K, d_v, group, budget=1.0)
                cells.append(f"{K // 1024}k {r['A_nw'] / r['T']:.2f}/{r['gamma_max']:.2f}/{f['eps_ovl']:.1%}")
            print(f"  {card:13} {name:20} " + "  ".join(cells))


def main(argv: list[str]) -> None:
    n = load(argv[1] if len(argv) > 1 else "/tmp/nofold.py")
    world(n)
    contexts(n)


if __name__ == "__main__":
    main(sys.argv)
