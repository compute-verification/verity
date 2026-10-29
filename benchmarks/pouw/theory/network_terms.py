"""HASH: the network term of every fold or sketch design, at egress caps of 32 and 320 bits a token.

The term is the lead's (`Pouw.Lead.networkGap`): within a forward's cap, an outsourced block needs every coin bit its
words read, so the bits bound the MACs an outside machine can do (n_max MACs a coin bit), and each outsourced MAC takes
the credit the design gives a MAC (W_mac, W1 units). Its share of stock is n_max · W_mac · cap / fwdMacs, on
Llama-3.2-1B's linears (`fwdMacs` = 973,078,528 MACs a token).

- **n_max is the coin layout's, not the seal's.** What an outsourced block must receive is its coins; what it returns is
  one tree root for any number of leaves (in M0 the served bytes needn't return). So a design that keeps the lead's
  coin lanes (21 bits a 64-row tile, step and column, 28 data lanes) keeps n_max = 256/3 (`lead_cells_macs_le`), and a
  fold or sketch changes only W_mac. Mirror-π's row coins (a byte a row, step and lane; column weights a byte a column,
  step and lane, shared by every row; the replay half drawing none) carry (32 − ℓ)/(4ℓ) · m·N/(m + N) MACs a bit on an
  m × N block, and (32 − ℓ)·64/(4ℓ) with a row coin per 64-column tile.
- **The hashing route.** An outside hasher sent a register's bits returns its compressions' credit: 3,515.38 · 32 / 1,024
  W1 a bit at ShaWrites' price (109.9), below every design's coin route here.
- **W_mac** is priced on the Lean basis (ShaWrites' registers a compression, `Pouw.Lead.honestRegs`) for the lead and the
  thin fold, call by call; the other designs' W_mac is 32S/K for the words plus their lane's published h.

Run: `python network_terms.py [--out DIR]`. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

SHA_WORDS = 3515.38          # registers a 128-byte words compression (`shaWordsRegs`)
SHA_A = 3449.05              # registers an A compression (`shaARegs`)
W1_REG = 32                  # W1 units a register (H32)
FWD_MACS = 973_078_528       # Llama-3.2-1B's linears, MACs a token (`Layout.fwdMacs`)
CALLS = [("qkv", 3072, 2048), ("o", 2048, 2048), ("gate_up", 16384, 2048), ("down", 2048, 8192)]   # (n, K), 16 layers
N_LEAD = 256 / 3
CAPS = (32, 320)
H_LEAD_BRIEF = (120.0, 230.0)

#: the per-period basis (N-CARD's, as ASSUME's interim): H200 int8 at 40% MFU, ANDs a MAC, 600-s windows
H200_MACS_S = 9.89e14 * 0.40
ANDS_A_MAC = 129.94
WINDOW_S = 600
PERIODS = {"window": 1, "week": 7 * 144, "month": 30 * 144, "year": 365 * 144}
DATACENTER = 100_000
WORLD_MACS_YEAR = 4.68e30


def steps(K: int) -> int:
    """`Pouw.Lead.stepsFor`: 21 data lanes at step 0 and 28 at each later step."""
    return 1 + math.ceil((K - 21) / 28)


def call_macs(n: int, K: int) -> int:
    return n * K


def hash_word(K: int) -> float:
    """W1 a MAC to hash one 32-bit word a cell at ShaWrites' price."""
    return SHA_WORDS / K


def a_hash(n: int) -> float:
    """W1 a MAC to hash A's rows (rows · K / 128 compressions over m · n · K MACs, rows = m)."""
    return W1_REG * SHA_A / (128 * n)


def lead_w(n: int, K: int) -> dict:
    S = steps(K)
    return {"words": W1_REG * S / K, "hash": SHA_WORDS * S / K, "A": a_hash(n)}


def thin_w(n: int, K: int, sealed_words: float) -> dict:
    """The thin fold: the words (FoldMaterial, a write a step), the fold (ThinFoldOpt, ½ an instruction a word), the seal
    (`sealed_words` words hashed a cell: σ and O′ a cell is 2; σ a row of 8 with each O′ is 1.125), and A."""
    S = steps(K)
    return {"words": W1_REG * S / K, "fold": 16 * S / K, "seal": sealed_words * hash_word(K), "A": a_hash(n)}


def total(parts: dict) -> float:
    return sum(parts.values())


def share(n_max: float, w_mac: float, cap: float) -> float:
    return n_max * w_mac * cap / FWD_MACS


def mirror_nmax(ell: int, m: int, N: int) -> float:
    """Mirror-π's coins on an m × N block over all S steps: a byte a (row, step < S/2, lane) and a byte a (column,
    step < S/2, lane); the replay half reads the same coins. MACs (32 − ℓ)·S·m·N over 8ℓ·(S/2)·(m + N) bits."""
    return (32 - ell) * m * N / (4 * ell * (m + N))


def mirror_tile_nmax(ell: int) -> float:
    """The same with the row coin drawn per 64-column tile: (32 − ℓ)·64 MACs a row over 4ℓ bits, as rows grow."""
    return (32 - ell) * 64 / (4 * ell)


def per_period(sh: float) -> dict:
    window_ands = H200_MACS_S * WINDOW_S * ANDS_A_MAC
    world_window = WORLD_MACS_YEAR * ANDS_A_MAC / PERIODS["year"]
    out = {}
    for scope, base in (("H200", window_ands), ("datacenter", window_ands * DATACENTER), ("world", world_window)):
        out[scope] = {p: sh * base * k for p, k in PERIODS.items()}
    return out


def designs() -> list[dict]:
    rows = []
    mac_w = {name: call_macs(n, K) for name, n, K in CALLS}
    tot = sum(mac_w.values())

    def lean_row(name, fn, premise, rank):
        per_call = {c: total(fn(n, K)) for c, n, K in CALLS}
        parts = {c: {k: round(v, 4) for k, v in fn(n, K).items()} for c, n, K in CALLS}
        w_max = max(per_call.values())
        w_avg = sum(per_call[c] * mac_w[c] for c in per_call) / tot
        rows.append({"design": name, "n_max": N_LEAD, "n_max_source": "the lead's coin lanes (proved)",
                     "w_mac_by_call": {c: round(v, 4) for c, v in per_call.items()}, "parts_by_call": parts,
                     "w_mac_max": w_max, "w_mac_mac_weighted": w_avg, "premise": premise, "rank": rank,
                     "basis": "Lean (ShaWrites registers)"})

    lean_row("lead (every word hashed)", lead_w, "ShaWrites", "4a")
    lean_row("thin fold, seal (σ, O′) a cell", lambda n, K: thin_w(n, K, 2.0), "FoldMaterial, ThinFoldOpt", "5")
    lean_row("thin fold, seal σ a row of 8 cells and each O′", lambda n, K: thin_w(n, K, 1.125),
             "FoldMaterial, ThinFoldOpt", "5")
    lean_row("thin fold as ASSUME priced (σ alone a cell)", lambda n, K: thin_w(n, K, 1.0),
             "FoldMaterial, ThinFoldOpt", "5")
    lean_row("thin fold as ASSUME priced (σ alone a row of 8)", lambda n, K: thin_w(n, K, 0.125),
             "FoldMaterial, ThinFoldOpt", "5")

    words = W1_REG * steps(2048) / 2048
    for name, h, nm, src, prem, rank in (
            ("multiply–xor fold (N-FLOOR, c4, 8-bit row coin)", (2.94, 2.94), N_LEAD,
             "the lead's coin lanes; the row coin adds bits", "FOLD-OPT", "4"),
            ("xor-16 (N-SHALB)", (8.22, 8.22), N_LEAD, "the lead's coin lanes", "XorWrites_16", "open"),
            ("mirror-π quad-l4-w4, row coins per 64-column tile", (2.37, 4.55), mirror_tile_nmax(4),
             "a byte a (row, step < S/2, lane, 64-column tile)", "SketchWrites2", "4a+"),
            ("mirror-π quad-l3-w4, row coins per 64-column tile", (2.37, 4.55), mirror_tile_nmax(3),
             "a byte a (row, step < S/2, lane, 64-column tile)", "SketchWrites2", "4a+")):
        rows.append({"design": name, "n_max": nm, "n_max_source": src, "h": list(h),
                     "w_mac_max": words + h[1], "w_mac_min": words + h[0], "premise": prem, "rank": rank,
                     "basis": "32S/K + the lane's published h"})

    for m in (64, 512, 8192, 65536):
        nm_call = {c: mirror_nmax(4, m, n) for c, n, K in CALLS}
        nm_max = max(nm_call.values())
        nm_avg = sum(nm_call[c] * mac_w[c] for c in nm_call) / tot
        rows.append({"design": f"mirror-π quad-l4-w4 as published, {m} rows a call", "n_max": nm_max,
                     "n_max_mac_weighted": nm_avg, "n_max_by_call": {c: round(v, 1) for c, v in nm_call.items()},
                     "n_max_source": "a byte a (row, step < S/2, lane); column weights shared by every row",
                     "h": [2.37, 4.55], "w_mac_max": words + 4.55, "w_mac_min": words + 2.37,
                     "premise": "SketchWrites2", "rank": "4a+", "basis": "32S/K + the lane's published h"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = designs()
    hash_route = SHA_WORDS * W1_REG / 1024
    fails = []
    for r in rows:
        lo = r.get("w_mac_min", r.get("w_mac_mac_weighted"))
        hi = r["w_mac_max"]
        r["share"] = {str(c): [share(r["n_max"], lo, c), share(r["n_max"], hi, c)] for c in CAPS}
        if "n_max_mac_weighted" in r:
            r["share_mac_weighted_bits"] = {str(c): [share(r["n_max_mac_weighted"], lo, c),
                                                     share(r["n_max_mac_weighted"], hi, c)] for c in CAPS}
        r["w1_a_bit"] = [r["n_max"] * lo, r["n_max"] * hi]
        r["coin_route_above_hash_route"] = r["n_max"] * lo >= hash_route
        r["per_period_ands_cap32_high"] = per_period(r["share"]["32"][1])
        r["per_period_ands_cap320_high"] = per_period(r["share"]["320"][1])
    lead = rows[0]
    if not lead["w_mac_max"] <= 128.69 or not lead["share"]["32"][1] <= 3.62e-4:
        fails.append(f"the lead's W_mac {lead['w_mac_max']:.3f} or share {lead['share']['32'][1]:.3e} exceeds the "
                     "proved 128.69 / 0.0362% (`llama_betaLead_le`, `llama_forward`)")
    res = {"fwd_macs": FWD_MACS, "caps_bits_a_token": list(CAPS), "hash_route_w1_a_bit": hash_route,
           "steps": {str(K): steps(K) for K in (2048, 8192)}, "designs": rows, "fails": fails}
    text = json.dumps(res, indent=1)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "network_terms.json"), "w") as f:
            f.write(text)
    print(f"{'design':58s} {'n_max':>8s} {'W_mac':>15s} {'cap 32':>21s} {'cap 320':>21s}")
    for r in rows:
        lo = r.get("w_mac_min", r.get("w_mac_mac_weighted"))
        s32, s320 = r["share"]["32"], r["share"]["320"]
        print(f"{r['design'][:58]:58s} {r['n_max']:8.1f} {lo:7.2f}–{r['w_mac_max']:<7.2f} "
              f"{100 * s32[0]:9.5f}–{100 * s32[1]:<9.5f}% {100 * s320[0]:9.4f}–{100 * s320[1]:<9.4f}%")
    print("FAIL" if fails else "OK", len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
