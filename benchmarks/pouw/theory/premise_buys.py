#!/usr/bin/env python3
"""What each accelerator premise buys NCP's T1 gap on the H200 (the PoUW sweep's SW-PREMISES table).

T1 (inbox CORE.md §2a): cost(H) − min cost(adversary) ≤ 1% of cost(stock), in priced work (W1 units: one dense int8 MAC is
one unit, every instruction its price a register written), hashing credited under ShaSpeed, H the best honest program.
Today's bound is DistinctWritesMH100 under H32 (`Pouw/Assumptions/Dimension/H100W.lean`): the adversary pays at least 32
a committed, pairwise-distinct, noise-dependent 32-bit word, and nothing else. NCP commits a word every k32 of its
3K-wide stacked product X·Yᵀ, so every gap term is honest work that bound doesn't force:

- `final_word`: each dot product's last word is A·Bᵀ, salt-independent, free in MH100w's free set: 32·Σmn / ΣmnK.
- `forming`: X's 3mK and Y's 3nK entries at p units an entry (p = 8: one packed add, four entries a register at H32's
  32), uncharged because MH100w's wgmma writes any function and reads no operand: 3p·Σ(mK + nK) / ΣmnK.
- `derivation`: each noise entry is (XOF byte & 63) − 31 (`PouwNoiseByte`), one LOP3 a packed register of E₁ and F₁:
  8·Σ(mK + nK) / ΣmnK (an estimate of honest's instruction count).
- `before_noise`: A's and B's commitment is hashed before the call's noise, free in the same model: 0.42–0.72× stock
  (an estimate carried from the core's smell check, not computed here).
- `decode`: below N 64 a k32 register costs honest more than H32's 32 (`DECODE_PRICE`), while the floor stays 32.
- `uncommitted(L)`: committing a word every L > 32 leaves L/32 − 1 of every L/32 tensor writes unforced: 3(1 − 32/L).
  With every instruction class semantic and algebraic sufficiency at L (rank 3), the counting sketch credits a running
  word 64 − 1024/L of honest's L (`uncommitted_semantic(L)`); only a matmul lower bound credits all of L (rank 4 to 5).
- `sign_fix`: the u8 × s8 wgmma takes one operand signed, and one IADD3 can't form signed bytes of B + F (the carry out
  of a byte depends on its noise for 63 of the 127 int7 values), so honest writes that operand's registers twice where
  a forced-write bound charges once. Per linear honest takes the cheaper route: U, Y signed, 3p·nK more; or U′, X
  signed, 3p·mK more plus the row init's sum of A, p·mK (an estimate). No premise set closes it.

The premise sets are cumulative (`SETS`): S0 today; S1 adds the atomic call (the free values are the call's own words,
not their span; with every device of the period counted); S2 makes every instruction class semantic: TensorSemantics
(a tensor instruction computes the captured exact product of its operand tiles, output (i, j) from row i and column j
only, at its measured price at its shape), lane semantics with fan-in, oblivious memory moves and algebraic
sufficiency, with the forming counting lemma (the tensor class alone buys nothing: one `other` instruction still writes
any word at 32); S3 adds the one-instruction minimality of the noise's derivation (a CPU check over the lane
instructions' semantics) with its sharing lemma. Each set's residual is the sum of the terms it leaves open, at every
shape (`SHAPES`) and T.

Committing closes the same two terms without a premise (a committed, salt-moving register that equals no coin word is
a DistinctWrites target): X and Y in place of TensorSemantics' counting lemma, E₁ and F₁ in place of
OneInstructionMinimal. `commit_instead_hashing` is the credited hashing each adds, 3 and 1 bytes an operand entry at
`HASH_BYTE`.

T2 at interval L scales hashing's share: T2(L) = T2(32)·(1 − h + h·32/L). ShaSpeed's tolerance is the relative error in
the hashing price that keeps its slack within T1's 1%: δ = 1% / (hashing units a useful MAC), times L/32.

The per-period totals scale the shares to one H200, the reference datacenter (100,000 H200s) and the world, at
CORE-TIME's stock-work-covered figures (`PERIOD_STOCK`, AND gates); only the shares are this script's arithmetic.

  uv run --no-project --with-editable benchmarks/pouw/exact_kit python -B benchmarks/pouw/theory/premise_buys.py
      [--json OUT] [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import json
import sys

from exact_kit.checks import Checks

FLOOR = 32
P_FORM = 8.0
P_DERIVE = 8.0
#: Llama-3.2-1B's block linears (name, n, K), formed apart and fused as vLLM serves them.
LLAMA_APART = (("q", 2048, 2048), ("k", 512, 2048), ("v", 512, 2048), ("o", 2048, 2048), ("gate", 8192, 2048),
               ("up", 8192, 2048), ("down", 2048, 8192))
LLAMA_FUSED = (("qkv", 3072, 2048), ("o", 2048, 2048), ("gate_up", 16384, 2048), ("down", 2048, 8192))
#: A DeepSeek-V4-like layer's linears (estimate): a dense K = n = 7,168 projection, and a routed expert's fused gate-up
#: (n 4,096, K 7,168) and down (n 7,168, K 2,048), each expert seeing T·8/256 of T tokens.
V4_DENSE = (("dense", 7168, 7168),)
V4_EXPERT = (("expert_gate_up", 4096, 7168), ("expert_down", 7168, 2048))
EXPERT_SHARE = 8 / 256
SHAPES = {"llama-3.2-1b apart": (LLAMA_APART, 1.0), "llama-3.2-1b fused": (LLAMA_FUSED, 1.0),
          "v4-like dense": (V4_DENSE, 1.0), "v4-like routed expert": (V4_EXPERT, EXPERT_SHARE)}
TS = (8192, 512, 64, 8)
#: Honest's least price a k32 register below 64 tokens (CORE-H32's H200 table): mma.sync m16n8k32 with the tokens on
#: its N 8, 48.07 (wgmma m64n8k32 would be 144), and below 8 tokens the same tile padded, 48.07·8/m a useful word. At
#: 64 tokens and more, wgmma's 32. The decode term is a share of stock's mnK, like the others.
DECODE_PRICE = 48.07
DECODE_N = 8
#: Hashing-before-noise (estimate, a share of stock).
BEFORE_NOISE = (0.42, 0.72)
#: Hashing's priced units a useful MAC at k32 words (estimate), and its share of t_H, with T2 at T 8,192 (sync, async).
HASH_UNITS = (670.0, 800.0)
HASH_SHARE = 0.984
T2_K32 = {"sync": 155.0, "async": 186.0}
LS = (32, 64, 128, 256)
#: Credited hashing's price a byte (W1 units): hm96-sha512 rows beside the served GEMM (measured, SW-GPU) and HASH-38's
#: bar (an estimate). Committing X and Y instead of TensorSemantics' counting lemma, or E₁ and F₁ instead of
#: OneInstructionMinimal, hashes 3 and 1 bytes an entry of A and B.
HASH_BYTE = (1369.0, 1774.85)
#: Stock work covered, AND gates, a window (600 s), week, month (30 d) and year, at NCP's tensor peak (CORE-TIME).
PERIOD_STOCK = {"one H200": (2.57e19, 2.59e22, 1.11e23, 1.35e24),
                "datacenter (100,000 H200)": (2.57e24, 2.59e27, 1.11e28, 1.35e29),
                "world": (3.86e27, 3.89e30, 1.67e31, 2.03e32)}
PERIODS = ("window", "week", "month", "year")

#: The premises each set adds, and the terms it closes (cumulative).
SETS = {
    "S0 today (H32, DistinctWrites)": ((), ()),
    "S1 + atomic call": (("atomic call", "PeriodInventory"), ("final_word", "before_noise")),
    "S2 + every class semantic": (("TensorSemantics", "LaneSemantics", "FanIn", "MemoryMovesOblivious",
                                   "algebraic sufficiency", "forming counting lemma"), ("forming", "decode")),
    "S3 + minimal derivation": (("OneInstructionMinimal", "sharing lemma"), ("derivation",)),
}


def terms(linears, share: float, t: int, p_form: float = P_FORM, p_derive: float = P_DERIVE) -> dict[str, float]:
    m = max(1.0, t * share)
    stock = sum(m * n * k for _, n, k in linears)
    operands = sum(m * k + n * k for _, n, k in linears)
    price = DECODE_PRICE * max(1.0, DECODE_N / m) if m < 64 else None
    return {"m": m, "operand_bytes": operands / stock,
            "final_word": FLOOR * sum(m * n for _, n, _ in linears) / stock,
            "forming": 3 * p_form * operands / stock,
            "derivation": p_derive * operands / stock,
            "before_noise": BEFORE_NOISE[0],
            "decode": 3 * (price - FLOOR) / FLOOR if price else 0.0,
            "sign_fix": sum(min(3 * p_form * n * k, 4 * p_form * m * k) for _, n, k in linears) / stock}


def residual(t: dict[str, float], closed: set[str], noise: bool = True) -> float:
    keys = ("final_word", "forming", "derivation", "before_noise", "decode", "sign_fix")
    return sum(t[k] for k in keys if k not in closed and (noise or k != "before_noise"))


def uncommitted(L: int) -> float:
    return 3 * (1 - FLOOR / L)


def uncommitted_semantic(L: int) -> float:
    return 3 * (1 - (2 * FLOOR - FLOOR * FLOOR / L) / L)


def t2(L: int, base: float) -> float:
    return base * (1 - HASH_SHARE + HASH_SHARE * FLOOR / L)


def sha_tolerance(L: int, h: float) -> float:
    return 0.01 / h * L / FLOOR


def main() -> int:
    checks = Checks()
    out = checks.args[checks.args.index("--json") + 1] if "--json" in checks.args else None

    table: dict[str, dict] = {}
    for shape, (linears, share) in SHAPES.items():
        for t in TS:
            tm = terms(linears, share, t)
            closed: set[str] = set()
            row = {"terms": tm, "residual": {}}
            for name, (_, closes) in SETS.items():
                closed |= set(closes)
                row["residual"][name] = residual(tm, closed)
            table[f"{shape}, T {t}"] = row

    la, lf = table["llama-3.2-1b apart, T 8192"]["terms"], table["llama-3.2-1b fused, T 8192"]["terms"]
    checks.check("the final word over Llama-3.2-1B's linears is 32·Σn / ΣnK = 1.239% at every T (CORE-TIME)",
                 all(abs(table[f"llama-3.2-1b apart, T {t}"]["terms"]["final_word"] - 32 * 23552 / 60817408) < 1e-12
                     for t in TS) and abs(la["final_word"] - 0.01239) < 5e-6, f"{la['final_word']:.4%}")
    l512, l64 = (table[f"llama-3.2-1b apart, T {t}"]["terms"] for t in (512, 64))
    checks.check("forming at p 8 is 1.10% at T 8,192, 5.50% at T 512 and 38.3% at T 64 apart, 0.86% fused (CORE-TIME, "
                 "CORE-SMELL)", round(la["forming"], 4) == 0.0110 and round(l512["forming"], 4) == 0.0550
                 and round(l64["forming"], 3) == 0.383 and round(lf["forming"], 4) == 0.0086,
                 f"{la['forming']:.2%}, {l512['forming']:.2%}, {l64['forming']:.2%}; fused {lf['forming']:.2%}")
    checks.check("the noise's derivation is 0.37% apart and 0.29% fused at T 8,192 and 12.8% at T 64 (CORE-SMELL C28)",
                 round(la["derivation"], 4) == 0.0037 and round(lf["derivation"], 4) == 0.0029
                 and round(l64["derivation"], 3) == 0.128,
                 f"{la['derivation']:.2%}, {lf['derivation']:.2%}, {l64['derivation']:.2%}")

    crossing = [b for b in range(-63, 64) if any((b - 31) % 256 + r >= 256 for r in range(64))
                and any((b - 31) % 256 + r < 256 for r in range(64))]
    checks.check("an add whose byte sum is ≡ B − 31 + r (mod 256) carries out for some r in 0..63 and not others for 63 "
                 "of the 127 int7 B, exactly [−32, 30]: those signed bytes take a second write",
                 crossing == list(range(-32, 31)), f"{len(crossing)} of 127")
    l512f, v4d = table["llama-3.2-1b fused, T 512"]["terms"], table["v4-like dense, T 8192"]["terms"]
    checks.check("the sign fix is route U′'s 32·ΣK/ΣnK = 0.754% over Llama-3.2-1B fused at T 512, and route U's 24/m = "
                 "0.293% on V4's dense linear at T 8,192",
                 abs(l512f["sign_fix"] - 32 * 14336 / 60817408) < 1e-12 and abs(v4d["sign_fix"] - 24 / 8192) < 1e-12,
                 f"{l512f['sign_fix']:.3%}, {v4d['sign_fix']:.3%}; Llama fused T 8,192 {lf['sign_fix']:.3%}")

    s = list(SETS)
    print("  (shape, T)                         m      final   forming  derive  decode  sign  |  S0       S1       S2"
          "       S3")
    for key, row in table.items():
        tm, r = row["terms"], row["residual"]
        print(f"  {key:34s} {tm['m']:6.0f} " + " ".join(f"{tm[k]:7.2%}" for k in ("final_word", "forming",
                                                                                   "derivation", "decode", "sign_fix"))
              + " | " + " ".join(f"{v:8.2%}" for v in r.values()))
    checks.check("today (S0) misses T1 at every shape and T: hashing before the noise alone is 42% of stock (estimate)",
                 all(row["residual"][s[0]] > 0.01 for row in table.values()))
    s1 = {k: row["residual"][s[1]] for k, row in table.items()}
    t1_s1 = sorted(k for k, v in s1.items() if v <= 0.01)
    checks.check("the atomic call (S1) alone meets T1 at no shape and T: forming, derivation and the sign fix stay "
                 "above 1% even on V4's dense K = n = 7,168 at T 8,192", t1_s1 == [],
                 f"{s1['v4-like dense, T 8192']:.2%} (V4 dense); Llama fused T 8,192 "
                 f"{s1['llama-3.2-1b fused, T 8192']:.2%}")
    s2 = {k: row["residual"][s[2]] for k, row in table.items()}
    t1_s2 = sorted(k for k, v in s2.items() if v <= 0.01)
    checks.check("with every class semantic (S2) T1 holds at T 8,192 on Llama-3.2-1B fused and apart and on V4's dense "
                 "linear; the derivation's 8/m still breaks it on a routed expert's 256 rows, at T 512's 64 rows and "
                 "at decode", t1_s2 == ["llama-3.2-1b apart, T 8192", "llama-3.2-1b fused, T 8192",
                                        "v4-like dense, T 8192"],
                 f"Llama fused T 8,192 {s2['llama-3.2-1b fused, T 8192']:.3%}, T 512 "
                 f"{s2['llama-3.2-1b fused, T 512']:.2%}; V4 expert T 8,192 {s2['v4-like routed expert, T 8192']:.2%}")
    s3 = {k: row["residual"][s[3]] for k, row in table.items()}
    over_s3 = sorted(k for k, v in s3.items() if v > 0.01)
    checks.check("with the derivation's minimality and its sharing lemma (S3) only the sign fix remains; it meets T1 "
                 "fused and on V4's shapes, but Llama-3.2-1B served apart (k and v at n 512) misses at T 512 and below",
                 all(v == table[k]["terms"]["sign_fix"] for k, v in s3.items())
                 and over_s3 == ["llama-3.2-1b apart, T 512", "llama-3.2-1b apart, T 64", "llama-3.2-1b apart, T 8"],
                 f"fused T 8,192 {s3['llama-3.2-1b fused, T 8192']:.3%}, T 512 {s3['llama-3.2-1b fused, T 512']:.3%};"
                 f" apart T 512 {s3['llama-3.2-1b apart, T 512']:.3%}")

    commit_alt = {key: {"X and Y": [3 * row["terms"]["operand_bytes"] * p for p in HASH_BYTE],
                        "E1 and F1": [row["terms"]["operand_bytes"] * p for p in HASH_BYTE]}
                  for key, row in table.items()}
    for key in ("llama-3.2-1b fused, T 8192", "llama-3.2-1b fused, T 512", "llama-3.2-1b fused, T 64"):
        print(f"  committing instead (credited hashing, W1 units a stock MAC), {key}: "
              + "; ".join(f"{k} {lo:.2f}–{hi:.2f}" for k, (lo, hi) in commit_alt[key].items()))
    ca = commit_alt["llama-3.2-1b fused, T 8192"]
    checks.check("committing X and Y instead of the forming counting lemma hashes 3·Σ(mK + nK)/ΣmnK bytes a MAC: "
                 "1.5–1.9 W1 units a MAC of credited hashing at T 8,192 (Llama fused), against about 670–800 at k32",
                 1.4 < ca["X and Y"][0] < ca["X and Y"][1] < 2.0 and ca["E1 and F1"][1] < 0.7,
                 f"X, Y {ca['X and Y'][0]:.2f}–{ca['X and Y'][1]:.2f}; E1, F1 {ca['E1 and F1'][0]:.2f}–"
                 f"{ca['E1 and F1'][1]:.2f}")

    unc = {L: uncommitted(L) for L in LS}
    checks.check("committing every L > 32 under H32 alone leaves 3(1 − 32/L) of stock unforced: 150% at L 64; "
                 "OwnProducts (rank 4) is what closes it", unc[32] == 0 and unc[64] == 1.5,
                 "  ".join(f"L {L} {v:.0%}" for L, v in unc.items()))
    unc_sem = {L: uncommitted_semantic(L) for L in LS}
    checks.check("with every class semantic and algebraic sufficiency (rank 3) a running word is credited 64 − 1024/L: "
                 "32 at L 32, and 75%, 169% and 230% of stock still unforced at L 64, 128 and 256, so at rank ≤ 3 the "
                 "commit stays every k32 and T2's L buy is rank 4 to 5",
                 unc_sem[32] == 0 and unc_sem[64] == 0.75 and all(unc_sem[L] > 0.01 for L in LS if L > 32),
                 "  ".join(f"L {L} {v:.0%}" for L, v in unc_sem.items()))
    t2s = {mode: {L: t2(L, b) for L in LS} for mode, b in T2_K32.items()}
    tol = {L: (sha_tolerance(L, HASH_UNITS[1]), sha_tolerance(L, HASH_UNITS[0])) for L in LS}
    checks.check("hashing at k32 makes ShaSpeed's tolerance 0.00125–0.0015% of the hashing price; L 64 doubles it and "
                 "brings T2 to about 79× (sync) and 95× (async) from 155× and 186× (estimates)",
                 abs(tol[32][1] - 0.01 / 670) < 1e-15 and 78 < t2s["sync"][64] < 80 and 94 < t2s["async"][64] < 96,
                 f"L 32 {tol[32][0]:.5%}–{tol[32][1]:.5%}; L 64 T2 {t2s['sync'][64]:.1f}× / {t2s['async'][64]:.1f}×")

    periods: dict[str, dict] = {}
    for key in ("llama-3.2-1b fused, T 8192", "llama-3.2-1b fused, T 512"):
        r = table[key]["residual"]
        periods[key] = {scope: {name: dict(zip(PERIODS, (share * x for x in stock)))
                                for name, share in r.items()} | {"T1 allowance": dict(zip(PERIODS, (0.01 * x
                                                                                                    for x in stock)))}
                        for scope, stock in PERIOD_STOCK.items()}
    w = periods["llama-3.2-1b fused, T 8192"]["world"]
    checks.check("per period the residuals scale linearly: the world's week of T1 allowance is 3.89e28 AND gates",
                 abs(w["T1 allowance"]["week"] - 3.89e28) < 1e24,
                 f"world week, Llama fused T 8,192: S1 {w[s[1]]['week']:.2e}, S2 {w[s[2]]['week']:.2e}")

    if checks.mutations:
        cheap = terms(LLAMA_FUSED, 1.0, 8192, p_form=P_FORM / 4)
        checks.mutation("at a quarter of forming's floor price (an entry a byte-wide register) S1 would pass at T 8,192: "
                        "forming's price decides S1's verdict", residual(cheap, {"final_word", "before_noise"}) < 0.01
                        < s1["llama-3.2-1b fused, T 8192"], f"{residual(cheap, {'final_word', 'before_noise'}):.2%}")
        no_dec = dict(table["llama-3.2-1b fused, T 8"]["terms"], decode=0.0)
        checks.mutation("dropping the decode term would leave S1 at T 8 dominated by forming alone: the term is counted",
                        residual(no_dec, {"final_word", "before_noise"}) < s1["llama-3.2-1b fused, T 8"])
        no_sign = dict(table["llama-3.2-1b apart, T 512"]["terms"], sign_fix=0.0)
        closed_s3 = {c for _, closes in SETS.values() for c in closes}
        checks.mutation("dropping the sign fix would put S3 at 0 and pass Llama-3.2-1B apart at T 512: the term is counted",
                        residual(no_sign, closed_s3) == 0 < 0.01 < s3["llama-3.2-1b apart, T 512"])
    if out:
        with open(out, "w") as f:
            json.dump({"sets": {k: {"adds": v[0], "closes": v[1]} for k, v in SETS.items()}, "table": table,
                       "uncommitted": unc, "uncommitted_semantic": unc_sem, "t2": t2s, "sha_tolerance": tol, "periods": periods,
                       "commit_instead_hashing": commit_alt,
                       "estimates": {"before_noise": BEFORE_NOISE, "hash_units": HASH_UNITS, "hash_share": HASH_SHARE,
                                     "t2_k32": T2_K32, "expert_share": EXPERT_SHARE, "derivation_price": P_DERIVE,
                                     "sign_fix_route_u_prime_row_init": P_FORM}},
                      f, indent=1)
            f.write("\n")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
