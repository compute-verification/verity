"""O-BITS (ACCEL, CMP-112): what each register writer costs a 32-bit word, against the INT8 tensor core's k32 word, and
section 4(g)'s register-write bound, on the H200 SXM and the RTX PRO 6000 (sm_120).

A writer is one warp instruction: `per` units of its rate, `words` the 32-bit registers it writes (A6), `distinct` how
many of them can hold different values (a broadcast load writes 128 registers but 4 values), the `pipe` it shares and
whether it is a `head` (it computes a value; a copy, LDS, LDSM or SHFL, moves one). Every writer also takes one of the
SM's 4 issue slots a clock (A2).

Three readings of the rates (units a SM-clock):

- `ds`: the DS-90 / DS-120 sheets as `machine-model sheet` derives them today (`data/devices/sm_90/sheet.toml`,
  `data/census/DS-120.csv`); a class with no row there (b1, DFMA on the H200, IMAD.WIDE, IADD.64 and DMUL on sm_120,
  LDS.128 on the H200) is priced at its issue slot alone.
- `sheet`: ACCEL CALIB's theta (cursor/compute-research-accel-calib-ea24, `data/devices/sm_90/theta.toml` and
  `data/theta.toml`), its `sheet` value, or `measured` where no document gives one (b1 on the H200).
- `measured`: CALIB's `measured` value, or `sheet` where it has none.

A `*` marks a shape CALIB has not timed: its `measured` is the sheet's. A writer's word ratio is its time a distinct
word over the k32 MMA's at the same reading's tensor rate (1 = the tensor core's word, ½ = twice as fast). The 4(g)
program maximizes distinct words written a SM-clock over every writer at once, under issue and each pipe; `c_g` is the
k32 tensor core's words a SM-clock over that maximum.

    uv run python benchmarks/pouw/theory/o_bits_rates.py
"""
from __future__ import annotations

from fractions import Fraction as F

import numpy as np
from scipy.optimize import linprog

ISSUE = 4  # warp instructions a SM-clock (A2: 4 SMSPs, one each)

# (name, pipe, per, words, distinct, head, {reading: rate a SM-clock or None for issue-only})
# Rates: MACs, lanes (results), FP16 FMAs, bytes or warp instructions a SM-clock, as `per` counts them.
H200 = [
    ("wgmma m64n256k32 s8", "tensor", 131072, 4096, 4096, True, {"ds": 4096, "sheet": 4096, "measured": 4095.6}),
    ("mma.sync m16n8k32 s8", "tensor", 4096, 128, 128, True, {"ds": 4096, "sheet": 4096, "measured": 2687.1}),
    ("mma.sync m16n8k16 s8", "tensor", 2048, 128, 128, True, {"ds": 4096, "sheet": 4096, "measured": 1995.7}),
    ("mma.sync m8n8k16 s8 *", "tensor", 1024, 64, 64, True, {"ds": 4096, "sheet": 4096, "measured": 4096}),
    ("mma.sync m16n8k8 f16 (f32 acc) *", "tensor", 1024, 128, 128, True, {"ds": 2048, "sheet": 2048, "measured": 2048}),
    ("mma.sync m16n8k4 tf32 *", "tensor", 512, 128, 128, True, {"ds": 1024, "sheet": 1024, "measured": 1024}),
    ("wgmma m64n256k256 b1", "tensor", 1048576, 4096, 4096, True, {"ds": None, "sheet": 32764.6, "measured": 32764.6}),
    ("mma.sync m16n8k256 b1", "tensor", 32768, 128, 128, True, {"ds": None, "sheet": 21501, "measured": 21501}),
    ("DFMA (two words a lane)", "fp64", 32, 64, 64, True, {"ds": None, "sheet": 64, "measured": 64}),
    ("IMAD.WIDE (two words a lane)", "imad", 32, 64, 64, True, {"ds": 64, "sheet": 32, "measured": 32}),
    ("IMAD", "imad", 32, 32, 32, True, {"ds": 64, "sheet": 64, "measured": 63.99}),
    ("IMAD.HI", "imad", 32, 32, 32, True, {"ds": 64, "sheet": 32, "measured": 30.6}),
    ("IADD3/LOP3/SHF/LEA.HI/PRMT/BMSK", "alu", 32, 32, 32, True, {"ds": 64, "sheet": 64, "measured": 63.01}),
    ("POPC", "xu", 32, 32, 32, True, {"ds": None, "sheet": 16, "measured": 16}),
    ("FFMA", "fma", 32, 32, 32, True, {"ds": None, "sheet": 128, "measured": 127.9}),
    ("HFMA2", "fma", 64, 32, 32, True, {"ds": None, "sheet": 256, "measured": 256}),
    ("ATOMS.ADD (shared-memory add)", "atom", 32, 32, 32, True, {"ds": None, "sheet": None, "measured": None}),
    ("SHFL (copy)", "shfl", 32, 32, 32, False, {"ds": 32, "sheet": 32, "measured": 32.0}),
    ("LDS.128 (copy)", "smem", 512, 128, 128, False, {"ds": None, "sheet": 128, "measured": 127.96}),
]

SM120 = [
    ("mma.sync m16n8k32 s8", "tensor", 4096, 128, 128, True, {"ds": 1024, "sheet": 1024, "measured": 1023.8}),
    ("mma.sync m16n8k16 s8", "tensor", 2048, 128, 128, True, {"ds": 1024, "sheet": 1024, "measured": 512}),
    ("mma.sync m8n8k16 s8 *", "tensor", 1024, 64, 64, True, {"ds": 1024, "sheet": 1024, "measured": 1024}),
    ("mma.sync m16n8k4 tf32", "tensor", 512, 128, 128, True, {"ds": 256, "sheet": 256, "measured": 256.0}),
    ("IADD.64 (two words a lane)", "alu", 32, 64, 64, True, {"ds": None, "sheet": 64, "measured": 62.5}),
    ("DMUL (two words a lane)", "fp64", 32, 64, 64, True, {"ds": None, "sheet": 2, "measured": 2}),
    ("IMAD.WIDE (two words a lane)", "imad", 32, 64, 64, True, {"ds": None, "sheet": 32, "measured": 31.8}),
    ("IMAD", "imad", 32, 32, 32, True, {"ds": 64, "sheet": 64, "measured": 64}),
    ("IMAD.HI", "imad", 32, 32, 32, True, {"ds": 64, "sheet": 32, "measured": 32}),
    ("IDP.4A (dp4a)", "imad", 32, 32, 32, True, {"ds": None, "sheet": 64, "measured": 64}),
    ("IADD3", "alu", 32, 32, 32, True, {"ds": 64, "sheet": 128, "measured": 64}),
    ("LOP3/SHF/LEA.HI/PRMT/BMSK", "alu", 32, 32, 32, True, {"ds": 64, "sheet": 64, "measured": 64}),
    ("POPC", "xu", 32, 32, 32, True, {"ds": 16, "sheet": 16, "measured": 16}),
    ("FFMA", "fma", 32, 32, 32, True, {"ds": None, "sheet": 128, "measured": 124.3}),
    ("HFMA2", "fma", 64, 32, 32, True, {"ds": None, "sheet": 128, "measured": 125.9}),
    ("ATOMS.ADD (shared-memory add)", "atom", 32, 32, 32, True, {"ds": None, "sheet": None, "measured": None}),
    ("SHFL (copy)", "shfl", 32, 32, 32, False, {"ds": 32, "sheet": 32, "measured": 16.0}),
    ("LDS.128 (copy)", "smem", 512, 128, 128, False, {"ds": 512, "sheet": 128, "measured": 127.6}),
    ("LDS.128 broadcast (copy)", "smem", 1, 128, 4, False, {"ds": 1, "sheet": 1, "measured": 0.4528}),
]

CARDS = {"H200": H200, "sm_120": SM120}
READINGS = ("ds", "sheet", "measured")


def instr_time(per, rate) -> float:
    """SM-clocks of the writer's pipe a warp instruction takes, or the issue slot alone (¼) with no rate."""
    return 0.25 if rate is None else max(0.25, per / rate)


def k32_word_time(card, reading) -> float:
    """SM-clocks a word of the k32 tensor core takes at the reading's s8 rate (the wgmma's on the H200)."""
    name, _, per, words, _, _, rates = card[0]
    return per / rates[reading] / words


def ratios(card, reading):
    t = k32_word_time(card, reading)
    return [(w[0], w[5], instr_time(w[2], w[6][reading]) / w[4] / t) for w in card]


def small_k(w) -> bool:
    return w[1] == "tensor" and w[2] / w[3] < 32


def lp(card, reading, heads_only: bool, no_small_k: bool = False):
    """4(g): max distinct words a SM-clock, writers at once, under issue and each pipe (A2, A4, G2)."""
    ws = [w for w in card if (w[5] or not heads_only) and not (no_small_k and small_k(w))]
    pipes = sorted({w[1] for w in ws})
    A, b = [[1.0] * len(ws)], [ISSUE]
    for p in pipes:
        A.append([(w[2] / w[6][reading] if (w[1] == p and w[6][reading]) else 0.0) for w in ws])
        b.append(1.0)
    res = linprog(c=-np.array([w[4] for w in ws], dtype=float), A_ub=np.array(A), b_ub=np.array(b), bounds=(0, None),
                  method="highs")
    assert res.status == 0, res.message
    mix = {ws[i][0]: round(float(x), 3) for i, x in enumerate(res.x) if x > 1e-9}
    return -res.fun, mix, res.ineqlin.marginals


# Lane instructions a word of the designs' folds (`machine_model/data/devices/*/designs`): duo is LOP3, LEA.HI and IMAD;
# LI is a LOP3 on one word and an IMAD on the next (the H200's headline dither-1s-li-l32).
FOLDS = {"duo": {"alu": 2.0, "imad": 1.0}, "li": {"alu": 0.5, "imad": 0.5}}


def _row(card, prefix):
    return next(w for w in card if w[0].startswith(prefix))


def _lane_rate(card, pipe, reading) -> float:
    """Lane instructions a SM-clock of the pipe's LOP3 or IMAD row (issue's 128 with no rate)."""
    w = _row(card, ("IADD3/LOP3", "LOP3/")) if pipe == "alu" else next(w for w in card if w[0] == "IMAD")
    return w[6][reading] or ISSUE * 32


def honest_dither(card, reading, fold: str = "li") -> float:
    """Words a SM-clock of the honest kernel: the k32 tensor core's words with the fold's lane instructions a word, under
    the tensor pipe, each lane pipe and issue (the MMA's own issue slot included)."""
    name, _, per, words, _, _, rates = card[0]
    f = FOLDS[fold]
    issue_per_word = sum(f.values()) / 32 + 1 / words
    return min(rates[reading] / per * words, *(_lane_rate(card, p, reading) / n for p, n in f.items()),
               ISSUE / issue_per_word)


def concurrency(card, reading, fold: str = "li") -> tuple[float, float]:
    """R-ACCEL-1 with other work priced on the tensor core alone: the tensor pipe runs other work at its peak and the
    transcript's words come from Four Russians on the lanes (A = B = 0: 3 entries a word, 4 to a LDS.128 lane, 2 IADD3
    and the fold). Returns (Four Russians words a SM-clock, that over the tensor peak's words): the strategy's credited
    work plus other work over the honest optimum's is 1 plus the second. Lookups are priced conflict-free."""
    name, _, per, words, _, _, rates = card[0]
    issue = ISSUE - rates[reading] / per
    f = FOLDS[fold]
    lds = _row(card, "LDS.128 (copy)")
    lds_t = instr_time(lds[2], lds[6][reading])
    iadd = _row(card, ("IADD3",))
    alu_t = 2 / (iadd[6][reading] or ISSUE * 32) + f["alu"] / _lane_rate(card, "alu", reading)
    imad_t = f["imad"] / _lane_rate(card, "imad", reading)
    issue_t = (3 / 4 + 2 + sum(f.values())) / 32
    fr = min(issue / issue_t, 1 / (3 / 128 * lds_t), 1 / alu_t, 1 / imad_t)
    return fr, fr / (rates[reading] / per * words)


def main() -> None:
    for cname, card in CARDS.items():
        print(f"\n== {cname}: a writer's time a distinct word over the k32 tensor core's (head: computes a value)")
        print(f"{'writer':38s} {'head':>4s} " + " ".join(f"{r:>9s}" for r in READINGS))
        rs = {r: ratios(card, r) for r in READINGS}
        for i, w in enumerate(card):
            print(f"{w[0]:38s} {'yes' if w[5] else 'no':>4s} " + " ".join(f"{rs[r][i][2]:9.3f}" for r in READINGS))
        for r in READINGS:
            heads = [x for x in rs[r] if x[1]]
            nonring = [x for x in heads if not x[0].startswith(("wgmma", "mma.sync"))]
            m, mn = min(heads, key=lambda x: x[2]), min(nonring, key=lambda x: x[2])
            print(f"  {r:8s} c_head = {m[2]:.3f} ({m[0]}); non-ring heads only {mn[2]:.3f} ({mn[0]})")
        print(f"-- 4(g) on {cname}: distinct words a SM-clock B, every writer at once; c_g = the k32 tensor core's / B "
              f"(no-small-k: without the MMA shapes of k·n/words < 32)")
        for r in READINGS:
            k32 = 1 / k32_word_time(card, r)
            hd = honest_dither(card, r)
            print(f"  {r:8s} k32 peak {k32:5.1f} words a SM-clock; honest Dither " +
                  ", ".join(f"{f} {honest_dither(card, r, f):5.1f}" for f in FOLDS))
            for heads_only, nsk in ((True, False), (False, False), (True, True), (False, True)):
                best, mix, _ = lp(card, r, heads_only, nsk)
                tag = ("heads" if heads_only else "all  ") + (" no-small-k" if nsk else "           ")
                print(f"  {r:8s} {tag} B = {best:7.1f}  c_g = {k32 / best:.3f}  (against honest Dither's with LI, "
                      f"{hd:5.1f}: {hd / best:.3f})  mix {mix}")
        print(f"-- R-ACCEL-1 on {cname}, other work on the tensor core alone: Four Russians words a SM-clock on the lanes "
              f"beside it, and the strategy's gain over the honest optimum")
        for r in READINGS:
            print(f"  {r:8s} " + "; ".join(f"{f}: {fr:5.1f} words, +{g:.1%}" for f in FOLDS
                                         for fr, g in [concurrency(card, r, f)]))
    print("\nSHA-512 a word (section 'barrier'): ~3,570 32-bit ALU instructions a 1,024-bit block, 111.6 a word")
    for cname, alu, k32 in (("H200", 64, 128), ("sm_120", 64, 32)):
        g = (3570 / 32) / alu / (1 / k32)
        print(f"  {cname}: {g:.0f}x the k32 tensor core's word time on the ALU pipe alone")
    print(f"\nTwo clocks: the H200's tensor peak is quoted at 1830 MHz and its lane pipes at 1980 MHz, so per second a lane "
          f"writer's ratio is {float(F(1830, 1980)):.3f} of its per-clock ratio")


if __name__ == "__main__":
    main()
