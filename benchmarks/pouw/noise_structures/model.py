"""PoUW noise structures on the H200: the floor at A = B = 0, honest de-noising, noise generation and committed bytes.

Every figure is an estimate from the rates below (theta.toml sm_90 and the runs it cites), not a timing. Costs are in
G-times: a call's plain s8 GEMM, m·n·K MACs at the s8 rate, is 1. CUDA-core work is priced at its issue rate against
s8's, SHA-512 at its measured compression rate, HBM at its measured bandwidth. A structure's floor is the cheapest
cheater route found at A = B = 0 per credited MAC, an upper bound on the true floor; `check.py` checks every route's
algebra exactly.

`uv run benchmarks/pouw/noise_structures/model.py --out DIR` writes DIR/structures.json (every call class × m × structure,
the best parameters of each) and DIR/structures.md (the tables).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check import mac_ands, sparse_hi, wraps

CLK = 1.769e9  # measured event clock under the served GEMM, run b57f
SMS = 132
S8 = 4096.0  # s8 wgmma MACs a SM-clock (theta tensor.s8.k32, measured 4095.6)
# b1 .and.popc against the honest s8 GEMM. r20261028-090323-2c56 (#6216) measured BMMA at 21,845 MACs a SM-clock, 8.00×
# mma.sync s8's 2,731 but 5.33× wgmma s8's 4,096 (S8): 3/16 against the served GEMM. 1/8 is the default, a margin to
# the cheater; `--r-b 0.1875` prices the measured b1 against wgmma.
R_B = 1 / 8
SP24 = 0.5  # 2:4 sparse s8 wgmma, A in registers: 8191 logical MACs a SM-clock (dc/README, wgsprs_s8)
ALU = 64.0  # IADD3/LOP3 results a SM-clock (theta alu.iadd3 sheet 64, alu.lop3 measured 63.01)
ADD = S8 / (2 * ALU)  # MAC-times a sparse add: IADD3 folds two adds an instruction
OP = S8 / ALU  # MAC-times a dense ALU instruction
SHA_C = 0.018327  # SHA-512 compressions a SM-clock (theta sha512, measured)
PRG_B = S8 / (SHA_C * 64)  # MAC-times a PRG byte (64 output bytes a compression): about 3,492
HASH_BPS = 521e9  # SHA-512 over 16 KiB rows, the whole card (b921; note:compute-research/20261028T1600Z-finding-commit-bytes)
R_S8 = S8 * SMS * CLK  # MACs a second
HASH_B = R_S8 / HASH_BPS  # MAC-times a hashed byte: about 1,836
HBM = 4.26e12  # bytes a second, measured (theta hbm)
WRAP_8B = {"p_W": 0.0119, "p_X": 0.0016}  # Llama-3.1-8B real weights and activations at ±80 (note 20261026T0225Z-draft-accel-bet-hybrid-2)
WRAP_G = {"alu": ADD, "measured": 3.5 / 0.01}  # G a unit wrap rate: an IADD3-bound sparse add, or v31's wrap2 at m 8192 (r20261028-190744-7676)
WRAP_COST = ["measured"]  # model.py --wrap-cost
HASH_CORUN_BPS = 66.5e9  # SHA-512 beside the GEMM (the co-run share; GD-MODEL's corun_share, an estimate)

# Proof blocks (C-Flock): ANDs a block, the cap, and the prover's price
AND_MAC_32 = 271342 / 2048  # ScaledMmS8_v1's K-2048 coordinate, measured lowering count (r20261028-192825-df06, GD-PARTITION)
OPEN_B = 454  # ANDs a byte of an operand row opened in the block (TOP-138; an estimate)
ROW1 = 0.167e6  # ANDs a one-block row opened (GD-MODEL: three of them are 17% of the 3.0 M s8 coordinate block)
FOLD_WORD_ANDS = 32  # ANDs to fold one word in-circuit (an 8-bit ARX step: add, multiply by a constant, rotate; an estimate)
K_CAP = 27  # bits a block, log2: compose's K_MAX, live/src/circuit.rs IN_RANGE_K_LOG, Stmt.InRange
K_FLOOR = 22  # a block is at least 2^22 bits (GD-PROVER)
BITS_AND = 2.0  # bits a block AND (GD-PROVER: SHA-512 1.50, hm96 2.26, s8 blocks 1.87–3.27 after padding; an estimate)
DRAWS = 2773  # blocks a population draws a window at δ 2^-40, ε 0.01 (GD-MODEL, Λ/ε)
O_AND, F0, F1, M_MAX = 1.087e-8, 3.03, 3.51e-6, 35  # GD-MODEL's costfit over r20261018-060433-b55e (sm_120; E for the H200)
SESSION = (2.05, 1.14e-6, 1.20e-8)  # a + b·s + c·s·P GPU-s a session: GD-PROVER's warm refit of r20261018-060433-b55e (RTX PRO 6000)

CALLS = {"qkv": (3072, 2048), "o": (2048, 2048), "gate_up": (16384, 2048), "down": (2048, 8192), "lm_head": (128256, 2048)}
MS = (4, 64, 512, 8192)
LAYER = ("qkv", "o", "gate_up", "down")
LAYERS = 16
STEP_CELLS = {k: sum(CALLS[c][0] for c in LAYER if CALLS[c][1] == k) * LAYERS for k in (2048, 8192)}  # cells a token, by K


@dataclasses.dataclass(frozen=True)
class Call:
    name: str
    m: int
    n: int
    k: int

    @property
    def macs(self) -> float:
        return self.m * self.n * self.k

    @property
    def t_g(self) -> float:
        return self.macs / R_S8

    @property
    def t_stock(self) -> float:
        return max(self.t_g, self.n * self.k / HBM)


_HI: dict = {}
_WR: dict = {}


def hi_res(perm: str, t: int, amp: int) -> float:
    """The merged factor's 2:4 residual density (entries of hi past each 4-group's first two), from `check.sparse_hi`."""
    key = (perm, t, amp)
    if key not in _HI:
        rng = np.random.default_rng(7 + t + 131 * amp)
        r = sparse_hi(rng, 512, 4, 256, t, perm if perm != "none" else "segment", amp)
        _HI[key] = (r["hi_density"], r["residual_density"])
    return _HI[key]


def wrap_rates(amp: int) -> dict:
    """Wrap rates at ±amp: synthetic operands (check.wraps), scaled so ±80 meets the measured Llama-3.1-8B rates."""
    if not _WR:
        rows = wraps(np.random.default_rng(3), 128, 8192, [16, 32, 48, 64, 80, 127])
        _WR.update({r["amp"]: r for r in rows})
    s = _WR[amp]
    return {"p_X": s["p_X"] * WRAP_8B["p_X"] / _WR[80]["p_X"], "p_W": s["p_W"] * WRAP_8B["p_W"] / _WR[80]["p_W"]}


def c_segment(perm: str, t: int, amp: int, word_bits: int, words: int, dither_bits: int = 1) -> tuple[float, str]:
    """The cheater's s8 limb-products a period of L' for one word, beyond the binary product, and the route.

    The word's increment is d_A,s d_B,sᵀ + F Uᵀ + V Y_sᵀ with F = D_s + V M_s (check.hyb). F Uᵀ: one limb if F fits int8,
    or the word is 8 bits (F mod 2^8 as u8); else a dense low limb plus the high limb dense or 2:4 sparse with a CUDA
    residual; under per-segment π, D_s Uᵀ and t·(Vπ_s⁻¹)Uᵀ as two dense limbs. Unpermuted: V Uᵀ is shared by every
    segment, so D_s Uᵀ + V Y_sᵀ + t·VUᵀ costs 2 + 1/words."""
    fmax = t * (amp + 2 ** dither_bits - 1)
    if word_bits <= 8 or fmax <= 127:
        return 2.0, "F·Uᵀ one limb (F mod 2^8 or F in int8) + V·Y_sᵀ"
    if perm == "none":
        return 2 + 1 / words, "D_s·Uᵀ + V·Y_sᵀ dense, V·Uᵀ once a call"
    dens, res = hi_res(perm, t, amp)
    sparse = 1 + SP24 + res * ADD  # low limb, 2:4 high limb, and res·L' residual adds a cell at ADD MAC-times each
    routes = {"F lo dense + F hi 2:4 + residual adds": sparse, "F lo + hi dense": 2.0}
    if perm == "segment":
        routes["D_s·Uᵀ + t·(Vπ_s⁻¹)·Uᵀ dense"] = 2.0
    name = min(routes, key=routes.get)
    return routes[name] + 1, f"{name} ({routes[name]:.2f}) + V·Y_sᵀ (1); hi density {dens:.3f}, residual {res:.4f}"


@dataclasses.dataclass
class Row:
    structure: str
    call: str
    m: int
    n: int
    k: int
    params: dict
    floor: float
    route: str
    credited: float
    tensor: float
    cuda: float
    prg: float
    prg_weight_per_call: float
    hash_words: float
    hash_o: float
    hbm: float
    words_bytes_cell: float
    committed: str
    block: str
    premise: str
    gd_model: dict = dataclasses.field(default_factory=dict)
    credit_on: str = ""
    serial: float = 0.0
    overlap: float = 0.0
    objective: float = 0.0

    def finish(self, c: Call) -> "Row":
        """Time over the stock call: serialized max(tensor, HBM) + CUDA + SHA, and overlapped max of all four."""
        tg = c.t_g
        t_tensor = tg * (1 + self.tensor)
        t_hbm = self.hbm * c.n * c.k / HBM
        t_cuda = tg * self.cuda
        t_sha = tg * (self.prg + self.prg_weight_per_call + self.hash_words)
        self.serial = (max(t_tensor, t_hbm) + t_cuda + t_sha) / c.t_stock - 1
        self.overlap = max(t_tensor, t_hbm, t_cuda, t_sha) / c.t_stock - 1
        self.objective = self.serial / self.credited if self.credited > 0 else math.inf
        return self


def lprimes(k: int) -> list[int]:
    return [lp for lp in (32, 64, 128, 256, 512, 1024, 2048, 4096) if lp < k and k % lp == 0]


def gdm(scheme: str, k: int, noise: str, row: "Row", unit_cells: int, seg: int | None = None, lp: int | None = None,
        word_bits: int = 8) -> dict:
    """The row in GD-MODEL's `pouw` keys (golden/model.py), plus the keys it lacks (`word_bits`, `forced_frac`)."""
    return {"scheme": scheme, "K": k, "L": seg, "L_prime": lp, "noise": noise,
            "denoise_g_frac": round(row.tensor + row.cuda, 4), "unit_cells": unit_cells, "word_bits": word_bits,
            "words_bytes_cell": row.words_bytes_cell, "forced_frac": round(min(1.0, row.floor) * row.credited, 4)}


def hyb(c: Call, name: str, perm: str, word_bits: int, amp_v: int = 80, amp_u: int = 80, v_bytes: int = 4,
        fold: bool = False, dither_bits: int = 1, weight_per_call: bool = False, receipt_bits: int = 8,
        lanes: int = 1, wrap_cost: str | None = None, v32: bool = False) -> list[Row]:
    """Dither-style periodic noise: V, U period L', words every L = t·L' (W of them).

    v32: Dither v3.2's layout, T_s for s < ⌈K/L⌉ only (T_S is implied by O, not committed), so W·L < K and a call
    credits at most (K − L)/K. Otherwise a word at K is committed too ("v3.2 plus a committed T_S").

    Two honest routes for O: the peel (the noised GEMM over the whole K, then 4L'/K of tail products and a wrap correction
    over the wrapped entries, priced at `WRAP_G[wrap_cost]` G a unit wrap rate), or the recompute that Dither v3.2's served
    call runs (the noised GEMM only over the credited f·K, O by the plain GEMM: f G of tensor and no wrap correction).

    Only V's amplitude enters the cheater's limbs (F = D_s + V M_s); U's sets only the weight-side wrap rate, since U is
    the right factor at one s8 limb whatever its width, and the H200 has no narrower integer or FP format faster than s8.
    v_bytes is the PRG bytes a V entry: 4 in dither.py's `_noise` (one u32 a value), 1 for a byte-wide draw."""
    wrap_cost = wrap_cost or WRAP_COST[0]
    rb = dither_bits * dither_bits * R_B
    wr = {"p_X": wrap_rates(amp_v)["p_X"], "p_W": wrap_rates(amp_u)["p_W"]}
    noise_key = "periodic" if perm == "none" else "permuted"
    rows = []
    for lp in lprimes(c.k):
        for t in range(1, 9):
            for w in range(1, c.k // (t * lp) + 1):
                cs, route = c_segment(perm, t, amp_v, word_bits, w, dither_bits)
                seg = t * lp
                floor = (rb * seg + min(cs * lp, (1 - rb) * seg)) / seg
                credit = w * seg
                if v32 and credit >= c.k:
                    continue
                if floor < 1 - 1e-9:
                    continue
                f = credit / c.k
                base = (2 * OP + ADD) / c.n
                wrap = WRAP_G[wrap_cost] * (wr["p_W"] + wr["p_X"])
                prg = (c.m * c.k * dither_bits / 8 + v_bytes * c.m * lp) * PRG_B / c.macs
                wpc = (c.n * c.k * dither_bits / 8 + 4 * c.n * lp) * PRG_B / c.macs if weight_per_call else 0.0
                extra = (2 * OP / c.m if weight_per_call else 0) + (w * 3 * OP / c.k if fold else 0)
                wb = receipt_bits / 8 / lanes if fold else w * word_bits / 8
                for how in ("peel", "recompute"):
                    if how == "peel":
                        tensor = 4 * lp / c.k + (SP24 if wrap > SP24 else 0)
                        cuda = base + (0 if wrap > SP24 else wrap) + extra
                        hbm = 1 + 3 * lp / c.k + 4 * wr["p_W"]
                    else:
                        tensor, cuda, hbm = f, base + extra, 1 + f
                    r = Row(
                        name, c.name, c.m, c.n, c.k,
                        {"lp": lp, "t": t, "W": w, "word_bits": word_bits, "amp_v": amp_v, "amp_u": amp_u, "v_bytes": v_bytes,
                         "perm": perm, "dither_bits": dither_bits, "fold": fold, "route": how},
                        floor, f"binary {rb:.3f}/MAC + per word min(separation {cs:.2f}·L', direct): {route}", f,
                        tensor, cuda, prg, wpc, wb * HASH_B / c.k, 4 * HASH_B / c.k, hbm, wb,
                        (f"1 folded {receipt_bits}-bit receipt per {lanes} cell(s) over {w} virtual {word_bits}-bit prefix words a cell, and O (32-bit)" if fold else
                         f"{w} prefix words T_s at {word_bits} bits a cell, and O (32-bit); weight root + per-period W′, U"
                         + (", Y_W, wrap list" if how == "peel" else "")
                         + (" (the word at K included: v3.2 plus a committed T_S)" if credit == c.k else "")),
                        f"{lanes} cell(s)' whole K (lanes form)" if fold else f"one cell × one segment (L = {seg}); the O block opens the whole rows",
                        ("ARX fold unpredictable without its inputs (not SHA-512: a new named premise)" if fold else "SAK on noised operands"),
                    ).finish(c)
                    r.gd_model = gdm("dither", c.k, noise_key, r, lanes if fold else 1, seg, lp, word_bits)
                    r.credit_on = ("A and B: each segment's word is the noised served product (A + d_A + V)(B + d_B + Uπ)ᵀ over it, "
                                   "so a cheater computes the served product segment by segment, plus a 4L'/K peel")
                    rows.append(r)
    return rows


def two_gemm(c: Call, word_bits: int = 8, dither_bits: int = 1, lanes: int = 1) -> list[Row]:
    """O = A·Bᵀ plain; one noised word per cell, T = wrap8(A + d_A)·Rᵀ mod 2^b, R fresh int8 along K, per period, stored.

    A word mod 2^b over a d-bit dither needs the b1 plane pairs (i, j) with i + j < b (check.planes): 8 at d = 1, b = 8.
    lanes > 1 folds that many cells' words into one receipt (J10-R's lane fold), a premise and a lane-sized proof block."""
    pairs = sum(1 for i in range(dither_bits) for j in range(8) if i + j < word_bits)
    floor = min(1.0, pairs * R_B)
    if floor < 1 - 1e-9:
        return []
    fold = lanes > 1
    r = Row(
        "fresh-R two-GEMM" + (f" + fold of {lanes} cells" if fold else ""), c.name, c.m, c.n, c.k,
        {"word_bits": word_bits, "dither_bits": dither_bits, "lanes": lanes},
        floor, f"s8 direct = {pairs} b1 plane products = two 2:4 halves: all {floor:.2f}", 1.0,
        1.0, (1.5 * OP) / c.n + (3 * OP / c.k if fold else 0), c.m * c.k * dither_bits / 8 * PRG_B / c.macs, 0.0,
        word_bits / 8 / lanes * HASH_B / c.k, 4 * HASH_B / c.k, 2.0, word_bits / 8 / lanes,
        (f"1 receipt over {lanes} cells' words" if fold else f"1 word T at {word_bits} bits a cell")
        + " (and O, plain); weight root + per-period R (n×K int8, +1× weight memory)",
        f"{lanes} cells' whole K" if fold else "one cell's whole K",
        "SAK on noised operands, with R fixed for the period (one operand known ahead)"
        + ("; the receipt fold, as J10-R" if fold else ""),
    ).finish(c)
    r.gd_model = gdm("two_gemm", c.k, "fresh", r, lanes, word_bits=word_bits)
    r.credit_on = "A and R: (A + d_A)·Rᵀ, a product beside the served A·Bᵀ (smell-check Q1)"
    return [r]


def fresh_u(c: Call) -> list[Row]:
    """X = A + d_A, W′ = wrap8(B + d_B + U) with U fresh int8 along K, drawn once a period and stored; one 8-bit word
    T = X·W′ᵀ mod 2^8 a cell, and O by the plain GEMM. JR2's noise without its fold, or HYB at t = 1 with V = 0: the
    nearest design to two-GEMM whose word is the served product's own operands, noised."""
    r = Row(
        "fresh-U per period, 8-bit word", c.name, c.m, c.n, c.k, {"word_bits": 8, "u_amp": 64},
        1.0, "d_A·W′ᵀ mod 2^8: 8 b1 planes or s8 direct, 1.00 (check.planes)", 1.0,
        1.0, 1.5 * OP / c.n, c.m * c.k / 8 * PRG_B / c.macs, 0.0,
        HASH_B / c.k, 4 * HASH_B / c.k, 2.0, 1.0,
        "1 word at 8 bits a cell, and O (32-bit); weight root + per-period W′ (n×K int8, +1× weight memory)",
        "one cell's whole K", "SAK on noised operands, with U fixed for the period (one operand known ahead)",
    ).finish(c)
    r.gd_model = gdm("fresh_u", c.k, "fresh", r, 1)
    r.credit_on = "A and B: (A + d_A)(B + d_B + U)ᵀ; the work forced beyond the served O is X·Uᵀ"
    return [r]


def jr2(c: Call, per_call: bool) -> list[Row]:
    """J10-R v2 (#6369): X = A + b, W′ = B + c + U with U fresh at every K entry (±64), O′ = T − X·Uᵀ, receipts by the
    duo/fx1 fold over the lane's sets. Per call as the PR has it, or with c and U drawn once a period and stored."""
    wr = wrap_rates(64)
    wrap_w = min(ADD * wr["p_W"], SP24)
    cuda = 1.5 * OP / c.n + (0 if wrap_w == SP24 else wrap_w) + (2 * OP / c.m if per_call else 0)
    wpc = c.n * c.k * 9 / 8 * PRG_B / c.macs if per_call else 0.0
    r = Row(
        "JR2 fresh U " + ("per call (#6369)" if per_call else "per period"), c.name, c.m, c.n, c.k, {"u_amp": 64},
        1.0, "b·(c+U)ᵀ: s8 direct or 8 b1 planes, both 1.00 (U fresh, no fold)", 1.0,
        1.0 + (SP24 if wrap_w == SP24 else 0), cuda, c.m * c.k / 8 * PRG_B / c.macs, wpc,
        0.0, 4 * HASH_B / c.k, 1.0 if per_call else 2.0, 0.0,
        "receipts (2(KS+1) words a lane) folded over T_0..T_(KS−1) and O′; the z/y records are the values' commitment",
        "a whole tile (32×32 cells × K)", "the duo/fx1 receipt fold, as J10-R v1",
    ).finish(c)
    r.gd_model = gdm("jr2", c.k, "fresh", r, 1024, word_bits=32)
    r.credit_on = ("A and B: X·(B + c + U)ᵀ; the work forced beyond O is X·Uᵀ. Assumes the receipts are computed at serve "
                   "time for every tile; as J10-R serves today, no noised product runs at serve time and it earns 0")
    return [r]


def fresh_v(c: Call) -> list[Row]:
    """X = A + d_A + V with V fresh int8 per entry, per call; W′ = B + d_B. One 8-bit word. O = T − V·W′ᵀ."""
    wr = wrap_rates(80)
    r = Row(
        "fresh-V (per call)", c.name, c.m, c.n, c.k, {"word_bits": 8},
        1.0, "V·d_Bᵀ mod 2^8: 8 b1 planes or s8 direct, 1.00", 1.0,
        1.0, (2 * OP + ADD) / c.n + ADD * wr["p_X"], c.m * c.k * 9 / 8 * PRG_B / c.macs, 0.0,
        HASH_B / c.k, 4 * HASH_B / c.k, 1.0, 1.0,
        "1 word at 8 bits a cell, and O (32-bit)", "one cell's whole K", "SAK on noised operands",
    ).finish(c)
    r.gd_model = gdm("fresh_v", c.k, "fresh", r, 1)
    r.credit_on = "A and B: (A + d_A + V)(B + d_B)ᵀ; the work forced beyond the served O is V·W′ᵀ"
    return [r]


ENG = {"amp_u": 16, "v_bytes": 1}  # U narrow (weight wraps ≈ 0) and V one PRG byte an entry: neither changes a floor

STRUCTS = {
    "HYB per-tile π, 16-bit (Dither v3.2 as specified)": lambda c, s: hyb(c, s, "tile", 16, v32=True),
    "HYB per-segment π, 16-bit (v3.2 noise and layout)": lambda c, s: hyb(c, s, "segment", 16, v32=True),
    "HYB per-tile π, 16-bit, U ±16, V 1 B": lambda c, s: hyb(c, s, "tile", 16, **ENG),
    "HYB per-segment π, 16-bit, U ±16, V 1 B": lambda c, s: hyb(c, s, "segment", 16, **ENG),
    "HYB no π, 16-bit, U ±16, V 1 B": lambda c, s: hyb(c, s, "none", 16, **ENG),
    "HYB per-segment π, 8-bit, U ±16, V 1 B": lambda c, s: hyb(c, s, "segment", 8, **ENG),
    "HYB per-segment π, 2-bit dithers, U ±16, V 1 B": lambda c, s: hyb(c, s, "segment", 16, dither_bits=2, **ENG),
    "HYB per-segment π, weight noise per call": lambda c, s: hyb(c, s, "segment", 16, weight_per_call=True, **ENG),
    "HYB per-segment π + ARX fold, 8-bit receipt a cell": lambda c, s: hyb(c, s, "segment", 16, fold=True, **ENG),
    "HYB per-segment π + ARX fold of 32 cells": lambda c, s: hyb(c, s, "segment", 16, fold=True, lanes=32, **ENG),
    "fresh-R two-GEMM, 8-bit word": lambda c, s: two_gemm(c),
    "fresh-R two-GEMM + fold of 32 cells": lambda c, s: two_gemm(c, lanes=32),
    "JR2 fresh U per call (#6369)": lambda c, s: jr2(c, True),
    "JR2 fresh U per period": lambda c, s: jr2(c, False),
    "fresh-V per call, 8-bit word": lambda c, s: fresh_v(c),
    "fresh-U per period, 8-bit word": lambda c, s: fresh_u(c),
}


_MAC: dict = {}


def mac_and(word_bits: int) -> float:
    """ANDs a MAC into a word_bits accumulator: the measured 32-bit lowering scaled by `check.mac_ands`' ratio."""
    if not _MAC:
        _MAC.update({b: mac_ands(np.random.default_rng(5), b)["ands"] for b in (8, 16, 32)})
    return AND_MAC_32 * _MAC[word_bits] / _MAC[32]


# name: (STRUCTS row its parameters come from, block along K, right operand public, word bits, lanes fold)
TILE_STRUCTS = {
    "HYB per-segment π, 16-bit words": ("HYB per-segment π, 16-bit, U ±16, V 1 B", "segment", False, 16, False),
    "HYB per-segment π, 8-bit words": ("HYB per-segment π, 8-bit, U ±16, V 1 B", "segment", False, 8, False),
    "HYB per-segment π + lanes fold": ("HYB per-segment π + ARX fold of 32 cells", "segment", False, 16, True),
    "fresh-R two-GEMM, 8-bit word": ("fresh-R two-GEMM, 8-bit word", "whole", True, 8, False),
    "fresh-R two-GEMM + lanes fold": ("fresh-R two-GEMM + fold of 32 cells", "whole", True, 8, True),
    "fresh-V, 8-bit word": ("fresh-V per call, 8-bit word", "whole", False, 8, False),
    "fresh-U per period, 8-bit word": ("fresh-U per period, 8-bit word", "whole", False, 8, False),
    "TOP-138's model (J10-R lanes, 300 ANDs a MAC)": (None, "whole", False, 32, True),
}
SIDES = (1, 2, 4, 8, 16, 32)


def opened(rows: int, nbytes: int, k: int, row_bytes: int | None) -> int:
    """Bytes opened to read nbytes from each of `rows` operand rows of K bytes, in value rows of row_bytes (None: exact)."""
    if row_bytes is None:
        return rows * nbytes
    if row_bytes >= k:
        return math.ceil(rows / (row_bytes // k)) * row_bytes
    return rows * math.ceil(nbytes / row_bytes) * row_bytes


def block(name: str, k: int, r: int, c: int, p: dict, row_bytes: int | None = None) -> dict:
    """One draw's proof blocks for an r×c tile: (ANDs a draw, the largest block's ANDs, blocks a draw, serve bytes a cell).

    Without a fold a block is the tile over one segment (HYB) or the whole K, opening r left rows and c right rows (none if
    the right operand is the verifier's coins) and the cells' words in one-block rows. With the lanes fold a draw is the
    tile's whole credited K as r lanes (one tile row each, its words written as a registered value) and a combine that
    folds the tile's words into its one receipt."""
    _, along, right_public, wb, fold = TILE_STRUCTS[name]
    if p is None:
        draw = 300 * r * c * k * (1 + 1.5 * (1 / r + 1 / c))
        return {"draw": draw, "largest": draw / r, "blocks": r + 1 if r > 1 else 1, "bytes_cell": 0.0}
    words = p.get("W", 1)
    seg = p["t"] * p["lp"] if along == "segment" else k
    a, wbytes = mac_and(wb), wb / 8
    right = lambda cols, n: 0 if right_public else opened(cols, n, k, row_bytes)  # noqa: E731
    if not fold:
        nw = 2 if along == "segment" else 1
        ands = r * c * seg * a + OPEN_B * (opened(r, seg, k, row_bytes) + right(c, seg)) + ROW1 * math.ceil(r * c * nw * wbytes / 128)
        return {"draw": ands, "largest": ands, "blocks": 1, "bytes_cell": words * wbytes}
    span = words * seg
    lane = c * span * a + OPEN_B * (opened(1, span, k, row_bytes) + right(c, span)) + ROW1 * math.ceil(c * words * wbytes / 128)
    comb = ROW1 * (math.ceil(r * c * words * wbytes / 128) + 1) + FOLD_WORD_ANDS * r * c * words
    return {"draw": r * lane + comb, "largest": max(lane, comb), "blocks": r + 1, "bytes_cell": 1 / (r * c),
            "fold_words_cell": words}


def prove(b: dict) -> dict:
    """GPU-s a window for one population's DRAWS draws, as C-Flock proves them (GD-PROVER): sessions of k blocks padded to
    P = max(8, 2^⌈log2 k⌉), k ≤ 2^(35 − k_log), each a + b·s + c·s·P for blocks of s ANDs, s floored at 2^22 bits; the
    session count that minimizes the total. `gpu_s_gdmodel` is GD-MODEL's costfit, draws·o·ANDs + statements·(f0 + f1·s)."""
    k_log = max(K_FLOOR, math.ceil(math.log2(b["largest"] * BITS_AND)))
    blocks = DRAWS * b["blocks"]
    s = max(b["draw"] / b["blocks"], 2 ** K_FLOOR / BITS_AND)
    a_, b_, c_ = SESSION
    kmax = 2 ** max(0, M_MAX - k_log)
    best_cost = math.inf
    for kk in (2 ** i for i in range(0, int(math.log2(kmax)) + 1)):
        n = math.ceil(blocks / kk)
        per = math.ceil(blocks / n)
        cost = n * (a_ + b_ * s + c_ * s * max(8, 2 ** math.ceil(math.log2(per))))
        best_cost = min(best_cost, cost)
    stmts = math.ceil(blocks / kmax)
    return {"k_log": k_log, "statements": stmts, "gpu_s": best_cost,
            "gpu_s_gdmodel": DRAWS * O_AND * b["draw"] + stmts * (F0 + F1 * b["largest"])}


def tiles(rows: list[dict], row_bytes: int | None = None) -> dict:
    """Each structure's tile sweep: K 2048 and K 8192 populations, every r×c, and the optimum for a window of S steps.

    The objective is GD-MODEL's J up to its t_stock: proving GPU-s a window + S × (the step's word hashing + fold ALU
    work in GPU-s). Everything else a structure costs doesn't depend on the tile."""
    rep = {}
    for name, (src, *_rest) in TILE_STRUCTS.items():
        params = {}
        for k, call in ((2048, "qkv"), (8192, "down")):
            if src is None:
                params[k] = None
                continue
            r0 = [r for r in rows if r["structure"] == src and r["call"] == call and r["m"] == 8192]
            params[k] = r0[0]["params"] if r0 else None
        if src is not None and None in params.values():
            continue
        sweep = []
        for r in SIDES:
            for c in SIDES:
                parts = {k: block(name, k, r, c, params[k], row_bytes) for k in (2048, 8192)}
                pv = {k: prove(b) for k, b in parts.items()}
                serve = {}
                for t in (512, 8192):
                    by = sum(STEP_CELLS[k] * t * parts[k]["bytes_cell"] for k in parts)
                    fold = sum(STEP_CELLS[k] * t * parts[k].get("fold_words_cell", 0) * 3 for k in parts) / (ALU * SMS * CLK)
                    serve[t] = {"bytes": by, "gpu_s": by / HASH_BPS + fold, "gpu_s_corun": by / HASH_CORUN_BPS + fold}
                sweep.append({"r": r, "c": c, "k_log": max(v["k_log"] for v in pv.values()),
                              "ands_draw": {k: parts[k]["draw"] for k in parts}, "ands_largest": {k: parts[k]["largest"] for k in parts},
                              "prove_gpu_s": sum(v["gpu_s"] for v in pv.values()),
                              "prove_gpu_s_gdmodel": sum(v["gpu_s_gdmodel"] for v in pv.values()),
                              "statements": {k: pv[k]["statements"] for k in pv},
                              "serve": serve})
        opt = {}
        for t in (512, 8192):
            for hash_key in ("gpu_s", "gpu_s_corun"):
                for s in (1, 4, 1e3, 1e4, 1e5, 1e6):
                    j = lambda e: e["prove_gpu_s"] + s * e["serve"][t][hash_key]  # noqa: E731
                    free = min(sweep, key=j)
                    out = {"free": (free["r"], free["c"], free["k_log"], j(free))}
                    for cap in (27, 28, 30):
                        ok = [e for e in sweep if e["k_log"] <= cap]
                        e = min(ok, key=j)
                        out[f"cap{cap}"] = (e["r"], e["c"], e["k_log"], j(e))
                    out["cap_binds"] = out["cap27"][:2] != out["free"][:2]
                    opt[f"T{t} {'co-run' if hash_key == 'gpu_s_corun' else 'card'} hash, S {s:g}"] = out
        rep[name] = {"params": params, "sweep": sweep, "optimum": opt}
    return rep


T_STOCK_8192 = 26.32e-3  # b57f's stock step at T 8,192, s (GD-MODEL's calibration of r20261028-080559-b57f)
TILE_OF = {  # STRUCTS row: (TILE_STRUCTS row, the tile its serve row assumes)
    "HYB per-segment π, 16-bit, U ±16, V 1 B": ("HYB per-segment π, 16-bit words", (1, 1)),
    "HYB per-segment π, 8-bit, U ±16, V 1 B": ("HYB per-segment π, 8-bit words", (1, 1)),
    "HYB per-segment π + ARX fold of 32 cells": ("HYB per-segment π + lanes fold", (1, 32)),
    "fresh-R two-GEMM, 8-bit word": ("fresh-R two-GEMM, 8-bit word", (1, 1)),
    "fresh-R two-GEMM + fold of 32 cells": ("fresh-R two-GEMM + lanes fold", (1, 32)),
    "fresh-V per call, 8-bit word": ("fresh-V, 8-bit word", (1, 1)),
    "fresh-U per period, 8-bit word": ("fresh-U per period, 8-bit word", (1, 1)),
}


def j_pouw(steps: dict, tl: dict) -> dict:
    """PoUW's share of J at T 8,192 (GD-MODEL's J: serve overhead over t_stock + proving GPU-s over stock GPU-s a
    window), for a window of S steps: the step's extra time over b57f's stock step, plus its PoUW blocks' proving."""
    out = {}
    for s, (tname, (r, c)) in TILE_OF.items():
        if s not in steps.get(8192, {}) or tname not in tl:
            continue
        e = next(e for e in tl[tname]["sweep"] if (e["r"], e["c"]) == (r, c))
        serve = steps[8192][s]["extra_ms_step"] / 1e3 / T_STOCK_8192
        out[s] = {"tile": f"{r}×{c}", "k_log": e["k_log"], "over_cap": e["k_log"] > K_CAP, "serve_term": serve,
                  "prove_gpu_s": e["prove_gpu_s"],
                  **{f"J S {n:g}": serve + e["prove_gpu_s"] / (n * T_STOCK_8192) for n in (1, 1e3, 1e5)}}
    return out


def forced_table() -> dict:
    """HYB's forced fraction of G per credited MAC at A = B = 0 (the best attack found), by π, word bits and t = 1..8:
    min(1, r_b + c/t) at ±80 and 1-bit dithers, for both b1 prices. Unpermuted noise at W = 1; it falls to 2 + 1/W limbs
    as W grows."""
    return {f"r_b {rb:g}, {perm} π, {wb}-bit": [round(min(1.0, rb + c_segment(perm, t, 80, wb, 1)[0] / t), 3)
                                                 for t in range(1, 9)]
            for rb in (1 / 8, 3 / 16) for perm in ("tile", "segment", "none") for wb in (16, 8)}


def best(rows: list[Row], min_credit: float) -> Row | None:
    ok = [r for r in rows if r.credited >= min_credit - 1e-9]
    return min(ok, key=lambda r: (r.objective, -r.credited)) if ok else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--min-credit", type=float, default=0.75)
    p.add_argument("--wrap-cost", choices=sorted(WRAP_G), default="measured")
    p.add_argument("--r-b", type=float, default=R_B, help="b1 MAC-time in s8 G (1/8, or 0.1875 against wgmma)")
    a = p.parse_args()
    WRAP_COST[0] = a.wrap_cost
    globals()["R_B"] = a.r_b
    out = []
    for name, (n, k) in CALLS.items():
        for m in MS:
            c = Call(name, m, n, k)
            for s, fn in STRUCTS.items():
                r = best(fn(c, s), a.min_credit)
                if r is not None:
                    r.structure = s
                    out.append(dataclasses.asdict(r))
    steps = {}
    for m in MS:
        for s in STRUCTS:
            rs = [r for r in out if r["structure"] == s and r["m"] == m and r["call"] in LAYER]
            if len(rs) < len(LAYER):
                continue
            stock = sum(Call(r["call"], m, r["n"], r["k"]).t_stock for r in rs)
            extra = sum(r["serial"] * Call(r["call"], m, r["n"], r["k"]).t_stock for r in rs)
            cred = sum(r["credited"] * Call(r["call"], m, r["n"], r["k"]).macs for r in rs) / sum(Call(r["call"], m, r["n"], r["k"]).macs for r in rs)
            steps.setdefault(m, {})[s] = {"overhead_over_linear_stock": extra / stock, "credited_share": cred,
                                          "extra_ms_step": LAYERS * extra * 1e3}
    rep = {"rates": {"clock_hz": CLK, "s8_macs_sm_clk": S8, "r_b": R_B, "sp24_cost": SP24, "alu_ops_sm_clk": ALU,
                     "macs_per_sparse_add": ADD, "macs_per_prg_byte": PRG_B, "macs_per_hashed_byte": HASH_B, "hbm_Bps": HBM,
                     "wrap_anchor_8B": WRAP_8B},
           "wrap_rates": {amp: wrap_rates(amp) for amp in (16, 32, 48, 64, 80, 127)},
           "hi_limb": {f"{k[0]} t{k[1]} a{k[2]}": v for k, v in _HI.items()},
           "min_credit": a.min_credit, "wrap_cost": a.wrap_cost, "rows": out, "steps": steps}
    rep["blocks"] = {"and_mac": {b: mac_and(b) for b in (8, 16, 32)}, "open_b": OPEN_B, "row1": ROW1, "k_cap": K_CAP,
                     "k_floor": K_FLOOR, "bits_per_and": BITS_AND, "draws": DRAWS,
                     "session_gd_prover": dict(zip("abc", SESSION)),
                     "costfit_gd_model": {"o": O_AND, "f0": F0, "f1": F1, "m_max": M_MAX},
                     "fold_word_ands": FOLD_WORD_ANDS, "step_cells_token": STEP_CELLS}
    rep["forced_frac"] = forced_table()
    rep["tiles"] = {"exact rows": tiles(out), "16 KiB value rows": tiles(out, 16384)}
    rep["j_pouw_8192"] = {rb: j_pouw(steps, tl) for rb, tl in rep["tiles"].items()}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "structures.json").write_text(json.dumps(rep, indent=1, default=str))
    (a.out / "structures.md").write_text(render(rep))
    print(render(rep))
    return 0


def render(rep: dict) -> str:
    lines = ["# PoUW noise structures on the H200 (estimates from theta rates; floor: the best attack found at A = B = 0 per "
             f"credited MAC, an upper bound on the true floor; b1 at {rep['rates']['r_b']:g} G)", ""]
    for m in MS:
        lines += [f"## m = {m}", "",
                  "| call | structure | params | floor | credited | tensor | CUDA | PRG/call | weight PRG/call | hash words | hash O | HBM× | serial | overlap |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rep["rows"]:
            if r["m"] != m:
                continue
            pr = r["params"]
            ps = (f"L′{pr['lp']} t{pr['t']} W{pr['W']} {pr['word_bits']}b" if "lp" in pr else
                  ",".join(f"{k}{v}" for k, v in pr.items()))
            lines.append(f"| {r['call']} | {r['structure']} | {ps} | {r['floor']:.2f} | {r['credited']:.2f} | {r['tensor']:.3f} | "
                         f"{r['cuda']:.3f} | {r['prg']:.3f} | {r['prg_weight_per_call']:.2f} | {r['hash_words']:.3f} | "
                         f"{r['hash_o']:.3f} | {r['hbm']:.2f} | {r['serial']:.2f} | {r['overlap']:.2f} |")
        lines.append("")
    lines += ["## A 16-layer step's linears (qkv, o, gate_up, down), each at its best parameters", "",
              "| T | structure | overhead / stock linear time | credited share | extra ms a step |", "|---|---|---|---|---|"]
    for m, d in rep["steps"].items():
        for s, v in sorted(d.items(), key=lambda kv: kv[1]["overhead_over_linear_stock"]):
            lines.append(f"| {m} | {s} | {v['overhead_over_linear_stock']:.2f} | {v['credited_share']:.3f} | {v['extra_ms_step']:.2f} |")
    for rb, jd in rep.get("j_pouw_8192", {}).items():
        lines += ["", f"## PoUW's share of J at T 8,192 ({rb}; serve term over b57f's 26.32 ms, proving over S steps' stock GPU-s)", "",
                  "| structure | tile (k_log) | serve term | proving GPU-s a window | J, S 1 | J, S 1e3 | J, S 1e5 |", "|---|---|---|---|---|---|---|"]
        for s, v in sorted(jd.items(), key=lambda kv: kv[1]["J S 1000"]):
            lines.append(f"| {s} | {v['tile']} ({v['k_log']}{', over 2^27' if v['over_cap'] else ''}) | {v['serve_term']:.3f} | "
                         f"{v['prove_gpu_s']:.0f} | {v['J S 1']:.0f} | {v['J S 1000']:.2f} | {v['J S 100000']:.3f} |")
    if "forced_frac" in rep:
        lines += ["", "## HYB's forced fraction of G per credited MAC at A = B = 0 (best attack found), ±80, 1-bit dithers", "",
                  "| b1 price, π, word | " + " | ".join(f"t {t}" for t in range(1, 9)) + " |", "|---" * 9 + "|"]
        lines += [f"| {k} | " + " | ".join(f"{x:.3f}" for x in v) + " |" for k, v in rep["forced_frac"].items()]
    shown = {(1, 1), (1, 8), (1, 32), (4, 8), (8, 8), (16, 16), (32, 32)}
    for rb, tl in rep.get("tiles", {}).items():
        lines += ["", f"## Proof blocks by tile ({rb}; K 2048 and K 8192 populations, {DRAWS} draws each a window; "
                  f"k_log of bits at {BITS_AND} a block AND, floor 2^{K_FLOOR}; GD-PROVER's session fit, RTX PRO 6000)", "",
                  "| structure | tile | M ANDs a draw K2048 / K8192 | largest block k_log | proving GPU-s a window (GD-MODEL's costfit) | words hashed a T 8192 step, ms (card / co-run) |",
                  "|---|---|---|---|---|---|"]
        for name, d in tl.items():
            for e in d["sweep"]:
                if (e["r"], e["c"]) in shown:
                    sv = e["serve"][8192]
                    lines.append(f"| {name} | {e['r']}×{e['c']} | {e['ands_draw'][2048] / 1e6:.2f} / {e['ands_draw'][8192] / 1e6:.2f} | "
                                 f"{e['k_log']}{' (over 2^27)' if e['k_log'] > K_CAP else ''} | {e['prove_gpu_s']:.0f} ({e['prove_gpu_s_gdmodel']:.0f}) | "
                                 f"{sv['gpu_s'] * 1e3:.2f} / {sv['gpu_s_corun'] * 1e3:.2f} |")
        lines += ["", f"### Optimum tile r×c (k_log) for a window of S steps at T 8,192 ({rb})", "",
                  "| structure | hash | S 1 | S 1e3 | S 1e4 | S 1e5 | S 1e6 | cap binds at any S shown |", "|---|---|---|---|---|---|---|---|"]
        for name, d in tl.items():
            for hk in ("card", "co-run"):
                cells, binds = [], False
                for s in ("1", "1000", "10000", "100000", "1e+06"):
                    o = d["optimum"][f"T8192 {hk} hash, S {s}"]
                    binds |= o["cap_binds"]
                    fr, cp = o["free"], o["cap27"]
                    cells.append(f"{cp[0]}×{cp[1]} ({cp[2]})" + (f", free {fr[0]}×{fr[1]} ({fr[2]})" if o["cap_binds"] else ""))
                lines.append(f"| {name} | {hk} | " + " | ".join(cells) + f" | {'yes' if binds else 'no'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
