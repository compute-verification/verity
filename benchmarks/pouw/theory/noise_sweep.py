"""Noising and de-noising designs under DistinctWrites with H32: each design's T1 gap terms, T2 and T3 inputs, and checks.

The bound (`DistinctWritesMH100Floor`, Security/Definitions/Pouw/Assumptions/Dimension/H100W.lean) charges the adversary
32 units for each committed 32-bit word that is pairwise distinct on the salt grid and equal on it to no free value.
Every term below is a share of stock's m·n·K int8 MACs, in priced work: one dense int8 MAC is 1 unit, and every register
write costs at least 32 (H32). A design's T1 gap is the honest work that no such word forces:

    gap = (forming_X − forced_X) + (forming_Y − forced_Y) + de-noise + [final word free]·32/K + dup·α

- α: the stacked product's width over K. Every k32 step's accumulator is committed, so the tensor work α is forced,
  less the duplicate or free fraction dup.
- forming: honest register writes that build the operands (noise derivation, adds, negations), at 32 a register, so
  p = 8 a packed int8 entry. Oblivious whole-word moves and loads are free (`Op100.move`).
- forced: the part of forming that commits as distinct non-free words when the formed operands are committed
  (candidate 1, `--c1`); 0 otherwise.
- final word: free under M0 (free set: constants, salt-independent functions of the input, raw coin words) when the
  noise cancels, so the last word is A·Bᵀ; under M1 (the call's own words, CORE-TIME's lemma, unproved) it is a target.

Two families avoid both the final word and most forming:

- served sparse noise (`ssn_*`): each k32 step holds 28 data entries and one register of noise bits against a public
  pattern W₀, so every word carries fresh noise, the final word is the served output A·Bᵀ + noise, and the only forming
  is one mask a row a step;
- stacked periodic noise (`per2`): X = [A + V·P | V⊗L], Y = [B; −limbs of fold(B)], exact, with only weights-only
  corrections.

`check()` verifies each stacked design's exact cancellation, the impossibility of a two-sided 2-block cancellation with
scalar coefficients, and the functional distinctness (pairwise distinct and non-free on a salt grid, M0's free set) of
every committed word on small instances, degenerate data included.

All prices and shapes are inputs: H32's 32 (measured on the H200), Llama-3.2-1B's linears, an assumed V4-like dense
and routed-expert shape, and CORE-NCP's measured step parts for the T2 estimate. Output: a JSON report on stdout or
`--json PATH`, and a text table with `--table`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

H32 = 32
P_ENTRY = H32 / 4
DATA_PER_STEP = 28

LLAMA = {"qkv": (3072, 2048), "o": (2048, 2048), "gate_up": (16384, 2048), "down": (2048, 8192)}
LLAMA_LAYERS = 16
V4_DENSE = {"dense": (7168, 7168)}
V4_EXPERT = {"e_gate_up": (4096, 7168), "e_down": (7168, 2048)}
V4_EXPERT_SHARE = 8 / 256
ROWS = {"T512_64tok": 64, "T512_512tok": 512, "T8192": 8192}

HASH_GBPS = 519.6
STEP = {"T8192": {"stock_ms": 37.77, "gemm_ms": 12.49, "tokens": 8192},
        "T512_64tok": {"stock_ms": 3.42, "gemm_ms": 0.571, "tokens": 64}}
LLAMA_MACS_TOKEN = sum(n * k for n, k in LLAMA.values()) * LLAMA_LAYERS

H200_MAC_S = 989.5e12
AND_PER_MAC = 2079 / 16
WORLD_MAC_YEAR = 9.36e30 / 2
DATACENTER = 100_000
PERIOD_S = {"window": 600, "week": 604_800, "month": 2_592_000, "year": 31_536_000}

PER2_LIMBS_REAL = 8
WEIGHT_SIGMA_INT8 = 25.0
QUANT_ERR_UNIT = 0.5 / math.sqrt(3)


def steps28(k: int) -> int:
    return math.ceil(k / DATA_PER_STEP)


@dataclass(frozen=True)
class Design:
    name: str
    noise: str
    alpha: Callable[[int], float]
    form_x: Callable[[int, int, int], float]
    form_y: Callable[[int, int, int], float]
    forced_x: Callable[[int, int, int], float]
    forced_y: Callable[[int, int, int], float]
    denoise: Callable[[int, int, int], float]
    final_free_m0: bool
    dup: str
    wrap: str
    t3: str
    fmt: str
    c1_bytes: Callable[[int, int, int], float]
    verdict: str = ""
    notes: list[str] = field(default_factory=list)


def _zero(m: int, n: int, k: int) -> float:
    return 0.0


def designs() -> list[Design]:
    s = steps28
    return [
        Design("ncp3", "X=[A+E1|A+E1R|-(A+E1(I+R))], Y=[B+F1;B-RF1;B+(I-R)F1], E,F in [-31,32] (ncp.py)",
               lambda k: 3.0, lambda m, n, k: 32 / n, lambda m, n, k: 32 / m,
               lambda m, n, k: 24 / n, lambda m, n, k: 24 / m, _zero, True, "data-independent",
               "int7 activations, Route U offset (Int7Int32UUnsigned, k <= 2^16)", "+3.68% 2K / +3.48% 8K (measured, CORE-NCP)",
               "F1 int7-ncp-v1", lambda m, n, k: 3 / n + 3 / m,
               "fails at T 512 (24/m + 8/m); 2.1% at T 8,192"),
        Design("os2", "X=[A+E|E], Y=[B;-B], E=(h&7)-4 one-sided, unsigned offset", lambda k: 2.0,
               lambda m, n, k: 2 * P_ENTRY / n, _zero, lambda m, n, k: 2 * P_ENTRY / n, _zero, _zero, True,
               "weights-only (zero or equal weight columns)", "activations clamped to +-124; u8 offset is a weights-only init",
               "~0 (clamp 124, estimate)", "F2 int8-tok124", lambda m, n, k: 2 / n,
               "dominated by per2 (alpha 2)"),
        Design("per2", f"X=[A+V.P | V x L], Y=[B; -limbs(fold32 B)], V in [-4,3] per row, L={PER2_LIMBS_REAL} limbs",
               lambda k: 1 + H32 * PER2_LIMBS_REAL / k,
               lambda m, n, k: P_ENTRY / n + 256 / (n * k), _zero,
               lambda m, n, k: P_ENTRY / n + 256 / (n * k), _zero, _zero, True,
               "weights-only for words; X registers repeat on zero A blocks", "activations clamped to +-124; limbs keep Y int8",
               "~0 (clamp 124, estimate)", "F2 int8-tok124", lambda m, n, k: 1 / n,
               "needs M1 for the final word; limbs from real weights unmeasured (estimate 8)"),
        Design("ssn_a", "per k32 step: 28 data + 4 bits a=h&0x01010101 per row; public W0 (step 1 injective in [-6,6]^4, then one-hot)",
               lambda k: H32 * s(k) / k, lambda m, n, k: H32 * s(k) / (n * k), _zero,
               lambda m, n, k: H32 * s(k) / (n * k), _zero, _zero, False, "data-independent",
               "none: stock int8 +-127, int32 holds 127^2 K + sum|W0| for K < 133,000", "~0: served noise std ~6 units vs ~330 (estimate)",
               "F3 ssn-a-v1", lambda m, n, k: 4 * s(k) / (n * k),
               "lead: no C1, no M1 needed"),
        Design("ssn2", "per k32 step: 28 data + 4 bits a per row and 4 bits b per column (both h&0x01010101)",
               lambda k: H32 * s(k) / k, lambda m, n, k: H32 * s(k) / (n * k),
               lambda m, n, k: H32 * s(k) / (m * k), lambda m, n, k: H32 * s(k) / (n * k),
               lambda m, n, k: H32 * s(k) / (m * k), _zero, False, "data-independent", "none (stock int8)",
               "~0: served noise std ~7 units (estimate)", "F5 ssn2-v1", lambda m, n, k: 4 * s(k) / (n * k) + 4 * s(k) / (m * k),
               "needs C1 at m 64 (Y masks 1.14/m)"),
        Design("nsf", "one k32 step first: 32 bits per row x public W0 (distinct weight-w columns), served",
               lambda k: 1 + H32 / k, lambda m, n, k: 8 * H32 / (n * k), _zero, lambda m, n, k: 8 * H32 / (n * k), _zero,
               _zero, False, "data-dependent within a chain (zero data blocks repeat a word)", "none (stock int8)",
               "~0: error <= w units", "F6 nsf-v1", lambda m, n, k: 32 / (n * k),
               "dup on zero blocks (massive-activation rows)"),
        Design("disp2s", "first 16 K-entries of A and B replaced by raw coins R, S (moves); restore [A_P|R]·[B_P;-S] in one k32 step",
               lambda k: 1 + H32 / k, _zero, lambda m, n, k: P_ENTRY * 16 / (m * k), _zero,
               lambda m, n, k: P_ENTRY * 16 / (m * k), _zero, True, "data-dependent within a chain",
               "none (stock int8, exact)", "0 (exact)", "F0 int8-channel127-v1", lambda m, n, k: 16 / (m * k),
               "needs M1; dup on zero blocks"),
        Design("perm", "k32-block order of K permuted per call (TMA coordinates)", lambda k: 1.0, _zero, _zero, _zero, _zero,
               _zero, True, "data-dependent (zero blocks, identical rows)", "none (exact)", "0 (exact)",
               "F0 int8-channel127-v1", _zero, "needs M1; dup; a noise-dependent address may price as `other` (8/n + 8/m)"),
        Design("lowrank1", "X=[A+u vᵀ | u x L], v public, u fresh per row; Kronecker and block-diagonal reduce to this or per2",
               lambda k: 1 + H32 * PER2_LIMBS_REAL / k, lambda m, n, k: 2 * P_ENTRY / n, _zero,
               lambda m, n, k: P_ENTRY / n, _zero, _zero, True, "weights-only", "clamp to fit u v",
               "~0 (estimate)", "F2 int8-tok124", lambda m, n, k: 1 / n, "dominated by per2 (a table select per register)"),
        Design("dither32", "Dither v3.2 noise (±80 periodic U, V; wrap8) with 32-bit words every k32 and O de-noised",
               lambda k: 1.0, lambda m, n, k: 3 * P_ENTRY / n, lambda m, n, k: 3 * P_ENTRY / m, _zero, _zero,
               lambda m, n, k: 128 / k + 1 / m, True, "data-independent", "wrap8 with wrap indicators",
               "0 (exact O)", "F0 int8-channel127-v1", _zero,
               "dominated: de-noise L'/K + 1/m and wrap forming unforced"),
        Design("rowsign", "row signs s_i on A, served after one sign fix an output", lambda k: 1.0,
               lambda m, n, k: P_ENTRY / n, _zero, lambda m, n, k: P_ENTRY / n, _zero, lambda m, n, k: H32 / k, False,
               "data-dependent (zero partial sums are free)", "none", "0", "F0 int8-channel127-v1", lambda m, n, k: 1 / n,
               "dominated: the per-output fix is 32/K"),
        Design("signs2", "column signs d on both A and B: (A d)(d B) = AB at every partial sum", lambda k: 1.0,
               lambda m, n, k: P_ENTRY / n, lambda m, n, k: P_ENTRY / m, _zero, _zero, _zero, True,
               "every word free (d^2 = 1)", "none", "0", "-", _zero, "rejected: words salt-independent"),
        Design("init", "one per-row accumulator init dither c_i", lambda k: 1.0, _zero, _zero, _zero, _zero, _zero, False,
               "fails: words c_i + D collide on equal partial sums", "none", "0", "-", _zero, "rejected"),
    ]


DUP_ESTIMATE = {"ncp3": 0.0, "os2": 0.0, "per2": 0.0, "ssn_a": 0.0, "ssn2": 0.0, "lowrank1": 0.0, "dither32": 0.0,
                "nsf": None, "disp2s": None, "perm": None, "rowsign": None, "signs2": 1.0, "init": None}


def gap_terms(d: Design, m: int, n: int, k: int, model: str, c1: bool) -> dict:
    a = d.alpha(k)
    fx, fy = d.form_x(m, n, k), d.form_y(m, n, k)
    cx, cy = (d.forced_x(m, n, k), d.forced_y(m, n, k)) if c1 else (0.0, 0.0)
    final = H32 / k if (d.final_free_m0 and model == "M0") else 0.0
    dup = DUP_ESTIMATE[d.name]
    terms = {"alpha": a, "forming_x": fx - cx, "forming_y": fy - cy, "denoise": d.denoise(m, n, k), "final_word": final,
             "dup": a * dup if dup is not None else 0.0}
    terms["gap"] = sum(v for key, v in terms.items() if key != "alpha")
    terms["dup_data_dependent"] = float(dup is None)
    return terms


def weighted(d: Design, shapes: dict, m: int, model: str, c1: bool, rows_scale: float = 1.0) -> dict:
    rows = max(1, round(m * rows_scale))
    total, acc = 0.0, {}
    for n, k in shapes.values():
        w = rows * n * k
        t = gap_terms(d, rows, n, k, model, c1)
        for key, v in t.items():
            acc[key] = acc.get(key, 0.0) + w * v
        total += w
    out = {key: v / total for key, v in acc.items()}
    out["rows"] = rows
    return out


def hash_bytes_per_mac(d: Design, m: int, n: int, k: int, c1: bool) -> float:
    return d.alpha(k) * 4 / H32 + (d.c1_bytes(m, n, k) if c1 else 0.0)


def t2_estimate(d: Design, point: str, c1: bool) -> float:
    p = STEP[point]
    m = p["tokens"]
    macs = sum(m * n * k for n, k in LLAMA.values()) * LLAMA_LAYERS
    byts = sum(m * n * k * hash_bytes_per_mac(d, m, n, k, c1) for n, k in LLAMA.values()) * LLAMA_LAYERS
    alpha = sum(m * n * k * d.alpha(k) for n, k in LLAMA.values()) * LLAMA_LAYERS / macs
    t_ms = p["stock_ms"] + (alpha - 1) * p["gemm_ms"] + byts / (HASH_GBPS * 1e9) * 1e3
    return t_ms / p["stock_ms"]


def ssn_a_w0(n: int, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """W0 for ssn_a: steps × 4 × n. Step 1 is an injective code of the column into [-6,6]^4 \\ {0} (base-13 digits of
    j + 1, skipping the all-zero code), so n ≤ 28,559; later steps are one-hot, rotating with the step."""
    s = steps28(k)
    w0 = np.zeros((s, 4, n), dtype=np.int64)
    zero = sum(6 * 13 ** e for e in range(4))
    for j in range(n):
        c = j + 1 if j + 1 < zero else j + 2
        digits = []
        for _ in range(4):
            digits.append(c % 13 - 6)
            c //= 13
        assert c == 0, "n too large for a 4-digit base-13 code"
        w0[0, :, j] = digits
        for st in range(1, s):
            w0[st, (j + st) % 4, j] = 1
    return w0


MASKS = {1: 0x01010101, 2: 0x03030303, 4: 0x0F0F0F0F}


def ssn_noise_std(k: int, n: int = 2048) -> dict:
    """Served noise std in accumulator units for ssn_a at mask widths 1, 2 and 4 bits a byte, and for ssn2 (1 bit)."""
    w0 = ssn_a_w0(min(n, 28560), k)
    w2 = (w0.astype(float) ** 2).sum(axis=(0, 1))
    quant = QUANT_ERR_UNIT * WEIGHT_SIGMA_INT8 * math.sqrt(k)
    out = {"K": k, "act_quant_std_units_estimate": quant, "ssn2_std_units": math.sqrt(0.75 * steps28(k))}
    for bits in MASKS:
        var = w2 * ((2 ** bits) ** 2 - 1) / 12
        out[f"ssn_a_b{bits}_std_units"] = float(np.sqrt(var).mean())
        out[f"ssn_a_b{bits}_var_share_of_quant"] = float(var.mean()) / quant ** 2
    return out


# ---------------------------------------------------------------- small-instance checks (M0 free set on a salt grid)

def _pack(regs: np.ndarray) -> np.ndarray:
    """Pack int8 entries 4 to a 32-bit register: (..., 4·r) → (..., r) as int64 codes."""
    u = (regs.astype(np.int64) & 0xFF).reshape(*regs.shape[:-1], -1, 4)
    return u[..., 0] | (u[..., 1] << 8) | (u[..., 2] << 16) | (u[..., 3] << 24)


def _words(x: np.ndarray, y: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
    """Committed accumulator words: the partial sum after every k32 step of x (salts × m × W) · y (salts × W × n)."""
    sa, m, w = x.shape
    assert w % 32 == 0
    prods = np.einsum("smk,skn->smnk", x.astype(np.int64), y.astype(np.int64))
    blk = prods.reshape(sa, m, y.shape[2], w // 32, 32).sum(-1).cumsum(-1)
    if init is not None:
        blk = blk + init[:, :, :, None]
    return blk


def _distinct(targets: list[np.ndarray], free: list[np.ndarray]) -> dict:
    """targets, free: arrays (salts × …). Count targets pairwise distinct on the grid and equal to no free function."""
    tv = np.concatenate([t.reshape(t.shape[0], -1) for t in targets], axis=1).T
    fv = np.concatenate([f.reshape(f.shape[0], -1) for f in free], axis=1).T if free else np.zeros((0, tv.shape[1]))
    const = (tv == tv[:, :1]).all(axis=1)
    fset = {row.tobytes() for row in fv.astype(np.int64)}
    seen, good, dup, freec = set(), 0, 0, 0
    for i, row in enumerate(tv.astype(np.int64)):
        b = row.tobytes()
        if const[i] or b in fset:
            freec += 1
        elif b in seen:
            dup += 1
        else:
            seen.add(b)
            good += 1
    n = len(tv)
    return {"words": n, "distinct_nonfree": good, "dup": dup, "free": freec, "bad_share": (dup + freec) / n}


def _data(case: str, m: int, n: int, k: int, rng: np.random.Generator, amax: int = 127) -> tuple[np.ndarray, np.ndarray]:
    a = np.clip(np.rint(rng.normal(0, 30, (m, k))), -amax, amax).astype(np.int64)
    b = np.clip(np.rint(rng.normal(0, 25, (k, n))), -127, 127).astype(np.int64)
    if case == "zero_row":
        a[0] = 0
    elif case == "identical_rows":
        a[1] = a[0]
    elif case == "massive_row":
        a[0] = 0
        a[0, 3] = amax
    elif case == "zero_blocks":
        a[:, 32:96] = 0
    elif case == "identical_cols":
        b[:, 1] = b[:, 0]
    elif case == "zero_wblock":
        b[32:64, 0] = 0
    return a, b


CASES = ["random", "zero_row", "identical_rows", "massive_row", "zero_blocks", "identical_cols", "zero_wblock"]


def _bits(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.integers(0, 2, shape, dtype=np.int64)


def sim_design(name: str, case: str, salts: int = 24, m: int = 4, n: int = 6, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    if name in ("ssn_a", "ssn2"):
        k = DATA_PER_STEP * 4
        a, b = _data(case, m, n, 128, rng)
        a, b = a[:, :k], b[:k]
        s = steps28(k)
        w0 = ssn_a_w0(n, k)
        xs, ys, raw, formed = [], [], [], []
        for _ in range(salts):
            bits_a = _bits(rng, (m, s, 4))
            xa = np.concatenate([a.reshape(m, s, DATA_PER_STEP), bits_a], axis=2).reshape(m, 32 * s)
            if name == "ssn_a":
                wy = w0
            else:
                wy = _bits(rng, (s, 4, n))
            yb = np.concatenate([b.reshape(s, DATA_PER_STEP, n), wy], axis=1).reshape(32 * s, n)
            xs.append(xa)
            ys.append(yb)
            raw.append(rng.integers(0, 2 ** 32, (m, s)))
            formed.append(_pack(bits_a.reshape(m, s * 4)))
        x, y = np.stack(xs), np.stack(ys)
        bias = np.rint(w0.sum(axis=(0, 1)) / 2).astype(np.int64) if name == "ssn_a" else np.full(n, s)
        init = -np.broadcast_to(bias, (salts, m, n))
        words = _words(x, y, init)
        out = _distinct([words, np.stack(formed)], [np.stack(raw)])
        exact = None
        err = words[:, :, :, -1] - (a @ b)[None]
        out["served_err_mean"] = float(err.mean())
        out["served_err_std"] = float(err.std())
        return out | {"exact": exact}
    k = 128
    amax = 63 if name == "ncp3" else (124 if name in ("os2", "per2") else 127)
    a, b = _data(case, m, n, k, rng, amax)
    xs, ys, formed, raw = [], [], [], []
    inits = None
    for _ in range(salts):
        if name == "ncp3":
            e1 = rng.integers(-31, 33, (m, k))
            f1 = rng.integers(-31, 33, (k, n))
            perm = rng.permutation(k // 4)
            r = np.zeros((k, k), dtype=np.int64)
            for i, p in enumerate(perm):
                r[4 * i:4 * i + 4, 4 * p:4 * p + 4] = np.eye(4, dtype=np.int64)
            e3 = e1 + e1 @ r
            f3 = f1 - r @ f1
            x = np.concatenate([a + e1, a + e1 @ r, -(a + e3)], axis=1)
            y = np.concatenate([b + f1, b - r @ f1, b + f3], axis=0)
            formed.append(np.concatenate([_pack(x), _pack(y.T)], axis=None))
        elif name == "os2":
            e = rng.integers(-4, 4, (m, k))
            x = np.concatenate([a + e, e], axis=1)
            y = np.concatenate([b, -b], axis=0)
            formed.append(_pack(x).ravel())
        elif name == "per2":
            v = rng.integers(-4, 4, (m, 32))
            fold = b.reshape(k // 32, 32, n).sum(0)
            limbs = max(1, math.ceil(np.abs(fold).max() / 127))
            parts = [np.floor_divide(fold + l, limbs) for l in range(limbs)]
            assert sum(parts).tolist() == fold.tolist()
            x = np.concatenate([a + np.tile(v, (1, k // 32))] + [v] * limbs, axis=1)
            y = np.concatenate([b] + [-p for p in parts], axis=0)
            formed.append(_pack(x[:, :k + 32]).ravel())
        elif name == "disp2s":
            rr = rng.integers(-128, 128, (m, 16))
            ss = rng.integers(-128, 128, (16, n))
            xa = a.copy()
            xa[:, :16] = rr
            yb = b.copy()
            yb[:16] = ss
            x = np.concatenate([xa, a[:, :16], rr], axis=1)
            y = np.concatenate([yb, b[:16], -ss], axis=0)
            raw.append(np.concatenate([_pack(rr).ravel(), _pack(ss.T).ravel()]))
            formed.append(_pack((-ss).T).ravel())
        elif name == "perm":
            p = rng.permutation(k // 32)
            idx = np.concatenate([np.arange(32 * q, 32 * q + 32) for q in p])
            x, y = a[:, idx], b[idx]
        elif name == "nsf":
            bits = _bits(rng, (m, 32))
            w0 = np.zeros((32, n), dtype=np.int64)
            for j in range(n):
                w0[j % 32, j] = 1
                w0[(j * 7 + 3) % 32, j] = 1
            x = np.concatenate([bits, a], axis=1)
            y = np.concatenate([w0, b], axis=0)
            formed.append(_pack(bits).ravel())
        elif name == "rowsign":
            sg = rng.choice([-1, 1], (m, 1))
            x, y = sg * a, b
        elif name == "signs2":
            d = rng.choice([-1, 1], (k,))
            x, y = a * d[None], d[:, None] * b
        elif name == "init":
            c = rng.integers(-2 ** 20, 2 ** 20, (m, 1))
            x, y = a, b
            inits = (inits or []) + [np.broadcast_to(c, (m, n))]
        else:
            raise ValueError(name)
        xs.append(x)
        ys.append(y)
    x, y = np.stack(xs), np.stack(ys)
    init = np.stack(inits) if inits else None
    words = _words(x, y, init)
    targets = [words] + ([np.stack(formed)] if formed else [])
    out = _distinct(targets, [np.stack(raw)] if raw else [])
    final = words[:, :, :, -1] - (init if init is not None else 0)
    out["exact"] = bool((final == (a @ b)[None]).all()) if name not in ("rowsign", "init", "nsf") else None
    return out


def two_block_cancellation(coef: range = range(-3, 4)) -> dict:
    """X = [A+E | αA+βE], Y = [B+F; γB+δF] cancels to AB iff 1+αγ=1, 1+αδ=0, 1+βγ=0, 1+βδ=0. No scalar solution."""
    sols = [(al, be, ga, de) for al in coef for be in coef for ga in coef for de in coef
            if al * ga == 0 and 1 + al * de == 0 and 1 + be * ga == 0 and 1 + be * de == 0]
    one_sided = [(al, be, ga) for al in coef for be in coef for ga in coef
                 if 1 + al * ga == 1 and 1 + be * ga == 0 and al == 0]
    return {"two_sided_solutions": sols, "one_sided_solutions": one_sided}


def check() -> list[dict]:
    res = []

    def ok(name: str, cond: bool, detail) -> None:
        res.append({"check": name, "pass": bool(cond), "detail": detail})

    tb = two_block_cancellation()
    ok("no two-sided 2-block scalar cancellation", tb["two_sided_solutions"] == [], tb["two_sided_solutions"])
    ok("one-sided 2-block [A+E|E]·[B;-B] cancels", (0, 1, -1) in tb["one_sided_solutions"], tb["one_sided_solutions"])
    for name in ("ncp3", "os2", "per2", "disp2s", "perm"):
        r = sim_design(name, "random")
        ok(f"{name} cancels exactly", r["exact"] is True, r["exact"])
    for name in ("ssn_a", "ssn2"):
        bad = {c: sim_design(name, c)["bad_share"] for c in CASES}
        ok(f"{name} words distinct and non-free on every data case", all(v == 0 for v in bad.values()), bad)
    ncp = {c: sim_design("ncp3", c) for c in CASES}
    ok("ncp3: on every data case only the final word (A·B) is free, no duplicates",
       all(r["dup"] == 0 and r["free"] == 4 * 6 for r in ncp.values()), {c: (r["dup"], r["free"]) for c, r in ncp.items()})
    for name, case in (("disp2s", "zero_blocks"), ("perm", "identical_rows"), ("nsf", "zero_blocks"),
                       ("signs2", "random"), ("init", "zero_blocks")):
        r = sim_design(name, case)
        ok(f"{name} loses words on {case} (mutation: the check sees data-dependent distinctness)", r["bad_share"] > 0,
           r["bad_share"])
    for name in ("os2", "per2"):
        r = sim_design(name, "random")
        ok(f"{name}: only the final word is free", r["free"] == 4 * 6 and r["dup"] == 0, r)
    r = sim_design("ssn_a", "random", salts=400)
    ok("ssn_a served error is zero-mean within 0.5", abs(r["served_err_mean"]) < 0.5, r["served_err_mean"])
    ok("ssn_a W0 step-1 code is injective and nonzero for n = 16,384",
       len({tuple(c) for c in ssn_a_w0(16384, 56)[0].T}) == 16384 and (np.abs(ssn_a_w0(16384, 56)[0]).sum(0) > 0).all(),
       "base-13 digits")
    return res


def report(c1_only: bool = False) -> dict:
    ds = designs()
    rows = []
    for d in ds:
        for model in ("M0", "M1"):
            for c1 in (False, True):
                entry = {"design": d.name, "model": model, "c1": c1}
                for point, m in ROWS.items():
                    entry[f"llama_{point}"] = weighted(d, LLAMA, m, model, c1)
                    entry[f"v4dense_{point}"] = weighted(d, V4_DENSE, m, model, c1)
                    entry[f"v4expert_{point}"] = weighted(d, V4_EXPERT, m, model, c1, V4_EXPERT_SHARE)
                entry["t2_T8192"] = t2_estimate(d, "T8192", c1)
                entry["t2_T512_64tok"] = t2_estimate(d, "T512_64tok", c1)
                rows.append(entry)
    sims = {d.name: {c: sim_design(d.name, c) for c in CASES} for d in ds if d.name not in ("lowrank1", "dither32")}
    per_dot = {}
    for d in ds:
        per_dot[d.name] = {lin: {"extra_macs": (d.alpha(k) - 1) * k, "words": d.alpha(k) * k / H32,
                                 "forming_x_units_per_row": d.form_x(1, n, k) * n * k,
                                 "forming_y_units_per_col_call": d.form_y(1, n, k) * k,
                                 "denoise_units": d.denoise(10 ** 9, n, k) * k}
                           for lin, (n, k) in LLAMA.items()}
    return {"designs": [{k: (v if not callable(v) else None) for k, v in d.__dict__.items() if not callable(v)} for d in ds],
            "rows": rows, "sims": sims, "per_dot_product": per_dot,
            "noise_std": [ssn_noise_std(k) for k in (2048, 8192)], "checks": check(),
            "params": {"H32": H32, "p_entry": P_ENTRY, "data_per_step": DATA_PER_STEP, "llama": LLAMA, "v4_dense": V4_DENSE,
                       "v4_expert": V4_EXPERT, "v4_expert_share": V4_EXPERT_SHARE, "rows": ROWS, "hash_GBps": HASH_GBPS,
                       "step": STEP, "per2_limbs_estimate": PER2_LIMBS_REAL}}


def periods(gap_share: float, alpha: float) -> dict:
    out = {}
    for scope, mac_s in (("one_h200", H200_MAC_S), ("datacenter", H200_MAC_S * DATACENTER),
                         ("world", WORLD_MAC_YEAR / PERIOD_S["year"])):
        out[scope] = {p: gap_share * mac_s / alpha * s * AND_PER_MAC for p, s in PERIOD_S.items()}
    return out


def table(rep: dict) -> str:
    lines = []
    hdr = ("design", "model", "C1", "L T512(64)", "L T512(512)", "L T8192", "V4d T512", "V4d T8192", "V4e T8192",
           "T2 8192", "T2 512")
    lines.append(" | ".join(hdr))
    for r in rep["rows"]:
        def f(key):
            return f"{100 * r[key]['gap']:.3f}%" + ("+dup" if r[key]["dup_data_dependent"] else "")
        lines.append(" | ".join([r["design"], r["model"], "y" if r["c1"] else "n", f("llama_T512_64tok"),
                                 f("llama_T512_512tok"), f("llama_T8192"), f("v4dense_T512_64tok"), f("v4dense_T8192"),
                                 f("v4expert_T8192"), f"{r['t2_T8192']:.1f}x", f"{r['t2_T512_64tok']:.1f}x"]))
    lines.append("")
    for c in rep["checks"]:
        lines.append(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--table", action="store_true", help="print the text table")
    args = ap.parse_args(argv)
    rep = report()
    lead = next(r for r in rep["rows"] if r["design"] == "ssn_a" and r["model"] == "M0" and not r["c1"])
    rep["periods_ssn_a_M0_noC1_llama_T8192"] = periods(lead["llama_T8192"]["gap"], lead["llama_T8192"]["alpha"])
    ncp = next(r for r in rep["rows"] if r["design"] == "ncp3" and r["model"] == "M0" and not r["c1"])
    rep["periods_ncp3_M0_noC1_llama_T8192"] = periods(ncp["llama_T8192"]["gap"], ncp["llama_T8192"]["alpha"])
    text = json.dumps(rep, indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    if args.table:
        print(table(rep))
    elif not args.json:
        print(text)
    return 0 if all(c["pass"] for c in rep["checks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
