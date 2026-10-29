"""The whole served step in W1: every component's stock and honest priced work, what DistinctWrites forces of it under
the component's best design, and the step's T1 (its coverage term beside the matmuls' terms), T2 and uncovered share.

noise_sweep.py prices one linear at a time. This prices every component of a served step, for Llama-3.2-1B and an
assumed V4-like MoE layer, at the points in `POINTS`:

- tensor components: the linears, QKᵀ, PV, the lm_head and the router's gate, at the cheapest instruction that computes
  each at its shape (`tc_best`: `wgmma` either orientation, `mma.sync` either orientation, with the padding each needs);
- lane components (`lane_ops`): the dequant epilogue, fused add + RMSNorm + per-token quant, RoPE, SiLU·mul, the
  quantizes, softmax (per score), the attention output's normalization, the MoE combine and router, and the sampler,
  each an instruction idiom per element, counted from the kernels' structure (an estimate until a SASS census);
- moves: the embedding gather and the KV-cache write are input-addressed moves, free (`Op100.move`) in the headline;
  `--lookup` prices them as L2 lookups instead.

Stock is the served stack: vLLM's W8A8 int8 linears (per-token dynamic activation scales, per-channel weight scales,
a dequant epilogue), FlashAttention-3 in bf16 over a bf16 KV cache, a bf16 lm_head, and separate norm, RoPE, SiLU·mul
and quantize kernels (`lane_ops(kind, "stock")`). Each tensor component is priced at its best instruction, so stock's
own padding (FA3's 64-row tile at decode) is not counted in its favour.

Prices are units a register written, three readings (`READINGS`): `floor` (H32's class table, 32 for every writer),
`paired` (the cheapest measured context: an ALU op beside an FMA-pipe op) and `alone` (the op's own measured
price). One dense int8 MAC is 1 unit and one bf16 MAC 2. DistinctWrites forces 32 a committed word that is distinct and
non-free on the salt grid; premises (`PREMISES`) raise that per class. Each op's output has a class:

- T: arithmetic. A target when its input carries live noise (the variant is `fused`: the f32 chain from a noised
  accumulator, before any rounding) and the word is committed; otherwise uncovered.
- M (MUFU) and C (I2F): targets like T, forced at 32 against a price of 256 and 128 unless MufuSpeed or CvtMinimal.
- S: a selection (FMNMX, IMNMX), equal to one of its inputs on almost every salt.
- R: a rounding to bf16 or int8, a pack, or a bf16 unpack. The served noise (at most 24 accumulator units) is below
  the rounding step on most elements, so the output is its salt-independent value (free under M0) there; `flip_shares`
  measures the share of elements whose output still varies, which the headline does not credit.
- X: a shuffle or a reduction to memory (a copy or a sum another word already holds); H: bf16x2 arithmetic (rounded).

Designs (`PLANS`): `linears` is today (ssn_1l on the linears, the rest stock and uncovered); `lean` adds lean lane
idioms; `best` adds an int8 lm_head under ssn_1l, int8 attention (`ssn_sh` on QKᵀ and PV), fused C-all lanes and
committed noise-carrying padding at small m; `best_p3` and `best_p4` add premises up to ranks 3 and 4a; `best_sh` puts
ssn_sh on the linears too; `best_pad64` pads every matmul below 64 rows to wgmma's 64; `best_dither` and
`best_p4_dither` round every int8 / u8 quantize as floor(x·s + u), u from an in-kernel xorshift stream seeded by a raw
coin word (`dithered`), which makes the rounded values and their packs targets at a T3 cost (`T3["dither"]`).

`ssn_sh` is ssn_1l's sibling for small n: step 1 holds 28 data lanes and 4 bits a row against the injective column
code, and every later step 28 data lanes and one register shared by every row, a raw coin word (a move), against W₀ =
[1, 0, 0, 0]; the accumulator's init cancels the shared bytes. Its forming is one LOP3 a row a call, at α = 32/28.

`check()` reproduces the served model's decode and prefill MAC shares and noise_sweep's ssn_1l terms, checks ssn_sh's and the int8 PV blocks'
distinctness on every data case (with mutations that must fail), the softmax chain's classes on a noisy row, the
magic-number conversion's exact range, the degree-3 exp2's error, that dithered rounds vary across salts where plain
ones don't (`sim_dither`) and the small-m instruction choice. Output: a JSON
report (`--json PATH`) and a text table (`--table`).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import noise_sweep as nsw

H32 = nsw.H32
READINGS = ("floor", "paired", "alone")
PRICE = {
    "FADD": (32, 32.26, 32.26), "FMUL": (32, 32.26, 32.26), "FFMA": (32, 32.26, 32.26), "FFMA3": (32, 48.09, 48.09),
    "IADD3": (32, 32.35, 64.13), "LOP3": (32, 32.31, 64.13), "SHF": (32, 32.31, 64.13), "PRMT": (32, 32.28, 64.13),
    "FMNMX": (32, 32.3, 64.13), "UNPK": (32, 32.28, 64.13), "F2FP": (32, 65.5, 65.5), "F2I": (32, 128.2, 128.2),
    "I2F": (32, 128.0, 128.0), "MUFU": (32, 256.0, 256.0), "SHFL": (32, 128.0, 128.0), "HFMA2": (32, 32.28, 32.28),
    "RED": (32, 1000.0, 1000.0), "LOOKUP": (32, 472.0, 472.0),
}
#: I2F.F32.S32 is unmeasured (F2I's 128.2 assumed); RED's is a measured lower bound for a reduction to memory
PRICE_ESTIMATED = ("I2F", "RED", "HFMA2", "FMNMX")
TENSOR = {"s8": {"kreg": 32, "mma": 48.07}, "bf16": {"kreg": 16, "mma": 48.07}}
REFINED_FLOOR = 32.26

#: name: (rank, what it credits)
PREMISES = {
    "MeasuredFloor": ("1", "every committed word costs at least 32.26 (the cheapest measured writer)"),
    "CvtMinimal": ("2", "I2F's function costs its measured price (checkable by a family search, derive_minimal-style)"),
    "TensorSemantics": ("3", "a tensor instruction's word costs its measured price at its shape"),
    "MufuSpeed": ("4a", "MUFU's functions (ex2, tanh, rsq, rcp) cost at least MUFU's measured 256"),
}

FA3_BLOCK = 128
I8_BLOCK = 112
EPT_STOCK = 8
EPT_LEAN = 32
NOISE_RANGE = 24


def _ri(reading: str) -> int:
    return READINGS.index(reading)


def wgmma_price(n: int, reading: str) -> float:
    """Units a register for a wgmma at width N: measured at 8, 16, 32 and ≥ 64 (144, 80, 48, 32.003), 16 + 1024/N between."""
    return 32.0 if reading == "floor" else max(32.003, 16 + 1024 / n)


def tc_best(m: int, dtype: str, reading: str) -> tuple[float, float, str]:
    """(units a tile MAC, tile MACs a real MAC, instruction) at the cheapest of wgmma (rows = m, M = 64), wgmma swapped
    (N = m rounded to 8), mma.sync m16n8k32 / m16n8k16 swapped (N 8) and mma.sync (M 16)."""
    kreg = TENSOR[dtype]["kreg"]
    mma = 32.0 if reading == "floor" else TENSOR[dtype]["mma"]

    def ceil(x: float, q: int) -> int:
        return q * math.ceil(x / q)

    opts = [(wgmma_price(256, reading) / kreg, ceil(m, 64) / m, "wgmma"),
            (wgmma_price(min(256, ceil(m, 8)), reading) / kreg, ceil(m, 8) / m, "wgmma swapped"),
            (mma / kreg, ceil(m, 8) / m, "mma.sync swapped"),
            (mma / kreg, ceil(m, 16) / m, "mma.sync")]
    if dtype == "s8" and reading != "floor":
        opts.append((32.01 / 16, ceil(m, 8) / m, "mma.sync m16n8k16 swapped"))
    return min(opts, key=lambda o: (o[0] * o[1], o[1]))


# ---------------------------------------------------------------- noising designs for one matmul

def ssn1l_terms(m: int, n: int, k: int, code_lanes: int = 4) -> dict:
    """noise_sweep's ssn_1l at a code of `code_lanes` lanes (5 reach n ≤ 371,291): α, forming and coins a MAC."""
    s = 1 + math.ceil(max(0, k - (32 - code_lanes)) / 31)
    words = nsw.ssn_1l_words(s + code_lanes - 4)
    return {"design": f"ssn_1l({code_lanes})", "alpha": H32 * s / k,
            "forming": H32 * nsw.ssn_1l_forming_writes(s) / (n * k),
            "coins": 4 * words * nsw.COIN_PRICED_BYTE / (n * k), "coin_bytes_call": 4 * words * m}


def steps_sh(k: int) -> int:
    return 1 + math.ceil(max(0, k - nsw.DATA_PER_STEP) / nsw.DATA_PER_STEP)


def ssn_sh_terms(m: int, n: int, k: int) -> dict:
    """ssn_sh: one LOP3 a row a call (step 1's bits) and S − 1 shared raw coin words a call."""
    s = steps_sh(k)
    coin_b = 0.5 * m + 4 * (s - 1)
    return {"design": "ssn_sh", "alpha": H32 * s / k, "forming": H32 / (n * k),
            "coins": coin_b * nsw.COIN_PRICED_BYTE / (m * n * k), "coin_bytes_call": coin_b}


# ---------------------------------------------------------------- lane idioms (op, count an element, class)

def lane_ops(kind: str, variant: str, exp: str = "mufu", dv: int = 64) -> list[tuple[str, float, str]]:
    red_s, red_l = 5 / EPT_STOCK, 5 / EPT_LEAN
    if kind == "epi":
        return {"stock": [("I2F", 1, "C"), ("FMUL", 2, "T"), ("F2FP", 0.5, "R")],
                "lean": [("I2F", 1, "C"), ("FMUL", 2, "T"), ("F2FP", 0.5, "R")],
                "fused": [("I2F", 1, "C"), ("FMUL", 2, "T")]}[variant]
    if kind == "epi_bf16":
        return [("F2FP", 0.5, "R")]
    if kind in ("norm", "norm_nores"):
        res = kind == "norm"
        add = [("UNPK", 2, "R"), ("FADD", 1, "T"), ("F2FP", 0.5, "R")] if res else [("UNPK", 1, "R")]
        if variant == "stock":
            return add + [("FFMA", 1, "T"), ("SHFL", red_s, "X"), ("FADD", red_s, "T"), ("FMUL", 1, "T"),
                          ("UNPK", 1, "R"), ("FMUL", 1, "T"), ("F2FP", 0.5, "R"),
                          ("UNPK", 1, "R"), ("FMNMX", 1, "S"), ("SHFL", red_s, "X"), ("FMNMX", red_s, "S"),
                          ("FMUL", 1, "T"), ("F2I", 1, "R"), ("PRMT", 0.75, "R")]
        if variant == "lean":
            return add + [("FFMA", 1, "T"), ("SHFL", red_l, "X"), ("FADD", red_l, "T"), ("FMUL", 1, "T"),
                          ("FMNMX", 1, "S"), ("SHFL", red_l, "X"), ("FMNMX", red_l, "S"), ("FFMA", 1, "R"),
                          ("PRMT", 0.75, "R")]
        epi = ([("FADD", 1, "T")] if res else []) + [("FFMA", 1, "T"), ("RED", 1 / 128, "X")]
        return epi + [("FMUL", 2, "T"), ("FMNMX", 1, "S"), ("SHFL", red_l, "X"), ("FMNMX", red_l, "S"),
                      ("FFMA", 1, "R"), ("PRMT", 0.75, "R")]
    if kind == "rope":
        return {"stock": [("UNPK", 2, "R"), ("FMUL", 1, "T"), ("FFMA3", 1, "T"), ("F2FP", 0.5, "R")],
                "lean": [("UNPK", 1, "R"), ("FMUL", 1, "T"), ("FFMA3", 1, "T"), ("F2FP", 0.5, "R")],
                "fused": [("FMUL", 2, "T"), ("FADD", 1, "T")]}[variant]
    if kind == "kvq":
        # q and k are Hadamard-rotated before they're quantized: log2(64) butterfly FADDs on the ~5/6 that are q or k
        return [] if variant != "fused" else [("FADD", 5, "T"), ("FMNMX", 1, "S"), ("SHFL", 2 / 16, "X"),
                                              ("FMNMX", 2 / 16, "S"), ("FFMA", 1, "R"), ("PRMT", 0.75, "R")]
    if kind == "silu":
        return {"stock": [("UNPK", 2, "R"), ("FMUL", 1, "T"), ("MUFU", 1, "M"), ("FADD", 1, "T"), ("MUFU", 1, "M"),
                          ("FMUL", 2, "T"), ("F2FP", 0.5, "R")],
                "lean": [("HFMA2", 1.5, "H"), ("MUFU", 0.5, "H")],
                "fused": [("FMUL", 1, "T"), ("MUFU", 1, "M"), ("FFMA", 1, "T"), ("FMUL", 1, "T")]}[variant]
    if kind == "quant":
        return {"stock": [("UNPK", 1, "R"), ("FMNMX", 1, "S"), ("SHFL", red_s, "X"), ("FMNMX", red_s, "S"),
                          ("FMUL", 1, "T"), ("F2I", 1, "R"), ("PRMT", 0.75, "R")],
                "lean": [("UNPK", 1, "R"), ("FMNMX", 1, "S"), ("SHFL", red_l, "X"), ("FMNMX", red_l, "S"),
                         ("FFMA", 1, "R"), ("PRMT", 0.75, "R")],
                "fused": [("FMNMX", 1, "S"), ("SHFL", red_l, "X"), ("FMNMX", red_l, "S"), ("FFMA", 1, "R"),
                          ("PRMT", 0.75, "R")]}[variant]
    if kind == "attn_out":
        return {"stock": [("FMUL", 1, "T"), ("F2FP", 0.5, "R")], "lean": [("FMUL", 1, "T"), ("F2FP", 0.5, "R")],
                "fused": [("FMUL", 1, "T")]}[variant]
    if kind == "softmax":
        if variant in ("stock", "lean"):
            b, q = FA3_BLOCK, FA3_BLOCK / 4
            return [("FMNMX", 1, "S"), ("FFMA", 1, "T"), ("MUFU", 1, "M"), ("FADD", 1, "T"), ("F2FP", 0.5, "R"),
                    ("FMUL", dv / b, "T"), ("SHFL", 2 / q, "X"), ("FMNMX", 2 / q, "S"), ("MUFU", 1 / q, "M"),
                    ("FMUL", 1 / q, "T")]
        b, q = I8_BLOCK, I8_BLOCK / 4
        ex = [("MUFU", 1, "M")] if exp == "mufu" else [
            ("FMNMX", 1, "S"), ("FADD", 2, "R"), ("FADD", 1, "T"), ("FFMA", 3, "T"), ("SHF", 1, "R"), ("IADD3", 1, "T")]
        return [("IADD3", 1, "T"), ("FADD", 1, "T"), ("FMUL", 1, "T"), ("FMNMX", 1, "S"), ("FFMA", 1, "T")] + ex + [
            ("FADD", 1, "T"), ("FFMA", 1, "R"), ("PRMT", 0.75, "R"), ("IADD3", dv / b, "T"), ("FADD", dv / b, "T"),
            ("FFMA3", dv / b, "T"), ("SHFL", 2 / q, "X"), ("FMNMX", 2 / q, "S"), ("MUFU", 1 / q, "M")]
    if kind == "combine":
        return {"stock": [("UNPK", 9, "R"), ("FADD", 8, "T"), ("F2FP", 0.5, "R")],
                "lean": [("UNPK", 9, "R"), ("FADD", 8, "T"), ("F2FP", 0.5, "R")],
                "fused": [("FADD", 8, "T")]}[variant]
    if kind == "router":
        return [("UNPK", 1, "R"), ("FMUL", 1, "T"), ("MUFU", 2, "M"), ("FADD", 2, "T"), ("FMNMX", 8, "S")]
    if kind == "sampler":
        return [("UNPK", 1, "R"), ("FMNMX", 3, "S")] if variant == "stock" else [("FMNMX", 3, "S")]
    if kind == "move":
        return [("LOOKUP", 1, "G")]
    raise ValueError(kind)


def dithered(ops: list[tuple[str, float, str]]) -> list[tuple[str, float, str]]:
    """Dithered rounding: each round to int8 / u8 (`FFMA` magic, class R) becomes FFMA(x, s, u) and a round-down magic
    FADD, with u ∈ [0, 1) from an in-kernel xorshift seeded by a raw coin word (6 ALU ops a 4 bytes) placed in a float's
    mantissa and offset (PRMT, FADD); the rounded values then vary with u, so they and their pack are targets."""
    out = []
    for op, cnt, cls in ops:
        if (op, cls) == ("FFMA", "R"):
            out += [("FFMA", cnt, "T"), ("FADD", cnt, "T"), ("LOP3", 1.5 * cnt, "T"), ("PRMT", cnt, "T"),
                    ("FADD", cnt, "T")]
        elif (op, cls) == ("PRMT", "R"):
            out.append(("PRMT", cnt, "T"))
        else:
            out.append((op, cnt, cls))
    return out


def lane_cost(ops, reading: str, premises: frozenset, reachable: bool) -> dict:
    """Honest units, forced units, committed words and the honest units no word forces, an element."""
    ri = _ri(reading)
    out = {"honest": 0.0, "forced": 0.0, "words": 0.0, "unforced": 0.0}
    for op, cnt, cls in ops:
        p = PRICE[op][ri]
        h, f = cnt * p, 0.0
        if reachable and cls in ("T", "M", "C"):
            base = REFINED_FLOOR if "MeasuredFloor" in premises else H32
            if cls == "M" and "MufuSpeed" in premises:
                base = PRICE["MUFU"][1]
            if cls == "C" and "CvtMinimal" in premises:
                base = PRICE["I2F"][1]
            if op == "FFMA3" and "MeasuredFloor" in premises:
                base = REFINED_FLOOR
            f = cnt * min(p, base)
            out["words"] += cnt
        out["honest"] += h
        out["forced"] += f
        if f == 0:
            out["unforced"] += h
    return out


# ---------------------------------------------------------------- models and points

LLAMA = {"d": 2048, "layers": 16, "heads": 32, "kv_heads": 8, "dk": 64, "dv": 64, "ffn": 8192, "vocab": 128_256,
         "linears": dict(nsw.LLAMA)}
V4 = {"d": 7168, "layers": 1, "heads": 128, "q_lora": 1536, "kv_lora": 512, "nope": 128, "rope": 64, "v": 128,
      "experts": 256, "top_k": 8, "shared": 1, "inter": 2048,
      "dense": {"q_a": (1536, 7168), "kv_a": (576, 7168), "q_b": (24576, 1536), "kv_b": (32768, 512),
                "o": (7168, 16384), "sh_gate_up": (4096, 7168), "sh_down": (7168, 2048)},
      "expert": {"e_gate_up": (4096, 7168), "e_down": (7168, 2048)}}

POINTS = {
    "T512_64": {"prompts": 4, "tokens": 16, "decode": False, "label": "T 512 at 64 real tokens: 4 × 16-token prompts"},
    "T512_512": {"prompts": 2, "tokens": 256, "decode": False, "label": "T 512 at 512 rows: 2 × 256-token prompts"},
    "T8192_8x1k": {"prompts": 8, "tokens": 1024, "decode": False, "label": "T 8,192: 8 × 1,024-token prompts"},
    "T8192_1x8k": {"prompts": 1, "tokens": 8192, "decode": False, "label": "T 8,192: one 8,192-token prompt"},
    "dec64_2k": {"prompts": 64, "tokens": 1, "context": 2048, "decode": True, "label": "decode, batch 64, context 2K"},
    "dec64_8k": {"prompts": 64, "tokens": 1, "context": 8192, "decode": True, "label": "decode, batch 64, context 8K"},
}

#: the served step's measured parts at T 8,192 (8 × 1K, sync): the step, noise_sweep's GEMM time and the attention
#: share; the rest is the step less those and the host gap
STOCK_T8192_MS = 37.77
GEMM_T8192_MS = nsw.STEP["T8192"]["gemm_ms"]
ATT_T8192_MS = 2.9
HOST_T8192_MS = 8.73
REST_T8192_MS = STOCK_T8192_MS - GEMM_T8192_MS - ATT_T8192_MS - HOST_T8192_MS
HOST_FIT_MS = (2.42, 0.666e-3)
HBM_B_S = 4.0e12
LAUNCH_MS = 4e-3


def point_shape(p: dict) -> dict:
    rows = p["prompts"] * p["tokens"]
    keys = (p["context"] + 1) if p["decode"] else (p["tokens"] + 1) / 2
    return {"rows": rows, "keys": keys, "lm_rows": p["prompts"], "decode": p["decode"],
            "q_per_call": 1 if p["decode"] else p["tokens"]}


@dataclass
class Comp:
    name: str
    family: str
    count: float
    kind: str = ""
    m: int = 0
    n: int = 0
    k: int = 0
    dk: int = 0
    dv: int = 0
    keys: float = 0.0
    rows: float = 0.0
    calls: float = 1.0


def components(model: str, point: str) -> list[Comp]:
    sh = point_shape(POINTS[point])
    t, keys = sh["rows"], sh["keys"]
    out = []
    if model == "llama":
        L, cfg = LLAMA["layers"], LLAMA
        for name, (n, k) in cfg["linears"].items():
            out.append(Comp(name, "linear", t * n * k * L, m=t, n=n, k=k, calls=L))
            out.append(Comp(f"epi_{name}", "lane", t * n * L, kind="epi"))
        out.append(Comp("norm", "lane", 2 * t * cfg["d"] * L, kind="norm"))
        out.append(Comp("rope", "lane", t * (cfg["heads"] + cfg["kv_heads"]) * cfg["dk"] * L, kind="rope"))
        qkv = t * (cfg["heads"] + 2 * cfg["kv_heads"]) * cfg["dk"] * L
        out.append(Comp("qkv_quant", "lane", qkv, kind="kvq"))
        scores = t * cfg["heads"] * keys * L
        gqa = cfg["heads"] // cfg["kv_heads"]
        m_att = min(64, sh["q_per_call"] * gqa)
        rows_att = t * cfg["heads"] * L
        out.append(Comp("qk", "qk", scores * cfg["dk"], m=m_att, dk=cfg["dk"], dv=cfg["dv"], keys=keys, rows=rows_att,
                        calls=L))
        out.append(Comp("pv", "pv", scores * cfg["dv"], m=m_att, dk=cfg["dk"], dv=cfg["dv"], keys=keys,
                        rows=rows_att, calls=L))
        out.append(Comp("softmax", "softmax", scores, kind="softmax", dv=cfg["dv"]))
        out.append(Comp("attn_out", "lane", t * cfg["heads"] * cfg["dv"] * L, kind="attn_out"))
        out.append(Comp("quant_o", "lane", t * cfg["d"] * L, kind="quant"))
        out.append(Comp("silu", "lane", t * cfg["ffn"] * L, kind="silu"))
        out.append(Comp("quant_down", "lane", t * cfg["ffn"] * L, kind="quant"))
        lm = sh["lm_rows"]
        out.append(Comp("final_norm", "lane", lm * cfg["d"], kind="norm_nores"))
        out.append(Comp("lm_head", "lm_head", lm * cfg["vocab"] * cfg["d"], m=lm, n=cfg["vocab"], k=cfg["d"]))
        out.append(Comp("lm_epi", "lm_epi", lm * cfg["vocab"], kind="epi"))
        out.append(Comp("sampler", "lane", lm * cfg["vocab"], kind="sampler"))
        out.append(Comp("embed", "move", t * cfg["d"] / 2, kind="move"))
        out.append(Comp("kv_write", "move", t * 2 * cfg["kv_heads"] * cfg["dk"] / 2 * L, kind="move"))
        return out
    cfg = V4
    for name, (n, k) in cfg["dense"].items():
        out.append(Comp(name, "linear", t * n * k, m=t, n=n, k=k))
        out.append(Comp(f"epi_{name}", "lane", t * n, kind="epi"))
    m_e = t * cfg["top_k"] / cfg["experts"]
    for name, (n, k) in cfg["expert"].items():
        out.append(Comp(name, "linear", t * cfg["top_k"] * n * k, m=max(1, round(m_e)), n=n, k=k,
                        calls=cfg["experts"]))
        out.append(Comp(f"epi_{name}", "lane", t * cfg["top_k"] * n, kind="epi"))
    out.append(Comp("router_gate", "router_gemm", t * cfg["experts"] * cfg["d"], m=t, n=cfg["experts"], k=cfg["d"]))
    out.append(Comp("router", "lane", t * cfg["experts"], kind="router"))
    out.append(Comp("norm", "lane", 2 * t * cfg["d"], kind="norm"))
    out.append(Comp("norm_lora", "lane", t * (cfg["q_lora"] + cfg["kv_lora"]), kind="norm_nores"))
    out.append(Comp("rope", "lane", t * (cfg["heads"] + 1) * cfg["rope"], kind="rope"))
    if sh["decode"]:
        dk, dv, m_att = cfg["kv_lora"] + cfg["rope"], cfg["kv_lora"], cfg["heads"]
    else:
        dk, dv, m_att = cfg["nope"] + cfg["rope"], cfg["v"], min(64, sh["q_per_call"])
    qkv = t * cfg["heads"] * (2 * (cfg["nope"] + cfg["rope"]) + cfg["v"])
    out.append(Comp("qkv_quant", "lane", qkv, kind="kvq"))
    scores = t * cfg["heads"] * keys
    rows_att = t * cfg["heads"]
    out.append(Comp("qk", "qk", scores * dk, m=m_att, dk=dk, dv=dv, keys=keys, rows=rows_att))
    out.append(Comp("pv", "pv", scores * dv, m=m_att, dk=dk, dv=dv, keys=keys, rows=rows_att))
    out.append(Comp("softmax", "softmax", scores, kind="softmax", dv=dv))
    out.append(Comp("attn_out", "lane", t * cfg["heads"] * dv, kind="attn_out"))
    out.append(Comp("quant_o", "lane", t * cfg["heads"] * cfg["v"], kind="quant"))
    n_ffn = cfg["top_k"] + cfg["shared"]
    out.append(Comp("silu", "lane", t * n_ffn * cfg["inter"], kind="silu"))
    out.append(Comp("quant_down", "lane", t * n_ffn * cfg["inter"], kind="quant"))
    out.append(Comp("combine", "lane", t * cfg["d"], kind="combine"))
    return out


# ---------------------------------------------------------------- plans

@dataclass(frozen=True)
class Plan:
    name: str
    linears: str = "ssn_1l"
    lm_head: str = "stock"
    attention: str = "stock"
    exp: str = "mufu"
    lanes: str = "stock"
    premises: frozenset = frozenset()
    commit_pad: bool = True
    small_m: str = "best"
    dither: bool = False
    note: str = ""


PLANS = [
    Plan("linears", note="today: ssn_1l on the linears; every other component stock and uncovered"),
    Plan("lean", lanes="lean", note="linears + the lean lane idioms (uncovered, but smaller)"),
    Plan("best", lm_head="ssn_1l", attention="i8", exp="best", lanes="fused",
         note="int8 lm_head (ssn_1l, 5-lane code), int8 attention (ssn_sh), fused C-all lanes, committed padding"),
    Plan("best_p3", lm_head="ssn_1l", attention="i8", exp="best", lanes="fused",
         premises=frozenset({"MeasuredFloor", "CvtMinimal", "TensorSemantics"}), note="best + premises of rank <= 3"),
    Plan("best_p4", lm_head="ssn_1l", attention="i8", exp="best", lanes="fused",
         premises=frozenset({"MeasuredFloor", "CvtMinimal", "TensorSemantics", "MufuSpeed"}),
         note="best + premises up to rank 4a (MufuSpeed)"),
    Plan("best_sh", linears="ssn_sh", lm_head="ssn_1l", attention="i8", exp="best", lanes="fused",
         note="best with ssn_sh on the linears too (forming and coins ~0, alpha 32/28)"),
    Plan("best_pad64", lm_head="ssn_1l", attention="i8", exp="best", lanes="fused", small_m="wgmma64",
         note="best with every matmul below 64 rows padded to wgmma's 64 with committed noise-carrying rows"),
    Plan("best_dither", lm_head="ssn_1l", attention="i8", exp="best", lanes="fused", small_m="wgmma64", dither=True,
         note="best_pad64 with dithered rounding of every int8 / u8 quantize (T3 cost)"),
    Plan("best_p4_dither", lm_head="ssn_1l", attention="i8", exp="best", lanes="fused", dither=True,
         premises=frozenset({"MeasuredFloor", "CvtMinimal", "TensorSemantics", "MufuSpeed"}),
         note="best_p4 with dithered rounding"),
]
PLAN = {p.name: p for p in PLANS}

T3 = {
    "linear": ("~0 (served noise std 6.0 / 9.2 units at K 2,048 / 8,192 vs ~330 quant)", 0.0),
    "lm_head": ("+0.03% (estimate: int7 lm_head's measured +0.07-0.08%, /4 for int8's halved step)", 0.0003),
    "qk": ("+0.07% (estimate: int7 QKᵀ with q and k Hadamard-rotated, measured +0.29% at 8K, /4)", 0.0007),
    "pv": ("+0.16% (estimate: int7 P·V with the sink in its own block, measured +0.65% at 8K, /4; int7 QKᵀ and P·V "
           "together measured +0.66%, so the sum with qk is conservative)", 0.0016),
    "silu": ("~0 (tanh.approx.f32: relative error <= 2^-10.99, below bf16's 2^-9; estimate)", 0.0),
    "softmax": ("~0 (poly exp2 checked below; P is u8 at 1/255)", 0.0),
    "dither": ("+1.2% (estimate: rounding variance x2 on every int8 activation, interpolated between int8 and int7 "
               "activations' measured +3.5%; P and QKᵀ's int8 inside the attention terms)", 0.012),
}


def tensor_comp(c: Comp, plan: Plan, reading: str, stock: bool) -> dict:
    """Honest, forced and stock units of a tensor component, its words, forming and coins."""
    fam = c.family
    dtype_stock = "bf16" if fam in ("qk", "pv", "lm_head", "router_gemm") else "s8"
    sp, spad, sinstr = tc_best(c.m, dtype_stock, reading)
    stock_units = c.count * sp * spad
    covered = not stock and (fam == "linear" or (fam == "lm_head" and plan.lm_head != "stock")
                             or (fam in ("qk", "pv") and plan.attention == "i8"))
    if not covered:
        return {"stock": stock_units, "honest": stock_units, "forced": 0.0, "words": 0.0, "forming": 0.0, "coins": 0.0,
                "coin_calls": [], "instr": sinstr, "design": "stock", "alpha": 1.0, "unforced": stock_units,
                "pad": spad}
    hp, hpad, hinstr = tc_best(c.m, "s8", reading)
    if plan.small_m == "wgmma64" and c.m < 64:
        hp, hpad, hinstr = wgmma_price(256, reading) / TENSOR["s8"]["kreg"], 64 / c.m, "wgmma, rows padded to 64"
    lanes = 4 if c.n <= nsw.SSN_CODE_N_MAX else 5
    if fam == "linear":
        t = ssn1l_terms(c.m, c.n, c.k, lanes) if plan.linears == "ssn_1l" else ssn_sh_terms(c.m, c.n, c.k)
        forming, coins = t["forming"] * c.count, t["coins"] * c.count
        coin_calls = [(t["coin_bytes_call"], c.calls * (c.count / (c.m * c.n * c.k * c.calls)))]
    elif fam == "lm_head":
        t = ssn1l_terms(c.m, c.n, c.k, lanes)
        forming, coins = t["forming"] * c.count, t["coins"] * c.count
        coin_calls = [(t["coin_bytes_call"], 1)]
    else:
        scores = c.count / (c.dk if fam == "qk" else c.dv)
        if fam == "qk":
            t = ssn_sh_terms(1, max(1.0, c.keys), c.dk)
            per_row = 1.0
        else:
            t = ssn_sh_terms(1, c.dv, I8_BLOCK)
            t["alpha"] = H32 * math.ceil(c.keys / nsw.DATA_PER_STEP) / c.keys
            per_row = math.ceil(c.keys / I8_BLOCK)
        forming = H32 * per_row * c.rows
        coin_b = 0.5 * per_row * c.rows
        coins = coin_b * nsw.COIN_PRICED_BYTE
        coin_calls = [(coin_b / c.calls, c.calls)]
        _ = scores
    a = t["alpha"]
    honest = c.count * a * hp * hpad
    tile_forced = hp if "TensorSemantics" in plan.premises else min(hp, H32 / TENSOR["s8"]["kreg"])
    pad_c = hpad if plan.commit_pad else 1.0
    forced = c.count * a * pad_c * tile_forced
    words = c.count * a * pad_c / TENSOR["s8"]["kreg"]
    return {"stock": stock_units, "honest": honest + forming, "forced": forced, "words": words, "forming": forming,
            "coins": coins, "coin_calls": coin_calls, "instr": hinstr, "design": t["design"], "alpha": a,
            "unforced": forming + (honest - forced if not plan.commit_pad else 0.0), "pad": hpad}


def lane_variant(c: Comp, plan: Plan) -> tuple[str, bool]:
    if c.kind == "kvq":
        return ("fused" if plan.attention == "i8" else "stock"), plan.attention == "i8"
    if c.kind == "sampler":
        return ("fused" if plan.lm_head != "stock" and plan.lanes == "fused" else "stock"), False
    if c.kind == "attn_out" and plan.lanes == "fused":
        return "fused", plan.attention == "i8"
    if c.family == "softmax":
        return ("i8" if plan.attention == "i8" else "stock"), plan.attention == "i8"
    if c.family == "lm_epi":
        if plan.lm_head == "stock":
            return "bf16", False
        return ("fused" if plan.lanes == "fused" else "stock"), plan.lanes == "fused"
    if c.kind in ("router", "move"):
        return "stock", False
    if plan.lanes == "fused":
        return "fused", c.kind != "quant"
    return plan.lanes, False


def lane_comp(c: Comp, plan: Plan, reading: str, stock: bool, lookup: bool) -> dict:
    sv = "stock"
    if c.family == "lm_epi":
        s_ops = lane_ops("epi_bf16", "stock")
    elif c.kind == "softmax":
        s_ops = lane_ops("softmax", "stock", dv=c.dv)
    else:
        s_ops = lane_ops(c.kind, sv)
    if c.kind == "move" and not lookup:
        s_ops = []
    st = lane_cost(s_ops, reading, frozenset(), False)
    if stock:
        h, design, ops = st, "stock", s_ops
    else:
        v, reach = lane_variant(c, plan)
        if c.family == "lm_epi" and v == "bf16":
            ops = lane_ops("epi_bf16", "stock")
        elif c.kind == "softmax":
            if v == "stock":
                ops = s_ops
            else:
                cands = {e: lane_ops("softmax", "i8", exp=e, dv=c.dv) for e in ("mufu", "poly")}
                if plan.dither:
                    cands = {e: dithered(o) for e, o in cands.items()}
                if plan.exp == "best":
                    e = min(cands, key=lambda e: (lambda r: (r["honest"] - r["forced"], r["honest"]))(
                        lane_cost(cands[e], reading, plan.premises, True)))
                else:
                    e = plan.exp
                ops, v = cands[e], f"i8_{e}" + ("_dither" if plan.dither else "")
        else:
            ops = s_ops if (c.kind == "move") else lane_ops(c.kind, v)
            if plan.dither and v == "fused" and any(cls == "R" for _, _, cls in ops):
                ops, reach, v = dithered(ops), True, "fused_dither"
        h = lane_cost(ops, reading, plan.premises, reach)
        design = v
    return {"stock": st["honest"] * c.count, "honest": h["honest"] * c.count, "forced": h["forced"] * c.count,
            "words": h["words"] * c.count, "unforced": h["unforced"] * c.count, "forming": 0.0, "coins": 0.0,
            "coin_calls": [], "design": design, "per_element": {"stock": st["honest"], "honest": h["honest"],
                                                                 "forced": h["forced"], "words": h["words"]},
            "ops": ops}


# ---------------------------------------------------------------- the step

def stock_ms(model: str, point: str) -> tuple[float, str]:
    """The stock step's time: measured at Llama T 8,192 (8 × 1K); else the measured parts scaled (an estimate)."""
    sh = point_shape(POINTS[point])
    t = sh["rows"]
    if model == "llama":
        if point == "T8192_8x1k":
            return STOCK_T8192_MS, "measured (the served step, sync)"
        host = HOST_FIT_MS[0] + HOST_FIT_MS[1] * t
        gemm = max(GEMM_T8192_MS * t / 8192, 1.24e9 / HBM_B_S * 1e3) + 0.525e9 / HBM_B_S * 1e3
        s_ref = 8192 * 512.5 * LLAMA["heads"] * LLAMA["layers"]
        scores = t * LLAMA["heads"] * sh["keys"] * LLAMA["layers"]
        if sh["decode"]:
            kv = t * sh["keys"] * LLAMA["layers"] * 2 * LLAMA["kv_heads"] * LLAMA["dk"] * 2
            att = kv / HBM_B_S * 1e3
        else:
            att = max(ATT_T8192_MS * scores / s_ref, LLAMA["layers"] * LAUNCH_MS)
        rest = max(REST_T8192_MS * t / 8192, LLAMA["layers"] * 8 * LAUNCH_MS)
        return host + gemm + att + rest, "estimate: host-gap fit + T 8,192's parts scaled, HBM floors"
    macs = sum(c.count for c in components("v4", point) if c.family == "linear")
    touched = V4["experts"] * (1 - (1 - V4["top_k"] / V4["experts"]) ** t)
    wbytes = sum(n * k for n, k in V4["dense"].values()) + touched * sum(n * k for n, k in V4["expert"].values())
    gemm_rate = LLAMA_MACS_T8192 / (GEMM_T8192_MS * 1e-3)
    gemm = max(macs / gemm_rate, wbytes / HBM_B_S) * 1e3
    att_c = [c for c in components("v4", point) if c.family in ("qk", "pv")]
    att_macs = sum(c.count for c in att_c)
    att_rate = 8192 * 512.5 * LLAMA["heads"] * LLAMA["layers"] * 128 / (ATT_T8192_MS * 1e-3)
    att = att_macs / att_rate * 1e3
    if sh["decode"]:
        att = max(att, t * sh["keys"] * (V4["kv_lora"] + V4["rope"]) * 2 / HBM_B_S * 1e3)
    lane_el = sum(c.count for c in components("v4", point) if c.family == "lane")
    rest = max(REST_T8192_MS * lane_el / LLAMA_LANE_EL_T8192, 10 * LAUNCH_MS)
    return gemm + att + rest, "estimate: one layer, no host gap; Llama T 8,192's rates, HBM floors"


LLAMA_MACS_T8192 = nsw.LLAMA_MACS_TOKEN * 8192
LLAMA_LANE_EL_T8192 = sum(c.count for c in components("llama", "T8192_8x1k") if c.family == "lane")


def evaluate(model: str, point: str, plan: Plan, reading: str = "paired", lookup: bool = False) -> dict:
    comps = components(model, point)
    rows, tot = [], {k: 0.0 for k in ("stock", "honest", "forced", "unforced", "words", "forming", "coins")}
    for c in comps:
        if c.family in ("linear", "lm_head", "qk", "pv", "router_gemm"):
            s = tensor_comp(c, plan, reading, stock=True)
            r = tensor_comp(c, plan, reading, stock=False)
            r["stock"] = s["stock"]
        else:
            r = lane_comp(c, plan, reading, stock=False, lookup=lookup)
        r["name"], r["family"], r["count"] = c.name, c.family, c.count
        fam = c.family if c.family in ("linear", "lm_head", "qk", "pv", "softmax") else c.kind
        r["t3"] = T3.get(fam, ("~0", 0.0)) if r["design"] not in ("stock", "bf16", "lean") else ("0 (stock)", 0.0)
        rows.append(r)
        for k in tot:
            tot[k] += r.get(k, 0.0)
    st = tot["stock"]
    matmul = {"linear", "lm_head", "qk", "pv"}
    covered_mm = [r for r in rows if r["family"] in matmul and r["design"] != "stock"]
    t1_mm = sum(r["honest"] - r["forced"] for r in covered_mm) / st
    cov = sum(r["honest"] - r["forced"] for r in rows if r not in covered_mm) / st
    coins = sum(r["coins"] for r in rows) / st
    lin = [r for r in rows if r["family"] == "linear"]
    lin_share = sum(r["stock"] for r in lin) / st
    groups = {}
    for r in rows:
        g = group_of(r["name"], r["family"])
        e = groups.setdefault(g, {"stock": 0.0, "honest": 0.0, "forced": 0.0, "unforced": 0.0, "words": 0.0,
                                  "designs": set()})
        for k in ("stock", "honest", "forced", "unforced", "words"):
            e[k] += r[k]
        e["designs"].add(r["design"])
    for g in groups.values():
        g["designs"] = sorted(g["designs"])
        g["stock_share"] = g["stock"] / st
        g["gap_share"] = (g["honest"] - g["forced"]) / st
        g["uncovered_share"] = g["unforced"] / st
    t2 = t2_of(model, point, rows, plan)
    t3 = sum(r["t3"][1] for r in rows if r["family"] in matmul | {"softmax"} or r["name"] == "silu") if any(
        r["design"] != "stock" for r in rows) else 0.0
    if plan.dither:
        t3 += T3["dither"][1]
    alpha = tot["honest"] / st
    return {"model": model, "point": point, "plan": plan.name, "reading": reading, "stock_units": st,
            "honest_units": tot["honest"], "alpha_w1": alpha, "linears_stock_share": lin_share,
            "t1_matmul": t1_mm, "coverage": cov, "t1": t1_mm + cov, "coins_uncredited": coins,
            "t1_coins_uncredited": t1_mm + cov + coins, "uncovered_share": tot["unforced"] / st,
            "slack_share": (tot["honest"] - tot["forced"] - tot["unforced"]) / st,
            "hash_bytes": 4 * tot["words"], "groups": groups, "t2": t2, "t3_estimate": t3,
            "components": [{k: v for k, v in r.items() if k not in ("ops", "coin_calls")} for r in rows]}


GROUPS = {"linear": "linears", "lm_head": "lm_head", "qk": "QK^T", "pv": "PV", "softmax": "softmax",
          "router_gemm": "router"}


def group_of(name: str, family: str) -> str:
    if family in GROUPS:
        return GROUPS[family]
    if name.startswith("epi_") or name == "lm_epi":
        return "dequant epilogue"
    return {"norm": "RMSNorm + residual + quant", "norm_lora": "RMSNorm + residual + quant",
            "final_norm": "RMSNorm + residual + quant", "rope": "RoPE", "qkv_quant": "attention quant (honest)",
"attn_out": "attention output", "quant_o": "quant",
            "quant_down": "quant", "silu": "SiLU*mul", "sampler": "sampler", "embed": "embedding + KV write",
            "kv_write": "embedding + KV write", "combine": "MoE combine", "router": "router"}[name]


def unit_rates() -> dict:
    """Units a second of each family at Llama T 8,192, from the measured parts (paired reading): the T2 time model."""
    rows = evaluate_stock_rows("llama", "T8192_8x1k")
    lin = sum(r for f, r in rows if f == "linear")
    att = sum(r for f, r in rows if f in ("qk", "pv", "softmax"))
    rest = sum(r for f, r in rows if f not in ("linear", "qk", "pv", "softmax"))
    return {"tensor": lin / (GEMM_T8192_MS * 1e-3), "attention": att / (ATT_T8192_MS * 1e-3),
            "lane": rest / (REST_T8192_MS * 1e-3)}


def evaluate_stock_rows(model: str, point: str) -> list[tuple[str, float]]:
    out = []
    for c in components(model, point):
        if c.family in ("linear", "lm_head", "qk", "pv", "router_gemm"):
            out.append((c.family, tensor_comp(c, PLAN["linears"], "paired", stock=True)["stock"]))
        else:
            out.append((c.family, lane_comp(c, PLAN["linears"], "paired", stock=True, lookup=False)["stock"]))
    return out


_RATES: dict = {}


def t2_of(model: str, point: str, rows: list[dict], plan: Plan) -> dict:
    """Step time over stock: the stock step, each component's extra honest work at its family's measured rate (work
    honest saves is not credited), every committed word exported and hashed (serial or overlapped), and each call's
    coin expansion."""
    if not _RATES:
        _RATES.update(unit_rates())
    base, how = stock_ms(model, point)
    extra = 0.0
    for r in rows:
        fam = {"linear": "tensor", "lm_head": "tensor", "router_gemm": "tensor", "qk": "attention", "pv": "attention",
               "softmax": "attention"}.get(r["family"], "lane")
        extra += max(0.0, r["honest"] - r["stock"]) / _RATES[fam] * 1e3
    byts = 4 * sum(r["words"] for r in rows)
    coin_ms = sum(nsw.coin_us(b) * n for r in rows for b, n in r.get("coin_calls", [])) * 1e-3
    out = {"stock_ms": base, "stock_ms_source": how, "extra_work_ms": extra, "coin_ms": coin_ms,
           "hash_gb": byts / 1e9}
    for key, u in nsw.HASH_UNITS_BYTE.items():
        out[f"hash_ms_{key}"] = byts * (u + nsw.EXPORT_UNITS_WORD / 4) / nsw.H200_MAC_S * 1e3
        out[f"t2_{key}"] = (base + extra + coin_ms + out[f"hash_ms_{key}"]) / base
    return out


# ---------------------------------------------------------------- small-instance checks

def _bytes_s8(w: np.ndarray) -> np.ndarray:
    """Signed bytes of 32-bit words (… × 4)."""
    u = np.stack([(w >> (8 * e)) & 0xFF for e in range(4)], axis=-1)
    return np.where(u > 127, u - 256, u)


def sim_ssn_sh(case: str, salts: int = 24, m: int = 4, n: int = 6, k: int = 4 * 28, seed: int = 1,
               mutate: str | None = None) -> dict:
    """ssn_sh on S = 1 + ⌈(k − 28)/28⌉ steps: words distinct and non-free (M0: raw coin words free). Mutations:
    `no_shared` drops W₀'s shared lane in step 3, `code_dup` gives two columns one code."""
    rng = np.random.default_rng(seed)
    a, b = nsw._data(case, m, n, 128, rng)
    a, b = a[:, :k], b[:k]
    s = steps_sh(k)
    code = nsw.ssn_a_w0(n, nsw.DATA_PER_STEP)[0].copy()
    if mutate == "code_dup":
        code[:, 1] = code[:, 0]
    d = nsw.DATA_PER_STEP
    apad = np.zeros((m, d * s), dtype=np.int64)
    bpad = np.zeros((d * s, n), dtype=np.int64)
    apad[:, :k], bpad[:k] = a, b
    ycols = [bpad[:d], code]
    lanes = []
    for t in range(1, s):
        lane = np.zeros((4, n), dtype=np.int64)
        if not (mutate == "no_shared" and t == 2):
            lane[0] = 1
        lanes.append(lane)
        ycols += [bpad[d * t:d * (t + 1)], lane]
    y1 = np.concatenate(ycols, axis=0)
    xs, raw, formed, inits = [], [], [], []
    bias = np.rint(code.sum(axis=0) / 2).astype(np.int64)
    for _ in range(salts):
        h = rng.integers(0, 2 ** 32, (m, 1), dtype=np.int64)
        r = rng.integers(0, 2 ** 32, (s - 1,), dtype=np.int64)
        bits = np.stack([(h[:, 0] >> (8 * e)) & 1 for e in range(4)], axis=1)
        rb = _bytes_s8(r)
        parts = [apad[:, :d], bits]
        for t in range(1, s):
            parts += [apad[:, d * t:d * (t + 1)], np.broadcast_to(rb[t - 1], (m, 4))]
        xs.append(np.concatenate(parts, axis=1))
        shared = sum(rb[t - 1] @ lanes[t - 1] for t in range(1, s))
        inits.append(np.broadcast_to(-(shared + bias), (m, n)))
        raw.append(np.concatenate([h[:, 0], r]))
        formed.append(nsw._pack(bits))
    x = np.stack(xs)
    y = np.broadcast_to(y1, (salts, *y1.shape))
    words = nsw._words(x, y, np.stack(inits))
    out = nsw._distinct([words, np.stack(formed)], [np.stack(raw)])
    err = words[:, :, :, -1] - (a @ b)[None]
    return out | {"served_err_mean": float(err.mean()), "served_err_std": float(err.std()), "steps": s}


PV_CASES = ["random", "zero_row", "identical_rows", "uniform_p", "onehot_p", "identical_cols", "zero_vblock"]


def sim_pv_blocks(case: str, salts: int = 24, m: int = 4, dv: int = 6, blocks: int = 3, steps: int = 2,
                  seed: int = 2, mutate: str | None = None) -> dict:
    """Int8 PV under ssn_sh, a fresh step-1 code a key block: u8 P (m × L) · s8 V (L × dv), each block's k32 words
    committed. Mutation `code_once` codes only block 1, so later blocks' words repeat on degenerate P."""
    rng = np.random.default_rng(seed)
    d = nsw.DATA_PER_STEP
    blk = d * steps
    L = blk * blocks
    p = rng.integers(0, 256, (m, L))
    v = np.clip(np.rint(rng.normal(0, 25, (L, dv))), -127, 127).astype(np.int64)
    if case == "zero_row":
        p[0] = 0
    elif case == "identical_rows":
        p[1] = p[0]
    elif case == "uniform_p":
        p[:] = 7
    elif case == "onehot_p":
        p[:] = 0
        p[:, 3] = 255
    elif case == "identical_cols":
        v[:, 1] = v[:, 0]
    elif case == "zero_vblock":
        v[blk:2 * blk] = 0
    code = nsw.ssn_a_w0(dv, d)[0]
    lane = np.zeros((4, dv), dtype=np.int64)
    lane[0] = 1
    all_words, raws, formed = [], [], []
    for bi in range(blocks):
        pb, vb = p[:, bi * blk:(bi + 1) * blk], v[bi * blk:(bi + 1) * blk]
        coded = not (mutate == "code_once" and bi > 0)
        ycols = []
        for t in range(steps):
            ycols += [vb[d * t:d * (t + 1)], code if (t == 0 and coded) else lane]
        y1 = np.concatenate(ycols, axis=0)
        xs, rw, fm, inits = [], [], [], []
        bias = np.rint(code.sum(axis=0) / 2).astype(np.int64) if coded else 0
        for _ in range(salts):
            h = rng.integers(0, 2 ** 32, (m,), dtype=np.int64)
            r = rng.integers(0, 2 ** 32, (steps,), dtype=np.int64)
            rb = _bytes_s8(r)
            bits = np.stack([(h >> (8 * e)) & 1 for e in range(4)], axis=1)
            parts, shared = [], 0
            for t in range(steps):
                if t == 0 and coded:
                    parts += [pb[:, :d], bits]
                else:
                    parts += [pb[:, d * t:d * (t + 1)], np.broadcast_to(rb[t], (m, 4))]
                    shared = shared + rb[t] @ lane
            xs.append(np.concatenate(parts, axis=1))
            inits.append(np.broadcast_to(-(shared + bias), (m, dv)))
            rw.append(np.concatenate([h, r]))
            fm.append(nsw._pack(bits) if coded else np.zeros((m, 1), dtype=np.int64))
        x = np.stack(xs)
        y = np.broadcast_to(y1, (salts, *y1.shape))
        all_words.append(nsw._words(x, y, np.stack(inits)))
        raws.append(np.stack(rw))
        if coded:
            formed.append(np.stack(fm))
    return nsw._distinct(all_words + formed, raws)


def poly_exp2_coeffs(deg: int = 3) -> np.ndarray:
    """Least-squares fit of 2^f on [0, 1) at Chebyshev nodes, constant term pinned to 1."""
    f = (np.cos(np.pi * (np.arange(512) + 0.5) / 512) + 1) / 2
    a = np.stack([f ** e for e in range(1, deg + 1)], axis=1)
    c, *_ = np.linalg.lstsq(a / (2 ** f)[:, None], (2 ** f - 1) / 2 ** f, rcond=None)
    return np.concatenate([[1.0], c])


def poly_exp2(y: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """The served poly exp2 in float32: clamp, floor by the magic add, the fractional part, Horner, exponent insert."""
    y = np.maximum(y.astype(np.float32), np.float32(-126.0))
    j = np.floor(y).astype(np.float32)
    f = (y - j).astype(np.float32)
    p = np.float32(coeffs[-1])
    for cf in coeffs[-2::-1]:
        p = (p * f + np.float32(cf)).astype(np.float32)
    bits = p.view(np.int32) + (j.astype(np.int32) << 23)
    return bits.view(np.float32)


def sim_softmax(salts: int = 64, keys: int = 96, rows: int = 3, seed: int = 4, exp: str = "mufu") -> dict:
    """One int8 attention block's softmax chain on rows of noisy scores: the share of each class's outputs that vary
    across salts (targets must, roundings mostly don't) and whether the varying targets are pairwise distinct."""
    rng = np.random.default_rng(seed)
    q = rng.integers(-127, 128, (rows, 64))
    kk = rng.integers(-127, 128, (keys, 64))
    acc = q @ kk.T
    code = nsw.ssn_a_w0(keys, nsw.DATA_PER_STEP)[0]
    bias = np.rint(code.sum(axis=0) / 2)
    sk = rng.uniform(0.5, 1.5, keys).astype(np.float32)
    sv = rng.uniform(0.5, 1.5, keys).astype(np.float32)
    c = np.float32(1.0 / 8 / 1e4 * 1.4427)
    co = coeffs = poly_exp2_coeffs()
    outs = {"T": [], "R": [], "S": []}
    for _ in range(salts):
        a = rng.integers(0, 2, (rows, 4))
        noisy = acc + a @ code - bias
        magic = (noisy + 0x4B400000).astype(np.int64)
        xf = (magic.astype(np.int32).view(np.float32) - np.float32(12582912.0)).astype(np.float32)
        s = (xf * sk[None]).astype(np.float32)
        mx = s.max(axis=1, keepdims=True)
        y = ((s - mx) * c).astype(np.float32)
        p = np.exp2(y).astype(np.float32) if exp == "mufu" else poly_exp2(y, co)
        lsum = np.cumsum(p, axis=1, dtype=np.float32)
        pu8 = np.rint(p * 255 * sv[None] / sv.max())
        outs["T"].append(np.concatenate([magic.reshape(rows, -1), xf.view(np.int32), s.view(np.int32),
                                         y.view(np.int32), p.view(np.int32), lsum[:, 1:].view(np.int32)], axis=1))
        outs["R"].append(pu8)
        outs["S"].append(np.broadcast_to(mx, s.shape).view(np.int32))
    res = {}
    for cls, v in outs.items():
        arr = np.stack(v).reshape(salts, -1)
        varies = ~(arr == arr[:1]).all(axis=0)
        res[f"{cls}_vary_share"] = float(varies.mean())
        if cls == "T":
            cols = arr[:, varies].T
            res["T_distinct_share"] = len({col.tobytes() for col in cols}) / max(1, len(cols))
            res["T_const"] = int((~varies).sum())
    s_all = np.stack(outs["S"])[:, :, 0]
    t_s = np.stack([o[:, 2 * keys:3 * keys] for o in outs["T"]])
    res["S_new_share"] = float(np.mean([not any((s_all[:, i] == t_s[:, i, j]).all() for j in range(keys))
                                        for i in range(rows)]))
    err = np.abs(poly_exp2(np.linspace(-30, 0, 20001, dtype=np.float32), coeffs) /
                 np.exp2(np.linspace(-30, 0, 20001)) - 1).max()
    res["poly_exp2_max_rel_err"] = float(err)
    return res


def flip_shares(samples: int = 4096, seed: int = 5) -> dict:
    """Share of a noised linear's outputs whose rounding to bf16 (after dequant) or to int8 (per-token quant over 2,048)
    varies over the 16 step-1 bit patterns: the R outputs that are targets on the grid, uncredited in the headline."""
    rng = np.random.default_rng(seed)
    k = 2048
    sigma = 30 * 25 * math.sqrt(k)
    acc = np.rint(rng.normal(0, sigma, (2, samples)))
    code = nsw.ssn_a_w0(samples, nsw.DATA_PER_STEP)[0]
    bias = np.rint(code.sum(axis=0) / 2)
    pats = np.array(list(itertools.product((0, 1), repeat=4)))
    vals = acc[0][None] + pats @ code - bias
    scale = np.float32(0.0123)
    f = (vals.astype(np.float32) * scale).astype(np.float32)
    bf = (f.view(np.uint32) + 0x7FFF + ((f.view(np.uint32) >> 16) & 1)) >> 16
    bf16_vary = float((~(bf == bf[:1]).all(axis=0)).mean())
    amax = np.abs(acc[0][:k]).max()
    q = np.rint(vals[:, :k] * 127 / amax)
    int8_vary = float((~(q == q[:1]).all(axis=0)).mean())
    return {"bf16_vary_share": bf16_vary, "int8_vary_share": int8_vary, "acc_sigma": sigma,
            "noise_range_units": NOISE_RANGE}


def _xorshift32(x: np.ndarray) -> np.ndarray:
    x = x ^ ((x << np.uint32(13)) & np.uint32(0xFFFFFFFF))
    x = x ^ (x >> np.uint32(17))
    return x ^ ((x << np.uint32(5)) & np.uint32(0xFFFFFFFF))


def sim_dither(salts: int = 48, samples: int = 2048, seed: int = 6, dither: bool = True) -> dict:
    """A noised linear's outputs quantized per token to int8, each rounded as floor(x·s + u) with u's byte from a
    xorshift32 stream seeded by a raw coin word (dither) or as rint(x·s) (not): the share of int8 values and of their
    4-byte packs that vary across salts (noise bits and coin word), and whether the varying packs are pairwise
    distinct."""
    rng = np.random.default_rng(seed)
    sigma = 30 * 25 * math.sqrt(2048)
    acc = np.rint(rng.normal(0, sigma, samples))
    code = nsw.ssn_a_w0(samples, nsw.DATA_PER_STEP)[0]
    bias = np.rint(code.sum(axis=0) / 2)
    s = np.float32(127 / np.abs(acc).max())
    lane = np.arange(samples // 4, dtype=np.uint32)
    qs, packs = [], []
    for _ in range(salts):
        vals = (acc + rng.integers(0, 2, 4) @ code - bias).astype(np.float32)
        coin = np.uint32(rng.integers(0, 2**32))
        st = _xorshift32(_xorshift32(coin ^ (lane * np.uint32(0x9E3779B9))))
        u = ((st[:, None] >> (np.uint32(8) * np.arange(4, dtype=np.uint32))) & np.uint32(0xFF)).reshape(-1)
        u = (u.astype(np.float32) + np.float32(0.5)) / np.float32(256)
        q = np.floor(vals * s + u) if dither else np.rint(vals * s)
        q = np.clip(q, -127, 127).astype(np.int8)
        qs.append(q)
        packs.append(q.view(np.uint32))
    q, pk = np.stack(qs), np.stack(packs)
    vq, vp = ~(q == q[:1]).all(axis=0), ~(pk == pk[:1]).all(axis=0)
    cols = pk[:, vp].T
    return {"int8_vary_share": float(vq.mean()), "pack_vary_share": float(vp.mean()),
            "pack_distinct_share": len({c.tobytes() for c in cols}) / max(1, len(cols)),
            "mean_abs_err_vs_rint": float(np.abs(q.astype(np.float64) - np.rint(
                (acc[None] * s)).clip(-127, 127)).mean())}


def check() -> list[dict]:
    res = []

    def ok(name: str, cond: bool, detail) -> None:
        res.append({"check": name, "pass": bool(cond), "detail": detail})

    L, lin = LLAMA, nsw.LLAMA_MACS_TOKEN
    head = L["vocab"] * L["d"]
    ncp = {144: (78.1, 21.1, 0.8), 288: (77.6, 20.9, 1.5), 2048: (71.0, 19.2, 9.8), 8192: (54.9, 14.8, 30.3)}
    got = {}
    for keys, want in ncp.items():
        att = L["heads"] * keys * (L["dk"] + L["dv"]) * L["layers"]
        tot = lin + head + att
        got[keys] = tuple(round(100 * x / tot, 1) for x in (lin, head, att))
    ok("decode MAC shares (linears / head / attention) at 144, 288, 2K and 8K keys",
       all(abs(g - w) <= 0.1 for keys in ncp for g, w in zip(got[keys], ncp[keys])), got)
    pre = {}
    for ctx, want in ((2048, 93.5), (8192, 78.4)):
        att = L["heads"] * (ctx + 1) / 2 * (L["dk"] + L["dv"]) * L["layers"]
        pre[ctx] = round(100 * lin / (lin + att), 1)
    ok("prefill linear shares at 2K and 8K", pre == {2048: 93.5, 8192: 78.4}, pre)
    ok("Llama-3.2-1B linears are 973.1M MACs a token", round(lin / 1e6, 1) == 973.1, lin)
    v4a = sum(n * k for n, k in V4["dense"].values()) - sum(n * k for n, k in (V4["dense"]["sh_gate_up"],
                                                                               V4["dense"]["sh_down"]))
    v4f = (V4["top_k"] + 1) * sum(n * k for n, k in V4["expert"].values())
    ok("V4-like layer: ~187M attention-linear and ~396M FFN MACs a token", abs(v4a / 1e6 - 187) < 1 and
       abs(v4f / 1e6 - 396) < 2, (v4a, v4f))
    d = {x.name: x for x in nsw.designs()}["ssn_1l"]
    mine = {name: ssn1l_terms(8192, n, k) for name, (n, k) in nsw.LLAMA.items()}
    theirs = {name: nsw.gap_terms(d, 8192, n, k, "M0", False) for name, (n, k) in nsw.LLAMA.items()}
    ok("ssn1l_terms reproduces noise_sweep's ssn_1l (alpha, forming, coins) on every Llama linear",
       all(abs(mine[x]["alpha"] - theirs[x]["alpha"]) < 1e-12 and abs(mine[x]["forming"] - theirs[x]["gap"]) < 1e-12
           and abs(mine[x]["coins"] - theirs[x]["coins"]) < 1e-12 for x in mine),
       {x: (mine[x]["forming"], theirs[x]["gap"]) for x in mine})
    lb = evaluate("llama", "T8192_8x1k", PLAN["linears"], "floor")
    lin_gap = sum(r["honest"] - r["forced"] for r in lb["components"] if r["family"] == "linear")
    lin_macs = sum(r["count"] for r in lb["components"] if r["family"] == "linear")
    ok("the step's linear term (floor reading) equals noise_sweep's weighted ssn_1l gap at T 8,192",
       abs(lin_gap / lin_macs - nsw.weighted(d, nsw.LLAMA, 8192, "M0", False)["gap"]) < 1e-9, lin_gap / lin_macs)
    for name in ("ssn_sh",):
        bad = {c: sim_ssn_sh(c)["bad_share"] for c in nsw.CASES}
        ok(f"{name} words distinct and non-free on every data case", all(v == 0 for v in bad.values()), bad)
    bad64 = {c: sim_ssn_sh(c, k=64)["bad_share"] for c in nsw.CASES}
    ok("ssn_sh at QK's head dim 64 (S = 3, 8 data lanes in step 3) distinct and non-free on every case",
       all(v == 0 for v in bad64.values()) and sim_ssn_sh("random", k=64)["steps"] == 3, bad64)
    r = sim_ssn_sh("zero_blocks", mutate="no_shared")
    ok("ssn_sh without step 3's shared lane loses words on zero_blocks (mutation)", r["bad_share"] > 0, r["bad_share"])
    r = sim_ssn_sh("identical_cols", mutate="code_dup")
    ok("ssn_sh with a repeated column code loses words on identical_cols (mutation)", r["bad_share"] > 0,
       r["bad_share"])
    r = sim_ssn_sh("random", salts=400)
    ok("ssn_sh served error is zero-mean within 0.5 (std ~3.7 units)", abs(r["served_err_mean"]) < 0.5,
       (r["served_err_mean"], r["served_err_std"]))
    pv = {c: sim_pv_blocks(c)["bad_share"] for c in PV_CASES}
    ok("int8 PV blocks (ssn_sh, a code a block) distinct and non-free on every P/V case",
       all(v == 0 for v in pv.values()), pv)
    pv_m = {c: sim_pv_blocks(c, mutate="code_once")["bad_share"] for c in ("identical_rows", "uniform_p")}
    ok("int8 PV coded only in block 1 loses words on identical or uniform P (mutation)",
       all(v > 0 for v in pv_m.values()), pv_m)
    for e in ("mufu", "poly"):
        sm = sim_softmax(exp=e)
        ok(f"softmax chain ({e} exp2): T outputs vary across salts but the row max's own y = 0 and p = 1, and the "
           "varying ones are pairwise distinct (the row sum starts from p_0 by a move)",
           sm["T_const"] <= 2 * 3 and sm["T_distinct_share"] == 1.0, sm)
    sm = sim_softmax()
    ok("softmax chain: the u8 P rounding (R) is salt-independent on most elements", sm["R_vary_share"] < 0.5, sm)
    dz, dr = sim_dither(), sim_dither(dither=False)
    ok("dithered int8 rounding: >= 90% of values and every 4-byte pack vary across salts, the packs pairwise "
       "distinct, error ~0.25 a step; rint varies on < 5%", dz["int8_vary_share"] >= 0.9 and dz["pack_vary_share"] == 1
       and dz["pack_distinct_share"] == 1 and dz["mean_abs_err_vs_rint"] < 0.3 and dr["int8_vary_share"] < 0.05,
       (dz, dr))
    ok("degree-3 exp2 within 2e-4 relative on [-30, 0] (u8 P's step is 3.9e-3)",
       sm["poly_exp2_max_rel_err"] < 2e-4, sm["poly_exp2_max_rel_err"])
    xs = np.array([-(2 ** 22), -1, 0, 1, 12345, 2 ** 22 - 1], dtype=np.int64)
    conv = ((xs + 0x4B400000).astype(np.int32).view(np.float32) - np.float32(12582912.0)).astype(np.float64)
    ok("magic-number conversion exact for |acc| < 2^22, which holds QK (127^2·64) and PV (255·127·112) blocks",
       bool((conv == xs).all()) and 127 ** 2 * 64 < 2 ** 22 and 255 * 127 * I8_BLOCK < 2 ** 22, conv.tolist())
    tc = {m: tc_best(m, "s8", "paired") for m in (64, 8, 4, 2)}
    tb = {m: tc_best(m, "bf16", "paired") for m in (64, 8, 4)}
    ok("small-m pricing: s8 1.0 at 64 rows, 1.502 / 3.0 / 6.0 a real MAC at 8 / 4 / 2; bf16 2.0 / 3.0 / 6.0",
       abs(tc[64][0] * tc[64][1] - 1.0) < 1e-3 and abs(tc[8][0] * tc[8][1] - 1.502) < 1e-3
       and abs(tc[4][0] * tc[4][1] - 3.004) < 2e-3 and abs(tc[2][0] * tc[2][1] - 6.0) < 1e-2
       and abs(tb[64][0] * tb[64][1] - 2.0) < 1e-3 and abs(tb[8][0] * tb[8][1] - 3.004) < 2e-3
       and abs(tb[4][0] * tb[4][1] - 6.009) < 2e-2, {**{f"s8_{m}": v for m, v in tc.items()},
                                                     **{f"bf16_{m}": v for m, v in tb.items()}})
    for plan in PLANS:
        for model in ("llama", "v4"):
            for point in POINTS:
                e = evaluate(model, point, plan)
                if not all(r["forced"] <= r["honest"] + 1e-6 for r in e["components"]):
                    ok(f"forced <= honest in every component ({plan.name}, {model}, {point})", False, point)
    ok("forced <= honest in every component, plan, model and point", True, "")
    return res


# ---------------------------------------------------------------- report

def report(lookup: bool = False) -> dict:
    out = {"points": {k: {**v, **point_shape(v)} for k, v in POINTS.items()}, "plans": [p.__dict__ | {
        "premises": sorted(p.premises)} for p in PLANS], "premises": PREMISES, "evals": []}
    for model in ("llama", "v4"):
        for point in POINTS:
            for plan in PLANS:
                for reading in READINGS:
                    e = evaluate(model, point, plan, reading, lookup)
                    if reading != "paired":
                        e.pop("components")
                    out["evals"].append(e)
    idioms = {}
    for kind in ("epi", "norm", "norm_nores", "rope", "kvq", "silu", "quant", "attn_out", "combine", "router",
                 "sampler"):
        idioms[kind] = {}
        for v in ("stock", "lean", "fused"):
            ops = lane_ops(kind, v)
            idioms[kind][v] = {"ops": ops, **{r: lane_cost(ops, r, frozenset(), v == "fused") for r in READINGS}}
    for v, e in (("stock", "mufu"), ("i8", "mufu"), ("i8", "poly")):
        ops = lane_ops("softmax", v, exp=e)
        idioms["softmax"] = idioms.get("softmax", {})
        idioms["softmax"][f"{v}_{e}"] = {"ops": ops, **{r: lane_cost(ops, r, frozenset(), v == "i8")
                                                        for r in READINGS}}
    out["idioms"] = idioms
    out["flip_shares"] = flip_shares()
    out["softmax_sim"] = {e: sim_softmax(exp=e) for e in ("mufu", "poly")}
    out["dither_sim"] = {"dither": sim_dither(), "rint": sim_dither(dither=False)}
    out["checks"] = check()
    best = {(e["model"], e["point"], e["plan"]): e for e in out["evals"] if e["reading"] == "paired"}
    out["periods"] = {f"{m}_{pt}_{pl}": {"t1": nsw.periods(best[(m, pt, pl)]["t1"], best[(m, pt, pl)]["alpha_w1"]),
                                         "uncovered": nsw.periods(best[(m, pt, pl)]["uncovered_share"],
                                                                  best[(m, pt, pl)]["alpha_w1"])}
                      for m in ("llama", "v4") for pt in ("T8192_8x1k", "dec64_2k")
                      for pl in ("linears", "best", "best_p4", "best_dither", "best_p4_dither")}
    out["params"] = {"price": PRICE, "price_estimated": PRICE_ESTIMATED, "refined_floor": REFINED_FLOOR,
                     "fa3_block": FA3_BLOCK, "i8_block": I8_BLOCK, "ept": [EPT_STOCK, EPT_LEAN],
                     "stock_t8192_ms": STOCK_T8192_MS, "gemm_t8192_ms": GEMM_T8192_MS, "att_t8192_ms": ATT_T8192_MS,
                     "host_t8192_ms": HOST_T8192_MS, "host_fit_ms": HOST_FIT_MS, "hbm_b_s": HBM_B_S,
                     "lookup_priced": lookup, "t3": T3}
    return out


def table(rep: dict) -> str:
    lines = []
    hdr = ("model", "point", "plan", "reading", "lin share", "T1 matmul", "coverage", "T1", "T1 +coins",
           "uncovered", "slack", "alpha W1", "T2 ser/ovl", "hash GB", "stock ms")
    lines.append(" | ".join(hdr))
    for e in rep["evals"]:
        t2 = e["t2"]
        lines.append(" | ".join([e["model"], e["point"], e["plan"], e["reading"],
                                 f"{100 * e['linears_stock_share']:.1f}%", f"{100 * e['t1_matmul']:.3f}%",
                                 f"{100 * e['coverage']:.2f}%", f"{100 * e['t1']:.2f}%",
                                 f"{100 * e['t1_coins_uncredited']:.2f}%", f"{100 * e['uncovered_share']:.2f}%",
                                 f"{100 * e['slack_share']:.2f}%", f"{e['alpha_w1']:.3f}",
                                 f"{t2['t2_serial']:.1f}x/{t2['t2_overlapped']:.1f}x", f"{t2['hash_gb']:.1f}",
                                 f"{t2['stock_ms']:.2f}"]))
    lines.append("")
    lines.append("groups (paired): model | point | plan | group | stock share | gap share | uncovered | designs")
    for e in rep["evals"]:
        if e["reading"] != "paired" or e["plan"] not in ("linears", "best", "best_p4", "best_p4_dither"):
            continue
        for g, v in sorted(e["groups"].items(), key=lambda kv: -kv[1]["stock_share"]):
            lines.append(" | ".join([e["model"], e["point"], e["plan"], g, f"{100 * v['stock_share']:.2f}%",
                                     f"{100 * v['gap_share']:.2f}%", f"{100 * v['uncovered_share']:.2f}%",
                                     ",".join(v["designs"])]))
    lines.append("")
    lines.append(f"flip shares: {rep['flip_shares']}")
    lines.append(f"softmax sim: {rep['softmax_sim']}")
    lines.append(f"dither sim: {rep['dither_sim']}")
    for c in rep["checks"]:
        lines.append(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--table", action="store_true", help="print the text table")
    ap.add_argument("--data", help="noise_data.py's JSON, passed to noise_sweep (its duplicate shares)")
    ap.add_argument("--lookup", action="store_true", help="price input-addressed moves (embedding, KV write) as L2 lookups")
    args = ap.parse_args(argv)
    if args.data:
        nsw.load_data(args.data)
    rep = report(args.lookup)
    text = json.dumps(rep, indent=1, default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else float(o))
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
