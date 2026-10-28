"""DeepSeek-V4-Pro's MoE routing and activation on a GPU (vLLM d9105ea80): the Triton router `dsv4_topk` (sqrtsoftplus scores, the
f32 `e_score_correction_bias`, top-6 of 384, renormalised, x2.5), the hash router of layers 0-2 (`ops.topk_hash_softplus_sqrt` over
`tid2eid`, the CUDA kernel `dsv4HashTopkSoftplusSqrt`) and the clamped SwiGLU (`_C.silu_and_mul_with_clamp`, limit 10), against
`registry.dsv4_moe`'s `MoeRouterSqrtSoftplus_v1{E=384,SCALE=2.5}`, `MoeRouterHash_v1{E=384,SCALE=2.5}` and `SiluMulClamp_v1{I,LIMIT=10.0}`.

One GPU process writes `<out>/dsv4_moe_capture.json`, the property record `dsv4_moe_capture` (`properties.record`), and `<out>/rows.npz`:
every captured row's inputs and both launches' outputs.  `ok` = every check below.  The device gate runs first and nothing else runs
when it fails (exit 3):

  gate     `fa2_target_capture_gpu.gate`: driver, compute capability and SM count.
  binary   the SASS and PTX archs vLLM's `_moe_C` and `_C` carry (informational).
  mufu     `fa2_target_capture_gpu.mufu`: MUFU ex2 / rcp are core's tables on all 2^32 inputs; and through Triton inline asm on
           this GPU: `lg2.approx.ftz.f32` and `lg2.approx.f32` on every 1.m of [1, 2) are `MufuLg2Mantissa_v1` (the RTX PRO 6000's
           measured words, `LG2_MEASURED`), lg2(2) = 1 and a NaN 0x7FFFFFFF; `sqrt.approx.ftz.f32` on every word of [1, 4) is
           core's sqrt table and on a sweep `mufu_sqrt_words`; `ex2.approx.f32` (not .ftz) on every word of [-150, -125] and a
           sweep is `Ex2Approx_v1`'s lowering; `div.full.f32` of 2.5 by the router's divisors is `MC.div_full`.
  launch   the kernel each path ran (torch.profiler): `_dsv4_topk_kernel`, `dsv4HashTopkSoftplusSqrt<int, int>`, and
           `act_and_mul_kernel<bf16, ..., silu_kernel, ..., act_first, use_vec, HAS_CLAMP>` with use_vec at d = 384 (TP8's
           3072 / 8, 16-byte vectors on sm_90) and the scalar fallback at d = 9.
  rows     `router_rows`: groups of 64 rows sharing one bias (the kernel reads one bias per launch), drawn to reach the router's
           edges: ties in the current and in the score, a bias that decides, score + bias == +0, the x > 20 select and its
           boundary, the ex2 halving below -126, subnormal, huge and non-finite logits and biases, fewer than six currents above
           -inf and none.  `hash_rows`: a token's six table ids in [0, E) (the checkpoint's values; repeats too) and logits whose
           six read words reach the weight's edges: the halving, y = 2 and y = 1, a zero sum, an infinite weight, NaN.
           `clamp_rows`: gate / up pairs at and around +-10, NaN, +-inf, +-0, subnormals, random words, and every gate word
           against 16 ups and every up word against 16 gates.  Each path runs twice; every output against
           `kernels.dsv4_moe_twin`'s `router_twin`, `hash_twin` and `clamp_twin` (the Programs restated on numpy arrays), and a
           sample against the Programs themselves.

`--check DIR` evaluates the Programs on every row of a capture's `rows.npz` (no GPU; about 0.2 core-seconds a router row, 0.03 a hash
or clamp row) and writes `<DIR>/program_check.json`.  `--verify RECORD` re-derives a record's digest and verdict.

Not modelled, so not captured: image tokens (`bias_vl` with `image_sentinel_lo`), padding rows (`is_padding`), and a table id
outside [0, E), which reads the next row's logits on the GPU.

It lives under tests/ because it seeds random generators and sets VLLM_BATCH_INVARIANT for its own process (lints P3, P7).

    cd integrations/vllm; PY=/workspace/jobs/venv312/bin/python      # PYTHONPATH as for the other captures
    $PY tests/properties/dsv4_moe_capture_gpu.py --out DIR --want-cc 9.0 --want-sms 132     # H200 (sm_90)
    $PY tests/properties/dsv4_moe_capture_gpu.py --check DIR --workers 4
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from verity.ml.circuits.ops import mufu as MU
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics.dsv4_moe_twin import clamp_twin, hash_twin, lg2_words, router_twin, score_words
from verity.ml.definitions.registry import dsv4_moe as D
from verity.ml.definitions.registry import mla_check as MC

SCHEMA = "verity-vllm/dsv4-moe-capture/v1"
RECORD = "dsv4_moe_capture.json"
ROWS_FILE = "rows.npz"
CHECK_FILE = "program_check.json"
E, TOPK, SCALE, LIMIT = 384, D.TOPK, 2.5, 10.0
GROUP = 64                    # router rows per launch, which share one bias
WIDTHS = (384, 9)             # the clamp's d: 3072 / TP8 takes the 16-byte vector path on sm_90, an odd d the scalar fallback
PATHS = ("router", "hash", *(f"clamp{d}" for d in WIDTHS))
KERNEL_NAMES = {"router": ("_dsv4_topk_kernel",), "hash": ("dsv4HashTopkSoftplusSqrt",),
                "clamp384": ("act_and_mul_kernel", "silu_kernel"), "clamp9": ("act_and_mul_kernel", "silu_kernel")}
#: act_and_mul_kernel's <act_first, use_vec, HAS_CLAMP, use_256b>, demangled and mangled
KERNEL_FLAGS = {"clamp384": ("true, true, true, false>", "Lb1ELb1ELb1ELb0E"), "clamp9": ("true, false, true, false>", "Lb1ELb0ELb1ELb0E")}

U32 = np.uint32
ONE, TWO, SIGN, PINF, NEG_INF, QNAN = 0x3F800000, 0x40000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FFFFFFF
NONFINITE = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF, PINF, NEG_INF], U32)
EXTREMES = np.array([0, SIGN, 1, SIGN | 1, 0x007FFFFF, 0x00800000, SIGN | 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF], U32)
HALVING = 0x42AEAC50                 # 87.336544: |x| log2(e) = 126, where ex2.approx halves its operand below
Y_TWO, Y_ONE = 0x33000000, 0x41851592  # |x| near 2^-25, where 1 + 2^(-|x| log2 e) rounds to 2; 16.6355, where it rounds to 1
BIAS_KINDS = ("zero", "small", "decisive", "ties", "cancel", "huge", "nonfinite", "mostly-ninf", "bits")
ROW_KINDS = ("gauss", "near", "ties", "cancel", "twenty", "low", "extremes", "nonfinite", "bits")
HASH_KINDS = ("gauss", "edges", "large", "zero-sum", "inf", "nan", "repeat", "bits")
CLAMP_KINDS = ("gauss", "limit", "specials", "silu", "bits", "sweep")
LIMIT_BF16 = D.limit_bits(LIMIT) >> 16


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _w(v) -> np.ndarray:
    return np.asarray(v, np.float32).view(U32)


def _ulps(w: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(w, np.int64) + k, 0, 0xFFFFFFFF).astype(U32)


# ----------------------------------------------------------------------------------------------------------------------- rows
def _bias(kind: str, rng: np.random.Generator, k: int) -> tuple[np.ndarray, np.ndarray | None]:
    """The k-th group of its kind's bias words, and for `cancel` the logits whose score each expert's bias cancels exactly (score +
    bias == +0)."""
    c = None
    if kind == "zero":
        b = np.zeros(E, U32)
    elif kind == "small":
        b = _w(rng.standard_normal(E) * 10.0 ** rng.uniform(-4, 0))
    elif kind == "decisive":
        b = _w(rng.standard_normal(E) * 2.0)
    elif kind == "ties":
        b = _w(rng.choice([0.0, 0.25, -0.5], E))
    elif kind == "cancel":
        c = _w(rng.standard_normal(E) * 3.0)
        b = score_words(c) ^ U32(SIGN)
    elif kind == "huge":
        b = _w(rng.standard_normal(E) * 1e-2)
        b[rng.choice(E, 16, replace=False)] = _w(rng.choice([1e30, -1e30, 3.4028235e38, -3.4028235e38, 1e7, -1e7, 16777216.0], 16))
    elif kind == "nonfinite":
        b = _w(rng.standard_normal(E) * 0.1)
        i = rng.choice(E, int(rng.integers(1, 41)), replace=False)
        b[i] = rng.choice(NONFINITE, len(i))
    elif kind == "mostly-ninf":
        b = np.full(E, NEG_INF, U32)
        i = rng.choice(E, k % 10, replace=False)
        b[i] = _w(rng.standard_normal(len(i)))
    else:
        b = rng.integers(0, 1 << 32, E, dtype=np.uint64).astype(U32)
    return b.astype(U32), c


def _logits(kind: str, rng: np.random.Generator, c: np.ndarray | None, n: int = E) -> np.ndarray:
    if kind == "gauss":
        return _w(rng.standard_normal(n) * rng.choice([0.5, 2.0, 4.0, 8.0]))
    if kind == "near":                                             # within a few ulps of one value: near-equal and equal scores
        v = [rng.standard_normal() * 3.0, rng.uniform(20.0, 40.0), -rng.uniform(80.0, 100.0), rng.standard_normal() * 1e-3]
        return _ulps(np.full(n, _w(v[int(rng.integers(4))])), rng.integers(-6, 7, n))
    if kind == "ties":
        return rng.choice(_w(rng.standard_normal(int(rng.integers(1, 6))) * 3.0), n)
    if kind == "cancel":            # the group's cancelled logits, some moved by an ulp or two
        x = (_w(rng.standard_normal(n) * 3.0) if c is None else c[:n]).copy()
        i = rng.choice(n, int(rng.integers(0, n // 2)), replace=False)
        x[i] = _ulps(x[i], rng.choice([-2, -1, 1, 2], len(i)))
        return x
    if kind == "twenty":                                           # the select x > 20 and its boundary, then large logits
        x = _ulps(np.full(n, _w(20.0)), rng.integers(-8, 9, n))
        i = rng.choice(n, n // 4, replace=False)
        x[i] = _w(rng.choice([1e3, 1e7, 1e30, 3.4028235e38, 87.0, 88.8], len(i)))
        return x
    if kind == "low":                                              # x log2 e around -126 and below: the halving, a subnormal or zero ex2
        x = _ulps(np.full(n, HALVING | SIGN), rng.integers(-40000, 40000, n))
        i = rng.choice(n, n // 3, replace=False)
        x[i] = _w(-rng.uniform(100.0, 200.0, len(i)))
        return x
    if kind == "extremes":
        more = _w(rng.choice([1e4, -1e4, 1e-30, -1e-30, 88.0, -88.0, 104.0, -104.0, 18.0, -18.0, 1e38, -1e38], n))
        return np.where(rng.integers(0, 2, n) == 0, rng.choice(EXTREMES, n), more).astype(U32)
    if kind == "nonfinite":                                        # 0, 1, 5, 6, 7 or more finite experts
        x = _w(rng.standard_normal(n) * 4.0)
        keep = int(rng.choice([0, 1, 5, 6, 7, int(rng.integers(8, n))]))
        i = rng.choice(n, n - keep, replace=False)
        x[i] = rng.choice(NONFINITE, len(i))
        return x
    return rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)


def router_rows(seed: int = 2026, n: int = 8192) -> dict[str, np.ndarray]:
    """`ceil(n / 64)` groups of 64 rows: group g's bias is `BIAS_KINDS[g % 9]`, its row r `ROW_KINDS[r % 9]`.  Returns `x` (rows, E),
    `b` (groups, E) and the kinds' indices."""
    rng = np.random.default_rng(seed)
    xs, bs, bk, rk = [], [], [], []
    for g in range(-(-n // GROUP)):
        b, c = _bias(BIAS_KINDS[g % len(BIAS_KINDS)], rng, g // len(BIAS_KINDS))
        bs.append(b)
        bk.append(g % len(BIAS_KINDS))
        for r in range(GROUP):
            xs.append(_logits(ROW_KINDS[r % len(ROW_KINDS)], rng, c))
            rk.append(r % len(ROW_KINDS))
    return {"x": np.stack(xs).astype(U32), "b": np.stack(bs).astype(U32), "bias_kind": np.array(bk, np.int8),
            "row_kind": np.array(rk, np.int8)}


def row_bias(R: dict[str, np.ndarray]) -> np.ndarray:
    """Each router row's bias (rows, E)."""
    return R["b"][np.arange(len(R["x"])) // GROUP]


def _read(kind: str, rng: np.random.Generator) -> np.ndarray:
    """The six logits a hash row's table ids read."""
    if kind == "gauss":
        return _w(rng.standard_normal(TOPK) * rng.choice([1.0, 4.0, 16.0]))
    if kind == "edges":
        edges = np.array([HALVING, Y_TWO, Y_ONE, 0x34000000, 0x3F800000], U32)
        x = _ulps(rng.choice(edges, TOPK), rng.integers(-64, 65, TOPK)) | (rng.integers(0, 2, TOPK).astype(U32) << 31)
        return np.where(rng.integers(0, 4, TOPK) == 0, rng.choice(EXTREMES, TOPK), x).astype(U32)
    if kind == "large":
        return _w(rng.choice([1e3, 1e10, 1e30, 3.4028235e38, -1e3, -1e30, 50.0, -50.0, 104.0, -104.0], TOPK))
    if kind == "zero-sum":                                         # every weight +0: the sum is 0 and the divisor 1
        return _w(-rng.uniform(104.0, 1e6, TOPK))
    if kind == "inf":                                              # one +inf: an infinite sum, its slot NaN and the others +0
        x = _w(rng.standard_normal(TOPK) * 4.0)
        x[rng.integers(TOPK)] = PINF
        x[rng.integers(TOPK)] = rng.choice([PINF, NEG_INF, x[0]])
        return x
    if kind == "nan":
        x = _w(rng.standard_normal(TOPK) * 4.0)
        i = rng.choice(TOPK, int(rng.integers(1, TOPK + 1)), replace=False)
        x[i] = rng.choice(NONFINITE[:4], len(i))
        return x
    if kind == "repeat":
        return _w(rng.standard_normal(TOPK) * 4.0)
    return rng.integers(0, 1 << 32, TOPK, dtype=np.uint64).astype(U32)


def hash_rows(seed: int = 2027, n: int = 8192, e: int = E) -> dict[str, np.ndarray]:
    """n token rows of e logits, row r of kind `HASH_KINDS[r % 8]`: `tokens` (n,) a permutation of [0, n), `tid2eid` (n, 6) the table
    (six distinct ids in [0, e) a row, or for `repeat` copies of fewer), and the six words the row's ids read set by its kind (the
    other logits normal draws)."""
    rng = np.random.default_rng(seed)
    tokens = rng.permutation(n).astype(np.int32)
    table = np.zeros((n, TOPK), np.int32)
    x = _w(rng.standard_normal((n, e)) * 3.0)
    kinds = np.arange(n) % len(HASH_KINDS)
    for r in range(n):
        kind = HASH_KINDS[kinds[r]]
        ids = rng.choice(e, TOPK, replace=False)
        if kind == "repeat":
            ids = rng.choice(ids[:int(rng.integers(1, TOPK))], TOPK)
        table[tokens[r]] = ids
        x[r, ids] = _read(kind, rng)
    return {"x": x.astype(U32), "tokens": tokens, "tid2eid": table, "kind": kinds.astype(np.int8)}


def hash_ids(Hr: dict[str, np.ndarray]) -> np.ndarray:
    """Each hash row's six ids (rows, 6): its token's table row."""
    return Hr["tid2eid"][Hr["tokens"]]


def _bf16(v) -> np.ndarray:
    return DR.f2fp_bf16(_w(v)).astype(np.uint16)


def _halves(kind: str, rng: np.random.Generator, n: int) -> np.ndarray:
    if kind == "gauss":
        return _bf16(rng.standard_normal(n) * rng.choice([1.0, 4.0, 16.0]))
    if kind == "limit":                                            # +-10 and a few bf16 ulps either side
        return (rng.choice([LIMIT_BF16, LIMIT_BF16 | 0x8000], n) + rng.integers(-3, 4, n)).astype(np.uint16)
    if kind == "specials":
        sp = np.array([0x7FC0, 0xFFC0, 0x7F81, 0xFFFF, 0x7F80, 0xFF80, 0, 0x8000, 1, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F,
                       LIMIT_BF16, LIMIT_BF16 | 0x8000], np.uint16)
        return np.where(rng.integers(0, 2, n) == 0, rng.choice(sp, n), _bf16(rng.standard_normal(n) * 8.0)).astype(np.uint16)
    if kind == "silu":                                             # expf's overflow and underflow, tiny gates
        return _bf16(rng.choice([-1.0, 1.0], n) * rng.choice([88.0, 89.0, 100.0, 1e4, 1e-20, 1e-38, 0.5], n))
    return rng.integers(0, 1 << 16, n).astype(np.uint16)


def clamp_sweep(d: int) -> np.ndarray:
    """Every gate word against 16 up words and every up word against 16 gate words, as rows (k, 2d) (the last padded with +0)."""
    probe = np.array([LIMIT_BF16, LIMIT_BF16 | 0x8000, LIMIT_BF16 + 1, LIMIT_BF16 - 1, 0x3F80, 0xBF80, 0, 0x8000, 0x7FC0, 0x7F80,
                      0xFF80, 0x0001, 0x4000, 0xC120, 0x3E00, 0xC2B0], np.uint16)
    every = np.arange(1 << 16, dtype=np.uint16)
    g = np.concatenate([np.tile(every, len(probe)), np.repeat(probe, 1 << 16)])
    u = np.concatenate([np.repeat(probe, 1 << 16), np.tile(every, len(probe))])
    k = -(-len(g) // d)
    pad = k * d - len(g)
    g, u = np.concatenate([g, np.zeros(pad, np.uint16)]), np.concatenate([u, np.zeros(pad, np.uint16)])
    return np.concatenate([g.reshape(k, d), u.reshape(k, d)], axis=1)


def clamp_rows(seed: int, n: int, d: int, sweep: bool) -> dict[str, np.ndarray]:
    """n rows of 2d bf16 words, row r's gate and up halves of kind `CLAMP_KINDS[r % 5]` each (the two drawn independently), then
    `clamp_sweep(d)` when asked."""
    rng = np.random.default_rng(seed)
    kinds = np.arange(n) % (len(CLAMP_KINDS) - 1)
    gu = np.stack([np.concatenate([_halves(CLAMP_KINDS[k], rng, d), _halves(CLAMP_KINDS[int(rng.integers(len(CLAMP_KINDS) - 1))], rng, d)])
                   for k in kinds]) if n else np.zeros((0, 2 * d), np.uint16)
    if sweep:
        s = clamp_sweep(d)
        gu, kinds = np.concatenate([gu, s]), np.concatenate([kinds, np.full(len(s), CLAMP_KINDS.index("sweep"))])
    return {"gu": gu.astype(np.uint16), "kind": kinds.astype(np.int8)}


def all_rows(seed: int = 2026, n: int = 8192, clamp_n: int = 2048) -> dict[str, dict[str, np.ndarray]]:
    """Every path's rows, keyed by path."""
    return {"router": router_rows(seed, n), "hash": hash_rows(seed + 1, n),
            "clamp384": clamp_rows(seed + 2, clamp_n, 384, True), "clamp9": clamp_rows(seed + 3, clamp_n // 2, 9, False)}


def expect(path: str, P: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The twin's outputs for one path's rows."""
    if path == "router":
        w, ids = router_twin(P["x"], row_bias(P), SCALE)
        return {"w": w, "ids": ids}
    if path == "hash":
        ids = hash_ids(P)
        return {"w": hash_twin(P["x"], ids, SCALE), "ids": ids}
    return {"out": clamp_twin(P["gu"], LIMIT)}


def digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _flat(rows: dict[str, dict[str, np.ndarray]], got: dict[str, dict[str, np.ndarray]] | None = None) -> dict[str, np.ndarray]:
    out = {f"{p}__{k}": v for p, P in rows.items() for k, v in P.items()}
    if got:
        out.update({f"{p}__out_{k}": v for p, G in got.items() for k, v in G.items()})
    return out


def _unflat(z) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    rows: dict[str, dict[str, np.ndarray]] = {}
    got: dict[str, dict[str, np.ndarray]] = {}
    for key in z.files:
        p, k = key.split("__", 1)
        (got.setdefault(p, {}).__setitem__(k[4:], z[key]) if k.startswith("out_") else rows.setdefault(p, {}).__setitem__(k, z[key]))
    return rows, got


# ----------------------------------------------------------------------------------------------------------------------- mufu
SQRT_IN = U32(ONE) + np.arange(1 << 24, dtype=U32)                     # every word of [1, 4): core's sqrt table's inputs
LG2_SPECIALS = np.array([TWO, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFFFFFFF, QNAN], U32)
EX2_LO, EX2_HI = 0xC2FA0000, 0xC3160000                               # -125 and -150


def _splitmix(n: int, salt: int) -> np.ndarray:
    z = np.arange(n, dtype=np.uint64) + (np.uint64(salt) << np.uint64(40))
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return (z ^ (z >> np.uint64(31))).astype(U32)


def lg2_inputs() -> np.ndarray:
    """Every 1.m of [1, 2)."""
    return U32(ONE) | np.arange(1 << 23, dtype=U32)


def ex2_inputs() -> np.ndarray:
    """Every word of [-150, -125] (the halving's boundary and its range) and a sweep of 2^20 words."""
    return np.concatenate([np.arange(EX2_LO, EX2_HI + 1, dtype=U32), _splitmix(1 << 20, 0xE2)])


def div_divisors() -> np.ndarray:
    """The router's divisors and more: 1, +inf, 2^-12, the scaling boundaries 2^-126 and 2^126 within 8 ulps, FLT_MAX, and 2^20
    positive normal words."""
    edges = np.concatenate([np.array([ONE, PINF, 0x39800000, 0x7F7FFFFF], U32),
                            _ulps(np.repeat(np.array([0x00800000, 0x7E800000], U32), 17), np.tile(np.arange(-8, 9), 2))])
    return np.concatenate([edges, U32(0x00800000) + _splitmix(1 << 20, 0xD1) % U32(0x7F000000)])


def mufu_expect() -> dict[str, np.ndarray]:
    """What the probe compares each instruction with."""
    sweep = _splitmix(1 << 20, 0x5C)
    return {"lg2": lg2_words(lg2_inputs()), "lg2_specials": np.array([ONE] + [QNAN] * 5, U32), "sqrt": MU.table("sqrt"),
            "sqrt_sweep_in": sweep, "sqrt_sweep": MU.mufu_sqrt_words(sweep, MU.table("sqrt")), "ex2": MC.ex2_approx(ex2_inputs()),
            "div": MC.div_full(np.full(len(div_divisors()), _w(SCALE), U32), div_divisors())}


_ASM = {"lg2_ftz": "lg2.approx.ftz.f32 $0, $1;", "lg2": "lg2.approx.f32 $0, $1;", "sqrt_ftz": "sqrt.approx.ftz.f32 $0, $1;",
        "ex2": "ex2.approx.f32 $0, $1;", "div_full": "div.full.f32 $0, $1, $2;"}


def _asm(op: str, *xs: np.ndarray) -> np.ndarray:
    """One PTX instruction element-wise over f32 words on the GPU (Triton inline asm)."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def one(X, Y, n, BLOCK: tl.constexpr, ASM: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        y = tl.inline_asm_elementwise(ASM, "=r,r", [tl.load(X + i, mask=i < n)], dtype=tl.float32, is_pure=True, pack=1)
        tl.store(Y + i, y, mask=i < n)

    @triton.jit
    def two(A, B, Y, n, BLOCK: tl.constexpr, ASM: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        y = tl.inline_asm_elementwise(ASM, "=r,r,r", [tl.load(A + i, mask=i < n), tl.load(B + i, mask=i < n)], dtype=tl.float32,
                                      is_pure=True, pack=1)
        tl.store(Y + i, y, mask=i < n)

    ts = [_device(x) for x in xs]
    y = torch.empty_like(ts[0])
    grid = (triton.cdiv(ts[0].numel(), 1024),)
    (one if len(ts) == 1 else two)[grid](*ts, y, ts[0].numel(), BLOCK=1024, ASM=_ASM[op])
    torch.cuda.synchronize()
    return _u32(y)


def _mismatch(got: np.ndarray, want: np.ndarray, x: np.ndarray | None = None) -> dict[str, Any]:
    bad = np.flatnonzero(got != want)
    return {"words": int(got.size), "mismatch": int(bad.size),
            "first": [[int(i), int(x[i]) if x is not None else None, int(got[i]), int(want[i])] for i in bad[:8]]}


def mufu_probe() -> dict[str, Any]:
    """MUFU.LG2, MUFU.SQRT, `ex2.approx.f32`'s lowering and `div.full.f32` on this GPU against the twin's models."""
    want = mufu_expect()
    rec: dict[str, Any] = {}
    for op in ("lg2_ftz", "lg2"):
        got = _asm(op, lg2_inputs())
        rec[op] = {**_mismatch(got, want["lg2"], lg2_inputs()), "sha256": hashlib.sha256(got.astype("<u4").tobytes()).hexdigest(),
                   "specials": _mismatch(_asm(op, LG2_SPECIALS), want["lg2_specials"], LG2_SPECIALS)}
        rec[op]["ok"] = rec[op]["mismatch"] == 0 and rec[op]["specials"]["mismatch"] == 0
    rec["lg2_ftz"]["measured_sha256"] = D.LG2_MEASURED["sha256"]
    rec["lg2_ftz"]["ok"] &= rec["lg2_ftz"]["sha256"] == D.LG2_MEASURED["sha256"]
    rec["sqrt_ftz"] = {**_mismatch(_asm("sqrt_ftz", SQRT_IN), want["sqrt"], SQRT_IN),
                       "sweep": _mismatch(_asm("sqrt_ftz", want["sqrt_sweep_in"]), want["sqrt_sweep"], want["sqrt_sweep_in"])}
    rec["sqrt_ftz"]["ok"] = rec["sqrt_ftz"]["mismatch"] == 0 and rec["sqrt_ftz"]["sweep"]["mismatch"] == 0
    rec["ex2"] = _mismatch(_asm("ex2", ex2_inputs()), want["ex2"], ex2_inputs())
    rec["ex2"]["ok"] = rec["ex2"]["mismatch"] == 0
    d = div_divisors()
    rec["div_full"] = _mismatch(_asm("div_full", np.full(len(d), _w(SCALE), U32), d), want["div"], d)
    rec["div_full"]["ok"] = rec["div_full"]["mismatch"] == 0
    rec["ok"] = all(rec[k]["ok"] for k in ("lg2_ftz", "lg2", "sqrt_ftz", "ex2", "div_full"))
    return rec


# ------------------------------------------------------------------------------------------------------------------------ gpu
def _u32(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int32).cpu().numpy().view(U32)


def _equal_f32_words(a, b) -> bool:
    """Whether two f32 tensors have the same words, including NaN payloads."""
    import torch

    return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))


def _device(a: np.ndarray, dtype=None):
    import torch

    t = torch.from_numpy(np.ascontiguousarray(a, U32).view(np.int32)).cuda()
    return t.view(torch.float32) if dtype is None else t


def _bf16_device(a: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(a, np.uint16).view(np.int16)).cuda().view(torch.bfloat16)


def launch_router(xd, bd) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

    w, ids = dsv4_topk(xd, bd, torch.int32, SCALE)
    torch.cuda.synchronize()
    return _u32(w), ids.cpu().numpy().astype(np.int32)


def launch_hash(xd, tokd, tabd) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from vllm import _custom_ops as ops

    n = xd.shape[0]
    w = torch.empty(n, TOPK, dtype=torch.float32, device="cuda")
    ids = torch.empty(n, TOPK, dtype=torch.int32, device="cuda")
    tei = torch.empty(n, TOPK, dtype=torch.int32, device="cuda")
    ops.topk_hash_softplus_sqrt(w, ids, tei, xd, renormalize=True, routed_scaling_factor=SCALE, e_score_correction_bias=None,
                                input_tokens=tokd, hash_indices_table=tabd)
    torch.cuda.synchronize()
    return _u32(w), ids.cpu().numpy().astype(np.int32)


def launch_clamp(gud) -> np.ndarray:
    import torch
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

    out = torch.empty(gud.shape[0], gud.shape[1] // 2, dtype=torch.bfloat16, device="cuda")
    torch.ops._C.silu_and_mul_with_clamp(out, gud, LIMIT, 1.0, 0.0)
    torch.cuda.synchronize()
    return out.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def kernel_names(fn: Callable[[], Any], path: str) -> dict[str, Any]:
    """The device kernels one call ran (torch.profiler), against the one the path dispatches to."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
    names = sorted({e.name for e in prof.events() if e.device_type == DeviceType.CUDA and "memcpy" not in e.name.lower()
                    and "memset" not in e.name.lower()})
    ok = len(names) == 1 and all(k in names[0] for k in KERNEL_NAMES[path])
    if path in KERNEL_FLAGS:
        ok = ok and any(f in names[0] for f in KERNEL_FLAGS[path])
    return {"names": [n[:400] for n in names], "ok": ok}


def capture(rows: dict[str, dict[str, np.ndarray]], reps: int = 2) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Every path's rows through its kernel `reps` times: the outputs, and whether the launches were deterministic and left their
    inputs unwritten."""
    import torch

    rec: dict[str, Any] = {}
    got: dict[str, dict[str, np.ndarray]] = {}
    R = rows["router"]
    outs: list[list[tuple[np.ndarray, np.ndarray]]] = []
    intact = True
    for g in range(len(R["b"])):
        xd, bd = _device(R["x"][g * GROUP:(g + 1) * GROUP]), _device(R["b"][g])
        before = (xd.clone(), bd.clone())
        outs.append([launch_router(xd, bd) for _ in range(reps)])
        intact &= _equal_f32_words(xd, before[0]) and _equal_f32_words(bd, before[1])
    rec["router"] = {"launches": reps * len(outs), "inputs_intact": intact, "deterministic": all(
        np.array_equal(o[0][0], r[0]) and np.array_equal(o[0][1], r[1]) for o in outs for r in o[1:])}
    got["router"] = {"w": np.concatenate([o[0][0] for o in outs]), "ids": np.concatenate([o[0][1] for o in outs])}
    Hr = rows["hash"]
    xd, tokd, tabd = _device(Hr["x"]), _device(Hr["tokens"].view(U32), torch.int32), _device(Hr["tid2eid"].view(U32), torch.int32)
    before = xd.clone()
    o = [launch_hash(xd, tokd, tabd) for _ in range(reps)]
    rec["hash"] = {"launches": reps, "inputs_intact": bool(torch.equal(xd.view(torch.int32), before.view(torch.int32))),
                   "deterministic": all(np.array_equal(o[0][0], r[0]) and np.array_equal(o[0][1], r[1]) for r in o[1:])}
    got["hash"] = {"w": o[0][0], "ids": o[0][1]}
    for path in PATHS[2:]:
        gud = _bf16_device(rows[path]["gu"])
        before = gud.clone()
        o = [launch_clamp(gud) for _ in range(reps)]
        rec[path] = {"launches": reps, "inputs_intact": bool(torch.equal(gud.view(torch.int16), before.view(torch.int16))),
                     "deterministic": all(np.array_equal(o[0], r) for r in o[1:])}
        got[path] = {"out": o[0]}
    return rec, got


def _kinds(path: str) -> tuple[str, ...]:
    return ROW_KINDS if path == "router" else HASH_KINDS if path == "hash" else CLAMP_KINDS


def _row_kind(path: str, P: dict[str, np.ndarray]) -> np.ndarray:
    return P["row_kind"] if path == "router" else P["kind"]


def compare(rows: dict[str, dict[str, np.ndarray]], got: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Every path's outputs against its twin, with the mismatching rows by kind and the first few."""
    rec: dict[str, Any] = {}
    for path, P in rows.items():
        want = expect(path, P)
        bad = np.zeros(len(next(iter(want.values()))), bool)
        for k, v in want.items():
            bad |= (v != got[path][k]).any(1)
        bad = np.flatnonzero(bad)
        kinds, rk = _kinds(path), _row_kind(path, P)
        rec[path] = {"rows": len(rk), "mismatch": len(bad),
                     "mismatch_by_kind": {kinds[k]: int(np.count_nonzero(rk[bad] == k)) for k in range(len(kinds))},
                     "first": [{"row": int(i), "kind": kinds[rk[i]], **{f"gpu_{k}": got[path][k][i].tolist() for k in want},
                                **{f"twin_{k}": want[k][i].tolist() for k in want}} for i in bad[:4]]}
    return rec


# -------------------------------------------------------------------------------------------------------------------- program
_PROGRAMS: dict[str, Any] = {}


def _program(path: str):
    from verity.ml.circuits.defs import bind

    if path not in _PROGRAMS:
        _PROGRAMS[path] = (bind(D.MoeRouterSqrtSoftplus, E=E, SCALE=SCALE) if path == "router" else
                           bind(D.MoeRouterHash, E=E, SCALE=SCALE) if path == "hash" else
                           bind(D.SiluMulClamp, I=int(path[5:]), LIMIT=LIMIT))
    return _PROGRAMS[path]


def program_inputs(path: str, P: dict[str, np.ndarray], i: int) -> list[int]:
    """Row i of a path as the Program's input words."""
    if path == "router":
        return [int(v) for v in P["x"][i]] + [int(v) for v in row_bias(P)[i]]
    if path == "hash":
        return [int(v) for v in P["x"][i]] + [int(v) for v in hash_ids(P)[i]]
    return [int(v) for v in P["gu"][i]]


def program_outputs(path: str, got: dict[str, np.ndarray], i: int) -> list[int]:
    """Row i's captured outputs as the Program returns them."""
    if path == "router":
        return got["w"][i].tolist() + got["ids"][i].tolist()
    if path == "hash":
        return got["w"][i].tolist()
    return got["out"][i].tolist()


def program_row(job: tuple[str, list[int]]) -> list[int]:
    """The Program of a path on one row's input words (the reference evaluator)."""
    from verity.ml.circuits.evaluation import evaluate

    path, words = job
    return [int(v) for v in evaluate(_program(path), words)]


def program_compare(rows: dict[str, dict[str, np.ndarray]], got: dict[str, dict[str, np.ndarray]], idx: dict[str, np.ndarray],
                    workers: int) -> dict[str, Any]:
    """The Programs on rows `idx[path]` of each path, against the captured outputs."""
    jobs = [(path, program_inputs(path, rows[path], int(i))) for path in idx for i in idx[path]]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        outs = list(ex.map(program_row, jobs, chunksize=max(1, len(jobs) // (4 * workers))))
    rec: dict[str, Any] = {}
    k = 0
    for path, ii in idx.items():
        mine, k = outs[k:k + len(ii)], k + len(ii)
        bad = [int(i) for i, o in zip(ii, mine) if o != program_outputs(path, got[path], int(i))]
        rec[path] = {"rows": len(ii), "mismatch": len(bad), "first": bad[:8]}
    rec["ok"] = all(rec[p]["mismatch"] == 0 for p in idx)
    return rec


def sample(path: str, P: dict[str, np.ndarray], k: int, seed: int) -> np.ndarray:
    """One row of each kind (for the router, each (bias kind, row kind) pair), then random rows, `k` in all (sorted)."""
    key = (P["bias_kind"][np.arange(len(P["x"])) // GROUP].astype(np.int64) * 16 + P["row_kind"] if path == "router"
           else _row_kind(path, P).astype(np.int64))
    _, first = np.unique(key, return_index=True)
    idx = np.sort(first)[:k]
    rest = np.setdiff1d(np.arange(len(key)), idx)
    extra = np.random.default_rng(seed).choice(rest, max(0, min(k - len(idx), len(rest))), replace=False)
    return np.sort(np.concatenate([idx.astype(np.int64), extra.astype(np.int64)]))


def check(d: str, workers: int, limit: int | None = None) -> dict[str, Any]:
    """`--check`: the Programs on every captured row of every path (or the first `limit` of each)."""
    rows, got = _unflat(np.load(Path(d) / ROWS_FILE))
    idx = {p: np.arange(len(_row_kind(p, rows[p])) if limit is None else min(limit, len(_row_kind(p, rows[p])))) for p in PATHS}
    print(f"[dsv4-moe] {_utc()} check: {sum(len(v) for v in idx.values())} rows on {workers} workers", flush=True)
    rec = {"rows_file_sha256": _sha256_file(Path(d) / ROWS_FILE), "started": _utc(), **program_compare(rows, got, idx, workers),
           "finished": _utc()}
    record = Path(d) / RECORD
    rec["record_digest"] = json.loads(record.read_text()).get("digest") if record.exists() else None
    (Path(d) / CHECK_FILE).write_text(json.dumps(rec, indent=1) + "\n")
    print(f"DSV4-MOE-PROGRAM-CHECK {'OK' if rec['ok'] else 'FAIL'} "
          + " ".join(f"{p}={rec[p]['rows']}!={rec[p]['mismatch']}" for p in PATHS), flush=True)
    return rec


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------ run
def host_keeps_subnormals() -> bool:
    """The twin's host float operations (adds, products, divides, sqrt, the f64 -> f32 roundings) need subnormals neither flushed nor
    read as zero."""
    tiny = np.array([1e-40], np.float32)
    return bool(tiny[0] != 0 and (tiny * np.float32(1.0))[0] != 0 and np.float64(1e-40).astype(np.float32) != 0)


def _verdict(doc: dict[str, Any]) -> bool:
    c = doc["capture"]
    return (doc["gate"]["ok"] and doc["host_subnormals"] and doc["mufu"].get("ok") is True and doc["mufu_probe"].get("ok") is True
            and all(doc["kernels"][p]["ok"] for p in PATHS)
            and all(c[p]["deterministic"] and c[p]["inputs_intact"] for p in PATHS)
            and all(doc["rows"][p]["mismatch"] == 0 for p in PATHS) and doc["program"]["ok"] is True)


def run(out: str, seed: int, n: int, clamp_n: int, min_driver: int, want_cc: str, want_sms: int, workers: int,
        program_sample: int) -> dict[str, Any]:
    from verity_vllm.engine import engine_profile as prof
    from verity_vllm.properties.record import seal as seal_record

    from tests.properties import fa2_target_capture_gpu as C
    from tests.properties import fa_tap_exactness_gpu as G

    env_flags = prof.apply_env()
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    import torch
    import vllm
    import vllm._custom_ops

    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"schema": SCHEMA, "utc": _utc(), "argv": sys.argv, "environment": G._environment(), "env_flags": env_flags,
                           "host_subnormals": host_keeps_subnormals(),
                           "config": {"E": E, "TOPK": TOPK, "SCALE": SCALE, "LIMIT": LIMIT, "clamp_widths": list(WIDTHS),
                                      "indices_dtype": "int32", "tid2eid_dtype": "int32", "seed": seed, "rows": n, "clamp_rows": clamp_n},
                           "gate": C.gate(min_driver, want_cc, want_sms)}
    print(f"[dsv4-moe] {_utc()} gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-MOE-CAPTURE GATE-FAIL", flush=True)
        raise SystemExit(3)
    p = torch.cuda.get_device_properties(0)
    lib = Path(vllm.__file__).parent
    doc["binary"] = {name: (C._archs(str(so[0])) if so else {"error": f"no vllm/{name}*.so"})
                     for name, so in (("_moe_C", sorted(lib.glob("_moe_C*.so"))), ("_C", sorted(lib.glob("_C*.so"))))}
    for name, b in doc["binary"].items():
        print(f"[dsv4-moe] {_utc()} binary {name} sass={b.get('sass_archs')} ptx={b.get('ptx_archs')}", flush=True)
    with tempfile.TemporaryDirectory() as work:
        doc["mufu"] = C.mufu(Path(work), f"sm_{p.major}{p.minor}")
    for k, b in doc["mufu"]["builds"].items():
        b.pop("tables_kept", None)
        print(f"[dsv4-moe] {_utc()} mufu {k}: ok={b.get('ok')} ex2!={b.get('ex2_entries_differing')} "
              f"rcp!={b.get('rcp_entries_differing')} {b.get('error', '')}", flush=True)
    doc["mufu_probe"] = mufu_probe()
    print(f"[dsv4-moe] {_utc()} mufu probe ok={doc['mufu_probe']['ok']} " + " ".join(
        f"{k}!={v['mismatch']}" for k, v in doc["mufu_probe"].items() if isinstance(v, dict)), flush=True)

    rows = all_rows(seed, n, clamp_n)
    doc["inputs"] = {p: {"rows": int(len(_row_kind(p, P))), "sha256": digest(*P.values())} for p, P in rows.items()}
    R, Hr = rows["router"], rows["hash"]
    probe = {"router": lambda: launch_router(_device(R["x"][:GROUP]), _device(R["b"][0])),
             "hash": lambda: launch_hash(_device(Hr["x"][:GROUP]), _device(Hr["tokens"][:GROUP].view(U32), torch.int32),
                                         _device(Hr["tid2eid"].view(U32), torch.int32)),
             **{p: (lambda p=p: launch_clamp(_bf16_device(rows[p]["gu"][:GROUP]))) for p in PATHS[2:]}}
    doc["kernels"] = {p: kernel_names(probe[p], p) for p in PATHS}
    for path, k in doc["kernels"].items():
        print(f"[dsv4-moe] {_utc()} kernel {path}: ok={k['ok']} {k['names']}", flush=True)
    doc["capture"], got = capture(rows)
    print(f"[dsv4-moe] {_utc()} captured {json.dumps(doc['capture'])}", flush=True)
    doc["rows"] = compare(rows, got)
    for path in PATHS:
        r = doc["rows"][path]
        print(f"[dsv4-moe] {_utc()} rows {path}: {r['rows']} rows, twin!={r['mismatch']} {r['mismatch_by_kind']}", flush=True)
        for m in r["first"]:
            print(f"[dsv4-moe]    {json.dumps(m)[:900]}", flush=True)
    np.savez_compressed(outdir / ROWS_FILE, **_flat(rows, got))
    doc["rows_file_sha256"] = _sha256_file(outdir / ROWS_FILE)
    idx = {p: sample(p, rows[p], program_sample, seed + 7) for p in PATHS}
    doc["program"] = program_compare(rows, got, idx, workers)
    print(f"[dsv4-moe] {_utc()} program: {json.dumps(doc['program'])}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"DSV4-MOE-CAPTURE {'OK' if doc['ok'] else 'FAIL'} {doc['digest']} "
          + " ".join(f"{p}!={doc['rows'][p]['mismatch']}" for p in PATHS)
          + f" mufu={doc['mufu'].get('ok')} probe={doc['mufu_probe'].get('ok')} program={doc['program']['ok']} {outdir / RECORD}",
          flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4's routers and clamped SwiGLU on this GPU against registry.dsv4_moe")
    ap.add_argument("--out", help=f"directory for {RECORD} and {ROWS_FILE}")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rows", type=int, default=8192, help="router rows, and hash rows")
    ap.add_argument("--clamp-rows", type=int, default=2048, help="random clamp rows at d = 384 (half as many at d = 9), before the sweep")
    ap.add_argument("--min-driver", type=int, default=550)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=4, help="processes for the Programs' reference evaluation")
    ap.add_argument("--program-sample", type=int, default=96, help="captured rows of each path the GPU run also checks against its Program")
    ap.add_argument("--check", metavar="DIR", help=f"evaluate the Programs on every row of DIR/{ROWS_FILE} (no GPU)")
    ap.add_argument("--limit", type=int, default=None, help="--check: only the first LIMIT rows of each path")
    ap.add_argument("--verify", metavar="RECORD", help="check a record's digest and verdict (no GPU)")
    a = ap.parse_args(argv)
    if a.verify:
        from verity_vllm.properties.record import PropertyRecord

        rec = PropertyRecord(json.loads(Path(a.verify).read_text()))
        problems = rec.problems()
        print(f"{a.verify}: {'intact, ok=' + str(rec.ok) if not problems else '; '.join(problems)} (digest {rec.digest})")
        return 1 if problems or not rec.ok else 0
    if a.check:
        return 0 if check(a.check, a.workers, a.limit)["ok"] else 1
    if not a.out:
        ap.error("--out is required (or --check DIR, --verify RECORD)")
    out = str(Path(a.out).resolve())
    os.chdir(Path(__file__).resolve().parents[2])
    doc = run(out, a.seed, a.rows, a.clamp_rows, a.min_driver, a.want_cc, a.want_sms, a.workers, a.program_sample)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
