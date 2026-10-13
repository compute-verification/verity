"""DeepSeek-V4-Pro's query norm + SWA cache insert and its indexer's query quant on Hopper against their Programs: the H200's
go/no-go for binding `registry.dsv4_qnorm_insert` (`Dsv4QNormRopeKvInsert_v1` for vLLM d9105ea80's
`torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`, as `DeepseekV4Attention._fused_qnorm_rope_kv_insert` calls it on
the `fp8_ds_mla` cache, attention.py:733-750; `Dsv4IndexerQ_v1` for `fused_indexer_q_rope_quant` and the CuTe DSL
`IndexerQFp8Kernel` it dispatches to, as the indexer calls it, attention.py:1018-1026) at DeepSeek-V4-Pro's statics on one TP8
rank: 16 query heads padded to FlashMLA's 64, the 64-token SWA page, 64 indexer heads of 128.

One GPU process writes `<out>/dsv4_qnorm_insert_capture.json`, the property record `dsv4_qnorm_insert_capture`
(`properties.record`); it prints one line per launch and per case, then `DSV4-QNORM-INSERT GO` or `NO-GO`.  `ok` (GO) = every
check below.  The device gate runs first and nothing else runs when it fails (exit 3).  A phase that raises records its error
and fails the verdict without stopping the others.  The operands are random at the served shapes, so no model weights are read.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `_select_dsv4_attn_cls` picks `DeepseekV4FlashMLAAttention`, whose layout is `fp8_ds_mla` (so its SWA cache is uint8
             and the insert is the op above), `get_padded_num_q_heads(16)` is 64, the op is registered, `has_cutedsl()` holds and
             the platform's FP8 is e4m3fn (so the indexer's FP8 path is the CuTe kernel), `dsa_indexer_uses_fp4` is False, and the
             installed attention.py, fused_indexer_q.py, fused_indexer_q_cutedsl.py and cute_utils are d9105ea80's (`SOURCES`).
  insert     per case (`INSERT_CASES`), the op on q (T, 16, 512) and kv (T, 512) bf16, a paged cache (blocks, `page_bytes(64)`)
             uint8 of random bytes, slot_mapping (n <= T: the DP-padded rows are normed and not inserted; some slots -1),
             positions int64 and the f32 cos_sin_cache (MAX_POS, 64), padded_heads 64, eps 1e-6, block_size 64; twice, on
             caches of two different random fills.  Every word of the returned q (T, 64, 512) against
             `dsv4_qnorm_insert_twin.insert` (the pad heads +0), every byte of the cache against the twin's rows scattered into
             the fill (`scatter`: a written byte where the twin writes it, every other byte the fill's), and each inserted row as
             FlashMLA's decode reads it (`dsv4_flashmla_capture_gpu.gather_rows`, `FlashMlaKvRow_v1`'s nope, scale and rope)
             against the twin's; on the first and the last token of each case and every token of `edge`, the reference
             `Dsv4QNormRopeKvInsert_v1{NH=16,...}` against the twin.  Fewer than 1024 tokens run the kernel's one-warp-per-slot
             grid, more its reduced grid (`INSERT_CUTOFF`); the cases take both.
  indexer    per case (`INDEXER_CASES`), `fused_indexer_q_rope_quant(positions, q (T, 64, 128) bf16, cos_sin_cache, weights
             (T, 64) bf16, 128^-0.5, 64^-0.5, use_fp4=False)`, twice: the dispatch reaches `fused_indexer_q_rope_quant_fp8_cutedsl`
             once a launch and runs coarsen 1 below 512 tokens and 4 from 512 (`COARSEN_CUTOFF`) on an f32 RoPE table; every
             byte of q_fp8 and every word of weights_out against `dsv4_qnorm_insert_twin.indexer_q`, and the reference
             `Dsv4IndexerQ_v1{H=64,...}` on the first and the last token and every token of `edge`.
  bind       the Programs bind at the served statics, the insert's row is `FlashMlaKvRow_v1`'s input (its nope and rope, its
             first seven scale bytes) and the indexer's outputs are `DsaMqaLogit_v1{H=64, D=128}`'s q and w.

The insert's cases: `decode` a few tokens; `padded` a DP-padded batch (slot_mapping shorter than q); `chunk` 1023 tokens and
`reduced` 1024 (the last of one grid and the first of the other); `long` 1536; `wide` words over many binades with NaN,
infinities and subnormals among them (`dsv4_qnorm_insert_twin._bf16`); `edge` the special words of `insert_edge`: zero heads
and rows, infinite and NaN elements, every KV block's amax at 448 2^k and one bf16 ulp either side (the scale byte's
ceil(log2) turns there), bf16's largest finite word (the squares overflow), subnormals, the RoPE's contraction (each rotated
pair (x, x) at a table row of s = c or s = -c, where a fused and an unfused rotation differ), and every special row of the
table (`EDGE_CS`).  The indexer's: `decode`, `short` 200 tokens, `last1` 511 (the last of coarsen 1), `coarse` 512, `odd` 777,
`wide` and `edge` (`indexer_edge`).

`sumsq_probe` is separate from those served-static cases and their totals: at the non-served eps=2^-140, one crafted query head
has inexact subnormal squares.  It is GO only when the kernel equals `Dsv4QNormRope_v1{D=512,DR=64,EPS=2^-140}` and differs from
the deliberately unfused FMUL-then-FADD sum-of-squares model on several output words.

GO needs every served case's words compared with 0 mismatches, the references sampled with 0 disagreements with the twin, the
separate sum-of-squares probe, every launch check, the selection and the bindings.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/dsv4_qnorm_insert_capture_gpu.py --out DIR [--scratch DIR] [--quick]
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import dsv4_flashmla_capture_gpu as M
from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.program.kernels import derived_rows as DW
from verity_vllm.program.kernels import dsv4_qnorm_insert_twin as W
from verity_vllm.program.kernels import flashmla_twin as FT
from verity_vllm.program.registry import dsv4_qnorm_insert as R
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/dsv4-qnorm-insert-capture/v1"
RECORD = "dsv4_qnorm_insert_capture.json"
NH, NHP, DN, DR, QB, EPS = (R.DSV4_INSERT[k] for k in ("NH", "NHP", "DN", "DR", "QB", "EPS"))
D = DN + DR
IH, ID, IDR, ISCALE = (R.DSV4_INDEXER[k] for k in ("H", "D", "DR", "SCALE"))
#: the indexer's softmax scale (head_dim^-0.5) and head scale (n_head^-0.5), attention.py:912, :1023-1024; the CuTe wrapper
#: multiplies them in double and passes the f32 rounding (fused_indexer_q_cutedsl.py:87)
SOFTMAX_SCALE, HEAD_SCALE = ID ** -0.5, IH ** -0.5
PBS = M.SWA_PBS
MAX_POS, ROPE_THETA = 1 << 16, 10000.0
EDGE_POS = MAX_POS - 16
INSERT_CUTOFF = 1024      # NUM_TOKEN_CUTOFF, fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:105, :624
COARSEN_CUTOFF = 512      # fused_indexer_q_cutedsl.py:83
#: sha256 of the files the Definitions were read from, at vLLM d9105ea8001e0a6d77a96327d17515bb5791fb36 (the .cu is compiled into
#: `_C_stable_libtorch`, whose SASS architectures `stack` records)
SOURCES = {
    "vllm/models/deepseek_v4/attention.py": "b6b32d5a98737ba2d7e619b66de0b055f6189f68d513d9dc1e11f05aaa362616",
    "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py": "f676cba24ff9923af08fb25986e65ad91b9e8c3da011463bf2b43e38531c24fe",
    "vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py":
        "784f011ace93597727d0d6ce8b320e8e941a367cf1cb00f7015b3246e12321bc",
    "vllm/cute_utils/__init__.py": "36d72f3916c6a5e850f721e0eb8a53cf1d91c8b7a34caa3ae6f67b936ccd3b1f",
    "vllm/cute_utils/cvt.py": "b7de75d6a24c833a89915ec4aaefb2378155a1931c4fe400db23d57800927104",
}
F32_ONE, F32_RSQRT2, F32_NAN, F32_INF, SIGN = 0x3F800000, 0x3F3504F3, 0x7FC00000, 0x7F800000, 1 << 31
#: the special rows of the table, at EDGE_POS + j: (cos, sin) words for every pair, ("full", +-1) a random full-mantissa cos in
#: [1/2, 1) with sin = +-cos, ("words", None) every entry an edge word
EDGE_CS: tuple[tuple[Any, Any], ...] = (
    (F32_ONE, 0), (F32_RSQRT2, F32_RSQRT2), (F32_RSQRT2, F32_RSQRT2 | SIGN), ("full", 1), ("full", -1), (SIGN, 0), (F32_ONE, 1),
    (0x71800000, 0x71800000), (F32_NAN, F32_ONE), (F32_INF, F32_INF | SIGN), ("words", None), (0x00400000, 0x00400000),
)
#: the KV blocks' amax exponents in `insert_edge` (amax 448 2^k); the first, 1.07e-4, sits just above the amax floor 1e-4 and
#: shares its scale byte 105
EDGE_K = (-22, -14, -1, 0, 1, 20, 110)
#: A non-served diagnostic eps and one lane repeated across the warp.  Its widened bf16 squares are subnormal and inexact; the
#: fused and unfused 16-element lane sums differ, and that difference survives the butterfly, rsqrt and bf16 output rounding.
SUMSQ_PROBE_EPS = 2.0 ** -140
SUMSQ_PROBE_LANE = (0x1C0A, 0x1BB3, 0x1DDC, 0x1A94, 0x1D97, 0x1B6F, 0x1C3C, 0x1B1F,
                    0x1C64, 0x1CFA, 0x1CA3, 0x1AB7, 0x1BBD, 0x1BBA, 0x1C40, 0x1BCC)
SUMSQ_PROBE_PADDED_HEADS = 8
#: case -> (tokens, inserted tokens, seed, operands)
INSERT_CASES = {"decode": (8, 8, 1, "act"), "padded": (48, 40, 2, "act"), "chunk": (1023, 1023, 3, "act"),
                "reduced": (1024, 1024, 4, "act"), "long": (1536, 1500, 5, "act"), "wide": (128, 128, 6, "wide"),
                "edge": (40, 40, 7, "edge")}
#: case -> (tokens, seed, operands)
INDEXER_CASES = {"decode": (8, 11, "act"), "short": (200, 12, "act"), "last1": (511, 13, "act"), "coarse": (512, 14, "act"),
                 "odd": (777, 15, "act"), "wide": (128, 16, "wide"), "edge": (40, 17, "edge")}
QUICK_INSERT, QUICK_INDEXER = ("decode", "reduced", "edge"), ("decode", "coarse", "edge")
#: tokens per twin job
JOB_TOKENS = 128
TOTALS = ("words", "mismatch", "reference", "twin_mismatch")
_error, _save, _load, _u16 = M._error, M._save, M._load, M._u16


def cases_of(quick: bool) -> tuple[tuple[str, str], ...]:
    ins = QUICK_INSERT if quick else tuple(INSERT_CASES)
    idx = QUICK_INDEXER if quick else tuple(INDEXER_CASES)
    return tuple(("insert", c) for c in ins) + tuple(("indexer", c) for c in idx)


def _flat(v) -> list[int]:
    return [x for e in v for x in _flat(e)] if isinstance(v, (list, tuple)) else [int(v)]


# --------------------------------------------------------------------------------------------------------------- the operands
@functools.cache
def cos_sin_cache() -> np.ndarray:
    """(MAX_POS, DR) f32 words, cos in the first half and sin in the second as both kernels read them: below EDGE_POS row p is
    cos(p t_i) | sin(p t_i), t_i = ROPE_THETA^(-i / 32); from EDGE_POS the special rows `EDGE_CS`, then ordinary rows."""
    half = DR // 2
    ang = np.arange(MAX_POS, dtype=np.float64)[:, None] * ROPE_THETA ** (-np.arange(half, dtype=np.float64) / half)
    cs = np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32).view(np.uint32)
    rng = np.random.default_rng(99)
    for j, (c, s) in enumerate(EDGE_CS):
        row = cs[EDGE_POS + j]
        if c == "full":
            m = rng.uniform(0.5, 1.0, half).astype(np.float32).view(np.uint32) | np.uint32(1)
            row[:half], row[half:] = m, (m if s > 0 else m | np.uint32(SIGN))
        elif c == "words":
            row[:] = rng.choice(W._F32_EDGE, DR)
        else:
            row[:half], row[half:] = c, s
    return cs


def _act(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """bf16 activations: each row (last axis) standard normal times 2^U(-6, 6)."""
    x = rng.standard_normal(shape) * np.exp2(rng.uniform(-6, 6, shape[:-1] + (1,)))
    return M._bf16(x.astype(np.float32))


def amax_word(k: int, ulp: int = 0) -> int:
    """The bf16 word of 448 2^k (1.75 2^(k + 8)), or ulp bf16 steps from it."""
    return ((127 + 8 + k) << 7 | 0x60) + ulp


def _block_at(rng: np.random.Generator, n: int, word: int) -> np.ndarray:
    """n bf16 words whose largest magnitude is ``word``'s (at a random index, a random sign), the rest below it."""
    v = np.array([word << 16], dtype=np.uint32).view(np.float32)[0]
    x = M._bf16((rng.uniform(-0.98, 0.98, n) * v).astype(np.float32))
    x[rng.integers(n)] = word | (int(rng.integers(2)) << 15)
    return x


def _pairs_equal(x: np.ndarray, base: int) -> None:
    """Each RoPE pair of x[..., base:] made (x, x): its odd element its even one."""
    x[..., base + 1::2] = x[..., base::2]


def _sub(rng: np.random.Generator, shape) -> np.ndarray:
    return (rng.integers(1, 0x80, shape, dtype=np.uint16) | (rng.integers(0, 2, shape, dtype=np.uint16) << 15)).astype(np.uint16)


def _slots(rng: np.random.Generator, n: int) -> tuple[np.ndarray, int]:
    """n distinct slots over ceil(n / 64) + 2 blocks, about 1 in 25 (and the second, when n > 1) -1; and the block count."""
    nb = -(-n // PBS) + 2
    s = rng.choice(nb * PBS, n, replace=False).astype(np.int64)
    s[rng.random(n) < 0.04] = -1
    if n > 1:
        s[1] = -1
    return s, nb


def sumsq_probe_operands() -> dict[str, np.ndarray]:
    """One head whose every lane is `SUMSQ_PROBE_LANE`, an identity RoPE row, and a skipped zero KV row."""
    q = np.tile(np.asarray(SUMSQ_PROBE_LANE, dtype=np.uint16), W.LANES).reshape(1, 1, D)
    cs = np.zeros((1, DR), dtype=np.uint32)
    cs[:, :DR // 2] = F32_ONE
    return {"q": q, "kv": np.zeros((1, D), dtype=np.uint16), "cs": cs, "positions": np.zeros(1, dtype=np.int64),
            "slots": np.full(1, -1, dtype=np.int64)}


def sumsq_probe_reference(q: np.ndarray, cs: np.ndarray, *, fused: bool) -> np.ndarray:
    """The probe head through the SASS chain, or through the FMUL-then-FADD sum-of-squares mutant when ``fused`` is false."""
    x = FT._bf16(np.asarray(q, dtype=np.uint16))
    lanes = x.reshape(W.LANES, D // W.LANES)
    acc = W.fmul(lanes[:, 0], lanes[:, 0])
    for i in range(1, lanes.shape[1]):
        product = W.fmul(lanes[:, i], lanes[:, i])
        acc = W.ffma(lanes[:, i], lanes[:, i], acc) if fused else W.fadd(acc, product)
    for h in (16, 8, 4, 2, 1):
        acc = W.fadd(acc[:h], acc[h:2 * h])
    r = DW.rsqrt_approx(W.ffma(acc[0], W.f32(1.0 / D), W.f32(SUMSQ_PROBE_EPS)))
    return DW.f2fp_bf16(W._rope(W.fmul(x, r), np.asarray(cs, dtype=np.uint32), DR))


def insert_operands(case: str) -> dict[str, Any]:
    """An insert case's operands: q (T, NH, D) and kv (T, D) bf16 words, positions (T,), cs (T, DR) the table's rows at them,
    slots (n,), two random fills (2, blocks, page_bytes(64))."""
    T_, n, seed, kind = INSERT_CASES[case]
    rng = np.random.default_rng(seed)
    if kind == "wide":
        q, kv = W._bf16(rng, (T_, NH, D)), W._bf16(rng, (T_, D))
    else:
        q, kv = _act(rng, (T_, NH, D)), _act(rng, (T_, D))
    pos = rng.integers(0, EDGE_POS, T_).astype(np.int64)
    if kind == "wide":
        pos[::2] = EDGE_POS + rng.integers(0, len(EDGE_CS), pos[::2].size)
    o = {"case": case, "q": q, "kv": kv, "positions": pos}
    if kind == "edge":
        insert_edge(rng, o)
    o["cs"] = cos_sin_cache()[o["positions"]]
    o["slots"], nb = _slots(rng, n)
    o["fills"] = rng.integers(0, 256, (2, nb, M.page_bytes(PBS)), dtype=np.uint8)
    return o


def insert_edge(rng: np.random.Generator, o: dict[str, Any]) -> None:
    """`edge`'s tokens.  0: every head and the row +0 (rsqrt of eps alone; every block the floor's scale byte 105).  1: heads 0-2
    an element +inf, -inf, NaN and head 3 every element NaN; KV blocks 0-2 likewise and block 3 every element NaN.  2-4: KV
    block b's amax 448 2^EDGE_K[b] exactly, one bf16 ulp above, one below.  5: bf16's largest finite word leading every head and
    every block, signs mixed (the squares overflow).  6: subnormals everywhere.  7-10: every RoPE pair (x, x), at the rows s = c
    and s = -c (EDGE_CS 1-4).  11-22: one token on each row of EDGE_CS.  23-39: `wide` words on special and ordinary rows."""
    q, kv, pos = o["q"], o["kv"], o["positions"]
    q[0], kv[0] = 0, 0
    for h, w in enumerate((0x7F80, 0xFF80, 0x7FC0)):
        q[1, h, rng.integers(D)] = w
        kv[1, h * QB + rng.integers(QB)] = w
    q[1, 3], kv[1, 3 * QB:4 * QB] = 0xFFC1, 0x7FC0
    for t, ulp in ((2, 0), (3, 1), (4, -1)):
        for b, k in enumerate(EDGE_K):
            kv[t, b * QB:(b + 1) * QB] = _block_at(rng, QB, amax_word(k, ulp))
    q[5, :, 0] = np.where(rng.random(NH) < 0.5, 0x7F7F, 0xFF7F)
    kv[5, :DN:QB] = np.where(rng.random(DN // QB) < 0.5, 0x7F7F, 0xFF7F)
    q[6], kv[6] = _sub(rng, (NH, D)), _sub(rng, (D,))
    for t, j in zip(range(7, 11), range(1, 5)):
        _pairs_equal(q[t], DN)
        _pairs_equal(kv[t], DN)
        pos[t] = EDGE_POS + j
    pos[11:11 + len(EDGE_CS)] = EDGE_POS + np.arange(len(EDGE_CS))
    lo = 11 + len(EDGE_CS)
    q[lo:], kv[lo:] = W._bf16(rng, q[lo:].shape), W._bf16(rng, kv[lo:].shape)
    pos[lo::2] = EDGE_POS + rng.integers(0, len(EDGE_CS), pos[lo::2].size)


def indexer_operands(case: str) -> dict[str, Any]:
    """An indexer case's operands: q (T, IH, ID) and w (T, IH) bf16 words, positions (T,), cs (T, IDR)."""
    T_, seed, kind = INDEXER_CASES[case]
    rng = np.random.default_rng(seed)
    if kind == "wide":
        q, w = W._bf16(rng, (T_, IH, ID)), W._bf16(rng, (T_, IH))
    else:
        q, w = _act(rng, (T_, IH, ID)), _act(rng, (T_, IH))
    pos = rng.integers(0, EDGE_POS, T_).astype(np.int64)
    if kind == "wide":
        pos[::2] = EDGE_POS + rng.integers(0, len(EDGE_CS), pos[::2].size)
    o = {"case": case, "q": q, "w": w, "positions": pos}
    if kind == "edge":
        indexer_edge(rng, o)
    o["cs"] = cos_sin_cache()[o["positions"]]
    return o


def indexer_edge(rng: np.random.Generator, o: dict[str, Any]) -> None:
    """`edge`'s tokens.  0: every head and weight +0.  1: heads 0-2 an element +inf, -inf, NaN and head 3 every element NaN, the
    weights edge words.  2-4: head h's amax 448 2^(h - 40) exactly, one bf16 ulp above, one below, on the identity row (the
    power of two the scale rounds up to turns there).  5-8: NoPE +0 and every RoPE pair (x, x), at the rows s = c and s = -c.
    9: subnormals.  10: bf16's largest finite word leading every head.  11-22: one token on each row of EDGE_CS.  23-39: `wide`
    words."""
    q, w, pos = o["q"], o["w"], o["positions"]
    q[0], w[0] = 0, 0
    for h, x in enumerate((0x7F80, 0xFF80, 0x7FC0)):
        q[1, h, rng.integers(ID)] = x
    q[1, 3] = 0xFFC1
    w[1] = np.resize(W._BF16_EDGE, IH)
    for t, ulp in ((2, 0), (3, 1), (4, -1)):
        for h in range(IH):
            q[t, h] = _block_at(rng, ID, amax_word(h - 40, ulp))
        pos[t] = EDGE_POS
    for t, j in zip(range(5, 9), range(1, 5)):
        q[t, :, :ID - IDR] = 0
        _pairs_equal(q[t], ID - IDR)
        pos[t] = EDGE_POS + j
    q[9], w[9] = _sub(rng, (IH, ID)), _sub(rng, (IH,))
    q[10, :, 0] = np.where(rng.random(IH) < 0.5, 0x7F7F, 0xFF7F)
    pos[11:11 + len(EDGE_CS)] = EDGE_POS + np.arange(len(EDGE_CS))
    lo = 11 + len(EDGE_CS)
    q[lo:], w[lo:] = W._bf16(rng, q[lo:].shape), W._bf16(rng, w[lo:].shape)
    pos[lo::2] = EDGE_POS + rng.integers(0, len(EDGE_CS), pos[lo::2].size)


# ----------------------------------------------------------------------------------------------------------------- the twin side
def written(slots: np.ndarray, nb: int) -> np.ndarray:
    """(nb, page_bytes(64)) bool: the bytes an insert at ``slots`` writes, each slot's 576 data bytes and 8 scale bytes."""
    data, scale = np.ones((len(slots), M.DATA_BYTES), np.uint8), np.ones((len(slots), M.SCALE_BYTES), np.uint8)
    return W.scatter(np.zeros((nb, M.page_bytes(PBS)), np.uint8), slots, data, scale, PBS).astype(bool)


def cache_checks(slots: np.ndarray, fills: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> dict[str, bool]:
    """The caches two inserts of the same operands left on fills 0 and 1: the bytes `written` equal between them, every other
    byte of the second its fill's."""
    mask = written(slots, g1.shape[0])
    return {"rerun": bool(np.array_equal(g1[mask], g2[mask])), "untouched": bool(np.array_equal(g2[~mask], fills[1][~mask]))}


def insert_twin(a: dict[str, np.ndarray], lo: int, hi: int):
    """The twin's insert of tokens [lo, hi): q (n, NHP, D), nope, scale, rope."""
    return W.insert(np.asarray(a["q"][lo:hi]), np.asarray(a["kv"][lo:hi]), np.asarray(a["cs"][lo:hi]), NHP=NHP, DR=DR, QB=QB,
                    EPS=EPS)


def twin_cache(a: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """The cache the first insert leaves: the twin's rows of the inserted tokens scattered into fill 0; and those rows."""
    slots = np.asarray(a["slots"])
    n = slots.size
    rows = W.kv_rows(np.asarray(a["kv"][:n]), np.asarray(a["cs"][:n]), DR=DR, QB=QB)
    return W.scatter(np.asarray(a["fills"][0]), slots, *W.cache_rows(*rows), PBS), rows


def indexer_twin(a: dict[str, np.ndarray], lo: int, hi: int):
    return W.indexer_q(np.asarray(a["q"][lo:hi]), np.asarray(a["cs"][lo:hi]), np.asarray(a["w"][lo:hi]), DR=IDR, SCALE=ISCALE)


def _refs(T_: int, case: str, quick: bool) -> list[int]:
    if case == "edge" and not quick:
        return list(range(T_))
    return [0] if quick else sorted({0, T_ - 1})


def insert_jobs(case: str, paths: dict[str, str], T_: int, quick: bool) -> list[dict[str, Any]]:
    """The cache job; q jobs of JOB_TOKENS tokens; reference jobs on the first and the last token (every token of `edge`; under
    --quick only the first)."""
    job = {"phase": "insert", "case": case, "paths": paths}
    jobs = [{**job, "kind": "cache"}]
    jobs += [{**job, "kind": "q", "tokens": [lo, min(T_, lo + JOB_TOKENS)]} for lo in range(0, T_, JOB_TOKENS)]
    return jobs + [{**job, "kind": "insert-ref", "token": t} for t in _refs(T_, case, quick)]


def indexer_jobs(case: str, paths: dict[str, str], T_: int, quick: bool) -> list[dict[str, Any]]:
    """q jobs of JOB_TOKENS tokens and reference jobs, as `insert_jobs`."""
    job = {"phase": "indexer", "case": case, "paths": paths}
    jobs = [{**job, "kind": "iq", "tokens": [lo, min(T_, lo + JOB_TOKENS)]} for lo in range(0, T_, JOB_TOKENS)]
    return jobs + [{**job, "kind": "indexer-ref", "token": t} for t in _refs(T_, case, quick)]


def _first(rec: dict[str, Any], **where) -> None:
    if rec["first_mismatch"] is None:
        rec["first_mismatch"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in where.items()}


def _tally(rec: dict[str, Any], want: np.ndarray, got: np.ndarray, what: str, offset: tuple[int, ...] = ()) -> None:
    want, got = np.asarray(want), np.asarray(got)
    bad = np.argwhere(want != got)
    rec["words"] += int(got.size)
    rec["mismatch"] += int(bad.shape[0])
    if bad.shape[0]:
        at = tuple(int(v) for v in bad[0])
        _first(rec, what=what, at=[o + i for o, i in zip(offset + (0,) * len(at), at)], want=int(want[at]), kernel=int(got[at]))


def compare_q(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Tokens [t0, t1): every word of the returned q (64 heads, the pads +0) against the twin."""
    a = _load(job["paths"])
    lo, hi = job["tokens"]
    _tally(rec, insert_twin(a, lo, hi)[0], a["q_out"][lo:hi], "q", (lo,))


def compare_cache(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Every byte of the first insert's cache against the twin's rows scattered into its fill, and each inserted row as FlashMLA
    reads it (nope, the seven UE8M0 bytes, rope) against the twin's."""
    a = _load(job["paths"])
    want, (nope, scale, rope) = twin_cache(a)
    _tally(rec, want, a["cache"], "cache")
    slots = np.asarray(a["slots"])
    ok = slots >= 0
    got = M.gather_rows(a["cache"], slots[ok], PBS)
    for what, w, g in zip(("flashmla_nope", "flashmla_scale", "flashmla_rope"), (nope[ok], scale[ok][:, :DN // QB], rope[ok]), got):
        _tally(rec, w, g, what)


def compare_insert_ref(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """One token: `Dsv4QNormRopeKvInsert_v1{NH=16,...}` (the reference) against the twin, every output word."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    t = job["token"]
    y = np.concatenate([np.asarray(v, dtype=np.int64).reshape(-1) for v in insert_twin(a, t, t + 1)])
    fn = bind(R.Dsv4QNormRopeKvInsert, **R.DSV4_INSERT)
    r = np.array(_flat(evaluate(fn, np.asarray(a["q"][t]).reshape(-1).tolist(), np.asarray(a["kv"][t]).tolist(),
                                np.asarray(a["cs"][t]).tolist())), dtype=np.int64)
    bad = np.flatnonzero(r != y)
    rec["reference"] += 1
    rec["twin_mismatch"] += int(bad.size)
    if bad.size:
        _first(rec, what="insert-ref", token=t, word=int(bad[0]), reference=int(r[bad[0]]), twin=int(y[bad[0]]))


def compare_iq(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Tokens [t0, t1): every byte of q_fp8 and every word of weights_out against the twin."""
    a = _load(job["paths"])
    lo, hi = job["tokens"]
    q8, w = indexer_twin(a, lo, hi)
    _tally(rec, q8, a["q_fp8"][lo:hi], "q_fp8", (lo,))
    _tally(rec, w, a["w_out"][lo:hi], "weights_out", (lo,))


def compare_indexer_ref(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """One token: `Dsv4IndexerQ_v1{H=64,...}` (the reference) against the twin."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    t = job["token"]
    q8, w = indexer_twin(a, t, t + 1)
    y = np.concatenate([q8.reshape(-1).astype(np.int64), w.reshape(-1).astype(np.int64)])
    fn = bind(R.Dsv4IndexerQ, **R.DSV4_INDEXER)
    r = np.array(_flat(evaluate(fn, np.asarray(a["q"][t]).reshape(-1).tolist(), np.asarray(a["cs"][t]).tolist(),
                                np.asarray(a["w"][t]).tolist())), dtype=np.int64)
    bad = np.flatnonzero(r != y)
    rec["reference"] += 1
    rec["twin_mismatch"] += int(bad.size)
    if bad.size:
        _first(rec, what="indexer-ref", token=t, word=int(bad[0]), reference=int(r[bad[0]]), twin=int(y[bad[0]]))


COMPARE = {"q": compare_q, "cache": compare_cache, "insert-ref": compare_insert_ref, "iq": compare_iq,
           "indexer-ref": compare_indexer_ref}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    try:
        COMPARE[job["kind"]](job, rec)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "error": _error(e)}
    return {"phase": job["phase"], "case": job["case"], **rec}


# ----------------------------------------------------------------------------------------------------------------- the GPU side
def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection() -> dict[str, Any]:
    """vLLM's DSV4 attention class and its cache layout on this device, the padded heads, the insert op, the indexer's FP8 CuTe
    dispatch and the sources."""
    import torch
    import vllm
    from vllm.config import get_current_vllm_config
    from vllm.models.deepseek_v4 import attention as A
    from vllm.models.deepseek_v4.common.ops import fused_indexer_q as FQ
    from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls
    from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4

    import vllm._C_stable_libtorch  # noqa: F401  -- registers torch.ops._C's stable ops

    cfg = get_current_vllm_config()
    cls = _select_dsv4_attn_cls(cfg)
    root = Path(vllm.__file__).parent.parent
    sha = {p: _sha256(str(root / p)) if (root / p).exists() else None for p in SOURCES}
    rec = {"attn_cls": cls.__name__, "fp8_ds_mla": bool(getattr(cls, "use_fp8_ds_mla_layout", None)),
           "cache_dtype": [str(v) for v in A._resolve_dsv4_kv_cache_dtype(True, "auto", None)],
           "padded_heads": int(DeepseekV4FlashMLAAttention.get_padded_num_q_heads(NH)),
           "op": hasattr(torch.ops._C, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"), "has_cutedsl": bool(FQ.has_cutedsl()),
           "fp8_dtype": str(FQ.current_platform.fp8_dtype()), "indexer_fp4": bool(dsa_indexer_uses_fp4(cfg)), "sources": sha}
    rec["checks"] = {"attn_cls": cls is DeepseekV4FlashMLAAttention, "fp8_ds_mla": rec["fp8_ds_mla"],
                     "cache_dtype": rec["cache_dtype"] == ["fp8_ds_mla", "torch.uint8"], "padded_heads": rec["padded_heads"] == NHP,
                     "op": rec["op"], "has_cutedsl": rec["has_cutedsl"], "fp8_dtype": rec["fp8_dtype"] == "torch.float8_e4m3fn",
                     "indexer_fp8": not rec["indexer_fp4"], "sources": sha == SOURCES}
    rec["ok"] = all(rec["checks"].values())
    return rec


def stack() -> dict[str, Any]:
    """What the kernels of record came from: torch, vLLM, the CuTe DSL, `_C_stable_libtorch` and its SASS architectures."""
    import torch
    import vllm

    import vllm._C_stable_libtorch as CS

    rec: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda, "vllm": getattr(vllm, "__version__", None),
                           "stable_C": getattr(CS, "__file__", None)}
    try:
        import cutlass
        from vllm.vllm_flash_attn.cute import utils as FU

        rec["cutlass_dsl"] = getattr(cutlass, "__version__", None)
        rec["flash_attn_cute_utils"] = {"file": FU.__file__, "sha256": _sha256(FU.__file__)}
    except Exception as e:  # noqa: BLE001
        rec["cutlass_dsl"] = {"error": _error(e)}
    try:
        rec["archs"] = C._archs(rec["stable_C"])
    except Exception as e:  # noqa: BLE001
        rec["archs"] = {"error": _error(e)}
    return rec


class Spy:
    """Records the positional arguments of each call of ``owner.name`` (a class's static method when ``static``) and calls it
    through; restores the attribute on exit."""

    def __init__(self, owner, name: str, static: bool = False):
        self.owner, self.name, self.static, self.calls = owner, name, static, []

    def __enter__(self) -> "Spy":
        self.raw = vars(self.owner)[self.name] if isinstance(self.owner, type) else getattr(self.owner, self.name)
        fn = getattr(self.owner, self.name)

        def spy(*a, **k):
            self.calls.append(a)
            return fn(*a, **k)

        setattr(self.owner, self.name, staticmethod(spy) if self.static else spy)
        return self

    def __exit__(self, *exc) -> None:
        setattr(self.owner, self.name, self.raw)


def _dev(a: np.ndarray, dtype=None):
    import torch

    t = torch.from_numpy(np.ascontiguousarray(a)).cuda()
    return t if dtype is None else t.view(dtype)


def launch_sumsq_probe() -> dict[str, Any]:
    """Run the non-served-eps query head and compare every word with the fused Definition and the unfused mutant."""
    import torch
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    import vllm._C_stable_libtorch  # noqa: F401

    o = sumsq_probe_operands()
    q, kv = _dev(o["q"].view(np.int16), torch.bfloat16), _dev(o["kv"].view(np.int16), torch.bfloat16)
    cs, pos, slots = _dev(o["cs"].view(np.float32)), _dev(o["positions"]), _dev(o["slots"])
    cache = _dev(np.zeros((1, M.page_bytes(PBS)), dtype=np.uint8))
    op = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    out = op(q, kv, cache, slots, pos, cs, SUMSQ_PROBE_PADDED_HEADS, SUMSQ_PROBE_EPS, PBS)
    torch.cuda.synchronize()
    got = _u16(out)[0, 0]
    fn = bind(R.Dsv4QNormRope, D=D, DR=DR, EPS=SUMSQ_PROBE_EPS)
    fused = np.asarray(_flat(evaluate(fn, o["q"][0, 0].tolist(), o["cs"][0].tolist())), dtype=np.uint16)
    fused_model = sumsq_probe_reference(o["q"][0, 0], o["cs"][0], fused=True)
    unfused = sumsq_probe_reference(o["q"][0, 0], o["cs"][0], fused=False)
    fused_bad = np.flatnonzero(got != fused)
    unfused_bad = np.flatnonzero(got != unfused)
    forms_bad = np.flatnonzero(fused != unfused)
    first = int(forms_bad[0]) if forms_bad.size else None
    checks = {"shape": tuple(out.shape) == (1, SUMSQ_PROBE_PADDED_HEADS, D) and out.dtype == torch.bfloat16,
              "pads": not bool(_u16(out)[0, 1:].any()), "inputs": bool(np.array_equal(_u16(q), o["q"])),
              "fused_model": bool(np.array_equal(fused_model, fused)), "kernel_is_fused": fused_bad.size == 0,
              "forms_separate": forms_bad.size >= 3, "kernel_is_not_unfused": unfused_bad.size > 0}
    return {"case": "insert/sumsq-probe", "eps": SUMSQ_PROBE_EPS, "served_statics": False,
            "scope": "diagnostic non-served eps; excluded from served case totals",
            "input_lane": list(SUMSQ_PROBE_LANE), "kernel_fused_mismatch": int(fused_bad.size),
            "kernel_unfused_mismatch": int(unfused_bad.size), "fused_unfused_difference": int(forms_bad.size),
            "first_difference": None if first is None else
                {"word": first, "kernel": int(got[first]), "fused": int(fused[first]), "unfused": int(unfused[first])},
            "checks": checks, "ok": all(checks.values())}


def launch_insert(work: Path, case: str, quick: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One insert case, twice on caches of two fills: its launch record and comparison jobs."""
    import torch

    import vllm._C_stable_libtorch  # noqa: F401

    o = insert_operands(case)
    T_ = o["q"].shape[0]
    q, kv = _dev(o["q"].view(np.int16), torch.bfloat16), _dev(o["kv"].view(np.int16), torch.bfloat16)
    cs, pos, slots = _dev(cos_sin_cache().view(np.float32)), _dev(o["positions"]), _dev(o["slots"])
    op = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert

    def run(fill: np.ndarray):
        cache = _dev(fill)
        return op(q, kv, cache, slots, pos, cs, NHP, EPS, PBS), cache

    out1, c1 = run(o["fills"][0])
    out2, c2 = run(o["fills"][1])
    torch.cuda.synchronize()
    g1, g2 = c1.cpu().numpy(), c2.cpu().numpy()
    cc = cache_checks(o["slots"], o["fills"], g1, g2)
    rec = {"phase": "insert", "case": case, "b": T_, "inserted": int(o["slots"].size), "skipped": int((o["slots"] < 0).sum()),
           "blocks": int(g1.shape[0]), "grid": "warp" if T_ < INSERT_CUTOFF else "reduced",
           "checks": {"shape": tuple(out1.shape) == (T_, NHP, D) and out1.dtype == torch.bfloat16,
                      "rerun": bool(torch.equal(out1.view(torch.int16), out2.view(torch.int16))) and cc["rerun"],
                      "untouched": cc["untouched"],
                      "inputs": bool(np.array_equal(_u16(q), o["q"]) and np.array_equal(_u16(kv), o["kv"]))}}
    paths = _save(work / f"insert-{case}", q=o["q"], kv=o["kv"], cs=o["cs"], slots=o["slots"], fills=o["fills"][:1],
                  q_out=_u16(out1), cache=g1)
    return rec, insert_jobs(case, paths, T_, quick)


def launch_indexer(work: Path, case: str, quick: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One indexer case, twice: its launch record and comparison jobs."""
    import torch
    from cutlass import Float32
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import fused_indexer_q_rope_quant
    from vllm.models.deepseek_v4.nvidia.ops import fused_indexer_q_cutedsl as K

    o = indexer_operands(case)
    T_ = o["q"].shape[0]
    q, w = _dev(o["q"].view(np.int16), torch.bfloat16), _dev(o["w"].view(np.int16), torch.bfloat16)
    cs, pos = _dev(cos_sin_cache().view(np.float32)), _dev(o["positions"])
    with Spy(K, "fused_indexer_q_rope_quant_fp8_cutedsl") as route, Spy(K.IndexerQFp8Kernel, "compile", static=True) as comp:
        q1, w1 = fused_indexer_q_rope_quant(pos, q, cs, w, SOFTMAX_SCALE, HEAD_SCALE, use_fp4=False)
        q2, w2 = fused_indexer_q_rope_quant(pos, q, cs, w, SOFTMAX_SCALE, HEAD_SCALE, use_fp4=False)
    torch.cuda.synchronize()
    coarsen = [int(c[4]) for c in comp.calls]
    want = 1 if T_ < COARSEN_CUTOFF else 4
    rec = {"phase": "indexer", "case": case, "b": T_, "route_calls": len(route.calls), "coarsen_compiled": coarsen,
           "checks": {"shape": tuple(q1.shape) == (T_, IH, ID) and q1.dtype == torch.float8_e4m3fn and tuple(w1.shape) == (T_, IH)
                      and w1.dtype == torch.float32,
                      "rerun": bool(torch.equal(q1.view(torch.uint8), q2.view(torch.uint8))
                                    and torch.equal(w1.view(torch.int32), w2.view(torch.int32))),
                      "route": len(route.calls) == 2,
                      "coarsen": len(coarsen) == 6 and coarsen[2::3] == [want, want] and all(c[3] is Float32 for c in comp.calls),
                      "inputs": bool(np.array_equal(_u16(q), o["q"]) and np.array_equal(_u16(w), o["w"]))}}
    paths = _save(work / f"indexer-{case}", q=o["q"], w=o["w"], cs=o["cs"], q_fp8=q1.view(torch.uint8).cpu().numpy(),
                  w_out=w1.view(torch.int32).cpu().numpy().view(np.uint32))
    return rec, indexer_jobs(case, paths, T_, quick)


def bind_check() -> dict[str, Any]:
    """The Programs bind at DeepSeek-V4-Pro's statics, and the seams: the insert's row is `FlashMlaKvRow_v1{448, 64, 64}`'s
    input, the indexer's outputs `DsaMqaLogit_v1{H=64, D=128}`'s q and w."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsa as Dsa
    from verity_vllm.program.registry import flashmla as F

    def row_seam() -> bool:
        out = bind(R.Dsv4QNormRopeKvInsert, **R.DSV4_INSERT).signature()[1].fields
        (_, nope), (_, scale), (_, rope), _ = bind(F.FlashMlaKvRow, DN=DN, DR=DR, QT=QB).signature()[0]
        return out[1] == nope and out[3] == rope and out[2].elem == scale.elem and out[2].n == scale.n + 1

    def logit_seam() -> bool:
        (_, qt), (_, wt), *_ = bind(Dsa.DsaMqaLogit, H=IH, D=ID).signature()[0]
        return tuple(bind(R.Dsv4IndexerQ, **R.DSV4_INDEXER).signature()[1].fields) == (qt, wt)

    targets = {
        "Dsv4QNormRopeKvInsert_v1{NH=16,NHP=64,DN=448,DR=64,QB=64,EPS=1e-06}":
            lambda: bool(bind(R.Dsv4QNormRopeKvInsert, **R.DSV4_INSERT).signature()),
        "Dsv4QNormRope_v1{D=512,DR=64,EPS=1e-06}": lambda: bool(bind(R.Dsv4QNormRope, D=D, DR=DR, EPS=EPS).signature()),
        "Dsv4KvRopeQuant_v1{DN=448,DR=64,QB=64}": lambda: bool(bind(R.Dsv4KvRopeQuant, DN=DN, DR=DR, QB=QB).signature()),
        "Dsv4IndexerQ_v1{H=64,D=128,DR=64,SCALE=...}": lambda: bool(bind(R.Dsv4IndexerQ, **R.DSV4_INDEXER).signature()),
        "Dsv4IndexerQHead_v1{D=128,DR=64,SCALE=...}":
            lambda: bool(bind(R.Dsv4IndexerQHead, D=ID, DR=IDR, SCALE=ISCALE).signature()),
        "insert row -> FlashMlaKvRow_v1{DN=448,DR=64,QT=64}": row_seam,
        "indexer q, w -> DsaMqaLogit_v1{H=64,D=128}": logit_seam,
    }
    out = {}
    for name, check in targets.items():
        try:
            out[name] = check()
        except Exception as e:  # noqa: BLE001
            out[name] = _error(e)
    return {"bindings": out, "ok": all(v is True for v in out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
def gpu(work: Path, outdir: Path, quick: bool, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The selection, the stack and every case's launches in order; each error goes to doc["errors"].  Returns the jobs."""
    import torch

    jobs: list[dict[str, Any]] = []

    def step(name: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            print(f"[dsv4-qnorm-insert] {name} ERROR {doc['errors'][name][:600]}", flush=True)
            return None

    doc["selection"] = step("selection", selection) or {"ok": False}
    print(f"[dsv4-qnorm-insert] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
    doc["stack"] = step("stack", stack) or {}
    for phase, case in cases_of(quick):
        launch = launch_insert if phase == "insert" else launch_indexer
        r = step(f"{phase}/{case}", lambda launch=launch, case=case: launch(work, case, quick))
        if r is not None:
            doc["launches"].append(r[0])
            jobs += r[1]
            print(f"[dsv4-qnorm-insert] {phase:7s} {case:7s} b={r[0]['b']} {json.dumps(r[0]['checks'])}", flush=True)
    doc["sumsq_probe"] = step("insert/sumsq-probe", launch_sumsq_probe) or {"case": "insert/sumsq-probe", "ok": False}
    p = doc["sumsq_probe"]
    print(f"[dsv4-qnorm-insert] probe insert/sumsq-probe {'GO' if p['ok'] else 'NO-GO'} eps={p.get('eps')} "
          f"kernel!=fused={p.get('kernel_fused_mismatch')} kernel!=unfused={p.get('kernel_unfused_mismatch')} "
          f"fused!=unfused={p.get('fused_unfused_difference')}", flush=True)
    torch.cuda.empty_cache()
    return jobs


def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["checks"]) and all(v is True for v in x["checks"].values())


def _words_ok(t: dict[str, Any]) -> bool:
    return t["words"] > 0 and t["mismatch"] == 0 and t["reference"] > 0 and t["twin_mismatch"] == 0


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's verdict: its launch checks, its words and its errors, so a NO-GO names what failed."""
    out: dict[str, dict[str, Any]] = {}
    for phase, case in cases_of(bool(doc.get("quick"))):
        key = f"{phase}/{case}"
        launches = [x for x in doc["launches"] if x["phase"] == phase and x["case"] == case]
        comps = [r for r in doc.get("comparisons", []) if r["phase"] == phase and r["case"] == case]
        t = {k: sum(r[k] for r in comps) for k in TOTALS}
        errors = [e for e in doc["errors"] if e == key or e == f"compare/{key}"]
        ok = bool(launches) and all(_launch_ok(x) for x in launches)
        out[key] = {"launches_ok": ok, "totals": t, "errors": errors, "ok": ok and _words_ok(t) and not errors}
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    return bool(doc["gate"]["ok"] and not doc["errors"] and doc["selection"].get("ok") is True and doc["bind"].get("ok") is True
                and doc.get("sumsq_probe", {}).get("ok") is True and all(c["ok"] for c in cases(doc).values())
                and _words_ok(doc["totals"]))


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    import torch  # noqa: F401
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "quick": quick, "environment": G._environment(),
                           "gate": C.gate(min_driver, want_cc, want_sms), "errors": {}, "launches": []}
    print(f"[dsv4-qnorm-insert] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-QNORM-INSERT GATE-FAIL", flush=True)
        raise SystemExit(3)
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        jobs = gpu(work, outdir, quick, doc)
    doc.setdefault("selection", {"ok": False})
    try:
        doc["bind"] = bind_check()
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    print(f"[dsv4-qnorm-insert] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
    results: dict[tuple, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(compare, jobs):
            if "error" in res:
                doc["errors"].setdefault(f"compare/{res['phase']}/{res['case']}", res["error"])
                continue
            agg = results.setdefault((res["phase"], res["case"]), {"phase": res["phase"], "case": res["case"],
                                                                  **{k: 0 for k in TOTALS}, "first_mismatch": None})
            for k in TOTALS:
                agg[k] += res[k]
            agg["first_mismatch"] = agg["first_mismatch"] or res["first_mismatch"]
    doc["comparisons"] = list(results.values())
    for r in doc["comparisons"]:
        print(f"[dsv4-qnorm-insert] words {r['phase']:7s} {r['case']:7s} words={r['words']} !={r['mismatch']} "
              f"ref={r['reference']} twin!={r['twin_mismatch']} first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[dsv4-qnorm-insert] case {key:15s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} "
              f"words={t['words']} !={t['mismatch']} ref={t['reference']} twin!={t['twin_mismatch']} errors={c['errors']}",
              flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"DSV4-QNORM-INSERT {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro's q-norm/RoPE/SWA insert and indexer q quant on this Hopper GPU "
                                             "against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (the cases' operands and outputs)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true",
                    help=f"the cases {QUICK_INSERT} and {QUICK_INDEXER}, the reference on one token each: a first signal in "
                         "minutes")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
