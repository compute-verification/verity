"""DeepSeek-V4-Pro's DSA indexer on Hopper against its Programs: the H200's go/no-go for binding `dsv4_indexer`'s Definitions (the
compressor's state score, its fused compress / RMSNorm / RoPE / ue8m0 quantisation / cache insert, the inverse-RoPE FP8 quantisation of
the attention output and the short-context fill), `dsa.DsaMqaLogit_v3{64,128}` (DeepGEMM `fp8_fp4_mqa_logits` and
`fp8_fp4_paged_mqa_logits` on the compressed FP8 indexer cache) and `dsa.DsaTopKShortRow_v1{1024}` (vLLM's `top_k_per_row_prefill`,
`cooperative_topk` and `persistent_topk`) at DeepSeek-V4-Pro's statics on one TP8 rank: a ratio-4 layer's indexer (64 heads of 128,
replicated, top 1024 of the compressed keys) and the o-projection's input (16 local heads of 512 in 2 groups of 8).  vLLM d9105ea80;
line numbers under `vllm/models/deepseek_v4/` unless named.

One GPU process writes `<out>/dsv4_indexer_capture.json`, the property record `dsv4_indexer_capture` (`properties.record`), and the
four Triton kernels' PTX (and SASS where Triton's bundled `cuobjdump` runs) under `<out>/ptx/`; it prints one line per phase, then
`DSV4-INDEXER GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first and nothing else runs when it fails (exit 3).
The Triton cache and DeepGEMM's JIT cache are in the scratch directory.  A phase that raises records its error and fails the verdict
without stopping the others.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  the FP8 indexer cache (`dsa_indexer_uses_fp4` False: MXFP4 needs sm_10x), no two-stage compressor
             (`_prefer_two_stage_compressor`, ROCm only), so the head-128 compressor is `compress_norm_rope_store_triton`'s
             `_fused_kv_compress_norm_rope_insert_indexer_attn` (compressor.py:420-423); f32 o-projection scales (`tma_aligned_scales`
             False, nvidia/ops/o_proj.py:24); the installed SOURCES the ones read (`sources`), since the phases below call the
             kernels' wrappers directly rather than through `DeepseekCompressor.forward`'s dispatch.
  score      `save_partial_states` as `DeepseekCompressor.forward` calls it (compressor.py:354-365, block size 4, state width 256), twice
             on fresh state caches, over three requests in one launch: SEQ tokens at positions 0..SEQ-1, EDGE_SEQ tokens from EDGE_START
             (its first state block holds four positions written before; the edge operands, `EDGE`) and EXTRA boundary tokens (half
             without a state slot): each written state row's score half against `Dsv4CompressorScore_v1{256}` of its kv_score row and
             `ape[pos % 4]`, its kv half a copy, every other word of the cache untouched.
  compress   `compress_norm_rope_store_triton` (compressor.py:430-452: head 128, rope 64, ratio 4 with overlap, quant block 128),
             twice, on the state the score launch wrote: every written indexer-cache entry (128 e4m3 bytes and the f32 scale) against
             `Dsv4IndexerCompress_v1{128,64,1e-06}` of its 8 state rows as the kernel reads them (`state_rows`), k_norm's bf16 weight
             and the rotary cache row (pos // 4) * 4; the cases `served` (request 0) and `edge` (request 1); the state unchanged, every
             byte of an unwritten entry untouched (the extra tokens without a cache slot write none).
  fill       `_fill_short_context_topk_indices` as attention.py:1005-1012 launches it (TOP_K 1024, ratio 4, 8 warps), twice, on
             FILL_POSITIONS and random int64 positions: each row against `Dsv4ShortContextTopK_v1{1024,4}`.
  inv_rope   `fused_inv_rope_fp8_quant` as `deep_gemm_fp8_o_proj` calls it (nvidia/ops/o_proj.py:48-57) on the first 16 heads of a
             64-head FlashMLA output buffer (attention.py:429-456), twice, over INV_TOKENS tokens (special heads and rotary rows): each
             (token, head) against `Dsv4InvRopeQuantHead_v1{4,128,64}` (its bytes [t, g, 512 i:512 (i + 1)] and scales [t, g, 4 i:4 i + 4]
             for head 8 g + i), the TMA padding rows' scales zero.
  logits     the prefill gather (`cp_gather_indexer_k_quant_cache`) of request 0's SEQ // 4 entries against the cache; `fp8_fp4_mqa_logits`
             (ks = 0, ke = (p + 1) // 4, the compressed keys before each token, clean_logits=False, sparse_attn_indexer.py:500-507) of
             e4m3 queries and f32 weights (the scale folded in, as `fused_indexer_q_rope_quant` makes them) on rows FULL_ROWS, random
             rows and random (row, key) pairs, and `fp8_fp4_paged_mqa_logits` (one context per DECODE_LENS entry, its query the boundary
             token's, sparse_attn_indexer.py:603-613) on every key: each logit against `DsaMqaLogit_v3{64,128}` (twin
             `glm53_dsa_capture_gpu.logit_rows`; the reference on a sample).
  topk       `top_k_per_row_prefill` on the served logits (twice) and on synthetic rows from a row start of 5; on the decode logits the
             kernel vLLM picks (sparse_attn_indexer.py:617-664: `cooperative_topk` on sm_90 for at most 64 rows); `persistent_topk`,
             `cooperative_topk` (10 and 8 rows) and the `top_k_per_row_decode` fallback (recorded) on synthetic rows of TOPK_LENS keys,
             tie rows and a negative length: every row of at most 1024 keys against `DsaTopKShortRow_v1{1024}` (closed form
             `glm53_dsa_capture_gpu.topk_short`, the reference on its lengths); a longer row a valid top-1024 set (recorded: no
             Definition states its order or ties, GLM-5.3's long-row work).
  ptx        the Triton cache holds one specialization of each of KERNELS: their PTX instruction counts, copied to `<out>/ptx/`.
  bind       the Programs bind at the served statics.

GO needs every phase's words compared with 0 mismatches and 0 unresolved (a non-finite logit accumulator; these operands keep every
logit finite), the reference with 0 disagreements wherever it ran (every instance of the score, compress, fill and inv_rope cases
unless --quick, a sample of the logits), deterministic reruns (the prefill logits on the words they write, `logits_rerun`), and the
checks above.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/dsv4_indexer_capture_gpu.py --out DIR [--scratch DIR] [--quick]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from tests.properties import glm53_dsa_capture_gpu as X
from verity_vllm.program.registry import dsv4_indexer as I
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/dsv4-indexer-capture/v1"
RECORD = "dsv4_indexer_capture.json"
#: DeepSeek-V4-Pro (deepseek-ai/DeepSeek-V4-Pro config.json) at TP8: the indexer's 64 heads of 128 with a 64-wide GPT-J RoPE, top 1024
#: of the ratio-4 compressed keys, its cache entry 128 e4m3 bytes and an f32 scale at BLOCK entries per block (GLM-5.3's layout)
H, D, ROPE, TOPK, RATIO, EPS = 64, 128, 64, 1024, 4, 1e-6
BLOCK, STRIDE = X.BLOCK, X.STRIDE
#: the compressor's state: 8 rows (overlap) of [kv | score], STATE_W f32 words each, STATE_BLOCK positions per state block
ROWS, STATE_W, STATE_BLOCK = I.ROWS, 2 * D, 4
#: the o-projection's input: 16 local heads of 512 (448 nope, 64 rope) in 2 groups of 8, quantised per 128; FlashMLA pads to 64 heads
NH, HD, NOPE, QG, GROUPS, PADDED_HEADS = 16, 512, 448, 128, 2, 64
CHUNKS, HPG = HD // QG, NH // GROUPS
MAXPOS = 8192
#: request 0: SEQ tokens at positions 0..SEQ-1 (SEQ // 4 entries, more than TOPK); request 1: EDGE_SEQ tokens from EDGE_START; request
#: 2: EXTRA boundary tokens, the first half without a state slot, the second without a cache slot.  NBS state blocks, NBK cache blocks
SEQ, EDGE_START, EDGE_SEQ, EXTRA = 4608, 6144, 256, 8
EXTRA_POSITIONS = (3, 7, 11, 15, 19, 23, 27, 31)
NBS, NBK = 1280, 24
#: the indexer's max_model_len is the engine's divided by the ratio (attention.py:918-920): 8192 tokens here
MAX_MODEL_LEN = 2048
#: request 1's edge operands by token offset: (what, name, offsets).  A name is the value of each word of those tokens' kv or score
#: rows, `xF` the rows scaled by F, `one_hot` -1e30 but 0 at the fourth token, `V@j` the value V at column j of the first token
EDGE = (("score", "zero", range(32, 40)), ("score", "inf", (50,)), ("score", "nan", (60,)), ("score", "neg_inf", range(64, 72)),
        ("kv", "zero", range(80, 88)), ("kv", "x2^70", range(88, 96)), ("kv", "x2^-140", range(96, 104)),
        ("kv", "nan@3", (110,)), ("kv", "inf@5", (111,)), ("kv", "neg_inf@7", (112,)), ("score", "x1e4", range(120, 128)),
        ("score", "one_hot", range(128, 136)), ("score", "inf", range(140, 148)), ("kv", "3e38@9", (150,)),
        ("kv", "x1e-30", range(160, 168)))
EDGE_VALUES = {"zero": 0.0, "inf": float("inf"), "nan": float("nan"), "neg_inf": float("-inf"), "3e38": 3e38}
EDGE_FACTORS = {"x2^70": 2.0 ** 70, "x2^-140": 2.0 ** -140, "x1e4": 1e4, "x1e-30": 1e-30}
#: the compressor's rotary base (config.json `compress_rope_theta`; the Definitions take the cache's words, so YaRN is left out)
THETA = 160000
#: rotary rows with special words: cos[2] inf, sin[5] NaN, every cos 0 and sin -1 (request 1's entries at p = row + 3 read them)
CS_SPECIAL = (EDGE_START + 8, EDGE_START + 16, EDGE_START + 24)
FULL_ROWS = (3, 4, 7, 8, 255, 256, 1023, 4095, 4096, 4099, 4100, SEQ - 1)
RANDOM_ROWS, RANDOM_PAIRS = 8, 50_000
DECODE_LENS = (1, 2, 63, 64, 65, 1023, 1024, 1025, SEQ // RATIO)
TOPK_LENS = (0, 1, 2, 1023, 1024, 1025, 2048)
#: not a multiple of 4: the TMA-aligned scale buffer has padding rows
INV_TOKENS = 510
FILL_POSITIONS = (*range(9), 11, 12, *range(4094, 4101), 4103, 1 << 20, 2 ** 31 - 1, 2 ** 31, 2 ** 32 + 3, 2 ** 62, 2 ** 63 - 2,
                  2 ** 63 - 1, -1, -2, -3, -4, -5, -8, -2 ** 31, -2 ** 63)
FILL_RANDOM = 32
#: instances per job: score rows, cache entries, (token, head)s, fill rows, logits entries
JOB_ROWS, JOB_ENTRIES, JOB_HEADS, JOB_FILL, JOB_LOGITS = 256, 32, 128, 64, 4096
QUICK_REF_EVERY = 16
TOTALS = X.TOTALS
SENTINEL_STATE, SENTINEL_K, SENTINEL_HEAD = 0x7FA5A5A5, 0xA5, 0x5A5A
KERNELS = ("_save_partial_states_kernel", "_fused_kv_compress_norm_rope_insert_indexer_attn", "_fused_inv_rope_fp8_quant_per_head",
           "_fill_short_context_topk_indices")
PTX_COUNT = re.compile(r"\b(fma\.rn(?:\.ftz)?\.f32|mul(?:\.rn)?(?:\.ftz)?\.f32|sub(?:\.rn)?(?:\.ftz)?\.f32|add(?:\.rn)?(?:\.ftz)?\.f32|"
                       r"neg(?:\.ftz)?\.f32|div\.(?:rn|full)(?:\.ftz)?\.f32|rsqrt\.approx(?:\.ftz)?\.f32|ex2\.approx(?:\.ftz)?\.f32|"
                       r"lg2\.approx(?:\.ftz)?\.f32|cvt\.rpi(?:\.ftz)?\.f32\.f32|cvt\.rn\.bf16\.f32|cvt\.rn\.satfinite\.e4m3x2\.f32|"
                       r"(?:max|min)(?:\.ftz)?(?:\.xorsign\.abs)?\.f32)(?=[\s;])")

#: the installed vLLM's files the Definitions and this script were read from, by sha256 at d9105ea80.  They pin the dispatch this
#: script calls past: `DeepseekCompressor.forward` sends head 512 to cuteDSL and the indexer's head 128 to
#: `compress_norm_rope_store_triton` (compressor.py:402-427; its class docstring still says the head-128 path is cuteDSL)
SOURCES = {"models/deepseek_v4/compressor.py": "338b89ec959f1b40f7210cbee01a8dce12afa1259195137feb1fb806011f61f7",
           "models/deepseek_v4/common/ops/fused_compress_quant_cache.py":
               "6b1f4c6b4861f9645c296cf990b879329042a9aa7fa0214eabd9e58f8ebc7987",
           "models/deepseek_v4/common/ops/save_partial_states.py": "27f5a42c352339627004cc767f33e113710c095a585c4b40ddbfda3a335d9363",
           "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py":
               "afc182aab04103430d220d5e146da5389f98ec1a814c1477e27f700cc605de59",
           "models/deepseek_v4/attention.py": "b6b32d5a98737ba2d7e619b66de0b055f6189f68d513d9dc1e11f05aaa362616",
           "models/deepseek_v4/nvidia/ops/o_proj.py": "ffb28e7cc44124bb2878e596617b70ec2659c7b26a2048a06facc343fa4d24e1",
           "model_executor/layers/sparse_attn_indexer.py": "470f78842e70f62fc089b0f1429683dcf1d6d07814e2ce8045d42e2841b9dde6"}

NAN = np.uint32(0x7FFFFFFF)
NEG_INF = np.uint32(0xFF800000)
M64 = (1 << 64) - 1
_error = X._error
_u8, _u16, _u32, _save, _load, _chunks = X._u8, X._u16, X._u32, X._save, X._load, X._chunks


def environment(work: Path) -> dict[str, str]:
    """The caches in scratch; set before vLLM is imported."""
    env = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TRITON_CACHE_DIR": str(work / "triton_cache"), "DG_JIT_CACHE_DIR": str(work / "deep_gemm")}
    os.environ.update(env)
    return env


def stack() -> dict[str, Any]:
    """`glm53_dsa_capture_gpu.stack` and whether cuteDSL is installed (then `fused_indexer_q_rope_quant` runs its cuteDSL kernel)."""
    from vllm.utils.import_utils import has_cutedsl

    return {**X.stack(), "has_cutedsl": bool(has_cutedsl())}


# ------------------------------------------------------------------------------------------------------------------- the twins
def _f(u) -> np.ndarray:
    return np.asarray(u, dtype=np.uint32).view(np.float32)


def _w(f) -> np.ndarray:
    return np.asarray(f, dtype=np.float32).view(np.uint32)


def _is_nan(u) -> np.ndarray:
    u = np.asarray(u, dtype=np.uint32)
    return ((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)


def _gpu(u) -> np.ndarray:
    """Every NaN the GPU's 0x7FFFFFFF."""
    u = np.asarray(u, dtype=np.uint32)
    return np.where(_is_nan(u), NAN, u)


class NumpyOps:
    """`dsv4_indexer.WORD`'s steps on numpy arrays of words, so the Definitions' bodies (`compress_column`, `norm_rope`, `k_quant`,
    `inv_rope`, `inv_rope_quant`) run vectorised over many instances: the f32 arithmetic of `F32AddV2_v1` .. `F32FmaFtzV2_v1` (numpy's
    IEEE single rounding, `derived_rows.fma_rn`'s exact FMA, each NaN 0x7FFFFFFF), MUFU and div.full through core's registered kernels
    (`verity.ml.circuits.ops.mufu`, exact on every encoding), the rest on the encodings."""

    @staticmethod
    def word(bits: int):
        return np.uint32(bits)

    @staticmethod
    def neg(x):
        return np.asarray(x, dtype=np.uint32) ^ np.uint32(0x80000000)

    @staticmethod
    def add(a, b):
        return _gpu(_w(_f(a) + _f(b)))

    @staticmethod
    def sub(a, b):
        return _gpu(_w(_f(a) - _f(b)))

    @staticmethod
    def mul(a, b):
        return _gpu(_w(_f(a) * _f(b)))

    @staticmethod
    def fma(a, b, c):
        from verity_vllm.program.kernels import derived_rows as DR

        return _gpu(DR.fma_rn(a, b, c))

    @staticmethod
    def fma_ftz(a, b, c):
        from verity_vllm.program.kernels import derived_rows as DR

        return _gpu(DR.fma_ftz(a, b, c))

    @staticmethod
    def rcp(b):
        from verity.ml.circuits.ops import mufu as MU

        return MU.mufu_rcp_words(MU.div_full_divisor_words(np.atleast_1d(np.asarray(b, dtype=np.uint32))), MU.table("rcp"))

    @staticmethod
    def scale_a(a, b):
        from verity.ml.circuits.ops import mufu as MU

        return MU.div_full_scale_a_words(*np.broadcast_arrays(np.atleast_1d(np.asarray(a, np.uint32)), np.atleast_1d(np.asarray(b, np.uint32))))

    @staticmethod
    def rsqrt(x):
        from verity.ml.circuits.ops import mufu as MU

        return MU.rsqrt_approx_words(np.atleast_1d(np.asarray(x, dtype=np.uint32)), MU.table("rsq"))

    @staticmethod
    def ex2(x):
        from verity.ml.circuits.ops import mufu as MU

        return MU.mufu_ex2_words(np.atleast_1d(np.asarray(x, dtype=np.uint32)), MU.table("ex2"))

    @staticmethod
    def fabs(x):
        return np.asarray(x, dtype=np.uint32) & np.uint32(0x7FFFFFFF)

    @staticmethod
    def _maxmin(a, b, larger: bool):
        """`max.f32` / `min.f32`: a NaN yields the other operand (two give 0x7FFFFFFF), max(+0, -0) = +0 and min(+0, -0) = -0."""
        a, b = np.broadcast_arrays(np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32))
        fa, fb = _f(a), _f(b)
        out = np.where(fa > fb if larger else fa < fb, a, b)
        out = np.where(fa == fb, a & b if larger else a | b, out)
        na, nb = _is_nan(a), _is_nan(b)
        return np.where(na & nb, NAN, np.where(na, b, np.where(nb, a, out)))

    def fmaxf(self, a, b):
        return self._maxmin(a, b, True)

    def fminf(self, a, b):
        return self._maxmin(a, b, False)

    @staticmethod
    def e4m3(x):
        from verity_vllm.program.kernels.fp8_twins import np_f32_to_e4m3_sat

        return np_f32_to_e4m3_sat(_f(x)).astype(np.uint32)

    @staticmethod
    def iadd(a, b):
        return ((np.asarray(a, dtype=np.uint64) + np.asarray(b, dtype=np.uint64)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    @staticmethod
    def iand(a, b):
        return np.asarray(a, dtype=np.uint32) & np.asarray(b, dtype=np.uint32)

    @staticmethod
    def isub(a, b):
        return ((np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)) & 0xFFFFFFFF).astype(np.uint32)

    @staticmethod
    def i2f(a):
        return _w(np.asarray(a, dtype=np.uint32).view(np.int32).astype(np.float32))

    @staticmethod
    def ceil(x):
        """`cvt.rpi.ftz.f32.f32`: a subnormal flushed to its signed zero, then rounded up (-1 < x < 0 gives -0), NaN 0x7FFFFFFF."""
        from verity_vllm.program.kernels import derived_rows as DR

        return _gpu(_w(np.ceil(_f(DR.ftz(x)))))

    @staticmethod
    def cvt(w):
        return np.asarray(w, dtype=np.uint32) << np.uint32(16)

    @staticmethod
    def bf16(x):
        from verity_vllm.program.kernels import derived_rows as DR

        return DR.f2fp_bf16(np.atleast_1d(np.asarray(x, dtype=np.uint32)))

    @staticmethod
    def fsel(c, a, b):
        return np.where(np.asarray(c, dtype=np.uint32) & 1, np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32))

    @staticmethod
    def gt(a, b):
        return (_f(a) > _f(b)).astype(np.uint32)

    @staticmethod
    def ile(a, b):
        return (np.asarray(a, dtype=np.uint32).view(np.int32) <= np.asarray(b, dtype=np.uint32).view(np.int32)).astype(np.uint32)


OPS = NumpyOps()
_NORM = SimpleNamespace(D=D, ROPE=ROPE, EPS=EPS)
_KQ, _GQ, _ROT = SimpleNamespace(N=D), SimpleNamespace(N=QG), SimpleNamespace(ROPE=ROPE)


def score_twin(score: np.ndarray, ape: np.ndarray) -> np.ndarray:
    """`Dsv4CompressorScore_v1{W}` over rows: score and ape (n, W) f32 words -> (n, W)."""
    with np.errstate(all="ignore"):
        return np.asarray(I.compressor_score(OPS, None, [score], [ape])[0], dtype=np.uint32)


def entries_twin(kv: np.ndarray, score: np.ndarray, w: np.ndarray, cs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`Dsv4IndexerCompress_v1{128,64,1e-06}` over n cache entries: kv and score (n, 8, 128) f32 words, w (128,) bf16 words, cs (n, 64)
    f32 words -> (n, 128) e4m3 bytes and (n,) f32 scales, through `dsv4_indexer`'s bodies."""
    kv, score, cs = (np.asarray(a, dtype=np.uint32) for a in (kv, score, cs))
    with np.errstate(all="ignore"):
        c = I.compress_column(OPS, None, [kv[:, r] for r in range(ROWS)], [score[:, r] for r in range(ROWS)])[0]
        v = I.norm_rope(OPS, _NORM, [c[:, j] for j in range(D)], [np.uint32(x) for x in np.asarray(w)], [cs[:, j] for j in range(ROPE)])
        out = I.k_quant(OPS, _KQ, v)
    n = kv.shape[0]
    return (np.stack([np.broadcast_to(x, (n,)) for x in out[:D]], axis=1).astype(np.uint8),
            np.broadcast_to(out[D], (n,)).astype(np.uint32))


def heads_twin(x: np.ndarray, cs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`Dsv4InvRopeQuantHead_v1{4,128,64}` over n heads: x (n, 512) bf16 words, cs (n, 64) f32 words -> (n, 512) e4m3 bytes and (n, 4)
    f32 scales."""
    x, cs = np.asarray(x, dtype=np.uint16), np.asarray(cs, dtype=np.uint32)
    n = x.shape[0]
    with np.errstate(all="ignore"):
        words = [OPS.cvt(x[:, i]) for i in range(HD - ROPE)]
        words += I.inv_rope(OPS, _ROT, [x[:, i] for i in range(HD - ROPE, HD)], [cs[:, j] for j in range(ROPE)])
        q, s = [], []
        for g in range(CHUNKS):
            out = I.inv_rope_quant(OPS, _GQ, words[g * QG:(g + 1) * QG])
            q += out[:QG]
            s.append(out[QG])
    return (np.stack([np.broadcast_to(v, (n,)) for v in q], axis=1).astype(np.uint8),
            np.stack([np.broadcast_to(v, (n,)) for v in s], axis=1).astype(np.uint32))


def fill_twin(pos) -> np.ndarray:
    """`_fill_short_context_topk_indices` restated: slot i of a row is i when i < (pos + 1) // 4 in int64 (wrapping, truncating), else
    -1 -> (n, TOPK) int32."""
    p = np.asarray(pos, dtype=np.int64)
    with np.errstate(over="ignore"):
        q = p + np.int64(1)
    num = q // RATIO + ((q % RATIO != 0) & (q < 0))
    j = np.arange(TOPK, dtype=np.int64)[None, :]
    return np.where(j < num[:, None], j, -1).astype(np.int32)


def state_rows(state: np.ndarray, block_table: np.ndarray, req: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 8 state rows the compress kernel reads for each boundary token (fused_compress_quant_cache.py:732-773): row r is position
    p - 7 + r of the token's request, at `block_table[req, q // 4]` offset q % 4 of a [blocks, 4, 512] f32 state; rows 0-3 its kv
    and score words [0:128] and [256:384], rows 4-7 [128:256] and [384:512]; a row before position 0 reads kv 0 and score -inf.
    -> kv, score (n, 8, 128) f32 words."""
    state = np.asarray(state, dtype=np.uint32)
    p = np.asarray(pos, dtype=np.int64)[:, None] - (ROWS - 1) + np.arange(ROWS)[None, :]
    valid = p >= 0
    pc = np.maximum(p, 0)
    rows = state[np.asarray(block_table)[np.asarray(req)[:, None], pc // STATE_BLOCK], pc % STATE_BLOCK]
    col = ((np.arange(ROWS) >= RATIO) * D)[None, :, None] + np.arange(D)[None, None, :]
    col = np.broadcast_to(col, (p.shape[0], ROWS, D))
    kv = np.where(valid[..., None], np.take_along_axis(rows, col, axis=2), np.uint32(0))
    sc = np.where(valid[..., None], np.take_along_axis(rows, col + STATE_W, axis=2), NEG_INF)
    return kv.astype(np.uint32), sc.astype(np.uint32)


def head_words(fp8: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`fused_inv_rope_fp8_quant`'s outputs (T, G, 8 * 512) bytes and (T, G, 8 * 4) f32 words as (T, 16, 512) and (T, 16, 4): head
    8 g + i is bytes [t, g, 512 i:512 (i + 1)] and scales [t, g, 4 i:4 i + 4]."""
    T = fp8.shape[0]
    return np.asarray(fp8).reshape(T, GROUPS * HPG, HD), np.asarray(scale).reshape(T, GROUPS * HPG, CHUNKS)


def topk_valid(vals: np.ndarray, n: int, idx: np.ndarray) -> dict[str, Any]:
    """`glm53_dsa_capture_gpu.topk_valid` at TOPK = 1024."""
    sel = idx[idx >= 0].astype(np.int64)
    ok = bool(sel.size == TOPK and np.unique(sel).size == sel.size and (sel < n).all())
    ties = None
    if ok:
        v = np.asarray(vals[:n], dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        mask[sel] = True
        thr = v[mask].min()
        ok = not bool((v[~mask] > thr).any())
        ties = int((v[~mask] == thr).sum())
    return {"valid": ok, "ties_left_out": ties, "ascending": bool(np.all(np.diff(sel) > 0))}


def check_topk(vals: np.ndarray, lens: np.ndarray, out: np.ndarray, start: int = 0) -> dict[str, Any]:
    """Rows of at most TOPK keys against `topk_short` (lengths clamped at 0); longer rows `topk_valid`."""
    lens = np.asarray(lens, dtype=np.int64)
    short = lens <= TOPK
    want = X.topk_short(np.maximum(lens[short], 0), TOPK)
    long_rows = [topk_valid(vals[i, start:], int(lens[i]), out[i]) for i in np.flatnonzero(~short)]
    return {"short_rows": int(short.sum()), "short_mismatch": int((want != out[short]).any(axis=1).sum()),
            "long_rows": len(long_rows), "long_valid": sum(r["valid"] for r in long_rows),
            "long_ascending": sum(r["ascending"] for r in long_rows), "long_with_ties": sum(bool(r["ties_left_out"]) for r in long_rows),
            "long": long_rows}


def logits_rerun(a, b, ke) -> tuple[bool, int]:
    """Two prefill logits runs (ks = 0, clean_logits=False): whether they agree on every word of each row's window [0, ke), the words
    the kernel writes, and how many words past the windows differ."""
    import torch

    window = torch.arange(a.shape[1], device=a.device)[None, :] < ke.to(a.device)[:, None]
    diff = a.view(torch.int32) != b.view(torch.int32)
    return not bool(diff[window].any()), int(diff[~window].sum())


# ------------------------------------------------------------------------------------------------------------------- operands
def layout(seed: int) -> dict[str, np.ndarray]:
    """The three requests' tokens: positions, request ids, the state block table and slots (block size 4), the cache block table
    and the compressed slots (`get_compressed_slot_mapping`: -1 off the boundaries), each block table shuffled."""
    rng = np.random.default_rng(seed)
    pos = np.concatenate([np.arange(SEQ), EDGE_START + np.arange(EDGE_SEQ), EXTRA_POSITIONS]).astype(np.int64)
    req = np.concatenate([np.zeros(SEQ), np.ones(EDGE_SEQ), np.full(EXTRA, 2)]).astype(np.int32)
    width = (EDGE_START + EDGE_SEQ) // STATE_BLOCK
    sbt = np.zeros((3, width), dtype=np.int32)
    perm = rng.permutation(NBS).astype(np.int32)
    n0, lo1 = SEQ // STATE_BLOCK, EDGE_START // STATE_BLOCK - 1
    sbt[0, :n0] = perm[:n0]
    sbt[1, lo1:] = perm[n0:n0 + width - lo1]
    n1 = n0 + width - lo1
    sbt[2, 3:8] = perm[n1:n1 + 5]
    sslot = sbt[req, pos // STATE_BLOCK].astype(np.int64) * STATE_BLOCK + pos % STATE_BLOCK
    sslot[SEQ + EDGE_SEQ:SEQ + EDGE_SEQ + EXTRA // 2] = -1
    kbt = np.zeros((3, width // BLOCK), dtype=np.int32)
    kperm = rng.permutation(NBK).astype(np.int32)
    kbt[0, :SEQ // RATIO // BLOCK] = kperm[:SEQ // RATIO // BLOCK]
    kbt[1, EDGE_START // RATIO // BLOCK] = kperm[SEQ // RATIO // BLOCK]
    e = pos // RATIO
    kslot = np.where((pos + 1) % RATIO == 0, kbt[req, e // BLOCK].astype(np.int64) * BLOCK + e % BLOCK, -1)
    kslot[req == 2] = -1
    prefilled = [(int(sbt[1, lo1]), r) for r in range(STATE_BLOCK)] + [(int(b), r) for b in sbt[2, 3:8] for r in range(STATE_BLOCK)]
    return {"pos": pos, "req": req, "sbt": sbt, "sslot": sslot, "kbt": kbt, "kslot": kslot, "prefilled": np.array(prefilled)}


def rotary_cache() -> np.ndarray:
    """An f32 [MAXPOS, 64] cache [cos 0..31 | sin 0..31] of GPT-J angles (THETA), and CS_SPECIAL's rows."""
    inv = float(THETA) ** (-np.arange(0, ROPE, 2) / ROPE)
    ang = np.arange(MAXPOS)[:, None] * inv[None, :]
    cs = np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
    a, b, c = CS_SPECIAL
    cs[a, 2] = np.inf
    cs[b, ROPE // 2 + 5] = np.nan
    cs[c, :ROPE // 2], cs[c, ROPE // 2:] = 0.0, -1.0
    return cs


def operands(seed: int, dev: str = "cuda") -> dict[str, Any]:
    """The compressor's inputs as `DeepseekCompressor.forward` has them (compressor.py:300-352): kv_score [T, 512] f32 (the
    `fused_wkv_wgate` output; kv | score), ape [4, 256] f32, k_norm's bf16 weight (w[3] = 0, w[100] = 64), the rotary cache; the
    initial state cache (every word SENTINEL_STATE but the prefilled rows: request 1's first block, request 2's blocks)."""
    import torch

    gen = torch.Generator(device=dev).manual_seed(seed)
    lay = layout(seed)
    T = lay["pos"].size

    def scaled(n, w, lo, hi):
        return torch.randn(n, w, generator=gen, device=dev) * torch.exp2(torch.empty(n, 1, device=dev).uniform_(lo, hi, generator=gen))

    kv_score = torch.empty(T, 2 * STATE_W, device=dev)
    kv_score[:, :STATE_W] = scaled(T, STATE_W, -3, 3)
    kv_score[:, STATE_W:] = 4 * torch.randn(T, STATE_W, generator=gen, device=dev)
    kv, sc = kv_score[:, :STATE_W], kv_score[:, STATE_W:]
    for what, name, offs in EDGE:
        t = [SEQ + k for k in offs]
        x = kv if what == "kv" else sc
        if name in EDGE_VALUES:
            x[t] = EDGE_VALUES[name]
        elif name in EDGE_FACTORS:
            x[t] *= EDGE_FACTORS[name]
        elif name == "one_hot":
            x[t] = -1e30
            x[t[3]] = 0.0
        else:
            v, col = name.split("@")
            x[t[0], int(col)] = EDGE_VALUES[v]
    w = (1 + 0.1 * torch.randn(D, generator=gen, device=dev)).to(torch.bfloat16)
    w[3], w[100] = 0.0, 64.0
    state = torch.full((NBS, STATE_BLOCK, 2 * STATE_W), SENTINEL_STATE, dtype=torch.int32, device=dev).view(torch.float32)
    n_pre = lay["prefilled"].shape[0]
    pre = torch.cat([scaled(n_pre, STATE_W, -3, 3), 4 * torch.randn(n_pre, STATE_W, generator=gen, device=dev)], dim=1)
    b, r = (torch.from_numpy(lay["prefilled"][:, i]).to(dev).long() for i in (0, 1))
    state[b, r] = pre
    return {**lay, "kv_score": kv_score, "ape": 0.5 * torch.randn(RATIO, STATE_W, generator=gen, device=dev), "w": w,
            "cs": torch.from_numpy(rotary_cache()).to(dev), "state0": state}


# ------------------------------------------------------------------------------------------------------------------ the phases
def sources(root: Path) -> dict[str, Any]:
    """Each of SOURCES under the vLLM package directory ``root``: True when its sha256 is the one read, False when it differs,
    None when it is missing."""
    files: dict[str, bool | None] = {}
    for rel, want in SOURCES.items():
        p = root / rel
        files[rel] = hashlib.sha256(p.read_bytes()).hexdigest() == want if p.is_file() else None
    return {"files": files, "ok": all(v is True for v in files.values())}


def selection() -> dict[str, Any]:
    import vllm
    from vllm.config import get_current_vllm_config
    from vllm.models.deepseek_v4.compressor import _prefer_two_stage_compressor
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4

    recipe, tma = compute_fp8_einsum_recipe()
    rec: dict[str, Any] = {"indexer_fp4": bool(dsa_indexer_uses_fp4(get_current_vllm_config())),
                           "two_stage_compressor": bool(_prefer_two_stage_compressor()),
                           "family_100": bool(current_platform.is_device_capability_family(100)),
                           "einsum_recipe": list(recipe), "tma_aligned_scales": bool(tma),
                           "sources": sources(Path(vllm.__file__).parent)}
    rec["ok"] = not (rec["indexer_fp4"] or rec["two_stage_compressor"] or rec["tma_aligned_scales"]) and rec["sources"]["ok"]
    return rec


def _ref_every(quick: bool) -> int:
    return QUICK_REF_EVERY if quick else 1


def phase_compress(work: Path, o: dict[str, Any], quick: bool) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """`save_partial_states` then `compress_norm_rope_store_triton`, twice on fresh caches: the score and compress records and jobs."""
    import torch
    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import compress_norm_rope_store_triton
    from vllm.models.deepseek_v4.common.ops.save_partial_states import save_partial_states

    dev = "cuda"
    T = o["pos"].size
    pos_t = torch.from_numpy(o["pos"]).to(dev)
    sslot_t, kslot_t = torch.from_numpy(o["sslot"]).to(dev), torch.from_numpy(o["kslot"]).to(dev)
    sbt_t, req_t = torch.from_numpy(o["sbt"]).to(dev), torch.from_numpy(o["req"]).to(dev)
    kv, score = o["kv_score"].split([STATE_W, STATE_W], dim=-1)
    runs = []
    for _ in range(2):
        state = o["state0"].clone()
        kc = torch.full((NBK, BLOCK, STRIDE), SENTINEL_K, dtype=torch.uint8, device=dev)
        save_partial_states(kv=kv, score=score, ape=o["ape"], positions=pos_t, state_cache=state, slot_mapping=sslot_t,
                            block_size=STATE_BLOCK, state_width=STATE_W, compress_ratio=RATIO, pdl_kwargs={"launch_pdl": False})
        torch.cuda.synchronize()
        saved = _u32(state)
        compress_norm_rope_store_triton(state_cache=state, num_actual=T, token_to_req_indices=req_t, positions=pos_t,
                                        slot_mapping=sslot_t, block_table=sbt_t, block_size=STATE_BLOCK, state_width=STATE_W,
                                        cos_sin_cache=o["cs"], kv_cache=kc, k_cache_metadata=SimpleNamespace(slot_mapping=kslot_t),
                                        pdl_kwargs={"launch_pdl": False}, head_dim=D, rope_head_dim=ROPE, compress_ratio=RATIO,
                                        overlap=True, use_fp4_cache=False, rms_norm_weight=o["w"], rms_norm_eps=EPS, quant_block=D,
                                        token_stride=D, scale_dim=4)
        torch.cuda.synchronize()
        runs.append({"saved": saved, "after": _u32(state), "k": _u8(kc), "kc": kc})
    init = _u32(o["state0"])
    pos, req, sslot, kslot = o["pos"], o["req"], o["sslot"], o["kslot"]
    written = np.flatnonzero(sslot >= 0)
    saved = runs[0]["saved"]
    flat_rows = saved.reshape(NBS * STATE_BLOCK, 2 * STATE_W)
    ks = _u32(o["kv_score"])
    touched = np.zeros(NBS * STATE_BLOCK, dtype=bool)
    touched[sslot[written]] = True
    rec_score = {"phase": "score", "case": "served", "tokens": int(T), "rows_written": int(written.size),
                 "checks": {"deterministic": bool(np.array_equal(saved, runs[1]["saved"])),
                            "kv_copied": bool(np.array_equal(flat_rows[sslot[written], :STATE_W], ks[written, :STATE_W])),
                            "unwritten_untouched": bool(np.array_equal(flat_rows[~touched], init.reshape(-1, 2 * STATE_W)[~touched])),
                            "rows_written": int(written.size) == T - EXTRA // 2}}
    ape = _u32(o["ape"])
    every = _ref_every(quick)
    paths = _save(work / "score", score=ks[written, STATE_W:], ape=ape[pos[written] % RATIO], got=flat_rows[sslot[written], STATE_W:],
                  tokens=written)
    jobs = [{"phase": "score", "case": "served", "kind": "score", "paths": paths, "rows": [a, b], "ref_every": every}
            for a, b in _chunks(written.size, JOB_ROWS)]
    entries = np.flatnonzero((sslot >= 0) & (kslot >= 0) & ((pos + 1) % RATIO == 0))
    k = runs[0]["k"]
    untouched = k.reshape(-1)[~X.slot_mask(NBK, kslot[entries])]
    rec_comp = {"phase": "compress", "case": "served", "entries": int((req[entries] == 0).sum()),
                "checks": {"deterministic": bool(np.array_equal(k, runs[1]["k"])),
                           "state_unchanged": all(np.array_equal(r["saved"], r["after"]) for r in runs),
                           "unwritten_untouched": bool((untouched == SENTINEL_K).all()),
                           "entries": int(entries.size) == (SEQ + EDGE_SEQ) // RATIO}}
    rec_edge = {"phase": "compress", "case": "edge", "entries": int((req[entries] == 1).sum()),
                "checks": {"entries": int((req[entries] == 1).sum()) == EDGE_SEQ // RATIO,
                           "deterministic": rec_comp["checks"]["deterministic"]}}
    w = _u16(o["w"])
    cs = _u32(o["cs"])
    for case, sel in (("served", entries[req[entries] == 0]), ("edge", entries[req[entries] == 1])):
        kvr, scr = state_rows(saved, o["sbt"], req[sel], pos[sel])
        got = [X.slot_bytes(k, kslot[t]) for t in sel]
        paths = _save(work / f"compress_{case}", kv=kvr, score=scr, w=w, cs=cs[(pos[sel] // RATIO) * RATIO],
                      q=np.stack([g[0] for g in got]), s=np.array([g[1] for g in got], dtype=np.uint32), tokens=pos[sel])
        jobs += [{"phase": "compress", "case": case, "kind": "entries", "paths": paths, "rows": [a, b],
                  "ref_every": every if case == "served" else 1} for a, b in _chunks(sel.size, JOB_ENTRIES)]
    return [rec_score, rec_comp, rec_edge], jobs, {"kcache": runs[0]["kc"]}


def fill_positions(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rand = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, FILL_RANDOM, dtype=np.int64, endpoint=True)
    return np.concatenate([np.array(FILL_POSITIONS, dtype=np.int64), rand, rng.integers(0, SEQ, FILL_RANDOM, dtype=np.int64)])


def phase_fill(work: Path, quick: bool) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """`_fill_short_context_topk_indices` (twice) over `fill_positions`."""
    import torch
    import triton
    from vllm.models.deepseek_v4.attention import _fill_short_context_topk_indices

    p = fill_positions()
    pt = torch.from_numpy(p).to("cuda")
    outs = []
    for _ in range(2):
        out = torch.full((p.size, TOPK), 7, dtype=torch.int32, device="cuda")
        _fill_short_context_topk_indices[(p.size,)](out, pt, TOP_K=TOPK, COMPRESS_RATIO=RATIO, PADDED_TOP_K=triton.next_power_of_2(TOPK),
                                                    num_warps=8)
        outs.append(out.cpu().numpy())
    rec = {"phase": "fill", "case": "served", "rows": int(p.size), "checks": {"deterministic": bool(np.array_equal(*outs))}}
    paths = _save(work / "fill", pos=p, got=outs[0])
    jobs = [{"phase": "fill", "case": "served", "kind": "fill", "paths": paths, "rows": [a, b], "ref_every": _ref_every(quick)}
            for a, b in _chunks(p.size, JOB_FILL)]
    return [rec], jobs, {}


def head_operands(seed: int, dev: str = "cuda") -> tuple[Any, np.ndarray]:
    """The attention output [INV_TOKENS, 16, 512] bf16 (random magnitudes per head; token 0's heads special: zeros, x 2^100, bf16
    subnormals, +inf at 5, NaN at 450 and -inf at 500 (rotary), one 2^20 outlier over 2^-20, multiples of 448) and the positions
    (tokens 1-3 at CS_SPECIAL)."""
    import torch

    gen = torch.Generator(device=dev).manual_seed(seed)
    T = INV_TOKENS
    x = torch.randn(T, NH, HD, generator=gen, device=dev) * torch.exp2(torch.empty(T, NH, 1, device=dev).uniform_(-4, 4, generator=gen))
    x[0, 0] = 0.0
    x[0, 1] *= 2.0 ** 100
    x[0, 2] *= 2.0 ** -130
    x[0, 3, 5] = float("inf")
    x[0, 4, 450] = float("nan")
    x[0, 5, 500] = float("-inf")
    x[0, 6] = 2.0 ** -20
    x[0, 6, 130] = 2.0 ** 20
    x[0, 7] = 448.0 * torch.exp2(torch.randint(-8, 8, (HD,), generator=gen, device=dev).float())
    pos = np.random.default_rng(seed).integers(0, MAXPOS, T).astype(np.int64)
    pos[1:4] = CS_SPECIAL
    return x.to(torch.bfloat16), pos


def phase_inv_rope(work: Path, o: dict[str, Any], quick: bool) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """`fused_inv_rope_fp8_quant` (twice) on the first NH heads of a PADDED_HEADS-head buffer."""
    import torch
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    T = INV_TOKENS
    xb, pos = head_operands(77)
    padded = torch.full((T, PADDED_HEADS, HD), SENTINEL_HEAD, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    padded[:, :NH] = xb
    x = padded[:, :NH, :]
    pt = torch.from_numpy(pos).to("cuda")
    rs = [fused_inv_rope_fp8_quant(x, pt, o["cs"], n_groups=GROUPS, heads_per_group=HPG, nope_dim=NOPE, rope_dim=ROPE, quant_group_size=QG,
                                   tma_aligned_scales=False) for _ in range(2)]
    torch.cuda.synchronize()
    fp8, scale = rs[0]
    Ta = -(-T // 4) * 4
    pad = torch.as_strided(scale, (GROUPS, Ta, HPG * CHUNKS), (HPG * CHUNKS * Ta, 1, Ta))[:, T:]
    rec = {"phase": "inv_rope", "case": "served", "tokens": T, "heads": NH,
           "checks": {"shapes": (list(fp8.shape) == [T, GROUPS, HPG * HD] and fp8.dtype == torch.float8_e4m3fn
                                 and list(scale.shape) == [T, GROUPS, HPG * CHUNKS] and scale.dtype == torch.float32
                                 and list(fp8.stride()) == [HPG * HD, T * HPG * HD, 1]),
                      "deterministic": all(np.array_equal(_u8(a), _u8(b)) for a, b in zip(rs[0], rs[1])),
                      "scale_padding_zero": bool((_u32(pad) == 0).all()) and pad.shape[1] == Ta - T,
                      "input_unchanged": bool((_u16(padded[:, NH:]) == SENTINEL_HEAD).all())}}
    q, s = head_words(_u8(fp8), _u32(scale))
    ids = np.arange(T * NH)
    paths = _save(work / "inv_rope", x=_u16(x).reshape(T * NH, HD), cs=np.repeat(_u32(o["cs"])[pos], NH, axis=0),
                  q=q.reshape(T * NH, HD), s=s.reshape(T * NH, CHUNKS), ids=ids)
    every = _ref_every(quick)
    jobs = [{"phase": "inv_rope", "case": "served", "kind": "heads", "paths": paths, "rows": [a, b], "ref_every": 1 if a < NH else every}
            for a, b in _chunks(T * NH, JOB_HEADS)]
    return [rec], jobs, {}


def queries(seed: int, n: int, dev: str = "cuda"):
    """e4m3 index queries [n, 64, 128] and f32 weights [n, 64] (their ue8m0 scale folded in, as `fused_indexer_q_rope_quant` makes
    them, fused_indexer_q.py); token 6's head 0 zero and token 7's weights zero."""
    import torch

    gen = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(n, H, D, generator=gen, device=dev) * torch.exp2(torch.randint(-3, 3, (n, H, 1), generator=gen, device=dev).float())
    q = q.clamp(-448, 448).to(torch.float8_e4m3fn)
    q[6, 0] = 0
    w = 0.05 * torch.randn(n, H, generator=gen, device=dev) * torch.exp2(torch.randint(-6, 1, (n, 1), generator=gen, device=dev).float())
    w[7] = 0
    return q, w


def phase_logits(work: Path, o: dict[str, Any], cst: dict[str, Any], quick: bool):
    """The prefill gather, `fp8_fp4_mqa_logits` and `fp8_fp4_paged_mqa_logits` on request 0's compressed cache."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata
    from vllm.utils.platform_utils import num_compute_units

    cache = cst["kcache"]
    n_keys = SEQ // RATIO
    bt = torch.from_numpy(o["kbt"][:1]).to("cuda").contiguous()
    k_quant = torch.empty(n_keys, D, dtype=torch.float8_e4m3fn, device="cuda")
    k_scale = torch.empty(n_keys, 4, dtype=torch.uint8, device="cuda")
    ops.cp_gather_indexer_k_quant_cache(cache, k_quant, k_scale, bt, torch.tensor([0, n_keys], dtype=torch.int32, device="cuda"))
    q, w = queries(5, SEQ)
    ks = torch.zeros(SEQ, dtype=torch.int32, device="cuda")
    ke = ((torch.arange(SEQ, device="cuda") + 1) // RATIO).to(torch.int32)
    kv = (k_quant, k_scale.view(torch.float32).squeeze(-1))
    logits = fp8_fp4_mqa_logits((q, None), kv, w, ks, ke, clean_logits=False)
    logits2 = fp8_fp4_mqa_logits((q, None), kv, w, ks, ke, clean_logits=False)
    B = len(DECODE_LENS)
    lens = torch.tensor(DECODE_LENS, dtype=torch.int32, device="cuda")
    tok = (RATIO * lens.long() - 1)
    q_dec = q[tok].reshape(B, 1, H, D)
    w_dec = w[tok].contiguous()
    seq_lens = lens.reshape(B, 1).contiguous()
    btd = bt.expand(B, -1).contiguous()
    sched = get_paged_mqa_logits_metadata(seq_lens, BLOCK, num_compute_units(torch.cuda.current_device()))
    dec = fp8_fp4_paged_mqa_logits((q_dec, None), cache.unsqueeze(-2), w_dec, seq_lens, btd, sched, max_model_len=MAX_MODEL_LEN,
                                   clean_logits=False)
    torch.cuda.synchronize()
    idx = _u8(cache)
    kbt = o["kbt"][0]
    slots = kbt[np.arange(n_keys) // BLOCK].astype(np.int64) * BLOCK + np.arange(n_keys) % BLOCK
    gq, gs = _u8(k_quant), _u32(k_scale.view(torch.float32)).reshape(-1)
    gather_ok = all(np.array_equal(gq[e], X.slot_bytes(idx, slots[e])[0]) and gs[e] == X.slot_bytes(idx, slots[e])[1] for e in range(n_keys))
    L, Ld = _u32(logits), _u32(dec)
    ken = ke.cpu().numpy()
    dec_eq = sum(int((Ld[b, :n] != L[RATIO * n - 1, :n]).sum()) for b, n in enumerate(DECODE_LENS))
    same, past_window = logits_rerun(logits, logits2, ke)
    recs = [{"phase": "logits", "case": "prefill", "shape": list(logits.shape),
             "checks": {"gather": bool(gather_ok), "deterministic": same,
                        "shape": list(logits.shape) == [SEQ, n_keys] and logits.dtype == torch.float32},
             "info": {"words_past_window_differing": past_window}},
            {"phase": "logits", "case": "decode", "shape": list(dec.shape), "lens": list(DECODE_LENS),
             "checks": {"shape": list(dec.shape) == [B, MAX_MODEL_LEN] and dec.dtype == torch.float32},
             "info": {"words_differing_from_prefill": dec_eq}}]
    rng = np.random.default_rng(7)
    rows = sorted(set(FULL_ROWS) | set(rng.choice(np.arange(RATIO - 1, SEQ), 2 if quick else RANDOM_ROWS, replace=False).tolist()))
    r_full = np.concatenate([np.full(ken[r], r) for r in rows])
    c_full = np.concatenate([np.arange(ken[r]) for r in rows])
    n_pairs = RANDOM_PAIRS // (10 if quick else 1)
    r_rand = rng.integers(RATIO - 1, SEQ, n_pairs)
    c_rand = (rng.random(n_pairs) * ken[r_rand]).astype(np.int64)
    rr, cc = np.concatenate([r_full, r_rand]), np.concatenate([c_full, c_rand])
    r_dec = np.concatenate([np.full(n, RATIO * n - 1) for n in DECODE_LENS])
    c_dec = np.concatenate([np.arange(n) for n in DECODE_LENS])
    g_dec = np.concatenate([Ld[b, :n] for b, n in enumerate(DECODE_LENS)])
    paths = _save(work / "logits", q=_u8(q), w=_u32(w), k=gq, ks=gs, r=rr, c=cc, got=L[rr, cc], rd=r_dec, cd=c_dec, gotd=g_dec)
    jobs = [{"phase": "logits", "case": "prefill", "kind": "logits", "paths": paths, "keys": ["r", "c", "got"], "range": [a, b]}
            for a, b in _chunks(rr.size, JOB_LOGITS)]
    jobs += [{"phase": "logits", "case": "decode", "kind": "logits", "paths": paths, "keys": ["rd", "cd", "gotd"], "range": [a, b]}
             for a, b in _chunks(r_dec.size, JOB_LOGITS)]
    return recs, jobs, {"logits": logits, "ke": ke, "decode": dec, "decode_lens": seq_lens}


def topk_rows(seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic decode rows of width MAX_MODEL_LEN: random values over TOPK_LENS keys (past each length +inf, which no kernel may
    read), a row of 1500 equal values, a row of 1100 keys with 1050 ones and the rest zeros, and a length of -3."""
    rng = np.random.default_rng(seed)
    lens = list(TOPK_LENS) + [1500, 1100, -3]
    names = [f"len{n}" for n in TOPK_LENS] + ["ties_all_equal", "ties_1050_ones", "negative"]
    vals = rng.standard_normal((len(lens), MAX_MODEL_LEN)).astype(np.float32)
    vals[len(TOPK_LENS)] = 1.0
    vals[len(TOPK_LENS) + 1] = 0.0
    vals[len(TOPK_LENS) + 1, rng.choice(1100, 1050, replace=False)] = 1.0
    for i, n in enumerate(lens):
        vals[i, max(n, 0):] = np.inf
    return vals, np.array(lens, dtype=np.int32), names


def phase_topk(o: dict[str, Any], lst: dict[str, Any] | None) -> dict[str, Any]:
    """vLLM's top-k kernels at TOPK 1024: prefill on the served logits (twice) and on synthetic rows from row start 5; on the decode
    logits the kernel vLLM picks; the decode kernels on synthetic rows; the reference `DsaTopKShortRow_v1{1024}` on its lengths."""
    import torch
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.sparse_attn_indexer import RADIX_TOPK_WORKSPACE_SIZE
    from vllm.platforms import current_platform
    from verity_vllm.program.registry import dsa

    rec: dict[str, Any] = {"cases": {}, "errors": {}}
    cooperative = bool(current_platform.has_device_capability(90) and not current_platform.is_device_capability_family(120))
    rec["cooperative_served"] = cooperative

    def ws():
        return torch.zeros(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device="cuda")

    if lst is None:
        rec["errors"]["prefill_served"] = "no served logits (the logits phase failed)"
    else:
        logits, ke = lst["logits"], lst["ke"]
        ks = torch.zeros(SEQ, dtype=torch.int32, device="cuda")
        outs = []
        for _ in range(2):
            out = torch.full((SEQ, TOPK), -1, dtype=torch.int32, device="cuda")
            ops.top_k_per_row_prefill(logits, ks, ke, out, SEQ, logits.stride(0), logits.stride(1), TOPK)
            outs.append(out.cpu().numpy())
        ken = ke.cpu().numpy()
        served = check_topk(logits.cpu().numpy(), ken, outs[0])
        short = ken <= TOPK
        served["deterministic_short"] = bool(np.array_equal(outs[0][short], outs[1][short]))
        served["deterministic_long"] = bool(np.array_equal(outs[0][~short], outs[1][~short]))
        rec["cases"]["prefill_served"] = served
        dec, sl = lst["decode"], lst["decode_lens"]
        name = "decode_served_cooperative" if cooperative and dec.stride(0) % 4 == 0 else "decode_served_persistent"
        try:
            out = torch.full((dec.shape[0], TOPK), -1, dtype=torch.int32, device="cuda")
            if name.endswith("cooperative"):
                torch.ops._C.cooperative_topk(dec, sl, out, ws(), TOPK, RATIO * max(DECODE_LENS))
            else:
                torch.ops._C.persistent_topk(dec, sl, out, ws(), TOPK, dec.shape[1])
            rec["cases"][name] = check_topk(dec.cpu().numpy(), np.array(DECODE_LENS), out.cpu().numpy())
        except Exception as e:  # noqa: BLE001
            rec["errors"][name] = _error(e)
            X._clear_launch_error()
    vals, lens, names = topk_rows(11)
    rows = len(lens)
    start = 5
    wide = np.full((rows, MAX_MODEL_LEN + start + 8), np.inf, dtype=np.float32)
    wide[:, start:start + MAX_MODEL_LEN] = vals
    wide[:, :start] = 1e30
    pl = torch.from_numpy(wide).to("cuda")
    pks = torch.full((rows,), start, dtype=torch.int32, device="cuda")
    pke = (pks + torch.from_numpy(np.maximum(lens, 0)).to("cuda")).to(torch.int32)
    pout = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
    ops.top_k_per_row_prefill(pl, pks, pke, pout, rows, pl.stride(0), pl.stride(1), TOPK)
    rec["cases"]["prefill_start5"] = check_topk(wide, np.maximum(lens, 0), pout.cpu().numpy(), start)
    dl = torch.from_numpy(vals).to("cuda")
    seq_lens = torch.from_numpy(lens).to("cuda").reshape(rows, 1).contiguous()
    kernels = [("persistent", rows, lambda d, s, out, m: torch.ops._C.persistent_topk(d, s, out, ws(), TOPK, d.shape[1]))]
    if cooperative:
        kernels += [(n_, k_, lambda d, s, out, m: torch.ops._C.cooperative_topk(d, s, out, ws(), TOPK, m))
                    for n_, k_ in (("cooperative", rows), ("cooperative_8rows", 8))]
    kernels.append(("decode_fallback", rows, lambda d, s, out, m: ops.top_k_per_row_decode(d, 1, s, out, d.shape[0], d.stride(0),
                                                                                             d.stride(1), TOPK)))
    for name, n, call in kernels:
        try:
            outs = []
            for _ in range(2):
                out = torch.full((n, TOPK), -1, dtype=torch.int32, device="cuda")
                call(dl[:n], seq_lens[:n], out, RATIO * int(lens[:n].max()))
                outs.append(out.cpu().numpy())
            r = check_topk(vals[:n], lens[:n], outs[0])
            r["deterministic_short"] = bool(np.array_equal(outs[0][lens[:n] <= TOPK], outs[1][lens[:n] <= TOPK]))
            rec["cases"][name] = r
        except Exception as e:  # noqa: BLE001
            rec["errors"][name] = _error(e)
            X._clear_launch_error()
    ref_lens = sorted({0, 1, 2, TOPK - 1, TOPK, -3})
    fn = bind(dsa.DsaTopKShortRow, TOPK=TOPK)
    rec["reference_lengths"] = ref_lens
    rec["reference_ok"] = all(list(evaluate(fn, n & 0xFFFFFFFF)) == (X.topk_short(np.array([max(n, 0)]), TOPK)[0].astype(np.int64)
                                                                     & 0xFFFFFFFF).tolist() for n in ref_lens)
    gating = ("prefill_served", "prefill_start5", "persistent") + (("cooperative", "cooperative_8rows") if cooperative else ())
    gating += tuple(k for k in rec["cases"] if k.startswith("decode_served"))
    rec["ok"] = bool(rec["reference_ok"] and all(g in rec["cases"] and rec["cases"][g]["short_rows"] > 0
                                                 and rec["cases"][g]["short_mismatch"] == 0 for g in gating)
                     and any(k.startswith("decode_served") for k in rec["cases"])
                     and rec["cases"]["prefill_served"]["deterministic_short"])
    return rec


# ------------------------------------------------------------------------------------------------------------------ comparisons
def _tally(rec: dict[str, Any], want: np.ndarray, got: np.ndarray, ids: np.ndarray, unit: str) -> None:
    want = np.asarray(want, dtype=np.int64).reshape(len(ids), -1)
    got = np.asarray(got, dtype=np.int64).reshape(len(ids), -1)
    bad = want != got
    rec["words"] += int(got.size)
    rec["mismatch"] += int(bad.sum())
    if bad.any() and rec["first_mismatch"] is None:
        r, w = (int(v) for v in np.argwhere(bad)[0])
        rec["first_mismatch"] = {unit: int(ids[r]), "word": w, "want": int(want[r, w]), "kernel": int(got[r, w])}


def _ref_rows(lo: int, hi: int, every: int) -> list[int]:
    return [i for i in range(lo, hi) if i == lo or i % every == 0]


def _reference(rec: dict[str, Any], fn, want: np.ndarray, lo: int, rows: list[int], args) -> None:
    from verity.ml.circuits.evaluation import evaluate

    for i in rows:
        y = evaluate(fn, *args(i))
        rec["reference"] += 1
        rec["twin_mismatch"] += [int(v) for v in y] != np.asarray(want[i - lo], dtype=np.int64).reshape(-1).tolist()


def compare_score(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """State rows [lo, hi): their score words against the twin, the reference `Dsv4CompressorScore_v1{256}` on every ref_every-th."""
    from verity.ml.circuits.defs import bind

    a = _load(job["paths"])
    lo, hi = job["rows"]
    want = score_twin(a["score"][lo:hi], a["ape"][lo:hi])
    _tally(rec, want, a["got"][lo:hi], a["tokens"][lo:hi], "token")
    _reference(rec, bind(I.Dsv4CompressorScore, W=STATE_W), want, lo, _ref_rows(lo, hi, job["ref_every"]),
               lambda i: (a["score"][i].tolist(), a["ape"][i].tolist()))


def compare_entries(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Cache entries [lo, hi): bytes and scale against the twin, the reference `Dsv4IndexerCompress_v1{128,64,1e-06}` on every
    ref_every-th."""
    from verity.ml.circuits.defs import bind

    a = _load(job["paths"])
    lo, hi = job["rows"]
    q, s = entries_twin(a["kv"][lo:hi], a["score"][lo:hi], a["w"], a["cs"][lo:hi])
    want = np.concatenate([q.astype(np.int64), s[:, None].astype(np.int64)], axis=1)
    got = np.concatenate([np.asarray(a["q"][lo:hi], dtype=np.int64), np.asarray(a["s"][lo:hi], dtype=np.int64)[:, None]], axis=1)
    _tally(rec, want, got, a["tokens"][lo:hi], "position")
    w = np.asarray(a["w"]).tolist()
    _reference(rec, bind(I.Dsv4IndexerCompress, D=D, ROPE=ROPE, EPS=EPS), want, lo, _ref_rows(lo, hi, job["ref_every"]),
               lambda i: (np.asarray(a["kv"][i]).reshape(-1).tolist(), np.asarray(a["score"][i]).reshape(-1).tolist(), w,
                          a["cs"][i].tolist()))


def compare_heads(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """(token, head)s [lo, hi) (id = token * 16 + head): bytes and scales against the twin, the reference
    `Dsv4InvRopeQuantHead_v1{4,128,64}` on every ref_every-th."""
    from verity.ml.circuits.defs import bind

    a = _load(job["paths"])
    lo, hi = job["rows"]
    q, s = heads_twin(a["x"][lo:hi], a["cs"][lo:hi])
    want = np.concatenate([q.astype(np.int64), s.astype(np.int64)], axis=1)
    got = np.concatenate([np.asarray(a["q"][lo:hi], dtype=np.int64), np.asarray(a["s"][lo:hi], dtype=np.int64)], axis=1)
    _tally(rec, want, got, a["ids"][lo:hi], "head_id")
    _reference(rec, bind(I.Dsv4InvRopeQuantHead, CHUNKS=CHUNKS, QG=QG, ROPE=ROPE), want, lo, _ref_rows(lo, hi, job["ref_every"]),
               lambda i: (a["x"][i].tolist(), a["cs"][i].tolist()))


def compare_fill(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Fill rows [lo, hi) against the twin, the reference `Dsv4ShortContextTopK_v1{1024,4}` on every ref_every-th."""
    from verity.ml.circuits.defs import bind

    a = _load(job["paths"])
    lo, hi = job["rows"]
    pos = np.asarray(a["pos"][lo:hi], dtype=np.int64)
    want = fill_twin(pos).astype(np.int64) & 0xFFFFFFFF
    _tally(rec, want, np.asarray(a["got"][lo:hi], dtype=np.int64) & 0xFFFFFFFF, pos, "position")
    _reference(rec, bind(I.Dsv4ShortContextTopK, TOPK=TOPK, RATIO=RATIO), want, lo, _ref_rows(lo, hi, job["ref_every"]),
               lambda i: (int(a["pos"][i]) & M64,))


def compare_logits(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Entries [a, b) of the prefill or decode logits: the twin against the kernel's words, `DsaMqaLogit_v3{64,128}` on the first two."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate
    from verity_vllm.program.registry import dsa

    a = _load(job["paths"])
    kr, kc, kg = job["keys"]
    lo, hi = job["range"]
    r, c = np.asarray(a[kr][lo:hi]), np.asarray(a[kc][lo:hi])
    got = np.asarray(a[kg][lo:hi], dtype=np.uint32)
    q, w, k, ks = np.asarray(a["q"])[r], np.asarray(a["w"])[r], np.asarray(a["k"])[c], np.asarray(a["ks"])[c]
    want, nonfin = X.logit_rows(q, w, k, ks)
    bad = np.flatnonzero((want != got) & ~nonfin)
    rec["words"] += int(got.size)
    rec["mismatch"] += int(bad.size)
    rec["unresolved"] += int(nonfin.sum())
    if bad.size and rec["first_mismatch"] is None:
        i = int(bad[0])
        rec["first_mismatch"] = {"row": int(r[i]), "key": int(c[i]), "want": int(want[i]), "kernel": int(got[i])}
    fn = bind(dsa.DsaMqaLogitV3, H=H, D=D)
    for i in range(min(2, got.size)):
        y = evaluate(fn, q[i].reshape(-1).tolist(), w[i].tolist(), k[i].tolist(), int(ks[i]))
        rec["reference"] += 1
        rec["twin_mismatch"] += y != (int(want[i]),)


COMPARE = {"score": compare_score, "entries": compare_entries, "heads": compare_heads, "fill": compare_fill, "logits": compare_logits}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    try:
        COMPARE[job["kind"]](job, rec)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "error": _error(e)}
    return {"phase": job["phase"], "case": job["case"], **rec}


# --------------------------------------------------------------------------------------------------------------- ptx and bind
def ptx(triton_cache: Path, outdir: Path) -> dict[str, Any]:
    """The Triton cache's PTX of KERNELS: ok when each has exactly one specialization (every launch here ran the served kernel).
    Each file's sha256 and arithmetic instruction counts; the PTX, and the SASS where Triton's bundled `cuobjdump` disassembles the
    cubin, are copied to `<out>/ptx/`."""
    dest = outdir / "ptx"
    dest.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"kernels": {}}
    try:
        import triton

        cand = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump"
        tool = str(cand) if cand.exists() else shutil.which("cuobjdump")
    except Exception:  # noqa: BLE001
        tool = shutil.which("cuobjdump")
    for kernel in KERNELS:
        entries = []
        for f in sorted(glob.glob(str(triton_cache / "**" / f"{kernel}*.ptx"), recursive=True)):
            text = Path(f).read_text(errors="replace")
            name = f"{kernel}.{hashlib.sha256(text.encode()).hexdigest()[:12]}"
            shutil.copy(f, dest / f"{name}.ptx")
            e = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "counts": dict(Counter(PTX_COUNT.findall(text))),
                 "maxntid": re.findall(r"\.maxntid\s+([\d, ]+)", text)}
            cubin = Path(f).with_suffix(".cubin")
            if tool and cubin.exists():
                try:
                    sass = subprocess.run([tool, "-sass", str(cubin)], capture_output=True, text=True, check=True).stdout
                    (dest / f"{name}.sass").write_text(sass)
                    e["sass_counts"] = dict(Counter(re.findall(r"\b(FFMA|FMUL|FADD|FMNMX|MUFU\.\w+|F2FP\.\w+(?:\.\w+)*)\b", sass)))
                except Exception as err:  # noqa: BLE001
                    e["sass_error"] = str(err)[-300:]
            entries.append(e)
        rec["kernels"][kernel] = entries
    rec["ok"] = all(len(rec["kernels"][k]) == 1 for k in KERNELS)
    return rec


def bind_check() -> dict[str, Any]:
    """The Programs bind at DeepSeek-V4-Pro's statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsa

    targets = {f"Dsv4CompressorScore_v1{{W={STATE_W}}}": lambda: bind(I.Dsv4CompressorScore, W=STATE_W),
               f"Dsv4CompressColumn_v1{{ROWS={ROWS}}}": lambda: bind(I.Dsv4CompressColumn, ROWS=ROWS),
               f"Dsv4IndexerNormRope_v1{{D={D},ROPE={ROPE},EPS={EPS}}}": lambda: bind(I.Dsv4IndexerNormRope, D=D, ROPE=ROPE, EPS=EPS),
               f"Dsv4IndexerKQuant_v1{{N={D}}}": lambda: bind(I.Dsv4IndexerKQuant, N=D),
               f"Dsv4IndexerCompress_v1{{D={D},ROPE={ROPE},EPS={EPS}}}": lambda: bind(I.Dsv4IndexerCompress, D=D, ROPE=ROPE, EPS=EPS),
               f"Dsv4InvRope_v1{{ROPE={ROPE}}}": lambda: bind(I.Dsv4InvRope, ROPE=ROPE),
               f"Dsv4InvRopeQuant_v1{{N={QG}}}": lambda: bind(I.Dsv4InvRopeQuant, N=QG),
               f"Dsv4InvRopeQuantHead_v1{{CHUNKS={CHUNKS},QG={QG},ROPE={ROPE}}}":
                   lambda: bind(I.Dsv4InvRopeQuantHead, CHUNKS=CHUNKS, QG=QG, ROPE=ROPE),
               f"Dsv4ShortContextTopK_v1{{TOPK={TOPK},RATIO={RATIO}}}": lambda: bind(I.Dsv4ShortContextTopK, TOPK=TOPK, RATIO=RATIO),
               f"DsaMqaLogit_v3{{H={H},D={D}}}": lambda: bind(dsa.DsaMqaLogitV3, H=H, D=D),
               f"DsaMqaLogits_v3{{N=64,H={H},D={D}}}": lambda: bind(dsa.DsaMqaLogitsV3, N=64, H=H, D=D),
               f"DsaTopKShortRow_v1{{TOPK={TOPK}}}": lambda: bind(dsa.DsaTopKShortRow, TOPK=TOPK)}
    out = {}
    for name, make in targets.items():
        try:
            out[name] = bool(make().signature())
        except Exception as e:  # noqa: BLE001
            out[name] = _error(e)
    return {"bindings": out, "ok": all(v is True for v in out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
def gpu(work: Path, outdir: Path, quick: bool, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every GPU phase in order; each phase's error goes to doc["errors"] (later phases that read it then fail too).  Fills
    doc["selection"], doc["stack"], doc["launches"], doc["topk"]; returns the comparison jobs."""
    import torch

    jobs: list[dict[str, Any]] = []
    st: dict[str, Any] = {}

    def step(name: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            print(f"[dsv4-idx] {name} ERROR {doc['errors'][name][:600]}", flush=True)
            return None

    doc["selection"] = step("selection", selection) or {}
    print(f"[dsv4-idx] selection {json.dumps(doc['selection'])}", flush=True)
    doc["stack"] = step("stack", stack) or {}
    o = step("operands", lambda: operands(20261013))
    for name, fn in (("compress", lambda: phase_compress(work, o, quick)), ("fill", lambda: phase_fill(work, quick)),
                     ("inv_rope", lambda: phase_inv_rope(work, o, quick)), ("logits", lambda: phase_logits(work, o, st["compress"], quick))):
        r = step(name, fn)
        if r is not None:
            doc["launches"] += r[0]
            jobs += r[1]
            st[name] = r[2]
            for x in r[0]:
                print(f"[dsv4-idx] {x['phase']} {x['case']:8s} {json.dumps(x['checks'])} {json.dumps(x.get('info', {}))[:600]}", flush=True)
    doc["topk"] = step("topk", lambda: phase_topk(o, st.get("logits"))) or {"ok": False}
    brief = {k: {f: v for f, v in g.items() if f != "long"} for k, g in doc["topk"].get("cases", {}).items()}
    print(f"[dsv4-idx] topk ok={doc['topk'].get('ok')} {json.dumps(brief)[:1500]} errors={list(doc['topk'].get('errors', {}))}",
          flush=True)
    del st, o
    torch.cuda.empty_cache()
    return jobs


def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["checks"]) and all(v is True for v in x["checks"].values())


def _words_ok(t: dict[str, Any]) -> bool:
    return t["words"] > 0 and t["mismatch"] == 0 and t["unresolved"] == 0 and t["reference"] > 0 and t["twin_mismatch"] == 0


CASES = (("score", "served"), ("compress", "served"), ("compress", "edge"), ("fill", "served"), ("inv_rope", "served"),
         ("logits", "prefill"), ("logits", "decode"))
#: the GPU step whose error fails a phase's cases
STEP = {"score": "compress", "compress": "compress", "fill": "fill", "inv_rope": "inv_rope", "logits": "logits"}


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each compared case's verdict: its launch checks, its words, and its errors, so a NO-GO names what failed."""
    out: dict[str, dict[str, Any]] = {}
    for phase, case in CASES:
        launches = [x for x in doc["launches"] if x["phase"] == phase and x["case"] == case]
        comps = [r for r in doc.get("comparisons", []) if r["phase"] == phase and r["case"] == case]
        t = {k: sum(r[k] for r in comps) for k in TOTALS}
        errors = [e for e in doc["errors"] if e == STEP[phase] or e == f"compare/{phase}/{case}"]
        ok = bool(launches) and all(_launch_ok(x) for x in launches)
        out[f"{phase}/{case}"] = {"launches_ok": ok, "totals": t, "errors": errors, "ok": ok and _words_ok(t) and not errors}
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    return bool(doc["gate"]["ok"] and not doc["errors"] and doc["selection"].get("ok") is True and doc["ptx"].get("ok") is True
                and doc["bind"].get("ok") is True and doc["topk"].get("ok") is True
                and all(c["ok"] for c in cases(doc).values()) and _words_ok(doc["totals"]))


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env = environment(work)
    import torch  # noqa: F401
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "quick": quick, "environment": G._environment(),
                           "gate": C.gate(min_driver, want_cc, want_sms), "errors": {}, "launches": []}
    print(f"[dsv4-idx] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-INDEXER GATE-FAIL", flush=True)
        raise SystemExit(3)
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        jobs = gpu(work, outdir, quick, doc)
    for k in ("selection", "topk"):
        doc.setdefault(k, {"ok": False})
    try:
        doc["ptx"] = ptx(work / "triton_cache", outdir)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["ptx"] = _error(e)
        doc["ptx"] = {"ok": False}
    print(f"[dsv4-idx] ptx {json.dumps(doc['ptx'])[:2000]}", flush=True)
    try:
        doc["bind"] = bind_check()
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    print(f"[dsv4-idx] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
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
        print(f"[dsv4-idx] words {r['phase']:8s} {r['case']:8s} words={r['words']} !={r['mismatch']} unresolved={r['unresolved']} "
              f"ref={r['reference']} twin!={r['twin_mismatch']} first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[dsv4-idx] case {key:18s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} "
              f"!={t['mismatch']} unresolved={t['unresolved']} ref={t['reference']} twin!={t['twin_mismatch']} errors={c['errors']}",
              flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"DSV4-INDEXER {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro's DSA indexer kernels on this Hopper GPU against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and ptx/")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, Triton and DeepGEMM caches)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true",
                    help=f"the reference on every {QUICK_REF_EVERY}th served instance and fewer logits: a first signal in minutes")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
