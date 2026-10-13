"""GLM-5.3's sparse attention on Hopper against its Programs: the H200's go/no-go for binding the DSA indexer's `dsa.DsaIndexerKRow_v1`
(Triton `_fused_norm_rope_kernel` program 0) and `dsa.DsaIndexerQ_v1` (`_fused_q_kernel` program 1), `dsa.DsaMqaLogit_v3` (DeepGEMM
`fp8_fp4_mqa_logits` and `fp8_fp4_paged_mqa_logits`), `dsa.DsaTopKShortRow_v1` (vLLM's `top_k_per_row_prefill`, `cooperative_topk`
and `persistent_topk`) and `sparse_mla.SparseMlaHead_v1` (FA3, `FLASH_ATTN_MLA_SPARSE`) at GLM-5.3's statics on one TP8 rank (8 MLA
heads; the indexer's 32 heads replicated), and the run that settles the RoPE contraction of the two Triton kernels.

One GPU process writes `<out>/glm53_dsa_capture.json`, the property record `glm53_dsa_capture` (`properties.record`), and the two
Triton kernels' PTX (and SASS where Triton's bundled `cuobjdump` runs) under `<out>/ptx/`; it prints one line per phase, a `ROPE`
line, then `GLM53-DSA GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first and nothing else runs when it fails
(exit 3).  vLLM runs as GLM-5.3's eager engine does on sm_90 (`VLLM_USE_DEEP_GEMM=0` for the FP8 linears; the indexer's DeepGEMM
kernels load regardless, `utils/deep_gemm.py` `_lazy_init`), with the Triton cache and DeepGEMM's JIT cache in the scratch directory.
A phase that raises records its error and fails the verdict without stopping the others.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `get_valid_backends` for GLM-5.3's MLA (head size 576, bf16, kv cache "auto", block 64, sparse, 8 heads) puts
             FLASH_ATTN_MLA_SPARSE first, and `is_fused_q_cutedsl_supported` refuses the bf16 query, so `_fused_q_kernel` runs.
  k          `fused_norm_rope` as `DeepseekV32Attention.forward` calls it (attention.py:338-366), twice on fresh caches, over SEQ tokens
             of one request at positions 0..SEQ-1 in a shuffled block table and EXTRA tokens at far positions (8 without a slot):
             every written indexer-cache slot (128 e4m3 bytes and the f32 scale) against `DsaIndexerKRow_v1{128,64,1e-06}` of the
             token's index_k row, k_norm's weight and bias and the token's rotary cache row; the top-k buffer all -1; every byte of
             an unwritten slot untouched; each written MLA cache row [kv_c_out | k_pe_out].  The index-K probe (`index_k_probe`):
             PROBE_LAUNCHES launches whose LayerNorm leaves exactly the bias (k = 0, w = 0), each rotary pair solved so that its
             small roped value lands within about 2^-17 of an e4m3 rounding midpoint, against the same Definition.
  q          `fused_q` (attention.py:378-392, bf16 query): each token's index_q_fp8 [32, 128] and index_weights_out [32] against
             `DsaIndexerQ_v1{32,128,64}`.
  rope       k_pe_out (`_fused_norm_rope_kernel` program 1) and q_pe_out (`_fused_q_kernel` program 0) of every token and of the
             subnormal probe (below): which of the candidate contractions the words are.
  logits     the prefill gather (`cp_gather_indexer_k_quant_cache`) against the cache slots; `fp8_fp4_mqa_logits` (ks = 0, ke = i + 1,
             clean_logits=False, sparse_attn_indexer.py:496-507) on rows FULL_ROWS, random rows and random (row, key) pairs, and
             `fp8_fp4_paged_mqa_logits` (one context per DECODE_LENS entry, its query the last token's, sparse_attn_indexer.py:600-613)
             on every key: each logit against `DsaMqaLogit_v3{32,128}` (twin `logit_rows`; the reference on a sample).
  topk       `top_k_per_row_prefill` on the served logits (twice) and on synthetic rows from a row start of 5; `persistent_topk`,
             `cooperative_topk` where vLLM calls it (not on sm_120; 10 and 8 rows, its clusters of 4 and 8 on sm_90), both
             sparse_attn_indexer.py:617-656, and the `top_k_per_row_decode` fallback (recorded) on synthetic rows
             of TOPK_LENS keys, tie rows and a negative length: every row of at most 2048 keys against `DsaTopKShortRow_v1{2048}`
             (closed form `topk_short`, the reference on its lengths); a longer row a valid top-2048 set (recorded: no Definition
             states its order or its ties).
  attn       FA3's sparse decode (`forward_mqa`, flashattn_mla_sparse.py:201-250) for ATTN_TOKENS on their prefill top-k rows: the
             index conversion (`triton_convert_req_index_to_global_index`) against `convert_rows`, FA3 twice, every (token, head)
             output against `SparseMlaHead_v1{T,64,512,64,...}` (T the row's valid count) of the token's q and its selected rows.
  ptx        the Triton cache holds one specialization of each of `_fused_norm_rope_kernel` and `_fused_q_kernel` (so the probes ran
             the served kernels): their PTX instruction counts, copied to `<out>/ptx/`.
  bind       the Programs bind at the served statics.

The RoPE contraction.  GLM-5.3's capture map named `RoPEHeadGptJ_v1` for the k_pe and q_pe RoPE (`fused-rope-triton`) before a run
measured it; run r20261012-222325-ce92 settled both on `RoPEHeadGptJ_v2`, which the map now names, and every run settles it again.
The kernels compute `r1 = x1*cos - x2*sin`, `r2 = x2*cos + x1*sin` in f32
from bf16 words (kernels.py:233-236, 718-731), so nvcc/ptxas may contract either product into an FMA: L_a = bf16(fma(x1, c, -RN(x2 s)))
(`RoPEHeadGptJ_v1`'s `RopeOut_v1`), L_b = bf16(fma(-x2, s, RN(x1 c))), L_n = bf16(RN(x1 c) - RN(x2 s)); H_a = bf16(fma(x2, c, RN(x1
s))) (`RoPEHeadGptJ_v1`'s `RopeOutAdd_v1`), H_b = bf16(fma(x1, s, RN(x2 c))) (`RoPEHeadGptJ_v2`, also measured on vLLM's
CUDA rotary kernel), H_n; each also flushing subnormals (`_ftz`).  A
product of two bf16 words is exact in f32 unless it leaves f32's normal range, so served operands cannot tell them apart: the
subnormal probe (`rope_probe_quads`) puts operand quadruples whose products are f32 subnormals beside a bf16 rounding tie, each
quadruple separating two forms, in k_pe (16 launches) and q_pe.  `ROPE k_pe=... q_pe=...` names the Definition the words are
(`rope_name`).  The index-K RoPE (program 0) rotates f32 values, where the forms differ on ordinary operands: its probe gates the k
phase through `DsaIndexerKRow_v1` (which states the form `fma(x, cos, RN(+-partner * sin))`), and `index_k_forms` records which form
the bytes are.

GO needs every phase's words compared with 0 mismatches and 0 unresolved (a non-finite logit accumulator; these operands keep every
logit finite), the reference sampled where a twin stands in with 0 disagreements, deterministic reruns (the prefill logits on the words
they write, `logits_rerun`), the checks above, and the RoPE
probe settled (one candidate form each for the even and the odd outputs of k_pe and of q_pe, or only flushing forms).

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_dsa_capture_gpu.py --out DIR [--scratch DIR] [--quick]
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
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-dsa-capture/v2"
RECORD = "glm53_dsa_capture.json"
#: GLM-5.3 (tests/check/fixtures/glm_moe_dsa/config.json) at TP8: the indexer's 32 heads of 128 with a 64-wide interleaved RoPE,
#: top 2048, its cache row 128 e4m3 bytes and a 4-byte scale (kernels.py:63-87); MLA's 8 local heads, rope 64, latent 512, nope 192
H, D, ROPE, TOPK, BLOCK = 32, 128, 64, 2048, 64
STRIDE = D + 4
NH, DV, DR, NOPE = 8, 512, 64, 192
Q_LORA, KV_LORA = 2048, 512
K_EPS, RMS_EPS = 1e-6, 1e-5
THETA, MAXPOS = 8000000, 1 << 20
SCALE = (NOPE + DR) ** -0.5
#: one request of SEQ tokens over SEQ // BLOCK of NB cache blocks, and EXTRA tokens at far positions (EXTRA_UNSLOTTED with slot -1)
SEQ, NB, EXTRA, EXTRA_UNSLOTTED = 3072, 64, 64, 8
MAX_MODEL_LEN = 4096
FULL_ROWS = (0, 1, 2, 5, 6, 63, 64, 65, 127, 128, 2047, 2048, 2049, 3071)
RANDOM_ROWS, RANDOM_PAIRS = 8, 50_000
DECODE_LENS = (1, 2, 63, 64, 65, 2048, 2049, 3000)
TOPK_LENS = (0, 1, 2, 2047, 2048, 2049, 4096)
ATTN_TOKENS = (0, 1, 4, 63, 64, 65, 127, 128, 129, 1000, 2046, 2047, 2048, 3071)
QUICK_ATTN_TOKENS = (0, 65, 2048)
QUICK_Q_TOKENS = 256
PROBE_LAUNCHES, PROBE_TOKENS = 16, 3
#: entries per logits job, cache slots per k job, tokens per q job
JOB_LOGITS, JOB_SLOTS, JOB_Q = 4096, 128, 8
#: the special rows: index_k rows 1-5 (zero, x 2^40, x 2^-40, bf16 subnormals, one outlier over tiny values), index_q heads 0-3 of
#: token 6 (zero, x 2^40, x 2^-40, bf16 subnormals)
K_SPECIAL, Q_SPECIAL_TOKEN = (1, 2, 3, 4, 5), 6
TOTALS = ("words", "mismatch", "unresolved", "reference", "twin_mismatch")
L_FORMS, H_FORMS = ("L_a", "L_b", "L_n"), ("H_a", "H_b", "H_n")
PTX_COUNT = re.compile(r"\b(fma\.rn(?:\.ftz)?\.f32|mul(?:\.rn)?(?:\.ftz)?\.f32|sub(?:\.rn)?(?:\.ftz)?\.f32|add(?:\.rn)?(?:\.ftz)?\.f32|"
                       r"neg(?:\.ftz)?\.f32|div\.rn(?:\.ftz)?\.f32|rsqrt\.approx(?:\.ftz)?\.f32|ex2\.approx(?:\.ftz)?\.f32|"
                       r"cvt\.rpi(?:\.ftz)?\.f32\.f32|cvt\.rn\.bf16\.f32|cvt\.rn\.satfinite\.e4m3x2\.f32|max(?:\.ftz)?\.f32)(?=[\s;])")
KERNELS = ("_fused_norm_rope_kernel", "_fused_q_kernel")


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()[-600:] + " | " + traceback.format_exc(limit=-3)[-900:]


# ------------------------------------------------------------------------------------------------------------------------- env
def environment(work: Path) -> dict[str, str]:
    """GLM-5.3's sm_90 engine env (TargetProfile `fp8_block_gemm="cutlass"`), the caches in scratch; set before vLLM is imported."""
    env = {"VLLM_USE_DEEP_GEMM": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TRITON_CACHE_DIR": str(work / "triton_cache"),
           "DG_JIT_CACHE_DIR": str(work / "deep_gemm")}
    os.environ.update(env)
    return env


def stack() -> dict[str, Any]:
    """What the kernels of record came from: DeepGEMM's module and version, Triton's, the top-k ops, PDL."""
    import torch
    import triton
    from vllm.platforms import current_platform
    from vllm.utils import deep_gemm as VDG
    from vllm.utils.import_utils import has_deep_gemm

    rec: dict[str, Any] = {"triton": triton.__version__, "has_deep_gemm": bool(has_deep_gemm()),
                           "pdl": bool(current_platform.is_arch_support_pdl())}
    try:
        dg = VDG._import_deep_gemm()
        rec["deep_gemm"] = {"file": getattr(dg, "__file__", None), "version": getattr(dg, "__version__", None)}
    except Exception as e:  # noqa: BLE001
        rec["deep_gemm"] = {"error": _error(e)}
    rec["ops"] = {name: hasattr(torch.ops._C, name) for name in ("cooperative_topk", "persistent_topk", "top_k_per_row_prefill",
                                                                  "top_k_per_row_decode")}
    return rec


# ---------------------------------------------------------------------------------------------------------------- word helpers
def _u8(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.uint8).cpu().numpy()


def _u16(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _u32(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int32).cpu().numpy().view(np.uint32)


def _save(d: Path, **arrays: np.ndarray) -> dict[str, str]:
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, a in arrays.items():
        np.save(d / f"{k}.npy", np.ascontiguousarray(a))
        out[k] = str(d / f"{k}.npy")
    return out


def _load(paths: dict[str, str]) -> dict[str, np.ndarray]:
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


def _chunks(n: int, step: int) -> list[tuple[int, int]]:
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def f32_word(v: float) -> int:
    return int(np.float32(v).view(np.uint32))


def bf16_word(v: float) -> int:
    """RNE of a value to bf16 (inputs here are f32-exact)."""
    u = f32_word(v)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


# ------------------------------------------------------------------------------------------------------- the indexer slot layout
def slot_bytes(cache: np.ndarray, slot: int) -> tuple[np.ndarray, int]:
    """One indexer-cache slot of a [blocks, BLOCK, STRIDE] byte cache: its D e4m3 bytes and its f32 scale word
    (`_fp8_quant_and_cache_write`: the block's D-byte rows first, then its BLOCK 4-byte scales)."""
    flat = cache.reshape(-1)
    blk, off = divmod(int(slot), BLOCK)
    base = blk * BLOCK * STRIDE
    q = flat[base + off * D: base + off * D + D]
    s = int(flat[base + BLOCK * D + off * 4: base + BLOCK * D + off * 4 + 4].view(np.uint32)[0])
    return q, s


def slot_mask(n_blocks: int, slots) -> np.ndarray:
    """The bytes of a [n_blocks, BLOCK, STRIDE] cache that writing ``slots`` touches."""
    m = np.zeros(n_blocks * BLOCK * STRIDE, dtype=bool)
    for slot in slots:
        blk, off = divmod(int(slot), BLOCK)
        base = blk * BLOCK * STRIDE
        m[base + off * D: base + off * D + D] = True
        m[base + BLOCK * D + off * 4: base + BLOCK * D + off * 4 + 4] = True
    return m


# ------------------------------------------------------------------------------------------------------------------- the twins
def logit_rows(q: np.ndarray, w: np.ndarray, k: np.ndarray, ks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`DsaMqaLogit_v3{H,D}` over n entries, each its own (q, w, k, ks): q (n, H, D) e4m3 bytes, w (n, H) f32 words, k (n, D) e4m3
    bytes, ks (n,) f32 words -> (n logit words, nonfinite mask).  Per head the D/32 `tc_dot32_e4m3` steps from +0 (key as A),
    `fmaxf(acc, +0)`, then the quad lanes' sums in ascending j (j = 0's product contracted into j = 1's add, `fma(r0, w0, r1 *
    w1)`, then FFMAs), `(s0 + s1) * ks` and `(v0 + v1) + (v2 + v3)`.  An entry whose dot steps flag a non-finite accumulator or an
    e4m3 NaN, or whose logit is NaN, is unresolved."""
    from verity_vllm.program.kernels import derived_rows as DR

    n, Hh, Dd = q.shape
    relu = np.empty((n, Hh), dtype=np.uint32)
    nonfin = np.zeros(n, dtype=bool)
    for h in range(Hh):
        acc = np.zeros(n, dtype=np.uint32)
        for s in range(Dd // 32):
            acc, nf = DR.tc_dot32_e4m3(acc, k[:, 32 * s:32 * s + 32], q[:, h, 32 * s:32 * s + 32])
            nonfin |= nf
        relu[:, h] = np.where(DR.as_f32(acc) > 0, acc, np.uint32(0))
    v = []
    for c in range(4):
        sums = []
        for pair in (0, 1):
            hs = [2 * c + pair + 8 * j for j in range(Hh // 8)]
            if len(hs) > 1:
                acc, rest = DR.fma_rn(relu[:, hs[0]], w[:, hs[0]], DR.mul_rn(relu[:, hs[1]], w[:, hs[1]])), hs[2:]
            else:
                acc, rest = DR.mul_rn(relu[:, hs[0]], w[:, hs[0]]), hs[1:]
            for h in rest:
                acc = DR.fma_rn(relu[:, h], w[:, h], acc)
            sums.append(acc)
        v.append(DR.mul_rn(DR.add_rn(sums[0], sums[1]), ks))
    out = DR.add_rn(DR.add_rn(v[0], v[1]), DR.add_rn(v[2], v[3]))
    return out, nonfin | np.isnan(DR.as_f32(out))


def topk_short(lens: np.ndarray, width: int = TOPK) -> np.ndarray:
    """`DsaTopKShortRow_v1{TOPK}` per row: slot j holds j when j < len, else -1."""
    j = np.arange(width, dtype=np.int64)[None, :]
    return np.where(j < np.asarray(lens, dtype=np.int64)[:, None], j, -1).astype(np.int32)


def topk_valid(vals: np.ndarray, n: int, idx: np.ndarray) -> dict[str, Any]:
    """Whether ``idx`` (TOPK int32) is a top-TOPK set of the n > TOPK values ``vals[:n]``: TOPK distinct keys below n, none
    unselected above the least selected; the ties at that threshold left out, and whether the indices ascend."""
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


def convert_rows(topk: np.ndarray, block_table: np.ndarray, block_stride_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """`triton_convert_req_index_to_global_index` (sparse_utils.py:398-511, one request, return_valid_counts) restated: entry tok of a
    row becomes `block_table[tok // BLOCK] * block_stride_rows + tok % BLOCK`, -1 when tok < 0 or its block is past the table; the
    valid entries move to the front in order (one tile of TOPK: an exclusive prefix sum), the rest -1; and each row's valid count."""
    out = np.full(topk.shape, -1, dtype=np.int32)
    counts = np.zeros(topk.shape[0], dtype=np.int32)
    nb = block_table.shape[0]
    for r, row in enumerate(np.asarray(topk, dtype=np.int64)):
        blk = np.where(row >= 0, row // BLOCK, -1)
        ok = (row >= 0) & (blk < nb)
        g = block_table[np.clip(blk, 0, nb - 1)].astype(np.int64) * block_stride_rows + row % BLOCK
        out[r, :int(ok.sum())] = g[ok]
        counts[r] = int(ok.sum())
    return out, counts


# ------------------------------------------------------------------------------------------------------------------ RoPE forms
NEG = np.uint32(0x80000000)


def rope_forms(x1, x2, c, s) -> dict[str, np.ndarray]:
    """Every candidate contraction (module docstring) of the interleaved RoPE on bf16 words (arrays of one shape): name -> bf16
    words of the even (L_*) or odd (H_*) output."""
    from verity_vllm.program.kernels import derived_rows as DR

    X1, X2, Cc, Ss = (DR.bf16_to_f32_words(np.asarray(v)) for v in (x1, x2, c, s))
    out = {}
    for suffix, mul, fma, add in (("", DR.mul_rn, DR.fma_rn, DR.add_rn), ("_ftz", DR.mul_ftz, DR.fma_ftz, DR.add_ftz)):
        out["L_a" + suffix] = fma(X1, Cc, mul(X2, Ss) ^ NEG)
        out["L_b" + suffix] = fma(X2 ^ NEG, Ss, mul(X1, Cc))
        out["L_n" + suffix] = add(mul(X1, Cc), mul(X2, Ss) ^ NEG)
        out["H_a" + suffix] = fma(X2, Cc, mul(X1, Ss))
        out["H_b" + suffix] = fma(X1, Ss, mul(X2, Cc))
        out["H_n" + suffix] = add(mul(X2, Cc), mul(X1, Ss))
    return {k: DR.f2fp_bf16(v) for k, v in out.items()}


def rope_forms_exact(x1: int, x2: int, c: int, s: int) -> dict[str, int]:
    """`rope_forms` of one quadruple through core's exact bit functions (the non-flushing forms)."""
    from verity.ml.circuits.ops.fp32 import f32_add_bits as add, f32_fma_bits as fma, f32_mul_bits as mul, f32_sub_bits as sub
    from verity.ml.circuits.ops.fp32 import BF16_NAN_TORCH, f32_to_bf16_rn_bits

    X1, X2, Cc, Ss = ((v & 0xFFFF) << 16 for v in (x1, x2, c, s))
    n = 0x80000000
    raw = {"L_a": fma(X1, Cc, mul(X2, Ss) ^ n), "L_b": fma(X2 ^ n, Ss, mul(X1, Cc)), "L_n": sub(mul(X1, Cc), mul(X2, Ss)),
           "H_a": fma(X2, Cc, mul(X1, Ss)), "H_b": fma(X1, Ss, mul(X2, Cc)), "H_n": add(mul(X2, Cc), mul(X1, Ss))}
    return {k: f32_to_bf16_rn_bits(v, BF16_NAN_TORCH) for k, v in raw.items()}


def rope_probe_quads(per_pair: int = 16, seed: int = 0) -> list[tuple[int, int, int, int]]:
    """Operand quadruples (x1, x2, cos, sin) as bf16 words, at least ``per_pair`` separating each of the six pairs of non-flushing
    forms (L_a/L_b, L_a/L_n, L_b/L_n and the H ones).  x = mx 2^-133 (mx < 256: bf16 subnormals and the least normals), cos and sin
    = (odd 129..255) 2^-16 and 2^-17 (or the exponents swapped), so x cos and x sin are integer and half-integer multiples of
    2^-149, f32 subnormals; the targets are the exact L or H at one unit 2^-150 beside a bf16 subnormal tie ((2k+1) 2^16 units), so
    rounding one product (a half-integer one) moves the result across the tie.  A quadruple is kept where its two forms differ."""
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for fs in (L_FORMS, H_FORMS) for i, a in enumerate(fs) for b in fs[i + 1:]]
    found: dict[tuple[str, str], list] = {p: [] for p in pairs}
    out: list[tuple[int, int, int, int]] = []
    mx = np.arange(1, 256)
    odd = np.arange(129, 256, 2)
    targets = [(2 * k + 1) * 2 ** 16 + d for k in range(3) for d in (-1, 1)]
    targets += [-t for t in targets]
    combos = [(kind, int(mc), int(ms)) for kind in (0, 1) for mc in odd for ms in odd]
    for i in rng.permutation(len(combos)):
        if all(len(v) >= per_pair for v in found.values()):
            break
        kind, mc, ms = combos[i]
        c, s = (bf16_word(mc * 2.0 ** -16), bf16_word(ms * 2.0 ** -17)) if kind == 0 else (bf16_word(mc * 2.0 ** -17),
                                                                                             bf16_word(ms * 2.0 ** -16))
        A, Bc = (2 * mc, ms) if kind == 0 else (mc, 2 * ms)
        for T in targets[rng.integers(len(targets))::5]:
            sols = []
            for g1 in (1, -1):
                for g2 in (1, -1):
                    num = T + g2 * Bc * mx
                    ok = (num % A == 0) & (num * g1 > 0)
                    sols += [("L", int(a), int(b), g1, g2) for a, b in zip(num[ok] // A * g1, mx[ok]) if 1 <= a <= 255]
                    num = T - g1 * Bc * mx
                    ok = (num % A == 0) & (num * g2 > 0)
                    sols += [("H", int(a), int(b), g1, g2) for a, b in zip(mx[ok], num[ok] // A * g2) if 1 <= b <= 255]
            for side, m1, m2, g1, g2 in sols[:2]:
                q = (m1 | (0x8000 if g1 < 0 else 0), m2 | (0x8000 if g2 < 0 else 0), c, s)
                f = rope_forms_exact(*q)
                useful = [p for p in pairs if p[0][0] == side and f[p[0]] != f[p[1]] and len(found[p]) < per_pair]
                for p in useful:
                    found[p].append(q)
                if useful and q not in out:
                    out.append(q)
    if not all(len(v) >= per_pair for v in found.values()):
        raise RuntimeError(f"rope_probe_quads: {({k: len(v) for k, v in found.items()})}")
    return out


def rope_matches(x1, x2, c, s, got_even, got_odd) -> dict[str, Any]:
    """The candidate forms that give every observed word: (L forms matching every even output, H forms every odd output)."""
    f = rope_forms(x1, x2, c, s)
    ge, go = np.asarray(got_even, dtype=np.uint16), np.asarray(got_odd, dtype=np.uint16)
    L = [k for k in f if k.startswith("L") and np.array_equal(f[k], ge)]
    Hm = [k for k in f if k.startswith("H") and np.array_equal(f[k], go)]
    return {"L": L, "H": Hm, "words": int(ge.size + go.size),
            "mismatch_v1": int((f["L_a"] != ge).sum() + (f["H_a"] != go).sum())}


def rope_settled(m: dict[str, Any]) -> bool:
    """One form each for the even and the odd outputs, or only flushing forms (which the probe's subnormal operands cannot tell
    apart, and no Definition here states)."""
    return all(m[k] and (len(m[k]) == 1 or all(f.endswith("_ftz") for f in m[k])) for k in ("L", "H"))


def rope_name(m: dict[str, Any]) -> str:
    """The Definition the words are: RoPEHeadGptJ_v1 = (L_a, H_a), RoPEHeadGptJ_v2 = (L_a, H_b)."""
    forms = (tuple(m["L"]), tuple(m["H"]))
    if forms == (("L_a",), ("H_a",)):
        return "RoPEHeadGptJ_v1"
    if forms == (("L_a",), ("H_b",)):
        return "RoPEHeadGptJ_v2"
    return f"no Definition (L in {list(forms[0])}, H in {list(forms[1])})"


# ------------------------------------------------------------------------------------------------------------- the index-K probe
def index_k_probe(launch: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """Launch ``launch``'s k_norm bias b (D f32 words) and its rotary cache row (ROPE bf16 words).  With k = 0 and w = 0 the
    LayerNorm's output is b exactly; b[ROPE] = 448 (outside the rotation) and the rest 0, so the ue8m0 scale is 1 and each slot byte is
    e4m3(roped).  Pair p's (x1, x2) = b[2p], b[2p+1] is solved (in f64, then rounded to f32) for roped targets (M1, M2), one a
    small e4m3 rounding midpoint +-(1 + (2m+1)/16) 2^e (e in -5..-2) and the other large (+-[64, 400]), alternating by pair: the
    products are about 100 and cancel, so which of them is rounded moves the small roped value by about 2^-17 around the midpoint
    and the e4m3 byte shows it.  cos and sin are bf16 of magnitude [0.25, 1) with random signs."""
    rng = np.random.default_rng([seed, launch])
    half = ROPE // 2
    c = rng.uniform(0.25, 1.0, half) * rng.choice([-1.0, 1.0], half)
    s = rng.uniform(0.25, 1.0, half) * rng.choice([-1.0, 1.0], half)
    cs = [bf16_word(v) for v in np.concatenate([c, s])]
    cf = np.array([np.float64(np.uint32(w << 16).view(np.float32)) for w in cs])
    b = np.zeros(D, dtype=np.float32)
    for p in range(half):
        small = (1 + (2 * int(rng.integers(8)) + 1) / 16) * 2.0 ** int(rng.integers(-5, -1)) * rng.choice([-1, 1])
        large = rng.uniform(64, 400) * rng.choice([-1, 1])
        m1, m2 = (small, large) if p % 2 == 0 else (large, small)
        cc, ss = cf[p], cf[half + p]
        den = cc * cc + ss * ss
        b[2 * p] = np.float32((cc * m1 + ss * m2) / den)
        b[2 * p + 1] = np.float32((cc * m2 - ss * m1) / den)
    b[ROPE] = 448.0
    return b.view(np.uint32).astype(int).tolist(), cs


def index_k_forms(b: np.ndarray, cs: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    """Which contraction of `roped = normed * cos + sign * partner * sin` (kernels.py:392) the probe's bytes are: A (the
    Definition's) = fma(x, cos, RN(+-partner sin)), B = fma(+-partner, sin, RN(x cos)), N = RN(x cos) + RN(+-partner sin), separately
    for even (L) and odd (H) elements; b (launches, D) f32 words, cs (launches, ROPE) bf16 words, got (launches, D) bytes."""
    from verity_vllm.program.kernels import derived_rows as DR
    from verity_vllm.program.kernels.fp8_twins import np_f32_to_e4m3_sat

    half = ROPE // 2
    x = np.asarray(b, dtype=np.uint32)[:, :ROPE]
    cos = DR.bf16_to_f32_words(np.asarray(cs)[:, :half])
    sin = DR.bf16_to_f32_words(np.asarray(cs)[:, half:])
    x1, x2 = x[:, 0::2], x[:, 1::2]
    forms = {"L_A": DR.fma_rn(x1, cos, DR.mul_rn(x2 ^ NEG, sin)), "L_B": DR.fma_rn(x2 ^ NEG, sin, DR.mul_rn(x1, cos)),
             "L_N": DR.add_rn(DR.mul_rn(x1, cos), DR.mul_rn(x2 ^ NEG, sin)),
             "H_A": DR.fma_rn(x2, cos, DR.mul_rn(x1, sin)), "H_B": DR.fma_rn(x1, sin, DR.mul_rn(x2, cos)),
             "H_N": DR.add_rn(DR.mul_rn(x2, cos), DR.mul_rn(x1, sin))}
    g = np.asarray(got, dtype=np.uint8)[:, :ROPE]
    want = {"L": g[:, 0::2], "H": g[:, 1::2]}
    out: dict[str, Any] = {"L": [], "H": [], "separating": {}}
    for k, v in forms.items():
        e = np_f32_to_e4m3_sat(DR.as_f32(v)).reshape(v.shape)
        if np.array_equal(e, want[k[0]]):
            out[k[0]].append(k)
        out["separating"][k] = int((e != np_f32_to_e4m3_sat(DR.as_f32(forms[k[0] + "_A"])).reshape(v.shape)).sum())
    return out


# ------------------------------------------------------------------------------------------------------------------- operands
def operands(seed: int) -> dict[str, Any]:
    """GLM-5.3's attention inputs at one TP8 rank, laid out as `DeepseekV32Attention.forward` makes them (attention.py:283-376):
    qkv_lora split [2048, 512, 64], wk_weights_proj's output split [128, 32], q [T, 8, 256] split [192, 64], ql_nope a transposed
    bmm output, index_q [T, 32, 128]; k_norm's f32 weight and bias, the two RMSNorm weights; the shared rotary cache."""
    import torch
    from vllm.model_executor.layers.rotary_embedding import get_rope

    gen = torch.Generator(device="cuda").manual_seed(seed)
    T, dev = SEQ + EXTRA, "cuda"

    def rows(*shape, lo=-3.0, hi=3.0):
        x = torch.randn(*shape, generator=gen, device=dev)
        return x * torch.exp2(torch.empty(shape[0], *([1] * (len(shape) - 1)), device=dev).uniform_(lo, hi, generator=gen))

    pos = torch.arange(T, device=dev, dtype=torch.int64)
    far = torch.randint(SEQ, MAXPOS, (EXTRA,), generator=gen, device=dev)
    far[-1] = MAXPOS - 1
    pos[SEQ:] = far
    perm = torch.randperm(NB, generator=gen, device=dev).to(torch.int32)
    bt = perm[:SEQ // BLOCK].reshape(1, -1).contiguous()
    tok = torch.arange(SEQ, device=dev)
    slots = torch.empty(T, dtype=torch.int64, device=dev)
    slots[:SEQ] = bt[0, tok // BLOCK].to(torch.int64) * BLOCK + tok % BLOCK
    free = (perm[SEQ // BLOCK:].to(torch.int64)[:, None] * BLOCK + torch.arange(BLOCK, device=dev)).reshape(-1)
    n_extra = EXTRA - EXTRA_UNSLOTTED
    slots[SEQ:SEQ + n_extra] = free[torch.randperm(free.numel(), generator=gen, device=dev)[:n_extra]]
    slots[SEQ + n_extra:] = -1
    qkv = rows(T, Q_LORA + KV_LORA + DR).to(torch.bfloat16)
    kwf = rows(T, D + H, lo=-2, hi=2)
    kwf[K_SPECIAL[0], :D] = 0
    kwf[K_SPECIAL[1], :D] *= 2.0 ** 40
    kwf[K_SPECIAL[2], :D] *= 2.0 ** -40
    kwf[K_SPECIAL[3], :D] *= 2.0 ** -130
    kwf[K_SPECIAL[4], :D] *= 2.0 ** -9
    kwf[K_SPECIAL[4], 7] = 2.0 ** 12
    kw = kwf.to(torch.bfloat16)
    iq = rows(T, H, D)
    t = Q_SPECIAL_TOKEN
    iq[t, 0] = 0
    iq[t, 1] *= 2.0 ** 40
    iq[t, 2] *= 2.0 ** -40
    iq[t, 3] *= 2.0 ** -130
    rope = get_rope(ROPE, MAXPOS, is_neox_style=False, rope_parameters={"rope_theta": THETA, "rope_type": "default"},
                    dtype=torch.bfloat16)
    o = {"pos": pos, "bt": bt, "slots": slots, "qkv": qkv, "kw": kw,
         "q_c": qkv[:, :Q_LORA], "kv_c": qkv[:, Q_LORA:Q_LORA + KV_LORA], "k_pe": qkv[:, Q_LORA + KV_LORA:],
         "index_k": kw[:, :D], "index_weights": kw[:, D:],
         "q_full": rows(T, NH, NOPE + DR).to(torch.bfloat16), "index_q": iq.to(torch.bfloat16).contiguous(),
         "ql_nope": rows(NH, T, DV).to(torch.bfloat16).transpose(0, 1),
         "k_w": (1 + 0.1 * torch.randn(D, generator=gen, device=dev)).float(), "k_b": (0.1 * torch.randn(D, generator=gen, device=dev)).float(),
         "q_rms_w": (1 + 0.1 * torch.randn(Q_LORA, generator=gen, device=dev)).to(torch.bfloat16),
         "kv_rms_w": (1 + 0.1 * torch.randn(KV_LORA, generator=gen, device=dev)).to(torch.bfloat16),
         "cs": rope.cos_sin_cache.to(dev), "k_scale": torch.ones(1, dtype=torch.float32, device=dev),
         "q_scale": torch.ones(1, dtype=torch.float32, device=dev)}
    o["q_pe"] = o["q_full"][..., NOPE:]
    return o


def caches(n_blocks: int, n_tokens: int) -> dict[str, Any]:
    """Fresh caches as the engine allocates them (indexer [blocks, 64, 132] bytes, MLA [blocks, 64, 576] bf16), every byte a sentinel
    (0xA5, 0x5A5A), and the top-k buffer [tokens, 2048] at 7."""
    import torch

    return {"idx": torch.full((n_blocks, BLOCK, STRIDE), 0xA5, dtype=torch.uint8, device="cuda"),
            "mla": torch.full((n_blocks, BLOCK, KV_LORA + DR), 0x5A5A, dtype=torch.int16, device="cuda").view(torch.bfloat16),
            "topk": torch.full((n_tokens, TOPK), 7, dtype=torch.int32, device="cuda")}


def norm_rope(o: dict[str, Any], c: dict[str, Any], cs_k, cs_i):
    """`fused_norm_rope` as attention.py:338-366 calls it (no PCP: index_k_out None); returns (q_c_out, kv_c_out, k_pe_out)."""
    import torch
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope

    kv_c_out, k_pe_out = torch.empty_like(o["kv_c"]), torch.empty_like(o["k_pe"])
    q_c_out = fused_norm_rope(o["pos"], o["q_c"], o["q_rms_w"], RMS_EPS, o["kv_c"], o["kv_rms_w"], RMS_EPS, o["k_pe"], cs_k, o["index_k"],
                              o["k_w"], o["k_b"], K_EPS, cs_i, c["topk"], slot_mapping=o["slots"], indexer_k_cache=c["idx"],
                              mla_kv_cache=c["mla"], mla_kv_cache_dtype="auto", mla_k_scale=o["k_scale"], has_indexer=True,
                              index_rope_interleave=True, kv_c_out=kv_c_out, k_pe_out=k_pe_out, index_k_out=None)
    return q_c_out, kv_c_out, k_pe_out


def fused_q_call(o: dict[str, Any], cs):
    """`fused_q` as attention.py:378-392 calls it with the bf16 query: (index_q_fp8, index_weights_out, q_pe_out)."""
    from vllm.models.deepseek_v32.common.kernels import fused_q

    return fused_q(o["pos"], o["q_pe"], cs, o["index_q"], cs, o["ql_nope"], o["q_scale"], o["index_weights"], D ** -0.5, H ** -0.5,
                   has_indexer=True, index_rope_interleave=True, quantize_mqa=False)


def _rms_info(x: np.ndarray, w: np.ndarray, got: np.ndarray, eps: float) -> dict[str, Any]:
    """Triton `_rms_norm` (kernels.py:25-31), no Definition here: the outputs against f64 sums of an f32 model, in bf16 ulps."""
    from verity_vllm.program.kernels import derived_rows as DR

    xf = DR.as_f32(DR.bf16_to_f32_words(x)).astype(np.float64)
    r = np.float32(1.0) / np.sqrt((np.mean(xf * xf, axis=1, keepdims=True) + eps).astype(np.float32))
    want = DR.f2fp_bf16(DR.as_u32((DR.as_f32(DR.bf16_to_f32_words(x)) * r) * DR.as_f32(DR.bf16_to_f32_words(w))))
    d = np.abs(want.astype(np.int64) - np.asarray(got, dtype=np.int64))
    return {"words": int(d.size), "equal": int((d == 0).sum()), "max_ulps": int(d.max())}


# ------------------------------------------------------------------------------------------------------------------- the phases
def selection() -> dict[str, Any]:
    import torch
    from vllm.platforms import current_platform
    from vllm.v1.attention.selector import AttentionSelectorConfig

    cfg = AttentionSelectorConfig(head_size=KV_LORA + DR, dtype=torch.bfloat16, kv_cache_dtype="auto", block_size=BLOCK, use_mla=True,
                                  use_sparse=True)
    valid, invalid = current_platform.get_valid_backends(current_platform.get_device_capability(), cfg, num_heads=NH)
    order = [c.backend.name for c in sorted(valid, key=lambda c: c.priority)]
    rec: dict[str, Any] = {"valid": order, "invalid": {b.name: r for b, r in invalid.items()}}
    try:
        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import is_fused_q_cutedsl_supported

        q_pe = torch.empty(1, NH, DR, dtype=torch.bfloat16, device="cuda")
        rec["fused_q_cutedsl"] = bool(is_fused_q_cutedsl_supported(q_pe, torch.empty(1, H, D, dtype=torch.bfloat16, device="cuda"),
                                                                   torch.empty(1, NH, DV, dtype=torch.bfloat16, device="cuda"),
                                                                   has_indexer=True, quantize_mqa=False))
    except Exception as e:  # noqa: BLE001
        rec["fused_q_cutedsl"] = _error(e)
    rec["ok"] = bool(order and order[0] == "FLASH_ATTN_MLA_SPARSE" and rec["fused_q_cutedsl"] is False)
    return rec


def phase_k(work: Path, o: dict[str, Any], quick: bool) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """The served `fused_norm_rope` launches (twice) and the index-K probe launches: records, k-slot jobs, and what later phases read."""
    import torch

    T = o["pos"].numel()
    c1, c2 = caches(NB, T), caches(NB, T)
    out1 = norm_rope(o, c1, o["cs"], o["cs"])
    out2 = norm_rope(o, c2, o["cs"], o["cs"])
    torch.cuda.synchronize()
    pos, slots = o["pos"].cpu().numpy(), o["slots"].cpu().numpy()
    idx = _u8(c1["idx"])
    mla = _u16(c1["mla"]).reshape(NB * BLOCK, KV_LORA + DR)
    kv_out, kpe_out = _u16(out1[1]), _u16(out1[2])
    written = np.flatnonzero(slots >= 0)
    q = np.stack([slot_bytes(idx, slots[t])[0] for t in written])
    s = np.array([slot_bytes(idx, slots[t])[1] for t in written], dtype=np.uint32)
    untouched = idx.reshape(-1)[~slot_mask(NB, slots[written])]
    mla_rows = mla[slots[written]]
    unwritten = np.setdiff1d(np.arange(NB * BLOCK), slots[written])
    cs_rows = _u16(o["cs"])[pos]
    rec = {"phase": "k", "case": "served", "tokens": int(T), "slots_written": int(written.size),
           "checks": {"deterministic": bool(torch.equal(c1["idx"], c2["idx"]) and torch.equal(c1["mla"].view(torch.int16), c2["mla"].view(torch.int16))
                                            and all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(out1, out2))),
                      "topk_cleared": bool((c1["topk"] == -1).all().item()),
                      "unwritten_untouched": bool((untouched == 0xA5).all() and (mla[unwritten] == 0x5A5A).all()),
                      "mla_rows": bool(np.array_equal(mla_rows, np.concatenate([kv_out[written], kpe_out[written]], axis=1)))},
           "info": {"q_c_rms": _rms_info(_u16(o["q_c"]), _u16(o["q_rms_w"]), _u16(out1[0]), RMS_EPS),
                    "kv_c_rms": _rms_info(_u16(o["kv_c"]), _u16(o["kv_rms_w"]), kv_out, RMS_EPS)}}
    w = np.broadcast_to(_u32(o["k_w"]), (written.size, D))
    b = np.broadcast_to(_u32(o["k_b"]), (written.size, D))
    paths = _save(work / "k_served", k=_u16(o["index_k"])[written], w=w, b=b, cs=cs_rows[written], q=q, s=s, tokens=written)
    jobs = [{"phase": "k", "case": "served", "kind": "kslots", "paths": paths, "rows": [r0, r1]}
            for r0, r1 in _chunks(written.size, JOB_SLOTS)]
    probe, pjobs = index_k_probe_launches(work, o)
    return [rec, probe], jobs + pjobs, {"caches": c1, "kv_c_out": out1[1], "k_pe_out": out1[2], "k_pe_probe": probe.pop("k_pe_out")}


def probe_operands(o: dict[str, Any], n: int, seed: int) -> dict[str, Any]:
    """The layout of `operands` at n tokens (positions 0..n-1, so the probe caches' rows are the tokens'), random values."""
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    p = dict(o)
    p["pos"] = torch.arange(n, device=dev, dtype=torch.int64)
    p["qkv"] = torch.randn(n, Q_LORA + KV_LORA + DR, generator=gen, device=dev).to(torch.bfloat16)
    p["kw"] = torch.randn(n, D + H, generator=gen, device=dev).to(torch.bfloat16)
    p["q_c"], p["kv_c"], p["k_pe"] = p["qkv"][:, :Q_LORA], p["qkv"][:, Q_LORA:Q_LORA + KV_LORA], p["qkv"][:, Q_LORA + KV_LORA:]
    p["index_k"], p["index_weights"] = p["kw"][:, :D], p["kw"][:, D:]
    p["q_full"] = torch.randn(n, NH, NOPE + DR, generator=gen, device=dev).to(torch.bfloat16)
    p["q_pe"] = p["q_full"][..., NOPE:]
    p["index_q"] = torch.randn(n, H, D, generator=gen, device=dev).to(torch.bfloat16)
    p["ql_nope"] = torch.randn(NH, n, DV, generator=gen, device=dev).to(torch.bfloat16).transpose(0, 1)
    return p


def rope_probe_inputs(n_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """The subnormal probe as kernel inputs: x [n_tokens, ROPE] bf16 words (pair p of token t is quadruple 32 t + p's (x1, x2)) and
    the probe rotary cache [n_tokens, ROPE] (row t: the quadruples' cos, then their sin)."""
    quads = rope_probe_quads()
    half = ROPE // 2
    quads = (quads * (-(-n_tokens * half // len(quads))))[:n_tokens * half]
    q = np.array(quads, dtype=np.uint16).reshape(n_tokens, half, 4)
    x = np.empty((n_tokens, ROPE), dtype=np.uint16)
    x[:, 0::2], x[:, 1::2] = q[..., 0], q[..., 1]
    return x, np.concatenate([q[..., 2], q[..., 3]], axis=1)


def index_k_probe_launches(work: Path, o: dict[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    """PROBE_LAUNCHES `fused_norm_rope` launches of PROBE_TOKENS tokens: index_k 0, k_norm weight 0 and launch j's bias and index-K
    cache row (`index_k_probe`, every row the same), k_pe the subnormal probe (`rope_probe_inputs`) on its cache; token t of launch j
    writes slot PROBE_TOKENS j + t of a one-block probe cache."""
    import torch

    n = PROBE_TOKENS
    x, cs_k = rope_probe_inputs(n)
    p = probe_operands(o, n, 4242)
    p["k_pe"].copy_(torch.from_numpy(x.view(np.int16)).to("cuda").view(torch.bfloat16))
    p["index_k"].zero_()
    p["k_w"] = torch.zeros(D, dtype=torch.float32, device="cuda")
    cs_k_t = torch.from_numpy(cs_k.view(np.int16)).to("cuda").view(torch.bfloat16).contiguous()
    c = caches(1, n)
    bs, css, kpe = [], [], []
    for j in range(PROBE_LAUNCHES):
        b, cs = index_k_probe(j)
        p["k_b"] = torch.from_numpy(np.array(b, dtype=np.uint32).view(np.float32)).to("cuda")
        cs_i = torch.from_numpy(np.array([cs] * n, dtype=np.uint16).view(np.int16)).to("cuda").view(torch.bfloat16)
        p["slots"] = torch.arange(n * j, n * j + n, dtype=torch.int64, device="cuda")
        c["topk"].fill_(7)
        _, _, k_pe_out = norm_rope(p, c, cs_k_t, cs_i)
        bs += [b] * n
        css += [cs] * n
        kpe.append(_u16(k_pe_out))
    torch.cuda.synchronize()
    idx = _u8(c["idx"])
    slots = np.arange(n * PROBE_LAUNCHES)
    q = np.stack([slot_bytes(idx, sl)[0] for sl in slots])
    s = np.array([slot_bytes(idx, sl)[1] for sl in slots], dtype=np.uint32)
    b_arr, cs_arr = np.array(bs, dtype=np.uint32), np.array(css, dtype=np.uint16)
    forms = index_k_forms(b_arr, cs_arr, q)
    rec = {"phase": "k", "case": "probe", "launches": PROBE_LAUNCHES, "slots_written": int(slots.size),
           "checks": {"scale_one": bool((s == 0x3F800000).all()), "k_pe_repeatable": all(np.array_equal(kpe[0], k) for k in kpe)},
           "info": {"index_k_forms": forms}, "k_pe_out": {"x": x, "cs": cs_k, "out": kpe[0]}}
    paths = _save(work / "k_probe", k=np.zeros((slots.size, D), dtype=np.uint16), w=np.zeros((slots.size, D), dtype=np.uint32),
                  b=b_arr, cs=cs_arr, q=q, s=s, tokens=slots)
    jobs = [{"phase": "k", "case": "probe", "kind": "kslots", "paths": paths, "rows": [r0, r1]} for r0, r1 in _chunks(slots.size, JOB_SLOTS)]
    return rec, jobs


def phase_q(work: Path, o: dict[str, Any], quick: bool) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """The served `fused_q` launch (twice) and the q_pe probe launch."""
    import torch

    r1 = fused_q_call(o, o["cs"])
    r2 = fused_q_call(o, o["cs"])
    torch.cuda.synchronize()
    T = o["pos"].numel()
    pos = o["pos"].cpu().numpy()
    tokens = np.arange(T)
    if quick:
        tokens = np.unique(np.concatenate([np.arange(8), np.random.default_rng(1).choice(T, QUICK_Q_TOKENS, replace=False)]))
    rec = {"phase": "q", "case": "served", "tokens": int(T), "compared": int(tokens.size),
           "checks": {"deterministic": all(torch.equal(a.view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a, b.view(torch.uint8)
                                                       if b.dtype == torch.float8_e4m3fn else b) for a, b in zip(r1, r2)),
                      "shapes": (list(r1[0].shape) == [T, H, D] and list(r1[1].shape) == [T, H] and r1[1].dtype == torch.float32
                                 and list(r1[2].shape) == [T, NH, DR] and r1[2].dtype == torch.bfloat16)}}
    paths = _save(work / "q_served", q=_u16(o["index_q"])[tokens], cs=_u16(o["cs"])[pos[tokens]], w=_u16(o["index_weights"])[tokens],
                  outq=_u8(r1[0])[tokens], outw=_u32(r1[1])[tokens], tokens=tokens)
    jobs = [{"phase": "q", "case": "served", "kind": "qtok", "paths": paths, "rows": [a, b]} for a, b in _chunks(tokens.size, JOB_Q)]
    x, cs = rope_probe_inputs(PROBE_TOKENS)
    p = probe_operands(o, PROBE_TOKENS, 4343)
    p["q_pe"].copy_(torch.from_numpy(np.repeat(x[:, None, :], NH, axis=1).view(np.int16)).to("cuda").view(torch.bfloat16))
    cs_t = torch.from_numpy(cs.view(np.int16)).to("cuda").view(torch.bfloat16).contiguous()
    pr = fused_q_call(p, cs_t)
    torch.cuda.synchronize()
    return [rec], jobs, {"index_q_fp8": r1[0], "index_weights_out": r1[1], "q_pe_out": r1[2],
                         "q_pe_probe": {"x": x, "cs": cs, "out": _u16(pr[2])}}


def phase_rope(o: dict[str, Any], kst: dict[str, Any], qst: dict[str, Any]) -> dict[str, Any]:
    """Which candidate contraction the k_pe and q_pe words are (served tokens and the subnormal probe), the reference
    `RoPEHeadGptJ_v1` on a sample of them against the L_a/H_a candidate words."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import ref_prims as RP

    pos = o["pos"].cpu().numpy()
    cs_rows = _u16(o["cs"])[pos]
    half = ROPE // 2
    out: dict[str, Any] = {}
    sets = {"k_pe": [(_u16(o["k_pe"]), cs_rows, _u16(kst["k_pe_out"])),
                     (kst["k_pe_probe"]["x"], kst["k_pe_probe"]["cs"], kst["k_pe_probe"]["out"])],
            "q_pe": [(_u16(o["q_pe"]).reshape(-1, ROPE), np.repeat(cs_rows, NH, axis=0), _u16(qst["q_pe_out"]).reshape(-1, ROPE)),
                     (np.repeat(qst["q_pe_probe"]["x"], NH, axis=0), np.repeat(qst["q_pe_probe"]["cs"], NH, axis=0),
                      qst["q_pe_probe"]["out"].reshape(-1, ROPE))]}
    fn = bind(RP.RoPEHeadGptJ, D=ROPE)
    for name, parts in sets.items():
        x = np.concatenate([p[0] for p in parts])
        cs = np.concatenate([p[1] for p in parts])
        got = np.concatenate([p[2] for p in parts])
        m = rope_matches(x[:, 0::2], x[:, 1::2], cs[:, :half], cs[:, half:], got[:, 0::2], got[:, 1::2])
        f = rope_forms(x[:, 0::2], x[:, 1::2], cs[:, :half], cs[:, half:])
        v1 = np.empty_like(got)
        v1[:, 0::2], v1[:, 1::2] = f["L_a"], f["H_a"]
        sample = sorted({0, 1, x.shape[0] - 1, parts[0][0].shape[0]} | set(range(parts[0][0].shape[0], x.shape[0], 7)))
        m["reference_rows"] = len(sample)
        m["twin_mismatch"] = sum(list(evaluate(fn, x[r].tolist(), cs[r].tolist())) != v1[r].tolist() for r in sample)
        m["probe_rows"] = int(parts[1][0].shape[0])
        m["settled"] = rope_settled(m)
        m["definition"] = rope_name(m)
        out[name] = m
    out["ok"] = all(out[k]["settled"] and out[k]["twin_mismatch"] == 0 and out[k]["reference_rows"] > 0 for k in ("k_pe", "q_pe"))
    return out


def phase_logits(work: Path, o: dict[str, Any], kst: dict[str, Any], qst: dict[str, Any], quick: bool):
    """The prefill gather, `fp8_fp4_mqa_logits` and `fp8_fp4_paged_mqa_logits` on the served cache and queries."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata
    from vllm.utils.platform_utils import num_compute_units

    cache = kst["caches"]["idx"]
    bt = o["bt"]
    k_quant = torch.empty(SEQ, D, dtype=torch.float8_e4m3fn, device="cuda")
    k_scale = torch.empty(SEQ, 4, dtype=torch.uint8, device="cuda")
    ops.cp_gather_indexer_k_quant_cache(cache, k_quant, k_scale, bt, torch.tensor([0, SEQ], dtype=torch.int32, device="cuda"))
    q = qst["index_q_fp8"][:SEQ]
    w = qst["index_weights_out"][:SEQ]
    ks = torch.zeros(SEQ, dtype=torch.int32, device="cuda")
    ke = torch.arange(1, SEQ + 1, dtype=torch.int32, device="cuda")
    logits = fp8_fp4_mqa_logits((q, None), (k_quant, k_scale.view(torch.float32).squeeze(-1)), w, ks, ke, clean_logits=False)
    logits2 = fp8_fp4_mqa_logits((q, None), (k_quant, k_scale.view(torch.float32).squeeze(-1)), w, ks, ke, clean_logits=False)
    B = len(DECODE_LENS)
    lens = torch.tensor(DECODE_LENS, dtype=torch.int32, device="cuda")
    q_dec = q[lens.long() - 1].reshape(B, 1, H, D)
    w_dec = w[lens.long() - 1].contiguous()
    seq_lens = lens.reshape(B, 1).contiguous()
    btd = bt.expand(B, -1).contiguous()
    sched = get_paged_mqa_logits_metadata(seq_lens, BLOCK, num_compute_units(torch.cuda.current_device()))
    dec = fp8_fp4_paged_mqa_logits((q_dec, None), cache.unsqueeze(-2), w_dec, seq_lens, btd, sched, max_model_len=MAX_MODEL_LEN,
                                   clean_logits=False)
    torch.cuda.synchronize()
    slots = o["slots"].cpu().numpy()[:SEQ]
    idx = _u8(cache)
    gq, gs = _u8(k_quant), _u32(k_scale.view(torch.float32)).reshape(-1)
    gather_ok = all(np.array_equal(gq[t], slot_bytes(idx, slots[t])[0]) and gs[t] == slot_bytes(idx, slots[t])[1] for t in range(SEQ))
    L = _u32(logits)
    Ld = _u32(dec)
    dec_eq = sum(int((Ld[b, :n] != L[n - 1, :n]).sum()) for b, n in enumerate(DECODE_LENS))
    same, past_window = logits_rerun(logits, logits2)
    recs = [{"phase": "logits", "case": "prefill", "shape": list(logits.shape),
             "checks": {"gather": bool(gather_ok), "deterministic": same,
                        "shape": list(logits.shape) == [SEQ, SEQ] and logits.dtype == torch.float32},
             "info": {"words_past_window_differing": past_window}},
            {"phase": "logits", "case": "decode", "shape": list(dec.shape), "lens": list(DECODE_LENS),
             "checks": {"shape": list(dec.shape) == [B, MAX_MODEL_LEN] and dec.dtype == torch.float32},
             "info": {"words_differing_from_prefill": dec_eq}}]
    rng = np.random.default_rng(7)
    rows = sorted(set(FULL_ROWS) | set(rng.choice(SEQ, 2 if quick else RANDOM_ROWS, replace=False).tolist()))
    r_full = np.concatenate([np.full(r + 1, r) for r in rows])
    c_full = np.concatenate([np.arange(r + 1) for r in rows])
    n_pairs = RANDOM_PAIRS // (10 if quick else 1)
    r_rand = rng.integers(0, SEQ, n_pairs)
    c_rand = (rng.random(n_pairs) * (r_rand + 1)).astype(np.int64)
    rr, cc = np.concatenate([r_full, r_rand]), np.concatenate([c_full, c_rand])
    r_dec = np.concatenate([np.full(n, n - 1) for n in DECODE_LENS])
    c_dec = np.concatenate([np.arange(n) for n in DECODE_LENS])
    g_dec = np.concatenate([Ld[b, :n] for b, n in enumerate(DECODE_LENS)])
    paths = _save(work / "logits", q=_u8(q), w=_u32(w), k=gq, ks=gs, r=rr, c=cc, got=L[rr, cc], rd=r_dec, cd=c_dec, gotd=g_dec)
    jobs = [{"phase": "logits", "case": "prefill", "kind": "logits", "paths": paths, "keys": ["r", "c", "got"], "range": [a, b]}
            for a, b in _chunks(rr.size, JOB_LOGITS)]
    jobs += [{"phase": "logits", "case": "decode", "kind": "logits", "paths": paths, "keys": ["rd", "cd", "gotd"], "range": [a, b]}
             for a, b in _chunks(r_dec.size, JOB_LOGITS)]
    return recs, jobs, {"logits": logits}


def logits_rerun(a, b) -> tuple[bool, int]:
    """Two prefill logits runs (ks = 0, ke = r + 1, clean_logits=False): whether they agree on every word of each row's window [0, r],
    the words the kernel writes, and how many words past the windows differ (unwritten: only clean_logits sets them, to -inf,
    deep_gemm.py:542)."""
    import torch

    window = torch.ones(a.shape, dtype=torch.bool, device=a.device).tril()
    diff = a.view(torch.int32) != b.view(torch.int32)
    return not bool(diff[window].any()), int(diff[~window].sum())


def _clear_launch_error() -> None:
    """A launch that fails leaves its error pending for the next CUDA call to report: take it here, not in the next kernel's case."""
    import torch

    try:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _topk_rows(gen_seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic decode rows of width MAX_MODEL_LEN: random values over TOPK_LENS keys (past each length +inf, which no kernel may
    read), a row of 3000 equal values, a row of 2100 keys with 2000 ones and the rest zeros, and a length of -3."""
    rng = np.random.default_rng(gen_seed)
    lens = list(TOPK_LENS) + [3000, 2100, -3]
    names = [f"len{n}" for n in TOPK_LENS] + ["ties_all_equal", "ties_2000_ones", "negative"]
    vals = rng.standard_normal((len(lens), MAX_MODEL_LEN)).astype(np.float32)
    vals[len(TOPK_LENS)] = 1.0
    vals[len(TOPK_LENS) + 1] = 0.0
    vals[len(TOPK_LENS) + 1, rng.choice(2100, 2000, replace=False)] = 1.0
    for i, n in enumerate(lens):
        vals[i, max(n, 0):] = np.inf
    return vals, np.array(lens, dtype=np.int32), names


def _check_topk(vals: np.ndarray, lens: np.ndarray, out: np.ndarray, start: int = 0) -> dict[str, Any]:
    """Rows of at most TOPK keys against `topk_short` (lengths clamped at 0); longer rows `topk_valid`."""
    lens = np.asarray(lens, dtype=np.int64)
    short = lens <= TOPK
    want = topk_short(np.maximum(lens[short], 0))
    long_rows = [topk_valid(vals[i, start:], int(lens[i]), out[i]) for i in np.flatnonzero(~short)]
    return {"short_rows": int(short.sum()), "short_mismatch": int((want != out[short]).any(axis=1).sum()),
            "long_rows": len(long_rows), "long_valid": sum(r["valid"] for r in long_rows),
            "long_ascending": sum(r["ascending"] for r in long_rows), "long_with_ties": sum(bool(r["ties_left_out"]) for r in long_rows),
            "long": long_rows}


def phase_topk(o: dict[str, Any], lst: dict[str, Any]) -> dict[str, Any]:
    """vLLM's top-k kernels: prefill on the served logits (twice) and on synthetic rows from row start 5; the decode kernels on
    synthetic rows; the reference `DsaTopKShortRow_v1{2048}` on the lengths the rows take."""
    import torch
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.sparse_attn_indexer import RADIX_TOPK_WORKSPACE_SIZE
    from vllm.platforms import current_platform
    from verity_vllm.program.registry import dsa

    rec: dict[str, Any] = {"cases": {}, "errors": {}}
    if lst is None:
        rec["errors"]["prefill_served"] = "no served logits (the logits phase failed)"
    else:
        logits = lst["logits"]
        ks = torch.zeros(SEQ, dtype=torch.int32, device="cuda")
        ke = torch.arange(1, SEQ + 1, dtype=torch.int32, device="cuda")
        outs = []
        for _ in range(2):
            out = torch.full((SEQ, TOPK), -1, dtype=torch.int32, device="cuda")
            ops.top_k_per_row_prefill(logits, ks, ke, out, SEQ, logits.stride(0), logits.stride(1), TOPK)
            outs.append(out.cpu().numpy())
        served = _check_topk(logits.cpu().numpy(), np.arange(1, SEQ + 1), outs[0])
        served["deterministic_short"] = bool(np.array_equal(outs[0][:TOPK], outs[1][:TOPK]))
        served["deterministic_long"] = bool(np.array_equal(outs[0][TOPK:], outs[1][TOPK:]))
        rec["cases"]["prefill_served"] = served
        rec["prefill_rows"] = outs[0]
    vals, lens, names = _topk_rows(11)
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
    rec["cases"]["prefill_start5"] = _check_topk(wide, np.maximum(lens, 0), pout.cpu().numpy(), start)
    dl = torch.from_numpy(vals).to("cuda")
    seq_lens = torch.from_numpy(lens).to("cuda").reshape(rows, 1).contiguous()
    # sparse_attn_indexer.py:617-623 at k = 2048, rows <= 64 and stride % 4 == 0 (both hold here)
    rec["cooperative_served"] = cooperative = bool(current_platform.has_device_capability(90)
                                                    and not current_platform.is_device_capability_family(120))
    kernels = [("persistent", rows, lambda d, sl, out, ws, m: torch.ops._C.persistent_topk(d, sl, out, ws, TOPK, d.shape[1]))]
    if cooperative:
        # on sm_90, 10 rows take clusters of 4 and 8 rows clusters of 8 (cooperative_topk.cu:94-102)
        kernels += [(name, n, lambda d, sl, out, ws, m: torch.ops._C.cooperative_topk(d, sl, out, ws, TOPK, m))
                    for name, n in (("cooperative", rows), ("cooperative_8rows", 8))]
    kernels.append(("decode_fallback", rows, lambda d, sl, out, ws, m: ops.top_k_per_row_decode(d, 1, sl, out, d.shape[0], d.stride(0),
                                                                                                d.stride(1), TOPK)))
    for name, n, call in kernels:
        try:
            outs = []
            for _ in range(2):
                out = torch.full((n, TOPK), -1, dtype=torch.int32, device="cuda")
                call(dl[:n], seq_lens[:n], out, torch.zeros(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device="cuda"),
                     int(lens[:n].max()))
                outs.append(out.cpu().numpy())
            r = _check_topk(vals[:n], lens[:n], outs[0])
            r["deterministic_short"] = bool(np.array_equal(outs[0][lens[:n] <= TOPK], outs[1][lens[:n] <= TOPK]))
            rec["cases"][name] = r
        except Exception as e:  # noqa: BLE001
            rec["errors"][name] = _error(e)
            _clear_launch_error()
    ref_lens = sorted({0, 1, 2, TOPK - 1, TOPK, -3})
    fn = bind(dsa.DsaTopKShortRow, TOPK=TOPK)
    rec["reference_lengths"] = ref_lens
    rec["reference_ok"] = all(list(evaluate(fn, n & 0xFFFFFFFF)) == (topk_short(np.array([max(n, 0)]))[0].astype(np.int64)
                                                                     & 0xFFFFFFFF).tolist() for n in ref_lens)
    gating = ("prefill_served", "prefill_start5", "persistent") + (("cooperative", "cooperative_8rows") if cooperative else ())
    rec["ok"] = bool(rec["reference_ok"] and all(g in rec["cases"] and rec["cases"][g]["short_rows"] > 0
                                                 and rec["cases"][g]["short_mismatch"] == 0 for g in gating)
                     and rec["cases"]["prefill_served"]["deterministic_short"])
    return rec


def phase_attn(work: Path, o: dict[str, Any], kst: dict[str, Any], qst: dict[str, Any], topk: dict[str, Any], quick: bool):
    """FA3's sparse decode on ATTN_TOKENS' prefill top-k rows, as `forward_mqa` calls it: through its module's
    ``flash_attn_varlen_func`` under the serving switch (`engine.sparse_mla`), which pins one split.  vLLM's own heuristic on the
    same call is recorded beside it (``heuristic_differs``), not checked."""
    import torch
    from vllm.v1.attention.backends.mla import flashattn_mla_sparse as FMS
    from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view, triton_convert_req_index_to_global_index

    from verity_vllm.engine import sparse_mla

    pin = sparse_mla.install(FMS)

    toks = np.array(QUICK_ATTN_TOKENS if quick else ATTN_TOKENS)
    S = toks.size
    rows = np.ascontiguousarray(topk["prefill_rows"][toks])
    mla = kst["caches"]["mla"]
    kv_rows, bsr = flat_kv_row_view(mla, BLOCK)
    topk_t = torch.from_numpy(rows).to("cuda")
    req = torch.zeros(S, dtype=torch.int32, device="cuda")
    glob_t, valid_t = triton_convert_req_index_to_global_index(req, o["bt"], topk_t, BLOCK_SIZE=BLOCK, BLOCK_STRIDE_ROWS=bsr,
                                                                NUM_TOPK_TOKENS=TOPK, return_valid_counts=True)
    gen = torch.Generator(device="cuda").manual_seed(99)
    q_nope = torch.randn(NH, S, DV, generator=gen, device="cuda").to(torch.bfloat16).transpose(0, 1)
    q_rope = qst["q_pe_out"][torch.from_numpy(toks).to("cuda")]
    k_cache = kv_rows[:, KV_LORA:].unsqueeze(1).unsqueeze(1)
    v_cache = kv_rows[:, :KV_LORA].unsqueeze(1).unsqueeze(1)
    cu = torch.arange(0, S + 1, dtype=torch.int32, device="cuda")

    def fa3(fn):
        return fn(q=q_rope, k=k_cache, v=v_cache, q_v=q_nope, max_seqlen_q=1, cu_seqlens_q=cu, max_seqlen_k=TOPK, seqused_k=valid_t,
                  block_table=glob_t, softmax_scale=SCALE, causal=True, fa_version=3)

    out1, out2 = fa3(FMS.flash_attn_varlen_func), fa3(FMS.flash_attn_varlen_func)
    heuristic = fa3(pin.original) if pin is not None else None
    torch.cuda.synchronize()
    want_g, want_n = convert_rows(rows, o["bt"].cpu().numpy()[0], bsr)
    g, n = glob_t.cpu().numpy(), valid_t.cpu().numpy()
    kvr = _u16(kv_rows)
    qn, qr, out = _u16(q_nope), _u16(q_rope), _u16(out1)
    rec = {"phase": "attn", "case": "fa3_sparse", "tokens": toks.tolist(), "valid": n.tolist(), "block_stride_rows": int(bsr),
           "switch": {sparse_mla.ENV: os.environ.get(sparse_mla.ENV), "num_splits": sparse_mla.num_splits()},
           "heuristic_differs": None if heuristic is None else int((_u16(heuristic) != out).sum()),
           "checks": {"conversion": bool(np.array_equal(g, want_g) and np.array_equal(n, want_n)),
                      "deterministic": bool(torch.equal(out1.view(torch.int16), out2.view(torch.int16))),
                      "shape": list(out1.shape) == [S, NH, DV] and out1.dtype == torch.bfloat16,
                      "one_split": sparse_mla.num_splits() == 1}}
    jobs = []
    for i, t in enumerate(toks):
        paths = _save(work / f"attn_{t}", qn=qn[i], qr=qr[i], kv=kvr[g[i, :n[i]]], out=out[i])
        jobs += [{"phase": "attn", "case": f"token{t}", "kind": "attn", "paths": paths, "T": int(n[i]), "head": h} for h in range(NH)]
    return [rec], jobs


# ------------------------------------------------------------------------------------------------------------------ comparisons
def compare_kslots(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Cache slots [r0, r1): their bytes and scale against `DsaIndexerKRow_v1{128,64,1e-06}` (the reference itself)."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsa

    a = _load(job["paths"])
    fn = bind(dsa.DsaIndexerKRow, D=D, ROPE=ROPE, EPS=K_EPS)
    for r in range(*job["rows"]):
        y = np.array(evaluate(fn, a["k"][r].tolist(), a["w"][r].tolist(), a["b"][r].tolist(), a["cs"][r].tolist()), dtype=np.int64)
        got = np.concatenate([np.asarray(a["q"][r], dtype=np.int64), [int(a["s"][r])]])
        bad = np.flatnonzero(y != got)
        rec["words"] += D + 1
        rec["reference"] += 1
        rec["mismatch"] += int(bad.size)
        if bad.size and rec["first_mismatch"] is None:
            rec["first_mismatch"] = {"token": int(a["tokens"][r]), "word": int(bad[0]), "want": int(y[bad[0]]), "kernel": int(got[bad[0]])}


def compare_qtok(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Tokens [r0, r1): index_q_fp8 and index_weights_out against `DsaIndexerQ_v1{32,128,64}` (the reference itself)."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsa

    a = _load(job["paths"])
    fn = bind(dsa.DsaIndexerQ, H=H, D=D, ROPE=ROPE)
    for r in range(*job["rows"]):
        y = np.array(evaluate(fn, np.asarray(a["q"][r]).reshape(-1).tolist(), a["cs"][r].tolist(), a["w"][r].tolist()), dtype=np.int64)
        got = np.concatenate([np.asarray(a["outq"][r], dtype=np.int64).reshape(-1), np.asarray(a["outw"][r], dtype=np.int64)])
        bad = np.flatnonzero(y != got)
        rec["words"] += H * D + H
        rec["reference"] += 1
        rec["mismatch"] += int(bad.size)
        if bad.size and rec["first_mismatch"] is None:
            rec["first_mismatch"] = {"token": int(a["tokens"][r]), "word": int(bad[0]), "want": int(y[bad[0]]), "kernel": int(got[bad[0]])}


def compare_logits(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Entries [a, b) of the prefill or decode logits: the twin against the kernel's words, the reference on the first two."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsa

    a = _load(job["paths"])
    kr, kc, kg = job["keys"]
    lo, hi = job["range"]
    r, c = np.asarray(a[kr][lo:hi]), np.asarray(a[kc][lo:hi])
    got = np.asarray(a[kg][lo:hi], dtype=np.uint32)
    q, w, k, ks = np.asarray(a["q"])[r], np.asarray(a["w"])[r], np.asarray(a["k"])[c], np.asarray(a["ks"])[c]
    want, nonfin = logit_rows(q, w, k, ks)
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


def compare_attn(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """One (token, head) of FA3's sparse output against `SparseMlaHead_v1{T,...}` (the reference itself)."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import sparse_mla as M

    a = _load(job["paths"])
    h, T = job["head"], job["T"]
    fn = bind(M.SparseMlaHead, T=T, **{k: v for k, v in M.SPARSE_MLA_GLM53.items() if k != "NH"})
    y = np.array(evaluate(fn, a["qn"][h].tolist(), a["qr"][h].tolist(), np.asarray(a["kv"]).reshape(-1).tolist()), dtype=np.int64)
    got = np.asarray(a["out"][h], dtype=np.int64)
    bad = np.flatnonzero(y != got)
    rec["words"] += DV
    rec["reference"] += 1
    rec["mismatch"] += int(bad.size)
    if bad.size and rec["first_mismatch"] is None:
        rec["first_mismatch"] = {"head": h, "T": T, "word": int(bad[0]), "want": int(y[bad[0]]), "kernel": int(got[bad[0]])}


COMPARE = {"kslots": compare_kslots, "qtok": compare_qtok, "logits": compare_logits, "attn": compare_attn}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    try:
        COMPARE[job["kind"]](job, rec)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "error": _error(e)}
    return {"phase": job["phase"], "case": job["case"], **rec}


# --------------------------------------------------------------------------------------------------------------- ptx and bind
def ptx(triton_cache: Path, outdir: Path) -> dict[str, Any]:
    """The Triton cache's PTX of the two kernels: ok when each has exactly one specialization (every launch here, the probes'
    included, ran the served kernel).  Each file's sha256 and arithmetic instruction counts; the PTX, and the SASS where
    Triton's bundled `cuobjdump` disassembles the cubin, are copied to `<out>/ptx/`."""
    dest = outdir / "ptx"
    dest.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"kernels": {}}
    tool = None
    try:
        import triton

        cand = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump"
        tool = str(cand) if cand.exists() else shutil.which("cuobjdump")
    except Exception:  # noqa: BLE001
        tool = shutil.which("cuobjdump")
    for kernel in KERNELS:
        files = sorted(glob.glob(str(triton_cache / "**" / f"{kernel}*.ptx"), recursive=True))
        entries = []
        for f in files:
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
    """The Programs bind at GLM-5.3's statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_rope as BR
    from verity_vllm.program.registry import dsa
    from verity_vllm.program.registry import sparse_mla as M

    mla = {k: v for k, v in M.SPARSE_MLA_GLM53.items() if k != "NH"}
    targets = {f"DsaIndexerKRow_v1{{D={D},ROPE={ROPE},EPS={K_EPS}}}": lambda: bind(dsa.DsaIndexerKRow, D=D, ROPE=ROPE, EPS=K_EPS),
               f"DsaIndexerQ_v1{{H={H},D={D},ROPE={ROPE}}}": lambda: bind(dsa.DsaIndexerQ, H=H, D=D, ROPE=ROPE),
               f"DsaMqaLogit_v3{{H={H},D={D}}}": lambda: bind(dsa.DsaMqaLogitV3, H=H, D=D),
               f"DsaMqaLogits_v3{{N=64,H={H},D={D}}}": lambda: bind(dsa.DsaMqaLogitsV3, N=64, H=H, D=D),
               f"DsaTopKShortRow_v1{{TOPK={TOPK}}}": lambda: bind(dsa.DsaTopKShortRow, TOPK=TOPK),
               f"RoPEHeadGptJ_v2{{D={DR}}}": lambda: bind(BR.RoPEHeadGptJV2, D=DR),
               f"RoPEGptJ_v2{{NHEADS={NH},D={DR}}}": lambda: bind(BR.RoPEGptJV2, NHEADS=NH, D=DR)}
    for T in (1, 64, 65, TOPK):
        targets[f"SparseMlaHead_v1{{T={T},...}}"] = lambda T=T: bind(M.SparseMlaHead, T=T, **mla)
    targets[f"SparseMla_v1{{T={TOPK},NH={NH},...}}"] = lambda: bind(M.SparseMla, T=TOPK, **M.SPARSE_MLA_GLM53)
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
    doc["selection"], doc["launches"], doc["rope"], doc["topk"]; returns the comparison jobs."""
    import torch

    jobs: list[dict[str, Any]] = []
    st: dict[str, Any] = {}

    def step(name: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            print(f"[glm53-dsa] {name} ERROR {doc['errors'][name][:600]}", flush=True)
            return None

    doc["selection"] = step("selection", selection) or {}
    print(f"[glm53-dsa] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
    doc["stack"] = step("stack", stack) or {}
    o = step("operands", lambda: operands(20261005))
    for name, fn in (("k", lambda: phase_k(work, o, quick)), ("q", lambda: phase_q(work, o, quick))):
        r = step(name, fn)
        if r is not None:
            doc["launches"] += r[0]
            jobs += r[1]
            st[name] = r[2]
            for x in r[0]:
                print(f"[glm53-dsa] {name} {x['case']:8s} {json.dumps(x['checks'])} {json.dumps(x.get('info', {}))[:600]}", flush=True)
    doc["rope"] = step("rope", lambda: phase_rope(o, st["k"], st["q"])) or {"ok": False}
    r = step("logits", lambda: phase_logits(work, o, st["k"], st["q"], quick))
    if r is not None:
        doc["launches"] += r[0]
        jobs += r[1]
        st["logits"] = r[2]
        for x in r[0]:
            print(f"[glm53-dsa] logits {x['case']:8s} {json.dumps(x['checks'])} {json.dumps(x.get('info', {}))}", flush=True)
    topk = step("topk", lambda: phase_topk(o, st.get("logits")))
    if topk is not None and "prefill_rows" in topk:
        st["topk"] = {"prefill_rows": topk.pop("prefill_rows")}
    doc["topk"] = topk or {"ok": False}
    brief = {k: {f: v for f, v in g.items() if f != "long"} for k, g in doc["topk"].get("cases", {}).items()}
    print(f"[glm53-dsa] topk ok={doc['topk'].get('ok')} {json.dumps(brief)[:1500]} "
          f"errors={list(doc['topk'].get('errors', {}))}", flush=True)
    r = step("attn", lambda: phase_attn(work, o, st["k"], st["q"], st["topk"], quick))
    if r is not None:
        doc["launches"] += r[0]
        jobs += r[1]
        print(f"[glm53-dsa] attn {json.dumps(r[0][0]['checks'])} valid={r[0][0]['valid']}", flush=True)
    del st, o
    torch.cuda.empty_cache()
    return jobs


def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["checks"]) and all(v is True for v in x["checks"].values())


def _words_ok(t: dict[str, Any]) -> bool:
    return t["words"] > 0 and t["mismatch"] == 0 and t["unresolved"] == 0 and t["reference"] > 0 and t["twin_mismatch"] == 0


PHASES = ("k", "q", "logits", "attn")
CASES = (("k", "served"), ("k", "probe"), ("q", "served"), ("logits", "prefill"), ("logits", "decode"), ("attn", None))


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each compared case's verdict (k/served, k/probe, q/served, logits/prefill, logits/decode, attn): its launch checks, its
    words, and its errors, so a NO-GO names what failed."""
    out: dict[str, dict[str, Any]] = {}
    for phase, case in CASES:
        key = f"{phase}/{case}" if case else phase
        launches = [x for x in doc["launches"] if x["phase"] == phase and (case is None or x["case"] == case)]
        comps = [r for r in doc.get("comparisons", []) if r["phase"] == phase and (case is None or r["case"] == case)]
        t = {k: sum(r[k] for r in comps) for k in TOTALS}
        errors = [e for e in doc["errors"] if e == phase or e.startswith(f"compare/{phase}/")
                  and (case is None or e.startswith(f"compare/{phase}/{case}"))]
        out[key] = {"launches_ok": bool(launches) and all(_launch_ok(x) for x in launches), "totals": t, "errors": errors,
                    "ok": bool(launches) and all(_launch_ok(x) for x in launches) and _words_ok(t) and not errors}
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    return bool(doc["gate"]["ok"] and not doc["errors"] and doc["selection"].get("ok") is True and doc["ptx"].get("ok") is True
                and doc["bind"].get("ok") is True and doc["rope"].get("ok") is True and doc["topk"].get("ok") is True
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
    print(f"[glm53-dsa] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-DSA GATE-FAIL", flush=True)
        raise SystemExit(3)
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        jobs = gpu(work, outdir, quick, doc)
    for k in ("selection", "rope", "topk"):
        doc.setdefault(k, {"ok": False})
    rope = doc["rope"]
    print(f"[glm53-dsa] rope {json.dumps({k: v for k, v in rope.items() if k != 'ok'})[:2000]}", flush=True)
    try:
        doc["ptx"] = ptx(work / "triton_cache", outdir)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["ptx"] = _error(e)
        doc["ptx"] = {"ok": False}
    print(f"[glm53-dsa] ptx {json.dumps(doc['ptx'])[:2000]}", flush=True)
    try:
        doc["bind"] = bind_check()
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    print(f"[glm53-dsa] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
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
        print(f"[glm53-dsa] words {r['phase']:6s} {r['case']:10s} words={r['words']} !={r['mismatch']} unresolved={r['unresolved']} "
              f"ref={r['reference']} twin!={r['twin_mismatch']} first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[glm53-dsa] case {key:16s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} "
              f"!={t['mismatch']} unresolved={t['unresolved']} ref={t['reference']} twin!={t['twin_mismatch']} errors={c['errors']}",
              flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    names = {k: rope.get(k, {}).get("definition") for k in ("k_pe", "q_pe")}
    forms = {k: (rope.get(k, {}).get("L"), rope.get(k, {}).get("H")) for k in ("k_pe", "q_pe")}
    print(f"ROPE k_pe={names['k_pe']} q_pe={names['q_pe']} forms={json.dumps(forms)} settled={rope.get('ok')}", flush=True)
    print(f"GLM53-DSA {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's sparse-attention indexer and sparse MLA on this Hopper GPU against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and ptx/")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, Triton and DeepGEMM caches)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true",
                    help=f"q on {QUICK_Q_TOKENS} tokens, fewer logits, attention on tokens {QUICK_ATTN_TOKENS}: a first signal in minutes")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
