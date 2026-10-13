"""DeepSeek-V4-Pro's head-512 compressor on Hopper against its Programs: the H200's go/no-go for binding `dsv4_compressor`'s
Definitions (vLLM d9105ea80's CuTe DSL `compress_norm_rope_store_cutedsl`, which writes the compressed rows FlashMLA's decode and
prefill read) and `dsv4_indexer.Dsv4CompressorScore_v1{W}` (the state score `save_partial_states` writes before it) at
DeepSeek-V4-Pro's statics on one TP8 rank: a ratio-4 layer (state width 1024, the 8-row overlapping window, one fused kernel) and
a ratio-128 layer (state width 512, the 128-row window, the split compress and store), each into an `fp8_ds_mla` cache.  Line
numbers under `vllm/models/deepseek_v4/` unless named ("cutedsl" is nvidia/ops/sparse_attn_compress_cutedsl.py).

One GPU process writes `<out>/dsv4_compressor_capture.json`, the property record `dsv4_compressor_capture` (`properties.record`),
and the three CuTe DSL kernels' PTX and cubin (and SASS where `cuobjdump` runs) under `<out>/ptx/`; it prints one line per launch
and per case, then `DSV4-COMPRESSOR GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first and nothing else
runs when it fails (exit 3).  A layer that raises records its error and fails its cases without stopping the other.  The operands
are random at the served shapes (`operands`), so no model weights are read.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `_select_dsv4_attn_cls` picks `DeepseekV4FlashMLAAttention`, whose cache is `fp8_ds_mla` (`use_fp8_ds_mla_layout`,
             attention.py:148, so `store_full_kv` is False, compressor.py:390-399), the platform is CUDA and `has_cutedsl()` holds
             (head 512 then takes `compress_norm_rope_store_cutedsl`, compressor.py:403-414), the installed nvidia-cutlass-dsl is
             the 4.6.2 vLLM pins (requirements/cuda.txt) and whose PTX was read, and the installed SOURCES are the files read,
             since the launches call the kernels' wrappers directly rather than through `DeepseekCompressor.forward`.
  launches   per layer (`LAYERS`), as `forward` makes them (compressor.py:354-454): `save_partial_states` (state block 4 or 8,
             state width coff * 512) then `compress_norm_rope_store_cutedsl` (head 512, rope 64, quant block 64, token stride 576,
             8 scale bytes, the cache (nb, page block, 584) uint8 of pages `page_bytes(page block)`), twice on fresh caches, over
             three requests in one launch: request 0 from position 0; request 1 from a later position, the state rows before it
             written earlier, its operands the edge words (`EDGE`); request 2 tokens the kernels must skip (no state slot, no
             cache slot, or a cache slot off a boundary).  The state's pages are padded past their rows (stride(0) STATE_PAGE
             words), as the state shares the KV cache's pages (compressor.py:154-160), so the kernels' strided reads are checked.
  score      each written state row's score half against `Dsv4CompressorScore_v1{W}` of its score row and `ape[pos % ratio]`
             (twin `flashmla_twin.fadd`, the reference on a sample), its kv half a copy, every other word untouched.
  compress   every written cache row (448 E4M3 NoPE bytes, 64 bf16 RoPE words, 7 UE8M0 bytes and the pad 0) against
             `dsv4_compressor_twin.compress_c4` / `compress_c128` of the state rows the kernel reads (`state_rows`: on ratio 4 the
             rows at positions below 0 skipped, so position 3's window is 4 rows), the bf16 norm weight and the rotary row
             (pos // ratio) * ratio; the reference `Dsv4MlaCompressC4_v1{ROWS,...}` / `Dsv4MlaCompressC128_v1{8,16,...}` on every
             entry (every 16th served one under --quick); the state unchanged, every other byte of the cache untouched.
  columns    ratio 128 only: `split_kv_compress_norm_rope_insert_sparse_attn_cutedsl` as the served call makes it, with its f32
             `compressed_kv` filled with a sentinel so it can be read: every column of every token the compress kernel takes
             against `dsv4_compressor_twin.c128_columns` (the reference `Dsv4MlaC128Column_v1{8,16}` on every column of every
             token, of every 16th served one under --quick), every other row the sentinel, and the cache it writes the served
             call's, byte for byte.
  ptx        each of the three kernels compiled once (their `compile` caches hold one entry), its PTX and cubin kept
             (`environment`'s CUTE_DSL_KEEP), the PTX for sm_90a and its arithmetic instruction counts PTX_COUNTS, the ones
             the Definitions were read from.
  bind       the Programs bind at the served statics.

GO needs every case's words compared with 0 mismatches, the references with 0 disagreements with the twins, every launch check,
the selection, the PTX and the bindings.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/dsv4_compressor_capture_gpu.py --out DIR [--scratch DIR] [--quick]
"""
from __future__ import annotations

import argparse
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
from typing import Any, NamedTuple

import numpy as np

from tests.properties import dsv4_flashmla_capture_gpu as FX
from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.program.kernels import dsv4_compressor_twin as T
from verity_vllm.program.kernels import flashmla_twin as FT
from verity_vllm.program.registry import dsv4_compressor as R
from verity_vllm.program.registry import dsv4_indexer as I
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/dsv4-compressor-capture/v1"
RECORD = "dsv4_compressor_capture.json"
#: DeepSeek-V4-Pro (config.json at b5968e91): head 512 = NoPE 448 | RoPE 64, one UE8M0 scale per 64, `rms_norm_eps`
D, QB, EPS = (R.DSV4_COMPRESSOR[k] for k in ("D", "QB", "EPS"))
ROPE, NOPE = QB, D - QB
W128, R128 = R.DSV4_COMPRESSOR["W"], R.DSV4_COMPRESSOR["R"]
#: the `fp8_ds_mla` slot: 576 value bytes (token_stride) and 8 scale bytes (scale_dim), compressor.py:288-293
DATA_BYTES, SCALE_BYTES, TOKEN_BYTES = FX.DATA_BYTES, FX.SCALE_BYTES, FX.TOKEN_BYTES
WORDS = NOPE + ROPE + SCALE_BYTES
page_bytes = FX.page_bytes


class Layer(NamedTuple):
    ratio: int
    width: int    # the state width coff * 512: kv | score each this wide
    block: int    # state rows per state block (compressor.py:162-165)
    window: int   # state rows a compressed row reads
    pbs: int      # cache rows per page: 256 positions / ratio (the decode's `LAYERS`)


LAYERS = {"c4": Layer(4, 2 * D, 4, 8, FX.LAYERS["c4"][0]), "c128": Layer(128, D, 8, 128, FX.LAYERS["c128"][0])}
#: each state block's page in f32 words: a ratio-4 KV page (`page_bytes(64)` / 4), past both layers' block rows
STATE_PAGE = page_bytes(64) // 4
#: per layer: request 0 (start, tokens); request 1 (start, tokens, state rows written before it)
REQUESTS = {"c4": ((0, 2048), (6144, 256, 4)), "c128": ((0, 4096), (4160, 2048, 64))}
#: request 2's tokens: (position, has a state slot, has a cache slot); none is compressed (a ratio-128 one with a state slot but
#: no cache slot gets its column, which `columns` reads, but no row)
EXTRA = {"c4": ((3, False, True), (7, False, True), (11, True, False), (15, True, False), (16, True, True), (21, True, True)),
         "c128": ((127, False, True), (255, True, False), (200, True, True), (383, False, True))}
#: state blocks and cache pages per layer; the spare ones no token reads or writes
NBS, NBK = {"c4": 640, "c128": 900}, {"c4": 16, "c128": 32}
SEEDS = {"c4": 41, "c128": 42}
MAXPOS = 8192
#: the compressor's rotary base (config.json `compress_rope_theta`; the Definitions take the cache's words, so YaRN is left out)
THETA = 160000


def _window(k: int, rows) -> tuple[int, ...]:
    """Request 1's token offsets of rows ``rows`` of its ratio-128 window k (window 0 starts 64 rows before its first token)."""
    return tuple(128 * k + r - REQUESTS["c128"][1][2] for r in rows)


#: request 1's edge operands by token offset: (what, name, offsets).  A name is the value of each word of those tokens' kv or score
#: rows, `xF` the rows scaled by F, `one_hot` -1e30 but 0 at the fourth token, `V@j` the value V at column j of the first token.
#: Ratio 4's window of boundary offset 4k + 3 is offsets 4k - 4 .. 4k + 3, its rows 0-3 reading each token's first half and rows
#: 4-7 its second; ratio 128's window k is its 8 warps of 16 rows
EDGE = {
    "c4": (("score", "zero", range(32, 40)), ("score", "inf", (50,)), ("score", "nan", (60,)), ("score", "neg_inf", range(64, 72)),
           ("kv", "zero", range(80, 88)), ("kv", "x2^70", range(88, 96)), ("kv", "x2^-140", range(96, 104)),
           ("kv", "nan@3", (110,)), ("kv", "inf@517", (111,)), ("kv", "neg_inf@7", (112,)), ("score", "x1e4", range(120, 128)),
           ("score", "one_hot", range(128, 136)), ("score", "inf", range(140, 148)), ("kv", "3e38@521", (154,)),
           ("kv", "x1e-30", range(160, 168)), ("score", "neg_inf", range(176, 180))),
    "c128": (("score", "zero", _window(1, range(128))), ("score", "inf", _window(2, (40,))), ("score", "nan", _window(3, (77,))),
             ("score", "neg_inf", _window(4, range(16, 32))), ("kv", "zero", _window(5, range(128))),
             ("kv", "x2^70", _window(6, range(128))), ("kv", "x2^-140", _window(7, range(128))), ("kv", "nan@3", _window(8, (5,))),
             ("score", "one_hot", _window(9, range(128))), ("score", "x1e4", _window(10, range(128))),
             ("kv", "x1e-30", _window(11, range(128))), ("kv", "inf@5", _window(12, (6,))), ("kv", "neg_inf@7", _window(13, (7,))),
             ("kv", "3e38@9", _window(14, (100,)))),
}
EDGE_VALUES = {"zero": 0.0, "inf": float("inf"), "nan": float("nan"), "neg_inf": float("-inf"), "3e38": 3e38}
EDGE_FACTORS = {"x2^70": 2.0 ** 70, "x2^-140": 2.0 ** -140, "x1e4": 1e4, "x1e-30": 1e-30}
#: rotary rows with special words, (ratio 4's, ratio 128's) compressed positions of request 1: cos[2] inf, sin[5] NaN, every cos 0
#: and sin -1
CS_SPECIAL = ((6152, 4096), (6160, 5376), (6168, 5504))
#: the norm weight's special words (bf16): 0 and 64 in the NoPE, -1 and 2^-8 in the RoPE
W_SPECIAL = {3: 0x0000, 100: 0x4280, 470: 0xBF80, 511: 0x3B80}
SENTINEL_STATE, SENTINEL_K = 0x7FA5A5A5, 0xA5
#: instances per job: state rows; cache entries per layer
JOB_ROWS, JOB_ENTRIES = 256, {"c4": 32, "c128": 4}
QUICK_REF_EVERY = 16
TOTALS = ("words", "mismatch", "reference", "twin_mismatch")
CUTLASS_DSL = "4.6.2"
#: the arithmetic of each kernel's PTX as compiled for sm_90a by nvidia-cutlass-dsl 4.6.2 (C4 fused; C128 compress; its store)
PTX_COUNT = re.compile(r"\b(fma\.rn(?:\.ftz)?\.f32|mul(?:\.rn)?(?:\.ftz)?\.f32|sub(?:\.rn)?(?:\.ftz)?\.f32|add(?:\.rn)?(?:\.ftz)?\.f32|"
                       r"neg(?:\.ftz)?\.f32|abs(?:\.ftz)?\.f32|div\.(?:rn|full|approx)(?:\.ftz)?\.f32|rcp\.(?:rn|approx)(?:\.ftz)?\.f32|"
                       r"rsqrt\.approx(?:\.ftz)?\.f32|ex2\.approx(?:\.ftz)?\.f32|cvt\.rn\.bf16x2\.f32|cvt\.rn\.bf16\.f32|"
                       r"cvt\.rn\.satfinite\.e4m3x2\.f32|(?:max|min)(?:\.ftz)?(?:\.NaN)?(?:\.xorsign\.abs)?\.f32|shfl\.sync\.bfly\.b32|"
                       r"selp\.f32|setp\.\w+\.f32)(?=[\s;])")
PTX_COUNTS = {
    "c4": {"abs.f32": 8, "add.f32": 7, "cvt.rn.bf16x2.f32": 8, "cvt.rn.satfinite.e4m3x2.f32": 4, "div.rn.f32": 9,
           "ex2.approx.ftz.f32": 128, "fma.rn.f32": 141, "max.f32": 84, "mul.f32": 228, "rsqrt.approx.ftz.f32": 1, "selp.f32": 8,
           "setp.ge.f32": 8, "shfl.sync.bfly.b32": 8, "sub.f32": 132},
    "c128": {"add.f32": 44, "div.rn.f32": 2, "ex2.approx.ftz.f32": 34, "fma.rn.f32": 32, "max.f32": 38, "mul.f32": 38,
             "shfl.sync.bfly.b32": 18, "sub.f32": 34},
    "store": {"abs.f32": 2, "add.f32": 8, "cvt.rn.bf16x2.f32": 2, "cvt.rn.satfinite.e4m3x2.f32": 1, "div.rn.f32": 1, "fma.rn.f32": 3,
              "max.f32": 9, "mul.f32": 10, "rsqrt.approx.ftz.f32": 1, "selp.f32": 2, "setp.ge.f32": 2, "shfl.sync.bfly.b32": 13,
              "sub.f32": 1},
}
#: the installed vLLM's files the Definitions and this script were read from, by sha256 at d9105ea80
SOURCES = {"models/deepseek_v4/compressor.py": "338b89ec959f1b40f7210cbee01a8dce12afa1259195137feb1fb806011f61f7",
           "models/deepseek_v4/common/ops/save_partial_states.py": "27f5a42c352339627004cc767f33e113710c095a585c4b40ddbfda3a335d9363",
           "models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py":
               "a91fb006cfc49cf207e03ed9626123077e75c84aefadef717469df8408cf8c7f",
           "models/deepseek_v4/attention.py": "b6b32d5a98737ba2d7e619b66de0b055f6189f68d513d9dc1e11f05aaa362616"}
NEG_INF = np.uint32(0xFF800000)
_error, _save, _load = FX._error, FX._save, FX._load


def _u8(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.uint8).cpu().numpy()


def _u32(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int32).cpu().numpy().view(np.uint32)


def _chunks(n: int, step: int) -> list[tuple[int, int]]:
    return [(i, min(n, i + step)) for i in range(0, n, step)]


# ------------------------------------------------------------------------------------------------------------------ operands
def layout(layer: str, seed: int) -> dict[str, np.ndarray]:
    """The three requests' tokens: positions, request ids, the state block table and slots, the cache block table and the cache
    slots (`get_compressed_slot_mapping`: -1 off the boundaries; request 2's as `EXTRA` says), each block table shuffled; the
    (block, row) of each state row written before the launch."""
    L = LAYERS[layer]
    rng = np.random.default_rng(seed + 1)
    (s0, n0), (s1, n1, pre) = REQUESTS[layer]
    extra = EXTRA[layer]
    pos = np.concatenate([np.arange(s0, s0 + n0), np.arange(s1, s1 + n1), [p for p, _, _ in extra]]).astype(np.int64)
    req = np.concatenate([np.zeros(n0), np.ones(n1), np.full(len(extra), 2)]).astype(np.int32)
    top = max(p for p, _, _ in extra) + 1
    sbt = np.zeros((3, (s1 + n1) // L.block + 1), dtype=np.int32)
    perm, k = rng.permutation(NBS[layer]).astype(np.int32), 0
    for r, (a, b) in enumerate(((s0, s0 + n0), (s1 - pre, s1 + n1), (0, top))):
        lo, hi = a // L.block, -(-b // L.block)
        sbt[r, lo:hi] = perm[k:k + hi - lo]
        k += hi - lo
    sslot = sbt[req, pos // L.block].astype(np.int64) * L.block + pos % L.block
    e = pos // L.ratio
    kbt = np.zeros((3, (s1 + n1) // L.ratio // L.pbs + 1), dtype=np.int32)
    kperm, k = rng.permutation(NBK[layer]).astype(np.int32), 0
    for r, (a, b) in enumerate(((s0, s0 + n0), (s1, s1 + n1), (0, top))):
        lo, hi = a // L.ratio // L.pbs, -(-(b // L.ratio + 1) // L.pbs)
        kbt[r, lo:hi] = kperm[k:k + hi - lo]
        k += hi - lo
    kslot = np.where((pos + 1) % L.ratio == 0, kbt[req, e // L.pbs].astype(np.int64) * L.pbs + e % L.pbs, -1)
    for i, (p, has_state, has_cache) in enumerate(extra):
        t = n0 + n1 + i
        sslot[t] = sslot[t] if has_state else -1
        kslot[t] = int(kbt[2, e[t] // L.pbs]) * L.pbs + int(e[t] % L.pbs) if has_cache else -1
    rows = [(s1 - pre + i, 1) for i in range(pre)] + [(p, 2) for p in range(top)]
    prefilled = np.array([(sbt[r, p // L.block], p % L.block) for p, r in rows], dtype=np.int64)
    return {"pos": pos, "req": req, "sbt": sbt, "sslot": sslot, "kbt": kbt, "kslot": kslot, "prefilled": prefilled}


def compressed(o: dict[str, Any], layer: str) -> np.ndarray:
    """The tokens whose row the kernel writes: a state slot, a cache slot, a window's end (cutedsl:166-184, 1316-1331)."""
    return (o["sslot"] >= 0) & (o["kslot"] >= 0) & ((o["pos"] + 1) % LAYERS[layer].ratio == 0)


def columned(o: dict[str, Any]) -> np.ndarray:
    """The tokens whose column the ratio-128 compress kernel writes: a state slot and a window's end (cutedsl:1003-1013)."""
    return (o["sslot"] >= 0) & ((o["pos"] + 1) % LAYERS["c128"].ratio == 0)


def rotary_cache() -> np.ndarray:
    """An f32 [MAXPOS, 64] cache [cos 0..31 | sin 0..31] of GPT-J angles (THETA), and CS_SPECIAL's rows, as words."""
    inv = float(THETA) ** (-np.arange(0, ROPE, 2) / ROPE)
    ang = np.arange(MAXPOS)[:, None] * inv[None, :]
    cs = np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
    (a4, a128), (b4, b128), (c4, c128) = CS_SPECIAL
    cs[[a4, a128], 2] = np.inf
    cs[[b4, b128], ROPE // 2 + 5] = np.nan
    cs[[c4, c128], :ROPE // 2], cs[[c4, c128], ROPE // 2:] = 0.0, -1.0
    return cs.view(np.uint32)


def _apply_edges(kv: np.ndarray, score: np.ndarray, base: int, table) -> None:
    for what, name, offs in table:
        t = [base + k for k in offs]
        x = kv if what == "kv" else score
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


def operands(layer: str, seed: int) -> dict[str, Any]:
    """The compressor's inputs as `DeepseekCompressor.forward` has them (compressor.py:310-365), as numpy words: kv and score
    [T, width] f32 (`fused_wkv_wgate`'s split output), ape [ratio, width] f32, the norm's bf16 weight (W_SPECIAL), the rotary
    cache; the initial state (n blocks, STATE_PAGE) f32, every word SENTINEL_STATE but the rows written before the launch."""
    L = LAYERS[layer]
    lay = layout(layer, seed)
    rng = np.random.default_rng(seed)
    n = lay["pos"].size

    def scaled(m: int) -> np.ndarray:
        return (rng.standard_normal((m, L.width)) * np.exp2(rng.uniform(-3, 3, (m, 1)))).astype(np.float32)

    kv, score = scaled(n), (4 * rng.standard_normal((n, L.width))).astype(np.float32)
    with np.errstate(over="ignore"):
        _apply_edges(kv, score, REQUESTS[layer][0][1], EDGE[layer])
    w = FX._bf16(1 + 0.1 * rng.standard_normal(D))
    for i, v in W_SPECIAL.items():
        w[i] = v
    state = np.full((NBS[layer], STATE_PAGE), SENTINEL_STATE, dtype=np.uint32)
    k = lay["prefilled"].shape[0]
    rows = np.concatenate([scaled(k), (4 * rng.standard_normal((k, L.width))).astype(np.float32)], axis=1).view(np.uint32)
    state_rows_of(state, layer)[lay["prefilled"][:, 0], lay["prefilled"][:, 1]] = rows
    return {**lay, "kv": kv.view(np.uint32), "score": score.view(np.uint32),
            "ape": (0.5 * rng.standard_normal((L.ratio, L.width))).astype(np.float32).view(np.uint32), "w": w,
            "cs": rotary_cache(), "state0": state}


def state_rows_of(state: np.ndarray, layer: str) -> np.ndarray:
    """A (n blocks, STATE_PAGE) state as the kernels index it: [block, row, kv | score]."""
    L = LAYERS[layer]
    return state[:, :L.block * 2 * L.width].reshape(state.shape[0], L.block, 2 * L.width)


def state_rows(state: np.ndarray, block_table: np.ndarray, req: np.ndarray, pos: np.ndarray, layer: str):
    """The window's state rows the kernel reads for each token: row r is position p - window + 1 + r of its request, at
    `block_table[req, q // block]` row q % block, its kv words [(r // ratio) * 512 + j] and its score words that plus the width
    (cutedsl:192-245, 1080-1117); a row at a position below 0 is skipped (ratio 4 at p = 3).  -> kv, score (n, window, 512) words
    (a skipped row kv 0, score -inf) and the rows read (n,), the last ones of the window."""
    L = LAYERS[layer]
    s = state_rows_of(np.asarray(state, dtype=np.uint32), layer)
    p = np.asarray(pos, dtype=np.int64)[:, None] - (L.window - 1) + np.arange(L.window)[None, :]
    valid = p >= 0
    pc = np.maximum(p, 0)
    rows = s[np.asarray(block_table)[np.asarray(req)[:, None], pc // L.block], pc % L.block]
    col = ((np.arange(L.window) // L.ratio) * D)[None, :, None] + np.arange(D)[None, None, :]
    col = np.broadcast_to(col, (p.shape[0], L.window, D))
    kv = np.where(valid[..., None], np.take_along_axis(rows, col, axis=2), np.uint32(0))
    sc = np.where(valid[..., None], np.take_along_axis(rows, col + L.width, axis=2), NEG_INF)
    return kv.astype(np.uint32), sc.astype(np.uint32), valid.sum(axis=1)


def slot_words(cache: np.ndarray, slots, pbs: int) -> np.ndarray:
    """The rows at cache slots ``slots`` of a (n pages, page_bytes(pbs)) byte cache: (n, 520) [448 NoPE bytes | 64 RoPE bf16
    words | 8 scale bytes], the order a row Definition returns."""
    c = np.asarray(cache)
    blk, off = np.divmod(np.asarray(slots, dtype=np.int64), pbs)
    data = c[:, :pbs * DATA_BYTES].reshape(c.shape[0], pbs, DATA_BYTES)[blk, off]
    sc = c[:, pbs * DATA_BYTES:pbs * TOKEN_BYTES].reshape(c.shape[0], pbs, SCALE_BYTES)[blk, off]
    rope = np.ascontiguousarray(data[:, NOPE:]).view("<u2")
    return np.concatenate([data[:, :NOPE], rope, sc], axis=1).astype(np.int64)


def written_bytes(n_pages: int, pbs: int, slots) -> np.ndarray:
    """The bytes of a (n_pages, page_bytes(pbs)) cache that writing rows at ``slots`` touches: 576 value bytes and 8 scale bytes."""
    page = page_bytes(pbs)
    m = np.zeros(n_pages * page, dtype=bool)
    for slot in slots:
        blk, off = divmod(int(slot), pbs)
        m[blk * page + off * DATA_BYTES: blk * page + (off + 1) * DATA_BYTES] = True
        m[blk * page + pbs * DATA_BYTES + off * SCALE_BYTES: blk * page + pbs * DATA_BYTES + (off + 1) * SCALE_BYTES] = True
    return m


def entries_twin(layer: str, kv: np.ndarray, score: np.ndarray, nrows: np.ndarray, w: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """`dsv4_compressor_twin`'s rows of n entries: kv and score (n, window, 512) words, each entry's last ``nrows`` rows read ->
    (n, 520) words."""
    kv, score, cs, nrows = (np.asarray(a) for a in (kv, score, cs, nrows))
    out = np.zeros((kv.shape[0], WORDS), dtype=np.int64)
    with np.errstate(all="ignore"):
        for k in np.unique(nrows):
            sel = nrows == k
            a, b = kv[sel, kv.shape[1] - k:], score[sel, kv.shape[1] - k:]
            q, r, e = (T.compress_c4(a, b, w, cs[sel], QB, EPS) if layer == "c4"
                       else T.compress_c128(a, b, w, cs[sel], W128, R128, QB, EPS))
            out[sel] = np.concatenate([q, r, e], axis=1)
    return out


def entry_fn(layer: str, rows: int):
    from verity.ml.circuits.defs import bind

    if layer == "c4":
        return bind(R.Dsv4MlaCompressC4, ROWS=rows, D=D, QB=QB, EPS=EPS)
    return bind(R.Dsv4MlaCompressC128, W=W128, R=R128, D=D, QB=QB, EPS=EPS)


# ----------------------------------------------------------------------------------------------------------------- the GPU side
def environment(work: Path) -> dict[str, str]:
    """The Triton cache (`save_partial_states`) and the CuTe DSL's dump and cache directories in scratch, and CUTE_DSL_KEEP=
    ptx,cubin; set before vLLM, and so cutlass, is imported (the DSL reads its environment once).  Without KEEP a compiled
    kernel's `__ptx__` and `__cubin__` are None (cutlass 4.6.2's `JitFunctionArtifacts` holds only the kept files), and with it
    the DSL bypasses its own cache, so each kernel the launches compile is compiled afresh and keeps its PTX and cubin."""
    env = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TRITON_CACHE_DIR": str(work / "triton_cache"), "CUTE_DSL_KEEP": "ptx,cubin",
           "CUTE_DSL_DUMP_DIR": str(work / "cute_dsl_dump"), "CUTE_DSL_CACHE_DIR": str(work / "cute_dsl_cache")}
    for k in ("CUTE_DSL_DUMP_DIR", "CUTE_DSL_CACHE_DIR"):
        Path(env[k]).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    return env


def sources(root: Path) -> dict[str, Any]:
    """Each of SOURCES under the vLLM package directory ``root``: True when its sha256 is the one read, False when it differs,
    None when it is missing."""
    files: dict[str, bool | None] = {}
    for rel, want in SOURCES.items():
        p = root / rel
        files[rel] = hashlib.sha256(p.read_bytes()).hexdigest() == want if p.is_file() else None
    return {"files": files, "ok": all(v is True for v in files.values())}


def _version(dist: str) -> str | None:
    import importlib.metadata as M

    try:
        return M.version(dist)
    except M.PackageNotFoundError:
        return None


def selection() -> dict[str, Any]:
    """vLLM's DSV4 attention class and its cache layout on this device, the CuTe DSL path and its version, the sources."""
    import vllm
    from vllm.config import get_current_vllm_config
    from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import has_cutedsl

    cls = _select_dsv4_attn_cls(get_current_vllm_config())
    rec: dict[str, Any] = {"attn_cls": cls.__name__, "fp8_ds_mla": bool(cls.use_fp8_ds_mla_layout),
                           "cuda": bool(current_platform.is_cuda()), "has_cutedsl": bool(has_cutedsl()),
                           "cutlass_dsl": _version("nvidia-cutlass-dsl"), "sources": sources(Path(vllm.__file__).parent)}
    rec["checks"] = {"attn_cls": cls is DeepseekV4FlashMLAAttention, "fp8_ds_mla": rec["fp8_ds_mla"], "cuda": rec["cuda"],
                     "has_cutedsl": rec["has_cutedsl"], "cutlass_dsl": rec["cutlass_dsl"] == CUTLASS_DSL,
                     "sources": rec["sources"]["ok"]}
    rec["ok"] = all(rec["checks"].values())
    return rec


def stack() -> dict[str, Any]:
    """What the kernels of record came from: torch, vLLM, Triton, the CuTe DSL and quack (its fake-tensor helper)."""
    import torch
    import vllm

    return {"torch": torch.__version__, "cuda": torch.version.cuda, "vllm": getattr(vllm, "__version__", None),
            **{d: _version(d) for d in ("triton", "nvidia-cutlass-dsl", "quack-kernels")}}


def _ref_every(quick: bool) -> int:
    return QUICK_REF_EVERY if quick else 1


def launch_layer(work: Path, layer: str, quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One layer's launches and what they wrote: the launch records and comparison jobs."""
    o = operands(layer, SEEDS[layer])
    return observe(work, layer, o, *launches(layer, o), quick)


def launches(layer: str, o: dict[str, Any]) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray] | None]:
    """One layer's launches as `forward` makes them, twice on fresh caches: per run the state words after `save_partial_states`
    ("saved") and after the compress ("after") and the cache bytes ("k"); on ratio 128 the split call on the first run's saved
    state, its f32 `compressed_kv` words ("cols"), state ("state") and cache ("k"), else None."""
    import torch
    from vllm.models.deepseek_v4.common.ops.save_partial_states import save_partial_states
    from vllm.models.deepseek_v4.nvidia.ops import sparse_attn_compress_cutedsl as K

    L = LAYERS[layer]
    dev = "cuda"
    n, nbs, nbk = o["pos"].size, NBS[layer], NBK[layer]

    def t(a: np.ndarray):
        a = np.ascontiguousarray(a)
        return torch.from_numpy(a.view(np.int32) if a.dtype == np.uint32 else a.view(np.int16) if a.dtype == np.uint16 else a).to(dev)

    pos, sslot, kslot, sbt, req = (t(o[k]) for k in ("pos", "sslot", "kslot", "sbt", "req"))
    kv, score, ape, cs = (t(o[k]).view(torch.float32) for k in ("kv", "score", "ape", "cs"))
    w = t(o["w"]).view(torch.bfloat16)
    pdl = {"launch_pdl": False}

    def fresh(state: np.ndarray):
        sbuf = t(state).view(torch.float32)
        kbuf = torch.full((nbk, page_bytes(L.pbs)), SENTINEL_K, dtype=torch.uint8, device=dev)
        return (sbuf, sbuf[:, :L.block * 2 * L.width].view(nbs, L.block, 2 * L.width), kbuf,
                kbuf[:, :L.pbs * TOKEN_BYTES].view(nbk, L.pbs, TOKEN_BYTES))

    runs = []
    for _ in range(2):
        sbuf, state, kbuf, kc = fresh(o["state0"])
        save_partial_states(kv=kv, score=score, ape=ape, positions=pos, state_cache=state, slot_mapping=sslot, block_size=L.block,
                            state_width=L.width, compress_ratio=L.ratio, pdl_kwargs=pdl)
        torch.cuda.synchronize()
        saved = _u32(sbuf)
        K.compress_norm_rope_store_cutedsl(state_cache=state, num_actual=n, token_to_req_indices=req, positions=pos,
                                           slot_mapping=sslot, block_table=sbt, block_size=L.block, state_width=L.width,
                                           cos_sin_cache=cs, kv_cache=kc, k_cache_metadata=SimpleNamespace(slot_mapping=kslot),
                                           pdl_kwargs=pdl, head_dim=D, rope_head_dim=ROPE, compress_ratio=L.ratio,
                                           overlap=L.ratio == 4, use_fp4_cache=False, rms_norm_weight=w, rms_norm_eps=EPS,
                                           quant_block=QB, token_stride=DATA_BYTES, scale_dim=SCALE_BYTES, store_full_kv=False,
                                           store_full_fp8=False, fp8_scale=None)
        torch.cuda.synchronize()
        runs.append({"saved": saved, "after": _u32(sbuf), "k": _u8(kbuf)})
    if L.ratio != 128:
        return runs, None
    sbuf, state, kbuf, kc = fresh(runs[0]["saved"])
    ckv = torch.full((n, D), SENTINEL_STATE, dtype=torch.int32, device=dev).view(torch.float32)
    K.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
        state, req, pos, sslot, sbt, L.block, ckv, w, EPS, cs, kc, kslot, kc.shape[1], kc.stride(0), head_size=D,
        state_width=L.width, rope_head_dim=ROPE, fp8_max=448.0, quant_block=QB, token_stride=DATA_BYTES, scale_dim=SCALE_BYTES,
        compress_ratio=L.ratio, overlap=False)
    torch.cuda.synchronize()
    return runs, {"cols": _u32(ckv), "state": _u32(sbuf), "k": _u8(kbuf)}


def observe(work: Path, layer: str, o: dict[str, Any], runs: list[dict[str, np.ndarray]], split: dict[str, np.ndarray] | None,
            quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`launches`' words checked and saved under ``work``: the launch records and the comparison jobs."""
    L = LAYERS[layer]
    nbs, nbk = NBS[layer], NBK[layer]
    saved, k, init = runs[0]["saved"], runs[0]["k"], o["state0"]
    p, r_, ss, ks = o["pos"], o["req"], o["sslot"], o["kslot"]
    flat = lambda s: state_rows_of(s, layer).reshape(nbs * L.block, 2 * L.width)  # noqa: E731
    written = np.flatnonzero(ss >= 0)
    touched = np.zeros(nbs * L.block, dtype=bool)
    touched[ss[written]] = True
    pad = slice(L.block * 2 * L.width, None)
    jobs: list[dict[str, Any]] = []
    every = _ref_every(quick)
    rec_score = {"phase": "score", "case": layer, "rows_written": int(written.size),
                 "checks": {"deterministic": bool(np.array_equal(saved, runs[1]["saved"])),
                            "kv_copied": bool(np.array_equal(flat(saved)[ss[written], :L.width], o["kv"][written])),
                            "unwritten_untouched": bool(np.array_equal(flat(saved)[~touched], flat(init)[~touched])
                                                        and np.array_equal(saved[:, pad], init[:, pad])),
                            "rows_written": int(written.size) == p.size - sum(not s for _, s, _ in EXTRA[layer])}}
    paths = _save(work / f"score-{layer}", score=o["score"][written], ape=o["ape"][p[written] % L.ratio],
                  got=flat(saved)[ss[written], L.width:], tokens=p[written])
    jobs += [{"phase": "score", "case": layer, "kind": "score", "paths": paths, "rows": [a, b], "ref_every": every}
             for a, b in _chunks(written.size, JOB_ROWS)]
    act = compressed(o, layer)
    (_, n0), (_, n1, _) = REQUESTS[layer]
    want = {0: n0 // L.ratio, 1: (REQUESTS[layer][1][0] + n1) // L.ratio - REQUESTS[layer][1][0] // L.ratio}
    rec_comp = {"phase": "compress", "case": layer, "entries": int((act & (r_ == 0)).sum()),
                "checks": {"deterministic": bool(np.array_equal(k, runs[1]["k"])),
                           "state_unchanged": all(np.array_equal(r["saved"], r["after"]) for r in runs),
                           "unwritten_untouched": bool((k.reshape(-1)[~written_bytes(nbk, L.pbs, ks[act])] == SENTINEL_K).all()),
                           "entries": int((act & (r_ == 0)).sum()) == want[0] and not (act & (r_ == 2)).any()}}
    rec_edge = {"phase": "compress", "case": f"{layer}-edge", "entries": int((act & (r_ == 1)).sum()),
                "checks": {"entries": int((act & (r_ == 1)).sum()) == want[1], "deterministic": rec_comp["checks"]["deterministic"]}}
    recs = [rec_score, rec_comp, rec_edge]
    for case, r in ((layer, 0), (f"{layer}-edge", 1)):
        sel = np.flatnonzero(act & (r_ == r))
        kvr, scr, nrows = state_rows(saved, o["sbt"], r_[sel], p[sel], layer)
        paths = _save(work / f"compress-{case}", kv=kvr, score=scr, nrows=nrows, w=o["w"], cs=o["cs"][(p[sel] // L.ratio) * L.ratio],
                      got=slot_words(k, ks[sel], L.pbs), tokens=p[sel])
        jobs += [{"phase": "compress", "case": case, "kind": "entries", "layer": layer, "paths": paths, "rows": [a, b],
                  "ref_every": every if r == 0 else 1} for a, b in _chunks(sel.size, JOB_ENTRIES[layer])]
    if split is not None:
        cols = split["cols"]
        col = columned(o)
        recs.append({"phase": "columns", "case": layer, "columns": int((col & (r_ != 1)).sum()),
                     "checks": {"split_is_served": bool(np.array_equal(split["k"], k)),
                                "state_unchanged": bool(np.array_equal(split["state"], saved)),
                                "untouched": bool((cols[~col] == SENTINEL_STATE).all()),
                                "columns": int((col & (r_ != 1)).sum()) == want[0] + 1}})
        recs.append({"phase": "columns", "case": f"{layer}-edge", "columns": int((col & (r_ == 1)).sum()),
                     "checks": {"columns": int((col & (r_ == 1)).sum()) == want[1]}})
        for case, sel in ((layer, np.flatnonzero(col & (r_ != 1))), (f"{layer}-edge", np.flatnonzero(col & (r_ == 1)))):
            kvr, scr, _ = state_rows(saved, o["sbt"], r_[sel], p[sel], layer)
            paths = _save(work / f"columns-{case}", kv=kvr, score=scr, got=cols[sel], tokens=p[sel])
            jobs += [{"phase": "columns", "case": case, "kind": "columns", "paths": paths, "rows": [a, b],
                      "ref_every": every if case == layer else 1} for a, b in _chunks(sel.size, JOB_ENTRIES[layer])]
    return recs, jobs


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


def _flat_out(y) -> list[int]:
    out: list[int] = []
    for x in y:
        out += [int(v) for v in x] if isinstance(x, (list, tuple)) else [int(x)]
    return out


def compare_score(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """State rows [lo, hi): their score words against `fadd(score, ape)`, the reference `Dsv4CompressorScore_v1{W}` on every
    ref_every-th."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    lo, hi = job["rows"]
    want = FT.fadd(a["score"][lo:hi], a["ape"][lo:hi])
    _tally(rec, want, a["got"][lo:hi], a["tokens"][lo:hi], "position")
    fn = bind(I.Dsv4CompressorScore, W=int(a["score"].shape[1]))
    for i in _ref_rows(lo, hi, job["ref_every"]):
        y = evaluate(fn, a["score"][i].tolist(), a["ape"][i].tolist())
        rec["reference"] += 1
        rec["twin_mismatch"] += int(_flat_out(y) != want[i - lo].astype(np.int64).tolist())


def compare_entries(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Cache rows [lo, hi): every word against the twin, the reference `Dsv4MlaCompressC4_v1{ROWS,...}` or
    `Dsv4MlaCompressC128_v1{8,16,...}` on every ref_every-th."""
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    lo, hi = job["rows"]
    layer = job["layer"]
    nrows, w = np.asarray(a["nrows"]), np.asarray(a["w"])
    want = entries_twin(layer, a["kv"][lo:hi], a["score"][lo:hi], nrows[lo:hi], w, a["cs"][lo:hi])
    _tally(rec, want, a["got"][lo:hi], a["tokens"][lo:hi], "position")
    for i in _ref_rows(lo, hi, job["ref_every"]):
        k = int(nrows[i])
        kv, sc = (np.asarray(a[x][i])[a[x].shape[1] - k:] for x in ("kv", "score"))
        y = evaluate(entry_fn(layer, k), kv.reshape(-1).tolist(), sc.reshape(-1).tolist(), w.tolist(), a["cs"][i].tolist())
        rec["reference"] += 1
        rec["twin_mismatch"] += int(_flat_out(y) != want[i - lo].tolist())


def compare_columns(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Ratio-128 columns of tokens [lo, hi): every word against `c128_columns`, the reference `Dsv4MlaC128Column_v1{8,16}` on every
    column of every ref_every-th."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    lo, hi = job["rows"]
    kv, sc = (np.swapaxes(np.asarray(a[x][lo:hi]), -1, -2) for x in ("kv", "score"))
    with np.errstate(all="ignore"):
        want = T.c128_columns(kv, sc, W128, R128)
    _tally(rec, want, a["got"][lo:hi], a["tokens"][lo:hi], "position")
    fn = bind(R.Dsv4MlaC128Column, W=W128, R=R128)
    for i in _ref_rows(lo, hi, job["ref_every"]):
        y = [_flat_out(evaluate(fn, kv[i - lo, j].tolist(), sc[i - lo, j].tolist()))[0] for j in range(D)]
        rec["reference"] += 1
        rec["twin_mismatch"] += int(y != want[i - lo].astype(np.int64).tolist())


COMPARE = {"score": compare_score, "entries": compare_entries, "columns": compare_columns}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    try:
        COMPARE[job["kind"]](job, rec)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "error": _error(e)}
    return {"phase": job["phase"], "case": job["case"], **rec}


# --------------------------------------------------------------------------------------------------------------- ptx and bind
def ptx_counts(text: str) -> dict[str, int]:
    return dict(sorted(Counter(PTX_COUNT.findall(text)).items()))


def ptx(outdir: Path) -> dict[str, Any]:
    """The three kernels the launches compiled, fetched from their `compile` caches with the wrappers' own arguments (the caches'
    sizes unchanged: each holds the one served specialization), their PTX's target and arithmetic against PTX_COUNTS; the PTX,
    cubin and (where `cuobjdump` runs) SASS copied to `<out>/ptx/`."""
    import torch
    from vllm.models.deepseek_v4.nvidia.ops import sparse_attn_compress_cutedsl as K

    kernels = {"c4": K.SparseAttnCompressNormRopeStoreC4Kernel, "c128": K.SparseAttnCompressC128Block8Kernel,
               "store": K.SparseAttnNormRopeStoreKernel}
    sizes = lambda: {k: c.compile.cache_info().currsize for k, c in kernels.items()}  # noqa: E731
    before = sizes()
    L4, L128 = LAYERS["c4"], LAYERS["c128"]
    got = {"c4": K.SparseAttnCompressNormRopeStoreC4Kernel.compile(
        head_size=D, state_width=L4.width, rope_head_dim=ROPE, fp8_max=448.0, quant_block=QB, token_stride=DATA_BYTES,
        scale_dim=SCALE_BYTES, kv_block_stride=page_bytes(L4.pbs), compress_ratio=L4.ratio, overlap=True,
        norm_weight_dtype=K._TORCH_TO_CUTE[torch.bfloat16])}
    got["c128"], got["store"] = K.compile_split_sparse_attn_cutedsl(
        D, L128.width, L128.block, ROPE, 448.0, QB, DATA_BYTES, SCALE_BYTES, L128.pbs, page_bytes(L128.pbs), L128.ratio, False,
        torch.bfloat16, store_full_kv=False, store_full_fp8=False)
    dest = outdir / "ptx"
    dest.mkdir(parents=True, exist_ok=True)
    tool = shutil.which("cuobjdump")
    rec: dict[str, Any] = {"cache_sizes": {"before": before, "after": sizes()}, "keep": os.environ.get("CUTE_DSL_KEEP"),
                           "kernels": {}}
    for name, c in got.items():
        text, cubin = getattr(c, "__ptx__", None), getattr(c, "__cubin__", None)
        e: dict[str, Any] = {"ptx": text is not None, "cubin": cubin is not None}
        if text is not None:
            text = text.decode() if isinstance(text, bytes) else text
            (dest / f"{name}.ptx").write_text(text)
            e.update(sha256=hashlib.sha256(text.encode()).hexdigest(), target=re.findall(r"\.target\s+(\S+)", text),
                     counts=ptx_counts(text))
        if cubin is not None:
            (dest / f"{name}.cubin").write_bytes(cubin)
            if tool:
                try:
                    sass = subprocess.run([tool, "-sass", str(dest / f"{name}.cubin")], capture_output=True, text=True,
                                          check=True).stdout
                    (dest / f"{name}.sass").write_text(sass)
                    e["sass_counts"] = dict(Counter(re.findall(r"\b(FFMA|FMUL|FADD|FMNMX|MUFU\.\w+|F2FP\.\w+(?:\.\w+)*)\b", sass)))
                except Exception as err:  # noqa: BLE001
                    e["sass_error"] = str(err)[-300:]
        rec["kernels"][name] = e
    rec["checks"] = {"one_specialization": before == rec["cache_sizes"]["after"] == {k: 1 for k in kernels},
                     "kept": all(e["ptx"] and e["cubin"] for e in rec["kernels"].values()),
                     "target": all(e.get("target") == ["sm_90a"] for e in rec["kernels"].values()),
                     "counts": all(rec["kernels"][k].get("counts") == PTX_COUNTS[k] for k in kernels)}
    rec["ok"] = all(rec["checks"].values())
    return rec


def bind_check() -> dict[str, Any]:
    """The Programs bind at DeepSeek-V4-Pro's statics, and the decode's row reader at the row they write."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import flashmla as F

    row = {"D": D, "QB": QB, "EPS": EPS}
    targets = {
        "Dsv4CompressorScore_v1{W=1024}": lambda: bind(I.Dsv4CompressorScore, W=LAYERS["c4"].width),
        "Dsv4CompressorScore_v1{W=512}": lambda: bind(I.Dsv4CompressorScore, W=LAYERS["c128"].width),
        "Dsv4MlaC4Column_v1{ROWS=8}": lambda: bind(R.Dsv4MlaC4Column, ROWS=8),
        "Dsv4MlaC4Column_v1{ROWS=4}": lambda: bind(R.Dsv4MlaC4Column, ROWS=4),
        "Dsv4MlaC128Column_v1{W=8,R=16}": lambda: bind(R.Dsv4MlaC128Column, W=W128, R=R128),
        "Dsv4MlaNormRopeQuant_v1{RATIO=4,...}": lambda: bind(R.Dsv4MlaNormRopeQuant, RATIO=4, **row),
        "Dsv4MlaNormRopeQuant_v1{RATIO=128,...}": lambda: bind(R.Dsv4MlaNormRopeQuant, RATIO=128, **row),
        "Dsv4MlaCompressC4_v1{ROWS=8,...}": lambda: entry_fn("c4", 8),
        "Dsv4MlaCompressC4_v1{ROWS=4,...}": lambda: entry_fn("c4", 4),
        "Dsv4MlaCompressC128_v1{W=8,R=16,...}": lambda: entry_fn("c128", LAYERS["c128"].window),
        "FlashMlaKvRow_v1{DN=448,DR=64,QT=64}": lambda: bind(F.FlashMlaKvRow, DN=NOPE, DR=ROPE, QT=QB),
    }
    out = {}
    for name, make in targets.items():
        try:
            out[name] = bool(make().signature())
        except Exception as e:  # noqa: BLE001
            out[name] = _error(e)
    return {"bindings": out, "ok": all(v is True for v in out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
def gpu(work: Path, outdir: Path, quick: bool, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The selection, the stack and each layer's launches in order; each error goes to doc["errors"].  Returns the jobs."""
    import torch

    jobs: list[dict[str, Any]] = []

    def step(name: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            print(f"[dsv4-cmp] {name} ERROR {doc['errors'][name][:600]}", flush=True)
            return None

    doc["selection"] = step("selection", selection) or {"ok": False}
    print(f"[dsv4-cmp] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
    doc["stack"] = step("stack", stack) or {}
    for layer in LAYERS:
        r = step(layer, lambda layer=layer: launch_layer(work, layer, quick))
        if r is not None:
            doc["launches"] += r[0]
            jobs += r[1]
            for x in r[0]:
                print(f"[dsv4-cmp] {x['phase']:8s} {x['case']:9s} {json.dumps(x['checks'])}", flush=True)
    torch.cuda.empty_cache()
    return jobs


CASES = (("score", "c4"), ("score", "c128"), ("compress", "c4"), ("compress", "c4-edge"), ("compress", "c128"),
         ("compress", "c128-edge"), ("columns", "c128"), ("columns", "c128-edge"))


def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["checks"]) and all(v is True for v in x["checks"].values())


def _words_ok(t: dict[str, Any]) -> bool:
    return t["words"] > 0 and t["mismatch"] == 0 and t["reference"] > 0 and t["twin_mismatch"] == 0


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's verdict: its launch checks, its words and its errors (its layer's launch, its comparisons), so a NO-GO names
    what failed."""
    out: dict[str, dict[str, Any]] = {}
    for phase, case in CASES:
        launches = [x for x in doc["launches"] if x["phase"] == phase and x["case"] == case]
        comps = [r for r in doc.get("comparisons", []) if r["phase"] == phase and r["case"] == case]
        t = {k: sum(r[k] for r in comps) for k in TOTALS}
        errors = [e for e in doc["errors"] if e == case.split("-")[0] or e == f"compare/{phase}/{case}"]
        ok = bool(launches) and all(_launch_ok(x) for x in launches)
        out[f"{phase}/{case}"] = {"launches_ok": ok, "totals": t, "errors": errors, "ok": ok and _words_ok(t) and not errors}
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    return bool(doc["gate"]["ok"] and not doc["errors"] and doc["selection"].get("ok") is True and doc["ptx"].get("ok") is True
                and doc["bind"].get("ok") is True and all(c["ok"] for c in cases(doc).values()) and _words_ok(doc["totals"]))


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
    print(f"[dsv4-cmp] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-COMPRESSOR GATE-FAIL", flush=True)
        raise SystemExit(3)
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        jobs = gpu(work, outdir, quick, doc)
    doc.setdefault("selection", {"ok": False})
    for name, fn in (("ptx", lambda: ptx(outdir)), ("bind", bind_check)):
        try:
            doc[name] = fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            doc[name] = {"ok": False}
    print(f"[dsv4-cmp] ptx {json.dumps(doc['ptx'])[:2000]}", flush=True)
    print(f"[dsv4-cmp] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
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
        print(f"[dsv4-cmp] words {r['phase']:8s} {r['case']:9s} words={r['words']} !={r['mismatch']} ref={r['reference']} "
              f"twin!={r['twin_mismatch']} first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[dsv4-cmp] case {key:18s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} "
              f"!={t['mismatch']} ref={t['reference']} twin!={t['twin_mismatch']} errors={c['errors']}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"DSV4-COMPRESSOR {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro's head-512 compressor on this Hopper GPU against its Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and ptx/")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (the operands, outputs and Triton cache)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true",
                    help=f"the reference on every {QUICK_REF_EVERY}th served instance: a first signal in minutes")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
