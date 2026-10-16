"""DeepSeek-V4-Pro's window (SWA) and ratio-128 (HCA) decode index rows on Hopper against their Programs: the H200's go/no-go for
binding `dsv4_decode_metadata.Dsv4SwaIndices_v1{SW,BS,NB}` (vLLM d9105ea80's Triton `_compute_swa_indices_and_lens_kernel`, the
window's slot ids of the paged SWA cache and their count, which FlashMLA's sparse decode reads as `indices` and `topk_length`) and
`Dsv4C128aGlobalTopk_v1{W,R,BS,NB}` (`_build_c128a_topk_metadata_kernel`'s decode branch, an HCA layer's compressed slot ids and
their count, `extra_indices_in_kvcache` and `extra_topk_length`) at DeepSeek-V4-Pro's statics on one TP8 rank: the window of 128
at block 64 over the SWA group's block-table row, and ratio 128 at block 2 over the HCA group's, each row at max_model_len
1,048,576 and at 4096.  Line numbers under `vllm/` unless named.

One GPU process writes `<out>/dsv4_decode_metadata_capture.json`, the property record `dsv4_decode_metadata_capture`
(`properties.record`), and the kernels' PTX (and SASS where Triton's bundled `cuobjdump` runs) under `<out>/ptx/`; it prints one
line per launch and per case, then `DSV4-DECODE-METADATA GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs
first and nothing else runs when it fails (exit 3).  A case that raises records its error and fails its cases without stopping the
others.  The operands are the integers a decode step holds (`requests`), so no model is loaded.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `_select_dsv4_attn_cls` picks `DeepseekV4FlashMLAAttention`, whose decode (models/deepseek_v4/nvidia/flashmla.py:
             174-264) reads the SWA builder's rows (209-210) and the HCA builder's (204-207): its `swa_backend_cls` is
             `DeepseekSparseSWAFlashMLABackend` of `DeepseekSparseSWAFlashMLAMetadataBuilder`, which keeps the
             `DeepseekSparseSWAMetadataBuilder.build` that launches `_COMPUTE_SWA_INDICES_AND_LENS_KERNEL` (v1/attention/backends/
             mla/sparse_swa.py:593-609, an instance of `ComputeSWAIndicesAndLensKernel` over `_compute_swa_indices_and_lens_kernel`)
             at kernel block sizes multiples of 64; its `backend_cls` `DeepseekV4FlashMLABackend` of `DeepseekV4FlashMLAMetadataBuilder`,
             which keeps `build` and `_build_c128a_metadata` (models/deepseek_v4/sparse_mla.py:179-278) at kernel block 256; the
             installed Triton is the 3.7.1 whose PTX was read, and the installed SOURCES are the files read, since the launches call
             the launchers directly rather than through the builders.
  swa        per case, `_COMPUTE_SWA_INDICES_AND_LENS_KERNEL` as the builder calls it (window_size = index_width = 128, the lengths'
             buffer as the unused visible arguments, block size 64, token_offset 0): the rows a (tokens + 3, 1, 128) int32 buffer and
             the lengths (tokens + 8,), sentinel-filled; `query_start_loc`, `seq_lens` and `token_to_req_indices` int32 buffers
             longer than their words, the block table a [:requests] slice of an int32 table NB wide, `is_valid_token` a bool view of
             bytes (555-556).  `served`: 16 requests (max_num_seqs; no MTP) of one decode token each, at positions from 0 to
             max_model_len - 1, each block row the request's block ids and junk words past them; two padding tokens (byte 0).
             `edge`: twelve requests of 1 to 4 tokens in a shuffled map, at the window's, a block's and the row's edges and at the
             negative positions below them; `query_start_loc` words that wrap int32; bytes 0, 1, 2, 0x80 and 0xFF; block rows whose
             products by 64 wrap int32 (2^25 and up, INT32_MAX, negative words, random words).
  c128a      per case, `build_c128a_topk_metadata` (sparse_mla.py:295-358) as the HCA builder calls it (ratio 128, block size 2,
             max_compressed_tokens the step's active width, the buffers c128a_max_compressed wide, 150-177) on the decode tokens and
             two prefill tokens after them: the decode rows, lengths and prefill rows into sentinel buffers.  `served`: 16 requests
             at positions from 0 to max_model_len - 1 (the step's width 8192 at 2^20, 128 at 4096) and two padding tokens (slot -1);
             `one`: the last of them alone and `full`: the 16 without padding, the kernel's num_decode_tokens == 1 and multiple of
             16 specializations, which a served step of one request and of max_num_seqs compile.  `edge`: six requests' 42 tokens
             in a shuffled map at the width 1024 of a step whose longest sequence is 65,664 to 131,199 tokens (a row 8192 wide
             written to 1024; at 4096, 128), at the lane counts' edges (0, 1, the block, the width, past it where the row holds it)
             and the int64 ends (-1, -R - 1, INT64_MIN, INT64_MAX), slots of every sign and the int64 ends, block rows whose
             products by 2 wrap int32.
             Each launch twice, then the kernel itself once at the launcher's arguments into sentinel-filled buffers at an offset:
             every token's row and its length against `dsv4_decode_metadata_twin` (the kernel's stores) and the reference on every
             token (every 16th served one under --quick); the launches the same words, the outputs where the decode reads them (the
             returned views of the HCA buffers, [:tokens, :width]), the HCA prefill rows the kernel's prefill branch, the operands
             unchanged, and every other word of the sentinel buffers untouched.
  ptx        the Triton cache's PTX of each kernel for sm_90a, each file one of PTX_SPECS and each of those present (the strides,
             widths and block sizes are multiples of 16 or not alike in every case; the HCA kernel compiles PTX_COUNTS at 18 and
             42 decode tokens, again at 16, and PTX_ONE, whose decode addresses fold, at 1), holding PTX_COUNTS: the arithmetic the
             Definitions were read from (the window's p // 64 and the row's j // 2 as div.s32 by the runtime block size and p % BS
             as p - (p // BS) * BS, no remainder; the window's byte as one ld.global.b8, and.b16 and setp.ne.b16 and its start as
             max.s32; the row's (position + 1) // 128 as div.s64 or div.u32, min.s64, the two int64 loads, the count as popc.b32,
             redux.sync.add.s32 and two shfl.sync.bfly.b32).
  bind       the Programs bind at the served statics and the edge width.

GO needs every case's words compared with 0 mismatches, the references with 0 disagreements with the twin, every launch check,
the selection, the PTX and the bindings.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/dsv4_decode_metadata_capture_gpu.py --out DIR [--scratch DIR] [--quick]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import dsv4_compressor_capture_gpu as CX
from tests.properties import dsv4_flashmla_capture_gpu as FX
from tests.properties import dsv4_indices_capture_gpu as IX
from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.program.kernels import dsv4_decode_metadata_twin as T
from verity_vllm.program.registry import dsv4_decode_metadata as R
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/dsv4-decode-metadata-capture/v1"
RECORD = "dsv4_decode_metadata_capture.json"
I32_MIN, I32_MAX, M32 = -(1 << 31), (1 << 31) - 1, 0xFFFFFFFF
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
#: per case, its kernel and the max_model_len its block-table row serves
MAX_LEN = {"swa": 1 << 20, "swa-4096": 4096, "c128": 1 << 20, "c128-4096": 4096}
PHASE = {"swa": "swa", "swa-4096": "swa", "c128": "c128a", "c128-4096": "c128a"}
CASES_L = tuple(MAX_LEN)
#: the HCA builder's kernel block (sparse_mla.py:54) and ratio; the SWA cache's block (sparse_swa.py:92)
KERNEL_BLOCK, HCA_R, SWA_BLOCK = 256, 128, 64
#: the edge step's width where it is not the served one: a step whose longest sequence is 65,664 to 131,199 tokens
EDGE_W = {"c128": 1024}
#: the served positions, one decode token per request (16 = max_num_seqs): each case's from 0 to max_model_len - 1
POSITIONS = {"swa": (0, 1, 126, 127, 128, 255, 1000, 4095, 65535, 100000, 262143, 524287, 999999, (1 << 20) - 65, (1 << 20) - 2,
                     (1 << 20) - 1),
             "swa-4096": (0, 1, 63, 64, 126, 127, 128, 191, 1000, 2047, 2500, 3000, 3967, 4031, 4094, 4095),
             "c128": (0, 126, 127, 128, 255, 383, 1000, 65535, 131071, 262143, 524287, 524415, 999999, (1 << 20) - 129,
                      (1 << 20) - 2, (1 << 20) - 1),
             "c128-4096": (0, 126, 127, 128, 255, 256, 383, 1000, 1023, 2047, 2048, 2500, 3000, 3967, 4094, 4095)}
PADDING, PREFILL = 2, 2
#: each phase's steps: the HCA kernel's also `one` (the served step's last request alone) and `full` (its 16 requests, no
#: padding), the num_decode_tokens Triton passes as the constant 1 and as a multiple of 16, as a served step of one request and of
#: max_num_seqs do; the window kernel takes no token count
KINDS = {"swa": ("served", "edge"), "c128a": ("served", "edge", "one", "full")}
SEEDS = {"served": 301, "edge": 302, "one": 401, "full": 402}
#: the SWA edge step's requests' query lengths, and the HCA edge step's requests and tokens per request
SWA_EDGE_QLEN = (1, 2, 3, 4) * 3
EDGE_REQUESTS, EDGE_PER_REQUEST = 6, 7
EDGE_BYTES = (0, 1, 2, 0x80, 0xFF)
NUM_BLOCKS = 1 << 18
SENTINEL = 0x5A5A5A5A
#: entries past the words of `query_start_loc`, `seq_lens`, `token_to_req_indices` and the positions and slots, rows of the
#: block table past the requests', bytes past the tokens', rows past the tokens' in the launchers' row buffers, entries past them
#: in the length buffers, and the direct launch's offsets (rows, entries) into its sentinel buffers
REQ_PAD, TABLE_PAD, VALID_PAD, ROWS_PAD, LENS_PAD, WS_OFF, LENS_OFF = 4, 2, 4, 3, 8, 1, 4
TRITON_BLOCK = 1024
DEVICE = "cuda"
JOB_TOKENS = 8
QUICK_REF_EVERY = 16
TOTALS = ("words", "mismatch", "reference", "twin_mismatch")
TRITON = "3.7.1"
KERNELS = {"swa": "_compute_swa_indices_and_lens_kernel", "c128a": "_build_c128a_topk_metadata_kernel"}
PTX_COUNT = re.compile(r"\b(div\.[su]\d+|rem\.[su]\d+|mul\.lo\.s\d+|mad\.lo\.s32|mul\.wide\.s32|mad\.wide\.s32|min\.s64|max\.s32|"
                       r"popc\.b32|redux\.sync\.add\.s32|shfl\.sync\.bfly\.b32|ld\.global\.b8|ld\.global\.b64|setp\.ne\.b16|and\.b16|"
                       r"shr\.[su]\d+|shl\.b\d+|sub\.s32)(?=[\s;])")
#: Triton 3.7.1's sm_90a PTX of each kernel at the served arguments and four warps: eight lanes a thread
PTX_COUNTS = {"swa": {"and.b16": 1, "div.s32": 8, "ld.global.b8": 1, "mad.lo.s32": 8, "mad.wide.s32": 15, "max.s32": 2,
                      "mul.lo.s32": 11, "mul.wide.s32": 1, "setp.ne.b16": 1, "shl.b64": 1, "shr.u32": 2, "sub.s32": 12},
              "c128a": {"and.b16": 3, "div.s32": 8, "div.s64": 1, "div.u32": 1, "ld.global.b64": 2, "mad.lo.s32": 8,
                        "mad.wide.s32": 13, "min.s64": 1, "mul.lo.s32": 11, "popc.b32": 1, "redux.sync.add.s32": 1,
                        "shfl.sync.bfly.b32": 2, "shl.b16": 7, "shl.b32": 3, "shl.b64": 2, "shr.u32": 1, "sub.s32": 9}}
#: the HCA kernel at num_decode_tokens == 1, which Triton passes as a constant: only token 0 decodes, so the decode branch's
#: addresses fold (no shl.b64, one mul.lo.s32 and mad.wide.s32 fewer) and a prefill token's row token_idx - 1 is an offset (one
#: sub.s32 fewer); the words' arithmetic is PTX_COUNTS["c128a"]'s, which a multiple of 16 also compiles to
PTX_ONE = {**{k: v for k, v in PTX_COUNTS["c128a"].items() if k != "shl.b64"}, "mad.wide.s32": 12, "mul.lo.s32": 10, "sub.s32": 8}
#: each kernel's specializations the cases compile, by name: every one must appear in the cache, and every file must be one
PTX_SPECS = {"swa": {"served": PTX_COUNTS["swa"]},
             "c128a": {"num_decode_tokens": PTX_COUNTS["c128a"], "num_decode_tokens=1": PTX_ONE}}
#: the installed vLLM's files the Definitions and this script were read from, by sha256 at d9105ea80
SOURCES = {"v1/attention/backends/mla/sparse_swa.py": "442673bbedfc44d415e26da12e7ea7427690cd4677972ea2c16065b9011d00d2",
           "models/deepseek_v4/sparse_mla.py": "98b829059c5381e48a60545c019878b86c0fc2314664664eb685215ec2129251",
           "models/deepseek_v4/nvidia/flashmla.py": "28f636943e330702ce1d02ad1a6a242237bbce2503ee55f167274afa7d1ca0fe",
           "models/deepseek_v4/attention.py": "b6b32d5a98737ba2d7e619b66de0b055f6189f68d513d9dc1e11f05aaa362616",
           "model_executor/warmup/jit_warmup_triton_helper.py": "85d01d62ab52d8bedab5201dc885ef3fd1694b772b241c258e18fb135ad84458"}
_error, _save, _load = FX._error, FX._save, FX._load
_chunks, _tally, _ref_rows, _flat_out, _version = CX._chunks, CX._tally, CX._ref_rows, CX._flat_out, CX._version
_launch_ok, _words_ok = CX._launch_ok, CX._words_ok
environment, stack = IX.environment, IX.stack


def case_name(case: str, kind: str) -> str:
    return case if kind == "served" else f"{case}-{kind}"


def statics(case: str, kind: str = "served") -> dict[str, int]:
    st = dict(R.DSV4_DECODE_METADATA[case])
    if kind == "edge" and case in EDGE_W:
        st["W"] = EDGE_W[case]
    return st


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def stride(case: str) -> int:
    """The HCA buffers' width, c128a_max_compressed = cdiv(cdiv(max_model_len, 128), 128) * 128 (sparse_mla.py:150-158)."""
    return _cdiv(_cdiv(MAX_LEN[case], HCA_R), 128) * 128


def _w(x) -> np.ndarray:
    return (np.asarray(x, dtype=np.int64) & M32).astype(np.uint32)


def _w64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64).astype(np.uint64)


# ------------------------------------------------------------------------------------------------------------------ operands
def block_rows(rng, counts, nb: int) -> np.ndarray:
    """Per request its ``counts[r]`` distinct block ids below NUM_BLOCKS, then junk words, as words (requests, nb)."""
    rows = rng.integers(0, 1 << 32, (len(counts), nb), dtype=np.uint64).astype(np.int64)
    for r, n in enumerate(counts):
        rows[r, :n] = rng.choice(NUM_BLOCKS, n, replace=False)
    return _w(rows)


def edge_table(nb: int, bs: int, rows: int, rng) -> np.ndarray:
    """``rows`` block rows of six kinds in turn: block numbers 2^31 / BS + j (each product by BS wraps int32); INT32_MAX,
    INT32_MIN, -1, 0, 1 and the wrap's neighbours; random words; ids below NUM_BLOCKS; every word -1; random words."""
    wrap = (1 << 31) // bs
    special = np.resize(np.array([I32_MAX, I32_MIN, -1, 0, 1, 2 * wrap - 1, wrap, wrap - 1]), nb)
    words = rng.integers(0, 1 << 32, (2, nb), dtype=np.uint64).astype(np.int64)
    kinds = np.stack([wrap + np.arange(nb), special, words[0], rng.integers(0, NUM_BLOCKS, nb), np.full(nb, -1), words[1]])
    return _w(kinds[np.arange(rows) % len(kinds)])


def swa_edges(case: str) -> tuple[int, ...]:
    """The SWA edge step's requests' newest positions: the window's, a block's and the row's edges."""
    st = statics(case)
    SW, BS, span = st["SW"], st["BS"], st["NB"] * st["BS"]
    return (0, 1, SW - 2, SW - 1, SW, BS - 1, BS, BS + 1, 2 * BS + 5, span - SW, span - 2, span - 1)


def swa_requests(case: str, kind: str, seed: int) -> dict[str, Any]:
    """One SWA decode step's operands as words: `qsl` (requests + 1), `seq` (requests), `req` and `valid` (tokens), the block
    table (requests, NB), and each token's position `pos` (the kernel's (seq - (qe - qs)) + t - qs)."""
    rng = np.random.default_rng(seed)
    st = statics(case)
    if kind == "served":
        pos = np.asarray(POSITIONS[case], dtype=np.int64)
        nreq = pos.size
        table = block_rows(rng, [_cdiv(int(p) + 1, st["BS"]) for p in pos], st["NB"])
        req = np.concatenate([np.arange(nreq), np.zeros(PADDING)]).astype(np.int32)
        valid = np.concatenate([np.ones(nreq), np.zeros(PADDING)]).astype(np.uint8)
        return {"qsl": _w(np.arange(nreq + 1)), "seq": _w(pos + 1), "req": req, "valid": valid, "table": table,
                "pos": np.concatenate([pos, pos[:1].repeat(PADDING)])}
    edges = np.asarray(swa_edges(case), dtype=np.int64)
    qlen = np.asarray(SWA_EDGE_QLEN, dtype=np.int64)
    nreq = qlen.size
    req = rng.permutation(np.repeat(np.arange(nreq), qlen)).astype(np.int32)
    t = np.arange(req.size)
    c = np.asarray([edges[r] - t[req == r].max() for r in range(nreq)])
    qsl = (I32_MAX - 9) + np.concatenate([[0], np.cumsum(qlen)])
    seq = c + qsl[1:]
    valid = rng.permutation(np.resize(np.array(EDGE_BYTES + (1, 1, 1)), req.size)).astype(np.uint8)
    return {"qsl": _w(qsl), "seq": _w(seq), "req": req, "valid": valid, "table": edge_table(st["NB"], st["BS"], nreq, rng),
            "pos": c[req] + t}


def c128_edge_positions(case: str, n: int, rng) -> np.ndarray:
    """n int64 positions: each giving q = (position + 1) // 128 at the lane counts' edges (0, 1, the block and its neighbours, the
    row's last lanes, past the width where the row holds the width) at both ends of its R, the int64 ends (INT64_MAX - 1 and 2^40,
    whose rows are full, where the row holds the width; INT64_MAX's position + 1 wraps negative), then draws."""
    st = statics(case, "edge")
    W, R, BS, NB = st["W"], st["R"], st["BS"], st["NB"]
    top = min(W, BS * NB)
    held = W <= BS * NB
    qs = [0, 1, BS - 1, BS, BS + 1, top - 1, top] + ([W + 1, 2 * W] if held else [])
    ends = [-1, -2, -R, -R - 1, I64_MIN, I64_MIN + 1, I64_MAX] + ([I64_MAX - 1, 1 << 40] if held else [])
    kinds = [q * R + off - 1 for q in qs for off in (0, R - 1)] + ends
    draws = rng.integers(0, top + 1, n - len(kinds)) * R + rng.integers(0, R, n - len(kinds)) - 1
    return rng.permutation(np.concatenate([np.asarray(kinds, dtype=np.int64), draws]))


def c128_requests(case: str, kind: str, seed: int) -> dict[str, Any]:
    """One HCA step's operands: the decode tokens' `req`, then the PREFILL prefill tokens; every token's int64 position `pos` and
    slot `slot` as uint64 words (decode tokens first); the block table (requests, NB) as words.  `one` is the served step's last
    request alone (its row the longest) and `full` its 16 requests without padding."""
    rng = np.random.default_rng(seed)
    st = statics(case, kind)
    W, R, BS, NB = st["W"], st["R"], st["BS"], st["NB"]
    pre = np.asarray([min(W * R, MAX_LEN[case]) - 1, 300], dtype=np.int64)
    if kind != "edge":
        pos = np.asarray(POSITIONS[case][-1:] if kind == "one" else POSITIONS[case], dtype=np.int64)
        nreq, pad = pos.size, PADDING if kind == "served" else 0
        table = block_rows(rng, [_cdiv(min((int(p) + 1) // R, W), BS) for p in pos], NB)
        slot = rng.integers(0, 1 << 31, nreq)
        return {"req": np.concatenate([np.arange(nreq), np.zeros(pad)]).astype(np.int32), "table": table,
                "pos": _w64(np.concatenate([pos, np.zeros(pad), pre])),
                "slot": _w64(np.concatenate([slot, np.full(pad, -1), rng.integers(0, 1 << 31, PREFILL)]))}
    n = EDGE_REQUESTS * EDGE_PER_REQUEST
    req = rng.permutation(np.repeat(np.arange(EDGE_REQUESTS), EDGE_PER_REQUEST)).astype(np.int32)
    slots = np.resize(np.array([5, 0, -1, -2, I64_MIN, I64_MAX, 1 << 40]), n)
    slot = rng.permutation(np.where(rng.random(n) < 0.3, rng.integers(0, 1 << 40, n), slots))
    return {"req": req, "table": edge_table(NB, BS, EDGE_REQUESTS, rng),
            "pos": _w64(np.concatenate([c128_edge_positions(case, n, rng), pre])),
            "slot": _w64(np.concatenate([slot, rng.integers(0, 1 << 31, PREFILL)]))}


def requests(case: str, kind: str, seed: int) -> dict[str, Any]:
    return (swa_requests if PHASE[case] == "swa" else c128_requests)(case, kind, seed)


def prefill_rows(pos: np.ndarray, W: int, R: int) -> np.ndarray:
    """The kernel's prefill branch (sparse_mla.py:417-427) on int64 position words: lane j < W is j where j < (position + 1) // R
    (int64, truncating), else -1, as words."""
    t = (np.asarray(pos, dtype=np.uint64) + np.uint64(1)).view(np.int64)
    q = np.where(t < 0, (t + (R - 1)) // R, t // R)
    j = np.arange(W, dtype=np.int64)
    return _w(np.where(j < q[:, None], j, -1))


# ----------------------------------------------------------------------------------------------------------------- the GPU side
def sources(root: Path) -> dict[str, Any]:
    """Each of SOURCES under the vLLM package directory ``root``: True when its sha256 is the one read, False when it differs,
    None when it is missing."""
    files: dict[str, bool | None] = {}
    for rel, want in SOURCES.items():
        p = root / rel
        files[rel] = hashlib.sha256(p.read_bytes()).hexdigest() == want if p.is_file() else None
    return {"files": files, "ok": all(v is True for v in files.values())}


def selection() -> dict[str, Any]:
    """vLLM's DSV4 attention class, the builders its decode reads and the launchers they call, the kernel block sizes, Triton, the
    sources."""
    import vllm
    from vllm.config import get_current_vllm_config
    from vllm.models.deepseek_v4 import sparse_mla as SM
    from vllm.models.deepseek_v4.nvidia import flashmla as F
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls
    from vllm.v1.attention.backends.mla import sparse_swa as SW

    cls = _select_dsv4_attn_cls(get_current_vllm_config())
    swa, hca = cls.swa_backend_cls, cls.backend_cls
    sizes = {"swa": [getattr(b, "base", b) for b in swa.get_supported_kernel_block_sizes()],
             "hca": [getattr(b, "base", b) for b in hca.get_supported_kernel_block_sizes()]}
    rec: dict[str, Any] = {"attn_cls": cls.__name__, "swa_backend": getattr(swa, "__name__", repr(swa)), "backend": hca.__name__,
                           "block_sizes": sizes, "triton": _version("triton"), "sources": sources(Path(vllm.__file__).parent)}
    swa_b, hca_b = swa.get_builder_cls(), hca.get_builder_cls()
    rec["checks"] = {"attn_cls": cls is F.DeepseekV4FlashMLAAttention,
                     "swa_builder": swa is F.DeepseekSparseSWAFlashMLABackend and swa_b is F.DeepseekSparseSWAFlashMLAMetadataBuilder
                     and swa_b.build is SW.DeepseekSparseSWAMetadataBuilder.build,
                     "swa_kernel": isinstance(SW._COMPUTE_SWA_INDICES_AND_LENS_KERNEL, SW.ComputeSWAIndicesAndLensKernel)
                     and SW.ComputeSWAIndicesAndLensKernel.kernel is SW._compute_swa_indices_and_lens_kernel,
                     "hca_builder": hca is SM.DeepseekV4FlashMLABackend and hca_b is SM.DeepseekV4FlashMLAMetadataBuilder
                     and hca_b.build is SM.DeepseekV4SparseMLAMetadataBuilder.build
                     and hca_b._build_c128a_metadata is SM.DeepseekV4SparseMLAMetadataBuilder._build_c128a_metadata,
                     "block_size": sizes == {"swa": [SWA_BLOCK], "hca": [KERNEL_BLOCK]}
                     and R.DSV4_DECODE_METADATA["swa"]["BS"] == SWA_BLOCK
                     and R.DSV4_DECODE_METADATA["c128"]["BS"] == KERNEL_BLOCK // HCA_R,
                     "triton": rec["triton"] == TRITON, "sources": rec["sources"]["ok"]}
    rec["ok"] = all(rec["checks"].values())
    return rec


def _ref_every(quick: bool) -> int:
    return QUICK_REF_EVERY if quick else 1


def _dev(a: np.ndarray):
    """A host array on the device, 4- and 8-byte words as int32 and int64."""
    import torch

    a = np.ascontiguousarray(a)
    return torch.from_numpy(a.view({4: np.int32, 8: np.int64}[a.itemsize]) if a.itemsize in (4, 8) else a).to(DEVICE)


def _pad(a: np.ndarray, rows: int, fill: int) -> np.ndarray:
    return np.concatenate([a, np.full((rows,) + a.shape[1:], fill, dtype=a.dtype)])


def _sentinel(*shape: int):
    import torch

    return torch.full(shape, SENTINEL, dtype=torch.int32, device=DEVICE)


def launch_swa(case: str, o: dict[str, Any]) -> dict[str, Any]:
    """`_COMPUTE_SWA_INDICES_AND_LENS_KERNEL` on one step's operands as the SWA builder passes them, twice into fresh sentinel
    buffers (rows (tokens + 3, 1, SW), lengths (tokens + 8,)), then its kernel at the launcher's arguments into sentinel buffers at
    an offset (rows (tokens + 2, SW), lengths (tokens + 8,)): the buffers as words, and whether every operand buffer is the bytes
    it was."""
    import torch
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _COMPUTE_SWA_INDICES_AND_LENS_KERNEL,
        _compute_swa_indices_and_lens_kernel,
    )

    st = statics(case)
    SW, n, nreq = st["SW"], o["req"].size, o["seq"].size
    bufs = {"qsl": _dev(_pad(o["qsl"], REQ_PAD, SENTINEL)), "seq": _dev(_pad(o["seq"], REQ_PAD, SENTINEL)),
            "req": _dev(_pad(o["req"].astype(np.uint32), REQ_PAD, SENTINEL)),
            "table": _dev(_pad(o["table"], TABLE_PAD, SENTINEL)), "valid": _dev(_pad(o["valid"], VALID_PAD, 0xA5))}
    before = {k: v.clone() for k, v in bufs.items()}
    table_t, valid_t = bufs["table"][:nreq], bufs["valid"].view(torch.bool)[:n]
    runs = []
    for _ in range(2):
        ws, wl = _sentinel(n + ROWS_PAD, 1, SW), _sentinel(n + LENS_PAD)
        _COMPUTE_SWA_INDICES_AND_LENS_KERNEL(ws, wl, SW, ws.shape[-1], wl, wl, bufs["qsl"], bufs["seq"], bufs["req"], valid_t,
                                             table_t, st["BS"], num_tokens=n, token_offset=0)
        torch.cuda.synchronize()
        runs.append({"ws": CX._u32(ws).reshape(n + ROWS_PAD, SW), "wl": CX._u32(wl)})
    ws, wl = _sentinel(n + WS_OFF + 1, SW), _sentinel(n + LENS_OFF + 4)
    out, lens = ws[WS_OFF:WS_OFF + n], wl[LENS_OFF:LENS_OFF + n]
    _compute_swa_indices_and_lens_kernel[(n,)](out, out.stride(0), lens, SW, SW, lens, lens, bufs["qsl"], bufs["seq"], bufs["req"],
                                               valid_t, table_t, table_t.stride(0), st["BS"], 0, HAS_IMAGE=False,
                                               TRITON_BLOCK_SIZE=TRITON_BLOCK)
    torch.cuda.synchronize()
    return {"runs": runs, "direct": {"ws": CX._u32(ws), "wl": CX._u32(wl)},
            "operands": all(bool(torch.equal(before[k], bufs[k])) for k in bufs)}


def launch_c128(case: str, kind: str, o: dict[str, Any]) -> dict[str, Any]:
    """`build_c128a_topk_metadata` on one step's operands as the HCA builder passes them, twice into fresh sentinel buffers (decode
    rows (tokens + 3, stride), lengths (tokens + 8,), prefill rows (PREFILL + 3, stride)), then its kernel at the launcher's
    arguments into sentinel buffers at an offset: the buffers as words, whether each launch returned the buffers' [:tokens,
    :width] views, and whether every operand buffer is the bytes it was."""
    import torch
    from vllm.models.deepseek_v4.sparse_mla import _build_c128a_topk_metadata_kernel, build_c128a_topk_metadata

    st, S = statics(case, kind), stride(case)
    W, nd, nreq = st["W"], o["req"].size, o["table"].shape[0]
    n = o["pos"].size
    bufs = {"pos": _dev(_pad(o["pos"], REQ_PAD, SENTINEL)), "slot": _dev(_pad(o["slot"], REQ_PAD, SENTINEL)),
            "req": _dev(_pad(np.concatenate([o["req"], np.zeros(n - nd, dtype=np.int32)]).astype(np.uint32), REQ_PAD, SENTINEL)),
            "table": _dev(_pad(o["table"], TABLE_PAD, SENTINEL))}
    before = {k: v.clone() for k, v in bufs.items()}
    pos_t, table_t = bufs["pos"][:n], bufs["table"][:nreq]
    runs, views = [], True
    for _ in range(2):
        gd, dl, pf = _sentinel(n + ROWS_PAD, S), _sentinel(n + LENS_PAD), _sentinel(PREFILL + ROWS_PAD, S)
        g, d, p = build_c128a_topk_metadata(pos_t, st["R"], nd, bufs["req"], table_t, st["BS"], bufs["slot"], gd, dl, pf,
                                            max_compressed_tokens=W)
        torch.cuda.synchronize()
        views &= bool(g.data_ptr() == gd.data_ptr() and tuple(g.shape) == (nd, W) and g.stride(0) == S
                      and d.data_ptr() == dl.data_ptr() and tuple(d.shape) == (nd,)
                      and p.data_ptr() == pf.data_ptr() and tuple(p.shape) == (n - nd, W) and p.stride(0) == S)
        runs.append({"gd": CX._u32(gd), "dl": CX._u32(dl), "pf": CX._u32(pf)})
    gd, dl, pf = _sentinel(n + WS_OFF + 1, S), _sentinel(n + LENS_OFF + 4), _sentinel(PREFILL + WS_OFF + 1, S)
    g, d, p = gd[WS_OFF:], dl[LENS_OFF:], pf[WS_OFF:]
    _build_c128a_topk_metadata_kernel[(n,)](g, g.stride(0), d, p, p.stride(0), pos_t, st["R"], W, nd, bufs["req"], table_t,
                                            table_t.stride(0), st["BS"], bufs["slot"], BLOCK_SIZE=TRITON_BLOCK)
    torch.cuda.synchronize()
    return {"runs": runs, "views": views, "direct": {"gd": CX._u32(gd), "dl": CX._u32(dl), "pf": CX._u32(pf)},
            "operands": all(bool(torch.equal(before[k], bufs[k])) for k in bufs)}


def launch(case: str, kind: str, o: dict[str, Any]) -> dict[str, Any]:
    return launch_swa(case, o) if PHASE[case] == "swa" else launch_c128(case, kind, o)


def launches(case: str, seeds: dict[str, int]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ops = {kind: requests(case, kind, seed) for kind, seed in seeds.items()}
    return ops, {kind: launch(case, kind, ops[kind]) for kind in ops}


def launch_case(work: Path, case: str, quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    i = CASES_L.index(case)
    return observe(work, case, *launches(case, {kind: SEEDS[kind] + 2 * i for kind in KINDS[PHASE[case]]}), quick)


def _outside(shape: tuple[int, ...], *spans: tuple[int, int]) -> np.ndarray:
    """A mask of ``shape`` true outside the box ``spans`` (one (lo, hi) per axis)."""
    m = np.ones(shape, dtype=bool)
    m[tuple(slice(lo, hi) for lo, hi in spans)] = False
    return m


def _untouched(a: np.ndarray, *spans: tuple[int, int]) -> bool:
    return bool((a[_outside(a.shape, *spans)] == SENTINEL).all())


def observe_swa(case: str, kind: str, o: dict[str, Any], L: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    st = statics(case)
    n, (r1, r2), d = o["req"].size, L["runs"], L["direct"]
    checks = {"deterministic": all(np.array_equal(r1[k], r2[k]) for k in ("ws", "wl")),
              "direct": bool(np.array_equal(d["ws"][WS_OFF:WS_OFF + n], r1["ws"][:n])
                             and np.array_equal(d["wl"][LENS_OFF:LENS_OFF + n], r1["wl"][:n])),
              "untouched": all(_untouched(r["ws"], (0, n), (0, st["SW"])) and _untouched(r["wl"], (0, n)) for r in (r1, r2))
              and _untouched(d["ws"], (WS_OFF, WS_OFF + n), (0, st["SW"])) and _untouched(d["wl"], (LENS_OFF, LENS_OFF + n)),
              "operands": L["operands"] is True}
    return checks, r1["ws"][:n], r1["wl"][:n]


def observe_c128(case: str, kind: str, o: dict[str, Any], L: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    st = statics(case, kind)
    W, nd, n = st["W"], o["req"].size, o["pos"].size
    (r1, r2), d = L["runs"], L["direct"]
    checks = {"deterministic": all(np.array_equal(r1[k], r2[k]) for k in ("gd", "dl", "pf")),
              "views": L["views"] is True,
              "direct": bool(np.array_equal(d["gd"][WS_OFF:WS_OFF + nd, :W], r1["gd"][:nd, :W])
                             and np.array_equal(d["dl"][LENS_OFF:LENS_OFF + nd], r1["dl"][:nd])
                             and np.array_equal(d["pf"][WS_OFF:WS_OFF + n - nd, :W], r1["pf"][:n - nd, :W])),
              "untouched": all(_untouched(r["gd"], (0, nd), (0, W)) and _untouched(r["dl"], (0, nd))
                               and _untouched(r["pf"], (0, n - nd), (0, W)) for r in (r1, r2))
              and _untouched(d["gd"], (WS_OFF, WS_OFF + nd), (0, W)) and _untouched(d["dl"], (LENS_OFF, LENS_OFF + nd))
              and _untouched(d["pf"], (WS_OFF, WS_OFF + n - nd), (0, W)),
              "prefill": bool(np.array_equal(r1["pf"][:n - nd, :W], prefill_rows(o["pos"][nd:], W, st["R"]))),
              "operands": L["operands"] is True}
    return checks, r1["gd"][:nd, :W], r1["dl"][:nd]


def observe(work: Path, case: str, ops: dict[str, Any], rows: dict[str, dict[str, Any]],
            quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`launches`' words checked and saved under ``work``: the launch records and the comparison jobs."""
    phase = PHASE[case]
    recs: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for kind, L in rows.items():
        o, name = ops[kind], case_name(case, kind)
        checks, got_rows, got_lens = (observe_swa if phase == "swa" else observe_c128)(case, kind, o, L)
        nd = o["req"].size
        rec = {"phase": phase, "case": name, "tokens": nd, "requests": int(o["table"].shape[0]), "checks": checks}
        if phase == "swa":
            rec["padding"] = int((o["valid"] == 0).sum())
            arrays = {k: o[k] for k in ("qsl", "seq", "req", "valid", "table", "pos")}
        else:
            rec["padding"] = int((T.s64(o["slot"][:nd]) < 0).sum())
            arrays = {"req": o["req"], "table": o["table"], "pos": o["pos"][:nd], "slot": o["slot"][:nd]}
        recs.append(rec)
        paths = _save(work / f"{phase}-{name}", **arrays, got_rows=got_rows, got_lens=got_lens)
        every = _ref_every(quick) if kind in ("served", "full") else 1
        jobs += [{"phase": phase, "case": name, "statics": statics(case, kind), "paths": paths, "rows": [a, b], "ref_every": every}
                 for a, b in _chunks(nd, JOB_TOKENS)]
    return recs, jobs


# ------------------------------------------------------------------------------------------------------------------ comparisons
def compare_swa(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Tokens [lo, hi): each window row and its length against the twin, the reference `Dsv4SwaIndices_v1` on every ref_every-th."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    lo, hi = job["rows"]
    st = job["statics"]
    req = np.asarray(a["req"][lo:hi])
    t = _w(np.arange(lo, hi))
    qsl = np.asarray(a["qsl"])
    qs, qe, seq = qsl[req], qsl[req + 1], np.asarray(a["seq"])[req]
    blocks, valid = np.asarray(a["table"])[req], np.asarray(a["valid"][lo:hi])
    rows, lens, declined = T.swa_indices(t, qs, qe, seq, blocks, valid, **st)
    if declined.any():
        raise ValueError(f"tokens with a live lane outside their block row: {(np.flatnonzero(declined) + lo)[:8].tolist()}")
    want = np.concatenate([rows, lens[:, None]], axis=1)
    got = np.concatenate([a["got_rows"][lo:hi], np.asarray(a["got_lens"][lo:hi])[:, None]], axis=1)
    _tally(rec, want, got, np.arange(lo, hi), "token")
    fn = bind(R.Dsv4SwaIndices, **st)
    for i in _ref_rows(lo, hi, job["ref_every"]):
        k = i - lo
        y = evaluate(fn, int(t[k]), int(qs[k]), int(qe[k]), int(seq[k]), [int(v) for v in blocks[k]], int(valid[k]))
        rec["reference"] += 1
        rec["twin_mismatch"] += int(_flat_out(y) != want[k].astype(np.int64).tolist())


def compare_c128(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Decode tokens [lo, hi): each row and its length against the twin, the reference `Dsv4C128aGlobalTopk_v1` on every
    ref_every-th."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    lo, hi = job["rows"]
    st = job["statics"]
    pos, slot = np.asarray(a["pos"][lo:hi]), np.asarray(a["slot"][lo:hi])
    blocks = np.asarray(a["table"])[np.asarray(a["req"][lo:hi])]
    rows, lens, declined = T.c128a_global_topk(pos, slot, blocks, **st)
    if declined.any():
        raise ValueError(f"tokens with a live lane past their block row: {(np.flatnonzero(declined) + lo)[:8].tolist()}")
    want = np.concatenate([rows, lens[:, None]], axis=1)
    got = np.concatenate([a["got_rows"][lo:hi], np.asarray(a["got_lens"][lo:hi])[:, None]], axis=1)
    _tally(rec, want, got, np.arange(lo, hi), "token")
    fn = bind(R.Dsv4C128aGlobalTopk, **st)
    for i in _ref_rows(lo, hi, job["ref_every"]):
        k = i - lo
        y = evaluate(fn, int(pos[k]), int(slot[k]), [int(v) for v in blocks[k]])
        rec["reference"] += 1
        rec["twin_mismatch"] += int(_flat_out(y) != want[k].astype(np.int64).tolist())


def compare(job: dict[str, Any]) -> dict[str, Any]:
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    try:
        (compare_swa if job["phase"] == "swa" else compare_c128)(job, rec)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "error": _error(e)}
    return {"phase": job["phase"], "case": job["case"], **rec}


# --------------------------------------------------------------------------------------------------------------- ptx and bind
def ptx_counts(text: str) -> dict[str, int]:
    return dict(sorted(Counter(PTX_COUNT.findall(text)).items()))


def ptx(triton_cache: Path, outdir: Path) -> dict[str, Any]:
    """The Triton cache's PTX of KERNELS: each file's target, sha256, PTX_COUNT counts and the PTX_SPECS specialization they are;
    ok when every file is one of its kernel's and each of those has a file.  The PTX, and the SASS where Triton's bundled
    `cuobjdump` disassembles the cubin, are copied to `<out>/ptx/`."""
    dest = outdir / "ptx"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import triton

        cand = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump"
        tool = str(cand) if cand.exists() else shutil.which("cuobjdump")
    except Exception:  # noqa: BLE001
        tool = shutil.which("cuobjdump")
    rec: dict[str, Any] = {"kernels": {}}
    for key, kernel in KERNELS.items():
        entries = []
        for f in sorted(glob.glob(str(triton_cache / "**" / f"{kernel}.ptx"), recursive=True)):
            text = Path(f).read_text(errors="replace")
            sha = hashlib.sha256(text.encode()).hexdigest()
            name = f"{key}.{len(entries)}.{sha[:12]}"
            shutil.copy(f, dest / f"{name}.ptx")
            counts = ptx_counts(text)
            e: dict[str, Any] = {"sha256": sha, "target": re.findall(r"\.target\s+(\S+)", text), "counts": counts,
                                 "spec": next((s for s, c in PTX_SPECS[key].items() if c == counts), None)}
            cubin = Path(f).with_suffix(".cubin")
            if tool and cubin.exists():
                try:
                    sass = subprocess.run([tool, "-sass", str(cubin)], capture_output=True, text=True, check=True).stdout
                    (dest / f"{name}.sass").write_text(sass)
                except Exception as err:  # noqa: BLE001
                    e["sass_error"] = str(err)[-300:]
            entries.append(e)
        rec["kernels"][key] = entries
    rec["checks"] = {"target": all(e["target"] == ["sm_90a"] for es in rec["kernels"].values() for e in es),
                     **{key: all(e["spec"] for e in es) and {e["spec"] for e in es} == set(PTX_SPECS[key])
                        for key, es in rec["kernels"].items()}}
    rec["ok"] = all(rec["checks"].values())
    return rec


def bind_check() -> dict[str, Any]:
    """The Programs bind at DeepSeek-V4-Pro's statics and the edge width."""
    from verity.ml.circuits.defs import bind

    out = {}
    for case in CASES_L:
        fn = R.Dsv4SwaIndices if PHASE[case] == "swa" else R.Dsv4C128aGlobalTopk
        for kind in ("served", "edge"):
            st = statics(case, kind)
            name = fn.name + "_v1{" + ",".join(f"{k}={v}" for k, v in st.items()) + "}"
            try:
                out[name] = bool(bind(fn, **st).signature())
            except Exception as e:  # noqa: BLE001
                out[name] = _error(e)
    return {"bindings": out, "ok": all(v is True for v in out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
def gpu(work: Path, outdir: Path, quick: bool, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The selection, the stack and each case's launches in order; each error goes to doc["errors"].  Returns the jobs."""
    import torch

    jobs: list[dict[str, Any]] = []

    def step(name: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            print(f"[dsv4-dec-meta] {name} ERROR {doc['errors'][name][:600]}", flush=True)
            return None

    doc["selection"] = step("selection", selection) or {"ok": False}
    print(f"[dsv4-dec-meta] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
    doc["stack"] = step("stack", stack) or {}
    for case in CASES_L:
        r = step(case, lambda case=case: launch_case(work, case, quick))
        if r is not None:
            doc["launches"] += r[0]
            jobs += r[1]
            for x in r[0]:
                print(f"[dsv4-dec-meta] {x['phase']:5s} {x['case']:14s} {json.dumps(x['checks'])}", flush=True)
        torch.cuda.empty_cache()
    return jobs


CASES = tuple((PHASE[c], case_name(c, k)) for c in CASES_L for k in KINDS[PHASE[c]])


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's verdict: its launch checks, its words and its errors (its case's launch, its comparisons), so a NO-GO names
    what failed."""
    out: dict[str, dict[str, Any]] = {}
    for base, phase, case in ((c, PHASE[c], case_name(c, k)) for c in CASES_L for k in KINDS[PHASE[c]]):
        launches_ = [x for x in doc["launches"] if x["phase"] == phase and x["case"] == case]
        comps = [r for r in doc.get("comparisons", []) if r["phase"] == phase and r["case"] == case]
        t = {k: sum(r[k] for r in comps) for k in TOTALS}
        errors = [e for e in doc["errors"] if e == base or e == f"compare/{phase}/{case}"]
        ok = bool(launches_) and all(_launch_ok(x) for x in launches_)
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
    print(f"[dsv4-dec-meta] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-DECODE-METADATA GATE-FAIL", flush=True)
        raise SystemExit(3)
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        jobs = gpu(work, outdir, quick, doc)
    doc.setdefault("selection", {"ok": False})
    for name, fn in (("ptx", lambda: ptx(work / "triton_cache", outdir)), ("bind", bind_check)):
        try:
            doc[name] = fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            doc[name] = {"ok": False}
    print(f"[dsv4-dec-meta] ptx {json.dumps(doc['ptx'])[:2000]}", flush=True)
    print(f"[dsv4-dec-meta] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
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
        print(f"[dsv4-dec-meta] words {r['phase']:5s} {r['case']:14s} words={r['words']} !={r['mismatch']} ref={r['reference']} "
              f"twin!={r['twin_mismatch']} first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[dsv4-dec-meta] case {key:20s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} "
              f"!={t['mismatch']} ref={t['reference']} twin!={t['twin_mismatch']} errors={c['errors']}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"DSV4-DECODE-METADATA {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro's SWA and HCA decode index rows on this Hopper GPU against their "
                                             "Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and ptx/")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (the operands, outputs and Triton cache)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true",
                    help=f"the reference on every {QUICK_REF_EVERY}th served token: a first signal in minutes")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
