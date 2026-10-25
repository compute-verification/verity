"""DeepSeek-V4-Pro's ratio-4 (CSA) decode index row on Hopper against its Program: the H200's go/no-go for binding
`dsv4_decode_indices.Dsv4GlobalTopk_v1{TOPK,BS,NB}` (vLLM d9105ea80's Triton `compute_global_topk_indices_and_lens`, the slot ids
of the paged compressed cache and their count that FlashMLA's sparse decode reads as `extra_indices_in_kvcache` and
`extra_topk_length`) at DeepSeek-V4-Pro's statics on one TP8 rank: the indexer's top-k of 1024, 64 compressed rows per block (the
kernel block 256 over ratio 4), and the block table's row at max_model_len 1,048,576 (4096 blocks) and at 4096 (16).  Line
numbers under `vllm/models/deepseek_v4/` unless named.

One GPU process writes `<out>/dsv4_decode_index_row_capture.json`, the property record `dsv4_decode_index_row_capture`
(`properties.record`), and the kernel's PTX (and SASS where Triton's bundled `cuobjdump` runs) under `<out>/ptx/`; it prints one
line per launch and per case, then `DSV4-DECODE-INDEX-ROW GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs
first and nothing else runs when it fails (exit 3).  A case that raises records its error and fails its cases without stopping
the others.  The operands are the integers a decode step holds (`requests`), so no model is loaded.  The kernel reads no position:
its operands are the indexer's top-k row, the request's block-table row and the token's valid byte.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `_select_dsv4_attn_cls` picks `DeepseekV4FlashMLAAttention`, whose ratio-4 decode (nvidia/flashmla.py:186-207) calls
             the `compute_global_topk_indices_and_lens` of common/ops/cache_utils.py at block size `block_size // 4`, its
             backend `DeepseekV4FlashMLABackend` (flashmla.py:54) of kernel block size 256 (sparse_mla.py:54); the installed
             Triton is the 3.7.1 whose PTX was read, and the installed SOURCES are the files read, since the launches call the
             function directly rather than through the attention layer.
  row        per case, `compute_global_topk_indices_and_lens` (cache_utils.py:435-466) as the decode calls it: the top-k rows a
             [:tokens] slice of an int32 buffer TOPK wide (`topk_indices_buffer`, nvidia/model.py:1305-1309), the rows past the
             tokens' the sentinel; `token_to_req_indices` an int32 buffer longer than the tokens, its tail the sentinel; the block
             table a [:requests] slice of an int32 table NB wide; `is_valid_token` a bool view of bytes (v1/attention/backends/
             mla/sparse_swa.py:498-502, 555-556).  `served`: 16 requests (max_num_seqs) of one decode token each, at positions
             from 0 (an empty row) to max_model_len - 1 (a full row whose indices reach the row's last block), each top-k row
             distinct ids below the token's compressed count (position + 1) // 4, the newest among them, and -1 past them, each
             block row the request's block ids and junk words past them; two padding tokens after them (byte 0, request 0).
             `edge`: six requests' tokens in a shuffled map, several to a request; bytes 0, 1, 2, 0x80 and 0xFF (a bool tensor's
             bytes, true when nonzero); rows empty, short, full, of the negative kinds (-1, -2, -BS, INT32_MIN, random), of the
             block edges and the row's last slot, and draws; block rows whose products by BS wrap int32 (2^25 and up, INT32_MAX,
             negative words, random words).  Each launch twice, then the kernel itself once at the wrapper's arguments into
             sentinel-filled buffers: every token's TOPK slots and its length against `dsv4_decode_indices_twin.global_topk` (the
             kernel's stores) and the reference `Dsv4GlobalTopk_v1` on every token (every 16th served one under --quick); the
             launches the same words, the outputs (tokens, TOPK) and (tokens,), the operands unchanged, and every other word of
             the sentinel buffers untouched.
  ptx        the Triton cache's PTX of the kernel for sm_90a, at least one specialization per NB (the block table's stride is a
             constexpr), each holding PTX_COUNTS: the arithmetic the Definition was read from (`//` and `%` by 64 as shifts and
             masks, 16 shr.s32, 10 shr.u32, 8 sub.s32; the product as shl.b32; the count as popc.b32, redux.sync.add.s32 and two
             shfl.sync.bfly.b32; the byte as one ld.global.b8; no division, remainder or mul.lo).
  bind       the Programs bind at the served statics.

GO needs every case's words compared with 0 mismatches, the references with 0 disagreements with the twin, every launch check,
the selection, the PTX and the bindings.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/dsv4_decode_index_row_capture_gpu.py --out DIR [--scratch DIR] [--quick]
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

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
from verity.ml.definitions.numerics import dsv4_decode_indices_twin as T
from verity.ml.definitions.registry import dsv4_decode_indices as R
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/dsv4-decode-index-row-capture/v1"
RECORD = "dsv4_decode_index_row_capture.json"
I32_MIN, I32_MAX, M32 = -(1 << 31), (1 << 31) - 1, 0xFFFFFFFF
#: the backend's kernel block size (sparse_mla.py:54) and the layer's compress ratio: BS = 256 // 4
KERNEL_BLOCK, RATIO = 256, 4
#: per case, the max_model_len its block-table row serves: NB = cdiv(max_model_len, 256)
MAX_LEN = {"c4": 1 << 20, "c4-4096": 4096}
CASES_L = tuple(MAX_LEN)
#: the served positions, one decode token per request (16 = max_num_seqs): each case's from 0 to max_model_len - 1
POSITIONS = {"c4": (0, 2, 3, 7, 255, 256, 1000, 4091, 4095, 65535, 100000, 262143, 524287, 999999, (1 << 20) - 2, (1 << 20) - 1),
             "c4-4096": (0, 1, 3, 4, 63, 255, 256, 257, 1000, 2047, 3000, 4000, 4091, 4092, 4094, 4095)}
PADDING = 2
#: the edge chunk's requests and tokens per request
EDGE_REQUESTS, EDGE_PER_REQUEST = 6, 7
EDGE_BYTES = (0, 1, 2, 0x80, 0xFF)
NUM_BLOCKS = 1 << 18
SENTINEL = 0x5A5A5A5A
#: rows of the top-k buffer past the tokens', entries of `token_to_req_indices` past them, rows of the block table past the
#: requests', bytes past the valid tokens', and the sentinel rows and lengths around the direct launch's outputs
TOPK_PAD, REQ_PAD, TABLE_PAD, VALID_PAD, WS_OFF, LENS_OFF = 3, 4, 2, 4, 1, 4
TRITON_BLOCK = 1024
DEVICE = "cuda"
JOB_TOKENS = 16
QUICK_REF_EVERY = 16
TOTALS = ("words", "mismatch", "reference", "twin_mismatch")
TRITON = "3.7.1"
KERNELS = {"row": "_compute_global_topk_indices_and_lens_kernel"}
PTX_COUNT = re.compile(r"\b(shr\.[su]\d+|shl\.b32|sub\.s32|and\.b32|selp\.b32|popc\.b32|redux\.sync\.add\.s32|shfl\.sync\.bfly\.b32|"
                       r"ld\.global\.b8|mad\.wide\.s32|mul\.wide\.s32|mul\.lo\.s32|mul\.hi\.[su]\d+|div\.[su]\d+|rem\.[su]\d+|"
                       r"min\.[su]\d+|max\.[su]\d+)(?=[\s;])")
#: Triton 3.7.1's sm_90a PTX of the kernel at top-k 1024, block size 64 and four warps, the same at both block-table strides
#: (4096 and 16; they differ in one shl.b32's amount): eight words a thread
PTX_COUNTS = {"row": {"and.b32": 14, "ld.global.b8": 1, "mad.wide.s32": 9, "mul.wide.s32": 1, "popc.b32": 1, "redux.sync.add.s32": 1,
                      "selp.b32": 9, "shfl.sync.bfly.b32": 2, "shl.b32": 12, "shr.s32": 16, "shr.u32": 10, "sub.s32": 8}}
#: the installed vLLM's files the Definition and this script were read from, by sha256 at d9105ea80
SOURCES = {"models/deepseek_v4/common/ops/cache_utils.py": "cf62718e8aa2b9c8dcac1fedd478a27c8e47925c908f621c015bd2c46de458f8",
           "models/deepseek_v4/nvidia/flashmla.py": "28f636943e330702ce1d02ad1a6a242237bbce2503ee55f167274afa7d1ca0fe",
           "models/deepseek_v4/sparse_mla.py": "98b829059c5381e48a60545c019878b86c0fc2314664664eb685215ec2129251",
           "v1/attention/backends/mla/sparse_swa.py": "442673bbedfc44d415e26da12e7ea7427690cd4677972ea2c16065b9011d00d2"}
_error, _save, _load = FX._error, FX._save, FX._load
_chunks, _tally, _ref_rows, _flat_out, _version = CX._chunks, CX._tally, CX._ref_rows, CX._flat_out, CX._version
_launch_ok, _words_ok = CX._launch_ok, CX._words_ok
environment, stack = IX.environment, IX.stack


def statics(case: str) -> dict[str, int]:
    return dict(R.DSV4_DECODE_INDICES[case])


def _w(x) -> np.ndarray:
    return (np.asarray(x, dtype=np.int64) & M32).astype(np.uint32)


# ------------------------------------------------------------------------------------------------------------------ operands
def served_rows(case: str, rng) -> tuple[np.ndarray, np.ndarray]:
    """Per served request its top-k row (min(q, TOPK) distinct ids below its compressed count q = (position + 1) // 4, q - 1 among
    them, then -1) and its block row (its cdiv(q, BS) block ids, then junk words), as words."""
    st = statics(case)
    topk = np.full((len(POSITIONS[case]), st["TOPK"]), -1, dtype=np.int64)
    blocks = rng.integers(0, 1 << 32, (len(POSITIONS[case]), st["NB"]), dtype=np.uint64).astype(np.int64)
    for r, p in enumerate(POSITIONS[case]):
        q = (p + 1) // RATIO
        k = min(q, st["TOPK"])
        if k:
            ids = rng.choice(q, k, replace=False)
            if q - 1 not in ids:
                ids[rng.integers(0, k)] = q - 1
            topk[r, :k] = ids
        n = -(-q // st["BS"])
        blocks[r, :n] = rng.choice(NUM_BLOCKS, n, replace=False)
    return _w(topk), _w(blocks)


def edge_table(case: str, rng) -> np.ndarray:
    """The edge chunk's block rows: block numbers 2^25 + j (each product by BS wraps); INT32_MAX, INT32_MIN, -1, 0, 1, 2^26 - 1,
    2^25, 2^25 - 1 in turn; random words; ids below NUM_BLOCKS; every word -1; random words."""
    nb = statics(case)["NB"]
    special = np.resize(np.array([I32_MAX, I32_MIN, -1, 0, 1, (1 << 26) - 1, 1 << 25, (1 << 25) - 1]), nb)
    words = rng.integers(0, 1 << 32, (2, nb), dtype=np.uint64).astype(np.int64)
    return _w(np.stack([(1 << 25) + np.arange(nb), special, words[0], rng.integers(0, NUM_BLOCKS, nb), np.full(nb, -1), words[1]]))


def edge_row(kind: str, case: str, rng) -> np.ndarray:
    """One edge top-k row of ``kind``: `empty` (every index -1), `negative` (the negative kinds), `short` (a prefix of 1, 63, 64,
    65 or 1023 indices, then -1), `full`, `edges` (the block edges, the row's last slot NB * BS - 1 and the negatives around
    them), or `mix` (the twin's draws, a valid one past the row folded into it)."""
    st = statics(case)
    K, BS, span = st["TOPK"], st["BS"], st["NB"] * st["BS"]
    if kind == "empty":
        x = np.full(K, -1)
    elif kind == "negative":
        x = np.where(rng.random(K) < 0.5, rng.choice([-1, -2, -BS, -BS - 1, I32_MIN, I32_MIN + 1, -span], K),
                     rng.integers(I32_MIN, 0, K))
    elif kind == "short":
        k = int(rng.choice([1, 63, 64, 65, K - 1]))
        x = np.where(np.arange(K) < k, rng.integers(0, span, K), -1)
    elif kind == "full":
        x = rng.integers(0, span, K)
        x[rng.integers(0, K)] = span - 1
    elif kind == "edges":
        x = rng.choice([0, 1, BS - 1, BS, BS + 1, span - BS, span - 1, -1, -2, -BS, I32_MIN], K)
    else:
        x = T.s32(T.indices(rng, (1, K), BS, st["NB"]))[0]
        x = np.where(x >= span, x % span, x)
    return _w(x)


EDGE_KINDS = ("empty", "negative", "short", "full", "edges", "mix", "mix")


def requests(case: str, kind: str, seed: int) -> dict[str, Any]:
    """One decode step's operands: per token its top-k row (tokens, TOPK) and valid byte as words, its request `req`, and its
    position (-1 where the chunk holds none); the block table (requests, NB) as words."""
    rng = np.random.default_rng(seed)
    if kind == "served":
        topk, table = served_rows(case, rng)
        st = statics(case)
        pad = _w(np.where(rng.random((PADDING, st["TOPK"])) < 0.3, -1, rng.integers(0, st["NB"] * st["BS"], (PADDING, st["TOPK"]))))
        nreq = len(POSITIONS[case])
        return {"topk": np.concatenate([topk, pad]), "table": table,
                "req": np.concatenate([np.arange(nreq), np.zeros(PADDING)]).astype(np.int32),
                "valid": np.concatenate([np.ones(nreq), np.zeros(PADDING)]).astype(np.uint8),
                "pos": np.concatenate([POSITIONS[case], np.full(PADDING, -1)]).astype(np.int64)}
    table = edge_table(case, rng)
    req = np.repeat(np.arange(EDGE_REQUESTS), EDGE_PER_REQUEST).astype(np.int32)
    kinds = np.resize(np.array(EDGE_KINDS), req.size)
    topk = np.stack([edge_row(str(k), case, rng) for k in kinds])
    order = rng.permutation(req.size)
    valid = rng.permutation(np.resize(np.array(EDGE_BYTES + (1, 1)), req.size)).astype(np.uint8)
    return {"topk": topk[order], "table": table, "req": req[order], "valid": valid, "pos": np.full(req.size, -1, dtype=np.int64),
            "kinds": kinds[order]}


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
    """vLLM's DSV4 attention class, the row its ratio-4 decode calls, its backend's kernel block size, Triton, the sources."""
    import vllm
    from vllm.config import get_current_vllm_config
    from vllm.models.deepseek_v4 import sparse_mla as SM
    from vllm.models.deepseek_v4.common.ops import cache_utils as CU
    from vllm.models.deepseek_v4.nvidia import flashmla as F
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls

    cls = _select_dsv4_attn_cls(get_current_vllm_config())
    sizes = cls.backend_cls.get_supported_kernel_block_sizes()
    rec: dict[str, Any] = {"attn_cls": cls.__name__, "backend": cls.backend_cls.__name__,
                           "block_sizes": [b if isinstance(b, int) else repr(b) for b in sizes], "triton": _version("triton"),
                           "sources": sources(Path(vllm.__file__).parent)}
    rec["checks"] = {"attn_cls": cls is F.DeepseekV4FlashMLAAttention, "backend": cls.backend_cls is SM.DeepseekV4FlashMLABackend,
                     "row": F.compute_global_topk_indices_and_lens is CU.compute_global_topk_indices_and_lens,
                     "block_size": rec["block_sizes"] == [KERNEL_BLOCK] and KERNEL_BLOCK // RATIO == R.DSV4_DECODE_INDICES["c4"]["BS"],
                     "triton": rec["triton"] == TRITON, "sources": rec["sources"]["ok"]}
    rec["ok"] = all(rec["checks"].values())
    return rec


def _ref_every(quick: bool) -> int:
    return QUICK_REF_EVERY if quick else 1


def _dev(a: np.ndarray):
    """A host array on the device, 4-byte words as int32."""
    import torch

    a = np.ascontiguousarray(a)
    return torch.from_numpy(a.view(np.int32) if a.itemsize == 4 else a).to(DEVICE)


def launch(case: str, o: dict[str, Any]) -> dict[str, Any]:
    """`compute_global_topk_indices_and_lens` on one step's operands as the decode passes them, twice, then its kernel at the
    wrapper's arguments into sentinel buffers (rows (tokens + 2, TOPK), lengths (tokens + 8,)): each launch's outputs as words,
    their shapes, the direct launch's buffers, and whether every operand buffer is the bytes it was."""
    import torch
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _compute_global_topk_indices_and_lens_kernel,
        compute_global_topk_indices_and_lens,
    )

    st = statics(case)
    n, nreq = o["topk"].shape[0], o["table"].shape[0]
    pad = lambda a, rows, fill: np.concatenate([a, np.full((rows,) + a.shape[1:], fill, dtype=a.dtype)])  # noqa: E731
    bufs = {"topk": _dev(pad(o["topk"], TOPK_PAD, SENTINEL)), "req": _dev(pad(o["req"].astype(np.uint32), REQ_PAD, SENTINEL)),
            "table": _dev(pad(o["table"], TABLE_PAD, SENTINEL)), "valid": _dev(pad(o["valid"], VALID_PAD, 0xA5))}
    before = {k: v.clone() for k, v in bufs.items()}
    topk_t, table_t, valid_t = bufs["topk"][:n], bufs["table"][:nreq], bufs["valid"].view(torch.bool)[:n]
    args = (topk_t, bufs["req"], table_t, st["BS"], valid_t)
    runs = []
    for _ in range(2):
        rows, lens = compute_global_topk_indices_and_lens(*args)
        torch.cuda.synchronize()
        runs.append({"rows": CX._u32(rows), "lens": CX._u32(lens), "shapes": [list(rows.shape), list(lens.shape)]})
    ws = torch.full((n + WS_OFF + 1, st["TOPK"]), SENTINEL, dtype=torch.int32, device=DEVICE)
    wl = torch.full((n + LENS_OFF + 4,), SENTINEL, dtype=torch.int32, device=DEVICE)
    out, lens_out = ws[WS_OFF:WS_OFF + n], wl[LENS_OFF:LENS_OFF + n]
    _compute_global_topk_indices_and_lens_kernel[(n,)](out, out.stride(0), lens_out, topk_t, topk_t.stride(0), topk_t.shape[-1],
                                                       bufs["req"], table_t, table_t.stride(0), st["BS"], valid_t,
                                                       TRITON_BLOCK_SIZE=TRITON_BLOCK)
    torch.cuda.synchronize()
    return {"runs": runs, "direct": {"ws": CX._u32(ws), "wl": CX._u32(wl)},
            "operands": all(bool(torch.equal(before[k], bufs[k])) for k in bufs)}


def launches(case: str, seeds: tuple[int, int]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    ops = {kind: requests(case, kind, seed) for kind, seed in zip(("served", "edge"), seeds)}
    return ops, {kind: launch(case, ops[kind]) for kind in ops}


def launch_case(work: Path, case: str, quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    i = CASES_L.index(case)
    return observe(work, case, *launches(case, (201 + 2 * i, 202 + 2 * i)), quick)


def observe(work: Path, case: str, ops: dict[str, Any], rows: dict[str, dict[str, Any]],
            quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`launches`' words checked and saved under ``work``: the launch records and the comparison jobs."""
    st = statics(case)
    recs: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for kind, L in rows.items():
        o, name = ops[kind], case if kind == "served" else f"{case}-edge"
        n, (r1, r2), d = o["topk"].shape[0], L["runs"], L["direct"]
        others = np.ones(d["ws"].shape[0], dtype=bool)
        others[WS_OFF:WS_OFF + n] = False
        lens_others = np.ones(d["wl"].shape[0], dtype=bool)
        lens_others[LENS_OFF:LENS_OFF + n] = False
        recs.append({"phase": "row", "case": name, "tokens": n, "requests": int(o["table"].shape[0]),
                     "padding": int((o["valid"] == 0).sum()),
                     "checks": {"deterministic": all(np.array_equal(r1[k], r2[k]) for k in ("rows", "lens")),
                                "shapes": r1["shapes"] == [[n, st["TOPK"]], [n]] and r2["shapes"] == r1["shapes"],
                                "direct": bool(np.array_equal(d["ws"][WS_OFF:WS_OFF + n], r1["rows"])
                                               and np.array_equal(d["wl"][LENS_OFF:LENS_OFF + n], r1["lens"])),
                                "untouched": bool((d["ws"][others] == SENTINEL).all() and (d["wl"][lens_others] == SENTINEL).all()),
                                "operands": L["operands"] is True}})
        paths = _save(work / f"row-{name}", topk=o["topk"], table=o["table"], req=o["req"], valid=o["valid"], pos=o["pos"],
                      got_rows=r1["rows"], got_lens=r1["lens"])
        every = _ref_every(quick) if kind == "served" else 1
        jobs += [{"phase": "row", "case": name, "statics": st, "paths": paths, "rows": [a, b], "ref_every": every}
                 for a, b in _chunks(n, JOB_TOKENS)]
    return recs, jobs


# ------------------------------------------------------------------------------------------------------------------ comparisons
def compare_row(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Tokens [lo, hi): each row of slots and its length against the twin, the reference `Dsv4GlobalTopk_v1` on every
    ref_every-th."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    lo, hi = job["rows"]
    st = job["statics"]
    topk, valid = np.asarray(a["topk"][lo:hi]), np.asarray(a["valid"][lo:hi])
    blocks = np.asarray(a["table"])[np.asarray(a["req"][lo:hi])]
    rows, lens, declined = T.global_topk(topk, blocks, valid, **st)
    if declined.any():
        raise ValueError(f"tokens with a valid index past their block row: {(np.flatnonzero(declined) + lo)[:8].tolist()}")
    want = np.concatenate([rows, lens[:, None]], axis=1)
    got = np.concatenate([a["got_rows"][lo:hi], np.asarray(a["got_lens"][lo:hi])[:, None]], axis=1)
    _tally(rec, want, got, np.arange(lo, hi), "token")
    fn = bind(R.Dsv4GlobalTopk, **st)
    for i in _ref_rows(lo, hi, job["ref_every"]):
        y = evaluate(fn, [int(v) for v in topk[i - lo]], [int(v) for v in blocks[i - lo]], int(valid[i - lo]))
        rec["reference"] += 1
        rec["twin_mismatch"] += int(_flat_out(y) != want[i - lo].astype(np.int64).tolist())


def compare(job: dict[str, Any]) -> dict[str, Any]:
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    try:
        compare_row(job, rec)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "error": _error(e)}
    return {"phase": job["phase"], "case": job["case"], **rec}


# --------------------------------------------------------------------------------------------------------------- ptx and bind
def ptx_counts(text: str) -> dict[str, int]:
    return dict(sorted(Counter(PTX_COUNT.findall(text)).items()))


def ptx(triton_cache: Path, outdir: Path) -> dict[str, Any]:
    """The Triton cache's PTX of KERNELS: each file's target, sha256 and PTX_COUNT counts; ok when there is one file per NB at
    least and every file holds PTX_COUNTS["row"].  The PTX, and the SASS where Triton's bundled `cuobjdump` disassembles the
    cubin, are copied to `<out>/ptx/`."""
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
            e: dict[str, Any] = {"sha256": sha, "target": re.findall(r"\.target\s+(\S+)", text), "counts": ptx_counts(text)}
            cubin = Path(f).with_suffix(".cubin")
            if tool and cubin.exists():
                try:
                    sass = subprocess.run([tool, "-sass", str(cubin)], capture_output=True, text=True, check=True).stdout
                    (dest / f"{name}.sass").write_text(sass)
                except Exception as err:  # noqa: BLE001
                    e["sass_error"] = str(err)[-300:]
            entries.append(e)
        rec["kernels"][key] = entries
    row = rec["kernels"]["row"]
    rec["checks"] = {"target": all(e["target"] == ["sm_90a"] for es in rec["kernels"].values() for e in es),
                     "row": len(row) >= len(CASES_L) and all(e["counts"] == PTX_COUNTS["row"] for e in row)}
    rec["ok"] = all(rec["checks"].values())
    return rec


def bind_check() -> dict[str, Any]:
    """The Program binds at DeepSeek-V4-Pro's statics."""
    from verity.ml.circuits.defs import bind

    out = {}
    for case in CASES_L:
        st = statics(case)
        name = "Dsv4GlobalTopk_v1{" + ",".join(f"{k}={v}" for k, v in st.items()) + "}"
        try:
            out[name] = bool(bind(R.Dsv4GlobalTopk, **st).signature())
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
            print(f"[dsv4-idx-row] {name} ERROR {doc['errors'][name][:600]}", flush=True)
            return None

    doc["selection"] = step("selection", selection) or {"ok": False}
    print(f"[dsv4-idx-row] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
    doc["stack"] = step("stack", stack) or {}
    for case in CASES_L:
        r = step(case, lambda case=case: launch_case(work, case, quick))
        if r is not None:
            doc["launches"] += r[0]
            jobs += r[1]
            for x in r[0]:
                print(f"[dsv4-idx-row] {x['phase']:4s} {x['case']:13s} {json.dumps(x['checks'])}", flush=True)
        torch.cuda.empty_cache()
    return jobs


CASES = tuple(("row", n) for c in CASES_L for n in (c, f"{c}-edge"))


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's verdict: its launch checks, its words and its errors (its case's launch, its comparisons), so a NO-GO names
    what failed."""
    out: dict[str, dict[str, Any]] = {}
    for phase, case in CASES:
        launches_ = [x for x in doc["launches"] if x["phase"] == phase and x["case"] == case]
        comps = [r for r in doc.get("comparisons", []) if r["phase"] == phase and r["case"] == case]
        t = {k: sum(r[k] for r in comps) for k in TOTALS}
        errors = [e for e in doc["errors"] if e == case.removesuffix("-edge") or e == f"compare/{phase}/{case}"]
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
    print(f"[dsv4-idx-row] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-DECODE-INDEX-ROW GATE-FAIL", flush=True)
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
    print(f"[dsv4-idx-row] ptx {json.dumps(doc['ptx'])[:2000]}", flush=True)
    print(f"[dsv4-idx-row] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
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
        print(f"[dsv4-idx-row] words {r['phase']:4s} {r['case']:13s} words={r['words']} !={r['mismatch']} ref={r['reference']} "
              f"twin!={r['twin_mismatch']} first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[dsv4-idx-row] case {key:18s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} "
              f"!={t['mismatch']} ref={t['reference']} twin!={t['twin_mismatch']} errors={c['errors']}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"DSV4-DECODE-INDEX-ROW {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro's ratio-4 decode index row on this Hopper GPU against its Program")
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
