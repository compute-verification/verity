"""DeepSeek-V4-Pro's prefill index builders on Hopper against their Programs: the H200's go/no-go for binding `dsv4_indices`'
`Dsv4CombineTopkSwa_v1{TOPK,RATIO,WINDOW,WIDTH}` (vLLM d9105ea80's Triton `combine_topk_swa_indices`, the row of gathered-workspace
indices and its length FlashMLA's sparse prefill reads) and `dsv4_indexer.Dsv4ShortContextTopK_v1{K,128}` (the ratio-128 top-k row
`_build_c128a_topk_metadata_kernel` writes for a prefill token) at DeepSeek-V4-Pro's statics on one TP8 rank: a ratio-4 layer
(TOP_K 1024 in a row of 1152) and a ratio-128 layer at three `active_topk_width`s K (256 in a buffer 8192 wide, 896 in its own, and
8192), each row round_up(K + 128, 128) wide, the window 128.  Line numbers under `vllm/models/deepseek_v4/` unless named.

One GPU process writes `<out>/dsv4_indices_capture.json`, the property record `dsv4_indices_capture` (`properties.record`), and the
two Triton kernels' PTX (and SASS where Triton's bundled `cuobjdump` runs) under `<out>/ptx/`; it prints one line per launch and
per case, then `DSV4-INDICES GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first and nothing else runs when
it fails (exit 3).  A case that raises records its error and fails its cases without stopping the others.  The operands are the
integers a schedule holds (`requests`), so no model is loaded.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `_select_dsv4_attn_cls` picks `DeepseekV4FlashMLAAttention`, whose prefill (nvidia/flashmla.py:266-389) calls the
             `combine_topk_swa_indices` of common/ops/cache_utils.py, with no image width unless asked (`max_image_tokens` 0),
             the rows padded to 128 (`_SPARSE_PREFILL_TOPK_ALIGNMENT`) and ratio 128's K to 128 (`_C128A_TOPK_ALIGNMENT`); the
             installed Triton is the 3.7.1 whose PTX was read, and the installed SOURCES are the files read, since the launches
             call the two functions directly rather than through the attention layer.
  c128a      per ratio-128 case, `build_c128a_topk_metadata` as `_build_c128a_metadata` calls it with no decode token
             (sparse_mla.py:222-286, 295-358): the int64 positions of the case's served requests, SPECIAL_POSITIONS and draws,
             into a prefill buffer of the case's width filled with a sentinel.  Each row's first K words against `short_rows`
             (slot i is i when i < (position + 1) // 128, truncating, else -1) and the reference `Dsv4ShortContextTopK_v1{K,128}`
             on every row (every 16th served one under --quick); every other word of the buffers untouched, the returned rows
             the buffer's [:n, :K] view.
  combine    per case, `combine_topk_swa_indices` as the prefill calls it (flashmla.py:291-389): one chunk of requests, its
             `query_start_loc`, `seq_lens` and `gather_lens` slices of longer tensors (the chunk's base nonzero), `out` rows
             inside a sentinel-filled workspace, the top-k rows a [rows, :TOP_K] slice of a buffer the case's width.  `served`:
             requests from position 0 and late in a context (past the top-k's saturation at ratio 4), their gather lengths, N
             and M as `get_prefill_chunk_plan` sizes them (v1/attention/backends/mla/sparse_swa.py:214-285), the top-k the
             indexer's selection (ratio 4: ids below the token's compressed count, -1, and junk past it) or the c128a rows.
             `edge`: M = 2^30 + 1 (M * b wraps from b = 2), N = 2^31 - 200 (N + swa_start wraps), random 32-bit top-k words,
             and requests at negative positions, at both int32 ends (the Definition's domain: no position past INT32_MAX - 1 or
             below INT32_MIN + 127), past the top-k's saturation at every K, and with a gather longer than the sequence.  Each
             launch twice on fresh workspaces and once without `out`: every token's row and length against
             `dsv4_indices_twin.combine` (the kernel's stores) and the reference `Dsv4CombineTopkSwa_v1` on every token (every
             16th served one under --quick); the three launches the same words, the workspace's other rows untouched.
  ptx        the Triton cache's PTX of the two kernels, each for sm_90a: every combine specialization (at least one per TOP_K)
             holds PTX_COUNTS["combine"], the arithmetic the Definition was read from (one min, two max, two arithmetic shifts
             and one logical, the mul.lo of M * b, the 64-bit address mad.wide, no division), and the c128a builder its one
             int64 truncating division and min (PTX_COUNTS["c128a"]).
  bind       the Programs bind at the served statics.

GO needs every case's words compared with 0 mismatches, the references with 0 disagreements with the twins, every launch check,
the selection, the PTX and the bindings.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/dsv4_indices_capture_gpu.py --out DIR [--scratch DIR] [--quick]
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

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
from typing import Any, NamedTuple

import numpy as np

from tests.properties import dsv4_compressor_capture_gpu as CX
from tests.properties import dsv4_flashmla_capture_gpu as FX
from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity.ml.definitions.numerics import dsv4_indices_twin as T
from verity.ml.definitions.registry import dsv4_indexer as I
from verity.ml.definitions.registry import dsv4_indices as R
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/dsv4-indices-capture/v1"
RECORD = "dsv4_indices_capture.json"
#: DeepSeek-V4-Pro (config.json at b5968e91): `sliding_window` 128; rows and ratio 128's K padded to 128 (cache_utils.py:525,
#: sparse_mla.py:33)
WINDOW, ALIGN = R.DSV4_INDICES["c4"]["WINDOW"], 128
I32_MIN, I32_MAX, M32 = T.I32_MIN, T.I32_MAX, 0xFFFFFFFF
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1


class Case(NamedTuple):
    topk: int     # TOP_K: the top-k rows' width (`topk_indices.shape[-1]`, flashmla.py:304)
    ratio: int
    buffer: int   # the top-k buffer's row stride: the indexer's `index_topk`, or ratio 128's `c128a_max_compressed`
    max_len: int  # the context the case serves (`max_model_len`; at K 256, the step's longest sequence)


#: ratio 4 (`index_topk` 1024); ratio 128 at K 256 (a step whose sequences are below 257 * 128, `max_model_len` 2^20), at K 896
#: (`max_model_len` 114688, so `c128a_max_compressed` 896) and at K 8192 (`max_model_len` 2^20)
CASES_L = {"c4": Case(1024, 4, 1024, 1 << 20), "c128-256": Case(256, 128, 8192, 257 * 128 - 1),
           "c128-896": Case(896, 128, 896, 114688), "c128-8192": Case(8192, 128, 8192, 1 << 20)}
#: per case, the served chunk's requests (seq_len, query_len): one from position 0, one ending at the case's longest context,
#: a three-token one
SERVED = {"c4": ((1500, 1500), (4300, 600), (3, 3)), "c128-256": ((1000, 1000), (32895, 1000), (3, 3)),
          "c128-896": ((1000, 1000), (114688, 1000), (3, 3)), "c128-8192": ((1000, 1000), (1 << 20, 1000), (3, 3))}
#: the edge chunk's M and N
EDGE_M, EDGE_N = (1 << 30) + 1, (1 << 31) - 200
#: int64 positions the c128a builder takes beside the served ones (K * 128 relative ones added per case)
SPECIAL_POSITIONS = (I64_MIN, I64_MIN + 1, I32_MIN - 1, -129, -128, -127, -2, -1, 0, 1, 126, 127, 128, 254, 255, 256, I32_MAX,
                     I32_MAX + 1, (1 << 40) + 127, I64_MAX - 1, I64_MAX)
SENTINEL = 0x5A5A5A5A
#: offsets of the chunk's slices in their longer tensors (`query_start_loc[num_decodes + chunk_start:]` and the like), the base
#: of its `query_start_loc`, the rows of the top-k buffer before the chunk's, and the sentinel rows around the workspace's
QSL_OFF, SEQ_OFF, GATHER_OFF, QSL_BASE, TOPK_OFF, WS_OFF, LENS_OFF = 3, 5, 1, 37, 5, 1, 4
JOB_TOKENS = 64
QUICK_REF_EVERY = 16
TOTALS = ("words", "mismatch", "reference", "twin_mismatch")
TRITON = "3.7.1"
KERNELS = {"combine": "kernel", "c128a": "_build_c128a_topk_metadata_kernel"}
PTX_COUNT = re.compile(r"\b(min\.[su]\d+|max\.[su]\d+|shr\.[su]\d+|mul\.lo\.s32|mul\.hi\.[su]\d+|sub\.s32|mad\.wide\.s32|div\.[su]\d+|"
                       r"rem\.[su]\d+)(?=[\s;])")
#: Triton 3.7.1's sm_90a PTX: the combine's every op of PTX_COUNT at every TOP_K and alignment read (1024 / 4; 256, 896, 8192 /
#: 128); of the c128a builder the prefill row's division and min (its decode branch's divisions vary with the specialization)
PTX_COUNTS = {"combine": {"mad.wide.s32": 4, "max.s32": 2, "min.s32": 1, "mul.lo.s32": 5, "shr.s32": 2, "shr.u32": 1, "sub.s32": 6},
              "c128a": {"div.s64": 1, "min.s64": 1}}
#: the installed vLLM's files the Definitions and this script were read from, by sha256 at d9105ea80
SOURCES = {"models/deepseek_v4/common/ops/cache_utils.py": "cf62718e8aa2b9c8dcac1fedd478a27c8e47925c908f621c015bd2c46de458f8",
           "models/deepseek_v4/sparse_mla.py": "98b829059c5381e48a60545c019878b86c0fc2314664664eb685215ec2129251",
           "models/deepseek_v4/nvidia/flashmla.py": "28f636943e330702ce1d02ad1a6a242237bbce2503ee55f167274afa7d1ca0fe"}
_error, _save, _load = FX._error, FX._save, FX._load
_chunks, _tally, _ref_rows, _flat_out, _version = CX._chunks, CX._tally, CX._ref_rows, CX._flat_out, CX._version
_launch_ok, _words_ok = CX._launch_ok, CX._words_ok


def width(topk: int) -> int:
    """The row's width: round_up(TOP_K + WINDOW, 128) (flashmla.py:316, cache_utils.py:548-553)."""
    return -(-(topk + WINDOW) // ALIGN) * ALIGN


def statics(case: str) -> dict[str, int]:
    c = CASES_L[case]
    return {"TOPK": c.topk, "RATIO": c.ratio, "WINDOW": WINDOW, "WIDTH": width(c.topk)}


def _w(x) -> np.ndarray:
    return (np.asarray(x, dtype=np.int64) & M32).astype(np.uint32)


# ------------------------------------------------------------------------------------------------------------------ operands
def plan(reqs, ratio: int) -> tuple[np.ndarray, int, int]:
    """`get_prefill_chunk_plan`'s gather lengths query_len + clamp(prefix, 0, WINDOW - 1), N its largest seq_len // ratio and M
    N plus its largest gather, for one chunk of requests (seq_len, query_len)."""
    seq, ql = (np.array([r[i] for r in reqs], dtype=np.int64) for i in (0, 1))
    gather = ql + np.clip(seq - ql, 0, WINDOW - 1)
    n = int((seq // ratio).max())
    return gather, n, n + int(gather.max())


def edge_requests(c: Case) -> tuple[tuple[int, int, int], ...]:
    """The edge chunk's requests (seq_len, query_len, gather_len): negative positions; INT32_MAX - 3 .. INT32_MAX - 1; INT32_MIN +
    127 .. + 176 (pos - 127 not wrapping); across the top-k's saturation at K * ratio; a gather longer than the sequence."""
    s = c.topk * c.ratio
    return ((5, 40, 5), (I32_MAX, 3, 130), (I32_MIN + WINDOW - 1 + 50, 50, -7), (s + 200, 400, 527), (300, 300, 1000))


def requests(case: str, kind: str, seed: int) -> dict[str, Any]:
    """One chunk's operands: per request seq_len, query_len, gather_len; M, N; per token its position (start_pos + its index,
    cache_utils.py:684-697), request b, mb = mul.lo(M, b) and gather_start = seq_len - gather_len as words; and the top-k rows
    (tokens, TOP_K) as words, or None for ratio 128's served chunk, whose rows the c128a builder writes."""
    c = CASES_L[case]
    rng = np.random.default_rng(seed)
    if kind == "served":
        reqs = SERVED[case]
        gather, n, m = plan(reqs, c.ratio)
        seq, ql = (np.array([r[i] for r in reqs], dtype=np.int64) for i in (0, 1))
    else:
        er = edge_requests(c)
        seq, ql, gather = (np.array([r[i] for r in er], dtype=np.int64) for i in (0, 1, 2))
        n, m = EDGE_N, EDGE_M
    b = np.repeat(np.arange(seq.size), ql)
    pos = (seq - ql)[b] + (np.arange(b.size) - np.concatenate([[0], np.cumsum(ql)[:-1]])[b])
    o = {"seq": seq, "qlen": ql, "gather": gather, "M": m, "N": n, "pos": pos, "b": b, "mb": _w(m * b), "n": _w(np.full(b.size, n)),
         "gs": _w((seq - gather)[b])}
    if kind == "served" and c.ratio == 128:
        o["topk"] = None
    elif kind == "served":
        q = np.minimum((pos + 1) // c.ratio, c.topk)
        ids = rng.integers(-1, np.maximum(q, 1)[:, None], (pos.size, c.topk))
        o["topk"] = _w(np.where(np.arange(c.topk)[None, :] < q[:, None], ids, rng.integers(0, 1 << 32, (pos.size, c.topk))))
    else:
        words = rng.integers(0, 1 << 32, (pos.size, c.topk), dtype=np.uint64).astype(np.uint32)
        special = _w(rng.choice([0, 1, -1, I32_MIN, I32_MAX], (pos.size, c.topk)))
        o["topk"] = np.where(rng.random((pos.size, c.topk)) < 0.1, special, words)
    return o


def c128a_positions(case: str, served_pos: np.ndarray, seed: int) -> np.ndarray:
    """The c128a builder's int64 positions: the served chunk's, then SPECIAL_POSITIONS, those around K * 128, and 64 draws."""
    c = CASES_L[case]
    s = c.topk * c.ratio
    rng = np.random.default_rng(seed + 1)
    near = [s - 129, s - 128, s - 1, s, s + 1, 2 * s]
    return np.concatenate([served_pos, SPECIAL_POSITIONS, near, rng.integers(0, 2 * s, 64)]).astype(np.int64)


def short_rows(pos, K: int, ratio: int) -> np.ndarray:
    """`_build_c128a_topk_metadata_kernel`'s prefill rows (sparse_mla.py:417-427): slot i of a token is i when i < min((position + 1)
    // ratio, K) in int64 (div.s64 truncates; position + 1 wraps at INT64_MAX), else -1.  -> (tokens, K) words."""
    with np.errstate(over="ignore"):
        t = np.asarray(pos, dtype=np.int64) + np.int64(1)
    q = np.where(t >= 0, t // ratio, -1)
    i = np.arange(K, dtype=np.int64)[None, :]
    return _w(np.where(i < q[:, None], i, -1))


# ----------------------------------------------------------------------------------------------------------------- the GPU side
def environment(work: Path) -> dict[str, str]:
    """The Triton cache in scratch, so the PTX check reads only this process's compiles; set before vLLM is imported."""
    env = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TRITON_CACHE_DIR": str(work / "triton_cache")}
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


def selection() -> dict[str, Any]:
    """vLLM's DSV4 attention class, the combine its prefill calls, the paddings and the image width's default, Triton, the
    sources."""
    import inspect

    import vllm
    from vllm.config import get_current_vllm_config
    from vllm.models.deepseek_v4 import sparse_mla as SM
    from vllm.models.deepseek_v4.common.ops import cache_utils as CU
    from vllm.models.deepseek_v4.nvidia import flashmla as F
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls

    cls = _select_dsv4_attn_cls(get_current_vllm_config())
    rec: dict[str, Any] = {"attn_cls": cls.__name__, "image_width": inspect.signature(CU.combine_topk_swa_indices)
                           .parameters["max_image_tokens"].default, "row_alignment": CU._SPARSE_PREFILL_TOPK_ALIGNMENT,
                           "c128a_alignment": SM._C128A_TOPK_ALIGNMENT, "triton": _version("triton"),
                           "sources": sources(Path(vllm.__file__).parent)}
    rec["checks"] = {"attn_cls": cls is F.DeepseekV4FlashMLAAttention,
                     "combine": F.combine_topk_swa_indices is CU.combine_topk_swa_indices, "image_width": rec["image_width"] == 0,
                     "row_alignment": rec["row_alignment"] == ALIGN, "c128a_alignment": rec["c128a_alignment"] == ALIGN,
                     "triton": rec["triton"] == TRITON, "sources": rec["sources"]["ok"]}
    rec["ok"] = all(rec["checks"].values())
    return rec


def stack() -> dict[str, Any]:
    import torch
    import vllm

    return {"torch": torch.__version__, "cuda": torch.version.cuda, "vllm": getattr(vllm, "__version__", None),
            "triton": _version("triton")}


def _ref_every(quick: bool) -> int:
    return QUICK_REF_EVERY if quick else 1


def _i32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(_w(a)).view(np.int32)


def launch_c128a(case: str, pos: np.ndarray) -> dict[str, Any]:
    """`build_c128a_topk_metadata` on int64 positions ``pos``, none a decode token, twice on fresh sentinel buffers: the prefill
    buffer (tokens + 2, buffer) and the decode buffers as words, whether the returned rows were the buffer's [:n, :K] view, and
    that view as a device tensor (the served chunk's top-k)."""
    import torch
    from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata

    c = CASES_L[case]
    dev, n = "cuda", pos.size
    tp = torch.from_numpy(np.ascontiguousarray(pos)).to(dev)
    runs, view, local = [], True, None
    for _ in range(2):
        pf = torch.full((n + 2, c.buffer), SENTINEL, dtype=torch.int32, device=dev)
        gd = torch.full((1, c.buffer), SENTINEL, dtype=torch.int32, device=dev)
        dl = torch.full((1,), SENTINEL, dtype=torch.int32, device=dev)
        _, _, local = build_c128a_topk_metadata(tp, c.ratio, 0, torch.zeros(n, dtype=torch.int32, device=dev),
                                                torch.zeros((1, 1), dtype=torch.int32, device=dev), 256 // c.ratio,
                                                torch.zeros(n, dtype=torch.int64, device=dev), gd, dl, pf, max_compressed_tokens=c.topk)
        torch.cuda.synchronize()
        view &= bool(local.data_ptr() == pf.data_ptr() and tuple(local.shape) == (n, c.topk) and local.stride(0) == c.buffer)
        runs.append({"pf": CX._u32(pf), "gd": CX._u32(gd), "dl": CX._u32(dl)})
    return {"runs": runs, "view": view, "local": local}


def launch_combine(case: str, o: dict[str, Any], topk_t) -> dict[str, Any]:
    """`combine_topk_swa_indices` on one chunk, as the prefill calls it, twice into fresh sentinel workspaces (rows (tokens + 3,
    WIDTH), lengths (tokens + 8,)) and once without `out`; ``topk_t`` the device top-k rows, or None to put ``o["topk"]`` in a
    [rows, :TOP_K] slice of a buffer the case's width.  -> the workspaces and the allocated rows as words, and whether each
    launch returned its `out` views."""
    import torch
    from vllm.models.deepseek_v4.common.ops.cache_utils import combine_topk_swa_indices

    c, st = CASES_L[case], statics(case)
    dev, n, nreq, wd = "cuda", o["pos"].size, o["seq"].size, st["WIDTH"]
    qsl = np.concatenate([[0, 7, 19][:QSL_OFF], QSL_BASE + np.concatenate([[0], np.cumsum(o["qlen"])])])
    tq = torch.from_numpy(_i32(qsl)).to(dev)[QSL_OFF:QSL_OFF + nreq + 1]
    ts = torch.from_numpy(_i32(np.concatenate([np.full(SEQ_OFF, 11), o["seq"]]))).to(dev)[SEQ_OFF:]
    tg = torch.from_numpy(_i32(np.concatenate([np.full(GATHER_OFF, 13), o["gather"]]))).to(dev)[GATHER_OFF:]
    if topk_t is None:
        buf = torch.full((n + TOPK_OFF, c.buffer), SENTINEL, dtype=torch.int32, device=dev)
        topk_t = buf[TOPK_OFF:, :c.topk]
        topk_t.copy_(torch.from_numpy(np.ascontiguousarray(o["topk"]).view(np.int32)).to(dev))
    args = (topk_t, tq, ts, tg, WINDOW, c.ratio, c.topk, int(o["M"]), int(o["N"]))
    runs, views = [], True
    for _ in range(2):
        ws = torch.full((n + WS_OFF + 2, wd), SENTINEL, dtype=torch.int32, device=dev)
        wl = torch.full((n + LENS_OFF + 4,), SENTINEL, dtype=torch.int32, device=dev)
        out = (ws[WS_OFF:WS_OFF + n], wl[LENS_OFF:LENS_OFF + n])
        rows, lens = combine_topk_swa_indices(*args, out=out)
        torch.cuda.synchronize()
        views &= bool(rows.data_ptr() == out[0].data_ptr() and lens.data_ptr() == out[1].data_ptr())
        runs.append({"ws": CX._u32(ws), "wl": CX._u32(wl)})
    rows, lens = combine_topk_swa_indices(*args)
    torch.cuda.synchronize()
    return {"runs": runs, "views": views, "alloc": {"rows": CX._u32(rows), "lens": CX._u32(lens)},
            "topk": CX._u32(topk_t) if o["topk"] is None else o["topk"]}


def launches(case: str, seeds: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """One case's launches: its operands, the c128a launch (ratio 128) or None, and per chunk kind its combine launch."""
    ops = {kind: requests(case, kind, seed) for kind, seed in zip(("served", "edge"), seeds)}
    c128a, topk_t = None, None
    if CASES_L[case].ratio == 128:
        pos = c128a_positions(case, ops["served"]["pos"], seeds[0])
        c128a = launch_c128a(case, pos)
        c128a["pos"] = pos
        topk_t = c128a.pop("local")[:ops["served"]["pos"].size]
    combine = {"served": launch_combine(case, ops["served"], topk_t), "edge": launch_combine(case, ops["edge"], None)}
    return ops, c128a, combine


def launch_case(work: Path, case: str, quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = (101 + 2 * list(CASES_L).index(case), 102 + 2 * list(CASES_L).index(case))
    return observe(work, case, *launches(case, seeds), quick)


def observe(work: Path, case: str, ops: dict[str, Any], c128a: dict[str, Any] | None, combine: dict[str, dict[str, Any]],
            quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`launches`' words checked and saved under ``work``: the launch records and the comparison jobs."""
    c, st = CASES_L[case], statics(case)
    every = _ref_every(quick)
    recs: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    if c128a is not None:
        pos, (r1, r2) = c128a["pos"], c128a["runs"]
        n = pos.size
        rest = np.ones(r1["pf"].shape, dtype=bool)
        rest[:n, :c.topk] = False
        recs.append({"phase": "c128a", "case": case, "rows": n,
                     "checks": {"deterministic": all(np.array_equal(r1[k], r2[k]) for k in r1), "view": c128a["view"],
                                "untouched": bool((r1["pf"][rest] == SENTINEL).all() and (r1["gd"] == SENTINEL).all()
                                                  and (r1["dl"] == SENTINEL).all())}})
        paths = _save(work / f"c128a-{case}", pos=pos, got=r1["pf"][:n, :c.topk])
        served = ops["served"]["pos"].size
        jobs += [{"phase": "c128a", "case": case, "kind": "c128a", "K": c.topk, "ratio": c.ratio, "paths": paths, "rows": [a, b],
                  "ref_every": every if b <= served else 1} for a, b in _chunks(n, JOB_TOKENS)]
    for kind, L in combine.items():
        o, name = ops[kind], case if kind == "served" else f"{case}-edge"
        n, (r1, r2) = o["pos"].size, L["runs"]
        got_rows, got_lens = r1["ws"][WS_OFF:WS_OFF + n], r1["wl"][LENS_OFF:LENS_OFF + n]
        others = np.ones(r1["ws"].shape[0], dtype=bool)
        others[WS_OFF:WS_OFF + n] = False
        lens_others = np.ones(r1["wl"].shape[0], dtype=bool)
        lens_others[LENS_OFF:LENS_OFF + n] = False
        recs.append({"phase": "combine", "case": name, "tokens": n, "M": int(o["M"]), "N": int(o["N"]),
                     "positions": [int(o["pos"].min()), int(o["pos"].max())],
                     "checks": {"deterministic": all(np.array_equal(r1[k], r2[k]) for k in r1), "views": L["views"],
                                "alloc_is_out": bool(L["alloc"]["rows"].shape == (n, st["WIDTH"])
                                                     and np.array_equal(L["alloc"]["rows"], got_rows)
                                                     and np.array_equal(L["alloc"]["lens"], got_lens)),
                                "untouched": bool((r1["ws"][others] == SENTINEL).all() and (r1["wl"][lens_others] == SENTINEL).all())}})
        paths = _save(work / f"combine-{name}", topk=L["topk"], pos=_w(o["pos"]), mb=o["mb"], n=o["n"], gs=o["gs"], got_rows=got_rows,
                      got_lens=got_lens)
        jobs += [{"phase": "combine", "case": name, "kind": "combine", "statics": st, "paths": paths, "rows": [a, b],
                  "ref_every": every if kind == "served" else 1} for a, b in _chunks(n, JOB_TOKENS)]
    return recs, jobs


# ------------------------------------------------------------------------------------------------------------------ comparisons
def compare_combine(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Tokens [lo, hi): each row and length against the twin, the reference `Dsv4CombineTopkSwa_v1` on every ref_every-th."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    lo, hi = job["rows"]
    st = job["statics"]
    args = [np.asarray(a[k][lo:hi]) for k in ("topk", "pos", "mb", "n", "gs")]
    rows, lens, declined = T.combine(*args, **st)
    if declined.any():
        raise ValueError(f"positions outside the Definition's domain: {T.s32(args[1][declined])[:8].tolist()}")
    want = np.concatenate([rows, lens[:, None]], axis=1)
    got = np.concatenate([a["got_rows"][lo:hi], np.asarray(a["got_lens"][lo:hi])[:, None]], axis=1)
    _tally(rec, want, got, T.s32(args[1]), "position")
    fn = bind(R.Dsv4CombineTopkSwa, **st)
    for i in _ref_rows(lo, hi, job["ref_every"]):
        y = evaluate(fn, [int(v) for v in a["topk"][i]], *(int(a[k][i]) for k in ("pos", "mb", "n", "gs")))
        rec["reference"] += 1
        rec["twin_mismatch"] += int(_flat_out(y) != want[i - lo].astype(np.int64).tolist())


def compare_c128a(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Rows [lo, hi): each against `short_rows`, the reference `Dsv4ShortContextTopK_v1{K,ratio}` on every ref_every-th."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    lo, hi = job["rows"]
    pos = np.asarray(a["pos"][lo:hi])
    want = short_rows(pos, job["K"], job["ratio"])
    _tally(rec, want, a["got"][lo:hi], pos, "position")
    fn = bind(I.Dsv4ShortContextTopK, TOPK=job["K"], RATIO=job["ratio"])
    for i in _ref_rows(lo, hi, job["ref_every"]):
        y = evaluate(fn, int(a["pos"][i]) & ((1 << 64) - 1))
        rec["reference"] += 1
        rec["twin_mismatch"] += int(_flat_out(y) != want[i - lo].astype(np.int64).tolist())


COMPARE = {"combine": compare_combine, "c128a": compare_c128a}


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


def ptx(triton_cache: Path, outdir: Path) -> dict[str, Any]:
    """The Triton cache's PTX of KERNELS: each file's target, sha256 and PTX_COUNT counts; ok when every combine file holds
    PTX_COUNTS["combine"] and there is one per TOP_K at least, and every c128a file holds PTX_COUNTS["c128a"]'s ops at their
    counts.  The PTX, and the SASS where Triton's bundled `cuobjdump` disassembles the cubin, are copied to `<out>/ptx/`."""
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
    combine, c128a = rec["kernels"]["combine"], rec["kernels"]["c128a"]
    want = PTX_COUNTS["c128a"]
    rec["checks"] = {"target": all(e["target"] == ["sm_90a"] for es in rec["kernels"].values() for e in es),
                     "combine": len(combine) >= len({c.topk for c in CASES_L.values()})
                     and all(e["counts"] == PTX_COUNTS["combine"] for e in combine),
                     "c128a": bool(c128a) and all({k: e["counts"].get(k, 0) for k in want} == want for e in c128a)}
    rec["ok"] = all(rec["checks"].values())
    return rec


def bind_check() -> dict[str, Any]:
    """The Programs bind at DeepSeek-V4-Pro's statics."""
    from verity.ml.circuits.defs import bind

    targets = {}
    for case in CASES_L:
        st = statics(case)
        targets["Dsv4CombineTopkSwa_v1{" + ",".join(f"{k}={v}" for k, v in st.items()) + "}"] = \
            lambda st=st: bind(R.Dsv4CombineTopkSwa, **st)
        if st["RATIO"] == 128:
            targets[f"Dsv4ShortContextTopK_v1{{TOPK={st['TOPK']},RATIO=128}}"] = \
                lambda k=st["TOPK"]: bind(I.Dsv4ShortContextTopK, TOPK=k, RATIO=128)
    out = {}
    for name, make in targets.items():
        try:
            out[name] = bool(make().signature())
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
            print(f"[dsv4-idx-rows] {name} ERROR {doc['errors'][name][:600]}", flush=True)
            return None

    doc["selection"] = step("selection", selection) or {"ok": False}
    print(f"[dsv4-idx-rows] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
    doc["stack"] = step("stack", stack) or {}
    for case in CASES_L:
        r = step(case, lambda case=case: launch_case(work, case, quick))
        if r is not None:
            doc["launches"] += r[0]
            jobs += r[1]
            for x in r[0]:
                print(f"[dsv4-idx-rows] {x['phase']:8s} {x['case']:15s} {json.dumps(x['checks'])}", flush=True)
        torch.cuda.empty_cache()
    return jobs


CASES = tuple([("c128a", c) for c in CASES_L if CASES_L[c].ratio == 128]
              + [("combine", n) for c in CASES_L for n in (c, f"{c}-edge")])


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's verdict: its launch checks, its words and its errors (its layer case's launch, its comparisons), so a NO-GO
    names what failed."""
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
    print(f"[dsv4-idx-rows] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-INDICES GATE-FAIL", flush=True)
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
    print(f"[dsv4-idx-rows] ptx {json.dumps(doc['ptx'])[:2000]}", flush=True)
    print(f"[dsv4-idx-rows] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
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
        print(f"[dsv4-idx-rows] words {r['phase']:8s} {r['case']:15s} words={r['words']} !={r['mismatch']} ref={r['reference']} "
              f"twin!={r['twin_mismatch']} first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[dsv4-idx-rows] case {key:24s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} "
              f"!={t['mismatch']} ref={t['reference']} twin!={t['twin_mismatch']} errors={c['errors']}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"DSV4-INDICES {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro's prefill index builders on this Hopper GPU against their Programs")
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
