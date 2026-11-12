"""GLM-5.3's sparse read of the FP8 cache on Hopper, captured: FlashMLA 6bc49418's `sparse_decode_fwd` at its V32 instance
(`flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<V32, 64>>` and its combine) as vLLM d9105ea80's
`FlashMLASparseImpl._forward_fp8_kv_mixed_batch` (flashmla_sparse.py:826) calls it on one TP8 rank: every token of a step, decode
and prefill alike, as one batch of one request with s_q = T query rows, 8 real heads padded to 64 with zeros, a 576-wide key
whose first 512 words are the value, 2048 indices a token into an `fp8_ds_mla` cache of 656-byte rows (512 E4M3 bytes, four f32
scales, 64 bf16 words), softmax_scale 256^-0.5, no sink and no lengths.

Three stages, each a process of its own, so a GPU is held for the launches alone and the comparison can be run again from the
saved words without one:

  sass     no GPU.  `cuobjdump -sass` of each sm_90a image of vLLM's `_flashmla_C` library: under `<out>/sass/`, the listing of
           every function of the sparse FP8 decode, the combine and the planner, gzipped, the images that hold them (`elf/`), and
           `index.json`: the library's sha256 and architectures, every sm_90a function it carries, and per kept function its
           image, sha256, instruction count and floating-point opcodes.  `instance` names the functions of the V32, 64-head kernel.
  launch   one GPU process.  The device gate (`fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs) runs first and nothing
           else runs when it fails (exit 3).  The cache: every request's rows through its block table, written by vLLM's own
           write (`deepseek_v32.common.kernels.fused_norm_rope` under `fp8_ds_mla`, on random inputs: zero rows, rows far below
           the norm's epsilon and ordinary ones), a few requests' rows and the planted rows by numpy, and block 0 poisoned (the
           row an index of -1 addresses, NaN in every field).  Per case, `_forward_fp8_kv_mixed_batch` itself on a stand-in for
           the backend that holds only what the method reads, with the arguments it hands `flash_mla_with_kvcache` recorded;
           then that call again on fresh planner metadata into two outputs filled with two different sentinels (every word
           written, deterministic, the planner rerun) and once with a sink of -inf on every head (no sink is a sink of -inf).
           The dense prefill's read-back (`cp_gather_and_upconvert_fp8_kv_cache`) of the probed blocks, twice into two sentinels.
           Everything the kernel read and wrote is saved under `<out>/capture/`: the cache, and per case the queries, the
           indices before and after vLLM's conversion, all 64 heads' output words, the lse and the planner's metadata.
  compare  no GPU.  The planner's metadata against `flashmla_twin.sched_meta` at num_sm_parts = max(132 // T, 1); every output
           word of every token against `flashmla_dsv32_twin.decode` at the step's SPLITS: the 8 real heads and one pad head
           against the twin, the other 55 pad heads against that pad head's words; the probes; and, for ``--definition N``
           tokens a case (every token of `edge`), the 8 real heads against `FlashMlaV32Decode_v1` itself, bound at GLM-5.3's
           statics and the step's SPLITS and run through the reference evaluator (about two minutes a token on one core).
           Writes `<out>/glm53_flashmla_fp8_capture.json` and prints, per case and in total, the words compared and the words
           that differ; exit 1 unless every check held and no word differs.

No stage catches a failure: a launch or a comparison that does not finish is its own traceback and non-zero exit, and it
leaves no `capture/launch.json` (launch) and no record (compare).  A missing listing alone is recorded, as `sass.ok` false.

The cases.  `dT`: T decode tokens, one query row a request (a third of them requests of 2048 rows or more), each with the 2048
rows its request's indexer would name (all of them, in a random order, then -1, when the request has fewer), at every T where
the planner's cut of the 32 blocks changes
(`DECODE_T`: 32 splits of one block at T <= 3 down to one split, and no combine, from T = 67).  `prefill`: a step of 320 tokens,
a long request's last positions, a short request's every position (1 to 100 visible rows) and decode tokens beside them.
`prefill45`: 45 positions of one request (two splits).  `edge` (16 tokens, 7 splits) and `edgeflat` (the same 16 and 64 more, one
split): the planted rows (`PLANTED`), indices of -1 scattered, in whole blocks, in a whole split and everywhere, one valid
index, one row repeated, queries at +-bf16 max, zero and subnormal.  `probe` (320 tokens) and `probesplit` (32): one valid
index a token, so a pad head's output (its query is zero, the row's weight exactly 1) is the row as the kernel staged it;
those words are compared with the read-back's of the same row, which is the question the probes answer: do the two reads of
one cache row agree.

    cd adapters/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_flashmla_fp8_capture_gpu.py sass --out DIR
    $PY tests/properties/glm53_flashmla_fp8_capture_gpu.py launch --out DIR [--quick]
    $PY tests/properties/glm53_flashmla_fp8_capture_gpu.py compare --out DIR [--workers N] [--definition N]
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import argparse
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
import os
import re
import shutil
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import dsv4_flashmla_capture_gpu as D
from tests.properties import dsv4_marlin_capture_gpu as M
from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity.definitions.kernels import tc as CK
from verity.definitions.numerics import derived_rows as DRW
from verity.definitions.numerics import flashmla_twin as T
from verity.definitions.numerics import flashmla_dsv32_twin as V
from verity_vllm.properties.record import digest_of

SCHEMA = "verity-vllm/glm53-flashmla-fp8-capture/v1"
RECORD = "glm53_flashmla_fp8_capture.json"
#: GLM-5.3 (config.json) at TP8: 64 / 8 = 8 local heads padded to h_q = 64 (flashmla_sparse.py:587-590), kv_lora_rank 512 and
#: qk_rope_head_dim 64 (the key 576 wide, the value its first 512), index_topk 2048, softmax scale qk_head_dim^-0.5 = 1/16
NH, H_Q, DN, DR, QT, BN, TOPK, SCALE = 8, 64, 512, 64, 128, 64, 2048, 0.0625
DK, DV = DN + DR, DN
#: the `fp8_ds_mla` row: 512 E4M3 bytes | four f32 scales, one per 128 elements | 64 bf16 words; 64 rows a block
ROW_BYTES, PBS, Q_LORA, RMS_EPS = 656, 64, 2048, 1e-5
SEED = 0x5EED
#: the planner cuts a token's 32 blocks by T alone: each of these T is the first or the last of a cut
DECODE_T = (1, 3, 4, 6, 7, 10, 11, 13, 14, 16, 17, 18, 19, 22, 23, 26, 27, 33, 34, 44, 45, 66, 67, 133)
QUICK_T = (1, 7, 34, 67)
FIXED_LENS = (4096, 2500, 100, 1, 63, 64, 65, 2047, 2048, 2049, 127, 129, 1000, 3000)
#: requests (the planted one apart) and how many of them are long: lengths uniform over [2048, 4096], the rest log-uniform from 1
REQS, QUICK_REQS = (133, 40), (67, 20)
PLANT_BLOCKS = 2
PLANT_ROWS = PLANT_BLOCKS * PBS
JOB_TOKENS = 4
#: the cases whose every token is held to the Definition's own evaluator, whatever ``--definition`` (above 0) says: the planted rows
DEFINITION_WHOLE = ("edge",)
#: the architecture an H200 loads from the library's fat binary
SASS_ARCH = "sm_90a"
SASS_KEEP = re.compile(r"fp8_sparse|sparse_fp8|mla_combine_kernel|get_mla_metadata_kernel")
#: the V32, 64-head instance's template arguments as the Itanium ABI mangles them (`ModelType::V32` is 0; `ModelType` is
#: declared outside any namespace, `L9ModelType0E`, and the nested form is taken too)
SASS_INSTANCE = re.compile(r"flash_fwd_splitkv_mla_fp8_sparse_kernel.*9ModelTypeE?0ELi64E")
F32 = np.float32
U32 = np.uint32


def _f32w(x) -> np.ndarray:
    return np.asarray(x, dtype=F32).view(U32)


# ------------------------------------------------------------------------------------------------------------------- the rows
def pack_rows(nope, scale, rope) -> np.ndarray:
    """(n, 512) e4m3 bytes, (n, 4) f32 words and (n, 64) bf16 words as (n, 656) bytes."""
    n = np.asarray(nope).shape[0]
    rows = np.zeros((n, ROW_BYTES), dtype=np.uint8)
    rows[:, :DN] = nope
    rows[:, DN:DN + 16] = np.ascontiguousarray(np.asarray(scale, dtype="<u4")).view(np.uint8).reshape(n, 16)
    rows[:, DN + 16:] = np.ascontiguousarray(np.asarray(rope, dtype="<u2")).view(np.uint8).reshape(n, 2 * DR)
    return rows


def unpack_rows(rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.ascontiguousarray(rows, dtype=np.uint8)
    n = rows.shape[0]
    return (rows[:, :DN].copy(), np.ascontiguousarray(rows[:, DN:DN + 16]).view("<u4").reshape(n, DN // QT).astype(U32),
            np.ascontiguousarray(rows[:, DN + 16:]).view("<u2").reshape(n, DR).astype(np.uint16))


def rows_at(cache, gidx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rows global indices name in a cache (blocks, 64, 656); an index of -1 gives row 0, which no reader may use."""
    g = np.asarray(gidx, dtype=np.int64)
    g = np.where(g == -1, 0, g)
    return unpack_rows(np.asarray(cache).reshape(-1, ROW_BYTES)[g])


def numpy_rows(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n finite rows: every E4M3 code but the two NaNs, scales log-uniform over [2^-12, 2^4) with full mantissas, normal RoPE."""
    nope = rng.integers(0, 256, (n, DN), dtype=np.uint8)
    nope[(nope & 0x7F) == 0x7F] = 0x3E
    scale = _f32w(np.exp2(rng.uniform(-12, 4, (n, DN // QT))))
    return nope, scale, D._bf16(rng.standard_normal((n, DR)).astype(F32))


#: the planted rows, by row of the planted request: (name, {tile: f32 scale word} or None, what else)
PLANTED_SCALES: dict[int, tuple[str, tuple[int, int, int, int]]] = {
    0: ("hand: scale 1 + 2^-8 + 2^-20 in every tile; element 0 is 1.75", (0x3F808008,) * 4),
    1: ("scales on a tie to bf16 and beside it", (0x3F808000, 0x3F818000, 0x3F808001, 0x3F807FFF)),
    2: ("scales that round to bf16 +inf, to bf16 max and on the tie below", (0x7F7FFFFF, 0x7F7F8000, 0x7F7F7FFF, 0x7F7E8000)),
    3: ("the write's floor and subnormal scales", (0x00800000, 0x00000001, 0x00008000, 0x00018000)),
    4: ("zero and negative scales", (0x00000000, 0x80000000, 0xBF800000, 0xC2F6E979)),
    5: ("infinite and NaN scales", (0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001)),
    6: ("NaN scales of other patterns", (0xFFC00000, 0x7FFFFFFF, 0x3F800000, 0x3F800000)),
    8: ("E4M3 +-448 under scales that overflow bf16", (0x7B800000, 0x7F000000, 0x3F800000, 0x40490FDB)),
    9: ("E4M3 subnormals under the smallest scales", (0x00800000, 0x00800001, 0x01000000, 0x3F800000)),
    10: ("bytes +-0 under 1, -1, +inf and NaN", (0x3F800000, 0xBF800000, 0x7F800000, 0x7FC00000)),
}
PLANTED = {**{r: n for r, (n, _) in PLANTED_SCALES.items()},
           7: "E4M3 NaN bytes 0x7F and 0xFF in a row of finite scales",
           11: "RoPE words +-inf, NaN, subnormal, -0 and bf16 max",
           12: "RoPE words all subnormal",
           13: "rows 13 to 31: scales whose low 16 bits are 0x7FFF, 0x8000 or 0x8001",
           32: "rows 32 to 127: scales log-uniform over [2^-40, 2^20)"}
HAND_ROW, HAND_BYTE, HAND_SCALE = 0, 0x3E, 0x3F808008


def planted_rows(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nope, scale, rope = numpy_rows(rng, PLANT_ROWS)
    for r, (_, words) in PLANTED_SCALES.items():
        scale[r] = np.array(words, dtype=U32)
    nope[HAND_ROW, 0] = HAND_BYTE
    nope[7, 17], nope[7, 300] = 0x7F, 0xFF
    nope[8] = np.where(rng.random(DN) < 0.5, 0x7E, 0xFE)
    nope[9] = rng.integers(1, 8, DN, dtype=np.uint8) | (rng.integers(0, 2, DN, dtype=np.uint8) << 7)
    nope[10] = rng.integers(0, 2, DN, dtype=np.uint8) << 7
    rope[11, :12] = np.array([0x7F80, 0xFF80, 0x7FC0, 0x7FFF, 0xFFC1, 0x0001, 0x8001, 0x007F, 0x8000, 0x7F7F, 0xFF7F, 0x0000],
                             dtype=np.uint16)
    rope[12] = rng.integers(1, 0x80, DR, dtype=np.uint16) | (rng.integers(0, 2, DR, dtype=np.uint16) << 15)
    low = np.array([0x7FFF, 0x8000, 0x8001], dtype=U32)
    scale[13:32] = (_f32w(np.exp2(rng.uniform(-10, 3, (19, DN // QT)))) & U32(0xFFFF0000)) | low[rng.integers(0, 3, (19, DN // QT))]
    scale[32:] = _f32w(np.exp2(rng.uniform(-40, 20, (PLANT_ROWS - 32, DN // QT))))
    return nope, scale, rope


# ----------------------------------------------------------------------------------------------------------------- the layout
def layout(quick: bool) -> dict[str, Any]:
    """The requests: `lens` rows each, `table` its block table over a shuffle of blocks 1.. (block 0 is the poison block), and
    `numpy` the requests whose rows numpy writes.  The last request is the planted one: `PLANT_ROWS` planted rows, then 4096."""
    rng = np.random.default_rng(SEED)
    n, n_long = QUICK_REQS if quick else REQS
    lens = list(FIXED_LENS) + [int(v) for v in rng.integers(2048, 4097, n_long)]
    lens += [int(v) for v in np.clip(np.rint(np.exp(rng.uniform(0, np.log(4096), n - len(lens)))), 1, 4096)]
    lens.append(PLANT_ROWS + 4096)
    nb = [-(-v // PBS) for v in lens]
    blocks = 1 + sum(nb)
    perm = 1 + rng.permutation(blocks - 1)
    table = np.zeros((len(lens), max(nb)), dtype=np.int32)
    off = 0
    for r, k in enumerate(nb):
        table[r, :k] = perm[off:off + k]
        off += k
    by_numpy = np.array([r % 16 == 15 for r in range(len(lens))], dtype=bool)
    by_numpy[-1] = True
    return {"lens": np.array(lens, dtype=np.int32), "table": table, "blocks": blocks, "numpy": by_numpy}


def slots(lay: dict[str, Any], r: int, pos) -> np.ndarray:
    """The cache rows of request r's positions (-1 stays -1): what vLLM's conversion computes."""
    p = np.asarray(pos, dtype=np.int64)
    ok = p >= 0
    q = np.where(ok, p, 0)
    return np.where(ok, lay["table"][r, q // PBS].astype(np.int64) * PBS + q % PBS, -1).astype(np.int32)


def host_cache(lay: dict[str, Any]) -> np.ndarray:
    """The cache before vLLM's write: numpy rows in every request's rows, the planted rows, and block 0 NaN in every field."""
    rng = np.random.default_rng(SEED + 1)
    cache = np.zeros((lay["blocks"] * PBS, ROW_BYTES), dtype=np.uint8)
    cache[:PBS] = pack_rows(np.full((PBS, DN), 0x7F, dtype=np.uint8), np.full((PBS, DN // QT), 0x7FC00000, dtype=U32),
                            np.full((PBS, DR), 0x7FC0, dtype=np.uint16))
    for r, n in enumerate(lay["lens"]):
        cache[slots(lay, r, np.arange(int(n)))] = pack_rows(*numpy_rows(rng, int(n)))
    cache[slots(lay, len(lay["lens"]) - 1, np.arange(PLANT_ROWS))] = pack_rows(*planted_rows(rng))
    return cache.reshape(lay["blocks"], PBS, ROW_BYTES)


# ------------------------------------------------------------------------------------------------------------------ the cases
def case_names(quick: bool) -> list[str]:
    return [f"d{t}" for t in (QUICK_T if quick else DECODE_T)] + (["prefill45", "edge", "probesplit"] if quick else
                                                                   ["prefill", "prefill45", "edge", "edgeflat", "probe", "probesplit"])


def _queries(rng: np.random.Generator, t: int) -> np.ndarray:
    gain = np.array([0.25, 1.0, 4.0, 16.0] * (NH // 4), dtype=F32)[None, :, None]
    return D._bf16(rng.standard_normal((t, NH, DK)).astype(F32) * gain)


def _row(rng: np.random.Generator, visible: int) -> np.ndarray:
    """The 2048 positions an indexer names among `visible` rows: every one in a random order then -1, or 2048 of them."""
    idx = np.full(TOPK, -1, dtype=np.int32)
    k = min(visible, TOPK)
    idx[:k] = rng.permutation(visible)[:k]
    return idx


def probe_rows(lay: dict[str, Any]) -> list[tuple[int, int]]:
    """(request, position) of each probed row: the planted request's planted blocks, block 0 of the two longest requests vLLM
    writes and of one numpy writes."""
    plant = len(lay["lens"]) - 1
    written = [r for r in np.argsort(-lay["lens"][:-1], kind="stable") if not lay["numpy"][r]][:2]
    by_numpy = [r for r in range(plant) if lay["numpy"][r] and lay["lens"][r] >= PBS][:1]
    return [(plant, p) for p in range(PLANT_ROWS)] + [(int(r), p) for r in [*written, *by_numpy] for p in range(PBS)]


def case_operands(case: str, lay: dict[str, Any]) -> dict[str, np.ndarray]:
    """A case's operands as numpy words: q (T, 8, 576) bf16, idx (T, 2048) positions in the token's request, req (T,)."""
    rng = np.random.default_rng([SEED, int.from_bytes(hashlib.sha256(case.encode()).digest()[:4], "little")])
    lens, plant = lay["lens"], len(lay["lens"]) - 1
    if case.startswith("d") and case[1:].isdigit():
        t = int(case[1:])
        full = lens[:plant] >= TOPK
        k = max(-(-t // 3), t - int((~full).sum()))
        req = rng.permutation(np.concatenate([rng.permutation(np.flatnonzero(full))[:k],
                                              rng.permutation(np.flatnonzero(~full))[:t - k]])).astype(np.int32)
        idx = np.stack([_row(rng, int(lens[r])) for r in req])
    elif case == "prefill":
        long_, short = 0, 2
        toks = [(long_, p) for p in range(int(lens[long_]) - 192, int(lens[long_]))] + [(short, p) for p in range(int(lens[short]))]
        toks += [(int(r), int(lens[r]) - 1) for r in rng.permutation(np.arange(3, plant))[:320 - len(toks)]]
        req = np.array([r for r, _ in toks], dtype=np.int32)
        idx = np.stack([_row(rng, p + 1) for _, p in toks])
    elif case == "prefill45":
        req = np.full(45, 6, dtype=np.int32)
        idx = np.stack([_row(rng, p + 1) for p in range(int(lens[6]) - 45, int(lens[6]))])
    elif case in ("edge", "edgeflat"):
        return edge(rng, lay, 80 if case == "edgeflat" else 16)
    elif case in ("probe", "probesplit"):
        rows = probe_rows(lay)
        if case == "probesplit":
            rows = rows[:16] + rows[PLANT_ROWS:PLANT_ROWS + 16]
        req = np.array([r for r, _ in rows], dtype=np.int32)
        idx = np.full((len(rows), TOPK), -1, dtype=np.int32)
        idx[np.arange(len(rows)), rng.integers(0, TOPK, len(rows))] = [p for _, p in rows]
    else:
        raise ValueError(case)
    return {"q": _queries(rng, req.size), "idx": idx, "req": req}


def edge(rng: np.random.Generator, lay: dict[str, Any], t: int) -> dict[str, np.ndarray]:
    """`edge`'s tokens, all of the planted request (positions 0..127 the planted rows, the rest finite): 0 the hand row among
    2047 finite rows; 1 the tie scales; 2 the E4M3 NaN row; 3 the rows whose scales overflow, vanish or are negative; 4 the
    infinite and NaN scales; 5 -1 on a third of its indices; 6 all -1; 7 one valid index; 8 one row 2048 times; 9 heads 0 and 1
    at +-bf16 max; 10 real heads +0; 11 subnormal queries; 12 its first block and another whole block -1; 13 its first five
    blocks -1 (a whole split at 16 tokens); 14 the RoPE rows; 15 every planted row.  Tokens past 15 are ordinary."""
    plant = len(lay["lens"]) - 1
    n = int(lay["lens"][plant])
    q = _queries(rng, t)
    idx = np.stack([PLANT_ROWS + _row(rng, n - PLANT_ROWS) for _ in range(t)]).astype(np.int32)
    for tok, rows in ((0, [0]), (1, [1, *range(13, 32)]), (2, [7]), (3, [2, 3, 4, 8, 9]), (4, [5, 6, 10]), (14, [11, 12]),
                      (15, list(range(PLANT_ROWS)))):
        idx[tok, rng.permutation(TOPK)[:len(rows)]] = rows
    idx[5, rng.random(TOPK) < 1 / 3] = -1
    idx[6] = -1
    keep = idx[7, 700]
    idx[7] = -1
    idx[7, 700] = keep
    idx[8] = idx[8, 0]
    q[9, 0], q[9, 1] = 0x7F7F, 0xFF7F
    q[10] = 0
    q[11] = rng.integers(1, 0x80, (NH, DK), dtype=np.uint16) | (rng.integers(0, 2, (NH, DK), dtype=np.uint16) << 15)
    idx[12, :BN] = -1
    idx[12, 9 * BN:10 * BN] = -1
    idx[13, :5 * BN] = -1
    return {"q": q, "idx": idx, "req": np.full(t, plant, dtype=np.int32)}


def global_indices(lay: dict[str, Any], o: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([slots(lay, int(r), row) for r, row in zip(o["req"], o["idx"])])


# ------------------------------------------------------------------------------------------------------------------ the SASS
def flashmla_library() -> str:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("no vllm package on the path")
    so = sorted(Path(spec.submodule_search_locations[0]).glob("_flashmla_C*.so"))
    if not so:
        raise RuntimeError("no _flashmla_C*.so beside vllm")
    return str(so[0])


def split_sass(lines) -> Any:
    """(arch, function, its lines) for each function of a `cuobjdump -sass` listing."""
    arch, name, buf = None, None, []
    for line in lines:
        m = re.match(r"\s*arch = (sm_\w+)", line)
        f = re.match(r"\s*Function : (\S+)", line) or re.match(r"\s*\.section\s+\.text\.([^\s,]+)", line)
        if m or f or line.startswith("Fatbin") or re.match(r"\s*\.section\s", line):
            if name is not None:
                yield arch, name, buf
            name, buf = (f.group(1) if f else None), []
            if m:
                arch = m.group(1)
        if name is not None:
            buf.append(line)
    if name is not None:
        yield arch, name, buf


def sass(out: str) -> dict[str, Any]:
    """The library's CUDA images for `SASS_ARCH`, each disassembled alone, since a toolkit's disassembler stops at the first
    image of an architecture it does not know and the library holds sm_100's too.  Kept under ``<out>/sass``: the listing of
    each function `SASS_KEEP` names, and under ``elf/`` each image that holds one."""
    outdir = Path(out) / "sass"
    elf = outdir / "elf"
    elf.mkdir(parents=True, exist_ok=True)
    so, tool = flashmla_library(), C._cuda_tool("cuobjdump")
    rec: dict[str, Any] = {"library": C._archs(so), "arch": SASS_ARCH, "images": [], "functions": {SASS_ARCH: {}}, "kept": [],
                           "instance": []}
    x = C._run([tool, "-xelf", "all", so], cwd=elf)
    rec["xelf_rc"] = x.returncode
    cubins = sorted(elf.glob(f"*.{SASS_ARCH}.cubin"), key=lambda p: int(p.name.split(".")[-3]))
    for other in set(elf.glob("*.cubin")) - set(cubins):
        other.unlink()
    filt = shutil.which("c++filt")
    for cubin in cubins:
        p = C._run([tool, "-sass", str(cubin)])
        img: dict[str, Any] = {"file": cubin.name, "sha256": C._sha256_file(str(cubin)), "rc": p.returncode, "functions": 0, "kept": 0}
        if p.returncode:
            img["error"] = (p.stdout + p.stderr)[-300:]
        for _, name, buf in split_sass(p.stdout.splitlines(keepends=True)):
            text = "".join(buf)
            n_ins = len(re.findall(r"/\*[0-9a-f]{4,}\*/\s+\S", text))
            rec["functions"][SASS_ARCH][name] = n_ins
            img["functions"] += 1
            if not SASS_KEEP.search(name):
                continue
            path = outdir / f"{len(rec['kept']):03d}.{SASS_ARCH}.sass.gz"
            with gzip.open(path, "wt") as f:
                f.write(text)
            ent = {"file": path.name, "arch": SASS_ARCH, "image": cubin.name, "function": name, "instructions": n_ins,
                   "sha256": hashlib.sha256(text.encode()).hexdigest(), "fp_ops": M.sass_census(text)}
            if filt:
                ent["demangled"] = C._run([filt, name]).stdout.strip()[:2000]
            rec["kept"].append(ent)
            img["kept"] += 1
            if SASS_INSTANCE.search(name):
                rec["instance"].append(ent["file"])
        if not img["kept"]:
            cubin.unlink()
        rec["images"].append(img)
    rec["ok"] = bool(cubins and all(i["rc"] == 0 for i in rec["images"]) and rec["instance"])
    (outdir / "index.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[glm53-flashmla-fp8] sass library={rec['library'].get('sha256')} archs={rec['library'].get('sass_archs')} "
          f"images={len(cubins)} failing={[i['file'] for i in rec['images'] if i['rc']]} "
          f"functions={len(rec['functions'][SASS_ARCH])} kept={len(rec['kept'])} instance={rec['instance']}",
          flush=True)
    for e in rec["kept"]:
        print(f"[glm53-flashmla-fp8] sass {e['file']} ins={e['instructions']} {e['function'][:150]} fp={json.dumps(e['fp_ops'])}",
              flush=True)
    return rec


# ---------------------------------------------------------------------------------------------------------------- the GPU side
def selection() -> dict[str, Any]:
    """What vLLM's sparse backend would do with 8 heads and an FP8 cache, and whose interface it calls."""
    from vllm.v1.attention.backends.mla import flashmla_sparse as FS
    from vllm.v1.attention.ops import flashmla as VF

    import vllm.third_party.flashmla.flash_mla_interface as FI

    sup = VF.is_flashmla_sparse_supported()
    sha = hashlib.sha256(Path(FI.__file__).read_bytes()).hexdigest()
    fn = FS.flash_mla_with_kvcache
    rec = {"padded_heads": int(FS.FlashMLASparseImpl._compute_fp8_decode_padded_heads(NH)),
           "min_heads_for_bf16_prefill": int(FS.MIN_HEADS_FOR_BF16_PREFILL), "sparse_supported": list(sup),
           "interface": FI.__file__, "interface_sha256": sha, "decode_fn": f"{fn.__module__}.{fn.__name__}"}
    rec["checks"] = {"padded_heads": rec["padded_heads"] == H_Q, "mixed_batch": NH < rec["min_heads_for_bf16_prefill"],
                     "sparse_supported": bool(sup[0]), "interface": sha == D.INTERFACE_SHA256,
                     "decode_fn": rec["decode_fn"] == "vllm.third_party.flashmla.flash_mla_interface.flash_mla_with_kvcache"}
    rec["ok"] = all(rec["checks"].values())
    return rec


def write_cache(cache_t, lay: dict[str, Any], dev: str) -> dict[str, Any]:
    """vLLM's write of every row of the requests numpy does not write: `fused_norm_rope` under `fp8_ds_mla`, as
    `DeepseekV32Attention` calls it for a layer without an indexer, on random inputs: one row in 16 zero, one in 8 far below
    the norm's epsilon, the rest with gains in [1/4, 4), under norm weights with gains in [1/8, 8)."""
    import torch
    from vllm.models.deepseek_v32.common import kernels as K

    g = torch.Generator(device=dev).manual_seed(SEED + 2)
    bf = torch.bfloat16

    def randn(*shape):
        return torch.randn(*shape, device=dev, generator=g)

    def rand(*shape):
        return torch.rand(*shape, device=dev, generator=g)

    half = DR // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=dev) / half))
    freqs = torch.einsum("i,j->ij", torch.arange(8192, dtype=torch.float32, device=dev), inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    qw = randn(Q_LORA).to(bf)
    kvw = (randn(DN) * torch.exp2(6 * rand(DN) - 3)).to(bf)
    pos = np.concatenate([np.arange(int(n)) for r, n in enumerate(lay["lens"]) if not lay["numpy"][r]])
    slot = np.concatenate([slots(lay, r, np.arange(int(n))) for r, n in enumerate(lay["lens"]) if not lay["numpy"][r]])
    step = 8192
    topk_buf = torch.full((step, TOPK), 7, dtype=torch.int32, device=dev)
    for lo in range(0, pos.size, step):
        n = min(step, pos.size - lo)
        u = rand(n, 1)
        gain = torch.where(u < 1 / 16, torch.zeros_like(u),
                           torch.where(u < 3 / 16, torch.exp2(-8 - 12 * rand(n, 1)), torch.exp2(4 * rand(n, 1) - 2)))
        kv_c, k_pe = (randn(n, DN) * gain).to(bf), randn(n, DR).to(bf)
        K.fused_norm_rope(
            torch.from_numpy(pos[lo:lo + n]).to(dev), randn(n, Q_LORA).to(bf), qw, RMS_EPS, kv_c, kvw, RMS_EPS,
            k_pe, cos_sin, None, None, None, RMS_EPS, None, topk_buf[:n],
            slot_mapping=torch.from_numpy(slot[lo:lo + n].astype(np.int64)).to(dev), indexer_k_cache=None, mla_kv_cache=cache_t,
            mla_kv_cache_dtype="fp8_ds_mla", mla_k_scale=None, has_indexer=False, index_rope_interleave=False,
            kv_c_out=torch.empty_like(kv_c), k_pe_out=torch.empty_like(k_pe), index_k_out=None)
    torch.cuda.synchronize()
    return {"by": f"{K.fused_norm_rope.__module__}.fused_norm_rope", "rows": int(pos.size),
            "topk_buffer_untouched": bool((topk_buf == 7).all().item())}


def build_cache(lay: dict[str, Any], dev: str, doc: dict[str, Any]):
    import torch

    host = host_cache(lay)
    cache_t = torch.from_numpy(host).to(dev)
    rec = write_cache(cache_t, lay, dev)
    cache = cache_t.cpu().numpy()
    flat, hflat = cache.reshape(-1, ROW_BYTES), host.reshape(-1, ROW_BYTES)
    mine = np.zeros(flat.shape[0], dtype=bool)
    for r, n in enumerate(lay["lens"]):
        if not lay["numpy"][r]:
            mine[slots(lay, r, np.arange(int(n)))] = True
    changed = np.any(flat != hflat, axis=1)
    _, scale, _ = unpack_rows(flat[mine])
    rec.update(rows_changed=int(changed[mine].sum()), other_rows_changed=int(changed[~mine].sum()),
               rows_at_the_floor_scale=int(np.any(scale == 0x00800000, axis=1).sum()),
               distinct_scale_exponents=int(np.unique(scale >> 23).size))
    rec["ok"] = rec["rows_changed"] == rec["rows"] and rec["other_rows_changed"] == 0
    doc["write"] = rec
    return cache_t, cache


def launch_case(case: str, lay: dict[str, Any], ctx: dict[str, Any], outdir: Path) -> dict[str, Any]:
    """One case: `_forward_fp8_kv_mixed_batch` on the stand-in, its call into FlashMLA recorded, and that call again."""
    import torch
    from vllm.v1.attention.backends.mla import flashmla_sparse as FS

    dev = ctx["dev"]
    o = case_operands(case, lay)
    t = o["q"].shape[0]
    q = torch.from_numpy(o["q"].view(np.int16)).to(dev).view(torch.bfloat16)
    idx, req = torch.from_numpy(o["idx"]).to(dev), torch.from_numpy(o["req"]).to(dev)
    sched, _ = FS.get_mla_metadata(cache_seqlens=ctx["topk_tokens"][:1], num_q_tokens_per_head_k=t * H_Q, topk=TOPK,
                                   num_heads_q=H_Q, num_heads_k=1, is_fp8_kvcache=True)
    meta = FS.FlashMLASparseMetadata.FP8KernelMetadata(scheduler_metadata=sched, cache_lens=ctx["max_model_len"][:1],
                                                       dummy_block_table=ctx["dummy_block_table"][:1])
    md = types.SimpleNamespace(req_id_per_token=req, block_table=ctx["block_table"], block_size=PBS, fp8_extra_metadata=meta)
    calls: list[tuple[dict[str, Any], Any]] = []
    real = FS.flash_mla_with_kvcache

    def recorder(*args, **kw):
        ret = real(*args, **kw)
        calls.append(({"args": args, **kw}, ret))
        return ret

    FS.flash_mla_with_kvcache = recorder
    try:
        out, lse_ret = FS.FlashMLASparseImpl._forward_fp8_kv_mixed_batch(ctx["stub"], q, ctx["cache"], idx, md)
    finally:
        FS.flash_mla_with_kvcache = real
    kw, (raw, raw_lse) = calls[0]
    kw = {k: v for k, v in kw.items() if k != "args"}
    torch.cuda.synchronize()
    tsm, ns = sched.tile_scheduler_metadata.cpu().numpy(), sched.num_splits.cpu().numpy()
    gidx = kw["indices"][0].cpu().numpy()
    d = outdir / case
    d.mkdir(parents=True, exist_ok=True)
    for k, a in (("q", o["q"]), ("idx", o["idx"]), ("req", o["req"]), ("gidx", gidx.astype(np.int32)), ("out", D._u16(raw[0])),
                 ("lse", raw_lse.contiguous().view(torch.int32).cpu().numpy().view(U32)), ("metas", tsm[:, :7].astype(np.int32)),
                 ("num_splits", ns.astype(np.int32))):
        np.save(d / f"{k}.npy", np.ascontiguousarray(a))

    def same(a, b) -> bool:
        return bool(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)))

    def direct(sentinel: int, **more):
        buf = torch.full((1, t, H_Q, DV), D._i16(sentinel), dtype=torch.int16, device=dev).view(torch.bfloat16)
        m, _ = FS.get_mla_metadata()
        ret, lse = real(**{**kw, "tile_scheduler_metadata": m, "out": buf, **more})
        torch.cuda.synchronize()
        return buf, ret, lse, m

    def rerun() -> dict[str, bool]:
        b1, r1, l1, m1 = direct(D.SENTINELS[0])
        b2, _, _, _ = direct(D.SENTINELS[1])
        return {"rerun": same(raw, b1) and same(b1, b2) and same(r1, b1),
                "lse_rerun": bool(torch.equal(raw_lse.contiguous().view(torch.int32), l1.contiguous().view(torch.int32))),
                "planner_rerun": bool(torch.equal(sched.tile_scheduler_metadata[:, :7], m1.tile_scheduler_metadata[:, :7])
                                      and torch.equal(sched.num_splits, m1.num_splits))}

    def sink_ninf() -> bool:
        b3, _, _, _ = direct(D.SENTINELS[0], attn_sink=torch.full((H_Q,), float("-inf"), dtype=torch.float32, device=dev))
        return same(raw, b3)

    def planner() -> bool:
        want_m, want_ns = T.sched_meta([TOPK], topk=TOPK, parts=parts)
        want = np.array([m.words() for m in want_m], dtype=np.int32)
        return bool(tsm.shape[0] == want.shape[0] and np.array_equal(tsm[:, :7], want) and ns.tolist() == list(want_ns))

    parts = T.num_sm_parts(H_Q, s_q=t)
    again = rerun()
    rec = {"case": case, "tokens": t, "num_sm_parts": parts, "num_splits": ns.tolist(),
           "call": {k: (list(v.shape) if hasattr(v, "shape") else v) for k, v in kw.items() if k != "tile_scheduler_metadata"},
           "call_dtypes": {k: str(v.dtype) for k, v in kw.items() if hasattr(v, "dtype")}, "lse_shape": list(raw_lse.shape),
           "splits": list(T.request_splits([T.SchedMeta(*(int(v) for v in m[:7])) for m in tsm], [int(v) for v in ns], 0,
                                           TOPK // BN)),
           "checks": {
               "one_call": len(calls) == 1 and not calls[0][0]["args"],
               "arguments": bool(sorted(kw) == sorted(["q", "k_cache", "block_table", "head_dim_v", "cache_seqlens",
                                                       "tile_scheduler_metadata", "is_fp8_kvcache", "indices", "softmax_scale"])
                                 and kw["head_dim_v"] == DV and kw["softmax_scale"] == SCALE and kw["is_fp8_kvcache"] is True
                                 and list(kw["q"].shape) == [1, t, H_Q, DK] and list(kw["indices"].shape) == [1, t, TOPK]
                                 and list(kw["k_cache"].shape) == [lay["blocks"], PBS, 1, ROW_BYTES]
                                 and kw["k_cache"].dtype == torch.uint8),
               "pad_heads_zero": same(kw["q"][0, :, :NH], q)
                                 and not bool(kw["q"][0, :, NH:].contiguous().view(torch.int16).any().item()),
               "indices": bool(np.array_equal(gidx, global_indices(lay, o))),
               "returned": lse_ret is None and same(out, raw[0, :, :NH]) and list(raw.shape) == [1, t, H_Q, DV],
               **again,
               "sink_ninf": sink_ninf(),
               "sched_shape": tuple(tsm.shape) == (parts, 8) and tuple(ns.shape) == (2,),
               "sched": planner()}}
    torch.cuda.empty_cache()
    return rec


def readback(lay: dict[str, Any], ctx: dict[str, Any], outdir: Path) -> dict[str, Any]:
    """`cp_gather_and_upconvert_fp8_kv_cache` over the probed blocks as one request whose block table they are: (rows, 576)."""
    import torch
    from vllm import _custom_ops as ops

    dev = ctx["dev"]
    rows = probe_rows(lay)
    blocks = sorted({int(lay["table"][r, p // PBS]) for r, p in rows})
    n = len(blocks) * PBS
    table = torch.tensor([blocks], dtype=torch.int32, device=dev)
    starts = torch.zeros(1, dtype=torch.int32, device=dev)
    bufs = []
    for s in D.SENTINELS:
        dst = torch.full((n, DK), D._i16(s), dtype=torch.int16, device=dev).view(torch.bfloat16)
        ops.cp_gather_and_upconvert_fp8_kv_cache(ctx["cache"], dst, table, starts, 1)
        bufs.append(dst)
    torch.cuda.synchronize()
    d = outdir / "readback"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "rows.npy", D._u16(bufs[0]))
    np.save(d / "blocks.npy", np.array(blocks, dtype=np.int32))
    return {"blocks": blocks, "rows": n, "rerun": bool(torch.equal(bufs[0].view(torch.int16), bufs[1].view(torch.int16)))}


def launch(out: str, quick: bool, min_driver: int, want_cc: str, want_sms: int) -> dict[str, Any]:
    outdir = Path(out) / "capture"
    outdir.mkdir(parents=True, exist_ok=True)
    import torch
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    # a launch that does not finish leaves no launch record: nothing catches a failure, which is this process's traceback
    (outdir / "launch.json").unlink(missing_ok=True)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "quick": quick, "environment": G._environment(),
                           "gpu_lease_uuid": os.environ.get("GPU_LEASE_UUID"), "gate": C.gate(min_driver, want_cc, want_sms),
                           "launches": [], "cases": case_names(quick)}
    print(f"[glm53-flashmla-fp8] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / "launch.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print("GLM53-FLASHMLA-FP8 GATE-FAIL", flush=True)
        raise SystemExit(3)

    dev = "cuda"
    lay = layout(quick)
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        doc["selection"] = selection()
        print(f"[glm53-flashmla-fp8] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
        doc["stack"] = D.stack()
        cache_t, cache = build_cache(lay, dev, doc)
        np.save(outdir / "cache.npy", cache)
        for k in ("lens", "table", "numpy"):
            np.save(outdir / f"layout_{k}.npy", lay[k])
        print(f"[glm53-flashmla-fp8] cache blocks={lay['blocks']} requests={len(lay['lens'])} write={json.dumps(doc['write'])[:900]}",
              flush=True)
        from vllm.v1.attention.backends.mla import flashmla_sparse as FS

        stub = types.SimpleNamespace(dcp_world_size=1, need_to_return_lse_for_decode=False, softmax_scale=SCALE,
                                     fp8_decode_padded_heads=FS.FlashMLASparseImpl._compute_fp8_decode_padded_heads(NH))
        stub._fp8_flash_mla_kernel = types.MethodType(FS.FlashMLASparseImpl._fp8_flash_mla_kernel, stub)
        ctx = {"dev": dev, "cache": cache_t, "stub": stub, "block_table": torch.from_numpy(lay["table"]).to(dev),
               "topk_tokens": torch.full((1,), TOPK, dtype=torch.int32, device=dev),
               "max_model_len": torch.full((1,), 1 << 20, dtype=torch.int32, device=dev),
               "dummy_block_table": torch.empty((1, 1), dtype=torch.int32, device=dev)}
        for case in doc["cases"]:
            r = launch_case(case, lay, ctx, outdir)
            doc["launches"].append(r)
            print(f"[glm53-flashmla-fp8] launch {case:10s} T={r['tokens']} parts={r['num_sm_parts']} splits={r['splits']} "
                  f"{json.dumps(r['checks'])}", flush=True)
        doc["readback"] = readback(lay, ctx, outdir)
        print(f"[glm53-flashmla-fp8] readback {json.dumps(doc['readback'])}", flush=True)
    (outdir / "launch.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"[glm53-flashmla-fp8] launch done: {len(doc['launches'])} of {len(doc['cases'])} cases", flush=True)
    return doc


# ------------------------------------------------------------------------------------------------------------- the comparison
def _case(outdir: Path, case: str) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p, mmap_mode="r") for p in sorted((outdir / case).glob("*.npy"))}


def compare_tokens(job: dict[str, Any]) -> dict[str, Any]:
    """Tokens [t0, t1) of a case: the 8 real heads and pad head 8 against the twin at the step's SPLITS; pad heads 9..63 against
    pad head 8's words."""
    rec = {"case": job["case"], "words": 0, "differ": 0, "twin_words": 0, "twin_differ": 0, "pad_words": 0, "pad_differ": 0,
           "first": None}
    outdir = Path(job["dir"])
    a, cache = _case(outdir, job["case"]), np.load(outdir / "cache.npy", mmap_mode="r")
    metas = [T.SchedMeta(*(int(v) for v in m)) for m in np.asarray(a["metas"])]
    splits = T.request_splits(metas, [int(v) for v in np.asarray(a["num_splits"])], 0, TOPK // BN)
    for t in range(*job["tokens"]):
        g = np.asarray(a["gidx"][t])
        q = np.zeros((NH + 1, DK), dtype=np.uint16)
        q[:NH] = a["q"][t]
        y = V.decode(q, *rows_at(cache, g), g, SPLITS=splits, QT=QT, BN=BN, DV=DV, SCALE=SCALE)
        got = np.asarray(a["out"][t], dtype=np.uint16)
        bad = np.argwhere(y != got[:NH + 1])
        pad = int((got[NH + 1:] != got[NH]).sum())
        rec["twin_words"] += int(y.size)
        rec["twin_differ"] += int(bad.shape[0])
        rec["pad_words"] += int(got[NH + 1:].size)
        rec["pad_differ"] += pad
        if bad.shape[0] and rec["first"] is None:
            h, i = (int(v) for v in bad[0])
            rec["first"] = {"token": t, "head": h, "word": i, "twin": int(y[h, i]), "kernel": int(got[h, i]),
                            "splits": list(splits)}
    rec["words"], rec["differ"] = rec["twin_words"] + rec["pad_words"], rec["twin_differ"] + rec["pad_differ"]
    return rec


def definition_tokens(tokens: int, n: int, whole: bool) -> list[int]:
    """The tokens of a case of ``tokens`` held to the Definition's evaluator: all of them, or ``n`` spread from first to last."""
    if whole or n >= tokens:
        return list(range(tokens)) if n > 0 else []
    return sorted({round(i * (tokens - 1) / max(n - 1, 1)) for i in range(n)})


def definition_token(job: dict[str, Any]) -> dict[str, Any]:
    """One token's 8 real heads against `FlashMlaV32Decode_v1` at GLM-5.3's statics and the step's SPLITS, through the
    reference evaluator: on the captured queries, the cache rows the token's indices name (any row for an index of -1) and
    the indices."""
    rec = {"case": job["case"], "token": job["token"], "words": 0, "differ": 0, "first": None}
    from verity.circuits.defs import bind
    from verity.circuits.evaluation import evaluate
    from verity.definitions.registry import flashmla_dsv32 as R

    outdir, t = Path(job["dir"]), job["token"]
    a, cache = _case(outdir, job["case"]), np.load(outdir / "cache.npy", mmap_mode="r")
    metas = [T.SchedMeta(*(int(v) for v in m)) for m in np.asarray(a["metas"])]
    splits = T.request_splits(metas, [int(v) for v in np.asarray(a["num_splits"])], 0, TOPK // BN)
    target = bind(R.FlashMlaV32Decode, NH=NH, SPLITS=splits, DN=DN, DR=DR, QT=QT, BN=BN, DOT=R.FLASHMLA_GLM53["DOT"], SCALE=SCALE)
    g = np.asarray(a["gidx"][t])
    nope, scale, rope = rows_at(cache, g)
    y = np.asarray(evaluate(target, np.asarray(a["q"][t], dtype=np.uint16).reshape(-1), nope.reshape(-1), scale.reshape(-1),
                            rope.reshape(-1), g.astype(np.int64).astype(U32)), dtype=np.uint16).reshape(NH, DV)
    got = np.asarray(a["out"][t][:NH], dtype=np.uint16)
    bad = np.argwhere(y != got)
    rec.update(target=target.id, words=int(y.size), differ=int(bad.shape[0]))
    if bad.shape[0]:
        h, i = (int(v) for v in bad[0])
        rec["first"] = {"token": t, "head": h, "word": i, "definition": int(y[h, i]), "kernel": int(got[h, i])}
    return rec


def dense_rows(nope, scale) -> np.ndarray:
    """The read-back's arithmetic as its source states it (cache_kernels.cu:1360-1385, quant_utils.cuh:152-176): each E4M3
    element times its tile's f32 scale in f32, then rounded to bf16."""
    x = CK.e4m3_to_f32_batch(np.asarray(nope))[0]
    return DRW.f2fp_bf16(T.fmul(x, np.repeat(np.asarray(scale, dtype=U32), QT, axis=-1)))


def compare_probes(outdir: Path, cases: list[str]) -> dict[str, Any]:
    """The two reads of one row: a probe token's output on a pad head against the read-back's words of that row.  A pad head's
    query is zero, so its score of the one valid row is 0, the row's weight is exactly 1 and its output is the row as the
    sparse kernel staged it (-0 as +0); a real head's weight is 2^r for the residual r its score's product with the scale
    leaves under the FMA, and is not compared here.  The probe cannot tell -0 from +0, so a word the read-back has as -0 and
    the probe as +0 is counted apart (`zero_sign_words`) and is no difference.  Rows whose key holds a NaN or an infinity (in
    a staged word, or in a RoPE word, which makes the score NaN) are counted apart: the probe does not show them staged."""
    lay = {k: np.load(outdir / f"layout_{k}.npy") for k in ("lens", "table", "numpy")}
    cache = np.load(outdir / "cache.npy", mmap_mode="r")
    rb, blocks = np.load(outdir / "readback" / "rows.npy"), np.load(outdir / "readback" / "blocks.npy").tolist()
    plant = len(lay["lens"]) - 1
    planted = set(slots(lay, plant, np.arange(PLANT_ROWS)).tolist())
    rec: dict[str, Any] = {"cases": {}, "rows": 0, "rows_differ": 0, "words": 0, "words_differ": 0, "one_ulp": 0,
                           "zero_sign_words": 0, "special_rows": 0, "special_words_differ": 0, "rope_words": 0, "rope_differ": 0,
                           "sparse_model_differ": 0, "dense_model_differ": 0, "by_writer": {}, "examples": []}
    for case in cases:
        a = _case(outdir, case)
        c = {"tokens": int(a["out"].shape[0]), "pad_heads_disagree": 0}
        for t in range(c["tokens"]):
            g = np.asarray(a["gidx"][t])
            slot = int(g[g != -1][0])
            pads = np.asarray(a["out"][t][NH:], dtype=np.uint16)
            got = pads[0]
            back = rb[blocks.index(slot // PBS) * PBS + slot % PBS]
            nope, scale, rope = rows_at(cache, [slot])
            staged = V.kv_rows(nope, scale, rope, [True], QT)[0, :DN]
            dense = dense_rows(nope, scale)[0]
            special = bool(np.any((staged & 0x7F80) == 0x7F80) or np.any((rope[0] & 0x7F80) == 0x7F80))
            c["pad_heads_disagree"] += int(np.any(pads != got))
            zero_sign = (got == 0) & (back[:DN] == 0x8000)
            diff = (got != back[:DN]) & ~zero_sign
            r = int(a["req"][t])
            writer = "planted" if slot in planted else "numpy" if lay["numpy"][r] else "vllm"
            w = rec["by_writer"].setdefault(writer, {"rows": 0, "rows_differ": 0, "words": 0, "words_differ": 0, "one_ulp": 0,
                                                     "zero_sign_words": 0, "least_in_a_row": None, "most_in_a_row": 0})
            rec["rope_words"] += DR
            rec["rope_differ"] += int((back[DN:] != rope[0]).sum())
            rec["sparse_model_differ"] += int((got != np.where(staged == 0x8000, 0, staged)).sum()) if not special else 0
            rec["dense_model_differ"] += int((back[:DN] != dense).sum())
            if special:
                rec["special_rows"] += 1
                rec["special_words_differ"] += int(diff.sum())
                continue
            n = int(diff.sum())
            one = int((np.abs(got.astype(np.int32) - back[:DN].astype(np.int32))[diff] == 1).sum())
            for d in (rec, w):
                d["rows"] += 1
                d["rows_differ"] += int(n > 0)
                d["words"] += DN
                d["words_differ"] += n
                d["one_ulp"] += one
                d["zero_sign_words"] += int(zero_sign.sum())
            w["least_in_a_row"] = n if w["least_in_a_row"] is None else min(w["least_in_a_row"], n)
            w["most_in_a_row"] = max(w["most_in_a_row"], n)
            if n and len(rec["examples"]) < 6:
                i = int(np.flatnonzero(diff)[0])
                rec["examples"].append({"case": case, "token": t, "writer": writer, "element": i, "byte": int(nope[0, i]),
                                        "scale": int(scale[0, i // QT]), "sparse": int(got[i]), "readback": int(back[i])})
            if writer == "planted" and int(nope[0, 0]) == HAND_BYTE and int(scale[0, 0]) == HAND_SCALE:
                rec["hand"] = {"byte": HAND_BYTE, "scale": HAND_SCALE, "sparse": int(got[0]), "readback": int(back[0]),
                               "row_words_differ": int(diff.sum())}
        rec["cases"][case] = c
    return rec


def compare(out: str, workers: int, definition: int = 0) -> dict[str, Any]:
    root, outdir = Path(out), Path(out) / "capture"
    # a comparison that does not finish leaves no record: a worker's failure is raised here, with its traceback
    (root / RECORD).unlink(missing_ok=True)
    doc = json.loads((outdir / "launch.json").read_text())
    try:
        s = json.loads((root / "sass" / "index.json").read_text())
        doc["sass"] = {"ok": s["ok"], "instance": s["instance"], "library": s["library"],
                       "kept": [{k: e.get(k) for k in ("file", "arch", "image", "function", "instructions", "sha256", "fp_ops")}
                                for e in s["kept"]]}
    except (OSError, ValueError, KeyError) as e:  # no listing, or not one `sass` finished: the record says so and is not ok
        doc["sass"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    done = [x["case"] for x in doc["launches"]]
    jobs = [{"dir": str(outdir), "case": x["case"], "tokens": [lo, min(x["tokens"], lo + JOB_TOKENS)]}
            for x in doc["launches"] for lo in range(0, x["tokens"], JOB_TOKENS)]
    print(f"[glm53-flashmla-fp8] comparing {sum(x['tokens'] for x in doc['launches'])} tokens of {len(done)} cases in {len(jobs)} "
          f"jobs on {workers} workers", flush=True)
    keys = ("words", "differ", "twin_words", "twin_differ", "pad_words", "pad_differ")
    comps: dict[str, dict[str, Any]] = {c: {"case": c, **{k: 0 for k in keys}, "first": None} for c in done}
    with ProcessPoolExecutor(max_workers=max(workers, 1), mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in (ex.map(compare_tokens, jobs) if workers else map(compare_tokens, jobs)):
            agg = comps[res["case"]]
            for k in keys:
                agg[k] += res[k]
            agg["first"] = agg["first"] or res["first"]
        djobs = [{"dir": str(outdir), "case": x["case"], "token": t} for x in doc["launches"]
                 for t in definition_tokens(x["tokens"], definition, x["case"] in DEFINITION_WHOLE)]
        print(f"[glm53-flashmla-fp8] holding {len(djobs)} tokens to FlashMlaV32Decode_v1's evaluator", flush=True)
        held: dict[str, Any] = {"asked": definition, "tokens": 0, "words": 0, "differ": 0, "targets": [], "cases": {}, "first": None}
        for res in (ex.map(definition_token, djobs) if workers else map(definition_token, djobs)):
            c = held["cases"].setdefault(res["case"], {"tokens": [], "words": 0, "differ": 0})
            c["tokens"].append(res["token"])
            for d in (held, c):
                d["words"] += res["words"]
                d["differ"] += res["differ"]
            held["tokens"] += 1
            held["first"] = held["first"] or (res["first"] and {"case": res["case"], **res["first"]})
            if res["target"] not in held["targets"]:
                held["targets"].append(res["target"])
            print(f"[glm53-flashmla-fp8] definition {res['case']:10s} token {res['token']} words={res['words']} "
                  f"differ={res['differ']}", flush=True)
    doc["definition"] = held
    doc["comparisons"] = list(comps.values())
    doc["totals"] = {k: sum(c[k] for c in comps.values()) for k in keys}
    doc["reads"] = compare_probes(outdir, [c for c in ("probe", "probesplit") if c in done])
    launches_ok = {x["case"]: all(v is True for v in x["checks"].values()) for x in doc["launches"]}
    for c in doc["comparisons"]:
        c["launch_ok"] = launches_ok[c["case"]]
        print(f"[glm53-flashmla-fp8] case {c['case']:10s} words={c['words']} differ={c['differ']} (twin {c['twin_words']}/"
              f"{c['twin_differ']}, pad heads {c['pad_words']}/{c['pad_differ']}) launch_ok={c['launch_ok']} "
              f"first={json.dumps(c['first'])}", flush=True)
    # `errors`: a launch file written when a failed step was recorded there instead of raised
    ok = bool(doc["gate"]["ok"] and not doc.get("errors") and doc["selection"].get("ok") is True and doc["sass"].get("ok") is True
              and doc.get("write", {}).get("ok") is True and done == doc["cases"] and all(launches_ok.values())
              and doc.get("readback", {}).get("rerun") is True and doc["totals"]["words"] > 0 and doc["totals"]["differ"] == 0
              and held["differ"] == 0 and (definition == 0 or held["words"] > 0))
    body = {**{k: v for k, v in doc.items() if k != "digest"}, "ok": ok}
    doc = {**body, "digest": digest_of(body)}
    (root / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    r = doc["reads"]
    if r:
        print(f"[glm53-flashmla-fp8] two reads of one row: rows={r['rows']} differ={r['rows_differ']} words={r['words']} "
              f"differ={r['words_differ']} (by one bf16 ulp {r['one_ulp']}; -0 read as +0, no difference, {r['zero_sign_words']}) "
              f"by_writer={json.dumps(r['by_writer'])} special_rows={r['special_rows']} "
              f"hand={json.dumps(r.get('hand'))} rope {r['rope_words']}/{r['rope_differ']} sparse_model_differ="
              f"{r['sparse_model_differ']} dense_model_differ={r['dense_model_differ']} examples={json.dumps(r['examples'][:3])}",
              flush=True)
    print(f"[glm53-flashmla-fp8] FlashMlaV32Decode_v1's evaluator: tokens={held['tokens']} words={held['words']} "
          f"differ={held['differ']} first={json.dumps(held['first'])}", flush=True)
    print(f"GLM53-FLASHMLA-FP8 {'HOLDS' if ok else 'FAILS'} {doc['digest']} words={doc['totals']['words']} "
          f"differ={doc['totals']['differ']} cases={len(done)}/{len(doc['cases'])} launches_failing="
          f"{[c for c, v in launches_ok.items() if not v]}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's FlashMLA sparse read of the FP8 cache on this Hopper GPU, captured and "
                                             "compared with its twin")
    ap.add_argument("stage", choices=("sass", "launch", "compare"))
    ap.add_argument("--out", required=True, help=f"directory for sass/, capture/ and {RECORD}")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers(), help="comparison processes; 0 compares in this one")
    ap.add_argument("--quick", action="store_true", help="fewer requests and cases: a first signal in a few minutes")
    ap.add_argument("--definition", type=int, default=0, metavar="N",
                    help="compare: also hold N tokens a case (and every token of the planted rows' case) to FlashMlaV32Decode_v1's "
                         "own evaluator, about two minutes a token on one core")
    a = ap.parse_args(argv)
    if a.stage == "sass":
        return 0 if sass(a.out)["ok"] else 1
    if a.stage == "launch":
        launch(a.out, a.quick, a.min_driver, a.want_cc, a.want_sms)
        return 0
    return 0 if compare(a.out, a.workers, a.definition)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
