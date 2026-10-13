"""DeepSeek-V4-Pro's sparse MLA decode on Hopper against its Programs: the H200's go/no-go for binding `flashmla.FlashMlaDecode_v1`
(FlashMLA 6bc49418's `sparse_decode_fwd`, `flash_fwd_splitkv_mla_fp8_sparse_kernel<MODEL1, 64>` and its combine, as vLLM d9105ea80's
`DeepseekV4FlashMLAAttention._forward_decode` calls it) at DeepSeek-V4-Pro's statics on one TP8 rank: 16 real heads padded to 64,
the 128-token sliding window over an `fp8_ds_mla` cache, the compressed rows of a compress-ratio 4 and 128 layer, and the sink.

One GPU process writes `<out>/dsv4_flashmla_capture.json`, the property record `dsv4_flashmla_capture` (`properties.record`); it
prints one line per launch and per case, then `DSV4-FLASHMLA GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs
first and nothing else runs when it fails (exit 3).  A phase that raises records its error and fails the verdict without stopping
the others.  The operands are random at the served shapes (`case_operands`), so no model weights are read.

  gate       `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  selection  `_select_dsv4_attn_cls` picks `DeepseekV4FlashMLAAttention` on this device, its `get_padded_num_q_heads(16)` is 64,
             `is_flashmla_sparse_supported()` holds, and the installed `vllm.third_party.flashmla.flash_mla_interface` is the
             interface vLLM's build vendors from FlashMLA 6bc49418 (`INTERFACE_SHA256`).
  decode     per case (`CASES`), `flash_mla_with_kvcache` called as flashmla.py:248-266 calls it: q (b, 1, 64, 512) bf16 with the
             pad heads zero, the SWA cache (nb, 64, 1, 584) uint8 (a page of `page_bytes(64)`), indices (b, 1, 128) and
             topk_length, the sink (64,) f32 with the pad heads -inf, softmax_scale 512^-0.5, and on a compressed layer the extra
             cache, its indices and extra_topk_length; twice on one `FlashMLASchedMeta` into outputs filled with two different
             sentinels (every word written, deterministic) and once on a fresh one (the planner reruns to the same words).  The
             planner's `tile_scheduler_metadata` and `num_splits` against `flashmla_twin.sched_meta`, and every output word of
             every token and all 64 heads against `flashmla_twin.decode` at the token's SPLITS (`request_splits` of the GPU's
             metadata); on a few tokens of each case (the most and the fewest splits, every token of `edge`) the reference
             `FlashMlaDecode_v1{NH=16,...}` over the real heads against the twin.
  bind       the Programs bind at the served statics.

The cases: `swa` a window-only layer (compress_ratio 0) over swa_len 0..128; `c4` a compress-ratio 4 layer (extra page block 64,
1024 indices, the indexer's top-k); `c128` a compress-ratio 128 layer (extra page block 2, 128 indices as a slice of a 1024-wide
buffer, as vLLM's `c128a` indices are); `lone` one token of 1152 rows, cut by the planner into its most splits; `batch` 160 tokens
of random lengths (requests that share a partition); `edge` special words: UE8M0 scales 0x00 / 0xFE / 0xFF, E4M3 NaN bytes, -1
indices within a length and valid ones past it, repeated indices, overflowing, zero and subnormal queries, an empty request, and
sinks +-inf, NaN and +-1e30.

GO needs every case's words compared with 0 mismatches, the reference sampled with 0 disagreements with the twin, every launch
check, the selection and the bindings.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/dsv4_flashmla_capture_gpu.py --out DIR [--scratch DIR] [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.program.kernels import flashmla_twin as T
from verity_vllm.program.kernels.fp8_twins import np_f32_to_e4m3_sat
from verity_vllm.program.registry import flashmla as F
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/dsv4-flashmla-capture/v1"
RECORD = "dsv4_flashmla_capture.json"
#: DeepSeek-V4-Pro (config.json at b5968e91) at TP8: 128 / 8 = 16 local heads padded to h_q = 64 (flashmla.py:78-85), head_dim
#: 512 = NoPE 448 | RoPE 64, sliding_window 128, index_topk 1024; `FLASHMLA_DSV4` holds the Definitions' statics
H_Q = 64
NH, DN, DR, QT, BN, SCALE = (F.FLASHMLA_DSV4[k] for k in ("NH", "DN", "DR", "QT", "BN", "SCALE"))
D = DN + DR
WINDOW, C4_TOPK = 128, 1024
#: the `fp8_ds_mla` (MODEL1) token: 448 E4M3 + 64 bf16 bytes in the block's data region, 8 scale bytes (7 UE8M0 and a pad) in its
#: scale region; a page is padded to a multiple of 576 bytes (FlashMLA tests/quant.py:140-147, splitkv_mla.cuh:553-557)
TOKEN_BYTES, DATA_BYTES, SCALE_BYTES = 584, 576, 8
SWA_PBS = 64
#: the extra caches: (page block, index width, index buffer width).  C4: 256-token blocks / 4; C128: 256 / 128, and its indices
#: `buffer[:T, :W]` with W = min(max(next_pow2(max_seq_len / 128), 128), c128a_max_compressed) (here 128 of a 1024-wide buffer)
LAYERS: dict[str, tuple[int, int, int] | None] = {"swa": None, "c4": (64, C4_TOPK, C4_TOPK), "c128": (2, 128, 1024)}
#: blocks of each cache; the last block of each is reserved for `edge`'s special rows
CACHE_BLOCKS = {"swa": 48, "c4": 40, "c128": 512}
#: sha256 of vllm/third_party/flashmla/flash_mla_interface.py as cmake/external_projects/flashmla.cmake writes it from FlashMLA
#: 6bc49418c5ead572ff0339191ddf3b155749e183 (the shim `import vllm._flashmla_C` before `flash_mla_cuda = torch.ops._flashmla_C`)
INTERFACE_SHA256 = "84601897d0f3a748ef6a8046ed28285586c8db672ac08b71178e52574a3c57e5"
SENTINELS = (0x7FBF, 0xFFBF)     # bf16 NaNs the kernel never writes (its NaNs are 0x7FFF)
F_NINF = 0xFF800000

#: (swa_len, extra_len) per token
SWA_LENS = tuple((n, None) for n in (0, 1, 2, 63, 64, 65, 127, 128))
C4_LENS = ((128, 1024), (128, 1), (5, 1000), (128, 0), (64, 64), (128, 513), (1, 1024), (100, 257))
C128_LENS = ((128, 0), (128, 1), (77, 2), (128, 64), (128, 65), (3, 128))
LONE_LENS = ((128, 1024),)
BATCH, QUICK_BATCH = 160, 24
EDGE_LENS = ((128, 1024), (128, 300), (100, 200), (128, 1024), (128, 64), (0, 0), (128, 128), (64, 700))
EDGE_SINKS = (0x7F800000, F_NINF, 0x7FC00000, 0, 0x7149F2CA, 0xF149F2CA, 1, 0x42B00000, 0xC2B00000, 0x3F800000, 0x80000000,
              0x3E800000, 0x41200000, 0xC1200000, 0x00800000, 0xBF800000)
#: case -> (layer, seed)
CASE_LAYERS = {"swa": ("swa", 1), "c4": ("c4", 2), "c128": ("c128", 3), "lone": ("c4", 4), "batch": ("c4", 5), "edge": ("c4", 6)}
CASES = tuple(("decode", c) for c in CASE_LAYERS)
#: tokens per twin job
JOB_TOKENS = 4
TOTALS = ("words", "mismatch", "reference", "twin_mismatch")


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()[-600:] + " | " + traceback.format_exc(limit=-3)[-900:]


def _save(d: Path, **arrays: np.ndarray) -> dict[str, str]:
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, a in arrays.items():
        np.save(d / f"{k}.npy", np.ascontiguousarray(a))
        out[k] = str(d / f"{k}.npy")
    return out


def _load(paths: dict[str, str]) -> dict[str, np.ndarray]:
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


def _u16(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _bf16(x: np.ndarray) -> np.ndarray:
    """RNE of f32 values to bf16 words (finite inputs)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


# ------------------------------------------------------------------------------------------------------------------- the cache
def page_bytes(pbs: int) -> int:
    return -(-pbs * TOKEN_BYTES // DATA_BYTES) * DATA_BYTES


def pack_cache(nope: np.ndarray, scale: np.ndarray, rope: np.ndarray, pbs: int, fill: np.ndarray | None = None) -> np.ndarray:
    """Rows (n, DN) e4m3, (n, 7) UE8M0 and (n, DR) bf16 words as a MODEL1 cache (n / pbs, page_bytes(pbs)) of bytes; the bytes no
    field holds (the eighth scale byte, the page's padding) are ``fill``'s, else 0."""
    nb = nope.shape[0] // pbs
    c = np.zeros((nb, page_bytes(pbs)), dtype=np.uint8) if fill is None else np.array(fill, dtype=np.uint8)
    data = c[:, :pbs * DATA_BYTES].reshape(nb, pbs, DATA_BYTES)
    data[..., :DN] = nope.reshape(nb, pbs, DN)
    data[..., DN:] = np.ascontiguousarray(rope, dtype="<u2").reshape(nb, pbs, DR).view(np.uint8)
    c[:, pbs * DATA_BYTES:pbs * TOKEN_BYTES].reshape(nb, pbs, SCALE_BYTES)[..., :DN // QT] = scale.reshape(nb, pbs, DN // QT)
    return c


def gather_rows(cache: np.ndarray, idx, pbs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rows the kernel reads for indices ``idx`` (block idx // pbs, offset idx % pbs): (n, DN) e4m3, (n, 7) UE8M0, (n, DR) bf16;
    a -1 row is zeros (the kernel zeroes it)."""
    cache = np.asarray(cache)
    nb = cache.shape[0]
    idx = np.asarray(idx, dtype=np.int64)
    ok = idx != -1
    i = np.where(ok, idx, 0)
    blk, off = i // pbs, i % pbs
    data = cache[:, :pbs * DATA_BYTES].reshape(nb, pbs, DATA_BYTES)[blk, off]
    sc = cache[:, pbs * DATA_BYTES:pbs * TOKEN_BYTES].reshape(nb, pbs, SCALE_BYTES)[blk, off, :DN // QT]
    nope = np.where(ok[:, None], data[:, :DN], 0).astype(np.uint8)
    rope = np.where(ok[:, None], np.ascontiguousarray(data[:, DN:]).view("<u2"), 0).astype(np.uint16)
    return nope, np.where(ok[:, None], sc, 0).astype(np.uint8), rope


def quantize_rows(x: np.ndarray, rope: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """f32 NoPE rows (n, DN) as vLLM's cache writer and FlashMLA's quant.py quantize them: per 64-element tile the UE8M0 scale
    2^ceil(log2(max(amax / 448, 1e-4))) and the e4m3 bytes of x / scale; RoPE (n, DR) f32 as bf16 words."""
    t = np.asarray(x, dtype=np.float32).reshape(x.shape[0], DN // QT, QT)
    inv = np.maximum(np.abs(t).max(axis=-1) / np.float32(448.0), np.float32(1e-4)).astype(np.float32)
    e = np.ceil(np.log2(inv)).astype(np.int64)
    s = np.ldexp(np.float32(1.0), e).astype(np.float32)
    nope = np_f32_to_e4m3_sat((t / s[..., None]).astype(np.float32)).reshape(x.shape[0], DN)
    return nope, (e + 127).astype(np.uint8), _bf16(rope)


def random_cache(rng: np.random.Generator, layer: str) -> np.ndarray:
    """A cache of CACHE_BLOCKS[layer] blocks of random rows (row gains log-uniform in [1/4, 4]), its padding random bytes."""
    pbs = SWA_PBS if layer == "swa" else LAYERS[layer][0]
    n = CACHE_BLOCKS[layer] * pbs
    gain = np.exp2(rng.uniform(-2, 2, (n, 1))).astype(np.float32)
    x = (rng.standard_normal((n, DN)) * gain).astype(np.float32)
    r = (rng.standard_normal((n, DR)) * gain).astype(np.float32)
    nope, scale, rope = quantize_rows(x, r)
    fill = rng.integers(0, 256, (n // pbs, page_bytes(pbs)), dtype=np.uint8)
    return pack_cache(nope, scale, rope, pbs, fill)


# ------------------------------------------------------------------------------------------------------------------ the operands
def _slots(rng, layer: str, k: int) -> np.ndarray:
    pbs = SWA_PBS if layer == "swa" else LAYERS[layer][0]
    usable = (CACHE_BLOCKS[layer] - 1) * pbs
    return rng.choice(usable, size=k, replace=k > usable).astype(np.int32)


def case_lens(case: str, quick: bool) -> tuple[tuple[int, int | None], ...]:
    if case == "batch":
        rng = np.random.default_rng(CASE_LAYERS[case][1] + 100)
        b = QUICK_BATCH if quick else BATCH
        swa = np.where(rng.random(b) < 0.75, WINDOW, rng.integers(1, WINDOW + 1, b))
        return tuple((int(s), int(e)) for s, e in zip(swa, rng.integers(0, C4_TOPK + 1, b)))
    return {"swa": SWA_LENS, "c4": C4_LENS, "c128": C128_LENS, "lone": LONE_LENS, "edge": EDGE_LENS}[case]


def case_operands(case: str, quick: bool = False) -> dict[str, Any]:
    """A case's launch operands as numpy words: q (b, 64, D) bf16 (pad heads +0), sink (64,) f32 (pad heads -inf), the SWA cache,
    swa_idx (b, WINDOW) with -1 past each length, swa_lens; on a compressed layer the extra cache, extra_buf (b, buffer width)
    whose first `width` columns are the indices, extra_lens."""
    layer, seed = CASE_LAYERS[case]
    rng = np.random.default_rng(seed)
    lens = case_lens(case, quick)
    b = len(lens)
    head_gain = np.array([0.25, 1.0, 4.0, 16.0] * (NH // 4), dtype=np.float32)[None, :, None]
    q = np.zeros((b, H_Q, D), dtype=np.uint16)
    q[:, :NH] = _bf16(rng.standard_normal((b, NH, D)).astype(np.float32) * head_gain)
    sink = np.full(H_Q, F_NINF, dtype=np.uint32)
    sink[:NH] = (rng.standard_normal(NH) * 2).astype(np.float32).view(np.uint32)
    o: dict[str, Any] = {"case": case, "layer": layer, "q": q, "sink": sink, "swa_cache": random_cache(rng, "swa"),
                         "swa_lens": np.array([s for s, _ in lens], dtype=np.int32)}
    o["swa_idx"] = np.full((b, WINDOW), -1, dtype=np.int32)
    for t, (s, _) in enumerate(lens):
        o["swa_idx"][t, :s] = _slots(rng, "swa", s)
    if layer != "swa":
        pbs, width, buf = LAYERS[layer]
        o["extra_cache"] = random_cache(rng, layer)
        o["extra_lens"] = np.array([e for _, e in lens], dtype=np.int32)
        o["extra_buf"] = rng.integers(-1, 1 << 20, (b, buf), dtype=np.int32)      # past `width`: never read
        o["extra_buf"][:, :width] = -1
        for t, (_, e) in enumerate(lens):
            o["extra_buf"][t, :e] = _slots(rng, layer, e)
    if case == "edge":
        edge(rng, o)
    return o


def _reserved(o: dict[str, Any], cache: str, k: int) -> np.ndarray:
    """k slots of the reserved last block of a cache (rows no other token reads)."""
    pbs = SWA_PBS if cache == "swa_cache" else LAYERS[o["layer"]][0]
    first = (o[cache].shape[0] - 1) * pbs
    return np.arange(first, first + k, dtype=np.int32)


def _write_row(o: dict[str, Any], cache: str, slot: int, nope=None, scale=None, rope=None) -> None:
    pbs = SWA_PBS if cache == "swa_cache" else LAYERS[o["layer"]][0]
    blk, off = divmod(int(slot), pbs)
    page = o[cache][blk]
    if nope is not None:
        page[off * DATA_BYTES:off * DATA_BYTES + DN] = nope
    if rope is not None:
        page[off * DATA_BYTES + DN:(off + 1) * DATA_BYTES] = np.asarray(rope, dtype="<u2").view(np.uint8)
    if scale is not None:
        page[pbs * DATA_BYTES + off * SCALE_BYTES:pbs * DATA_BYTES + off * SCALE_BYTES + DN // QT] = scale


def edge(rng: np.random.Generator, o: dict[str, Any]) -> None:
    """`edge`'s special words, each in rows of the caches' reserved blocks or in its own token: token 0 a window row with UE8M0
    scales 0x00 / 0xFE / 0x01 / 0x7F / 0x80; token 1 an E4M3 NaN byte (0x7F) in a valid window row and 0xFF in a valid compressed
    row; token 2 -1 within both lengths and valid indices past them (the rows the kernel reads up to its 64-row blocks) onto rows
    of scale 0xFF; token 3 heads 0 and 1 at +-bf16 max (overflowing scores); token 4 real heads all +0; token 5 an empty
    request; token 6 one row repeated through its window and bf16 subnormal queries and RoPE; token 7 its window rows all -1
    bar one."""
    o["sink"][:NH] = np.array(EDGE_SINKS, dtype=np.uint32)
    sw, ex = _reserved(o, "swa_cache", 8), _reserved(o, "extra_cache", 2)
    nan_scale = np.full(DN // QT, 0xFF, dtype=np.uint8)
    _write_row(o, "swa_cache", sw[0], scale=np.array([0x00, 0xFE, 0x01, 0x7F, 0x80, 0x00, 0xFE], dtype=np.uint8))
    o["swa_idx"][0, 5] = sw[0]
    nan_row = rng.integers(0, 0x7E, DN, dtype=np.uint8)
    nan_row[17] = 0x7F
    _write_row(o, "swa_cache", sw[1], nope=nan_row)
    o["swa_idx"][1, 100] = sw[1]
    nan_row = nan_row.copy()
    nan_row[17], nan_row[300] = 0, 0xFF
    _write_row(o, "extra_cache", ex[1], nope=nan_row)
    o["extra_buf"][1, 10] = ex[1]
    s2, e2 = EDGE_LENS[2]
    o["swa_idx"][2, 7] = -1
    o["extra_buf"][2, 3] = -1
    _write_row(o, "swa_cache", sw[2], scale=nan_scale)
    _write_row(o, "extra_cache", ex[0], scale=nan_scale)
    o["swa_idx"][2, s2:] = sw[2]
    o["extra_buf"][2, e2:-(-e2 // BN) * BN] = ex[0]
    o["q"][3, 0], o["q"][3, 1] = 0x7F7F, 0xFF7F
    o["q"][4, :NH] = 0
    sub = rng.integers(1, 0x80, (NH, D), dtype=np.uint16) | (rng.integers(0, 2, (NH, D), dtype=np.uint16) << 15)
    o["q"][6, :NH] = sub
    _write_row(o, "swa_cache", sw[3], rope=rng.integers(1, 0x80, DR, dtype=np.uint16))
    o["swa_idx"][6, :EDGE_LENS[6][0]] = sw[3]
    o["swa_idx"][7, :] = -1
    o["swa_idx"][7, 40] = sw[4]


def extra_width(o: dict[str, Any]) -> int:
    return LAYERS[o["layer"]][1] if o["layer"] != "swa" else 0


def token_inputs(a: dict[str, np.ndarray], t: int, metas, num_splits) -> dict[str, Any]:
    """`FlashMlaDecode_v1`'s inputs for token t of a case's saved arrays: its N rows in kernel order (the window's NW, then
    ceil64(extra_len) compressed rows), their bytes and indices, the lengths, and its SPLITS from the planner's output."""
    swa = int(a["swa_lens"][t])
    extra = int(a["extra_lens"][t]) if "extra_lens" in a else None
    nw, n = T.request_blocks(swa, extra)
    idx = np.concatenate([np.asarray(a["swa_idx"][t][:nw]), np.asarray(a["extra_idx"][t][:n - nw]) if n > nw else
                          np.zeros(0, dtype=np.int32)]).astype(np.int32)
    w = gather_rows(a["swa_cache"], idx[:nw], SWA_PBS)
    if n > nw:
        x = gather_rows(a["extra_cache"], idx[nw:], int(a["extra_pbs"][0]))
        w = tuple(np.concatenate([p, r]) for p, r in zip(w, x))
    lens = [np.uint32(swa)] + ([np.uint32(extra)] if n > nw else [])
    return {"q": np.asarray(a["q"][t]), "nope": w[0], "scale": w[1], "rope": w[2], "idx": idx.view(np.uint32), "lens": lens,
            "sink": np.asarray(a["sink"]), "NW": nw, "N": n, "SPLITS": T.request_splits(metas, num_splits, t, n // BN)}


def _metas(a: dict[str, np.ndarray]) -> tuple[list[T.SchedMeta], list[int]]:
    return [T.SchedMeta(*(int(v) for v in m)) for m in np.asarray(a["metas"])], [int(v) for v in np.asarray(a["num_splits"])]


def twin_out(a: dict[str, np.ndarray], t: int, metas, num_splits) -> tuple[dict[str, Any], np.ndarray]:
    x = token_inputs(a, t, metas, num_splits)
    y = T.decode(x["q"], x["nope"], x["scale"], x["rope"], x["idx"], x["lens"], x["sink"], SPLITS=x["SPLITS"], NW=x["NW"], QT=QT,
                 BN=BN, SCALE=SCALE)
    return x, y


def twin_sched(o: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The port's planner output for a case: (parts, 7) metadata words and num_splits (b + 1,)."""
    w = extra_width(o)
    metas, ns = T.sched_meta(o["swa_lens"].tolist(), o["extra_lens"].tolist() if w else None, topk=WINDOW, extra_topk=w,
                             parts=T.num_sm_parts(H_Q))
    return np.array([m.words() for m in metas], dtype=np.int32), np.array(ns, dtype=np.int32)


def case_arrays(o: dict[str, Any], out: np.ndarray, metas: np.ndarray, num_splits: np.ndarray) -> dict[str, np.ndarray]:
    """What a case's comparison jobs read: the operands, the kernel's output words and its planner's metadata."""
    arr = {"q": o["q"], "sink": o["sink"], "swa_cache": o["swa_cache"], "swa_idx": o["swa_idx"], "swa_lens": o["swa_lens"],
           "out": out, "metas": metas, "num_splits": num_splits}
    if o["layer"] != "swa":
        arr.update(extra_cache=o["extra_cache"], extra_idx=o["extra_buf"][:, :extra_width(o)], extra_lens=o["extra_lens"],
                   extra_pbs=np.array([LAYERS[o["layer"]][0]]))
    return arr


def case_jobs(case: str, paths: dict[str, str], num_splits: np.ndarray, quick: bool) -> list[dict[str, Any]]:
    """Twin jobs of JOB_TOKENS tokens over every token; reference jobs on the tokens with the most and the fewest splits (every
    token of `edge`; under --quick only the fewest)."""
    per = np.diff(np.asarray(num_splits))
    b = per.size
    jobs = [{"phase": "decode", "case": case, "kind": "twin", "paths": paths, "tokens": [lo, min(b, lo + JOB_TOKENS)]}
            for lo in range(0, b, JOB_TOKENS)]
    refs = {int(np.argmin(per))} if quick else ({int(np.argmax(per)), int(np.argmin(per))} if case != "edge" else set(range(b)))
    return jobs + [{"phase": "decode", "case": case, "kind": "ref", "paths": paths, "token": t} for t in sorted(refs)]


# ----------------------------------------------------------------------------------------------------------------- the GPU side
def selection() -> dict[str, Any]:
    """vLLM's DSV4 attention class on this device, its padded heads, FlashMLA sparse support and the vendored interface."""
    from vllm.config import get_current_vllm_config
    from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls
    from vllm.v1.attention.ops import flashmla as VF

    import vllm.third_party.flashmla.flash_mla_interface as FI

    cls = _select_dsv4_attn_cls(get_current_vllm_config())
    sup = VF.is_flashmla_sparse_supported()
    sha = hashlib.sha256(Path(FI.__file__).read_bytes()).hexdigest()
    rec = {"attn_cls": cls.__name__, "padded_heads": int(DeepseekV4FlashMLAAttention.get_padded_num_q_heads(NH)),
           "sparse_supported": list(sup), "interface": FI.__file__, "interface_sha256": sha,
           "decode_fn": f"{VF.flash_mla_with_kvcache.__module__}.{VF.flash_mla_with_kvcache.__name__}"}
    rec["checks"] = {"attn_cls": cls is DeepseekV4FlashMLAAttention, "padded_heads": rec["padded_heads"] == H_Q,
                     "sparse_supported": bool(sup[0]), "interface": sha == INTERFACE_SHA256,
                     "decode_fn": rec["decode_fn"] == "vllm.third_party.flashmla.flash_mla_interface.flash_mla_with_kvcache"}
    rec["ok"] = all(rec["checks"].values())
    return rec


def stack() -> dict[str, Any]:
    """What the kernels of record came from: torch, vLLM, the `_flashmla_C` library and its SASS architectures."""
    import torch
    import vllm

    import vllm._flashmla_C as FC

    rec: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda, "vllm": getattr(vllm, "__version__", None),
                           "flashmla_C": getattr(FC, "__file__", None)}
    try:
        rec["archs"] = C._archs(rec["flashmla_C"])
    except Exception as e:  # noqa: BLE001
        rec["archs"] = {"error": _error(e)}
    return rec


def launch_case(work: Path, case: str, quick: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One case's launches as `_forward_decode` makes them; its launch record and comparison jobs."""
    import torch
    from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata

    o = case_operands(case, quick)
    b, w = o["q"].shape[0], extra_width(o)
    dev = "cuda"

    def cache(a: np.ndarray, pbs: int):
        t = torch.from_numpy(a).to(dev)
        return t[:, :pbs * TOKEN_BYTES].view(a.shape[0], pbs, TOKEN_BYTES).unsqueeze(-2)

    q = torch.from_numpy(o["q"].view(np.int16)).to(dev).view(torch.bfloat16)
    sink = torch.from_numpy(o["sink"].view(np.float32)).to(dev)
    swa_k = cache(o["swa_cache"], SWA_PBS)
    swa_idx = torch.from_numpy(o["swa_idx"]).to(dev).view(b, 1, WINDOW)
    swa_lens = torch.from_numpy(o["swa_lens"]).to(dev)
    extra = {}
    if w:
        buf = torch.from_numpy(o["extra_buf"]).to(dev)
        extra = {"extra_k_cache": cache(o["extra_cache"], LAYERS[o["layer"]][0]),
                 "extra_indices_in_kvcache": buf[:b, :w].view(b, 1, -1),
                 "extra_topk_length": torch.from_numpy(o["extra_lens"]).to(dev)}

    def run(meta, sentinel: int):
        out = torch.full((b, H_Q, D), sentinel - (1 << 16) if sentinel >= 1 << 15 else sentinel, dtype=torch.int16,
                         device=dev).view(torch.bfloat16)
        ret, lse = flash_mla_with_kvcache(q=q.unsqueeze(1), k_cache=swa_k, block_table=None, head_dim_v=512,
                                          tile_scheduler_metadata=meta, cache_seqlens=None, is_fp8_kvcache=True, indices=swa_idx,
                                          topk_length=swa_lens, softmax_scale=SCALE, attn_sink=sink, out=out.unsqueeze(1), **extra)
        return out, ret, lse

    meta, _ = get_mla_metadata()
    out1, ret1, lse = run(meta, SENTINELS[0])
    out2, _, _ = run(meta, SENTINELS[1])
    meta2, _ = get_mla_metadata()
    out3, _, _ = run(meta2, SENTINELS[0])
    torch.cuda.synchronize()
    tsm, ns = meta.tile_scheduler_metadata.cpu().numpy(), meta.num_splits.cpu().numpy()
    want_m, want_ns = twin_sched(o)
    parts = T.num_sm_parts(H_Q)
    rec = {"phase": "decode", "case": case, "layer": o["layer"], "b": b, "extra_width": w, "num_splits": ns.tolist(),
           "lse_shape": list(lse.shape),
           "checks": {"shape": list(out1.shape) == [b, H_Q, D] and tuple(tsm.shape) == (parts, 8) and tuple(ns.shape) == (b + 1,),
                      "sched": bool(np.array_equal(tsm[:, :7], want_m) and np.array_equal(ns, want_ns)),
                      "rerun": bool(torch.equal(out1.view(torch.int16), out2.view(torch.int16))),
                      "planner": bool(torch.equal(out1.view(torch.int16), out3.view(torch.int16))
                                      and torch.equal(meta.tile_scheduler_metadata[:, :7], meta2.tile_scheduler_metadata[:, :7])
                                      and torch.equal(meta.num_splits, meta2.num_splits)),
                      "returned": bool(torch.equal(ret1.reshape(b, H_Q, D).view(torch.int16), out1.view(torch.int16)))}}
    if not rec["checks"]["sched"]:
        bad = np.flatnonzero(np.any(tsm[:, :7] != want_m, axis=1)) if tsm.shape[0] == want_m.shape[0] else []
        rec["sched_first_bad"] = {"part": int(bad[0]), "gpu": tsm[bad[0], :7].tolist(), "port": want_m[bad[0]].tolist()} \
            if len(bad) else {"gpu_num_splits": ns.tolist()[:40], "port_num_splits": want_ns.tolist()[:40]}
    paths = _save(work / case, **case_arrays(o, _u16(out1), tsm[:, :7].astype(np.int32), ns.astype(np.int32)))
    return rec, case_jobs(case, paths, ns, quick)


# ------------------------------------------------------------------------------------------------------------------ comparisons
def compare_twin(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Tokens [t0, t1): every output word of all 64 heads against `flashmla_twin.decode` at the token's SPLITS."""
    a = _load(job["paths"])
    metas, ns = _metas(a)
    for t in range(*job["tokens"]):
        x, y = twin_out(a, t, metas, ns)
        got = np.asarray(a["out"][t], dtype=np.uint16)
        bad = np.argwhere(y != got)
        rec["words"] += int(got.size)
        rec["mismatch"] += int(bad.shape[0])
        if bad.shape[0] and rec["first_mismatch"] is None:
            h, i = (int(v) for v in bad[0])
            rec["first_mismatch"] = {"token": t, "head": h, "word": i, "want": int(y[h, i]), "kernel": int(got[h, i]),
                                     "splits": list(x["SPLITS"]), "N": x["N"]}


def compare_ref(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """One token's real heads: `FlashMlaDecode_v1{NH=16,...}` (the reference) against the twin."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    a = _load(job["paths"])
    metas, ns = _metas(a)
    t = job["token"]
    x, y = twin_out(a, t, metas, ns)
    fn = bind(F.FlashMlaDecode, SPLITS=tuple(x["SPLITS"]), NW=x["NW"], **F.FLASHMLA_DSV4)
    r = evaluate(fn, x["q"][:NH].reshape(-1).tolist(), x["nope"].reshape(-1).tolist(), x["scale"].reshape(-1).tolist(),
                 x["rope"].reshape(-1).tolist(), x["idx"].tolist(), *[int(v) for v in x["lens"]], x["sink"][:NH].tolist())
    r = np.array(r, dtype=np.int64)
    bad = np.flatnonzero(r != y[:NH].reshape(-1))
    rec["reference"] += 1
    rec["twin_mismatch"] += int(bad.size)
    if bad.size and rec["first_mismatch"] is None:
        h, i = divmod(int(bad[0]), D)
        rec["first_mismatch"] = {"token": t, "head": h, "word": i, "reference": int(r[bad[0]]), "twin": int(y[h, i])}


COMPARE = {"twin": compare_twin, "ref": compare_ref}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None}
    try:
        COMPARE[job["kind"]](job, rec)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "error": _error(e)}
    return {"phase": job["phase"], "case": job["case"], **rec}


def bind_check() -> dict[str, Any]:
    """The Programs bind at DeepSeek-V4-Pro's statics: a window-only token (two blocks, one split), a full compressed token cut in
    two, the lone token's split, combine and head shapes, the KV row."""
    from verity.ml.circuits.defs import bind

    st = {k: v for k, v in F.FLASHMLA_DSV4.items() if k != "NH"}
    blk = {"D": D, "BN": BN, "DOT": F.FLASHMLA_DSV4["DOT"], "SCALE": SCALE}
    targets = {
        "FlashMlaKvRow_v1{DN=448,DR=64,QT=64}": lambda: bind(F.FlashMlaKvRow, DN=DN, DR=DR, QT=QT),
        "FlashMlaSplit_v1{NB=2,PARTIAL=False,...}": lambda: bind(F.FlashMlaSplit, NB=2, PARTIAL=False, **blk),
        "FlashMlaSplit_v1{NB=1,PARTIAL=True,...}": lambda: bind(F.FlashMlaSplit, NB=1, PARTIAL=True, **blk),
        "FlashMlaCombine_v1{NS=18,D=512}": lambda: bind(F.FlashMlaCombine, NS=18, D=D),
        "FlashMlaDecodeHead_v1{SPLITS=(0,9,18),...}": lambda: bind(F.FlashMlaDecodeHead, SPLITS=(0, 9, 18), **blk),
        "FlashMlaDecode_v1{NH=16,SPLITS=(0,2),NW=128,...}": lambda: bind(F.FlashMlaDecode, NH=NH, SPLITS=(0, 2), NW=128, **st),
        "FlashMlaDecode_v1{NH=16,SPLITS=(0,9,18),NW=128,...}": lambda: bind(F.FlashMlaDecode, NH=NH, SPLITS=(0, 9, 18), NW=128,
                                                                            **st),
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
    """The selection, the stack and every case's launches in order; each error goes to doc["errors"].  Returns the jobs."""
    import torch

    jobs: list[dict[str, Any]] = []

    def step(name: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            print(f"[dsv4-flashmla] {name} ERROR {doc['errors'][name][:600]}", flush=True)
            return None

    doc["selection"] = step("selection", selection) or {"ok": False}
    print(f"[dsv4-flashmla] selection {json.dumps(doc['selection'])[:1500]}", flush=True)
    doc["stack"] = step("stack", stack) or {}
    for _, case in CASES:
        r = step(f"decode/{case}", lambda case=case: launch_case(work, case, quick))
        if r is not None:
            doc["launches"].append(r[0])
            jobs += r[1]
            print(f"[dsv4-flashmla] decode {case:6s} b={r[0]['b']} {json.dumps(r[0]['checks'])} "
                  f"splits={np.diff(r[0]['num_splits']).tolist()[:24]}", flush=True)
    torch.cuda.empty_cache()
    return jobs


def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["checks"]) and all(v is True for v in x["checks"].values())


def _words_ok(t: dict[str, Any]) -> bool:
    return t["words"] > 0 and t["mismatch"] == 0 and t["reference"] > 0 and t["twin_mismatch"] == 0


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's verdict: its launch checks, its words and its errors, so a NO-GO names what failed."""
    out: dict[str, dict[str, Any]] = {}
    for phase, case in CASES:
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
                and all(c["ok"] for c in cases(doc).values()) and _words_ok(doc["totals"]))


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    import torch  # noqa: F401
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "quick": quick, "environment": G._environment(),
                           "gate": C.gate(min_driver, want_cc, want_sms), "errors": {}, "launches": []}
    print(f"[dsv4-flashmla] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-FLASHMLA GATE-FAIL", flush=True)
        raise SystemExit(3)
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        jobs = gpu(work, outdir, quick, doc)
    doc.setdefault("selection", {"ok": False})
    try:
        doc["bind"] = bind_check()
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    print(f"[dsv4-flashmla] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
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
        print(f"[dsv4-flashmla] words {r['phase']} {r['case']:6s} words={r['words']} !={r['mismatch']} ref={r['reference']} "
              f"twin!={r['twin_mismatch']} first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[dsv4-flashmla] case {key:14s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} "
              f"!={t['mismatch']} ref={t['reference']} twin!={t['twin_mismatch']} errors={c['errors']}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"DSV4-FLASHMLA {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro's FlashMLA sparse decode on this Hopper GPU against its Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (the cases' operands and outputs)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true",
                    help=f"a batch of {QUICK_BATCH} and the reference on one token per case: a first signal in minutes")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
