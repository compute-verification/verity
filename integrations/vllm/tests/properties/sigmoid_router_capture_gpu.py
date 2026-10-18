"""GLM-5.3's router on a GPU: vLLM's fused sigmoid `noaux_tc` kernel (`_moe_C.grouped_topk` with scoring_func 1, which
`fused_grouped_topk` launches for `GroupedTopKRouter`) at one group, top-8 of 256 experts, renormalized, x2.5, on f32 logits and an
f32 correction bias, against `registry.moe.MoeRouterSigmoidBias_v1{E=256,TOPK=8,SCALE=2.5,KERNEL}`.

One GPU process writes `<out>/sigmoid_router_capture.json`, the property record `sigmoid_router_capture` (`properties.record`), and
`<out>/rows.npz`: every captured row's logits, each launch group's bias and both launches' outputs.  `ok` = every check below.  The
device gate runs first and nothing else runs when it fails (exit 3):

  gate     `fa2_target_capture_gpu.gate`: driver, compute capability and SM count.
  binary   the SASS and PTX archs vLLM's `_moe_C` carries (informational).
  mufu     `fa2_target_capture_gpu.mufu`: MUFU ex2 / rcp, which the sigmoid reads, are core's tables on all 2^32 inputs.
  launch   the kernel each launch ran (torch.profiler): `single_group_topk_block_kernel<float, float, int, SIGMOID, 256, 8>` for at most
           1024 tokens, `single_group_topk_warp_kernel<...>` for more (`grouped_topk_kernels.cu` `launch`).
  rows     `rows(seed, n)`: groups of 64 rows sharing one bias (the kernel reads one bias per expert for a whole launch), drawn to reach
           the router's edges: ties in the score and in score + bias, a bias that decides, score + bias == 0, the sigmoid's polynomial,
           MUFU and saturated ranges and their boundaries, subnormal, huge and non-finite logits and biases, fewer finite experts
           than slots and none.  Each group runs as one block launch and, tiled past 1024 tokens, as one warp launch, twice each.
           Every output against `twin` (the Program restated on numpy arrays) and a sample against the Program itself.
  sweep    every logit word whose sigmoid reads MUFU (|x| in [0x3F99999A, 0x41902CB4), both signs, 65.9M words) through both launches,
           eight per row with the rest NaN, bias 0, not renormalized, x1: the kernel's weights are the sigmoids themselves.

`--check DIR` evaluates the Program on every row of a capture's `rows.npz` through both launches (no GPU; about 0.33 core-seconds a
row and launch) and writes `<DIR>/program_check.json`.  `--verify RECORD` re-derives a record's digest and verdict.

It lives under tests/ because it seeds random generators and sets VLLM_BATCH_INVARIANT for its own process (lints P3, P7).

    cd integrations/vllm; PY=/workspace/jobs/venv312/bin/python      # PYTHONPATH as for the other captures
    $PY tests/properties/sigmoid_router_capture_gpu.py --out DIR --want-cc 12.0 --want-sms 188    # RTX PRO 6000 Blackwell (sm_120)
    $PY tests/properties/sigmoid_router_capture_gpu.py --out DIR --want-cc 9.0 --want-sms 132     # H200 (sm_90)
    $PY tests/properties/sigmoid_router_capture_gpu.py --check DIR --workers 4
"""
from __future__ import annotations

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
from typing import Any

import numpy as np
from verity.ml.circuits.ops import mufu as MU
from verity_vllm.program.kernels import derived_rows as DR

SCHEMA = "verity-vllm/sigmoid-router-capture/v1"
RECORD = "sigmoid_router_capture.json"
ROWS_FILE = "rows.npz"
CHECK_FILE = "program_check.json"
E, TOPK, SCALE = 256, 8, 2.5
GROUP = 64                    # rows per launch group, which share one bias
BLOCK_TOKENS = 1024           # `launch`: at 256 experts, at most 1024 tokens run the block kernel and more the warp kernel
LAUNCHES = ("block", "warp")
KERNEL_NAMES = {"block": "single_group_topk_block_kernel", "warp": "single_group_topk_warp_kernel"}
MUFU_LO, MUFU_HI = 0x3F99999A, 0x41902CB4    # |x| in [MUFU_LO, MUFU_HI): the sigmoid reads MUFU.EX2 and MUFU.RCP
SWEEP_ROWS = 1 << 19

U32 = np.uint32
ONE, HALF, SIGN = 0x3F800000, 0x3F000000, 0x80000000
NEG_INF, PINF, QNAN = 0xFF800000, 0x7F800000, 0x7FC00000
NONFINITE = np.array([QNAN, 0xFFC00000, 0x7F800001, 0xFFFFFFFF, PINF, NEG_INF], U32)
EXTREMES = np.array([0, SIGN, 1, SIGN | 1, 0x007FFFFF, 0x00800000, SIGN | 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF], U32)
BIAS_KINDS = ("zero", "small", "decisive", "ties", "cancel", "huge", "nonfinite", "mostly-nonfinite", "bits")
ROW_KINDS = ("gauss", "near", "ties", "cancel", "edges", "extremes", "nonfinite", "bits")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _w(v) -> np.ndarray:
    return np.asarray(v, np.float32).view(U32)


def _add(a, b) -> np.ndarray:
    return DR.fma_rn(a, U32(ONE), b)


def _mul(a, b) -> np.ndarray:
    return DR.fma_rn(a, b, U32(SIGN))


def _finite(w) -> np.ndarray:
    return (np.asarray(w, U32) & U32(PINF)) != PINF


# ----------------------------------------------------------------------------------------------------------------------- twin
def sigmoid_words(x) -> np.ndarray:
    """`moe_twins.nv_sigmoid_bits` (`NvSigmoid_v1`, the SASS of the kernel's sigmoid) on an array of f32 words, in integer arithmetic."""
    x = np.asarray(x, U32)
    h = _mul(x, U32(HALF))
    ah = h & U32(0x7FFFFFFF)
    nan = ah > PINF
    poly = (ah < 0x3F19999A) | nan
    big = (ah >= 0x41102CB4) & ~nan
    a = _mul(ah, U32(0x4038AA3B))
    s = _mul(h, h)
    p = DR.fma_rn(s, U32(0x3C80F082), U32(0xBD563CAE))
    e = MU.mufu_ex2_words(a, MU.table("ex2"))
    p = DR.fma_rn(s, p, U32(0x3E085941))
    d = _add(e, U32(ONE))
    p = DR.fma_rn(s, p, U32(0xBEAAA9ED))
    q = DR.fma_rn(s, p, U32(0))
    r = MU.mufu_rcp_words(d, MU.table("rcp"))
    t = np.where(big, U32(ONE), DR.fma_rn(r, U32(0xC0000000), U32(ONE))) | (h & U32(SIGN))
    t = np.where(poly, DR.fma_rn(h, q, h), t)
    return DR.fma_rn(t, U32(HALF), U32(HALF))


def _keys(sel: np.ndarray, expert: np.ndarray) -> np.ndarray:
    """The kernel's 64-bit keys `(TwiddleIn(score) << 32) | (65535 - expert)`: distinct per row, larger is better."""
    tw = np.where(sel >> 31 != 0, ~sel, sel | U32(SIGN)).astype(np.uint64)
    return (tw << np.uint64(32)) | (np.uint64(65535) - expert.astype(np.uint64))


def twin(x, b, launch: str, renormalize: bool = True, scale: float = SCALE) -> tuple[np.ndarray, np.ndarray]:
    """`single_group_topk_{launch}_kernel<float, float, int, SIGMOID, 256, 8>` on rows x (n, E) under bias b (E,) or (n, E): weights
    (n, 8) f32 words and ids (n, 8).  The keys are distinct, so the kernel's warp reductions return the 8 largest in order; the rest is
    `MoeRouterSigmoidBias_v1` term by term: `preprocess_score`, the unbiased score (block) or score - bias (warp) of each slot, the
    32-lane butterfly sum, `scale / (sum + 1e-20)` and one product per slot, 1/8 each when no expert is finite."""
    x = np.asarray(x, U32)
    n, e = x.shape
    b = np.broadcast_to(np.asarray(b, U32), x.shape)
    fin = _finite(x) & _finite(b)
    sig = sigmoid_words(x)
    biased = _add(sig, b)
    sel = np.where(fin, np.where((biased & U32(0x7FFFFFFF)) == 0, U32(0), biased), U32(NEG_INF)).astype(U32)
    ids = np.argsort(~_keys(sel, np.arange(e)), axis=1, kind="stable")[:, :TOPK]
    s = np.take_along_axis(sel, ids, 1)
    ok = s != NEG_INF
    if launch == "block":
        u = np.take_along_axis(np.where(fin, sig, U32(0)), ids, 1)
    else:
        u = _add(s, np.take_along_axis(b, ids, 1) ^ U32(SIGN))
    u = np.where(ok, u, U32(0)).astype(U32)
    lanes = np.zeros((n, 32), U32)
    lanes[:, :TOPK] = u
    for m in (16, 8, 4, 2, 1):
        lanes = _add(lanes, lanes[:, np.arange(32) ^ m])
    q = DR.div_rn(_w(scale), _add(lanes[:, :1], _w(1e-20))) if renormalize else _w(scale)
    w = np.where(ok, _mul(u, q), U32(0))
    w = np.where(ok.any(axis=1, keepdims=True), w, _w(1.0 / TOPK) if renormalize else U32(0))
    return w.astype(U32), ids.astype(np.int32)


# ----------------------------------------------------------------------------------------------------------------------- rows
def _bias(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    """A group's bias words, and for `cancel` the logits whose sigmoid each expert's bias cancels exactly (score + bias == +0)."""
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
        b = sigmoid_words(c) ^ U32(SIGN)
    elif kind == "huge":
        b = _w(rng.standard_normal(E) * 1e-2)
        b[rng.choice(E, 16, replace=False)] = _w(rng.choice([1e30, -1e30, 3.4028235e38, -3.4028235e38, 1e7, -1e7, 3e7, -3e7,
                                                            16777216.0, 33554432.0], 16))
    elif kind == "nonfinite":
        b = _w(rng.standard_normal(E) * 0.1)
        i = rng.choice(E, int(rng.integers(1, 41)), replace=False)
        b[i] = rng.choice(NONFINITE, len(i))
    elif kind == "mostly-nonfinite":
        b = np.full(E, PINF, U32)
        i = rng.choice(E, int(rng.integers(0, 10)), replace=False)
        b[i] = _w(rng.standard_normal(len(i)))
    else:
        b = rng.integers(0, 1 << 32, E, dtype=np.uint64).astype(U32)
    return b.astype(U32), c


def _ulps(w: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.clip(w.astype(np.int64) + k, 0, 0xFFFFFFFF).astype(U32)


def _row(kind: str, rng: np.random.Generator, c: np.ndarray | None) -> np.ndarray:
    if kind == "gauss":
        return _w(rng.standard_normal(E) * rng.choice([0.5, 2.0, 4.0, 8.0]))
    if kind == "near":                                             # within a few ulps of one value: near-equal and equal sigmoids
        v = [rng.standard_normal() * 3.0, rng.uniform(9.0, 20.0) * rng.choice([-1.0, 1.0]), rng.standard_normal() * 1e-3]
        return _ulps(np.full(E, _w(v[int(rng.integers(3))])), rng.integers(-6, 7, E))
    if kind == "ties":
        return rng.choice(_w(rng.standard_normal(int(rng.integers(1, 6))) * 3.0), E)
    if kind == "cancel":            # the group's cancelled logits, some moved by an ulp or two (half the rows only down: zeros lead)
        x = (_w(rng.standard_normal(E) * 3.0) if c is None else c).copy()
        i = rng.choice(E, int(rng.integers(0, E // 2)), replace=False)
        k = rng.choice([-2, -1, 1, 2], len(i))
        if rng.integers(2):
            k = -np.abs(k) * np.where(x[i] >> 31 != 0, -1, 1)
        x[i] = _ulps(x[i], k)
        return x
    if kind == "edges":                                            # the sigmoid's range boundaries, both signs, +-8 ulps
        edges = np.array([MUFU_LO, MUFU_HI, 0x3F19999A, 0x41102CB4], U32)
        return _ulps(rng.choice(edges, E), rng.integers(-8, 9, E)) | (rng.integers(0, 2, E).astype(U32) << 31)
    if kind == "extremes":
        more = _w(rng.choice([1e4, -1e4, 1e-30, -1e-30, 88.0, -88.0, 104.0, -104.0, 18.0, -18.0, 1e38, -1e38], E))
        return np.where(rng.integers(0, 2, E) == 0, rng.choice(EXTREMES, E), more).astype(U32)
    if kind == "nonfinite":                                        # 0, 1, 7, 8, 9 or more finite experts
        x = _w(rng.standard_normal(E) * 4.0)
        keep = int(rng.choice([0, 1, 7, 8, 9, int(rng.integers(10, E))]))
        i = rng.choice(E, E - keep, replace=False)
        x[i] = rng.choice(NONFINITE, len(i))
        return x
    return rng.integers(0, 1 << 32, E, dtype=np.uint64).astype(U32)


def rows(seed: int = 2026, n: int = 10_240) -> dict[str, np.ndarray]:
    """`ceil(n / 64)` groups of 64 rows: group g's bias is `BIAS_KINDS[g % 9]`, its row r `ROW_KINDS[r % 8]`.  Returns `x` (rows, E),
    `b` (groups, E) and the kinds' indices."""
    rng = np.random.default_rng(seed)
    xs, bs, bk, rk = [], [], [], []
    for g in range(-(-n // GROUP)):
        b, c = _bias(BIAS_KINDS[g % len(BIAS_KINDS)], rng)
        bs.append(b)
        bk.append(g % len(BIAS_KINDS))
        for r in range(GROUP):
            xs.append(_row(ROW_KINDS[r % len(ROW_KINDS)], rng, c))
            rk.append(r % len(ROW_KINDS))
    return {"x": np.stack(xs).astype(U32), "b": np.stack(bs).astype(U32), "bias_kind": np.array(bk, np.int8),
            "row_kind": np.array(rk, np.int8)}


def row_bias(R: dict[str, np.ndarray]) -> np.ndarray:
    """Each row's bias (rows, E)."""
    return R["b"][np.arange(len(R["x"])) // GROUP]


def digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------- sweep
def sweep_words() -> np.ndarray:
    """Every logit word the sigmoid reads MUFU on, both signs, padded to a multiple of 8 with `MUFU_LO`."""
    pos = np.arange(MUFU_LO, MUFU_HI, dtype=U32)
    w = np.concatenate([pos, pos | U32(SIGN)])
    return np.concatenate([w, np.full(-len(w) % 8, MUFU_LO, U32)])


def sweep_chunk(vals: np.ndarray, row0: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows of NaN logits holding 8 words each, row r's at experts `8 (row0 + r) mod 256 + (0..7)`: (x (R, E), the experts (R, 8))."""
    R = len(vals)
    pos = ((row0 + np.arange(R, dtype=np.int64))[:, None] * 8) % E + np.arange(8)
    x = np.full((R, E), QNAN, U32)
    np.put_along_axis(x, pos, vals, 1)
    return x, pos


def sweep_expect(vals: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`twin` of a sweep chunk (bias 0, not renormalized, x1) from its 8 finite words a row: the sigmoids, largest first."""
    sig = sigmoid_words(vals)
    order = np.argsort(~_keys(sig, pos), axis=1, kind="stable")
    return np.take_along_axis(sig, order, 1), np.take_along_axis(pos, order, 1).astype(np.int32)


# ------------------------------------------------------------------------------------------------------------------------ gpu
def _u32(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int32).cpu().numpy().view(U32)


def _device(a: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(a, U32).view(np.int32)).cuda().view(torch.float32)


def _launch(xd, bd, renormalize: bool, scale: float) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        fused_grouped_topk,
    )

    w, ids = fused_grouped_topk(hidden_states=xd[:, :1], gating_output=xd, topk=TOPK, renormalize=renormalize,
                                e_score_correction_bias=bd, num_expert_group=1, topk_group=1, scoring_func="sigmoid",
                                routed_scaling_factor=scale)
    torch.cuda.synchronize()
    return _u32(w), ids.cpu().numpy().astype(np.int32)


def run_launch(xd, bd, launch: str, renormalize: bool = True, scale: float = SCALE) -> tuple[np.ndarray, np.ndarray, bool]:
    """The rows of `xd` through `launch`: block = launches of at most 1024 tokens; warp = one launch of more than 1024, the rows tiled
    until there are.  (weights, ids, every tile's outputs equal)."""
    M = xd.shape[0]
    if launch == "block":
        parts = [_launch(xd[i:i + BLOCK_TOKENS], bd, renormalize, scale) for i in range(0, M, BLOCK_TOKENS)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), True
    reps = BLOCK_TOKENS // M + 1
    w, ids = _launch(xd.repeat(reps, 1) if reps > 1 else xd, bd, renormalize, scale)
    w, ids = w.reshape(reps, M, TOPK), ids.reshape(reps, M, TOPK)
    return w[0], ids[0], bool((w == w[0]).all() and (ids == ids[0]).all())


def kernel_names(xd, bd, launch: str) -> dict[str, Any]:
    """The device kernels one launch ran (torch.profiler), against the one `launch` dispatches to at 256 experts and top-8."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run_launch(xd, bd, launch)
    names = sorted({e.name for e in prof.events() if e.device_type == DeviceType.CUDA and "topk" in e.name.lower()})
    ok = len(names) == 1 and KERNEL_NAMES[launch] in names[0] and ("256, 8>" in names[0] or "Li256ELi8E" in names[0])
    return {"names": [n[:300] for n in names], "ok": ok}


def capture_rows(R: dict[str, np.ndarray], reps: int = 2) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Every group through both launches `reps` times: the outputs, and whether the launches were deterministic, every warp tile
    agreed and no launch wrote its logits."""
    import torch

    x, b = R["x"], R["b"]
    out: dict[str, list[np.ndarray]] = {f"{lnch}_{k}": [] for lnch in LAUNCHES for k in ("w", "ids")}
    rec: dict[str, Any] = {"groups": len(b), "launches": 0, "deterministic": True, "tiles_equal": True, "logits_intact": True}
    for g in range(len(b)):
        xd, bd = _device(x[g * GROUP:(g + 1) * GROUP]), _device(b[g])
        before = xd.clone()
        for launch in LAUNCHES:
            got = [run_launch(xd, bd, launch) for _ in range(reps)]
            rec["launches"] += reps
            rec["deterministic"] &= all(np.array_equal(got[0][0], o[0]) and np.array_equal(got[0][1], o[1]) for o in got[1:])
            rec["tiles_equal"] &= all(o[2] for o in got)
            out[f"{launch}_w"].append(got[0][0])
            out[f"{launch}_ids"].append(got[0][1])
        rec["logits_intact"] &= bool(torch.equal(xd.view(torch.int32), before.view(torch.int32)))
    return rec, {k: np.concatenate(v) for k, v in out.items()}


def compare_rows(R: dict[str, np.ndarray], got: dict[str, np.ndarray]) -> dict[str, Any]:
    """Every row's outputs against `twin`, by launch, with the first mismatches."""
    rec: dict[str, Any] = {}
    b = row_bias(R)
    for launch in LAUNCHES:
        w, ids = twin(R["x"], b, launch)
        bad = np.flatnonzero((w != got[f"{launch}_w"]).any(1) | (ids != got[f"{launch}_ids"]).any(1))
        rec[launch] = {"rows": len(w), "mismatch": len(bad), "mismatch_by_row_kind": {
            ROW_KINDS[k]: int(np.count_nonzero(R["row_kind"][bad] == k)) for k in range(len(ROW_KINDS))},
            "first": [{"row": int(i), "bias_kind": BIAS_KINDS[R["bias_kind"][i // GROUP]], "row_kind": ROW_KINDS[R["row_kind"][i]],
                       "gpu_w": got[f"{launch}_w"][i].tolist(), "twin_w": w[i].tolist(),
                       "gpu_ids": got[f"{launch}_ids"][i].tolist(), "twin_ids": ids[i].tolist()} for i in bad[:4]]}
    return rec


def sweep(launches=LAUNCHES, chunk_rows: int = SWEEP_ROWS) -> dict[str, Any]:
    """`sweep_words` through each launch against `sweep_expect`: the mismatching words and the first few."""
    words = sweep_words()
    total = len(words) // 8
    bd = _device(np.zeros(E, U32))
    rec: dict[str, Any] = {"words": len(words), "rows": total, "chunk_rows": chunk_rows,
                           **{lnch: {"mismatch_rows": 0, "tiles_equal": True, "first": []} for lnch in launches}}
    for row0 in range(0, total, chunk_rows):
        vals = words[row0 * 8:(row0 + chunk_rows) * 8].reshape(-1, 8)
        x, pos = sweep_chunk(vals, row0)
        ew, eids = sweep_expect(vals, pos)
        xd = _device(x)
        for launch in launches:
            w, ids, same = run_launch(xd, bd, launch, renormalize=False, scale=1.0)
            bad = np.flatnonzero((w != ew).any(1) | (ids != eids).any(1))
            r = rec[launch]
            r["mismatch_rows"] += len(bad)
            r["tiles_equal"] &= same
            for i in bad[:max(0, 4 - len(r["first"]))]:
                r["first"].append({"row": int(row0 + i), "words": vals[i].tolist(), "gpu_w": w[i].tolist(), "twin_w": ew[i].tolist(),
                                   "gpu_ids": ids[i].tolist(), "twin_ids": eids[i].tolist()})
        del xd
        print(f"[sigmoid-router] {_utc()} sweep rows {min(row0 + chunk_rows, total)}/{total} "
              + " ".join(f"{lnch}!={rec[lnch]['mismatch_rows']}" for lnch in launches), flush=True)
    rec["ok"] = all(rec[lnch]["mismatch_rows"] == 0 and rec[lnch]["tiles_equal"] for lnch in launches)
    return rec


# -------------------------------------------------------------------------------------------------------------------- program
_PROGRAMS: dict[str, Any] = {}


def _program(launch: str):
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import moe

    if launch not in _PROGRAMS:
        _PROGRAMS[launch] = bind(moe.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=SCALE, KERNEL=launch)
    return _PROGRAMS[launch]


def program_row(job: tuple[str, np.ndarray, np.ndarray]) -> list[int]:
    """`MoeRouterSigmoidBias_v1{E=256,TOPK=8,SCALE=2.5,KERNEL=launch}` on one row (the reference evaluator): 8 weights, 8 ids."""
    from verity.ml.circuits.evaluation import evaluate

    launch, x, b = job
    return [int(v) for v in evaluate(_program(launch), [int(v) for v in x] + [int(v) for v in b])]


def program_compare(R: dict[str, np.ndarray], got: dict[str, np.ndarray], idx: np.ndarray, workers: int) -> dict[str, Any]:
    """The Program on rows `idx` through both launches, against the captured outputs."""
    b = row_bias(R)
    jobs = [(launch, R["x"][i], b[i]) for launch in LAUNCHES for i in idx]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        outs = list(ex.map(program_row, jobs, chunksize=max(1, len(jobs) // (4 * workers))))
    rec: dict[str, Any] = {"rows": len(idx)}
    for k, launch in enumerate(LAUNCHES):
        mine = outs[k * len(idx):(k + 1) * len(idx)]
        bad = [int(i) for i, o in zip(idx, mine)
               if o != got[f"{launch}_w"][i].tolist() + got[f"{launch}_ids"][i].tolist()]
        rec[launch] = {"mismatch": len(bad), "first": bad[:8]}
    rec["ok"] = all(rec[launch]["mismatch"] == 0 for launch in LAUNCHES)
    return rec


def sample(R: dict[str, np.ndarray], k: int, seed: int) -> np.ndarray:
    """One row of each (bias kind, row kind) pair that exists, then random rows, `k` in all (sorted)."""
    first: dict[tuple[int, int], int] = {}
    for i in range(len(R["x"])):
        first.setdefault((int(R["bias_kind"][i // GROUP]), int(R["row_kind"][i])), i)
    idx = sorted(first.values())[:k]
    rest = np.setdiff1d(np.arange(len(R["x"])), idx)
    extra = np.random.default_rng(seed).choice(rest, max(0, min(k - len(idx), len(rest))), replace=False)
    return np.sort(np.concatenate([np.asarray(idx, np.int64), extra.astype(np.int64)]))


def check(d: str, workers: int, limit: int | None = None) -> dict[str, Any]:
    """`--check`: the Program on every captured row (or the first `limit`) through both launches."""
    z = np.load(Path(d) / ROWS_FILE)
    R = {k: z[k] for k in ("x", "b", "bias_kind", "row_kind")}
    got = {f"{lnch}_{k}": z[f"{lnch}_{k}"] for lnch in LAUNCHES for k in ("w", "ids")}
    n = len(R["x"]) if limit is None else min(limit, len(R["x"]))
    print(f"[sigmoid-router] {_utc()} check: {n} rows x {len(LAUNCHES)} launches on {workers} workers", flush=True)
    rec = {"rows_file_sha256": _sha256_file(Path(d) / ROWS_FILE), "started": _utc(),
           **program_compare(R, got, np.arange(n), workers), "finished": _utc()}
    record = Path(d) / RECORD
    rec["record_digest"] = json.loads(record.read_text()).get("digest") if record.exists() else None
    (Path(d) / CHECK_FILE).write_text(json.dumps(rec, indent=1) + "\n")
    print(f"SIGMOID-ROUTER-PROGRAM-CHECK {'OK' if rec['ok'] else 'FAIL'} rows={n} "
          + " ".join(f"{lnch}!={rec[lnch]['mismatch']}" for lnch in LAUNCHES), flush=True)
    return rec


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------ run
def host_keeps_subnormals() -> bool:
    """The twin's few host float operations (the divide, the f64 -> f32 rounding) need subnormals neither flushed nor read as zero."""
    tiny = np.array([1e-40], np.float32)
    return bool(tiny[0] != 0 and (tiny * np.float32(1.0))[0] != 0 and np.float64(1e-40).astype(np.float32) != 0)


def _verdict(doc: dict[str, Any]) -> bool:
    c, s = doc["capture"], doc.get("sweep") or {}
    return (doc["gate"]["ok"] and doc["host_subnormals"] and doc["mufu"].get("ok") is True
            and all(doc["kernels"][lnch]["ok"] for lnch in LAUNCHES)
            and c["deterministic"] and c["tiles_equal"] and c["logits_intact"]
            and all(doc["rows"][lnch]["mismatch"] == 0 for lnch in LAUNCHES) and doc["program"]["ok"] is True
            and s.get("ok") is True)


def run(out: str, seed: int, n: int, min_driver: int, want_cc: str, want_sms: int, workers: int, program_sample: int,
        do_sweep: bool) -> dict[str, Any]:
    from verity_vllm.engine import engine_profile as prof
    from verity_vllm.properties.record import seal as seal_record

    from tests.properties import fa2_target_capture_gpu as C
    from tests.properties import fa_tap_exactness_gpu as G

    env_flags = prof.apply_env()
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    import torch
    import vllm
    import vllm._custom_ops
    from vllm import envs

    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"schema": SCHEMA, "utc": _utc(), "argv": sys.argv, "environment": G._environment(), "env_flags": env_flags,
                           "fused_grouped_topk": bool(envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK), "host_subnormals": host_keeps_subnormals(),
                           "config": {"E": E, "TOPK": TOPK, "SCALE": SCALE, "n_group": 1, "topk_group": 1, "renormalize": True,
                                      "scoring_func": "sigmoid", "dtype": "float32", "bias_dtype": "float32", "seed": seed, "rows": n},
                           "gate": C.gate(min_driver, want_cc, want_sms)}
    print(f"[sigmoid-router] {_utc()} gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("SIGMOID-ROUTER-CAPTURE GATE-FAIL", flush=True)
        raise SystemExit(3)
    p = torch.cuda.get_device_properties(0)
    moe_so = sorted(Path(vllm.__file__).parent.glob("_moe_C*.so"))
    doc["binary"] = C._archs(str(moe_so[0])) if moe_so else {"error": "no vllm/_moe_C*.so"}
    print(f"[sigmoid-router] {_utc()} binary _moe_C sass={doc['binary'].get('sass_archs')} ptx={doc['binary'].get('ptx_archs')}", flush=True)
    with tempfile.TemporaryDirectory() as work:
        doc["mufu"] = C.mufu(Path(work), f"sm_{p.major}{p.minor}")
    for k, b in doc["mufu"]["builds"].items():
        b.pop("tables_kept", None)
        print(f"[sigmoid-router] {_utc()} mufu {k}: ok={b.get('ok')} ex2!={b.get('ex2_entries_differing')} "
              f"rcp!={b.get('rcp_entries_differing')} {b.get('error', '')}", flush=True)

    R = rows(seed, n)
    doc["inputs"] = {"rows": len(R["x"]), "groups": len(R["b"]), "sha256": digest(R["x"], R["b"])}
    xd0, bd0 = _device(R["x"][:GROUP]), _device(R["b"][0])
    doc["kernels"] = {lnch: kernel_names(xd0, bd0, lnch) for lnch in LAUNCHES}
    for lnch, k in doc["kernels"].items():
        print(f"[sigmoid-router] {_utc()} kernel {lnch}: ok={k['ok']} {k['names']}", flush=True)
    doc["capture"], got = capture_rows(R)
    print(f"[sigmoid-router] {_utc()} captured {json.dumps(doc['capture'])}", flush=True)
    doc["rows"] = compare_rows(R, got)
    for lnch in LAUNCHES:
        r = doc["rows"][lnch]
        print(f"[sigmoid-router] {_utc()} rows {lnch}: {r['rows']} rows, twin!={r['mismatch']} {r['mismatch_by_row_kind']}", flush=True)
        for m in r["first"]:
            print(f"[sigmoid-router]    {json.dumps(m)[:900]}", flush=True)
    np.savez_compressed(outdir / ROWS_FILE, **R, **got)
    doc["rows_file_sha256"] = _sha256_file(outdir / ROWS_FILE)
    idx = sample(R, program_sample, seed + 1)
    doc["program"] = program_compare(R, got, idx, workers)
    print(f"[sigmoid-router] {_utc()} program on {len(idx)} rows: {json.dumps(doc['program'])}", flush=True)
    doc["sweep"] = sweep() if do_sweep else {"skipped": True}
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"SIGMOID-ROUTER-CAPTURE {'OK' if doc['ok'] else 'FAIL'} {doc['digest']} rows={doc['inputs']['rows']} "
          + " ".join(f"{lnch}!={doc['rows'][lnch]['mismatch']}" for lnch in LAUNCHES)
          + f" sweep={doc['sweep'].get('ok')} mufu={doc['mufu'].get('ok')} {outdir / RECORD}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's sigmoid noaux_tc router kernel on this GPU against MoeRouterSigmoidBias_v1")
    ap.add_argument("--out", help=f"directory for {RECORD} and {ROWS_FILE}")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rows", type=int, default=10_240)
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="12.0")
    ap.add_argument("--want-sms", type=int, default=188)
    ap.add_argument("--workers", type=int, default=4, help="processes for the Program's reference evaluation")
    ap.add_argument("--program-sample", type=int, default=96, help="captured rows the GPU run also checks against the Program")
    ap.add_argument("--no-sweep", action="store_true", help="skip the MUFU-range sweep (the record is then not ok)")
    ap.add_argument("--check", metavar="DIR", help=f"evaluate the Program on every row of DIR/{ROWS_FILE} (no GPU)")
    ap.add_argument("--limit", type=int, default=None, help="--check: only the first LIMIT rows")
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
    doc = run(out, a.seed, a.rows, a.min_driver, a.want_cc, a.want_sms, a.workers, a.program_sample, not a.no_sweep)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
