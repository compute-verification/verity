"""Word-for-word checks of the MLA captures (`mla_capture`) against numpy models of the kernels as they compile on sm_120
(Triton 3.7.1, vLLM d9105ea80), the arithmetic `mla.TritonMlaDecode`, `mla.MlaBmm` and `mla.MoeSharedCombine` compute:

    verity-vllm mla-capture check --dir DIR [--out REPORT.json]

`decode_rows` is `_fwd_grouped_kernel_stage1` per (request, head, split) and `_fwd_kernel_stage2`, read from their PTX and SASS:

* scores: one ascending chain of 36 `mma.sync.m16n8k16` steps from the zero accumulator, `q[0:512] . k[0:512]` then
  `q[512:576] . k[512:576]` (the `qpe` dot takes the first as its accumulator), each step the Hopper BF16 pipeline (core's
  `sm120.mma.m16n8k16.bf16` model, `HOPPER_BF16_WGMMA_K16`); then `RN(qk * sm_scale)`; a key past the split's end is -inf;
* per block of 32 keys: `n = max(rowmax(qk), e_max)` (`max.f32`), `re = EXP(e_max - n)`, `p = EXP(qk - n)`, `acc = RN(acc * re)`
  then two steps `acc = DOT(acc, bf16(p[16 s:16 s + 16]), v[16 s:16 s + 16, d])`, `e_sum = fma(e_sum, re, rowsum(p))`;
  `EXP(x)` is `ex2.approx.f32(RN(x * 0x3FB8AA3B))`, which ptxas lowers to `x < -126 ? MUFU.EX2(x * 0.5)^2 : MUFU.EX2(x)`
  (`ex2_approx`); `rowsum` is the warp tree `((x0+x1)+(x4+x5)) + ((x2+x3)+(x6+x7))` over each warp's eight columns, then
  `(w0 + w2) + (w1 + w3)` over the four warps (shared memory at row * 4 + warp, butterflies 2 then 1);
* a split's outputs: `tv = div.full(acc, e_sum)` and `tl = e_max + logf(e_sum)` (libdevice `__nv_logf`);
* stage 2 over the nonempty splits in order: `n = max(tl, e_max)`, `old = EXP(e_max - n)`, `el = EXP(tl - n)`,
  `acc = fma(acc, old, RN(el * tv))`, `e_sum = fma(e_sum, old, el)`; `o = cvt.rn.bf16(div.full(acc, e_sum))` and
  `lse = cvt.rn.bf16(e_max + logf(e_sum))`. One split's unrolled first iteration (`fma(el, tv, old * 0)`) computes the same words.

`bmm_rows` is `bmm_kernel` (BLOCK_K 64, `tl.dot` chained over K): per head `Gemm_v3{K, N, HopperBF16WgmmaDot16}`. `combine_rows` is
ATen's `fused *= rsf; shared + fused` on bf16 tensors: `bf16(f32(fused) * f32(rsf))`, then `bf16(f32(shared) + f32(fused))`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from verity.ml.kernels import tc as CK
from verity.ml.circuits.ops import mufu as MU
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16
from verity_vllm.program.kernels import dense_rows as DRW
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels.libdevice_sampling import nv_logf

U32 = np.uint32
LOG2E = U32(0x3FB8AA3B)
NEG_INF = U32(0xFF800000)
HALF = U32(0x3F000000)
ONE = U32(0x3F800000)
QNAN = U32(0x7FFFFFFF)
BLOCK = 32
DL = 512
HOPPER = "HopperBF16WgmmaDot16_v1"


def f32(u):
    return np.ascontiguousarray(u, dtype=U32).view(np.float32)


def u32(x):
    return np.ascontiguousarray(x, dtype=np.float32).view(U32)


def is_nan(u):
    u = np.asarray(u, dtype=U32)
    return (u & U32(0x7FFFFFFF)) > U32(0x7F800000)


def step(acc, a, b):
    """One Hopper / sm_120 BF16 k16 step on every lane: acc (...), a and b (..., 16) bf16 words (core's total model)."""
    shape = np.shape(acc)
    a, b = np.broadcast_arrays(np.asarray(a, np.uint16), np.asarray(b, np.uint16))
    r, _ = CK.group_sum_total_batch(HOPPER_BF16_WGMMA_K16, np.asarray(acc, U32).reshape(-1), a.reshape(-1, 16), b.reshape(-1, 16))
    return r.reshape(shape)


def dot_chain(x, w):
    """`DotBf16_v3{K, HopperBF16WgmmaDot16}` of every row of x (M, K) against every row of w (N, K): f32 words (M, N). The C++
    chain where it claims the coordinate, core's total step on the rest (a non-finite operand, accumulator or result)."""
    x, w = np.ascontiguousarray(x, np.uint16), np.ascontiguousarray(w, np.uint16)
    out, nonfin = DRW.chain_rows(x, w, HOPPER)
    redo = nonfin | ((out & U32(0x7F800000)) == U32(0x7F800000))
    if redo.any():
        i, j = np.nonzero(redo)
        acc = np.zeros(i.size, U32)
        for s in range(0, x.shape[1], 16):
            acc = step(acc, x[i, s:s + 16], w[j, s:s + 16])
        out[i, j] = acc
    return out


def fmax(a, b):
    """PTX `max.f32`: a NaN operand yields the other, two NaNs 0x7FFFFFFF, `max(+0, -0) = +0`."""
    a, b = np.broadcast_arrays(np.asarray(a, U32), np.asarray(b, U32))
    r = np.where(f32(b) > f32(a), b, a)
    zero = ((a | b) & U32(0x7FFFFFFF)) == 0
    r = np.where(zero, a & b, r)
    na, nb = is_nan(a), is_nan(b)
    return np.where(na & nb, QNAN, np.where(na, b, np.where(nb, a, r))).astype(U32)


def mul(a, b):
    return u32(f32(a) * f32(b))


def add(a, b):
    return u32(f32(a) + f32(b))


def sub(a, b):
    return u32(f32(a) - f32(b))


def fma(a, b, c):
    return DR.fma_rn(a, b, c)


def ex2_approx(t):
    """`ex2.approx.f32` (not .ftz) as ptxas 12.9 emits it on sm_120: `t < -126` (ordered) halves t before MUFU.EX2 and squares
    the result (FMUL, subnormals kept)."""
    t = np.asarray(t, U32)
    small = f32(t) < np.float32(-126.0)
    y = MU.mufu_ex2_words(np.where(small, mul(t, HALF), t), MU.table("ex2"))
    return np.where(small, mul(y, y), y).astype(U32)


def exp_tl(x):
    """Triton's `tl.exp` on f32: `ex2.approx.f32(RN(x * log2(e)))`."""
    return ex2_approx(mul(x, LOG2E))


def div_full(a, b):
    """`div.full.f32`: `RN(DivFullScaleA(a, b) * DivFullRcp(b))`."""
    a, b = np.broadcast_arrays(np.asarray(a, U32), np.asarray(b, U32))
    return mul(MU.div_full_scale_a_words(a, b), MU.mufu_rcp_words(MU.div_full_divisor_words(b), MU.table("rcp")))


def logf(x):
    return u32(nv_logf(f32(x)))


def row_max(qk):
    """The max over the last axis (32 columns) by `max.f32`; its order does not matter."""
    m = qk[..., 0]
    for j in range(1, qk.shape[-1]):
        m = fmax(m, qk[..., j])
    return m


def row_sum(p):
    """The block's row sum as the kernel's reduction tree computes it (`tt.reduce` over an mma layout, warps [1, 4])."""
    x = f32(p)
    warps = []
    for w in range(4):
        c = [x[..., 8 * w + j] for j in range(8)]
        warps.append(((c[0] + c[1]) + (c[4] + c[5])) + ((c[2] + c[3]) + (c[6] + c[7])))
    return u32((warps[0] + warps[2]) + (warps[1] + warps[3]))


def bf16_of(u):
    return DR.f2fp_bf16(u).astype(np.uint16)


def split_ranges(T: int, S: int) -> list[tuple[int, int]]:
    """The kernel's split bounds: `cdiv(T, S)` keys each, the last clipped; empty splits are skipped by both stages."""
    per = -(-T // S)
    return [(s * per, min(s * per + per, T)) for s in range(S)]


def stage1(qk: np.ndarray, kv: np.ndarray, lo: int, hi: int, DL: int = DL) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One split of one request for every head: qk (H, T) scaled scores, kv (T, DL + DR) -> (tv (H, DL), e_max (H,),
    e_sum (H,))."""
    H = qk.shape[0]
    e_max = np.full(H, NEG_INF, U32)
    e_sum = np.zeros(H, U32)
    acc = np.zeros((H, DL), U32)
    for n0 in range(lo, hi, BLOCK):
        cols = np.arange(n0, n0 + BLOCK)
        valid = cols < hi
        blk = np.where(valid[None, :], qk[:, np.minimum(cols, hi - 1)], NEG_INF)
        v = np.where(valid[:, None], kv[np.minimum(cols, hi - 1), :DL], np.uint16(0))        # (32, DL)
        n = fmax(row_max(blk), e_max)
        re = exp_tl(sub(e_max, n))
        p = exp_tl(sub(blk, n[:, None]))
        acc = mul(acc, re[:, None])
        pb = bf16_of(p)                                                                     # (H, 32)
        vt = np.ascontiguousarray(v.T)                                                      # (512, 32)
        for s in range(0, BLOCK, 16):
            acc = step(acc, pb[:, None, s:s + 16], vt[None, :, s:s + 16])
        e_sum = fma(e_sum, re, row_sum(p))
        e_max = n
    return div_full(acc, e_sum[:, None]), e_max, e_sum


def decode_rows(q: np.ndarray, kv: np.ndarray, splits: int, scale: int = 0x3D800000, DL: int = DL) -> dict:
    """One request: q (H, DL + DR) and kv (T, DL + DR) bf16 words -> o (H, DL) and lse (H,) bf16 words, logits (H, S, DL + 1)
    f32 words (zero for an empty split, which the kernel leaves unwritten)."""
    H, T = q.shape[0], kv.shape[0]
    qk = mul(dot_chain(q, kv), U32(scale))
    logits = np.zeros((H, splits, DL + 1), U32)
    present = np.zeros(splits, bool)
    e_max = np.full(H, NEG_INF, U32)
    e_sum = np.zeros(H, U32)
    acc = np.zeros((H, DL), U32)
    for s, (lo, hi) in enumerate(split_ranges(T, splits)):
        if hi <= lo:
            continue
        tv, m, l = stage1(qk, kv, lo, hi, DL)
        tl = add(m, logf(l))
        logits[:, s, :DL], logits[:, s, DL] = tv, tl
        present[s] = True
        n = fmax(tl, e_max)
        old = exp_tl(sub(e_max, n))
        el = exp_tl(sub(tl, n))
        acc = fma(acc, old[:, None], mul(el[:, None], tv))
        e_sum = fma(e_sum, old, el)
        e_max = n
    o = bf16_of(div_full(acc, e_sum[:, None]))
    lse = bf16_of(add(e_max, logf(e_sum)))
    return {"o": o, "lse": lse, "logits": logits, "present": present}


def bmm_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`bmm_kernel`: a (NH, M, K), b (NH, K, N) bf16 words -> (NH, M, N), per head `Gemm_v3{K, N, HopperBF16WgmmaDot16}`."""
    return np.stack([bf16_of(dot_chain(a[h], b[h].T)) for h in range(a.shape[0])])


def combine_rows(routed: np.ndarray, shared: np.ndarray, rsf: float, nan: int) -> np.ndarray:
    """ATen on CUDA bf16: `fused = bf16(f32(routed) * f32(rsf))`, then `bf16(f32(shared) + f32(fused))`; every NaN `nan`."""
    def bf(u):
        r = DR.f2fp_bf16(u).astype(np.uint16)
        return np.where(is_nan(u), np.uint16(nan), r)
    fused = bf(mul(DR.bf16_to_f32_words(routed), u32(np.float32(rsf))))
    return bf(add(DR.bf16_to_f32_words(shared), DR.bf16_to_f32_words(fused)))


# --- comparisons -------------------------------------------------------------------------------------------------------------

def _first(got: np.ndarray, want: np.ndarray, bad: np.ndarray, n: int = 4) -> list:
    return [{"at": [int(i) for i in idx], "got": int(got[idx]), "model": int(want[idx])} for idx in zip(*np.nonzero(bad))][:n]


def _words_equal(got, want, f32_words: bool = False):
    """Equal words, NaNs equal by NaN-ness for f32 words (their payloads are not part of any output)."""
    if f32_words:
        return (got == want) | (is_nan(got) & is_nan(want))
    return got == want


def check_decode(case: dict, out: dict, inputs) -> dict:
    q, kvs = inputs
    S = int(case["splits"])
    rows = bad_rows = words_bad = logits_bad = lse_bad = 0
    first: list = []
    for b, kv in enumerate(kvs):
        m = decode_rows(q[b], kv, S)
        ok_o = _words_equal(out["o"][b], m["o"])
        lg = out["logits"][b][:, m["present"], :]
        ok_l = _words_equal(lg, m["logits"][:, m["present"], :], f32_words=True)
        ok_lse = out["lse"][b] == m["lse"]
        rows += q.shape[1]
        bad_rows += int((~ok_o.all(axis=1)).sum())
        words_bad += int((~ok_o).sum())
        logits_bad += int((~ok_l).sum())
        lse_bad += int((~ok_lse).sum())
        if not ok_o.all() and len(first) < 4:
            first += [{"request": b, **f} for f in _first(out["o"][b], m["o"], ~ok_o)]
        if not ok_l.all() and len(first) < 4:
            first += [{"request": b, "logits": True, **f} for f in _first(lg, m["logits"][:, m["present"], :], ~ok_l)]
    return {"rows": rows, "mismatched_rows": bad_rows, "o_words_bad": words_bad, "logits_words_bad": logits_bad,
            "lse_bad": lse_bad, "first": first[:6]}


def check_bmm(case: dict, out: dict, inputs) -> dict:
    a, b = inputs
    m = bmm_rows(a, b)
    bad = out["c"] != m
    return {"rows": int(a.shape[0] * a.shape[1]), "o_words_bad": int(bad.sum()),
            "mismatched_rows": int(bad.any(axis=2).sum()), "first": _first(out["c"], m, bad)}


def check_combine(case: dict, out: dict, inputs) -> dict:
    """ATen's bf16 combine with NaN 0x7FFF (F2FP), the word sm_120 writes; 0x7FC0's count is reported beside it."""
    routed, shared = inputs
    m = combine_rows(routed, shared, case["rsf"], 0x7FFF)
    bad = out["out"] != m
    other = int((out["out"] != combine_rows(routed, shared, case["rsf"], 0x7FC0)).any(axis=1).sum())
    return {"rows": int(routed.shape[0]), "words_bad": int(bad.sum()), "mismatched_rows": int(bad.any(axis=1).sum()),
            "first": _first(out["out"], m, bad), "rows_bad_at_nan_0x7fc0": other}


def check_mufu(d: Path) -> dict:
    """The 2^20-word sweep of `ex2.approx.f32` (as ptxas lowers it) and `ex2.approx.ftz.f32` against the models."""
    x = np.load(d / "mufu_sweep_in.npz")["x"].astype(U32)
    rep = {}
    for name, model in (("ex2", ex2_approx), ("ex2_ftz", lambda t: MU.mufu_ex2_words(t, MU.table("ex2")))):
        y = np.load(d / f"mufu_sweep_{name}.npz")["y"].astype(U32)
        bad = y != model(x)
        rep[name] = {"words": int(x.size), "mismatches": int(bad.sum()),
                     "first": [[int(x[i]), int(y[i]), int(model(x[i:i + 1])[0])] for i in np.flatnonzero(bad)[:4]]}
    return rep


CHECK = {"decode": check_decode, "bmm": check_bmm, "combine": check_combine}


def check(d: Path, inputs: dict, only: list[int] | None = None) -> dict:
    """Every case of the capture directory d against its model, `inputs[op](case)` rebuilding a case's inputs
    (`mla_capture.INPUTS`); `summary.mismatched_rows` counts output rows (an attention row is one head of one request; a bmm
    row one head's output row; a combine row one token) with any differing word."""
    man = json.loads((d / "manifest.json").read_text())
    rep: dict = {"suite": man["suite"], "device": man["device"], "vllm": man["vllm"], "triton": man["triton"], "cases": []}
    if man["suite"] == "mufu":
        rep["mufu"] = check_mufu(d)
        bad = sum(v["mismatches"] for v in rep["mufu"].values())
        rep["summary"] = {"rows": sum(v["words"] for v in rep["mufu"].values()), "mismatched_rows": bad}
        return rep
    rows = mismatched = 0
    for i, case in enumerate(man["cases"]):
        if only is not None and i not in only:
            continue
        op = case.get("op", "decode")
        out = dict(np.load(d / f"case_{i}.npz"))
        r = CHECK[op](case, out, inputs[op](case))
        rows += r["rows"]
        mismatched += r["mismatched_rows"]
        rep["cases"].append({"case": i, **{k: v for k, v in case.items() if k != "seqlens"},
                             "T": [min(case["seqlens"]), max(case["seqlens"])] if "seqlens" in case else None, **r})
    rep["summary"] = {"rows": rows, "mismatched_rows": mismatched}
    return rep
