"""Qwen3-235B-A22B-FP8's block-scaled FP8 kernels on one sm_120 GPU (cc 12.0) against CPU models: probe P1 of the 235B FP8 TP8
rtxpro6000 row, the evidence the sm_120 block-FP8 Definitions will cite.

vLLM runs as that row's engine does with the dense linears pinned to CUTLASS: `VLLM_USE_DEEP_GEMM=0` (TargetProfile `fp8_block_gemm=
"cutlass"`, target_profile.py `env()`; DeepGEMM off also turns E8M0 off, so every scale is an f32 word), `VLLM_BATCH_INVARIANT=1`,
custom ops on.  One GPU process writes one JSON verdict line per case (stdout, prefixed `VERDICT `, and `<out>/verdicts.jsonl`), the
property record `<out>/blackwell_fp8_capture.json` (`blackwell_fp8_capture`, `properties.record`), and last `FP8-SM120 GO` or `NO-GO`.  On a
device that is not cc 12.0 it writes one `skip` verdict and exits 3; a NO-GO exits 1.

  selection  `init_fp8_linear_kernel` at the row's two FP8 linears selects CutlassFp8BlockScaledMMKernel, and TritonFp8BlockScaledMMKernel
             under `--linear-backend triton` (kernel_config.linear_backend), each with QuantFP8.forward_cuda and use_ue8m0 False;
             `select_fp8_moe_backend` at block [64, 64] selects TRITON / TritonExperts.
  cutlass    (a) `_C.cutlass_scaled_mm` blockwise, through the selected kernel's apply_weights twice, at one TP8 rank's linears (`DENSE`)
             and M in `DENSE_M`, which spans sm_120's three dispatch variants (scaled_mm_blockwise_sm120_fp8_dispatch.cuh: swap-AB
             128x32x128 at M <= 64, pingpong 64x128x128 at M <= 256, cooperative 128x128x128 above).  The profiler names the CUTLASS
             kernels each launch ran; the three M classes must run three different ones.  Model `cutlass`.
  triton     (b) the same shapes through TritonFp8BlockScaledMMKernel (`w8a8_triton_block_scaled_mm`; its config is vLLM's default unless
             a tuned file exists for the device, and the record says which).  Model `triton` at that config's BLOCK_SIZE_K.
  moe        (c) `fused_experts` (E 128, top 8, hidden 4096, one TP8 rank's I = 192) at block [64, 64], twice, every `fused_moe_kernel`
             launch intercepted: w13's (token, slot) rows against `triton` at G 64 and the launch's BLOCK_SIZE_K, w2's with the routed
             weight's FMUL; batch invariance must run every launch at 64x64x32 without swap-AB.  The two `per_token_group_quant_fp8`
             calls of a run (w13's input and, after `silu_and_mul`, w2's; group 64: the fused per-block quant needs [128, 128],
             triton_moe.py:471-495) are recorded, must feed the launches, and are checked as (d).
  quant      (d) `per_token_group_quant_fp8` (`_C.per_token_group_fp8_quant`, use_ue8m0 False) at G 64 and 128, row- and column-major
             scales, against `Fp8GroupQuant_v1{K,G}` (twin `derived_rows.fp8_group_quant_row`).  The dense cases check the quant their
             GEMM received too.

The models (`block_row`): per K tile of KB, `d` = KB/32 chained k32 steps `BlackwellE4m3QmmaDot32_v1` from +0 (vectorised by core's
registered kernel); `cutlass` then acc = FFMA(d, FMUL(sx, sw), acc), `triton` acc = FFMA(FMUL(d, sx), sw, acc); the tile's scales are
block k*KB // G's; FMUL(acc, g) for a routed weight; then `cvt.rn.bf16.f32`.  On a sample of every launch's coordinates the plain-integer
chain of the registered primitives must give the same words (`reference_word`), and the competing models (`ALTERNATIVES`: Hopper's
14-bit step, the other scale orders, an unfused mul + add) are scored on the same rows: each must be refuted by a word of its phase, or
the capture could not tell it from the model.

Operands are random e4m3 (every code but NaN) and adversarial: a zero row, one outlier over subnormal codes, a subnormal scale, and from
M >= 7 a cancellation row (two +-448 products cancelling over small terms in every 32-group, against weight rows n % 8 == 1 built for
it: the adder-width discriminator), a saturated row (every activation byte +-448), and a row whose odd scale groups negate the even
ones, against weight blocks whose odd groups copy the even ones under scales a few ulps apart (`pair_groups`: the scale-order and
FFMA discriminator, which random operands rarely reach through the bf16 rounding).  GO = every case GO (0 mismatches and 0 unresolved
words, the twins equal to the reference, deterministic reruns, the expected kernels, tiles and statics), the selections, the three
CUTLASS kernels, every alternative refuted, and no error.  A case's verdict keeps its first mismatch (coordinates, words, and which
alternative, if any, gives the kernel's word).

    cd integrations/vllm; PY=/workspace/venv312/bin/python      # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/fp8_sm120_capture_gpu.py --out DIR [--scratch DIR] [--quick] [--only cutlass,triton,moe,quant]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from tests.properties import glm53_block_fp8_capture_gpu as H
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/blackwell-fp8-capture/v1"
RECORD = "blackwell_fp8_capture.json"
VERDICTS = "verdicts.jsonl"
WANT_CC = (12, 0)
HIDDEN = 4096
GROUP = 128
#: Qwen3-235B-A22B's FP8 linears on one TP8 rank, (name, K, N): 64 query and 4 KV heads of 128, so 8 + 2 * 1 heads per rank (fewer KV
#: heads than ranks are replicated).  The router (`mlp.gate`) and lm_head are bf16, and every layer's MLP is the MoE.
DENSE = (("qkv_proj", HIDDEN, (8 + 2) * 128), ("o_proj", 8 * 128, HIDDEN))
#: the row's decode (1) and prefill (256), and both sides of sm_120's dispatch thresholds
DENSE_M = (1, 2, 64, 65, 256, 257, 300)
QUICK_DENSE_M = (1, 65, 257)
NUM_EXPERTS, TOP_K, MOE_I, MOE_G = 128, 8, 1536 // 8, 64
#: (name, tokens, hot experts: 0 routes each token to 8 of all 128); the row's decode is `decode`, its prefill `prefill`
MOE = (("decode", 1, 0), ("small", 8, 0), ("block", 64, 0), ("prefill", 256, 16))
QUICK_MOE = ("decode", "small")
#: (K, G) of the quant phase: the dense inputs at 128, the expert inputs (hidden, one rank's I) at 64
QUANT = ((HIDDEN, 128), (8 * 128, 128), (HIDDEN, 64), (MOE_I, 64))
QUANT_M = 300
FORMS = ("cutlass", "triton", "triton_swap", "mul_add")
#: each model's competitors, (scale form, k32 step) by name
ALTERNATIVES: dict[str, dict[str, tuple[str, str]]] = {
    "cutlass": {"hopper_step": ("cutlass", "hopper"), "triton_order": ("triton", "blackwell"),
                "triton_swap_order": ("triton_swap", "blackwell"), "unfused": ("mul_add", "blackwell")},
    "triton": {"hopper_step": ("triton", "hopper"), "swap_order": ("triton_swap", "blackwell"),
               "cutlass_order": ("cutlass", "blackwell"), "unfused": ("mul_add", "blackwell")},
}
#: sm_120's blockwise dispatch classes by M (scaled_mm_blockwise_sm120_fp8_dispatch.cuh:261-276)
DISPATCH = (("swap_ab", 64), ("pingpong", 256), ("cooperative", None))
TOTALS = ("words", "mismatch", "unresolved", "quant_rows", "quant_mismatch", "reference_coords", "reference_quant_rows", "twin_mismatch",
          "alt_words")
SPECIAL_ROWS = (0, 2, 3, 4, 5, 6)


def dispatch_class(M: int) -> str:
    return next(name for name, top in DISPATCH if top is None or M <= top)


# ---------------------------------------------------------------------------------------------------------------------- models
def _step(dot: str):
    """The vectorised k32 e4m3 step: acc (n,) f32 words, a/b (n, 32) e4m3 bytes -> (n,) f32 words."""
    if dot == "blackwell":
        from verity.ml.kernels import tc as K
        from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32

        return lambda acc, a, b: np.asarray(K.group_sum_total_batch(BLACKWELL_SM120_E4M3_M16N8K32, acc, a, b)[0], dtype=np.uint32)
    if dot == "hopper":
        from verity.ml.definitions.numerics.derived_rows import tc_dot32_e4m3

        return lambda acc, a, b: tc_dot32_e4m3(acc, a, b)[0]
    raise ValueError(f"unknown k32 step {dot!r}")


def _check(K: int, G: int, KB: int) -> None:
    if K % G or G % KB or KB % 32:
        raise ValueError(f"block GEMM statics K={K} G={G} KB={KB}: K a multiple of G, G of KB, KB of 32")


def block_row(xq, sx, w, ws, *, G: int, KB: int, form: str = "triton", dot: str = "blackwell", g: int | None = None,
              chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """One token row of a block-scaled e4m3 GEMM: xq (K,) e4m3 bytes, sx (K/G,) f32 words, w (N, K) e4m3 bytes, ws (cdiv(N, G), K/G)
    f32 words -> (N bf16 words, nonfinite mask).  Per K tile k of KB, `d` = KB/32 chained `dot` steps from +0, a = sx[kb], b =
    ws[n // G][kb] with kb = k*KB // G, and

        cutlass       acc = FFMA(d, FMUL(a, b), acc)    CUTLASS's blockwise mainloop (`ScaledMmFp8BlockCoordinate` with KB = G)
        triton        acc = FFMA(FMUL(d, a), b, acc)    Triton's `accumulator += tl.dot(a, b) * a_s * b_s` (`triton_block_acc`)
        triton_swap   acc = FFMA(FMUL(d, b), a, acc)    the same with the scales swapped
        mul_add       acc = FADD(FMUL(d, FMUL(a, b)), acc), unfused (no kernel computes it: a discriminator)

    from acc = +0; then acc = FMUL(acc, g) when g is given, and `cvt.rn.bf16.f32`."""
    from verity.ml.definitions.numerics.derived_rows import add_rn, f2fp_bf16, fma_rn, mul_rn

    if form not in FORMS:
        raise ValueError(f"unknown scale form {form!r}")
    xq = np.asarray(xq, dtype=np.uint8).reshape(-1)
    K = xq.size
    _check(K, G, KB)
    w = np.asarray(w, dtype=np.uint8).reshape(-1, K)
    N = w.shape[0]
    sx = np.asarray(sx, dtype=np.uint32).reshape(K // G)
    ws = np.asarray(ws, dtype=np.uint32).reshape(-(-N // G), K // G)
    step = _step(dot)
    out = np.empty(N, dtype=np.uint16)
    nonfin = np.zeros(N, dtype=bool)
    for n0 in range(0, N, chunk):
        n1 = min(N, n0 + chunk)
        n = n1 - n0
        wc, rows = w[n0:n1], np.arange(n0, n1) // G
        acc = np.zeros(n, dtype=np.uint32)
        for k in range(K // KB):
            d = np.zeros(n, dtype=np.uint32)
            for st in range(KB // 32):
                lo = k * KB + 32 * st
                d = step(d, np.broadcast_to(xq[lo:lo + 32], (n, 32)), wc[:, lo:lo + 32])
            kb = k * KB // G
            a, b = np.broadcast_to(sx[kb], (n,)), ws[rows, kb]
            if form == "cutlass":
                acc = fma_rn(d, mul_rn(a, b), acc)
            elif form == "triton":
                acc = fma_rn(mul_rn(d, a), b, acc)
            elif form == "triton_swap":
                acc = fma_rn(mul_rn(d, b), a, acc)
            else:
                acc = add_rn(mul_rn(d, mul_rn(a, b)), acc)
        if g is not None:
            acc = mul_rn(acc, np.broadcast_to(np.uint32(g), (n,)))
        out[n0:n1] = f2fp_bf16(acc)
        nonfin[n0:n1] = (acc & 0x7F800000) == 0x7F800000
    return out, nonfin


def reference_word(xq, sx, wrow, wsrow, *, G: int, KB: int, form: str, g: int | None = None) -> int:
    """`block_row`'s word at one output coordinate (`dot` Blackwell's) as the plain-integer chain of the registered primitives
    `BlackwellE4m3QmmaDot32_v1`, `F32Mul_v1`, `F32Fma_v1`, `F32Add_v1` and `F2fpBf16_v1`; wrow the coordinate's weight row, wsrow its
    weight-scale row."""
    from verity.ml.circuits.ops import fp32, prims

    xq, wrow = [int(v) for v in np.asarray(xq).reshape(-1)], [int(v) for v in np.asarray(wrow).reshape(-1)]
    sx, wsrow = [int(v) for v in np.asarray(sx).reshape(-1)], [int(v) for v in np.asarray(wsrow).reshape(-1)]
    _check(len(xq), G, KB)
    mul, fma = fp32.F32Mul.evaluate, fp32.F32Fma.evaluate
    acc = 0
    for k in range(len(xq) // KB):
        d = 0
        for st in range(KB // 32):
            lo = k * KB + 32 * st
            d = prims.BlackwellE4m3QmmaDot32.evaluate(d, *xq[lo:lo + 32], *wrow[lo:lo + 32])
        a, b = sx[k * KB // G], wsrow[k * KB // G]
        acc = {"cutlass": lambda: fma(d, mul(a, b), acc), "triton": lambda: fma(mul(d, a), b, acc),
               "triton_swap": lambda: fma(mul(d, b), a, acc), "mul_add": lambda: fp32.F32Add.evaluate(mul(d, mul(a, b)), acc)}[form]()
    if g is not None:
        acc = mul(acc, int(g))
    return prims.F2fpBf16.evaluate(acc)


# ------------------------------------------------------------------------------------------------------------------ comparison
def _zero() -> dict[str, Any]:
    return {**{k: 0 for k in TOTALS}, "alternatives": {}, "first_mismatch": None}


def _row_operands(a: dict[str, np.ndarray], job: dict[str, Any], i: int):
    """(xq, sx, w, ws, g, kernel words) of row i: a token row of a dense launch, or a (token, slot) pair of an expert launch."""
    if job["phase"] != "moe":
        return a["q"][i], a["s"][i], a["w"], a["ws"], None, np.asarray(a["out"][i])
    t, u = i // job["top_k"], int(a["umap"][int(a["ids"][i])])
    return a["A"][t], a["As"][t], a["B"][u], a["Bs"][u], (int(a["tw"][i]) if job["mul"] else None), np.asarray(a["C"][i])


def compare_gemm(job: dict[str, Any]) -> dict[str, Any]:
    """Rows [i0, i1) of one GEMM launch against `block_row` at the job's statics; on its reference rows the plain-integer chain on
    three coordinates and every alternative on all of them; on a dense launch the activation quant it received, too."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.numerics import derived_rows as DR
    from verity.ml.definitions.registry import fp8 as F

    a = H._load(job["paths"])
    K, N, Gs, KB, form = job["K"], job["N"], job["G"], job["KB"], job["form"]
    rec = _zero()
    rec["alternatives"] = {name: 0 for name in ALTERNATIVES[form]}
    for i in range(*job["range"]):
        xq, sx, w, ws, g, got = _row_operands(a, job, i)
        want, nonfin = block_row(xq, sx, w, ws, G=Gs, KB=KB, form=form, g=g)
        bad = np.flatnonzero((want != got) & ~nonfin)
        rec["words"] += N
        rec["mismatch"] += int(bad.size)
        rec["unresolved"] += int(nonfin.sum())
        if bad.size and rec["first_mismatch"] is None:
            n = int(bad[0])
            explained = [name for name, (f, d) in ALTERNATIVES[form].items()
                         if int(block_row(xq, sx, w[n:n + 1], ws[n // Gs:n // Gs + 1], G=Gs, KB=KB, form=f, dot=d, g=g)[0][0]) == int(got[n])]
            rec["first_mismatch"] = {"row": i, "column": n, "want": int(want[n]), "kernel": int(got[n]), "explained_by": explained}
        if i in job["reference"]:
            for name, (f, d) in ALTERNATIVES[form].items():
                alt, alt_nonfin = block_row(xq, sx, w, ws, G=Gs, KB=KB, form=f, dot=d, g=g)
                rec["alternatives"][name] += int(((alt != got) & ~nonfin & ~alt_nonfin).sum())
            rec["alt_words"] += N
            for n in sorted({0, N - 1, (i * 7919) % N}):
                rec["reference_coords"] += 1
                rec["twin_mismatch"] += reference_word(xq, sx, w[n], ws[n // Gs], G=Gs, KB=KB, form=form, g=g) != int(want[n])
        if job["phase"] != "moe":
            q, s = DR.fp8_group_quant_row(a["x"][i], G=Gs)
            rec["quant_rows"] += 1
            bad_q = not (np.array_equal(q, a["q"][i]) and np.array_equal(s, a["s"][i]))
            rec["quant_mismatch"] += bad_q
            if bad_q and rec["first_mismatch"] is None:
                rec["first_mismatch"] = {"row": i, "quant": True}
            if i in job["reference"]:
                rec["reference_quant_rows"] += 1
                rec["twin_mismatch"] += list(evaluate(bind(F.Fp8GroupQuant, K=K, G=Gs), a["x"][i].tolist())) != q.tolist() + s.tolist()
    return {"phase": job["phase"], "case": job["case"], "launch": job.get("launch"), "range": job["range"], **rec}


def compare_quant(job: dict[str, Any]) -> dict[str, Any]:
    """Rows [i0, i1) of one quant call: (q, scale) against `Fp8GroupQuant_v1{K,G}`'s twin, the reference on the job's rows."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.numerics import derived_rows as DR
    from verity.ml.definitions.registry import fp8 as F

    a = H._load(job["paths"])
    K, Gq = job["K"], job["G"]
    rec = _zero()
    for i in range(*job["range"]):
        q, s = DR.fp8_group_quant_row(a["x"][i], G=Gq)
        rec["quant_rows"] += 1
        bad = not (np.array_equal(q, a["q"][i]) and np.array_equal(s, a["s"][i]))
        rec["quant_mismatch"] += bad
        if bad and rec["first_mismatch"] is None:
            col = np.flatnonzero(q != np.asarray(a["q"][i]))
            rec["first_mismatch"] = {"row": i, "quant": True, "column": int(col[0]) if col.size else None}
        if i in job["reference"]:
            rec["reference_quant_rows"] += 1
            rec["twin_mismatch"] += list(evaluate(bind(F.Fp8GroupQuant, K=K, G=Gq), a["x"][i].tolist())) != q.tolist() + s.tolist()
    return {"phase": job["phase"], "case": job["case"], "launch": job.get("launch"), "range": job["range"], **rec}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return compare_quant(job) if job["kind"] == "quant" else compare_gemm(job)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "launch": job.get("launch"), "error": H._error(e)}


def gemm_jobs(job: dict[str, Any], n_rows: int, special: tuple[int, ...]) -> list[dict[str, Any]]:
    """Row chunks of one GEMM launch; the reference and the alternatives run on the special rows and the last."""
    ref = sorted({r for r in (*special, n_rows - 1) if 0 <= r < n_rows})
    return [{**job, "kind": "gemm", "range": [r0, r1], "reference": [r for r in ref if r0 <= r < r1]}
            for r0, r1 in H._chunks(n_rows, job["N"] * job["K"] // 32)]


def quant_jobs(job: dict[str, Any], n_rows: int) -> list[dict[str, Any]]:
    ref = sorted({r for r in (*SPECIAL_ROWS, n_rows - 1) if 0 <= r < n_rows})
    step = max(1, 4096 * 64 // job["K"])
    return [{**job, "kind": "quant", "range": [r0, min(n_rows, r0 + step)], "reference": [r for r in ref if r0 <= r < r0 + step]}
            for r0 in range(0, n_rows, step)]


# ------------------------------------------------------------------------------------------------------------------- operands
def _activation(gen, M: int, K: int, G: int):
    """`glm53_block_fp8_capture_gpu._activation`'s rows (row 1 zero, row 2 one outlier over subnormal codes, row 3 a subnormal scale),
    and from M >= 7: row 4 small values with 1.0 at k % 32 in {0, 1} (two +448 codes in every 32-group), row 5 +-1.0 (every code
    +-448), row 6 random with every odd group of G the negation of the group before it (`pair_groups`)."""
    import torch

    x = H._activation(gen, M, K).float()
    if M >= 7:
        x[4] = torch.randn(K, generator=gen, device="cuda") * 2.0 ** -6
        x[4, 0::32] = 1.0
        x[4, 1::32] = 1.0
        x[5] = torch.where(torch.randn(K, generator=gen, device="cuda") < 0, -1.0, 1.0)
        for gk in range(1, K // G, 2):
            x[6, gk * G:(gk + 1) * G] = -x[6, (gk - 1) * G:gk * G]
    return x.to(torch.bfloat16).contiguous()


def pair_groups(w, ws_bits, G: int) -> None:
    """In place, on e4m3 bytes w (..., N, K) and f32 scale words ws_bits (..., cdiv(N, G), K/G), numpy or torch: in every odd block of
    weight rows (nb % 2 == 1) each odd K group of G copies the group before it, with that group's scale plus 1 to 7 ulps.  Against
    activation row 6 the two groups' terms cancel to a few ulps of either, so each scale form's rounding reaches the bf16 word: the
    scale-order and FFMA discriminator."""
    for nb in range(1, ws_bits.shape[-2], 2):
        r0, r1 = nb * G, min(w.shape[-2], (nb + 1) * G)
        for gk in range(1, w.shape[-1] // G, 2):
            w[..., r0:r1, gk * G:(gk + 1) * G] = w[..., r0:r1, (gk - 1) * G:gk * G]
            ws_bits[..., nb, gk] = ws_bits[..., nb, gk - 1] + 1 + (nb + gk) % 7


def _weights(gen, shape: tuple[int, ...]):
    """Random e4m3 codes but NaN, with every row n % 8 == 1 built to cancel against activation row 4: +448 at k % 32 == 0, -448 at
    k % 32 == 1, codes of exponent field <= 3 elsewhere."""
    import torch

    b = H._e4m3(gen, shape).view(torch.uint8).clone()
    r = b[..., 1::8, :]
    r &= 0x9F
    r[..., 0::32] = 0x7E
    r[..., 1::32] = 0xFE
    return b.view(torch.float8_e4m3fn)


def _routing(gen, M: int, hot: int):
    import torch

    pool = torch.randperm(NUM_EXPERTS, generator=gen, device="cuda")[:hot] if hot else torch.arange(NUM_EXPERTS, device="cuda")
    ids = torch.stack([pool[torch.randperm(pool.numel(), generator=gen, device="cuda")[:TOP_K]] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, TOP_K, generator=gen, device="cuda", dtype=torch.float32) + 0.05
    return ids.contiguous(), (w / w.sum(dim=1, keepdim=True)).contiguous()


def _logical(t, rows: int, cols: int):
    """A captured operand as (rows, cols), transposing the kernel's layout when it holds the other one; (tensor, transposed)."""
    if list(t.shape) == [rows, cols]:
        return t.contiguous(), False
    if list(t.shape) == [cols, rows]:
        return t.t().contiguous(), True
    raise ValueError(f"operand of shape {list(t.shape)} is neither ({rows}, {cols}) nor its transpose")


# ------------------------------------------------------------------------------------------------------------------ GPU phases
def environment(triton_cache: Path) -> dict[str, str]:
    env = {"VLLM_USE_DEEP_GEMM": "0", "VLLM_BATCH_INVARIANT": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TRITON_CACHE_DIR": str(triton_cache)}
    os.environ.update(env)
    return env


def vllm_config(linear_backend: str):
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.config.kernel import KernelConfig

    return set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]),
                                              kernel_config=KernelConfig(linear_backend=linear_backend)))


EXPECTED_KERNEL = {"cutlass": ("auto", "CutlassFp8BlockScaledMMKernel"), "triton": ("triton", "TritonFp8BlockScaledMMKernel")}


def _linear_kernel(K: int, N: int, name: str | None = None):
    import torch
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key

    return init_fp8_linear_kernel(activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, GROUP)),
                                  weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(GROUP, GROUP)),
                                  input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=(N, K), module_name=name)


def selection() -> dict[str, Any]:
    import torch

    rec: dict[str, Any] = {}
    for phase, (backend, want) in EXPECTED_KERNEL.items():
        with vllm_config(backend):
            got = {}
            for name, K, N in DENSE:
                k = _linear_kernel(K, N, name)
                q = getattr(k, "quant_fp8", None)
                got[name] = {"kernel": type(k).__name__, "quant_forward": getattr(getattr(q, "_forward_method", None), "__name__", None),
                             "use_ue8m0": getattr(q, "use_ue8m0", None)}
        rec[phase] = got
        rec[f"{phase}_ok"] = all(v["kernel"] == want and v["quant_forward"] == "forward_cuda" and v["use_ue8m0"] is False
                                 for v in got.values())
    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import select_fp8_moe_backend
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key, kFp8Dynamic128Sym

        cfg = FusedMoEConfig(num_experts=NUM_EXPERTS, experts_per_token=TOP_K, hidden_dim=HIDDEN, intermediate_size=MOE_I,
                             num_local_experts=NUM_EXPERTS, num_logical_experts=NUM_EXPERTS, activation=MoEActivation.SILU, device="cuda",
                             routing_method=RoutingMethodType.Renormalize, moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                             in_dtype=torch.bfloat16)
        # the keys `Fp8MoEMethod` selects with (quantization/fp8.py:536-540): the refined block in the weight key, and kFp8Dynamic128Sym
        # for the activations whatever the block (the Triton kernel quantizes them at the weight key's block_k); a (1, 64) activation
        # key matches no backend
        backend, cls = select_fp8_moe_backend(config=cfg, weight_key=create_fp8_quant_key(static=True, group_shape=GroupShape(MOE_G, MOE_G)),
                                              activation_key=kFp8Dynamic128Sym, allow_vllm_cutlass=False)
        rec["moe"] = {"backend": getattr(backend, "name", str(backend)), "experts": getattr(cls, "__name__", None)}
        rec["moe_ok"] = rec["moe"] == {"backend": "TRITON", "experts": "TritonExperts"}
    except Exception as e:  # noqa: BLE001
        rec["moe"], rec["moe_ok"] = {"error": H._error(e)}, False
    return rec


def _cuda_kernels(fn) -> list[str]:
    """The CUDA kernels one call of fn launches, by name (the profiler's)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.key for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA})


def triton_config(K: int, N: int, M: int) -> dict[str, Any]:
    """The config `w8a8_triton_block_scaled_mm` launches at (M, K, N): a tuned file's nearest entry, else its default."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import get_w8a8_block_fp8_configs

    tuned = get_w8a8_block_fp8_configs(N, K, GROUP, GROUP)
    if tuned:
        return {"tuned": True, **tuned[min(tuned, key=lambda x: abs(x - M))]}
    return {"tuned": False, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": GROUP, "BLOCK_SIZE_K": GROUP, "GROUP_SIZE_M": 32, "num_warps": 4,
            "num_stages": 2}


def dense_launch(work: Path, phase: str, name: str, K: int, N: int, M: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """One dense case: the phase's kernel's apply_weights, twice, the second under the profiler; (launch record, job)."""
    import torch

    with vllm_config(EXPECTED_KERNEL[phase][0]):
        kernel = _linear_kernel(K, N)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        layer = torch.nn.Module()
        w0, ws0 = _weights(gen, (N, K)), H._scales(gen, (H.cdiv(N, GROUP), K // GROUP), -12, -6)
        pair_groups(w0.view(torch.uint8), ws0.view(torch.int32), GROUP)
        layer.weight = torch.nn.Parameter(w0, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(ws0, requires_grad=False)
        layer.input_scale = None
        kernel.process_weights_after_loading(layer)
        x = _activation(gen, M, K, GROUP)
        seen: list[dict[str, Any]] = []
        inner = kernel.apply_block_scaled_mm

        def tap(A, B, As, Bs):
            seen.append({"A": A.clone(), "B": B, "As": As.contiguous().clone(), "Bs": Bs.contiguous().clone()})
            return inner(A, B, As, Bs)

        kernel.apply_block_scaled_mm = tap
        out1 = kernel.apply_weights(layer, x).clone()
        outs: list = []
        kernels = _cuda_kernels(lambda: outs.append(kernel.apply_weights(layer, x).clone()))
    op = seen[0]
    w, w_t = _logical(op["B"].view(torch.uint8), N, K)
    ws, ws_t = _logical(op["Bs"], H.cdiv(N, GROUP), K // GROUP)
    cfg = triton_config(K, N, M) if phase == "triton" else None
    KB = cfg["BLOCK_SIZE_K"] if cfg else GROUP
    launch = {"phase": phase, "case": name, "K": K, "N": N, "M": M, "kernel": type(kernel).__name__, "dispatch": dispatch_class(M),
              "cuda_kernels": kernels, "triton_config": cfg, "KB": KB, "B_transposed": w_t, "Bs_transposed": ws_t,
              "deterministic": bool(torch.equal(out1, outs[0])),
              "operands_ok": (list(op["A"].shape) == [M, K] and list(op["As"].shape) == [M, K // GROUP] and op["A"].dtype == torch.float8_e4m3fn
                              and out1.dtype == torch.bfloat16 and kernel.__class__.__name__ == EXPECTED_KERNEL[phase][1]),
              "statics_ok": KB % 32 == 0 and GROUP % KB == 0}
    job = {"phase": phase, "case": name, "K": K, "N": N, "M": M, "G": GROUP, "KB": KB, "form": phase,
           "paths": H._save(work / f"{phase}_{name}_M{M}", x=H._u16(x), q=H._u8(op["A"]), s=H._u32(op["As"]), w=w.cpu().numpy(),
                            ws=H._u32(ws), out=H._u16(out1))}
    return launch, job


def moe_launches(work: Path, name: str, M: int, hot: int, seed: int) -> tuple[list[dict], list[dict]]:
    """One MoE case: `fused_experts` twice at block [64, 64] with every expert launch and quant call recorded; (launches, jobs)."""
    import torch
    from vllm.model_executor.layers.fused_moe import utils as MU
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    gen = torch.Generator(device="cuda").manual_seed(seed)
    E, I, Gm = NUM_EXPERTS, MOE_I, MOE_G
    w1, w2 = _weights(gen, (E, 2 * I, HIDDEN)), _weights(gen, (E, HIDDEN, I))
    s1 = H._scales(gen, (E, 2 * I // Gm, HIDDEN // Gm), -12, -6)
    s2 = H._scales(gen, (E, HIDDEN // Gm, I // Gm), -12, -6)
    pair_groups(w1.view(torch.uint8), s1.view(torch.int32), Gm)
    pair_groups(w2.view(torch.uint8), s2.view(torch.int32), Gm)
    x = _activation(gen, M, HIDDEN, Gm)
    ids, tw = _routing(gen, M, hot)
    qc = fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, block_shape=[Gm, Gm])
    quants: list[dict[str, Any]] = []
    inner_q = MU.per_token_group_quant_fp8

    def qtap(A, group_size, *args, **kw):
        q, s = inner_q(A, group_size, *args, **kw)
        quants.append({"label": icp.label, "x": A.clone(), "q": q.clone(), "s": s.contiguous().clone(), "G": int(group_size)})
        return q, s

    MU.per_token_group_quant_fp8 = qtap
    try:
        with H.Interceptor() as icp:
            ys = []
            for run in (1, 2):
                icp.label = f"run{run}"
                ys.append(fused_experts(x, w1, w2, tw, ids, activation=MoEActivation.SILU, global_num_experts=E, quant_config=qc).clone())
    finally:
        MU.per_token_group_quant_fp8 = inner_q
    torch.cuda.synchronize()
    runs = {lab: [L for L in icp.launches if L["label"] == lab] for lab in ("run1", "run2")}
    qruns = {lab: [q for q in quants if q["label"] == lab] for lab in ("run1", "run2")}
    ids_flat = ids.cpu().numpy().astype(np.int64).reshape(-1)
    used = np.unique(ids_flat)
    umap = np.full(E, -1, dtype=np.int64)
    umap[used] = np.arange(used.size)
    q_ok = all(len(qruns[lab]) == 2 and [q["G"] for q in qruns[lab]] == [Gm, Gm] for lab in qruns)
    launches, jobs = [], []
    for j, kind in enumerate(("w13", "w2")):
        L, twin = runs["run1"][j], runs["run2"][j]
        K, N = int(L["A"].shape[1]), int(L["B"].shape[1])
        n_pairs = int(L["A"].shape[0]) * L["top_k"]
        fed = q_ok and torch.equal(L["A"], qruns["run1"][j]["q"]) and torch.equal(L["A_scale"].contiguous(), qruns["run1"][j]["s"])
        rec = {"phase": "moe", "case": name, "launch": kind, "M": M, "hot": hot, "K": K, "N": N, "pairs": n_pairs, "config": L["config"],
               "KB": L["KB"], "SWAP": L["SWAP"], "block_shape": L["block_shape"], "flags": L["flags"], "experts_used": int(used.size),
               "quant_calls": {lab: [q["G"] for q in qruns[lab]] for lab in qruns}, "quant_feeds_launch": bool(fed),
               "deterministic": bool(torch.equal(L["C"], twin["C"]) and torch.equal(ys[0], ys[1])),
               "assignment_ok": H._assignment_ok(L, ids_flat)}
        rec["statics_ok"] = (L["block_shape"] == [Gm, Gm] and L["flags"] == [True, False, False, False, False] and not L["bias"]
                             and L["mul_routed_weight"] == (kind == "w2") and L["top_k"] == (TOP_K if kind == "w13" else 1)
                             and n_pairs == M * TOP_K and L["KB"] == 32 and not L["SWAP"] and L["config"] == H.BATCH_INVARIANT_CONFIG
                             and K == (HIDDEN if kind == "w13" else I) and N == (2 * I if kind == "w13" else HIDDEN))
        rec["operands_ok"] = bool(fed)
        launches.append(rec)
        sel = torch.as_tensor(used, device="cuda")
        d = work / f"moe_{name}_{kind}"
        arrays = {"A": H._u8(L["A"]), "As": H._u32(L["A_scale"]), "B": H._u8(L["B"].view(torch.uint8)[sel]), "Bs": H._u32(L["B_scale"][sel]),
                  "ids": ids_flat, "umap": umap, "tw": H._u32(tw).reshape(-1), "C": H._u16(L["C"].reshape(n_pairs, N))}
        job = {"phase": "moe", "case": name, "launch": kind, "K": K, "N": N, "G": Gm, "KB": L["KB"], "form": "triton",
               "mul": L["mul_routed_weight"], "top_k": L["top_k"], "paths": H._save(d, **arrays)}
        if rec["statics_ok"]:
            jobs += gemm_jobs(job, n_pairs, tuple(r * TOP_K for r in SPECIAL_ROWS))
        if q_ok:
            Q = qruns["run1"][j]
            qd = work / f"moe_{name}_{kind}_input"
            rows = int(Q["x"].reshape(-1, Q["x"].shape[-1]).shape[0])
            qjob = {"phase": "moe", "case": name, "launch": f"{kind}_input", "K": int(Q["x"].shape[-1]), "G": Gm,
                    "paths": H._save(qd, x=H._u16(Q["x"].reshape(rows, -1)), q=H._u8(Q["q"].reshape(rows, -1)),
                                     s=H._u32(Q["s"].reshape(rows, -1)))}
            jobs += quant_jobs(qjob, rows)
    del w1, w2
    torch.cuda.empty_cache()
    return launches, jobs


def quant_launch(work: Path, K: int, Gq: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = _activation(gen, QUANT_M, K, Gq)
    q1, s1 = per_token_group_quant_fp8(x, Gq, use_ue8m0=False)
    q2, s2 = per_token_group_quant_fp8(x, Gq, use_ue8m0=False)
    q3, s3 = per_token_group_quant_fp8(x, Gq, column_major_scales=True, use_ue8m0=False)
    torch.cuda.synchronize()
    case = f"K{K}/G{Gq}"
    launch = {"phase": "quant", "case": case, "K": K, "G": Gq, "M": QUANT_M, "operands_ok": list(s1.shape) == [QUANT_M, K // Gq],
              "deterministic": bool(torch.equal(q1, q2) and torch.equal(s1, s2)),
              "layouts_agree": bool(torch.equal(q1, q3) and torch.equal(s1, s3.contiguous()))}
    job = {"phase": "quant", "case": case, "K": K, "G": Gq,
           "paths": H._save(work / f"quant_K{K}_G{Gq}", x=H._u16(x), q=H._u8(q1), s=H._u32(s1))}
    return launch, quant_jobs(job, QUANT_M)


# --------------------------------------------------------------------------------------------------------------------- verdict
def case_key(x: dict[str, Any]) -> str:
    if x["phase"] in ("cutlass", "triton"):
        return f"{x['phase']}/{x['case']}/M{x['M']}"
    return f"{x['phase']}/{x['case']}"


def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["deterministic"] and x.get("operands_ok", True) and x.get("statics_ok", True) and x.get("assignment_ok") is not False
                and x.get("layouts_agree", True))


def _words_ok(t: dict[str, Any], phase: str) -> bool:
    quant = t["quant_rows"] > 0 and t["quant_mismatch"] == 0 and t["reference_quant_rows"] > 0
    if phase == "quant":
        return quant and t["twin_mismatch"] == 0
    return (t["words"] > 0 and t["mismatch"] == 0 and t["unresolved"] == 0 and t["reference_coords"] > 0 and t["twin_mismatch"] == 0
            and quant)


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's verdict: its launches' checks, its words (every comparison of its key), its first mismatch."""
    out: dict[str, dict[str, Any]] = {}
    for x in doc["launches"]:
        c = out.setdefault(case_key(x), {"phase": x["phase"], "launches_ok": True, "totals": {k: 0 for k in TOTALS}, "first_mismatch": None})
        c["launches_ok"] = c["launches_ok"] and _launch_ok(x)
    for e in doc["errors"]:
        if e.split("/")[0] in ("cutlass", "triton", "moe", "quant") and e not in out:
            out[e] = {"phase": e.split("/")[0], "launches_ok": False, "totals": {k: 0 for k in TOTALS}, "first_mismatch": None}
    for r in doc.get("comparisons", []):
        c = out.get(case_key(r))
        if c is not None:
            for k in TOTALS:
                c["totals"][k] += r[k]
            c["first_mismatch"] = c["first_mismatch"] or r["first_mismatch"]
    for key, c in out.items():
        errors = [e for e in doc["errors"] if e == key or e.startswith(f"compare/{key}")]
        c["ok"] = c["launches_ok"] and _words_ok(c["totals"], c["phase"]) and not errors
    return out


def alternatives(doc: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per GEMM phase, the words each alternative model got wrong over the run."""
    out: dict[str, dict[str, int]] = {}
    for r in doc.get("comparisons", []):
        for name, n in r.get("alternatives", {}).items():
            out.setdefault(r["phase"], {}).setdefault(name, 0)
            out[r["phase"]][name] += n
    return out


def dispatch(doc: dict[str, Any]) -> dict[str, Any]:
    """Per CUTLASS shape, the CUDA kernels named `cutlass` that each dispatch class ran; ok when every shape ran all three classes,
    each class one set of kernels, and the classes' sets differ."""
    out: dict[str, Any] = {}
    for x in doc["launches"]:
        if x["phase"] == "cutlass":
            names = frozenset(k for k in x["cuda_kernels"] if "cutlass" in k.lower())
            out.setdefault(x["case"], {}).setdefault(x["dispatch"], set()).add(names)
    ok = bool(out) and all(set(by) == {c for c, _ in DISPATCH} for by in out.values())
    ok = ok and all(len(s) == 1 and next(iter(s)) for by in out.values() for s in by.values())
    ok = ok and all(len({next(iter(s)) for s in by.values()}) == len(by) for by in out.values())
    return {"classes": {case: {cls: [sorted(n) for n in s] for cls, s in by.items()} for case, by in out.items()}, "ok": ok}


def refuted(doc: dict[str, Any], only: set[str]) -> bool:
    """Every alternative of every GEMM phase that ran got some word wrong."""
    alts = doc["alternatives"]
    return all(alts.get(p, {}).get(n, 0) > 0 for p in only & {"cutlass", "triton", "moe"}
               for n in ALTERNATIVES["cutlass" if p == "cutlass" else "triton"])


def verdict(doc: dict[str, Any], only: set[str]) -> bool:
    sel = doc["selection"]
    return bool(doc["gate"]["ok"] and not doc["errors"] and all(sel.get(f"{p}_ok") is True for p in only & {"cutlass", "triton", "moe"})
                and ("cutlass" not in only or doc["dispatch"]["ok"]) and {x["phase"] for x in doc["launches"]} == only
                and doc["cases"] and all(c["ok"] for c in doc["cases"].values()) and refuted(doc, only))


def verdict_line(key: str, c: dict[str, Any]) -> dict[str, Any]:
    return {"case": key, "verdict": "GO" if c["ok"] else "NO-GO", "launches_ok": c["launches_ok"], **c["totals"],
            "first_mismatch": c["first_mismatch"]}


# ------------------------------------------------------------------------------------------------------------------------- run
def device_present() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0) == WANT_CC)
    except Exception:  # noqa: BLE001
        return False


def _emit(outdir: Path, line: dict[str, Any]) -> None:
    text = json.dumps(line, default=str)
    print(f"VERDICT {text}", flush=True)
    with open(outdir / VERDICTS, "a") as f:
        f.write(text + "\n")


def run(out: str, scratch: str | None, min_driver: int, want_sms: int, workers: int, quick: bool, only: set[str]) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / VERDICTS).write_text("")
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "quick": quick, "only": sorted(only), "errors": {}}
    if not device_present():
        doc["gate"] = {"ok": False, "skip": f"no cc {WANT_CC[0]}.{WANT_CC[1]} CUDA device"}
        _emit(outdir, {"case": "gate", "verdict": "skip", "reason": doc["gate"]["skip"]})
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("FP8-SM120 SKIP", flush=True)
        raise SystemExit(3)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    doc["env"] = environment(work / "triton_cache")
    doc["environment"] = G._environment()
    gate = C.gate(min_driver, f"{WANT_CC[0]}.{WANT_CC[1]}", want_sms)
    if not want_sms:
        gate["checks"]["sms"] = True
        gate["ok"] = all(gate["checks"].values())
    doc["gate"] = gate
    _emit(outdir, {"case": "gate", "verdict": "GO" if gate["ok"] else "NO-GO", **gate})
    if not gate["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("FP8-SM120 NO-GO gate", flush=True)
        raise SystemExit(1)
    doc["selection"], doc["launches"], jobs = {}, [], []
    try:
        doc["selection"] = selection()
    except Exception as e:  # noqa: BLE001
        doc["errors"]["selection"] = H._error(e)
    _emit(outdir, {"case": "selection", "verdict": "GO" if all(doc["selection"].get(f"{p}_ok") for p in only & {"cutlass", "triton", "moe"})
                   and "selection" not in doc["errors"] else "NO-GO", **doc["selection"]})
    seed = 12000
    for phase in ("cutlass", "triton"):
        if phase in only:
            for name, K, N in DENSE:
                for M in (QUICK_DENSE_M if quick else DENSE_M):
                    seed += 1
                    try:
                        launch, job = dense_launch(work, phase, name, K, N, M, seed)
                        doc["launches"].append(launch)
                        jobs += gemm_jobs(job, M, SPECIAL_ROWS) if launch["statics_ok"] else []
                        print(f"[fp8-sm120] {phase} {name} M={M} kernel={launch['kernel']} KB={launch['KB']} det={launch['deterministic']} "
                              f"cuda={[k[:80] for k in launch['cuda_kernels']]}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        doc["errors"][f"{phase}/{name}/M{M}"] = H._error(e)
    if "moe" in only:
        for name, M, hot in MOE:
            if quick and name not in QUICK_MOE:
                continue
            seed += 1
            try:
                launches, mjobs = moe_launches(work, name, M, hot, seed)
                doc["launches"] += launches
                jobs += mjobs
                for x in launches:
                    print(f"[fp8-sm120] moe {name} {x['launch']} K={x['K']} N={x['N']} config={x['config']} KB={x['KB']} SWAP={x['SWAP']} "
                          f"statics={x['statics_ok']} quant={x['quant_calls']} det={x['deterministic']}", flush=True)
            except Exception as e:  # noqa: BLE001
                doc["errors"][f"moe/{name}"] = H._error(e)
    if "quant" in only:
        for K, Gq in QUANT:
            seed += 1
            try:
                launch, qjobs = quant_launch(work, K, Gq, seed)
                doc["launches"].append(launch)
                jobs += qjobs
            except Exception as e:  # noqa: BLE001
                doc["errors"][f"quant/K{K}/G{Gq}"] = H._error(e)
    print(f"[fp8-sm120] comparing {len(jobs)} jobs on {workers} workers", flush=True)
    results: dict[tuple, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for job, res in zip(jobs, ex.map(compare, jobs)):
            if "error" in res:
                doc["errors"][f"compare/{case_key({**job, 'M': job.get('M')})}/{res.get('launch')}"] = res["error"]
                continue
            key = (res["phase"], res["case"], res.get("launch"), job.get("M"))
            agg = results.setdefault(key, {"phase": res["phase"], "case": res["case"], "launch": res.get("launch"), "M": job.get("M"),
                                           **_zero()})
            for k in TOTALS:
                agg[k] += res[k]
            for name, n in res["alternatives"].items():
                agg["alternatives"][name] = agg["alternatives"].get(name, 0) + n
            agg["first_mismatch"] = agg["first_mismatch"] or res["first_mismatch"]
    doc["comparisons"] = list(results.values())
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["alternatives"] = alternatives(doc)
    doc["dispatch"] = dispatch(doc) if "cutlass" in only else {"ok": None}
    doc["cases"] = cases(doc)
    for key, c in doc["cases"].items():
        _emit(outdir, verdict_line(key, c))
    _emit(outdir, {"case": "dispatch", "verdict": "GO" if doc["dispatch"]["ok"] in (True, None) else "NO-GO", **doc["dispatch"]})
    _emit(outdir, {"case": "alternatives", "refuted": doc["alternatives"]})
    doc = seal_record(doc, verdict(doc, only))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"FP8-SM120 {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Qwen3-235B-A22B-FP8's block-FP8 kernels on this sm_120 GPU against their CPU models")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and {VERDICTS}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, Triton cache); default the system temp")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-sms", type=int, default=0, help="the device's SM count, 0 for any cc 12.0 device")
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"dense M in {QUICK_DENSE_M}, MoE cases {QUICK_MOE}")
    ap.add_argument("--only", default="cutlass,triton,moe,quant", help="comma list of phases: cutlass, triton, moe, quant")
    a = ap.parse_args(argv)
    only = set(a.only.split(","))
    if only - {"cutlass", "triton", "moe", "quant"}:
        ap.error(f"unknown phases {sorted(only - {'cutlass', 'triton', 'moe', 'quant'})}")
    doc = run(a.out, a.scratch, a.min_driver, a.want_sms, a.workers, a.quick, only)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
