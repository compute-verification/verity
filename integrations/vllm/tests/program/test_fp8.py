"""CPU tests of the FP8 registry (`verity/ml/definitions/registry/fp8.py`) on saved H100 operands.

Data (`tests/program/data/`, recorded on an H100):
* `fp8_saved_operands.npz` — `_C.cutlass_scaled_mm` on random e4m3 operands at three B1 shapes (K=1536/N=2048 with bias,
  K=1536/N=1536, K=8960/N=1536): the sampled weight rows, the activation rows + scales, the kernel's bf16 outputs
  (`gpu/tc-fp8/vllm_fp8_kernels_probe.py --gemm`);
* `fp8_production_row_l0_qkv.npz` — three real rows of the production run (profile 0ca394ec17b9, request 10ae824e…,
  positions 0, 63, 130; layer 0 qkv_proj): the bf16 norm output `x`, the captured `x_q` and `s_x` (fp8_quant ports 0/1),
  the captured output row, plus eight committed e4m3 weight rows, the committed f32 weight scale and the bf16 bias.

The tests evaluate the registered functions gate by gate (`verity.ml.circuits.evaluation`) and compare words.  Python >= 3.12
(`veritor.core.silicon_total_fp8`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 12):  # pragma: no cover
    pytest.skip("veritor.core needs Python 3.12", allow_module_level=True)

from verity.ml.circuits import Program, bind
from verity_vllm.correspondence.capture_identities import _coll_range
from verity.ml.circuits.evaluation import evaluate_region
from verity.ml.definitions.registry import fp8 as F
from verity.ml.definitions.numerics import fp8_twins
from verity.ml.definitions.registry import prims as P

DATA = Path(__file__).parent / "data"


def _evaluate(program: Program, arrays: list[np.ndarray]) -> list[list[int]]:
    """Bind the program's Input collections (in parameter order) to `arrays` and return every root call's returns."""
    circ = program.circuit
    t: dict[int, int] = {}
    for i, arr in enumerate(arrays):
        lo, hi = _coll_range(circ, program.input_coll(i))
        flat = np.ascontiguousarray(arr).reshape(-1)
        assert hi - lo == flat.size, (i, hi - lo, flat.size)
        for g, v in zip(range(lo, hi), flat.tolist()):
            t[g] = int(v)
    outs = []
    for call in circ.root_calls():
        evaluate_region(circ, call, t)
        outs.append([t[g] for g in call.returns])
    return outs


def _evaluate_returns(program: Program, arrays: list[np.ndarray]) -> list[int]:
    """Like `_evaluate`, but returns the program function's OWN return words (its body's `ret` refs resolved to root gates),
    whatever the root-call order of its children."""
    from verity.ml.circuits.layout import resolve
    circ = program.circuit
    t: dict[int, int] = {}
    for i, arr in enumerate(arrays):
        lo, hi = _coll_range(circ, program.input_coll(i))
        flat = np.ascontiguousarray(arr).reshape(-1)
        assert hi - lo == flat.size, (i, hi - lo, flat.size)
        for g, v in zip(range(lo, hi), flat.tolist()):
            t[g] = int(v)
    for call in circ.root_calls():
        evaluate_region(circ, call, t)
    ret = program.root.body.ret
    return [t[resolve(circ.root_scope, ret[i])] for i in range(ret.count)]


# ------------------------------------------------------------------------------------------------------------
# the conversion
# ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value,word", [
    (448.0, 0x7E), (-448.0, 0xFE), (449.0, 0x7E), (1e9, 0x7E), (float("inf"), 0x7E), (float("-inf"), 0xFE),
    (1.0, 0x38), (-1.0, 0xB8), (1.0625, 0x38), (1.0625001, 0x39), (1.1875, 0x3A),
    (2.0 ** -6, 0x08), (2.0 ** -9, 0x01), (2.0 ** -10, 0x00), (2.0 ** -10 * 1.001, 0x01), (-(2.0 ** -11), 0x80),
    (7 * 2.0 ** -9, 0x07), (7.5 * 2.0 ** -9, 0x08), (6.5 * 2.0 ** -9, 0x06), (0.0, 0x00), (-0.0, 0x80),
])
def test_f32_to_e4m3_sat_words(value: float, word: int) -> None:
    assert F.F32ToE4m3Sat.evaluate(int(P.bits_of(np.float32(value)))) == word


def test_f32_to_e4m3_sat_nan_is_canonical_positive() -> None:
    # measured on the H100 over all 2^32 words: every NaN input (either sign) -> 0x7F
    assert F.F32ToE4m3Sat.evaluate(0x7FC00000) == 0x7F
    assert F.F32ToE4m3Sat.evaluate(0xFFC00000) == 0x7F
    assert F.F32ToE4m3Sat.evaluate(0x7F800001) == 0x7F


def test_numpy_twin_matches_scalar_model() -> None:
    rng = np.random.default_rng(3)
    u = rng.integers(0, 1 << 32, 20000, dtype=np.uint64).astype(np.uint32)
    u = np.concatenate([u, (np.arange(0, 1 << 16, 7, dtype=np.uint32) << 16)])  # a spread of bf16-representable words
    got = fp8_twins.np_f32_to_e4m3_sat(u.view(np.float32))
    want = np.array([F.F32ToE4m3Sat.evaluate(int(w)) for w in u], dtype=np.uint8)
    assert np.array_equal(got, want)


# ------------------------------------------------------------------------------------------------------------
# the scaled GEMM on saved cutlass_scaled_mm operands
# ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("group", ["g0", "g1", "g3"])
def test_scaled_mm_fp8_matches_cutlass(group: str) -> None:
    d = np.load(DATA / "fp8_saved_operands.npz")
    K, N, bias_on = int(d[f"{group}_K"]), int(d[f"{group}_N"]), bool(int(d[f"{group}_bias"]))
    xq, sx, w, sw, b, y = d[f"{group}_xq"], d[f"{group}_sx"], d[f"{group}_w"], d[f"{group}_sw"], d[f"{group}_b"], d[f"{group}_y"]
    ncols = w.shape[0]
    prog = Program(bind(F.ScaledMmFp8, K=K, N=ncols, BIAS=bias_on))
    for r in range(xq.shape[0]):
        args = [xq[r], np.array([P.bits_of(np.float32(sx[r]))], np.uint32), w, np.array([P.bits_of(np.float32(sw[0]))], np.uint32)]
        if bias_on:
            args.append(b)
        outs = _evaluate(prog, args)
        got = np.array([v for o in outs for v in o], dtype=np.uint16)
        assert np.array_equal(got, y[r]), (group, r, [hex(v) for v in got], [hex(v) for v in y[r]])


# ------------------------------------------------------------------------------------------------------------
# the whole linear on real production rows (computed x_q and s_x, then eight output coordinates)
# ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t", [0, 63, 130])
def test_quant_linear_fp8_on_production_row(t: int) -> None:
    d = np.load(DATA / "fp8_production_row_l0_qkv.npz")
    x, xq, sx, y = d[f"t{t}_x"], d[f"t{t}_xq"], d[f"t{t}_sx"], d[f"t{t}_y"]
    cols, w_rows, sw, b = d["cols"], d["w_rows"], d["sw"], d["b"]
    prog = Program(bind(F.QuantLinearFp8, K=1536, N=len(cols), BIAS=True))
    outs = _evaluate(prog, [x, w_rows, sw, b])
    got_xq = np.array(outs[0][:1536], dtype=np.uint8)
    got_sx = outs[0][1536]
    got_y = np.array([v for o in outs[1:] for v in o], dtype=np.uint16)
    assert np.array_equal(got_xq, xq)  # port 0: the e4m3 row, 1536 words
    assert got_sx == int(sx[0])  # port 1: the runtime-derived scale, a COMPUTED word
    assert np.array_equal(got_y, y[cols])  # the scaled GEMM output at the eight sampled coordinates


def test_numpy_twin_of_the_quant_kernel_on_production_rows() -> None:
    d = np.load(DATA / "fp8_production_row_l0_qkv.npz")
    for t in (0, 63, 130):
        xq, sx = fp8_twins.np_dynamic_per_token_quant(d[f"t{t}_x"][None, :])
        assert np.array_equal(xq[0], d[f"t{t}_xq"])
        assert int(P.bits_of(sx[0])) == int(d[f"t{t}_sx"][0])


def test_hopper_e4m3_step_is_not_ieee() -> None:
    """The pinned step truncates the accumulator to the 14-bit adder: acc + 0 != acc for most words (accid family)."""
    zeros = [0] * 64
    acc = int(P.bits_of(np.float32(1.0 + 2 ** -20)))
    assert F.HopperE4m3QgmmaDot32.evaluate(acc, *zeros) != acc
    assert F.HopperE4m3QgmmaDot32.evaluate(int(P.bits_of(np.float32(1.0))), *zeros) == int(P.bits_of(np.float32(1.0)))
    one = 0x38  # e4m3 1.0
    assert F.HopperE4m3QgmmaDot32.evaluate(0, *([one] * 32 + [one] * 32)) == int(P.bits_of(np.float32(32.0)))


# ------------------------------------------------------------------------------------------------------------
# the BLOCK-SCALED linear (Qwen3-4B-Instruct-2507-FP8 on Hopper): `_C.per_token_group_fp8_quant` + blockwise
# `_C.cutlass_scaled_mm`, on complete operand sets recorded on the H100:
# "p" = (M=1, K=256, N=256) -> the swap_ab Pingpong variant, "c" = (M=4, K=384, N=256) -> the Cooperative variant.
# ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tag", ["p", "c"])
def test_fp8_group_quant_matches_kernel(tag: str) -> None:
    d = np.load(DATA / "fp8_block_operands.npz")
    x, xq, xs = d[f"{tag}_x"], d[f"{tag}_xq"], d[f"{tag}_xs"]
    M, K = x.shape
    prog = Program(bind(F.Fp8GroupQuant, K=K, G=F.FP8_BLOCK))
    for m in range(M):
        got = _evaluate_returns(prog, [x[m]])
        assert np.array_equal(np.array(got[:K], dtype=np.uint8), xq[m]), (tag, m)             # port 0: the e4m3 words
        assert np.array_equal(np.array(got[K:], dtype=np.uint32), xs[m]), (tag, m)            # port 1: the K/128 group scales (computed)


@pytest.mark.parametrize("tag", ["p", "c"])
def test_scaled_mm_fp8_block_matches_cutlass(tag: str) -> None:
    d = np.load(DATA / "fp8_block_operands.npz")
    xq, xs, w, ws, y = d[f"{tag}_xq"], d[f"{tag}_xs"], d[f"{tag}_w"], d[f"{tag}_ws"], d[f"{tag}_y"]
    M, K = xq.shape
    N = w.shape[0]
    prog = Program(bind(F.ScaledMmFp8Block, K=K, N=N, G=F.FP8_BLOCK))
    for m in range(M):
        got = np.array(_evaluate_returns(prog, [xq[m], xs[m], w, ws]), dtype=np.uint16)
        assert np.array_equal(got, y[m]), (tag, m, int((got != y[m]).sum()))


def test_scaled_mm_fp8_block_wrong_scale_fails() -> None:
    """A perturbed weight-block scale (one ulp on one block) must change output words: the scales are load-bearing inputs."""
    d = np.load(DATA / "fp8_block_operands.npz")
    xq, xs, w, ws, y = d["p_xq"], d["p_xs"], d["p_w"], d["p_ws"], d["p_y"]
    K, N = xq.shape[1], w.shape[0]
    ws2 = ws.copy()
    ws2[0, 0] = np.uint32(int(ws2[0, 0]) ^ 0x00080000)   # a mantissa bit of the block-(0,0) scale
    prog = Program(bind(F.ScaledMmFp8Block, K=K, N=N, G=F.FP8_BLOCK))
    got = np.array(_evaluate_returns(prog, [xq[0], xs[0], w, ws2]), dtype=np.uint16)
    assert not np.array_equal(got[:128], y[0][:128])      # the perturbed block's columns move
    assert np.array_equal(got[128:], y[0][128:])           # the other column block is untouched (block locality of the scale)


def test_fp8_block_promotion_is_ffma_ascending() -> None:
    """The adversarial operands of the H100 FP8 capture's adversarial cases: the H100 words agree with FFMA(temp, FMUL(sfa, sfb), acc)
    in ascending k and disagree with mul+add / descending k (cases_run.json: 160/160, 160/160, 240/240) — here the Definition itself is
    checked to separate the hypotheses on one such case (tile 0 puts acc on a bf16 midpoint, tile 1 adds ~half an f32 ulp)."""
    one, onehalf = 0x38, 0x3C
    K, G = 256, 128
    xq = np.zeros(K, dtype=np.uint8)
    xq[0] = xq[G] = one
    w = np.zeros(K, dtype=np.uint8)
    w[0], w[G] = one, onehalf
    sx = np.array([P.bits_of(np.float32(1.0))] * 2, dtype=np.uint32)
    mid = int(P.bits_of(np.float32(1.0 + 2.0 ** -8)))                 # a bf16 midpoint
    s1 = int(P.bits_of(np.float32(2.0 ** -25))) | 0x2AAAAB              # 2^-25 * RN(4/3): 1.5 * s1 = 2^-24 (1 + 2^-25), inexact
    sw = np.array([mid, s1], dtype=np.uint32)
    prog = Program(bind(F.ScaledMmFp8BlockCoordinate, K=K, G=G))
    got = _evaluate_returns(prog, [xq, sx, w, sw])[0]
    # FFMA keeps the 2^-49 tail: 1 + 2^-8 + 2^-24 + tail rounds UP to 1 + 2^-8 + 2^-23 -> bf16 RNE 1 + 2^-7 = 0x3F81; mul+add (RN(t*s) = 2^-24
    # exactly) lands on the f32 tie -> 1 + 2^-8 -> bf16 tie -> even 1.0 = 0x3F80.
    assert got == 0x3F81
