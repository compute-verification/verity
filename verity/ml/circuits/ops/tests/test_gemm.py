"""`GemmCoordinate_v2{K, DOT}` through the oracle on 30 authenticated RTX 4090 GEMM coordinates.

Ported from veritor ``verity.ir/tests/test_composition.py`` §3 (`test_gemm_on_ampere_captured_coordinates`), with the
numpy fixture loading replaced by `array('H')`.  The recorded cross-architecture numbers (Hopper step on the same operands:
same bf16 word on all 30, f32 accumulators differ on 26/30) are pinned as in the source.
"""
from __future__ import annotations

import json
import re
import struct
from array import array
from pathlib import Path

import pytest

from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits.evaluation import evaluate_call
from verity.ml.circuits.program import Program
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.silicon.cast import f32_to_bf16_hw_word
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16, tc_dot_total
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16

GEMM_FIX = Path(__file__).parent / "fixtures" / "gemm-B1-b3eba618b1b5-20260907T1731Z"

_STANDALONE = CompositeDefinition("Standalone", 1, ("FN",), lambda S: S.FN.signature(),
                                  lambda B, S, *args: B.call(S.FN, *args),
                                  doc="root wrapper: one call of FN over Input gates", register=False)


def _standalone(spec):
    """A program whose root is exactly one call of `spec`; returns (program, the FunctionCall)."""
    prog = Program(bind(_STANDALONE, FN=spec))
    calls = [c for c in prog.circuit.root_calls() if not c.is_primitive]
    assert len(calls) == 1, calls
    return prog, calls[0]


def _ev(spec, *arrays) -> tuple[int, ...]:
    """Evaluate one call of `spec` on word arrays (flattened in parameter order) with the oracle."""
    prog, call = _standalone(spec)
    return evaluate_call(prog.circuit, call, [int(v) for a in arrays for v in a])


def _u16(path: Path) -> array:
    a = array("H")
    a.frombytes(path.read_bytes())
    if struct.pack("=H", 1) != b"\x01\x00":  # the files are little-endian
        a.byteswap()
    return a


def _gemm_fixtures():
    for d in sorted(p for p in GEMM_FIX.iterdir() if p.is_dir()):
        f = json.loads((d / "fixture.json").read_text())
        x = _u16(d / "x.u16")
        if f["x_arg_cols"] is not None:
            lo, hi = f["x_arg_cols"]
            x = x[lo:hi]
        yield f, x, _u16(d / "w_row.u16"), _u16(d / "out_row.u16")


def test_thirty_fixtures_present() -> None:
    fs = list(_gemm_fixtures())
    assert len(fs) == 30
    assert {f["family"] for f, *_ in fs} == {"down_proj", "gate_up_proj", "lm_head", "o_proj", "qkv_proj"}
    assert {f["phase"] for f, *_ in fs} == {"prefill", "decode"}
    assert {f["K"] for f, *_ in fs} == {1536, 8960}


def test_gemm_on_ampere_captured_coordinates() -> None:
    """30 authenticated B1 GEMM coordinates (5 families x prefill/decode, K = 1536 / 8960): `GemmCoordinate_v2{K, DOT}` with
    the Ampere step gives the committed word every time. The cross - the Hopper step on the same operands - gives the
    same bf16 word on all 30 (recorded) although the f32 accumulators differ on 26/30: the step difference is below the
    bf16 store rounding on these rows."""
    n = exact = cross_bf16 = cross_f32 = 0
    for f, x, w, out in _gemm_fixtures():
        K, j = f["K"], f["j"]
        assert len(x) == K == len(w)
        ya = _ev(bind(G.GemmCoordinate, K=K, **G.GEMM_AMPERE), x, w)[0]
        yh = _ev(bind(G.GemmCoordinate, K=K, **G.GEMM_HOPPER), x, w)[0]
        aa = _ev(bind(G.DotBf16, K=K, **G.GEMM_AMPERE), x, w)[0]
        ah = _ev(bind(G.DotBf16, K=K, **G.GEMM_HOPPER), x, w)[0]
        n += 1
        exact += int(ya == out[j] == f["committed_word"])
        cross_bf16 += int(yh != out[j])
        cross_f32 += int(aa != ah)
    assert n == 30 and exact == 30
    assert (cross_bf16, cross_f32) == (0, 26), (cross_bf16, cross_f32)  # recorded: veritor composition.md §3


def test_ir_evaluation_is_the_direct_chain() -> None:
    """The oracle's answer for `GemmCoordinate_v2` is literally K/16 `tc_dot_total` steps then `cvt.rn.bf16.f32`: the Program
    adds structure, not semantics.  Also fixes the proof-block shape the proofs target."""
    for f, x, w, out in list(_gemm_fixtures())[:4]:
        K = f["K"]
        acc = 0
        for s in range(K // 16):
            acc = tc_dot_total(AMPERE_BF16_M16N8K16, acc, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
        assert _ev(bind(G.DotBf16, K=K, DOT=P.AmpereBF16TcDot16), x, w) == (acc,)
        assert _ev(bind(G.GemmCoordinate, K=K, DOT=P.AmpereBF16TcDot16), x, w) == (f32_to_bf16_hw_word(acc),)
        acc = 0
        for s in range(K // 16):
            acc = tc_dot_total(HOPPER_BF16_WGMMA_K16, acc, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
        assert _ev(bind(G.DotBf16, K=K, DOT=P.HopperBF16WgmmaDot16), x, w) == (acc,)


def test_gemm_coordinate_circuit_shape() -> None:
    """One proof block = K/16 step gates + one cast gate (+ the zero constant): the shape the census counts over."""
    prog, call = _standalone(bind(G.GemmCoordinate, K=1536, DOT=P.AmpereBF16TcDot16))
    circ = prog.circuit
    ids: dict[str, int] = {}
    for g in circ[call.interval[0]:call.interval[1]]:
        ids[g.prim.id] = ids.get(g.prim.id, 0) + 1
    assert ids == {"Const32[0x00000000]_v1": 1, "AmpereBF16TcDot16_v2": 1536 // 16, "F2fpBf16_v1": 1}, ids
    # the inputs (2 x 1536 bf16 words) are the only gates outside the call
    assert circ.size == len(call) + 2 * 1536
    assert all(circ[i].prim.id == "Input16_v1" for i in range(2 * 1536))
    # a step gate reads the previous accumulator and 32 input words; the cast reads the last step
    steps = [g for g in circ[call.interval[0]:call.interval[1]] if g.prim.id == "AmpereBF16TcDot16_v2"]
    assert all(len(g.operands) == 33 for g in steps)
    for prev, g in zip(steps, steps[1:]):
        assert g.operands[0] == prev.index
    cast = next(g for g in circ[call.interval[0]:call.interval[1]] if g.prim.id == "F2fpBf16_v1")
    assert list(cast.operands) == [steps[-1].index] and call.returns == (cast.index,)


def test_gemm_batches_the_coordinate() -> None:
    """`Gemm_v2{K, N, DOT}` on a tiny problem equals N independent coordinates."""
    import random

    rng = random.Random(3)
    K, N = 32, 3
    x = [rng.getrandbits(16) & 0x7F7F for _ in range(K)]
    w = [[rng.getrandbits(16) & 0x7F7F for _ in range(K)] for _ in range(N)]
    y = _ev(bind(G.Gemm, K=K, N=N, DOT=P.AmpereBF16TcDot16), x, [v for row in w for v in row])
    assert len(y) == N
    for i in range(N):
        assert y[i] == _ev(bind(G.GemmCoordinate, K=K, DOT=P.AmpereBF16TcDot16), x, w[i])[0]


# ---- the sm_120 FP8 / FP4 coordinates -----------------------------------------------------------------------------------

def _fp4_operands(rng, K: int, block: int, lo: int, hi: int):
    return ([rng.getrandbits(4) for _ in range(K)], [rng.getrandbits(4) for _ in range(K)],
            [rng.randrange(lo, hi) for _ in range(K // block)], [rng.randrange(lo, hi) for _ in range(K // block)])


def test_sm120_coordinates_are_the_direct_chain() -> None:
    """Each sm_120 coordinate is its pinned model folded over K in ascending k from +0, then `cvt.rn.bf16.f32`."""
    import random

    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32, BLACKWELL_SM120_MXF4, BLACKWELL_SM120_NVF4
    from verity.ml.circuits.silicon.total_fp8 import tc_dot_total_e4m3

    rng = random.Random(4)
    for _ in range(5):
        K = 128
        x, w = [rng.getrandbits(8) & 0x7E for _ in range(K)], [rng.getrandbits(8) & 0xFE for _ in range(K)]
        acc = 0
        for s in range(K // 32):
            acc = tc_dot_total_e4m3(BLACKWELL_SM120_E4M3_M16N8K32, acc, x[32 * s:32 * s + 32], w[32 * s:32 * s + 32])
        assert _ev(bind(G.GemmCoordinateE4m3, K=K, **G.GEMM_SM120_E4M3), x, w) == (f32_to_bf16_hw_word(acc),)
        for coord, binding, model, lo, hi in ((G.GemmCoordinateNvf4, G.GEMM_SM120_NVF4, BLACKWELL_SM120_NVF4, 0x20, 0x48),
                                              (G.GemmCoordinateMxf4, G.GEMM_SM120_MXF4, BLACKWELL_SM120_MXF4, 110, 140)):
            ops = _fp4_operands(rng, 256, model.block, lo, hi)
            assert _ev(bind(coord, K=256, **binding), *ops) == (f32_to_bf16_hw_word(model.chain(0, *ops)),)


def test_sm120_coordinate_circuit_shapes() -> None:
    """One proof block = K/k step gates + one cast gate (+ the zero constant), each step reading the previous accumulator."""
    for coord, binding, K, k, width in ((G.GemmCoordinateE4m3, G.GEMM_SM120_E4M3, 256, 32, 1 + 2 * 32),
                                        (G.GemmCoordinateNvf4, G.GEMM_SM120_NVF4, 256, 64, 1 + 2 * 64 + 2 * 4),
                                        (G.GemmCoordinateMxf4, G.GEMM_SM120_MXF4, 256, 64, 1 + 2 * 64 + 2 * 2)):
        prog, call = _standalone(bind(coord, K=K, **binding))
        gates = prog.circuit[call.interval[0]:call.interval[1]]
        ids: dict[str, int] = {}
        for g in gates:
            ids[g.prim.id] = ids.get(g.prim.id, 0) + 1
        dot = binding["DOT"].id
        assert ids == {"Const32[0x00000000]_v1": 1, dot: K // k, "F2fpBf16_v1": 1}, ids
        steps = [g for g in gates if g.prim.id == dot]
        assert all(len(g.operands) == width for g in steps)
        for prev, g in zip(steps, steps[1:]):
            assert g.operands[0] == prev.index
        cast = next(g for g in gates if g.prim.id == "F2fpBf16_v1")
        assert list(cast.operands) == [steps[-1].index] and call.returns == (cast.index,)


def test_nvfp4_coordinate_replays_every_recorded_rtx5090_vector() -> None:
    """The 1,601 autoproof dot statements (K = 64 / 128 / 256 from a zero accumulator), through the coordinate: the recorded
    accumulator, then `cvt.rn.bf16.f32`."""
    root = next(p for p in Path(__file__).resolve().parents if (p / "uv.lock").exists())
    cases = json.loads((root / "verity" / "ml" / "circuits" / "silicon" / "captures" / "autoproof-probe-vectors.json").read_text())["nvfp4_m16n8k64"]["cases"]
    assert len(cases) == 1601
    for c in cases[::40]:
        ops = ([int(ch, 16) for ch in c["a"]], [int(ch, 16) for ch in c["b"]], list(bytes.fromhex(c["sa"])), list(bytes.fromhex(c["sb"])))
        assert _ev(bind(G.GemmCoordinateNvf4, K=len(c["a"]), **G.GEMM_SM120_NVF4), *ops) == (f32_to_bf16_hw_word(int(c["d"], 16)),)


def test_the_s8_coordinate_scales_as_steps_int8_linear_and_the_row_is_its_coordinates() -> None:
    """`ScaledMmS8Coordinate_v1` is bf16_rn((f32_rn(acc) * sx) * sw) on the exact integer sum, `benchmarks/ppl/steps.py`'s
    `int8_linear` order (numpy's float32 products), and `ScaledMmS8_v1` is the coordinates over W's rows. Some rows' scales
    are searched for so that `sw` first (CUTLASS c2x's order) rounds to another bf16 word: random scales almost never tell
    the two orders apart."""
    import numpy as np

    from verity.ml.circuits import evaluation as E

    def bf16_rn(v):
        u = v.view(np.uint32).astype(np.uint64)
        return (u + 0x7FFF + ((u >> 16) & 1)) >> 16

    rng = np.random.default_rng(31)
    n, K, N = 300, 128, 3
    x = rng.integers(-128, 128, (n, K)).astype(np.int8)
    w = rng.integers(-128, 128, (n, N, K)).astype(np.int8)
    sx = (np.abs(rng.standard_normal(n)) / 127).astype(np.float32)
    sw = (np.abs(rng.standard_normal((n, N))) * 0.02 / 127).astype(np.float32)
    acc = (x[:, None, :].astype(np.int64) * w).sum(-1).astype(np.int32)
    f = acc[:64, :1].astype(np.float32)
    cx = (np.abs(rng.standard_normal((64, 1 << 16))) / 127).astype(np.float32)
    cw = (np.abs(rng.standard_normal((64, 1 << 16))) * 0.02 / 127).astype(np.float32)
    apart = bf16_rn((f * cx) * cw) != bf16_rn(cx * (cw * f))
    rows = np.flatnonzero(apart.any(1))
    assert len(rows) >= 8
    sx[rows], sw[rows, 0] = cx[rows, apart[rows].argmax(1)], cw[rows, apart[rows].argmax(1)]
    y = (acc.astype(np.float32) * sx[:, None]) * sw
    want = np.vectorize(f32_to_bf16_hw_word, otypes=[np.uint64])(y.view(np.uint32))
    c2x = sx[rows, None] * (sw[rows] * acc[rows].astype(np.float32))
    assert (np.vectorize(f32_to_bf16_hw_word, otypes=[np.uint64])(c2x.view(np.uint32))[:, 0] != want[rows, 0]).all()
    u8, u32 = (lambda a: a.view(np.uint8)), (lambda a: a.view(np.uint32))
    coord = bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM120_S8)
    got = E.evaluate_batch(coord, u8(x), u32(sx), u8(w[:, 0]), u32(sw[:, 0]), kernel="reference")
    assert np.array_equal(got.astype(np.uint64), want[:, 0])
    row = E.evaluate_batch(bind(G.ScaledMmS8, K=K, N=N, **G.GEMM_SM120_S8), u8(x), u32(sx), u8(w), u32(sw), kernel="reference")
    assert np.array_equal(row.astype(np.uint64), want)
    accw = E.evaluate_batch(bind(G.GemmAccumulatorS8, K=K, **G.GEMM_SM120_S8), u8(x), u8(w[:, 0]), kernel="reference")
    assert np.array_equal(accw.astype(np.uint32).view(np.int32), acc[:, 0])


def test_the_row_split_s8_coordinate_is_the_coordinate_on_its_parts_concatenated() -> None:
    """`ScaledMmS8CoordinateRows_v1{K, R}` is `ScaledMmS8Coordinate_v1{K}` on x's and W's row's R parts, for R = 1, 2 and 4.
    At K = 65,536 (the demo's MLP down projection) R = 2 makes each s8 operand one `UNIT_CLASS_MAX` row, where the unsplit
    coordinate's are twice it."""
    import numpy as np

    from verity.ml.circuits import evaluation as E
    from verity.ml.circuits.partition_v2 import UNIT_CLASS_MAX

    rng = np.random.default_rng(33)
    n, K = 400, 128
    x, w = (rng.integers(0, 256, (n, K), dtype=np.uint8) for _ in range(2))
    sx = (np.abs(rng.standard_normal(n)) / 127).astype(np.float32).view(np.uint32)
    sw = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    want = E.evaluate_batch(bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM90_S8), x, sx, w, sw, kernel="reference")
    for R in (1, 2, 4):
        rows = bind(G.ScaledMmS8CoordinateRows, K=K, R=R, **G.GEMM_SM90_S8)
        assert [p for p, _ in rows.signature()[0]] == [*(f"x{r}" for r in range(R)), "sx", *(f"wrow{r}" for r in range(R)), "sw"]
        got = E.evaluate_batch(rows, *np.split(x, R, axis=1), sx, *np.split(w, R, axis=1), sw, kernel="reference")
        assert np.array_equal(got, want), R

    def bits(fn):
        return [sum(t.leaf_widths()) for _, t in fn.signature()[0]]

    assert bits(bind(G.ScaledMmS8CoordinateRows, K=65536, R=2, **G.GEMM_SM90_S8)) == [UNIT_CLASS_MAX] * 2 + [32] + [UNIT_CLASS_MAX] * 2 + [32]
    assert bits(bind(G.ScaledMmS8Coordinate, K=65536, **G.GEMM_SM90_S8)) == [2 * UNIT_CLASS_MAX, 32, 2 * UNIT_CLASS_MAX, 32]


@pytest.mark.parametrize("K,R", [(128, 1), (128, 2), (128, 4), pytest.param(16384, 2, marks=pytest.mark.slow),
                                 pytest.param(65536, 8, marks=pytest.mark.slow)])
def test_the_residual_coordinate_is_the_unfused_coordinate_then_residual_add(K, R) -> None:
    """`ScaledMmS8ResidualCoordinateRows_v1{K, R}` is the chain it fuses: `ScaledMmS8CoordinateRows_v1{K, R}`'s word, then
    `Bf16AddRow_v1{N = 1}` of the residual stream's word r and it. At K = 128 for R = 1, 2 and 4, and at the frozen shape's
    two projections back to the width, wo (K = 16,384, R = 2) and wdown (K = 65,536, R = 8); r random, every special word
    (signed zeros, subnormals, the largest finites, infinities, NaNs) and the coordinate's own word negated (the exact +0)."""
    import numpy as np

    from verity.ml.circuits import evaluation as E
    from verity.ml.circuits.ops import rows as RO

    rng = np.random.default_rng(35 + K + R)
    n = 1000 if K == 128 else 32
    x, w = (rng.integers(0, 256, (n, K), dtype=np.uint8) for _ in range(2))
    sx = (np.abs(rng.standard_normal(n)) / 127).astype(np.float32).view(np.uint32)
    sw = (np.abs(rng.standard_normal(n)) * 0.02 / 127).astype(np.float32).view(np.uint32)
    ports = [*np.split(x, R, axis=1), sx, *np.split(w, R, axis=1), sw]
    y = E.evaluate_batch(bind(G.ScaledMmS8CoordinateRows, K=K, R=R, **G.GEMM_SM90_S8), *ports, kernel="reference")
    y = y.reshape(n).astype(np.uint16)
    special = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0,
                        0x7F81, 0xFFFF], np.uint16)
    s = len(special)
    r = rng.integers(0, 1 << 16, n, dtype=np.uint32).astype(np.uint16)
    r[:s], r[s:2 * s] = special, y[s:2 * s] ^ np.uint16(0x8000)
    want = E.evaluate_batch(bind(RO.Bf16AddRow, N=1), r[:, None], y[:, None], kernel="reference").reshape(n)
    got = E.evaluate_batch(bind(G.ScaledMmS8ResidualCoordinateRows, K=K, R=R, **G.GEMM_SM90_S8), *ports, r, kernel="reference")
    assert np.array_equal(got.reshape(n).astype(np.uint16), want.astype(np.uint16))
    finite = (y[s:2 * s] & 0x7F80) != 0x7F80
    assert finite.any() and (want[s:2 * s][finite] == 0).all()


def test_the_padded_coordinate_is_the_coordinate_on_zero_padded_operands_at_every_padded_k() -> None:
    """`GemmCoordinatePad_v1{T, K}` is `GemmCoordinate_v2{K}` on x and W's row with words T to K - 1 bf16 +0, and the same word at
    every K >= T: a k16 step on +0 products changes only a -0 accumulator (to +0) and a NaN's payload (`F2fpBf16` returns one
    NaN for all), and no step returns -0, so a chain from +0 never holds it. So attention's PV at T <= 32 keys is exact at
    K = 32 whatever k extent the kernel pads its masked keys to."""
    import numpy as np

    from verity.ml.circuits import evaluation as E
    from verity.ml.kernels import tc as KT

    coord = bind(G.GemmCoordinate, K=32, **G.GEMM_HOPPER)
    xs, ws = KT.sample_tc(np.random.default_rng(34), 3000, E.batch.signature_of(coord)[0])
    for T in (1, 5, 16, 17, 31, 32):
        def pad(a):
            return np.concatenate([a[:, :T], np.zeros((len(a), 32 - T), a.dtype)], axis=1)

        want = E.evaluate_batch(coord, pad(xs), pad(ws))
        for k in (32, 48, 64):
            pad_k = bind(G.GemmCoordinatePad, T=T, K=k, **G.GEMM_HOPPER)
            assert np.array_equal(E.evaluate_batch(pad_k, xs[:, :T], ws[:, :T]), want), (T, k)
            ref = E.evaluate_batch(pad_k, xs[:200, :T], ws[:200, :T], kernel="reference")
            assert np.array_equal(ref, want[:200]), (T, k)
    step = G.GEMM_HOPPER["DOT"]
    acc, a, b = KT.sample_tc(np.random.default_rng(35), 50_000, E.batch.signature_of(step)[0])
    edge = np.array([0x80000000, 0, 1, 0x80000001, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFFFFFFF], np.uint32)
    acc = np.concatenate([np.asarray(acc, np.uint32), edge])
    z = np.zeros((len(acc), 16), np.uint16)
    out = E.evaluate_batch(step, acc, z, z).astype(np.uint32)
    nan = (acc & 0x7FFFFFFF) > 0x7F800000
    assert np.array_equal(out[~nan], np.where(acc == 0x80000000, 0, acc)[~nan]) and (out[nan] == 0x7FFFFFFF).all()
    assert (acc == 0x80000000).sum() > 1000
    assert not (E.evaluate_batch(step, acc[:50_000], a, b).astype(np.uint32) == 0x80000000).any()


def test_the_pv_row_is_the_padded_coordinate_on_each_of_v_s_columns() -> None:
    """`PvRow_v1{T, K=32, D=64}`'s dim d is `GemmCoordinatePad_v1{T, K=32}` of p and (v0[d], ..., v{T-1}[d]), at T = 1, 8 and
    32, on the reference; the registered kernel gives the same words. Each key row is one 1,024-bit (128-byte) port, over the
    111 bytes C-Flock needs of some row of a unit, and the output is one 1,024-bit row."""
    import numpy as np

    from verity.ml.circuits import evaluation as E
    from verity.ml.kernels import tc as KT

    n, D = 6, 64
    coord = bind(G.GemmCoordinate, K=32, **G.GEMM_HOPPER)
    xs, ws = KT.sample_tc(np.random.default_rng(36), n * D, E.batch.signature_of(coord)[0])
    for T in (1, 8, 32):
        p, v = xs[:n, :T], ws.reshape(n, D, 32)[:, :, :T].transpose(0, 2, 1)
        vs = [np.ascontiguousarray(v[:, t]) for t in range(T)]
        row = bind(G.PvRow, T=T, K=32, D=D, **G.GEMM_HOPPER)
        got = E.evaluate_batch(row, p, *vs, kernel="reference").reshape(n, D)
        pad = bind(G.GemmCoordinatePad, T=T, K=32, **G.GEMM_HOPPER)
        cols = [E.evaluate_batch(pad, p, np.ascontiguousarray(v[:, :, d]), kernel="reference").reshape(n) for d in range(D)]
        assert np.array_equal(got, np.stack(cols, axis=1)), T
        assert np.array_equal(E.evaluate_batch(row, p, *vs).reshape(n, D), got), T
        params, ret = row.signature()
        assert [(name, sum(t.leaf_widths())) for name, t in params] == [("p", 16 * T), *((f"v{t}", 1024) for t in range(T))]
        assert sum(ret.leaf_widths()) == 1024


def test_the_padded_pv_row_is_the_pv_row_at_its_key_count() -> None:
    """At each key count n in {1, 2, 5, 17, 31, 32} of T = 32 (K = 32, D = 64), on the reference:

    - `PvRow_v1{T=32}` on p padded with +0 words and V padded with all-zero key rows is `PvRow_v1{T=n}` on the first n keys,
      word for word (both are `GemmCoordinate_v2{K=32}` on the same operands), but one inf word in an absent key's V row
      makes that dim 0x7FFF on every head (+0 x inf is a NaN product), so that padding is exact only on zero rows;
    - `PvRowPad_v1{T=32}` at n is `PvRow_v1{T=n}` whatever the absent keys hold, here inf p words and inf and NaN V words,
      on the reference and on its kernel."""
    import numpy as np

    from verity.ml.circuits import evaluation as E
    from verity.ml.kernels import tc as KT

    heads, D, T = 8, 64, 32
    coord = bind(G.GemmCoordinate, K=32, **G.GEMM_HOPPER)
    xs, ws = KT.sample_tc(np.random.default_rng(37), heads * D, E.batch.signature_of(coord)[0])
    p, v = xs[:heads], ws.reshape(heads, D, 32).transpose(0, 2, 1)
    rows = lambda a: [np.ascontiguousarray(a[:, t]) for t in range(T)]  # noqa: E731
    full = bind(G.PvRow, T=T, K=32, D=D, **G.GEMM_HOPPER)
    masked = bind(G.PvRowPad, T=T, K=32, D=D, **G.GEMM_HOPPER)
    for n in (1, 2, 5, 17, 31, 32):
        want = E.evaluate_batch(bind(G.PvRow, T=n, K=32, D=D, **G.GEMM_HOPPER), p[:, :n], *rows(v)[:n],
                                kernel="reference").reshape(heads, D)
        keep = np.arange(T) < n
        p0, v0 = np.where(keep, p, 0).astype(p.dtype), np.where(keep[None, :, None], v, 0).astype(v.dtype)
        assert np.array_equal(E.evaluate_batch(full, p0, *rows(v0), kernel="reference").reshape(heads, D), want), n
        if n < T:
            inf = v0.copy()
            inf[:, n, 0] = 0x7F80
            got = E.evaluate_batch(full, p0, *rows(inf), kernel="reference").reshape(heads, D)
            assert (got[:, 0] == 0x7FFF).all() and np.array_equal(got[:, 1:], want[:, 1:]), n
        junk_p, junk_v = p.copy(), v.copy()
        junk_p[:, n:] = 0x7F80
        junk_v[:, n:, 0], junk_v[:, n:, 1] = 0x7F80, 0x7FC0
        counts = np.full(heads, n, np.uint16)
        got = E.evaluate_batch(masked, junk_p, *rows(junk_v), counts, kernel="reference").reshape(heads, D)
        assert np.array_equal(got, want), n
        assert np.array_equal(E.evaluate_batch(masked, junk_p, *rows(junk_v), counts).reshape(heads, D), want), n
    params, ret = masked.signature()
    assert [(name, sum(t.leaf_widths())) for name, t in params] == [("p", 512), *((f"v{t}", 1024) for t in range(T)), ("n", 16)]
    assert sum(ret.leaf_widths()) == 1024


def test_the_padded_pv_row_keeps_the_keys_under_its_unsigned_count() -> None:
    """`PvRowPad_v1{T=4}` keeps key t exactly when t < n, n unsigned: with p all 1.0 and key t's V row 2^t, the output is
    2^min(n, 4) - 1 in every dim, on its kernel at every u16 n and on the reference at n = 0..9 and at 0x7FFF, 0x8000,
    0x8001 and 0xFFFF (words whose i32 widening `Bf16ToF32_v1` makes negative)."""
    import numpy as np

    from verity.ml.circuits import evaluation as E

    T, D = 4, 2
    pad = bind(G.PvRowPad, T=T, K=32, D=D, **G.GEMM_HOPPER)
    prefix = np.array([0x0000, 0x3F80, 0x4040, 0x40E0, 0x4170])  # +0, 1, 3, 7, 15
    for kernel, counts in ((None, np.arange(1 << 16)), ("reference", np.array([*range(10), 0x7FFF, 0x8000, 0x8001, 0xFFFF]))):
        counts = counts.astype(np.uint16)
        p = np.full((len(counts), T), 0x3F80, np.uint16)
        vs = [np.full((len(counts), D), (127 + t) << 7, np.uint16) for t in range(T)]
        got = E.evaluate_batch(pad, p, *vs, counts, **({"kernel": kernel} if kernel else {})).reshape(len(counts), D)
        assert np.array_equal(got, np.repeat(prefix[np.minimum(counts, T)][:, None], D, axis=1)), kernel


def test_the_padded_pv_row_reads_its_key_count_as_a_position() -> None:
    """`PvRowPad`'s n is a position port (`KEY_COUNT`), so it is a run input declared kind position passed through untouched:
    a unit over it builds, and a Program that computes n, or passes an input it doesn't declare a position, is refused."""
    from verity.ml.circuits.ports import PortError
    from verity.ml.circuits.program import Program

    pad = bind(G.PvRowPad, T=2, K=32, D=64, **G.GEMM_HOPPER)
    assert G.PvRowPad.ports is G.KEY_COUNT and G.KEY_COUNT.params["n"].kind == "position"
    Program(pad)
    sig = pad.signature()
    computed = CompositeDefinition("PvRowPadComputedCount", 1, (), lambda S: sig,
                                   lambda B, S, *args: B.call(pad, *args[:-1], B.call(P.F32ToBf16Rn, B.call(P.Bf16ToF32, args[-1]))),
                                   ports=G.KEY_COUNT, register=False)
    with pytest.raises(PortError, match="E_PORT_KIND.*parameter 'n'.*never computes its position"):
        Program(computed)
    undeclared = CompositeDefinition("PvRowPadUndeclaredCount", 1, (), lambda S: sig, lambda B, S, *args: B.call(pad, *args),
                                     register=False)
    with pytest.raises(PortError, match="E_PORT_KIND.*parameter 'n'.*reads a run input"):
        Program(undeclared)


def test_the_padded_row_split_and_pv_forms_refuse_t_r_and_d_outside_their_range() -> None:
    for T in (0, -1, 33):
        with pytest.raises(ValueError, match=re.escape(f"T={T} is not in 1..K=32")):
            bind(G.GemmCoordinatePad, T=T, K=32, **G.GEMM_HOPPER)
        for pv in (G.PvRow, G.PvRowPad):
            with pytest.raises(ValueError, match=re.escape(f"T={T} is not in 1..K=32")):
                bind(pv, T=T, K=32, D=64, **G.GEMM_HOPPER)
    for R in (0, 3, 256):
        for rows in (G.ScaledMmS8CoordinateRows, G.ScaledMmS8ResidualCoordinateRows):
            with pytest.raises(ValueError, match=re.escape(f"R={R} does not divide K=128")):
                bind(rows, K=128, R=R, **G.GEMM_SM90_S8)
    for D in (0, -64):
        for pv in (G.PvRow, G.PvRowPad):
            with pytest.raises(ValueError, match=re.escape(f"D={D} is not a head width")):
                bind(pv, T=8, K=32, D=D, **G.GEMM_HOPPER)


# ---- the K a Definition binds ---------------------------------------------------------------------------------------------

#: each GEMM Definition, its step width k and the rest of a binding
FAMILIES = [(G.DotBf16, 16, G.GEMM_HOPPER), (G.GemmCoordinate, 16, G.GEMM_AMPERE),
            (G.GemmCoordinatePad, 16, {"T": 1, **G.GEMM_HOPPER}), (G.PvRow, 16, {"T": 1, "D": 2, **G.GEMM_HOPPER}),
            (G.PvRowPad, 16, {"T": 1, "D": 2, **G.GEMM_HOPPER}), (G.Gemm, 16, {"N": 3, **G.GEMM_AMPERE}),
            (G.GemmCoordinateE4m3, 32, G.GEMM_SM120_E4M3), (G.GemmCoordinateNvf4, 64, G.GEMM_SM120_NVF4),
            (G.GemmCoordinateMxf4, 64, G.GEMM_SM120_MXF4), (G.GemmAccumulatorS8, 32, G.GEMM_SM120_S8),
            (G.ScaledMmS8Coordinate, 32, G.GEMM_SM120_S8), (G.ScaledMmS8, 32, {"N": 2, **G.GEMM_SM120_S8}),
            (G.ScaledMmS8CoordinateRows, 32, {"R": 1, **G.GEMM_SM90_S8}),
            (G.ScaledMmS8ResidualCoordinateRows, 32, {"R": 1, **G.GEMM_SM90_S8})]
S8_FAMILIES = FAMILIES[9:]


@pytest.mark.parametrize("defn,k,rest", FAMILIES, ids=[d.name for d, *_ in FAMILIES])
def test_bind_refuses_a_k_that_is_not_a_multiple_of_the_step(defn, k, rest) -> None:
    """K = k + 1 (33 for the s8 Definitions) and K = 3k/2 would lose their last products to `range(K // k)`. The refusal says
    the multiple to pad to, whether K is bound with the step or after it."""
    for K in (k + 1, 3 * k // 2):
        why = (f"{defn.id} refuses this binding: K={K} is not a multiple of the {k}-wide step, so K // {k} steps would drop "
               f"the last {K % k} of its {K} products: pad every operand with zero codes to K={2 * k} ")
        with pytest.raises(ValueError, match=re.escape(why)):
            bind(defn, K=K, **rest)
        with pytest.raises(ValueError, match=re.escape(why)):
            bind(defn, **rest).bind(K=K)


@pytest.mark.parametrize("defn,k,rest", S8_FAMILIES, ids=[d.name for d, *_ in S8_FAMILIES])
def test_the_s8_definitions_refuse_k_from_2_17(defn, k, rest) -> None:
    """From `S8_K_LIMIT` = 2^17 an s8 sum can leave s32 (`test_integer_mma`), and K = 2^17 - 32 is the largest K that binds.
    The limit is the s8 Definitions' alone."""
    assert G.S8_K_LIMIT == 1 << 17
    for K in (1 << 17, (1 << 17) + 32, 1 << 20):
        with pytest.raises(ValueError, match=re.escape(f"K={K} is at or past 131072, the least K whose sum can leave the s32 "
                                                       f"accumulator")) as e:
            bind(defn, K=K, **rest)
        assert str(e.value).endswith("Split the k range into accumulators of at most K=131040 each")
    assert bind(defn, K=(1 << 17) - 32, **rest).complete
    assert bind(G.GemmCoordinateE4m3, K=1 << 20, **G.GEMM_SM120_E4M3).complete


#: (gates, standalone program digest) at valid K, recorded before `bind` refused any K: the refusals change no gate of a K
#: they admit. The s8 Definitions at K = 2^17 - 32, the largest they admit.
VALID = {
    bind(G.DotBf16, K=48, **G.GEMM_HOPPER): (4, "0ff626e0396e9b12f29c519e092261491771bc1403e63195431b9592220c17c64a3853633c2ca9a0f00e7783444b6fe17e8159f79d0a349c55551e7fc352730a"),
    bind(G.GemmCoordinate, K=8960, **G.GEMM_AMPERE): (562, "239bc2a3c130cb7496b07bb590c4a2bf55957b6214b17a9ff629abd5b9c83211eb4e484d7440a2815407528ff5c64e623b9470dba081874f8e2df9c8792d9b3e"),
    bind(G.Gemm, K=1536, N=3, **G.GEMM_AMPERE): (294, "588acc04cbf29bd700425da767dba4c3a5f7aa8fc9a6407c30de4251895151378c86d2f08db89302ba661656780b2e416c92db938bffd458b95e4a59aaac6c55"),
    bind(G.GemmCoordinateE4m3, K=8192, **G.GEMM_SM120_E4M3): (258, "b9a41ad5ff51169a596a3b73d5e64bc2ee29dce9562ea7c318c8c7f42459a8621bbf5573b9afc0354a38b4efd2bb8bfc309c29d62083f211d6ba9d88dd6a938b"),
    bind(G.GemmCoordinateNvf4, K=2048, **G.GEMM_SM120_NVF4): (34, "546196109f4e09acc4b44308a38098cf5295f3c359df7cf364b45af67c95b005ff95d02cdc73e7a9724c93b7c3e712a706a27dfc1406c0992463015ceca6bfcc"),
    bind(G.GemmCoordinateMxf4, K=2048, **G.GEMM_SM120_MXF4): (34, "26e9e86cb631488f8531c90d6f5e60eaa90ca3e488d43633fdfbe80a884cca7d7198984bb5573fbe4d24e49c9110e40c2748f0300e8ac4ebbd171a6399f92bbc"),
    bind(G.GemmAccumulatorS8, K=131040, **G.GEMM_SM120_S8): (4096, "6c58a4fdb90b9e4029279a9a762e9370f102143753eaa4d262be60a456a8abb3c1a941ab01051f71eb8b5f089d1f3a5f996c67c9ea2b8e6eeb1746d54f662991"),
    bind(G.ScaledMmS8Coordinate, K=131040, **G.GEMM_SM120_S8): (4100, "dfa0738a72365f7920185ffdb6da23d8d17ffd6fb4ba384c5c5cb1f0833cf16e1ca1c14cd78c75c8e3199d531cb438ecaa1cba5b0a1e328fed08ed29314fcabd"),
    bind(G.ScaledMmS8, K=131040, N=2, **G.GEMM_SM120_S8): (8200, "5eaef04a9fbb5fea1e993d98afb123c640c232b2b7ff44536f4966cff6aa481c3898b0210fc1e4e9c710aaddbf2411f7fa29c084bf4c27d07931959ac25bdc90"),
}


@pytest.mark.parametrize("spec", VALID, ids=[s.id for s in VALID])
def test_a_valid_k_binds_to_its_recorded_digest(spec) -> None:
    prog, _ = standalone_call(spec)
    assert (spec.gates, codec.program_digest(codec.encode_program(prog))) == VALID[spec]


# ---- the s8 binding of each card ------------------------------------------------------------------------------------------

def _chain(model, a, b) -> int:
    """``model``'s step chained over one row from a zero accumulator, as a word."""
    c = 0
    for i in range(0, len(a), model.k):
        c = model.step(c, a[i:i + model.k], b[i:i + model.k])
    return c


def test_s8_gemm_binds_the_registered_step_for_every_card_that_computes_it_and_refuses_any_other() -> None:
    """sm_90's and sm_100's s8 steps are `BLACKWELL_SM120_S8_M16N8K32`'s function, so each binds `BlackwellS8ImmaDot32_v1`, and
    a GEMM bound on sm_90 is sm_120's, id for id. A wrapping, u8 or k = 16 step is another function and is refused."""
    from dataclasses import replace

    from verity.ml.circuits.silicon.integer import (BLACKWELL_SM100_S8_K32, BLACKWELL_SM120_S8_M16N8K32,
                                                    BLACKWELL_SM120_S8_M16N8K32_WRAP, BLACKWELL_SM120_U8S8_M16N8K32,
                                                    HOPPER_SM90_S8_K32)
    for model in (HOPPER_SM90_S8_K32, BLACKWELL_SM100_S8_K32, BLACKWELL_SM120_S8_M16N8K32):
        assert G.s8_gemm(model) == G.GEMM_SM120_S8 == {"DOT": P.BlackwellS8ImmaDot32}
    assert G.GEMM_SM90_S8 == G.GEMM_SM120_S8
    for defn, rest in ((G.GemmAccumulatorS8, {}), (G.ScaledMmS8Coordinate, {}), (G.ScaledMmS8, {"N": 2})):
        assert bind(defn, K=512, **rest, **G.GEMM_SM90_S8).id == bind(defn, K=512, **rest, **G.GEMM_SM120_S8).id
    for model in (BLACKWELL_SM120_S8_M16N8K32_WRAP, replace(HOPPER_SM90_S8_K32, satfinite=False),
                  BLACKWELL_SM120_U8S8_M16N8K32, replace(HOPPER_SM90_S8_K32, k_per_instruction=16)):
        with pytest.raises(ValueError, match="BlackwellS8ImmaDot32_v1 is not its step"):
            G.s8_gemm(model)


@pytest.mark.parametrize("K", [512, 1536, 2048, 6144])
def test_the_sm90_accumulator_is_hoppers_step_chain_bit_for_bit(K) -> None:
    """`GemmAccumulatorS8` bound with `GEMM_SM90_S8`, at Ks H1 serves on sm_90, on random int8 rows and on the rows of largest
    products (-128 x -128, -128 x 127, 127 x 127, mixed), is `HOPPER_SM90_S8_K32`'s step chained over K/32 instructions, the
    exact sum, and the chain without ``.satfinite`` (h1u's ``mma.sync``): below 2^17 no partial sum leaves s32."""
    from dataclasses import replace

    import numpy as np

    from verity.ml.circuits import evaluation as E
    from verity.ml.circuits.silicon.integer import HOPPER_SM90_S8_K32, s32_value

    rng = np.random.default_rng(K)
    x = rng.integers(-128, 128, (24, K)).astype(np.int8)
    w = rng.integers(-128, 128, (24, K)).astype(np.int8)
    x[0], w[0] = -128, -128
    x[1], w[1] = -128, 127
    x[2], w[2] = 127, 127
    x[3], w[3] = rng.choice([-128, 127], K), rng.choice([-128, 127], K)
    got = E.evaluate_batch(bind(G.GemmAccumulatorS8, K=K, **G.GEMM_SM90_S8), x.view(np.uint8), w.view(np.uint8),
                           kernel="reference").astype(np.uint32)
    wrap = replace(HOPPER_SM90_S8_K32, satfinite=False)
    for r in range(len(x)):
        a, b = x[r].view(np.uint8).tolist(), w[r].view(np.uint8).tolist()
        exact = int((x[r].astype(np.int64) * w[r]).sum())
        assert s32_value(int(got[r])) == s32_value(_chain(HOPPER_SM90_S8_K32, a, b)) == s32_value(_chain(wrap, a, b)) == exact
    assert s32_value(int(got[0])) == 16384 * K and s32_value(int(got[1])) == -16256 * K


def test_the_sm90_steps_agree_up_to_the_largest_k_and_part_at_2_17() -> None:
    """At K = 2^17 - 512, the largest K a multiple of 512 below `S8_K_LIMIT`, all -128 rows sum to 2,139,095,040 under the
    clamped and the wrapping chain alike. At K = 2^17 the sum 2^31 leaves s32: the clamped chain gives 2^31 - 1, the wrapping
    one -2^31, and the s8 Definitions refuse that K, so no bound GEMM tells the two sm_90 forms apart."""
    from dataclasses import replace

    from verity.ml.circuits.silicon.integer import HOPPER_SM90_S8_K32, s32_value

    wrap = replace(HOPPER_SM90_S8_K32, satfinite=False)
    K = (1 << 17) - 512
    row = [0x80] * K
    assert s32_value(_chain(HOPPER_SM90_S8_K32, row, row)) == s32_value(_chain(wrap, row, row)) == 16384 * K == 2_139_095_040
    row = [0x80] * (1 << 17)
    assert (s32_value(_chain(HOPPER_SM90_S8_K32, row, row)), s32_value(_chain(wrap, row, row))) == ((1 << 31) - 1, -(1 << 31))
    with pytest.raises(ValueError, match="at or past 131072"):
        bind(G.GemmAccumulatorS8, K=1 << 17, **G.GEMM_SM90_S8)
