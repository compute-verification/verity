"""`GemmCoordinate_v2{K, DOT}` through the IR oracle on 30 authenticated RTX 4090 GEMM coordinates.

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
from verity.ml.circuits.evaluate import evaluate_call
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
    """Evaluate one call of `spec` on word arrays (flattened in parameter order) with the IR oracle."""
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
    """The oracle's answer for `GemmCoordinate_v2` is literally K/16 `tc_dot_total` steps then `cvt.rn.bf16.f32`: the IR
    adds structure, not semantics.  Also fixes the verification-unit shape the proof work targets."""
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
    """One proof unit = K/16 step gates + one cast gate (+ the zero constant): the shape the census counts over."""
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
    """One proof unit = K/k step gates + one cast gate (+ the zero constant), each step reading the previous accumulator."""
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


# ---- the K a Definition binds ---------------------------------------------------------------------------------------------

#: each GEMM Definition, its step width k and the rest of a binding
FAMILIES = [(G.DotBf16, 16, G.GEMM_HOPPER), (G.GemmCoordinate, 16, G.GEMM_AMPERE), (G.Gemm, 16, {"N": 3, **G.GEMM_AMPERE}),
            (G.GemmCoordinateE4m3, 32, G.GEMM_SM120_E4M3), (G.GemmCoordinateNvf4, 64, G.GEMM_SM120_NVF4),
            (G.GemmCoordinateMxf4, 64, G.GEMM_SM120_MXF4), (G.GemmAccumulatorS8, 32, G.GEMM_SM120_S8),
            (G.ScaledMmS8Coordinate, 32, G.GEMM_SM120_S8), (G.ScaledMmS8, 32, {"N": 2, **G.GEMM_SM120_S8})]
S8_FAMILIES = FAMILIES[6:]


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
