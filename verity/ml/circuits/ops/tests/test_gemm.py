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


# ---- the BF16 binding of each card ----------------------------------------------------------------------------------------

def test_bf16_gemm_binds_the_registered_step_for_every_card_that_computes_it_and_refuses_any_other() -> None:
    """Each registered BF16 instruction binds the step whose function its model computes: Hopper's one group of 16 on sm_90,
    sm_100 and sm_120, Ampere's two groups of 8 on sm_80 and sm_89. sm_120's binding is `GEMM_HOPPER`, so a GEMM bound on sm_120
    is Hopper's, id for id. A step with another adder width, floor, grouping or format is refused."""
    from dataclasses import replace

    from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
    from verity.ml.circuits.silicon.models import HOPPER_BF16_M16N8K16, HOPPER_E4M3_K32

    want = {"sm80": G.GEMM_AMPERE, "sm89": G.GEMM_AMPERE, "sm90": G.GEMM_HOPPER, "sm100": G.GEMM_HOPPER, "sm120": G.GEMM_HOPPER}
    bf16 = {i: ins for i, ins in INSTRUCTIONS.items() if ins.a_dtype == "bf16" and ins.model is not None}
    assert {"sm80.mma.m16n8k16.bf16", "sm89.mma.m16n8k16.bf16", "sm90.mma.m16n8k16.bf16", "sm90.wgmma.m64n8k16.bf16",
            "sm100.mma.m16n8k16.bf16", "sm120.mma.m16n8k16.bf16"} <= set(bf16)
    for i, ins in bf16.items():
        assert G.bf16_gemm(ins.model) == want[i.split(".")[0]], i
    assert INSTRUCTIONS["sm120.mma.m16n8k16.bf16"].status == "pinned"
    assert G.GEMM_SM120_BF16 == G.GEMM_HOPPER == {"DOT": P.HopperBF16WgmmaDot16}
    for defn, rest in ((G.DotBf16, {}), (G.GemmCoordinate, {}), (G.Gemm, {"N": 2})):
        assert bind(defn, K=128, **rest, **G.GEMM_SM120_BF16).id == bind(defn, K=128, **rest, **G.GEMM_HOPPER).id
    for model in (replace(HOPPER_BF16_M16N8K16, width=25), replace(HOPPER_BF16_M16N8K16, zero_exponent=-132),
                  replace(HOPPER_BF16_M16N8K16, groups=(8, 8)), HOPPER_E4M3_K32):
        with pytest.raises(ValueError, match="no registered BF16 step is its step"):
            G.bf16_gemm(model)


def _attention_rows(rng, n: int, K: int, pad: int):
    """``n`` pairs of BF16 rows of length ``K`` whose last ``pad`` entries are zero codes (a KV length K - pad, zero-padded to a
    multiple of 16): normal magnitudes, and rows with NaN, both infinities, inf x 0, -0 everywhere, subnormals, an exact
    cancellation, and sums past FP32's range with and without sign changes."""
    import numpy as np

    live = K - pad
    def bf16(v):
        return (np.asarray(v, np.float32).view(np.uint32) >> 16).astype(np.uint16)
    x, w = bf16(rng.standard_normal((n, K))), bf16(rng.standard_normal((n, K)))
    x[0, rng.integers(live)] = 0x7FC1
    x[1, rng.integers(live)], w[1] = 0x7F80, 0x3F80
    x[2, :2], w[2, :2] = (0x7F80, 0xFF80), 0x3F80
    x[3, 0], w[3, 0] = 0xFF80, 0x0000
    x[4], w[4] = 0x8000, 0x3F80
    x[5] = rng.integers(1, 0x80, K).astype(np.uint16) | (rng.integers(0, 2, K).astype(np.uint16) << 15)
    w[5] = bf16(rng.uniform(2.0 ** 100, 2.0 ** 120, K))
    half = live // 2
    x[6, half:2 * half], w[6, half:2 * half] = x[6, :half] ^ 0x8000, w[6, :half]
    x[6, 2 * half:live] = 0
    x[7], w[7] = 0x7F7F, 0x7F7F
    x[8], w[8] = 0x7F7F | (np.arange(K, dtype=np.uint16) % 2 << 15), 0x7F7F
    x[:, live:], w[:, live:] = 0, 0
    return x, w


@pytest.mark.parametrize("K,pad", [(16, 15), (64, 0), (128, 0), (1040, 15), (4096, 0)])
def test_the_sm120_coordinate_is_the_registered_models_total_chain_at_attention_shapes(K, pad) -> None:
    """At QK^T's K (head dims 64 and 128) and PV's (a KV length of 1, 1025 and 4096, zero-padded to a multiple of 16),
    `GemmCoordinate_v2` and `DotBf16_v2` bound with `GEMM_SM120_BF16` are the registry's sm_120 model folded over K in
    ascending k from +0 by `total.tc_dot_total`, then `cvt.rn.bf16.f32`, on random rows and on NaN, infinities, -0,
    subnormals, cancellation and overflow; the numpy twin gives the reference's words."""
    import numpy as np

    from verity.ml.circuits import evaluation as E
    from verity.ml.circuits.silicon.instructions import INSTRUCTIONS

    model = INSTRUCTIONS["sm120.mma.m16n8k16.bf16"].model
    x, w = _attention_rows(np.random.default_rng(K), 12, K, pad)
    want_acc = []
    for r in range(len(x)):
        acc = 0
        for s in range(0, K, 16):
            acc = tc_dot_total(model, acc, x[r, s:s + 16].tolist(), w[r, s:s + 16].tolist())
        want_acc.append(acc)
    want = [f32_to_bf16_hw_word(a) for a in want_acc]
    assert want[0] == want[3] == 0x7FFF and want[1] == want[7] == 0x7F80 and want[4] == 0
    assert want[2] == (0x7FFF if K - pad > 1 else 0x7F80)
    coord, dot = bind(G.GemmCoordinate, K=K, **G.GEMM_SM120_BF16), bind(G.DotBf16, K=K, **G.GEMM_SM120_BF16)
    for kernel in ("reference", "auto"):
        assert E.evaluate_batch(coord, x, w, kernel=kernel).astype(np.uint32).tolist() == want, kernel
        assert E.evaluate_batch(dot, x, w, kernel=kernel).astype(np.uint32).tolist() == want_acc, kernel


def test_the_sm120_step_replays_the_rtx_pro_6000_capture() -> None:
    """The RTX PRO 6000's specials, signed-zero and adder-width cancellation tiles (`fixtures/tc-sm120-bf16-2026-10-25`), one
    `mma.sync.m16n8k16` each: every word is `GEMM_SM120_BF16`'s step on its operands, through the step's kernel and, every
    seventh word, the primitive itself. Ampere's step misses 384 of the cancellation family's words."""
    import numpy as np

    from verity.ml.kernels.tc import group_sum_total_batch

    root = Path(__file__).parent / "fixtures" / "tc-sm120-bf16-2026-10-25"
    step = G.GEMM_SM120_BF16["DOT"]
    for name, words in (("tiles_specials", 6144), ("tiles_signed_zero", 32768), ("tiles_adder_width_cancellation", 4096)):
        z = np.load(root / f"{name}.npz")
        A, Bt, C, D = z["A"], z["Bt"], z["C"], z["D"]
        a = np.repeat(A, Bt.shape[1], axis=1).reshape(-1, 16)
        b = np.tile(Bt, (1, A.shape[1], 1)).reshape(-1, 16)
        acc, want = C.reshape(-1), D.reshape(-1)
        assert acc.size == words
        got, _ = group_sum_total_batch(HOPPER_BF16_WGMMA_K16, acc, a, b)
        assert np.array_equal(got, want), name
        for i in range(0, words, 7):
            assert step.evaluate(int(acc[i]), *a[i].tolist(), *b[i].tolist()) == int(want[i]), (name, i)
        if name == "tiles_adder_width_cancellation":
            ampere, _ = group_sum_total_batch(AMPERE_BF16_M16N8K16, acc, a, b)
            assert int((ampere != want).sum()) == 384
