"""`GemmCoordinate_v2{K, DOT}` through the IR oracle on 30 authenticated RTX 4090 GEMM coordinates.

Ported from veritor ``verity.ir/tests/test_composition.py`` §3 (`test_gemm_on_ampere_captured_coordinates`), with the
numpy fixture loading replaced by `array('H')`.  The recorded cross-architecture numbers (Hopper step on the same operands:
same bf16 word on all 30, f32 accumulators differ on 26/30) are pinned as in the source.
"""
from __future__ import annotations

import json
import struct
from array import array
from pathlib import Path


from verity.ml.circuits.defs import CompositeDefinition, bind
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
