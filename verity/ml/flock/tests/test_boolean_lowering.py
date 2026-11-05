"""Flock lowers the Boolean BF16 Definitions (`verity.ml.circuits.ops.boolean.gemm`) and NVFP4 quantization's casts and the bit AND
(`verity.ml.circuits.ops.boolean.scalar`) by their builders: each step and cast and the AND lower to exactly the circuit of the piece
their word primitive lowers to (the same ANDs and the same nonzeros of the AND rows), and the coordinate to
`GemmCoordinate_v2`'s, whose words it evaluates to. A Boolean Definition whose builder commits keeps its commits as rows, so
its AND operands stay a few bits however many activations are chained."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import gemm as G
from verity.ml.kernels import tc as K
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops import scalar as S
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import scalar as BSC
from verity.ml.circuits.types import Array
from verity_flock import ir_lower as IL
from verity_flock.gf2 import Circuit


def _lower(target, widths):
    C = Circuit(cse=True)
    ins = [C.input(f"in{j}", w) for j, w in enumerate(widths)]
    outs = IL.lower(C, target, ins)
    C.output("out", [b for o in outs for b in o])
    return C, outs


@pytest.mark.parametrize("boolean,word,widths", [
    (BG.F2fpBf16, P.F2fpBf16, [32]),
    (BG.AmpereBF16TcDot16, P.AmpereBF16TcDot16, [32] + [16] * 32),
    (BG.HopperBF16WgmmaDot16, P.HopperBF16WgmmaDot16, [32] + [16] * 32),
    (BSC.E4m3ToF32, P.E4m3ToF32, [8]),
    (BSC.F32ToE2m1Sat, P.F32ToE2m1Sat, [32]),
    (BSC.BitAnd, S.BitAnd, [1, 1]),
], ids=["f2fp", "ampere", "hopper", "e4m3-to-f32", "f32-to-e2m1", "bit-and"])
def test_a_traced_definition_lowers_to_its_pieces_circuit(boolean, word, widths) -> None:
    Cw, _ = _lower(word, widths)
    Cb, _ = _lower(bind(boolean), [1] * sum(widths))
    cw, cb = Cw.counts(), Cb.counts()
    assert (cb["ands"], cb["nnz_AB"]) == (cw["ands"], cw["nnz_AB"])
    assert cb["ands"] == boolean.traced().counts["and"]


#: (step binding, ANDs of GemmCoordinate_v2's lowering) at K = 16 and K = 64
COORDINATES = [("ampere", G.GEMM_AMPERE, BG.GEMM_AMPERE, {16: 8034, 64: 32814}),
               ("hopper", G.GEMM_HOPPER, BG.GEMM_HOPPER, {16: 7639, 64: 31255})]


@pytest.mark.parametrize("name,word_b,bool_b,v2", COORDINATES, ids=[c[0] for c in COORDINATES])
def test_the_coordinate_lowers_to_gemm_coordinate_v2s_circuit(name, word_b, bool_b, v2) -> None:
    """The first step's builder runs on the zero accumulator's constant forms, as the word primitive's piece does, so its
    branches on constants (`fp`, `tc_step`) drop the same ANDs."""
    for k, ands in v2.items():
        cw = _lower(bind(G.GemmCoordinate, K=k, **word_b), [16] * (2 * k))[0].counts()
        cb = _lower(bind(BG.GemmCoordinate, K=k, **bool_b), [1] * (2 * k * 16))[0].counts()
        assert cw["ands"] == ands
        assert (cb["ands"], cb["nnz_AB"]) == (cw["ands"], cw["nnz_AB"])


def test_the_lowered_coordinate_computes_gemm_coordinate_v2() -> None:
    k, T = 32, 256
    word = bind(G.GemmCoordinate, K=k, **G.GEMM_AMPERE)
    params, _ = E.batch.signature_of(word)
    x, w = K.sample_tc(np.random.default_rng(3), T, params)
    want = E.evaluate_batch(word, x, w).astype(np.uint64)
    C, outs = _lower(bind(BG.GemmCoordinate, K=k, **BG.GEMM_AMPERE), [1] * (2 * k * 16))
    bits = np.concatenate([x, w], axis=1).astype(np.uint64)
    inputs = {f"in{j}": IL.slice_words(bits[:, j // 16] >> np.uint64(j % 16) & np.uint64(1), 1) for j in range(2 * k * 16)}
    got_s, fail = C.evaluate(inputs, T)
    assert not fail
    assert np.array_equal(IL.unslice_words(got_s["out"], T), want)


W = 16
WORD = Array(W, BL.BIT)


def _mix(C, s, x):
    """A round whose state an XOR would carry from use to use: each bit the parity of two state bits and one of x's, plus
    the AND of two such parities, committed."""
    t = [s[i] ^ s[(i + 1) % W] ^ x[i] for i in range(W)]
    return C.commit([t[i] ^ C.AND(t[(i + 1) % W], t[(i + 3) % W]) for i in range(W)])


MIX = definition("TestCommitMix", 1, _mix, (("s", WORD), ("x", WORD)), WORD, register=False)


def _chain(B, S, s, x):
    for i in range(S.N):
        s = B.call(MIX, s, x[i])
    return s


CHAIN = CompositeDefinition("TestCommitMixChain", 1, ("N",), lambda S: ((("s", WORD), ("x", Array(S.N, WORD))), WORD), _chain,
                            register=False)
SPREAD = CompositeDefinition("TestCommitMixSpread", 1, ("N",), lambda S: ((("s", Array(S.N, WORD)), ("x", Array(S.N, WORD))),
                                                                         Array(S.N, WORD)),
                             lambda B, S, s, x: B.batch(MIX, s, x, axes=(0, 0)), register=False)


def _bits(rng, T, *shape):
    return rng.integers(0, 2, size=(T, *shape), dtype=np.uint64)


def _pack(bits):
    return (bits << np.arange(bits.shape[-1], dtype=np.uint64)).sum(axis=-1)


def _lower_chain(n, reads=False):
    """``CHAIN{N=n}`` lowered into one circuit, one input a bit (every gate of the call read when ``reads``, so every
    activation gate by gate)."""
    d = bind(CHAIN, N=n)
    C = Circuit()
    _prog, call = IL.standalone(d)
    wires = {a: C.input(f"in{j}", 1) for j, a in enumerate(call.args)}
    IL.lower_gates(C, call, wires, reads=call.gate_indices() if reads else ())
    C.output("out", [wires[r][0] for r in call.returns])
    return d, C


def test_a_committing_definition_lowers_by_its_builder_and_its_nonzeros_per_and_stay_flat() -> None:
    """Chained, the builder's commits are rows: n activations are n times one (its trace's ANDs and a commit a bit), every AND
    operand a few bits. Lowered gate by gate, as the traced gates alone give it, the same ANDs less the commits read forms
    that grow with every round."""
    per = MIX.traced().counts["and"]
    T, rng = 64, np.random.default_rng(20261016)
    nnz = {}
    for reads in (False, True):
        for n in (1, 2, 4):
            d, C = _lower_chain(n, reads)
            c = C.counts()
            assert c["ands"] == n * (per + (0 if reads else W))
            nnz[reads, n] = c["nnz_AB"] / c["ands"]
            s, x = _bits(rng, T, W), _bits(rng, T, n, W)
            cols = np.concatenate([s, x.reshape(T, -1)], axis=1)
            got_s, fail = C.evaluate({f"in{j}": IL.slice_words(cols[:, j], 1) for j in range(cols.shape[1])}, T)
            assert not fail
            assert np.array_equal(IL.unslice_words(got_s["out"], T), _pack(E.evaluate_batch(d, s, x).astype(np.uint64)))
    assert nnz[False, 1] == nnz[False, 2] == nnz[False, 4]
    assert nnz[True, 4] > 2 * nnz[True, 1]


def test_a_unit_lowering_keeps_its_activations_commits() -> None:
    """`class_statement.unit_lowering`, a PoUW statement's lowering, on u16 rows: the chain's rows are its ANDs and commits,
    and its laid-out rows compute the reference's outputs."""
    from verity_flock import class_statement as CS

    n, T = 4, 64
    d = bind(CHAIN, N=n)
    low, out_words = CS.unit_lowering(d, "unit/test-commit-mix", [("s", ["s"]), ("x", ["x"])])
    assert out_words == [1]
    assert low.layout.counts["ands"] == n * (MIX.traced().counts["and"] + W)
    rng = np.random.default_rng(7)
    s, x = _bits(rng, T, W), _bits(rng, T, n, W)
    got, _cut, ok = low.evaluate(np.concatenate([_pack(s)[:, None], _pack(x)], axis=1))
    assert int(ok) == (1 << T) - 1
    assert np.array_equal(got[:, 0], _pack(E.evaluate_batch(d, s, x).astype(np.uint64)))


def test_a_template_unit_holding_an_activation_lowers_it_by_its_builder() -> None:
    """`ir_lower.units` splits a batch of activations into one unit each, and the unit circuit is the builder's."""
    n, T = 4, 32
    d = bind(SPREAD, N=n)
    low = IL.lowering(d, "unit/test-commit-mix-spread")
    assert low.units.n == n and low.layout.counts["ands"] == MIX.traced().counts["and"] + W
    rng = np.random.default_rng(11)
    s, x = _bits(rng, T, n, W), _bits(rng, T, n, W)
    got, _cut, ok = low.evaluate(np.concatenate([s.reshape(T, -1), x.reshape(T, -1)], axis=1))
    assert int(ok) == (1 << (T * n)) - 1
    assert np.array_equal(got, E.evaluate_batch(d, s, x).astype(np.uint64).reshape(T, -1))


def _rounds(C, S, s, *x):
    for i in range(S.N):
        s = _mix(C, s, x[i])
    return s


def _rounds_body(B, S, s, x):
    ps, ret = [WORD, Array(S.N, WORD)], WORD
    return TR.emit(TR.trace(lambda C, *e: _rounds(C, S, *e), ps, ret), ps, ret, B, s, x)


#: N rounds of `_mix` in one Boolean Definition with a static, built as `sampling_boolean.traced_family` builds its own: the
#: builder takes ``S`` before the elements
ROUNDS = CompositeDefinition("TestCommitMixRounds", 1, ("N",), lambda S: ((("s", WORD), ("x", Array(S.N, WORD))), WORD),
                             _rounds_body, register=False)
ROUNDS.build = _rounds  # type: ignore[attr-defined]


def test_a_definition_with_statics_lowers_by_its_builder_on_them() -> None:
    """The lowering runs ``build(C, S, *elements)``: N rounds keep N rounds' commits, and compute the reference."""
    per, T, rng = MIX.traced().counts["and"], 64, np.random.default_rng(5)
    for n in (1, 2, 4):
        d = bind(ROUNDS, N=n)
        C, _ = _lower(d, [1] * ((n + 1) * W))
        assert C.counts()["ands"] == n * (per + W)
        s, x = _bits(rng, T, W), _bits(rng, T, n, W)
        cols = np.concatenate([s, x.reshape(T, -1)], axis=1)
        got_s, fail = C.evaluate({f"in{j}": IL.slice_words(cols[:, j], 1) for j in range(cols.shape[1])}, T)
        assert not fail
        assert np.array_equal(IL.unslice_words(got_s["out"], T), _pack(E.evaluate_batch(d, s, x).astype(np.uint64)))


@pytest.mark.parametrize("n", [2, 3, 8, 33])
def test_an_n_input_gate_lowers_to_n_minus_1_ands_or_none(n) -> None:
    T = 64
    bits = np.random.default_rng(n).integers(0, 2, size=(T, n), dtype=np.uint64)
    for gate, ands, want in ((BL.and_gate(n), n - 1, bits.all(axis=1)), (BL.xor_gate(n), 0, bits.sum(axis=1) & 1)):
        C, _ = _lower(gate, [1] * n)
        assert C.counts()["ands"] == ands
        got_s, fail = C.evaluate({f"in{j}": IL.slice_words(bits[:, j], 1) for j in range(n)}, T)
        assert not fail and np.array_equal(IL.unslice_words(got_s["out"], T), want.astype(np.uint64))
