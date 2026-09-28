"""C-Flock lowers the Boolean BF16 Definitions (`verity.ml.circuits.ops.boolean.gemm`) and NVFP4 quantization's casts and the bit AND
(`verity.ml.circuits.ops.boolean.scalar`) gate by gate, with no piece substituted for them: each step and cast and the AND lower to exactly
the circuit of the piece their word primitive lowers to (the same ANDs and the same nonzeros of the AND rows), and the
coordinate evaluates to `GemmCoordinate_v2`'s words.

The coordinate costs a few ANDs more than `GemmCoordinate_v2`'s lowering: there the first step's piece runs on the zero
accumulator's constant forms, and the builder's own branches on constants (`fp`, `tc_step`) drop ANDs that a gate-level circuit,
traced on a generic accumulator, keeps after constant folding. The difference is the first step's alone, so it is the same at
every K."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G
from verity.ml.kernels import tc as K
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops import scalar as S
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import scalar as BSC
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


#: (step binding, ANDs of GemmCoordinate_v2's lowering, ANDs more for GemmCoordinate_v3's) at K = 16 and K = 64
COORDINATES = [("ampere", G.GEMM_AMPERE, BG.GEMM_AMPERE, {16: 8034, 64: 32814}, 28),
               ("hopper", G.GEMM_HOPPER, BG.GEMM_HOPPER, {16: 7639, 64: 31255}, 31)]


@pytest.mark.parametrize("name,word_b,bool_b,v2,more", COORDINATES, ids=[c[0] for c in COORDINATES])
def test_the_coordinate_costs_its_first_steps_specialization(name, word_b, bool_b, v2, more) -> None:
    for k, ands in v2.items():
        Cw, _ = _lower(bind(G.GemmCoordinate, K=k, **word_b), [16] * (2 * k))
        Cb, _ = _lower(bind(BG.GemmCoordinate, K=k, **bool_b), [1] * (2 * k * 16))
        assert Cw.counts()["ands"] == ands
        assert Cb.counts()["ands"] == ands + more


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


@pytest.mark.parametrize("n", [2, 3, 8, 33])
def test_an_n_input_gate_lowers_to_n_minus_1_ands_or_none(n) -> None:
    from verity.ml.circuits import boolean as BL

    T = 64
    bits = np.random.default_rng(n).integers(0, 2, size=(T, n), dtype=np.uint64)
    for gate, ands, want in ((BL.and_gate(n), n - 1, bits.all(axis=1)), (BL.xor_gate(n), 0, bits.sum(axis=1) & 1)):
        C, _ = _lower(gate, [1] * n)
        assert C.counts()["ands"] == ands
        got_s, fail = C.evaluate({f"in{j}": IL.slice_words(bits[:, j], 1) for j in range(n)}, T)
        assert not fail and np.array_equal(IL.unslice_words(got_s["out"], T), want.astype(np.uint64))
