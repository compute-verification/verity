from __future__ import annotations

import functools

import numpy as np
import pytest

from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import program_digest
from verity.circuits.evaluation.reference import evaluate, evaluate_region
from verity.circuits.ops import product as PR


@functools.cache
def small(registered=True, tokens=1):
    if tokens == 1:
        return PR.product(k=16, s=2, units=4, block=2, k_log=22, g=2, registered=registered)
    w, x = PR.example(4, 16)
    return PR.product(w, np.stack([x, x[::-1]]), k=16, s=2, units=2, block=2, k_log=22, g=2, registered=registered)


@functools.cache
def cut(lo: int, hi: int, registered=True, tokens=1):
    p = small(registered, tokens)
    bits = R.Bits(R.program(p.description))
    return R.restrict(p.description, p.partition, [(lo, hi)], bits), bits


def trace(p, outputs=None):
    C, t = p.program.circuit, {}
    for pi, words in enumerate((p.weight.flat, p.x.flat)):
        t.update(zip((r[2] for r in p.program.input_coll(pi).refs), (int(v) for v in words), strict=True))
    evaluate_region(C, C[0:C.size], t)
    outs = C.root_outputs()
    assert [t[o] for o in outs] == [int(v) for v in p.outputs.flat]
    if outputs is not None:
        t.update(zip(outs, (int(v) for v in outputs.flat)))
    return t


def bit(t, bits, x):
    g = bits.gate_of(x)
    return (t[g] >> bits.out[g].index(x)) & 1


def as_float(words):
    return (np.asarray(words, np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize("tokens", [1, 2])
def test_its_outputs_are_each_slice_of_the_product_by_the_template_in_token_slice_order(tokens):
    p = small(tokens=tokens)
    xs, slices = p.x.reshape(tokens, 16), 4 // tokens
    assert p.outputs.shape == (4, 2) and p.blocks == [(0, 2), (2, 4)] and p.m == 25
    for u in range(4):
        rows = [int(v) for v in p.weight[2 * (u % slices):2 * (u % slices) + 2].flat]
        assert evaluate(p.template, [int(v) for v in xs[u // slices]], rows) == tuple(int(v) for v in p.outputs[u])
    ref = as_float(xs).astype(np.float64) @ as_float(p.weight).astype(np.float64).T
    assert np.allclose(as_float(p.outputs.reshape(-1)), ref.reshape(-1), rtol=2 ** -7, atol=2 ** -10)


@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("registered", [True, False])
def test_the_program_decodes_with_its_weight_registered_or_a_run_input_and_its_row_a_run_input(registered, tokens):
    p = small(registered, tokens)
    D = R.program(p.description)
    assert D.circuit.size == p.program.circuit.size and len(p.digest) == 128 and p.digest == program_digest(p.program)
    assert p.program.bindings == ("registration" if registered else "run", "run")


@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("registered", [True, False])
def test_its_partition_object_validates_at_one_level_and_at_two(registered, tokens):
    p = small(registered, tokens)
    assert PO.validate(p.partition).ok and PO.validate(p.partition_levels).ok
    assert PO.blocks(p.partition) == p.blocks and PO.levels(p.partition_levels) == [[(0, 4)], p.blocks]
    with pytest.raises(ValueError, match="blocks-levels"):
        PO.blocks(p.partition_levels)


@pytest.mark.parametrize("registered", [True, False])
@pytest.mark.parametrize("lo, hi", [(0, 2), (2, 4)])
@pytest.mark.parametrize("tokens", [1, 2])
def test_a_block_holds_on_the_values_and_not_on_one_output_word_changed(lo, hi, registered, tokens):
    p = small(registered, tokens)
    b, bits = cut(lo, hi, registered, tokens)
    text = b.listing()
    t = trace(p)
    assert R.evaluate(text, [bit(t, bits, x) for x in b.input_bits]) == [bit(t, bits, x) for x in b.output_bits]
    wrong = p.outputs.copy()
    wrong[lo, 0] ^= 1
    t = trace(p, wrong)
    assert R.evaluate(text, [bit(t, bits, x) for x in b.input_bits]) != [bit(t, bits, x) for x in b.output_bits]


def test_a_real_weight_slice_is_passed_as_its_words_or_its_values():
    w, x = PR.example(4, 16)
    assert (PR.product(w.view(np.int16), x, k=16, units=4, block=2, k_log=22, g=2).outputs ==
            PR.product(as_float(w), as_float(x), k=16, units=4, block=2, k_log=22, g=2).outputs).all()


def test_a_block_under_the_floor_or_past_the_range_is_refused():
    with pytest.raises(ValueError, match="pack-floor"):
        PR.product(k=16, units=4, block=2, k_log=20, g=2)
    with pytest.raises(ValueError, match="past 2"):
        PR.product(k=16, units=1024, block=1024, k_log=26, g=2)


@pytest.mark.parametrize("kw", [{"k": 16}, {"s": 2}, {"dot": PR.G.GEMM_AMPERE}, {"k": 16, "k_log": 22}])
def test_a_unit_other_than_the_default_names_its_k_log_and_g(kw):
    with pytest.raises(ValueError, match="unit-size"):
        PR.product(units=4, block=2, **kw)


def test_the_default_is_256_units_in_blocks_at_the_least_zero_knowledge_m():
    p = PR.product()
    assert p.outputs.shape == (256, 1) and len(p.blocks) == 16 and p.m == 25
    assert p.digest == ("71cdbf7c67ae3f054fc9a5fcd43b2c3dcb7017cc33306857431f8c42faa0cb59"
                        "dec407780be910cba076ceabf019362487f9a55a8a31f66e7b10b755fd72d4fb")
    assert PO.validate(p.partition).ok and PO.validate(p.partition_levels).ok
