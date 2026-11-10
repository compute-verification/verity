"""The word view of a Boolean Definition with a port of single bits, without circuit-check's catalog."""
from __future__ import annotations

import numpy as np

from circuit_check import checks as CK


def test_a_port_of_single_bits_is_compared_through_its_bits(monkeypatch):
    """`DsaWaksmanOrder_v1`'s settings are 21 one-bit words, which `boolean.bit_type` leaves one 21-bit element on bits, so
    `evaluate_words` refuses the word Definition's vectors: the word view compares `DsaWaksmanOrder_v2` with it bit by bit, and a
    Boolean version that flips one bit of every output word is caught."""
    import verity.circuits.evaluation.bits as EB
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa_topk as BT

    t = bind(BT.DsaWaksmanOrderV2, N=9, TOPK=8)
    F = CK.Findings()
    r = CK.word_view(t, CK.Options(), F)
    assert r["via"] == "word:DsaWaksmanOrder_v1{N=9,TOPK=8}" and r["compared"] and r["mismatches"] == 0 and not F.failures

    real = EB.evaluate_bits

    def flipped(*args, **kw):
        out = np.array(real(*args, **kw))
        out[..., 0] ^= 1
        return out

    monkeypatch.setattr(EB, "evaluate_bits", flipped)
    F = CK.Findings()
    r = CK.word_view(t, CK.Options(), F)
    assert r["mismatches"] == r["compared"] and [f["code"] for f in F.failures] == ["word-mismatch"]


def _xor_bits(word: bool | str):
    """x ^ y on four bits, with ``x ^ y`` on 4-bit words as its word Definition when ``word``: a primitive, or an unbound
    composite calling it when ``word == "composite"``."""
    from verity.circuits import boolean as BL
    from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
    from verity.circuits.refs import Coll, concat
    from verity.circuits.types import Array, Value

    nib = Array(4, BL.BIT)

    def body(B, S, x, y):
        return B.batch(BL.XOR, Coll(Array(4, Array(2, BL.BIT)), concat([c.refs for i in range(4) for c in (x[i], y[i])])), axes=(0,))

    d = CompositeDefinition("CircuitCheckXorNibble", 1, (), lambda S: ((("x", nib), ("y", nib)), nib), body, register=False)
    if word:
        d.word = PrimitiveDefinition("CircuitCheckXorNibbleWord", 1, (("x", Value(4)), ("y", Value(4))), Value(4), lambda x, y: x ^ y,
                                     register=False)
    if word == "composite":
        xor = d.word
        d.word = CompositeDefinition("CircuitCheckXorNibbleWordCall", 1, (), lambda S: ((("x", Value(4)), ("y", Value(4))), Value(4)),
                                     lambda B, S, x, y: B.call(xor, x, y), register=False)
    return bind(d)


def test_a_word_view_past_its_budget_still_compares_random_vectors():
    """The word view draws 16 x `Options.vectors` within 16 x `Options.reference_budget`; a word Definition too large for that
    takes as many as the budget pays for, at least `Options.min_vectors`, half of them random; and at least
    `WORD_RANDOM_FLOOR` of them are random whatever the options."""
    t = _xor_bits(word=True)
    F = CK.Findings()
    r = CK.word_view(t, CK.Options(vectors=8), F)
    assert (r["compared"], r["edge_vectors"], r["mismatches"]) == (128, 6, 0) and not F.failures
    r = CK.word_view(t, CK.Options(vectors=8, reference_budget=0), F)
    assert (r["compared"], r["edge_vectors"], r["mismatches"]) == (4, 2, 0) and not F.failures
    for opts in (CK.Options(vectors=8, reference_budget=0, min_vectors=0), CK.Options(vectors=0, min_vectors=0)):
        r = CK.word_view(t, opts, F)
        assert r["compared"] - r["edge_vectors"] >= CK.WORD_RANDOM_FLOOR == 2 and r["mismatches"] == 0 and not F.failures


def test_a_word_definition_named_unbound_is_bound_for_its_budget():
    """A word Definition named unbound (a composite without parameters, as `ArgmaxStep_v1` is to `ArgmaxStep_v2`) has its gates
    counted bound, for the budget of its vectors."""
    F = CK.Findings()
    r = CK.word_view(_xor_bits(word="composite"), CK.Options(vectors=8), F)
    assert r["via"] == "word:CircuitCheckXorNibbleWordCall_v1" and r["compared"] == 128 and r["mismatches"] == 0 and not F.failures


def test_a_boolean_composite_over_the_input_budget_is_compared_through_its_word_view():
    """An operand over the input budget that no lazy reader takes fails, unless the Definition is a Boolean composite with a word
    view: its reference is its gates, total on every input, and the word view compares them, so its correspondence is a warning."""
    opts = CK.Options(vectors=8, input_budget=2)
    e = CK.check_definition(_xor_bits(word=True), opts, pins={}, as_call=False)
    assert not e["failures"] and "unchecked" in e["correspondence"]
    (w,) = [w for w in e["warnings"] if w["check"] == "correspondence"]
    assert w["code"] == "unchecked" and w["name"].startswith("correspondence/unchecked CircuitCheckXorNibble_v1")
    assert w["operands"] == ["x", "y"] and "operand 'x' (4 leaves), 'y' (4 leaves) over the input budget (2)" in w["name"]
    assert w["name"].endswith(
        "a Boolean Definition's reference is its gates, total on every input, and its word view compares them with "
        "CircuitCheckXorNibbleWord_v1")
    (wv,) = [r for r in e["correspondence"]["realizations"] if r["via"].startswith("word:")]
    assert wv["compared"] == 128 and wv["mismatches"] == 0
    e = CK.check_definition(_xor_bits(word=False), opts, pins={}, as_call=False)
    assert [f["code"] for f in e["failures"]] == ["unprobed-table"]
