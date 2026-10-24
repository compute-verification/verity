"""The word view of a Boolean Definition with a port of single bits, without circuit-check's catalog."""
from __future__ import annotations

import numpy as np

from circuit_check import checks as CK


def test_a_port_of_single_bits_is_compared_through_its_bits(monkeypatch):
    """`DsaWaksmanOrder_v1`'s settings are 21 one-bit words, which `boolean.bit_type` leaves one 21-bit element on bits, so
    `evaluate_words` refuses the word Definition's vectors: the word view compares `DsaWaksmanOrder_v2` with it bit by bit, and a
    Boolean version that flips one bit of every output word is caught."""
    import verity.ml.circuits.evaluation.bits as EB
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_dsa_topk as BT

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
