"""`verity.ml.commitments.gates.lms`: the Boolean LMS verifier accepts every signature `lms.sign_at` makes at
(N, W) = (64, 1) on trees of height 1 and 2, refuses each with one thing wrong (a chain value, C, a path node, the leaf index,
the message, I, the root) exactly where `lms.verify` does, its parts cost the gates pinned here, and Flock lowers a part
by the builder its trace was taken from."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.commitments.gates import lms as G
from verity.ml.crypto import lms as R

SEED = bytes(range(64))
#: (part, its statics) -> (And, Xor, Not) gates of its traced circuit at N = 64, W = 1
PARTS = {
    ("Sha512Compress", ()): (57947, 286724, 4948),
    ("LmsMessage", (64, 1)): (113102, 554271, 11735),
    ("LmsChainStep", (64, 1)): (56814, 275348, 6267),
    ("LmsLeaf", (64,)): (56266, 274055, 6283),
    ("LmsNode", (64,)): (113103, 553285, 11231),
    ("LmsEqual", (64,)): (511, 512, 512),
}


def _flip(b: bytes) -> bytes:
    return bytes([b[0] ^ 1]) + b[1:]


def _cases(p: R.Params) -> tuple[list, list[int]]:
    """Every leaf's signature of its own role digest, then each with one thing wrong."""
    k = R.keygen(p, SEED)
    pk = R.public_key(k)
    bad_I, bad_root = _flip(pk[:16]) + pk[16:], pk[:16] + _flip(pk[16:])
    rows, want = [], []
    for q in range(2 ** p.H):
        m = R.certificate_message(k.I, bytes(32), "responses", "egress", q, b"record")
        s = R.sign_at(k, q, m)
        lv = q % p.H
        cases = [(pk, m, s),
                 (pk, m, R.Signature(q, s.C, s.y[:3] + (_flip(s.y[3]),) + s.y[4:], s.path)),
                 (pk, m, R.Signature(q, _flip(s.C), s.y, s.path)),
                 (pk, m, R.Signature(q, s.C, s.y, s.path[:lv] + (_flip(s.path[lv]),) + s.path[lv + 1:])),
                 (pk, m, R.Signature(q ^ 1, s.C, s.y, s.path)),
                 (pk, _flip(m), s), (bad_I, m, s), (bad_root, m, s)]
        for args in cases:
            rows.append(G.arguments(p, *args))
            want.append(int(R.verify(p, *args)))
    return rows, want


@pytest.mark.slow
@pytest.mark.parametrize("H", [1, 2])
def test_the_verifier_agrees_with_the_reference(H) -> None:
    """About a minute at H = 1 (the parts' first evaluation): the only check of the whole verifier at the deployed N, whose
    two-block message hash and tree nodes the toy binding circuit-check runs never reaches."""
    p = R.Params(64, 1, H)
    rows, want = _cases(p)
    assert want == [1, 0, 0, 0, 0, 0, 0, 0] * 2 ** H
    cols = [np.array([r[k] for r in rows], dtype=np.uint8) for k in range(7)]
    got = np.asarray(E.evaluate_batch(G.verifier(64, 1, H), *cols)).reshape(-1).tolist()
    assert got == want


@pytest.mark.slow
def test_the_parts_cost_the_pinned_gates() -> None:
    for (name, statics), (ands, xors, nots) in PARTS.items():
        assert getattr(G, name).traced(*statics).counts == {"and": ands, "xor": xors, "not": nots}, name


@pytest.mark.parametrize("name, statics", [("Sha512Compress", ()), ("LmsEqual", (64,))])
def test_a_part_lowers_by_the_builder_its_trace_was_taken_from(name, statics) -> None:
    """Flock lowers a part's activation by its builder (`ir_lower._boolean_activation`: ``build(C, *e)`` for a part without
    statics, ``build(C, S, *e)`` with), whose commits keep a compression's forms short; called so, it traces to the part's
    own gates."""
    d = getattr(G, name)
    S = SimpleNamespace(**dict(zip(d.statics, statics)))
    params, ret = d.signature(S)
    build = (lambda C, *e: d.build(C, S, *e)) if d.statics else d.build
    got, want = TR.trace(build, [t for _, t in params], ret), d.traced(*statics)
    assert (got.gates, got.outputs) == (want.gates, want.outputs)


@pytest.mark.slow
def test_the_verifier_is_boolean() -> None:
    assert BL.is_boolean(G.verifier(64, 1, 3))


def test_the_verifier_is_typed_by_its_shape() -> None:
    v = G.verifier(64, 1, 3)
    assert v is G.verifier(64, 1, 3)
    assert [t.leaves for _, t in v.signature()[0]] == [128, 512, 512, 3, 512, 522 * 512, 3 * 512]
    for n in (32, 48):
        with pytest.raises(ValueError):
            G.verifier(n, 1, 1).signature()


def test_malformed_arguments_are_refused() -> None:
    p = R.Params(64, 1, 1)
    k = R.keygen(p, SEED)
    s = R.sign_at(k, 0, bytes(64))
    with pytest.raises(ValueError, match="the message 64"):
        G.arguments(p, R.public_key(k), bytes(63), s)
    with pytest.raises(ValueError, match="chain values"):
        G.arguments(p, R.public_key(k), bytes(64), R.Signature(2, s.C, s.y, s.path))
