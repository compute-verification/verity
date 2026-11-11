"""Security harness — the four validity conditions of a commitment, as executable tests over every
commitment construction in `schemes.SCHEMES`:

  (1) binding under the primitive        — an opening authenticates only the committed bytes (altered value rejected)
  (2) binds identity/position, not a multiset — reordered positions rejected; a commitment to the same multiset in another order has a
                                            different root; equal values at two positions are distinguishable by position
  (3) opens against the exact prechallenge root — the verifier holds R fixed BEFORE the challenge (the challenge is derived from R
                                            and the coin opened after it); a root recomputed after the fact (altered value → R') is not
                                            accepted as R and yields another challenge
  (4) negative tests                     — altered / reordered / substituted / truncated stream all rejected

Every test is parameterised over the schemes; a sibling lane's engine joins by adding an adapter in `schemes.py`.
"""

from __future__ import annotations

import random

import pytest

from verity_vllm.commit.challenge import challenge_positions
from tests.acquire.schemes import SCHEMES

CTX = b"ctx/request-A/step-3"
CTX_B = b"ctx/request-B/step-3"
#: the run's positions coin, as the auditor host's challenger would open it after the root
COIN = b"positions coin"


def values_of(n: int, seed: int = 1, width: int = 8) -> list[bytes]:
    rng = random.Random(seed)
    return [rng.randbytes(width) for _ in range(n)]


@pytest.fixture(params=SCHEMES, ids=lambda s: s.name)
def scheme(request):
    return request.param


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 9, 64, 129])
def test_roundtrip_every_position_opens_against_original_root(scheme, n):
    vals = values_of(n)
    c = scheme.commit(vals, CTX)
    for i in range(n):
        assert scheme.verify(c.root, c.n, CTX, scheme.open(c, i)), (scheme.name, n, i)


def test_altered_value_rejected(scheme):
    vals = values_of(37)
    c = scheme.commit(vals, CTX)
    for i in (0, 17, 36):
        op = scheme.open(c, i)
        bad = bytearray(vals[i])
        bad[0] ^= 0x01
        assert not scheme.verify(c.root, c.n, CTX, scheme.with_value(op, bytes(bad)))
        assert not scheme.verify(c.root, c.n, CTX, scheme.with_value(op, vals[i] + b"\0"))       # length change
        assert not scheme.verify(c.root, c.n, CTX, scheme.with_value(op, vals[i][:-1]))


def test_reordered_positions_rejected(scheme):
    """A value presented at another position (its own path, another index; or another position's path) is refused."""
    vals = values_of(40)
    c = scheme.commit(vals, CTX)
    for i, j in ((0, 1), (5, 20), (39, 38), (7, 7 ^ 8)):
        op_i, op_j = scheme.open(c, i), scheme.open(c, j)
        assert not scheme.verify(c.root, c.n, CTX, scheme.with_index(op_i, j))                    # value_i, path_i at index j
        assert not scheme.verify(c.root, c.n, CTX, scheme.with_index(scheme.with_path(op_i, op_j[2] if isinstance(op_j, tuple) else op_j.path), j))  # value_i, path_j at j
        assert not scheme.verify(c.root, c.n, CTX, scheme.with_value(op_j, vals[i]))               # value_i at position j with path_j


def test_equal_multiset_other_order_is_another_commitment(scheme):
    """Identity/position binding: the same multiset of values committed in a different order has a different root, and its openings
    do not authenticate against the original root even where the value happens to be equal."""
    vals = values_of(33)
    perm = list(range(33))
    random.Random(7).shuffle(perm)
    c = scheme.commit(vals, CTX)
    c_perm = scheme.commit([vals[p] for p in perm], CTX)
    assert c_perm.root != c.root
    c_sorted = scheme.commit(sorted(vals), CTX)
    assert c_sorted.root != c.root
    # an opening from the permuted tree at the position where value_i now sits, against the ORIGINAL root: refused
    for i in range(0, 33, 4):
        k = perm.index(i)
        op = scheme.open(c_perm, k)
        assert not scheme.verify(c.root, c.n, CTX, op)
        assert not scheme.verify(c.root, c.n, CTX, scheme.with_index(op, i))


def test_equal_values_at_two_positions_are_distinguished(scheme):
    vals = values_of(16)
    vals[3] = vals[11] = b"\x42" * 8
    c = scheme.commit(vals, CTX)
    op3, op11 = scheme.open(c, 3), scheme.open(c, 11)
    assert scheme.verify(c.root, c.n, CTX, op3) and scheme.verify(c.root, c.n, CTX, op11)
    # same bytes, wrong position's path: refused (the path authenticates the index, not just the bytes)
    p11 = op11[2] if isinstance(op11, tuple) else op11.path
    p3 = op3[2] if isinstance(op3, tuple) else op3.path
    assert not scheme.verify(c.root, c.n, CTX, scheme.with_path(op3, p11))
    assert not scheme.verify(c.root, c.n, CTX, scheme.with_path(op11, p3))


def test_truncated_stream_rejected(scheme):
    """Committing a prefix of the stream is another commitment: a smaller N, a different root; openings from the truncated tree are
    refused against the original (root, N), and the original openings are refused under the truncated (root, N)."""
    vals = values_of(50)
    c = scheme.commit(vals, CTX)
    c_tr = scheme.commit(vals[:-1], CTX)
    assert c_tr.root != c.root and c_tr.n == c.n - 1
    for i in (0, 10, 48):
        assert not scheme.verify(c.root, c.n, CTX, scheme.open(c_tr, i))
        assert not scheme.verify(c_tr.root, c_tr.n, CTX, scheme.open(c, i))
    # the last leaf simply has no opening in the truncated tree
    with pytest.raises(Exception):
        scheme.open(c_tr, 49)
    # count mismatch alone (right root, wrong N) is refused
    assert not scheme.verify(c.root, c.n - 1, CTX, scheme.open(c, 0))
    assert not scheme.verify(c.root, c.n + 1, CTX, scheme.open(c, 0))


def test_root_recomputed_after_the_fact_is_not_the_prechallenge_root(scheme):
    """Protocol order: the verifier fixes R, opens the coin after it, derives the challenge from both, then receives openings.  A
    prover that alters a value after seeing the challenge and recomputes R' (i) produces openings that fail against R, and (ii) R'
    would have produced a different challenge — so R' can never be passed off as the prechallenge root."""
    vals = values_of(64)
    c = scheme.commit(vals, CTX)
    R = c.root                                              # fixed by the verifier BEFORE the challenge
    chal = challenge_positions(R, [c.n], 12, source=COIN)
    assert all(s == 0 and 0 <= i < c.n for s, i in chal)
    # honest openings at the challenged positions verify
    for _s, i in chal:
        assert scheme.verify(R, c.n, CTX, scheme.open(c, i))
    # adversary: alter the value at the first challenged position, recompute the root
    _s, i0 = chal[0]
    vals2 = list(vals)
    vals2[i0] = bytes(b ^ 0xFF for b in vals[i0])
    c2 = scheme.commit(vals2, CTX)
    assert c2.root != R
    assert not scheme.verify(R, c.n, CTX, scheme.open(c2, i0))            # opening from the recomputed tree fails against R
    assert scheme.verify(c2.root, c2.n, CTX, scheme.open(c2, i0))         # (it is a fine opening of R' — which the verifier never accepted)
    assert challenge_positions(c2.root, [c2.n], 12, source=COIN) != chal   # and R' would not have produced this challenge


def test_context_binding(scheme):
    """The same values under another context (request/step/domain) are another commitment; cross-context openings are refused."""
    vals = values_of(20)
    c = scheme.commit(vals, CTX)
    cb = scheme.commit(vals, CTX_B)
    assert c.root != cb.root
    for i in (0, 9, 19):
        assert not scheme.verify(c.root, c.n, CTX_B, scheme.open(c, i))
        assert not scheme.verify(cb.root, cb.n, CTX, scheme.open(cb, i))
    # NOTE (property, not a requirement): where the context lives only in the root wrapper (C0 primitive, native_host) an opening of
    # `cb` is also a valid opening of `c` under CTX — the bare trees are identical because the bytes are; `veritor.protocol.merkle`
    # binds the domain into every leaf/node hash instead, so there the same cross-opening is refused.  Both satisfy (3): the verifier
    # holds (root, ctx) and a root never authenticates under another context.
    cross = scheme.verify(c.root, c.n, CTX, scheme.open(cb, 0))
    assert cross in (True, False)
