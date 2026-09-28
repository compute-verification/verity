"""Two labels from one stored item (`docs/p3-instantiation.md` §1, `lean/submissions/sponge-dense`).

An item breaks the storage bound when it is smaller than two labels yet yields two missing labels, each of which alone
would not give the other.  The attacks run on real lower-layer labels of `B(14, 4)`: they succeed on the XOR sponge (a
state, walked back with `Π⁻¹` to the public first state) and on Miyaguchi–Preneel (the pre-image of one absorption),
and fail on the overwrite chain, whose chaining value finishes only its own row and whose parent-revealing items are
whole call states of two labels or more.  The designer's state game confirms each mode on dense and band rows.
"""

from __future__ import annotations

import random

import pytest

import pous_state_game as game
from verity.core.protocols.pous.labels import mask, salt_value, tag, w_labels
from verity.core.protocols.pous.oracle import Tweak, run
from verity.core.protocols.pous.params import Params
from verity.core.protocols.pous.primitives import (CAPACITY_BITS, FeistelP, FeistelPi, MiyaguchiPreneel, OverwriteChain, Primitives,
                                    XorSponge, header, pad_block)
from verity.core.protocols.pous.scheme import encode_prog

PARAMS = Params(14, 80, 4)
M = PARAMS.m
TWO_LABELS = 2 * M


def _world(h):
    prims = Primitives(h, FeistelP(M))
    w = random.Random(12).randbytes(PARAMS.w_bytes)
    salt = salt_value(b"\x0c" * 24)
    [(c1, _)], _ = run(encode_prog(PARAMS, h, int.from_bytes(w, "little"), salt), prims)
    return prims, w_labels(PARAMS, int.from_bytes(w, "little"))[0], tag(salt, 0), c1


def _row_steps(h, t, v, c1, steps):
    """Drive row `v`'s key derivation (lower layer) for `steps` calls, from the true labels: the preprocessor's view."""
    st = h.stepper(t, 0, v, list(PARAMS.parents(v)))
    for _ in range(steps):
        st.feed(h.answer(st.query(dict(enumerate(c1)))))
    return st


def _finish(h, st, known):
    while not st.done:
        st.feed(h.answer(st.query(known)))
    return st.key


def test_xor_sponge_state_yields_two_labels():
    h = XorSponge(PARAMS, FeistelPi(M + CAPACITY_BITS))
    prims, w, t, c1 = _world(h)
    v = 9
    item = _row_steps(h, t, v, c1, 2).state
    assert M + CAPACITY_BITS < TWO_LABELS
    known = {u: c1[u] for u in range(PARAMS.n) if u not in (v - 1, v)}
    s1 = h.pi.forward(int.from_bytes(header(t, 0, v, 4), "little"))
    back = h.pi.inverse(item)
    assert back >> M == s1 >> M, "Π⁻¹ walks back to the public first state"
    parent = (back ^ s1) & mask(M)
    st = h.stepper(t, 0, v, list(PARAMS.parents(v)))
    st.i, st.state = 2, item
    key = _finish(h, st, known)
    assert parent == c1[v - 1] and prims.p.forward(Tweak(t, 0, v), w[v] ^ key) == c1[v]


def test_miyaguchi_preneel_pre_image_yields_two_labels():
    """On band rows (`k = 4`): the pre-image of row 6's absorption of its oldest parent, 2, is a checkpoint and a
    masked copy of `c_2` (the state game's `('x', 6, 4)` at D = 3)."""
    h = MiyaguchiPreneel(PARAMS, FeistelPi(M + CAPACITY_BITS))
    prims, w, t, c1 = _world(h)
    v, a = 6, 2
    st = _row_steps(h, t, v, c1, 4)
    item = st.state ^ c1[a]
    known = {u: c1[u] for u in range(PARAMS.n) if u not in (a, v)}
    prefix = _row_steps(h, t, v, [known.get(u, 0) for u in range(PARAMS.n)], 4).state
    parent = (item ^ prefix) & mask(M)
    s = h.pi.forward(item) ^ item
    x = s ^ pad_block(M)
    key = (h.pi.forward(x) ^ x) & mask(M)
    assert parent == c1[a] and prims.p.forward(Tweak(t, 0, v), w[v] ^ key) == c1[v]
    assert M + CAPACITY_BITS < TWO_LABELS


def test_overwrite_chain_yields_one_label_per_state():
    h = OverwriteChain(PARAMS, FeistelPi(2 * M + CAPACITY_BITS))
    prims, w, t, c1 = _world(h)
    v = 9
    st = _row_steps(h, t, v, c1, 1)
    item = st.h
    known = {u: c1[u] for u in range(PARAMS.n) if u not in (v - 1, v)}
    h0 = h.initial(t, 0, v, 4)
    for r_guess in (0, st.r ^ 1, random.Random(3).getrandbits(M)):
        back = h.pi2.inverse(r_guess | (item << M))
        assert back >> M != h0 and back & mask(M) != c1[v - 1], "without the dropped r_1 there is no way back"
    key = _finish(h, st, known)
    assert prims.p.forward(Tweak(t, 0, v), w[v] ^ key) == c1[v], "the chaining value finishes its own row"
    whole = c1[v - 1] | (h0 << M)
    assert whole.bit_length() <= 2 * M + CAPACITY_BITS and 2 * M + CAPACITY_BITS >= TWO_LABELS, \
        "the item that also reveals the parent is a whole call input, at least two labels"


CASES = [("dense", None, 8, 2), ("dense", None, 10, 3), ("band4", 4, 14, 3), ("band4", 4, 14, 4), ("band4", 4, 14, 5),
         ("band6", 6, 18, 4)]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-B{c[2]}-D{c[3]}")
def test_state_game(case):
    name, k, b, d = case
    sponge, mp, over = (game.check(mode, b, k, d) for mode in ("sponge", "mp", "over"))
    assert sponge is not None and sponge[0] < 2, "the XOR sponge breaks everywhere"
    if name == "dense":
        assert mp is None
    else:
        assert mp is not None and mp[0] < 2 and mp[2][0] == "x", "Miyaguchi–Preneel breaks on band rows"
    assert over is None or over[0] >= 2, "the overwrite chain has no item under two labels worth two"
