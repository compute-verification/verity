"""The dense scheme (`verity.core.protocols.pous.dense`, Lean's `chainDense`): round trips, the construction recomputed from the Lean
definitions' text, its costs, the secure-first parameters, and the whole flow through setup, per-tensor decode and the
timed audit with the parameters' own codec."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from verity.core.protocols.pous import dense, setup
from verity.core.protocols.pous.audit import HonestResponder, UnsignedRounds, challenge_key, run_timed, simulate
from verity.core.protocols.pous.codec import encode_segments, split
from verity.core.protocols.pous.labels import salt_value, tag
from verity.core.protocols.pous.layout import Layout, decode_tensor
from verity.core.protocols.pous.oracle import run
from verity.core.protocols.pous.primitives import OverwriteStepper, Primitives, pad_block
from verity.core.protocols.pous.verifier import VerifierKey

PARAMS = dense.DenseParams(8, 64, 3)
SALT = bytes(range(24))


def _w(params, seed):
    return random.Random(seed).randbytes(params.w_bytes)


@pytest.mark.parametrize("backend", ["reference", "ideal"])
def test_round_trip(backend):
    prims = dense.reference(PARAMS) if backend == "reference" else dense.ideal(PARAMS, b"dense-seed")
    w = _w(PARAMS, 1)
    for s, seg in enumerate(split(PARAMS, w)):
        blocks = dense.encode_segment(PARAMS, prims, SALT, s, seg)
        assert len(blocks) == 8 and all(len(b) == 8 for b in blocks)
        assert dense.decode_segment(PARAMS, prims, SALT, s, blocks) == seg
    first = [b for seg in encode_segments(dense.DENSE, PARAMS, prims, w, SALT) for b in seg]
    assert first == [b for seg in encode_segments(dense.DENSE, PARAMS, prims, w, SALT) for b in seg]
    assert len(set(first)) == len(first), "segments with the same data differ by their tag"


def test_construction_is_lean_chain_dense():
    """C_i = r of the pad call of the overwrite chain from h_0 = 0^ℓ ‖ tag_s ‖ u256(i), absorbing C_{i-1}, ..., C_0
    (Lean `labelOv`, `ovKey`, `iv`, `dom`, `absorbed`), xor W_i."""
    prims = dense.reference(PARAMS)
    mb = PARAMS.m // 8
    w = _w(PARAMS, 2)
    s = 1
    seg = list(split(PARAMS, w))[s]
    tag_s = tag(salt_value(SALT), s).to_bytes(32, "little")
    pad = b"\x01" + bytes(mb - 2) + b"\x80"
    c: list[bytes] = []
    for i in range(8):
        h = bytes(mb) + tag_s + i.to_bytes(32, "little")
        for b in [c[u] for u in range(i - 1, -1, -1)] + [pad]:
            out = prims.h.pi2.forward(int.from_bytes(b + h, "little")).to_bytes(2 * mb + 64, "little")
            r, h = out[:mb], out[mb:]
        c.append(bytes(x ^ y for x, y in zip(r, seg[i * mb:(i + 1) * mb])))
    assert dense.encode_segment(PARAMS, prims, SALT, s, seg) == c


def test_costs():
    """Label i costs i + 1 calls of Π₂; decode runs the rows in parallel: B rounds, B(B + 1)/2 calls."""
    params = dense.DenseParams(8, 64)
    prims = dense.reference(params)
    t = tag(salt_value(SALT), 0)
    w = [random.Random(3).getrandbits(64) for _ in range(8)]
    c, cost = run(dense.encode_segment_prog(params, prims.h, t, w), prims)
    assert cost.depth == cost.work == 36 and cost.kinds == {"PiQuery": 36}
    back, cost = run(dense.decode_segment_prog(params, prims.h, t, c), prims)
    assert back == w and cost.depth == 8 and cost.work == 36


def test_secure_first_parameters():
    p = dense.POINT_64K
    assert (p.blocks_per_segment, p.m, p.segment_bytes) == (512, 524_288, 1 << 25), "dense_meets_64's point"
    assert (p.challenges, p.certified_rounds, p.call_ns) == (111, 5, 241_500) and p.codec is dense.DENSE
    assert dense.reference(p).h.pi2.width == 2 * 524_288 + 512
    assert list(p.parents(3)) == [0, 1, 2]
    assert dense.dom(5, 7) == 5 | (7 << 256)


def test_end_to_end_with_the_parameters_codec():
    """Setup, per-tensor decode, the timed audit and the simulator, every codec taken from the parameters."""
    base = dense.DenseParams(8, 64)
    sizes = [("embed", 100), ("norm", 3), ("proj", 150)]
    layout = Layout.of(base, sizes)
    params = layout.params
    prims = dense.reference(params)
    data = {name: random.Random(len(name)).randbytes(size) for name, size in sizes}
    w = layout.pack(data)
    commitment = setup.commit(params, w)
    salt = setup.draw_salt(b"verifier-secret", commitment)
    segs = list(split(params, w))
    server = [dense.encode_segment(params, prims, salt, s, g) for s, g in enumerate(segs)]
    vk = setup.verifier_key(params, prims, w, commitment, salt)
    assert vk == VerifierKey.from_segments(params, salt, server)
    blocks = [b for seg in server for b in seg]
    assert all(decode_tensor(layout, prims, salt, name, blocks.__getitem__) == data[name] for name in data)
    key = challenge_key(b"audit", vk)
    timed = UnsignedRounds(vk, key, clock=itertools.count(0, 1000).__next__)
    assert timed.k == 111
    assert run_timed(timed, blocks.__getitem__).accepted
    wrong = run_timed(UnsignedRounds(vk, key, k=5, clock=itertools.count(0, 1000).__next__), lambda i: blocks[(i + 1) % len(blocks)])
    assert not wrong.accepted
    from verity.core.protocols.pous.scheme import Encoding

    enc = Encoding(params, salt, tuple(int.from_bytes(b, "little") for b in blocks))
    result = simulate(vk, HonestResponder(enc), prims, key)
    assert result.accepted and len(result.answers) == 111


def test_rejects():
    prims = dense.reference(PARAMS)
    seg = bytes(PARAMS.segment_bytes)
    for bad in (dict(s=3), dict(salt=b"short"), dict(w_seg=seg[:-1])):
        args = dict(params=PARAMS, prims=prims, salt=SALT, s=0, w_seg=seg) | bad
        with pytest.raises(ValueError):
            dense.encode_segment(**args)
    with pytest.raises(ValueError):
        dense.DenseParams(0, 64)
    with pytest.raises(ValueError):
        prims.h.stepper(0, 1, 0, [])


class ToyF:
    """Lean `toyF`: `x ↦ a·x + b mod 2^w`, a = 11400714819323198485, b = 12345 + w."""

    name = "lean-toy-f"

    def __init__(self, width):
        self.width = width

    def forward(self, x):
        return (11400714819323198485 * x + 12345 + self.width) % (1 << self.width)

    def inverse(self, y):
        a = 11400714819323198485 % (1 << self.width)
        return ((y - 12345 - self.width) * pow(a, -1, 1 << self.width)) % (1 << self.width)


LEAN = json.loads((Path(__file__).parent / "fixtures" / "lean_dense_vectors.json").read_text())


@dataclass(frozen=True, slots=True)
class SmallCapacity(dense.DenseChain):
    """The chain with Lean's 4-bit `domSmall` (a 2-bit salt, then the index) in a 4-bit capacity, as evaluated."""

    capacity: int = 4

    def __post_init__(self) -> None:
        assert self.pi2.width == 2 * self.params.m + self.capacity

    def initial(self, t: int, i: int) -> int:
        return (t | ((i % 4) << 2)) << self.params.m


def _chain(case, chain):
    b, ell = case["B"], case["l"]
    params = dense.DenseParams(b, ell)
    h = chain(params, ToyF(2 * ell + case["c"]))
    prims = Primitives(h, None)
    salt, w = int(case["salt"]), int(case["w"])
    blocks = [(w >> (i * ell)) & ((1 << ell) - 1) for i in range(b)]
    c, _ = run(dense.encode_segment_prog(params, h, salt, blocks), prims)
    assert [str(x) for x in c] == case["blocks"]
    back, _ = run(dense.decode_segment_prog(params, h, salt, c), prims)
    assert str(sum(x << (i * ell) for i, x in enumerate(back))) == case["decode"] == case["w"]


@pytest.mark.parametrize("case", LEAN["small"], ids=lambda c: f"B{c['B']}-l{c['l']}-c4")
def test_labels_match_lean_small_capacity(case):
    _chain(case, SmallCapacity)


@pytest.mark.parametrize("case", LEAN["real"], ids=lambda c: f"B{c['B']}-l{c['l']}-c512")
def test_labels_match_lean_real_dom(case):
    _chain(case, dense.DenseChain)


@pytest.mark.parametrize("case", LEAN["layout"], ids=lambda c: f"l{c['l']}")
def test_layout_matches_lean(case):
    """`dom`, `iv` and `padBlk` at the real capacity, and `ovKey` over four blocks (the absorption order)."""
    ell, salt = case["l"], int(case["salt"])
    real = dense.DenseChain(dense.DenseParams(1, ell), ToyF(2 * ell + 512))
    assert str(dense.dom(salt, 3)) == case["dom3"] and str(real.initial(salt, 3)) == case["iv3"]
    assert str(pad_block(ell)) == case["pad"]
    small = SmallCapacity(dense.DenseParams(1, ell), ToyF(2 * ell + 4))
    st = OverwriteStepper(ell, small.initial(int(case["small_salt"]), 2), [0, 1, 2])
    labels = {0: 5, 1: 3, 2: 6}
    while not st.done:
        st.feed(small.answer(st.query(labels)))
    assert str(st.key) == case["key4"]
