"""The finite share of the correspondence vectors (`vectors.finite_share`) and the finite counts each comparison records: the
rows outside the share are the vectors sampled without it bit for bit, the share's words are finite in every float format of
their width at working scale, and a wide float Definition's word view is counted as sampled with and without the share."""
from __future__ import annotations

import functools
import hashlib

import numpy as np

from circuit_check import checks as CK
from circuit_check import targets as TG
from circuit_check import vectors as VX
from verity.circuits.types import Array, Tuple, Value

#: a wide float Definition a catalog binding checks, its word view's 1,024 vectors over 16- and 32-bit leaves
ATTN = "AttnBlock_v15{D=16,NVIS=5,FIRST=True,BN=16,DOT=HopperBF16WgmmaDot16_v2}"
#: every width `vectors` treats apart: 8, 16, 32 and 64 bits, a width no float has (24), the object path (a tuple with 128 bits)
MIXED = [Value(8), Array(3, Value(16)), Value(32), Value(64), Array(2, Value(24)), Tuple(Value(32), Value(128))]
#: sha256 of the vectors sampled before the finite share, rows outside it: ATTN's word view and MIXED's 40 vectors at seed 7
KEPT = {"attn": "652ee110f57e255e2c30bc749a29c14d32ebf6386c79f4ed3a38054889fe7319",
        "mixed": "2493db2966b32beaedf10f793fb219acb7879f784eaff9ec66e8c0994ad77226"}
#: and of MIXED's 24- and 128-bit leaves on the share's rows, which the share leaves as they were sampled
MIXED_OTHER = "660cb184177812bf991db3ee047db7918ec80f4144e538809776c4909870b9b0"


def _digest(arrays, rows) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        h.update(",".join(str(int(v)) for v in a.reshape(a.shape[0], -1)[rows].reshape(-1)).encode() + b";")
    return h.hexdigest()


def _kept(n: int, edge: int) -> list[int]:
    return sorted(set(range(n)) - set(VX.finite_share(n, edge)))


def _attn_word():
    t = CK.resolve(TG.definition(ATTN))
    word = CK._word_or_none(t)
    pw, _rw = CK.signature_of(word)
    n, edge = CK._word_vectors(word, pw, CK.DEFAULTS)
    return t, pw, n, edge


def _finite_by_numpy(x: int, w: int) -> bool:
    """A word's finiteness in every format of `FLOATS` at its width, read through numpy's own floats (E5M2 is FP16's top byte,
    BF16 FP32's top half, E4M3 is NaN at S.1111.111 alone)."""
    if w == 8:
        return (x & 0x7F) != 0x7F and bool(np.isfinite(np.array([x << 8], np.uint16).view(np.float16))[0])
    if w == 16:
        return bool(np.isfinite(np.array([x << 16], np.uint32).view(np.float32))[0]
                    and np.isfinite(np.array([x], np.uint16).view(np.float16))[0])
    if w == 32:
        return bool(np.isfinite(np.array([x], np.uint32).view(np.float32))[0])
    return bool(np.isfinite(np.array([x], np.uint64).view(np.float64))[0])


def test_the_rows_outside_the_finite_share_are_sampled_as_without_it():
    _t, pw, n, edge = _attn_word()
    new, old = VX.vectors(pw, n, CK.DEFAULTS.seed + 2, edge), VX.vectors(pw, n, CK.DEFAULTS.seed + 2, edge, finite=False)
    assert _digest(new, _kept(n, edge)) == _digest(old, _kept(n, edge)) == KEPT["attn"]
    E = VX.edge_count(MIXED)
    new, old = VX.vectors(MIXED, 40, 7), VX.vectors(MIXED, 40, 7, finite=False)
    assert _digest(new, _kept(40, E)) == _digest(old, _kept(40, E)) == KEPT["mixed"]
    share = list(VX.finite_share(40, E))
    other = [np.asarray(new[4]).reshape(40, -1), np.asarray(new[5]).reshape(40, -1)[:, 1:]]
    assert share == [27, 29, 31, 33, 35, 37, 39] and _digest(other, share) == MIXED_OTHER
    assert VX.finite_rows(MIXED, new, 40)[share].all()


def test_the_finite_share_is_half_the_random_rows():
    assert list(VX.finite_share(10, 4)) == [5, 7, 9] and list(VX.finite_share(10, 5)) == [5, 7, 9]
    assert list(VX.finite_share(4, 4)) == [] == list(VX.finite_share(4, 9))
    for n, edge in ((64, 17), (64, 26), (1024, 26), (65, 0)):
        assert abs(2 * len(VX.finite_share(n, edge)) - (n - edge)) <= 1


def test_the_finite_reading_is_every_format_of_the_width():
    for w in (8, 16):
        words = np.arange(1 << w, dtype=np.uint64)[:, None]
        got = VX.finite_words(words, [w])
        assert got.tolist() == [_finite_by_numpy(int(x), w) for x in words[:, 0]]
    rng = np.random.default_rng(1)
    for w in (32, 64):
        words = [*VX.edges(w), *(int(x) for x in rng.integers(0, (1 << w) - 1, 2000, dtype=np.uint64, endpoint=True))]
        got = VX.finite_words(np.array(words, dtype=np.uint64)[:, None], [w])
        assert got.tolist() == [_finite_by_numpy(x, w) for x in words]
    assert VX.finite_words(np.array([[0xFFFFFF, (1 << 128) - 1]], dtype=object), [24, 128]).tolist() == [True]


def test_the_finite_share_is_at_working_scale_in_every_float_width():
    for w, (e_bits, bias, spread) in VX._WORKING.items():
        params = [Array(50, Value(w)), Value(w)]
        E = VX.edge_count(params)
        arrays = VX.vectors(params, E + 40, 3)
        share = list(VX.finite_share(E + 40, E))
        assert len(share) == 20 and VX.finite_rows(params, arrays, E + 40)[share].all()
        words = np.concatenate([np.asarray(a).reshape(E + 40, -1)[share] for a in arrays], axis=1).astype(np.uint64)
        exp = (words >> np.uint64(w - 1 - e_bits)) & np.uint64((1 << e_bits) - 1)
        assert (np.abs(exp.astype(np.int64) - bias) <= spread).all()
        assert len(np.unique(words)) > min(900, ((2 * spread + 1) << (w - e_bits)) // 2)
        assert [_finite_by_numpy(int(x), w) for x in words.reshape(-1)[:200]] == [True] * 200


def test_an_operand_sampled_one_vector_at_a_time_is_finite_on_the_share():
    pt, edge, seed = Array(5000, Value(16)), 20, 11
    for i in range(edge + 6):
        f = VX.wide_words(pt, i, edge, seed, 2)
        words = np.array([[f(j) for j in range(pt.leaves)]], dtype=np.uint64)
        assert f.finite == bool(VX.finite_words(words, [16] * pt.leaves)[0])
        if i >= edge:
            assert f.finite == (i % 2 == 1)
        if i >= edge and i % 2 == 0:
            assert words[0].tolist() == VX._random(np.random.default_rng([seed, 2, i]), 16, (1, pt.leaves))[0].tolist()


def test_a_wide_float_word_view_counts_its_finite_vectors(monkeypatch):
    """`ATTN`'s word view: sampled without the share (as before it) 13 of its 1,024 vectors have every input leaf finite in
    every format and 11 give every output finite, all of them all-edge vectors, which is a ``vectors/few-finite`` warning;
    reading a 16-bit leaf as BF16 alone the inputs are 17 (four edge vectors hold a 16-bit word finite as BF16 and not as FP16:
    BF16's largest of either sign and two FP16 NaNs).  With the share, 512 inputs (those 13 and the 499 rows of the share) and
    260 outputs are finite, and nothing warns."""
    t, pw, n, edge = _attn_word()
    F = CK.Findings()
    rec = CK.word_view(t, CK.DEFAULTS, F)
    assert (rec["compared"], rec["mismatches"], rec["finite_inputs"], rec["finite_outputs"]) == (1024, 0, 512, 260)
    assert len(VX.finite_share(n, edge)) == 499 and not any(w["code"] == "few-finite" for w in F.warnings)

    monkeypatch.setattr(CK, "vectors", functools.partial(VX.vectors, finite=False))
    F = CK.Findings()
    rec = CK.word_view(t, CK.DEFAULTS, F)
    assert (rec["compared"], rec["mismatches"], rec["finite_inputs"], rec["finite_outputs"]) == (1024, 0, 13, 11)
    assert [(w["check"], w["code"]) for w in F.warnings] == [("vectors", "few-finite")]
    old = VX.vectors(pw, n, CK.DEFAULTS.seed + 2, edge, finite=False)
    monkeypatch.setattr(VX, "NONFINITE", {**VX.NONFINITE, 16: (0x7F80,)})
    assert int(VX.finite_rows(pw, old, n).sum()) == 17
