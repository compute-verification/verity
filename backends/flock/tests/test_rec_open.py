"""`verity_flock.rec_open`, V*'s opening check: its reference agrees with `rec_vstar`'s leaf and climb on a real tree, and
the M0 unit and the Definition agree with the reference on an honest opening of either side and on a forged one."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from verity_flock import rec_open as R
from verity_flock import rec_vstar as V


def _salt(i: int) -> bytes:
    return hashlib.shake_256(b"rec-open-salt" + i.to_bytes(4, "little")).digest(V.SALT)


def _tree(depth: int, lanes: int):
    rows = [[(i * 1000 + k) * 0x9E3779B97F4A7C15 % (1 << 128) for k in range(lanes)] for i in range(1 << depth)]
    salts = [_salt(i) for i in range(1 << depth)]
    layers = [[V.leaf(r, s) for r, s in zip(rows, salts)]]
    while len(layers[-1]) > 1:
        lo = layers[-1]
        layers.append([hashlib.sha512(lo[i] + lo[i + 1]).digest() for i in range(0, len(lo), 2)])
    return rows, salts, layers


def _opening(rows, salts, layers, pos: int, H: int):
    row, aux = R.pack(rows[pos], salts[pos], [layers[h][(pos >> h) ^ 1] for h in range(H)], pos)
    return row, aux, R.expected(layers[H][pos >> H], pos, H)


def test_the_reference_climbs_to_the_node_its_position_names():
    rows, salts, layers = _tree(4, 8)
    assert R.row_words(8) == 64 and R.aux_words(1) == 192 and R.aux_words(3) == 256
    for H in (1, 2, 4):
        for pos in range(16):
            row, aux, want = _opening(rows, salts, layers, pos, H)
            assert R.open_ref(8, H, row, aux) == want
            q = V.Query(0, 0, 0, pos, 4, 4 - H, layers[H][pos >> H], rows[pos], salts[pos],
                        [layers[h][(pos >> h) ^ 1] for h in range(H)])
            assert V.opens(q)
    row, aux, want = _opening(rows, salts, layers, 5, 2)
    for at in (0, R.SALT_WORDS - 1, R.SALT_WORDS + 3, R.dirs_at(2)):
        bad = list(aux)
        bad[at] ^= 1
        assert R.open_ref(8, 2, row, bad) != want, at
    assert R.open_ref(8, 2, [row[0] ^ 1] + row[1:], aux) != want


@pytest.mark.parametrize(("bad", "error"), [((8, 0), ValueError), ((12, 1), ValueError), ((8, 17), ValueError),
                                            ((8.0, 1), TypeError), ((True, 1), TypeError)])
def test_the_shape_is_checked(bad, error):
    with pytest.raises(error):
        R.open_ref(*bad, [], [])


def _cases():
    rows, salts, layers = _tree(1, 8)
    honest = [_opening(rows, salts, layers, pos, 1) for pos in (0, 1)]
    row, aux, want = honest[0]
    forged = (row, aux[:R.SALT_WORDS] + [w ^ 0x8000 for w in aux[R.SALT_WORDS:R.SALT_WORDS + 1]] + aux[R.SALT_WORDS + 1:], want)
    return honest + [forged]


def _bits(words):
    return [(w >> j) & 1 for w in words for j in range(16)]


@pytest.mark.slow
def test_the_m0_unit_computes_the_reference():
    """The unit (`forms`, 8 lanes, one level) on both sides of an honest opening and on a forged sibling, bit-sliced."""
    cases = _cases()
    C, ins, outs = R.unit_circuit(8, 1)
    vals = {nm: [0] * 16 for nm in ins}
    for t, (row, aux, _) in enumerate(cases):
        for nm, w in zip(ins, row + aux):
            for j in range(16):
                vals[nm][j] |= ((w >> j) & 1) << t
    got, fail = C.evaluate(vals, len(cases))
    assert fail == 0
    for t, (row, aux, want) in enumerate(cases):
        words = [sum(((got[nm][j] >> t) & 1) << j for j in range(16)) for nm in outs]
        assert words == R.open_ref(8, 1, row, aux)
        assert (words == want) == (t < 2)


@pytest.mark.slow
def test_the_definition_computes_the_reference():
    """`RecOpen{8, 1}` evaluated bit-sliced from its descriptor, on the same cases as the unit."""
    from verity.primitives.circuits.evaluation import evaluate_batch

    cases = _cases()
    d = R.definition(8, 1)
    row = np.array([np.array(_bits(r), dtype=np.uint8).reshape(-1, 16) for r, _, _ in cases])
    aux = np.array([np.array(_bits(a), dtype=np.uint8).reshape(-1, 16) for _, a, _ in cases])
    out = np.asarray(evaluate_batch(d, row, aux)).reshape(len(cases), R.OUT_WORDS, 16)
    for t, (r, a, want) in enumerate(cases):
        words = [int(sum(int(b) << j for j, b in enumerate(w))) for w in out[t]]
        assert words == R.open_ref(8, 1, r, a)
        assert (words == want) == (t < 2)
