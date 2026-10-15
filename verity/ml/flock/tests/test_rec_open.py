"""`verity_flock.rec_open`, V*'s opening check: its reference agrees with `rec_vstar`'s plain leaf and climb on a real tree and
gives the top node's registered row, its running sums add up to the consistency sums the inner verifier's final check reads,
and the M0 unit and the Definition agree with the reference on an honest opening of either side and on forged ones (a
sibling, the public side bits), at every height from H = 0 (the leaf is the top) to H = 22 (the side bits run into dirs'
second word)."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest
from verity.ml.circuits.ops.boolean import gf2k as G
from verity_flock import rec_live as RL
from verity_flock import rec_open as R
from verity_flock import rec_vstar as V


def _tree(depth: int, lanes: int):
    rows = [[(i * 1000 + k) * 0x9E3779B97F4A7C15 % (1 << 128) for k in range(lanes)] for i in range(1 << depth)]
    layers = [[V.leaf(r) for r in rows]]
    while len(layers[-1]) > 1:
        lo = layers[-1]
        layers.append([hashlib.sha512(lo[i] + lo[i + 1]).digest() for i in range(0, len(lo), 2)])
    return rows, layers


def _opening(rows, layers, pos: int, H: int):
    row, aux = R.pack(rows[pos], [layers[h][(pos >> h) ^ 1] for h in range(H)])
    return row, aux, R.dirs_words(pos, H), R.expected(layers[H][pos >> H])


def _acc_coef(lanes: int, seed: int) -> tuple[list[int], list[int]]:
    g = random.Random(seed)
    acc = R.words([g.getrandbits(128), g.getrandbits(128)]) + [g.getrandbits(16) for _ in range(R.ACC_WORDS - 16)]
    return acc, R.words([g.getrandbits(128) for _ in range(2 * lanes)])


def test_the_reference_climbs_to_the_top_nodes_registered_row():
    rows, layers = _tree(4, 8)
    assert R.row_words(8) == 64 and R.aux_words(1) == R.aux_words(2) == 64 and R.aux_words(3) == 128 and R.coef_words(8) == 128
    assert R.compressions(8, 1) == 4 and R.compressions(64, 8) == 25
    acc, coef = _acc_coef(8, 0)
    for H in (1, 2, 4):
        for pos in range(16):
            row, aux, dirs, want = _opening(rows, layers, pos, H)
            assert want == RL.top_rows(layers[H][pos >> H])[0]
            assert R.open_ref(8, H, row, aux, acc, coef, dirs)[0] == want
            q = V.Query(0, 0, 0, pos, 4, 4 - H, layers[H][pos >> H], rows[pos], None,
                        [layers[h][(pos >> h) ^ 1] for h in range(H)])
            assert V.opens(q)
    row, aux, dirs, want = _opening(rows, layers, 5, 2)
    for at in (0, 31, 32, 63):
        bad = list(aux)
        bad[at] ^= 1
        assert R.open_ref(8, 2, row, bad, acc, coef, dirs)[0] != want, at
    assert R.open_ref(8, 2, [row[0] ^ 1] + row[1:], aux, acc, coef, dirs)[0] != want
    for d in (dirs[0] ^ 1, dirs[0] ^ 2):
        assert R.open_ref(8, 2, row, aux, acc, coef, [d] + dirs[1:])[0] != want
    assert R.open_ref(8, 2, row, aux, acc, coef, [dirs[0] ^ 4] + dirs[1:])[0] == want


def test_the_running_sums_add_the_levels_consistency_sums_at_the_verifiers_coefficients():
    """Openings j = 0 .. Q-1 of one level, chained from acc = 0 with c_r[l] = κ_{r, l} · ea[j]: the last acc_out holds
    Σ_l κ_{r, l} · Σ_j ea[j] · row_j[l], the final check's share of the level's consistency sums, and its other words are
    the first acc's."""
    g = random.Random(7)
    lanes, Q = 16, 5
    rows, layers = _tree(3, lanes)
    kappa = [[g.getrandbits(128) for _ in range(lanes)] for _ in range(2)]
    ea = [g.getrandbits(128) for _ in range(Q)]
    tail = [g.getrandbits(16) for _ in range(R.ACC_WORDS - 16)]
    acc = [0] * 16 + tail
    for j in range(Q):
        row, aux, dirs, want = _opening(rows, layers, j, 1)
        coef = R.words([G.mul(kappa[r][l], ea[j]) for r in range(2) for l in range(lanes)])
        out, acc = R.open_ref(lanes, 1, row, aux, acc, coef, dirs)
        assert out == want
    cons = [0] * lanes
    for j in range(Q):
        for l in range(lanes):
            cons[l] ^= G.mul(ea[j], rows[j][l])
    want = [0, 0]
    for r in range(2):
        for l in range(lanes):
            want[r] ^= G.mul(kappa[r][l], cons[l])
    assert R.elements(acc[:16]) == want and acc[16:] == tail


@pytest.mark.parametrize(("bad", "error"), [((0, 1), ValueError), ((12, 1), ValueError), ((8, 33), ValueError),
                                            ((8, -1), ValueError), ((8.0, 1), TypeError), ((True, 1), TypeError),
                                            ((8, 1.0), TypeError)])
def test_the_shape_is_checked(bad, error):
    with pytest.raises(error):
        R.open_ref(*bad, [], [], [], [], [])


HEIGHTS = (0, 1, 16, 17, 22)


def _path(H: int, pos: int, seed: int, lanes: int = 8):
    """An opening of a tree we never build: a random row and H random siblings climb from position ``pos`` to the top
    `rec_vstar.climb` gives. ``(row, sibs, top)``."""
    g = random.Random(seed)
    row = [g.getrandbits(128) for _ in range(lanes)]
    sibs = [g.randbytes(64) for _ in range(H)]
    return row, sibs, V.climb(V.leaf(row), pos, sibs)


def _flip(dirs: list[int], h: int) -> list[int]:
    out = list(dirs)
    out[h // 16] ^= 1 << (h % 16)
    return out


@pytest.mark.parametrize("H", HEIGHTS)
def test_every_height_climbs_by_its_side_bits_and_no_other(H):
    """From H = 0, where the leaf is the top and ``dirs`` is not read, to H = 22, whose side bits 16 .. 21 are ``dirs``' word
    1: the honest opening gives its top's row and `rec_vstar.opens` agrees; each side bit below H flipped, a sibling or the
    row forged gives another; a side bit at or above H, and ``aux`` past its H siblings (all of it at H = 0), is not read."""
    pos = 0x2DB6A5 & ((1 << H) - 1)
    rowv, sibs, top = _path(H, pos, H)
    row, aux = R.pack(rowv, sibs)
    dirs = R.dirs_words(pos, H)
    acc, coef = _acc_coef(8, H)
    want = R.expected(top)
    assert len(aux) == R.aux_words(H) == 64 * max(1, (H + 1) // 2) and len(dirs) == R.DIRS_WORDS
    assert dirs[0] == pos & 0xFFFF and dirs[1] == pos >> 16 and not any(dirs[2:])
    assert R.compressions(8, H) == 2 + 2 * H
    assert V.opens(V.Query(0, 0, 0, pos, H, 0, top, rowv, None, sibs))
    assert R.open_ref(8, H, row, aux, acc, coef, dirs)[0] == want
    if H == 0:
        assert want == R.expected(V.leaf(rowv))
    for h in range(H):
        assert R.open_ref(8, H, row, aux, acc, coef, _flip(dirs, h))[0] != want, h
    for h in {H, 31}:
        assert R.open_ref(8, H, row, aux, acc, coef, _flip(dirs, h))[0] == want, h
    for at in (0, 32 * H - 1, len(aux) - 1):
        bad = list(aux)
        bad[at % len(aux)] ^= 1
        assert (R.open_ref(8, H, row, bad, acc, coef, dirs)[0] == want) == (at % len(aux) >= 32 * H), at
    assert R.open_ref(8, H, [row[0] ^ 1] + row[1:], aux, acc, coef, dirs)[0] != want
    assert R.dirs_words(pos | 1 << H, H) == dirs


def _cases(H: int):
    """Two honest openings (side bit 0 of either value, and above 16 bit 16 of either), then forgeries whose ``want`` is the
    honest node: a sibling (the row at H = 0), side bit 0 and, above 16, side bit 16, which is ``dirs``' word 1."""
    m = (1 << H) - 1
    honest = []
    for t, pos in enumerate((0x15555 & m, 0x0AAAA & m)):
        rowv, sibs, top = _path(H, pos, 100 + t)
        honest.append((*R.pack(rowv, sibs), R.dirs_words(pos, H), R.expected(top), *_acc_coef(8, t)))
    row, aux, dirs, want, acc, coef = honest[0]
    forged = [(row, [aux[0] ^ 0x8000] + aux[1:], dirs, want, acc, coef) if H else
              ([row[0] ^ 0x8000] + row[1:], aux, dirs, want, acc, coef)]
    forged += [(row, aux, _flip(dirs, h), want, acc, coef) for h in (0, 16) if h < H]
    return honest + forged


def _bits(words):
    return [(w >> j) & 1 for w in words for j in range(16)]


@pytest.mark.slow
@pytest.mark.parametrize("H", [0, 1, 17])
def test_the_m0_unit_computes_the_reference(H):
    """The unit (`forms`, 8 lanes) on two honest openings and on the forged ones, bit-sliced: at H = 0 (no climb, no
    ``aux``), one level, and 17 (the first height whose side bits reach ``dirs``' word 1)."""
    cases = _cases(H)
    C, ins, outs = R.unit_circuit(8, H)
    vals = {nm: [0] * 16 for nm in ins}
    for t, (row, aux, dirs, _, acc, coef) in enumerate(cases):
        for nm, w in zip(ins, row + aux + acc + coef + dirs, strict=True):
            for j in range(16):
                vals[nm][j] |= ((w >> j) & 1) << t
    got, fail = C.evaluate(vals, len(cases))
    assert fail == 0
    for t, (row, aux, dirs, want, acc, coef) in enumerate(cases):
        words = [sum(((got[nm][j] >> t) & 1) << j for j in range(16)) for nm in outs]
        out, acc_out = R.open_ref(8, H, row, aux, acc, coef, dirs)
        assert words == out + acc_out
        assert (out == want) == (t < 2)


@pytest.mark.slow
@pytest.mark.parametrize("H", [0, 1, 17])
def test_the_definition_computes_the_reference(H):
    """`RecOpen{8, H}` evaluated bit-sliced from its descriptor, on the same cases as the unit."""
    from verity.ml.circuits.evaluation import evaluate_batch

    cases = _cases(H)
    d = R.definition(8, H)
    assert d.id == f"RecOpen_v4{{LANES=8,H={H}}}"
    port = lambda i: np.array([np.array(_bits(c[i]), dtype=np.uint8).reshape(-1, 16) for c in cases])
    res = np.asarray(evaluate_batch(d, port(0), port(1), port(4), port(5), port(2)))
    res = res.reshape(len(cases), R.OUT_WORDS + R.ACC_WORDS, 16)
    out, acc_out = res[:, :R.OUT_WORDS], res[:, R.OUT_WORDS:]
    word = lambda w: int(sum(int(b) << j for j, b in enumerate(w)))
    for t, (r, a, dirs, want, acc, coef) in enumerate(cases):
        ref = R.open_ref(8, H, r, a, acc, coef, dirs)
        assert [word(w) for w in out[t]] == ref[0] and [word(w) for w in acc_out[t]] == ref[1]
        assert (ref[0] == want) == (t < 2)
