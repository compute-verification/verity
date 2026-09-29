"""H-1T's circuit (`verity_pouw.circuit.h1t`, layout A): its forming and chain are Track H's pinned `h1t_ref` and `chain`
bit for bit (`tests/vectors/h1t_ref.json`), its Program evaluates to its reference, its partition passes the whole-call
`validate_unit_cut`, and every PoUW unit is behind the pinned Z separator."""
from __future__ import annotations

import json
import random
import struct
from collections import Counter
from pathlib import Path

import pytest

from verity.evaluation import evaluate
from verity_pouw.circuit import h1t
from verity_pouw.circuit import h1t_reference as R
from verity_pouw.circuit import h1t_words as HW
from verity_pouw.circuit import partition as P
from verity_pouw.protocol import DomainError

VECTORS = json.loads((Path(__file__).parent / "vectors" / "h1t_ref.json").read_text())


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _legal(rng: random.Random) -> int:
    """A finite E4M3 code with |x| <= 416."""
    mag = rng.randrange(0, 0x7E)
    return 0 if mag == 0 else mag | (rng.getrandbits(1) << 7)


def call(m: int, k: int, n: int, seed: int = 7, *, a=None, rowids=None, stripids=None) -> R.Call:
    rng = random.Random(seed)
    a = tuple(tuple(_legal(rng) for _ in range(k)) for _ in range(m)) if a is None else a
    b = tuple(tuple(_legal(rng) for _ in range(k)) for _ in range(n))
    return R.Call(a, tuple(_f32(rng.uniform(1e-3, 2e-2)) for _ in range(m)), b, tuple(_f32(rng.uniform(1e-3, 2e-2)) for _ in range(n)),
                  tuple(_f32(rng.uniform(-1, 1)) >> 16 for _ in range(n)), bytes(rng.randrange(256) for _ in range(32)),
                  3, 5, 2, 11, tuple(range(40, 40 + m // 16)) if stripids is None else tuple(stripids),
                  tuple(range(640, 640 + m)) if rowids is None else tuple(rowids))


def run(c: R.Call) -> tuple[list[list[int]], list[list[int]]]:
    out = evaluate(h1t.linear(c.shape), *c.args())
    m, n, per = c.shape.m, c.shape.n, h1t.D * h1t.D * h1t.words_per_cell(c.shape.k)
    tiles = (m // h1t.D) * (n // h1t.D)
    assert len(out) == m * n + per * tiles
    return ([list(out[i * n:(i + 1) * n]) for i in range(m)],
            [list(out[m * n + t * per:m * n + (t + 1) * per]) for t in range(tiles)])


# ---- Track H's map, bit for bit ---------------------------------------------------------------------------------------

def test_the_primitives_are_track_h_s_pinned_map():
    """Per slice: the floor from the 29 real lanes, x1 and x2 per real lane, v and −v per tag lane, from the raw salt
    words in the honest kernel's bit layout. The vectors are Track H's own `h1t_ref` on those words."""
    for case in VECTORS["cases"]:
        T = case["T"]
        for i, row in enumerate(case["x"]):
            xr = row + [0] * (h1t.REAL * T - case["k"])
            x1, x2 = [], []
            for t in range(T):
                x, r = xr[h1t.REAL * t:h1t.REAL * (t + 1)], case["salt_words"][i][t]
                e = HW.PouwH1tSliceExp.evaluate(*x)
                for lane in range(h1t.REAL):
                    x1.append(HW.PouwH1tX1.evaluate(x[lane], e, r[lane // 2], lane % 2))
                    x2.append(HW.PouwH1tX2.evaluate(x[lane], e, r[lane // 2], lane % 2))
                for word, sel in ((r[17], 1), (r[16], 0), (r[16], 1)):
                    v = HW.PouwH1tTag.evaluate(e, word, sel)
                    x1.append(v)
                    x2.append(HW.PouwE4m3Neg.evaluate(v))
            assert (x1, x2) == (case["x1"][i], case["x2"][i])


def test_the_reference_is_track_h_s_map_and_chain():
    """The reference's forming, B′'s slices with their tag tuples, and each cell's chain equal Track H's `h1t_ref`,
    `layout_weights` and `chain`: the credited I_1 … I_2T, R_2 … R_(2T−2), and Z = R_2T."""
    for case in VECTORS["cases"]:
        T = case["T"]
        assert [list(h1t.tag_tuple(j)) for j in range(len(case["w"]))] == case["tags"]
        for i, row in enumerate(case["x"]):
            x1, x2 = R.form_row(row, case["salt_words"][i])
            assert (x1, x2) == (case["x1"][i], case["x2"][i])
            for j, w in enumerate(case["w"]):
                words, z = R.cell_words(x1, x2, R.weight_slices(w, h1t.tag_tuple(j), T))
                assert words == case["words"][i][j][:4 * T - 3] and z == case["words"][i][j][-1]


def test_the_primitives_are_total():
    """Specified on legal inputs (finite, |x| <= 416); a NaN code reads as 0 and the floor byte is taken mod 256."""
    assert HW.PouwH1tSliceExp.evaluate(*([0x7F] + [0] * 28)) == HW.E_BIAS - 2
    for e in (0, 255):
        HW.PouwH1tX1.evaluate(0xFF, e, 0xFFFFFFFF, 1)
        HW.PouwH1tTag.evaluate(e, 0, 0)


# ---- the Program ------------------------------------------------------------------------------------------------------

def test_the_program_evaluates_to_the_reference():
    c = call(16, 96, 16)
    ref = R.evaluate(c)
    y, words = run(c)
    assert words == ref.words and y == ref.out


@pytest.mark.slow
def test_the_program_evaluates_to_the_reference_on_two_strips_and_a_long_k():
    """Two strips of each kind (two keys), and k = 232 (T = 8, the last slice 0 on 0 lanes past k)."""
    for shape in ((32, 96, 32), (16, 232, 16)):
        c = call(*shape, seed=5)
        ref = R.evaluate(c)
        y, words = run(c)
        assert words == ref.words and y == ref.out


def test_index_binding_separates_identical_strips():
    """Two identical strips get different salts, so different words, because the strip and row indices are in the key
    and the stream (X-SPW-2); with equal indices they would be credited twice."""
    strip = tuple(tuple((7 * i + j) % 0x7D for j in range(96)) for i in range(16))
    ref = R.evaluate(call(32, 96, 16, a=strip + strip))
    assert ref.words[0] != ref.words[1]
    same = R.evaluate(call(32, 96, 16, a=strip + strip, rowids=list(range(16)) * 2, stripids=[9, 9]))
    assert same.words[0] == same.words[1]


def test_shapes_outside_the_circuit_are_refused():
    for shape in ((24, 96, 16), (16, 29, 16), (16, 100, 16), (16, 40_000, 16)):
        with pytest.raises(DomainError):
            h1t.linear(h1t.Shape(*shape))


# ---- the partition ----------------------------------------------------------------------------------------------------

def _check(shape: h1t.Shape):
    c = P.h1t_circuit(shape)
    return c, P.circuit_cut(c)


def test_the_whole_call_passes_validate_unit_cut():
    """Two strips of each kind: every computing gate certified once, reads across units only through committed values,
    and no value computed in two units."""
    c, uc = _check(h1t.Shape(32, 96, 32))
    assert uc.verdict.ok, uc.verdict.codes
    assert Counter(u.role for u in uc.units) == {"key": 2, "form_x": 2, "tile": 4, "dequant": 1024}


def test_the_separator_is_z_and_the_width_rule_holds_with_the_pinned_port():
    c, uc = _check(h1t.Shape(32, 96, 32))
    sep = P.circuit_separator(c, uc)
    assert sep.ok and sep.per_tile_bits == 32 * h1t.D * h1t.D and sep.reached_pouw == 0 and sep.words_delivered == 0
    w = P.circuit_width_rule(c, uc=uc)
    assert w.ok and {uc.units[u].role for u in w.checked} == {"dequant"} and w.exempt == len(uc.units) - 1024


def test_delivered_words_fail_the_width_rule():
    """Were the tiles' words served beside y, the tiles and everything upstream would reach a served output other than
    through Z, and fail."""
    c, uc = _check(h1t.Shape(16, 96, 16))
    w = P.circuit_width_rule(c, served=len(uc.graph.outputs), uc=uc)
    assert not w.ok and {uc.units[u].role for u, _, _ in w.too_wide} == set(P.H1T_POUW_ROLES)
