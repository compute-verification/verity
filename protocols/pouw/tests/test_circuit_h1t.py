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
    out = evaluate(h1t.linear(c.shape, c.bias is not None), *c.args())
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


# ---- the verifier's inputs, layout and window -------------------------------------------------------------------------

def _anchors(c: R.Call):
    from verity_pouw.circuit.anchors import H1tAnchors

    return H1tAnchors(c.salt, c.forward, c.layer, c.site, c.weight, tuple(c.rowids), tuple(range(c.shape.n)))


def test_the_verifier_builds_every_anchored_input():
    """From its anchors, the verifier's inputs are exactly the reference call's: the strip ids are its rows' // 16, and the
    tag tuples its columns'."""
    from verity_pouw.circuit import traced_h1t

    c = call(32, 96, 32)
    t = traced_h1t(32, 96, 32)
    assert t.definition is h1t.linear(c.shape) and t.served == 32 * 32
    assert t.call_args(_anchors(c), *c.args()[:5]) == c.args()


def test_a_prover_s_anchored_inputs_are_refused():
    """Repeated strip or row ids (X-SPW-2's duplicate), another column's tags, another call index or weight id: each is
    refused, as are anchors whose rows are not aligned strips or whose columns the tags don't tell apart."""
    from verity_pouw.circuit.anchors import H1tAnchors

    c = call(32, 96, 16)
    an, fn = _anchors(c), h1t.linear(c.shape)
    an.check_call(fn, c.args())
    names = [p for p, _t in fn.signature()[0]]
    moved = {"stripids": [9, 9], "rowids": list(range(16)) * 2, "tags": [x for j in range(1, 17) for x in h1t.tag_tuple(j)],
             "forward": [4, 0], "weight": [12, 0]}
    for name, value in moved.items():
        args = list(c.args())
        assert args[names.index(name)] != value
        args[names.index(name)] = value
        with pytest.raises(DomainError):
            an.check_call(fn, args)
    with pytest.raises(DomainError):
        H1tAnchors(c.salt, 3, 5, 2, 11, tuple(range(8, 40)), tuple(range(16))).check(c.shape)
    with pytest.raises(DomainError):
        H1tAnchors(c.salt, 3, 5, 2, 11, tuple(c.rowids), (h1t.TAG_COLUMNS, *range(15))).check(c.shape)


def test_the_verifier_refuses_shapes_outside_the_granted_scope():
    """`TTH1T` is granted for layout A at 16 × 16 tiles with k <= 32,768; the verifier builds no Program outside it."""
    from verity_pouw.circuit import traced_h1t

    for shape in ((24, 96, 16), (16, 96, 24), (16, 32_776, 16), (16, 29, 16)):
        with pytest.raises(DomainError):
            traced_h1t(*shape)


def test_calls_on_one_weight_compose_without_recompute():
    """Two calls on one weight in one root (`H1tWindow`), each with its own anchored indices: no value is computed in two
    units, since every key gate reads its strip id and every tile gate its call's X strip."""
    uc = P.circuit_cut(P.h1t_window_circuit(h1t.Shape(16, 96, 16), 2))
    assert uc.verdict.ok, uc.verdict.codes
    assert Counter(u.role for u in uc.units) == {"key": 2, "form_x": 2, "tile": 2, "dequant": 512}


def test_the_layout_is_the_verifier_s():
    """Each tile's closure is the tile, its X strip's forming unit and that strip's key (t ∈ cl t). `WorkLaw()` draws it,
    opening the rows its units read, B′'s included. A prover's layout mapping a tile to its X strip (C1) is refused, by
    the verifier's comparison and by `draw`."""
    from dataclasses import replace

    from verity.randomness import derive
    from verity_pouw.circuit import plan as PL
    from verity_pouw.circuit import traced_h1t

    t = traced_h1t(32, 96, 32)
    lay = t.layout()
    us = lay.units
    for tile, cl in lay.closure.items():
        assert [us[u].role for u in cl] == ["tile", "form_x", "key"] and us[cl[1]].where[0] == us[tile].where[0]
    assert [s.name for s in lay.strata] == list(dict.fromkeys(u.template for u in us))       # one per template (X-SPC-84)
    assert PL.work_table(lay) == {s.name: {"work": lay.tile_work() if s.role == "tile" else 0, "floor": 1} for s in lay.strata}
    key = derive(b"verifier-secret", PL.DRAW_DOMAIN, {"receipt": b"r"})
    plan = PL.draw(key, lay, PL.WorkLaw())
    assert set(lay.closure) <= set(plan.drawn) <= set(plan.proved)
    assert set(plan.openings) == {("a", i) for i in range(32)} | {("b", j) for j in range(32)}
    t.check_layout(lay)
    tile = min(lay.closure)
    xs = lay.closure[tile][1]
    headed_by_x = replace(lay, closure={**lay.closure, tile: (xs, *(u for u in lay.closure[tile] if u not in (tile, xs)))})
    strata = tuple(replace(s, units=tuple(xs if u == tile else u for u in s.units)) if s.role == "tile" else s
                   for s in lay.strata)
    x_as_tile = replace(lay, strata=strata, closure={(xs if u == tile else u): c for u, c in lay.closure.items()})
    for bad in (headed_by_x, x_as_tile):
        with pytest.raises(DomainError):
            t.check_layout(bad)
        with pytest.raises(DomainError):
            PL.draw(key, bad, PL.WorkLaw())


def test_the_width_rule_refuses_an_empty_served_set():
    c, uc = _check(h1t.Shape(16, 96, 16))
    with pytest.raises(ValueError):
        P.circuit_width_rule(c, served=0, uc=uc)


def _window(t, calls, source=bytes(range(32))):
    """A window over ``calls`` (`H1tAnchors`, commitment) of ``t``'s shape: its `Receipt` and the `Ledger` opened from it,
    with the fresh source replaced by ``source`` so the draws are reproducible."""
    from unittest import mock

    from verity_pouw.circuit import anchors as A

    receipt = A.Receipt(tuple((an, t.shape, commitment) for an, commitment in calls))
    with mock.patch.object(A, "_fresh_source", lambda: source):
        return receipt, A.Ledger.open(receipt)


def test_equal_calls_under_one_key_draw_different_tiles():
    """X-SPC-96 on H-1T: each stratum's subset is keyed by the call's (forward, position) with its template, so equal calls
    under the window's one key draw different tiles; `Traced.draw` draws over the verifier's own layout (X-SPC-97), under
    the key of the ledger opened from the window's receipt. A call the receipt doesn't list, or one already drawn, is
    refused before it draws (C1, M1)."""
    import hashlib

    from verity_pouw.circuit import plan as PL
    from verity_pouw.circuit import traced_h1t
    from verity_pouw.circuit.anchors import H1tAnchors

    c = call(64, 96, 64)
    t, law = traced_h1t(64, 96, 64), PL.WorkLaw(4)
    lay = t.layout()
    ans = {f: H1tAnchors(c.salt, f, c.layer, c.site, c.weight, tuple(c.rowids), tuple(range(64))) for f in (3, 4, 5, 6, 7)}
    _receipt, window = _window(t, [(ans[f], hashlib.sha256(b"call %d" % f).digest()) for f in (3, 4, 5, 6)])
    tiles = lambda plan: [u for u in plan.drawn if lay.units[u].role == "tile"]  # noqa: E731
    drawn = {f: tiles(t.draw(law, ans[f], window)) for f in (3, 4, 5, 6)}
    assert drawn[3] == tiles(PL.draw(window.key, lay, law, ans[3].call_index))
    assert all(drawn[f] != drawn[3] for f in (4, 5, 6))
    for again, match in ((ans[3], "repeated"), (ans[7], "not on the window's receipt")):
        with pytest.raises(DomainError, match=match):
            t.draw(law, again, window)


# ---- the served output: vLLM's epilogue order --------------------------------------------------------------------------

def scaled_epilogue(z: int, s: int, t: int, bias: int | None) -> int:
    """vLLM's `cutlass_scaled_mm` epilogue as the vLLM integration pins it (`verity_vllm.program.registry.fp8`
    `ScaledMmFp8Coordinate`, 0 mismatches against `_C.cutlass_scaled_mm` on the production capture): the batch-invariant
    dispatch swaps the scales, so Compute0 = RN(s_i·acc) with the row (token) scale, and Compute1 = FFMA(t_j, ·, f32(b_j))
    (`ScaledEpilogueBias`) or RN(t_j·) (`ScaledEpilogue`), then `cvt.rn.bf16.f32`. Written here on core's bit-level
    evaluators, since `protocols/pouw` reads nothing of the integration."""
    from verity.ml.fp32 import bf16_to_f32_bits, f32_fma_bits, f32_mul_bits
    from verity.ml.tc.cast import f32_to_bf16_hw_word

    m = f32_mul_bits(s, z)
    return f32_to_bf16_hw_word(f32_mul_bits(t, m) if bias is None else f32_fma_bits(t, m, bf16_to_f32_bits(bias)))


@pytest.mark.parametrize("shape,bias", [((16, 96, 16), True), ((16, 96, 16), False), ((32, 96, 32), True),
                                        ((16, 232, 16), True), ((16, 232, 16), False)])
def test_the_served_output_is_vllm_s_scaled_epilogue(shape, bias):
    """y is what vLLM serves: each output is `scaled_epilogue` of the chain's Z, the row scale, the column scale and the
    bias (none for a linear without one), bit for bit."""
    c = call(*shape)
    if not bias:
        c = R.Call(**{**c.__dict__, "bias": None})
    ref = R.evaluate(c)
    y, _words = run(c)
    want = [[scaled_epilogue(ref.z[i][j], c.s[i], c.t[j], c.bias[j] if bias else None) for j in range(shape[2])]
            for i in range(shape[0])]
    assert y == want == ref.out


def test_the_dequantization_unit_is_the_epilogue_on_hard_words():
    """`H1tDequant{BIAS}` alone, against `scaled_epilogue` on words across the exponent range, subnormals and ties, and
    the case that separates the two variants: Z = −0 without a bias serves −0, and an FFMA with a +0 bias would serve +0.
    The branch's earlier order, bf16(RN(RN(s·t)·Z)) then a bf16 add, differs from the served one on many of these."""
    import struct

    rng = random.Random(29)

    def f32(x):
        return struct.unpack("<I", struct.pack("<f", x))[0]

    zs = [f32(rng.uniform(-1, 1) * 2.0 ** rng.randint(-140, 100)) for _ in range(400)] + [0, 0x80000000, 1, 0x807FFFFF]
    old_differs = 0
    for z in zs:
        s, t, b = f32(rng.uniform(1e-4, 4) * 2.0 ** rng.randint(-20, 20)), f32(rng.uniform(1e-4, 4)), rng.getrandbits(15)
        assert evaluate(h1t.dequant_template(True), z, s, t, b)[0] == scaled_epilogue(z, s, t, b)
        assert evaluate(h1t.dequant_template(False), z, s, t)[0] == scaled_epilogue(z, s, t, None)
        from verity.ir.defs import REGISTRY

        old = REGISTRY["Bf16Add_v1"].evaluate(REGISTRY["F32ToBf16Rn_v1"].evaluate(
            REGISTRY["F32Mul_v2"].evaluate(REGISTRY["F32Mul_v2"].evaluate(s, t), z)), b)
        old_differs += old != scaled_epilogue(z, s, t, b)
    assert old_differs > 0
    one = f32(1.0)
    assert evaluate(h1t.dequant_template(False), 0x80000000, one, one)[0] == 0x8000
    assert evaluate(h1t.dequant_template(True), 0x80000000, one, one, 0)[0] == 0x0000


def test_a_key_drawn_at_its_floor_opens_its_strip_s_rows():
    """H-1T's key reads its strip's A rows itself, so a key drawn at its own floor for a strip no drawn tile covers is
    proved over rows the plan must open (X-SPC-110): every proved key's and X unit's strip rows are opened, and some of the
    plans here draw such a key."""
    from verity_pouw.circuit import plan as PL
    from verity_pouw.circuit import traced_h1t
    from verity_pouw.circuit.anchors import H1tAnchors

    c = call(48, 64, 32)
    t = traced_h1t(48, 64, 32)
    lay = t.layout()
    an = H1tAnchors(c.salt, c.forward, c.layer, c.site, c.weight, tuple(c.rowids), tuple(range(32)))
    lonely = 0
    for i in range(60):
        _receipt, window = _window(t, [(an, bytes(32))], source=bytes([i]) * 32)
        plan = t.draw(PL.WorkLaw(2), an, window)
        units = [lay.units[u] for u in plan.proved]
        tile_strips = {u.where[0] for u in units if u.role == "tile"}
        for u in units:
            if u.role in ("key", "form_x"):
                strip = u.where[0] if u.where else u.member
                assert {("a", 16 * strip + r) for r in range(16)} <= set(plan.openings)
                lonely += u.role == "key" and strip not in tile_strips
    assert lonely > 0


def test_a_window_s_defaults_cover_every_call_and_its_inputs_are_the_verifier_s():
    """X-SPC-111: a window's separator and width rule read every call's y by default (`Circuit.calls`), and
    `anchors.h1t_window_inputs` builds its anchored inputs from each call's `H1tAnchors`, refusing a repeated call index or
    calls of another salt, weight id or columns."""
    from verity_pouw.circuit.anchors import H1tAnchors, h1t_window_inputs

    shape = h1t.Shape(16, 96, 16)
    c = P.h1t_window_circuit(shape, 2)
    uc = P.circuit_cut(c)
    sep, w = P.circuit_separator(c, uc), P.circuit_width_rule(c, uc=uc)
    assert sep.ok and sep.per_tile_bits == 32 * 256 and w.ok and len(w.checked) == 2 * 256
    c1, c2 = call(16, 96, 16), call(16, 96, 16)
    an = [H1tAnchors(c1.salt, f, c1.layer, c1.site, c1.weight, tuple(c1.rowids), tuple(range(16))) for f in (3, 4)]
    got = h1t_window_inputs(an, shape)
    names = [p for p, _t in h1t.linear(shape).signature()[0]]
    assert got["salt"] == c1.args()[names.index("salt")] and got["tags"] == c1.args()[names.index("tags")]
    assert got["forward"] == [3, 0, 4, 0] and got["rowids"] == list(c1.rowids) * 2
    for bad in ([an[0], an[0]], [an[0], H1tAnchors(bytes(32), 4, 5, 2, 11, tuple(c1.rowids), tuple(range(16)))],
                [an[0], H1tAnchors(c1.salt, 4, 5, 2, 12, tuple(c1.rowids), tuple(range(16)))]):
        with pytest.raises(DomainError):
            h1t_window_inputs(bad, shape)


def test_y_is_drawn_at_option_b_s_floors():
    """Root's option B (X-SPC-105) on H-1T: in a window, each call's `H1tDequant` has floor ⌈K_y·n_c/N⌉ in its work table,
    and every other zero-work template 1."""
    from verity_pouw.circuit import plan as PL
    from verity_pouw.circuit import traced_h1t

    lay = traced_h1t(32, 96, 32).layout()
    (law, *_) = PL.window_laws([lay] * 100)
    table, deq = PL.work_table(lay, law), next(s for s in lay.strata if s.role == "dequant")
    assert table[deq.name] == {"work": 0, "floor": -(-PL.RECORD_K // 100)} and law.sizes(lay.strata)[deq.name] == 278
    assert all(table[s.name]["floor"] == 1 for s in lay.strata if s.role != "dequant")
