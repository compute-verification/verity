"""fold_xor_search.py: its menu is the sheet's, its forms compute what they model, and its CEGIS proves and finds."""
from __future__ import annotations

import itertools
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

POUW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POUW / "theory"))
sys.path.insert(0, str(POUW))
try:
    import fold_xor_search as S
    import z3
    from facet_break.loads import class_of
finally:
    sys.path.remove(str(POUW / "theory"))
    sys.path.remove(str(POUW))


def _val(e) -> int:
    return z3.simplify(e).as_long()


def test_cover_is_the_sheets_menu():
    """Every opcode the search covers is one DS-100 prices on the IMAD or FP pipe, every form it names is modelled, and
    what's off the menu isn't; fold_xor.py's menu and its 78's SASS forms are among the covered."""
    want = {op: "FFMA" if S.COVER[op] in S.FP_OPS else "IMAD" for op in S.COVER}
    assert {op: class_of(op) for op in S.COVER} == want
    assert set(S.COVER.values()) == set(S.FORMS) == set(S.INT_OPS + S.FP_OPS)
    assert all(class_of(op) not in ("IMAD", "FFMA") for op in S.OFF_MENU)
    assert {"IMAD", "IMAD.HI.U32", "IDP.4A.U8.U8", "IDP.2A.LO.U16.U8", "FFMA2", "FADD2", "FMUL2", "FFMA"} <= set(S.COVER)


def test_forms_compute_what_they_model():
    """The integer forms against Python on concrete words, with the modifiers off and on, and an FFMA lane against
    numpy's binary32 add (b = 1.0) and its canonical NaN."""
    off = {n: z3.BoolVal(False) for n in ("sa", "sb", "na", "nb", "np", "nc", "inv", "cx", "hi")}
    rng = np.random.default_rng(7)
    for a, b, c, h in rng.integers(0, 2**32, size=(64, 4), dtype=np.uint64).tolist():
        o = [S.bv(a), S.bv(b), S.bv(c), S.bv(h)]
        assert _val(S.imad(o, off)[0]) == (a * b + c) % 2**32
        on = {**off, "np": z3.BoolVal(True), "nc": z3.BoolVal(True), "cx": z3.BoolVal(True)}
        assert _val(S.imad(o, on)[0]) == (-(a * b) - c + 1) % 2**32
        s = a * b + (h << 32 | c)
        assert [_val(v) for v in S.imad_hi(o, off)] == [s % 2**32, (s >> 32) % 2**32]
        sa = a - 2**32 if a >> 31 else a
        signed = {**off, "sa": z3.BoolVal(True)}
        assert _val(S.imad_hi(o, signed)[1]) == ((sa * b + (h << 32 | c)) >> 32) % 2**32
        dp = c + sum(((a >> 8 * i) & 255) * ((b >> 8 * i) & 255) for i in range(4))
        assert _val(S.idp4a(o, off)[0]) == dp % 2**32
    m = {n + "0": z3.BoolVal(False) for n in ("ftz", "sat", "satz", "fmz", "fmzx", "absa", "absb", "absc", "nega",
                                               "negb", "negc")}
    m["rm0"] = z3.RNE()
    one = S.bv(0x3F800000)
    for x, y in rng.integers(0, 2**32, size=(32, 2), dtype=np.uint64).tolist():
        fx, fy = (np.frombuffer(struct.pack("<I", v), dtype=np.float32)[0] for v in (x, y))
        with np.errstate(all="ignore"):
            r = np.float32(fx + fy)
        want = 0x7FFFFFFF if np.isnan(r) else struct.unpack("<I", r.tobytes())[0]
        assert _val(S.lane(S.bv(x), one, S.bv(y), m, "0")) == want


def test_an_fp_lane_on_nan_inputs_is_its_constants():
    """reduce_nan's premise: with x and y binary32 NaN patterns, an FFMA lane whose operands are x, y or constants (an
    earlier FP result is one, by induction) computes the same word on any two such pairs, under every modifier."""
    m = S.modifiers("FFMA2", "t")
    X1, X2, Y1, Y2 = (z3.BitVec(n, 32) for n in ("X1", "X2", "Y1", "Y2"))
    nan = z3.And(*(S.nan_bits(v) for v in (X1, X2, Y1, Y2)))
    ks = [z3.BitVec(f"k{i}", 32) for i in range(3)]
    for picks in itertools.product("xyc", repeat=3):
        o1 = [{"x": X1, "y": Y1, "c": k}[p] for p, k in zip(picks, ks)]
        o2 = [{"x": X2, "y": Y2, "c": k}[p] for p, k in zip(picks, ks)]
        s = z3.Solver()
        s.add(nan, S.lane(*o1, m, "0") != S.lane(*o2, m, "0"))
        assert s.check() == z3.unsat, picks


def test_reduce_nan_makes_fp_results_constants():
    """An FP-first sequence becomes its integer suffix, each FP result read as that instruction's constant and the
    later results renumbered; an FP form after an integer one doesn't reduce."""
    assert S.reduce_nan(("FFMA2", ("IMAD", (0, 2)))) == (("IMAD", (0, 2)),)
    assert S.reduce_nan(("FFMA2", ("IMAD", (3, 4)))) == (("IMAD", (2, 2)),)
    assert S.reduce_nan(("FFMA2", ("IMAD.HI", (1, 4)), ("IMAD", (4, 5)))) == (("IMAD.HI", (1, 2)), ("IMAD", (2, 3)))
    assert S.reduce_nan(("FFMA2", ("IMAD.HI", (1, 4)), ("IMAD", (2, 6)))) == (("IMAD.HI", (1, 2)), ("IMAD", (4, 4)))
    assert S.reduce_nan(("FFMA2", ("IDP.2A", (4, 0)))) == (("IDP.2A", (2, 0)),)
    assert S.reduce_nan((("IMAD", (0, 1)), "FFMA2", ("IMAD", (2, 3)))) is None


def test_splits_are_every_program():
    """A sequence's `splits` set its first instruction's addend slots to every source, and `mod_splits` its modifiers
    to every assignment but inv and cx both set, which computes what neither does."""
    seq = (("IMAD.HI", (0, 1)), ("IMAD", (2, 3)))
    cases = S.splits(seq)
    assert sorted((f[0][2], f[0][3]) for f in cases) == sorted(itertools.product(range(3), repeat=2))
    assert len({S.fix_label(f) for f in cases}) == 9 and S.splits(("FFMA2", ("IMAD", (0, 2)))) == [()]
    for op, n in (("IMAD", 12), ("IMAD.HI", 192)):
        sub = S.mod_splits(((op, (0, 1)),), ({2: 0},))
        assert len(sub) == n == len({S.fix_label(f) for f in sub})
        assert all(f[0][2] == 0 and not (f[0]["inv"] and f[0]["cx"]) for f in sub)
        assert {S.fix_label(f, mods=False) for f in sub} == {"s2=x"}
    off = {n: z3.BoolVal(False) for n in ("sa", "sb", "na", "nb", "np", "nc", "inv", "cx")}
    both = {**off, "inv": z3.BoolVal(True), "cx": z3.BoolVal(True)}
    rng = np.random.default_rng(11)
    for a, b, c, h in rng.integers(0, 2**32, size=(16, 4), dtype=np.uint64).tolist():
        o = [S.bv(a), S.bv(b), S.bv(c), S.bv(h)]
        for nc in (False, True):
            m0, m1 = {**off, "nc": z3.BoolVal(nc)}, {**both, "nc": z3.BoolVal(nc)}
            assert _val(S.imad(o, m0)[0]) == _val(S.imad(o, m1)[0])
            assert [_val(v) for v in S.imad_hi(o, m0)] == [_val(v) for v in S.imad_hi(o, m1)]


def test_swap_is_x_and_y_exchanged():
    """`swap` permutes k = 2's sequences, an involution; on IMAD.HI[0, 1] it reorders the multiplicands, and a modifier
    case's per-operand modifiers move with them, so the image computes the case's first results at (y, x)."""
    seqs = S.sequences(2, True)
    assert {S.swap(s) for s in seqs} == set(seqs) and all(S.swap(S.swap(s)) == s for s in seqs)
    assert S.swap((("IMAD.HI", (1, 2)), ("IDP.2A", (0, 3)))) == (("IMAD.HI", (0, 2)), ("IDP.2A", (1, 3)))
    seq = (("IMAD.HI", (0, 1)), ("IMAD", (2, 3)))
    fix = ({2: 0, 3: 2, "sa": True, "sb": False, "na": False, "nb": True, "np": True, "nc": False, "inv": False,
            "cx": True},)
    img = S.swap_fix(seq, fix)[0]
    assert img == {**fix[0], 2: 1, "sa": False, "sb": True, "na": True, "nb": False}
    mods = lambda f: {k: z3.BoolVal(v) for k, v in f.items() if isinstance(k, str)}
    rng = np.random.default_rng(13)
    for x, y, c in rng.integers(0, 2**32, size=(16, 3), dtype=np.uint64).tolist():
        here = S.imad_hi([S.bv(x), S.bv(y), S.bv(x), S.bv(c)], mods(fix[0]))
        there = S.imad_hi([S.bv(y), S.bv(x), S.bv(x), S.bv(c)], mods(img))
        assert [_val(v) for v in here] == [_val(v) for v in there]


def test_merge_reads_through_the_swap():
    """A sequence is unsat when its image is, or when each case is (by itself, its image or each sub-case); with one
    sub-case short, it and its image stay open, and with it, both are unsat."""
    seqs = S.sequences(1, True)
    assert S.merge(1, [{"ops": S.label(S.swap(s)), "result": "unsat"} for s in seqs])["unsat"] == len(seqs) == 27
    seq = (("IMAD.HI", (0, 2)),)
    img = S.swap(seq)
    assert img == (("IMAD.HI", (1, 2)),)
    rest = [{"ops": S.label(s), "result": "unsat"} for s in seqs if s not in (seq, img)]
    rest.append({"ops": S.label(img), "result": "unknown"})
    cases = S.splits(seq)
    recs = [{"ops": f"{S.label(img)} /{S.fix_label(S.swap_fix(seq, f))}", "result": "unsat"} for f in cases[1:]]
    subs = [{"ops": f"{S.label(seq)} /{S.fix_label(g)}", "result": "unsat"} for g in S.mod_splits(seq, cases[0])]
    m = S.merge(1, rest + recs + subs[1:])
    assert m["unsat"] == 25 and m["open"][0] == f"{S.label(seq)} /{S.fix_label(cases[0])}"
    assert subs[0]["ops"] in m["subs"] and len(m["subs"]) == 2
    m = S.merge(1, rest + recs + subs)
    assert m["unsat"] == 27 and not m["open"], m["open"]
    summary = {"ops": S.label(seq), "result": "unsat", "cases": 1}
    assert S.merge(1, rest + recs + subs[1:] + [summary])["unsat"] == 25


@pytest.mark.slow
def test_by_word_and_anchored_are_the_products():
    """`ByWord`'s v·X and `anchored`'s A_i·B_i equal the products they stand for: ByWord at wraps, negatives and dense
    words against random X, and anchored on an x-slice's last multiplicands at random programs (selectors, constants,
    modifiers, x0 and the first product's word), against A_i·B_i itself."""
    rng = np.random.default_rng(17)
    for w in (32, 64):
        X = z3.BitVec(f"X{w}", w)
        by = S.ByWord(X)
        vs = [0, 1, 3, 1 << 31, (1 << w) - 1, (1 << w) - (1 << 20), ((1 << w) - (1 << 32) + 5) % (1 << w)]
        vs += rng.integers(0, 2**63, size=8, dtype=np.uint64).tolist()
        for v in vs:
            v %= 1 << w
            for x in rng.integers(0, 2**63, size=4, dtype=np.uint64).tolist():
                got = z3.substitute(by(v), (X, z3.BitVecVal(x, w)))
                assert _val(got) == v * x % (1 << w), (w, hex(v))
    seq = (("IMAD.HI", (0, 2)), ("IMAD.HI", (2, 3)))
    prog, cons = S.skeleton(seq, ({2: 1, 3: 0},))
    (_, s1, m1), (_, s2, m2) = prog
    X0, P1 = z3.BitVec("X0", 32), z3.BitVec("P1", 64)

    def factors(y):
        pool = [X0, S.bv(y)] + S.imad_hi([S.mux(s, [X0, S.bv(y), c]) for s, c in s1], m1, P1)
        return S.hi_factors([S.mux(s, pool + [c]) for s, c in s2], m2)

    a0, b0 = factors(0)
    by = (S.ByWord(a0), S.ByWord(b0))
    free = [v for _, slots, _ in prog for s, c in slots for v in (s, c) if not isinstance(v, int)]
    mods = [v for _, _, m in prog for v in m.values()]
    for y in (1, 1 << 17, 0xDEADBEEF):
        a, b = factors(y)
        p = S.anchored([a0, a], [b0, b], cons, 64, by)
        for _ in range(8):
            sub = [(v, z3.BitVecVal(int(rng.integers(0, 5)) if v.size() == 4 else int(rng.integers(0, 2**32)),
                                    v.size())) for v in free]
            sub += [(v, z3.BoolVal(bool(rng.integers(0, 2)))) for v in mods]
            sub += [(X0, S.bv(int(rng.integers(0, 2**32)))), (P1, z3.BitVecVal(int(rng.integers(0, 2**63)), 64))]
            assert _val(z3.substitute(p, *sub)) == _val(z3.substitute(a * b, *sub)), hex(y)


def test_values_is_every_value():
    """`values` lists every value of a term under its constraints, and None past its limit."""
    b, c = z3.Bools("b c")
    t = z3.If(b, S.bv(3), S.bv(5)) + z3.If(c, S.bv(0), S.bv(10))
    assert sorted(S.values([], t, 8)) == [3, 5, 13, 15]
    assert sorted(S.values([b], t, 8)) == [3, 13]
    assert S.values([], t, 3) is None


def test_shared_axis_is_the_word_the_products_read():
    """CEGIS shares x when the first instruction reads x alone, and with `by_mult` when both its multiplicands are x
    whatever its addends; y likewise; neither when a multiplicand is the other word or a constant, except that with
    `pow2` a word beside the constant is shared."""
    hi = lambda ab: (("IMAD.HI", ab), ("IMAD.HI", (3, 3)))
    for by_mult in (False, True):
        assert S.shared_axis(hi((0, 0)), ({2: 0, 3: 2},), by_mult) == 0
        assert S.shared_axis(hi((0, 2)), ({2: 0, 3: 0},), by_mult) == 0
        assert S.shared_axis(hi((0, 2)), ({2: 1, 3: 2},), by_mult) is None
        assert S.shared_axis(hi((0, 1)), ({2: 0, 3: 0},), by_mult) is None
        assert S.shared_axis(hi((0, 0)), (), by_mult) is None
    assert S.shared_axis(hi((0, 0)), ({2: 1, 3: 2},)) is None
    assert S.shared_axis(hi((0, 0)), ({2: 1, 3: 2},), True) == 0
    assert S.shared_axis(hi((1, 1)), ({2: 0, 3: 2},), True) == 1
    assert S.shared_axis(hi((0, 2)), ({2: 1, 3: 1},), True) is None
    assert S.shared_axis(hi((0, 2)), ({2: 1, 3: 1},), pow2=True) == 0
    assert S.shared_axis(hi((0, 1)), ({2: 0, 3: 0},), pow2=True) is None


def test_collision_inputs_agree_on_the_first_results():
    """On x·x + {y:C}, `collision_inputs` pairs inputs whose first results agree at other constants too and whose
    x ⊕ y differ."""
    import random
    seq = (("IMAD.HI", (0, 0)), ("IMAD.HI", (2, 3)))
    fix = ({2: 2, 3: 1, **dict.fromkeys(("cx", "inv", "na", "nb", "nc", "np", "sa", "sb"), False)},)
    got = S.collision_inputs(seq, fix, random.Random(5), n=2)
    assert len(got) == 4
    mods = {k: z3.BoolVal(v) for k, v in fix[0].items() if isinstance(k, str)}
    for (x1, y1), (x2, y2) in zip(got[::2], got[1::2]):
        assert x1 ^ y1 != x2 ^ y2
        for c in (0, 7, S.M32):
            first = lambda x, y: [_val(v) for v in S.imad_hi([S.bv(x), S.bv(x), S.bv(c), S.bv(y)], mods)]
            assert first(x1, y1) == first(x2, y2)


@pytest.mark.slow
def test_a_modifier_case_is_unsat():
    """A case that is unknown at 600 s unsplit (the first IMAD's addend y) is unsat once its modifiers are set; and one
    unknown at 60 s from the edge pairs (x² + {C:y}, then its high half squared) is unsat from shared values of x; one
    unknown at 120 s is unsat from colliding inputs, and one from x at powers of two."""
    seq = (("IMAD", (0, 0)), ("IMAD.HI", (2, 3)))
    fix = ({2: 1, "np": False, "nc": True, "inv": False, "cx": False},)
    assert S.cegis(seq, fix=fix)["result"] == "unsat"
    seq = (("IMAD.HI", (0, 0)), ("IMAD.HI", (3, 3)))
    fix = ({2: 1, 3: 2, **dict.fromkeys(("cx", "inv", "na", "nb", "nc", "np", "sa"), False), "sb": True},)
    # z3 takes another path once any timeout is set, as every run sets one: 3 s here against 47 s with none
    assert S.cegis(seq, timeout_ms=10**9, fix=fix, by_mult=True)["result"] == "unsat"
    seq = (("IMAD.HI", (0, 0)), ("IMAD.HI", (2, 3)))
    fix = ({2: 2, 3: 1, **dict.fromkeys(("cx", "inv", "na", "nb", "nc", "np", "sa", "sb"), False)},)
    assert S.cegis(seq, timeout_ms=10**9, fix=fix, collide=True)["result"] == "unsat"
    seq = (("IMAD.HI", (0, 2)), ("IMAD.HI", (0, 2)))
    fix = ({2: 1, 3: 1, **dict.fromkeys(("cx", "inv", "na", "nc", "np", "sa", "sb"), False), "nb": True},)
    assert S.cegis(seq, timeout_ms=10**9, fix=fix, pow2=True)["result"] == "unsat"


@pytest.mark.slow
def test_an_x_slice_is_unsat_and_finds_slices_that_exist():
    """A whole case whose 192 modifier sub-cases stayed unknown at 120 s each is unsat on its x-slice at 0x55555555
    (12 s), with the anchor product free; and the slice of x + y + K, which x·1 + y computes, is slice-sat."""
    seq = (("IMAD.HI", (0, 2)), ("IMAD.HI", (2, 4)))
    r = S.anchored_cegis(seq, ({2: 1, 3: 2},), 0x55555555, timeout_ms=10**9, free_anchor=True)
    assert r["result"] == "unsat" and r["ops"] == "IMAD.HI[0, 2] IMAD.HI[2, 4] /s2=y,s3=C", r
    seq = (("IMAD", (0, 2)), ("IMAD", (2, 3)))
    assert S.anchored_cegis(seq, ({2: 1},), 0x55555555, timeout_ms=10**9, tname="add",
                            free_anchor=True)["result"] == "slice-sat"


@pytest.mark.slow
def test_x_times_y_first_cases_are_unsat_from_the_axes():
    """Two x·y-first cases whose sub-cases stayed unknown at 300 s from colliding inputs are unsat from `axis_inputs`:
    one whole (its first product 0 on the axes, so its high half is a constant there), and one of the other's
    sub-cases (its high half y on x = 0, so the next product by a constant is a shift)."""
    seq = (("IMAD.HI", (0, 1)), ("IMAD.HI", (3, 4)))
    assert S.cegis(seq, timeout_ms=10**9, fix=({2: 0, 3: 2},), axes=True)["result"] == "unsat"
    fix = ({2: 0, 3: 1, **dict.fromkeys(("cx", "inv", "na", "nb", "nc", "np", "sa", "sb"), False)},)
    assert S.cegis(seq, timeout_ms=10**9, fix=fix, axes=True)["result"] == "unsat"


@pytest.mark.slow
def test_a_window_settles_the_hard_family_and_finds_windows_that_exist():
    """A whole case whose 192 sub-cases stayed unknown at 120 s each (x·C + {C:y}, then its low half times its high)
    is unsat on a window where its last product is affine, its count included (about 46 s, the one thing no cheaper
    test checks); the same window of y ⊕ 0 (y itself), of y ⊕ 3 (its steps equal) and of y + c* is window-sat; the
    count is sat one step below the breakpoints its multiplicands have; and a case whose first product reads y isn't
    taken."""
    seq, fix = (("IMAD.HI", (0, 2)), ("IMAD.HI", (2, 3))), ({2: 1, 3: 2},)
    r = S.window_check(seq, fix, timeout_ms=10**9)
    assert r["result"] == "unsat" and r["ops"] == "IMAD.HI[0, 2] IMAD.HI[2, 3] /s2=y,s3=C", r
    for c, t in ((0, "xor"), (3, "xor"), (S.SLICES[0], "add")):
        assert S.window_check(seq, fix, c, t, 10**9, count=None)["result"] == "window-sat", (c, t)
    for ops, count in ((seq, (1, 2)), ((("IMAD.HI", (0, 2)), ("IMAD.HI", (3, 4))), (3, 1))):
        r = S.window_check(ops, fix, timeout_ms=10**9, count=count)
        assert r["result"] == "unknown" and r["stage"] == "count of multiplicand 0: sat", r
    assert S.window_check((("IMAD", (0, 2)), ("IMAD", (2, 3))), ({2: 1},), tname="add")["result"] == "window-sat"
    with pytest.raises(ValueError):
        S.window_check((("IMAD.HI", (0, 1)), ("IMAD.HI", (3, 4))), ({2: 0, 3: 2},))


@pytest.mark.slow
def test_no_one_instruction_program():
    """k = 1: no single menu instruction computes x ^ y ^ K for any K (every sequence unsat)."""
    r = S.search(1)
    assert r["proved"] and r["sequences"] == 27, r["tally"]


@pytest.mark.slow
def test_it_finds_programs_that_exist():
    """The search isn't vacuous: it finds x + y + K (one IMAD) and hi32(x·y) + K (one IMAD.HI on x and y)."""
    assert S.cegis(("IMAD",), "add")["result"] == "FOUND"
    assert S.cegis((("IMAD.HI", (0, 1)),), "mulhi")["result"] == "FOUND"
