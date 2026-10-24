"""padrev's adversarial tests for lane `pad`'s stage TINY CIRCUITS (lane/r16-pad e6ac4a075) — **padded baseline (experimental)**.

Independent of pad's own suite (`test_lifted_tiny.py`): the oracle for the specified class is padrev's reading of LIFTING-SPEC
v0.4 §2/§3.2/§9(a) written from the spec's letter (`_Ref` below), not pad's code; the vectors are padrev's grid (Active
extremes, ⊥, INV, three other `1‖nonzero` words) and a fresh RNG seed.  Each test names the attack of LIFTING-REVIEW §3 it
runs.  A test marked `xfail(strict=True)` documents an OPEN finding (F-pad-NN in LIFTING-REVIEW §5): it flips to a failure
the moment pad fixes it, so the marker must then be removed.

Skips cleanly on a tree without `verity_vllm.program.registry.lifted` (the module lives on lane/r16-pad); needs the in-tree MUFU
tables like `test_serve3.py` for the LServe_v1 cases.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

L = pytest.importorskip("verity_vllm.program.registry.lifted")

from verity.ml.circuits import Program  # noqa: E402
from verity_vllm.program.registry import b1, pad_prims as PP, prims as P, ref_prims as R  # noqa: E402
from verity_vllm.program.kernels import serve3_reference as ref  # noqa: E402
from verity_vllm.program.registry.b1 import weights_type  # noqa: E402
from tests.program import test_lifted_tiny as T  # noqa: E402  (pad's fixture harness: _lserve, EOS_FOR, Y_RAW, ...)
from tests.program.test_serve3 import D_MAX, EOS_FOR, LP, TINY, Y_RAW, _inputs  # noqa: E402
from verity.ml.circuits.types import Array, Value  # noqa: E402

RNG = np.random.default_rng(20260920)


# ---------------------------------------------------------------------------------------------------------
# padrev's reference of the spec's letter (§2, §3.1, §3.2, §9a) — the oracle; no code shared with lifted.py
# ---------------------------------------------------------------------------------------------------------

class _V1AndV2:
    """The lifted relation under test = `Lifted[p]_v1` (the propagation relation, v0.5.2 A3 — FACTS under v0.6 A9.4) and, on the SAME
    vectors, `Lifted[p]_v2` at `enable = Active(1)` must agree with it (v0.6 A9.3: the enabled result is v0.5.2's relation)."""

    def __init__(self, p):
        self.v1, self.v2 = L.lifted_prim_v1(p), L.lifted_prim(p)
        self.id, self.lift_class, self.ret, self.params = self.v2.id, getattr(self.v1, "lift_class", None), self.v1.ret, self.v1.params

    def evaluate(self, *ops):
        r = self.v1.evaluate(*ops)
        assert self.v2.evaluate(*ops, L.enc_active(1, 1)) == r, (self.id, ops)
        assert self.v2.evaluate(*ops, L.enc_active(1, 0)) == L.enc_bot(self.v1.ret.w - 1), (self.id, ops)  # disabled → ⊥ whatever the operands
        return r



class _Ref:
    @staticmethod
    def bot(w):
        return 1 << w

    @staticmethod
    def inv(w):
        return (1 << w) + 1

    @staticmethod
    def tag(w, x):
        return (x >> w) & 1

    @staticmethod
    def payload(w, x):
        return x & ((1 << w) - 1)

    @classmethod
    def valid(cls, w, x):
        return 0 <= x < (1 << (w + 1)) and (cls.tag(w, x) == 0 or cls.payload(w, x) == 0)

    @classmethod
    def strict(cls, f, in_ws, out_w, *xs):
        if any(not cls.valid(w, x) for w, x in zip(in_ws, xs)):
            return cls.inv(out_w)
        if any(x == cls.bot(w) for w, x in zip(in_ws, xs)):
            return cls.bot(out_w)
        return f(*[cls.payload(w, x) for w, x in zip(in_ws, xs)])

    @classmethod
    def lselect(cls, w, c, a, b):
        if not (cls.valid(1, c) and cls.valid(w, a) and cls.valid(w, b)):
            return cls.inv(w)  # §9(a): "LSelect(c, INV, b) = INV even when c selects b"
        if c == cls.bot(1):
            return cls.bot(w)
        return a if c == 1 else b  # the untaken branch is not consulted

    @classmethod
    def lisactive(cls, w, x):
        if not cls.valid(w, x):
            return cls.inv(1)
        return 1 if cls.tag(w, x) == 0 else 0

    @classmethod
    def lgatherrow(cls, w, idx, rows):
        """In range only: OOR is a boundary fact (SPEC [P3]); the test compares OOR against the registered original."""
        if not cls.valid(32, idx) or any(not cls.valid(w, r) for r in rows):
            return cls.inv(w)
        if idx == cls.bot(32):
            return cls.bot(w)
        i = idx if idx < (1 << 31) else idx - (1 << 32)
        if not (0 <= i < len(rows)):
            raise IndexError(i)
        return rows[i]

    @classmethod
    def lsteplive(cls, eos, prev):
        if not cls.valid(32, prev) or any(not cls.valid(32, e) for e in eos):
            return cls.inv(1)
        if prev == cls.bot(32):
            return 0
        return 0 if prev in set(eos) else 1


def _grid(w):
    """Active extremes, ⊥, INV, and three other 1‖nonzero words (the attack A4 vectors)."""
    g = {"act0": 0, "act1": 1, "actmax": (1 << w) - 1, "bot": 1 << w, "inv": (1 << w) + 1, "garb_all": (1 << (w + 1)) - 1}
    if w >= 2:
        g["garb_hi"] = (1 << w) | (1 << (w - 1))
        g["garb_mid"] = (1 << w) | 0b10
    return g


def _rand(w, n):
    return [int(x) for x in RNG.integers(0, 1 << w, n, dtype=np.uint64)]


# ---------------------------------------------------------------------------------------------------------
# A2 / A4 — the specified class against padrev's oracle on the full grid
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("w", [1, 16, 32])
def test_lselect_equals_the_spec_letter_on_the_full_grid(w):
    """A2 (selector branch evaluating ⊥ must not change an Active result) and A4 (`1‖nonzero` never read as ⊥ or Active):
    all 4 × |grid|² vectors, condition patterns 00/01/10/11 included."""
    s = L.l_select(w)
    g = _grid(w)
    for c in range(4):
        for (na, a), (nb, b) in itertools.product(g.items(), g.items()):
            assert s.evaluate(c, a, b) == _Ref.lselect(w, c, a, b), (w, c, na, nb)
    # the Active result is the payload of the taken branch bit for bit, whatever the untaken branch is (⊥ included)
    for a in _rand(w, 16):
        assert s.evaluate(1, a, _Ref.bot(w)) == a and s.evaluate(0, _Ref.bot(w), a) == a


@pytest.mark.parametrize("w", [1, 16, 32])
def test_lisactive_and_lbot_equal_the_spec_letter(w):
    ia = L.l_is_active(w)
    for name, x in _grid(w).items():
        assert ia.evaluate(x) == _Ref.lisactive(w, x), (w, name)
    assert L.l_bot(w).evaluate() == _Ref.bot(w)
    assert ia.evaluate(L.l_bot(w).evaluate()) == 0  # activity as a value: Active(0), never ⊥


def test_lsteplive_equals_the_spec_letter_on_the_grid_including_eos_set_members():
    """A6 (activity is a function of the committed VALUE): every prev pattern × EOS sets that contain the token, do not,
    contain ⊥ (never matches) or contain INV (poisons to INV_1)."""
    for E in (1, 2, 3):
        sl = L.l_step_live(E)
        g = _grid(32)
        toks = [7, 17, 0xFFFFFFFF]
        for name, prev in g.items():
            for eos in itertools.product([7, 17, 0xFFFFFFFF, _Ref.bot(32), _Ref.inv(32)], repeat=E):
                assert sl.evaluate(prev, *eos) == _Ref.lsteplive(eos, prev), (E, name, eos)
        for t in toks:  # an Active token continues iff it is not in the set; the set is read by VALUE
            assert sl.evaluate(t, *([t] * E)) == 0 and sl.evaluate(t, *([(t + 1) & 0xFFFFFFFF] * E)) == 1


def test_lgatherrow_equals_the_spec_letter_and_oor_equals_the_registered_original():
    """Indexing (A2 family): a ⊥ or garbage row that is NOT selected must not change an Active read; a ⊥ selected row is ⊥;
    INV anywhere is INV; a ⊥ index is ⊥; an Active out-of-range index agrees with `GatherBf16x{N}`'s registered extension
    (payload agreement, F-pad-05) and is never INV/⊥."""
    N = 4
    lg = L.l_gather_row(16, N)
    orig = P.gather_bf16(N)
    g16 = _grid(16)
    base = _rand(16, N)
    for pos in range(N):
        for name, word in g16.items():
            rows = list(base)
            rows[pos] = word
            for idx in list(range(N)) + [_Ref.bot(32), _Ref.inv(32), (1 << 32) | 5]:
                got = lg.evaluate(idx, *rows)
                want = _Ref.lgatherrow(16, idx, rows)
                assert got == want, (pos, name, idx)
                if L.valid(32, idx) and idx != _Ref.bot(32) and all(L.valid(16, r) for r in rows):
                    # payload agreement with the original wherever both are defined (Active index, no ⊥/INV row)
                    if all(r < (1 << 16) for r in rows):
                        assert got == orig.evaluate(idx, *rows)
    for oor in (N, N + 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):  # Active out-of-range: the registered original's value, tag 0
        got = lg.evaluate(oor, *base)
        assert got == orig.evaluate(oor, *base) == 0x7FFF and L.tag(16, got) == 0


# ---------------------------------------------------------------------------------------------------------
# A7 / carry into the tag — the strict class on padrev's vectors
# ---------------------------------------------------------------------------------------------------------

STRICT = [p for p in list(P.ALL_SCALAR) + list(PP.ALL_NEW) if p.id not in ("SelectI32_v1", "SelectF32_v1", "SelectBf16_v1")]


@pytest.mark.parametrize("p", STRICT, ids=lambda p: p.id)
def test_strict_class_on_padrev_vectors_never_touches_the_tag_and_inv_beats_bot_in_every_order(p):
    """Attack: 'active numerics changed by the lifting arithmetic' / 'carry-out into the tag bit'.  For 64 fresh random
    vectors and the all-ones / MSB-only / MSB-and-LSB patterns: lifted == original bit for bit and the result is < 2^w_out
    (tag 0).  Then every arrangement of {⊥, INV, garbage} over the operand positions: INV wins whenever present, else ⊥."""
    lp = _V1AndV2(p)
    ws = [w for _, t in p.params for w in t.leaf_widths()]
    wo = p.ret.w
    pats = [lambda w: (1 << w) - 1, lambda w: 1 << (w - 1), lambda w: (1 << (w - 1)) | 1, lambda w: 0]
    vecs = [[f(w) for w in ws] for f in pats] + [[_rand(w, 1)[0] for w in ws] for _ in range(64)]
    for ops in vecs:
        want = p.evaluate(*ops)
        assert 0 <= want < (1 << wo), (p.id, ops, want)  # the frozen evaluator is total on its width (else the tag would flip)
        assert lp.evaluate(*ops) == want == _Ref.strict(p.evaluate, ws, wo, *ops)
    marks = {"bot": lambda w: 1 << w, "inv": lambda w: (1 << w) + 1, "garb": lambda w: (1 << w) | ((1 << w) - 1)}
    n = len(ws)
    if n <= 4:  # every arrangement (≤ 256)
        assigns = [a for a in itertools.product(["act", "bot", "inv", "garb"], repeat=n) if any(x != "act" for x in a)]
    else:  # wide prims (AmpereBF16TcDot16: 33 leaves): every single position × mark, every ordered pair of two marks, and 64 random
        assigns = [tuple("act" if j != i else m for j in range(n)) for i in range(n) for m in marks]
        assigns += [tuple(m1 if j == i else m2 if j == k else "act" for j in range(n))
                    for i, k in itertools.permutations(range(n), 2) if abs(i - k) <= 2 or (i, k) == (0, n - 1)
                    for m1, m2 in itertools.product(marks, repeat=2)]
        assigns += [tuple(str(x) for x in RNG.choice(["act", "bot", "inv", "garb"], size=n)) for _ in range(64)]
    for assign in assigns:
        if all(a == "act" for a in assign):
            continue
        ops = [vecs[4][i] if a == "act" else marks[a](w) for i, (a, w) in enumerate(zip(assign, ws))]
        want = (1 << wo) + 1 if any(a in ("inv", "garb") for a in assign) else 1 << wo
        assert lp.evaluate(*ops) == want == _Ref.strict(p.evaluate, ws, wo, *ops), (p.id, assign)


# ---------------------------------------------------------------------------------------------------------
# the closed specified list is keyed on the indexing FAMILY, not on one name (F-pad-07)
# ---------------------------------------------------------------------------------------------------------


def test_every_registered_gather_family_lifts_to_lgatherrow_not_to_the_strict_wrapper():
    """F-pad-07 (CLOSED lane/r16-pad b68c8797e): was lifted strictly at e6ac4a075 — an unselected ⊥ row poisoned the read."""
    g = R.gather_f32(4)
    lg = _V1AndV2(g)
    assert lg.lift_class == "specified", lg.id
    rows = [1, 2, 3, _Ref.bot(32)]
    assert lg.evaluate(0, *rows) == 1  # the unselected ⊥ row must not poison an Active read (A5)
    assert lg.evaluate(3, *rows) == _Ref.bot(32)
    assert lg.evaluate(9, 1, 2, 3, 4) == g.evaluate(9, 1, 2, 3, 4) == 0x7FC00000  # payload agreement with ITS registered OOR


def test_an_out_of_range_extension_cannot_be_invented_for_an_unregistered_width():
    """F-pad-08 (CLOSED b68c8797e): at e6ac4a075 the kwarg registered X silently for an unregistered width."""
    with pytest.raises((TypeError, ValueError)):
        L.l_gather_row(8, 4, oor_payload=0x12)


def test_a_registered_extension_cannot_be_overridden_and_an_unregistered_width_is_refused():
    with pytest.raises(ValueError):
        L.l_gather_row(16, 4, oor_payload=0x1234)  # disagrees with GatherBf16x's 0x7FFF
    with pytest.raises(TypeError):
        L.l_gather_row(8, 4)  # no registered original gather family at payload width 8 (F-pad-05: nothing invented)
    g32 = L.l_gather_row(32, 4)  # 32 IS a registered family now (GatherF32x): its extension is the original's
    assert g32.evaluate(9, 1, 2, 3, 4) == R.gather_f32(4).evaluate(9, 1, 2, 3, 4) == 0x7FC00000


# ---------------------------------------------------------------------------------------------------------
# LServe_v1 on TINY-PAD — A3 (deletion disguised), A6 (activity from the value), P-1/P-5 (D_MAX-independence)
# ---------------------------------------------------------------------------------------------------------


def _materialize(prog, prompt_words, weight_words):
    t = ref.materialize(prog, [prompt_words, weight_words])
    return t, [t[g] for g in prog.circuit.root_outputs()]


def test_the_active_prefix_does_not_depend_on_the_declared_maximum():
    """P-1/P-5: the same request programmed at D_MAX = 8 and at D_MAX = 5 (both ≥ a_r) has the same Active prefix, the same
    a_r and gate-for-gate equal step bodies for s ≤ 5 — the declared maximum pads, it never changes an active value."""
    eos = (EOS_FOR[3],)
    p8, t8, out8 = T._lserve(eos)
    prompt, w = _inputs()
    p5 = Program(L.lserve(TINY, LP, 5, eos, lag=0))
    t5, out5 = _materialize(p5, L.lift_words(Array(LP, Value(32)), list(prompt)), L.lift_words(weights_type(TINY), list(w)))
    assert out8[:6] == out5 and out5[:4] == Y_RAW[:4] and out5[4:] == [L.enc_bot(32)] * 2
    s8, s5 = T._steps(p8), T._steps(p5)
    for s in range(1, 6):
        c8, c5 = s8[s], s5[s]
        assert c8.hi - c8.lo == c5.hi - c5.lo
        assert [t8[g] for g in range(c8.lo, c8.hi)] == [t5[g] for g in range(c5.lo, c5.hi)], s


def test_an_eos_id_that_is_never_emitted_changes_nothing_and_the_emitted_one_decides(request):
    """A6: activity is a function of the committed token's VALUE and the declared set only.  Adding a never-emitted id to the
    set leaves every gate of every step identical; the set that contains the token emitted at event 3 makes a_r = 4."""
    p_a, t_a, out_a = T._lserve((EOS_FOR[3],))
    p_b, t_b, out_b = T._lserve((EOS_FOR[3], 0x7FFFFFFF))
    assert out_a == out_b
    sa, sb = T._steps(p_a), T._steps(p_b)
    for s in range(1, D_MAX + 1):
        assert [t_a[g] for g in range(sa[s].lo, sa[s].hi)] == [t_b[g] for g in range(sb[s].lo, sb[s].hi)], s
    p_c, t_c, out_c = T._lserve((0x7FFFFFFF,))  # never emitted: every event Active, the cap decides
    assert sum(L.is_active(32, x) for x in out_c) == D_MAX + 1 and [L.payload(32, x) for x in out_c] == Y_RAW
    assert out_a[:4] == out_c[:4] and all(x == L.enc_bot(32) for x in out_a[4:])


def test_a_bot_prompt_token_is_refused_at_the_boundary_and_cannot_shorten_a_request():
    """A3 (post-hoc deletion disguised as 'not required'): a ⊥ inside a prescribed input is refused by name; if the boundary is
    bypassed the strict closure makes the whole transcript ⊥ — there is no way to make a request look like a shorter Active one
    by declaring part of its prompt inactive."""
    prompt, w = _inputs()
    words = L.lift_words(Array(LP, Value(32)), list(prompt))
    words[LP - 2] = L.enc_bot(32)
    with pytest.raises(L.BoundaryFailure, match="prescribed input must be Active"):
        for i, tk in enumerate(words):
            L.check_boundary(32, tk, what=f"prompt[{i}]", prescribed=True, index_range=TINY["V"])
    prog = Program(L.lserve(TINY, LP, 2, (EOS_FOR[3],), lag=0))
    _, out = _materialize(prog, words, L.lift_words(weights_type(TINY), list(w)))
    assert all(x == L.enc_bot(32) for x in out), out  # never a shortened Active transcript
    words_inv = L.lift_words(Array(LP, Value(32)), list(prompt))
    words_inv[0] = L.inv(32)
    _, out_inv = _materialize(prog, words_inv, L.lift_words(weights_type(TINY), list(w)))
    assert all(x == L.inv(32) for x in out_inv), out_inv  # INV reaches every output, never absorbed (A4 end to end)


def test_lifted_program_outputs_are_one_namespace_and_the_step_inputs_are_the_previous_outputs():
    """P-2 (one value namespace): the Program's outputs are D_MAX+1 lifted i32 words — no activity array, no Value<2> output; the
    only Value<2> root gates are the D_MAX `live` guards, each reading the previous step's OUTPUT gate (never an Input)."""
    prog, _, out = T._lserve((EOS_FOR[3],))
    outs = prog.circuit.root_outputs()
    assert len(outs) == D_MAX + 1 and len(set(outs)) == D_MAX + 1
    assert all(prog.circuit.gate(g).prim.ret.w == 33 for g in outs)
    named = T._root_named(prog)
    lives = [c for n, c in named.items() if n.endswith("/live")]
    assert len(lives) == D_MAX and all(prog.circuit.gate(c.lo).prim.ret.w == 2 for c in lives)
    two_bit_roots = [c for c in prog.circuit.root_calls() if all(prog.circuit.gate(g).prim.ret.w == 2 for g in range(c.lo, c.hi))]
    # the 1-bit lifted root values: the D_MAX `live` guards + the Continuation_v1 controller bits (R17-32: cap `emit`, cap `is_eos`) --
    # nothing else; none of them is an OUTPUT (the outputs stay D_MAX+1 lifted i32 words: one value namespace)
    cap = D_MAX + 1
    # + the root enable constant `enable/one` (v0.6 B7 / A9.1: the root is always enabled)
    assert len(two_bit_roots) == D_MAX + 2 * cap + 1
    assert sorted(c.node.name for c in two_bit_roots) == sorted(["enable/one"] + [f"s={s}/live" for s in range(1, D_MAX + 1)] + [f"slot={t}/emit" for t in range(1, cap + 1)]
                                                              + [f"slot={t}/is_eos" for t in range(1, cap + 1)])
    assert not (set(outs) & {g for c in two_bit_roots for g in range(c.lo, c.hi)})
    t = T._lserve((EOS_FOR[3],))[1]
    for s in range(1, D_MAX + 1):  # A6 by value: guard and input token are the spec's functions of the PREVIOUS OUTPUT word
        live, tok_in = named[f"s={s}/live"], named[f"s={s}/tok_in"]
        assert live.hi - live.lo == 1 and tok_in.hi - tok_in.lo == 1
        assert t[live.lo] == _Ref.lsteplive((EOS_FOR[3],), out[s - 1]) == (1 if s - 1 < 3 else 0)
        assert t[tok_in.lo] == _Ref.lselect(32, t[live.lo], out[s - 1], L.enc_bot(32))
        if s <= 3:
            assert t[tok_in.lo] == out[s - 1] and L.is_active(32, out[s - 1])
        elif s == 4:  # the transition: out[3] is the ACTIVE committed EOS, tok_in_4 is the first ⊥ (DP-pad-01 option (a))
            assert L.is_active(32, out[3]) and L.payload(32, out[3]) == EOS_FOR[3] and t[tok_in.lo] == L.enc_bot(32)
        else:
            assert t[tok_in.lo] == out[s - 1] == L.enc_bot(32)


# ---------------------------------------------------------------------------------------------------------
# §7 served-prefix rule — the boundary of the committed array (A3/A4)
# ---------------------------------------------------------------------------------------------------------


def test_assign_inactive_refuses_every_shape_of_disguised_deletion():
    eos, n = (17,), 9
    assert L.assign_inactive("r", [5, 7, 17], eos, n, lag=0) == [5, 7, 17] + [L.enc_bot(32)] * 6
    assert L.assign_inactive("r", [17], eos, n, lag=0) == [17] + [L.enc_bot(32)] * 8  # the prefill's own event is EOS: a_r = 1
    assert L.assign_inactive("r", [5] * 9, eos, n, lag=0) == [5] * 9  # the cap: every event Active
    with pytest.raises(L.ServedPrefixFailure, match="execution past EOS"):
        L.assign_inactive("r", [17, 5], eos, n, lag=0)
    with pytest.raises(L.ServedPrefixFailure, match="declared activity not executed"):
        L.assign_inactive("r", [5, 7], eos, n, lag=0)  # stopped without EOS before the cap
    with pytest.raises(L.ServedPrefixFailure):
        L.assign_inactive("r", [], eos, n, lag=0)
    with pytest.raises(L.ServedPrefixFailure):
        L.assign_inactive("r", [5] * 10, eos, n, lag=0)


def test_a_bot_or_inv_word_inside_the_served_prefix_is_refused_by_name():
    """F-pad-09 (CLOSED b68c8797e): at e6ac4a075 only enc_active's bare ValueError caught it."""
    for bad in (L.enc_bot(32), L.inv(32), (1 << 32) | 17):
        with pytest.raises((L.ServedPrefixFailure, L.BoundaryFailure)):
            L.assign_inactive("r", [5, bad, 17], (17,), 9, lag=0)  # an EOS follows, so the prefix rule alone does not catch it


def test_activity_prefix_reads_only_active_members_of_the_set():
    """A4 at the committed boundary: an INV word whose payload is the EOS id is NOT an EOS (nor Active); a ⊥ is not an EOS."""
    assert L.activity_prefix([5, (1 << 32) | 17, 7], (17,), 9) == 9
    assert L.activity_prefix([5, 1 << 32, 7], (17,), 9) == 9
    assert L.activity_prefix([5, 17, 7], (17,), 9) == 2
    assert L.served_count([5, 17, 1 << 32, 7]) == 2 and L.served_count([(1 << 32) | 17, 5]) == 0


# ---------------------------------------------------------------------------------------------------------
# F-pad-10 — the S-migration hook (SPEC §5 [D4], LIFTING-STATUS P4/P5, M-0070): which value counts a live sampler row?
# ---------------------------------------------------------------------------------------------------------


def test_the_live_count_hook_must_read_the_steps_input_token_not_the_previous_output():
    """SPEC §5 [D4] writes `n_live_t = Σ_r LIsActive(out_r[t−1])`.  At step `a_r` (the step after the committed EOS) the previous
    output IS the EOS — an Active token — so `LIsActive(out[a_r−1]) = 1` although the request has no sampler row at `a_r`
    (vLLM removes a finished request before the next step; `LStepLive(EOS) = Active(0)`; under `GUARD=in` `tok_in_{a_r} = ⊥`).
    The hook over-counts every EOS-stopped request by exactly one engine step, which changes `S_t` for its batch-mates at that
    step.  Measured here on TINY-PAD with the EOS at event 3 (`a_r = 4`): the per-step liveness the Program itself computes
    (`live_s`, equivalently `LIsActive(tok_in_s)`) is the value that agrees with the served length `n_r = a_r`; the §5 formula
    does not.  A test that PASSES = the discrepancy is real (F-pad-10, OPEN); it names the hook that must replace it."""
    prog, t, out = T._lserve((EOS_FOR[3],))
    a_r = L.activity_prefix(out, (EOS_FOR[3],), D_MAX + 1)
    assert a_r == 4 and L.served_count(out) == a_r
    named = T._root_named(prog)
    ia = L.l_is_active(32)
    spec_hook = [1] + [ia.evaluate(out[s - 1]) for s in range(1, D_MAX + 1)]              # SPEC §5 [D4] as written
    live_from_program = [1] + [t[named[f"s={s}/live"].lo] for s in range(1, D_MAX + 1)]   # LStepLive(out[s−1], E) = live_s
    input_hook = [1] + [ia.evaluate(t[named[f"s={s}/tok_in"].lo]) for s in range(1, D_MAX + 1)]  # LIsActive(tok_in_s)
    served = [1 if s < a_r else 0 for s in range(D_MAX + 1)]  # the sampler rows vLLM actually runs: events 0..a_r−1
    assert live_from_program == input_hook == served
    assert spec_hook != served and spec_hook[a_r] == 1 and served[a_r] == 0 and spec_hook[:a_r] == served[:a_r] and spec_hook[a_r + 1:] == served[a_r + 1:]
    assert sum(spec_hook) == sum(served) + 1  # one phantom live row per EOS-stopped request, at its step a_r


# ---------------------------------------------------------------------------------------------------------
# PAD-04 candidate (F-pad-11) — LSplitsFor against padrev's own reading of vLLM's doubling loop; the F-pad-10 hook end to end
# ---------------------------------------------------------------------------------------------------------


def _splits_ref(n: int, sms: int) -> int:
    """stoch M-0093 (corrected): the largest power of two S ≤ 32 with S·n ≤ num_SMs; the loop doubles while the NEXT S still fits."""
    s = 1
    while s * 2 <= 32 and (s * 2) * n <= sms:
        s *= 2
    return s


@pytest.mark.parametrize("sms", [142, 132, 108, 8, 1])
def test_lsplitsfor_equals_the_doubling_loop_on_the_whole_domain_and_is_bot_only_at_zero(sms):
    if not hasattr(L, "l_splits_for"):
        pytest.skip("PAD-04 candidate not on this tree")
    ls = L.l_splits_for(sms)
    assert getattr(ls, "lift_class", None) == "specified" and getattr(ls, "candidate", None) == "PAD-04"
    for n in range(1, 65):
        got = ls.evaluate(n)
        assert L.tag(6, got) == 0 and got == _splits_ref(n, sms), (sms, n, got)
        assert got in (1, 2, 4, 8, 16, 32)
    assert ls.evaluate(0) == L.enc_bot(6)  # the specified rule: no live row, no sampling event
    assert ls.evaluate(L.enc_bot(7)) == L.enc_bot(6) and ls.evaluate(L.inv(7)) == L.inv(6) and ls.evaluate((1 << 7) | 5) == L.inv(6)
    for n in (65, 100, 127):  # outside the split pipeline: refused by name, never an invented S, never INV
        with pytest.raises(ValueError, match="not the split pipeline"):
            ls.evaluate(n)
    for bad in (0, 65):
        with pytest.raises(ValueError):
            L.assert_splits_domain(bad)
    assert L.assert_splits_domain(64) == 64
    if sms == 142:  # stoch's table (M-0067), spot-checked independently
        assert [ls.evaluate(n) for n in (4, 5, 8, 9, 17, 18, 35, 36, 64)] == [32, 16, 16, 8, 8, 4, 4, 2, 2]
    if sms == 132:
        assert [ls.evaluate(n) for n in (16, 17, 33, 34, 66 if False else 64)] == [8, 4, 4, 2, 2]


def test_the_s_operand_fed_from_the_input_token_hook_is_bot_exactly_at_the_phantom_step():
    """F-pad-10 end to end on TINY (one request, EOS at event 3): with the corrected hook `Σ LIsActive(tok_in_s)` the S operand is
    Active(32) for live steps and ⊥ from `a_r` on; the retired hook `Σ LIsActive(out[s−1])` would have produced S = 32 at `a_r`."""
    if not hasattr(L, "l_splits_for"):
        pytest.skip("PAD-04 candidate not on this tree")
    prog, t, out = T._lserve((EOS_FOR[3],))
    named = T._root_named(prog)
    ia, ls = L.l_is_active(32), L.l_splits_for(142)
    for s in range(1, D_MAX + 1):
        n_live = ia.evaluate(t[named[f"s={s}/tok_in"].lo])          # the corrected hook (one request)
        n_phantom = ia.evaluate(out[s - 1])                           # the retired hook
        assert ls.evaluate(n_live) == (32 if s < 4 else L.enc_bot(6))
        if s == 4:
            assert n_phantom == 1 and ls.evaluate(n_phantom) == 32  # the phantom S the retired hook would have committed


# ---------------------------------------------------------------------------------------------------------
# DP-pad-02 — padrev's reference for the guard's LAG (the async-scheduling placeholder step), written BEFORE any ruling
# ---------------------------------------------------------------------------------------------------------


def padrev_live_ref(tokens, eos, cap, lag):
    """live_s for s in 0..cap-1 from the committed events alone: event 0 unconditionally live; step s's INPUT is live iff no EOS
    was committed at an event ≤ s−1−lag (lag 0: the step after the EOS is ⊥; lag 1: the engine feeds the EOS forward once)."""
    eos = set(int(e) for e in eos)
    a_r = cap
    for s in range(1, cap):
        x = tokens[s - 1]
        if _Ref.valid(32, x) and _Ref.tag(32, x) == 0 and _Ref.payload(32, x) in eos:
            a_r = s
            break
    return [1 if s < a_r + lag else 0 for s in range(cap)], a_r


def test_lag_reference_lag0_equals_the_programs_live_values_and_lag1_shifts_by_exactly_one():
    prog, t, out = T._lserve((EOS_FOR[3],))
    named = T._root_named(prog)
    live_prog = [1] + [t[named[f"s={s}/live"].lo] for s in range(1, D_MAX + 1)]
    live0, a_r = padrev_live_ref(out, (EOS_FOR[3],), D_MAX + 1, 0)
    live1, a_r1 = padrev_live_ref(out, (EOS_FOR[3],), D_MAX + 1, 1)
    assert a_r == a_r1 == 4 and live0 == live_prog  # the implemented Program IS lag 0
    assert live1 == [1] * 5 + [0] * (D_MAX - 4) and sum(live1) == sum(live0) + 1  # lag 1: one placeholder step, at a_r exactly
    # the served-prefix rule under a lag: n_r = a_r + lag — exec M-0140's record (12 events, EOS at event 10) is lag 1 and NOTHING else
    assert 11 + 1 == 12 and 11 + 0 != 12


def test_a_lagged_lserve_if_present_matches_the_reference():
    """DP-pad-02 (b) as built by pad (859dc5d22, `lag=1`): the live values must equal `padrev_live_ref(…, 1)` (one placeholder step at
    `a_r`), out[a_r] must be an Active token (the placeholder's sample, = the unlifted y_{a_r}) with out[a_r+1:] = ⊥; and — PAD-05
    sub-question, padrev's acceptance property — NO step whose input token is ⊥ carries an Active(1) live value (the guard reads the
    FED token `tok_in_{s−1}`, not `out[s−2]`, which would report a live placeholder-successor)."""
    import inspect
    if "lag" not in inspect.signature(L.lserve).parameters:
        pytest.skip("no lagged LServe on this tree")
    prompt, w = _inputs()
    prog = Program(L.lserve(TINY, LP, D_MAX, (EOS_FOR[3],), lag=1))
    t = ref.materialize(prog, [L.lift_words(Array(LP, Value(32)), list(prompt)), L.lift_words(weights_type(TINY), list(w))])
    designated = [t[g] for g in prog.circuit.root_outputs()]  # R17-32: the return is the DESIGNATED output o_t = Select(emit_t, y_t, bot)
    named = T._root_named(prog)
    # the committed events (the raw transcript): prefill's token, then each step's sampled token
    out = [t[named["prefill"].returns[0]]] + [t[named[f"s={s}/step"].returns[0]] for s in range(1, D_MAX + 1)]
    assert "s=1/live" not in named and "s=1/tok_in" not in named  # lag 1: step 1 is unguarded (the engine feeds event 0's token unconditionally)
    live_raw = {s: t[named[f"s={s}/live"].lo] for s in range(2, D_MAX + 1)}
    tok_in = {s: t[named[f"s={s}/tok_in"].lo] for s in range(2, D_MAX + 1)}
    live_prog = [1, 1] + [1 if live_raw[s] == 1 else 0 for s in range(2, D_MAX + 1)]
    live1, a_r = padrev_live_ref(out, (EOS_FOR[3],), D_MAX + 1, 1)
    assert a_r == 4 and live_prog == live1 == [1] * 5 + [0] * (D_MAX - 4)
    assert L.is_active(32, out[a_r]) and all(x == L.enc_bot(32) for x in out[a_r + 1:]) and all(L.is_active(32, x) for x in out[:a_r])
    # the designated output serves exactly a_r (the EOS included) and never the placeholder's sample (Continuation_v1 s5 C)
    assert designated == out[:a_r] + [L.enc_bot(32)] * (D_MAX + 1 - a_r) and L.served_count(designated) == a_r == L.served_count(out) - 1
    emit_prog = [t[named[f"slot={tt}/emit"].lo] for tt in range(1, D_MAX + 2)]
    assert emit_prog == [1] * a_r + [0] * (D_MAX + 1 - a_r) and live_prog == [1] * (a_r + 1) + [0] * (D_MAX - a_r)  # (e, r) = (0, 1) exactly at slot a_r + 1
    # the placeholder's sample is the unlifted Program's own y_{a_r} (a value of the execution, not invented): compare with lag 0's step count
    p0, t0, out0 = T._lserve((0x7FFFFFFF,))  # never stops: every event is the unlifted transcript
    assert out[:a_r + 1] == out0[:a_r + 1]
    # PAD-05 acceptance property
    for s in range(2, D_MAX + 1):
        if tok_in[s] == L.enc_bot(32):
            assert live_raw[s] != 1, (s, live_raw[s])  # never Active(1) on a ⊥ input
    # and the served-prefix rule with the lag: n_r = a_r + 1 is the ONLY accepted served length
    served = [L.payload(32, x) for x in out[:a_r + 1]]
    assert L.assign_inactive("r", served, (EOS_FOR[3],), D_MAX + 1, lag=1) == out
    for n in (a_r, a_r + 2):
        with pytest.raises(L.ServedPrefixFailure):
            L.assign_inactive("r", [L.payload(32, x) for x in out0[:n]] if n > a_r else served[:n], (EOS_FOR[3],), D_MAX + 1, lag=1)
    with pytest.raises(L.ServedPrefixFailure):
        L.assign_inactive("r", served, (EOS_FOR[3],), D_MAX + 1, lag=0)  # a lag-1 record checked as lag 0: execution past EOS
