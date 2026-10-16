"""Lane `pad` (R16) — stage TINY of the ⊥-lifted, fully padded baseline (**padded baseline (experimental)**; nothing here is
a panel-row result).  Specification `out/gen/r15/LIFTING-SPEC.md` v0.4; agreed/proposal split `out/gen/r16/LIFTING-STATUS.md`;
candidates PAD-01 (unused `1‖nonzero` words) and PAD-02 (widened VU outputs) in `out/gen/r16/DECISIONS-PENDING.md`.

Every group has positive AND negative cases (BRIEF §7B):

* encoding / decoding of lifted values (§2; A3) and the observable-boundary check (PAD-01 rule 1);
* arithmetic — the strict class generated mechanically from every registered scalar primitive (§3.1): payload agreement bit
  for bit, ⊥ on any ⊥ operand, `INV` on any invalid operand, `INV` beats ⊥;
* selection / indexing — `LSelect`, `LIsActive`, `LBot`, `LGatherRow` (§3.2) incl. `Select(true, 7, ⊥) = 7`, an inactive
  unselected row never poisons a read, an inactive index yields ⊥;
* composites lifted by node replay (§3.3): address stability, payload agreement, measured strictness, a ⊥ table row read
  through `Lifted[Embedding]`;
* state — `LServe_v1` on the TINY-PAD fixture of `test_serve3.py` (§4–§6): prefix property gate for gate against the frozen
  `Serve_v1{STEPS=n}`, every inactive step ⊥ (strict closure), the EOS transition, the cap, the EOS **set**, the §7 served-prefix
  rule reproducing the committed output array from served tokens and its two named failures;
* malformed representations under the PAD-01 candidate: never absorbed, never canonicalised, reach the boundary as `INV`;
* PAD-02: the lifted width rule beside the untouched 32-bit validator; the codec round trip of a lifted Program.

Fixture: TINY-PAD (`test_serve3.TINY`, LP=4, D_MAX=8; raw greedy samples `[15, 23, 26, 17, 26, 17, 26, 17, 26]`, so EOS 15 / 17 /
63 realise 0 / 3 / 8 live decode steps).  Needs the in-tree MUFU tables like `test_serve3.py`.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.ml.circuits import Program, bind  # noqa: E402
from verity.ml.circuits.codec import decode_program, encode_program, program_digest  # noqa: E402
from verity.ml.circuits.defs import REGISTRY, PrimitiveDefinition  # noqa: E402
from verity_vllm.program.registry import b1, lifted as L, pad_prims as PP, prims as P, serve3  # noqa: E402
from verity_vllm.program.kernels import serve3_reference as ref  # noqa: E402
from verity_vllm.program.registry.b1 import weights_type  # noqa: E402
from tests.program.test_serve3 import D_MAX, EOS_FOR, LP, TINY, Y_RAW, _inputs, _v1  # noqa: E402
from verity.ml.circuits.types import Array, Record, Tuple, Value  # noqa: E402

I32, BF16, F32 = Value(32), Value(16), Value(32)
N_EVENTS = D_MAX + 1  # prefill event + D_MAX decode events (the Serve_v2 skeleton; SPEC R3 maps this to Workload_v1's T = cap−1)
BOT32, BOT16, INV32, INV16 = L.enc_bot(32), L.enc_bot(16), L.inv(32), L.inv(16)
RNG = np.random.default_rng(1601)


def _rand(w: int, n: int) -> list[int]:
    return [int(x) for x in RNG.integers(0, 1 << w, n, dtype=np.uint64)]


def _bad(w: int) -> int:
    """A `1‖nonzero` word that is not `INV_w` itself (at w = 1 the only such word IS INV_1 = 0b11)."""
    if w == 1:
        return 0b11
    return (1 << w) | (2 + int(RNG.integers(0, (1 << w) - 2)))


# ---------------------------------------------------------------------------------------------------------
# encoding / decoding (§2, A3) and the boundary check (PAD-01 rule 1)
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("w", [1, 2, 8, 16, 32])
def test_encoding_constants_and_validity(w):
    assert L.enc_bot(w) == 1 << w and L.inv(w) == (1 << w) + 1
    assert L.tag(w, L.enc_bot(w)) == 1 and L.payload(w, L.enc_bot(w)) == 0
    for v in ([0, 1, (1 << w) - 1] + _rand(w, 8)):
        x = L.enc_active(w, v)
        assert x == v and L.tag(w, x) == 0 and L.payload(w, x) == v and L.valid(w, x) and L.is_active(w, x)
    assert L.valid(w, L.enc_bot(w)) and L.is_bot(w, L.enc_bot(w)) and not L.is_active(w, L.enc_bot(w))
    # exactly 2^w + 1 valid patterns among the 2^{w+1} (exhaustive for small w; sampled otherwise)
    if w <= 8:
        assert sum(L.valid(w, x) for x in range(1 << (w + 1))) == (1 << w) + 1
    for x in [L.inv(w)] + [_bad(w) for _ in range(8)]:
        assert not L.valid(w, x) and not L.is_bot(w, x) and not L.is_active(w, x)  # negative: neither Active nor ⊥
    assert not L.valid(w, 1 << (w + 1))  # not a w+1-bit word at all
    with pytest.raises(ValueError):
        L.enc_active(w, 1 << w)


def test_bit_lifts_to_two_bits():
    """`Value<1>` lifts to `Value<2>`: 00 = Active(0), 01 = Active(1), 10 = ⊥, 11 = INVALID."""
    assert L.lift_type(Value(1)) == Value(2)
    assert [L.valid(1, x) for x in range(4)] == [True, True, True, False]
    assert L.enc_bot(1) == 0b10 and L.inv(1) == 0b11


def test_lift_type_is_a_functor_on_leaves():
    t = Record(("a", Array(3, I32)), ("b", Tuple(BF16, Array(2, Value(1)))))
    lt = L.lift_type(t)
    assert lt == Record(("a", Array(3, Value(33))), ("b", Tuple(Value(17), Array(2, Value(2)))))
    assert lt.leaves == t.leaves and lt.leaf_widths() == [w + 1 for w in t.leaf_widths()]
    assert L.unlift_type(lt) == t
    assert L.lift_words(t, [1, 2, 3, 4, 1, 0]) == [1, 2, 3, 4, 1, 0] and L.bot_words(t) == [1 << 32] * 3 + [1 << 16, 2, 2]
    with pytest.raises(TypeError):
        L.unlift_type(Value(1))  # negative: a 1-bit word cannot be a lifted word
    with pytest.raises(ValueError):
        L.lift_words(t, [1, 2])


def test_boundary_check_names_its_rejections():
    assert L.check_boundary(32, 7, what="x") == 7 and L.check_boundary(32, BOT32, what="x") == BOT32
    with pytest.raises(L.BoundaryFailure, match="invalid representation"):
        L.check_boundary(32, INV32, what="committed[3]")
    with pytest.raises(L.BoundaryFailure, match="invalid representation"):
        L.check_boundary(16, _bad(16))
    with pytest.raises(L.BoundaryFailure, match="prescribed input must be Active"):
        L.check_boundary(32, BOT32, prescribed=True, what="prompt[0]")  # negative: ⊥ is not a prescribed value
    assert L.check_boundary(32, 63, prescribed=True, index_range=64) == 63
    with pytest.raises(L.BoundaryFailure, match=r"index 64 out of range"):
        L.check_boundary(32, 64, prescribed=True, index_range=64)  # SPEC [P3]: range is a boundary fact
    with pytest.raises(L.BoundaryFailure, match=r"index -1 out of range"):
        L.check_boundary(32, 0xFFFFFFFF, index_range=64)
    with pytest.raises(L.BoundaryFailure, match="not a 33-bit word"):
        L.check_boundary(32, 1 << 33)


# ---------------------------------------------------------------------------------------------------------
# arithmetic — the strict class (§3.1), mechanically over every registered scalar primitive
# ---------------------------------------------------------------------------------------------------------

SPECIFIED_ORIGINALS = {"SelectI32_v1", "SelectF32_v1", "SelectBf16_v1"}  # their lift is LSelect (§3.2), tested below
STRICT_PRIMS = [p for p in list(P.ALL_SCALAR) + list(PP.ALL_NEW) if p.id not in SPECIFIED_ORIGINALS]


def _widths(p):
    return [w for _, t in p.params for w in t.leaf_widths()]


@pytest.mark.parametrize("p", STRICT_PRIMS, ids=lambda p: p.id)
def test_strict_payload_agreement_bot_and_inv(p):
    """For every registered scalar primitive `F`: `Lifted[F](0‖v…) = 0‖F(v…)` bit for bit; any ⊥ operand → exactly `2^w`;
    any invalid operand → exactly `INV_w`, whichever position, and INV beats ⊥."""
    lp = L.lifted_prim_v1(p)  # the total relation (v0.6 A9.4: the strict rows are FACTS of this relation; the enable is tested below)
    ws = _widths(p)
    assert lp.id == f"Lifted[{p.id}]_v1" and lp.ret == Value(p.ret.w + 1)
    assert [t for _, t in lp.params] == [L.lift_type(t) for _, t in p.params]
    assert getattr(lp, "lift_class") == "strict" and lp.lifted_from is p
    bot_out, inv_out = L.enc_bot(p.ret.w), L.inv(p.ret.w)
    special = {16: [0x0000, 0x3F80, 0x7FFF, 0x7FC0, 0x8000, 0xFF80], 32: [0, 1, 0x3F800000, 0x7F800000, 0x7FC00000, 0x80000000, 0xFFFFFFFF, 7], 1: [0, 1]}
    for trial in range(12):
        ops = [int(special[w][trial % len(special[w])]) if trial < 6 else _rand(w, 1)[0] for w in ws]
        want = p.evaluate(*ops)
        got = lp.evaluate(*ops)
        assert got == want and L.tag(p.ret.w, got) == 0, (p.id, ops)  # payload agreement, Active out
        for i in range(len(ws)):
            bops = list(ops)
            bops[i] = L.enc_bot(ws[i])
            assert lp.evaluate(*bops) == bot_out, (p.id, i)  # ⊥ in any one position → ⊥
            iops = list(ops)
            iops[i] = L.inv(ws[i]) if trial % 2 else _bad(ws[i])
            assert lp.evaluate(*iops) == inv_out, (p.id, i)  # invalid in any one position → INV (not ⊥, not Active)
            if len(ws) > 1:
                both = list(bops)
                both[(i + 1) % len(ws)] = _bad(ws[(i + 1) % len(ws)])
                assert lp.evaluate(*both) == inv_out  # INV beats ⊥ regardless of order
        assert lp.evaluate(*[L.enc_bot(w) for w in ws]) == bot_out  # all ⊥ → ⊥


@pytest.mark.parametrize("p", STRICT_PRIMS, ids=lambda p: p.id)
def test_enabled_lift_is_the_relation_at_enable_1_bot_at_enable_0_whatever_the_operands_inv_otherwise(p):
    """R17-43 / LIFTING-SPEC v0.6 A9.3, B7 — the lift OF RECORD `Lifted[F]_v2(x…, enable)`: `enable = Active(1)` → the v1 total relation
    (payload agreement / ⊥ / INV rows as facts); `enable = Active(0)` → exactly `⊥` WHATEVER the operands — Active, ⊥ or INV words alike
    (the value is produced inside a disabled computation; nothing about its operands is read); `enable ∈ {⊥, INV_1}` → `INV_w` (an enable
    is a Boolean word owned by the parent, A9.2).  The enable is the LAST operand and the only new one."""
    lp, lv1 = L.lifted_prim(p), L.lifted_prim_v1(p)
    ws = _widths(p)
    assert lp.id == f"Lifted[{p.id}]_v2" and lp.ret == Value(p.ret.w + 1) and lp is L.lifted_prim(p)
    assert lp.params[:-1] == lv1.params and lp.params[-1] == (L.ENABLE_PARAM, Value(2)) and L.has_enable(lp) and not L.has_enable(lv1)
    assert getattr(lp, "lift_class") == "strict" and lp.lifted_from is p and lp.lifted_v1 is lv1
    bot_out, inv_out = L.enc_bot(p.ret.w), L.inv(p.ret.w)
    for trial in range(8):
        ops = [_rand(w, 1)[0] for w in ws]
        variants = [ops] + [ops[:i] + [L.enc_bot(ws[i])] + ops[i + 1:] for i in range(len(ws))] + \
                   [ops[:i] + [L.inv(ws[i])] + ops[i + 1:] for i in range(len(ws))] + [[_bad(w) for w in ws]]
        for v in variants:
            assert lp.evaluate(*v, 1) == lv1.evaluate(*v), (p.id, v)  # enabled: the registered relation, including its ⊥/INV facts
            assert lp.evaluate(*v, 0) == bot_out, (p.id, v)  # disabled: ⊥, even on INV operands — no operand is read
            assert lp.evaluate(*v, L.enc_bot(1)) == inv_out and lp.evaluate(*v, 0b11) == inv_out, (p.id, v)  # not an enable word
        assert L.tag(p.ret.w, lp.evaluate(*ops, 1)) == 0


def test_strict_class_is_generated_not_authored_and_originals_untouched():
    """The lifted evaluator calls the frozen one; the original definitions are unchanged and still registered; a lifted
    definition cannot be lifted again; constants lift to the existing `Const<w+1>[0‖bits]` family."""
    a = L.lifted_prim_v1(P.Bf16Add)
    assert a is L.lifted_prim_v1(P.Bf16Add) and REGISTRY[a.id] is a and REGISTRY["Bf16Add_v1"] is P.Bf16Add
    assert P.Bf16Add.params == (("a", BF16), ("b", BF16)) and P.Bf16Add.ret == BF16  # frozen
    with pytest.raises(TypeError, match="already a lifted"):
        L.lifted_prim_v1(a)
    with pytest.raises(TypeError, match="already a lifted"):
        L.lifted_prim(a)
    with pytest.raises(TypeError, match="already a lifted"):
        L.lifted_prim(L.lifted_prim(P.Bf16Add))
    c = L.lifted_prim_v1(b1.const(32, 0xFF800000))
    assert c is b1.const(33, 0xFF800000) and c.id == "Const33[0xff800000]_v1" and c.evaluate() == 0xFF800000 and L.tag(32, c.evaluate()) == 0
    # v0.6 A9.3: a constant produced INSIDE a body owns the body's enable — `Lifted[Const32[k]_v1]_v2(enable)` is Active(k) enabled, ⊥ disabled
    ce = L.lifted_prim(b1.const(32, 0xFF800000))
    assert ce.id == "Lifted[Const32[0xff800000]_v1]_v2" and ce.params == ((L.ENABLE_PARAM, Value(2)),) and REGISTRY[ce.id] is ce
    assert ce.evaluate(1) == 0xFF800000 and ce.evaluate(0) == BOT32 and ce.evaluate(L.enc_bot(1)) == INV32 and ce.evaluate(0b11) == INV32
    # an Input is the Program's boundary, produced by no body: it owns no enable and lifts as before
    from verity.ml.circuits.program import input_prim
    assert L.lifted_prim(input_prim(32)) is input_prim(33) and not L.has_enable(L.lifted_prim(input_prim(32)))
    # a registered input lifts to a registered input: widening keeps its binding time
    assert L.lifted_prim(input_prim(32, "registration")) is input_prim(33, "registration")
    # a specified lifted-v1 member is not lifted AGAIN by the v1 factory; the v2 factory gives it its enabled form (v0.6 B7: a select or a
    # ⊥ constant INSIDE a disabled body is ⊥ -- `Lifted[LBot32_v1]_v2(enable)`), which is itself final
    with pytest.raises(TypeError):
        L.lifted_prim_v1(L.l_bot(32))
    eb = L.lifted_prim(L.l_bot(32))
    assert eb.id == "Lifted[LBot32_v1]_v2" and eb.evaluate(1) == eb.evaluate(0) == BOT32 and eb.evaluate(0b11) == INV32
    with pytest.raises(TypeError, match="already a lifted"):
        L.lifted_prim(eb)


# ---------------------------------------------------------------------------------------------------------
# selection / indexing — the specified class (§3.2)
# ---------------------------------------------------------------------------------------------------------


def test_lselect_relation():
    s = L.l_select(32)
    assert s.id == "LSelect32_v1" and L.lifted_prim_v1(P.SelectI32) is s and L.lifted_prim_v1(P.SelectF32) is s
    assert L.lifted_prim_v1(P.SelectBf16) is L.l_select(16)
    # v0.6 B7: inside a body the select is `Lifted[SelectI32_v1]_v2(c, a, b, enable)` — LSelect32's relation enabled, ⊥ disabled
    se = L.lifted_prim(P.SelectI32)
    assert se.id == "Lifted[SelectI32_v1]_v2" and se.lifted_v1 is s and se.lift_class == "specified" and L.lifted_prim(P.SelectF32) is not se
    assert se.evaluate(1, 7, BOT32, 1) == 7 and se.evaluate(0, BOT32, 7, 1) == 7 and se.evaluate(1, 7, INV32, 1) == INV32
    assert se.evaluate(1, 7, 8, 0) == BOT32 and se.evaluate(0b11, INV32, 8, 0) == BOT32 and se.evaluate(1, 7, 8, 0b10) == INV32
    assert s.evaluate(1, 7, BOT32) == 7  # Select(true, 7, ⊥) = 7 — the untaken branch may be inactive (A5)
    assert s.evaluate(0, BOT32, 7) == 7
    assert s.evaluate(1, BOT32, 7) == BOT32 and s.evaluate(0, 7, BOT32) == BOT32  # the taken branch's ⊥ is the value
    assert s.evaluate(L.enc_bot(1), 7, 8) == BOT32  # ⊥ condition → ⊥ (never a guess)
    for c in (0, 1):  # payload agreement with the frozen select on Active operands
        for a, b in zip(_rand(32, 6), _rand(32, 6)):
            assert s.evaluate(c, a, b) == P.SelectI32.evaluate(c, a, b)
    # negatives (PAD-01 rule 2): INV anywhere → INV, even in the untaken branch, even when the condition is ⊥
    assert s.evaluate(1, 7, INV32) == INV32 and s.evaluate(0, INV32, 7) == INV32
    assert s.evaluate(0b11, 7, 8) == INV32 and s.evaluate(L.enc_bot(1), _bad(32), 8) == INV32
    assert s.evaluate(1, 7, _bad(32)) == INV32  # a random 1‖nonzero is canonicalised to INV, never to ⊥/Active


def test_lis_active_and_lbot():
    ia = L.l_is_active(32)
    assert ia.id == "LIsActive32_v1" and ia.ret == Value(2)
    assert ia.evaluate(7) == 1 and ia.evaluate(0) == 1 and ia.evaluate(0xFFFFFFFF) == 1
    assert ia.evaluate(BOT32) == 0  # Active(0), never ⊥: activity as a value (the S-migration hook)
    assert ia.evaluate(INV32) == 0b11 and ia.evaluate(_bad(32)) == 0b11  # negative: INV_1
    for w in (1, 16, 32):
        bp = L.l_bot(w)
        assert bp.id == f"LBot{w}_v1" and bp.evaluate() == L.enc_bot(w) and bp.params == () and getattr(bp, "family") == "Const"
    assert L.l_bot(32).evaluate() != b1.const(33, 0).evaluate()  # ⊥ is not Active(0)


def test_lgather_row_indexing():
    g = L.l_gather_row(16, 8)
    assert g.id == "LGatherRow16x8_v1" and L.lifted_prim_v1(P.gather_bf16(64)) is L.l_gather_row(16, 64)
    ge = L.lifted_prim(P.gather_bf16(8))  # v0.6 B7: inside a body, `Lifted[GatherBf16x8_v1]_v2(token, column, enable)`
    assert ge.id == "Lifted[GatherBf16x8_v1]_v2" and ge.lifted_v1 is g and ge.lift_class == "specified"
    rows = _rand(16, 8)
    lazy = lambda idx, rs: g.evaluate_lazy(lambda j, _a=[idx] + list(rs): _a[j])  # noqa: E731
    for i in range(8):
        assert g.evaluate(i, *rows) == rows[i] == P.gather_bf16(8).evaluate(i, *rows) == lazy(i, rows)  # payload agreement
    poisoned = list(rows)
    poisoned[5] = BOT16
    assert g.evaluate(3, *poisoned) == rows[3] == lazy(3, poisoned)  # an inactive UNSELECTED row never poisons the read
    assert g.evaluate(5, *poisoned) == BOT16 == lazy(5, poisoned)  # reading the inactive row yields that row's ⊥
    assert g.evaluate(BOT32, *rows) == BOT16 == lazy(BOT32, rows)  # an inactive index yields ⊥, whatever the rows hold
    # out-of-range Active index: payload agreement with the ORIGINAL's registered total extension — `GatherBf16x{V}` (prims.py):
    # "out-of-range or negative token -> 0x7FFF (a total, documented extension; production would fault)" — forced by A6/§3.3,
    # nothing invented here (F-pad-05); never INV ([P3]: range is a boundary fact).  Measured against the original, both evaluators.
    orig = P.gather_bf16(8)
    assert "out-of-range or negative" in orig.doc and "token -> 0x7FFF (a total, documented extension" in orig.doc
    for oor in (8, 9, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):  # past the end, i32 max, i32 min (negative), -1
        assert g.evaluate(oor, *rows) == orig.evaluate(oor, *rows) == 0x7FFF == lazy(oor, rows), oor
    assert L.l_gather_row(16, 8).oor_payload == 0x7FFF == L.registered_oor_extension(orig) and g.lifted_from_id == "GatherBf16x8_v1"
    with pytest.raises(TypeError, match="no registered original gather family"):
        L.l_gather_row(8, 4)  # a payload width with no registered original gets no invented constant (F-pad-05)
    with pytest.raises(TypeError, match="no registered original gather family"):
        L.l_gather_row(8, 4, oor_payload=0x12)  # …and none can be invented through the API either (F-pad-08)
    with pytest.raises(ValueError, match="!= the original GatherBf16x4_v1's 0x7fff"):
        L.l_gather_row(16, 4, oor_payload=0x1234)  # an explicit extension may only confirm the original's
    with pytest.raises(L.BoundaryFailure, match="out of range"):
        L.check_boundary(32, 8, index_range=8)  # …and the range is rejected where validity is, at the boundary
    # negatives: INV anywhere → INV (index, selected row, unselected row — PAD-01 rule 2 literally), eager and lazy alike
    bad = list(rows)
    bad[1] = INV16
    assert g.evaluate(INV32, *rows) == INV16 == lazy(INV32, rows)
    assert g.evaluate(1, *bad) == INV16 == lazy(1, bad) and g.evaluate(6, *bad) == INV16 == lazy(6, bad)
    assert g.evaluate(BOT32, *bad) == INV16 == lazy(BOT32, bad)
    bad[1] = _bad(16)
    assert g.evaluate(6, *bad) == INV16
    # the enabled gather: the relation at enable 1 (in range, ⊥ row, out of range, INV), ⊥ at enable 0 whatever the index/rows
    for i in (3, 5, 8, 0xFFFFFFFF):
        assert ge.evaluate(i, *poisoned, 1) == g.evaluate(i, *poisoned) and ge.evaluate(i, *poisoned, 0) == BOT16
    assert ge.evaluate(1, *bad, 1) == INV16 and ge.evaluate(1, *bad, 0) == BOT16 and ge.evaluate(1, *rows, 0b11) == INV16


def test_every_registered_gather_family_is_specified_with_its_own_extension():
    """F-pad-07: the specified class is keyed on the indexing FAMILY (the gather shape), not on the name `GatherBf16x` — the f32
    reference gather `GatherF32x{V}_v1` (ref_prims, OOR 0x7FC00000) lifts to `LGatherRow32x{V}` with ITS extension, read from
    the original; a strict lift would let an unselected ⊥ row poison an Active read (A5).  Two originals of one `(w, N)` with
    different extensions cannot share the id: refused by name."""
    from verity_vllm.program.registry import ref_prims as R
    g32 = R.gather_f32(4)
    lg = L.lifted_prim_v1(g32)
    assert lg is L.l_gather_row(32, 4) and lg.id == "LGatherRow32x4_v1" and getattr(lg, "lift_class", None) == "specified"
    assert L.lifted_prim(g32).lifted_v1 is lg and L.lifted_prim(g32).id == "Lifted[GatherF32x4_v1]_v2"
    assert lg.oor_payload == 0x7FC00000 == L.registered_oor_extension(g32) and lg.lifted_from_id == "GatherF32x4_v1"
    rows = [1, 2, 3, BOT32]
    assert lg.evaluate(0, *rows) == 1 and lg.evaluate(3, *rows) == BOT32 and lg.evaluate(BOT32, 1, 2, 3, 4) == BOT32
    for oor in (4, 5, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
        assert lg.evaluate(oor, 1, 2, 3, 4) == g32.evaluate(oor, 1, 2, 3, 4) == 0x7FC00000  # payload agreement with ITS original
    assert lg.evaluate(INV32, 1, 2, 3, 4) == L.inv(32) and lg.evaluate(1, 1, INV32, 3, 4) == L.inv(32)
    # the two registered families have different extensions and never collide: (16, N) ↔ GatherBf16x, (32, N) ↔ GatherF32x
    assert L.l_gather_row(16, 4).oor_payload == 0x7FFF != L.l_gather_row(32, 4).oor_payload
    # a same-(w, N) original with a different extension cannot take the id: its own names the extension
    fake = PrimitiveDefinition("GatherFakex4", 1, (("token", Value(32)), ("column", Array(4, Value(32)))), Value(32),
                               lambda tok, *col: col[tok] if 0 <= tok < 4 else 0x12345678)
    assert L.registered_oor_extension(fake) == 0x12345678
    lf = L.l_gather_row_of(fake)
    assert lf.id == "LGatherRow32x4o12345678_v1" and lf.oor_payload == 0x12345678 and lf.lifted_from_id == "GatherFakex4_v1"
    assert L.l_gather_row(32, 4) is lg and lg.oor_payload == 0x7FC00000
    # an original whose out-of-range behaviour is not ONE constant has no registered total extension: nothing is invented
    two = PrimitiveDefinition("GatherTwox4", 1, (("token", Value(32)), ("column", Array(4, Value(32)))), Value(32),
                              lambda tok, *col: col[tok] if 0 <= tok < 4 else (tok & 0xFFFFFFFF))
    with pytest.raises(TypeError, match="no single total out-of-range extension"):
        L.registered_oor_extension(two)
    with pytest.raises(TypeError, match="not a registered gather family"):
        L.registered_oor_extension(P.F32Add)


def test_a_bot_or_inv_word_inside_the_served_prefix_is_refused_by_name():
    """F-pad-09: a served token is a committed Active word; ⊥ / INV / over-width words inside the prefix are BoundaryFailures at
    the served-token boundary (named), even when an EOS follows so that the prefix rule alone would pass."""
    for bad in (BOT32, INV32, (1 << 32) | 17, 1 << 33):
        with pytest.raises(L.BoundaryFailure, match="served token 1"):
            L.assign_inactive("r", [5, bad, 17], (17,), 9, lag=0)
    assert L.assign_inactive("r", [5, 6, 17], (17,), 9, lag=0) == [5, 6, 17] + [BOT32] * 6


def test_lstep_live_continuation_relation():
    sl = L.l_step_live(3)
    assert sl.id == "LStepLive3_v1" and sl.ret == Value(2)
    eos = (128001, 128008, 128009)
    assert sl.evaluate(42, *eos) == 1  # Active, not EOS → step continues
    for e in eos:
        assert sl.evaluate(e, *eos) == 0  # any member of the SET stops (P2), not only the first
    assert sl.evaluate(BOT32, *eos) == 0  # ⊥ previous token → Active(0): once inactive, stays inactive
    assert sl.evaluate(0, *eos) == 1
    assert sl.evaluate(INV32, *eos) == 0b11 and sl.evaluate(42, eos[0], INV32, eos[2]) == 0b11  # negatives
    assert sl.evaluate(_bad(32), *eos) == 0b11
    assert L.l_step_live(1).evaluate(17, 17) == 0 and L.l_step_live(1).evaluate(16, 17) == 1


def _check_specified_list():
    """Only the specified families — the five of v0.4 (+ the labelled PAD-04 candidate `LSplitsForSMS`), the two Continuation_v1 members
    (R17-32, SPEC B6) and `LFixedSelect<E>x<w>` (R17-42 §3, MOE-PAD-SPEC §2; STAGE1-GAPS G2) — are non-strict; everything registered as
    `Lifted[…]` is the strict wrapper.  The Continuation members and LFixedSelect are materialised here so the list is checked with them in it."""
    import re
    L.l_is_eos(1), L.l_emit_next(1), L.l_emit_next(0), L.l_fixed_select(4, 16), L.l_enable()  # + LEnable_v1 (v0.6 B7; moe M-1226 one-liner)
    allowed = re.compile(r"^(LSelect\d+|LIsActive\d+|LBot\d+|LGatherRow\d+x\d+(?:o[0-9a-f]+)?|LStepLive\d+|LSplitsForSMS\d+|LIsEOS\d+|LEmitNext\[H=\d+\]|LFixedSelect\d+x\d+|LEnable)_v1$")
    for fid in L.specified_ids():
        assert allowed.match(fid), fid
    for fid, d in list(REGISTRY.defs.items()):
        if fid.startswith("Lifted[") and hasattr(d, "evaluate"):
            if fid.endswith("_v2"):  # v0.6 B7: the enabled wrapper of a v1 lifted relation — its class is the wrapped relation's
                assert getattr(d, "lift_enabled", False) is True and getattr(d, "lift_class", None) in ("strict", "specified", "constant"), fid
                assert d.lifted_v1.id in L.specified_ids() or d.lifted_v1.id.startswith(("Lifted[", "Const")), fid
            else:
                assert getattr(d, "lift_class", None) == "strict", fid
    assert PP.BitAnd.id not in L.specified_ids() and L.lifted_prim(PP.BitAnd).lift_class == "strict"


def test_specified_list_is_closed():
    """`_check_specified_list` in a fresh interpreter: the registry is process-global, so a lifted relation another test in this process
    resolved would be checked with the ones this module materialises."""
    import subprocess
    code = "from tests.program.test_lifted_tiny import _check_specified_list; _check_specified_list()"
    p = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-4000:]


# ---------------------------------------------------------------------------------------------------------
# composites lifted by node replay (§3.3)
# ---------------------------------------------------------------------------------------------------------

GEMM = bind(b1.GemmV2, K=32, N=64, DOT=P.AmpereBF16TcDot16)
EMB = bind(b1.Embedding, V=64, H=32)
TSEL = bind(b1.TokenSelect, V=64)
STEP = bind(L.StepBody, C=TINY, T=LP + 3, GEMM=b1.GEMM_AMPERE, ATTENTION=b1.ATTENTION_FA2, NORM=b1.norm_bundle(TINY))


def _same_refs(a, b) -> bool:
    return a.to_json() == b.to_json()


@pytest.mark.parametrize("sp", [GEMM, EMB, TSEL, STEP], ids=lambda s: s.base_id)
def test_lift_def_address_stability(sp):
    """global M-0472 R1: same node indices, same operand runs, same offsets, same return refs; only types and spec ids change."""
    lsp = L.lift_def(sp)
    # v0.6 B7: `_v2`, SAME gate count; v0.7 A11 (G1): the class vector `ORD` is the one extra static (0 here: every parameter offered lifted)
    assert lsp.id == f"Lifted[{sp.base_id}]_v2" + sp.id[len(sp.base_id):-1] + ",ORD=0}" and lsp.gates == sp.gates
    assert L.unlifted_id(lsp.id) == sp.id
    ob, lb = sp.body, lsp.body
    assert lb.offsets == ob.offsets and len(lb.nodes) == len(ob.nodes) and _same_refs(lb.ret, ob.ret)
    # the lifted signature is the lifted original's plus the enable, LAST (B7); every node the body owns receives it as its last operand
    assert lb.ret_type == L.lift_type(ob.ret_type) and [t for _, t in lb.params] == [L.lift_type(t) for _, t in ob.params] + [Value(2)]
    assert lb.params[-1][0] == L.ENABLE_PARAM
    en = lb.params.__len__() - 1
    for on, ln in zip(ob.nodes, lb.nodes):
        assert (ln.form, ln.n, ln.n_xs, ln.name) == (on.form, on.n, on.n_xs, on.name)
        assert ln.out == L.lift_type(on.out)
        assert len(ln.args) == len(on.args) + 1 and all(_same_refs(la.refs, oa.refs) and la.type == L.lift_type(oa.type)
                                                      for la, oa in zip(ln.args[:-1], on.args))
        assert ln.args[-1].type == Value(2) and ln.args[-1].refs.to_json() == L.Affine("p", en, 0, 1).to_json()  # the call's own enable, threaded (a_local = 1)
        assert ln.axes == on.axes + ((None,) if on.form == "batch" else ()), ln.name  # a batch shares the enable across its members
        fid = ln.fn.id
        assert fid.startswith("Lifted[") and fid.split("{")[0].endswith("_v2") and L.has_enable(ln.fn), fid
    assert L.lift_def(sp) is lsp  # cached specialization


def _run(sp, inputs: list[list[int]]) -> tuple[Program, dict, list[int]]:
    prog = Program(sp)
    t = ref.materialize(prog, inputs)
    return prog, t, [t[g] for g in prog.circuit.root_outputs()]


def _runl(lsp, inputs: list[list[int]], enable: int = 1) -> tuple[Program, dict, list[int]]:
    """Run a LIFTED composite as a Program: its last parameter is the enable (v0.6 B7), a one-leaf Boolean-word input."""
    return _run(lsp, list(inputs) + [[enable]])


def _body_agrees(p0, t0, p1, t1) -> bool:
    """Gate for gate: the lifted Program (one more root input, the enable, after the data inputs) assigns every BODY gate the word the
    unlifted one does (tag 0 → the same int), and every data input the same word."""
    n_in = p0.input_gates  # the inputs are the leading gates; the lifted Program's one extra input gate is the enable's
    return p1.gates == p0.gates + 1 and all(t1[g] == t0[g] for g in range(n_in)) and all(t1[g + 1] == t0[g] for g in range(n_in, p0.gates))


def test_composite_payload_agreement_and_measured_strictness():
    """Gate-for-gate: the lifted Program has the same gate indices as the unlifted one, and on all-Active inputs every gate's
    payload agrees bit for bit (tag 0).  On all-⊥ data operands with Active weights every return leaf is ⊥ (strictness measured
    on the denoted circuit, not assumed)."""
    x, w = _rand(16, 32), _rand(16, 64 * 32)
    p0, t0, y0 = _run(GEMM, [x, w])
    p1, t1, y1 = _runl(L.lift_def(GEMM), [L.lift_words(Array(32, BF16), x), L.lift_words(Array(64, Array(32, BF16)), w)])
    assert _body_agrees(p0, t0, p1, t1) and y1 == y0
    _, _, yb = _runl(L.lift_def(GEMM), [L.bot_words(Array(32, BF16)), L.lift_words(Array(64, Array(32, BF16)), w)])
    assert yb == [BOT16] * 64  # ⊥ activations × Active weights → ⊥ (weights stay Active, the row is inactive) — a FACT of the relation (A9.4)
    # v0.6 A9.3: the DISABLED call — Active activations × Active weights, enable 0 → every value the call owns is ⊥; nothing it reads matters
    pd, td, yd = _runl(L.lift_def(GEMM), [L.lift_words(Array(32, BF16), x), L.lift_words(Array(64, Array(32, BF16)), w)], enable=0)
    assert yd == [BOT16] * 64 and all(L.is_bot(16, td[g]) or L.is_bot(32, td[g]) for g in range(32 + 64 * 32 + 1, pd.gates))
    logits = _rand(16, 64)
    _, _, s0 = _run(TSEL, [logits])
    _, _, s1 = _runl(L.lift_def(TSEL), [L.lift_words(Array(64, BF16), logits)])
    assert s1 == s0
    _, _, sb = _runl(L.lift_def(TSEL), [L.bot_words(Array(64, BF16))])
    assert sb == [BOT32]  # an inactive step's sampler yields ⊥ before any LSelect (SPEC §5)
    _, _, sp_ = _runl(L.lift_def(TSEL), [[logits[0]] + [BOT16] * 63])
    assert sp_ == [BOT32]  # one ⊥ logit poisons the argmax scan: strict, as the scan's Select conditions become ⊥
    _, _, sd = _runl(L.lift_def(TSEL), [L.lift_words(Array(64, BF16), logits)], enable=0)
    assert sd == [BOT32]  # disabled: ⊥ on Active logits


def test_lifted_embedding_reads_through_an_inactive_table_row():
    """Indexing through a composite: `Lifted[Embedding]` on an Active token reads its row even when another row is ⊥; a ⊥ token
    yields a ⊥ vector; a ⊥ selected row yields ⊥; an INV row anywhere poisons (PAD-01 rule 2)."""
    table = _rand(16, 64 * 32)
    tok = 5
    _, _, h0 = _run(EMB, [[tok], table])
    lt = L.lift_words(Array(64, Array(32, BF16)), table)
    _, _, h1 = _runl(L.lift_def(EMB), [[tok], lt])
    assert h1 == h0
    poisoned = list(lt)
    poisoned[7 * 32:8 * 32] = [BOT16] * 32  # row 7 inactive
    _, _, h2 = _runl(L.lift_def(EMB), [[tok], poisoned])
    assert h2 == h0  # unselected inactive row: no effect
    _, _, h3 = _runl(L.lift_def(EMB), [[7], poisoned])
    assert h3 == [BOT16] * 32  # the selected row is inactive → ⊥ vector
    _, _, h4 = _runl(L.lift_def(EMB), [[BOT32], lt])
    assert h4 == [BOT16] * 32  # inactive token → ⊥ vector
    poisoned[9 * 32] = INV16
    _, _, h5 = _runl(L.lift_def(EMB), [[tok], poisoned])
    assert h5[0] == INV16 and all(v == table[tok * 32 + j] for j, v in enumerate(h5[1:], 1))  # only column 0 sees the INV word
    _, _, h6 = _runl(L.lift_def(EMB), [[tok], lt], enable=0)
    assert h6 == [BOT16] * 32  # v0.6 A9.3: the disabled read is ⊥ — the Active table (an outside input) is untouched and unread


# ---------------------------------------------------------------------------------------------------------
# state — LServe_v1 on the TINY-PAD fixture (§4–§6)
# ---------------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _lserve(eos: tuple[int, ...]):
    prompt, w = _inputs()
    prog = Program(L.lserve(TINY, LP, D_MAX, eos, lag=0))
    for i, tk in enumerate(prompt):  # PAD-01 rule 1 at the prescribed-input boundary
        L.check_boundary(32, tk, what=f"prompt[{i}]", prescribed=True, index_range=TINY["V"])
    t = ref.materialize(prog, [L.lift_words(Array(LP, I32), list(prompt)), L.lift_words(weights_type(TINY), list(w))])
    return prog, t, [t[g] for g in prog.circuit.root_outputs()]


def _root_named(prog):
    return {c.node.name: c for c in prog.circuit.root_calls() if c.node.name}


def _steps(prog):
    return {int(c.node.name.split("=")[1].split("/")[0]): c for c in prog.circuit.root_calls()
            if c.node.name and c.node.name.endswith("/step")}


def _body_named(prog):
    """`t=…` body nodes of the lifted Program: the prefill's and each step's children."""
    m = {}
    for c in prog.circuit.root_calls():
        nm = c.node.name
        if nm == "prefill" or (nm and nm.endswith("/step")):
            for k in c.children():
                if k.node.name:
                    m[k.node.name] = k
    return m


def _tag_of(prog, t, g: int) -> int:
    """The tag of a LIFTED gate's word; an ordinary gate (G1: an input, an unwidened constant) has no tag -- 0 by definition (never ⊥)."""
    prim = prog.circuit.gate(g).prim
    if not L.is_lifted_prim(prim):
        return 0
    return t[g] >> (prim.ret.w - 1)


@pytest.mark.parametrize("n", [0, 3, D_MAX])
def test_lserve_prefix_property_gate_for_gate(n):
    """SPEC §6: the Active prefix of the lifted Program equals, payload for payload and gate for gate under the static name
    map, the frozen `Serve_v1{TINY, LP, STEPS=n}` — every gate Active (tag 0), no live read ever touches a ⊥ row."""
    p1, t1, y1 = _v1(n)
    p3, t3, out = _lserve((EOS_FOR[n],))
    assert sum(L.is_active(32, x) for x in out) == n + 1
    m1, m3 = {c.node.name: c for c in p1.circuit.root_calls() if c.node.name}, _body_named(p3)
    covered = 0
    for name, c1 in m1.items():
        c3 = m3[name]
        assert c3.hi - c3.lo == c1.hi - c1.lo, name
        for g3, g1 in zip(range(c3.lo, c3.hi), range(c1.lo, c1.hi)):
            assert t3[g3] == t1[g1], (name, g3)  # payload agreement AND tag 0 (an Active word is its payload)
        covered += c1.hi - c1.lo
    assert covered == p1.gates - p1.input_gates
    assert [L.payload(32, x) for x in out[:n + 1]] == y1


def fid_is_strict(prim) -> bool:
    return getattr(prim, "lift_class", None) == "strict"


def _is_const(prim) -> bool:
    return getattr(prim, "family", None) == "Const"


@pytest.mark.parametrize("ell", [0, 3])
def test_lserve_bot_steps_are_bot_at_every_gate_they_own_by_enablement(ell):
    """R17-43 / LIFTING-SPEC v0.6 A9.3 + A13.1 (replaces the propagation reading of DP-pad-01 (a)): with the committed EOS at `out[ell]`,
    `a_r = ell + 1`.  For every step `s ≥ a_r` — **including `a_r` itself** — `live_s = Active(0)` is the step's ENABLE: every gate the
    step owns is ⊥ — the sampled `y_s`, every new K/V row leaf, every intermediate, **every constant produced inside the step and every
    weights-only subcomputation** (v0.5.2 left those Active; under enablement nothing a disabled call produces is Active).  Each such
    gate's LAST operand is the step's `live` word (0).  For `s < a_r` every gate is Active and reads `live = 1`.  `tok_in_s = ⊥` still
    (CONTINUATION-SPEC `u`, A13.2) but is no longer what makes the step inactive.  No `1‖nonzero` word exists anywhere."""
    p3, t3, out = _lserve((EOS_FOR[ell],))
    a_r = ell + 1
    assert out[:a_r] == Y_RAW[:a_r] and all(x == BOT32 for x in out[a_r:])
    assert all(L.valid(p3.circuit.gate(g).prim.ret.w - 1, v) for g, v in t3.items() if L.is_lifted_prim(p3.circuit.gate(g).prim))  # G1: ordinary gates have no tag
    steps, root = _steps(p3), _root_named(p3)
    for s, c in sorted(steps.items()):
        r = c.returns
        y, rows = t3[r[0]], [t3[g] for g in r[1:]]
        live_g = root[f"s={s}/live"].returns[0]
        live = t3[live_g]
        tok_in = t3[root[f"s={s}/tok_in"].returns[0]]
        enables = {t3[p3.circuit.gate(g).operands[-1]] for g in range(c.lo, c.hi) if L.has_enable(p3.circuit.gate(g).prim)}
        n_enabled_gates = sum(1 for g in range(c.lo, c.hi) if L.has_enable(p3.circuit.gate(g).prim))
        assert n_enabled_gates == c.hi - c.lo, s  # every gate the step owns takes the enable (no operand-less constant survives inside)
        if s < a_r:  # active step: the input token is the previous committed token, every gate Active, every enable 1
            assert live == 1 and tok_in == out[s - 1] and L.is_active(32, y) and all(L.is_active(16, x) for x in rows), s
            assert all(_tag_of(p3, t3, g) == 0 for g in range(c.lo, c.hi)), s  # no ⊥ read on a live path
            assert enables == {1}, s
        else:  # s >= a_r: DISABLED — ⊥ at every gate the step owns, constants and weights-only cones included; a_r is NOT half-Active
            assert live == 0 and tok_in == BOT32 and y == BOT32 and all(x == BOT16 for x in rows), s
            assert enables == {0}, s
            not_bot = [(g, p3.circuit.gate(g).prim.id) for g in range(c.lo, c.hi) if _tag_of(p3, t3, g) != 1]
            assert not_bot == [], (s, not_bot[:5])
            assert any(_is_const(getattr(p3.circuit.gate(g).prim, "lifted_v1", None)) for g in range(c.lo, c.hi)), s  # a constant IS inside
    # the exact-count rule the DENSE B1 test T-3(ii) is defined against: every returned leaf of every step s >= a_r is enc(⊥)
    leaves_per_step = 1 + 2 * TINY["NL"] * TINY["KVH"] * TINY["D"]
    n_bot_leaves = sum(1 for s, c in steps.items() for g in c.returns if _tag_of(p3, t3, g))
    assert n_bot_leaves == (D_MAX - a_r + 1) * leaves_per_step
    assert sum(1 for x in out if L.is_bot(32, x)) == N_EVENTS - a_r


def test_lserve_transition_and_prefix_theorem():
    """`out[ell] = EOS` is an Active committed token (stop evidence); `LStepLive(out[ell]) = Active(0)` makes `tok_in[a_r] = ⊥`
    and the whole step `a_r` ⊥ (its sampler yields ⊥ on ⊥ logits — nothing Active is discarded); thereafter `LStepLive(⊥) =
    Active(0)` and `LSelect(Active(0), ⊥, ⊥) = ⊥` — activity is a prefix (no Active event follows a ⊥ one)."""
    p3, t3, out = _lserve((17,))
    root = _root_named(p3)
    assert out[3] == 17 and L.is_active(32, out[3])
    assert [t3[root[f"s={s}/live"].returns[0]] for s in range(1, D_MAX + 1)] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert [t3[root[f"s={s}/tok_in"].returns[0]] for s in range(1, D_MAX + 1)] == out[:3] + [BOT32] * 5
    assert t3[_steps(p3)[4].returns[0]] == BOT32 == out[4]  # the sampled y of step a_r IS ⊥ (no output guard needed)
    seen_bot = False
    for x in out:
        if L.is_bot(32, x):
            seen_bot = True
        else:
            assert not seen_bot and L.is_active(32, x)
    assert L.activity_prefix(out, (17,), N_EVENTS) == 4 == L.served_count(out)


def test_live_count_hook_reads_the_steps_input_activity_not_the_previous_output():
    """F-pad-10 (padrev M-0102): the S-migration operand `n_live_t = Σ_r live_{r,t}` must count the rows the sampler actually runs.
    On TINY with EOS at event 3 (`a_r = 4`): the Program's `live_s` (= `LIsActive(tok_in_s)` under GUARD=in) is `[1,1,1,0,0,0,0,0]`
    — the served rows; the retired hook `LIsActive(out[s−1])` over-counts by one at `a_r` (the committed EOS is Active)."""
    p3, t3, out = _lserve((17,))
    root = _root_named(p3)
    live = [t3[root[f"s={s}/live"].returns[0]] for s in range(1, D_MAX + 1)]
    tok_in = [t3[root[f"s={s}/tok_in"].returns[0]] for s in range(1, D_MAX + 1)]
    assert live == [L.l_is_active(32).evaluate(t) for t in tok_in] == [1, 1, 1, 0, 0, 0, 0, 0]
    wrong = [L.l_is_active(32).evaluate(out[s - 1]) for s in range(1, D_MAX + 1)]
    assert wrong == [1, 1, 1, 1, 0, 0, 0, 0] and sum(wrong) == sum(live) + 1  # the phantom row at a_r
    # the served rows of this request: prefill + 3 decode events; the hook agrees with served_count − 1 decode rows
    assert sum(live) == L.served_count(out) - 1


def test_lsplits_for_candidate_pad04():
    """CANDIDATE PAD-04 / F-pad-11: `LSplitsForSMS<SMS>_v1` — payload agreement with stoch's `SplitsFor_v1` (the verbatim vLLM
    doubling loop) for every `1 ≤ n ≤ 64` at 142 / 132 / 108 SMs; `Active(0)` → ⊥ (no live row, no event); ⊥ → ⊥; INV → INV_6;
    `n > 64` refused by name (structural bound, asserted at build); the table flips stoch named (17, 34, 35, 67.. are beyond 64)."""
    from verity_vllm.program.registry import topp_geometry
    for sms in (142, 132, 108):
        lp, up = L.l_splits_for(sms), L.splits_for_prim(sms)
        assert lp.id == f"LSplitsForSMS{sms}_v1" and up.id == f"SplitsForSMS{sms}_v1" and lp.lifted_from is up
        assert lp.lift_class == "specified" and lp.candidate == "PAD-04" and lp.id in L.specified_ids()
        assert lp.params == (("n_live", Value(L.N_LIVE_BITS + 1)),) and lp.ret == Value(L.SPLITS_BITS + 1)
        for n in range(1, 65):
            s = topp_geometry.splits_for(n, sms)
            assert lp.evaluate(n) == up.evaluate(n) == s and 1 <= s <= 32 and s & (s - 1) == 0  # Active(n) → 0‖S, bit for bit
            assert s == max(x for x in (1, 2, 4, 8, 16, 32) if x * n <= sms or x == 1)
        assert lp.evaluate(0) == L.enc_bot(L.SPLITS_BITS)  # the specified rule
        assert lp.evaluate(L.enc_bot(L.N_LIVE_BITS)) == L.enc_bot(L.SPLITS_BITS)
        assert lp.evaluate(L.inv(L.N_LIVE_BITS)) == L.inv(L.SPLITS_BITS) and lp.evaluate((1 << L.N_LIVE_BITS) | 5) == L.inv(L.SPLITS_BITS)
        with pytest.raises(ValueError, match="not the split pipeline"):
            lp.evaluate(65)
        with pytest.raises(ValueError, match="outside the domain"):
            up.evaluate(0)
        with pytest.raises(ValueError, match="outside the domain"):
            up.evaluate(65)
    assert [L.l_splits_for(142).evaluate(n) for n in (1, 4, 5, 8, 9, 17, 18, 35, 36, 64)] == [32, 32, 16, 16, 8, 8, 4, 4, 2, 2]
    assert [L.l_splits_for(132).evaluate(n) for n in (1, 4, 5, 8, 9, 16, 17, 33, 34, 64)] == [32, 32, 16, 16, 8, 8, 4, 4, 2, 2]
    assert L.l_splits_for(142).evaluate(17) == 8 != L.l_splits_for(132).evaluate(17) == 4  # the two profiles disagree at 17
    assert L.assert_splits_domain(64) == 64 and L.assert_splits_domain(1) == 1
    for bad in (0, 65, 128):
        with pytest.raises(ValueError, match="at most 64 live rows"):
            L.assert_splits_domain(bad)
    with pytest.raises(ValueError, match="not a target profile"):
        L.l_splits_for(0)
    # codec: the candidate's ids resolve lazily like the other five
    from verity.ml.circuits.codec import _lookup_primitive
    assert _lookup_primitive(REGISTRY, "LSplitsForSMS142_v1") is L.l_splits_for(142)
    assert _lookup_primitive(REGISTRY, "SplitsForSMS142_v1") is L.splits_for_prim(142)


def test_lserve_guard_choice_is_labelled_and_the_other_option_is_refused():
    """The guard placement is a static of the Program (in its spec id, hence its digest): only `GUARD="in"` exists; the spec's
    letter (`"out"`) is refused by name, never silently built."""
    sp = L.lserve(TINY, LP, D_MAX, (17,), lag=0)
    assert "GUARD=in" in sp.id.replace("'", "").replace('"', "") or sp.bindings["GUARD"] == "in"
    assert L.GUARDS_IMPLEMENTED == ("in",)
    with pytest.raises(ValueError, match="GUARD='out' is not implemented"):
        Program(L.lserve(TINY, LP, D_MAX, (17,), lag=0, guard="out"))
    with pytest.raises(ValueError, match="not implemented"):
        Program(L.lserve(TINY, LP, D_MAX, (17,), lag=0, guard="both"))


def test_lserve_length_cap_all_active():
    p3, t3, out = _lserve((63,))
    assert out == Y_RAW and all(L.is_active(32, x) for x in out)
    assert L.activity_prefix(out, (63,), N_EVENTS) == N_EVENTS == L.served_count(out)
    # the only tag-1 word in the whole transcript is the named ⊥ constant itself
    bots = [g for g, v in t3.items() if _tag_of(p3, t3, g)]
    assert len(bots) == 1 and p3.circuit.gate(bots[0]).prim.id == "LBot32_v1"


def test_lserve_eos_set_semantics():
    """The declared EOS SET (P2): any member stops; the set's order is irrelevant; a set is a different Program (digest)."""
    _, _, a = _lserve((17, 63))
    _, _, b = _lserve((63, 17))
    assert a == b and L.activity_prefix(a, (17, 63), N_EVENTS) == 4
    _, _, c = _lserve((63, 15))
    assert c == [15] + [BOT32] * D_MAX  # stopped on the second member
    assert L.lserve(TINY, LP, D_MAX, (17, 63), lag=0).id != L.lserve(TINY, LP, D_MAX, (17,), lag=0).id
    with pytest.raises(ValueError, match="non-empty"):
        Program(L.lserve(TINY, LP, D_MAX, (), lag=0))  # negative: an empty EOS set is not a declaration
    with pytest.raises(ValueError, match="MAX_POS"):
        Program(L.lserve(TINY, LP, TINY["MAX_POS"] - LP + 1, (17,), lag=0))


def test_lserve_structure_and_digests():
    """The fixed structure: D_MAX steps present regardless of EOS; gates = Serve_v1's + 2 per step (LStepLive, LSelect) + |E|
    EOS constants + 1 ⊥ constant; digests distinct from Serve_v1/v2/v3 and across EOS set / D_MAX; frozen definitions untouched."""
    sp1 = bind(b1.Serve, C=TINY, LP=LP, STEPS=D_MAX)
    for eos in ((15,), (17,), (63,), (17, 63)):
        sp = L.lserve(TINY, LP, D_MAX, eos, lag=0)
        # the guard's gates + the Continuation_v1 controller's (R17-32: 3 per slot over cap = D_MAX + 1 slots) — PAD-03 (address stability across them)
        assert sp.gates == sp1.gates + 2 * D_MAX + len(eos) + 2 + 3 * (D_MAX + 1) == sp1.gates + L.continuation_gate_count(D_MAX + 1, len(eos), 0)  # +2: bot32, enable/one
        prog = Program(sp)
        assert len(_steps(prog)) == D_MAX
        names = [c.node.name for c in prog.circuit.root_calls() if c.node.name]
        # root body = prefill, |E| eos consts, bot32, slot 1's controller (emit const, is_eos, out, next emit), then per step exactly
        # (live, tok_in, step) + the slot's controller (is_eos, out, next emit while a slot follows): no live array, no other node
        from tests.program.test_lifted_dense_b1 import _lserve_names
        assert names == _lserve_names(D_MAX, len(eos))
    d = lambda s: program_digest(Program(s))  # noqa: E731
    l17 = d(L.lserve(TINY, LP, D_MAX, (17,), lag=0))
    assert l17 != d(sp1) and l17 != d(b1.serve_v2(TINY, LP, D_MAX)) and l17 != d(serve3.serve_v3({**TINY, "EOS_ID": 17}, LP, D_MAX))
    assert l17 != d(L.lserve(TINY, LP, D_MAX, (63,), lag=0)) and l17 != d(L.lserve(TINY, LP, D_MAX - 1, (17,), lag=0))
    assert serve3.serve_v3({**TINY, "EOS_ID": 7}, LP, D_MAX).gates == 56936 and sp1.gates == 56361  # frozen budgets (spec §14.1)
    assert "EOS_ID" not in TINY


def test_served_prefix_rule_reproduces_the_committed_array_and_names_failures():
    """SPEC §7 (P1/P5): from the SERVED tokens alone, the pre-challenge rule assigns `0‖tok` to the served prefix and ⊥ after,
    admissible iff `n_r = a_r`; it reproduces the Program's committed `out` bit for bit on the three fixtures.  Negatives: a
    record that serves past an EOS, one that stops without an EOS short of the cap, and an empty/over-long record."""
    for n in (0, 3, D_MAX):
        eos = (EOS_FOR[n],)
        _, _, out = _lserve(eos)
        served = [L.payload(32, x) for x in out if L.is_active(32, x)]
        assert L.assign_inactive("r0", served, eos, N_EVENTS, lag=0) == out
    with pytest.raises(L.ServedPrefixFailure, match="execution past EOS"):
        L.assign_inactive("r1", Y_RAW[:6], (17,), N_EVENTS, lag=0)  # served 6 events but event 3 was the EOS
    with pytest.raises(L.ServedPrefixFailure, match="declared activity not executed"):
        L.assign_inactive("r2", Y_RAW[:3], (17,), N_EVENTS, lag=0)  # stopped before the EOS and below the cap (F-pm-24's shape)
    with pytest.raises(L.ServedPrefixFailure, match="declared activity not executed"):
        L.assign_inactive("r3", Y_RAW[:5], (63,), N_EVENTS, lag=0)  # no EOS ever: activity is the whole cap
    with pytest.raises(L.ServedPrefixFailure):
        L.assign_inactive("r4", [], (17,), N_EVENTS, lag=0)
    with pytest.raises(L.ServedPrefixFailure):
        L.assign_inactive("r5", Y_RAW + [1], (63,), N_EVENTS, lag=0)
    # the rule never consults a flag: the same served tokens under a different declared set give a different verdict
    assert L.assign_inactive("r6", Y_RAW[:4], (17,), N_EVENTS, lag=0)[4:] == [BOT32] * 5
    with pytest.raises(L.ServedPrefixFailure, match="declared activity not executed"):
        L.assign_inactive("r7", Y_RAW[:4], (63,), N_EVENTS, lag=0)


# ---------------------------------------------------------------------------------------------------------
# malformed representations under the PAD-01 candidate, end to end
# ---------------------------------------------------------------------------------------------------------


def test_malformed_word_reaches_the_boundary_as_inv_never_as_a_canonical_value():
    """A `1‖nonzero` prompt word is rejected at the prescribed boundary; if the check were bypassed, honest evaluation carries
    it to every observable output as `INV` — it never turns into ⊥ or an Active value anywhere in its cone."""
    prompt, w = _inputs()
    bad = list(prompt)
    bad[0] = L.enc_bot(32) | 5
    with pytest.raises(L.BoundaryFailure, match="invalid representation"):
        L.check_boundary(32, bad[0], what="prompt[0]", prescribed=True)
    prog = Program(L.lserve(TINY, LP, D_MAX, (17,), lag=0))
    t = ref.materialize(prog, [bad, L.lift_words(weights_type(TINY), list(w))])
    out = [t[g] for g in prog.circuit.root_outputs()]
    assert out == [INV32] * N_EVENTS  # INV at every event: the pattern is neither absorbed by LSelect/LStepLive nor by ⊥
    # and a ⊥ prompt word (also not a prescribed value) is rejected too, while the Program would treat it as inactivity
    with pytest.raises(L.BoundaryFailure, match="prescribed input must be Active"):
        L.check_boundary(32, BOT32, what="prompt[0]", prescribed=True)
    # positive counterpart: with the (valid) fixture prompt no INV exists anywhere in the transcript
    p3, t3, _ = _lserve((17,))
    assert not any(v == L.inv(p3.circuit.gate(g).prim.ret.w - 1) for g, v in t3.items())


# ---------------------------------------------------------------------------------------------------------
# PAD-02 — the lifted width rule beside the untouched validator; the codec round trip
# ---------------------------------------------------------------------------------------------------------


def test_lifted_width_summary_is_the_true_width_rule_r17_6b():
    """R17-6(b) (SPEC v0.5 A4): the width of a lifted set is Σ (w+1), admissibility is the unlifted rule at the lifted parameter; the R16
    PAD-02 candidate (payload rule + n_max) is WITHDRAWN and its API gone."""
    assert not hasattr(L, "validate_width_lifted") and not hasattr(L, "N_MAX")
    r = L.lifted_width_summary([32])
    assert r["ok"] and r["w_out_lifted"] == 33 and r["w_out_payload"] == 32 and r["n_out"] == 1 and r["grown_by_bits"] == 1
    assert not r["active_reference_rule_on_lifted_width_ok"]  # the 32-bit rule on the true width fails BY DESIGN — reported, never the verdict
    assert r["kind"] == L.USE_KIND_LIFTED == "vu-partition-lifted" and r["rules_version"] == L.LIFTED_RULES_VERSION == "lifted-v1"
    # the constant's status is a STATEMENT, never a bare number: accepted by revb (M-0042 B1) on the first real-row use document, the coord ruling
    # on the value itself still pending -- both named in the string (was "PROPOSED" before 18:00Z 2026-09-20)
    assert r["w_max_lifted"] == L.W_MAX_LIFTED == 64 and r["w_max_active_reference"] == L.W_MAX == 32
    assert r["w_max_lifted_status"] == L.W_MAX_LIFTED_STATUS and "accepted-revb-M-0042" in r["w_max_lifted_status"] and "pending" in r["w_max_lifted_status"]
    assert L.lifted_width_summary([32, 1])["ok"] and L.lifted_width_summary([32, 1])["w_out_lifted"] == 35  # no payload rule: 35 <= 64
    assert L.lifted_width_summary([8, 8, 8])["ok"] and L.lifted_width_summary([8, 8, 8])["n_out"] == 3          # no n_max
    r2 = L.lifted_width_summary([32, 32])
    assert not r2["ok"] and r2["w_out_lifted"] == 66 and r2["violations"] == ["w_out_lifted 66 > w_max_lifted 64"]
    assert L.lifted_width_summary([31, 31])["ok"] and L.lifted_width_summary([31, 31])["w_out_lifted"] == 64
    assert L.lifted_width_summary([32, 32], w_max_lifted=66)["ok"]  # the parameter is a parameter
    with pytest.raises(ValueError):
        L.lifted_width_summary([0])
    from verity.ml.circuits import partition
    assert getattr(partition, "validate_width", None) is not None  # the ONE validator: lifted_width_admissibility calls it at W_MAX_LIFTED


def test_codec_round_trip_of_a_lifted_program():
    prog = Program(L.lserve(TINY, LP, D_MAX, (17,), lag=0))
    desc = encode_program(prog)
    d0 = program_digest(desc)
    dec = decode_program(desc, REGISTRY)
    assert dec.gates == prog.gates and program_digest(dec) == d0
    fids = set(desc["definitions"])
    fams = {f.split("{", 1)[0] for f in fids}  # G1 (v0.7 A11): a lifted primitive reading an ordinary operand carries `{ORD=<mask>}` on its id
    assert {"LSelect32_v1", "LBot32_v1", "LStepLive1_v1", "Lifted[GatherBf16x64_v1]_v2", "Lifted[Bf16Add_v1]_v2", "Const2[0x1]_v1"} <= fams  # v0.6 B7: `_v2` inside bodies
    assert "LStepLive1_v1{ORD=2}" in fids and "Lifted[GatherBf16x64_v1]_v2{ORD=2}" in fids  # the EOS set and the embedding table are ordinary
    assert all(L.unlifted_id(f) == f.split("{", 1)[0] for f in fids if "{ORD=" in f and f.startswith("L") and not f.startswith("Lifted["))
    assert not any(f in ("Serve_v3", "PadGuard_v1", "BitAnd_v1") for f in fids)  # no live-bit chain anywhere (SPEC §4 (2))


# ---------------------------------------------------------------------------------------------------------
# R17-42 §7 Stage 1 (revb PADDED-ACCEPTANCE S-r17b-S1): the named items whose test was implicit until now
# ---------------------------------------------------------------------------------------------------------


def test_stage1_state_step_is_select_update_new_old_and_preserves_old_bit_identically():
    """Stage-1 item 4 'state preservation' as a tiny circuit: the state transition is `Select(update, new, old)` (LSelect, non-strict) --
    with `update = Active(0)` the state is the OLD word bit for bit even when the pending `new` value is bot (nothing happened, so the
    state is not bot); with `update = Active(1)` it is the new word; an inactive `update` (the request itself inactive) is bot, which
    a step of ANOTHER request never sees (the cross-request fixture is test_lifted_mixed.py)."""
    for w in (1, 16, 32):
        st = L.l_select(w)
        for old in _rand(w, 8):
            assert st.evaluate(0, L.enc_bot(w), old) == old  # no update: old state preserved, even beside a bot candidate
            for new in _rand(w, 3):
                assert st.evaluate(0, new, old) == old and st.evaluate(1, new, old) == new
                assert st.evaluate(1, new, L.enc_bot(w)) == new  # first write into an inactive slot: the new word
            assert st.evaluate(L.enc_bot(1), _rand(w, 1)[0], old) == L.enc_bot(w)  # the request inactive: its state is bot, not 'old'
        assert st.evaluate(0, L.enc_bot(w), L.enc_bot(w)) == L.enc_bot(w)  # never-written state of an inactive request stays bot


def test_stage1_flip_one_payload_bit_of_bot_is_named_by_every_class_and_never_reads_as_bot():
    """Stage-1 item 5 'malformed encodings', revb's test: flip ONE payload bit of a bot word -> the word is `1||nonzero`; every class of
    the lifted relation names it (INV out; the encoding predicates say neither Active nor bot); it is never rewritten (the operand the
    evaluator saw is the raw word; the caller still holds the raw bits to commit)."""
    for w in (1, 16, 32):
        for k in range(w):
            x = L.enc_bot(w) ^ (1 << k)
            raw = x
            assert not L.valid(w, x) and not L.is_bot(w, x) and not L.is_active(w, x) and L.tag(w, x) == 1 and L.payload(w, x) != 0
            with pytest.raises(L.BoundaryFailure, match="invalid representation"):
                L.check_boundary(w, x, what="x", prescribed=False)
            assert x == raw  # nothing normalised the caller's word
        x32 = L.enc_bot(32) ^ (1 << 3)
        x16 = L.enc_bot(16) ^ (1 << 3)
        # strict class: any primitive
        assert L.lifted_prim_v1(P.Bf16Add).evaluate(x16, 0x3F80) == INV16 and L.lifted_prim_v1(P.Bf16Add).evaluate(0x3F80, x16) == INV16
        assert L.lifted_prim(P.Bf16Add).evaluate(x16, 0x3F80, 1) == INV16 and L.lifted_prim(P.Bf16Add).evaluate(x16, 0x3F80, 0) == BOT16  # enabled: INV; disabled: ⊥
        # selection: in the condition, the taken and the untaken branch
        assert L.l_select(32).evaluate(1, x32, 7) == INV32 and L.l_select(32).evaluate(1, 7, x32) == INV32
        assert L.l_select(32).evaluate(0b11, 7, 8) == INV32
        # activity read
        assert L.l_is_active(32).evaluate(x32) == 0b11
        # indexing: as the index and as a row (selected or not)
        g = L.l_gather_row(16, 4)
        assert g.evaluate(x32, *[0x3F80] * 4) == INV16
        assert g.evaluate(1, x16, 0x3F80, 0x3F80, 0x3F80) == INV16 and g.evaluate(1, 0x3F80, x16, 0x3F80, 0x3F80) == INV16
        # continuation: as the previous token and as an EOS-set member
        sl = L.l_step_live(1)
        assert sl.evaluate(x32, 17) == 0b11 and sl.evaluate(5, x32) == 0b11
        # the canonical bot itself is NOT malformed: the same operations read it as inactivity
        assert L.lifted_prim(P.Bf16Add).evaluate(BOT16, 0x3F80, 1) == BOT16 and L.l_is_active(32).evaluate(BOT32) == 0
        assert g.evaluate(1, x16, BOT16, 0x3F80, 0x3F80) == INV16  # INV beats bot in an unselected row too (A3: INV anywhere -> INV)


@pytest.mark.slow
def test_stage1_reference_evaluator_exposes_the_complete_assignment_every_gate_with_its_reading(tmp_path):
    """Stage-1 cross-cutting: `serve3_reference.materialize` assigns EVERY gate (not only outputs); `lifted_dump.rows` reads the complete
    assignment back with each gate's class / width / raw word / A3 reading; on the tiny serve fixture with EOS at the raw event the
    bot population is exactly the inactive suffix of the transcript and no INV word exists anywhere."""
    from verity_vllm.program.lifting import dump as D
    prog, t, out = _lserve((Y_RAW[1],))  # EOS = the second decode's raw token: events 0..1 active, then bot (LAG 0: the EOS step is active)
    rs = list(D.rows(prog, t))
    s = D.summary(rs)
    assert s["complete"] and s["gates"] == prog.gates and len(t) == prog.gates
    assert s["by_reading"].get("inv", 0) == 0 and s["by_class"]["specified"] > 0 and s["by_class"]["strict"] > 0
    assert s["outputs"] == N_EVENTS and sum(1 for r in rs if r["is_output"] and r["reading"]["kind"] == "bot") == out.count(BOT32) > 0
    n_active_ev = sum(1 for x in out if x != BOT32)
    assert [r["reading"]["kind"] for r in rs if r["is_output"]] == ["active"] * n_active_ev + ["bot"] * (N_EVENTS - n_active_ev)
    assert all(r["width"] is not None and r["width"] >= 2 for r in rs if r["class"] in ("strict", "specified"))  # lifted widths w+1
    assert all(r["word"] == t[r["gate"]] for r in rs)  # raw words, byte for byte the transcript's
    s2 = D.dump(prog, t, tmp_path / "dump.jsonl")
    assert s2 == s and sum(1 for _ in open(tmp_path / "dump.jsonl")) == prog.gates


def test_stage1_lfixed_select_one_object_every_route_z_r_incl_bot_out_of_range_named_revb_3a_3c():
    """R17-42 §3 `FixedSelect(r_j, z_0..z_{E-1})` as the specified member `LFixedSelect<E>x<w>_v1` (MOE-PAD-SPEC §2; STAGE1-GAPS G2), against
    revb PADDED-ACCEPTANCE Stage 1 item 3: (a) one object for every route -- the same Definition (id, params, ret, evaluator) whatever `r`
    is; (b) output = `z_r` for every admitted `r`, INCLUDING an inactive slot (-> bot), an unselected bot slot never poisons the read;
    (c) `r` out of range (E, E+1, -1, i32 min/max) -> INV_w NAMED, never a wrap, never a default slot; `r` bot -> bot; INV anywhere -> INV."""
    E = 4
    for w in (16, 32):
        fs = L.l_fixed_select(E, w)
        assert fs.id == f"LFixedSelect{E}x{w}_v1" and fs is L.l_fixed_select(E, w) and fs.id in L.specified_ids()
        assert [t for _, t in fs.params] == [Value(33), Array(E, Value(w + 1))] and fs.ret == Value(w + 1)
        assert getattr(fs, "lift_class") == "specified" and "named failure" in getattr(fs, "lift_reason")
        bot, inv_ = L.enc_bot(w), L.inv(w)
        z = _rand(w, E)
        for r in range(E):  # (b) every admitted route: z_r bit for bit; the SAME object (a)
            assert fs.evaluate(r, *z) == z[r]
            assert fs.evaluate_lazy(lambda i, _r=r: _r if i == 0 else z[i - 1]) == z[r]
            zb = list(z)
            zb[(r + 1) % E] = bot  # an UNSELECTED slot inactive: no effect on the read
            assert fs.evaluate(r, *zb) == z[r]
            zs = list(z)
            zs[r] = bot  # the SELECTED slot inactive: the read IS bot (the slot's own check fails, not a wrap)
            assert fs.evaluate(r, *zs) == bot
        assert fs.evaluate(L.enc_bot(32), *z) == bot  # r bot (an inactive token): nothing selected
        assert fs.evaluate(L.enc_bot(32), *([bot] * E)) == bot
        for oor in (E, E + 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF):  # (c) out of range: NAMED, not z[oor % E], not a default
            assert fs.evaluate(oor, *z) == inv_ and fs.evaluate_lazy(lambda i, _o=oor: _o if i == 0 else z[i - 1]) == inv_
        assert not hasattr(fs, "oor_payload") and getattr(fs, "oor_named") == "INV"  # no registered extension VALUE (unlike LGatherRow)
        # INV anywhere -> INV: the route, a selected row, an unselected row, a bot route beside an INV row
        assert fs.evaluate(L.inv(32), *z) == inv_ and fs.evaluate(_bad(32), *z) == inv_
        zi = list(z); zi[1] = inv_
        assert fs.evaluate(1, *zi) == inv_ and fs.evaluate(0, *zi) == inv_ and fs.evaluate(L.enc_bot(32), *zi) == inv_
        zi[1] = _bad(w)
        assert fs.evaluate(3, *zi) == inv_
    # contrast, on the record: LGatherRow inherits the ORIGINAL family's registered out-of-range VALUE (a weight-bank read has one)
    g = L.l_gather_row(16, E)
    assert g.evaluate(E, *_rand(16, E)) == 0x7FFF and g.oor_payload == 0x7FFF
    # (a) at the Program level: two Programs that differ only in the route VALUE have the same digest -- one object for every route
    fs = L.l_fixed_select(E, 16)
    from verity.ml.circuits.defs import composite
    from verity.ml.circuits.types import Array as _A

    @composite("FixedSelectFixture", 1, [], lambda S: ((("r", Value(33)), ("z", _A(E, Value(17)))), Value(17)), conformance="protocol-defined")
    def FixedSelectFixture(B, S, r, z):
        return B.call(fs, r, z)
    prog = Program(bind(FixedSelectFixture))
    d0 = program_digest(encode_program(prog))
    z = _rand(16, E)  # bf16 payloads for the Value<17> fixture
    outs = []
    for r in range(E):
        t = ref.materialize(prog, [[r], z])
        outs.append(t[prog.circuit.root_outputs()[0]])
        assert program_digest(encode_program(prog)) == d0 and prog.gates == Program(bind(FixedSelectFixture)).gates
    assert outs == z
    dec = decode_program(encode_program(prog), REGISTRY)  # the codec resolves the member by id (cold path)
    assert program_digest(dec) == d0


def test_stage1_cold_decode_resolves_the_continuation_members_by_id():
    """`LIsEOS<E>_v1` / `LEmitNext[H=h]_v1` (R17-32) and `LFixedSelect<E>x<w>_v1` resolve through the codec's lazy family table -- a Program
    decoded in a fresh registry state (no prior materialisation) finds them by id; the digest round-trips."""
    from verity.ml.circuits.codec import _lookup_primitive
    for fid, want in (("LIsEOS1_v1", L.l_is_eos(1)), ("LIsEOS3_v1", L.l_is_eos(3)), ("LEmitNext[H=1]_v1", L.l_emit_next(1)),
                      ("LEmitNext[H=0]_v1", L.l_emit_next(0)), ("LFixedSelect4x16_v1", L.l_fixed_select(4, 16)), ("LFixedSelect8x32_v1", L.l_fixed_select(8, 32))):
        assert _lookup_primitive(REGISTRY, fid) is want, fid


def test_lenable_is_total_never_bot_and_a_disabled_parent_wins_whatever_the_local_condition():
    """v0.6 A9.1 / B7 `LEnable_v1(parent, local)` (S-r17b-40 N2: an enable is Active(0) or Active(1), never ⊥): parent Active(0) → Active(0)
    for EVERY local word (Active, ⊥ -- the local condition's own gate is disabled -- INV); parent Active(1) → the local Boolean; an ill-formed
    enable (⊥/INV parent, or ⊥/INV local under an enabled parent) → INV_1, named, never a silent Active(0)."""
    le = L.l_enable()
    assert le.id == "LEnable_v1" and le.ret == Value(2) and [t for _, t in le.params] == [Value(2), Value(2)]
    for local in (0, 1, 0b10, 0b11):
        assert le.evaluate(0, local) == 0, local
    assert le.evaluate(1, 0) == 0 and le.evaluate(1, 1) == 1
    for parent in (0b10, 0b11):
        for local in (0, 1, 0b10, 0b11):
            assert le.evaluate(parent, local) == 0b11, (parent, local)
    assert le.evaluate(1, 0b10) == le.evaluate(1, 0b11) == 0b11
    # the codec resolves it by id (a cold decode of a padded Program)
    from verity.ml.circuits.codec import _lookup_primitive
    assert _lookup_primitive(REGISTRY, "LEnable_v1") is le


def test_enabled_fixed_select_is_the_a12_slot_combine_member_bot_under_a_disabled_parent_inv_on_oor():
    """A12 / G2 under B7: `Lifted[FixedSelect<E>x<w>_v1]_v2(r, z_1..z_E, enable)`: enable Active(1) → the G2 relation (selected row; OOR r →
    INV_w NAMED -- F1 of S-r17b-40: never ⊥ on an out-of-range id; r = ⊥ → ⊥); enable Active(0) → ⊥ whatever r and the rows."""
    fs = L.l_fixed_select(4, 32)
    lfs = L.lifted_prim(fs)  # a specified member under an enable: Lifted[LFixedSelect4x32_v1]_v2
    rows = [11, 22, 33, 44]
    assert fs.evaluate(2, *rows) == 33 and fs.evaluate(L.enc_bot(32), *rows) == L.enc_bot(32)
    for oor in (4, 5, 1 << 20, (1 << 32) - 1):
        assert fs.evaluate(oor, *rows) == L.inv(32), oor
    assert lfs.id == "Lifted[LFixedSelect4x32_v1]_v2" and lfs.params[-1] == (L.ENABLE_PARAM, L.ENABLE_T) and lfs is L.lifted_prim(fs)
    assert lfs.evaluate(2, *rows, 1) == 33 and lfs.evaluate(2, *rows, 0) == L.enc_bot(32) and lfs.evaluate(7, *rows, 1) == L.inv(32)
    assert lfs.evaluate(7, *rows, 0) == L.enc_bot(32) and lfs.evaluate(2, *rows, 0b11) == L.inv(32)
    # the same for the plain select (LSelect under a disabled parent is ⊥ whatever it would have selected) and the codec resolves both
    ls = L.lifted_prim(L.l_select(32))
    assert ls.id == "Lifted[LSelect32_v1]_v2" and ls.evaluate(1, 5, 6, 1) == 5 and ls.evaluate(1, 5, 6, 0) == L.enc_bot(32)
    from verity.ml.circuits.codec import _lookup_primitive
    assert _lookup_primitive(REGISTRY, lfs.id) is lfs and _lookup_primitive(REGISTRY, ls.id) is ls
    with pytest.raises(TypeError, match="already a lifted"):
        L.lifted_prim(lfs)


def test_b10_new_node_classes_of_the_padded_dense_request_are_named_with_counts():
    """v0.6 B10 (S-r17b-40 N3): every NEW node class of Lift(P) has a name and a count: guard (`guard_gate_count`, includes the one enable
    gate `enable/one` = `enable_gate_count`), controller (`continuation_gate_count - guard_gate_count = 3·cap`), state selects (0 for the
    dense chain).  Checked on the request component: the non-mapped nodes are exactly guard + controller."""
    from tests.program.test_lifted_request import CAP as _CAP, component as _component_fixture  # noqa: F401
    for cap, n_eos, lag in ((3, 1, 0), (9, 2, 1), (5, 1, 0)):
        assert L.enable_gate_count(cap, n_eos, lag) == 1 and L.state_select_count(cap, n_eos, lag) == 0
        assert L.continuation_gate_count(cap, n_eos, lag) == L.guard_gate_count(cap, n_eos, lag) + 3 * cap
        assert L.guard_gate_count(cap, n_eos, lag) == 2 * max(cap - 1 - L.check_lag(lag), 0) + n_eos + 1 + L.enable_gate_count(cap, n_eos, lag)


def test_register_hand_lift_is_honoured_on_the_registry_and_the_decoded_path_and_refuses_a_wrong_interface_by_name():
    """moe M-1063 (MOE-PAD route A): a hand-authored padded lift stands in for `Lifted[twin]_v2` on BOTH paths.  Twin `HandTwin_v1{N}`:
    `y[i] = x[i] + c` with a constant c inside (strict replay: all Active).  Hand lift `LHandTwin_v1{N}(x: Array<N, Value<33>>, enable) ->
    Array<N, Value<33>>`: y[i] = enabled lift of x[i] + c -- and, to show it is the HAND body that runs, it returns ⊥ for i = N-1 whatever
    the enable (the author's construction, not the replay's).  Checked: (1) `lift_def(bind(twin))` on the registry path IS the hand lift;
    (2) a decoded specialization of the twin (encode -> decode) lifts to the hand lift too; (3) evaluation: enable 1 -> hand body, enable 0
    -> ⊥ everywhere; (4) a lift with a wrong interface (no enable / wrong ret) is refused by name at bind time; (5) statics must match."""
    from verity.ml.circuits.codec import decode_program, encode_program
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.circuits.refs import coll_concat
    from verity_vllm.program.registry.prims import I32

    def twin_sig(S):
        return (("x", Array(S.N, I32)),), Array(S.N, I32)

    def twin_body(B, S, x):
        c = B.call(b1.const(32, 7), name="c")
        return coll_concat(Array(S.N, I32), [B.call(P.I32Add, x[i], c, name=f"y{i}") for i in range(S.N)])

    twin = CompositeDefinition("HandTwin", 1, ("N",), twin_sig, twin_body, doc="test twin", conformance="protocol-defined")

    def hand_sig(S):
        return (("x", Array(S.N, Value(33))), (L.ENABLE_PARAM, L.ENABLE_T)), Array(S.N, Value(33))

    def hand_body(B, S, x, enable):
        c = B.call(L.lifted_prim(b1.const(32, 7)), enable, name="c")
        ys = [B.call(L.lifted_prim(P.I32Add), x[i], c, enable, name=f"y{i}") for i in range(S.N - 1)]
        ys.append(B.call(L.l_bot(32), name="hand_bot"))  # the author's construction: the last slot is ⊥ by design
        return coll_concat(Array(S.N, Value(33)), ys)

    hand = CompositeDefinition("LHandTwin", 1, ("N",), hand_sig, hand_body, doc="hand lift", conformance="protocol-defined")
    wrong = CompositeDefinition("LHandTwinWrong", 1, ("N",), lambda S: ((("x", Array(S.N, Value(33))),), Array(S.N, Value(33))),
                                lambda B, S, x: x, doc="no enable", conformance="protocol-defined")
    with pytest.raises(TypeError, match="statics differ"):
        L.register_hand_lift(twin, CompositeDefinition("LHandTwinStatics", 1, ("N", "M"), hand_sig, hand_body, register=False))
    assert L.register_hand_lift(twin, hand) is hand and L.hand_lift_of("HandTwin_v1") is hand
    with pytest.raises(TypeError, match="one lift per id"):
        L.register_hand_lift(twin, wrong)
    # (1) registry path
    sp = bind(twin, N=3)
    lsp = L.lift_def(sp)
    assert lsp.defn is hand and lsp.bindings == {"N": 3} and L.lift_composite(twin) is hand and L.lift_callee(twin) is hand
    # (2) decoded path: the twin's Program encoded and decoded -> its root is body-only -> lift_def gives the hand lift, same bindings
    # (a Program's root INLINES its specialization's body, so the twin must be a CALLEE to be a decoded node: an outer composite calls it)
    outer = CompositeDefinition("HandOuter", 1, ("N",), twin_sig, lambda B, S, x: B.call(bind(twin, N=S.N), x, name="twin"),
                                doc="calls the twin", conformance="protocol-defined")
    dec = decode_program(encode_program(Program(bind(outer, N=3))), REGISTRY)
    node = next(n for n in dec.root.body.nodes if getattr(n.fn, "defn", None) is not None and n.fn.defn.id == "HandTwin_v1")
    assert L._is_decoded(node.fn) and node.fn.bindings == {"N": 3}
    ldec_twin = L.lift_def(node.fn)
    assert ldec_twin.defn is hand and ldec_twin.bindings == {"N": 3}
    ldec = L.lift_def(dec.root)  # the decoded ROOT lifts by replay; its twin callee lifts to the hand lift through lift_callee
    lnode = next(n for n in ldec.body.nodes if getattr(n.fn, "defn", None) is hand)
    assert lnode.fn.bindings == {"N": 3}
    # evaluate the lifted decoded root: its one PARAM is the enable; the decoded root's own `Input33` node (x) is prescribed gate by gate
    from verity.ml.circuits.evaluate import evaluate_region
    from verity.ml.circuits.layout import resolve
    pd = L.as_program(ldec)
    td = {resolve(pd.circuit.root_scope, pd.input_coll(0).refs[0]): 1}
    x_gates = [g for g in range(pd.gates) if getattr(pd.circuit.gate(g).prim, "family", None) == "Input" and g not in td]
    assert len(x_gates) == 3
    td.update(zip(x_gates, [1, 2, 3]))
    evaluate_region(pd.circuit, pd.circuit[0:pd.gates], td)
    assert [td[g] for g in pd.circuit.root_outputs()] == [8, 9, L.enc_bot(32)]
    # (3) evaluation
    pl = Program(lsp)
    t1 = ref.materialize(pl, [[1, 2, 3], [1]])
    o1 = [t1[g] for g in pl.circuit.root_outputs()]
    assert o1 == [8, 9, L.enc_bot(32)]  # the HAND body (⊥ in the last slot by construction), not the replay (which would give 10)
    t0 = ref.materialize(pl, [[1, 2, 3], [0]])
    assert [t0[g] for g in pl.circuit.root_outputs()] == [L.enc_bot(32)] * 3
    # (4) a wrong interface is refused by name at bind time, on both paths
    twin2 = CompositeDefinition("HandTwin2", 1, ("N",), twin_sig, twin_body, doc="test twin 2", conformance="protocol-defined")
    L.register_hand_lift(twin2, wrong)
    with pytest.raises(TypeError, match="not the twin's lifted interface"):
        L.lift_def(bind(twin2, N=2))
    outer2 = CompositeDefinition("HandOuter2", 1, ("N",), twin_sig, lambda B, S, x: B.call(bind(twin2, N=S.N), x, name="twin2"),
                                 doc="calls twin2", conformance="protocol-defined")
    dec2 = decode_program(encode_program(Program(bind(outer2, N=2))), REGISTRY)
    with pytest.raises(TypeError, match="not the twin's lifted interface"):
        L.lift_def(dec2.root)  # the root's replay reaches the decoded twin2 callee and refuses there


def test_every_lift_of_a_primitive_declares_its_tables():
    """The data guard sees a table only through its primitive's declaration, and a lift runs its base's evaluator."""
    from verity.ml.circuits.ops.mufu import MufuEx2Ftz
    lp = L.lifted_prim(MufuEx2Ftz)
    assert MufuEx2Ftz.tables == (("ex2", "library"),)
    assert L.lifted_prim_v1(MufuEx2Ftz).tables == lp.tables == L.with_ordinary(lp, 1).tables == MufuEx2Ftz.tables
