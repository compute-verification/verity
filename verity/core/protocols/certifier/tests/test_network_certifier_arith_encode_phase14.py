"""REDTEAM phase 14, target 1 against ENC-CODE's `ChargeCount_v1{N}` (#3767, the compile slice's drop-in charge:
`(book, carry, en[N], bit[N]) -> (ok, bits[CW], carry')`). The unit is a faithful pure function of its inputs — the
existing suite checks it equals `code_v1`'s charge and that the circuit equals the reference. So the under-counting
surface is the three witness inputs `en`, `bit` and `carry`, which the unit does not and cannot bind to the replay. This
file maps that surface against the reference and the compiled circuit:

- Clearing `en` on a slot the replay asked under-counts: one fewer enabled slot lowers `n_arrived` and drops the
  decision's own bits, and because `run` advances only on enabled slots, the later decisions' contexts shift too. The
  unit charges the lower figure without complaint (ok stays 1), so nothing in the unit ties `en` to the scan's asks.
- Padding with `en = 0` slots is inert: `run` and the per-decision charge depend only on the enabled decisions in order,
  so moving the disabled slots around leaves the charge and `carry'` unchanged. That specific attack ("en=0 slots that
  shift the run") does not work at the unit — a good property, not a gap.
- The charge depends on the witness `carry`: a different `carry.run` gives a different charge, so a prover picks the one
  that charges least. The unit takes the carry as given; binding `carry'` of one window to `carry` of the next is the
  committed state chain's job, which isn't landed yet (the lead's note).

So: no finding against the unit; the finding is a boundary one. The tie of `en`/`bit` to the scan's open-step asks and
the committed decisions (replay_step.py's composition steps 1 and 3) is the compile slice's, and the slice is **not
wired** to the charge unit in this tree (`replay_step` imports no `arith_encode`). Until it is, and until the carry
chains across windows, a prover under-counts by the witness inputs. Owner: ENC-CODE (the carry chain) and ADV-SPEC
(wiring `en`/`bit` to the replay in the compile slice).
"""
from __future__ import annotations

import pytest

from verity.core.protocols.certifier import arith_encode as AE
from verity.core.protocols.certifier import replay_step as RS

N = 14  # the compile slice's class (P·A = 14 slots a window)


def _book():
    """A book whose n_arrived bins and KT contexts make the under-counts visible; `charge_book`-shaped totals."""
    arrived = (50, 40, 30, 20, 10)  # bins 0 … bitlen(N)
    zeros = ((10, 100),) * len(AE.ZERO_COUNTS)
    return AE.ChargeBook(arrived, 200, zeros)


def _pad(xs):
    return list(xs) + [0] * (N - len(xs))


def test_phase14_clearing_an_asked_slots_en_under_counts():
    """A window of four asked decisions; clearing the second slot's `en` (a decision the replay asked) charges fewer
    bits with ok still 1. The unit doesn't bind `en` to the asks, so it accepts the under-count."""
    book = _book()
    en, bit = _pad([1, 1, 1, 1]), _pad([1, 0, 0, 1])
    ok, honest, _ = AE.reference_charge(N, book, AE.Carry(), en, bit)
    cleared, bit_c = _pad([1, 0, 1, 1]), bit
    ok_c, under, _ = AE.reference_charge(N, book, AE.Carry(), cleared, bit_c)
    assert ok == 1 and ok_c == 1, "both are accepted: the unit raises no refusal for the dropped ask"
    assert under < honest, f"clearing an asked slot's en under-counts: {under} < {honest}"
    # the compiled circuit agrees: this is the unit's own charge, not a reference-only artifact
    g_honest, g_under = AE.evaluate_charges(N, [(book, AE.Carry(), en, bit), (book, AE.Carry(), cleared, bit_c)])
    assert (g_honest[0], g_honest[1]) == (1, honest) and (g_under[0], g_under[1]) == (1, under)


def test_phase14_clearing_en_also_shifts_the_run_of_later_decisions():
    """`run` advances only on enabled slots, so clearing one asked slot's `en` changes the run the later decisions are
    charged at — the carry the next window inherits differs too."""
    book = _book()
    bit = _pad([0, 0])  # 0-bits run the counter up; the run is RUNS-1 = 2 capped, so stay below it
    _, _, cy_full = AE.reference_charge(N, book, AE.Carry(), _pad([1, 1]), bit)
    _, _, cy_cut = AE.reference_charge(N, book, AE.Carry(), _pad([1, 0]), bit)
    assert (cy_full.run, cy_cut.run) == (2, 1), f"clearing a slot shifts the run carried on: {cy_full.run} vs {cy_cut.run}"


def test_phase14_en_zero_padding_does_not_shift_the_run():
    """Two windows with the same enabled decisions in the same order but the disabled (`en = 0`) slots placed
    differently charge identically and carry the same run: padding with en=0 slots cannot shift the run."""
    book = _book()
    # enabled decisions, in order, are (1, 0, 1) in both; only the en=0 slots move
    a_en, a_bit = _pad([1, 0, 1, 0, 1]), _pad([1, 0, 0, 0, 1])
    b_en, b_bit = _pad([1, 1, 1, 0, 0]), _pad([1, 0, 1, 0, 0])
    oa, ba, ca = AE.reference_charge(N, book, AE.Carry(), a_en, a_bit)
    ob, bb, cb = AE.reference_charge(N, book, AE.Carry(), b_en, b_bit)
    assert oa == ob == 1 and (ba, ca.run, ca.counts) == (bb, cb.run, cb.counts), \
        "en=0 placement is inert: same charge and carry"


def test_phase14_the_charge_depends_on_the_chosen_carry():
    """The carry is witness. With the KT counts skewed so context 2 is cheap for a 1-bit and context 0 dear, a prover
    who hands `carry.run = 2` is charged fewer bits than the honest `run = 0` for the same decisions — so the carry must
    be bound across windows (the committed state chain), which isn't landed."""
    skew = (((1 << 15) - 1, 0), (0, 0), (0, (1 << 15) - 1))  # ctx 0: all 0s (a 1-bit is dear); ctx 2: all 1s (cheap)
    book = _book()
    en, bit = _pad([1, 1, 1, 1]), _pad([1, 1, 1, 1])
    charges = {}
    for run in (0, 2):
        ok, bits, _ = AE.reference_charge(N, book, AE.Carry(skew, run), en, bit)
        assert ok == 1
        charges[run] = bits
    assert charges[2] < charges[0], f"choosing carry.run=2 under-counts: {charges[2]} < {charges[0]}"
    g0, g2 = AE.evaluate_charges(N, [(book, AE.Carry(skew, 0), en, bit), (book, AE.Carry(skew, 2), en, bit)])
    assert g0[1] == charges[0] and g2[1] == charges[2], "the compiled circuit charges the chosen carry too"


def test_phase14_the_compile_slice_is_not_wired_to_the_charge_unit():
    """replay_step.py (the compile unit's core) describes the charge in its composition but imports no `arith_encode`,
    so nothing in this tree ties `en`/`bit` to the scan's asks and committed decisions. The binding is still to be built."""
    assert not hasattr(RS, "arith_encode") and not hasattr(RS, "build_charge") and not hasattr(RS, "ChargeCount")
