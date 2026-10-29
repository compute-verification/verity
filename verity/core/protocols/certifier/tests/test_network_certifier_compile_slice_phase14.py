"""The compile slice's charge inputs: `ChargeCount_v1{N}` alone is a faithful function of its witness ``en``, ``bit``
and ``carry``, so a prover under-counts by clearing an asked slot's enable or choosing the carry until the slice binds
them. In `compile_slice.CompileSlice_v1` none of the three is witness: ``en`` is i < ``count`` and ``bit`` is ``adv``'s
decisions, the committed advice that ReplayLoad holds the scanned window row to, and the carry is ``cin``, which
`charge_chain.ChargeLink_v1` holds to ``cout`` and the verifier chains window to window (`charge_chain.check_chain`).
Each case runs on w212 at its vLLM engine class (`test_network_certifier_compile_slice_w212.py`)."""
from __future__ import annotations

import pytest

from verity.core.protocols.certifier import arith_encode as AE, charge_chain as CC
from verity.core.protocols.certifier import compile_slice as CS
from verity.core.protocols.certifier.tests.test_network_certifier_compile_slice import BOOK
from verity.core.protocols.certifier.tests.test_network_certifier_compile_slice_w212 import CARRY, CLASS, FITTED, OWN
from verity.core.protocols.certifier.tests.test_network_certifier_compile_slice_w212 import _load
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.ml.commitments import hm96

PROGRAM, REGISTRATION = "compile-slice-phase14", "cd" * 64


@pytest.fixture(scope="module")
def x():
    return _load(CLASS)


def _refused(c: CS.Case) -> set[str]:
    return {f for f, ok in c.flags.items() if not ok}


def test_no_enable_or_carry_is_witness(x):
    """The unit's rows hold no enable, context, frequency or carry the prover chooses: ``wit`` is the scan's state,
    window, arrivals, bits (and hits and cache under prefix caching), and the charge reads ``adv``, ``cin`` and
    ``cout``, rows the verifier holds by their b‖c."""
    F = CS.fields(x.S)
    assert [name for name, _n, _w in F["wit"]] == ["state", "window", "arr", "bits"]
    assert [name for name, _n, _w in F["adv"]] == ["dec", "count", "steps"]
    assert F["cin"] == F["cout"] == [("carry", 1, CC.ROW_BITS)]


def test_clearing_an_asked_enable_is_refused(x):
    """Clearing an asked slot's enable is, in the slice, advice whose count leaves out a decision the scan asks: the
    last (request 310's admission) or any earlier one (request 309's). It charges less, and the unit refuses it by
    ``load`` while the window row keeps the decision, and by ``steps`` (and ``rows``) when the window row drops it
    too, since the scan's ask then reads past the count."""
    kw = {"served": x.served, "carry": CARRY}
    honest = CS.case(x.S, x.sl, BOOK, **kw)
    dec = x.sl.decisions
    for k in (13, 6):
        cut = dec[:k] + dec[k + 1:]
        kept, read = CS.case(x.S, x.sl, BOOK, advice=cut, **kw), CS.case(x.S, x.sl, BOOK, advice=cut, window=cut, **kw)
        assert kept.length < honest.length, k
        assert (_refused(kept), _refused(read)) == ({"load"}, {"steps", "rows"}), k


def test_the_run_is_the_committed_decisions(x):
    """The run each decision is charged at moves over the committed decisions in commit order from ``cin``'s: ``cout``
    is that carry', and any other carry-out (the run a cleared enable leaves, 1 in place of 0) is refused by
    ``code``."""
    honest = CS.case(x.S, x.sl, BOOK, served=x.served, carry=CARRY)
    cb, _prior = AE.charge_book(BOOK, x.S.D)
    en, bits = [1] * 14 + [0] * (x.S.D - 14), list(x.sl.decisions) + [0] * (x.S.D - 14)
    assert honest.carry == AE.reference_charge(x.S.D, cb, CARRY, en, bits)[2] and honest.carry.run == 0
    shifted = AE.Carry(honest.carry.counts, 1)
    assert _refused(CS.case(x.S, x.sl, BOOK, served=x.served, carry=CARRY, cout=shifted)) == {"code"}


def test_the_padding_is_inert():
    """The class's padding slots (en 0 past the count: 18 at D = 32) change neither the charge nor the carry: w212 at its
    own statics (D = 14) and padded into the class charge 27 bits to the same carry'."""
    own, cls = _load(OWN), _load(CLASS)
    a, b = (CS.case(y.S, y.sl, BOOK, served=y.served, carry=CARRY) for y in (own, cls))
    assert (a.verdict, a.length, a.carry) == (b.verdict, b.length, b.carry) == (CS.ACCEPT, 27, b.carry)


def test_a_chosen_carry_is_refused(x):
    """A carry-in of the prover's choosing (counts fitted to w212, 25 bits in place of 27) is accepted by the unit, which
    takes any carry-in; the verifier's chain refuses it (``CH-carry``) whether the prover commits it
    under a fresh salt or the public start salt; and a window whose carry-out claims such a restart for the next one has
    its own unit refused by ``code``."""
    honest = CS.case(x.S, x.sl, BOOK, served=x.served, carry=CARRY)
    restart = CS.case(x.S, x.sl, BOOK, served=x.served, carry=FITTED)
    assert restart.verdict == CS.ACCEPT and restart.length == 25 < honest.length
    before = CC.commit_window(PROGRAM, BOOK, CARRY, REGISTRATION)
    ok = CC.commit_window(PROGRAM, BOOK, honest.carry, REGISTRATION, carry=before)
    assert CC.check_chain([CC.openings(before), CC.openings(ok)], PROGRAM, BOOK) is None
    for salt in (hm96.SHA512.fresh_salt(), CC.start_salt(PROGRAM)):
        forged = produced.commit(PROGRAM, CC.VALUE, [CC.row(CC.encode(FITTED)), CC.row(CC.encode(restart.carry))],
                                 REGISTRATION, [salt, hm96.SHA512.fresh_salt()], CC.WORD_BITS, CC.SCHEMA)
        got = CC.check_chain([CC.openings(before), CC.openings(forged)], PROGRAM, BOOK)
        assert got == "CH-carry: window 1's carry-in isn't window 0's carry-out"
    claimed = CS.case(x.S, x.sl, BOOK, served=x.served, carry=CARRY, cout=FITTED)
    assert _refused(claimed) == {"code"}


def test_the_slice_wires_the_charge(synth_cases):
    """The circuit, not only its reference, ties the charge to the scan: on the demonstration slice the unit's verdict
    word refuses advice that drops an asked decision (``load``) and a forged carry-out (``code``), and accepts a
    restarted carry-in at its lower charge, which only the chain refuses."""
    st, cases = synth_cases
    got = CS.evaluate(st, list(cases.values()))
    assert got == [c.verdict for c in cases.values()] == [CS.ACCEPT, CS.ACCEPT & ~1, CS.ACCEPT & ~4, CS.ACCEPT]
    assert cases["restart"].length < cases["honest"].length


@pytest.fixture(scope="module")
def synth_cases():
    from types import SimpleNamespace

    st, sl = CS.demo()
    S = SimpleNamespace(**st)
    honest = CS.case(S, sl, BOOK)
    served = list(honest.steps)
    return st, {"honest": honest, "cleared": CS.case(S, sl, BOOK, served=served, advice=sl.decisions[:-1]),
                "carry": CS.case(S, sl, BOOK, served=served, cout=AE.first_carry(BOOK)),
                "restart": CS.case(S, sl, BOOK, served=served, carry=AE.Carry(((0, 0),) * AE.RUNS, 0))}
