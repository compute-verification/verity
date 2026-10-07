"""The outer firewall's agreement harness (`verifier/firewall_outer_agree.py`) on hand-made transcripts: a check's outcome is
recomputed from its affine form over the tape's pads, and a case's expectation holds only when the firewall stopped where
the control says and the contract cut at that item for the control's reason. The transcripts themselves come from
`flock-circuit`'s recorder (`live/src/firewall_transcript.rs`), run by the outer agreement run."""
from __future__ import annotations

import importlib.util
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1] / "verifier"


def _harness():
    spec = importlib.util.spec_from_file_location("firewall_outer_agree", VERIFIER / "firewall_outer_agree.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OA = _harness()


def h(v: int) -> str:
    return f"{v:032x}"


def transcript(items, refused=None, at=None, stopped=None, pads=(0x1234, 0xBEEF << 70), start=None):
    return {"tape": {"pads": [h(p) for p in pads]}, "items": items,
            "firewall": {"gate": {"refused": refused, "at": at, "from": start}, "stopped": stopped, "recorder_errors": []}}


def check(ok, c, a):
    return {"check": {"relation": "r", "pads": [0], "ok": ok, "form": {"c": h(c), "a": [[i, h(x)] for i, x in a]}}}


def test_gmul_is_the_ghash_polynomials():
    assert OA.gmul(1 << 127, 2) == 0x87
    assert OA.gmul(3, 3) == 5


def test_a_checks_outcome_is_recomputed_from_its_form():
    p0, p1 = 0x1234, 0xBEEF << 70
    a = 7
    holds = OA.gmul(a, p0) ^ p1
    t = transcript([check(True, holds, [(0, a), (1, 1)]), check(False, holds ^ 1, [(0, a), (1, 1)]), check(True, 1, [(0, a)])])
    assert OA.recheck(t) == (2, 3)


def test_a_control_must_stop_where_the_firewall_did_and_for_its_reason():
    why = "rep 0 algebraic phase round 0 frame 1: label: not the value the statement fixes"
    t = transcript([{"fixed": {}}] * 10, refused="FIREWALL-REFUSED: algebraic phase, round 0, frame 1: label \"glock\"", at=9)
    assert OA.judge("label", t, {"accepted": False, "released": 9, "why": why}) == (True, "")
    assert not OA.judge("label", t, {"accepted": False, "released": 8, "why": why})[0]
    assert not OA.judge("label", t, {"accepted": False, "released": 9, "why": "the session ended without Finish"})[0]
    assert not OA.judge("label", t, {"accepted": True, "released": 10, "why": None})[0]
    assert not OA.judge("root", t, {"accepted": False, "released": 9, "why": why})[0]


def test_the_prover_stopping_is_judged_at_the_first_failing_check():
    t = transcript([{"masked": {}}, check(True, 0, []), check(False, 1, []), check(False, 1, [])],
                   stopped="PROVER-STOPPED: the batched constraint fails (⟨c, h⟩ ≠ t)")
    assert OA.judge("algebra", t, {"accepted": False, "released": 2, "why": "the relation r fails"}) == (True, "")
    assert not OA.judge("algebra", t, {"accepted": False, "released": 3, "why": "the relation r fails"})[0]
    assert not OA.judge("honest", t, {"accepted": True, "released": 4, "why": None})[0]


def test_an_honest_session_is_accepted_with_nothing_stopped():
    t = transcript([{"finish": {}}])
    assert OA.judge("honest", t, {"accepted": True, "released": 1, "why": None}) == (True, "")
    assert not OA.judge("honest", t, {"accepted": False, "released": 0, "why": "x"})[0]


def test_every_session_is_held_to_the_honest_sessions_public_input(tmp_path):
    """`public_input`: the honest sessions' public input, which must be the same in each; a control's may name more, but a
    statement value it gives a name the honest one fixes otherwise, or a relation the honest one lacks, is a miss; and
    without an honest session nothing fixes the public input."""
    import json

    import pytest

    def session(d, k, statement, relations=("r",), computes=("σ",)):
        d.mkdir(exist_ok=True)
        (d / f"session-{k}.json").write_text("{}")
        pub = {"kind": "verity/firewall-public/v0", "schedule": None, "relations": list(relations), "statement": statement,
               "computes": list(computes)}
        (d / f"public-{k}.json").write_text(json.dumps(pub))
        return d / f"session-{k}.json"

    hon, ctl = tmp_path / "honest", tmp_path / "label"
    honest = [session(hon, k, {"Register": "00"}) for k in range(2)]
    odd = session(hon, 2, {"Register": "01"})
    more = session(ctl, 0, {"Register": "00", "the firewall's refusal": "01"})
    clash = session(ctl, 1, {"Register": "01"})
    extra = session(ctl, 2, {"Register": "00"}, relations=("r", "the firewall's own"))
    pub, bad = OA.public_input([("honest", honest + [odd]), ("label", [more, clash, extra])])
    assert pub["statement"] == {"Register": "00"} and pub["relations"] == ["r"]
    assert set(bad) == {str(odd), str(clash), str(extra)} and str(more) not in bad
    with pytest.raises(SystemExit):
        OA.public_input([("label", [more])])


def fx(what, by="statement", pads=(), value="00"):
    return {"fixed": {"what": what, "by": by, "want": value, "sent": value, "pads": list(pads)}}


def mx(pad, value="00"):
    return {"masked": {"pad": pad, "value": value, "sent": value}}


#: an honest session's items, as the outer recorder writes them (the Commit's σ before its roots, a rep's inner proof before
#: its proof)
HONEST = [fx("Register"), fx("Hello", "own"), fx("rep 0: Open"), fx("the session's Commit: σ (root_f)", "own"),
          fx("the session's Commit: the level-0 roots", "recheck"), fx("the session's Link: y"),
          fx("rep 0 algebraic phase round 0 frame 0: label"), mx(0), mx(1),
          {"coins": {"stream": "circuit/rep0", "round": 0, "coins": [], "relayed": []}},
          mx(2), {"check": {"relation": "r", "pads": [0, 1, 2], "ok": True, "form": {"c": h(0), "a": []}}},
          fx("rep 0 proof: shape"), fx("rep 0 proof: the masked zerocheck and lincheck", "own", (0, 1, 2)), {"finish": {}}]


def test_an_honest_session_with_its_items_reordered_dropped_duplicated_or_cut_short_is_a_miss(tmp_path):
    """`sequence`: the honest sessions' item sequence is one and the same, and an honest session is all of it. Under the
    outer public input (no schedule) the contract accepts each of these, so only the harness refuses them, each at the item
    where it leaves the honest order."""
    import json

    seq = [OA.place(it) for it in HONEST]
    swap = HONEST[:5] + [HONEST[6], HONEST[5]] + HONEST[7:]
    moved = [HONEST[0]] + HONEST[2:7] + [HONEST[1]] + HONEST[7:]
    variants = {
        "reordered": (swap, 5, "fixed 'rep 0 algebraic phase round 0 frame 0: label' by statement"),
        "dropped": (HONEST[:9] + HONEST[10:], 9, "masked under pad 2"),
        "duplicated": (HONEST[:8] + [HONEST[7]] + HONEST[8:], 8, "masked under pad 0"),
        "moved": (moved, 1, "fixed 'rep 0: Open' by statement"),
        "early-finish": (HONEST[:7] + [{"finish": {}}], 7, "finish"),
        "cut-short": (HONEST[:-1], len(HONEST) - 1, "the end"),
        "longer": (HONEST + [HONEST[-1]], len(HONEST), "finish"),
    }
    assert OA.sequence_miss(transcript(HONEST), seq, True) == ""
    for name, (items, at, got) in variants.items():
        why = OA.sequence_miss(transcript(items), seq, True)
        assert why.startswith(f"its item {at} is {got}, where the honest sessions have "), (name, why)

    d = tmp_path / "honest"
    d.mkdir()
    files = []
    for k, items in enumerate([HONEST, HONEST] + [v[0] for v in variants.values()]):
        (d / f"session-{k}.json").write_text(json.dumps(transcript(items)))
        files.append(d / f"session-{k}.json")
    got, bad = OA.sequence([("honest", files)])
    assert got == seq and set(bad) == {str(f) for f in files[2:]}


def test_a_controls_items_are_the_honest_ones_up_to_its_cut_and_its_candidate_where_it_leaves():
    """A control's items before the candidate the firewall held back (`firewall.gate.from`) are the honest sequence's; the
    candidate is the honest run where its items leave, which need not be `from` (the Commit's roots are checked before σ goes
    out, a proof before its inner proof's rounds), and its dependencies may be fewer pads (as of the firewall's check)."""
    seq = [OA.place(it) for it in HONEST]
    tampered = lambda it: {k: {**v, "sent": "ff"} for k, v in it.items()}
    ctl = lambda items, start, **kw: transcript(items, refused="FIREWALL-REFUSED: x", at=len(items) - 1, start=start, **kw)
    root = ctl(HONEST[:3] + [tampered(HONEST[4])], 3)
    label = ctl(HONEST[:6] + [tampered(HONEST[6])], 6)
    proof = ctl(HONEST[:11] + [HONEST[12], fx("rep 0 proof: the masked zerocheck and lincheck", "own", (0, 1), "ff")], 11)
    stand_in = ctl(HONEST[:6] + [HONEST[6], {"fixed": {**fx("the firewall's refusal: x", "own")["fixed"], "stand_in": True}}], 6)
    stopped = transcript(HONEST[:12], stopped="PROVER-STOPPED: x")
    for t in (root, label, proof, stand_in, stopped):
        assert OA.sequence_miss(t, seq, False) == ""

    dropped = ctl(HONEST[:1] + HONEST[2:3] + [tampered(HONEST[4])], 2)
    assert OA.sequence_miss(dropped, seq, False).startswith("its item 1 is fixed 'rep 0: Open' by statement")
    twice = ctl(HONEST[:11] + [HONEST[12], HONEST[12]], 11)
    assert OA.sequence_miss(twice, seq, False).startswith("its held-back candidate's item 12 is fixed 'rep 0 proof: shape'")
    more_pads = ctl(HONEST[:11] + [HONEST[12], fx("rep 0 proof: the masked zerocheck and lincheck", "own", (0, 1, 2, 3))], 11)
    assert OA.sequence_miss(more_pads, seq, False).startswith("its held-back candidate's item 12 is")
    nowhere = ctl(HONEST[:6] + [fx("Register")], 6)
    assert "which no honest session sends after item 6" in OA.sequence_miss(nowhere, seq, False)
    swapped = transcript(HONEST[:7] + [HONEST[8], HONEST[7]], stopped="PROVER-STOPPED: x")
    assert OA.sequence_miss(swapped, seq, False).startswith("its item 7 is masked under pad 1")
    unmarked = ctl(HONEST[:3] + [tampered(HONEST[4])], None)
    assert "firewall.gate.from" in OA.sequence_miss(unmarked, seq, False)
