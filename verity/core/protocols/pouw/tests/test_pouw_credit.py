"""`credit.py`'s rules on a served window (RT4-INT-32, RT4-INT-34; PRF-132): each mutation of `pouw_credit_cases` is
refused by the rule it aims at, its message naming the fix, and the honest windows are accepted. The Lean
(`PouwCheck.Credit`) is held to the same cases by `lean/check/tests/test_difftest.py`."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from verity.core.protocols.pouw import credit as Cr

PKG = Path(__file__).resolve().parents[1] / "lean/check/tests"
for _name in ("pouw_rounds_cases", "pouw_credit_cases"):
    _spec = importlib.util.spec_from_file_location(_name, PKG / f"{_name}.py")
    sys.modules[_name] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules[_name])
cases = sys.modules["pouw_credit_cases"]

SEEDS = range(3)

#: each `credited` mutation's refusal: its code and its reason's start
CREDIT_EXPECT = {
    "changed": ("draw", Cr.CHANGED), "withdrawn": ("draw", Cr.CHANGED), "row-changed": ("draw", Cr.CHANGED),
    "stale-root": ("draw", Cr.DRAW_ROOT), "no-bit": ("credit", "manifest row "), "bit-form": ("credit", "manifest row "),
    "no-z": ("credit", "call "), "index": ("manifest", "manifest row "), "row-form": ("manifest", "manifest row "),
    "calls-form": ("manifest", Cr.CALLS_FORM), "salt-form": ("manifest", Cr.SALT_FORM),
    "served-form": ("manifest", Cr.SERVED_FORM), "rounds-form": ("rounds", "round "), "draw-form": ("draw", Cr.DRAW_FORM),
    "no-draw": ("draw", Cr.NO_DRAW), "draw-coin": ("coin", "the draw coin: "),
    "key-form": ("coin", "the verifier's challenger key is malformed"),
    "forged-ledger": ("coin", "the challenger's session is not trusted"),
}
#: each pin mutation's refusal (RT4-INT-40): its reason holds the fix
PIN_FIX = {
    "pin-not-main": "which is not a commit of main's first-parent history: repin the host to a main commit",
    "repinned": "the host was repinned mid-session; repin the host to a main commit between sessions",
    "no-pin": "names no pinned tree: repin the host to a main commit",
    "pin-form": "names a pinned tree that is not a commit (40 lowercase hex): repin the host to a main commit",
    "no-main": "not trusted: the verifier holds no copy of main's history",
    "main-form": "not trusted: the verifier's copy of main's history is malformed: ",
}
CREDIT_EXPECT.update({name: ("coin", "the challenger's session is not trusted: ") for name in PIN_FIX})
OUTCOME_EXPECT = {
    "missing": ("missing", "drawn unit "), "failed": ("failed", "drawn unit "), "twice": ("outcome", "unit "),
    "undrawn": ("outcome", "unit "), "drawn-form": ("outcome", Cr.DRAWN_FORM), "proofs-form": ("outcome", Cr.PROOFS_FORM),
}
UNTIED = "Register the A value whose rows each round's root committed"
A_LINK_EXPECT = {
    "untied": UNTIED, "late-rows": UNTIED, "rewritten": UNTIED, "round-root": "`credit.round_root`", "gap": "`credit.cover`",
    "extra": "`credit.cover`", "swapped": "`credit.cover`", "misaligned": "`credit.cover`",
    "unknown": "which is no registered A value", "no-a": Cr.NO_A, "twice": " twice", "entry-form": "has no root, domain",
    "blocks": "rounds' equal blocks", "log-form": Cr.LOG_FORM, "round-form": "inner root is not",
    "segments-form": "A segments are not", "entries-form": Cr.ENTRIES_FORM,
}


def _credited(name: str, seed: int):
    d = cases.credit_instance(random.Random(f"{name}/{seed}"), name)
    return d, Cr.credited(*cases.credit_args(d))


def test_every_mutation_is_aimed():
    assert set(CREDIT_EXPECT) == set(cases.CREDIT_NAMES)
    assert set(OUTCOME_EXPECT) == set(cases.OUTCOME_NAMES)
    assert set(A_LINK_EXPECT) == set(cases.A_LINK_NAMES)


@pytest.mark.parametrize("seed", SEEDS)
def test_honest_window_credits_its_bits(seed):
    d, (got, refused) = _credited("honest", seed)
    assert refused is None
    assert got == [i for i, row in enumerate(d["calls"]) if row[Cr.CREDIT]]


@pytest.mark.parametrize("seed", SEEDS)
def test_credit_bit_changed_after_draw_is_refused(seed):
    """RT4-INT-32 (a): a verdict withdrawn (or granted) after the draw coin, the draw root recomputed over the new bits,
    is not the root the challenger's log shows the draw followed."""
    for name in ("withdrawn", "changed"):
        _, (got, refused) = _credited(name, seed)
        assert got is None and refused == ("draw", Cr.CHANGED)
        assert "Commit every call's credit bit in its manifest row before the draw root" in refused[1]


@pytest.mark.parametrize("seed", SEEDS)
def test_credited_call_needs_a_bit_and_a_z_root(seed):
    for name, fix in (("no-bit", "commits each call's verdict as `credit`"), ("no-z", "carries its z table's root")):
        _, (_, refused) = _credited(name, seed)
        assert refused[0] == "credit" and fix in refused[1]


@pytest.mark.parametrize("seed", SEEDS)
def test_failed_or_missing_drawn_unit_refuses_the_window(seed):
    """RT4-INT-32 (b): no partial credit."""
    assert Cr.outcome(**cases.outcome_instance(random.Random(f"honest/{seed}"))) is None
    missing = Cr.outcome(**cases.outcome_instance(random.Random(f"missing/{seed}"), "missing"))
    failed = Cr.outcome(**cases.outcome_instance(random.Random(f"failed/{seed}"), "failed"))
    assert missing[0] == "missing" and "Prove and verify every drawn unit" in missing[1]
    assert failed[0] == "failed" and "refuses the whole window" in failed[1]
    assert all("no partial credit" in r[1] for r in (missing, failed))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_tree_not_tied_to_the_round_roots_is_refused(seed):
    """RT4-INT-34: an A tree registered over rows other than those each round's root committed before its coin."""
    assert Cr.a_link(**cases.a_link_instance(random.Random(f"honest/{seed}"))) is None
    for name in ("untied", "late-rows", "rewritten"):
        refused = Cr.a_link(**cases.a_link_instance(random.Random(f"{name}/{seed}"), name))
        assert refused[0] == "a-link" and UNTIED in refused[1]


@pytest.mark.parametrize("name", sorted(CREDIT_EXPECT))
def test_credit_mutation(name):
    code, start = CREDIT_EXPECT[name]
    for seed in SEEDS:
        _, (got, refused) = _credited(name, seed)
        assert got is None and refused[0] == code and refused[1].startswith(start), (seed, refused)


@pytest.mark.parametrize("name", sorted(PIN_FIX))
def test_a_window_whose_host_ran_a_commit_outside_mains_first_parent_history_is_not_credited(name):
    """RT4-INT-40: the draw coin counts only through a session every ledger entry of which the host wrote at one commit
    of the verifier's copy of main's first-parent history."""
    for seed in SEEDS:
        _, (got, refused) = _credited(name, seed)
        assert got is None and refused[0] == "coin" and PIN_FIX[name] in refused[1], (seed, refused)


@pytest.mark.parametrize("name", sorted(OUTCOME_EXPECT))
def test_outcome_mutation(name):
    code, start = OUTCOME_EXPECT[name]
    for seed in range(5):
        refused = Cr.outcome(**cases.outcome_instance(random.Random(f"{name}/{seed}"), name))
        assert refused[0] == code and refused[1].startswith(start), (seed, refused)


@pytest.mark.parametrize("name", sorted(A_LINK_EXPECT))
def test_a_link_mutation(name):
    for seed in range(5):
        refused = Cr.a_link(**cases.a_link_instance(random.Random(f"{name}/{seed}"), name))
        assert refused[0] == "a-link" and A_LINK_EXPECT[name] in refused[1], (seed, refused)


def test_cover_is_the_fewest_aligned_subtrees():
    for lo, hi in ((0, 1), (0, 8), (3, 11), (5, 6), (6, 21), (16, 48)):
        segs = Cr.cover(lo, hi)
        at = lo
        for level, index in segs:
            assert index << level == at
            at += 1 << level
        assert at == hi
    assert Cr.cover(0, 8) == [(3, 0)] and Cr.cover(3, 11) == [(0, 3), (2, 1), (1, 4), (0, 10)]
