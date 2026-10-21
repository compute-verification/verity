"""`credit.py`'s rules on a served window and its period (RT4-INT-32, -34 and -42; PRF-132): each mutation of
`pouw_credit_cases` is refused by the rule it aims at, its message naming the fix, and the honest windows and periods are
accepted. The drawn units are the draw coin's over the window's own layout, whose strata are the compiled
`h1window.program`'s. The Lean (`PouwCheck.Credit`) is held to the same cases by `lean/check/tests/test_difftest.py`."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw.circuit import plan

PKG = Path(__file__).resolve().parents[1] / "lean/check/tests"
for _name, _dir in (("pouw_rounds_cases", PKG), ("pouw_credit_cases", PKG), ("test_circuit_h1window", Path(__file__).parent)):
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_name, _dir / f"{_name}.py")
        sys.modules[_name] = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(sys.modules[_name])
cases, rc, h1 = (sys.modules[n] for n in ("pouw_credit_cases", "pouw_rounds_cases", "test_circuit_h1window"))

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
NOT_DRAWN = "has a proof outcome and was not drawn"
DRAWN_EXPECT = {
    "coin-form": ("draw", Cr.COIN_FORM), "args-form": ("draw", Cr.DRAW_ARGS), "layout-form": ("layout", Cr.LAYOUT_FORM),
    "named-twice": ("layout", "is named twice"), "unknown-linear": ("layout", "which is no linear of the window"),
    "uncalled": ("layout", "is never called"), "slot": ("layout", "(h1window.slot)"),
    "kn": ("layout", "n whole blocks of the tile's"), "blocks": ("layout", "is read by some call of K"),
}
OUTCOME_EXPECT = {
    "missing": ("missing", "Prove and verify every drawn unit"), "failed": ("failed", "refuses the whole window"),
    "twice": ("outcome", "a retry never reproves it"), "undrawn": ("outcome", NOT_DRAWN), "chosen": ("outcome", NOT_DRAWN),
    "proofs-form": ("outcome", Cr.PROOFS_FORM),
}
WINDOW_EXPECT = {
    "manifest-count": ("manifest", "each call of the window's Program is one manifest row"),
    "chosen": ("outcome", NOT_DRAWN), "failed": ("failed", "no partial credit"), "missing": ("missing", "no partial credit"),
    "layout-form": ("layout", Cr.LAYOUT_FORM), "credit-changed": ("draw", Cr.CHANGED), "credit-no-draw": ("draw", Cr.NO_DRAW),
    "credit-forged-ledger": ("coin", "the challenger's session is not trusted"),
}
PERIOD_EXPECT = {
    "no-close": ("period", "present the ledger after its close"), "incomplete": ("period", "closed incomplete"),
    "missing": ("missing", "present them all, in the close's order"),
    "order": ("order", "in the order its close lists them"), "extra": ("period", "only its close's windows count"),
    "foreign": ("period", "only the windows its close lists count"), "unopened": ("period", "opens no period"),
    "name": ("period", "the period is not named"), "window-refused": ("window", "so the period earns nothing"),
    "windows-form": ("period", Cr.WINDOWS_FORM), "forged-ledger": ("coin", "the auditor's ledger is not trusted"),
    "key-form": ("coin", "the verifier's challenger key is malformed"),
    "other-circuit": ("circuit", "prove the period's declared circuit"),
    "no-circuit": ("circuit", "commits no cap or circuit"), "descriptor-form": ("circuit", Cr.DESCRIPTOR_HEX),
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
    assert set(DRAWN_EXPECT) == set(cases.DRAWN_NAMES) - {"budget"}
    assert set(OUTCOME_EXPECT) == set(cases.OUTCOME_NAMES)
    assert set(WINDOW_EXPECT) == set(cases.WINDOW_NAMES)
    assert set(PERIOD_EXPECT) == set(cases.PERIOD_NAMES)
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
    assert _outcome("honest", seed) is None
    missing, failed = _outcome("missing", seed), _outcome("failed", seed)
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


@pytest.mark.parametrize("name", sorted(OUTCOME_EXPECT))
def test_outcome_mutation(name):
    code, part = OUTCOME_EXPECT[name]
    for seed in range(5):
        refused = _outcome(name, seed)
        assert refused[0] == code and part in refused[1], (seed, refused)


@pytest.mark.parametrize("name", sorted(DRAWN_EXPECT))
def test_drawn_mutation(name):
    code, part = DRAWN_EXPECT[name]
    for seed in range(5):
        got, refused = _drawn(name, seed)
        assert got is None and refused[0] == code and part in refused[1], (seed, refused)


@pytest.mark.parametrize("name", sorted(WINDOW_EXPECT))
def test_window_mutation(name):
    code, part = WINDOW_EXPECT[name]
    for seed in SEEDS:
        got, refused = _window(name, seed)
        assert got is None and refused[0] == code and part in refused[1], (seed, refused)


@pytest.mark.parametrize("name", sorted(PERIOD_EXPECT))
def test_period_mutation(name):
    code, part = PERIOD_EXPECT[name]
    for seed in SEEDS:
        got, refused = _period(name, seed)
        assert got is None and refused[0] == code and part in refused[1], (seed, refused)


def _drawn(name: str, seed: int):
    d = cases.drawn_instance(random.Random(f"{name}/{seed}"), name)
    return Cr.drawn(*(d[k] for k in cases.DRAWN_KEYS))


def _outcome(name: str, seed: int):
    d = cases.outcome_instance(random.Random(f"{name}/{seed}"), name)
    return Cr.outcome(*(d[k] for k in cases.OUTCOME_KEYS))


def _window(name: str, seed: int):
    return Cr.window(*cases.window_args(cases.window_instance(random.Random(f"{name}/{seed}"), name)))


def _period(name: str, seed: int):
    d = cases.period_instance(random.Random(f"{name}/{seed}"), name)
    return Cr.period_credit(*(d[k] for k in cases.PERIOD_KEYS), rc.lms_verify)


def test_the_budget_is_the_record_k():
    """K = min(`RECORD_K`, N): `plan.RECORD_K`, the least K with (1 − ε)^K ≤ δ at ε = 0.1% and δ = 2⁻⁴⁰."""
    assert Cr.RECORD_K == plan.RECORD_K == plan.record_k(plan.Fraction(1, 1000), plan.Fraction(1, 2 ** 40))
    for seed in range(2):
        got, refused = _drawn("budget", seed)
        ss, _ = Cr.strata(cases.drawn_instance(random.Random(f"budget/{seed}"), "budget")["layout"])
        assert refused is None and sum(hi - lo for s in ss for lo, hi in s["units"]) > Cr.RECORD_K
        assert Cr.RECORD_K <= len(got) < Cr.RECORD_K + len(ss) and got == sorted(set(got))


@pytest.mark.parametrize("seed", SEEDS)
def test_the_drawn_units_are_the_draw_coins(seed):
    """`credit.outcome` derives the drawn units from the draw coin: each stratum's units are its own, its count the work
    law's k_s, and another coin, stream, registration or receipt draws another set; never a list handed in."""
    d = cases.drawn_instance(random.Random(f"honest/{seed}"))
    args = [d[k] for k in cases.DRAWN_KEYS]
    got, refused = Cr.drawn(*args)
    ss, _ = Cr.strata(d["layout"])
    Cr.law(ss)
    assert refused is None and got == sorted(set(got))
    for s in ss:
        assert sum(any(lo <= u < hi for lo, hi in s["units"]) for u in got) == s["k"]
    big = {**d, "layout": cases.layout(random.Random(f"wide/{seed}"), big=True)}
    base, _ = Cr.drawn(*(big[k] for k in cases.DRAWN_KEYS))
    for k, v in (("coin", "ab" * 64), ("stream", "another"), ("registration", "cd" * 64), ("receipt", {"receipts": "x"})):
        assert Cr.drawn(*({**big, k: v}[x] for x in cases.DRAWN_KEYS))[0] != base, k


def test_the_strata_are_the_compiled_programs():
    """`credit.strata` over a layout is `partition.strata` of the compiled `h1window.program` and its query: the same
    units in the Program's order, one stratum per K's template, each unit's work its tile's R·C·K MACs."""
    for linears, calls, m in ((h1.LINEARS, h1.CALLS, 8), (h1.LINEARS, h1.CALLS[::-1], 16),
                              ((("k_proj", 32, 16), ("q_proj", 32, 16)), (("k_proj", 0), ("q_proj", 0)), 8)):
        prog = h1.HW.program(linears, calls, m, h1.TILE, unit=h1.unit)
        q = h1.HW.query(prog)
        ids = dict(zip(sorted({k for _n, k, _c in linears}), q["params"]["templates"]))
        ss, refused = Cr.strata({"calls": [list(c) for c in calls], "linears": [list(x) for x in linears], "m": m,
                                 "tile": list(h1.TILE)})
        assert refused is None
        assert [(ids[int(s["template"][1:])], s["units"]) for s in ss] == h1.P.strata(prog, q)
        assert [s["work"] for s in ss] == [h1.R * h1.C * int(s["template"][1:]) for s in ss]
        assert all(s["floor"] == 1 for s in ss)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_window_is_credited_only_with_every_drawn_unit_accepted(seed):
    d = cases.window_instance(random.Random(f"honest/{seed}"))
    got, refused = Cr.window(*cases.window_args(d))
    assert refused is None and got == [i for i, row in enumerate(d["calls"]) if row[Cr.CREDIT]]


@pytest.mark.parametrize("seed", SEEDS)
def test_a_period_counts_only_closed_and_complete(seed):
    """RT4-INT-42: a closed, complete period earns each window's credit, in its close's order; with no close, a window
    incomplete, missing or out of order, or one of no period among it, it earns nothing."""
    d = cases.period_instance(random.Random(f"honest/{seed}"))
    got, refused = Cr.period_credit(*(d[k] for k in cases.PERIOD_KEYS), rc.lms_verify)
    assert refused is None and got == [[i for i, row in enumerate(w["calls"]) if row[Cr.CREDIT]] for w in d["windows"]]
    for name in ("no-close", "incomplete", "missing", "order", "foreign"):
        assert _period(name, seed)[0] is None, name


@pytest.mark.parametrize("seed", SEEDS)
def test_a_window_proving_another_circuit_refuses_the_period(seed):
    """A window whose proofs were verified against a circuit other than the one the period's open committed counts
    toward no period (`challenger.credited_session`), so its period earns nothing, though every window is otherwise
    credited; so does a period that committed no circuit."""
    d = cases.period_instance(random.Random(f"other-circuit/{seed}"), "other-circuit")
    for w in d["windows"]:
        assert Cr.window(d["key"], d["ledger"], *(w[k] for k in cases.WINDOW_ARGS[2:]), rc.lms_verify)[1] is None
    got, refused = Cr.period_credit(*(d[k] for k in cases.PERIOD_KEYS), rc.lms_verify)
    assert got is None and refused[0] == "circuit" and "is not" in refused[1] and "the one period" in refused[1]
    assert "doesn't count toward it, so the period earns nothing" in refused[1]
    got, refused = _period("no-circuit", seed)
    assert got is None and refused[0] == "circuit"


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
