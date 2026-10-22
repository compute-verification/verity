"""`credit.py`'s rules on a served window and its period (RT4-INT-32, -34 and -42; PRF-132): each mutation of
`pouw_credit_cases` is refused by the rule it aims at, its message naming the fix, and the honest windows and periods are
accepted. The drawn units are the draw coin's over the window's own layout, read from the descriptor of the circuit the
period committed (`credit.layout_of`), whose strata are the compiled `h1window.program`'s. The Lean (`PouwCheck.Credit`) is held to the same cases by `lean/check/tests/test_difftest.py`."""
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

from fractions import Fraction

import pytest

from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import plan
from verity.ml.circuits import codec

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
FEWER = "units are fewer than K = "
NO_K = "commits no draw K (draw_k"
DRAWN_EXPECT = {
    "coin-form": ("draw", Cr.COIN_FORM), "args-form": ("draw", Cr.DRAW_ARGS), "layout-form": ("layout", Cr.LAYOUT_FORM),
    "named-twice": ("layout", "is named twice"), "unknown-linear": ("layout", "which is no linear of the window"),
    "uncalled": ("layout", "is never called"), "slot": ("layout", "(h1window.slot)"),
    "kn": ("layout", "n whole blocks of the tile's"), "blocks": ("layout", "is read by some call of K"),
    "k-form": ("budget", Cr.K_FORM), "k-past-units": ("budget", FEWER),
}
OUTCOME_EXPECT = {
    "missing": ("missing", "Prove and verify every drawn unit"), "failed": ("failed", "refuses the whole window"),
    "twice": ("outcome", "a retry never reproves it"), "undrawn": ("outcome", NOT_DRAWN), "chosen": ("outcome", NOT_DRAWN),
    "proofs-form": ("outcome", Cr.PROOFS_FORM),
}
WINDOW_EXPECT = {
    "manifest-count": ("manifest", "each call of the window's Program is one manifest row"),
    "chosen": ("outcome", NOT_DRAWN), "failed": ("failed", "no partial credit"), "missing": ("missing", "no partial credit"),
    "descriptor-hex": ("layout", Cr.DESCRIPTOR_HEX), "descriptor-form": ("layout", Cr.DESCRIPTOR_FORM),
    "not-window": ("layout", Cr.NOT_WINDOW), "template": ("layout", "is not h1window's"),
    "tile-ks": ("layout", "h1window binds one template per K"), "body": ("layout", Cr.BODY),
    "layout-form": ("layout", Cr.LAYOUT_FORM), "record-layout": ("layout", "which is no linear of the window"),
    "alt-layout": ("layout", Cr.BODY),
    "credit-changed": ("draw", Cr.CHANGED), "credit-no-draw": ("draw", Cr.NO_DRAW),
    "credit-forged-ledger": ("coin", "the challenger's session is not trusted"),
    "k-form": ("budget", Cr.K_FORM), "k-past-units": ("budget", FEWER),
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
    "no-k": ("budget", NO_K), "k-form": ("budget", NO_K), "k-past-units": ("window", FEWER),
    "other-k": ("window", "so the period earns nothing"),
    "alt-layout": ("circuit", "prove the period's declared circuit"),
    "bad-circuit": ("window", "so the period earns nothing: layout: "),
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


def test_a_draw_at_the_record_k():
    """At a committed K = `plan.RECORD_K` (the least K with (1 − ε)^K ≤ δ at ε = 0.1% and δ = 2⁻⁴⁰) over a larger
    population, the work law draws K units, each stratum's k_s rounded up, so at most one more a stratum."""
    assert cases.PLAN_K == plan.RECORD_K == plan.record_k(plan.Fraction(1, 1000), plan.Fraction(1, 2 ** 40))
    for seed in range(2):
        got, refused = _drawn("budget", seed)
        ss, _ = Cr.strata(cases.drawn_instance(random.Random(f"budget/{seed}"), "budget")["layout"])
        assert refused is None and sum(hi - lo for s in ss for lo, hi in s["units"]) > plan.RECORD_K
        assert plan.RECORD_K <= len(got) < plan.RECORD_K + len(ss) and got == sorted(set(got))


@pytest.mark.parametrize("seed", SEEDS)
def test_the_drawn_units_are_the_draw_coins(seed):
    """`credit.outcome` derives the drawn units from the draw coin: each stratum's units are its own, its count the work
    law's k_s, and another coin, stream, registration or receipt draws another set; never a list handed in."""
    d = cases.drawn_instance(random.Random(f"honest/{seed}"))
    args = [d[k] for k in cases.DRAWN_KEYS]
    got, refused = Cr.drawn(*args)
    ss, _ = Cr.strata(d["layout"])
    assert Cr.law(ss, d["k"]) is None
    assert refused is None and got == sorted(set(got))
    for s in ss:
        assert sum(any(lo <= u < hi for lo, hi in s["units"]) for u in got) == s["k"]
    big = {**d, "layout": cases.layout(random.Random(f"wide/{seed}"), big=True), "k": 100}
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


def _compiled(linears, calls, m) -> bytes:
    """The descriptor bytes of the compiled `h1window.program` over the test tile, as the period's ``circuit`` hashes
    them (`codec.program_digest`'s)."""
    desc = codec.encode_program(h1.HW.program(linears, calls, m, h1.TILE, unit=h1.unit))
    return codec.canonical_json({k: v for k, v in desc.items() if k != "annotations"})


COMPILED = ((h1.LINEARS, h1.CALLS, 8), (h1.LINEARS, h1.CALLS[::-1], 16),
            ((("k_proj", 32, 16), ("q_proj", 32, 16)), (("k_proj", 0), ("q_proj", 0)), 8))


def test_the_layout_is_the_committed_circuits():
    """`credit.layout_of` reads the compiled `h1window.program`'s own arguments back from its descriptor, the bytes the
    period's ``circuit`` is the SHA-512 of, and their strata are `partition.strata` of that Program under its query."""
    for linears, calls, m in COMPILED:
        lay, refused = Cr.layout_of(_compiled(linears, calls, m))
        assert refused is None
        assert lay == {"calls": [list(c) for c in calls], "linears": [list(x) for x in linears], "m": m,
                       "tile": list(h1.TILE)}
        prog = h1.HW.program(linears, calls, m, h1.TILE, unit=h1.unit)
        q = h1.HW.query(prog)
        ids = dict(zip(sorted({k for _n, k, _c in linears}), q["params"]["templates"]))
        assert [(ids[int(s["template"][1:])], s["units"]) for s in Cr.strata(lay)[0]] == h1.P.strata(prog, q)


def test_an_alternative_layout_with_the_same_descriptor_refuses():
    """The layout is the committed circuit's, never the window's evidence: a window presenting a layout beside its
    descriptor is not a window's evidence; a compiled descriptor whose record names another layout (another slot,
    another n, the calls in another order) over the same body is refused (`credit.BODY`), so another layout can't bring
    other strata or another drawn set; and the descriptor of the other layout is another circuit than the period's."""
    linears, calls, m = COMPILED[0]
    raw = _compiled(linears, calls, m)
    lay, _ = Cr.layout_of(raw)
    for alt in ({**lay, "m": 2 * m}, {**lay, "linears": [[lay["linears"][0][0], lay["linears"][0][1],
                                                         lay["linears"][0][2] + h1.C],
                                                         *lay["linears"][1:]]},
                {**lay, "calls": lay["calls"][1:] + lay["calls"][:1]}):
        assert Cr.strata(alt)[1] is None and Cr.strata(alt)[0] != Cr.strata(lay)[0]
        desc = json.loads(raw)
        tiles = json.loads(dict(Cr._record(desc["program_fn"]))["TILES"])
        desc["program_fn"] = cases.record(tiles, alt)
        assert Cr.layout_of(codec.canonical_json(desc)) == (None, ("layout", Cr.BODY))
    for seed in SEEDS:
        d = cases.period_instance(random.Random(f"honest/{seed}"))
        d["windows"][0] = {**d["windows"][0], "layout": lay}
        assert Cr.period_credit(*(d[k] for k in cases.PERIOD_KEYS), rc.lms_verify) == (None, ("period", Cr.WINDOWS_FORM))
        for name, code in (("alt-layout", "circuit"), ("bad-circuit", "window")):
            got, refused = _period(name, seed)
            assert got is None and refused[0] == code, (name, refused)


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
    assert refused is None
    assert got["windows"] == [[i for i, row in enumerate(w["calls"]) if row[Cr.CREDIT]] for w in d["windows"]]
    for name in ("no-close", "incomplete", "missing", "order", "foreign"):
        assert _period(name, seed)[0] is None, name


@pytest.mark.parametrize("seed", SEEDS)
def test_a_window_proving_another_circuit_refuses_the_period(seed):
    """A window whose proofs were verified against a circuit other than the one the period's open committed counts
    toward no period (`challenger.credited_session`), so its period earns nothing, though every window is otherwise
    credited; so does a period that committed no circuit."""
    d = cases.period_instance(random.Random(f"other-circuit/{seed}"), "other-circuit")
    big_k, _ = Cr.committed_k(d["ledger"], d["name"])
    for w in d["windows"]:
        args = (w[k] for k in cases.WINDOW_ARGS[2:-1])
        assert Cr.window(d["key"], d["ledger"], *args, big_k, rc.lms_verify)[1] is None
    got, refused = Cr.period_credit(*(d[k] for k in cases.PERIOD_KEYS), rc.lms_verify)
    assert got is None and refused[0] == "circuit" and "is not" in refused[1] and "the one period" in refused[1]
    assert "doesn't count toward it, so the period earns nothing" in refused[1]
    got, refused = _period("no-circuit", seed)
    assert got is None and refused[0] == "circuit"


def _work(layout: dict) -> int:
    ss, _ = Cr.strata(layout)
    return sum(s["work"] * sum(hi - lo for lo, hi in s["units"]) for s in ss)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_reported_eps_is_window_samplings_at_the_committed_k(seed):
    """PoUW carries no ε target: the period's open commits K, and the credit reports the ε that K establishes at
    `DELTA` over each window's credited work C, `window.k_prime` inverted at (C, W, 1), W the window's work: K′ at that
    ε is at most K, and at the next grid point down more than K, and a window whose draw establishes no ε <= 1 reports
    none. At K = `plan.RECORD_K` and u = 1 it is at most 0.1%."""
    d = cases.period_instance(random.Random(f"honest/{seed}"))
    got, refused = Cr.period_credit(*(d[k] for k in cases.PERIOD_KEYS), rc.lms_verify)
    big_k, _ = Cr.committed_k(d["ledger"], d["name"])
    assert refused is None and got["k"] == big_k and got["delta"] == str(Cr.DELTA)
    grid = Fraction(1, 1 << Wd.EPS_BITS)
    for w, calls, eps in zip(d["windows"], got["windows"], got["eps"], strict=True):
        lay, _ = Cr.layout_of(bytes.fromhex(w["descriptor"]))
        (r, c), ks = lay["tile"], {name: (k, n) for name, k, n in lay["linears"]}
        units = [((lay["m"] // r) * (ks[lin][1] // c), r * c * ks[lin][0]) for lin, _blk in lay["calls"]]
        total = _work(lay)
        assert sum(s * u for s, u in units) == total
        credited = sum(units[i][0] * units[i][1] for i in calls)
        if not credited or Wd.k_prime(Fraction(1), Cr.DELTA, credited, total, 1) > big_k:
            assert eps is None
            continue
        eps = Fraction(eps)
        assert eps == Wd.eps_at(big_k, Cr.DELTA, credited, total, 1)
        assert Wd.k_prime(eps, Cr.DELTA, credited, total, 1) <= big_k
        assert eps == grid or Wd.k_prime(eps - grid, Cr.DELTA, credited, total, 1) > big_k
    at_record = Cr.established({"calls": [["w", 0]], "linears": [["w", 64, 1 << 20]], "m": 1, "tile": [1, 1]}, [0],
                               plan.RECORD_K)
    assert at_record <= Fraction(1, 1000)
    assert plan.record_k(at_record, Cr.DELTA) <= plan.RECORD_K < plan.record_k(at_record - grid, Cr.DELTA)


def test_the_reported_eps_is_of_the_credited_work():
    """ε is a share of the credited work, not of the window's: with 8 of 9 equal calls credited (u = 8/9) the K′ of
    ε = 0.1% at `DELTA` is 31,178, and its ε is at most 0.1%, where taking credited = W would report 8/9 of that, too
    optimistic by 9/8. A window credited nothing establishes no ε."""
    lay = {"calls": [["w", 0]] * 9, "linears": [["w", 4096, 1 << 20]], "m": 16, "tile": [16, 16]}
    size, work = Cr.call_work(lay)[0]
    total = 9 * size * work
    k = Wd.k_prime(Fraction(1, 1000), Cr.DELTA, 8 * size * work, total, 1)
    eps = Cr.established(lay, list(range(8)), k)
    assert k == 31_178 and Fraction(9999, 10_000_000) < eps <= Fraction(1, 1000)
    assert eps == Wd.eps_at(k, Cr.DELTA, 8 * size * work, total, 1)
    assert Fraction(9, 8) * Wd.eps_at(k, Cr.DELTA, total, total, 1) == pytest.approx(eps, rel=1e-5)
    assert Cr.established(lay, [], k) is None


def test_the_record_k_establishes_about_0_1125_percent_at_u_eight_ninths():
    """At K = `plan.RECORD_K` = 27,713, the K that establishes 0.1% of a window's work at `DELTA`, a window credited 8
    of its 9 equal calls reports about 0.1125%, 9/8 of it: the miss bound reads only the missed work εC/W, and
    `Properties.Pouw.Credit.EpsAtSound` holds at the ε `eps_at` answers, so its K′ is at most 27,713 and the grid
    point below needs more."""
    lay = {"calls": [["w", 0]] * 9, "linears": [["w", 4096, 1 << 20]], "m": 16, "tile": [16, 16]}
    size, work = Cr.call_work(lay)[0]
    total, credited = 9 * size * work, 8 * size * work
    eps = Cr.established(lay, list(range(8)), plan.RECORD_K)
    assert eps == Wd.eps_at(plan.RECORD_K, Cr.DELTA, credited, total, 1)
    assert Fraction(1_124, 1_000_000) < eps < Fraction(1_126, 1_000_000)
    assert Wd.k_prime(eps, Cr.DELTA, credited, total, 1) <= plan.RECORD_K
    assert Wd.k_prime(eps - Fraction(1, 1 << Wd.EPS_BITS), Cr.DELTA, credited, total, 1) > plan.RECORD_K
    assert Fraction(8, 9) * eps == pytest.approx(Wd.eps_at(plan.RECORD_K, Cr.DELTA, total, total, 1), rel=1e-5)


def _worst_miss(strata: list[tuple[int, int, int]], eps: Fraction, total: int) -> Fraction:
    """The largest probability that the work law's draw (each stratum (n_s, w_s, k_s) drawn k_s of its n_s units
    without replacement) misses a set holding at least a share `eps` of the work `total`, over two strata: for each
    count b_1 of the second stratum's units in the set, the fewest of the first's that reach the share (a set's miss
    only falls as it grows)."""
    (n0, w0, k0), (n1, w1, k1) = strata

    def miss(n: int, k: int, b: int) -> Fraction:
        return Fraction(math.comb(n - b, k), math.comb(n, k))

    worst = Fraction(0)
    for b1 in range(n1 + 1):
        b0 = max(0, -(-(eps * total - b1 * w1) // w0))
        if b0 <= n0:
            worst = max(worst, miss(n0, k0, b0) * miss(n1, k1, b1))
    return worst


def test_the_eps_holds_at_the_work_laws_cap():
    """At K = n the work law draws a stratum of little work only in part, so the cap is K >= W, not the unit count:
    1,000 units of work 1 and 1,000 of work 1,000, all credited, drawn at K = n = 2,000. The reported ε, `eps_at` at
    (C, W, 1), about 1.38%, holds: every set holding that share of the work is missed with probability at most δ. The
    unit-count form `eps_at(K, δ, C, n, w_max)` would report 2^-32, yet a set of 500 work-1 units, 0.05% of the work,
    is missed with probability about 1/4."""
    lay = {"calls": [["a", 0], ["b", 0]], "linears": [["a", 1, 1000], ["b", 1000, 1000]], "m": 1, "tile": [1, 1]}
    ss, why = Cr.strata(lay)
    assert why is None and Cr.law(ss, 2000) is None
    total = sum(s["work"] * (s["units"][0][1] - s["units"][0][0]) for s in ss)
    drawn = [(s["units"][0][1] - s["units"][0][0], s["work"], s["k"]) for s in ss]
    assert total == 1_001_000 and drawn == [(1000, 1, 2), (1000, 1000, 1000)]
    eps = Cr.established(lay, [0, 1], 2000)
    assert eps == Wd.eps_at(2000, Cr.DELTA, total, total, 1) and Fraction(137, 10_000) < eps < Fraction(138, 10_000)
    assert _worst_miss(drawn, eps, total) <= Cr.DELTA
    by_units = Wd.eps_at(2000, Cr.DELTA, total, 2000, 1000)
    assert by_units == Fraction(1, 1 << Wd.EPS_BITS)
    assert _worst_miss(drawn, Fraction(500, total), total) > Fraction(1, 5) > Cr.DELTA


@pytest.mark.parametrize("light, heavy", [((1000, 1), (1000, 1000)), ((400, 1), (100, 100)), ((600, 2), (200, 300)),
                                          ((300, 1), (300, 64)), ((500, 1), (50, 1000))])
def test_the_unit_count_form_is_refuted_at_its_cap(light, heavy):
    """The ε of `eps_at(K, δ, C, n, w_max)`, the unit count n in place of the work W, is refuted wherever the work law
    draws a stratum of light work only in part at K = n: there it reports 2^-32, and a set of light units holding that
    share of the work is missed with probability above 9/10. At the same draw the work form (C, W, 1) holds, every set
    of its share missed with probability at most δ (`Properties.Pouw.Credit.WorkDrawMisses`)."""
    (n_a, w_a), (n_b, w_b) = light, heavy
    lay = {"calls": [["a", 0], ["b", 0]], "linears": [["a", w_a, n_a], ["b", w_b, n_b]], "m": 1, "tile": [1, 1]}
    n = n_a + n_b
    ss, why = Cr.strata(lay)
    assert why is None and Cr.law(ss, n) is None
    drawn = [(s["units"][0][1] - s["units"][0][0], s["work"], s["k"]) for s in ss]
    total = sum(size * w for size, w, _k in drawn)
    assert drawn[0][0] == n_a and drawn[0][2] < n_a and drawn[1][2] == n_b
    by_units = Wd.eps_at(n, Cr.DELTA, total, n, w_b)
    assert by_units == Fraction(1, 1 << Wd.EPS_BITS) and _worst_miss(drawn, by_units, total) > Fraction(9, 10)
    eps = Cr.established(lay, [0, 1], n)
    assert eps == Wd.eps_at(n, Cr.DELTA, total, total, 1) and _worst_miss(drawn, eps, total) <= Cr.DELTA


@pytest.mark.parametrize("seed", SEEDS)
def test_a_period_open_with_no_k_refuses(seed):
    """K is the proof service's selection, committed in the period-open before any draw: a period whose open commits
    none, or a malformed one, earns nothing."""
    for name in ("no-k", "k-form"):
        got, refused = _period(name, seed)
        assert got is None and refused[0] == "budget" and NO_K in refused[1] and "before any draw" in refused[1]


@pytest.mark.parametrize("seed", SEEDS)
def test_a_drawn_count_other_than_k_refuses(seed):
    """A window is drawn at its period's committed K: one with fewer units than K (its draw would be sized at N, not
    K) is refused, and so is one whose proofs are of a draw at another K."""
    got, refused = _window("k-past-units", seed)
    assert got is None and refused[0] == "budget" and FEWER in refused[1]
    for name in ("k-past-units", "other-k"):
        got, refused = _period(name, seed)
        assert got is None and refused[0] == "window" and "so the period earns nothing" in refused[1], (name, refused)
    assert FEWER in _period("k-past-units", seed)[1][1]


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
