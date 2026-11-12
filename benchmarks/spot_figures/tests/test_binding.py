"""The two-stage first-hit finder's bound (`binding`): its arithmetic is exact in every form, each term per sampled
replay block is the one stated at each design the options are compared at, with every run at 2^80 calls of SHA-512,
in calls and converted to compressions, and every line the command prints says what it covers."""
import json
from dataclasses import replace
from fractions import Fraction

import binding as b
import cards
import pytest

#: the least and the most accepted reruns a wait of a one-table session, at 2^10 and 2^23 level-0 symbols
SMALL, TABLE = 1507, 12343986
#: a session of the 200 tables the code admits
TABLES = 200
#: the most sessions holding one gate the designs are compared at
SESSIONS = 167
#: GLM-5.3's K at one proof block a stratum, the design with no limit on a stratum's sample
UNCAPPED = 27191776
#: K at the two ends of the designs the cost rows price, 2^40.23 and 2^43.23 rounded up
LEAST, MOST = b.natural("2^40.23"), b.natural("2^43.23")
#: the compressions of 2^80 calls, each as long as a 64 KiB row
ROWS = b.BUDGET * b.compressions(65536)
#: the priced design by position: its sample's mean gates, 2^37.69 rounded up, and its proof blocks
MEAN, RUN = b.natural("2^37.69"), 5523986
#: the sessions of that sample at one table a session, counted two ways
MEMBERS = (2893, 15694)
#: the runs of a year, one every 600 s on one machine
YEAR = 52560
#: the priced design per sampled replay block with every block's input bits counted, each an upper bound: its
#: gates, its classes and its sessions
GATES, CLASSES, MOST_SESSIONS = b.natural("2^41.81"), b.natural("2^26.24"), 15694
#: the blocks that sample checks, one session each, and the most gates one of its blocks needs
CHECKED, LARGEST = 10046500, 632128


def tenths(p: Fraction) -> Fraction:
    """−log₂ p rounded to a tenth of a bit, half up."""
    return Fraction((b.centibits(p) + 5) // 10, 10)


def at(K: int, k: int, d: int = 1, T: int = b.BUDGET) -> Fraction:
    return b.bound(b.Design(K=K, k=k, d=d, T=T))


def by_position(tables: int, K: int = MEAN, sessions: int = RUN, T: int = b.BUDGET) -> Fraction:
    """The first-hit term at K_s = K_m = K and no calls of the run's own."""
    return b.terms(b.Design(K=K, k=b.queries(tables * 2 ** 23), sessions=sessions, T=T))["first hit"]


def members(sessions: int, **more) -> b.Design:
    return b.Design(K=MEAN, k=TABLE, sessions=sessions, **more)


def test_by_gate_the_bound_is_its_three_terms_exactly():
    d = b.Design(K=3, k=1, d=2, T=5, open_calls=7, p_ref=Fraction(1, 8), opens=4, M=3)
    assert (d.form, d.reruns, d.calls) == ("gate", 211, 2 * 3 * (5 * (1 + 2 * 211) + 2 * 7))
    assert b.terms(d) == {"collision": Fraction(2 * 3 * 2129, 2 ** 256), "refused": Fraction(1, 2), "cap": Fraction(1)}
    assert b.bound(d) == Fraction(2 * 3 * 2129, 2 ** 256) + Fraction(3, 2)
    assert all(isinstance(t, Fraction) for t in b.terms(b.Design(K=1, k=1)).values())
    assert b.bound(b.Design(K=1, k=1)) == Fraction(2 * 212 * b.BUDGET, 2 ** 256)
    assert set(b.premises(d)) == set(b.terms(d))


def test_by_position_the_first_hit_term_is_the_finders_expected_calls_exactly():
    d = b.Design(K=3, k=1, T=5, sessions=2, Km=7, units=11, open_calls=13)
    assert (d.form, d.own) == ("position", 11 * 13)
    assert d.calls == 4 * (5 * (1 + 211 * 2 + 3 + 211 * 7) + (1 + 3) * 11 * 13)
    assert b.terms(d)["first hit"] == Fraction(d.calls, 2 ** 256)
    plain = b.Design(K=3, k=1, T=5, sessions=2)
    assert plain.calls == 4 * 5 * (1 + 2 * 211 + (1 + 211) * 3) == b.Design(K=3, k=1, T=5, sessions=2, Km=3).calls
    assert b.Design(K=3, k=1, T=10, sessions=2).calls == 2 * plain.calls
    assert plain.calls < b.Design(K=3, k=1, T=5, sessions=2, open_calls=1).calls
    assert plain.calls < b.Design(K=4, k=1, T=5, sessions=2).calls < b.Design(K=4, k=1, T=5, sessions=3).calls


def test_by_position_the_bound_is_its_four_terms_each_under_its_premise():
    d = b.Design(K=3, k=1, T=4, sessions=2, rounds=1, gamma=2, kappa=Fraction(1, 2 ** 9))
    strict, expected = b.terms(d), b.terms(d, "ExpectedCR")
    assert list(strict) == ["sessions", "rewinding", "conflict", "first hit"] == list(b.premises(d))
    assert strict["sessions"] == Fraction(1, 2 ** 9) and strict["conflict"] == Fraction(2 * 4 * (1 + 2 * 211), 2 ** 256)
    # r·γ·(2T)²/2^513 = 8²/2^512 and r·γ·2T/2^256 = 16/2^256, both squares
    assert strict["rewinding"] == Fraction(8, 2 ** 256) and expected["rewinding"] == Fraction(4, 2 ** 128)
    assert {name: p for name, p in expected.items() if name != "rewinding"} == {
        name: p for name, p in strict.items() if name != "rewinding"}
    assert b.bound(d) == sum(strict.values()) < b.bound(d, "ExpectedCR") == sum(expected.values())
    assert "StrictBirthday" in b.premises(d)["rewinding"] and "ExpectedCR" in b.premises(d, "ExpectedCR")["rewinding"]
    unstated = b.Design(K=3, k=1, T=4, sessions=2)
    assert (unstated.rewinds, unstated.gamma, b.terms(unstated)["sessions"]) == (4 + 4402 * 2, 2 ** 51, 0)
    assert "not given" in b.premises(unstated)["sessions"]
    with pytest.raises(ValueError, match="premise at B"):
        b.terms(d, "Birthday")


def test_the_root_is_at_or_above_and_as_near_as_its_step():
    assert b.root_up(Fraction(1, 4)) == Fraction(1, 2) and b.root_up(Fraction(0)) == 0
    r = b.root_up(Fraction(2))
    assert (r - Fraction(1, b.HASH)) ** 2 < 2 <= r ** 2
    r = b.root_up(Fraction(1, 3))
    assert (r - Fraction(1, 3 * b.HASH)) ** 2 < Fraction(1, 3) <= r ** 2
    with pytest.raises(ValueError, match="not negative"):
        b.root_up(Fraction(-1))


def test_the_bound_is_linear_in_the_gates_and_the_budget_and_rises_with_the_sessions():
    one = at(UNCAPPED, SMALL)
    assert at(2 * UNCAPPED, SMALL) == 2 * one == at(UNCAPPED, SMALL, T=2 * b.BUDGET)
    assert one < at(UNCAPPED, SMALL, d=2) < 2 * one
    assert at(UNCAPPED, TABLE) / one == Fraction(1 + 1 + 210 * TABLE, 1 + 1 + 210 * SMALL)


@pytest.mark.parametrize("design, bad", [
    (dict(K=0, k=1), "K"), (dict(K=1, k=0), "k"), (dict(K=1, k=1, d=0), "d"), (dict(K=1, k=1, T=0), "T"),
    (dict(K=1.5, k=1), "K"), (dict(K=1, k=1, open_calls=-1), "open_calls"),
    (dict(K=1, k=1, p_ref=Fraction(3, 2)), "p_ref"), (dict(K=1, k=1, opens=5), "M"),
    (dict(K=1, k=1, sessions=0), "sessions"), (dict(K=1, k=1, sessions=1, d=2), "by position the bound reads no d"),
    (dict(K=1, k=1, sessions=1, opens=1, M=1), "by position the bound reads no opens"),
    (dict(K=1, k=1, sessions=1, p_ref=Fraction(1, 2)), "by position the bound reads no p_ref"),
    (dict(K=1, k=1, Km=2), "by gate the bound reads no Km"), (dict(K=1, k=1, units=2), "by gate the bound reads no units"),
    (dict(K=1, k=1, rounds=5, kappa=Fraction(1, 2)), "by gate the bound reads no rounds, kappa"),
    (dict(K=1, k=1, gamma=2), "by gate the bound reads no gamma"), (dict(K=1, k=1, sessions=1, Km=0), "Km"),
    (dict(K=1, k=1, by_class=True), "by gate the bound reads no by_class"),
    (dict(K=1, k=1, sessions=1, units=0), "units"), (dict(K=1, k=1, sessions=1, rounds=0), "rounds"),
    (dict(K=1, k=1, sessions=1, gamma=0), "gamma"), (dict(K=1, k=1, sessions=1, kappa=2), "kappa")])
def test_a_design_outside_the_bounds_domain_is_refused_by_name(design, bad):
    with pytest.raises(ValueError, match=bad):
        b.Design(**design)


def test_queries_is_the_ceiling_at_every_session_size():
    table = {10: 1507, 11: 3014, 12: 6028, 13: 12055, 14: 24110, 15: 48219, 18: 385750, 19: 771500, 20: 1542999,
             21: 3085997, 22: 6171993, 23: 12343986}
    assert {d: b.queries(2 ** d) for d in table} == table
    assert (b.queries(2 ** 10), b.queries(2 ** 23)) == (SMALL, TABLE)
    assert b.queries(TABLES * 2 ** 23) == 2468797139 <= TABLES * TABLE
    assert Fraction("2.7182818283") < b.E_LOW < b.E_HIGH < Fraction("2.7182818286")
    assert b.E_HIGH - b.E_LOW < Fraction(1, 10 ** 40)
    with pytest.raises(ValueError, match="at least one symbol"):
        b.queries(0)


def test_compressions_counts_the_padding():
    assert [b.compressions(n) for n in (0, 111, 112, 128, 239, 240, 65536)] == [1, 1, 2, 2, 2, 3, 513]


def test_centibits_is_the_floor_exactly():
    assert b.centibits(Fraction(1, 2 ** 100)) == 10000
    assert b.centibits(Fraction(1, 2 ** 100) + Fraction(1, 2 ** 400)) == 9999
    assert b.centibits(Fraction(1, 2 ** 100) - Fraction(1, 2 ** 400)) == 10000
    assert b.centibits(Fraction(1, 3)) == 158 and b.centibits(Fraction(3)) == -159 and b.centibits(Fraction(1)) == 0
    assert b.centibits(Fraction(0)) is None
    assert [b.bits(p) for p in (Fraction(1, 2 ** 100), Fraction(1, 3), Fraction(3), Fraction(0))] == [
        "2^-100.00", "2^-1.58", "2^1.59", "0"]
    assert [b.log2(n) for n in (1, 3, 2 ** 80, 513 * 2 ** 80)] == ["0.00", "1.58", "80.00", "89.00"]
    with pytest.raises(ValueError, match="not negative"):
        b.centibits(Fraction(-1, 2))


def test_a_count_as_written():
    assert b.natural("3*2^80") == 3 * 2 ** 80 and b.natural("12,343,986") == TABLE and b.natural("2**10") == 1024
    assert (LEAST - 1) ** 100 < 2 ** 4023 <= LEAST ** 100 and (MOST - 1) ** 100 < 2 ** 4323 <= MOST ** 100
    assert b.probability("1/2^300") == Fraction(1, 2 ** 300) and b.probability("0") == 0
    with pytest.raises(ValueError):
        b.natural("0")


def test_the_cards_gates_are_its_stratas_bits():
    card = {"value": {"strata": [{"block": "A", "in_bits": 5, "out_bits": 16},
                                 {"block": "B", "in_bits": 0, "out_bits": 3.0}]}}
    assert b.card_gates(card) == 24
    card["value"]["strata"].append({"block": "C", "out_bits": 3})
    with pytest.raises(ValueError, match="C: the workload card gives no whole in_bits"):
        b.card_gates(card)


def test_a_monitoring_period_sums_its_runs_each_of_its_first_stages_replay_blocks():
    d = b.Design(K=UNCAPPED, k=SMALL)
    assert {name: n for name, (n, _) in b.periods(d).items()} == {"week": 1008, "month": 4320, "year": 52560}
    assert all(p == n * b.bound(d) for n, p in b.periods(d).values())
    assert all(p == 6 * n * b.bound(d) for n, p in b.periods(d, first_stage=6).values())
    assert b.periods(d, machines=12500)["year"][0] == 12500 * 52560
    two = members(MEMBERS[0])
    assert b.periods(two, first_stage=4, at_B="ExpectedCR")["week"] == (
        1008, 1008 * b.bound(b.at_run(two, 4), "ExpectedCR"))
    with pytest.raises(ValueError, match="at least one replay block"):
        b.periods(d, first_stage=0)


def test_a_run_has_its_first_stages_replay_blocks_sessions_and_gates():
    d = b.Design(K=3, k=1, T=5, sessions=2, Km=7, units=11, open_calls=13)
    run = b.at_run(d, 6)
    assert (run.K, run.Km, run.sessions, run.T, run.own, run.k) == (18, 42, 12, 5, 143, 1) and b.at_run(d, 1) == d
    assert run.rewinds == 4 + 4402 * 12 and b.at_run(b.Design(K=3, k=1, sessions=2, rounds=9), 6).rewinds == 9
    gate = b.Design(K=3, k=1, d=2, p_ref=Fraction(1, 8), opens=4, M=3)
    assert b.at_run(gate, 6) == b.Design(K=18, k=1, d=2, p_ref=Fraction(1, 8), opens=4, M=3)
    assert b.terms(b.at_run(gate, 6))["collision"] == 6 * b.terms(gate)["collision"]
    one, six = members(MEMBERS[1]), b.at_run(members(MEMBERS[1]), 6)
    assert b.terms(one)["first hit"] < b.terms(six)["first hit"] < 6 * b.terms(one)["first hit"]
    assert b.terms(one)["rewinding"] < b.terms(six)["rewinding"] < 6 * b.terms(one)["rewinding"]


def test_the_waits_term_at_each_limit():
    assert b.CAPS == (128, 192) and [b.bits(b.waits(cap)) for cap in b.CAPS] == ["2^-125.99", "2^-183.64"]
    e_128 = 1 / b.E_LOW ** 128
    exact = [(1 + b.INNER) * (Fraction(2, 2 ** cap) + e_128) for cap in b.CAPS]
    assert all(0 <= b.waits(cap) - x < Fraction(1, 2 ** 511) for cap, x in zip(b.CAPS, exact))
    assert b.INNER ** 1024 >= 2 and b.bits(e_128) == "2^-184.66"
    assert b.waits(192, 3 * 2 ** 10, 3 * 2 ** 10) < Fraction(1, 2 ** 189) < b.waits(192)
    assert b.waits(192, 1031, 1031) == b.waits(192) and b.waits(192, 2 ** 10, 2 ** 11) < b.waits(192)
    with pytest.raises(ValueError, match="positive k"):
        b.waits(192, 0)
    assert (b.runs(600), b.runs(601), b.runs(0), b.runs(1200, 600, 3)) == (1, 2, 0, 6)


# The designs the options are compared at, per sampled replay block, a run at 2^80 compressions.

def test_no_limit_on_a_stratums_sample_at_the_smallest_session():
    assert tenths(at(UNCAPPED, SMALL)) == Fraction("132.0")
    assert b.bits(at(UNCAPPED, SMALL)) == "2^-132.03"


def test_the_priced_designs_at_the_smallest_session():
    assert (tenths(at(LEAST, SMALL)), tenths(at(MOST, SMALL))) == (Fraction("116.5"), Fraction("113.5"))


def test_the_priced_designs_at_one_table_a_session():
    cells = (tenths(at(LEAST, TABLE, SESSIONS)), tenths(at(MOST, TABLE, SESSIONS)))
    assert cells == (Fraction("96.1"), Fraction("93.1"))


def test_the_priced_designs_at_the_tables_the_code_admits():
    k = b.queries(TABLES * 2 ** 23)
    assert (tenths(at(LEAST, k, SESSIONS)), tenths(at(MOST, k, SESSIONS))) == (Fraction("88.5"), Fraction("85.5"))
    assert (tenths(at(LEAST, TABLES * TABLE, SESSIONS)), tenths(at(MOST, TABLES * TABLE, SESSIONS))) == (
        Fraction("88.5"), Fraction("85.5"))


def test_compressions_at_calls_of_64_kib_cost_nine_bits():
    assert ROWS == 513 * 2 ** 80 == b.converted(b.Design(K=1, k=1)).T
    k = b.queries(TABLES * 2 ** 23)
    cells = [(UNCAPPED, SMALL, 1, "123.0"), (LEAST, SMALL, 1, "107.5"), (MOST, SMALL, 1, "104.5"),
             (LEAST, TABLE, SESSIONS, "87.1"), (MOST, TABLE, SESSIONS, "84.1"),
             (LEAST, k, SESSIONS, "79.5"), (MOST, k, SESSIONS, "76.5")]
    assert [str(tenths(at(K, q, d, ROWS))) for K, q, d, _ in cells] == [str(Fraction(c)) for *_, c in cells]
    assert all(at(K, q, d, ROWS) == 513 * at(K, q, d) for K, q, d, _ in cells)


def test_a_year_of_runs_costs_15_7_bits():
    year = b.runs(365 * 86400)
    assert year == 52560 and tenths(Fraction(1, year)) == Fraction("15.7")
    cells = [(UNCAPPED, SMALL, 1, "116.3"), (LEAST, SMALL, 1, "100.8"), (MOST, SMALL, 1, "97.8"),
             (LEAST, TABLE, SESSIONS, "80.4"), (MOST, TABLE, SESSIONS, "77.4")]
    for K, q, d, cell in cells:
        (n, p), one = b.periods(b.Design(K=K, k=q, d=d))["year"], at(K, q, d)
        assert (n, p) == (year, year * one) and tenths(p) == Fraction(cell)


def test_the_served_models_card_keeps_100_bits_at_the_smallest_session_over_a_year():
    K = b.card_gates(cards.load("workload", "GLM-5.3"))
    out = b.report(b.Design(K=K, k=b.queries(2 ** 10)))
    assert [c["at_B"] for c in out["block"]] == [None] and out["columns"] == ["gates, calls", "gates, compressions"]
    assert all(c["meets"] for span in ("block", "run") for c in out[span][0]["columns"].values())
    assert all(c["meets"] for cells in out["periods"].values() for row in cells for c in row["columns"].values())
    year = b.report(b.Design(K=K, k=TABLE, d=SESSIONS))["periods"]["year"][0]
    assert not year["columns"]["gates, calls"]["meets"]


def test_by_position_the_four_terms_at_one_table_a_session():
    """The sample's sessions counted two ways, each term per sampled replay block and the sums per run."""
    assert members(MEMBERS[0]).reruns == 2592237061 and [members(n).rewinds for n in MEMBERS] == [12734990, 69084992]
    cells = {MEMBERS[0]: ("2^-50.19", "2^-138.19", "2^-132.23", "2^-105.03"),
             MEMBERS[1]: ("2^-48.97", "2^-136.97", "2^-129.79", "2^-105.03")}
    for sessions, (expected, strict, conflict, first) in cells.items():
        d = members(sessions)
        got = b.terms(d)
        assert b.bits(b.terms(d, "ExpectedCR")["rewinding"]) == expected
        assert [b.bits(got[name]) for name in ("rewinding", "conflict", "first hit")] == [strict, conflict, first]
        assert got["sessions"] == 0 and b.bits(b.bound(d)) == "2^-105.03"
        assert b.bits(b.bound(d, "ExpectedCR")) == expected
        assert [b.bits(b.bound(b.at_run(d, n))) for n in (4, 6)] == ["2^-103.03", "2^-102.45"]
        assert b.bits(b.periods(d, first_stage=6)["year"][1]) == "2^-86.77"
        assert b.terms(members(sessions, T=ROWS))["first hit"] == 513 * got["first hit"]
    assert b.bits(b.bound(members(MEMBERS[0], kappa=Fraction(1, 2 ** 203)))) == "2^-105.03"


def test_by_position_the_runs_own_opening_checks_are_counted():
    none = members(MEMBERS[1])
    for own in (2 ** 60, 2 ** 80):
        d = members(MEMBERS[1], units=2 ** 20, open_calls=own // 2 ** 20)
        assert d.own == own and d.calls - none.calls == 4 * (1 + MEAN) * own
        assert b.bits(b.terms(d)["first hit"]) == "2^-105.03"
    assert b.bits(Fraction(4 * (1 + MEAN) * 2 ** 80, 2 ** 256)) == "2^-136.30"


def test_in_compressions_each_call_is_taken_as_long_as_the_longest():
    d = members(MEMBERS[1], units=3, open_calls=5)
    long = b.converted(d)
    assert (long.T, long.open_calls, long.own) == (513 * d.T, 513 * 5, 513 * 15)
    assert b.converted(d, 111) == d and b.converted(d, 112).T == 2 * d.T
    calls, each = b.terms(d), b.terms(long)
    assert each["first hit"] == 513 * calls["first hit"] and each["sessions"] == calls["sessions"]
    assert [b.bits(each[name]) for name in ("rewinding", "conflict", "first hit")] == [
        "2^-127.97", "2^-120.78", "2^-96.03"]
    assert b.bits(b.terms(long, "ExpectedCR")["rewinding"]) == "2^-44.47"
    gate = b.Design(K=LEAST, k=TABLE, d=SESSIONS, open_calls=7)
    assert b.bound(b.converted(gate)) == 513 * b.bound(gate)


def test_by_classes_is_by_position_at_the_classes_one_reveal_opens_whole():
    d = b.by_classes(b.Design(K=GATES, k=TABLE, sessions=MOST_SESSIONS, Km=2 * GATES), CLASSES)
    assert (d.form, d.K, d.Km, d.sessions) == ("classes", CLASSES, None, MOST_SESSIONS)
    assert b.terms(d) == b.terms(b.Design(K=CLASSES, k=TABLE, sessions=MOST_SESSIONS))
    assert "by classes" in b.premises(d)["first hit"] and "by classes" not in b.premises(members(RUN))["first hit"]
    assert b.by_classes(members(RUN), CLASSES, 2 * CLASSES).Km == 2 * CLASSES


def test_the_priced_design_by_gates_and_by_classes_in_calls_and_in_compressions():
    """The first-hit term at the priced design's upper bounds, per sampled replay block and per run at a padded
    first stage of 4 and of 6 replay blocks: by gates it is above 2^-100 a run, by classes below it in calls and in
    compressions."""
    gates = b.Design(K=GATES, k=TABLE, sessions=MOST_SESSIONS)
    classes = b.by_classes(gates, CLASSES)
    cells = [(gates, ("2^-100.91", "2^-98.91", "2^-98.33"), (True, False, False)),
             (b.converted(gates), ("2^-91.91", "2^-89.91", "2^-89.33"), (False, False, False)),
             (classes, ("2^-116.48", "2^-114.48", "2^-113.90"), (True, True, True)),
             (b.converted(classes), ("2^-107.48", "2^-105.48", "2^-104.90"), (True, True, True))]
    goal = Fraction(1, 2 ** 100)
    for counted, cell, meets in cells:
        got = [b.terms(b.at_run(counted, n))["first hit"] for n in (1, 4, 6)]
        assert [b.bits(p) for p in got] == list(cell) and [p <= goal for p in got] == list(meets)
    width = b.Design(K=b.natural("2^39.15"), k=TABLE, sessions=MOST_SESSIONS)
    by_width = [b.terms(b.at_run(d, n))["first hit"] for d in (width, b.by_classes(width, b.natural("2^23.83")))
                for n in (4, 6)]
    assert [b.bits(p) for p in by_width] == ["2^-101.57", "2^-100.99", "2^-116.89", "2^-116.31"]


def test_by_position_the_terms_at_the_bounds_of_a_one_table_session():
    """At 2^24 sessions, 2^42 gates counted either way and 2^80 calls of the run's own, the first-hit term is at
    most 2^-100, and the rewinding term under `StrictBirthday` at B at most 2^-131."""
    d = b.Design(K=2 ** 42, k=TABLE, sessions=2 ** 24, open_calls=2 ** 80)
    got = b.terms(d)
    assert (d.reruns, d.rewinds) == (2592237061, 4 + 4402 * 2 ** 24)
    assert got["first hit"] <= Fraction(1, 2 ** 100) and b.bits(got["first hit"]) == "2^-100.72"
    assert got["rewinding"] <= Fraction(1, 2 ** 131) and b.bits(got["rewinding"]) == "2^-131.94"
    assert b.bits(b.terms(d, "ExpectedCR")["rewinding"]) == "2^-43.94"


def test_by_position_the_priced_design_at_one_32_and_200_tables_a_session():
    """The first-hit term's cell under the three readings: calls, compressions at calls of 64 KiB, a year of
    runs."""
    cells = {1: ("105.0", "96.0", "89.4"), 32: ("100.0", "91.0", "84.4"), TABLES: ("97.4", "88.4", "81.7")}
    for tables, (block, rows, year) in cells.items():
        one = by_position(tables)
        assert by_position(tables, T=ROWS) == 513 * one
        assert [tenths(p) for p in (one, 513 * one, YEAR * one)] == [Fraction(c) for c in (block, rows, year)]
    goal = Fraction(1, 2 ** 100)
    assert by_position(32) <= goal < by_position(33)
    assert [b.bits(by_position(n)) for n in (1, 32, TABLES)] == ["2^-105.03", "2^-100.03", "2^-97.39"]
    assert [b.bits(by_position(1, sessions=n)) for n in (1, *MEMBERS, RUN)] == ["2^-105.03"] * 4


def test_by_position_the_other_designs_cells():
    """The designs beside the priced one: their mean gates and sessions, at one table a session and at 200."""
    cells = [("2^38.69", 11047962, "104.0", "96.4"), ("2^41.82", 6389664, "100.9", "93.3"),
             ("2^34.69", 690498, "108.0", "100.4"), ("2^33.69", 345249, "109.0", "101.4"),
             ("2^21.61", 15, "121.1", "113.5")]
    for mean, sessions, one, most in cells:
        K = b.natural(mean)
        assert [tenths(by_position(n, K, sessions)) for n in (1, TABLES)] == [Fraction(one), Fraction(most)]
    least = b.natural("2^21.61")
    assert tenths(YEAR * by_position(1, least, 15)) == Fraction("105.4")
    assert tenths(YEAR * by_position(TABLES, least, 15)) == Fraction("97.8")


def test_by_position_the_most_mean_gates_32_tables_leave():
    """A session of 32 tables has k = 395007543, and the sample may then need 227711235758 gates in the mean and
    no more."""
    k = b.queries(32 * 2 ** 23)
    most = b.most_gates(b.Design(K=1, k=k, sessions=RUN), 100)
    goal = Fraction(1, 2 ** 100)
    assert (k, most) == (395007543, 227711235758) and MEAN <= most
    assert by_position(32, most) <= goal < by_position(32, most + 1)
    assert b.most_gates(b.Design(K=1, k=k, sessions=RUN), 100, runs=YEAR) == 0
    assert b.log2(b.most_gates(b.Design(K=1, k=TABLE, sessions=RUN), 100, runs=YEAR)) == "26.98"
    six = b.most_gates(b.Design(K=1, k=TABLE, sessions=MOST_SESSIONS), 100, first_stage=6)
    assert b.terms(b.at_run(b.Design(K=six, k=TABLE, sessions=MOST_SESSIONS), 6))["first hit"] <= goal < b.terms(
        b.at_run(b.Design(K=six + 1, k=TABLE, sessions=MOST_SESSIONS), 6))["first hit"] and b.log2(six) == "40.14"
    assert b.most_gates(b.Design(K=5, k=k, sessions=RUN, Km=9), 100) == most
    own = b.most_gates(b.Design(K=1, k=TABLE, sessions=RUN, open_calls=2 ** 80), 100)
    at_own = b.Design(K=own, k=TABLE, sessions=RUN, open_calls=2 ** 80)
    assert b.terms(at_own)["first hit"] <= goal < b.terms(b.Design(K=own + 1, k=TABLE, sessions=RUN,
                                                                   open_calls=2 ** 80))["first hit"]


def test_the_most_gates_a_target_leaves():
    cells = [(SMALL, 1, "56.72"), (SMALL, SESSIONS, "49.34"), (TABLE, 1, "43.72"), (TABLE, SESSIONS, "36.34")]
    for k, d, cell in cells:
        most = b.most_gates(b.Design(K=1, k=k, d=d), 100)
        goal = Fraction(1, 2 ** 100)
        assert at(most, k, d) <= goal < at(most + 1, k, d) and b.log2(most) == cell
    d = b.Design(K=1, k=SMALL)
    assert b.most_gates(d, 100, runs=52560) == b.most_gates(d, 100) // 52560
    assert b.most_gates(b.Design(K=1, k=1, opens=1, M=0), 100) == 0


def covered(lines: list[str]) -> bool:
    """Every line says what it covers: a sampled replay block, or a run at a padded first stage."""
    return all(b.BLOCK_UNIT in line or "per run at a padded first stage of K0 = " in line for line in lines)


def test_the_command_prints_by_gate_the_design_its_terms_and_each_period(capsys):
    b.main(["--card", "GLM-5.3", "--symbols", "2^10"])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == ("by gate, per sampled replay block: K = 27191776 gates (2^24.69), d = 1, openCalls = 0, "
                      "T = 2^80.00 calls of SHA-512 in every run, the prover's and the verifier's, k = 1507, "
                      "e_s = 316471")
    assert out[1] == ("each figure per sampled replay block or per run at a padded first stage of K0 = 1 replay "
                      "blocks, in 2 columns: by gates, calls; by gates, compressions. Calls are as the Lean says; "
                      "compressions are converted at the longest call (65536 bytes, 513 compressions). A run has 1 "
                      "times the sessions and the gates. ! is above 2^-100")
    assert [line.split()[0] for line in out[2:-2]] == ["collision", "refused", "cap", "sum", "sum", "week", "month",
                                                       "year", "waits", "waits"]
    assert all(out[n].split()[1:3] == ["2^-132.03", "2^-123.02"] for n in (2, 5, 6)) and covered(out)
    assert [out[n].split()[1:3] for n in (10, 11)] == [["2^-125.99"] * 2, ["2^-183.64"] * 2]
    assert "at 2^128·k" in out[10] and "at 2^192·k" in out[11] and all("in no sum" in out[n] for n in (10, 11))
    assert out[-2].startswith("by gates, calls, the collision term beside the other two is at most 2^-100 up to K = "
                              "2^56.72 per sampled replay block")
    assert out[-1].startswith("by gates, compressions, the collision term beside the other two is at most 2^-100 up "
                              "to K = 2^47.72 per sampled replay block")
    b.main(["--K", "2^40.23", "--k", "12343986", "--d", "167", "--json"])
    report = json.loads(capsys.readouterr().out)
    block = report["block"][0]
    assert report["design"]["K"] == LEAST and block["unit"] == "per sampled replay block"
    calls, each = block["columns"]["gates, calls"], block["columns"]["gates, compressions"]
    assert (calls["bound"], each["bound"]) == ("2^-96.11", "2^-87.11")
    assert not calls["meets"] and Fraction(calls["exact"]) == at(LEAST, TABLE, SESSIONS)
    b.main(["--K", "2^40.23", "--symbols", "2^23", "--tables", "200", "--d", "167", "--json"])
    assert json.loads(capsys.readouterr().out)["block"][0]["columns"]["gates, compressions"]["bound"] == "2^-79.46"
    with pytest.raises(SystemExit):
        b.main(["--card", "GLM-5.3", "--symbols", "2^10", "--classes", "5"])
    assert "--classes counts a design by position" in capsys.readouterr().err


def test_the_command_prints_by_position_each_term_with_its_premise_and_what_it_covers(capsys):
    b.main(["--K", "2^37.69", "--sessions", "15694", "--symbols", "2^23", "--first-stage", "6"])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == ("by position, per sampled replay block: N_s = 15694 sessions, K_s = 221727998429 gates "
                      "(2^37.69), K_m = 221727998429 gates (2^37.69), c = n_PU·openCalls = 1·0 = 0, "
                      "T = 2^80.00 calls of SHA-512 in every run, the prover's and the verifier's, k = 12343986, "
                      "e_s = 2592237061")
    assert "A run has 6 times the sessions and the gates." in out[1]
    assert [line.split()[:3] for line in out[2:11]] == [
        ["sessions", "0", "0"], ["rewinding", "2^-48.97!", "2^-44.47!"], ["rewinding", "2^-136.97", "2^-127.97"],
        ["conflict", "2^-129.79", "2^-120.78"], ["first", "hit", "2^-105.03"], ["sum", "2^-48.97!", "2^-44.47!"],
        ["sum", "2^-105.03", "2^-96.03!"], ["sum", "2^-47.68!", "2^-43.18!"], ["sum", "2^-102.45", "2^-93.45!"]]
    assert covered(out) and len(out) == 21
    premise = ["not given, counted as 0", "`ExpectedCR` at B", "`StrictBirthday` at B",
               "`ExpectedCR` at the conflict finder", "`ExpectedCR` at the two first-hit finders, at every limit"]
    assert all(text in line for text, line in zip(premise, out[2:7]))
    assert all("per run at a padded first stage of K0 = 6 replay blocks" in line for line in out[9:19])
    assert [line.split()[0] for line in out[11:19]] == ["week", "week", "month", "month", "year", "year", "waits",
                                                        "waits"]
    assert out[-2].startswith("by gates, calls, the first-hit term at K_s = K_m = K is at most 2^-100 up to K = "
                              "2^42.72 per sampled replay block; 2^40.14 per run at a padded first stage of K0 = 6")
    assert out[-1].startswith("by gates, compressions, the first-hit term at K_s = K_m = K is at most 2^-100 up to "
                              "K = 2^33.72 per sampled replay block")
    b.main(["--K", "2^37.69", "--sessions", "2893", "--symbols", "2^23", "--units", "2^20", "--open-calls", "2^40",
            "--kappa", "1/2^203", "--longest-call", "111", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["design"]["form"] == "position" and report["design"]["own_calls"] == 2 ** 60
    assert report["longest_call"] == {"bytes": 111, "compressions": 1}
    assert [(c["term"], c["columns"]["gates, calls"]["bound"]) for c in report["terms"]][::4] == [
        ("sessions", "2^-203.00"), ("first hit", "2^-105.03")]
    rows = report["terms"] + report["block"] + report["run"]
    assert all(c["columns"]["gates, calls"] == c["columns"]["gates, compressions"] for c in rows)
    assert {c["at_B"]: c["columns"]["gates, calls"]["meets"] for c in report["block"]} == {
        "ExpectedCR": False, "StrictBirthday": True}
    assert all(c["unit"] == "per sampled replay block" for c in report["terms"] + report["block"])
    assert report["most_gates"]["gates, calls"] == report["most_gates"]["gates, compressions"]
    assert [(c["cap"], c["k"], c["bound"]) for c in report["waits"]] == [(128, 1024, "2^-125.99"),
                                                                        (192, 1024, "2^-183.64")]


def test_the_command_prints_four_columns_when_the_classes_are_stated(capsys):
    b.main(["--K", "2^41.81", "--classes", "2^26.24", "--sessions", "15694", "--symbols", "2^23", "--first-stage",
            "6"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("by position, per sampled replay block: N_s = 15694 sessions, K_s = 3855352734248 gates "
                             "(2^41.81), K_m = 3855352734248 gates (2^41.81), K_cls = 79255076 classes (2^26.24), ")
    assert "in 4 columns: by gates, calls; by gates, compressions; by classes, calls; by classes, compressions." in out[1]
    assert out[6].split()[:6] == ["first", "hit", "2^-100.91", "2^-91.91!", "2^-116.48", "2^-107.48"]
    assert out[10].split()[:5] == ["sum", "2^-98.33!", "2^-89.33!", "2^-113.90", "2^-104.90"]
    assert out[16].split()[:5] == ["year", "2^-82.65!", "2^-73.64!", "2^-98.22!", "2^-89.21!"]
    assert covered(out) and len(out) == 23 and [line.split(",")[0] for line in out[-4:]] == [
        "by gates", "by gates", "by classes", "by classes"]


def test_the_gates_as_a_count_gives_them_are_the_checked_blocks_times_the_largest_blocks_needs():
    """Counted as checked blocks times the largest block's needs, the priced design's gates are 2^42.53 where their
    mean is 2^41.81, and the first-hit term is 0.72 bits weaker: under 2^-100 a sampled replay block, above it a
    run."""
    mean = b.Design(K=GATES, k=TABLE, sessions=CHECKED)
    counted = b.as_counted(mean, LARGEST)
    assert counted.K == CHECKED * LARGEST == 6350673952000 and b.log2(counted.K) == "42.53"
    assert b.as_counted(replace(mean, sessions=MOST_SESSIONS), LARGEST, CHECKED).K == counted.K
    assert b.as_counted(replace(mean, Km=2 * GATES), LARGEST).Km is None
    goal = Fraction(1, 2 ** 100)
    assert b.bits(b.terms(counted)["first hit"]) == "2^-100.19" and b.terms(counted)["first hit"] <= goal
    assert [b.bits(b.bound(b.at_run(counted, n))) for n in (4, 6)] == ["2^-98.19", "2^-97.61"]
    assert b.bits(b.bound(b.at_run(b.converted(counted), 6))) == "2^-88.61"
    assert [b.bits(b.bound(b.at_run(mean, n))) for n in (4, 6)] == ["2^-98.91", "2^-98.33"]
    with pytest.raises(ValueError, match="by position"):
        b.as_counted(b.Design(K=GATES, k=TABLE), LARGEST)
    with pytest.raises(ValueError, match="positive"):
        b.as_counted(mean, 0)


def test_the_report_gives_each_count_as_given_and_as_counted():
    mean = b.Design(K=GATES, k=TABLE, sessions=CHECKED)
    out = b.report(mean, first_stage=6, classes=CLASSES, largest=LARGEST, largest_classes=132)
    assert out["columns"] == [f"{count}, {unit}" for count in ("gates", "gates as counted", "classes",
                                                               "classes as counted")
                              for unit in ("calls", "compressions")]
    assert out["design"]["counted"] == {
        "gates as counted": {"checked": CHECKED, "largest": LARGEST, "K": CHECKED * LARGEST},
        "classes as counted": {"checked": CHECKED, "largest": 132, "K": CHECKED * 132}}
    run = {column: c["bound"] for column, c in out["run"][-1]["columns"].items()}
    assert run["gates as counted, calls"] == "2^-97.61" and run["gates, calls"] == "2^-98.33"
    assert run["classes as counted, calls"] == b.bits(b.bound(b.at_run(b.by_classes(mean, CHECKED * 132), 6)))
    assert list(out["most_gates"]) == ["gates, calls", "gates, compressions", "classes, calls",
                                       "classes, compressions"]
    assert b.report(mean)["design"]["counted"] == {} and len(b.report(mean)["columns"]) == 2


def test_the_command_prints_the_gates_as_counted_beside_the_mean(capsys):
    b.main(["--K", "2^41.81", "--sessions", str(CHECKED), "--largest", str(LARGEST), "--symbols", "2^23",
            "--first-stage", "6"])
    out = capsys.readouterr().out.splitlines()
    assert out[1] == ("gates as counted, per sampled replay block: K_s = K_m = 10046500 checked blocks times the "
                      "632128 gates the largest block needs = 6350673952000 gates (2^42.53); as given, K_s and K_m "
                      "are means over the coins")
    assert "in 4 columns: by gates, calls; by gates, compressions; by gates as counted, calls; by gates as counted, " \
           "compressions." in out[2]
    assert out[7].split()[:6] == ["first", "hit", "2^-100.91", "2^-91.91!", "2^-100.19", "2^-91.19!"]
    assert out[11].split()[:5] == ["sum", "2^-98.33!", "2^-89.33!", "2^-97.61!", "2^-88.61!"]
    assert covered(out) and len(out) == 22 and [line.split(",")[0] for line in out[-2:]] == ["by gates", "by gates"]


@pytest.mark.parametrize("argv, message", [
    (["--K", "2^41.81", "--symbols", "2^23", "--largest", "632128"], "--largest counts a design by position"),
    (["--K", "2^41.81", "--symbols", "2^23", "--sessions", "5", "--checked", "7"], "--checked is the count"),
])
def test_the_command_refuses_a_count_it_cannot_place(argv, message, capsys):
    with pytest.raises(SystemExit):
        b.main(argv)
    assert message in capsys.readouterr().err
