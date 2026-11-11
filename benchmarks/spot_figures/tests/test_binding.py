"""The two-stage finder's bound (`binding`): its arithmetic is exact, and its value at each design the binding
options are compared at, with the prover at 2^80 SHA-512 compressions, is the one stated, to a tenth of a bit."""
import json
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
#: GLM-5.3's K at one proof block a stratum, the design with no cap on a stratum's sample
UNCAPPED = 27191776
#: K at the two ends of the designs the cost rows price, 2^40.23 and 2^43.23 rounded up
LEAST, MOST = b.natural("2^40.23"), b.natural("2^43.23")
#: T when 2^80 counts calls on 64 KiB rows
ROWS = b.BUDGET * b.compressions(65536)


def tenths(p: Fraction) -> Fraction:
    """−log₂ p rounded to a tenth of a bit, half up."""
    return Fraction((b.centibits(p) + 5) // 10, 10)


def at(K: int, k: int, d: int = 1, T: int = b.BUDGET) -> Fraction:
    return b.bound(b.Design(K=K, k=k, d=d, T=T))


def test_the_bound_is_its_three_terms_exactly():
    d = b.Design(K=3, k=1, d=2, T=5, open_calls=7, p_ref=Fraction(1, 8), opens=4, M=3)
    assert (d.reruns, d.cost) == (211, 5 * (1 + 2 * 211) + 2 * 7)
    assert b.terms(d) == {"collision": Fraction(2 * 3 * 2129, 2 ** 256), "refused": Fraction(1, 2), "cap": Fraction(1)}
    assert b.bound(d) == Fraction(2 * 3 * 2129, 2 ** 256) + Fraction(3, 2)
    assert all(isinstance(t, Fraction) for t in b.terms(b.Design(K=1, k=1)).values())
    assert b.bound(b.Design(K=1, k=1)) == Fraction(2 * 212 * b.BUDGET, 2 ** 256)


def test_the_bound_is_linear_in_the_gates_and_the_budget_and_rises_with_the_sessions():
    one = at(UNCAPPED, SMALL)
    assert at(2 * UNCAPPED, SMALL) == 2 * one == at(UNCAPPED, SMALL, T=2 * b.BUDGET)
    assert one < at(UNCAPPED, SMALL, d=2) < 2 * one
    assert at(UNCAPPED, TABLE) / one == Fraction(1 + 1 + 210 * TABLE, 1 + 1 + 210 * SMALL)


@pytest.mark.parametrize("design, bad", [
    (dict(K=0, k=1), "K"), (dict(K=1, k=0), "k"), (dict(K=1, k=1, d=0), "d"), (dict(K=1, k=1, T=0), "T"),
    (dict(K=1.5, k=1), "K"), (dict(K=1, k=1, open_calls=-1), "open_calls"),
    (dict(K=1, k=1, p_ref=Fraction(3, 2)), "p_ref"), (dict(K=1, k=1, opens=5), "M")])
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


def test_a_monitoring_period_sums_its_audits():
    d = b.Design(K=UNCAPPED, k=SMALL)
    assert {name: n for name, (n, _) in b.periods(d).items()} == {"week": 1008, "month": 4320, "year": 52560}
    assert all(p == n * b.bound(d) for n, p in b.periods(d).values())
    assert b.periods(d, machines=12500)["year"][0] == 12500 * 52560
    assert (b.audits(600), b.audits(601), b.audits(0), b.audits(1200, 600, 3)) == (1, 2, 0, 6)


# The designs the binding options are compared at, one audit, the prover at 2^80 compressions.

def test_no_cap_on_a_stratums_sample_at_the_smallest_session():
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


def test_calls_on_64_kib_rows_cost_nine_bits():
    assert ROWS == 513 * 2 ** 80
    k = b.queries(TABLES * 2 ** 23)
    cells = [(UNCAPPED, SMALL, 1, "123.0"), (LEAST, SMALL, 1, "107.5"), (MOST, SMALL, 1, "104.5"),
             (LEAST, TABLE, SESSIONS, "87.1"), (MOST, TABLE, SESSIONS, "84.1"),
             (LEAST, k, SESSIONS, "79.5"), (MOST, k, SESSIONS, "76.5")]
    assert [str(tenths(at(K, q, d, ROWS))) for K, q, d, _ in cells] == [str(Fraction(c)) for *_, c in cells]
    assert all(at(K, q, d, ROWS) == 513 * at(K, q, d) for K, q, d, _ in cells)


def test_a_year_of_audits_costs_15_7_bits():
    year = b.audits(365 * 86400)
    assert year == 52560 and tenths(Fraction(1, year)) == Fraction("15.7")
    cells = [(UNCAPPED, SMALL, 1, "116.3"), (LEAST, SMALL, 1, "100.8"), (MOST, SMALL, 1, "97.8"),
             (LEAST, TABLE, SESSIONS, "80.4"), (MOST, TABLE, SESSIONS, "77.4")]
    for K, q, d, cell in cells:
        (n, p), one = b.periods(b.Design(K=K, k=q, d=d))["year"], at(K, q, d)
        assert (n, p) == (year, year * one) and tenths(p) == Fraction(cell)


def test_the_served_models_card_keeps_100_bits_at_the_smallest_session_over_a_year():
    K = b.card_gates(cards.load("workload", "GLM-5.3"))
    out = b.report(b.Design(K=K, k=b.queries(2 ** 10)))
    assert out["audit"]["meets"] and all(c["meets"] for c in out["periods"].values())
    assert not b.report(b.Design(K=K, k=TABLE, d=SESSIONS))["periods"]["year"]["meets"]


def test_the_most_gates_a_target_leaves():
    cells = [(SMALL, 1, "56.72"), (SMALL, SESSIONS, "49.34"), (TABLE, 1, "43.72"), (TABLE, SESSIONS, "36.34")]
    for k, d, cell in cells:
        most = b.most_gates(b.Design(K=1, k=k, d=d), 100)
        goal = Fraction(1, 2 ** 100)
        assert at(most, k, d) <= goal < at(most + 1, k, d) and b.log2(most) == cell
    d = b.Design(K=1, k=SMALL)
    assert b.most_gates(d, 100, audits=52560) == b.most_gates(d, 100) // 52560
    assert b.most_gates(b.Design(K=1, k=1, opens=1, M=0), 100) == 0


def test_the_command_prints_the_design_and_each_period(capsys):
    b.main(["--card", "GLM-5.3", "--symbols", "2^10"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("T = 2^80.00 compressions, d = 1, k = 1507, e_s = 316471")
    assert out[1].split() == ["one", "audit", out[2].split()[1]] and out[2].split()[0] == "collision"
    assert [line.split()[0] for line in out[-4:-1]] == ["week", "month", "year"]
    assert out[-1].startswith("at most 2^-100 up to K = ")
    b.main(["--K", "2^40.23", "--k", "12343986", "--d", "167", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["design"]["K"] == LEAST and report["audit"]["bound"] == "2^-96.11" and not report["audit"]["meets"]
    assert Fraction(report["audit"]["exact"]) == at(LEAST, TABLE, SESSIONS)
    b.main(["--K", "2^40.23", "--symbols", "2^23", "--tables", "200", "--d", "167", "--row-bytes", "65536", "--json"])
    assert json.loads(capsys.readouterr().out)["audit"]["bound"] == "2^-79.46"
