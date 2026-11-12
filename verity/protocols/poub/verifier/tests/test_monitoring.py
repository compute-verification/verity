"""The monitoring period's table (`monitoring`): a session's error is the whole error its verifier's ledger gives it,
the sketch's term included, and never the answer's `eps` alone; a period's row carries that error summed over its
sessions and, where the sum is 1 or more, prints "no bound at this period" and no amount; the count is the least within δ
of the count bound's C(T, M)·p^M, the reference year at 10⁻⁶ leaves one session uncovered where the sum T·p doesn't
reach δ, the per-window figure is the session's openings bound at δ/T, its error charging each session's terms beyond
δ₀ once, and a rejected session pools nothing."""

from __future__ import annotations

import json
import math
from fractions import Fraction

import pytest

from verity.protocols import reporting
from verity.protocols.poub.verifier import audit, codec, coins, ledger, monitoring, period, pooled, timing

D0 = Fraction(1, 2 ** 40)
EPS = Fraction(2 ** 24 + 2 * audit.SAMPLES, 2 ** 64)
ALL_DRAWN = {"verdict": "accept", "reason": None, "public": {"eps": str(EPS), "q_bytes": 34 * codec.BLOCK_BYTES},
             "profile": {"delta": str(D0), "gaps": [], "receipts": 34, "windows": [[17, 17], [17, 17]],
                         "wrong": "0"}}
"""An accepted run of the subprotocol at 64 KiB: two windows of 17 receipts, every one drawn, at the eps `certify`
publishes for it, its two openings coins at `samples` each."""


def ledger_error(eps: Fraction = EPS, *, passes: int = 2, sketch: float = 0.0, binding: float = 0.0) -> dict:
    """The error a verifier's ledger gives a session whose answer publishes `eps` (`ledger.Gamma.to_json`'s `error`):
    the horizon's, δ₀ and AuditRO's slack over `passes` passes, with the sketch's term and the sample's two; its
    per-window total is unread here."""
    draw = ledger.SamplingTerms(float(eps - D0), binding)
    horizon = float(D0) + passes * 2.0 ** -audit.SLACK_LOG2 + sketch + draw.total
    return {"draw": draw.to_json(), "horizon": horizon, "sketch": sketch, "windows": None}


NO_SKETCH = ledger_error()
"""`ALL_DRAWN`'s error under a receipt that is no sketch: its `eps` and two passes' slack."""
P = Fraction(NO_SKETCH["horizon"])


def partly_drawn() -> dict:
    """A session whose two windows of 100,000 receipts drew 33,273 each, its outputs as `certify` makes them."""
    windows = [(100_000, 33_273), (100_000, 33_273)]
    b = pooled.bound(windows, D0)
    wrong = -(-b.wrong.numerator // b.wrong.denominator)
    return {"verdict": "accept", "reason": None,
            "public": {"eps": str(Fraction(2 ** 24 + 66_546, 2 ** 64)), "q_bytes": (200_000 - wrong) * codec.BLOCK_BYTES},
            "profile": {"delta": str(D0), "gaps": [], "windows": [list(w) for w in windows]}}


@pytest.mark.parametrize("t,p,delta", [(1, Fraction(1, 2), Fraction(1, 2)), (10, Fraction(1, 100), Fraction(1, 100)),
                                       (1008, EPS, Fraction(1, 10 ** 6)), (10 ** 6, Fraction(1, 10 ** 5), Fraction(1, 100)),
                                       (657_000_000, EPS, Fraction(1, 10 ** 6)), (5, Fraction(9, 10), Fraction(1, 10 ** 6))])
def test_the_count_is_the_least_within_delta(t, p, delta):
    m, bound = monitoring.short_sessions(t, p, delta)
    def term(k: int) -> Fraction:
        return math.comb(t, k) * p ** k

    if m <= t:
        assert bound == term(m) <= delta
    assert all(term(k) > delta for k in range(1, min(m, t + 1)))


def test_one_short_session_is_ruled_out_exactly_when_the_union_is():
    for t in (1, 1008, 10 ** 9):
        for delta in reporting.DELTAS:
            assert (monitoring.short_sessions(t, EPS, delta)[0] == 1) == (t * EPS <= delta)


def test_the_reference_year_at_one_in_a_million_leaves_one_session_uncovered():
    """12,500 machines with a session every 600 s: the row's error T·p ≈ 6·10⁻⁴ is within 1% but not 10⁻⁶, where the
    count still certifies all but one session."""
    year = monitoring.table(ALL_DRAWN, NO_SKETCH, 12_500, 600)["periods"]["year"]
    t = 12_500 * (365 * 86400 // 600)
    assert year["sessions"] == t
    assert year["error"] == float(t * P) and year["q_bytes"] == t * 34 * codec.BLOCK_BYTES and "no_bound" not in year
    one, tiny = year["1/100"]["pooled"], year["1/1000000"]["pooled"]
    assert (one["short_sessions_at_most"], one["certified_sessions_at_least"]) == (0, t)
    assert (tiny["short_sessions_at_most"], tiny["certified_sessions_at_least"]) == (1, t - 1)
    assert tiny["q_bytes_at_least"] == (t - 1) * 34 * codec.BLOCK_BYTES
    assert Fraction(1, 100) > t * P > Fraction(1, 10 ** 6)


def test_a_week_of_one_machine_certifies_every_session():
    week = monitoring.table(ALL_DRAWN, NO_SKETCH, 1, 600)["periods"]["week"]
    assert week["sessions"] == 1008
    for d in ("1/100", "1/1000000"):
        assert week[d]["pooled"]["q_bytes_at_least"] == 1008 * 34 * codec.BLOCK_BYTES
        assert week[d]["per_window"]["q_bytes_each"] == 34 * codec.BLOCK_BYTES


def test_the_per_window_figure_is_the_openings_bound_at_delta_over_t():
    a = partly_drawn()
    r = monitoring.table(a, ledger_error(Fraction(a["public"]["eps"])), 12_500, 600)
    for row in r["periods"].values():
        t = row["sessions"]
        for d in reporting.DELTAS:
            b = pooled.bound([tuple(w) for w in a["profile"]["windows"]], d / t)
            each = (200_000 - -(-b.wrong.numerator // b.wrong.denominator)) * codec.BLOCK_BYTES
            got = row[str(d)]["per_window"]
            assert got["q_bytes_each"] == each
            assert got["q_bytes"] == t * each
            q = a["public"]["q_bytes"]
            assert each <= q if d / t <= D0 else each >= q


def test_the_per_window_error_charges_each_sessions_terms_beyond_its_own_delta():
    """`ALL_DRAWN` drew 34 receipts, but its error charges its two openings coins at `samples` each, and the per-window
    figure's error charges every term beyond δ₀ once a session, not the draws made (2.4·10⁻⁶ over the year, not
    1.2·10⁻⁹)."""
    year = monitoring.table(ALL_DRAWN, NO_SKETCH, 12_500, 600)["periods"]["year"]
    t = year["sessions"]
    for d in monitoring.DELTAS:
        want = d + t * (P - D0)
        assert want > d + t * Fraction(2 * audit.SAMPLES, 2 ** 64), "the passes' slack is charged too"
        assert year[str(d)]["per_window"]["except_with"] == f"δ + T·(error − δ₀) = {float(want):.3g}"
    assert year["1/1000000"]["per_window"]["except_with"].endswith("= 3.37e-06")


def test_a_rejected_session_pools_nothing():
    r = monitoring.table({"verdict": "reject", "reason": "the session was rejected: payload-path", "public": None,
                          "profile": None}, None, 1, 600)
    assert r["periods"] is None and "payload-path" in r["why"]
    assert r["except_also"] == monitoring.EXCEPT, "a table that pools nothing still says what its δ leaves out"


SHAPE = {"windows": 88, "samples": 3_025, "passes": 88 * 64, "pace_k": 64, "rows": 16, "blocks": 130_023_424,
         "receipts": 8_321_499_136, "span_ns": 282_849_213_684}
"""A paced sketch session's shape on the stated node: 88 windows of 64 passes over a store of 130,023,424 blocks, one
openings coin a window at 3,025 samples, 16 sign rows, and its span."""


def shaped() -> tuple[dict, dict]:
    """`SHAPE`'s answer as `certify` makes it, and its error as the ledger computes it on `ledger.Node.stated()`: each
    term by the ledger's own function of the shape, added in `ledger.Gamma.error_horizon`'s order."""
    node, w, s = ledger.Node.stated(), SHAPE["windows"], SHAPE["samples"]
    c = math.ceil(node.sha512_per_s * SHAPE["span_ns"] / 1e9)
    sketch = float(timing.sketch_failure(c, SHAPE["passes"], SHAPE["rows"], blocks=SHAPE["blocks"],
                                         outside=node.sketch_outside))
    every = ledger.window_receipts(audit.Pace(SHAPE["pace_k"], 60 * 10 ** 9), SHAPE["blocks"], 0, 0)
    draw = ledger.SamplingTerms(coins.openings_coins(w, 1) * s * 2.0 ** -ledger.REJECTION_LOG2,
                                ledger.binding_over(coins.openings_coins(w, 1), every, s, node.sketch_outside * c))
    horizon = float(D0) + SHAPE["passes"] * 2.0 ** -audit.SLACK_LOG2 + sketch + draw.total
    windows = [(SHAPE["receipts"], s)] * w
    b = pooled.bound(windows, D0)
    q = (w * SHAPE["receipts"] - -(-b.wrong.numerator // b.wrong.denominator)) * codec.BLOCK_BYTES
    return ({"verdict": "accept", "reason": None, "public": {"eps": str(period.eps(w, 1, s)), "q_bytes": q},
             "profile": {"delta": str(D0), "gaps": [], "windows": [list(x) for x in windows]}},
            {"draw": draw.to_json(), "horizon": horizon, "sketch": sketch, "windows": None})


def test_a_sketch_sessions_whole_error_is_the_ledgers_and_is_hundreds_of_times_its_eps():
    """The answer's `eps` is δ₀ and the rejection, 2,130,427/2^61 (9.24·10⁻¹³). The whole error adds the sketch's term
    at the node's outside-compute factor, 2^20, which is 99.8% of it: 3.86·10⁻¹⁰, 418 times the `eps`. The table reads
    the whole error, so a period's row sums that and not the `eps`."""
    answer, error = shaped()
    eps = Fraction(answer["public"]["eps"])
    assert ledger.Node.stated().sketch_outside == timing.SKETCH_OUTSIDE == 2 ** 20
    assert eps == Fraction(2_130_427, 2 ** 61) and f"{float(eps):.4g}" == "9.239e-13"
    assert error["horizon"] == 3.8625798563375336e-10 and error["sketch"] == 3.853340220371482e-10
    assert error["sketch"] / error["horizon"] == pytest.approx(0.9976, abs=1e-4)
    assert error["horizon"] / float(eps) == pytest.approx(418.06, abs=0.01)
    table = monitoring.table(answer, error, 12_500, 600)
    o = table["observed"]
    assert (o["eps"], o["error"]) == (str(eps), error["horizon"])
    assert o["error_terms"] == {"sketch": error["sketch"], "binding": error["draw"]["binding"],
                                "rejection": error["draw"]["rejection"]}
    assert f"except with {error['horizon']:.4g}, its whole error" in table["scaled"][1]
    for row in table["periods"].values():
        assert row["error"] == float(row["sessions"] * Fraction(error["horizon"])) > float(row["sessions"] * eps)
    assert [f"{row['error']:.3g}" for row in table["periods"].values()] == ["0.00487", "0.0209", "0.254"]


def test_a_row_whose_summed_error_is_1_or_more_prints_no_bound_and_no_amount():
    """50,000 machines of `SHAPE`'s sessions: a year's 2,628,000,000 sessions sum to 1.015, so the year's row has no
    amount, neither its total nor a figure at any δ, while the month's, at 0.083, has them all. By the answer's `eps`
    alone the year would have summed to 0.0024 and printed a total."""
    answer, error = shaped()
    periods = monitoring.table(answer, error, 50_000, 600)["periods"]
    year, month = periods["year"], periods["month"]
    assert year["sessions"] == 2_628_000_000 and year["sessions"] * Fraction(answer["public"]["eps"]) < Fraction(1, 400)
    assert set(year) == {"seconds", "sessions", "sessions_per_machine", "basis", "error", "no_bound"}
    assert year["no_bound"] == f"{reporting.NO_BOUND}: the row's error, 1.015, is 1 or more"
    assert year["error"] == pytest.approx(1.0150859862455037) and "q_bytes" not in json.dumps(year)
    assert month["error"] == pytest.approx(0.08343172489689073) and "no_bound" not in month
    assert month["q_bytes"] == month["sessions"] * answer["public"]["q_bytes"]
    for d in monitoring.DELTAS:
        assert month[str(d)]["pooled"]["q_bytes_at_least"] > 0 and month[str(d)]["per_window"]["q_bytes"] > 0


def test_the_per_window_figure_has_no_amount_where_its_own_error_is_1_or_more():
    """A week of one machine whose sessions sum to 0.995: the row keeps its total, and the per-window figure, whose
    error adds its δ, keeps its amount at 10⁻⁶ and has none at 1/100, where it sums to 1.005."""
    sketch = float(Fraction(995, 1000) / 1008 - P)
    week = monitoring.table(ALL_DRAWN, ledger_error(sketch=sketch), 1, 600)["periods"]["week"]
    assert week["error"] == pytest.approx(0.995) and week["q_bytes"] == 1008 * 34 * codec.BLOCK_BYTES
    small, large = week["1/1000000"]["per_window"], week["1/100"]["per_window"]
    assert small["q_bytes"] == week["q_bytes"] and "no_bound" not in small
    assert set(large) == {"at", "except_with", "no_bound"} and large["no_bound"].startswith(reporting.NO_BOUND)
    assert large["except_with"] == "δ + T·(error − δ₀) = 1"


def test_an_error_that_is_not_the_sessions_ledgers_is_refused():
    with pytest.raises(ValueError, match="reads the session's whole error from its verifier's ledger"):
        monitoring.table(ALL_DRAWN, None, 1, 600)
    with pytest.raises(ValueError, match="reads the session's whole error from its verifier's ledger"):
        monitoring.table(ALL_DRAWN, {"horizon": float(EPS)}, 1, 600)
    below = ledger_error() | {"horizon": float(D0)}
    with pytest.raises(ValueError, match="is below the answer's eps, .*: it is not this session's"):
        monitoring.table(ALL_DRAWN, below, 1, 600)


def test_the_cli_reads_a_legs_bandwidth_json_and_the_sessions_ledger(tmp_path, capsys):
    f, g = tmp_path / "bandwidth.json", tmp_path / "ledger.json"
    f.write_text(json.dumps({"answer": ALL_DRAWN, "q_bytes": ALL_DRAWN["public"]["q_bytes"]}))
    g.write_text(json.dumps({"declared": {"gamma": {"spec": {"error": NO_SKETCH}, "measured": {"error": NO_SKETCH}}}}))
    args = [str(f), "--machines", "1", "--window-s", "600", "--out", str(tmp_path / "m.json")]
    assert monitoring.main([*args, "--ledger", str(g)]) == 0
    got = json.loads((tmp_path / "m.json").read_text())
    assert got == json.loads(capsys.readouterr().out)
    assert got["periods"]["month"]["sessions"] == 30 * 144 and got["observed"]["error"] == NO_SKETCH["horizon"]
    g.write_text(json.dumps({"gamma": {"spec": {"error": NO_SKETCH}}}))
    assert monitoring.main([*args, "--ledger", str(g)]) == 0, "ledger.declared's own JSON reads the same"
    with pytest.raises(SystemExit) as refused:
        monitoring.main(args)
    assert refused.value.code == 2 and "--ledger" in capsys.readouterr().err


def test_each_period_is_labelled_as_one_session_scaled_up():
    """No period is measured: each row says it is the one session scaled to the period's sessions, and the statement
    is named by its Lean declaration and file alone."""
    r = monitoring.table(ALL_DRAWN, NO_SKETCH, 12_500, 600)
    for name, row in r["periods"].items():
        assert row["basis"] == f"one session scaled up to {row['sessions']} sessions, not a measured {name}"
    assert set(r["statement"]) == {"name", "file"}
    # no δ or error of a table is read alone: beside its deltas it says the term each leaves out, in the ledger's words
    assert r["deltas"] and r["except_also"] == monitoring.EXCEPT
    assert "δ + Σ_t adv_t" in r["except_also"]
    assert "(SHA-512 stream as PRF: not priced, not assumed)" in r["except_also"]
