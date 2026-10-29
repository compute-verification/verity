"""The monitoring period's pooled step (`monitoring`): the count is the least within δ of the count bound's C(T, M)·p^M, the
reference year at 10⁻⁶ leaves one session uncovered where the union T·eps doesn't reach δ, the per-window figure is the
session's openings bound at δ/T, its error charging each session's draw term at its `samples`, and a rejected session
pools nothing."""

from __future__ import annotations

import json
import math
from fractions import Fraction

import pytest

from verity.core.protocols.remote_bandwidth import audit, codec, monitoring, pooled

EPS = Fraction(2 ** 24 + 2 * audit.SAMPLES, 2 ** 64)
ALL_DRAWN = {"verdict": "accept", "reason": None, "public": {"eps": str(EPS), "q_bytes": 34 * codec.BLOCK_BYTES},
             "profile": {"delta": str(Fraction(1, 2 ** 40)), "gaps": [], "receipts": 34, "windows": [[17, 17], [17, 17]],
                         "wrong": "0"}}
"""An accepted run of the subprotocol at 64 KiB: two windows of 17 receipts, every one drawn, at the eps `certify`
publishes for it, its two openings coins at `samples` each."""


def partly_drawn() -> dict:
    """A session whose two windows of 100,000 receipts drew 33,273 each, its outputs as `certify` makes them."""
    windows = [(100_000, 33_273), (100_000, 33_273)]
    b = pooled.bound(windows, Fraction(1, 2 ** 40))
    wrong = -(-b.wrong.numerator // b.wrong.denominator)
    return {"verdict": "accept", "reason": None,
            "public": {"eps": str(Fraction(2 ** 24 + 66_546, 2 ** 64)), "q_bytes": (200_000 - wrong) * codec.BLOCK_BYTES},
            "profile": {"delta": str(Fraction(1, 2 ** 40)), "gaps": [], "windows": [list(w) for w in windows]}}


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
        for delta in monitoring.DELTAS:
            assert (monitoring.short_sessions(t, EPS, delta)[0] == 1) == (t * EPS <= delta)


def test_the_reference_year_at_one_in_a_million_leaves_one_session_uncovered():
    """12,500 machines with a session every 600 s: T·eps ≈ 6·10⁻⁴ is within 1% but not 10⁻⁶, where the count still
    certifies all but one session."""
    year = monitoring.table(ALL_DRAWN, 12_500, 600)["periods"]["year"]
    t = 12_500 * (365 * 86400 // 600)
    assert year["sessions"] == t
    one, tiny = year["1/100"]["pooled"], year["1/1000000"]["pooled"]
    assert (one["short_sessions_at_most"], one["certified_sessions_at_least"]) == (0, t)
    assert (tiny["short_sessions_at_most"], tiny["certified_sessions_at_least"]) == (1, t - 1)
    assert tiny["q_bytes_at_least"] == (t - 1) * 34 * codec.BLOCK_BYTES
    assert t * EPS > Fraction(1, 10 ** 6)


def test_a_week_of_one_machine_certifies_every_session():
    week = monitoring.table(ALL_DRAWN, 1, 600)["periods"]["week"]
    assert week["sessions"] == 1008
    for d in ("1/100", "1/1000000"):
        assert week[d]["pooled"]["q_bytes_at_least"] == 1008 * 34 * codec.BLOCK_BYTES
        assert week[d]["per_window"]["q_bytes_each"] == 34 * codec.BLOCK_BYTES


def test_the_per_window_figure_is_the_openings_bound_at_delta_over_t():
    a = partly_drawn()
    r = monitoring.table(a, 12_500, 600)
    for row in r["periods"].values():
        t = row["sessions"]
        for d in monitoring.DELTAS:
            b = pooled.bound([tuple(w) for w in a["profile"]["windows"]], d / t)
            each = (200_000 - -(-b.wrong.numerator // b.wrong.denominator)) * codec.BLOCK_BYTES
            got = row[str(d)]["per_window"]
            assert got["q_bytes_each"] == each
            assert got["q_bytes"] == t * each
            q, d0 = a["public"]["q_bytes"], Fraction(1, 2 ** 40)
            assert each <= q if d / t <= d0 else each >= q


def test_the_per_window_error_charges_each_sessions_draw_term_at_its_samples():
    """`ALL_DRAWN` drew 34 receipts, but its eps charges its two openings coins at `samples` each, and the per-window
    figure's error charges that term once a session, not the draws made (2.4·10⁻⁶ over the year, not 1.2·10⁻⁹)."""
    year = monitoring.table(ALL_DRAWN, 12_500, 600)["periods"]["year"]
    t = year["sessions"]
    for d in monitoring.DELTAS:
        want = d + t * Fraction(2 * audit.SAMPLES, 2 ** 64)
        assert year[str(d)]["per_window"]["except_with"] == f"δ + T·(eps − δ₀) = {float(want):.3g}"
    assert year["1/1000000"]["per_window"]["except_with"].endswith("= 3.37e-06")


def test_a_rejected_session_pools_nothing():
    r = monitoring.table({"verdict": "reject", "reason": "the session was rejected: payload-path", "public": None,
                          "profile": None}, 1, 600)
    assert r["periods"] is None and "payload-path" in r["why"]


def test_the_cli_reads_a_legs_bandwidth_json(tmp_path, capsys):
    f = tmp_path / "bandwidth.json"
    f.write_text(json.dumps({"answer": ALL_DRAWN, "q_bytes": ALL_DRAWN["public"]["q_bytes"]}))
    assert monitoring.main([str(f), "--machines", "1", "--window-s", "600", "--out", str(tmp_path / "m.json")]) == 0
    got = json.loads((tmp_path / "m.json").read_text())
    assert got == json.loads(capsys.readouterr().out)
    assert got["periods"]["month"]["sessions"] == 30 * 144


def test_each_period_is_labelled_as_one_session_scaled_up():
    """No period is measured: each row says it is the one session scaled to the period's sessions, and the statement
    is named by its Lean declaration and file alone."""
    r = monitoring.table(ALL_DRAWN, 12_500, 600)
    for name, row in r["periods"].items():
        assert row["basis"] == f"one session scaled up to {row['sessions']} sessions, not a measured {name}"
    assert set(r["statement"]) == {"name", "file"}
