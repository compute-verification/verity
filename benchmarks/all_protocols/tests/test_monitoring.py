"""The result per monitoring period, over all compute:

* the pooled total is ``IntegrityHorizon``'s ``ln(1/δ)/q``, about 74 wrong units at k = 64 of n = 1024 and δ = 1%, the
  same over a week, a month and a year, so its share of the period's units falls as the period grows;
* the per-window figure beside it is ``IntegrityPeriod``'s least threshold at δ/T a window, exact, and it grows with T;
* a window that rejects rejects the monitoring period, and with no accepted window nothing is scaled;
* the deployment is read from the census, and a missing one names where to add it;
* the network certifier's K is pooled over the same periods, one machine's windows and the deployment's machines;
* the summary line leads with the week, the month and the year at both δ.
"""
from __future__ import annotations

import math
from fractions import Fraction

import pytest

from verity_all_protocols import compose as C, monitoring as M

DEP = {"id": "test-dc", "machines": 10, "window_s": 3600}


def row(verdict: str, n: int = 1024, kd: int = 64) -> dict:
    if verdict != "accept":
        return {"verdict": verdict}
    return {"verdict": "accept", "answer": {"profile": {"drawn": {"unit": kd},
                                                        "levels": [{"name": "unit", "size": n}]}}}


def report(w: int, verdict: str, composed: str = "pending-ports") -> dict:
    return {"window": w, "protocols": {"integrity": {**row(verdict), "owner": "compute-accounting"}},
            "composed": {"verdict": composed}}


def test_the_pooled_total_is_ln_one_over_delta_over_q():
    q = Fraction(64, 1024)
    assert M.pooled_bound(q, Fraction(1, 100)) == 73 == math.ceil(16 * math.log(100)) - 1
    assert M.pooled_bound(q, Fraction(1, 10**6)) == 221


@pytest.mark.parametrize("delta", [Fraction(1, 100), Fraction(1, 10**6), Fraction(1, 100 * 657_000_000)])
def test_the_per_window_threshold_is_the_least_one_whose_sampling_term_is_within_delta(delta):
    n, kd = 1024, 64
    k0 = M.window_threshold(n, kd, delta)
    assert Fraction(math.comb(n - k0, kd), math.comb(n, kd)) <= delta < Fraction(math.comb(n - k0 + 1, kd), math.comb(n, kd))


def test_each_period_pools_one_total_and_gives_the_per_window_figure_beside_it():
    i = M.integrity([row("accept"), row("accept")], DEP)
    assert i["observed"] == {"windows": 2, "accepted": 2, "not_accepted": 0, "verdict": "accept", "units": 2048,
                             "n": 1024, "kd": 64, "q": "1/16", "pooled": i["observed"]["pooled"]}
    week, month, year = (i["periods"][p] for p in ("week", "month", "year"))
    assert (week["windows"], month["windows"], year["windows"]) == (1680, 7200, 87600)
    for d in ("1/100", "1/1000000"):
        pooled = [p[d]["pooled"]["wrong_units_at_most"] for p in (week, month, year)]
        per_window = [p[d]["per_window"]["wrong_units_at_most"] for p in (week, month, year)]
        assert len(set(pooled)) == 1 and per_window == sorted(per_window) and per_window[0] < per_window[-1]
        assert float(year[d]["pooled"]["share"]) < float(week[d]["pooled"]["share"])
        assert "ε_ks" in week[d]["except_with"]
    assert week["1/100"]["pooled"]["wrong_units_at_most"] == 73
    assert any("test-dc" in s for s in i["scaled"]) and any("IntegrityHorizon" in s for s in i["proved"])


def test_a_rejected_window_rejects_the_monitoring_period():
    m = M.monitoring([report(0, "accept"), report(1, "reject", "reject"), {"window": 2, "error": "the loop: x"}], DEP)
    assert m["integrity"]["observed"]["verdict"] == "reject" and m["integrity"]["observed"]["not_accepted"] == 1
    assert m["composed"] == {"verdict": "reject", "windows": 2, "errors": 1}
    assert m["protocols"]["integrity"] == {"verdicts": ["accept", "reject"], "owner": "compute-accounting"}
    assert [w["window"] for w in m["windows"]] == [0, 1, 2]
    assert M.integrity([row("reject")], DEP)["periods"] is None


def test_the_deployment_is_the_census_one():
    d = M.deployment()
    assert d["id"] == M.REFERENCE and d["machines"] == 12500 and d["window_s"] == 600
    with pytest.raises(KeyError, match="add it there"):
        M.deployment("no-such-dc")


def test_the_line_leads_with_every_period_at_both_deltas():
    lead = M.lead(M.monitoring([report(0, "accept")], DEP))
    assert set(lead) == {"observed", "week", "month", "year", "composed"}
    assert set(lead["year"]) == {"1/100", "1/1000000"}
    assert lead["year"]["1/100"]["pooled"] == 73 and lead["year"]["1/100"]["per_window"] > 73


def test_the_network_certifiers_k_is_pooled_over_the_deployments_periods():
    q = {"K": 24, "eps": "1/100", "windows": 1, "schedules": 4, "cells": 3,
         "audits": [{"units": 3, "drawn": 3, "free": 1, "jitter": 2}]}
    windows = [report(0, "accept"), report(1, "accept")]
    for w, (verdict, quantity) in zip(windows, (("accept", q), ("reject", None))):
        w["protocols"]["communication"] = {"verdict": verdict, "protocol": C.CERTIFIER, "quantity": quantity,
                                           "owner": "network-accounting"}
    c = M.monitoring(windows, DEP)["communication"]
    assert (c["periods"], c["accepted"], c["refused"], c["period_s"]) == (2, 1, 1, DEP["window_s"])
    assert [c[p]["periods"] for p, _ in M.PERIODS] == [168, 720, 8760]
    for p, _ in M.PERIODS:
        assert set(c[p]["pooled"]) == set(c[p]["fleet"]) == {str(d) for d in M.DELTAS}
        assert c[p]["fleet"]["1/100"]["machines"] == DEP["machines"]
    assert c["week"]["pooled"]["1/100"] == C.horizon_leak([q], Fraction(1, 100), 168)
