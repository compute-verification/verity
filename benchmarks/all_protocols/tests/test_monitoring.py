"""The result per monitoring period, over all compute:

* the pooled total is ``IntegrityHorizon``'s ``ln(1/δ)/q``, about 74 wrong units at k = 64 of n = 1024 and δ = 1%, the
  same over a week, a month and a year, so its share of the period's units falls as the period grows;
* the per-window figure beside it is ``IntegrityPeriod``'s least threshold at δ/T a window, exact, and it grows with T;
* a window that rejects rejects the monitoring period, and with no accepted window nothing is scaled;
* the deployment is read from the census, and a missing one names where to add it;
* the network certifier's K per machine is pooled over the same periods, one machine's windows and the deployment's
  machines, beside the communication resource row;
* the summary line leads with the week, the month and the year at both δ;
* space pools to a count (#4418's ``pooledShort``): at p = 1% + 2^-128 a machine-week of 60 s windows certifies at least
  9,948 of its 10,080 at δ = 1% and 1,000 machine-years 520,337,076 of 5.256·10⁸, memory's published figures;
* bandwidth fails over a period with probability ``T · eps``, not ``eps``;
* communication pays each window's own terms every window and the faulty proof blocks once (#4367).
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
    q = {"K": 28, "eps": "1/100", "windows": 1, "schedules": 4, "cells": 3,
         "audits": [{"units": 3, "drawn": 3, "free": 1, "jitter": 2, "faulty_sets": 2, "faulty_bits": 16}]}
    windows = [report(0, "accept"), report(1, "accept")]
    for w, (verdict, quantity) in zip(windows, (("accept", q), ("reject", None))):
        w["protocols"]["communication"] = {"verdict": verdict, "protocol": C.CERTIFIER, "quantity": quantity,
                                           "owner": "network-accounting"}
    m = M.monitoring(windows, DEP)
    c = m["communication"]
    assert m["resources"]["communication"]["observed"]["accepted"] == 1
    assert (c["periods"], c["accepted"], c["refused"], c["period_s"]) == (2, 1, 1, DEP["window_s"])
    assert [c[p]["periods"] for p, _ in M.PERIODS] == [168, 720, 8760]
    for p, _ in M.PERIODS:
        assert set(c[p]["pooled"]) == set(c[p]["fleet"]) == {str(d) for d in M.DELTAS}
        assert c[p]["fleet"]["1/100"]["machines"] == DEP["machines"]
    assert c["week"]["pooled"]["1/100"] == C.horizon_leak([q], Fraction(1, 100), 168)


MEMORY_P = Fraction(1, 100) + Fraction(1, 2**128)


@pytest.mark.parametrize("t, delta, certified", [(10_080, Fraction(1, 100), 9_948), (10_080, Fraction(1, 10**6), 9_923),
                                                 (525_600, Fraction(1, 100), 520_124),
                                                 (525_600_000, Fraction(1, 100), 520_337_076),
                                                 (525_600_000, Fraction(1, 10**6), 520_332_005)])
def test_space_pools_to_memorys_count(t, delta, certified):
    assert t - M.short_windows(t, MEMORY_P, delta) + 1 == certified


def resource(name: str, quantity: dict | None, verdict: str = "accept", protocol: str | None = None) -> dict:
    return {"verdict": verdict, "quantity": quantity if verdict == "accept" else None, "protocol": protocol}


def test_the_space_row_certifies_a_count_of_the_periods_windows():
    dep = {"id": "one-machine-60s", "machines": 1, "window_s": 60}
    r = M.space([resource("space", {"certified": True, "eps": str(MEMORY_P), "s_bits": 4096})] * 2, dep)
    assert r["periods"]["week"]["windows"] == 10_080
    assert r["periods"]["week"]["1/100"]["certified_windows_at_least"] == 9_948
    assert r["periods"]["week"]["1/100"]["uncovered_at_most"] == "0.0131"
    assert r["statement"]["pr"] == 4418 and "FixedBefore" in r["statement"]["hypothesis"]
    assert M.space([resource("space", None, "reject")], dep)["periods"] is None


def test_bandwidth_fails_with_t_times_eps():
    dep = {"id": "one-machine-60s", "machines": 1, "window_s": 60}
    eps = Fraction(2**24 + 64, 2**64)
    r = M.bandwidth([resource("bandwidth", {"eps": str(eps), "q_bytes": 1 << 20})], dep)
    assert float(r["periods"]["week"]["except_with"]) == pytest.approx(float(10_080 * eps), rel=1e-3)
    assert r["periods"]["year"]["vacuous"] is False and r["periods"]["week"]["q_bytes_each"] == 1 << 20


def test_communication_pays_the_faulty_units_once():
    public = {"K": 1000, "cells": 3, "schedules": 7, "eps": "1/100", "windows": 2,
              "audits": [{"free": 0, "jitter": 10, "faulty_sets": 400, "faulty_bits": 580, "units": 1024, "drawn": 64,
                          "wrong_units": 36, "delta": "1/100"}]}
    r = M.communication([resource("communication", public, protocol=M.CERTIFIER)], DEP)
    week = r["periods"]["week"]
    t = week["windows"]
    pooled = week["1/100"]["pooled"]
    assert pooled["every_window_bits"] == 20 and pooled["faulty_units_at_most"] == 74
    assert pooled["faulty_once_bits"] == 74 * (16 + math.ceil(math.log2(t * 1024 + 1)))
    assert pooled["K_bits"] == 20 * t + pooled["faulty_once_bits"] < week["per_window"]["K_bits"] == 1000 * t
    assert "hesc" in r["statement"]["conditional"]
    other = M.communication([resource("communication", {"K": 1}, protocol="verity/network-certificate/v4")], DEP)
    assert other["periods"] is None


def test_the_line_carries_each_resource_row_the_run_has():
    rep = report(0, "accept")
    rep["protocols"]["space"] = resource("space", {"certified": True, "eps": "1/100", "s_bits": 8})
    rep["protocols"]["bandwidth"] = resource("bandwidth", None, "missing")
    lead = M.lead(M.monitoring([rep], DEP))
    assert set(lead["space"]["week"]) == {"1/100", "1/1000000"} and lead["bandwidth"] == {"verdict": "missing"}
