"""`wipe_monitoring`: one accepted wipe over a week, a month and a year of a deployment. The pooled count is the least M
with C(T, M)·p^M within δ, against hand counts at small T; an HBM-only wipe's table names its tier and device, every wipe
meeting its bound except with T·eps; the cadence is a parameter; an uncounted wipe stays vacuous, since every egress
record is refused, a covered one over the session's own real-time window too; a rejected wipe pools nothing; the periods
are the reporting grid's `PERIODS`."""

from __future__ import annotations

import json
import math
from fractions import Fraction

import pytest

import pous_test_signer as ts
from verity.protocols.pous.device import memory_challenger as mc
from verity.protocols.pous.verifier import streamed_wipe as sw
from verity.protocols.pous.verifier import wipe_monitoring as WM
from verity.protocols.pous.verifier import wipe_party as WP
from verity.protocols.pous.verifier import wipe_period as W
from verity.protocols import reporting

HBM = sw.WipeParams(64, W.READS, 1, 1_000, ((WM.HBM, 64 * sw.LABEL_BYTES),))
STORE = {"backend": "cuda", "device": "NVIDIA H200", "tier": WM.HBM, "uuid": "GPU-test", "nvml_total_bytes": 64 * 64,
         "cuda_total_bytes": 64 * 64, "total_bytes": 64 * 64}
LINK = WP.link("control", "n0", "outside", "eth0")


def _answer(egress_bits: int) -> dict:
    """An accepted HBM-only wipe of `HBM`, its egress signed as `egress_bits`, with the placement a GPU leg records."""
    w = W.run("n0", HBM, signer=ts.TestSigner(b"wipe monitoring", 2), log=ts.session_log(), schemes=ts.SCHEMES,
              protocol=mc.TEST_PREFIX + W.PROTOCOL, clock=iter(range(10 ** 9, 10 ** 10)).__next__,
              coin_secret=b"coins", egress_bits=egress_bits)
    assert w.verdict == "accept"
    where = WP.placement("n0", "auditor", store=STORE, links=[LINK])
    return w.answer | {"profile": w.answer["profile"] | {"placement": where}}


@pytest.fixture(scope="module")
def counted():
    return _answer(2 * sw.EGRESS_UNIT_BITS)


@pytest.fixture(scope="module")
def uncounted():
    return _answer(W.UNCOUNTED)


@pytest.mark.parametrize("t,p,delta,m", [
    (1, Fraction(1, 2), Fraction(1, 2), 1),       # 1·1/2 ≤ 1/2
    (5, Fraction(0), Fraction(1, 100), 1),        # nothing fails
    (7, Fraction(1, 10), Fraction(1, 100), 4),    # 7/10, 21/100, 35/1000 > 1/100 ≥ 35/10^4
    (10, Fraction(1, 5), Fraction(1, 1000), 8),   # C(10, 7)/5^7 = 0.001536 > 1/1000 ≥ C(10, 8)/5^8 = 0.0001152
    (3, Fraction(1, 2), Fraction(1, 100), 4),     # 3/2, 3/4, 1/8: no count within δ, so all 3 may fail
])
def test_the_pooled_count_is_the_hand_counted_least_within_delta(t, p, delta, m):
    got, bound = WM.short_sessions(t, p, delta)
    assert got == m
    step = WM.pooled_step(t, p, delta)
    assert step["failing_at_most"] == min(m - 1, t) and step["meeting_at_least"] == t - min(m - 1, t)
    assert bound == (1 if m > t else math.comb(t, m) * p ** m)


def test_an_hbm_only_wipe_over_a_week_a_month_and_a_year(counted):
    public = counted["public"]
    assert public["certified"] and public["eps"] == str(W.EPS) and public["egress_mib"] == 2
    r = WM.table(counted, 3, deployment="three H200 nodes")
    assert r["format"] == WM.FORMAT and r["deployment"] == {"id": "three H200 nodes", "machines": 3, "cadence_s": 86400}
    assert r["memory"] == {"tiers": [{"tier": "gpu-hbm", "bytes": 64 * 64}], "tier": "gpu-hbm", "hbm_only": True,
                           "devices": [{"tier": "gpu-hbm", "device": "NVIDIA H200", "uuid": "GPU-test"}]}
    f, m = public["foreign_bits"], public["memory_bits"]
    assert m == 64 * 512 and r["each"] | {"eps_float": None} == {
        "foreign_bits": f, "memory_bits": m, "wiped_bits": m - f, "wiped_share": f"{(m - f) / m:.4g}", "eps": str(W.EPS),
        "eps_float": None, "k": W.READS, "r": public["r"], "egress_mib": 2, "certified": True}
    for name, days in (("week", 7), ("month", 30), ("year", 365)):
        row = r["periods"][name]
        t = 3 * days
        assert (row["wipes"], row["wipes_per_machine"]) == (t, days)
        for d in reporting.DELTAS:
            assert row[str(d)]["failing_at_most"] == 0 and row[str(d)]["meeting_at_least"] == t
        assert row["per_wipe"]["except_with"] == f"T·eps = {float(t * W.EPS):.3g}"
    assert r["leaves_out"][0] == WM.EXCEPT and any(s.startswith("link control") for s in r["leaves_out"][1:])


def test_the_cadence_is_a_parameter(counted):
    half = WM.table(counted, 2, 43_200)["periods"]
    assert [half[n]["wipes"] for n in ("week", "month", "year")] == [28, 120, 1460]
    rare = WM.table(counted, 2, 30 * 86400)["periods"]
    assert "doesn't fit a week" in rare["week"]["why"] and rare["month"]["wipes"] == 2


def test_the_periods_are_the_reporting_grids():
    assert WM.PERIODS is reporting.PERIODS


def test_the_table_pools_at_the_answers_eps(counted):
    """A week of one machine at eps = 1/10 is the hand count above: T = 7, so at most 3 fail at δ = 1%."""
    loose = counted | {"public": counted["public"] | {"eps": "1/10"}}
    week = WM.table(loose, 1)["periods"]["week"]
    assert week["1/100"]["failing_at_most"] == 3 and week["per_wipe"]["except_with"] == "T·eps = 0.7"


RECORD = {"format": WM.EGRESS, "covered": True, "egress_mib": 5, "egress_bits": 5 * sw.EGRESS_UNIT_BITS,
          "link": "uplink", "delta": "0", "why": []}
"""A covered `wipe-egress/v1` record: its writer's verdict, its count and its link."""
WINDOW = {"t0_unix_ns": 1_793_088_000_123_456_789, "t1_unix_ns": 1_793_088_183_000_001_000, "pad_ns": 10 ** 9}
"""The wipe's own session in real time, as its challenger's side reports it beside the signed session
(`wipe_party.report`), in the fields a record's writer gives its window."""


def test_an_uncounted_wipe_stays_vacuous_and_a_covered_egress_record_is_refused(counted, uncounted):
    vacuous = WM.table(uncounted, 1)
    assert vacuous["periods"] is None and "outside 0 to 32766" in vacuous["why"]
    with pytest.raises(ValueError) as refusal:
        WM.counted(uncounted, RECORD)
    said = str(refusal.value)
    assert said == WM.REFUSED and said.startswith(f"the {WM.EGRESS} record is refused")
    assert "its monotonic clock (CLOCK_MONOTONIC ns" in said and "(t0_unix_ns, t1_unix_ns) is real time (Unix ns)" in said
    assert "until what the memory challenger signs ties its session to the span the record counts" in said
    assert "no --egress" in said
    for accepted in (uncounted, counted):
        with pytest.raises(ValueError, match="nothing signed says the record covers this wipe's session"):
            WM.table(accepted, 1, egress=RECORD)


def test_a_record_over_the_sessions_own_real_time_window_is_refused_too(uncounted):
    """The challenger signs the session's times on its monotonic clock, and the window that says when they were in real
    time is under no signature. So a record over exactly that window is refused as any other is: holding a record's
    `t0_unix_ns` and `t1_unix_ns` against the signed times would compare two clocks."""
    with pytest.raises(ValueError, match="comparing the two would compare two clocks"):
        WM.counted(uncounted, RECORD | WINDOW)
    with pytest.raises(ValueError, match="comparing the two would compare two clocks"):
        WM.table(uncounted, 1, egress=RECORD | WINDOW)


def test_a_rejected_wipe_pools_nothing():
    r = WM.table({"verdict": "reject", "reason": "a late read", "public": None, "profile": None}, 1)
    assert r["periods"] is None and "a late read" in r["why"]


def test_the_cli_reads_a_runs_wipe_json_and_refuses_an_egress_record(tmp_path, capsys, counted):
    (tmp_path / "wipe.json").write_text(json.dumps({"answer": counted, "uncertified": []}))
    argv = [str(tmp_path / "wipe.json"), "--machines", "2", "--out", str(tmp_path / "m.json")]
    assert WM.main(argv) == 0
    written = (tmp_path / "m.json").read_text()
    got = json.loads(written)
    assert capsys.readouterr().out == written
    assert got["periods"]["month"]["wipes"] == 60 and got["each"]["egress_mib"] == 2 and got["egress"] is None
    (tmp_path / "egress.json").write_text(json.dumps(RECORD | WINDOW))
    with pytest.raises(ValueError, match="record is refused"):
        WM.main([*argv, "--egress", str(tmp_path / "egress.json")])
    assert (tmp_path / "m.json").read_text() == written and capsys.readouterr().out == ""
