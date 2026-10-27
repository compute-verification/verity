"""`wipe_monitoring`: one accepted wipe over a week, a month and a year of a deployment. The pooled count is the least M
with C(T, M)·p^M within δ, against hand counts at small T; an HBM-only wipe's table names its tier and device, every wipe
meeting its bound except with T·eps; the cadence is a parameter; an uncounted wipe is vacuous until a covered egress
record counts it; a rejected wipe pools nothing."""

from __future__ import annotations

import json
import math
from fractions import Fraction

import pytest

import pous_test_signer as ts
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw
from verity.core.protocols.pous import wipe_monitoring as WM
from verity.core.protocols.pous import wipe_party as WP
from verity.core.protocols.pous import wipe_period as W

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
        for d in WM.DELTAS:
            assert row[str(d)]["failing_at_most"] == 0 and row[str(d)]["meeting_at_least"] == t
        assert row["per_wipe"]["except_with"] == f"T·eps = {float(t * W.EPS):.3g}"
    assert r["leaves_out"][0] == WM.EXCEPT and any(s.startswith("link control") for s in r["leaves_out"][1:])


def test_the_cadence_is_a_parameter(counted):
    half = WM.table(counted, 2, 43_200)["periods"]
    assert [half[n]["wipes"] for n in ("week", "month", "year")] == [28, 120, 1460]
    rare = WM.table(counted, 2, 30 * 86400)["periods"]
    assert "doesn't fit a week" in rare["week"]["why"] and rare["month"]["wipes"] == 2


def test_the_table_pools_at_the_answers_eps(counted):
    """A week of one machine at eps = 1/10 is the hand count above: T = 7, so at most 3 fail at δ = 1%."""
    loose = counted | {"public": counted["public"] | {"eps": "1/10"}}
    week = WM.table(loose, 1)["periods"]["week"]
    assert week["1/100"]["failing_at_most"] == 3 and week["per_wipe"]["except_with"] == "T·eps = 0.7"


def test_an_uncounted_wipe_is_vacuous_until_an_egress_record_counts_it(uncounted):
    vacuous = WM.table(uncounted, 1)
    assert vacuous["periods"] is None and "outside 0 to 32766" in vacuous["why"]
    record = {"format": WM.EGRESS, "covered": True, "egress_mib": 5, "egress_bits": 5 * sw.EGRESS_UNIT_BITS,
              "link": "warden", "delta": "0", "why": []}
    r = WM.table(uncounted, 1, egress=record)
    assert r["each"]["foreign_bits"] == W.certify(HBM, 5)["foreign_bits"] and r["each"]["certified"]
    assert r["egress"] == {"link": "warden", "egress_bits": 5 * sw.EGRESS_UNIT_BITS, "egress_mib": 5, "delta": "0",
                           "covered": True}
    assert r["periods"]["year"]["1/1000000"]["meeting_at_least"] == 365
    with pytest.raises(ValueError, match="not a covered"):
        WM.counted(uncounted, record | {"covered": False, "why": ["a window lacks its certificate"]})


def test_an_egress_record_replaces_only_an_uncounted_egress(counted, uncounted):
    record = {"format": WM.EGRESS, "covered": True, "egress_mib": 5}
    with pytest.raises(ValueError, match="counted when it was signed"):
        WM.counted(counted, record)
    other = uncounted | {"profile": uncounted["profile"] | {"k": W.READS - 1}}
    with pytest.raises(ValueError, match="profile doesn't give its outputs"):
        WM.counted(other, record)


def test_a_rejected_wipe_pools_nothing():
    r = WM.table({"verdict": "reject", "reason": "a late read", "public": None, "profile": None}, 1)
    assert r["periods"] is None and "a late read" in r["why"]


def test_the_cli_reads_a_legs_wipe_json(tmp_path, capsys, uncounted):
    (tmp_path / "wipe.json").write_text(json.dumps({"answer": uncounted, "uncertified": []}))
    (tmp_path / "egress.json").write_text(json.dumps({"format": WM.EGRESS, "covered": True, "egress_mib": 1}))
    assert WM.main([str(tmp_path / "wipe.json"), "--machines", "2", "--egress", str(tmp_path / "egress.json"),
                    "--out", str(tmp_path / "m.json")]) == 0
    got = json.loads((tmp_path / "m.json").read_text())
    assert got == json.loads(capsys.readouterr().out)
    assert got["periods"]["month"]["wipes"] == 60 and got["each"]["egress_mib"] == 1
