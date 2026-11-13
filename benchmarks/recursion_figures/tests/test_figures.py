"""The session's figures (`figures`): one recursion bench result's session scaled over the census's fleets and the
monitoring periods, with each figure's basis."""
from __future__ import annotations

import json
import math

import figures as F
import pytest

PARTS = ["L0", "L1", "alg-p0", "cap"]


def test_log2_binom_sum_is_the_stream_bound():
    assert math.isclose(F.log2_binom_sum(5, 2), 4.0)
    assert F.log2_binom_sum(7, 7) == 7.0 and F.log2_binom_sum(7, 9) == 7.0
    assert math.isclose(F.log2_binom_sum(12, 1), math.log2(13))
    assert math.isclose(F.log2_binom_sum(3000, 40), math.log2(sum(math.comb(3000, i) for i in range(41))))


def _doc(parts=4, ands=1000, n=3, per_vu=1) -> dict:
    return {"staged": {"n": n, "units_per_vu": per_vu, "unit_ands": ands, "subcircuit": "s", "unit": "u"},
            "rows": [{"step": "oprove", "stage": f"outer {p}", "proof_bytes": 100} for p in PARTS[:parts]],
            "release": {"ok": True, "schedule_ms": 9000, "parts": [{"part": p, "slack_ms": 500} for p in PARTS[:parts]]}}


def test_figures_scale_one_session_to_every_machine_window():
    times = {s: {"wall_s": 10.0, "cpu_s": 60.0} for s in F.SESSION_STEPS}
    fig = F.figures(_doc(), times)
    s = fig["session"]
    assert (s["wall_s"], s["cpu_s"], s["statement"]["ands"], s["parts"]) == (50.0, 300.0, 3000, 4)
    fl = F.fleets()
    dc, world = fig["fleets"]["datacenter"], fig["fleets"]["world"]
    w = dc["fleet"]["window_s"]
    assert dc["window"]["cost"]["cores_per_machine"] == 300.0 / w and dc["window"]["cost"]["fits_window"]
    assert dc["window"]["leak"]["bits"] == math.log2(5)
    t = math.log2(fl["world"]["gpus"] * w) + F.HASH_RATE_LOG2
    kappa = F.log2_add(F.KAPPA_LOG2, F.KNOWLEDGE_LOG2)
    assert dc["window"]["ks"] == world["window"]["ks"]
    assert dc["window"]["ks"]["world"]["error_log2"] == round(F.log2_add(kappa, t + F.EPS_LOG2), 2)
    year = dict(F.PERIODS)["year"]
    y = dc["periods"]["year"]
    R = year // w
    assert y["sessions"] == fl["datacenter"]["machines"] * R and y["covered_ands"] == y["sessions"] * 3000
    assert y["cost"]["cpu_s"] == y["sessions"] * 300.0 and y["cost"]["core_years"] == y["sessions"] * 300.0 / year
    assert y["leak"]["bits_per_stream"] == F.log2_binom_sum(R * 4, R)
    assert y["leak"]["bits"] == fl["datacenter"]["machines"] * y["leak"]["bits_per_stream"]
    for a in F.ADVERSARIES:
        T = math.log2(fl[a]["gpus"] * year) + F.HASH_RATE_LOG2
        k = y["ks"][a]
        assert k["error_log2"] == round(F.log2_add(math.log2(y["sessions"]) + kappa, T + F.EPS_LOG2), 2)
        assert math.isclose(k["share_per_and_log2"], k["error_log2"] - math.log2(y["covered_ands"]), abs_tol=0.02)
    assert world["periods"]["year"]["sessions"] == fl["world"]["machines"] * R
    assert F.figures(_doc(), times, budget=2)["fleets"]["world"]["periods"]["week"]["leak"]["budget"] == 2
    assert "datacenter | year" in F.markdown(fig)


def test_the_ks_error_at_the_census_pools_t_eps_over_the_period():
    """κ + t·ε pooled: against the world's calls, the window, week, month and year; against the datacenter's own, the
    year; and the year's share of the sessions, a machine-window each, for the datacenter and the world."""
    times = {s: {"wall_s": 10.0, "cpu_s": 60.0} for s in F.SESSION_STEPS}
    fl = F.figures(_doc(), times)["fleets"]
    world = [fl["world"]["window"]["ks"]["world"]["error_log2"]] + [
        fl["world"]["periods"][p]["ks"]["world"]["error_log2"] for p in ("week", "month", "year")]
    assert [round(x, 1) for x in world] == [-107.9, -97.9, -95.8, -92.2]
    assert round(fl["datacenter"]["periods"]["year"]["ks"]["datacenter"]["error_log2"], 1) == -100.4
    assert [round(fl[f]["periods"]["year"]["ks"]["world"]["share_per_session_log2"], 1)
            for f in ("datacenter", "world")] == [-121.5, -129.7]


def test_the_served_window_counts_every_element_of_its_blocks():
    """The served window's statement: 3 instances of `SiluMul_v2{I=8192}`, each a VU of 8,192 silu-mul units of 1,380
    ANDs, so 33,914,880 ANDs, and every per-AND figure over that count."""
    times = {s: {"wall_s": 10.0, "cpu_s": 60.0} for s in F.SESSION_STEPS}
    fig = F.figures(_doc(ands=1380, n=3, per_vu=8192), times)
    assert fig["session"]["statement"]["ands"] == 33_914_880
    dc = fig["fleets"]["datacenter"]
    assert dc["window"]["cost"]["cpu_s_per_and"] == 300.0 / 33_914_880
    assert dc["periods"]["year"]["covered_ands"] == dc["periods"]["year"]["sessions"] * 33_914_880
    assert "3 x 8192 x u (3.39e+07 ANDs proved)" in F.markdown(fig)


def test_a_statement_without_units_per_vu_has_no_and_count():
    doc = _doc()
    del doc["staged"]["units_per_vu"]
    times = {s: {"wall_s": 10.0, "cpu_s": 60.0} for s in F.SESSION_STEPS}
    fig = F.figures(doc, times)
    assert fig["session"]["statement"]["ands"] is None
    assert fig["fleets"]["datacenter"]["window"]["cost"]["cpu_s_per_and"] is None


def test_a_session_missing_a_step_has_no_total():
    times = {s: {"wall_s": 1.0, "cpu_s": 1.0} for s in F.SESSION_STEPS[:-1]}
    fig = F.figures(_doc(), times)
    assert fig["session"]["cpu_s"] is None and fig["fleets"]["world"]["periods"]["year"]["cost"]["cpu_s"] is None


def test_a_result_gives_its_figures_and_its_table(tmp_path, capsys):
    """`of_result` and the command line read a result's `session_times` and its release budget, as the bench writes them."""
    times = {s: {"wall_s": 10.0, "cpu_s": 60.0, "peak_rss_gb": 1.0} for s in F.SESSION_STEPS}
    doc = _doc() | {"session_times": times, "params": {"release_budget": "2"}}
    fig = F.of_result(doc)
    assert fig == F.figures(_doc(), times, budget=2)
    r = tmp_path / "recursion_e2e.json"
    r.write_text(json.dumps(doc))
    assert F.main([str(r)]) == 0 and "datacenter | year" in capsys.readouterr().out
    assert F.main([str(r), "--json"]) == 0 and json.loads(capsys.readouterr().out) == json.loads(json.dumps(fig))


def test_a_result_without_session_times_is_refused():
    with pytest.raises(ValueError, match="session_times"):
        F.of_result(_doc())
