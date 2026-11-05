"""The session's figures (`figures`) and the release side as `rec_profile` reads it from 85's RELEASE=1 records
(oprove's release.json and release-wall.tsv, the inner step's schedule and coin-tree audit)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import figures as F
import rec_profile as RP
from test_recursion_e2e import job

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


def _release(o: Path, ok: bool = True, held: bool = True) -> None:
    (o / "params.txt").write_text((o / "params.txt").read_text() + "release=1 release_budget=\n")
    op = o / "steps" / "oprove" / "r1" / "out" / "oprove"
    rows = [{"part": p, "window_ms": 1000, "release_ms": 2000 + 1000 * j, "ready_ms": 1500 + 1000 * j, "decided": "go",
             "record": ["release"], "slack_ms": 500, "loop_release_ms": 2000 + 1000 * j} for j, p in enumerate(PARTS)]
    (op / "release.json").write_text(json.dumps({"session": "r0000", "stream": "circuit/rep0", "start_ms": 1000, "parts": rows,
                                                 "released": PARTS, "stops": [], "inner_finished": True, "ok": ok,
                                                 "schedule_ms": 4000}))
    (op / "release-wall.tsv").write_text("".join(f"{p}\t{p}\t{1000 * j}\t{1000 * j + 400}\t0\n" for j, p in enumerate(PARTS)))
    i = o / "steps" / "inner" / "r1" / "out" / "inner"
    (i / "release.json").write_text(json.dumps({"parts": PARTS, "windows_ms": [1000] * 4}))
    (i / "tree-source.txt").write_text("local 127.0.0.1:9\n")
    (i / "tree-held.json").write_text(json.dumps({"ok": held}))


def test_the_release_side_is_read_and_holds_the_headline(tmp_path):
    o = job(tmp_path)
    _release(o)
    doc = RP.read(o)
    rel = doc["release"]
    assert rel["ok"] and rel["trees_held"] and rel["tree_source"].startswith("local")
    assert [p["prove_wall_s"] for p in rel["parts"]] == [0.4] * 4 and rel["schedule"]["parts"] == PARTS
    assert doc["headline"]["released"] is True
    md = RP.markdown(doc)
    assert "**Release side.**" in md and "| L0 | 1.0 | 1.0 | 0.5 | 0.5 | 0.4 | go | release |" in md
    assert "The session's figures" in md


def test_a_release_not_held_or_a_tree_not_the_challengers_is_no_accepted_run(tmp_path):
    for ok, held in ((False, True), (True, False)):
        o = job(tmp_path / f"{ok}{held}")
        _release(o, ok, held)
        doc = RP.read(o)
        assert doc["release"]["ok"] is False and doc["headline"]["released"] is False and doc["headline"]["ok"] is False


def test_release_1_without_a_report_is_no_accepted_run(tmp_path):
    o = job(tmp_path)
    (o / "params.txt").write_text((o / "params.txt").read_text() + "release=1\n")
    doc = RP.read(o)
    assert "release" not in doc and doc["headline"]["ok"] is False
