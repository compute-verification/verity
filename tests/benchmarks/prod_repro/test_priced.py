"""The serving cost priced from counts, on a synthetic job: unit times at four shapes, two lifetimes of ours and its
reference, and the top-k's counters."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import priced as Q
import pytest

#: (rows, keys): (base present, base cut, a call with every row failed, present then cut), µs over the fast kernel
SHAPES = {(6, 20000): (7.0, 12.0, 100.0, 22.0), (6, 131072): (12.0, 16.0, 1000.0, 50.0),
          (24, 20000): (10.0, 15.0, 100.0, 34.0), (24, 131072): (24.0, 28.0, 1000.0, 110.0)}


def units_file(tmp: Path) -> Path:
    lines = [{"K": 2048}]
    for (rows, keys), (bp, bc, fp, fc) in SHAPES.items():
        for dist, d in (("randn", 0.0), ("indexer", 0.5)):
            lines.append({"dist": dist, "rows": rows, "keys": keys, "fast_us": 10.0,
                          "present": {"added_us": bp + d, "per_call": {"rows_long": rows}},
                          "cut": {"added_us": bc + d, "per_call": {"rows_long": rows}}})
        lines.append({"dist": "shift32", "rows": rows, "keys": keys, "fast_us": 10.0,
                      "present": {"added_us": 900.0, "per_call": {"rows_bisected": rows}},
                      "cut": {"added_us": 40.0, "per_call": {"rows_refined": rows}}})
        for how in ("missed", "overflow", "garbage"):
            lines.append({"forced": how, "rows": rows, "keys": keys,
                          "present": {"us": fp}, "cut": {"us": fc if how != "garbage" else fp}})
    f = tmp / "topk_cut.jsonl"
    f.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return f


def lifetime(out: Path, name: str, seconds: float, *, mode: str | None, decode: list[tuple[int, int]] = (),
             prefill_t: float = 101.0, lead: bool = False) -> None:
    """A lifetime: rep 1 left out, rep 2 counted from t=100 for 10 s; in rep 2's window one decode step a pair of
    ``decode`` (requests, start), each request 6 rows, and one prefill step of 4,000 rows ending at ``prefill_t``,
    after a step of one short row at t=100.5 where ``lead``."""
    d = out / name
    for rep, t0 in ((1, 0.0), (2, 100.0)):
        p = d / "agentic@4" / f"rep-{rep}"
        p.mkdir(parents=True)
        (p / "meta.json").write_text(json.dumps({"load": "agentic@4", "rep": rep, "t0_unix": t0, "wall_s": 10.0}))
        (p / "result.json").write_text(json.dumps({"duration": seconds}))
    (d / "meta.json").write_text(json.dumps({"left_out": 1, "env": {"VERITY_DSA_TOPK": mode} if mode else {},
                                             "server_argv": ["--tensor-parallel-size", "8"]}))
    recs = [{"kind": "header", "config": {"speculative": {"num_speculative_tokens": 5}}}]
    if lead:
        recs.append({"kind": "step", "step": 98, "T": 1, "rows": [["s", 10, 1]], "t": 100.5})
    recs.append({"kind": "step", "step": 0, "T": 4000, "rows": [["p", 0, 4000]], "t": prefill_t})
    recs += [{"kind": "step", "step": i + 1, "T": 6 * n, "rows": [[f"r{j}", start, 6] for j in range(n)],
              "t": 102.0 + i} for i, (n, start) in enumerate(decode)]
    recs.append({"kind": "step", "step": 99, "T": 6, "rows": [["r0", 5000, 6]], "t": 5.0})
    (d / "server").mkdir()
    with gzip.open(d / "server" / "step_trace.1.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))


def job(tmp: Path, counters: dict | None, layers: list[dict] | None = None) -> Path:
    out = tmp / "out"
    for k in (1, 2):
        lifetime(out, f"prod-trace.{k}", 50.0, mode=None)
        lifetime(out, f"prod-record.{k}", 52.0, mode="canonical", decode=[(4, 131066), (1, 19994)])
    if counters is not None:
        (out / "topk_counters.json").write_text(json.dumps({"dirs": {f"prod-record.{k}": counters for k in (1, 2)}}))
    (out / "prod_repro.json").write_text(json.dumps({"run": "r-ab", "layers": {"each": layers or []}}))
    return out


def test_units_are_each_pass_range_over_the_distributions_and_a_failed_call_less_its_base(tmp_path):
    u = Q.units(units_file(tmp_path))
    assert u["present"]["base"][(6, 20000)] == (7.0, 7.5)
    assert u["cut"]["base"][(24, 131072)] == (28.0, 28.5)
    assert u["present"]["failed"][(6, 20000)] == (92.5, 93.0)
    assert u["cut"]["failed"][(24, 131072)] == (81.5, 82.0)


def test_at_is_linear_between_shapes_holds_rows_and_extends_keys(tmp_path):
    t = Q.units(units_file(tmp_path))["present"]["base"]
    assert Q.at(t, 6, 20000, 0) == 7.0
    assert Q.at(t, 15, 20000, 0) == pytest.approx(8.5)
    assert Q.at(t, 48, 20000, 0) == Q.at(t, 24, 20000, 0)
    assert Q.at(t, 6, 10000, 0) == 7.0
    assert Q.at(t, 6, 131072 + 111072, 0) == pytest.approx(17.0)


def test_steps_count_decode_rows_past_k_and_their_longest(tmp_path):
    out = job(tmp_path, None)
    s, n = Q.steps(out / "prod-record.1" / "server" / "step_trace.1.jsonl.gz")
    assert n == 6
    assert [(x["rows"], x["keys"]) for x in s] == [(0, 0), (24, 131072), (6, 20000), (6, 5006)]


def test_without_failed_rows_only_the_base_is_priced_and_no_point_stands(tmp_path, capsys):
    out = job(tmp_path, {"rows_long": 1, "rows_tied": 0, "rows_at": 1, "calls": 1})
    assert Q.main([str(out), "--units", str(units_file(tmp_path)), "--ours", "prod-record"]) == 0
    r = json.loads((out / "priced.json").read_text())
    ly = r["layers"][0]
    assert not ly["failed_priced"] and r["point"] is None and "no failed rows" in r["no_point"]
    # the counters predate rows_refined, so canonical was the bisecting pass: 22 calls of each counted decode step
    x = ly["each"][0]
    assert x["pass"] == "present" and x["decode_steps"] == 2
    assert x["base_s"]["lo"] == pytest.approx(22 * (24.0 + 7.0) / 1e6)
    assert ly["cost"]["lo"] == pytest.approx(22 * (24.0 + 7.0) / 1e6 / 50.0)


def test_failed_rows_are_placed_between_packed_and_one_a_call(tmp_path):
    u = Q.units(units_file(tmp_path))
    t = u["present"]["failed"]
    steps_ = [(24, 131072), (6, 20000)]
    f = Q.failure_s(steps_, 30, t, 6, calls=2)
    # packed: cheapest a row first, the 6-row calls (92.5/6 a row) then the 24-row ones (976/24)
    assert f["lo"] == pytest.approx((12 * 92.5 / 6 + 18 * 975.5 / 24) / 1e6)
    # one a call, the dearest first, each at the fewest rows timed: the two long calls, then the two short ones
    assert f["hi"] == pytest.approx((2 * 988.0 + 2 * 93.0) / 1e6)
    assert f["lo"] <= f["mid"] <= f["hi"]
    assert Q.failure_s(steps_, 0, t, 6) == dict.fromkeys(Q.ENDS, 0.0)


def test_the_point_is_the_priced_layer_and_the_measured_one_below_marked_modeled(tmp_path):
    below = {"arm": "prod-trace", "over": "prod", "value": 0.006, "ci95": [-0.01, 0.02], "pairs": 2}
    mine = {"arm": "prod-record", "over": "prod-trace", "value": 0.02, "ci95": [0.01, 0.03], "pairs": 2}
    out = job(tmp_path, {"rows_long": 1, "rows_tied": 0, "rows_bisected": 80, "rows_refined": 1600, "rows_at": 1,
                         "calls": 1}, [below, mine])
    assert Q.main([str(out), "--units", str(units_file(tmp_path)), "--ours", "prod-record"]) == 0
    r = json.loads((out / "priced.json").read_text())
    ly, p = r["layers"][0], r["point"]
    assert ly["failed_priced"] and ly["each"][0]["pass"] == "cut"
    # a card's share of the lifetime's rows, scaled to the counted pass's 30 of the trace's 36 decode rows
    assert ly["each"][0]["rows_failed_a_card"] == {"rows_bisected": pytest.approx(80 / 8 * 30 / 36),
                                                   "rows_refined": pytest.approx(1600 / 8 * 30 / 36)}
    c = ly["cost"]
    assert c["lo"] <= c["mid"] <= c["hi"]
    assert ly["measured"]["value"] == 0.02 and ly["measured"]["run"] == "r-ab"
    assert p["name"] == Q.METRIC and p["mark"] == "modeled" and p["tested_by"] == "r-ab"
    assert p["member"] == "glm-5.3-h200x8-prod-record-over-prod-priced"
    assert p["value"] == pytest.approx(100 * (c["mid"] + 0.006))
    assert p["ci95"] == pytest.approx([100 * (c["lo"] - 0.01), 100 * (c["hi"] + 0.02)])
    assert p["inputs"][0].startswith("modeled: ")


def test_a_lifetime_whose_server_ran_no_canonical_top_k_is_refused(tmp_path):
    out = job(tmp_path, None)
    with pytest.raises(SystemExit, match="no canonical top-k"):
        Q.main([str(out), "--units", str(units_file(tmp_path)), "--ours", "prod-trace"])


def steps_file(d: Path, lines: list[dict], stats=("rows_long", "rows_tied", "rows_bisected")) -> None:
    """The top-k's counts by step as `dsa_topk.steps_to` keeps them, beside the step trace."""
    (d / "server" / "dsa_topk_steps.7.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in [{"pid": 7, "stats": list(stats)}, *lines]))


def test_step_failure_is_placed_within_the_step_or_read_where_the_counts_say_how_many_calls_failed(tmp_path):
    t = Q.units(units_file(tmp_path))["present"]["failed"]
    # 12 failed rows among a step's 24-row calls: in one call, two (a request's 6 each), or twelve
    f = Q.step_failure_s(12, 24, 131072, t, 6)
    assert f["lo"] == pytest.approx(Q.at(t, 12, 131072, 0) / 1e6)
    assert f["hi"] == pytest.approx(12 * Q.at(t, 1, 131072, 1) / 1e6)
    assert f["lo"] <= f["mid"] <= f["hi"]
    read = Q.step_failure_s(12, 24, 131072, t, 6, read=3)
    assert read["lo"] == pytest.approx(3 * Q.at(t, 4, 131072, 0) / 1e6)
    assert read["hi"] == pytest.approx(3 * Q.at(t, 4, 131072, 1) / 1e6)
    # no more calls than a step makes, and none for no failed row
    assert Q.step_failure_s(100, 24, 131072, t, 6)["hi"] == pytest.approx(22 * Q.at(t, 100 / 22, 131072, 1) / 1e6)
    assert Q.step_failure_s(0, 24, 131072, t, 6) == dict.fromkeys(Q.ENDS, 0.0)


def test_counts_by_step_price_each_steps_own_failed_rows_on_its_slowest_card(tmp_path, capsys):
    out = job(tmp_path, {"rows_long": 1, "rows_tied": 0, "rows_bisected": 99999, "rows_at": 1, "calls": 1})
    for k in (1, 2):
        d = out / f"prod-record.{k}"
        steps_file(d, [
            # the prefill step: 30 failed rows on its slowest card, priced only at prefill shapes
            {"step": 0, "tokens": 4000, "device": "cuda:0", "rows_long": 9, "rows_tied": 0, "rows_bisected": 30},
            {"step": 0, "tokens": 4000, "device": "cuda:1", "rows_long": 9, "rows_tied": 0, "rows_bisected": 20},
            # the first decode step: 12 failed rows on cuda:1
            {"step": 1, "tokens": 24, "device": "cuda:0", "rows_long": 9, "rows_tied": 0, "rows_bisected": 6},
            {"step": 1, "tokens": 24, "device": "cuda:1", "rows_long": 9, "rows_tied": 0, "rows_bisected": 12},
            # the second's tokens are not the trace's: it joins nothing
            {"step": 2, "tokens": 7, "device": "cuda:0", "rows_long": 9, "rows_tied": 0, "rows_bisected": 5}])
    assert Q.main([str(out), "--units", str(units_file(tmp_path)), "--ours", "prod-record"]) == 0
    x = json.loads((out / "priced.json").read_text())["layers"][0]["each"][0]
    assert x["counts_by_step"] and x["rows_at"] is None and x["unjoined_steps"] == 1
    assert x["rows_failed_a_card"] == {"rows_bisected": 12} and x["prefill_rows_failed_a_card"] == {"rows_bisected": 30}
    t = Q.units(units_file(tmp_path))["present"]["failed"]
    assert x["failed_s"] == pytest.approx(Q.step_failure_s(12, 24, 131072, t, 6))
    assert x["prefill_s"] is None
    assert "from the top-k's counts by step" in capsys.readouterr().out


def test_prefill_steps_are_priced_only_at_prefill_unit_times_and_else_given_measured_apart(tmp_path, capsys):
    below = {"arm": "prod-trace", "over": "prod", "value": 0.006, "ci95": [-0.01, 0.02], "pairs": 2}
    out = tmp_path / "out"
    for k, extra in ((1, 0.5), (2, 1.0)):
        lifetime(out, f"prod-trace.{k}", 50.0, mode=None, lead=True)
        lifetime(out, f"prod-record.{k}", 52.0, mode="canonical", decode=[(4, 131066)], lead=True,
                 prefill_t=101.0 + extra)
    (out / "topk_counters.json").write_text(json.dumps({"dirs": {f"prod-record.{k}": {"rows_bisected": 8, "calls": 1}
                                                                  for k in (1, 2)}}))
    (out / "prod_repro.json").write_text(json.dumps({"run": "r-ab", "layers": {"each": [below]}}))
    units = str(units_file(tmp_path))
    assert Q.main([str(out), "--units", units, "--ours", "prod-record"]) == 0
    r = json.loads((out / "priced.json").read_text())
    ly = r["layers"][0]
    assert not ly["prefill_priced"] and "--prefill-units" in ly["prefill_refused"]
    # ours's prefill step took 0.5 s and 1.0 s longer than the reference's, of its 50 s
    assert ly["prefill_measured"]["value"] == pytest.approx(0.75 / 50) and ly["prefill_measured"]["mark"] == "measured"
    assert ly["prefill_measured"]["ci95"][0] < 0.75 / 50 < ly["prefill_measured"]["ci95"][1]
    p = r["point"]
    assert p["value"] == pytest.approx(100 * (ly["cost"]["mid"] + 0.75 / 50 + 0.006))
    assert any(i.startswith("measured, not priced: prod-record's prefill steps") for i in p["inputs"])
    assert "prefill steps not priced: no prefill unit time" in capsys.readouterr().out
    # with prefill unit times, the prefill step is priced at them and is no measured term of the point
    assert Q.main([str(out), "--units", units, "--prefill-units", units, "--ours", "prod-record"]) == 0
    r = json.loads((out / "priced.json").read_text())
    x = r["layers"][0]["each"][0]
    pb = Q.units(units_file(tmp_path))["present"]["base"]
    assert x["prefill_s"]["lo"] == pytest.approx(22 * Q.at(pb, 4000 - 2048, 4000, 0) / 1e6)
    assert not any(i.startswith("measured, not priced") for i in r["point"]["inputs"])


def test_the_measured_cost_adjusted_for_steps_is_paired_by_pass_and_marked_modelled(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    passes = []
    for life in (1, 2, 3):
        for rep, (steps, base) in enumerate(((2000, 80.0), (2200, 84.0)), start=2):
            passes.append({"arm": "prod-trace", "dir": f"prod-trace.{life}", "rep": rep, "steps": steps,
                           "seconds": base})
            # ours: 2% dearer at the same steps, and one pass of fewer steps, as seconds go with steps^0.5
            s = steps - 200 * (rep == 2)
            passes.append({"arm": "prod-record", "dir": f"prod-record.{life}", "rep": rep, "steps": s,
                           "seconds": 1.02 * base * (s / steps) ** 0.5 * (1 + 0.001 * life)})
        passes.append({"arm": "prod-record", "dir": f"prod-record.{life}", "rep": 1, "steps": 1, "seconds": 9.0,
                       "left_out": True})
    (out / "prod_repro.json").write_text(json.dumps({"ran": {"passes": {"passes": passes}}}))
    a = Q.adjusted(out, "prod-record", "prod-trace")
    assert a["mark"] == "modelled" and a["pairs"] == 3
    assert a["ci95"][0] < a["value"] < a["ci95"][1]
    assert 0.015 < a["value"] < 0.025
    assert Q.adjusted(out, "prod-record", "prod") is None
