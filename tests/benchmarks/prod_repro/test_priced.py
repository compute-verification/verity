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


def lifetime(out: Path, name: str, seconds: float, *, mode: str | None, decode: list[tuple[int, int]] = ()) -> None:
    """A lifetime: rep 1 left out, rep 2 counted from t=100 for 10 s; in rep 2's window one decode step a pair of
    ``decode`` (requests, start), each request 6 rows, and one prefill step of 4,000 rows."""
    d = out / name
    for rep, t0 in ((1, 0.0), (2, 100.0)):
        p = d / "agentic@4" / f"rep-{rep}"
        p.mkdir(parents=True)
        (p / "meta.json").write_text(json.dumps({"load": "agentic@4", "rep": rep, "t0_unix": t0, "wall_s": 10.0}))
        (p / "result.json").write_text(json.dumps({"duration": seconds}))
    (d / "meta.json").write_text(json.dumps({"left_out": 1, "env": {"VERITY_DSA_TOPK": mode} if mode else {},
                                             "server_argv": ["--tensor-parallel-size", "8"]}))
    recs = [{"kind": "header", "config": {"speculative": {"num_speculative_tokens": 5}}},
            {"kind": "step", "step": 0, "rows": [["p", 0, 4000]], "t": 101.0}]
    recs += [{"kind": "step", "step": i + 1, "rows": [[f"r{j}", start, 6] for j in range(n)], "t": 102.0 + i}
             for i, (n, start) in enumerate(decode)]
    recs.append({"kind": "step", "step": 99, "rows": [["r0", 5000, 6]], "t": 5.0})
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
