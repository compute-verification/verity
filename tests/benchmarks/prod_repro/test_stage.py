"""A prod_repro run's result as the serving cost's capture and minify records, on `test_priced`'s synthetic job."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import stage as G
from test_priced import job, units_file

BELOW = {"arm": "prod-trace", "over": "prod", "value": 0.006, "ci95": [-0.01, 0.02], "pairs": 2}
COUNTS = {"rows_long": 1, "rows_tied": 0, "rows_bisected": 80, "rows_at": 1, "calls": 1}


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261117-120000-abcd")
    return tmp_path


def result(run: Path) -> dict:
    return json.loads((run / "result.json").read_text())


def test_a_capture_holding_what_a_price_reads_is_a_passed_capture_record_naming_its_minify_run(run: Path):
    out = job(run, COUNTS, [BELOW])
    (run / "result.json").write_text(json.dumps({
        "schema": G.RESULT, "run_id": "r20261117-120000-abcd", "measurements": [{"name": "m"}],
        "workload_fingerprint": {"benchmark": "prod_repro"},
        "validation": {"status": "passed", "detail": "tokens: reproduced"}}))
    assert G.main(["capture", str(out), "--minify", "r20261117-110000-0001", "--exact", "r20261117-113000-0002"]) == 0
    r = result(run)
    assert r["kind"] == "serving-cost-capture/v1" and r["validation"]["status"] == "passed"
    assert r["workload_fingerprint"] == {"benchmark": "prod_repro", "minify": "r20261117-110000-0001",
                                         "exact": "r20261117-113000-0002"}
    assert "the run's own: tokens: reproduced" in r["validation"]["detail"] and r["measurements"] == [{"name": "m"}]
    assert r["validation"]["evidence"] == ["out/prod_repro.json"]


def test_a_capture_missing_a_part_of_what_a_price_reads_fails_naming_each(run: Path):
    out = job(run, None)
    for f in (out / "prod-record.2" / "server").glob("step_trace.*"):
        f.unlink()
    assert G.main(["capture", str(out), "--minify", "r20261117-110000-0001"]) == 1
    v = result(run)["validation"]
    assert v["status"] == "failed"
    for gap in ("prod-record.1 kept no top-k counts", "prod-record.2 kept no step trace",
                "the job measured no prod-trace over prod with an interval"):
        assert gap in v["detail"]
    assert G.capture_gaps(job(run / "b", COUNTS, [BELOW]), ours=("prod-record-bisect",)) == [
        "no lifetime of prod-record-bisect"]


def test_a_capture_whose_run_failed_its_own_gates_fails_and_one_of_another_kind_is_refused(run: Path):
    out = job(run, COUNTS, [BELOW])
    (run / "result.json").write_text(json.dumps({"schema": G.RESULT, "run_id": "r", "measurements": [],
                                                 "validation": {"status": "failed", "detail": "a token differs"}}))
    assert G.main(["capture", str(out), "--minify", "r20261117-110000-0001"]) == 1
    assert "the run's own validation failed: a token differs" in result(run)["validation"]["detail"]
    with pytest.raises(SystemExit, match="is a serving-cost-capture/v1 result"):
        G.main(["minify", str(out), "--model", "glm53-standin"])
    with pytest.raises(SystemExit, match="is not a run id"):
        G.main(["capture", str(out), "--minify", "out/minify"])
    with pytest.raises(SystemExit, match="not this research run's out/"):
        G.main(["capture", str(run / "elsewhere"), "--minify", "r20261117-110000-0001"])


def test_a_stand_ins_unit_costs_are_a_passed_minify_record_with_their_table_ranges_and_how(run: Path):
    out = run / "out"
    out.mkdir()
    rows = [json.loads(x) for x in units_file(run).read_text().splitlines()]
    rows[0] |= {"device": "NVIDIA H200", "vllm": "0.x", "passes": {"present": {"refine": False}, "cut": {}},
                "how": "device time a call inside a CUDA graph"}
    (out / G.UNITS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert G.main(["minify", str(out), "--model", "glm53-standin"]) == 0
    r = result(run)
    assert r["kind"] == "serving-cost-minify/v1" and r["validation"]["status"] == "passed"
    assert r["validation"]["evidence"] == ["out/units.jsonl", "out/units.json"]
    s = json.loads((out / G.SUMMARY).read_text())
    assert s["device"] == "NVIDIA H200" and s["how"].startswith("device time a call inside a CUDA graph")
    assert {"part": "top-k", "path": "present", "case": "base", "rows": 6, "keys": 20000, "us": [7.0, 7.5]} in s["units"]
    assert len(s["units"]) == 2 * 2 * 4 and s["unpriced"] == [["top-k", "prefill steps"]]


def test_a_minify_record_fails_off_a_stand_in_without_a_pass_its_pin_offers_or_short_of_its_last_shape(run: Path):
    out = run / "out"
    out.mkdir()
    assert G.minify_gaps(out, "glm53-standin") == (["no units.jsonl: topk_units.py did not run"], None)
    rows = [json.loads(x) for x in units_file(run).read_text().splitlines()]
    rows[0] |= {"device": "NVIDIA H200", "passes": {"present": {}, "cut": {}}}
    rows = [{k: v for k, v in r.items() if k != "cut"} for r in rows if "done_s" not in r]
    (out / G.UNITS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    gaps, s = G.minify_gaps(out, "glm53")
    assert gaps == ["glm53 is not a stand-in (serve_tax.MODELS' stand_in)",
                    "units.jsonl has no last record (done_s): topk_units.py did not time every shape",
                    "units.jsonl holds no cut times with and without a failed row"]
    assert {u["path"] for u in s["units"]} == {"present"}
    assert G.main(["minify", str(out), "--model", "glm53-standin"]) == 1
    assert result(run)["validation"]["status"] == "failed"
