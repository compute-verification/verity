"""`verity bench prove` as a command: its runner, the Lean verifier's runner's record read by the table, what a
runner records under a research run, the marks and target of the recorded figures, and the validation."""

from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import pytest
from prove_helpers import (
    ACCEPTED,
    DESIGN,
    REPO,
    RMSNORM,
    STAGE,
    VERIFIER,
    Verifier,
    _outside_a_research_run,  # noqa: F401  (autouse)
    designed,
    fixture,
    kept_proof,
    lean_ran,
    lean_record,
    recorded,
)
from verity_bench import kit, spot_status
from verity_bench import lean_verify as L
from verity_bench import prove as P
from verity_bench import prove_records as R
from verity_bench.cli import main

RUNNER = REPO / "benchmarks" / "prove" / "run.py"


# ---- the command ---------------------------------------------------------------------------------------------------------------


def test_verity_bench_prove_is_the_commands_runner(tmp_path, monkeypatch, capsys):
    """`verity bench prove` runs benchmarks/prove/run.py, which is `verity-bench prove`: the table from the store alone."""

    s, arts = recorded(tmp_path)
    shutil.rmtree(tmp_path / "producer")
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    # spot's holes block over the fixture runs, with no git and the fixture lock: test_spot_status.py decides the holes
    monkeypatch.setattr(spot_status, "commit_known", lambda root, commit: False)
    monkeypatch.setattr(spot_status, "LOCK", Path(__file__).parent / "fixtures" / "spot_status" / "lock.json")

    def run(*argv: str) -> tuple[int, str, str]:
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        with pytest.raises(SystemExit) as done:
            runpy.run_path(str(REPO / "benchmarks" / "prove" / "run.py"), run_name="__main__")
        out = capsys.readouterr()
        return done.value.code, out.out, out.err

    code, out, _ = run("--system", "flock", "--workload", "glm53", "--rows", "2")
    assert code == 0 and out.startswith("# verity bench prove: flock on glm53 (GLM-5.3)\n")
    assert "- prover count (`--provers`, a stated input): 1;" in out and "8 staged types measured at steady state at 1 prover" in out
    assert "## Every block proved" in out and "| 148 more block types |" in out
    assert "- host cores (`--cores`, a stated input): 10 CPU cores a card, node A's share" in out
    code, out, _ = run("--system", "flock", "--workload", "glm53", "--provers", "2", "--json")
    assert code == 0 and (json.loads(out)["provers"], json.loads(out)["cores"]) == (2, 10)
    code, out, _ = run("--system", "flock", "--workload", "glm53", "--json", "--steady-rows", str(tmp_path / "steady.json"))
    wrote = json.loads((tmp_path / "steady.json").read_text())["rows"]
    measured = {r["block"]: r for r in json.loads(out)["rows"] if r["source"] == P.STEADY}
    assert code == 0 and len(wrote) == 8 and {r["block"] for r in wrote} == set(measured)
    assert all(r["statement_seconds"] / r["blocks"] == pytest.approx(measured[r["block"]]["seconds"], rel=1e-12) and r["provers"] == 1
               for r in wrote)
    code, out, _ = run("--system", "flock", "--workload", "glm53", "--cores", "24", "--json")
    doc = json.loads(out)
    assert code == 0 and (doc["provers"], doc["cores"]) == (1, 24) and doc["totals"]["above_cores"]["block_types"] == 0
    code, _, err = run("--system", "flock", "--workload", "glm53", "--cores", "0", "--local-only")
    assert code == 2 and "--cores 0: a type is priced on a host of one core a card or more" in err
    code, out, _ = run("--system", "spot", "--workload", "glm53", "--sampling", arts[R.SAMPLING], "--json")
    doc = json.loads(out)
    assert code == 0 and doc["schema"] == P.SCHEMA and doc["totals"]["gpus_per_serving_gpu"] == pytest.approx(0.5420038985704952)
    assert doc["spot_holes"]["label"] == "SPOT"
    code, _, err = run("--system", "spot", "--workload", "glm53", "--sampling", arts[R.CENSUS], "--local-only")
    assert code == 2 and f"{arts[R.CENSUS]} isn't a sampling plan of record" in err
    with pytest.raises(SystemExit):
        main(["prove", "--system", "sp1", "--workload", "glm53"])
    assert "invalid choice: 'sp1'" in capsys.readouterr().err


def test_verity_bench_prove_lean_verify_is_the_lean_verifiers_runner_and_the_table_reads_its_record(tmp_path, monkeypatch, capsys):
    """`verity bench prove lean-verify` reaches the runner through the same file, and the record it writes, kept as a
    run of record's `out/lean-verify.json`, is what the table's two lines are from."""

    s, _ = recorded(tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    Verifier(monkeypatch, [(ACCEPTED, 9.0)])

    def run(*argv: str) -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        with pytest.raises(SystemExit) as done:
            runpy.run_path(str(REPO / "benchmarks" / "prove" / "run.py"), run_name="__main__")
        return done.value.code, capsys.readouterr().out

    out_dir = tmp_path / "lean-out"
    code, out = run("lean-verify", "--lean", VERIFIER, "--proof", str(kept_proof(tmp_path, stage=STAGE)), "--out",
                    str(out_dir), "--repeat", "1")
    assert code == 0 and out.startswith("LEAN-VERIFY ")
    lean_ran(s, tmp_path, "r20261110-000000-0001", json.loads((out_dir / L.RECORD).read_text()))
    code, out = run("--system", "flock", "--workload", "glm53")
    assert code == 0 and ", modeled, not measured. Measured: one block type, `RMSNormFusedCudaVector8_v1{" in out
    assert (": accepted on the provisional unit cover, units as stated, the Program not held to the circuit, by the Lean "
            "verifier's `verify` in 9 s (one run;") in out and "per token: not measured" not in out
    code, out = run("--system", "flock", "--workload", "glm53", "--json")
    t = json.loads(out)["totals"]
    assert (t["lean_verifier_seconds_per_token"]["status"], t["proof_bytes_per_token"]["status"]) == ("modeled", "modeled")
    assert t["lean_verify"][0]["seconds"] == [9.0] and t["lean_verify"][0]["proof_bytes"] == 500


def test_under_a_research_run_the_runner_records_spots_metrics_and_prints_the_same_table(tmp_path, monkeypatch, capsys):
    """Under a research run, `verity bench prove --system spot` writes run.py's metrics into the run's result through
    the kit: each total of SPOT's table at the plan of record in its unit with its mark first among its inputs, the Lean
    verifier's seconds and the proof's bytes with no value and why until a Lean verifier run of record measures a type
    and then modeled with what its accept holds, the prover GPUs a serving GPU at the exfiltration run of record's
    design, modeled, from that run, the six checks as the validation and the table as its evidence. What it prints is
    the same in a run as outside one, where it writes nothing; with every block proved it records nothing."""

    s, arts = recorded(tmp_path)
    designed(s, tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261112-000000-0000")
    # spot's holes block over the fixture runs, with no git and the fixture lock: test_spot_status.py decides the holes
    monkeypatch.setattr(spot_status, "commit_known", lambda root, commit: False)
    monkeypatch.setattr(spot_status, "LOCK", Path(__file__).parent / "fixtures" / "spot_status" / "lock.json")
    monkeypatch.setattr(kit, "host_stamp", lambda: {"card": "none", "gpus": [], "driver": None, "cuda": None,
                                                    "node": "cpu-node-0", "commit": "c" * 40, "why": "no nvidia-smi on PATH"})
    writers, writer = [], kit.writer

    def recording(out):
        writers.append(out)
        return writer(out)

    monkeypatch.setattr(kit, "writer", recording)
    metrics = runpy.run_path(str(RUNNER))["METRICS"]
    spot = ("--system", "spot", "--workload", "glm53")

    def run(*argv: str, within: Path | None = None) -> str:
        if within is None:
            monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
        else:
            monkeypatch.setenv("RESEARCH_RUN_DIR", str(within))
        monkeypatch.setattr(sys, "argv", ["run.py", *argv])
        with pytest.raises(SystemExit) as done:
            runpy.run_path(str(RUNNER), run_name="__main__")
        assert done.value.code == 0
        return capsys.readouterr().out

    def result(within: Path) -> dict:
        return json.loads((within / kit.RESULT).read_text())

    outside, doc = run(*spot), json.loads(run(*spot, "--json"))
    assert outside.startswith(R.render(doc)) and writers == []
    first = tmp_path / "run-1"
    assert run(*spot, within=first) == outside and writers == [str(first)]
    got, kept = result(first), json.loads((first / R.TABLE).read_text())
    assert got["run_id"] == "r20261112-000000-0000" and kept["rows"] and kept.items() <= doc.items()
    assert got["workload_fingerprint"] == {"source_sha": None, "benchmark": "prove", "system": "spot[flock]",
                                           "workload": "glm53", "sampling": arts[R.SAMPLING], "provers": 1, "cores": 10}
    assert got["measurement_files"] == [R.TABLE] and got["artifacts"] == []
    by = {m["name"]: m for m in got["measurements"]}
    assert list(by) == list(metrics) == [
        "benchmarks/prover-gpus-per-serving-gpu", "benchmarks/prover-gpu-seconds-per-token",
        "benchmarks/lean-verifier-seconds-per-token", "benchmarks/proof-bytes-per-token",
        "benchmarks/prover-gpus-per-serving-gpu-at-record"]
    assert [m["unit"] for m in by.values()] == [R.FIGURES[f] for f in metrics.values()]
    design = by["benchmarks/prover-gpus-per-serving-gpu-at-record"]
    assert design["value"] == pytest.approx(0.055147426108551965, rel=1e-12) and design["mark"] == "modeled"
    assert design["inputs"] == [f"modeled: {R._design(doc['at_record'])}", DESIGN] and "why" not in design
    # goal benchmarks/proof-system-performance's target is on the two prover GPUs a serving GPU and on no other
    # measurement; both are over it on these records
    assert R.GPUS_PER_SERVING_GPU_TARGET == 0.0545
    assert {m["name"]: (m["target"], m["verdict"]) for m in by.values()} == {
        "benchmarks/prover-gpus-per-serving-gpu": (0.0545, "misses"), "benchmarks/prover-gpu-seconds-per-token": (None, None),
        "benchmarks/lean-verifier-seconds-per-token": (None, None), "benchmarks/proof-bytes-per-token": (None, None),
        "benchmarks/prover-gpus-per-serving-gpu-at-record": (0.0545, "misses")}
    t, targets = doc["totals"], [*dict.fromkeys(x for i in doc["inputs"] for x in i["targets"])]
    assert arts[R.SAMPLING] in targets
    head, seconds = by["benchmarks/prover-gpus-per-serving-gpu"], by["benchmarks/prover-gpu-seconds-per-token"]
    assert head["value"] == pytest.approx(0.5420038985704952, rel=1e-12) and seconds["value"] == t["gpu_seconds_per_token"]
    assert head["mark"] == "modeled" and head["inputs"][:2] == [
        "modeled", (f"proving: modeled: steady-state measurements cover {t['steady_time_share']:.1%} of the proving "
                    f"time; range 0.186021 to 0.542004, the other end {t['range_basis']}")]
    off = ("card off its CPUs' NUMA node: 0 of 8 priced steady rows, 0.0000% of the proving time; on its node by the "
           f"runs' own records: 0 rows, 0.0000%; not recorded: 8 rows, {t['off_node']['not_recorded']['time_share']:.4%}; "
           f"priced from the single-session card, whose table records no CPUs: {t['off_node']['single']['rows']} rows, "
           f"{t['off_node']['single']['time_share']:.4%}")
    assert head["inputs"][2].startswith("serving rate (modeled, a stated input): MFU ") and head["inputs"][3:] == [off, *targets]
    assert seconds["mark"] == "modeled" and seconds["inputs"][0].startswith("modeled: ") and seconds["inputs"][1:] == [
        off, *targets]
    assert f"\n- {off}\n" in outside
    assert "why" not in head and head["ci95"] is None and head["host"]["node"] == "cpu-node-0"
    for name in ("benchmarks/lean-verifier-seconds-per-token", "benchmarks/proof-bytes-per-token"):
        assert by[name]["value"] is None and by[name]["mark"] == "modeled" and by[name]["inputs"] == ["modeled", *targets]
        assert by[name]["why"] == ("not measured: no Lean verifier run of record measures a block type "
                                   "(`run_of_record=prove-glm53-lean-verify`)")
    steady = next(x["block_types"] for x in doc["sources"] if x["source"] == P.STEADY)
    assert got["validation"] == {
        "status": "passed", "checker": "verity_bench.prove_records.CHECKS", "evidence": [R.TABLE],
        "detail": f"{steady} block type(s), {t['steady_time_share']:.1%} of the proving time, priced from steady-state "
                  "rows that each passed `accepted`, `exits`, `not_void`, `flat`, `no_throttling`, "
                  "`every_prover_warmed`; a row that didn't prices nothing and is named on its type's row"}

    lean_ran(s, tmp_path, "r20261110-000000-0001", lean_record("r20261110-000000-0001"))
    second = tmp_path / "run-2"
    run(*spot, within=second)
    lean = {m["name"]: m for m in result(second)["measurements"]}
    t = json.loads((second / R.TABLE).read_text())["totals"]
    for name, key, unit in (("benchmarks/lean-verifier-seconds-per-token", "lean_verifier_seconds_per_token", "s an AND gate"),
                            ("benchmarks/proof-bytes-per-token", "proof_bytes_per_token", "bytes a witness bit")):
        v = t[key]
        assert lean[name]["value"] == v["value"] and lean[name]["mark"] == "modeled" and "why" not in lean[name]
        own, family, fitted = (v["sources"][k] for k in ("measured", "family", "fitted"))
        assert own["block_types"] == 1 and fitted["block_types"] > 0 and v["fitted_rate"] > 0
        assert lean[name]["inputs"][0] == (
            f"modeled, not measured: own {own['share']:.1%} (1 block types), family {family['share'] or 0.0:.1%} "
            f"({family['block_types']} block types), fitted {fitted['share']:.1%} ({fitted['block_types']} block types) "
            f"(fitted rate {v['fitted_rate']:.4g} {unit}); measured on one block type, `{RMSNORM}` ({L.COVER_LABEL}, run "
            f"r20261110-000000-0001): {v['basis']}")
        assert "r20261110-000000-0001" in lean[name]["inputs"][1:]

    every = tmp_path / "run-3"
    run("--system", "flock", "--workload", "glm53", within=every)
    assert not every.exists()


def test_the_proving_totals_marks_give_the_plans_own_prices_where_its_record_states_them(tmp_path):
    mix = fixture("mix.json")
    s, _ = recorded(tmp_path, cells=[c | {"prove_s": 0.002 * c["taken"]} for c in mix["cells"]])
    doc = R.build("spot", "glm53", store=s.fresh(), root=REPO)
    head, seconds = R.measurements(doc, {"a/x": "gpus_per_serving_gpu", "a/y": "gpu_seconds_per_token"})
    marks = [head["inputs"][1], seconds["inputs"][0]]
    planned = doc["totals"]["planned"]
    assert [m.rsplit("; ", 1)[1] for m in marks] == [
        f"at the sampling plan's own prices, which its planner sized the sample at: {planned[k]:.6g}"
        for k in ("gpus_per_serving_gpu", "gpu_seconds_per_token")]


def test_the_prover_gpus_a_serving_gpu_take_the_weaker_of_provings_and_the_serving_rates_marks(tmp_path):
    """The headline is the proving total over a stated serving rate, so its mark is the weaker of the two's (measured,
    then partial, then modeled), and its inputs list both; the proving GPU-seconds keep proving's own."""

    mix = fixture("mix.json")
    s, _ = recorded(tmp_path, cells=[c | {"prove_s": 0.002 * c["taken"]} for c in mix["cells"]])
    doc = R.build("spot", "glm53", store=s.fresh(), root=REPO)
    assert R.STRENGTH == ("measured", "partial", "modeled") and set(R.STRENGTH) == set(kit.MARKS)
    metrics = {"a/x": "gpus_per_serving_gpu", "a/y": "gpu_seconds_per_token"}
    for proving, missing, serving, weaker in (("measured", [], "modeled", "modeled"), ("measured", [], "measured", "measured"),
                                              ("measured", ["t"], "measured", "partial"), ("modeled", [], "measured", "modeled"),
                                              ("measured", ["t"], "modeled", "modeled")):
        given = doc | {"totals": doc["totals"] | {"status": proving, "missing": missing},
                       "serving": doc["serving"] | {"status": serving}}
        head, seconds = R.measurements(given, metrics)
        own = "partial" if missing else proving
        assert (head["mark"], seconds["mark"]) == (weaker, own), (proving, missing, serving)
        assert head["inputs"][0] == weaker and head["inputs"][1].startswith(f"proving: {own}")
        assert head["inputs"][2].startswith(f"serving rate ({serving}, a stated input): ")
        assert seconds["inputs"][0].startswith(own)
        w = kit.writer(tmp_path / f"{proving}-{len(missing)}-{serving}")
        assert w.measurement(**head)["inputs"] == head["inputs"] and w.measurement(**seconds)["mark"] == own


def test_the_prover_gpus_a_serving_gpu_carry_the_goals_target_and_meet_it_at_or_under_it(tmp_path):
    """Both prover GPUs a serving GPU carry the goal's target, lower being better: a value at or under it `meets` it,
    one over it `misses` it, and a figure with no value carries the target and no verdict. The kit writes both."""

    s, _ = recorded(tmp_path)
    designed(s, tmp_path)
    doc = R.build("spot", "glm53", store=s.fresh(), root=REPO)
    metrics = {"a/x": "gpus_per_serving_gpu", "a/r": R.AT_RECORD, "a/y": "gpu_seconds_per_token"}
    target = R.GPUS_PER_SERVING_GPU_TARGET
    for value, verdict in ((target, "meets"), (0.9 * target, "meets"), (1.1 * target, "misses")):
        given = doc | {"totals": doc["totals"] | {"gpus_per_serving_gpu": value},
                       "at_record": doc["at_record"] | {"gpus_per_serving_gpu": value}}
        head, design, seconds = R.measurements(given, metrics)
        assert [(m["target"], m["verdict"]) for m in (head, design)] == [(target, verdict)] * 2, value
        assert "target" not in seconds and "verdict" not in seconds
        w = kit.writer(tmp_path / verdict / str(value))
        assert [(m["target"], m["verdict"]) for m in (w.measurement(**head), w.measurement(**design))] == [(target, verdict)] * 2
    unread = doc | {"at_record": doc["at_record"] | {"gpus_per_serving_gpu": None, "why": "no record"}}
    (design,) = R.measurements(unread, {"a/r": R.AT_RECORD})
    assert design["target"] == target and "verdict" not in design and design["value"] is None


def test_the_validation_fails_where_no_steady_state_row_prices_a_type_or_a_total_leaves_a_type_out():
    sources = [{"source": P.STEADY, "block_types": 3}, {"source": P.SINGLE, "block_types": 2}]
    doc = {"sources": sources, "totals": {"steady_time_share": 0.25, "missing": []}}
    assert R.validation(doc)[0] == "passed"
    status, detail = R.validation(doc | {"totals": {"steady_time_share": 0.25, "missing": [{"block": "X"}]}})
    assert status == "failed" and detail.endswith("; 1 block type(s) have no figure, and every total leaves them out")
    status, detail = R.validation(doc | {"sources": [{"source": P.STEADY, "block_types": 0}]})
    assert status == "failed" and detail.startswith("no steady-state row passed `accepted`, ")
    with pytest.raises(ValueError, match="no total of the table is named blocks_per_token"):
        R.measurements(doc, {"benchmarks/x": "blocks_per_token"})
