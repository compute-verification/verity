from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.serving_cost import run as S  # noqa: E402
from verity_bench import kit  # noqa: E402

MINIFY, EXACT, CAPTURE, PRICE, MEASURED, UNITS, REPLAY, CC, PROFILE = (f"r20261118-0000{i}0-000{i}" for i in range(1, 10))
#: (rows, keys): (base present, base cut, a call with every row failed, present then cut), µs over the fast kernel
SHAPES = {(6, 20000): (7.0, 12.0, 100.0, 22.0), (6, 131072): (12.0, 16.0, 1000.0, 50.0),
          (24, 20000): (10.0, 15.0, 100.0, 34.0), (24, 131072): (24.0, 28.0, 1000.0, 110.0)}
PREFILL = {(1024, 4096): (90.0, 95.0, 400.0, 120.0), (1024, 8192): (150.0, 160.0, 700.0, 210.0),
           (4096, 4096): (300.0, 320.0, 900.0, 400.0), (4096, 8192): (500.0, 520.0, 1500.0, 650.0)}
BINDINGS = ["DsaLatentRowDsMla_v1/glm53"]
MEMBER = "glm-5.3-h200-prod-record"
MEASURED_MEMBER = "glm-5.3-h200-prod-record-over-prod"


@pytest.fixture
def store(tmp_path, monkeypatch) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    monkeypatch.setattr(S, "fetch",
                        lambda run, paths=None: runs[run] if run in runs else pytest.fail(f"fetched {run}"))
    return runs


def record(tmp: Path, run: str, kind: str | None, *, status: str = "passed", fingerprint: dict | None = None,
           measurements: list | None = None) -> Path:
    d = tmp / run
    (d / "out").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "schema": kit.RESULT_SCHEMA, **({"kind": kind} if kind else {}), "run_id": run,
        "validation": {"status": status}, "workload_fingerprint": fingerprint or {},
        "measurements": measurements or []}))
    return d


def minify_record(tmp: Path, units: bool = True) -> Path:
    return units_record(tmp, MINIFY, S.KINDS["minify"], SHAPES, units=units)


def units_record(tmp: Path, run: str, kind: str, table: dict, *, units: bool = True, **fingerprint) -> Path:
    d = record(tmp, run, kind, fingerprint=fingerprint)
    lines: list[dict] = [{"K": 2048}]
    for (rows, keys), (bp, bc, fp, fc) in table.items():
        for dist, e in (("randn", 0.0), ("indexer", 0.5)):
            lines.append({"dist": dist, "rows": rows, "keys": keys,
                          "present": {"added_us": bp + e, "per_call": {"rows_long": rows}},
                          "cut": {"added_us": bc + e, "per_call": {"rows_long": rows}}})
        if units:
            for how in ("missed", "overflow"):
                lines.append({"forced": how, "rows": rows, "keys": keys, "present": {"us": fp}, "cut": {"us": fc}})
    (d / S.UNITS).write_text("".join(json.dumps(x) + "\n" for x in lines))
    return d


def lifetime(out: Path, name: str, seconds: float, mode: str | None, decode: tuple[tuple[int, int], ...] = (),
             prefill: bool = True) -> None:
    d = out / name
    for rep, t0 in ((1, 0.0), (2, 100.0)):
        p = d / "agentic@4" / f"rep-{rep}"
        p.mkdir(parents=True)
        (p / "meta.json").write_text(json.dumps({"load": "agentic@4", "rep": rep, "t0_unix": t0, "wall_s": 10.0}))
        (p / "result.json").write_text(json.dumps({"duration": seconds}))
    (d / "meta.json").write_text(json.dumps({"left_out": 1, "env": {"VERITY_DSA_TOPK": mode} if mode else {},
                                             "server_argv": ["--tensor-parallel-size", "8"]}))
    steps = [{"kind": "header", "num_speculative_tokens": 5},
             {"kind": "step", "step": 0, "rows": [[0, 4000]] if prefill else [], "t": 101.0}]
    steps += [{"kind": "step", "step": i + 1, "rows": [[start, 6] for _ in range(n)], "t": 102.0 + i}
              for i, (n, start) in enumerate(decode)]
    (d / "server").mkdir()
    with gzip.open(d / "server" / "steps.1.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in steps))


def capture_record(tmp: Path, *, below: bool = True, lifetimes: bool = True, prefill: bool = True,
                   **fingerprint) -> Path:
    fp = {"minify": MINIFY, "served": "glm-5.3", "card": "h200"} | fingerprint
    d = record(tmp, CAPTURE, S.KINDS["capture"], fingerprint={k: v for k, v in fp.items() if v is not None})
    out = d / "out"
    for k in (1, 2) if lifetimes else ():
        lifetime(out, f"prod-trace.{k}", 50.0, None, prefill=prefill)
        lifetime(out, f"prod-record.{k}", 52.0, "canonical", ((4, 131066), (1, 19994)), prefill)
    counters = {"rows_long": 1, "rows_bisected": 80, "rows_refined": 1600, "rows_at": 1, "calls": 1}
    (out / "topk_counters.json").write_text(json.dumps({"dirs": {f"prod-record.{k}": counters for k in (1, 2)}}))
    layers = [{"arm": "prod-record", "over": "prod-trace", "value": 0.02, "ci95": [0.01, 0.03], "pairs": 2}]
    if below:
        layers.append({"arm": "prod-trace", "over": "prod", "value": 0.006, "ci95": [-0.01, 0.02], "pairs": 2})
    (out / "prod_repro.json").write_text(json.dumps({"run": CAPTURE, "layers": {"each": layers}}))
    return d


def result(out: Path) -> dict:
    return json.loads((out / kit.RESULT).read_text())


def price(out: Path, *units: str) -> int:
    return S.main(["price", "--capture", CAPTURE, "--minify", MINIFY, "--ours", "prod-record", "--out", str(out),
                   *(x for u in units for x in ("--units", u))])


def replay_record(tmp: Path, status: str = "passed", **fingerprint) -> Path:
    return record(tmp, REPLAY, S.REPLAY_KIND, status=status,
                  fingerprint={"arm_dir": "/data/prod-record.1", "load": "agentic@4", "rep": 2} | fingerprint)


def circuit_check_record(tmp: Path, ok: object = True, rows: tuple[str, ...] = tuple(BINDINGS),
                         warned: tuple[str, ...] = ()) -> Path:
    d = record(tmp, CC, None)
    warning = {"name": "partition/unchecked:Dsa_v1", "check": "partition", "code": "unchecked",
               "definition": "Dsa_v1", "count": 1}
    (d / S.CIRCUIT_CHECK).write_text(json.dumps({
        "schema": S.CIRCUIT_CHECK_SCHEMA, "options": {},
        "summary": {"ok": ok, "failures": 0 if ok else 1, "failure_names": [] if ok else ["exhaustive"],
                    "unchecked": {"items": len(warned), "definitions": 1 if warned else 0},
                    "warnings": {"partition/unchecked": {"total": 1, "definitions": 1, "by_definition": {"Dsa_v1": 1}}}
                    if warned else {}},
        "targets": [{"kind": "definition", "target": r, "failures": [], "warnings": [warning] if r in warned else [],
                     "seconds": 0.1, "peak_gb": 0.1, "block": {}} for r in rows]}))
    return d


def exact(out: Path, *extra: str) -> int:
    return S.main(["exact", "--minify", MINIFY, "--replay", REPLAY, "--out", str(out), *extra])


@pytest.mark.parametrize("argv, names", [
    (["minify"], "give RUN, the run of vllm's stand-in"),
    (["exact"], "give --minify RUN, the run of vllm's stand-in"),
    (["capture"], "give RUN, the run of vllm's `benchmarks/prod_repro/run.sh` on the real model"),
    (["price", "--minify", MINIFY], "give --capture RUN, the run of vllm's `benchmarks/prod_repro/run.sh`"),
    (["price", "--capture", CAPTURE], "give --minify RUN, the run of vllm's stand-in"),
    (["confirm"], "give --price RUN, the run of `verity bench serving_cost price`"),
])
def test_a_stage_without_the_stored_record_it_reads_is_refused_naming_what_makes_it(store, argv, names):
    with pytest.raises(SystemExit, match=f"serving_cost {argv[0]}: no .* record: {names}"):
        S.main(argv)


def test_each_refusal_states_the_record_s_contract():
    with pytest.raises(SystemExit) as e:
        S.main(["capture"])
    said = str(e.value)
    assert ("a result of kind serving-cost-capture/v1, validation passed, holding the run's out/ as priced.py reads"
            in said)
    assert "`minify`" in said and "`exact` once an exact record exists" in said and "`served` and `card`" in said
    with pytest.raises(SystemExit, match="kind serving-cost-minify/v1, validation passed, holding out/units.jsonl"):
        S.main(["minify"])


def test_minify_prints_each_part_and_code_path_with_its_unit_cost(store, tmp_path, capsys):
    store[MINIFY] = minify_record(tmp_path)
    assert S.main(["minify", MINIFY]) == 0
    said = capsys.readouterr().out
    assert f"minify run {MINIFY}: a passed minify record" in said
    assert f"top-k, decode steps, present pass, no row failed, 6 rows of 20000 keys: 7.0 to 7.5 ({MINIFY})" in said
    assert ("top-k, decode steps, cut pass, every row failed, over the base, 24 rows of 131072 keys: 81.5 to 82.0"
            in said)
    assert "prefill shapes: no source covers them" in said


def test_minify_prints_a_units_run_s_costs_beside_its_own(store, tmp_path, capsys):
    store[MINIFY] = minify_record(tmp_path)
    store[UNITS] = units_record(tmp_path, UNITS, S.UNITS_KIND, PREFILL, shapes="prefill")
    assert S.main(["minify", MINIFY, "--units", UNITS]) == 0
    said = capsys.readouterr().out
    assert f"top-k, prefill steps, present pass, no row failed, 1024 rows of 4096 keys: 90.0 to 90.5 ({UNITS})" in said
    assert "prefill shapes: covered" in said


def test_minify_refuses_a_record_without_units_in_priced_s_form(store, tmp_path):
    store[MINIFY] = minify_record(tmp_path, units=False)
    with pytest.raises(SystemExit, match="is not in the form `priced.units` reads: .*no present times"):
        S.main(["minify", MINIFY])
    (store[MINIFY] / S.UNITS).unlink()
    with pytest.raises(SystemExit, match=f"minify run {MINIFY} holds no out/units.jsonl"):
        S.main(["minify", MINIFY])


def test_exact_passes_on_a_passed_replay_and_circuit_check_saying_what_it_could_not_hold(store, tmp_path, capsys):
    store[MINIFY], store[REPLAY] = minify_record(tmp_path), replay_record(tmp_path)
    store[CC] = circuit_check_record(tmp_path)
    out = tmp_path / "exact"
    assert exact(out, "--circuit-check", CC, "--bindings", *BINDINGS) == 0
    r = result(out)
    assert r["kind"] == "serving-cost-exact/v1" and r["validation"]["status"] == "passed"
    assert r["workload_fingerprint"].items() >= {"minify": MINIFY, "replay": REPLAY, "circuit_check": CC,
                                                 "replay_kind": "python", "held_to_minify": False}.items()
    assert r["workload_fingerprint"]["bindings"] == BINDINGS
    assert "`arm_dir` '/data/prod-record.1'" in r["validation"]["detail"]
    assert "0 of the 1 named bindings unchecked" in r["validation"]["detail"]
    assert "could not be held to minify run" in capsys.readouterr().out
    doc = json.loads((out / "exact.json").read_text())
    assert doc["replay"]["kind"] == "python"
    assert doc["circuit_check"]["unchecked"] == {"count": 0, "of": 1, "bindings": [],
                                                 "report": {"items": 0, "definitions": 0}}


def test_exact_reports_an_unchecked_binding_without_refusing(store, tmp_path):
    store[MINIFY], store[REPLAY] = minify_record(tmp_path), replay_record(tmp_path)
    store[CC] = circuit_check_record(tmp_path, warned=tuple(BINDINGS))
    out = tmp_path / "exact"
    assert exact(out, "--circuit-check", CC, "--bindings", *BINDINGS) == 0
    assert (f"1 of the 1 named bindings unchecked (a warning, which never fails: {BINDINGS[0]}; the report's "
            "`summary.unchecked`: 1 items on 1 definitions)") in result(out)["validation"]["detail"]
    assert json.loads((out / "exact.json").read_text())["circuit_check"]["unchecked"]["count"] == 1


def test_exact_holds_a_replay_that_names_its_minify_run(store, tmp_path):
    store[MINIFY], store[CC] = minify_record(tmp_path), circuit_check_record(tmp_path)
    store[REPLAY] = replay_record(tmp_path, minify=MINIFY, replay="lean")
    assert exact(tmp_path / "exact", "--circuit-check", CC, "--bindings", *BINDINGS) == 0
    fp = result(tmp_path / "exact")["workload_fingerprint"]
    assert fp["held_to_minify"] is True and fp["replay_kind"] == "lean"
    store[REPLAY] = replay_record(tmp_path / "other", minify="r20261118-000000-ffff")
    with pytest.raises(SystemExit, match=f"replayed minify run 'r20261118-000000-ffff''s copy, not {MINIFY}'s"):
        exact(tmp_path / "again", "--circuit-check", CC, "--bindings", *BINDINGS)


@pytest.mark.parametrize("ok, rows, why", [
    (False, tuple(BINDINGS), "failed over 1 targets, failing exhaustive"),
    (True, ("Other_v1/glm53",), f"failed over 1 targets, holding none of {BINDINGS[0]}"),
    ("true", tuple(BINDINGS), "failed over 1 targets"),
])
def test_exact_fails_where_circuit_check_fails_or_misses_a_binding(store, tmp_path, ok, rows, why):
    store[MINIFY], store[REPLAY] = minify_record(tmp_path), replay_record(tmp_path)
    store[CC] = circuit_check_record(tmp_path, ok, rows)
    assert exact(tmp_path / "exact", "--circuit-check", CC, "--bindings", *BINDINGS) == 1
    v = result(tmp_path / "exact")["validation"]
    assert v["status"] == "failed" and why in v["detail"]


def test_exact_refuses_without_a_circuit_check_report_naming_circuits(store, tmp_path):
    store[MINIFY], store[REPLAY] = minify_record(tmp_path), replay_record(tmp_path)
    with pytest.raises(SystemExit, match="no circuit-check verdict on the real model's bindings.*--out .*circuits' "
                                         "command for that set is coming$"):
        exact(tmp_path / "exact")
    with pytest.raises(SystemExit, match="no circuit-check verdict"):
        exact(tmp_path / "exact", "--circuit-check", CC)
    store[CC] = record(tmp_path, CC, None)
    with pytest.raises(SystemExit, match=f"run {CC} holds no circuit-check report at out/circuit-check.json"):
        exact(tmp_path / "exact", "--circuit-check", CC, "--bindings", *BINDINGS)


def test_exact_refuses_a_replay_that_is_not_a_passed_replay_record(store, tmp_path):
    store[MINIFY] = minify_record(tmp_path)
    with pytest.raises(SystemExit, match="no replay record: give --replay RUN, the run of proofs' "):
        S.main(["exact", "--minify", MINIFY])
    store[REPLAY] = replay_record(tmp_path, status="failed")
    with pytest.raises(SystemExit, match=f"run {REPLAY}'s replay record did not pass"):
        exact(tmp_path / "exact")


def test_capture_prints_what_a_passed_capture_record_holds(store, tmp_path, capsys):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    assert S.main(["capture", CAPTURE]) == 0
    said = capsys.readouterr().out
    assert f"capture run {CAPTURE}: a passed capture record of glm-5.3 on h200" in said
    assert f"minify run {MINIFY} (passed); exact run none named" in said
    assert "lifetimes: prod-record ×2, prod-trace ×2" in said
    assert "top-k counters: prod-record.1, prod-record.2" in said
    assert "prod-record over prod-trace +2.00%, prod-trace over prod +0.60%" in said


def test_capture_checks_the_exact_run_it_names(store, tmp_path, capsys):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path, exact=EXACT)
    store[EXACT] = record(tmp_path, EXACT, S.KINDS["exact"], status="failed")
    with pytest.raises(SystemExit, match=f"run {EXACT}'s exact record did not pass \\(validation 'failed'\\)"):
        S.main(["capture", CAPTURE])
    store[EXACT] = record(tmp_path / "again", EXACT, S.KINDS["exact"])
    assert S.main(["capture", CAPTURE]) == 0
    assert f"exact run {EXACT} (passed)" in capsys.readouterr().out


@pytest.mark.parametrize("fingerprint, lifetimes, why", [
    ({"minify": None}, True, "names no minify run"),
    ({"card": None}, True, "names no member parts .*`card` None"),
    ({"card": "NVIDIA H200"}, True, "names no member parts"),
    ({"card": "h900"}, True, "fingerprint's `card`: 0 GPU card files under .* have `short` 'h900': one of .*'h200'"),
    ({}, False, "holds no arm's lifetimes"),
])
def test_capture_refuses_a_record_that_breaks_the_contract(store, tmp_path, fingerprint, lifetimes, why):
    store[MINIFY] = minify_record(tmp_path)
    store[CAPTURE] = capture_record(tmp_path, lifetimes=lifetimes, **fingerprint)
    with pytest.raises(SystemExit, match=f"capture run {CAPTURE}'s .*{why}"):
        S.main(["capture", CAPTURE])


def test_profile_prints_a_passed_profile_records_cost_and_what_is_under_it(store, tmp_path, capsys):
    fp = {"profile": "coding-agent", "version": 1, "digest": "54ef9ec108314bdd", "model": "glm53", "card": "h200",
          "stock": "prod", "feature": "prod-record"}
    member = "glm-5.3-h200-prod-record-over-prod-coding-agent"
    held = [{"name": "vllm/serving-cost-glm53", "value": 11.11, "unit": "percent", "member": member,
             "ci95": [11.1, 11.12], "label": "our extra card time a token", "cost_unit": "lifetime",
             "reading": "ours slower"},
            {"name": "ttft_p50_s", "value": 0.2, "unit": "s", "member": member, "under": "vllm/serving-cost-glm53"}]
    store[PROFILE] = record(tmp_path, PROFILE, S.PROFILE_KIND, fingerprint=fp, measurements=held)
    assert S.main(["profile", PROFILE]) == 0
    said = capsys.readouterr().out.splitlines()
    assert said == [f"profile run {PROFILE}: a passed profile record, coding-agent v1 (54ef9ec10831), prod-record "
                    "beside prod on glm53.h200",
                    f"  our extra card time a token, {member}: +11.11% [+11.10, +11.12], a lifetime its unit: ours slower",
                    f"  beside it: ttft_p50_s {member} 0.2 s"]
    store[PROFILE] = record(tmp_path / "none", PROFILE, S.PROFILE_KIND, fingerprint=fp, measurements=held[1:])
    with pytest.raises(SystemExit, match="holds no figure of our extra card time a token"):
        S.main(["profile", PROFILE])
    store[PROFILE] = record(tmp_path / "failed", PROFILE, S.PROFILE_KIND, status="failed", fingerprint=fp)
    with pytest.raises(SystemExit, match=f"run {PROFILE}'s profile record did not pass"):
        S.main(["profile", PROFILE])


def test_a_path_in_place_of_a_run_id_is_refused(store, tmp_path):
    with pytest.raises(SystemExit, match="not a run id: .* never by a path"):
        S.main(["minify", str(minify_record(tmp_path))])


def test_a_record_of_another_kind_or_that_did_not_pass_is_refused(store, tmp_path):
    store[MINIFY] = record(tmp_path, MINIFY, "prod_repro/v1")
    with pytest.raises(SystemExit, match=f"run {MINIFY} holds no minify record .*'prod_repro/v1'"):
        S.main(["exact", "--minify", MINIFY])
    store[CAPTURE] = record(tmp_path, CAPTURE, S.KINDS["capture"], status="failed")
    with pytest.raises(SystemExit, match=f"run {CAPTURE}'s capture record did not pass \\(validation 'failed'\\)"):
        S.main(["capture", CAPTURE])


def test_price_refuses_a_capture_taken_against_another_minify(store, tmp_path):
    other = "r20261118-000000-ffff"
    store[MINIFY], store[other] = minify_record(tmp_path), minify_record(tmp_path / "other")
    store[CAPTURE] = capture_record(tmp_path, minify=other)
    with pytest.raises(SystemExit, match=f"was taken against minify run '{other}'"):
        price(tmp_path / "price")


def test_price_writes_its_own_metric_partial_with_both_runs_and_the_unpriced_paths(store, tmp_path):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    out = tmp_path / "price"
    assert price(out) == 0
    r = result(out)
    assert r["kind"] == S.KINDS["price"] and r["validation"]["status"] == "passed"
    assert r["workload_fingerprint"]["capture"] == CAPTURE and r["workload_fingerprint"]["minify"] == MINIFY
    (m,) = r["measurements"]
    assert m["name"] == "benchmarks/serving-cost-priced" and m["unit"] == "percent" and m["member"] == MEMBER
    assert m["mark"] == "partial" and m["inputs"] == ["partial", CAPTURE, MINIFY]
    assert m["unpriced"] == [["top-k", "prefill steps"]] and m["stock"] == "prod"
    assert {x["path"]: x["runs"] for x in m["priced_by"]}["decode steps, cut pass, no row failed"] == [MINIFY]
    priced = json.loads((out / "priced.json").read_text())
    layer = priced["layers"][0]
    assert m["value"] == pytest.approx(100 * (layer["cost"]["mid"] + layer["prefill_measured"]["value"] + 0.006))
    assert r["validation"]["evidence"] == ["priced.json"]


def test_price_of_a_capture_with_no_prefill_steps_is_modeled(store, tmp_path):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path, prefill=False)
    assert price(tmp_path / "price") == 0
    (m,) = result(tmp_path / "price")["measurements"]
    assert m["mark"] == "modeled" and m["unpriced"] == []


def test_price_with_a_units_run_at_prefill_shapes_prices_them_and_is_modeled(store, tmp_path):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    store[UNITS] = units_record(tmp_path, UNITS, S.UNITS_KIND, PREFILL, shapes="prefill")
    out = tmp_path / "price"
    assert price(out, UNITS) == 0
    r = result(out)
    (m,) = r["measurements"]
    assert m["mark"] == "modeled" and m["unpriced"] == [] and m["inputs"] == ["modeled", CAPTURE, MINIFY, UNITS]
    assert r["workload_fingerprint"]["units"] == [UNITS]
    by = {x["path"]: x["runs"] for x in m["priced_by"]}
    assert by["prefill steps, present pass, no row failed"] == [UNITS]
    assert by["decode steps, present pass, no row failed"] == [MINIFY]
    layer = json.loads((out / "priced.json").read_text())["layers"][0]
    assert layer["prefill_priced"] and layer["each"][0]["prefill_s"]["mid"] > 0
    assert m["value"] == pytest.approx(100 * (layer["cost"]["mid"] + 0.006))


def test_price_merges_two_sources_at_decode_shapes_and_refuses_where_they_differ(store, tmp_path):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    store[UNITS] = units_record(tmp_path, UNITS, S.UNITS_KIND, SHAPES, shapes="decode")
    assert price(tmp_path / "price", UNITS) == 0
    (m,) = result(tmp_path / "price")["measurements"]
    assert {x["path"]: x["runs"] for x in m["priced_by"]}["decode steps, cut pass, no row failed"] == [MINIFY, UNITS]
    other = dict(SHAPES) | {(6, 20000): (8.0, 12.0, 100.0, 22.0)}
    store[UNITS] = units_record(tmp_path / "other", UNITS, S.UNITS_KIND, other, shapes="decode")
    with pytest.raises(SystemExit, match=f"runs {MINIFY} and {UNITS} both give top-k's present pass, no row failed, "
                                         "at 6 rows of 20000 keys on decode steps a unit cost, and they differ"):
        price(tmp_path / "again", UNITS)


def test_price_refuses_sources_that_together_time_no_grid(store, tmp_path):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    store[UNITS] = units_record(tmp_path, UNITS, S.UNITS_KIND, {(6, 50000): (9.0, 14.0, 300.0, 30.0)}, shapes="decode")
    with pytest.raises(SystemExit, match=f"runs {MINIFY}, {UNITS} together time .* at no rows × keys grid, which "
                                         "pricing reads: none at 24 rows of 50000 keys"):
        price(tmp_path / "price", UNITS)


def test_price_refuses_a_units_run_that_names_no_shapes(store, tmp_path):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    store[UNITS] = units_record(tmp_path, UNITS, S.UNITS_KIND, PREFILL)
    with pytest.raises(SystemExit, match=f"units run {UNITS}'s fingerprint names no shapes it timed"):
        price(tmp_path / "price", UNITS)
    store[UNITS] = units_record(tmp_path / "other", UNITS, S.KINDS["minify"], PREFILL, shapes="prefill")
    with pytest.raises(SystemExit, match=f"run {UNITS} holds no units record .*kind {S.UNITS_KIND}"):
        price(tmp_path / "again", UNITS)


def test_price_with_no_measured_layer_below_stores_no_figure_and_fails(store, tmp_path):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path, below=False)
    out = tmp_path / "price"
    assert price(out) == 1
    r = result(out)
    assert r["validation"]["status"] == "failed" and "measured no prod-trace over prod" in r["validation"]["detail"]
    assert r["measurements"][0]["value"] is None and r["measurements"][0]["why"] == r["validation"]["detail"]


def priced_record(tmp: Path, value: float = 2.5) -> Path:
    return record(tmp, PRICE, S.KINDS["price"], measurements=[
        {"name": S.METRIC, "value": value, "mark": "partial", "member": MEMBER, "ci95": [1.0, 4.0],
         "unpriced": [["top-k", "prefill steps"]], "stock": "prod"}])


def measured_record(tmp: Path, mark: str = "measured", member: str = MEASURED_MEMBER) -> Path:
    return record(tmp, MEASURED, None, measurements=[
        {"name": "vllm/prod-serving-cost-glm53", "value": 3.3, "mark": mark, "member": member}])


def test_confirm_writes_measured_less_priced_against_the_price_s_member_over_its_stock(store, tmp_path):
    store[PRICE], store[MEASURED] = priced_record(tmp_path), measured_record(tmp_path)
    out = tmp_path / "confirm"
    assert S.main(["confirm", "--price", PRICE, "--measured", MEASURED, "--out", str(out)]) == 0
    r = result(out)
    (m,) = r["measurements"]
    assert r["kind"] == S.KINDS["confirm"] and m["name"] == S.DIFFERENCE and m["value"] == pytest.approx(0.8)
    assert m["inputs"] == ["modeled", MEASURED, PRICE] and m["member"] == MEMBER
    assert m["measured"]["member"] == MEASURED_MEMBER and m["priced"]["unpriced"] == [["top-k", "prefill steps"]]


def test_confirm_refuses_a_full_size_run_without_the_measured_member(store, tmp_path):
    store[PRICE], store[MEASURED] = priced_record(tmp_path), measured_record(tmp_path, "modeled", "x")
    with pytest.raises(SystemExit, match=f"holds no measured .* of member '{MEASURED_MEMBER}'; it holds 'x' "
                                         "\\(modeled\\): give --member"):
        S.main(["confirm", "--price", PRICE, "--measured", MEASURED, "--out", str(tmp_path / "c")])
    with pytest.raises(SystemExit, match="no full-size run: give --measured RUN"):
        S.main(["confirm", "--price", PRICE])


def test_confirm_reads_no_figure_from_a_price_under_another_id(store, tmp_path):
    store[PRICE] = record(tmp_path, PRICE, S.KINDS["price"], measurements=[
        {"name": "vllm/prod-serving-cost-glm53", "value": 2.5, "mark": "partial", "member": MEMBER}])
    with pytest.raises(SystemExit, match="holds no priced figure under benchmarks/serving-cost-priced"):
        S.main(["confirm", "--price", PRICE, "--measured", MEASURED])


def _stored(out: Path) -> str:
    r = result(out)
    volatile = {"commit", "run", "host", "version"}
    r["measurements"] = [{k: v for k, v in m.items() if k not in volatile} for m in r["measurements"]]
    doc = {"result": r, "priced": json.loads((out / "priced.json").read_text())}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("units, below, prefill, digest", [
    ((), True, True, "919cf2acb8bfa16c69c745dc43a83bd00406195b88a68f60ca10dc7c06c620d2"),
    ((UNITS,), True, True, "054ce6f725613251a5229743025756321c1d8367eec1dc286bb763921bcf2243"),
    ((), True, False, "b31d5b90a556ca54e271d634ba0e1b73b3af5c7b6b205f65807a6b2150d7ad5e"),
    ((), False, True, "b8cc7273efde2e4d3bfb3a1b6fb274ddba4135a1daaa349ca09fff3ae1073978"),
])
def test_price_stores_the_record_it_stored_before_it_called_priced_price(store, tmp_path, units, below, prefill, digest):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path, below=below, prefill=prefill)
    store[UNITS] = units_record(tmp_path, UNITS, S.UNITS_KIND, PREFILL, shapes="prefill")
    price(tmp_path / "price", *units)
    assert _stored(tmp_path / "price") == digest


def test_price_reads_a_units_run_as_stress_py_units_stores_it(store, tmp_path):
    from benchmarks.dsa_topk import units as U

    def timer(values: str, rows: int, keys: int, how: str | None) -> dict:
        us = 1e-4 * rows * keys / 64
        if how is None:
            return {"fast_us": us, **{p: {"ok": True, "us": 2 * us, "added_us": us, "per_call": {"rows_long": rows}}
                                      for p in U.PASSES.values()}}
        return {"copy_us": us, **{p: {"ok": True, "one_call": {"rows_long": rows, "rows_bisected": rows}, "us": 5 * us}
                                  for p in U.PASSES.values()}}

    run = tmp_path / UNITS
    assert U.write(run / "out", run / "result.json", "prefill", U.lines("prefill", timer), "a stand-in")
    store[MINIFY], store[CAPTURE], store[UNITS] = minify_record(tmp_path), capture_record(tmp_path), run
    out = tmp_path / "price"
    assert price(out, UNITS) == 0
    (m,) = result(out)["measurements"]
    assert m["mark"] == "modeled" and m["unpriced"] == [] and m["inputs"] == ["modeled", CAPTURE, MINIFY, UNITS]
    assert {x["path"]: x["runs"] for x in m["priced_by"]}["prefill steps, cut pass, every row failed, over the base"] \
        == [UNITS]


def test_price_refuses_arms_the_capture_did_not_serve(store, tmp_path):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    with pytest.raises(SystemExit, match="the capture served none of prod-record-bisect: give --ours an arm it served"):
        S.main(["price", "--capture", CAPTURE, "--minify", MINIFY, "--ours", "prod-record-bisect", "--out",
                str(tmp_path / "price")])


STEPS_RUN, OLDER_STEPS_RUN = "r20261118-000110-0010", "r20261118-000120-0011"


class LabelStore:
    """A stand-in for the store `labels_store` gives: its index's `labels` table, nothing pulled."""

    def __init__(self, labels: list[tuple[str, str, str, str]]):
        import sqlite3

        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE labels(target TEXT, key TEXT, value TEXT, by TEXT, ts TEXT, ref TEXT, path TEXT)")
        self.db.executemany("INSERT INTO labels VALUES (?, ?, ?, 'benchmarks', ?, NULL, ?)",
                            [(t, k, v, ts, f"{t}/{ts}") for t, k, v, ts in labels])
        self.index = self

    def sql(self, query: str, params: tuple = ()) -> list[tuple]:
        return self.db.execute(query, params).fetchall()


def steps_moved_out(store: dict[str, Path], tmp: Path, run: str) -> None:
    """The capture's public steps moved into a record of `run`'s, as `serve_tax.py steps-of-trace --to` lays them."""

    d = record(tmp, run, None)
    for f in sorted((store[CAPTURE] / "out").glob(S.STEPS)):
        dst = d / "out" / f.relative_to(store[CAPTURE] / "out")
        dst.parent.mkdir(parents=True, exist_ok=True)
        f.rename(dst)
    store[run] = d


def labelled(monkeypatch, labels: list[tuple[str, str, str, str]]) -> None:
    monkeypatch.setattr(S, "labels_store", lambda: LabelStore(labels))


def test_price_reads_a_captures_steps_from_the_record_labelled_steps_of_it_and_prices_as_its_own(
        store, tmp_path, monkeypatch, capsys):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    assert price(tmp_path / "own") == 0
    (own,) = result(tmp_path / "own")["measurements"]
    steps_moved_out(store, tmp_path / "steps", STEPS_RUN)
    assert not any((store[CAPTURE] / "out").glob(S.STEPS))
    labelled(monkeypatch, [(STEPS_RUN, S.STEPS_OF, CAPTURE, "2026-11-18T05:54:14Z"),
                           (STEPS_RUN, "note", f"public steps of {CAPTURE}", "2026-11-18T05:34:40Z")])
    capsys.readouterr()
    out = tmp_path / "price"
    assert price(out) == 0
    said = capsys.readouterr().out
    assert f"capture run {CAPTURE} holds no public steps; its steps are run {STEPS_RUN}'s" in said
    assert "newest" not in said
    r = result(out)
    (m,) = r["measurements"]
    assert m["value"] == own["value"] and m["inputs"] == [own["mark"], CAPTURE, STEPS_RUN, MINIFY]
    assert r["workload_fingerprint"]["steps"] == STEPS_RUN
    assert not any((store[CAPTURE] / "out").glob(S.STEPS))
    assert not any(p.is_symlink() for p in (store[CAPTURE] / "out").rglob("*"))


def test_price_takes_the_newest_of_two_records_labelled_steps_of_a_capture_and_says_so(
        store, tmp_path, monkeypatch, capsys):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    steps_moved_out(store, tmp_path / "steps", STEPS_RUN)
    store[OLDER_STEPS_RUN] = record(tmp_path / "older", OLDER_STEPS_RUN, None)
    labelled(monkeypatch, [(OLDER_STEPS_RUN, S.STEPS_OF, CAPTURE, "2026-11-18T05:00:00Z"),
                           (STEPS_RUN, S.STEPS_OF, CAPTURE, "2026-11-18T06:00:00Z"),
                           ("r20261118-000130-0012", S.STEPS_OF, CAPTURE, "2026-11-18T07:00:00Z"),
                           ("r20261118-000130-0012", S.STEPS_OF, "r20261118-000140-0013", "2026-11-18T08:00:00Z")])
    assert price(tmp_path / "price") == 0
    said = capsys.readouterr().out
    assert (f"its steps are run {STEPS_RUN}'s (labelled `steps_of={CAPTURE}` at 2026-11-18T06:00:00Z), the newest of "
            f"2 so labelled (also {OLDER_STEPS_RUN})") in said


def test_price_refuses_a_capture_with_neither_its_own_steps_nor_a_record_labelled_steps_of_it(
        store, tmp_path, monkeypatch):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    steps_moved_out(store, tmp_path / "steps", STEPS_RUN)
    labelled(monkeypatch, [(STEPS_RUN, S.STEPS_OF, "r20261118-000140-0013", "2026-11-18T05:54:14Z")])
    with pytest.raises(SystemExit, match=f"capture run {CAPTURE} holds no public steps .*, and no record is labelled "
                                         f"`steps_of={CAPTURE}`: make one with `benchmarks/serve_tax/serve_tax.py "
                                         f"steps-of-trace RUN_OUT --to DIR` over run {CAPTURE}'s fetched out/"):
        price(tmp_path / "price")


def test_price_refuses_a_record_labelled_steps_of_a_capture_that_holds_no_steps(store, tmp_path, monkeypatch):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    steps_moved_out(store, tmp_path / "steps", STEPS_RUN)
    store[OLDER_STEPS_RUN] = record(tmp_path / "older", OLDER_STEPS_RUN, None)
    labelled(monkeypatch, [(OLDER_STEPS_RUN, S.STEPS_OF, CAPTURE, "2026-11-18T09:00:00Z")])
    with pytest.raises(SystemExit, match=f"run {OLDER_STEPS_RUN} is labelled `steps_of={CAPTURE}` but holds no "
                                         "public steps"):
        price(tmp_path / "price")


def test_price_of_a_capture_holding_its_own_steps_reads_no_label(store, tmp_path, monkeypatch):
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    monkeypatch.setattr(S, "labels_store", lambda: pytest.fail("read the labels"))
    assert price(tmp_path / "price") == 0
    assert "steps" not in result(tmp_path / "price")["workload_fingerprint"]
