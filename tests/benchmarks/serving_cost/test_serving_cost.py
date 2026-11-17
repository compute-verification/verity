from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.serving_cost import run as S  # noqa: E402
from verity_bench import kit  # noqa: E402

MINIFY, EXACT, CAPTURE, PRICE, MEASURED = (f"r20261118-0000{i}0-000{i}" for i in range(1, 6))
#: (rows, keys): (base present, base cut, a call with every row failed, present then cut), µs over the fast kernel
SHAPES = {(6, 20000): (7.0, 12.0, 100.0, 22.0), (6, 131072): (12.0, 16.0, 1000.0, 50.0),
          (24, 20000): (10.0, 15.0, 100.0, 34.0), (24, 131072): (24.0, 28.0, 1000.0, 110.0)}
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
    d = record(tmp, MINIFY, S.KINDS["minify"])
    lines: list[dict] = [{"K": 2048}]
    for (rows, keys), (bp, bc, fp, fc) in SHAPES.items():
        for dist, e in (("randn", 0.0), ("indexer", 0.5)):
            lines.append({"dist": dist, "rows": rows, "keys": keys,
                          "present": {"added_us": bp + e, "per_call": {"rows_long": rows}},
                          "cut": {"added_us": bc + e, "per_call": {"rows_long": rows}}})
        if units:
            for how in ("missed", "overflow"):
                lines.append({"forced": how, "rows": rows, "keys": keys, "present": {"us": fp}, "cut": {"us": fc}})
    (d / S.UNITS).write_text("".join(json.dumps(x) + "\n" for x in lines))
    return d


def lifetime(out: Path, name: str, seconds: float, mode: str | None, decode: tuple[tuple[int, int], ...] = ()) -> None:
    d = out / name
    for rep, t0 in ((1, 0.0), (2, 100.0)):
        p = d / "agentic@4" / f"rep-{rep}"
        p.mkdir(parents=True)
        (p / "meta.json").write_text(json.dumps({"load": "agentic@4", "rep": rep, "t0_unix": t0, "wall_s": 10.0}))
        (p / "result.json").write_text(json.dumps({"duration": seconds}))
    (d / "meta.json").write_text(json.dumps({"left_out": 1, "env": {"VERITY_DSA_TOPK": mode} if mode else {},
                                             "server_argv": ["--tensor-parallel-size", "8"]}))
    steps = [{"kind": "header", "config": {"speculative": {"num_speculative_tokens": 5}}},
             {"kind": "step", "step": 0, "rows": [["p", 0, 4000]], "t": 101.0}]
    steps += [{"kind": "step", "step": i + 1, "rows": [[f"r{j}", start, 6] for j in range(n)], "t": 102.0 + i}
              for i, (n, start) in enumerate(decode)]
    (d / "server").mkdir()
    with gzip.open(d / "server" / "step_trace.1.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in steps))


def capture_record(tmp: Path, *, below: bool = True, lifetimes: bool = True, **fingerprint) -> Path:
    fp = {"minify": MINIFY, "served": "glm-5.3", "card": "h200"} | fingerprint
    d = record(tmp, CAPTURE, S.KINDS["capture"], fingerprint={k: v for k, v in fp.items() if v is not None})
    out = d / "out"
    for k in (1, 2) if lifetimes else ():
        lifetime(out, f"prod-trace.{k}", 50.0, None)
        lifetime(out, f"prod-record.{k}", 52.0, "canonical", ((4, 131066), (1, 19994)))
    counters = {"rows_long": 1, "rows_bisected": 80, "rows_refined": 1600, "rows_at": 1, "calls": 1}
    (out / "topk_counters.json").write_text(json.dumps({"dirs": {f"prod-record.{k}": counters for k in (1, 2)}}))
    layers = [{"arm": "prod-record", "over": "prod-trace", "value": 0.02, "ci95": [0.01, 0.03], "pairs": 2}]
    if below:
        layers.append({"arm": "prod-trace", "over": "prod", "value": 0.006, "ci95": [-0.01, 0.02], "pairs": 2})
    (out / "prod_repro.json").write_text(json.dumps({"run": CAPTURE, "layers": {"each": layers}}))
    return d


def result(out: Path) -> dict:
    return json.loads((out / kit.RESULT).read_text())


def price(out: Path) -> int:
    return S.main(["price", "--capture", CAPTURE, "--minify", MINIFY, "--ours", "prod-record", "--out", str(out)])


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
    assert "top-k, present pass, no row failed, 6 rows of 20000 keys: 7.0 to 7.5" in said
    assert "top-k, cut pass, every row failed, over the base, 24 rows of 131072 keys: 81.5 to 82.0" in said
    assert "unpriced: top-k, prefill steps" in said


def test_minify_refuses_a_record_without_units_in_priced_s_form(store, tmp_path):
    store[MINIFY] = minify_record(tmp_path, units=False)
    with pytest.raises(SystemExit, match="is not in the form `priced.units` reads: .*no present times"):
        S.main(["minify", MINIFY])
    (store[MINIFY] / S.UNITS).unlink()
    with pytest.raises(SystemExit, match=f"minify run {MINIFY} holds no out/units.jsonl"):
        S.main(["minify", MINIFY])


def test_exact_refuses_past_minify_saying_what_has_no_command(store, tmp_path):
    store[MINIFY] = minify_record(tmp_path)
    with pytest.raises(SystemExit, match="exact: no command makes its record yet.*circuit-check.*replay.py"):
        S.main(["exact", "--minify", MINIFY])


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
    ({}, False, "holds no arm's lifetimes"),
])
def test_capture_refuses_a_record_that_breaks_the_contract(store, tmp_path, fingerprint, lifetimes, why):
    store[MINIFY] = minify_record(tmp_path)
    store[CAPTURE] = capture_record(tmp_path, lifetimes=lifetimes, **fingerprint)
    with pytest.raises(SystemExit, match=f"capture run {CAPTURE}'s .*{why}"):
        S.main(["capture", CAPTURE])


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
    priced = json.loads((out / "priced.json").read_text())
    assert m["value"] == pytest.approx(100 * (priced["layers"][0]["cost"]["mid"] + 0.006))
    assert r["validation"]["evidence"] == ["priced.json"]


def test_price_with_nothing_unpriced_is_modeled(store, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "UNPRICED", ())
    store[MINIFY], store[CAPTURE] = minify_record(tmp_path), capture_record(tmp_path)
    assert price(tmp_path / "price") == 0
    (m,) = result(tmp_path / "price")["measurements"]
    assert m["mark"] == "modeled" and m["unpriced"] == []


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
