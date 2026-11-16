"""A point's results: the schema and identity, its metrics and their units, validate, tables and intervals."""
from __future__ import annotations

import json
import statistics
from dataclasses import replace

import pytest
from bench_standins import FakeFactory, inputs

from verity_bench.cli import main
from verity_bench.vllm import presets
from verity_bench.vllm import result as result_format
from verity_bench.vllm.arms import (
    Arm,
)
from verity_bench.vllm.presets import (
    expand_sweep,
)
from verity_bench.vllm.result import (
    IdentityMismatch,
    identity_mismatches,
    judge,
    merge,
    validate,
)
from verity_bench.vllm.runner import (
    METRICS,
    OPTIONAL_METRICS,
    _aggregate,
    run_benchmark,
    summarize,
)
from verity_bench.vllm.table import render
from verity_bench.vllm.timing import (
    decode_time_per_token,
    inter_token_latencies,
    percentile,
)


def test_a_sweep_that_samples_other_than_greedily_refuses(tmp_path, monkeypatch):
    quick = (presets.SPEC_ROOT / "quick.toml").read_text().replace('id = "quick"', 'id = "odd"')
    sampled = quick.replace("[output_tokens]", 'sampling = ["greedy", "stochastic"]\n\n[output_tokens]')
    (tmp_path / "odd.toml").write_text(sampled)
    monkeypatch.setattr(presets, "SPEC_ROOT", tmp_path)
    with pytest.raises(presets.ConfigError, match="every point samples greedily"):
        presets.load_sweep("odd")


def test_inter_token_latency_is_per_token_after_the_first_and_shares_a_multi_token_return():
    returns = [
        (1.0, {"a": 1}),
        (1.5, {"a": 2, "b": 1}),
        (2.0, {}),
        (2.5, {"a": 4, "b": 2}),
        (3.0, {"a": 4}),
    ]
    assert inter_token_latencies(returns) == [0.5, 0.5, 0.5, 1.0]
    assert inter_token_latencies([(1.0, {"a": 1})]) == []
    assert percentile([0.5, 0.5, 0.5, 1.0], 50) == 0.5
    assert percentile([0.5, 0.5, 0.5, 1.0], 99) == pytest.approx(0.985)
    assert percentile([2.0], 99) == 2.0 and percentile([], 50) is None


def test_result_schema_and_identity_strict_judge():
    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc")
    validate(doc)
    assert doc["identity"]["arms"][0]["engine_overrides"] == {"quantization": "fp8"}
    other = {**doc, "identity": {**doc["identity"], "git_commit": "def"}}
    with pytest.raises(IdentityMismatch, match="git_commit"):
        judge(doc, other)
    assert judge(doc, other, ("git_commit",))["accepted"]


def test_a_points_numbers_are_measurements_of_its_result(tmp_path, monkeypatch):
    preset, points, arms = inputs()
    doc = run_benchmark(preset, arms, points[:1], sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc")
    monkeypatch.setattr(result_format.kit, "host_stamp", lambda: {"commit": "abc", "card": "none"})
    (tmp_path / "result.json").write_text(json.dumps({"schema": "research/result/v0.1", "kind": "verity-vllm-benchmark/v1",
                                                     "run_id": "r1", "measurements": []}))
    result_format.record(doc, result_format.kit.Writer(tmp_path, "r1", result_format.kit.host_stamp()),
                         "out/benchmark.json")
    held = json.loads((tmp_path / "result.json").read_text())
    assert held["kind"] == "verity-vllm-benchmark/v1"
    got = {m["name"]: m for m in held["measurements"]}
    (row,) = doc["points"]
    other = next(a for a in row["arms"] if a != doc["baseline"])
    tpot = row["arms"][other]["metrics"]["time_per_output_token_s"]
    assert got[f"{other}.time_per_output_token_s"]["value"] == tpot["mean"]
    assert got[f"{other}.time_per_output_token_s"]["ci95"] == tpot["ci95"]
    assert got[f"{other}.time_per_output_token_s"]["unit"] == "s"
    assert got[f"{other}.time_per_output_token_s.ratio"]["value"] == tpot["ratio_to_stock"]["mean"]
    assert f"{doc['baseline']}.time_per_output_token_s.ratio" not in got
    assert {m["mark"] for m in got.values()} == {"measured"}
    assert {m["point"] for m in got.values()} == {row["point"]["id"]}
    assert len(got) == sum(len(a["metrics"]) * (1 if arm == doc["baseline"] else 2) for arm, a in row["arms"].items())


def test_every_metric_a_point_measures_has_a_unit():
    """`record` reads each metric's unit from `UNITS`, so a metric the runner measures without one would fail the
    record mid-write, after the run."""
    assert set(result_format.UNITS) == {*METRICS, *OPTIONAL_METRICS}


def test_ratio_table_puts_the_headline_cell_first():
    preset, points, arms = inputs()
    points = (replace(points[0], batch=8), points[0])
    doc = run_benchmark(preset, arms, points, sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc")
    lines = [line for line in render(doc).splitlines() if line.startswith("| b")]
    assert lines[0].startswith("| b32-p512-mixed **headline** | time / output token |")
    assert "2.000×" in lines[0]


def test_decode_time_per_token_counts_only_decode_only_forwards():
    steps = [
        {"forward": True, "s": 0.9, "prefill_tokens": 512, "decode_tokens": 0},
        {"forward": True, "s": 0.5, "prefill_tokens": 64, "decode_tokens": 7},
        {"forward": True, "s": 0.2, "prefill_tokens": 0, "decode_tokens": 8},
        {"forward": True, "s": 0.2, "prefill_tokens": 0, "decode_tokens": 8},
        {"forward": False, "s": 0.01, "prefill_tokens": 0, "decode_tokens": 0},
    ]
    assert decode_time_per_token(steps) == pytest.approx(0.4 / 16)
    assert decode_time_per_token([{**s, "prefill_tokens": None, "decode_tokens": None} for s in steps]) is None
    assert decode_time_per_token([]) is None
    # 23 decode tokens across all steps (7 + 8 + 8); a run that generated fewer had prompt tokens counted as decode
    assert decode_time_per_token(steps, generated=23) == pytest.approx(0.4 / 16)
    assert decode_time_per_token(steps, generated=22) is None


def test_the_decode_metric_is_reported_only_where_every_rep_has_it():
    arms = (Arm("stock"), Arm("candidate", control="stock"))
    point = expand_sweep("quick")[0]

    def rep(i, tpot, decode):
        metrics = {"time_per_output_token_s": tpot, "time_to_first_token_s": 0.1, "tokens_per_s": 1 / tpot,
                   "peak_gpu_memory_bytes": 1.0}
        return {"rep": i, "metrics": metrics if decode is None else {**metrics, "decode_time_per_token_s": decode}}

    full = {("stock", point.id): [rep(0, 1.0, 0.5), rep(1, 1.0, 0.5)],
            ("candidate", point.id): [rep(0, 2.0, 1.5), rep(1, 2.0, 1.5)]}
    [row] = _aggregate(arms, (point,), "stock", full)
    assert row["arms"]["candidate"]["metrics"]["decode_time_per_token_s"]["ratio_to_stock"]["mean"] == 3
    doc = {"baseline": "stock", "identity": {"arms": [{"id": "stock"}, {"id": "candidate"}]}, "points": [row]}
    assert "decode time / token" in render(doc)

    partial = {**full, ("candidate", point.id): [rep(0, 2.0, 1.5), rep(1, 2.0, None)]}
    [row] = _aggregate(arms, (point,), "stock", partial)
    assert "decode_time_per_token_s" not in row["arms"]["candidate"]["metrics"]
    doc["points"] = [row]
    assert "decode time / token" not in render(doc)


THREE_ARMS = (Arm("stock-fp8", precision="fp8"), Arm("a", control="stock-fp8"), Arm("b", control="stock-fp8"))


def bench(factory=None, *, arms=None, reps=2, points=None):
    preset, default_points, default_arms = inputs()
    return run_benchmark(preset, arms or default_arms, points or default_points, sweep="quick", reps=reps,
                         factory=factory or FakeFactory(), git_commit="abc")


@pytest.mark.parametrize("resident_failure", [None, "a"])
def test_drift_over_the_run_cancels_in_every_arms_mean_over_even_reps(resident_failure):
    """Each timed run is 1% slower than the one before (clocks, heat). Every arm's mean timed
    position is the same, so the means keep the arms' true 2x in either mode, and the paired ratios stay within the
    drift's second-order residue; blocks used to repeat its first cycle's order in the second, so the baseline always
    ran first and the last arm read 2.08x."""

    doc = bench(FakeFactory(drift=0.01, resident_failure=resident_failure), arms=THREE_ARMS, reps=4)
    assert doc["execution"]["interleave"] == ("blocks" if resident_failure else "resident")
    tpot = {a: doc["points"][0]["arms"][a]["metrics"]["time_per_output_token_s"] for a in ("stock-fp8", "a", "b")}
    for arm in ("a", "b"):
        assert tpot[arm]["mean"] / tpot["stock-fp8"]["mean"] == pytest.approx(2)
        assert tpot[arm]["ratio_to_stock"]["mean"] == pytest.approx(2, rel=0.005)


def test_a_blocks_engine_unlike_the_gated_one_fails_engine_identity_and_times_nothing_more():
    """In `blocks` the gate's engine and the timed engines are different lifetimes, and the
    identity recorded the gated one's KV pool, so a timed engine with another pool went unnoticed."""

    factory = FakeFactory(resident_failure="candidate", kv_after_gate={"candidate": 491_376})
    doc = bench(factory)
    [gate] = [g for g in doc["gates"] if g["kind"] == "engine-identity" and not g["passed"]]
    assert gate["arm"] == "candidate" and gate["differs"] == {"kv_cache_tokens": [458_752, 491_376]}
    assert doc["points"] == [] and not any(e[:2] == ("measure", "candidate") for e in factory.events)


def test_decode_time_charges_a_step_to_the_forward_it_waited_on_and_refuses_unseen_device_work():
    """Under async scheduling (output lag 1) a step's host time holds the previous forward's device
    time, so a decode-only step after a prefill carried the prefill: 1.1 s over 16 tokens instead of 0.4 s. Device
    work still running after the last step means the host steps missed it, and the metric is dropped."""

    steps = [
        {"forward": True, "s": 0.05, "prefill_tokens": 512, "decode_tokens": 0},
        {"forward": True, "s": 0.9, "prefill_tokens": 0, "decode_tokens": 8},
        {"forward": True, "s": 0.2, "prefill_tokens": 0, "decode_tokens": 8},
        {"forward": False, "s": 0.2, "prefill_tokens": 0, "decode_tokens": 0},
    ]
    assert decode_time_per_token(steps) == pytest.approx(1.1 / 16)
    assert decode_time_per_token(steps, output_lag=1) == pytest.approx(0.4 / 16)
    assert decode_time_per_token(steps, output_lag=1, device_tail_s=0.001) == pytest.approx(0.4 / 16)
    assert decode_time_per_token(steps, output_lag=1, device_tail_s=0.01) is None


def test_validate_refuses_dropped_reordered_or_edited_reps():
    """A result's points are recomputed from its reps, so a caller can't drop a slow rep, reorder,
    edit a mean or change `reps` and still have `judge` accept it."""

    doc = bench(reps=3)
    validate(doc)
    row = doc["points"][0]["arms"]["candidate"]
    reps = row["reps"]
    for forged in (
        {**row, "reps": reps[:2]},
        {**row, "reps": [reps[1], reps[0], reps[2]]},
        {**row, "metrics": {**row["metrics"], "tokens_per_s": {**row["metrics"]["tokens_per_s"], "mean": 99.0}}},
    ):
        bad = {**doc, "points": [{**doc["points"][0], "arms": {**doc["points"][0]["arms"], "candidate": forged}}]}
        with pytest.raises(ValueError):
            validate(bad)
    with pytest.raises(ValueError, match="reps"):
        validate({**doc, "execution": {**doc["execution"], "reps": 2}})
    with pytest.raises(IdentityMismatch, match="reps"):
        judge(doc, bench(reps=2))
    with pytest.raises(IdentityMismatch, match="interleave"):
        judge(doc, bench(FakeFactory(resident_failure="candidate"), reps=3))


def test_an_allow_hides_only_the_paths_it_names_and_judge_lists_what_it_hid():
    """`--allow precision` used to match every leaf named precision (each arm's as well as the
    preset's), and `--allow id` every arm id."""

    doc = bench()
    arms = [dict(a) for a in doc["identity"]["arms"]]
    arms[1].update(id="other", precision="int4", kernel_digest="digest-x")
    other = {**doc, "identity": {**doc["identity"], "arms": arms}}
    for allow in (("id",), ("precision",), ("kernel_digest",)):
        assert identity_mismatches(doc, other, allow)
    digest_only = {**doc, "identity": {**doc["identity"], "arms": [arms[0], {**doc["identity"]["arms"][1],
                                                                              "kernel_digest": "digest-x"}]}}
    verdict = judge(doc, digest_only, ("arms[*].kernel_digest",))
    assert verdict["differences"] == {"arms[1].kernel_digest": ["digest-candidate", "digest-x"]}


def test_table_merges_one_specs_point_results_only_when_their_identities_agree(tmp_path):
    """A dispatched spec runs each point as its own run, so a sweep's table is assembled from
    several results; `verity-bench table` refuses results whose identities differ anywhere but the point's own."""

    preset, (p32,), arms = inputs()
    p8 = replace(p32, batch=8)

    def point(p, identity, **kw):
        return run_benchmark(preset, arms, (p,), sweep="quick", reps=2, factory=FakeFactory(), git_commit="abc",
                             config_identity={"point_identity": identity, **kw})

    merged = merge([point(p8, "i8"), point(p32, "i32")])
    assert [p["point"]["id"] for p in merged["points"]] == ["b8-p512-mixed", "b32-p512-mixed"]
    assert merged["identity"]["config"]["point_identity"] == ["i8", "i32"]
    with pytest.raises(IdentityMismatch, match="uv_lock_sha256"):
        merge([point(p8, "i8", uv_lock_sha256="a"), point(p32, "i32", uv_lock_sha256="b")])
    with pytest.raises(ValueError, match="two results"):
        merge([point(p32, "i32"), point(p32, "i32")])
    files = []
    for i, doc in enumerate((point(p8, "i8"), point(p32, "i32"))):
        files.append(tmp_path / f"{i}.json")
        files[-1].write_text(json.dumps(doc))
    assert main(["table", *map(str, files), "--out", str(tmp_path / "merged.json")]) == 0
    validate(json.loads((tmp_path / "merged.json").read_text()))


def test_an_arm_cannot_be_its_own_control():
    """An arm that controlled itself would have its greedy gate compare it with itself and always pass."""

    with pytest.raises(ValueError, match="own control"):
        bench(arms=(Arm("stock-fp8", precision="fp8"), Arm("candidate", control="candidate")))


@pytest.mark.parametrize("kv_tokens", [{"candidate": None}, {"stock-fp8": None}, {"stock-fp8": None, "candidate": None}])
def test_an_unreported_kv_pool_fails_the_gate(kv_tokens):
    """The kv-pool gate used to be skipped when either engine didn't report its pool."""

    doc = bench(FakeFactory(kv_tokens=kv_tokens))
    [gate] = [g for g in doc["gates"] if g["kind"] == "kv-pool"]
    assert not gate["passed"] and "unchecked" in gate["detail"] and doc["points"] == []


def test_an_arm_whose_timed_tokens_differ_from_its_controls_keeps_no_points():
    """The greedy gate ran before timing only, so an arm could pass it and compute something else
    in its timed reps."""

    doc = bench(FakeFactory(timed_digest={"candidate": "other"}))
    [gate] = [g for g in doc["gates"] if g["kind"] == "timed-tokens" and g["control"]]
    assert not gate["passed"] and gate["reps"] == [0, 1] and doc["points"] == []
    validate(doc)


def test_an_arm_without_a_control_is_timed_on_its_own_gated_tokens_in_resident_mode():
    """An uncontrolled arm must still produce in every timed rep the tokens its gate saw from the same engine; blocks mode
    times other engines, so it has no such gate, and neither has an arm whose own gate says its tokens carry each pass's
    coins (`PER_PASS_TOKENS`: a Pearl-C protocol arm, whose every call's noise is seeded by its index across the epoch, so
    its timed reps never repeat its gated tokens), whose closing gate alone holds its timed reps."""

    doc = bench(FakeFactory(timed_digest={"stock-fp8": "other"}))
    [gate] = [g for g in doc["gates"] if g["kind"] == "timed-tokens" and g["arm"] == "stock-fp8"]
    assert (gate["against"], gate["passed"], gate["reps"]) == ("gate", False, [0, 1]) and doc["points"] == []
    good = bench()
    assert all(g["passed"] for g in good["gates"] if g["kind"] == "timed-tokens") and good["points"]
    blocks = bench(FakeFactory(resident_failure="candidate"))
    assert blocks["execution"]["interleave"] == "blocks"
    assert not [g for g in blocks["gates"] if g["kind"] == "timed-tokens" and g["control"] is None]
    arms = (Arm("stock-fp8", precision="fp8"), Arm("pearl"))
    per_pass = bench(FakeFactory(timed_digest={"pearl": "other"}, per_pass_tokens={"pearl"}), arms=arms)
    assert per_pass["execution"]["interleave"] == "resident" and per_pass["points"]
    assert [g["arm"] for g in per_pass["gates"] if g["kind"] == "timed-tokens"] == ["stock-fp8"]
    assert [g["passed"] for g in per_pass["gates"] if g["kind"] == "arm-closing" and g["arm"] == "pearl"] == [True]
    closing = bench(FakeFactory(timed_digest={"pearl": "other"}, per_pass_tokens={"pearl"}, timed_ok=False), arms=arms)
    assert closing["points"] == [] and not all(g["passed"] for g in closing["gates"] if g["kind"] == "arm-closing")
    validate(per_pass)
    good = bench()
    with pytest.raises(ValueError, match="lacks gates"):
        validate({**good, "gates": [g for g in good["gates"] if g["kind"] != "tokens"]})


def test_intervals_use_the_t_quantile_for_n_and_ratios_refuse_a_zero_or_missing_baseline_rep():
    """From n = 12 the interval used 1.96 instead of t (11 df: 2.201), too narrow; a zero baseline
    rep was dropped from the ratio, which then averaged the other reps; n = 1 rendered like a measured interval."""

    assert summarize([1.0])["n"] == 1 and "ci95" not in summarize([1.0])
    assert summarize([1.0, 3.0])["ci95"] == pytest.approx([2 - 12.706, 2 + 12.706])
    assert summarize([1.0, 2.0, 3.0])["ci95"] == pytest.approx([2 - 4.303 / 3**0.5, 2 + 4.303 / 3**0.5])
    values = [float(i % 2) for i in range(12)]
    half = 2.201 * statistics.stdev(values) / 12**0.5
    assert summarize(values)["ci95"] == pytest.approx([0.5 - half, 0.5 + half])

    arms = (Arm("stock"), Arm("candidate", control="stock"))
    point = expand_sweep("quick")[0]

    def rep(i, tpot):
        return {"rep": i, "metrics": {"time_per_output_token_s": tpot, "time_to_first_token_s": 0.1,
                                      "tokens_per_s": 1.0, "peak_gpu_memory_bytes": 1.0}}

    samples = {("stock", point.id): [rep(0, 0.0), rep(1, 1.0)], ("candidate", point.id): [rep(0, 2.0), rep(1, 9.0)]}
    [row] = _aggregate(arms, (point,), "stock", samples)
    assert row["arms"]["candidate"]["metrics"]["time_per_output_token_s"]["ratio_to_stock"] == {"mean": None, "n": 0}
    doc = {"baseline": "stock", "identity": {"arms": [{"id": "stock"}, {"id": "candidate", "control": "stock"}]},
           "points": [row]}
    assert "| time / output token | — | — |" in render(doc)
    with pytest.raises(ValueError, match="same reps"):
        _aggregate(arms, (point,), "stock", {**samples, ("stock", point.id): [rep(0, 1.0)]})
    assert "2.000× (n=1)" in render(bench(reps=1))
