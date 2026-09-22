from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm import commit, relations, render, required, rows, synthetic
from verity_vllm.negatives import NEGATIVE_IDS, NegativeHarness
from verity_vllm.rows import Common, RowError, validate_row, verdict

COMMON = Common("A0-abc123def456-20260907T0600Z", "A0", "abc123def456", "W2", "deadbeef")


def good_fields(**over):
    base = dict(
        workload={
            "requests": 2,
            "generated_positions": 8,
            "input_len_range": [4, 6],
            "peak_concurrency_achieved": 2,
        },
        production_changes="none",
        replay_restrictions=[],
        token_selection={"exact": 8, "attempted": 8},
        boundary_noninterference={
            "identical": True,
            "compared": 8,
            "mismatched_scalars": 0,
            "reference_method": "stock-vs-recording",
        },
        required_coverage={"bound": 14, "required": 14},
        authentication={"valid": 100, "attempted": 100},
        relation_checks={
            "same_runtime": {"exact": 0, "attempted": 0},
            "independent": {"exact": 40, "attempted": 40},
        },
        structural={"result": "ok", "failing_invariant": None},
        negative={"rejected": 9, "attempted": 9},
    )
    base.update(over)
    return base


def test_table1_pass_and_common_fields():
    row = rows.table1_row(COMMON, **good_fields())
    assert row["result"] == "PASS" and row["first_blocker"] is None
    assert set(rows.COMMON_FIELDS) <= set(row)
    assert validate_row(1, row) == []


@pytest.mark.parametrize(
    "over, result, blocker_prefix",
    [
        ({"authentication": {"valid": 99, "attempted": 100}}, "FAIL", "authentication"),
        ({"token_selection": {"exact": 7, "attempted": 8}}, "FAIL", "token_selection"),
        (
            {
                "relation_checks": {
                    "same_runtime": {"exact": 0, "attempted": 0},
                    "independent": {"exact": 39, "attempted": 40},
                }
            },
            "FAIL",
            "relation_checks.independent",
        ),
        (
            {
                "boundary_noninterference": {
                    "identical": False,
                    "compared": 8,
                    "mismatched_scalars": 1,
                    "reference_method": "x",
                }
            },
            "FAIL",
            "boundary_noninterference",
        ),
        ({"structural": {"result": "fail", "failing_invariant": "steps"}}, "FAIL", "structural"),
        ({"negative": {"rejected": 8, "attempted": 9}}, "FAIL", "negative"),
        ({"required_coverage": {"bound": 13, "required": 14}}, "PARTIAL", "required_coverage"),
        ({"required_coverage": {"bound": 0, "required": 0}}, "PARTIAL", "required_coverage"),
        (
            {
                "relation_checks": {
                    "same_runtime": {"exact": 0, "attempted": 0},
                    "independent": {"exact": 0, "attempted": 0},
                }
            },
            "PARTIAL",
            "relation_checks",
        ),
        ({"token_selection": {"exact": 0, "attempted": 0}}, "PARTIAL", "token_selection"),
        ({"negative": {"rejected": 0, "attempted": 0}}, "PARTIAL", "negative"),
    ],
)
def test_verdict_rule(over, result, blocker_prefix):
    row = rows.table1_row(COMMON, **good_fields(**over))
    assert row["result"] == result
    assert row["first_blocker"].startswith(blocker_prefix)


def test_fail_beats_partial():
    row = rows.table1_row(
        COMMON,
        **good_fields(
            authentication={"valid": 1, "attempted": 2},
            required_coverage={"bound": 1, "required": 2},
        ),
    )
    assert row["result"] == "FAIL"
    v = verdict(row)
    assert v.reasons[0].startswith("authentication") and any(
        r.startswith("required_coverage") for r in v.reasons
    )


def test_table2_rows_influence_verdict():
    t2_none = rows.table2_row(
        COMMON,
        op_id="attn.0",
        phase="decode",
        shape=["H"],
        dtype="bfloat16",
        layout_or_quant=None,
        runtime_impl="unresolved",
        primitive_semantics="attention.bf16.v1",
        decomposition="row",
        interface={"input_bytes": 0, "output_bits": 0, "range": None},
        origin="captured",
        binding={"present": 4, "unresolved": 0},
        checker="none",
        natural_checks={"exact": 0, "attempted": 0},
        stress_checks={"exact": 0, "attempted": 0, "domain": None},
        mutations={"rejected": 0, "attempted": 0},
        checker_work={"seconds": 0.0, "flops_or_bytes": 0.0, "includes_full_op": False},
        status="unimplemented",
        limitation="no executable attention relation",
    )
    row = rows.table1_row(COMMON, **good_fields(), table2_rows=[t2_none])
    assert row["result"] == "PARTIAL" and "attn.0" in row["first_blocker"]
    t2_bad = dict(
        t2_none,
        checker="independent",
        natural_checks={"exact": 3, "attempted": 4},
        status="mismatch",
    )
    row = rows.table1_row(COMMON, **good_fields(), table2_rows=[t2_bad])
    assert row["result"] == "FAIL"
    other_case = dict(t2_none, case="B1")
    row = rows.table1_row(COMMON, **good_fields(), table2_rows=[other_case])
    assert row["result"] == "PASS"


def test_blockers_only_without_evidence():
    nothing = good_fields(
        token_selection={"exact": 0, "attempted": 0},
        boundary_noninterference={
            "identical": False,
            "compared": 0,
            "mismatched_scalars": 0,
            "reference_method": "none",
        },
        authentication={"valid": 0, "attempted": 0},
        relation_checks={
            "same_runtime": {"exact": 0, "attempted": 0},
            "independent": {"exact": 0, "attempted": 0},
        },
        negative={"rejected": 0, "attempted": 0},
    )
    row = rows.table1_row(COMMON, **nothing, blocker={"kind": "BLOCKED", "reason": "no pod"})
    assert (row["result"], row["first_blocker"]) == ("BLOCKED", "no pod")
    row = rows.table1_row(COMMON, **nothing, blocker={"kind": "NOT_RUN"})
    assert row["result"] == "NOT_RUN"
    # with evidence present the blocker is ignored and the evidence decides
    row = rows.table1_row(COMMON, **good_fields(), blocker={"kind": "BLOCKED", "reason": "x"})
    assert row["result"] == "PASS"
    with pytest.raises(RowError):
        rows.table1_row(COMMON, **nothing, blocker={"kind": "MAYBE"})


def test_hand_assigned_pass_is_rejected():
    row = rows.table1_row(COMMON, **good_fields(required_coverage={"bound": 1, "required": 2}))
    assert row["result"] == "PARTIAL"
    forged = dict(row, result="PASS", first_blocker=None)
    problems = validate_row(1, forged)
    assert problems and "section 10 rule says PARTIAL" in problems[0]
    with pytest.raises(RowError):
        rows.require_valid(1, forged)


def test_validate_row_catches_schema_problems():
    row = rows.table1_row(COMMON, **good_fields())
    assert any("unknown field" in p for p in validate_row(1, dict(row, extra=1)))
    missing = dict(row)
    del missing["workload"]
    assert "workload: missing" in validate_row(1, missing)
    assert any(
        "expected int" in p
        for p in validate_row(1, dict(row, negative={"rejected": "9", "attempted": 9}))
    )
    assert any(
        "unknown field" in p
        for p in validate_row(1, dict(row, negative={"rejected": 9, "attempted": 9, "x": 1}))
    )
    assert any("must not be null" in p for p in validate_row(1, dict(row, production_changes=None)))
    assert any("not in" in p for p in validate_row(1, dict(row, result="MAYBE")))
    bool_as_int = dict(row, authentication={"valid": True, "attempted": 1})
    assert any("expected int" in p for p in validate_row(1, bool_as_int))
    assert any("non-empty string" in p for p in validate_row(1, dict(row, worker="")))
    t3 = {
        **COMMON.fields(),
        "negative_id": "N01",
        "attempted": 2,
        "rejected": 2,
        "unexpectedly_accepted": 1,
        "reproducer": None,
    }
    assert validate_row(3, t3) == ["rejected + unexpectedly_accepted > attempted"]


def test_table4_row_nullable_measurements():
    row = rows.table4_row(
        COMMON,
        partition_id="P0",
        instrumentation_mode="replay-commit",
        production_overhead_ratio=None,
        replay_commit_overhead_ratio=1.5,
        replay_vs_production_gpu_work=None,
        transparent_check_cost={"cpu_s": 0.1, "gpu_s": None},
        peak_replay_memory={"device_gb": None, "host_gb": 0.5, "delta_vs_reference": None},
        committed_logical_bytes=1234,
        processed_bytes={"hashed": 1234, "copied": 1234, "retained": 0, "emitted": 0},
        description_size_bytes=100,
        constructor_inputs_bytes=None,
        logical_size={"gates": None, "vus": 40, "rus": 8},
        lookup_cost_s=None,
        validation_cost_s=0.01,
        moe_padding={
            "inactive_units": None,
            "inactive_checks": None,
            "inactive_work_s": None,
            "inactive_bytes": None,
        },
        restrictions=["cpu-only toy"],
        spend={"gpu_hours": 0.0, "usd": 0.0},
    )
    assert validate_row(4, row) == []
    with pytest.raises(RowError):
        rows.table4_row(COMMON, partition_id="P0")  # everything else missing


def test_experiment_id_format():
    assert (
        rows.experiment_id("A0", "abc123def456", "20260907T0600Z")
        == "A0-abc123def456-20260907T0600Z"
    )


def test_jsonl_round_trip(tmp_path):
    row = rows.table1_row(COMMON, **good_fields())
    path = rows.write_jsonl(tmp_path / "r" / "table1.jsonl", [row])
    rows.append_row(path, 1, row)
    assert rows.read_jsonl(path) == [row, row]
    with pytest.raises(RowError):
        rows.append_row(path, 1, dict(row, result="MAYBE"))
    path.write_text("{not json\n")
    with pytest.raises(RowError):
        rows.read_jsonl(path)


# --- end to end on the toy: check + negatives -> four tables -> results.md ------------------------


@pytest.fixture(scope="module")
def toy_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    toy = synthetic.build_toy_run(root / "run", seed=3)
    reader = toy.reader()
    sample: list[str] = []
    for rid in reader.request_ids():
        sample += list(toy.description.expected_ids(reader.boundary(rid), reader.profile_id))
    report = relations.check(
        toy.description, reader, toy.commitments, sample, required_doc=toy.required
    )
    assert report.all_exact()
    negs = NegativeHarness(toy.run_dir, sample_size=6, seed=1).run()
    cov = required.coverage(toy.required, toy.description, reader)
    linkage = [
        relations.check_output_linkage(toy.description, reader, toy.commitments, rid)
        for rid in reader.request_ids()
    ]
    structural_problems = (
        reader.validate()
        + commit.verify_commitments(reader, toy.commitments)
        + [p for lr in linkage for p in lr.problems]
    )
    common = Common(
        rows.experiment_id("TOY", toy.profile_id, "20260907T0600Z"),
        "TOY",
        toy.profile_id,
        "W2",
        "toy",
    )
    sampler = report.families()["sampler"]["independent"]
    table3 = rows.table3_rows(common, negs)
    table2 = rows.table2_rows_from_report(
        common, report, toy.description, runtime_impl={"embed_tokens": "numpy"}
    )
    boundaries = [reader.boundary(rid) for rid in reader.request_ids()]
    prompt_lens = [len(b.prompt_token_ids) for b in boundaries]
    table1 = rows.table1_row(
        common,
        workload={
            "requests": len(boundaries),
            "generated_positions": sum(len(b.output_token_ids) for b in boundaries),
            "input_len_range": [min(prompt_lens), max(prompt_lens)],
            "peak_concurrency_achieved": 1,
        },
        production_changes="none",
        replay_restrictions=["synthetic numpy toy; no vLLM"],
        token_selection={"exact": sampler.exact, "attempted": sampler.attempted},
        boundary_noninterference={
            "identical": True,
            "compared": sum(lr.checked for lr in linkage),
            "mismatched_scalars": 0,
            "reference_method": "output-linkage (same capture; no independent production run)",
        },
        required_coverage={"bound": cov.bound, "required": cov.required},
        authentication=report.authentication(),
        relation_checks=report.relation_checks(),
        structural={
            "result": "ok" if not structural_problems else "fail",
            "failing_invariant": structural_problems[0] if structural_problems else None,
        },
        negative=rows.negative_summary(table3),
        table2_rows=table2,
    )
    table4 = rows.table4_row(
        common,
        partition_id=toy.required["partition_id"],
        instrumentation_mode="synthetic",
        production_overhead_ratio=None,
        replay_commit_overhead_ratio=None,
        replay_vs_production_gpu_work=None,
        transparent_check_cost={
            "cpu_s": sum(r.checker_seconds for r in report.results),
            "gpu_s": None,
        },
        peak_replay_memory={"device_gb": None, "host_gb": None, "delta_vs_reference": None},
        committed_logical_bytes=sum(r.output_bytes for r in report.results),
        processed_bytes={"hashed": None, "copied": None, "retained": None, "emitted": None},
        description_size_bytes=len(json.dumps(toy.description.to_json())),
        constructor_inputs_bytes=None,
        logical_size={"gates": None, "vus": len(sample), "rus": len(reader.ru_keys())},
        lookup_cost_s=None,
        validation_cost_s=None,
        moe_padding={
            "inactive_units": None,
            "inactive_checks": None,
            "inactive_work_s": None,
            "inactive_bytes": None,
        },
        restrictions=["cpu-only synthetic toy"],
        spend={"gpu_hours": 0.0, "usd": 0.0},
    )
    results_dir = root / "results"
    exp_dir = results_dir / common.experiment_id
    rows.write_jsonl(exp_dir / "table1.jsonl", [table1])
    rows.write_jsonl(exp_dir / "table2.jsonl", table2)
    rows.write_jsonl(exp_dir / "table3.jsonl", table3)
    rows.write_jsonl(exp_dir / "table4.jsonl", [table4])
    (exp_dir / "commitments.json").write_text(json.dumps(toy.commitments))
    (exp_dir / "log.txt").write_text("toy\n")
    return {
        "root": root,
        "results_dir": results_dir,
        "exp_dir": exp_dir,
        "common": common,
        "description": toy.description,
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "table4": table4,
        "report": report,
    }


def test_toy_tables_are_consistent(toy_results):
    t1, t2, t3 = toy_results["table1"], toy_results["table2"], toy_results["table3"]
    assert t1["result"] == "PASS", t1["first_blocker"]
    assert t1["negative"]["attempted"] == t1["negative"]["rejected"] > 0
    assert [r["negative_id"] for r in t3] == list(NEGATIVE_IDS)
    assert all(r["reproducer"] for r in t3)
    ops = {r["op_id"] for r in t2}
    assert {"embed_tokens", "sampler.select", "lm_head.gemm"} <= ops
    for r in t2:
        assert r["checker"] == "independent" and r["status"] == "exact"
        assert r["natural_checks"]["exact"] == r["natural_checks"]["attempted"] > 0
        assert r["binding"]["unresolved"] == 0 and r["checker_work"]["includes_full_op"] is False
    emb = next(r for r in t2 if r["op_id"] == "embed_tokens")
    assert emb["runtime_impl"] == "numpy" and emb["primitive_semantics"] == "embedding.row.v1"
    assert emb["interface"]["output_bits"] > 0
    assert {r["phase"] for r in t2} == {"prefill", "decode"}


def test_table2_from_report_marks_unimplemented_as_none(toy_results):
    report = toy_results["report"]
    common = toy_results["common"]
    doc = json.loads(json.dumps(report.to_json()))  # serialisable
    assert doc["statuses"] == {"exact": len(report.results)}
    patched = copy.deepcopy(report)
    flipped = False
    for r in patched.results:
        if r.op == "embed_tokens":
            r.status, r.detail = "unimplemented", "slot"
        elif r.op == "sampler.select" and r.status == "exact" and not flipped:
            r.status, flipped = "mismatch", True
    t2 = rows.table2_rows_from_report(common, patched, toy_results["description"])
    emb = [r for r in t2 if r["op_id"] == "embed_tokens"]
    assert emb and all(r["checker"] == "none" and r["status"] == "unimplemented" for r in emb)
    sampler = [r for r in t2 if r["op_id"] == "sampler.select" and r["status"] == "mismatch"]
    assert len(sampler) == 1
    assert sampler[0]["natural_checks"]["exact"] == sampler[0]["natural_checks"]["attempted"] - 1
    t1 = rows.table1_row(common, **good_fields(), table2_rows=t2)
    assert t1["result"] == "FAIL" and "sampler.select" in t1["first_blocker"]
    t1 = rows.table1_row(common, **good_fields(), table2_rows=emb)
    assert t1["result"] == "PARTIAL" and "embed_tokens" in t1["first_blocker"]


def test_render_results_md(toy_results):
    out = toy_results["root"] / "docs" / "vllm-poc" / "results.md"
    path, collected = render.render(toy_results["results_dir"], out)
    assert path == out and collected.issues == []
    text = out.read_text()
    exp = toy_results["common"].experiment_id
    assert (
        "## Table 1" in text
        and "## Table 2" in text
        and "## Table 3" in text
        and "## Table 4" in text
    )
    assert f"[{exp}](../../results/{exp}/)" in text  # relative link from docs/vllm-poc to results
    assert "| N01 |" in text and "| N14 |" in text
    assert "commitments.json, log.txt" in text
    assert "## Issues\n\n_none_" in text
    # deterministic
    render.render(toy_results["results_dir"], out)
    assert out.read_text() == text


def test_render_sorts_by_case_and_flags_forged_pass(toy_results, tmp_path):
    results_dir = tmp_path / "results"
    exp_dir = results_dir / toy_results["common"].experiment_id
    exp_dir.mkdir(parents=True)
    rows.write_jsonl(exp_dir / "table1.jsonl", [toy_results["table1"]])
    rows.write_jsonl(exp_dir / "table2.jsonl", toy_results["table2"])
    # a second experiment, case A0, sorted first; hand-assigned PASS with coverage missing
    forged_common = Common("A0-abc123def456-20260907T0700Z", "A0", "abc123def456", "W1", "r1")
    forged = rows.table1_row(
        forged_common, **good_fields(required_coverage={"bound": 1, "required": 2})
    )
    forged["result"] = "PASS"
    forged_dir = results_dir / forged_common.experiment_id
    rows.write_jsonl(forged_dir / "table1.jsonl", [forged])
    # and a broken row in a third experiment
    broken_dir = results_dir / "ZZ-000000000000-20260907T0800Z"
    broken_dir.mkdir()
    (broken_dir / "table3.jsonl").write_text(json.dumps({"negative_id": "N01"}) + "\n")
    out = tmp_path / "results.md"
    _, collected = render.render(results_dir, out, link_base="../fixtures/results")
    text = out.read_text()
    assert text.index("| [A0-") < text.index("| [TOY-")
    assert "**PASS** (rule: PARTIAL)" in text
    assert any("says PASS but the section 10 rule" in i for i in collected.issues)
    assert any("experiment_id: missing" in i for i in collected.issues)
    assert (
        "[A0-abc123def456-20260907T0700Z](../fixtures/results/A0-abc123def456-20260907T0700Z/)"
        in text
    )


def test_render_cli(toy_results, tmp_path):
    out = tmp_path / "results.md"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "verity_vllm.render",
            "--results",
            str(toy_results["results_dir"]),
            "--out",
            str(out),
            "--check",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 0, proc.stderr
    assert "1 experiments, 0 issues" in proc.stdout
    assert out.exists()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "verity_vllm.render",
            "--results",
            str(tmp_path / "nowhere"),
            "--out",
            str(out),
            "--check",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert proc.returncode == 1 and "does not exist" in proc.stderr


def test_escaping_and_formatting_helpers():
    assert render.esc("a|b\nc") == "a\\|b c"
    assert render.num(1.23456789) == "1.23"
    assert render.num(2130.0) == "2130" and render.num(2.5e9) == "2500000000"
    assert render.num(True) == "yes" and render.num(None) == "-"
    assert render.shape_cell(["H"]) == "[H]" and render.shape_cell([]) == "[]"
    assert render.shape_cell(None) == "-"
    assert render.compact({"a": 1, "b": None}) == "a=1, b=-"
    assert render.compact([1, [2, 3]]) == "1; 2; 3"
    assert render.frac({"exact": 1, "attempted": 2}, "exact", "attempted") == "1/2"
    assert render.frac(None, "a", "b") == "-"
