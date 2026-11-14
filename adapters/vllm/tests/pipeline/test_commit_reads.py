"""`verity_vllm.pipeline.commit` on records that leave a block out: the stop rule of record over versions that record no engine
arguments, the summary over an instrumented run with no Commit record, and the readers `main` reads a workload, a stop rule of
record, a collector's extra, a run's Commit record, an oracle compare and the versions through, and the engine's schedule of
record where the collector recorded no request rows.  A block that is absent, null, empty or zero reads alike: each is tested
for its truth.  The stand-in engine, the workload and the rows are `test_commit_main.py`'s.  CPU only."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.pipeline.test_commit_main import (
    PLUGIN,
    STOP_RULE,
    WORKLOAD,
    Clock,
    Record,
    control_row,
    engine,
    instrumented_row,
)
from verity_vllm.engine import vllm_adapter as va
from verity_vllm.pipeline import commit

ABSENT = "<absent>"
#: what stands in a block's place where a record has none: the block left out, null, and each empty or zero value
NOTHING: list[Any] = [ABSENT, None, {}, [], "", 0, False]
NOT_ATTESTED = "the Program declares LAG 1 but this engine does not attest async_scheduling: EOS stops fail by name"
OTHER_SCHEDULER = ("the Program declares LAG 1 but this engine resolved async_scheduling=False (LAG 0): not an execution of the declared "
                   "scheduler -- EOS stops fail by name")


def holding(key: str, value: Any) -> Record:
    """A record with `value` at `key`, or without the key (ABSENT)."""
    return {} if value is ABSENT else {key: value}


def stop_rule(tmp_path: Path, versions: Record) -> Record:
    """The stop rule of record of `WORKLOAD` on the stand-in engine, whose versions are `versions`."""
    (tmp_path / "workload.json").write_text(json.dumps(WORKLOAD))
    return commit.stop_rule_of_record(engine(Clock()), va.load_workload(tmp_path / "workload.json"), versions)


@pytest.mark.parametrize("kwargs", [*NOTHING, {"enforce_eager": True}])
def test_versions_that_record_no_scheduling_attest_none_and_the_lag_is_left_undeclared(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                                                         kwargs: Any) -> None:
    rule = stop_rule(tmp_path, holding("engine_kwargs", kwargs))
    assert rule == {**STOP_RULE, "lag": None, "engine_lag": None, "async_scheduling": None, "notes": [NOT_ATTESTED]}
    assert capsys.readouterr().out.splitlines() == [f"[delta] {NOT_ATTESTED}"]


def test_an_engine_that_resolved_the_declared_scheduling_keeps_the_declared_lag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert stop_rule(tmp_path, {"engine_kwargs": {"async_scheduling": True}}) == STOP_RULE
    assert capsys.readouterr().out == ""


def test_an_engine_that_resolved_another_scheduling_is_named_and_the_lag_is_left_undeclared(tmp_path: Path,
                                                                                              capsys: pytest.CaptureFixture[str]) -> None:
    rule = stop_rule(tmp_path, {"engine_kwargs": {"async_scheduling": False}})
    assert rule == {**STOP_RULE, "lag": None, "engine_lag": 0, "async_scheduling": False, "notes": [OTHER_SCHEDULER]}
    assert capsys.readouterr().out.splitlines() == [f"[delta] {OTHER_SCHEDULER}"]


def summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[Record]) -> Record:
    """The summary of `rows`, with no step rows, no versions and no declaration."""
    monkeypatch.setattr(commit, "time", Clock())
    return commit.summarize(rows, [], SimpleNamespace(workload=str(tmp_path / "workload.json"), committer=PLUGIN), {}, None)


@pytest.mark.parametrize("record", NOTHING)
def test_the_summary_of_a_run_without_a_commit_record_fails_and_lists_nothing_for_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                                      record: Any) -> None:
    row = control_row(tmp_path, 0, 0, arm="instrumented") | holding("commit", record)
    made = summary(tmp_path, monkeypatch, [row])
    assert (made["pairs"], made["commit_pass"]) == (0, False)
    assert (made["commit_fail_reasons"], made["expected_tensors"], made["committed_tensors"]) == ([None], [None], [None])


def test_the_summary_lists_each_instrumented_runs_tensor_counts_and_the_reason_of_each_that_failed(tmp_path: Path,
                                                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [instrumented_row(tmp_path, 0, 0, "aa", ok=False, why="openings"), instrumented_row(tmp_path, 1, 0, "bb")]
    made = summary(tmp_path, monkeypatch, rows)
    assert (made["pairs"], made["commit_pass"]) == (0, False)
    assert (made["commit_fail_reasons"], made["expected_tensors"], made["committed_tensors"]) == (["openings"], [3, 3], [3, 3])


@pytest.mark.parametrize("required", NOTHING)
def test_a_workload_that_requires_no_engine_argument_gives_none(required: Any) -> None:
    assert commit.required_engine_args(holding("engine_args_required", required)) == {}


def test_a_workloads_required_engine_arguments_are_the_ones_it_holds() -> None:
    required = {"max_num_seqs": 4}
    assert commit.required_engine_args({"engine_args_required": required}) is required


@pytest.mark.parametrize("srule", [*NOTHING, {"eos_token_ids": [2]}])
def test_a_stop_rule_of_record_that_read_no_rule_or_no_source_names_no_eos_source(srule: Any) -> None:
    assert commit.eos_source(holding("srule", srule)) is None


def test_a_stop_rule_of_record_names_the_source_its_rule_read_the_eos_ids_from() -> None:
    assert commit.eos_source({"srule": {"eos_token_ids": [2], "source": "generation_config.json"}}) == "generation_config.json"


@pytest.mark.parametrize("errors", NOTHING)
def test_a_collector_that_gives_no_count_and_lists_no_error_counts_none(errors: Any) -> None:
    assert commit.collector_errors_n(holding("errors", errors)) == 0


def test_a_collectors_count_is_its_own_and_where_it_gives_none_the_length_of_its_list() -> None:
    assert commit.collector_errors_n({"errors": ["e1", "e2"]}) == 2
    assert commit.collector_errors_n({"errors": ["e1", "e2"], "errors_n": 5}) == 5
    assert commit.collector_errors_n({"errors_n": None}) is None


@pytest.mark.parametrize("binding", [*NOTHING, {"error": "KeyError('step')"}, *({"coverage": c} for c in NOTHING[1:]),
                                     {"coverage": {"ok": False}}, {"coverage": {"missing_n": 1}}])
def test_a_run_whose_binding_records_no_covered_population_is_not_covered(binding: Any) -> None:
    assert commit.coverage_ok(holding("binding", binding)) is False


def test_a_run_whose_binding_covered_its_population_is_covered() -> None:
    assert commit.coverage_ok({"binding": {"coverage": {"ok": True, "missing_n": 0}}}) is True


@pytest.mark.parametrize("attribution", NOTHING)
def test_an_oracle_compare_that_made_no_attribution_gives_an_empty_one(attribution: Any) -> None:
    assert commit.attribution_of(holding("attribution", attribution)) == {}


def test_an_oracle_compares_attribution_is_the_one_it_holds() -> None:
    attribution = {"complete": True, "pairs": 3}
    assert commit.attribution_of({"attribution": attribution}) is attribution


@pytest.mark.parametrize("compilation", [*NOTHING, "eager", ["x"], {"compilation_mode": "NONE"}])
def test_versions_that_record_no_compilation_or_no_generated_kernels_give_none(compilation: Any) -> None:
    assert commit.generated_kernels_of(holding("compilation", compilation)) is None


def test_the_generated_kernels_are_the_ones_the_versions_compilation_holds() -> None:
    kernels = {"loaded": {"modules": 3}}
    assert commit.generated_kernels_of({"compilation": {"generated_kernels": kernels}}) is kernels


NO_ROWS = "no per-step request rows (single request or none recorded): declared arrive_steps"
#: the fields of the engine's schedule of record that say where each request ran and whether the declaration stands
PLACEMENT = ("declared_arrive_steps", "observed_arrive_steps", "equal_declared", "all_equal_declared", "scheduled_steps", "contiguous",
             "complete", "use", "why")


def placement(request_rows: Any, declared: dict[str, int], expected_steps: dict[str, int]) -> Record:
    """The engine's schedule of record for requests declared to arrive at `declared` and expected to run `expected_steps`."""
    of_record = commit.engine_schedule(request_rows, declared, expected_steps)
    return {field: of_record[field] for field in PLACEMENT}


@pytest.mark.parametrize("request_rows", [None, {}])
def test_a_collector_that_recorded_no_request_rows_leaves_the_declared_schedule(request_rows: Any) -> None:
    assert placement(request_rows, {"r0": 0}, {"r0": 32}) == {
        "declared_arrive_steps": {"r0": 0}, "observed_arrive_steps": {}, "equal_declared": None, "all_equal_declared": False,
        "scheduled_steps": {}, "contiguous": None, "complete": False, "use": "declared", "why": NO_ROWS}


def test_the_schedule_of_record_places_each_request_at_the_first_committed_step_it_ran_in() -> None:
    request_rows = {2: [("r1", 0, 1)], 0: [("r0", 0, 1)], 1: [("r0", 0, 1), ("r1", 1, 2)]}
    assert placement(request_rows, {"r0": 0, "r1": 5}, {"r0": 2, "r1": 2}) == {
        "declared_arrive_steps": {"r0": 0, "r1": 5}, "observed_arrive_steps": {"r0": 0, "r1": 1},
        "equal_declared": {"r0": True, "r1": False}, "all_equal_declared": False, "scheduled_steps": {"r0": 2, "r1": 2},
        "contiguous": True, "complete": True, "use": "observed", "why": None}
