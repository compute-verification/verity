"""What the config record reads of a row dir's documents when a block of one is left out (`verity_vllm.pipeline.config_record`).

Each part of the record is read from a document a stage wrote: the circuit database's index, a fold Match's verdict, a Commit's
summary, a line of `commit/runs.jsonl`, a Build's summary, a manifest, a sampled replay's result.  A block of such a document may be
absent, null or empty (a stage that stopped early, a writer that had nothing to state), and the record then states that part with
nothing in it.  Each test gives a block absent, null, empty and stated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verity_vllm.pipeline import config_record as CR

ROW = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"
#: `replay_of` of a result that states nothing.
NO_REPLAY: dict[str, Any] = {"ok": False, "result": None, "grade": None, "strata_by": None, "k": None, "picked": None, "evaluated": None,
                             "equal": None, "mismatch_n": None, "not_evaluated_n": None, "seed": None, "seed_source": None,
                             "differed": [], "by_family": None, "why": None}
#: One rank of a TP Build whose summary states nothing of it.
NO_RANK: dict[str, Any] = {"ok": False, "programs": {}, "census_digests": {}, "workload_digest": None}
NO_OPENINGS = {"n": None, "verified": None}
COUNTS = ("k", "picked", "evaluated", "equal", "mismatch_n", "not_evaluated_n")


def _put(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _runs(d: Path, *runs: dict[str, Any]) -> None:
    (d / "commit").mkdir(exist_ok=True)
    (d / "commit" / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))


def test_a_row_the_index_lists_with_nothing_or_does_not_list_is_on_the_trace(tmp_path):
    """`aot_of` is a switched row's `aot`; a row the index does not list, lists as null or lists with no `aot` has none."""
    assert CR.aot_of("r", tmp_path / "no_index.json") is None
    on_the_trace: list[dict[str, Any]] = [{}, {"r": None}, {"r": {}}, {"r": {"record": None}}, {"other": {"aot": {"trace": "t"}}}]
    for rows in on_the_trace:
        assert CR.aot_of("r", _put(tmp_path / "records.json", {"schema": CR.INDEX_SCHEMA, "rows": rows})) is None, rows
    aot = {"comparison": "art:c", "trace": "art:t"}
    assert CR.aot_of("r", _put(tmp_path / "records.json", {"schema": CR.INDEX_SCHEMA, "rows": {"r": {"aot": aot}}})) == aot


def test_a_fold_matchs_problems_are_its_first_sixteen_and_none_where_it_states_none(tmp_path):
    verdict = tmp_path / "match" / "fold_match" / "tp2_fold_match.json"
    stated_none: list[dict[str, Any]] = [{}, {"problems": None}, {"problems": []}]
    for problems in stated_none:
        _put(verdict, {"pass": True, "schema": "s", **problems})
        got = CR.match_part(tmp_path, 2)
        assert got is not None and (got["ok"], got["problems"]) == (True, []), problems
    _put(verdict, {"pass": False, "problems": [f"p{i}" for i in range(20)]})
    got = CR.match_part(tmp_path, 2)
    assert got is not None and (got["ok"], got["problems"]) == (False, [f"p{i}" for i in range(16)])


def test_a_commit_of_records_summary_reads_without_its_manifest_its_roots_and_its_replay():
    """`summary_part`: the required manifest, the run roots (a TP Commit's first, else every pair's), the per-rank roots and the
    sampled replay, each absent, null and empty."""
    blocks = ("required_manifest", "tp_run_roots", "run_roots", "per_rank_roots", "sampled_replay")
    left_out: list[dict[str, Any]] = [{}, dict.fromkeys(blocks), {"required_manifest": {}, "tp_run_roots": [], "run_roots": [],
                                                                  "per_rank_roots": [], "sampled_replay": {}}]
    no_replay = {"ok": False, "result": None, **dict.fromkeys(COUNTS[1:]), "population": None, "per_rank": None}
    for sm in left_out:
        commit, replay = CR.summary_part(sm)
        assert commit["manifest"] == {"manifest_digest": None, "identities": None, "program_digest_per_rank": None}, sm
        assert (commit["run_root"], commit["run_roots"], commit["per_rank"]) == (None, [], None), sm
        assert replay == no_replay, sm
    inner: list[dict[str, Any]] = [{}, {"sample": None, "population": None, "per_rank": None}, {"sample": {}, "population": {}, "per_rank": {}}]
    for part in inner:
        _, replay = CR.summary_part({"sampled_replay": {"result": True, **part}})
        assert replay == {**no_replay, "ok": True, "result": True}, part


def test_a_commit_of_records_summary_states_its_manifest_its_roots_and_its_replay():
    sample = {"picked": 4, "evaluated": 4, "equal": 4, "mismatch_n": 0, "not_evaluated_n": 0}
    sm = {"pass": True, "required_manifest": {"manifest_digest": "m", "identities": {"a": 1}, "program_digest_per_rank": ["p0", "p1"], "more": 1},
          "tp_run_roots": ["t", "t"], "run_roots": ["x"], "per_rank_roots": [{"0": "a0", "1": "a1"}, "no pair", {"0": "b0", "1": "b1"}],
          "sampled_replay": {"result": True, "sample": sample, "population": {"by_family": {"gemm": 9}},
                             "per_rank": {"10": {"result": True, "grade": "A", "more": 1}, "2": {"result": False, "grade": "B"}, "3": "no rank"}}}
    commit, replay = CR.summary_part(sm)
    assert commit["manifest"] == {"manifest_digest": "m", "identities": {"a": 1}, "program_digest_per_rank": ["p0", "p1"]}
    assert (commit["ok"], commit["run_root"], commit["run_roots"]) == (True, "t", ["t", "t"])
    assert commit["per_rank"] == {"0": {"run_roots": ["a0", "b0"]}, "1": {"run_roots": ["a1", "b1"]}}
    assert replay == {"ok": True, "result": True, **sample, "population": {"gemm": 9},
                      "per_rank": {"2": {"result": False, "grade": "B"}, "10": {"result": True, "grade": "A"}}}
    assert list(replay["per_rank"]) == ["2", "10"]
    no_tp_roots: list[dict[str, Any]] = [{}, {"tp_run_roots": None}, {"tp_run_roots": []}]
    for tp in no_tp_roots:
        commit, _ = CR.summary_part({"run_roots": ["x", "y"], **tp})
        assert (commit["run_root"], commit["run_roots"]) == (None, ["x", "y"]), tp
    assert CR.summary_part({"run_roots": ["x", "x"]})[0]["run_root"] == "x"
    _, replay = CR.summary_part({"sampled_replay": {"result": True, "sample": {"mismatch_n": 2}}})
    assert (replay["ok"], replay["mismatch_n"]) == (False, 2)
    _, replay = CR.summary_part({"sampled_replay": {"per_rank": {"0": "no rank", "1": None}}})
    assert replay["per_rank"] is None   # ranks none of which states a block are no ranks


def test_a_builds_part_reads_a_summary_without_its_workload_and_a_manifest_without_its_query(tmp_path):
    """A Build's `workload` that is absent or null is no part of it; one that is empty is a part that did not pass.  The query id is
    the manifest's `query.query_id`, None where the manifest or its `query` is absent, null or empty."""
    step = {"ok": True, "program_digest": "aa", "census_digest": "cc"}
    for workload, ok in (({}, True), ({"workload": None}, True), ({"workload": {}}, False)):
        _put(tmp_path / "build_summary.json", {"step": step, **workload})
        got = CR.build_part(tmp_path)
        assert (got["ok"], got["workload_digest"], got["programs"], got["census_digests"]) == (ok, None, {"step": "aa"}, {"step": "cc"}), workload
        assert (got["manifest_digest"], got["query_id"]) == (None, None)
    _put(tmp_path / "build_summary.json", {"step": step, "workload": {"ok": True, "workload_digest": "ww"}})
    no_query: list[dict[str, Any]] = [{}, {"query": None}, {"query": {}}]
    for query in no_query:
        _put(tmp_path / "manifest.json", {"manifest_digest": "mm", **query})
        got = CR.build_part(tmp_path)
        assert (got["ok"], got["workload_digest"], got["manifest_digest"], got["query_id"]) == (True, "ww", "mm", None), query
    _put(tmp_path / "manifest.json", {"manifest_digest": "mm", "query": {"query_id": "q"}})
    assert CR.build_part(tmp_path)["query_id"] == "q"
    _put(tmp_path / "manifest_header.json", {"manifest_digest": "hh", "query": {"query_id": "of the header"}})
    got = CR.build_part(tmp_path)
    assert (got["manifest_digest"], got["query_id"]) == ("hh", "of the header")


def test_a_sampled_replays_result_reads_without_its_sample_and_its_mismatches():
    left_out: list[dict[str, Any]] = [{}, {"sample": None, "mismatch": None, "mismatch_by": None}, {"sample": {}, "mismatch": [], "mismatch_by": {}}]
    for sr in left_out:
        assert CR.replay_of(sr) == NO_REPLAY, sr
        assert CR.replay_of({"result": True, **sr}) == {**NO_REPLAY, "ok": True, "result": True}, sr
    sr = {"result": True, "grade": "mismatch", "sample": {"k": 8, "mismatch_n": 70, "seed": 3, "strata_by": "family"},
          "mismatch": [{"i": i, "family": "gemm", "more": 1} for i in range(70)], "mismatch_by": {"by_family": {"gemm": 70}}, "why": "w"}
    assert CR.replay_of(sr) == {**NO_REPLAY, "result": True, "grade": "mismatch", "k": 8, "mismatch_n": 70, "seed": 3, "strata_by": "family",
                                "differed": [{"i": i, "family": "gemm"} for i in range(64)], "by_family": {"gemm": 70}, "why": "w"}


def test_a_commits_part_reads_a_run_without_its_commit_and_a_commit_without_its_blocks(tmp_path):
    """`commit_part` reads the last instrumented run of `commit/runs.jsonl`: its `commit`, and in it the openings, the coverage and
    the sampled replay."""
    no_commit: list[dict[str, Any]] = [{}, {"commit": None}, {"commit": {}}]
    for commit in no_commit:
        _runs(tmp_path, {"arm": "control", "commit": {"commit_pass": True}}, {"arm": "instrumented", "wall_s": 3.0, **commit})
        got, replay = CR.commit_part(tmp_path)
        assert got == {"ok": False, "reason": None, "runs": 1, "run_root": None, "openings": NO_OPENINGS, "coverage_ok": None, "wall_s": 3.0,
                       "tensors": {"expected": None, "committed": None}, "engine_steps": None, "host_mem_kb": None}, commit
        assert replay == NO_REPLAY, commit
    blocks = ("openings", "coverage", "sampled_replay")
    left_out: list[dict[str, Any]] = [{}, dict.fromkeys(blocks), dict.fromkeys(blocks, {})]
    for inner in left_out:
        _runs(tmp_path, {"arm": "instrumented", "commit": {"commit_pass": True, "run_root": "rr", **inner}})
        got, replay = CR.commit_part(tmp_path)
        assert (got["ok"], got["run_root"], got["openings"], got["coverage_ok"]) == (True, "rr", NO_OPENINGS, None), inner
        assert replay == NO_REPLAY, inner
    _runs(tmp_path, {"arm": "instrumented", "commit": {"commit_pass": True, "openings": {"n": 5, "verified": 4, "more": 1}, "coverage": {"ok": True},
                                                       "sampled_replay": {"result": True, "grade": "complete", "sample": {"k": 2}}}})
    got, replay = CR.commit_part(tmp_path)
    assert (got["openings"], got["coverage_ok"]) == ({"n": 5, "verified": 4}, True)
    assert replay == {**NO_REPLAY, "ok": True, "result": True, "grade": "complete", "k": 2}


def test_a_tp_builds_part_reads_a_summary_without_its_ranks_and_a_rank_without_its_workload(tmp_path):
    """`tp_build_part`: a rank the summary leaves out, null or empty has no Programs and did not pass, and so the Build did not."""
    assert CR.tp_build_part(tmp_path, 2)["per_rank"] == {"0": NO_RANK, "1": NO_RANK}
    no_ranks: list[dict[str, Any]] = [{}, {"ranks": None}, {"ranks": {}}]
    for ranks in no_ranks:
        _put(tmp_path / "build_summary.json", {"tp": {"world": 2}, **ranks})
        got = CR.tp_build_part(tmp_path, 2)
        assert (got["ok"], got["world"], got["per_rank"]) == (False, 2, {"0": NO_RANK, "1": NO_RANK}), ranks
        assert (got["manifest_digest"], got["query_id"], got["program_digest_per_rank"]) == (None, None, None)
    rank0 = {"build_step": {"ok": True, "program_digest": "s0", "census_digest": "c0"}, "workload": {"ok": True, "workload_digest": "w0"}}
    whole = {"ok": True, "programs": {"build_step": "s0"}, "census_digests": {"build_step": "c0"}, "workload_digest": "w0"}
    no_rank1: list[dict[str, Any]] = [{}, {"rank1": None}, {"rank1": {}}]
    for rank1 in no_rank1:
        _put(tmp_path / "build_summary.json", {"ranks": {"rank0": rank0, **rank1}})
        got = CR.tp_build_part(tmp_path, 2)
        assert (got["ok"], got["per_rank"]) == (False, {"0": whole, "1": NO_RANK}), rank1
    for workload, ok in (({}, True), ({"workload": None}, True), ({"workload": {}}, False)):
        _put(tmp_path / "build_summary.json", {"ranks": {"rank0": {"build_step": rank0["build_step"], **workload}}})
        got = CR.tp_build_part(tmp_path, 1)
        assert (got["ok"], got["per_rank"]) == (ok, {"0": {**whole, "ok": ok, "workload_digest": None}}), workload
    _put(tmp_path / "build_summary.json", {"ranks": {"rank0": rank0, "rank1": rank0}})
    no_query: list[dict[str, Any]] = [{}, {"query": None}, {"query": {}}]
    for query in no_query:
        _put(tmp_path / "manifest.json", {"manifest_digest": "mm", "program_digest_per_rank": ["s0", "s0"], **query})
        got = CR.tp_build_part(tmp_path, 2)
        assert (got["ok"], got["manifest_digest"], got["query_id"], got["program_digest_per_rank"]) == (True, "mm", None, ["s0", "s0"]), query
    _put(tmp_path / "manifest.json", {"query": {"query_id": "q"}})
    assert CR.tp_build_part(tmp_path, 2)["query_id"] == "q"


def test_a_tp_commits_part_reads_a_run_without_its_commit_and_a_commit_without_its_ranks(tmp_path):
    """`tp_commit_part`: with no rank stated the replay is not complete, so it did not pass, and its counts sum to 0."""
    no_commit: list[dict[str, Any]] = [{}, {"commit": None}, {"commit": {}}]
    for commit in no_commit:
        _runs(tmp_path, {"arm": "instrumented", "wall_s": 2.0, **commit})
        got, replay = CR.tp_commit_part(tmp_path, 2)
        assert got == {"ok": False, "reason": None, "runs": 1, "run_root": None, "openings": NO_OPENINGS, "tensors": {"committed": None},
                       "coverage_ok": None, "wall_s": 2.0, "engine_steps": None, "per_rank": {}}, commit
        assert replay == {"ok": False, "result": False, "grade": "", "strata_by": None, **dict.fromkeys(COUNTS, 0), "differed": [],
                          "by_family": {}, "why": None, "per_rank": {}}, commit
    left_out: list[dict[str, Any]] = [{}, {"per_rank": None, "openings": None}, {"per_rank": {}, "openings": {}}]
    for blocks in left_out:
        _runs(tmp_path, {"arm": "instrumented", "commit": {"commit_pass": True, "tp_run_root": "tt", **blocks}})
        got, replay = CR.tp_commit_part(tmp_path, 2)
        assert (got["ok"], got["run_root"], got["openings"], got["coverage_ok"], got["per_rank"]) == (True, "tt", NO_OPENINGS, None, {}), blocks
        assert (replay["ok"], replay["per_rank"]) == (False, {}), blocks


def test_a_tp_commits_part_reads_a_rank_without_its_blocks(tmp_path):
    """A rank's openings, `extra` (its tensor counts), `binding` and its coverage, and its sampled replay, each absent, null and
    empty: the rank's coverage is then not known, which the Commit's coverage reads as not passed."""
    blocks = ("openings", "extra", "binding", "sampled_replay")
    left_out: list[dict[str, Any]] = [{}, dict.fromkeys(blocks), dict.fromkeys(blocks, {}), {"binding": {"coverage": None}},
                                      {"binding": {"coverage": {}}}]
    for inner in left_out:
        rank = {"commit_pass": True, "run_root": "r", **inner}
        _runs(tmp_path, {"arm": "instrumented", "commit": {"commit_pass": True, "per_rank": {"0": rank, "1": rank}}})
        got, replay = CR.tp_commit_part(tmp_path, 2)
        bare = {"ok": True, "reason": None, "run_root": "r", "openings": NO_OPENINGS, "tensors": {"expected": None, "committed": None},
                "coverage_ok": None}
        assert (got["per_rank"], got["coverage_ok"]) == ({"0": bare, "1": bare}, False), inner
        assert (replay["per_rank"], replay["by_family"], replay["ok"]) == ({"0": NO_REPLAY, "1": NO_REPLAY}, {}, False), inner
    sr = {"result": True, "grade": "complete", "sample": dict.fromkeys(COUNTS, 2) | {"mismatch_n": 0, "not_evaluated_n": 0, "strata_by": "family"}}
    whole = {"commit_pass": True, "run_root": "r", "openings": {"n": 3, "verified": 3}, "extra": {"expected_tensors": 7, "committed_tensors": 6},
             "binding": {"coverage": {"ok": True}}, "sampled_replay": {**sr, "mismatch_by": {"by_family": {"gemm": 1}}}}
    _runs(tmp_path, {"arm": "instrumented", "commit": {"commit_pass": True, "openings": {"n": 6, "verified": 6, "more": 1},
                                                       "per_rank": {"0": whole, "1": {**whole, "sampled_replay": sr}}}})
    got, replay = CR.tp_commit_part(tmp_path, 2)
    assert got["per_rank"]["1"] == {"ok": True, "reason": None, "run_root": "r", "openings": {"n": 3, "verified": 3},
                                    "tensors": {"expected": 7, "committed": 6}, "coverage_ok": True}
    assert (got["coverage_ok"], got["openings"]) == (True, {"n": 6, "verified": 6})
    assert (replay["ok"], replay["result"], replay["grade"], replay["strata_by"]) == (True, True, "complete/complete", "family")
    assert {k: replay[k] for k in COUNTS} == {"k": 4, "picked": 4, "evaluated": 4, "equal": 4, "mismatch_n": 0, "not_evaluated_n": 0}
    assert (replay["by_family"], replay["per_rank"]["1"]["by_family"]) == ({"gemm": 1}, None)


def test_the_engine_of_record_is_none_where_the_run_recorded_no_replay(tmp_path):
    assert CR.engine_part(tmp_path) is None
    replay: list[Any] = [None, {}, {"engine_of_record": None}, {"engine_of_record": {}}]
    no_engine: list[dict[str, Any]] = [{}, {"commit": None}, {"commit": {}}, *({"commit": {"sampled_replay": sr}} for sr in replay)]
    for commit in no_engine:
        _runs(tmp_path, {"arm": "instrumented", **commit})
        assert CR.engine_part(tmp_path) is None, commit
    _runs(tmp_path, {"arm": "instrumented", "commit": {"sampled_replay": {"engine_of_record": {"VLLM_BATCH_INVARIANT": "1", "more": 2}}}})
    assert CR.engine_part(tmp_path) == {**dict.fromkeys(CR.ENGINE_KEYS), "VLLM_BATCH_INVARIANT": "1"}


def test_the_slowdown_needs_the_control_arm_and_is_none_per_phase_without_the_instrumented_one(tmp_path):
    arm = {"wall_s": 6.0, "step_wall_prefill_s": 3.0, "step_wall_decode_mean_s": 1.5, "engine_steps": 4}
    part = {"wall_s": 6.0, "prefill_s": 3.0, "decode_mean_s": 1.5, "engine_steps": 4}
    assert CR.timing_part(tmp_path) == {"slowdown": None}
    _runs(tmp_path, {"arm": "instrumented", **arm})
    assert CR.timing_part(tmp_path) == {"instrumented": part, "slowdown": None}
    _runs(tmp_path, {"arm": "control", **arm})
    assert CR.timing_part(tmp_path) == {"control": part, "slowdown": {"prefill": None, "decode": None, "wall": None}}
    _runs(tmp_path, {"arm": "control", **arm}, {"arm": "instrumented", **arm, "wall_s": 9.0, "step_wall_prefill_s": None})
    assert CR.timing_part(tmp_path)["slowdown"] == {"prefill": None, "decode": 1.0, "wall": 1.5}


def test_served_switches_that_failed_and_state_no_problem_still_fail_the_record_first(tmp_path):
    _put(tmp_path / "build_summary.json", {"step": {"ok": True, "program_digest": "aa"}})
    _put(tmp_path / "strict_word.json", {"ok": True})
    stated_none: list[dict[str, Any]] = [{}, {"problems": None}, {"problems": []}]
    for problems in stated_none:
        rec = CR.record(tmp_path, ROW, summary={"served": {"ok": False, **problems}})
        assert (rec["outcome"], rec["why"]) == ("FAIL", "served switches FAIL: "), problems
    rec = CR.record(tmp_path, ROW, summary={"served": {"ok": False, "problems": ["a", "b"]}})
    assert (rec["outcome"], rec["why"]) == ("FAIL", "served switches FAIL: a; b")
