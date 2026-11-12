"""The cross-rank check's refusals that no other test reaches (`verity_vllm.check.match.rank_match`): for each, the smallest
change to an input that passes, held to the check's name and to the failure's words.  `cross_rank_check` fails by name, so
a refusal that moves to another check, or loses its text, fails here; `commit_binding` states its failures in `failures`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.check.match import rank_match as RM

SITES = ((0, "all_reduce", "model.layers.0.self_attn.o_proj"), (1, "all_gather", "lm_head"))


def _colls(rank: int, world: int) -> list[dict]:
    """One rank's fold collectives of one step: an all_reduce (k 0) and an all_gather (k 1), with one peers field per other rank."""
    return [{"step": 0, "k": k, "kind": kind, "site": site, "rows": 1, "N": 8,
             "fields": [{"request": "r0", "rows": 1, "peer_rank": p} for p in range(world) if p != rank]} for k, kind, site in SITES]


def _recorder(rank: int, world: int) -> list[dict]:
    """The recorder's rows of the same two collectives: every rank received the same bytes, each sent its own partial."""
    return [{"step": 0, "k": k, "kind": kind, "module": site, "shape": [1, 8], "out_shape": [1, 8 * (world if kind == "all_gather" else 1)],
             "part_sha256": f"part-{rank}-{k}", "out_sha256": f"out-{k}"} for k, kind, site in SITES]


def _world(world: int = 2) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    return {r: _colls(r, world) for r in range(world)}, {r: _recorder(r, world) for r in range(world)}


def _fails(res: dict, name: str) -> list[str]:
    return next(c for c in res["checks"] if c["name"] == name)["fails"]


def _only(res: dict, name: str, words: str) -> None:
    """`name` is the one check that failed, and one of its failures holds `words`."""
    assert res["verdict"] == "FAIL" and res["failed"] == [name], res["failed"]
    assert any(words in f for f in _fails(res, name)), _fails(res, name)


@pytest.mark.parametrize("world", [2, 3])
def test_the_inputs_these_tests_change_pass(world):
    colls, recorder = _world(world)
    res = RM.cross_rank_check(colls, recorder=recorder, producers={r: {(0, 0): ["Gemm_v1{K=8}"], (0, 1): ["Gemm_v1{K=8}"]} for r in colls})
    assert res["schema"] == "tp2/rank_match/v1" and res["verdict"] == "PASS" and res["failed"] == [], res
    assert [c["name"] for c in res["checks"]] == ["count", "order", "op", "segmentation", "site", "shard_linkage", "producer", "step", "recorder", "value"]


def test_a_rank_that_states_one_collective_twice_is_refused_before_any_other_check():
    colls, _ = _world()
    colls[1].append(dict(colls[1][0]))
    res = RM.cross_rank_check(colls)
    assert res["verdict"] == "FAIL"
    assert res["checks"] == [{"name": "count", "pass": False, "fails": ["rank 1: duplicate (step, k) among its collectives"]}]


def test_collective_ordinals_that_skip_a_number_fail_count():
    colls, _ = _world()
    for r in colls:
        colls[r][1]["k"] = 2
    res = RM.cross_rank_check(colls)
    _only(res, "count", "rank 0 step 0: collective ordinals are [0, 2].., not 0..1")
    assert len(_fails(res, "count")) == 2


@pytest.mark.parametrize("peer, words", [(0, "names peer rank 0 (not another rank)"), (5, "names peer rank 5 (not another rank)")])
def test_a_peers_field_that_names_no_other_rank_fails_shard_linkage(peer, words):
    colls, _ = _world()
    colls[0][0]["fields"][0]["peer_rank"] = peer
    _only(RM.cross_rank_check(colls), "shard_linkage", "rank 0 (step 0, k 0): peers field for request 'r0' " + words)


def test_a_peers_field_of_a_request_the_peer_has_no_partial_of_fails_shard_linkage():
    colls, _ = _world()
    colls[0][0]["fields"][0]["request"] = "r9"
    res = RM.cross_rank_check(colls)
    assert res["failed"] == ["segmentation", "shard_linkage"], res["failed"]
    assert "rank 0 (step 0, k 0): peers field for request 'r9' has no own partial on rank 1" in _fails(res, "shard_linkage")


def test_beyond_two_ranks_every_request_needs_a_peers_field_for_every_other_rank():
    colls, _ = _world(3)
    del colls[0][0]["fields"][1]
    _only(RM.cross_rank_check(colls), "shard_linkage", "rank 0 (step 0, k 0): peers fields cover (request, peer) [('r0', 1)].., every request x every other rank required")


def test_an_own_partial_of_a_class_that_produces_no_partial_fails_producer():
    colls, _ = _world()
    producers = {r: {(0, 0): ["Softmax_v1{N=8}"], (0, 1): ["Gemm_v1{K=8}"]} for r in colls}
    res = RM.cross_rank_check(colls, producers=producers)
    _only(res, "producer", "rank 0 (step 0, k 0): own partial produced by ['Softmax_v1{N=8}'], not an admitted partial producer")


def test_a_recorder_with_one_row_twice_fails_recorder():
    colls, recorder = _world()
    recorder[0].append(dict(recorder[0][0]))
    _only(RM.cross_rank_check(colls, recorder=recorder), "recorder", "rank 0: duplicate (step, k) in the recorder")


def test_a_recorder_row_of_another_module_fails_recorder():
    colls, recorder = _world()
    recorder[1][0]["module"] = "model.layers.1.mlp.down_proj"
    words = "rank 1 (0, 0): recorder module 'model.layers.1.mlp.down_proj' vs fold site 'model.layers.0.self_attn.o_proj'"
    _only(RM.cross_rank_check(colls, recorder=recorder), "recorder", words)


def test_a_recorder_row_without_its_partials_digest_fails_value():
    colls, recorder = _world()
    del recorder[1][1]["part_sha256"]
    _only(RM.cross_rank_check(colls, recorder=recorder), "value", "rank 1 (0, 1): recorder row without a partial digest")


def test_a_provenance_record_is_not_bound_without_the_recorder_files_digests():
    colls, recorder = _world()
    provenance = {"pass": True, "collectives_checked": 2, "recorder_sha256": {"0": "h0", "1": "h1"}, "tensors_digested": 8}
    bound = RM.cross_rank_check(colls, recorder=recorder, provenance=provenance, recorder_sha256={0: "h0", 1: "h1"})
    assert bound["verdict"] == "PASS", bound["failed"]
    res = RM.cross_rank_check(colls, recorder=recorder, provenance=provenance)
    _only(res, "provenance", "(recorder_sha256=None): the provenance record cannot be bound to the rows loaded")


def test_a_fold_row_that_states_no_fields_has_no_rows_by_request():
    colls, recorder = _world()
    del colls[1][0]["fields"]
    res = RM.cross_rank_check(colls, recorder=recorder)
    assert res["verdict"] == "FAIL" and res["failed"] == ["segmentation", "shard_linkage"], res["failed"]
    assert _fails(res, "segmentation") == ["(step 0, k 0): rows by request {'r0': 1} vs {}"]
    assert _fails(res, "shard_linkage") == ["rank 0 (step 0, k 0): peers field for request 'r0' has no own partial on rank 1"]


def test_a_recorder_row_without_a_shape_has_neither_the_folds_nor_the_definitions():
    colls, recorder = _world()
    del recorder[0][0]["shape"]
    _only(RM.cross_rank_check(colls, recorder=recorder), "recorder", "rank 0 (0, 0): recorder partial shape None vs fold [1, 8]")
    colls, recorder = _world()
    del recorder[1][1]["out_shape"]
    _only(RM.cross_rank_check(colls, recorder=recorder), "value", "rank 1 (0, 1): received shape None vs the Definition's [1, 16]")


def test_a_failed_provenance_counts_the_mismatches_it_lists_where_it_states_no_count():
    colls, recorder = _world()
    digests = {0: "h0", 1: "h1"}
    bare = {"pass": False, "collectives_checked": 2, "recorder_sha256": {"0": "h0", "1": "h1"}, "tensors_digested": 8}
    res = RM.cross_rank_check(colls, recorder=recorder, provenance=bare, recorder_sha256=digests)
    assert res["failed"] == ["provenance"] and _fails(res, "provenance") == ["provenance FAIL: 0 mismatches, first null"]
    listed = dict(bare, mismatches=[{"step": 0, "k": 1}], digest_mismatches=[{"rank": 1, "k": 0}, {"rank": 1, "k": 1}])
    res = RM.cross_rank_check(colls, recorder=recorder, provenance=listed, recorder_sha256=digests)
    assert _fails(res, "provenance") == ['provenance FAIL: 1 mismatches, first {"step": 0, "k": 1}',
                                         '2 stored tensors do not digest to their recorder rows, first {"rank": 1, "k": 0}']
    stated = dict(listed, n_mismatches=7, n_digest_mismatches=9)
    res = RM.cross_rank_check(colls, recorder=recorder, provenance=stated, recorder_sha256=digests)
    assert _fails(res, "provenance") == ['provenance FAIL: 7 mismatches, first {"step": 0, "k": 1}',
                                         '9 stored tensors do not digest to their recorder rows, first {"rank": 1, "k": 0}']


# --- commit_binding -----------------------------------------------------------------------------------------------------------

def _match_dir(tmp_path: Path, world: int = 2) -> tuple[Path, dict, dict[int, list[dict]]]:
    """A Match directory whose fold Match passed: its two records and each rank's recorder rows.  Returns it with the binding and the rows."""
    colls, recorder = _world(world)
    rank_match = RM.cross_rank_check(colls, recorder=recorder)
    assert rank_match["verdict"] == "PASS" and len(rank_match["peer_binding"]) == 2 * world * (world - 1)
    (tmp_path / "fold_match").mkdir()
    (tmp_path / "capture").mkdir()
    (tmp_path / "fold_match" / "tp2_fold_match.json").write_text(json.dumps({"pass": True}))
    _write(tmp_path, rank_match, recorder)
    return tmp_path, rank_match, recorder


def _write(md: Path, rank_match: dict, recorder: dict[int, list[dict]]) -> None:
    (md / "fold_match" / "tp2_rank_match.json").write_text(json.dumps(rank_match))
    for r, rows in recorder.items():
        (md / "capture" / f"coll.rank{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


def _oracle(world: int = 2) -> dict[int, dict]:
    return {r: {"verdict": "PASS"} for r in range(world)}


@pytest.mark.parametrize("world", [2, 3])
def test_the_match_directory_these_tests_change_binds(tmp_path, world):
    md, rank_match, _ = _match_dir(tmp_path, world)
    res = RM.commit_binding(md, world=world, match_oracle=_oracle(world))
    assert res["schema"] == "verity/tp2/commit-binding/v1"
    assert res["result"] is True and res["failures"] == [] and res["n_bad_bindings"] == 0, res
    assert res["n_bindings"] == len(rank_match["peer_binding"]) and res["n_recorder"] == {str(r): 2 for r in range(world)}


def test_a_fold_match_record_that_is_not_json_is_a_failure_not_an_absence(tmp_path):
    md, _, _ = _match_dir(tmp_path)
    (md / "fold_match" / "tp2_rank_match.json").write_text("{")
    res = RM.commit_binding(md, match_oracle=_oracle())
    assert res["result"] is False and res["why"].startswith("fold Match record unreadable: JSONDecodeError"), res


def test_a_rank_whose_recorder_rows_are_gone_fails_the_binding(tmp_path):
    md, _, _ = _match_dir(tmp_path)
    (md / "capture" / "coll.rank1.jsonl").unlink()
    res = RM.commit_binding(md, match_oracle=_oracle())
    assert res["result"] is False and res["n_recorder"] == {"0": 2}
    assert res["failures"][0] == "rank 1: recorder rows coll.rank1.jsonl missing from the Match capture (the Commit oracle's source)"


@pytest.mark.parametrize("field, words", [("own_partial_sha256", "rank0:s0_k0: fold own part digest other != rank 0's recorded partial at (step 0, k 0)"),
                                          ("received_sha256", "rank0:s0_k0: received digest differs from the recorder at (step 0, k 0)")])
def test_a_binding_whose_own_digests_are_not_the_recorders_fails(tmp_path, field, words):
    md, rank_match, recorder = _match_dir(tmp_path)
    rank_match["peer_binding"]["rank0:s0_k0"][field] = "other"
    _write(md, rank_match, recorder)
    res = RM.commit_binding(md, match_oracle=_oracle())
    assert res["result"] is False and res["failures"] == [words] and res["n_bad_bindings"] == 1, res


def test_beyond_two_ranks_a_collective_without_a_binding_to_one_peer_fails(tmp_path):
    md, rank_match, recorder = _match_dir(tmp_path, world=3)
    del rank_match["peer_binding"]["rank0:s0_k1_r2"]
    _write(md, rank_match, recorder)
    res = RM.commit_binding(md, world=3, match_oracle=_oracle(3))
    assert res["result"] is False
    assert res["failures"] == ["rank 0: 1 recorder collectives without a fold binding to peer rank 2 (first [(0, 1)])"], res


def test_a_rank_match_record_that_states_no_peer_binding_binds_no_recorder_collective(tmp_path):
    md, rank_match, recorder = _match_dir(tmp_path)
    del rank_match["peer_binding"]
    _write(md, rank_match, recorder)
    res = RM.commit_binding(md, match_oracle=_oracle())
    assert res["result"] is False and res["n_bindings"] == 0
    assert res["failures"] == [f"rank {r}: 2 recorder collectives without a fold binding (first [(0, 0), (0, 1)]) -- the fold did not state every peer leaf"
                               for r in (0, 1)], res


def test_a_rank_without_a_block_of_the_commit_oracle_was_not_run(tmp_path):
    md, _, _ = _match_dir(tmp_path)
    res = RM.commit_binding(md, match_oracle={0: {"verdict": "PASS"}})
    assert res["result"] is False and res["n_bindings"] == 4
    words = "rank 1: Commit match oracle not run (committed partials vs the recorder's stored tensors): the binding has no committed value to bind to"
    assert res["failures"] == [words], res
    assert RM.commit_binding(md, match_oracle={"0": {"verdict": "PASS"}, "1": {"verdict": "PASS"}})["result"] is True
