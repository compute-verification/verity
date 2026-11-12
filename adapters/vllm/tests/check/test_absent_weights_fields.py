"""What the weights of record and their check make of a field that a record, a Program's result, the served digests or a
row's manifest leaves out.

A field that is absent, null or empty reads alike: a record that states no field roots an empty list, one that states no
shard has no pin, one that states no engine assumption assumes the default, and so on down the file.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from tests.check import test_weights_of_record as TW
from tests.check.test_absent_commit_fields import ABSENT, put
from tests.check.test_absent_link_fields import left_out
from verity_vllm.check import weights_of_record as WR
from verity_vllm.program import model as analytic

checkpoint = TW.checkpoint      # the tiny checkpoint and its manifest, as a fixture of this module too

ROTARY = "model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"
SERVED_ROTARY = ROTARY[len(TW.PREFIX):]
NO_PIN = "checkpoint shards are not the pinned ones (W1): no shard pin recorded (record built with --no-verify-shards?)"
A, B, W = "a" * 128, "b" * 128, "f" * 128


def index(checkpoint: Any) -> WR.CheckpointIndex:
    d, man, _ = checkpoint
    return WR.CheckpointIndex(str(d), checkpoints_manifest=str(man))


def honest(checkpoint: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The tiny checkpoint's record and the digests of the model served from it."""
    prog, fields = TW._program()
    return WR.weights_of_record(prog, index(checkpoint), prefix=TW.PREFIX), TW.live_digests(TW._served(checkpoint[2], fields))


def rotary(checkpoint: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """A record of two fields, the rotary table pinned by the analytic bound and one norm weight, and what a model that serves
    the table config.json gives states of both."""
    d, _, t = checkpoint
    cfg = json.loads((d / "config.json").read_text())
    cfg.update({"model_type": "llama", "num_hidden_layers": 1, "intermediate_size": 12, "vocab_size": 20, "rope_theta": 10000.0,
                "max_position_embeddings": 64, "head_dim": 4, "rms_norm_eps": 1e-5, "torch_dtype": "bfloat16"})
    (d / "config.json").write_text(json.dumps(cfg))
    C, _ = analytic.config_of(cfg)
    shape = [int(C["MAX_POS"]), analytic.rotary_dim(C)]
    prog, _ = TW._program()
    prog["instances"]["params"] = [["weights", f"{{{ROTARY}: Array<{shape[0]},Array<{shape[1]},Value<16>>>, model.model.norm.weight: Array<8,Value<16>>}}"]]
    norm = WR.to_served(t["model.norm.weight"][1], t["model.norm.weight"][0], 16)[0]
    live = TW.live_digests({"model.norm.weight": norm})
    live["params"][SERVED_ROTARY] = {"digest": "x", "dtype": "bfloat16", "shape": shape}
    live["raw"] = {SERVED_ROTARY: analytic.cos_sin_table(C)}
    return WR.weights_of_record(prog, index(checkpoint), prefix=TW.PREFIX), live


def table_entry(record: dict[str, Any]) -> dict[str, Any]:
    return next(f for f in record["fields"] if f["field"] == ROTARY)


# ------------------------------------------------------------------------------------------------------------------ a record's fields

@left_out([])
def test_a_record_that_states_no_field_has_the_root_of_an_empty_list(stated, checkpoint):
    record, _ = honest(checkpoint)
    assert WR.reroot(record) == record["root_of_record"]
    assert WR.reroot(put(record, ("fields",), stated)) == WR.root_of_fields([]) != record["root_of_record"]


@left_out([])
def test_two_records_differ_by_every_field_where_one_of_them_states_none(stated, checkpoint):
    record, _ = honest(checkpoint)
    names = sorted(f["field"] for f in record["fields"])
    bare = put(copy.deepcopy(record), ("fields",), stated)
    left, right = WR.diff_records(bare, record), WR.diff_records(record, bare)
    assert (left["only_a"], left["only_b"], left["n_fields"], left["same_weights_of_record"]) == ([], names, [0, len(names)], False)
    assert (right["only_a"], right["only_b"], right["n_fields"], right["same_weights_of_record"]) == (names, [], [len(names), 0], False)
    neither = WR.diff_records(bare, copy.deepcopy(bare))
    assert neither["n_fields"] == [0, 0] and neither["same_weights_of_record"] is True


# ---------------------------------------------------------------------------------------------------------------- a Program's result

@left_out({})
def test_a_program_without_a_result_is_the_one_rank_of_a_world_of_one_under_its_instances_digest(stated, checkpoint):
    prog, fields = TW._program()
    prog["instances"]["digest"] = "e" * 128
    record = WR.weights_of_record(put(prog, ("result",), stated), index(checkpoint), prefix=TW.PREFIX)
    assert record["tp"] == {"world": 1, "rank": 0} and record["program_digest"] == "e" * 128
    assert record["complete"] is True and record["n_pinned"] == len(fields)


@left_out([])
def test_a_result_that_names_no_attention_path_takes_the_prefix_the_checkpoint_gives(stated, checkpoint):
    prog, fields = TW._program()
    record = WR.weights_of_record(put(prog, ("result", "attention_impls_observed"), stated), index(checkpoint))
    assert record["wrapper_prefix"] == TW.PREFIX and record["complete"] is True and record["n_pinned"] == len(fields)


@pytest.mark.parametrize("result", [None, {}], ids=["null", "empty"])
def test_a_component_whose_result_states_nothing_is_recorded_and_stamped_by_its_instances_digest(result, checkpoint, tmp_path):
    d, man, _ = checkpoint
    prog, fields = TW._program()
    root = tmp_path / "row"
    TW._write_program(root / "build_request", prog, "d" * 128, 9, 2)
    (root / "build_request" / "result.json").write_text(json.dumps(result))
    record = WR.build_record(str(root / "build_request"), str(d), checkpoints_manifest=str(man))
    assert record["program_digest"] == "d" * 128 and record["tp"] == {"world": 1, "rank": 0}
    stamp = WR.stamp_of_record_set(record, str(root))["of_record_set"]
    assert stamp["components"] == [{"dir": "build_request", "program_digest": "d" * 128, "LP": None, "T": None, "struct_equal": True,
                                    "n_fields": len(fields), "of_record": True, "in_manifest": False}]
    assert stamp["program_digests"] == ["d" * 128] and stamp["why"] is None and record["of_record"] is True


# ----------------------------------------------------------------------------------------------------------- the analytic table

@left_out({})
def test_an_analytic_field_that_states_no_table_has_no_validator(stated, checkpoint):
    record, live = rotary(checkpoint)
    put(table_entry(record), ("analytic",), stated)
    res = WR.check(record, live)
    assert res["result"] is False and res["compared"] == 2
    assert res["analytic"] == [{"field": ROTARY, "served_name": SERVED_ROTARY, "table": None, "expected_shape": None, "status": "error",
                                "why": "no validator for analytic table None"}]


@left_out({})
def test_an_analytic_table_that_states_no_constant_is_an_error_by_name(stated, checkpoint):
    record, live = rotary(checkpoint)
    assert WR.check(record, live)["analytic"][0]["status"] == "pass"
    put(table_entry(record), ("analytic", "C"), stated)
    res = WR.check(record, live)
    assert res["result"] is False and res["analytic"][0]["status"] == "error"
    assert res["analytic"][0]["why"].startswith("check_cos_sin raised KeyError: ")
    assert res["why"].startswith(f"1 engine table(s) outside the frozen analytic bound: {SERVED_ROTARY}: check_cos_sin raised KeyError")


@left_out({})
def test_a_served_model_whose_table_words_were_not_kept_does_not_pass_the_analytic_field(stated, checkpoint):
    record, live = rotary(checkpoint)
    res = WR.check(record, put(live, ("raw",), stated))
    assert res["result"] is False and res["analytic"][0]["status"] == "missing" and res["compared"] == 2


# ------------------------------------------------------------------------------------------------------------------------ the check

@left_out({})
def test_a_served_model_that_states_no_parameter_lacks_every_pinned_field(stated, checkpoint):
    record, live = honest(checkpoint)
    res = WR.check(record, put(live, ("params",), stated))
    names = [f["field"] for f in record["fields"]]
    assert res["result"] is False and res["compared"] == 0 and res["missing"] == names and res["placeholders_deleted"] == []
    assert res["why"] == f"{len(names)} Program weight field(s) not found among the served parameters: " + ", ".join(names[:8])


@pytest.mark.parametrize("path", [("checkpoint",), ("checkpoint", "shards")], ids=["checkpoint", "shards"])
@left_out({})
def test_a_record_that_states_no_shard_has_no_pin_and_is_refused_by_name(path, stated, checkpoint):
    record, live = honest(checkpoint)
    res = WR.check(put(record, path, stated), live)
    assert res["result"] is False and res["shards_ok"] is False and res["why"] == NO_PIN and not res["shards"]
    assert res["mismatched"] == [] and res["missing"] == [] and res["compared"] == len(record["fields"])


def test_a_record_that_states_no_checkpoint_states_no_entry_and_no_shard_pin(checkpoint):
    record, live = honest(checkpoint)
    del record["checkpoint"]
    res = WR.check(record, live, checkpoint_of_record={"repo": "test/tiny", "revision": "abc"})
    assert (res["checkpoint_entry"], res["shards"], res["all_shards_pinned"], res["checkpoint_entry_of_record"]) == (None, None, None, False)


@left_out([])
def test_a_stamp_that_states_no_program_digest_is_empty_and_is_refused_by_name(stated, checkpoint):
    record, live = honest(checkpoint)
    record["of_record_set"] = put({"program_digests": ["d" * 128], "why": None}, ("program_digests",), stated)
    res = WR.check(record, live, program_digests_of_record={"d" * 128})
    assert res["result"] is False and res["program_of_record"] is False and res["why"] == "record's of-record stamp is empty"
    assert not res["of_record_set"]
    assert WR.check(record, live)["result"] is True         # asked of no Program of record, the stamp is not read


def test_a_check_states_the_stamps_program_digests_as_the_record_states_them(checkpoint):
    record, live = honest(checkpoint)
    assert WR.check(record, live)["of_record_set"] is None
    for stamp, stated in (({}, None), ({"program_digests": None}, None), ({"program_digests": []}, []), ({"program_digests": [A]}, [A])):
        assert WR.check(dict(record, of_record_set=stamp), live)["of_record_set"] == stated


@left_out({})
def test_a_record_whose_checkpoint_states_no_entry_is_of_no_checkpoint_of_record(stated, checkpoint):
    record, live = honest(checkpoint)
    put(record, ("checkpoint", "entry"), stated)
    res = WR.check(record, live, checkpoint_of_record={"repo": "test/tiny", "revision": "abc"})
    assert res["result"] is False and res["checkpoint_entry_of_record"] is False
    assert res["why"] == "record's checkpoint entry None != the workload's model test/tiny"
    res = WR.check(record, live, checkpoint_of_record={"revision": "abc"})
    assert res["result"] is False and res["why"] == "record's checkpoint revision None != the workload's pin abc"
    assert WR.check(record, live)["result"] is True         # asked of no checkpoint of record, the entry is not read


@left_out({})
def test_a_record_that_states_no_engine_assumption_assumes_the_default_kv_cache(stated, checkpoint):
    record, live = honest(checkpoint)
    put(record, ("engine_assumptions",), stated)
    assert WR.check(record, live, kv_cache_dtype="auto")["result"] is True
    res = WR.check(record, live, kv_cache_dtype="fp8_e4m3")
    assert res["result"] is False
    assert res["why"] == "record assumes kv_cache_dtype=auto; the engine runs kv_cache_dtype=fp8_e4m3 (kv scales not of record)"


@pytest.mark.parametrize("key", ["unpinned", "unresolved"])
@left_out([])
def test_a_record_that_lists_no_unpinned_or_unresolved_field_has_none(key, stated, checkpoint):
    record, live = honest(checkpoint)
    res = WR.check(put(record, (key,), stated), live)
    assert res["result"] is True and res[key] == [] and res["unpinned_note"].startswith("every declared weight field is pinned")


def test_a_check_states_the_records_unpinned_and_unresolved_fields_as_lists_of_its_own(checkpoint):
    record, live = honest(checkpoint)
    record["unpinned"], record["unresolved"] = ["x"], [{"field": "y", "why": "z"}]
    res = WR.check(record, live)
    assert res["unpinned"] == ["x"] and res["unpinned"] is not record["unpinned"]
    assert res["unresolved"] == [{"field": "y", "why": "z"}] and res["unresolved"] is not record["unresolved"]
    assert res["unpinned_note"].startswith("engine constants left unpinned by name")


# ------------------------------------------------------------------------------------------------------------------ a row's manifest

@pytest.mark.parametrize("key, stated, named", [(key, stated, named) for key, empty, named in (("component_digests", [], B), ("components", {}, A))
                                                for stated in (ABSENT, None, empty)])
def test_a_manifest_names_the_components_its_list_and_its_table_state_and_no_other(key, stated, named, checkpoint, tmp_path):
    d, man, _ = checkpoint
    prog, _ = TW._program()
    root = tmp_path / "row"
    TW._write_program(root / "build_request_LP4_T2", prog, A, 4, 2)
    TW._write_program(root / "build_request_LP9_T1", prog, B, 9, 1)
    manifest = {"program_digest": W, "component_digests": [A], "components": {"r0": {"program_digest": B}, "r1": None}}
    (root / "manifest.json").write_text(json.dumps(put(manifest, (key,), stated)))
    record = WR.stamp_of_record_set(WR.build_record(str(root / "build_request_LP4_T2"), str(d), checkpoints_manifest=str(man)), str(root))
    stamp = record["of_record_set"]
    assert stamp["manifest_digests"] == sorted({W, named}) and stamp["program_digests"] == sorted({A, B, W})
    assert stamp["why"] is None and record["of_record"] is True
    assert [c["in_manifest"] for c in stamp["components"]] == [named == A, named == B]
