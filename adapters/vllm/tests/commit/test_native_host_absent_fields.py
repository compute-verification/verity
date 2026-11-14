"""What `verity_vllm.commit.committer.native_host` makes of a record that leaves a field out, gives it null or gives it empty: an
expected manifest that names no TP sites, no counts for them, no MoE layers beside its plain ones and nothing ruled out; a step
bound without the model's geometry; a declared request without its prompt's ids; a schedule with no largest step; and a step
that staged nothing.  Each of the three means the same, and each case holds the value the code gives for it.  Offline (no CUDA)."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.commit import test_bound_phases as BP
from tests.commit import test_x03_identities as XI
from verity_vllm.commit.committer import native_host as NH

ABSENT = object()
#: a field left out, null, and each empty form the code reads as nothing
LEFT_OUT: tuple[Any, ...] = (ABSENT, None)


def _with(record: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """`record` with each field set, or left out where its value is `ABSENT`."""
    out = {k: v for k, v in record.items() if k not in fields}
    out.update({k: v for k, v in fields.items() if v is not ABSENT})
    return out


# --- an expected manifest's TP sites ------------------------------------------------------------------------------------------------------

PARTIAL = "model.layers.0.mlp.down_proj/part_rank0"
SITE = "model.layers.0.mlp.down_proj"


@pytest.mark.parametrize("sites", [*LEFT_OUT, []])
def test_a_manifest_that_names_no_tp_site_passes_a_step_without_partials_and_fails_one_with(sites: Any) -> None:
    man = _with({}, tp_sites=sites)
    assert NH.tp_partials_bad(man, Counter({"model.layers.0.mlp.down_proj": 1})) == {}
    assert NH.tp_partials_bad(man, Counter({PARTIAL: 2, "x/shard_rank1": 1})) == {"<undeclared tp source>": 3}


@pytest.mark.parametrize("counts", [*LEFT_OUT, {}])
def test_a_site_whose_manifest_gives_no_count_binds_one_partial_a_step(counts: Any) -> None:
    man = _with({"tp_sites": [SITE]}, tp_site_counts=counts)
    assert NH.tp_partials_bad(man, Counter({PARTIAL: 1})) == {}
    assert NH.tp_partials_bad(man, Counter({PARTIAL: 2})) == {SITE: 2}
    assert NH.tp_partials_bad(man, Counter()) == {SITE: 0}


def test_a_site_with_a_count_binds_that_many_and_another_rank_s_partial_is_named() -> None:
    man = {"tp_sites": [SITE], "tp_site_counts": {SITE: 2}, "tp_rank": 0}
    assert NH.tp_partials_bad(man, Counter({PARTIAL: 2})) == {}
    assert NH.tp_partials_bad(man, Counter({PARTIAL: 1})) == {SITE: 1}
    assert NH.tp_partials_bad(man, Counter({PARTIAL: 1, f"{SITE}/part_rank1": 1})) == {"<foreign rank tag>": 1}


# --- a step bound's model, and a declared request's prompt -------------------------------------------------------------------------------

ASZ = {"per_token_bytes": 100, "per_seq_bytes": 10, "per_step_bytes": 1}
ONE = [{"prompt_token_ids": [1, 2, 3], "arrive_step": 0, "max_tokens": 2}]
#: what a step bound without a model is read as: one head of 64, one layer, no hidden-value rows
PLAIN = {"heads": 1, "kv_heads": 1, "head_dim": 64, "layers": 1, "hidden_m1": False, "attn_backend": "fa2"}


@pytest.mark.parametrize("model", [*LEFT_OUT, {}])
def test_a_step_bound_without_its_model_is_the_plain_model_with_no_hidden_rows(model: Any) -> None:
    got = NH.schedule_stream_bound(_with(ASZ, model=model), ONE, {"max_tokens": 4})
    assert got == NH.schedule_stream_bound({**ASZ, "model": PLAIN}, ONE, {"max_tokens": 4})
    assert (got["run_bytes"], got["hidden_bytes"], got["steps"]) == (422, 0, 2)
    assert got["largest_step"] == {"step": 0, "tokens": 3, "seqs": 1, "prefill_seqs": 1, "bytes": 311, "hidden_bytes": 0}


def test_a_model_with_hidden_rows_counts_them() -> None:
    got = NH.schedule_stream_bound({**ASZ, "model": {**PLAIN, "hidden_m1": True}}, ONE, {"max_tokens": 4})
    assert got["hidden_bytes"] > 0 and got["run_bytes"] == 422 + got["hidden_bytes"]


@pytest.mark.parametrize("ids", [*LEFT_OUT, []])
def test_a_request_without_its_prompt_s_ids_is_as_long_as_it_says_or_one_token(ids: Any) -> None:
    said = NH.schedule_stream_bound(ASZ, [_with({"prompt_len": 7}, prompt_token_ids=ids)], None)
    assert said == NH.schedule_stream_bound(ASZ, [{"prompt_token_ids": list(range(7))}], None)
    assert said["largest_step"]["tokens"] == 7
    unsaid = NH.schedule_stream_bound(ASZ, [_with({}, prompt_token_ids=ids)], None)
    assert unsaid == NH.schedule_stream_bound(ASZ, [{"prompt_token_ids": [0]}], None)
    assert unsaid["largest_step"]["tokens"] == 1


def test_a_request_s_ids_win_over_the_length_it_says() -> None:
    got = NH.schedule_stream_bound(ASZ, [{"prompt_token_ids": [1, 2], "prompt_len": 7}], None)
    assert got["largest_step"]["tokens"] == 2


# --- a schedule's largest step ----------------------------------------------------------------------------------------------------------

def test_a_schedule_of_no_request_has_no_largest_step() -> None:
    assert NH.schedule_stream_bound(ASZ, [], None)["largest_step"] is None


@pytest.mark.parametrize("largest", [*LEFT_OUT, {}])
def test_the_host_bound_of_a_schedule_with_no_largest_step_names_none(largest: Any) -> None:
    ssb = _with(BP._s8(), largest_step=largest)
    got = NH.bounded_host_bound(ssb, engine_rss_bytes=2536 * BP.MiB, **BP.COMMON)
    whole = NH.bounded_host_bound(BP._s8(), engine_rss_bytes=2536 * BP.MiB, **BP.COMMON)
    where = got["dominant_contributor"]["largest_materialization"]
    assert {k: where[k] for k in ("step", "tokens", "step_bytes", "hidden_bytes")} == {"step": None, "tokens": None, "step_bytes": None, "hidden_bytes": None}
    assert "hidden stream export step None (None tok) 0 MiB of a 0 MiB step" in got["dominant_contributor"]["text"]
    for k in got:                                                           # nothing but what names the step differs
        if k != "dominant_contributor":
            assert got[k] == whole[k], k
    assert {k for k in whole["dominant_contributor"] if whole["dominant_contributor"][k] != got["dominant_contributor"][k]} == {"largest_materialization", "text"}


def test_the_host_bound_names_the_largest_step_a_schedule_has() -> None:
    ssb = BP._s8()
    where = NH.bounded_host_bound(ssb, engine_rss_bytes=2536 * BP.MiB, **BP.COMMON)["dominant_contributor"]["largest_materialization"]
    big = ssb["largest_step"]
    assert (where["step"], where["tokens"], where["step_bytes"], where["hidden_bytes"]) == (big["step"], big["tokens"], big["bytes"], big["hidden_bytes"])


# --- the identity check's manifest -------------------------------------------------------------------------------------------------------

def _moe_step(planes: int) -> list[str]:
    return [*XI._good_step(), *[f"moe/L0/p{i}" for i in range(planes)]]


def _checked(man: dict[str, Any], names: list[str]) -> dict[str, Any]:
    com = XI._com()
    com._layouts[0] = XI._layout(names)
    return com.identity_check(man)


@pytest.mark.parametrize("field", ["router_softmax_layers", "moe_hash_layers"])
@pytest.mark.parametrize("layers", [*LEFT_OUT, []])
def test_a_manifest_that_lists_no_layer_with_one_more_moe_tensor_wants_the_plain_count(field: str, layers: Any) -> None:
    man = _with({**XI._com().expected_manifest(), "moe_layers": [0], "moe_tensors_per_layer": 2}, **{field: layers})
    assert _checked(man, _moe_step(2))["ok"] is True
    bad = _checked(man, _moe_step(3))
    assert bad["ok"] is False and bad["violations"][0]["moe_layers_bad"] == {0: 3}


@pytest.mark.parametrize("field", ["router_softmax_layers", "moe_hash_layers"])
def test_a_layer_the_manifest_lists_for_one_more_moe_tensor_wants_it(field: str) -> None:
    man = {**XI._com().expected_manifest(), "moe_layers": [0], "moe_tensors_per_layer": 2, field: [0]}
    assert _checked(man, _moe_step(3))["ok"] is True
    bad = _checked(man, _moe_step(2))
    assert bad["ok"] is False and bad["violations"][0]["moe_layers_bad"] == {0: 2}


@pytest.mark.parametrize("gone", [*LEFT_OUT, []])
def test_the_check_s_record_counts_none_of_what_its_manifest_leaves_out(gone: Any) -> None:
    man = _with(XI._com().expected_manifest(), moe_layers=gone, tp_sites=gone, tp_sites_ruled_out=gone)
    got = _checked(man, XI._good_step())
    assert got["ok"] is True
    assert (got["manifest"]["moe_layers"], got["manifest"]["tp_sites"], got["manifest"]["tp_sites_ruled_out"]) == (0, 0, [])


def test_the_check_s_record_counts_what_its_manifest_lists() -> None:
    man = {**XI._com().expected_manifest(), "moe_layers": [0, 1], "moe_tensors_per_layer": 0, "tp_sites": [SITE], "tp_sites_ruled_out": ["a/b"]}
    got = _checked(man, [*XI._good_step(), PARTIAL])["manifest"]
    assert (got["moe_layers"], got["tp_sites"], got["tp_sites_ruled_out"]) == (2, 1, ["a/b"])


# --- a step's trace ---------------------------------------------------------------------------------------------------------------------

def _step(k: int) -> SimpleNamespace:
    return SimpleNamespace(step=k, root=bytes([k + 1]) * 64, n_tensors=0, n_bytes=0, n_leaves=0, layout_digest=bytes([k + 2]) * 64)


@pytest.mark.parametrize("layout", [ABSENT, None, [], "plan"])
def test_the_trace_of_a_step_that_staged_nothing_names_no_hidden_row(layout: Any) -> None:
    com = XI._com()
    if layout is not ABSENT:
        com._layouts[0] = XI._plan([]) if layout == "plan" else layout
    assert com._step_trace([_step(0)]) == [{"step": 0, "root": "0101010101010101", "tensors": 0, "bytes": 0, "leaves": 0, "layout": "0202020202020202", "fa2": None}]


@pytest.mark.parametrize("build", [XI._layout, XI._plan])
def test_the_trace_of_a_step_names_its_first_hidden_row(build: Any) -> None:
    com = XI._com()
    com._layouts[1] = build(XI._good_step())
    assert com._step_trace([_step(1)])[0]["fa2"] == "fa2.m1/L0/g1x64x1"
