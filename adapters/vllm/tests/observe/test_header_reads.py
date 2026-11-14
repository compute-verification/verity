"""What the fold's patterns read when a document they read is missing, null, empty or zero, or full: a run header's sampling,
backends, GPU and tensor-parallel facts, a step's KV groups, a sampler's scratch, the capture's external-buffers note and a Build's
`result.json`.  Each value of `EMPTY` in a document's place reads as the document left out.  A value of `NO_MAPPING` in the place
of the sampling block, the NCCL environment or the engine flags is refused by the read itself.

The readers: `verity_vllm.observe.fold.patterns.sampling`, `verity_vllm.observe.fold.profiles.gen_ov_sampling_patterns`,
`verity_vllm.observe.fold.patterns.fp8`, `verity_vllm.observe.fold.patterns.int8`, `verity_vllm.observe.fold.patterns.target`,
`verity_vllm.observe.fold.profiles.gen_shared_moe_patterns`, `verity_vllm.observe.fold.patterns.collectives`,
`verity_vllm.observe.fold.patterns.kv_cache`, `verity_vllm.observe.fold.resolver`, `verity_vllm.observe.fold.supplied` and
`verity_vllm.observe.fold.profiles.vllm_d9105ea80_sm89_eager`.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from verity.definitions.registry import tp_allreduce

from tests.observe.test_patterns_synthetic import V, context, decode_step
from verity_vllm.observe.events import Note, RunHeader
from verity_vllm.observe.fold.patterns import GeometryError
from verity_vllm.observe.fold.patterns.collectives import (
    allreduce_settings,
    capture_device,
)
from verity_vllm.observe.fold.patterns.fp8 import header_fp8_kernel
from verity_vllm.observe.fold.patterns.int8 import header_k_scales
from verity_vllm.observe.fold.patterns.kv_cache import kv_slots
from verity_vllm.observe.fold.patterns.sampling import (
    TokenSelectGumbelTopP,
    TokenSelectGumbelTwoStage,
    greedy_sampling,
    header_sampling,
    request_sampling,
)
from verity_vllm.observe.fold.patterns.target import header_gemm_arch_family
from verity_vllm.observe.fold.profiles import gen_ov_sampling_patterns as OV
from verity_vllm.observe.fold.profiles import vllm_d9105ea80_sm89_eager as PROF
from verity_vllm.observe.fold.profiles.gen_shared_moe_patterns import (
    header_gate_ll_bf16,
)
from verity_vllm.observe.fold.resolver import StepContext, external_buffers
from verity_vllm.observe.fold.supplied import program_registered_values
from verity_vllm.observe.fold.tree import step_events

#: what reads as no document: a null, and each empty or zero value
EMPTY: tuple = (None, {}, [], 0, "", False)

#: what is not empty and is no mapping: pairs, which `dict` would read as one, a string and a number
NO_MAPPING: tuple = ([("greedy", True)], (("greedy", True),), "no", 7)


def _absent(key: str) -> list[dict]:
    """What a document may hold where it holds no sub-document under `key`: no such key, or a value of `EMPTY`."""
    return [{}, *({key: empty} for empty in EMPTY)]


def _ctx(profile=None, model=None, workload=None, notes=None, scratch=None) -> StepContext:
    header = RunHeader("verity-capture/log/v1", "2026-09-16T00:00:00Z", "p", profile, model, workload, notes)
    return StepContext(header, None, None, [], {}, {}, None, scratch={} if scratch is None else scratch)


def test_the_runs_sampling_is_the_workloads_when_it_states_any_else_the_profiles():
    mine, pinned = {"greedy": False, "temperature": 0.7}, {"greedy": True, "temperature": 0.0}
    assert header_sampling(_ctx()) == {} and header_sampling(_ctx(profile={}, workload={})) == {}
    for workload in (None, *_absent("sampling")):
        assert header_sampling(_ctx(profile={"sampling": pinned}, workload=workload)) == pinned
        for profile in (None, *_absent("sampling")):
            assert header_sampling(_ctx(profile=profile, workload=workload)) == {}
    got = header_sampling(_ctx(profile={"sampling": pinned}, workload={"sampling": mine}))
    assert got == mine and got is not mine
    assert OV.header_sampling(_ctx(profile={"sampling": pinned}, workload={"sampling": {}})) == pinned


def test_greedy_sampling_names_what_is_not_greedy():
    def why(sampling):
        return greedy_sampling(_ctx(workload={"sampling": sampling}))
    assert greedy_sampling(_ctx()) == "the run header does not record greedy sampling" == why({"temperature": 0.0})
    assert why({"greedy": True, "temperature": 0.0}) is None and why({"greedy": True, "temperature": 0, "top_k": -1, "top_p": 1}) is None
    assert why({"greedy": True, "temperature": 0.5}) == "sampling temperature is 0.5, not 0"
    assert why({"greedy": True, "temperature": 0.0, "top_k": 5}) == "top_k/top_p sampling (5/None) is not the registered TokenSelect"
    assert greedy_sampling(_ctx(profile={"sampling": {"greedy": True, "temperature": 0.0}}, workload={})) is None


@pytest.mark.parametrize("block", NO_MAPPING)
@pytest.mark.parametrize("held_by", ["workload", "profile"])
def test_greedy_sampling_asks_the_sampling_block_itself_and_not_a_copy_of_it(held_by, block):
    """A sampling block that is no mapping has no `get`, whatever `dict` would make of it: the workload's, and the profile's."""
    ctx = _ctx(workload={"sampling": block}) if held_by == "workload" else _ctx(profile={"sampling": block}, workload={})
    with pytest.raises(AttributeError, match="has no attribute 'get'"):
        greedy_sampling(ctx)


def test_request_sampling_reads_each_requests_own_values_then_its_record_then_the_workloads():
    assert request_sampling(_ctx()) == {} == request_sampling(_ctx(workload={"sampling": {"temperature": 0.5}}))
    requests = [{"request_id": "a", "seed": 7, "top_p": 0.5}, {"request_id": "b"}]
    plain = request_sampling(_ctx(workload={"requests": requests}))
    assert plain == {"a": {"seed": 7, "temperature": 0.0, "top_p": 0.5, "top_k": 0}, "b": {"seed": None, "temperature": 0.0, "top_p": 1.0, "top_k": 0}}
    for empty in EMPTY:
        assert request_sampling(_ctx(workload={"requests": requests, "sampling": {"per_request": empty}, "children": empty})) == plain, empty
        assert request_sampling(_ctx(workload={"requests": empty, "sampling": {"temperature": 0.5}})) == {}, empty
    sampling = {"temperature": 0.7, "top_p": 0.9, "top_k": 4, "per_request": [{"request_id": "b", "temperature": 0.2, "top_k": 0}]}
    children = [{"request_id": "0_a", "parent": "a", "seed": 7}, {"request_id": "1_a", "parent": "a", "seed": 8},
                {"request_id": "0_z", "parent": "z", "seed": None}]
    got = request_sampling(_ctx(workload={"requests": requests, "sampling": sampling, "children": children}))
    assert got["a"] == {"seed": 7, "temperature": 0.7, "top_p": 0.5, "top_k": 4}
    assert got["b"] == {"seed": None, "temperature": 0.2, "top_p": 0.9, "top_k": 0}
    assert got["0_a"] == got["a"] and got["1_a"] == {**got["a"], "seed": 8} and got["0_z"] == {"seed": None}


def test_the_seeded_samplers_request_facts():
    assert OV.request_seeds(_ctx()) == {} == OV.request_seeds(_ctx(workload={"requests": [], "children": []}))
    workload = {"requests": [{"request_id": "a", "seed": "7"}, {"request_id": "b"}], "children": [{"request_id": "0_a", "seed": 8}, {"request_id": "1_a"}]}
    assert OV.request_seeds(_ctx(workload=workload)) == {"a": 7, "b": None, "0_a": 8, "1_a": None}
    assert OV.request_temperatures(_ctx()) == (0.0, {})
    for empty in EMPTY:
        assert OV.request_seeds(_ctx(workload={"requests": [{"request_id": "b"}], "children": empty})) == {"b": None}, empty
        assert OV.request_temperatures(_ctx(workload={"sampling": {"temperature": 1, "per_request": empty}})) == (1.0, {}), empty
        assert OV.sampling_filters_off({"top_k": -1, "per_request": empty}) is None, empty
    sampling = {"temperature": 1, "per_request": [{"request_id": "b", "temperature": 0}]}
    assert OV.request_temperatures(_ctx(workload={"sampling": sampling})) == (1.0, {"b": 0.0})
    assert OV.sampling_filters_off({}) is None and OV.sampling_filters_off({"top_k": -1, "top_p": 1.0}) is None
    assert OV.sampling_filters_off({"top_k": 3}) == "top_k/top_p (3/None)"
    assert OV.sampling_filters_off({"per_request": [{"request_id": "b"}, {"request_id": "c", "top_p": 0.5}]}) == "per-request top_k/top_p (None/0.5) for c"


@pytest.mark.parametrize("read,key,value,none", [
    (header_fp8_kernel, "fp8_block_gemm_kernel", "CutlassFp8BlockScaledMMKernel", None),
    (header_k_scales, "int8_k_scales", {"model.layers.0": [1.0]}, None),
    (header_gate_ll_bf16, "moe_gate_ll_bf16_gemm", True, False),
])
def test_a_backend_fact_is_read_only_where_the_header_states_it(read, key, value, none):
    for profile in (None, *_absent("backends"), {"backends": {"other": value}}, {"backends": {key: 3}}, {"backends": "no"}, "no"):
        assert read(_ctx(profile=profile)) is none, profile
    assert read(StepContext(None, None, None, [], {}, {}, None)) is none
    assert read(_ctx(profile={"backends": {key: value}})) == value


def test_the_tuned_gemm_family_is_the_headers_or_unsaid():
    for profile in (None, *_absent("gpu"), {"gpu": {"name": "x"}}, {"gpu": "no"}):
        assert header_gemm_arch_family(_ctx(profile=profile)) is ..., profile
    assert header_gemm_arch_family(StepContext(None, None, None, [], {}, {}, None)) is ...
    assert header_gemm_arch_family(_ctx(profile={"gpu": {"tuned_gemm_arch_family": "sm_90"}})) == "sm_90"
    assert header_gemm_arch_family(_ctx(profile={"gpu": {"tuned_gemm_arch_family": None}})) is None


def test_the_all_reduce_settings_and_device_a_capture_recorded():
    def header(profile=None, notes=None):
        return _ctx(profile=profile, notes=notes).header
    assert allreduce_settings(None) is None and allreduce_settings(header()) is None and capture_device(None) is None
    for tp2 in (*_absent("facts"), *({"facts": d} for d in _absent("parallel_config")), {"facts": {"parallel_config": {"disable_custom_all_reduce": 1}}}):
        assert allreduce_settings(header(notes={"tp2": tp2})) is None, tp2
    assert allreduce_settings(header(notes={"tp2": None})) is None and allreduce_settings(header(notes={})) is None
    recorded = {"disable_custom_all_reduce": True, "anything": 1}
    assert allreduce_settings(header(notes={"allreduce_settings": recorded, "tp2": None})) == recorded
    facts = {"parallel_config": {"disable_custom_all_reduce": False}}
    more: dict
    for more in ({}, {"nccl_env": None}, {"nccl_env": {}}):
        for profile in (None, *_absent("engine_flags")):
            assert allreduce_settings(header(profile, {"tp2": {"facts": {**facts, **more}}})) == tp_allreduce.settings_of(False, {})
    env, flags = {"NCCL_ALGO": "Ring", "A": "rank"}, {"A": "manifest"}
    got = allreduce_settings(header({"engine_flags": flags}, {"allreduce_settings": {}, "tp2": {"facts": {**facts, "nccl_env": env}}}))
    assert got == tp_allreduce.settings_of(False, {"NCCL_ALGO": "Ring", "A": "manifest"})
    for profile in (None, *_absent("gpu"), {"gpu": {"compute_capability": [9, 0]}}, {"gpu": {"sms": 132}}):
        assert capture_device(header(profile)) is None, profile
    assert capture_device(header({"gpu": {"compute_capability": [9, 0], "multi_processor_count": 132}})) == tp_allreduce.device_of([9, 0], 132)
    assert capture_device(header({"gpu": {"cc": [9, 0], "sms": 132}})) == tp_allreduce.device_of([9, 0], 132)


@pytest.mark.parametrize("part", NO_MAPPING)
@pytest.mark.parametrize("key", ["nccl_env", "engine_flags"])
def test_the_all_reduce_settings_unpack_the_nccl_environment_and_the_engine_flags_as_mappings(key, part):
    """Pairs are not read as the mapping they spell: each part is unpacked into the settings' environment, which takes a mapping."""
    facts = {"parallel_config": {"disable_custom_all_reduce": False}}
    if key == "nccl_env":
        header = _ctx(notes={"tp2": {"facts": {**facts, key: part}}}).header
    else:
        header = _ctx(profile={key: part}, notes={"tp2": {"facts": facts}}).header
    with pytest.raises(TypeError, match="is not a mapping"):
        allreduce_settings(header)


def test_a_kv_group_without_a_block_table_for_the_request_is_a_geometry_error():
    _events, start = decode_step()
    layer, tables = "model.layers.0.self_attn.attn", dict(start.block_tables)
    request = next(iter(tables))
    held: dict
    for held in (*_absent("block_tables"), {"block_tables": {"other": tables[request]}}):
        groups = [{"layers": [layer], "block_size": start.block_size, **held}, {"layers": ["other"], "block_size": start.block_size}]
        with pytest.raises(GeometryError, match="no block table for request"):
            kv_slots(context(dataclasses.replace(start, kv_groups=groups)), request, 1, layer)
    groups = [{"layers": [layer], "block_size": start.block_size, "block_tables": tables}, {"layers": ["other"]}]
    assert kv_slots(context(dataclasses.replace(start, kv_groups=groups)), request, 1, layer) == kv_slots(context(start), request, 1)
    with pytest.raises(GeometryError, match="no block table for request"):
        kv_slots(context(dataclasses.replace(start, block_tables={})), request, 1)


def test_a_sampler_without_its_first_stage_claims_no_argmax_and_no_gather():
    """The argmax and the gather of a step whose blockwise kernel no sampler has seen: each two-stage sampler leaves them to the
    patterns after it, with no state of its own, with the state it makes, with one that holds only another view's records, and
    with a value of `EMPTY` in the state's place or in the place of each of its tables."""
    events, start = decode_step(sampler="gumbel")
    roots = {root.op: root for root in step_events(events).roots}
    stochastic = {"greedy": False, "temperature": 0.7}
    samplers = [(TokenSelectGumbelTwoStage(vocab=V, block_size=PROF.SAMPLER_BLOCK_SIZE), None),
                (TokenSelectGumbelTopP(vocab=V), stochastic), (OV.TokenSampleGumbel(vocab=V), stochastic)]
    for sampler, sampling in samplers:
        for state in ("none", "made", "another view's"):
            ctx = context(start, None, **({} if sampling is None else {"sampling": sampling}))
            if state != "none":
                tables = [table for table in sampler._state(ctx).values() if isinstance(table, dict)]
                assert len(tables) >= 2
                if state != "made":
                    for table in tables:
                        table[-1] = {}
            assert sampler.match(roots["aten.argmax.default"], ctx) is None, (sampler.name, state)
            assert sampler.match(roots["aten.gather.default"], ctx) is None, (sampler.name, state)
        for empty in EMPTY:
            for whole in (True, False):
                ctx = context(start, None, **({} if sampling is None else {"sampling": sampling}))
                made = sampler._state(ctx)
                (key,) = ctx.scratch
                if whole:
                    ctx.scratch[key] = empty
                else:
                    made.update({name: empty for name, table in made.items() if isinstance(table, dict)})
                assert sampler.match(roots["aten.argmax.default"], ctx) is None, (sampler.name, empty, whole)
                assert sampler.match(roots["aten.gather.default"], ctx) is None, (sampler.name, empty, whole)


def test_the_external_buffers_note_gives_the_views_it_records_with_an_offset_and_strides():
    def device(alloc):
        return f"cuda:{alloc}"
    for data in (None, *_absent("buffers")):
        assert external_buffers(Note("external buffers", data), device) == {}
    buffers = {"kv_cache[a]": {"alloc": 3, "offset": 8, "shape": [2, 4], "strides": [4, 1], "dtype": "bfloat16"},
               "plain": {"alloc": 4, "shape": [2], "dtype": "int64"}}
    got = external_buffers(Note("external buffers", {"buffers": buffers}), device)
    assert list(got) == ["kv_cache[a]"]
    view = got["kv_cache[a]"]
    assert (view.alloc, view.offset, view.shape, view.strides, view.dtype, view.device) == (3, 8, (2, 4), (4, 1), "bfloat16", "cuda:3")


def test_check_header_compares_the_configuration_a_header_carries():
    for model in (None, *_absent("config")):
        assert PROF.check_header(_ctx(model=model).header) == []
    assert PROF.check_header(_ctx(model={"config": {"hidden_size": PROF.C["H"], "unknown": 1}}).header) == []
    assert PROF.check_header(_ctx(model={"config": {"hidden_size": PROF.C["H"] + 1}}).header) == [f"hidden_size={PROF.C['H'] + 1!r} != C[H]={PROF.C['H']!r}"]


def test_a_builds_programs_register_their_values_where_result_json_says_so(tmp_path):
    def build(name, doc):
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps(doc))
        return str(tmp_path / name)
    constants = [{}, *_absent("program"), *({"program": d} for d in _absent("supplied")), {"program": {"supplied": {"registration": None}}},
                 {"program": {"supplied": {"registration": {}}}}]
    held = [build(f"const{i}", doc) for i, doc in enumerate(constants)]
    registered = build("registered", {"program": {"supplied": {"registration": {"values": 3}}}})
    assert not program_registered_values([]) and not program_registered_values(held)
    assert all(not program_registered_values([d]) for d in held) and program_registered_values([registered, registered])
    with pytest.raises(SystemExit, match="some Programs register their values and some hold them as constants"):
        program_registered_values([held[0], registered])
    (tmp_path / "empty").mkdir()
    unreadable = [str(tmp_path / "empty"), build("list", []), build("text", "x")]
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "result.json").write_text("{")
    for d in (*unreadable, str(tmp_path / "broken")):
        with pytest.raises(SystemExit, match=r"no readable result\.json"):
            program_registered_values([registered, d])
