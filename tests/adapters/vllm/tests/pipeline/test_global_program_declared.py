"""What the workload Program declares of its row before a component's Program is read
(`verity_vllm.pipeline.global_program`): its requests in the manifest's order, its lag and the engine it models.  The requests
take the ids `r<k>` by position, a request listed out of its index or with no length is refused, and a row that serves with
the prefix cache states each request's cached rows and the requests that made them.  The lag is the declared
target's (1 under async scheduling, which is the pinned resolver's value when the target says nothing; 0 synchronous), each
component is held to it and to the pinned vLLM, and the first that disagrees is named.  The engine is the components' agreed
block under the conditions the Program models: eager for an eager label, the agreed compilation facts for a compiled one, and a
refusal by name where the components disagree, cache prefixes or state the other execution.  The stop rule is the set of the
ids given, the checkpoint's and the manifest's, each under its source; a manifest that switches EOS off or a row with no id says
so, a manifest's set that is not the checkpoint's is refused, and a checkpoint file that cannot be read is named with 120
characters of its error."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from verity_vllm.check.match.declaration import MODELLED_ENGINE
from verity_vllm.config import VLLM
from verity_vllm.pipeline import coverage_workloads as gw
from verity_vllm.pipeline import global_program as GP
from verity_vllm.program import prefix_plan as PP
from verity_vllm.records.workload import Workload, WorkloadSampling

LAG_SCHEMA = "verity/workload-program/v1/lag"
STOP_SCHEMA = "verity/workload-program/v1/stop-rule"
STOPS_AT_EOS = ("a request may stop before max_tokens only when its last served token is one of eos_token_ids (finish_reason stop/None); "
                "stop strings, aborts and drops are not modelled and FAIL G6")
GPU = {"compute_capability": [8, 9], "num_sms": 142}


class _CharTok:
    path, sha256 = Path("snap/tokenizer.json"), "5a" * 32

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(map(chr, ids))


DOC = gw.workload_doc(case="B0", model="org/m", tok=_CharTok(), profile="B8-mixed", ctx="C256", seed=3, tokenizer_revision="ab" * 20, sampling_seed=7)


def _requests(change=None):
    doc = copy.deepcopy(DOC)
    if change:
        change(doc["requests"])
    return GP.manifest_requests(Workload.of(doc, "w.json"))


def _component(tmp_path, name, declared=None, **result):
    d = tmp_path / name
    d.mkdir()
    (d / "result.json").write_text(json.dumps({**result, **({"target": {"declared": declared}} if declared is not None else {})}))
    return str(d)


def _manifest(**target):
    return {"engine": {"target": target}} if target else {}


def test_the_requests_are_the_record_s_in_its_order_under_the_ids_r_k_with_their_lengths_and_arrivals():
    rows = _requests()
    assert [list(r.values()) for r in rows] == [[f"r{i}", q["request_id"], i, q["prompt_len"], q["max_tokens"], q["arrive_step"]]
                                                 for i, q in enumerate(DOC["requests"])]
    assert len(rows) == 8 and [r["workload_request_id"] for r in rows] == [q["request_id"] for q in DOC["requests"]]
    assert [GP.shape_of(r) for r in rows] == [PP.shape(q["prompt_len"], q["max_tokens"] - 1, 0) for q in DOC["requests"]]
    assert any(r["arrive_step"] for r in rows) and GP.request_id_of(7) == GP.request_id_of("7") == "r7"
    undeclared = _requests(lambda reqs: [q.pop("arrive_step") for q in reqs] and [q.pop("index") for q in reqs])
    assert [r["arrive_step"] for r in undeclared] == [0] * 8 and [r["index"] for r in undeclared] == list(range(8))   # a request with neither


def test_a_row_that_serves_with_the_prefix_cache_states_each_request_s_cached_rows_and_the_requests_that_made_them():
    doc = copy.deepcopy(DOC)
    doc["engine"]["args"]["enable_prefix_caching"] = True
    for q, ids in zip(doc["requests"], ([1] * 40, [1] * 35, [1] * 16 + [2] * 20)):      # blocks of 16: two of the first's, then one of them
        q.update(prompt_token_ids=ids, prompt_len=len(ids))
    rows = GP.manifest_requests(Workload.of(doc, "w.json"))
    assert [(r["cached"], r["producers"]) for r in rows] == [(0, []), (32, ["r0", "r0"]), (16, ["r0"]), *[(0, [])] * 5]
    assert [list(r)[-2:] for r in rows] == [["cached", "producers"]] * 8 and not {"cached", "producers"} & {k for r in _requests() for k in r}


@pytest.mark.parametrize("change, refusal", [
    (lambda reqs: reqs[1].update(index=5), "manifest request 1 declares index 5: requests must be listed in index order (r<k> = position k)"),
    (lambda reqs: reqs.reverse(), "manifest request 0 declares index 7: requests must be listed in index order (r<k> = position k)"),
    (lambda reqs: reqs[2].update(max_tokens=0), "manifest request {2}: no prompt_len / max_tokens"),
    (lambda reqs: reqs[2].update(prompt_len=0, prompt_token_ids=[]), "manifest request {2}: no prompt_len / max_tokens"),
], ids=["an-index-not-its-place", "the-list-reversed", "no-token-to-make", "no-prompt"])
def test_a_request_out_of_its_index_or_with_no_length_is_refused_by_name(change, refusal):
    with pytest.raises(ValueError) as refused:
        _requests(change)
    assert refused.value.args == (refusal.replace("{2}", DOC["requests"][2]["request_id"]),)


def test_the_lag_is_the_declared_target_s_and_async_scheduling_when_the_target_says_nothing():
    for target, lag, explicit in (({}, 1, False), (GPU, 1, False), ({"async_scheduling": True}, 1, True), ({"async_scheduling": False}, 0, True)):
        out = GP.workload_lag(_manifest(**target), {})
        declared = GP.workload_target(target).async_scheduling_declaration()
        assert out == {"schema": LAG_SCHEMA, **declared, "rule": out["rule"], "components": {}, "refusal": None} and out["rule"]
        assert (out["value"], out["async_scheduling"], out["declared_explicitly"], out["vllm_revision"]) == (lag, bool(lag), explicit, VLLM)
    assert GP.workload_lag({"engine": {"target": None}}, {})["value"] == 1


@pytest.mark.parametrize("target, why", [
    ({"async_schedulin": True}, "workload target: unknown TargetProfile fields ['async_schedulin']"),
    ({"async_scheduling": "no"}, "async_scheduling must be None (the pinned resolver's value), true or false; got 'no'"),
])
def test_a_target_block_that_is_not_a_target_profile_gives_no_lag_and_says_why(tmp_path, target, why):
    never_read = {(256, 31): _component(tmp_path, "c0", {"async_scheduling": False})}
    assert GP.workload_lag(_manifest(**target), never_read) == {
        "schema": LAG_SCHEMA, "value": None, "refusal": f"the manifest's target block does not parse as a TargetProfile: {why}"}


def test_each_component_is_held_to_the_workload_s_lag_and_one_with_no_declared_target_is_not_checked(tmp_path):
    old, bare = _component(tmp_path, "old"), str(tmp_path / "no-result")
    same, said = _component(tmp_path, "same", GPU), _component(tmp_path, "said", {**GPU, "async_scheduling": True})
    out = GP.workload_lag(_manifest(), {(256, 31): said, (64, 7): same, (128, 15, 32): old, (8, 1): bare})
    held = {"async_scheduling": True, "lag": 1, "vllm_revision": VLLM}
    unchecked = {"declared_target": None, "lag": None, "note": "component carries no target.declared (pre-v3 derive): not checked"}
    assert out["refusal"] is None and out["components"] == {
        PP.shape_key((8, 1)): {"dir": bare, **unchecked}, PP.shape_key((64, 7)): {"dir": same, **held, "declared_explicitly": False},
        PP.shape_key((128, 15, 32)): {"dir": old, **unchecked}, PP.shape_key((256, 31)): {"dir": said, **held, "declared_explicitly": True}}
    assert list(out["components"]) == [PP.shape_key(s) for s in ((8, 1), (64, 7), (128, 15, 32), (256, 31))]      # in the shapes' order
    assert GP.component_declared_target(bare) is None and GP.component_spec_histogram(bare) is None
    assert GP.component_declared_target(same) == GPU and GP.component_spec_histogram(_component(tmp_path, "h", spec_histogram={"Gemm": 2})) == {"Gemm": 2}
    assert GP.component_spec_histogram(_component(tmp_path, "list", spec_histogram=["Gemm"])) is None           # a histogram that is no table


TWO_SCHEDULERS = ("component (LP=64, T=7) BAD was derived under a target resolving async_scheduling={a} (LAG {la}) but the workload's declared "
                  "target resolves async_scheduling={b} (LAG {lb}): one workload Program cannot be specialized for two schedulers")
COUPLED = "the lag declaration is coupled to the pinned implementation"


@pytest.mark.parametrize("workload, component, refusal", [
    ({}, {"async_scheduling": False}, TWO_SCHEDULERS.format(a=False, la=0, b=True, lb=1)),
    ({"async_scheduling": False}, {}, TWO_SCHEDULERS.format(a=True, la=1, b=False, lb=0)),
    ({}, {"vllm_revision": "0ld"}, f"component (LP=64, T=7) BAD was derived for vLLM 0ld but the workload's target pins {VLLM}: {COUPLED}"),
], ids=["a-synchronous-component", "a-synchronous-workload", "another-vllm"])
def test_the_first_component_of_another_scheduler_or_another_vllm_is_named_and_none_after_it_is_read(tmp_path, workload, component, refusal):
    first, bad, after = _component(tmp_path, "first", workload), _component(tmp_path, "bad", component), _component(tmp_path, "after", {"num_sms": "many"})
    out = GP.workload_lag(_manifest(**workload), {(256, 31): after, (64, 7): bad, (8, 1): first})
    assert out["refusal"] == refusal.replace("BAD", bad) and out["value"] == (0 if workload else 1)
    assert list(out["components"]) == [PP.shape_key((8, 1)), PP.shape_key((64, 7))] and out["components"][PP.shape_key((64, 7))]["dir"] == bad


@pytest.mark.parametrize("declared, why", [
    ({"x": 1, "a": 2}, "unknown TargetProfile fields ['a', 'x']"),
    ({"num_sms": "many"}, "invalid literal for int() with base 10: 'many'"),
])
def test_a_component_s_declared_target_that_is_not_a_target_profile_is_refused_and_is_not_among_the_components(tmp_path, declared, why):
    first, bad = _component(tmp_path, "first", {}), _component(tmp_path, "bad", declared)
    out = GP.workload_lag(_manifest(), {(64, 7): bad, (8, 1): first})
    assert out["refusal"] == f"component (LP=64, T=7) {bad}: target.declared does not parse as a TargetProfile: {why}"
    assert list(out["components"]) == [PP.shape_key((8, 1))]
    moe = GP.workload_moe_construction(_manifest(), {(64, 7): bad})
    assert moe["refusal"] == out["refusal"] and moe["components"] == {}


EAGER, COMPILED = "bi-eager", "bi-compiled"
FACTS = {"cudagraph_mode": "PIECEWISE", "compilation_mode": "3", "capture_sizes": [1, 2], "backend": "inductor"}
NORMAL = {"cudagraph_mode": "PIECEWISE", "mode": "VLLM_COMPILE", "capture_sizes": [1, 2], "backend": "inductor"}


def _dumped(*blocks):
    return sorted(json.dumps(b, sort_keys=True) for b in blocks)


def test_an_eager_label_models_the_program_s_own_conditions_over_the_components_agreed_engine_block():
    assert GP.modelled_engine(EAGER, {}) == GP.modelled_engine("", {}) == GP.modelled_engine(None, {"a": None, "b": {}}) == (MODELLED_ENGINE, None)
    block = {"attention": "fa2", "enforce_eager": True, "enable_prefix_caching": None}
    assert GP.modelled_engine(EAGER, {"a": {"engine": block}, "b": {"engine": dict(block)}, "c": {"engine": "not a block"}, "d": None}) == (
        {"attention": "fa2", **MODELLED_ENGINE}, None)
    assert MODELLED_ENGINE["enforce_eager"] is True and MODELLED_ENGINE["enable_prefix_caching"] is False


@pytest.mark.parametrize("label, comps, engine, refusal", [
    (EAGER, {"a": {"engine": {"x": 1}}, "b": {"engine": {"x": 2}}}, MODELLED_ENGINE,
     f"components disagree on applicability.engine {_dumped({'x': 1}, {'x': 2})}"),
    (COMPILED, {"a": {"engine": {"x": 1, **FACTS}}, "b": {"engine": {"x": 2, **FACTS}}}, MODELLED_ENGINE,
     f"components disagree on applicability.engine {_dumped({'x': 1, **FACTS}, {'x': 2, **FACTS})}"),
    (EAGER, {"a": {"engine": {"enable_prefix_caching": True}}}, MODELLED_ENGINE,
     "a component's applicability.engine declares enable_prefix_caching true, which GP-01 does not model"),
    (EAGER, {"a": {"engine": {"enforce_eager": False}}}, MODELLED_ENGINE,
     "components declare the eager execution 'bi-eager' but an engine block with enforce_eager false"),
    (COMPILED, {"a": {"engine": {"enforce_eager": True, **FACTS}}}, {**FACTS, **MODELLED_ENGINE, "enforce_eager": False, "compilation": None},
     "components declare the compiled execution 'bi-compiled' but an engine block with enforce_eager true"),
    (COMPILED, {"a": {"compilation": {"cudagraph_mode": "PIECEWISE"}}, "b": {"compilation": {"cudagraph_mode": "FULL"}}, "c": None},
     {**MODELLED_ENGINE, "enforce_eager": False, "compilation": None},
     f"components disagree on the compilation facts {_dumped({'cudagraph_mode': 'PIECEWISE'}, {'cudagraph_mode': 'FULL'})}"),
], ids=["two-engine-blocks", "two-engine-blocks-compiled", "a-prefix-cache", "eager-not-eager", "compiled-but-eager", "two-sets-of-facts"])
def test_components_that_disagree_cache_prefixes_or_state_the_other_execution_are_refused_by_name(label, comps, engine, refusal):
    assert GP.modelled_engine(label, comps) == (engine, refusal)


@pytest.mark.parametrize("comps", [{}, {"a": None}, {"a": {"engine": {"enforce_eager": False}}}, {"a": {"compilation": {"backend": "inductor"}}},
                                   {"a": {"engine": FACTS}, "b": {"other": 1}}, {"a": {"compilation": FACTS}, "b": {"compilation": {"mode": 3}}}],
                         ids=["none", "an-empty-one", "no-facts", "no-cudagraph-mode", "one-without-a-block", "one-without-the-mode"])
def test_a_compiled_label_without_every_component_s_compilation_facts_is_refused(comps):
    engine, refusal = GP.modelled_engine(COMPILED, comps)
    assert refusal == ("components declare the compiled execution 'bi-compiled' but no compilation facts (applicability.engine / "
                       "applicability.compilation with cudagraph_mode, compilation_mode|mode, capture_sizes, backend): the Program cannot state the "
                       "compiled execution it models")
    assert (engine["enforce_eager"], engine["compilation"]) == (False, None) and {k: engine[k] for k in MODELLED_ENGINE if k != "enforce_eager"} == {
        k: v for k, v in MODELLED_ENGINE.items() if k != "enforce_eager"}


def test_a_compiled_label_models_the_agreed_facts_under_one_spelling_from_either_block():
    for block, comps in (("engine", {"a": {"engine": FACTS}, "b": {"engine": dict(FACTS)}, "c": None}),
                         ("compilation", {"a": {"compilation": FACTS}, "b": {"compilation": NORMAL}})):
        engine, refusal = GP.modelled_engine(COMPILED, comps)
        declared = FACTS if block == "engine" else {}
        assert refusal is None and engine == {**declared, **MODELLED_ENGINE, "enforce_eager": False, "compilation": NORMAL}


def _sampling(**stated):
    return WorkloadSampling.of({"variant": "greedy", **stated}, "w.json")


def test_the_stop_rule_is_the_set_of_the_ids_given_the_checkpoint_s_and_the_manifest_s_each_under_its_source(tmp_path):
    def rule(ids, source, directory):
        return {"schema": STOP_SCHEMA, "eos_token_ids": ids, "source": source, "checkpoint_dir": directory, "rule": STOPS_AT_EOS}

    ck = str(tmp_path)
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [9, 2]}))
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 2}))
    files = {"generation_config.json": [2, 9], "config.json": [2]}
    assert GP.stop_rule(ck, None) == rule([2, 9], files, ck) and GP.stop_rule(None, [4], None) == rule([4], {"--eos-token-ids": [4]}, None)
    both = GP.stop_rule(ck, [7, 7, 3], _sampling(stop_token_ids=[42, 5]))
    assert both == rule([2, 3, 5, 7, 9, 42], {"--eos-token-ids": [3, 7], **files, "manifest.sampling.stop_token_ids": [5, 42]}, ck)
    assert list(both["source"]) == ["--eos-token-ids", "generation_config.json", "config.json", "manifest.sampling.stop_token_ids"]
    # the manifest's own set is the declaration of record: with no checkpoint, and beside one that agrees
    assert GP.stop_rule(None, None, _sampling(eos_token_ids=[2, 9, 2])) == rule([2, 9], {"manifest.sampling.eos_token_ids": [2, 9]}, None)
    assert GP.stop_rule(ck, [7], _sampling(eos_token_ids=[9, 2])) == rule([2, 7, 9], {"--eos-token-ids": [7], **files,
                                                                                      "manifest.sampling.eos_token_ids": [2, 9]}, ck)


def test_a_manifest_that_switches_eos_off_and_a_row_with_no_id_say_so_and_a_set_that_is_not_the_checkpoint_s_is_refused(tmp_path):
    for name, text in (("generation_config.json", json.dumps({"eos_token_id": [9, 2]})), ("config.json", "{")):
        (tmp_path / name).write_text(text)
    read = GP.stop_rule(str(tmp_path), None)["source"]
    assert read == {"generation_config.json": [2, 9], "config.json": f"unreadable: {_decode_error('{')!r}"}       # the file that is no JSON
    off = {"schema": STOP_SCHEMA, "eos_token_ids": [], "ignore_eos": True, "source": None, "checkpoint_dir": None,
           "rule": "manifest sampling.ignore_eos: every request must reach max_tokens; any early stop FAILS G6"}
    assert GP.stop_rule(None, None, _sampling(ignore_eos=True)) == off
    assert GP.stop_rule(str(tmp_path), [7], _sampling(ignore_eos=True, stop_token_ids=[5])) == {
        **off, "source": {"--eos-token-ids": [7], **read, "manifest.sampling.stop_token_ids": [5]}, "checkpoint_dir": str(tmp_path)}
    none = {"schema": STOP_SCHEMA, "eos_token_ids": None, "source": None, "checkpoint_dir": None,
            "note": "no EOS set declared: GM-01 fails closed on any request that stops before its max_tokens"}
    assert GP.stop_rule(None, None) == GP.stop_rule(None, [], _sampling(ignore_eos=False, stop_token_ids=[])) == none
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "generation_config.json").write_text(json.dumps({"bos_token_id": 1}))        # a file that states none is passed over
    (bare / "config.json").write_text("{")
    assert GP.stop_rule(str(bare), None) == {**none, "source": {"config.json": read["config.json"]}, "checkpoint_dir": str(bare)}
    (bare / "config.json").write_text(json.dumps({"eos_token_id": 2}))                    # the checkpoint's set is both files'
    assert GP.stop_rule(str(bare), None, _sampling(eos_token_ids=[2, 9])) == {
        "schema": STOP_SCHEMA, "eos_token_ids": None, "source": {"config.json": [2], "manifest.sampling.eos_token_ids": [2, 9]},
        "checkpoint_dir": str(bare), "refusal": "manifest sampling.eos_token_ids [2, 9] != the pinned checkpoint's eos_token_id set [2]: the "
                                                "manifest was pinned from another tokenizer/generation config"}


def _decode_error(text):
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return e
    raise AssertionError(f"{text!r} is JSON")


def test_a_checkpoint_file_that_cannot_be_read_is_named_with_120_characters_of_its_error(tmp_path):
    (tmp_path / "config.json").write_bytes(b"\xff" * 64)
    with pytest.raises(UnicodeDecodeError) as error, open(tmp_path / "config.json") as f:
        json.load(f)
    whole = f"unreadable: {error.value!r}"
    assert len(whole) > 120
    assert GP.stop_rule(str(tmp_path), None)["source"] == {"config.json": whole[:120]}
