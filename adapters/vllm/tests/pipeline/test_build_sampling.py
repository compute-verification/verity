"""The sampling policy a Build constructs a row's Program under (`verity_vllm.pipeline.build_sampling`): the policy an argument
or a workload file states, each form whole, with its source and the id length its seed rule reads; an argument and a file that
disagree; and the stochastic sampler's launch geometry, with each of its refusals whole."""
from __future__ import annotations

import json
import re
import types
from pathlib import Path

import pytest

from verity_vllm.pipeline import build_sampling as BS

STOCH = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "label": "stoch-t0.8-p0.95"}
PARAMS = {"temperature": 0.8, "top_p": 0.95}
VARIANTS = {"combo": {"temperature": 0.7, "top_p": 0.9, "top_k": 50}, "zero": {"temperature": 0.0}}
COMBO = {**STOCH, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "label": "combo"}
ONE = {**STOCH, "temperature": 1.0, "top_p": 1.0}
L40S = types.SimpleNamespace(num_sms=142)
GEOMETRY = {"num_sms": 142, "pipeline": "_apply_topp_split", "relation": "registered", "splits": 32,
            "rule": "S = registered_splits(num_SMs) at every sampling event, whatever the live batch: the served engine pins "
                    "vLLM's split count to it (engine.topp_pin)",
            "constant": "Const32[32]: the select's 6th operand at every sampling event (no `splits` input)"}


def _file(tmp_path, doc):
    p = tmp_path / "workload.json"
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return str(p)


def _refusal(sampling, hint, reason="sampler_geometry", **detail):
    return {"reason": reason, "phase": "applicability", "node": None, "module_path": None, "hint": hint,
            "detail": {"sampling": sampling, "samplers_bound": BS.SAMPLERS_BOUND, **detail}}


# --- the policy -----------------------------------------------------------------------------------------------------------------


def test_with_no_argument_and_no_workload_the_policy_is_greedy_by_default():
    assert BS.sampling_policy(None, None) == {"kind": "greedy", "source": "default"}
    assert BS.sampling_policy("", "") == {"kind": "greedy", "source": "default"}


@pytest.mark.parametrize("arg, policy", [
    ("greedy", {"kind": "greedy"}),
    (" greedy\n", {"kind": "greedy"}),
    ("stoch-t0.8-p0.95", STOCH),
    ("stoch-t1-p1-k50", {**ONE, "top_k": 50, "label": "stoch-t1-p1-k50"}),
    ("stoch-t0-p0.9", {"kind": "greedy"}),
    ('{"temperature": 0.0, "top_p": 0.9}', {"kind": "greedy"}),
    ('{"temperature": 0.5}', {**ONE, "temperature": 0.5, "label": "stoch-t0.5-p1"}),
    ("{}", {**ONE, "label": "stoch-t1-p1"}),
    (' {"temperature": 2, "top_p": 0.25, "top_k": null, "seeded": false, "rid": "7"}',
     {"kind": "stochastic", "temperature": 2.0, "top_p": 0.25, "top_k": 0, "seeded": False, "label": "stoch-t2-p0.25", "rid": 7}),
])
def test_an_argument_states_the_policy_by_its_name_or_as_json(arg, policy):
    assert BS.sampling_policy(arg, None) == {**policy, "source": "arg"}


@pytest.mark.parametrize("arg", ["nucleus", "stoch-t0.8", "stoch-t0.8-p0.95-k", "xstoch-t1-p1", "stoch-t1-p1-k5x"])
def test_an_argument_of_no_known_form_is_refused_with_the_forms(arg):
    message = f"unknown sampling policy {arg!r} (greedy | stoch-t<T>-p<P>[-k<K>] | JSON)"
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        BS.sampling_policy(arg, None)
    assert str(error.value) == message


@pytest.mark.parametrize("sweep, policy", [
    ({"sampling": "named", "sampling_params": {**PARAMS, "seeded": False}, "sampling_seeds": []}, {**STOCH, "seeded": False, "label": "named"}),
    ({"sampling": "named", "sampling_params": {**PARAMS, "seeded": False}, "sampling_seeds": [7]}, {**STOCH, "label": "named"}),
    ({"sampling": "named"}, {**ONE, "label": "named"}),
    ({"sampling": "named", "sampling_params": {"temperature": 0.0}}, {"kind": "greedy"}),
    ({"sampling": "greedy", "sampling_params": PARAMS}, {"kind": "greedy"}),
    ({"sampling": "greedy", "sampling_seeds": [7]}, {"kind": "greedy", "seeded": True}),
    ({"sampling": None, "sampling_params": PARAMS}, COMBO),
    ({}, COMBO),
])
def test_a_generated_workload_states_its_policy_in_its_sweep_block_and_one_with_none_by_its_requests(tmp_path, sweep, policy):
    w = _file(tmp_path, {"sweep": sweep, "variants": VARIANTS, "requests": [{"variant": "combo"}]})
    assert BS.sampling_policy(None, w) == {**policy, "source": "workload", "workload": w}


@pytest.mark.parametrize("doc, policy", [
    ({"requests": [{"variant": "combo"}, {"variant": "combo", "seed": 3}]}, COMBO),
    ({"sampling": {"variant": "combo"}, "requests": [{}, {"seed": 3}]}, COMBO),
    ({"sampling": {"variant": "combo"}, "requests": [{"variant": "zero"}]}, {"kind": "greedy"}),
    ({"requests": [{}, {"variant": "greedy"}]}, {"kind": "greedy"}),
    ({"requests": [{"seed": 3}]}, {**ONE, "label": "stoch-t1-p1"}),
    ({"requests": [{"variant": "greedy", "seed": 3}]}, {**ONE, "label": "greedy"}),
    ({"requests": [{"variant": "unnamed"}]}, {**ONE, "label": "unnamed"}),
    ({"variants": None, "sampling": None, "requests": [{"variant": "combo"}]}, {**ONE, "label": "combo"}),
    ({"requests": [{"variant": "combo"}, {}, {"variant": "zero"}, {"variant": "combo"}]},
     {"kind": "mixed", "policies": [{"kind": "greedy"}, COMBO]}),
    ({"sampling": {"variant": "combo"}, "requests": []}, COMBO),
    ({"sampling": {"variant": "combo"}}, COMBO),
    ({"sampling": {"variant": "unnamed"}}, {"kind": "greedy"}),
    ({"sampling": {"variant": None}}, {"kind": "greedy"}),
    ({}, {"kind": "greedy"}),
])
def test_a_workload_with_no_sweep_block_states_its_policy_by_its_requests_variants_and_seeds(tmp_path, doc, policy):
    w = _file(tmp_path, {"variants": VARIANTS, **doc})
    assert BS.sampling_policy(None, w) == {**policy, "source": "workload", "workload": w}


@pytest.mark.parametrize("ids, rid", [(["ab", "cd", None], {"rid": 2}), ([12, "ab"], {"rid": 2}), (["é"], {"rid": 2}), (["ab", "abc"], {}),
                                      ([None], {}), ([], {})])
def test_a_stochastic_policy_of_a_workload_carries_the_id_length_its_requests_share(tmp_path, ids, rid):
    requests = [{"request_id": i} for i in ids] + [{}]
    w = _file(tmp_path, {"sweep": {"sampling": "named", "sampling_params": PARAMS}, "requests": requests})
    assert BS.sampling_policy(None, w) == {**STOCH, "label": "named", **rid, "source": "workload", "workload": w}
    w = _file(tmp_path, {"sweep": {"sampling": "greedy"}, "requests": requests})
    assert BS.sampling_policy(None, w) == {"kind": "greedy", "source": "workload", "workload": w}


def test_an_argument_and_a_workload_that_agree_give_the_workloads_policy_and_two_that_do_not_are_refused(tmp_path):
    w = _file(tmp_path, {"sweep": {"sampling": "named", "sampling_params": PARAMS}})
    assert BS.sampling_policy("stoch-t0.8-p0.95", w) == {**STOCH, "label": "named", "source": "workload", "workload": w}
    for arg in ("greedy", "stoch-t0.8-p0.9", "stoch-t0.8-p0.95-k1", '{"temperature": 0.8, "top_p": 0.95, "seeded": false}',
                '{"temperature": 0.8, "top_p": 0.95, "rid": 2}'):
        message = f"--sampling {arg!r} disagrees with the manifest's sampling policy { {**STOCH, 'label': 'named'} } ({w})"
        with pytest.raises(ValueError, match=re.escape(message)) as error:
            BS.sampling_policy(arg, w)
        assert str(error.value) == message


# --- the stochastic sampler's geometry ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", [{"kind": "greedy", "top_k": 50}, {"kind": "mixed", "policies": []}, {}])
def test_a_policy_that_is_not_stochastic_has_no_geometry_and_no_refusal(sampling):
    given = dict(sampling)
    assert BS.sampler_geometry(given, None) is None and given == sampling


@pytest.mark.parametrize("num_sms, splits", [(142, 32), ("142", 32), (16, 16), (6, 4)])
def test_a_seeded_stochastic_policy_takes_the_registered_split_count_of_its_targets_sm_count(num_sms, splits):
    sampling = dict(STOCH)
    assert BS.sampler_geometry(sampling, types.SimpleNamespace(num_sms=num_sms), LP=8, T=4) is None
    assert sampling == {**STOCH, "splits": splits, "geometry": {**GEOMETRY, "num_sms": int(num_sms), "splits": splits, "constant":
                                                                 f"Const32[{splits}]: the select's 6th operand at every sampling event (no `splits` input)"}}


@pytest.mark.parametrize("requests", [[{}] * 64, [], None])
def test_a_workload_of_up_to_64_requests_or_of_none_stated_takes_the_geometry(tmp_path, requests):
    sampling = {**STOCH, "splits": 32, "top_k": None, "workload": _file(tmp_path, {"requests": requests})}
    assert BS.sampler_geometry(sampling, L40S) is None and sampling["geometry"] == GEOMETRY and sampling["splits"] == 32


@pytest.mark.parametrize("top_k", [50, "50"])
def test_a_top_k_is_refused_first_as_a_sampler_with_no_definition(top_k):
    sampling = {**STOCH, "top_k": top_k, "seeded": False, "workload": "/no/such/workload.json"}
    given = dict(sampling)
    assert BS.sampler_geometry(given, None) == _refusal(
        {**STOCH, "top_k": top_k, "seeded": False},
        "top_k = 50: vLLM's top-k path (`_topk_topp_kernel` / `_apply_topk_topp`) has no exact Definition; only the p-only split "
        "pipeline (`_apply_topp_split`) is modelled (TopPMask_v1{V})", reason="sampler", missing=["TopKMask"])
    assert given == sampling


@pytest.mark.parametrize("seeded", [{"seeded": False}, {"seeded": None}, {}])
def test_a_policy_with_no_declared_seed_is_refused_as_a_sampler(seeded):
    sampling = {k: v for k, v in STOCH.items() if k != "seeded"} | seeded
    assert BS.sampler_geometry(dict(sampling), L40S) == _refusal(
        sampling, "the policy has no explicit per-request seed: vLLM picks a random seed when `SamplingParams.seed` is None "
        "(SamplingStates.add_request), so the RNG stream is not a declared input and the sampled token is not reproducible from the "
        "manifest — declare `seed` per request (sweep `sampling_seeds`)", reason="sampler", missing=["declared seed"])


def test_a_workload_that_cannot_be_read_is_refused_with_its_path_and_the_error(tmp_path):
    w = _file(tmp_path, "not json")
    with pytest.raises(json.JSONDecodeError) as error:
        json.loads("not json")
    assert BS.sampler_geometry({**STOCH, "workload": w}, L40S) == _refusal(
        STOCH, f"cannot read the workload manifest {w!r} for the sampler batch: {error.value!r}")
    absent = str(tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError) as missing:
        Path(absent).read_text()
    assert BS.sampler_geometry({**STOCH, "workload": absent}, L40S) == _refusal(
        STOCH, f"cannot read the workload manifest {absent!r} for the sampler batch: {missing.value!r}")


def test_a_workload_of_65_requests_is_refused_as_past_the_split_pipelines_batch(tmp_path):
    sampling = {**STOCH, "workload": _file(tmp_path, {"requests": [{}] * 65})}
    assert BS.sampler_geometry(sampling, L40S) == _refusal(
        STOCH, "sampler batch 65 > _SPLIT_MAX_BATCH 64: vLLM runs `_topk_topp_kernel`, not the split pipeline")
    assert "splits" not in sampling and "geometry" not in sampling


@pytest.mark.parametrize("target", [types.SimpleNamespace(), types.SimpleNamespace(num_sms=None), None])
def test_a_target_that_declares_no_sm_count_is_refused(target):
    assert BS.sampler_geometry(dict(STOCH), target) == _refusal(
        STOCH, "the target profile declares no num_sms: the registered split count registered_splits(num_SMs) has no num_SMs")


@pytest.mark.parametrize("splits", [16, 32.0, "32", True, None])
def test_a_split_count_that_is_not_the_registered_one_as_an_int_is_refused_with_both(splits):
    sampling = {**STOCH, "splits": splits}
    assert BS.sampler_geometry(dict(sampling), L40S) == _refusal(
        sampling, f"unregistered split count: the session declares top-p split count {splits!r}, not the registered one, "
        "registered_splits(142) = 32: serving runs every launch at it (engine.topp_pin), and the Program carries it as Const32[32]")
