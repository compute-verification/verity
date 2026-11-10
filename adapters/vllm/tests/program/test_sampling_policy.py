"""The sampling policy is a structural input of the Bet A construction and an applicability condition; a workload whose sampler
the builder cannot bind is REFUSED, never built as the greedy Program (that would be a fail-open Build).

GPU-free and vLLM-free: the refusal happens in `derive_step.run` before any vLLM import, so these tests run anywhere `verity.ml.circuits`
imports.  Covered: policy parsing from `--sampling` and from the two manifest formats (sweep/workload.py `sweep.*`, legacy
`variants` + per-request `variant`/`seed`), disagreement between the two sources, the refusal record (`reason: "sampler"`,
`structural_inputs.sampling` = the declared policy, no artifact), applicability carrying the sampler, and greedy-by-omission recorded
as such.

Run: `PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/program/test_sampling_policy.py`
"""
from __future__ import annotations

import json
import os

import pytest

from verity_vllm.pipeline.build import SAMPLERS_BOUND, applicability_conditions, run, sampling_policy


def _sweep_manifest(tmp_path, sampling: str, params: dict, seeds: list[int]):
    p = tmp_path / "wl_sweep.json"
    p.write_text(json.dumps({"schema": "verity-vllm/workload/v2", "model": "unsloth/Llama-3.2-1B", "requests": [],
                             "sweep": {"row_id": "r", "sampling": sampling, "sampling_params": params, "sampling_seeds": seeds}}))
    return str(p)


def _legacy_manifest(tmp_path, requests: list[dict], top_variant: str | None = "greedy"):
    p = tmp_path / "wl_legacy.json"
    doc = {"schema": "verity-vllm/workload/v2", "model": "Qwen/Qwen2.5-1.5B",
           "variants": {"greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0}, "temp1": {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
                        "combo": {"temperature": 0.7, "top_p": 0.9, "top_k": 50}},
           "sampling": {"variant": top_variant, "seed": 0, "n": 1}, "requests": requests}
    p.write_text(json.dumps(doc))
    return str(p)


# ---------------------------------------------------------------------------------------------------------
# policy parsing
# ---------------------------------------------------------------------------------------------------------


def test_default_is_greedy_by_omission():
    pol = sampling_policy(None, None)
    assert pol["kind"] == "greedy" and pol["source"] == "default"


def test_arg_greedy_and_stoch_label():
    assert sampling_policy("greedy", None)["kind"] == "greedy"
    pol = sampling_policy("stoch-t0.8-p0.95", None)
    assert pol["kind"] == "stochastic" and pol["temperature"] == 0.8 and pol["top_p"] == 0.95 and pol["top_k"] == 0 and pol["source"] == "arg"
    pol = sampling_policy("stoch-t1-p1-k50", None)
    assert pol["top_k"] == 50 and pol["top_p"] == 1.0


def test_arg_json_temperature_zero_is_greedy():
    assert sampling_policy('{"temperature": 0.0, "top_p": 0.9}', None)["kind"] == "greedy"
    assert sampling_policy('{"temperature": 0.5}', None)["kind"] == "stochastic"


def test_unknown_arg_refused():
    with pytest.raises(ValueError):
        sampling_policy("nucleus", None)


def test_sweep_manifest_policy(tmp_path):
    wl = _sweep_manifest(tmp_path, "stoch-t0.8-p0.95", {"temperature": 0.8, "top_p": 0.95, "top_k": 0}, [1, 2])
    pol = sampling_policy(None, wl)
    assert pol["kind"] == "stochastic" and pol["temperature"] == 0.8 and pol["top_p"] == 0.95 and pol["seeded"] is True and pol["source"] == "workload"
    wl = _sweep_manifest(tmp_path, "greedy", {"temperature": 0.0, "top_p": 1.0, "top_k": 0}, [])
    assert sampling_policy(None, wl)["kind"] == "greedy"


def test_legacy_manifest_policy(tmp_path):
    wl = _legacy_manifest(tmp_path, [{"index": 0, "prompt_len": 32, "max_tokens": 16}], top_variant="greedy")
    assert sampling_policy(None, wl)["kind"] == "greedy"
    wl = _legacy_manifest(tmp_path, [{"index": 0, "prompt_len": 32, "max_tokens": 16, "seed": 0}], top_variant="combo")
    pol = sampling_policy(None, wl)
    assert pol["kind"] == "stochastic" and pol["temperature"] == 0.7 and pol["top_p"] == 0.9 and pol["top_k"] == 50
    # a batch mixing greedy and stochastic requests is not one policy: "mixed" (refused downstream, never built greedy)
    wl = _legacy_manifest(tmp_path, [{"index": 0, "variant": "greedy"}, {"index": 1, "seed": 0}], top_variant="temp1")
    assert sampling_policy(None, wl)["kind"] == "mixed"


def test_arg_and_manifest_must_agree(tmp_path):
    wl = _sweep_manifest(tmp_path, "stoch-t0.8-p0.95", {"temperature": 0.8, "top_p": 0.95, "top_k": 0}, [1])
    assert sampling_policy("stoch-t0.8-p0.95", wl)["kind"] == "stochastic"
    with pytest.raises(ValueError):
        sampling_policy("greedy", wl)


# ---------------------------------------------------------------------------------------------------------
# the refusal in the driver, and applicability
# ---------------------------------------------------------------------------------------------------------


def test_top_k_row_is_refused_before_export(tmp_path):
    """top-k has no Definition (`_topk_topp_kernel`): refused fail-closed with the reason, before any export."""
    out = str(tmp_path / "build")
    pol = sampling_policy("stoch-t0.7-p0.9-k50", None)
    res = run(out, 1, 8, "b1-eager-v3", "gloo", wrapper="request", LP=8, T=4, sampling=pol)
    assert res["ok"] is False
    assert res["unsupported"]["reason"] == "sampler" and res["unsupported"]["phase"] == "applicability"
    assert "top_k" in res["unsupported"]["hint"] and "TopKMask" in res["unsupported"]["detail"]["missing"]
    assert "program" not in res and not os.path.exists(os.path.join(out, "artifact.json"))
    man = json.load(open(os.path.join(out, "construction_manifest.json")))
    assert man["structural_inputs"]["sampling"]["kind"] == "stochastic" and man["structural_inputs"]["sampling"]["top_k"] == 50
    assert man["ok"] is False and man["unsupported"]["reason"] == "sampler"


def test_unseeded_stochastic_row_is_refused(tmp_path):
    """No declared seed -> vLLM draws one at random: the RNG stream is not a Program input; refused, never built."""
    pol = sampling_policy('{"temperature": 0.8, "top_p": 0.95, "seeded": false}', None)
    assert pol["kind"] == "stochastic" and pol["seeded"] is False
    res = run(str(tmp_path / "b"), 1, 8, "b1-eager-v3", "gloo", wrapper="request", LP=8, T=4, sampling=pol)
    assert res["ok"] is False and res["unsupported"]["reason"] == "sampler" and "seed" in res["unsupported"]["hint"]


def test_mixed_batch_is_refused(tmp_path):
    res = run(str(tmp_path / "b"), 1, 8, "b1-eager-v3", "gloo", wrapper="step", sampling={"kind": "mixed", "policies": []})
    assert res["ok"] is False and res["unsupported"]["reason"] == "sampler"


# ---------------------------------------------------------------------------------------------------------
# the sampler geometry: S is the registered split count of the target (`topp_geometry.registered_splits`), the constant the served
# engine pins -- never the live batch's, never a declared schedule
# ---------------------------------------------------------------------------------------------------------


def _requests_manifest(tmp_path, n: int, arrive=None, caps=None, seeds=True):
    from verity_vllm.pipeline.workload import SAMPLING
    p = tmp_path / f"wl_{n}.json"
    reqs = [{"request_id": f"r{i}", "index": i, "prompt_len": 8, "max_tokens": int((caps or [16] * n)[i]),
             "arrive_step": int((arrive or [0] * n)[i]), **({"seed": 1000 + i} if seeds else {})} for i in range(n)]
    p.write_text(json.dumps({"schema": "verity-vllm/workload/v2", "model": "unsloth/Llama-3.2-1B", "requests": reqs,
                             "sweep": {"row_id": "r", "sampling": "stoch-t0.8-p0.95", "sampling_params": SAMPLING["stoch-t0.8-p0.95"],
                                       "sampling_seeds": [1000 + i for i in range(n)] if seeds else []}}))
    return str(p)


def _l40s():
    from verity_vllm.program.frontend.target_profile import TargetProfile
    return TargetProfile.default().with_overrides(num_sms=142)


GEOM_KEYS = {"num_sms", "pipeline", "relation", "splits", "rule", "constant"}


@pytest.mark.parametrize("num_sms,S", [(188, 32), (142, 32), (132, 32), (16, 16), (6, 4)])
def test_every_workload_takes_the_registered_s_of_its_sm_count(tmp_path, num_sms, S):
    """The served engine runs every top-p launch at the registered split count (`engine.topp_pin`), so every stochastic row -- one
    request, B2, the uniform B8, the B8 whose live count would change vLLM's own S, staggered arrivals, no manifest at all -- takes
    S = registered_splits(num_SMs) = SplitsFor_v1(1, num_SMs) as a constant of its Program (`sampling["splits"]` the int, no `splits`
    input), for every wrapper shape and the step wrapper alike; the geometry carries nothing of the batch, so every component of every
    workload on the target declares the same (GP-01 agreement by construction)."""
    from verity_vllm.pipeline.build import sampler_geometry
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.frontend.vllm_meta import registered_policy_splits
    from verity.ml.definitions.registry.topp_geometry import registered_splits, splits_for_v1
    target = TargetProfile.default().with_overrides(num_sms=num_sms)
    cases = [dict(n=1), dict(n=2, arrive=[0, 5], caps=[16, 9]), dict(n=8), dict(n=8, caps=[16] * 4 + [3] * 4),
             dict(n=8, arrive=[0, 0, 0, 0, 3, 3, 6, 6], caps=[16] * 8), None]
    geoms = set()
    for c in cases:
        for lp_t in ((8, 15), (8, 9), (None, None)):
            if c is None:
                pol = sampling_policy(None, None)
                pol.update(kind="stochastic", seeded=True, temperature=0.8, top_p=0.95, top_k=0)
            else:
                pol = sampling_policy(None, _requests_manifest(tmp_path, c["n"], arrive=c.get("arrive"), caps=c.get("caps")))
            assert sampler_geometry(pol, target, LP=lp_t[0], T=lp_t[1]) is None
            assert pol["splits"] == S == registered_splits(num_sms) == splits_for_v1(1, num_sms) and registered_policy_splits(pol) == S
            g = pol["geometry"]
            assert set(g) == GEOM_KEYS, sorted(g)
            assert g["num_sms"] == num_sms and g["splits"] == S and g["relation"] == "registered"
            geoms.add(json.dumps(g, sort_keys=True))
    assert len(geoms) == 1
    # a policy whose int is not the registered S of its geometry, or that has no geometry, carries no registered S
    assert registered_policy_splits(dict(pol, splits=S * 2 if S < 32 else 16)) is None
    assert registered_policy_splits({k: v for k, v in pol.items() if k != "geometry"}) is None


def test_geometry_refusals_are_named(tmp_path):
    from verity_vllm.pipeline.build import sampler_geometry
    from verity_vllm.program.frontend.target_profile import TargetProfile
    # B > 64: vLLM runs `_topk_topp_kernel`, not the split pipeline
    pol = sampling_policy(None, _requests_manifest(tmp_path, 65))
    ref = sampler_geometry(pol, _l40s(), LP=8, T=15)
    assert ref is not None and ref["reason"] == "sampler_geometry" and "_SPLIT_MAX_BATCH" in ref["hint"] and "splits" not in pol
    # no declared seed
    pol = sampling_policy(None, _requests_manifest(tmp_path, 2, seeds=False))
    pol["seeded"] = False
    ref = sampler_geometry(pol, _l40s())
    assert ref is not None and ref["reason"] == "sampler" and "seed" in ref["hint"]
    # top-k
    pol = sampling_policy(None, _requests_manifest(tmp_path, 2))
    pol["top_k"] = 50
    assert sampler_geometry(pol, _l40s())["detail"]["missing"] == ["TopKMask"]
    # a target without num_sms has no registered S
    pol = sampling_policy(None, _requests_manifest(tmp_path, 2))
    class _NoSms:
        num_sms = None
    ref = sampler_geometry(pol, _NoSms())
    assert ref is not None and ref["reason"] == "sampler_geometry" and "num_sms" in ref["hint"]
    # greedy: nothing to derive
    assert sampler_geometry({"kind": "greedy"}, _l40s()) is None


def test_declared_live_counts_follow_arrivals_and_caps():
    from verity_vllm.pipeline.build import declared_live_counts
    reqs = [{"arrive_step": 0, "max_tokens": 4}, {"arrive_step": 1, "max_tokens": 2}, {"arrive_step": 5, "max_tokens": 1}]
    assert declared_live_counts(reqs) == [1, 2, 2, 1, 0, 1]


def test_make_wrapper_refuses_a_declared_s():
    """A numeric `splits` other than the registered S of the policy's geometry is refused by name."""
    pytest.importorskip("torch")
    from verity_vllm.program.frontend.vllm_meta import make_wrapper
    with pytest.raises(ValueError, match="must be the registered"):
        make_wrapper("request", object(), LP=8, T=4, sampling={"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": 32})
    with pytest.raises(ValueError, match="must be the registered"):
        make_wrapper("request", object(), LP=8, T=4, sampling={"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": 16,
                                                                "geometry": {"num_sms": 188}})
    with pytest.raises(ValueError, match="must be the registered"):
        make_wrapper("request", object(), LP=8, T=4, sampling={"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": [16] * 5})


def test_geometry_refusal_is_the_driver_refusal(tmp_path):
    pol = sampling_policy(None, _requests_manifest(tmp_path, 65))
    res = run(str(tmp_path / "b"), 1, 8, "b1-eager-v3", "gloo", wrapper="request", LP=8, T=15, sampling=pol, target=_l40s())
    assert res["ok"] is False and res["unsupported"]["reason"] == "sampler_geometry" and res["unsupported"]["phase"] == "applicability"


def test_applicability_carries_the_sampler():
    from verity_vllm.program.frontend.target_profile import TargetProfile
    prof = TargetProfile.default()
    g = applicability_conditions(prof, {}, {}, {}, {"kind": "greedy", "source": "default"})
    s = applicability_conditions(prof, {}, {}, {}, {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "source": "arg", "label": "x",
                                                    "splits": "derived"})
    assert g["sampler"] == {"kind": "greedy"} and g["token_select"] == SAMPLERS_BOUND["greedy"]
    # the stochastic select IS bound now (GumbelTopPTokenSelect_v1{V}); the policy is a condition of reuse, S is DERIVED
    assert s["sampler"] == {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}
    assert s["token_select"] == SAMPLERS_BOUND["stochastic"] and "GumbelTopPTokenSelect" in s["token_select"]
    assert g != s  # a greedy artifact is not reusable for a stochastic row (check_reuse names the differing field)
    # `source` / `label` / `workload` are provenance, not applicability: the same policy from two sources is one condition
    assert applicability_conditions(prof, {}, {}, {}, {"kind": "greedy", "source": "workload", "workload": "x.json"})["sampler"] == g["sampler"]
