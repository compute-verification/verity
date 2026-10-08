"""R13 X-05 (B-SAMPLING) — the sampling policy is a structural input of the Bet A construction and an applicability condition;
a row whose sampler the builder cannot bind is REFUSED, never built as the greedy Program (F-dse-08 was a fail-open Build).

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
    p.write_text(json.dumps({"schema": "verity-vllm/workload/v1", "model": "unsloth/Llama-3.2-1B", "requests": [],
                             "sweep": {"row_id": "r", "sampling": sampling, "sampling_params": params, "sampling_seeds": seeds}}))
    return str(p)


def _legacy_manifest(tmp_path, requests: list[dict], top_variant: str | None = "greedy"):
    p = tmp_path / "wl_legacy.json"
    doc = {"schema": "verity-vllm/workload/v1", "model": "Qwen/Qwen2.5-1.5B",
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
    """[R15 stoch] top-k has no Definition (`_topk_topp_kernel`): refused fail-closed with the reason, before any export."""
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
    """[R15 stoch] no declared seed -> vLLM draws one at random: the RNG stream is not a Program input; refused, never built."""
    pol = sampling_policy('{"temperature": 0.8, "top_p": 0.95, "seeded": false}', None)
    assert pol["kind"] == "stochastic" and pol["seeded"] is False
    res = run(str(tmp_path / "b"), 1, 8, "b1-eager-v3", "gloo", wrapper="request", LP=8, T=4, sampling=pol)
    assert res["ok"] is False and res["unsupported"]["reason"] == "sampler" and "seed" in res["unsupported"]["hint"]


def test_mixed_batch_is_refused(tmp_path):
    res = run(str(tmp_path / "b"), 1, 8, "b1-eager-v3", "gloo", wrapper="step", sampling={"kind": "mixed", "policies": []})
    assert res["ok"] is False and res["unsupported"]["reason"] == "sampler"


# ---------------------------------------------------------------------------------------------------------
# [R15 stoch -> R16 coord M-0169] the sampler geometry: S is a DERIVED input of the workload (S_t = SplitsFor_v1(|live(t)|, num_SMs)),
# the request Program's `splits` Input -- never a static, never a declared schedule
# ---------------------------------------------------------------------------------------------------------


def _requests_manifest(tmp_path, n: int, arrive=None, caps=None, seeds=True):
    from verity_vllm.pipeline.workload import SAMPLING
    p = tmp_path / f"wl_{n}.json"
    reqs = [{"request_id": f"r{i}", "index": i, "prompt_len": 8, "max_tokens": int((caps or [16] * n)[i]),
             "arrive_step": int((arrive or [0] * n)[i]), **({"seed": 1000 + i} if seeds else {})} for i in range(n)]
    p.write_text(json.dumps({"schema": "verity-vllm/workload/v1", "model": "unsloth/Llama-3.2-1B", "requests": reqs,
                             "sweep": {"row_id": "r", "sampling": "stoch-t0.8-p0.95", "sampling_params": SAMPLING["stoch-t0.8-p0.95"],
                                       "sampling_seeds": [1000 + i for i in range(n)] if seeds else []}}))
    return str(p)


def _l40s():
    from verity_vllm.program.frontend.target_profile import TargetProfile
    return TargetProfile.default().with_overrides(num_sms=142)


DERIVED_GEOM_KEYS = {"sampler_batch", "num_sms", "pipeline", "relation", "rule", "live", "splits_range_1_to_B", "input"}


def test_geometry_is_derived_for_every_batch_and_chronology(tmp_path, monkeypatch):
    """Every stochastic row of several requests -- B2, the uniform B8, the B8 whose live count changes S, staggered arrivals, a shape that
    is no declared request -- gets the SAME geometry: splits 'derived' + the relation; no schedule, no per-request field, no assumption.
    The R15 knob STOCH_DECLARED_CHRONOLOGY_S is dead (set or not: same result).  (One request: the constant, below.)"""
    from verity_vllm.pipeline.build import sampler_geometry
    cases = [dict(n=2, arrive=[0, 5], caps=[16, 9]), dict(n=8), dict(n=8, caps=[16] * 4 + [3] * 4),
             dict(n=8, arrive=[0, 0, 0, 0, 3, 3, 6, 6], caps=[16] * 8)]
    geoms: dict[int, set] = {}
    for env in (None, "1"):
        if env is None:
            monkeypatch.delenv("STOCH_DECLARED_CHRONOLOGY_S", raising=False)
        else:
            monkeypatch.setenv("STOCH_DECLARED_CHRONOLOGY_S", env)
        for c in cases:
            for lp_t in ((8, 15), (8, 9), (None, None)):          # request wrapper (a declared shape, an undeclared one) and the step wrapper
                pol = sampling_policy(None, _requests_manifest(tmp_path, c["n"], arrive=c.get("arrive"), caps=c.get("caps")))
                assert sampler_geometry(pol, _l40s(), LP=lp_t[0], T=lp_t[1]) is None
                assert pol["splits"] == "derived"
                g = pol["geometry"]
                assert set(g) == DERIVED_GEOM_KEYS, sorted(g)
                assert g["sampler_batch"] == c["n"] and g["num_sms"] == 142 and g["relation"] == "SplitsFor_v1" and g["live"] == "workload-derived"
                assert g["splits_range_1_to_B"] == [{2: 32, 8: 16}[c["n"]], 32]
                assert "request" not in g and "declared_live_counts" not in g and "assumption" not in g and "experimental" not in g
                geoms.setdefault(id(c), set()).add(json.dumps(g, sort_keys=True))
    # within one workload every component (every shape, the step wrapper, either env) declares the SAME geometry: GP-01 agreement by construction
    assert all(len(v) == 1 for v in geoms.values()), {k: len(v) for k, v in geoms.items()}


@pytest.mark.parametrize("num_sms,S", [(142, 32), (132, 32), (16, 16), (6, 4)])
def test_a_single_request_takes_the_constant_S_of_its_sm_count(tmp_path, num_sms, S):
    """[single-request workloads] A workload of one request has |live(t)| = 1 at every sampling event: S = SplitsFor_v1(1, num_SMs),
    computed from the target's SM count, is a constant of its Program (`sampling["splits"]` the int; no `splits` input), for every
    wrapper shape and the step wrapper alike."""
    from verity_vllm.pipeline.build import sampler_geometry
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.frontend.vllm_meta import single_request_splits
    from verity_vllm.program.registry.topp_geometry import splits_for_v1
    target = TargetProfile.default().with_overrides(num_sms=num_sms)
    for lp_t in ((8, 15), (8, 9), (None, None)):
        pol = sampling_policy(None, _requests_manifest(tmp_path, 1))
        assert sampler_geometry(pol, target, LP=lp_t[0], T=lp_t[1]) is None
        assert pol["splits"] == S == splits_for_v1(1, num_sms) and single_request_splits(pol) == S
        g = pol["geometry"]
        assert g["sampler_batch"] == 1 and g["num_sms"] == num_sms and g["splits"] == S and g["live"] == "single request"
        assert "input" not in g and g["relation"] == "SplitsFor_v1"
    # a policy whose int disagrees with its geometry, or whose geometry is several requests', is no single-request constant
    assert single_request_splits(dict(pol, splits=S * 2 if S < 32 else 16)) is None
    assert single_request_splits(dict(pol, geometry=dict(g, sampler_batch=2))) is None
    # without a workload manifest nothing establishes one request: the derived input stays
    pol = sampling_policy(None, None)
    pol.update(kind="stochastic", seeded=True, temperature=0.8, top_p=0.95, top_k=0)
    assert sampler_geometry(pol, target) is None and pol["splits"] == "derived"


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
    # a target without num_sms cannot derive S
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
    """The wrapper takes S as an INPUT; a numeric `splits` (R15's static / schedule) is refused by name."""
    pytest.importorskip("torch")
    from verity_vllm.program.frontend.vllm_meta import make_wrapper
    with pytest.raises(ValueError, match="must be 'derived'"):
        make_wrapper("request", object(), LP=8, T=4, sampling={"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": 32})
    with pytest.raises(ValueError, match="must be 'derived'"):
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
    # [R15 stoch] the stochastic select IS bound now (GumbelTopPTokenSelect_v1{V}); the policy is a condition of reuse, S is DERIVED (M-0169)
    assert s["sampler"] == {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}
    assert s["token_select"] == SAMPLERS_BOUND["stochastic"] and "GumbelTopPTokenSelect" in s["token_select"]
    assert g != s  # a greedy artifact is not reusable for a stochastic row (check_reuse names the differing field)
    # `source` / `label` / `workload` are provenance, not applicability: the same policy from two sources is one condition
    assert applicability_conditions(prof, {}, {}, {}, {"kind": "greedy", "source": "workload", "workload": "x.json"})["sampler"] == g["sampler"]
