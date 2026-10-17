"""`_C.fused_add_rms_norm`'s Definition is decided in one place, by the declared target (`frontend.targets.rmsnorm_fused_spec`):
the Build's vocabularies, the fold's `FusedAddRMSNorm` and the predictor all ask it.  Under `batch_invariant` it binds what they
bound before (`RMSNormFusedCuda_v2{N,EPS}`, or `RMSNormFusedCuda_v4{N}` under registered values); without it each side binds
the served width-8 kernel at the block selected from the launch row count and hidden size.  That path has no registered-epsilon twin."""

from __future__ import annotations

import dataclasses
import re

import pytest
import torch
from torch import nn

from verity.ml.circuits.defs import bind
from verity_vllm.observe.events import RunHeader
from verity_vllm.observe.fold.resolver import StepContext, Unsupported, resolve_step
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.predict.build import build_program
from verity_vllm.predict.phi import Phi
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, observe_target_profile
from verity_vllm.program.frontend.rules.vocab import PROFILES
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET
from verity_vllm.program.registry import b1
from verity_vllm.program.frontend import targets as T
from tests.observe.test_patterns_synthetic import by_spec, context, decode_step

NON_BI = DEFAULT_TARGET.with_overrides(batch_invariant=False)
GENERIC, EPS_INPUT = "RMSNormFusedCuda_v2{N=%d,EPS=1e-05}", "RMSNormFusedCuda_v4{N=%d}"
VECTOR8 = "RMSNormFusedCudaVector8_v1{N=%d,EPS=1e-05,BLOCK=%d}"
LEGACY_WIDTH8 = "RMSNormFusedCuda_v1{N=%d,EPS=1e-05}"
BUILDS = [("b1-eager", False), ("b1-eager", True), ("b1-eager-v2", False), ("b1-eager-v3", False), ("b1-eager-v3", True)]
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}


class _AddNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = ex.RMSNorm(16, 1e-5)

    def forward(self, x, res):
        return self.norm(x, res)


def _build(profile: str, registered: bool, target=None):
    with torch.device("meta"):
        m = _AddNorm()
    xs = tuple(torch.empty(2, 16, dtype=torch.bfloat16, device="meta") for _ in range(2))
    if target is not None:
        observe_target_profile(target)
    try:
        return derive_report(m, xs, mode="profile", profile=profile, name="AddNorm", registered_values=registered)
    finally:
        clear_observations()


def _built(r) -> set[str]:
    assert r.ok, r.unsupported
    return {n.fn.id for n in r.flat_definition.body.nodes}


def _fold(target, registered: bool = False):
    """One synthetic decode step (two fused norms at N=576) resolved under a run header that declares `target`."""
    events, start = decode_step()
    se = step_events(events)
    ctx = context(start, se.sample)
    h = ctx.header
    hdr = RunHeader(h.schema, h.created_utc, h.profile_id, {**h.profile, "target": target.to_json()}, h.model, h.workload, h.notes)
    prof = dataclasses.replace(ctx.profile, registered_values=True) if registered else ctx.profile
    return resolve_step(se, StepContext(hdr, ctx.start, ctx.sample, ctx.links, ctx.roots, ctx.config, prof))


def _predict(**target):
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target={"compute_capability": [8, 9], "num_sms": 128, **target},
              workload={"requests": []})
    return {re.match(r"\w+", k).group() for k in build_program(phi, "step").desc["definitions"]}


def test_the_selector_binds_the_kernel_the_target_serves():
    assert T.rmsnorm_fused_spec(DEFAULT_TARGET, 576, 1e-5).id == GENERIC % 576
    assert T.rmsnorm_fused_spec(DEFAULT_TARGET, 576, eps_input=True).id == EPS_INPUT % 576
    assert T.rmsnorm_fused_spec(NON_BI, 576, 1e-5, M=1).id == VECTOR8 % (576, 576)
    with pytest.raises(T.UnregisteredTarget, match="registered-epsilon twin"):
        T.rmsnorm_fused_spec(NON_BI, 576, eps_input=True)


def test_every_vocabulary_binds_the_role_through_the_target():
    with_role = {name: v for name, v in PROFILES.items() if name and v.has("rmsnorm_fused")}
    assert set(with_role) == {"b1-eager", "b1-eager-v2", "b1-eager-v3", "dense2-eager-v1"}
    for name, v in with_role.items():
        assert v.kind("rmsnorm_fused", N=576, EPS=1e-5, TARGET=DEFAULT_TARGET).id == GENERIC % 576, name
        assert v.kind("rmsnorm_fused", N=576, EPS=1e-5) is None, f"{name}: a rule always states the target"
        assert v.kind("rmsnorm_fused", N=576, EPS=1e-5, TARGET=NON_BI, M=1).id == VECTOR8 % (576, 576)
        if v.has("rmsnorm_fused_value"):
            assert v.kind("rmsnorm_fused_value", N=576, TARGET=DEFAULT_TARGET).id == EPS_INPUT % 576, name
            with pytest.raises(T.UnregisteredTarget, match="registered-epsilon twin"):
                v.kind("rmsnorm_fused_value", N=576, TARGET=NON_BI)


@pytest.mark.parametrize("profile,registered", BUILDS)
def test_the_build_binds_by_the_declared_target(profile, registered):
    assert _built(_build(profile, registered)) == {(EPS_INPUT if registered else GENERIC) % 16}
    result = _build(profile, registered, NON_BI).result
    if registered:
        assert result.reason == "numerics_unregistered" and "fused_add_rms_norm" in result.node
        assert "registered-epsilon twin" in result.hint
    else:
        assert _built(_build(profile, registered, NON_BI)) == {VECTOR8 % (16, 16)}


@pytest.mark.parametrize("registered", [False, True])
def test_the_fold_binds_by_the_record_target(registered):
    res = _fold(DEFAULT_TARGET, registered)
    assert not [r for r in res if isinstance(r, Unsupported)]
    assert len(by_spec(res)[(EPS_INPUT if registered else GENERIC) % 576]) == 2
    non_bi = _fold(NON_BI, registered)
    uns = [r for r in non_bi if isinstance(r, Unsupported)]
    if registered:
        assert len(uns) == 2 and all(u.op == "_C.fused_add_rms_norm.default" and "registered-epsilon twin" in u.reason for u in uns), uns
    else:
        assert not uns
        assert len(by_spec(non_bi)[VECTOR8 % (576, 576)]) == 2


def test_the_predictor_binds_by_the_declared_target():
    assert "RMSNormFusedCuda_v2" in _predict()
    non_bi = _predict(batch_invariant=False)
    assert {"RMSNormCudaVector8_v1", "RMSNormFusedCudaVector8_v1"} <= non_bi


def test_the_build_the_fold_and_the_predictor_ask_one_selector(monkeypatch):
    """Point the selector at another Definition: all three sides follow it, so none keeps a binding of its own."""
    asked = []

    def width8(profile, N, EPS=None, eps_input=False, M=1):
        asked.append(N)
        return bind(b1.RMSNormFusedCudaV1, N=N, EPS=EPS)

    monkeypatch.setattr(T, "rmsnorm_fused_spec", width8)
    assert _built(_build("b1-eager-v3", False)) == {LEGACY_WIDTH8 % 16}
    specs = by_spec(_fold(DEFAULT_TARGET))
    assert len(specs[LEGACY_WIDTH8 % 576]) == 2 and GENERIC % 576 not in specs
    names = _predict()
    assert "RMSNormFusedCuda_v1" in names and "RMSNormFusedCuda_v2" not in names
    assert {16, 576, 64} <= set(asked)
