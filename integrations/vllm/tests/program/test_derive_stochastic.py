"""The frontend binds the seeded temperature + top-p token select: `vllm_iface.gumbel_topp_token_select(logits, seed, T, p,
pos, splits)` -> `GumbelTopPTokenSelectSharedGreedy_v1{V}(row, Const32[p], seed, Const64[pos], Const32[T], splits[s])` per sampled step
(`GumbelTopPTokenSelect_v1{V}` under a declared `sampler_construction = "greedy-check-per-stage"`), the seed a Program INPUT and the split count S a request-owned Program INPUT `splits` (int32 [STEPS+1], entry s = the
derived S_t = SplitsFor_v1(|live(t)|, num_SMs) of sampling event s) -- the 6th operand, never a static of the Definition.

Positive: the stand-in `StochasticServe` (derive_examples) derives under the vLLM ruleset / `b1-eager-v3`: one composite per sampled
token (STEPS + 1), `seed` and `splits` parameters of the Program (in that order after the prompt), temperature / top_p / pos constants
with the declared bits, the token I32, each event's select reading ITS entry of `splits`.
Negatives: temperature 0 is refused by the rule (greedy is `TokenSelect_v1`; never relabelled); a seed created inside forward (a literal)
is hidden state (`hidden_state`), never a Program input; an S created inside forward (a literal / a Python int) is refused the same way
(S is the workload's derived value, not the wrapper's); the default ruleset (no vLLM bindings) has no rule for the op
(`custom_op_without_binding`).  Two seeds / two policies are two different Programs only where they should be: the seed and S are
INPUTS (same digest for any value — bound at run time); the policy is STRUCTURE (a different top_p is a different digest).

Run (pod, torch): `PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/program/test_derive_stochastic.py`
"""
from __future__ import annotations

import json
import struct

import pytest
import torch

from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.derive import Unsupported, derive, derive_report
from verity_vllm.program.frontend.rules import DEFAULT_RULES
from verity_vllm.program.frontend.rules.base import RuleSet
from verity_vllm.program.frontend.rules.vllm_bindings import clear_observations, observe_target_profile
from verity_vllm.program.frontend.rules.vllm_sampling import VLLM_SAMPLING_RULES
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET
from verity.ml.definitions.registry import b1

PROF = dict(mode="profile", profile="b1-eager-v3")


def vllm_ruleset() -> RuleSet:
    """The default rules + the sampling rule (what `vllm_bindings.vllm_ruleset()` adds for the real export; the BI norm rule of the
    full vLLM ruleset needs observed export facts the stand-in module does not have)."""
    return RuleSet("default+stoch", list(DEFAULT_RULES.rules) + list(VLLM_SAMPLING_RULES), list(DEFAULT_RULES.families))


def _tiny_config(**over):
    C = dict(b1.B0_CONFIG)
    C.update(NL=2, V=64, MAX_POS=64)
    C.update(over)
    return C


def _meta(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _inputs(LP: int, STEPS: int = 1):
    return (_meta(LP, dtype=torch.int32), _meta(1, dtype=torch.int64), _meta(STEPS + 1, dtype=torch.int32))


def _base(n) -> str:
    return getattr(n.fn, "base_id", None) or n.fn.id.split("[")[0].split("{")[0]


def _bits(x: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', x))[0]:08x}"


def _derive(C, LP, STEPS, **kw):
    with torch.device("meta"):
        m = ex.StochasticServe(C, STEPS, **kw)
    return derive_report(m, _inputs(LP, STEPS), **PROF, config=C, rules=vllm_ruleset(), name="StochServe")


def test_stochastic_serve_binds_one_select_per_sample_with_seed_and_splits_as_inputs():
    C = _tiny_config()
    LP, STEPS = 3, 2
    rep = _derive(C, LP, STEPS, temperature=0.8, top_p=0.95)
    assert rep.ok, rep.unsupported
    fsd = rep.flat_definition
    sel = [n for n in fsd.body.nodes if _base(n) == "GumbelTopPTokenSelectSharedGreedy_v1"]
    assert len(sel) == STEPS + 1
    assert all(n.fn.id == f"GumbelTopPTokenSelectSharedGreedy_v1{{V={C['V']}}}" for n in sel), sorted({n.fn.id for n in sel})
    # the seed is the second PARAMETER of the Program (a runtime input, int64 [1]) and the derived split count the third (int32 [STEPS+1]),
    # never constants
    params, _ret = fsd.signature()
    names = [n for n, _ in params]
    assert names[:3] == ["prompt", "seed", "splits"], names
    assert str(params[1][1]).replace(' ', '') == "Array<1,Value<64>>", params[1]   # one int64 seed
    assert str(params[2][1]).replace(' ', '') == f"Array<{STEPS + 1},Value<32>>", params[2]   # one int32 S per sampling event
    # each select's 6th operand is ITS entry of `splits` (event s -> splits[s]); no Const32[32] / Const32[16] stands in for S
    for s_i, n in enumerate(sel):
        op = n.args[5]
        assert op.type.leaves == 1 and str(op.type).replace(' ', '') == "Value<32>", (s_i, op.type)
        ref = op.refs[0]                       # (space, idx, leaf): the parameter space 'p', parameter 2 = splits, leaf s = event s
        assert ref[0] == "p" and ref[1] == 2 and ref[2] == s_i, (s_i, ref)
    consts_all = sorted({n.fn.id for n in fsd.body.nodes if _base(n).startswith("Const32")})
    assert not any(c in consts_all for c in ("Const32[0x00000020]_v1", "Const32[0x00000010]_v1")), consts_all
    # the policy and the row position are constants of the Program with the declared f32 / i64 bits
    consts = sorted({n.fn.id for n in fsd.body.nodes if _base(n).startswith("Const")})
    assert f"Const32[{_bits(0.8)}]_v1" in consts and f"Const32[{_bits(0.95)}]_v1" in consts, consts
    for k in range(STEPS + 1):
        assert f"Const64[0x{LP - 1 + k:016x}]_v1" in consts, (k, consts)
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "GumbelTopPTokenSelect[vllm]_v3"]
    assert len(apps) == STEPS + 1 and all(a["preconditions_checked"]["seed"] == "Program input (runtime)" for a in apps)
    assert [a["preconditions_checked"]["pos"] for a in apps] == [LP - 1 + k for k in range(STEPS + 1)]
    assert rep.provenance["mutation_effects"] == []


def test_seed_and_splits_are_inputs_policy_is_structure():
    C = _tiny_config()
    a = _derive(C, 3, 1, temperature=0.8, top_p=0.95)
    b = _derive(C, 3, 1, temperature=0.8, top_p=0.9)
    assert a.ok and b.ok
    assert a.digest != b.digest, "a different top_p is a different Program (structural input)"
    assert a.digest == _derive(C, 3, 1, temperature=0.8, top_p=0.95).digest
    # S is an INPUT: the Program is the same whatever split counts the workload derives (16 at 8 live, 32 at 1 live) -- the values are
    # bound at Match from the split kernels' constexpr and checked against SplitsFor_v1 there (no `{V,S}` static any more)
    ids = [n.fn.id for n in a.flat_definition.body.nodes]
    assert not any(i.startswith(("GumbelTopPTokenSelect", "TopPMask_v1{", "TopPMaskWordx")) and ",S=" in i for i in ids), ids
    assert f"GumbelTopPTokenSelectSharedGreedy_v1{{V={C['V']}}}" in ids


def test_a_declared_greedy_check_per_stage_binds_the_old_select():
    C = _tiny_config()
    shared = _derive(C, 3, 1, temperature=0.8, top_p=0.95)
    observe_target_profile(DEFAULT_TARGET.with_overrides(sampler_construction="greedy-check-per-stage"))
    try:
        old = _derive(C, 3, 1, temperature=0.8, top_p=0.95)
    finally:
        clear_observations()
    assert shared.ok and old.ok and shared.digest != old.digest
    ids = {n.fn.id for n in old.flat_definition.body.nodes}
    assert f"GumbelTopPTokenSelect_v1{{V={C['V']}}}" in ids and not any(i.startswith("GumbelTopPTokenSelectSharedGreedy") for i in ids)


def test_temperature_zero_is_not_a_stochastic_select():
    C = _tiny_config()
    with torch.device("meta"):
        m = ex.StochasticServe(C, 1, temperature=0.0, top_p=0.95)
    r = derive(m, _inputs(3), **PROF, config=C, rules=vllm_ruleset())
    assert isinstance(r, Unsupported), r
    assert r.reason == "custom_op_without_binding" and "greedy TokenSelect_v1" in r.hint, (r.reason, r.hint)


def test_literal_seed_is_hidden_state():
    C = _tiny_config()
    with torch.device("meta"):
        m = ex.StochasticServe(C, 1, literal_seed=True)
    r = derive(m, _inputs(3), **PROF, config=C, rules=vllm_ruleset())
    assert isinstance(r, Unsupported), r
    assert r.reason in ("hidden_state", "mutation", "custom_op_without_binding") and "lifted" in r.hint or r.reason == "hidden_state", (r.reason, r.hint)


@pytest.mark.parametrize("S", [32, 16])
def test_literal_splits_is_refused_like_a_literal_seed(S):
    """A wrapper that writes S itself (a literal tensor inside forward) is not a Program the Match can hold: S is the
    workload's derived value, read from the `splits` Input."""
    C = _tiny_config()
    with torch.device("meta"):
        m = ex.StochasticServe(C, 1, literal_splits=S)
    r = derive(m, _inputs(3), **PROF, config=C, rules=vllm_ruleset())
    assert isinstance(r, Unsupported), r
    assert r.reason in ("hidden_state", "mutation", "custom_op_without_binding"), (r.reason, r.hint)


def _derive_constant(C, LP, STEPS, S, **kw):
    with torch.device("meta"):
        m = ex.StochasticServe(C, STEPS, constant_S=S, **kw)
    return derive_report(m, _inputs(LP, STEPS)[:2], **PROF, config=C, rules=vllm_ruleset(), name="StochServe")


@pytest.mark.parametrize("S", [32, 16])
@pytest.mark.slow
def test_a_single_request_takes_S_as_a_constant(S):
    """[single-request workloads] A single-request workload's S = SplitsFor_v1(1, num_SMs) is a constant of its Program: no `splits`
    parameter, every select `GumbelTopPTokenSelect_v2{V,S}` (its keep word as word gates at S), and its 6th operand a `Const32[S]`
    emitted as the row's 4th constant (top_p, pos, temperature, S: the fold's operand order), read by that select alone."""
    C = _tiny_config()
    LP, STEPS = 3, 2
    rep = _derive_constant(C, LP, STEPS, S, temperature=0.8, top_p=0.95)
    assert rep.ok, rep.unsupported
    fsd = rep.flat_definition
    names = [n for n, _ in fsd.signature()[0]]
    assert names[:2] == ["prompt", "seed"] and "splits" not in names, names
    nodes = fsd.body.nodes
    sel = [k for k, n in enumerate(nodes) if _base(n) == "GumbelTopPTokenSelect_v2"]
    assert len(sel) == STEPS + 1 and not [n for n in nodes if _base(n) == "GumbelTopPTokenSelect_v1"]
    for e, k in enumerate(sel):
        assert nodes[k].fn.id.endswith(f",S={S}}}"), nodes[k].fn.id      # the select specialised on the constant: its keep word as word gates
        ref = nodes[k].args[5].refs[0]
        assert ref[0] == "n" and ref[1] == k - 1, (k, ref)
        assert [nodes[j].fn.id for j in range(k - 4, k)] == [f"Const32[{_bits(0.95)}]_v1", f"Const64[0x{LP - 1 + e:016x}]_v1",
                                                             f"Const32[{_bits(0.8)}]_v1", f"Const32[0x{S:08x}]_v1"]
        assert sum(1 for n in nodes for a in n.args for r in a.refs if r[0] == "n" and r[1] == k - 1) == 1   # read by this select alone
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "GumbelTopPTokenSelect[vllm]_v3"]
    assert all(a["preconditions_checked"]["S"].startswith(f"Const32[{S}]") for a in apps)
    assert rep.digest != _derive(C, LP, STEPS, temperature=0.8, top_p=0.95).digest       # another Program than the derived-S form
    assert rep.digest == _derive_constant(C, LP, STEPS, S, temperature=0.8, top_p=0.95).digest


@pytest.mark.parametrize("S", [0, 3, 64, -32])
def test_a_constant_S_is_a_launched_split_count(S):
    C = _tiny_config()
    with torch.device("meta"):
        m = ex.StochasticServe(C, 1, constant_S=S)
    r = derive(m, _inputs(3)[:2], **PROF, config=C, rules=vllm_ruleset())
    assert isinstance(r, Unsupported) and "launched split count" in r.hint, (r.reason, r.hint)


def test_default_ruleset_has_no_binding_for_the_op():
    C = _tiny_config()
    with torch.device("meta"):
        m = ex.StochasticServe(C, 1)
    r = derive(m, _inputs(3), **PROF, config=C)
    assert isinstance(r, Unsupported) and r.reason == "custom_op_without_binding", r


@pytest.mark.parametrize("bad", [dict(top_p=0.0), dict(top_p=1.5)])
def test_bad_policy_statics_are_refused(bad):
    C = _tiny_config()
    with torch.device("meta"):
        m = ex.StochasticServe(C, 1, **bad)
    r = derive(m, _inputs(3), **PROF, config=C, rules=vllm_ruleset())
    assert isinstance(r, Unsupported) and r.reason == "custom_op_without_binding", r
