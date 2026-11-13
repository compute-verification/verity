"""The frontend binds the seeded temperature + top-p token select: `vllm_iface.gumbel_topp_token_select(logits, seed, T, p,
pos, S)` -> `GumbelTopPTokenSelect_v2{V,S}(row, Const32[p], seed, Const64[pos], Const32[T], Const32[S])` per sampled step, the seed a
Program INPUT and S the registered split count, a constant of the Program.

Positive: the stand-in `StochasticServe` (derive_examples) derives under the vLLM ruleset / `b1-eager-v3`: one select per sampled
token (STEPS + 1), `seed` the Program's only parameter after the prompt, temperature / top_p / pos / S constants with the declared
bits, the token I32.  Negatives: temperature 0 is refused by the rule (greedy is `TokenSelect_v1`; never relabelled); a seed created
inside forward (a literal) is hidden state (`hidden_state`), never a Program input; an S that is not a launched split count is refused;
the default ruleset (no vLLM bindings) has no rule for the op (`custom_op_without_binding`).  The seed is an INPUT (same digest for
any value — bound at run time); the policy and S are STRUCTURE (a different top_p or S is a different digest).

Run (pod, torch): `PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/program/test_derive_stochastic.py`
"""
from __future__ import annotations

import struct

import pytest
import torch

from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.derive import Unsupported, derive, derive_report
from verity_vllm.program.frontend.rules import DEFAULT_RULES
from verity_vllm.program.frontend.rules.base import RuleSet
from verity_vllm.program.frontend.rules.vllm_sampling import VLLM_SAMPLING_RULES
from verity.definitions.registry import b1

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


def _inputs(LP: int):
    return (_meta(LP, dtype=torch.int32), _meta(1, dtype=torch.int64))


def _base(n) -> str:
    return getattr(n.fn, "base_id", None) or n.fn.id.split("[")[0].split("{")[0]


def _bits(x: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', x))[0]:08x}"


def _derive(C, LP, STEPS, **kw):
    with torch.device("meta"):
        m = ex.StochasticServe(C, STEPS, **kw)
    return derive_report(m, _inputs(LP), **PROF, config=C, rules=vllm_ruleset(), name="StochServe")


@pytest.mark.parametrize("S", [32, 16])
@pytest.mark.slow
def test_stochastic_serve_binds_one_select_per_sample_at_the_constant_S(S):
    """Every select is `GumbelTopPTokenSelect_v2{V,S}` (its keep word as word gates at S), its 6th operand a `Const32[S]` emitted as
    the row's 4th constant (top_p, pos, temperature, S: the fold's operand order), read by that select alone; the seed the Program's
    second parameter and no other."""
    C = _tiny_config()
    LP, STEPS = 3, 2
    rep = _derive(C, LP, STEPS, temperature=0.8, top_p=0.95, S=S)
    assert rep.ok, rep.unsupported
    fsd = rep.flat_definition
    params, _ret = fsd.signature()
    assert [n for n, _ in params][:2] == ["prompt", "seed"] and "splits" not in [n for n, _ in params], params
    assert str(params[1][1]).replace(' ', '') == "Array<1,Value<64>>", params[1]   # one int64 seed
    nodes = fsd.body.nodes
    sel = [k for k, n in enumerate(nodes) if _base(n) == "GumbelTopPTokenSelect_v2"]
    assert len(sel) == STEPS + 1 and not [n for n in nodes if _base(n).startswith("GumbelTopPTokenSelect") and _base(n) != "GumbelTopPTokenSelect_v2"]
    for e, k in enumerate(sel):
        assert nodes[k].fn.id.endswith(f",S={S}}}"), nodes[k].fn.id
        ref = nodes[k].args[5].refs[0]
        assert ref[0] == "n" and ref[1] == k - 1, (k, ref)
        assert [nodes[j].fn.id for j in range(k - 4, k)] == [f"Const32[{_bits(0.95)}]_v1", f"Const64[0x{LP - 1 + e:016x}]_v1",
                                                             f"Const32[{_bits(0.8)}]_v1", f"Const32[0x{S:08x}]_v1"]
        assert sum(1 for n in nodes for a in n.args for r in a.refs if r[0] == "n" and r[1] == k - 1) == 1   # read by this select alone
    apps = [a for a in rep.provenance["applications"] if a["rule"] == "GumbelTopPTokenSelect[vllm]_v4"]
    assert len(apps) == STEPS + 1 and all(a["preconditions_checked"]["seed"] == "Program input (runtime)" for a in apps)
    assert all(a["preconditions_checked"]["S"].startswith(f"Const32[{S}]") for a in apps)
    assert [a["preconditions_checked"]["pos"] for a in apps] == [LP - 1 + k for k in range(STEPS + 1)]
    assert rep.provenance["mutation_effects"] == []


def test_the_seed_is_an_input_policy_and_S_are_structure():
    C = _tiny_config()
    a = _derive(C, 3, 1, temperature=0.8, top_p=0.95)
    assert a.ok
    assert a.digest == _derive(C, 3, 1, temperature=0.8, top_p=0.95).digest
    assert a.digest != _derive(C, 3, 1, temperature=0.8, top_p=0.9).digest, "a different top_p is a different Program"
    assert a.digest != _derive(C, 3, 1, temperature=0.8, top_p=0.95, S=16).digest, "a different S is a different Program"


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


@pytest.mark.parametrize("S", [0, 3, 64, -32])
def test_S_is_a_launched_split_count(S):
    C = _tiny_config()
    with torch.device("meta"):
        m = ex.StochasticServe(C, 1, S=S)
    r = derive(m, _inputs(3), **PROF, config=C, rules=vllm_ruleset())
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
