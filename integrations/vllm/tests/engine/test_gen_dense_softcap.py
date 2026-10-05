"""[r8 lane E-cleanup 2026-09-18] `AttentionScopeSoftcap` (verity_vllm/engine/profiles/gen_dense_softcap_patterns.py): the FA2 scope of a
softcapped launch resolves to `AttentionSoftcap_v1{T,NH,KVH,D,BN,CAP}` with the base pattern's operands; softcap 0 / None is untouched;
a windowed + softcapped launch (Gemma-2 even layers) goes through the inert-window rule; bad softcaps are Unsupported with a reason."""
from __future__ import annotations

import importlib

import dataclasses
import math

import pytest

from verity_vllm.observe.fold.patterns import AttentionScope
from verity_vllm.engine.profiles import vllm_d9105ea80_sm89_eager as B1PROF
from verity_vllm.engine.profiles.gen_dense2_patterns import AttentionScopeInertWindow, with_dense2_patterns
from verity_vllm.engine.profiles.gen_dense_softcap_patterns import AttentionScopeSoftcap, with_softcap_patterns

gemma = importlib.import_module("verity_vllm.engine.profiles.derived_GEMMA2_2B")
from verity_vllm.observe.fold.patterns.target import target_of
from verity_vllm.observe.fold.resolver import Compute, Unsupported, resolve_step
from tests.engine.test_gen_dense2 import FA2, SCOPE, _fa2_edit
from tests.observe.test_patterns_synthetic import D, KVH, NH, _context_on_host, by_spec, context, decode_step
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.program.frontend.rules.vocab import vocabulary_for
from verity_vllm.program.frontend import targets as TG
from verity_vllm.program.registry.quarantine.dense import softcap as SC

BASE = "Attention_v3{T=33,NH=9,KVH=3,D=64,BN=128}"
CAPPED = "AttentionSoftcap_v1{T=33,NH=9,KVH=3,D=64,BN=128,CAP=50.0}"


def _ctx(start, sample, dense2=True):
    ctx = context(start, sample)
    ps = with_dense2_patterns(ctx.profile.patterns, ctx.config) if dense2 else list(ctx.profile.patterns)
    prof = dataclasses.replace(ctx.profile, patterns=with_softcap_patterns(ps))
    return dataclasses.replace(ctx, profile=prof)


def _resolve(dense2=True, **fa2_edits):
    events, start = decode_step()
    se = step_events(_fa2_edit(events, **fa2_edits) if fa2_edits else events)
    return se, resolve_step(se, _ctx(start, se.sample, dense2=dense2))


def test_softcap_scope_is_inserted_ahead_of_the_attention_scopes_it_refines():
    ps = with_softcap_patterns(with_dense2_patterns(B1PROF.PROFILE.patterns, B1PROF.C))
    kinds = [type(p) for p in ps]
    i, j, k = kinds.index(AttentionScopeSoftcap), kinds.index(AttentionScopeInertWindow), kinds.index(AttentionScope)
    assert i == j - 1 == k - 2 and kinds.count(AttentionScopeSoftcap) == 1
    assert ps[i].bn == ps[k].bn and ps[i].num_splits == ps[k].num_splits and ps[i].name == "attention.fa2_varlen_fwd.softcap"
    # without the dense2 patterns the softcap scope sits directly ahead of the base scope
    ps2 = with_softcap_patterns(B1PROF.PROFILE.patterns)
    kinds2 = [type(p) for p in ps2]
    assert kinds2.index(AttentionScopeSoftcap) == kinds2.index(AttentionScope) - 1 and len(ps2) == len(B1PROF.PROFILE.patterns) + 1


def test_gemma2_profile_registers_the_softcap_scope_first():
    names = [p.name for p in gemma.PROFILE.patterns]
    i = names.index("attention.fa2_varlen_fwd.softcap")
    assert names[i + 1] == "attention.fa2_varlen_fwd.inert_window" and names[i + 2] == "attention.fa2_varlen_fwd"
    assert gemma.PROFILE.notes["softcap_pattern"] == ["attention.fa2_varlen_fwd.softcap"]
    assert gemma.C["attn_logit_softcapping"] == 50.0 and gemma.C["BN"] == 64


def test_no_softcap_is_left_to_the_base_pattern():
    se, res = _resolve()
    assert not [r for r in res if isinstance(r, Unsupported)]
    assert BASE in by_spec(res) and not [s for s in by_spec(res) if s.startswith("AttentionSoftcap")]
    (c,) = by_spec(res)[BASE]
    assert c.pattern == "attention.fa2_varlen_fwd"


def test_softcapped_launch_resolves_to_attention_softcap_with_the_base_operands():
    se0, base = _resolve()
    se, res = _resolve(softcap=50.0)
    assert not [r for r in res if isinstance(r, Unsupported)]
    assert CAPPED in by_spec(res) and BASE not in by_spec(res)
    (c,), (c0,) = by_spec(res)[CAPPED], by_spec(base)[BASE]
    assert (c.operands, c.outputs, c.rows, c.attribution, c.consumed, c.kernels) == (c0.operands, c0.outputs, c0.rows, c0.attribution, c0.consumed, c0.kernels)
    assert c.pattern == "attention.fa2_varlen_fwd.softcap" and c.spec.bindings["CAP"] == 50.0
    assert set(c.consumed) == next(n for n in se.roots if n.op == SCOPE).ids()
    # the rest of the step is unchanged by the more specific scope (same specs apart from the attention body)
    others = lambda d: sorted(s for s in d if not s.startswith(("Attention_v3", "AttentionSoftcap")))
    assert others(by_spec(res)) == others(by_spec(base))


def test_softcap_definition_is_the_quarantined_one_and_builds_a_body():
    se, res = _resolve(softcap=50.0)
    (c,) = by_spec(res)[CAPPED]
    assert c.spec.defn is SC.AttentionSoftcap and SC.SEMANTIC_ID == "dense/AttentionSoftcap@v1"
    assert "quarantine" in (c.spec.defn.conformance or "")
    # CAP is a static of the body: the two constants differ from the base body's
    assert SC.softcap_mul_bits(64, 50.0) != SC.softcap_scale_log2_bits(50.0)


@pytest.mark.parametrize("left", [4095, 33])       # Gemma-2 (4096-1) and the boundary left == max_seqlen_k
def test_softcap_with_an_inert_window_goes_through_the_inert_window_rule(left):
    se, res = _resolve(softcap=50.0, window_size_left=left, window_size_right=0)
    assert not [r for r in res if isinstance(r, Unsupported)], [r.reason for r in res if isinstance(r, Unsupported)]
    (c,) = by_spec(res)[CAPPED]
    assert c.pattern == "attention.fa2_varlen_fwd.softcap"


def test_softcap_with_a_real_local_window_is_still_unsupported():
    se, res = _resolve(softcap=50.0, window_size_left=16, window_size_right=0)     # 16 < max_seqlen_k 33
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and uns[0].op == SCOPE and "sliding window" in uns[0].reason
    assert not [s for s in by_spec(res) if s.startswith("AttentionSoftcap")]


def test_softcap_without_dense2_patterns_still_resolves_the_windowless_launch():
    se, res = _resolve(dense2=False, softcap=30.0)
    assert not [r for r in res if isinstance(r, Unsupported)]
    assert "AttentionSoftcap_v1{T=33,NH=9,KVH=3,D=64,BN=128,CAP=30.0}" in by_spec(res)


@pytest.mark.parametrize("cap", [-50.0, float("nan"), float("inf")])
def test_bad_softcaps_are_unsupported_with_the_value_recorded(cap):
    se, res = _resolve(softcap=cap)
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and uns[0].op == SCOPE and "softcap" in uns[0].reason
    got = uns[0].detail["softcap"]
    assert got == cap or (math.isnan(cap) and math.isnan(got))


def test_base_facts_are_still_checked_under_a_softcap():
    se, res = _resolve(softcap=50.0, softmax_scale=0.5)         # D=64: the pinned scale is 0.125
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and "softmax_scale" in uns[0].reason and uns[0].detail.get("softcap") == 50.0
    se, res = _resolve(softcap=50.0, is_causal=False)
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and "causal" in uns[0].reason


def _resolve_on_host(cc, sms, registered_values=False, fa=None, **fa2_edits):
    """The softcap profile's resolution of the synthetic step on a host of capability `cc`; `fa` = the FA version its attention layers
    reported (a forced version)."""
    events, start = decode_step(num_sms=sms)
    se = step_events(_fa2_edit(events, **fa2_edits))
    base = _context_on_host(start, se.sample, sms, cc)
    if fa is not None:
        doc = {**base.header.profile, "backends": {**(base.header.profile.get("backends") or {}), "attention_flash_attn_version": fa}}
        base = dataclasses.replace(base, header=dataclasses.replace(base.header, profile=doc))
    prof = dataclasses.replace(base.profile, patterns=with_softcap_patterns(with_dense2_patterns(base.profile.patterns, base.config)),
                               registered_values=registered_values)
    ctx = dataclasses.replace(base, profile=prof)
    return ctx, resolve_step(se, ctx)


def test_on_the_rtx_pro_6000_a_softcapped_launch_is_the_builders_check_inf_softcap_chain():
    """On cc 12.0 (FA2's per-iteration `Check_inf`) the base chain is `Attention_v5`; the softcapped launch is `AttentionSoftcap_v2` with
    the same statics and CAP, the Definition the builder's `attention_dot` kind binds there, on the base pattern's operands."""
    ctx0, base = _resolve_on_host((12, 0), 188)
    ctx, res = _resolve_on_host((12, 0), 188, softcap=50.0)
    assert not [r for r in res if isinstance(r, Unsupported)], [r.reason for r in res if isinstance(r, Unsupported)]
    (c0,) = [c for k, v in by_spec(base).items() if k.startswith("Attention_v5{") for c in v]
    (c,) = [c for k, v in by_spec(res).items() if k.startswith("AttentionSoftcap") for c in v]
    tp = target_of(ctx)[0]
    geom = TG.attention_geometry(tp, D, 1, NH, KVH)
    built = vocabulary_for("profile", tp.numerics_profile).kind(
        "attention_dot", T=33, NH=NH, KVH=KVH, D=D, BN=geom["BN"], DOT=geom["DOT"], INV=geom["INV"],
        FA2_MASKED_FROM=TG.fa2_masked_from(tp, 33, D, 1, 0, 1), CAP=50.0)
    assert c.spec is built and built.base_id == "AttentionSoftcap_v2"
    assert c.spec is TG.attention_spec(tp, 33, NH, KVH, D, 1, row=0, own_rows=1, cap=50.0)
    assert {k: v for k, v in c.spec.bindings.items() if k != "CAP"} == c0.spec.bindings
    assert (c.operands, c.outputs, c.rows, c.attribution, c.consumed) == (c0.operands, c0.outputs, c0.rows, c0.attribution, c0.consumed)


@pytest.mark.parametrize("cc,sms,registered_values,fa", [((8, 9), 142, True, None), ((12, 0), 188, True, None), ((9, 0), 132, False, 2)])
def test_a_softcapped_launch_without_a_registered_softcap_chain_is_unsupported_by_name(cc, sms, registered_values, fa):
    """No softcapped twin of the target's chain is folded (the value-input chains `Attention_v10` / `_v11`, forced FA2's `Attention_v2` on
    Hopper): Unsupported by name, never the plain chain."""
    _, res = _resolve_on_host(cc, sms, registered_values=registered_values, fa=fa, softcap=50.0)
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and uns[0].op == SCOPE and "softcap" in uns[0].reason, [(u.op, u.reason) for u in uns]
    assert not [k for k in by_spec(res) if k.startswith("Attention")]


def test_synthetic_fa2_launch_is_the_varlen_fwd_schema():
    events, _ = decode_step()
    from verity_vllm.observe.events import DispatchEnter
    fa = next(e for e in events if isinstance(e, DispatchEnter) and e.op == FA2)
    assert "float softcap" in fa.schema
