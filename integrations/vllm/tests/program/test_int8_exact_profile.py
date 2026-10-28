"""`TargetProfile.int8_exact`: the layers on steps.py's plain int8 steps (`linears`, `attention` or `all`), declared, applied (vLLM
quantization `verity_int8` for the linears, the attention backend in vLLM's `CUSTOM` slot for attention), read back from every
layer, and absent from every undeclared profile's serialisation (their digests are the ones the profile had before the field
existed)."""
from __future__ import annotations

import types

import pytest

from verity_vllm.program.frontend.kernel_readback import kept, routers
from verity_vllm.program.frontend.target_profile import (TargetProfile, applied_kernels, int8_exact_of, parse_target_arg,
                                                         profile_mismatches)

from tests.program.test_target_profile import _applied

#: digests of these profiles computed by the profile before `int8_exact` existed
EARLIER_DIGESTS = [
    ({}, "86c255b375c6cf028a3bc3b7deeed96fd10b520956325accb8b541fbc90026b1"),
    ({"compute_capability": (12, 0), "num_sms": 188}, "5b84e6c0db8d7e6dbcfdb6338b16411bd0c29e4d7dc5e500d461c8f01f177b72"),
    ({"compute_capability": (9, 0), "num_sms": 132, "fp8_block_gemm": "cutlass"}, "183084111c1b2591eb169793bda67cd5f7da6a5f7e4ad16ea64ff8069867fcbb"),
    ({"compute_capability": (12, 0), "num_sms": 188, "moe_backend": "triton", "ir": "boolean"},
     "f5f40794ef63b5a1f1e0eaf1695855e9ca05bc05d08a4ebfa85c400a01e2c1b7"),
]

SM120 = TargetProfile(compute_capability=(12, 0), num_sms=188)
APPLIES = {"linears": {"quantization": "verity_int8"}, "attention": {"attention_backend": "CUSTOM"},
           "all": {"quantization": "verity_int8", "attention_backend": "CUSTOM"}}


@pytest.mark.parametrize("kw, digest", EARLIER_DIGESTS)
def test_undeclared_profiles_keep_their_digests(kw, digest):
    p = TargetProfile(**kw)
    assert "int8_exact" not in p.to_json() and p.digest() == digest and TargetProfile.from_json(p.to_json()) == p
    assert "quantization" not in p.engine_args() and "int8_exact" not in applied_kernels(p, _Model()) and p.int8_parts == ()


@pytest.mark.parametrize("scope", ["linears", "attention", "all"])
def test_a_declared_scope_round_trips_and_applies_its_parts(scope):
    t = SM120.with_overrides(int8_exact=scope)
    assert t.to_json()["int8_exact"] == scope and t.digest() != SM120.digest() and TargetProfile.from_json(t.to_json()) == t
    for execution in (True, False):
        kw = t.engine_args(execution=execution)
        assert {k: kw[k] for k in ("quantization", "attention_backend") if k in kw} == APPLIES[scope]
    assert t.env() == SM120.env() and t.execution_label == SM120.execution_label
    assert parse_target_arg(f'{{"int8_exact": "{scope}"}}', base=SM120) == t


def test_refused_declarations():
    for v in ("plain", "dither", "LINEARS", "", True, 1):
        with pytest.raises(ValueError, match="int8_exact must be"):
            TargetProfile(int8_exact=v)
    for scope in ("attention", "all"):
        with pytest.raises(ValueError, match="selects the attention backend"):
            TargetProfile(int8_exact=scope, attention_backend="FLASH_ATTN")
    assert TargetProfile(int8_exact="linears", attention_backend="FLASH_ATTN").engine_args()["attention_backend"] == "FLASH_ATTN"


@pytest.mark.parametrize("scope", ["linears", "attention", "all"])
def test_the_declared_scope_must_be_what_the_layers_ran(scope):
    t = SM120.with_overrides(int8_exact=scope)
    applied = _applied(t, int8_exact=scope, attention_backend=APPLIES[scope].get("attention_backend"))
    assert profile_mismatches(t, applied) == []
    others = [g for g in ("linears", "attention", "all", "<mixed>", None) if g != scope]
    for got in others:
        assert profile_mismatches(t, {**applied, "int8_exact": got}) == [f"int8_exact declared {scope!r}, applied {got!r}"]
    assert profile_mismatches(t, {k: v for k, v in applied.items() if k != "int8_exact"}) == [
        f"int8_exact declared {scope!r}, applied '<absent>'"]
    if "attention_backend" in APPLIES[scope]:
        assert profile_mismatches(t, {**applied, "attention_backend": None}) == ["attention_backend declared 'CUSTOM', applied None"]
    assert profile_mismatches(SM120, {**_applied(SM120), "int8_exact": "<mixed>"}) == []        # undeclared: not judged


LinearMethodBase = type("LinearMethodBase", (), {})
Int8 = type("Int8PlainLinearMethod", (LinearMethodBase,), {"int8_arm": "plain"})
Unquantized = type("UnquantizedLinearMethod", (LinearMethodBase,), {})
KVCache = type("BaseKVCacheMethod", (), {})
AttentionImpl = type("AttentionImpl", (), {})
Int8Attn = type("Int8AttentionImpl", (AttentionImpl,), {"int8_arm": "plain"})
FlashAttn = type("FlashAttentionImpl", (AttentionImpl,), {})


class _Layer:
    def __init__(self, qm=None, impl=None):
        self.quant_method = qm
        if impl is not None:
            self.impl = impl


class _Model:
    def __init__(self, *layers):
        self.layers = layers

    def modules(self):
        return iter((self, *self.layers))


def test_the_scope_is_read_back_from_every_linear_and_attention_layer():
    lin, att = _Layer(Int8()), _Layer(impl=Int8Attn())
    plain_lin, plain_att = _Layer(Unquantized()), _Layer(impl=FlashAttn())
    assert int8_exact_of(_Model(lin, lin, _Layer(KVCache()), _Layer(), plain_att)) == "linears"
    assert int8_exact_of(_Model(plain_lin, att, att)) == "attention"
    assert int8_exact_of(_Model(lin, att)) == "all"
    assert int8_exact_of(_Model(plain_lin, plain_att)) is None
    assert int8_exact_of(_Model()) is None
    assert int8_exact_of(_Model(lin, plain_lin, att)) == "<mixed>"                    # one linear on another method
    assert int8_exact_of(_Model(lin, att, plain_att)) == "<mixed>"                    # one attention layer on another impl
    t = SM120.with_overrides(int8_exact="all")
    assert applied_kernels(t, _Model(lin, att)) == {"fp8_block_gemm_kernel": None, "int8_exact": "all"}


FusedMoEMethodBase = type("FusedMoEMethodBase", (), {})
Int8Experts = type("Int8ExpertsMethod", (FusedMoEMethodBase,), {"int8_arm": "plain"})
UnquantizedExperts = type("UnquantizedFusedMoEMethod", (FusedMoEMethodBase,), {})


class _Node(_Layer):
    """A module with children, as torch's: a MoE block's router (`ReplicatedLinear`, quant_config None) and its runner."""

    def __init__(self, *kids, qm=None, **attrs):
        super().__init__(qm)
        self.kids = kids
        self.__dict__.update(attrs)

    def children(self):
        return iter(self.kids)

    def modules(self):
        yield self
        for k in self.kids:
            yield from (k.modules() if isinstance(k, _Node) else (k,))


def _moe(experts_method, router_out=8, quant_config=None, held=False):
    router = _Node(qm=Unquantized(), output_size=router_out, quant_config=quant_config)
    routed = _Node(qm=experts_method, global_num_experts=8)
    runner = _Node(routed, routed_experts=routed, gate=router if held else None)
    return _Node(router, runner) if not held else _Node(runner)


def test_moe_experts_count_as_linears_and_their_routers_do_not():
    lin, att = _Layer(Int8()), _Layer(impl=Int8Attn())
    assert int8_exact_of(_Node(lin, _moe(Int8Experts()), att)) == "all"
    assert int8_exact_of(_Node(lin, _moe(Int8Experts(), held=True), att)) == "all"          # the runner holds its gate
    assert int8_exact_of(_Node(lin, _moe(UnquantizedExperts()), att)) == "<mixed>"         # bf16 experts beside int8 linears
    assert int8_exact_of(_Node(lin, _moe(Int8Experts(), router_out=16), att)) == "<mixed>"  # not the experts' logits
    assert int8_exact_of(_Node(lin, _moe(Int8Experts(), quant_config=object()), att)) == "<mixed>"  # built under the config
    block = _moe(Int8Experts())
    assert routers(_Node(block)) == {id(block.kids[0])} and routers(_Node(block.kids[0])) == set()


class _Named(_Model):
    """A model with module names, its HF config's ``quantization_config`` and a ``packed_modules_mapping``."""

    packed_modules_mapping = {"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]}   # vLLM's: no wk_weights_proj

    def __init__(self, kept_names, **named):
        super().__init__(*named.values())
        self.named = named
        self.config = types.SimpleNamespace(quantization_config={"modules_to_not_convert": kept_names})

    def named_modules(self):
        return iter([("", self), *self.named.items()])


def test_a_linear_built_without_a_quant_config_is_kept_only_when_the_checkpoint_keeps_every_shard():
    lin = _Layer(Int8())
    fused = _Node(qm=Unquantized(), quant_config=None)
    idx = "model.layers.0.self_attn.indexer"
    both = [f"{idx}.wk", f"{idx}.weights_proj"]
    model = _Named(both, **{"model.layers.0.self_attn.o_proj": lin, f"{idx}.wk_weights_proj": fused})
    assert kept(model) == {id(fused)} and int8_exact_of(model) == "linears"
    assert int8_exact_of(_Named(both[:1], **model.named)) == "<mixed>"                       # one shard not kept
    under = _Node(qm=Unquantized(), quant_config=object())
    assert int8_exact_of(_Named(both, **{"model.layers.0.self_attn.o_proj": lin, f"{idx}.wk_weights_proj": under})) == "<mixed>"


def test_the_engine_refuses_layers_that_did_not_run_the_declared_scope(monkeypatch):
    pytest.importorskip("torch")
    from verity_vllm.engine import build as B
    monkeypatch.setattr(B.torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(major=12, minor=0, multi_processor_count=188))
    monkeypatch.setattr(B, "observed_flash_attn_version", lambda model: SM120.effective_flash_attn_version())
    monkeypatch.setattr(B, "compilation_facts", lambda llm: {})

    def mismatches(scope, *layers):
        return B.target_mismatches(types.SimpleNamespace(target=SM120.with_overrides(int8_exact=scope), llm=None, model=_Model(*layers)))
    assert mismatches("linears", _Layer(Int8()), _Layer(impl=FlashAttn())) == []
    assert mismatches("all", _Layer(Int8()), _Layer(impl=Int8Attn())) == []
    assert mismatches("all", _Layer(Int8()), _Layer(impl=FlashAttn())) == ["int8_exact declared 'all', layers report 'linears'"]
    assert mismatches("linears", _Layer(Int8()), _Layer(Unquantized())) == ["int8_exact declared 'linears', layers report '<mixed>'"]
    assert mismatches("attention") == ["int8_exact declared 'attention', layers report None"]


def test_the_engine_registers_each_part_and_refuses_a_compiled_int8_engine(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import json
    import sys

    from verity_vllm import engine as engine_pkg
    from verity_vllm.engine import build as B
    from verity_vllm.engine import engine_profile as prof
    registered = []
    for mod in ("int8_vllm", "int8_attention_vllm"):
        fake = types.SimpleNamespace(register=lambda mod=mod: registered.append(mod))
        monkeypatch.setitem(sys.modules, f"verity_vllm.engine.{mod}", fake)
        monkeypatch.setattr(engine_pkg, mod, fake, raising=False)
    monkeypatch.setattr(prof, "apply_env", lambda: {})
    monkeypatch.setattr(B.engine_env, "apply_target", lambda env: {})
    cps = tmp_path / "checkpoints.json"
    cps.write_text(json.dumps({"checkpoints": [{"repo": "r/int8", "role": "INT8", "downloaded": True, "local_path": str(tmp_path)}]}))
    for scope, parts in (("linears", ["int8_vllm"]), ("attention", ["int8_attention_vllm"]), ("all", ["int8_vllm", "int8_attention_vllm"])):
        registered.clear()
        t = SM120.with_overrides(int8_exact=scope)
        kw, _, _ = B.engine_kwargs_for(cps, case="INT8", target=t)
        assert {k: kw[k] for k in ("quantization", "attention_backend") if k in kw} == APPLIES[scope]
        assert kw["enforce_eager"] is True and registered == parts
        with pytest.raises(RuntimeError, match="runs eagerly only"):
            B.engine_kwargs_for(cps, case="INT8", target=t, engine_args={"enforce_eager": False})
        with pytest.raises(RuntimeError, match="runs eagerly only"):
            B.engine_kwargs_for(cps, case="INT8", target=t.with_overrides(compilation="piecewise"))
    registered.clear()
    kw, _, _ = B.engine_kwargs_for(cps, case="INT8", target=SM120)
    assert "quantization" not in kw and "attention_backend" not in kw and registered == []
