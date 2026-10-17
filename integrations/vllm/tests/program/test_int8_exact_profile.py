"""`TargetProfile.int8_exact`: declared, applied as vLLM quantization `verity_int8`, read back from every linear layer, and absent from
every undeclared profile's serialisation (their digests are the ones the profile had before the field existed)."""
from __future__ import annotations

import types

import pytest

from verity_vllm.program.frontend.target_profile import (TargetProfile, applied_kernels, int8_exact_arm_of, parse_target_arg,
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


@pytest.mark.parametrize("kw, digest", EARLIER_DIGESTS)
def test_undeclared_profiles_keep_their_digests(kw, digest):
    p = TargetProfile(**kw)
    assert "int8_exact" not in p.to_json() and p.digest() == digest and TargetProfile.from_json(p.to_json()) == p
    assert "quantization" not in p.engine_args() and "int8_exact_arm" not in applied_kernels(p, _Model())


def test_declared_plain_round_trips_and_applies_the_quantization_method():
    t = SM120.with_overrides(int8_exact="plain")
    assert t.to_json()["int8_exact"] == "plain" and t.digest() != SM120.digest() and TargetProfile.from_json(t.to_json()) == t
    assert t.engine_args()["quantization"] == "verity_int8" == t.engine_args(execution=False)["quantization"]
    assert t.env() == SM120.env() and t.execution_label == SM120.execution_label
    assert parse_target_arg('{"int8_exact": "plain"}', base=SM120) == t
    for v in ("dither", "PLAIN", "", True, 1):
        with pytest.raises(ValueError):
            TargetProfile(int8_exact=v)


def test_declared_arm_must_be_what_every_linear_ran():
    t = SM120.with_overrides(int8_exact="plain")
    applied = _applied(t, int8_exact_arm="plain")
    assert profile_mismatches(t, applied) == []
    for got in ("<none>", "<mixed>", None):
        assert profile_mismatches(t, {**applied, "int8_exact_arm": got}) == [f"int8_exact declared 'plain', applied {got!r}"]
    assert profile_mismatches(t, {k: v for k, v in applied.items() if k != "int8_exact_arm"}) == [
        "int8_exact declared 'plain', applied '<absent>'"]
    assert profile_mismatches(SM120, {**_applied(SM120), "int8_exact_arm": "<none>"}) == []        # undeclared: not judged


LinearMethodBase = type("LinearMethodBase", (), {})
Int8 = type("Int8PlainLinearMethod", (LinearMethodBase,), {"int8_arm": "plain"})
Unquantized = type("UnquantizedLinearMethod", (LinearMethodBase,), {})
KVCache = type("BaseKVCacheMethod", (), {})


class _Layer:
    def __init__(self, qm=None):
        self.quant_method = qm


class _Model:
    def __init__(self, *layers):
        self.layers = layers

    def modules(self):
        return iter((self, *self.layers))


def test_int8_arm_is_read_back_from_every_linear_layer():
    every = _Model(_Layer(Int8()), _Layer(Int8()), _Layer(KVCache()), _Layer())
    assert int8_exact_arm_of(every) == "plain"
    assert int8_exact_arm_of(_Model(_Layer(Int8()), _Layer(Unquantized()))) == "<mixed>"     # one linear on another method
    assert int8_exact_arm_of(_Model(_Layer(Unquantized()))) == "<none>"
    assert int8_exact_arm_of(_Model(_Layer(KVCache()))) is None                             # no linear layer
    t = SM120.with_overrides(int8_exact="plain")
    assert applied_kernels(t, every) == {"fp8_block_gemm_kernel": None, "int8_exact_arm": "plain"}


def test_the_engine_refuses_a_linear_that_did_not_run_the_declared_arm(monkeypatch):
    pytest.importorskip("torch")
    from verity_vllm.engine import build as B
    t = SM120.with_overrides(int8_exact="plain")
    monkeypatch.setattr(B.torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(major=12, minor=0, multi_processor_count=188))
    monkeypatch.setattr(B, "observed_flash_attn_version", lambda model: t.effective_flash_attn_version())
    monkeypatch.setattr(B, "compilation_facts", lambda llm: {})

    def engine(*layers):
        return types.SimpleNamespace(target=t, llm=None, model=_Model(*layers))
    assert B.target_mismatches(engine(_Layer(Int8()), _Layer(Int8()))) == []
    assert B.target_mismatches(engine(_Layer(Int8()), _Layer(Unquantized()))) == ["int8_exact declared 'plain', layers report '<mixed>'"]
    assert B.target_mismatches(engine(_Layer(Unquantized()))) == ["int8_exact declared 'plain', layers report '<none>'"]
    assert B.target_mismatches(engine()) == ["int8_exact declared 'plain', layers report None"]


def test_the_engine_registers_the_method_and_refuses_a_compiled_int8_engine(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import json
    import sys

    from verity_vllm import engine as engine_pkg
    from verity_vllm.engine import build as B
    from verity_vllm.engine import engine_profile as prof
    registered = []
    fake = types.SimpleNamespace(register=lambda: registered.append(1))
    monkeypatch.setitem(sys.modules, "verity_vllm.engine.int8_vllm", fake)
    monkeypatch.setattr(engine_pkg, "int8_vllm", fake, raising=False)
    monkeypatch.setattr(prof, "apply_env", lambda: {})
    monkeypatch.setattr(B.engine_env, "apply_target", lambda env: {})
    cps = tmp_path / "checkpoints.json"
    cps.write_text(json.dumps({"checkpoints": [{"repo": "r/int8", "role": "INT8", "downloaded": True, "local_path": str(tmp_path)}]}))
    t = SM120.with_overrides(int8_exact="plain")
    kw, _, _ = B.engine_kwargs_for(cps, case="INT8", target=t)
    assert kw["quantization"] == "verity_int8" and kw["enforce_eager"] is True and registered == [1]
    with pytest.raises(RuntimeError, match="runs eagerly only"):
        B.engine_kwargs_for(cps, case="INT8", target=t, engine_args={"enforce_eager": False})
    with pytest.raises(RuntimeError, match="runs eagerly only"):
        B.engine_kwargs_for(cps, case="INT8", target=t.with_overrides(compilation="piecewise"))
    kw, _, _ = B.engine_kwargs_for(cps, case="INT8", target=SM120)
    assert "quantization" not in kw and registered == [1, 1, 1]
