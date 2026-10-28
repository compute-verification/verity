"""`verity_vllm.LLM`: the options it accepts, the pin it resolves, the engine it builds and what `generate` returns.

* an option outside the supported list, or a value of the wrong type, raises ``TypeError`` naming it and the list;
* every supported option is documented in the class docstring and the README;
* ``model`` / ``revision`` resolve to the pin in ``manifests/checkpoints.json`` (an unpinned model or another revision
  raises; a repo pinned under two roles resolves to the downloaded one);
* the engine is built by ``vllm_adapter.build_engine`` with the pin's role, and only the options the caller set reach vLLM;
* ``generate`` returns vLLM's outputs unchanged and records the call;
* ``import verity_vllm`` imports neither torch nor vLLM.
"""
from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import verity_vllm
from verity_vllm import config
from verity_vllm.config import EnginePin
from verity_vllm import llm as L

LLAMA = ("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da")


def test_the_package_exports_llm_and_what_it_is_built_from():
    from verity_vllm.engine import spec
    from verity_vllm.program.frontend import target_profile

    assert verity_vllm.__all__ == ["EnginePin", "EngineSpec", "LLM", "TargetProfile"]
    assert verity_vllm.LLM is L.LLM and verity_vllm.EnginePin is EnginePin
    assert verity_vllm.EngineSpec is spec.EngineSpec and verity_vllm.TargetProfile is target_profile.TargetProfile
    with pytest.raises(AttributeError, match="no attribute 'Engine'"):
        _ = verity_vllm.Engine


@pytest.mark.parametrize("opt", ["tokenizer", "speculative_config"])
def test_an_unsupported_option_raises_with_its_name_and_the_supported_list(opt):
    with pytest.raises(TypeError) as e:
        L.LLM(LLAMA[0], revision=LLAMA[1], **{opt: None})
    msg = str(e.value)
    assert repr(opt) in msg
    assert all(name in msg for name in L.SUPPORTED)


@pytest.mark.parametrize("opt,value", [("max_num_seqs", True), ("max_num_seqs", "8"), ("gpu_memory_utilization", "0.5"),
                                       ("enforce_eager", 1), ("tensor_parallel_size", 2.0)])
def test_a_value_of_the_wrong_type_raises(opt, value):
    with pytest.raises(TypeError, match=repr(opt)):
        L.resolve_options({opt: value})


def test_options_default_to_the_documented_values():
    assert L.resolve_options({}) == {k: s.default for k, s in L.SUPPORTED.items()}
    assert L.resolve_options({"gpu_memory_utilization": 1})["gpu_memory_utilization"] == 1
    with pytest.raises(ValueError):
        L.resolve_options({"max_num_seqs": 0})


def test_every_supported_option_is_documented():
    readme = (config.ROOT / "README.md").read_text()
    for name in L.SUPPORTED:
        assert f"``{name}``" in L.LLM.__doc__, name
        assert f"`{name}`" in readme, name


def test_the_pin_is_the_checkpoints_entry():
    pin = EnginePin.resolve(*LLAMA)
    assert (pin.model, pin.revision, pin.case, pin.world) == (*LLAMA, "LLAMA32_1B", 1)
    with pytest.raises(ValueError, match="not pinned"):
        EnginePin.resolve("nobody/unpinned", LLAMA[1])
    with pytest.raises(ValueError, match="full commit sha"):
        EnginePin.resolve(LLAMA[0], LLAMA[1][:10])


def test_a_repo_pinned_under_two_roles_resolves_to_the_downloaded_one(tmp_path):
    rev = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
    ck = tmp_path / "checkpoints.json"
    ck.write_text(json.dumps({"checkpoints": [
        {"repo": "allenai/OLMoE-1B-7B-0924", "role": "M1/M2", "revision": rev, "local_path": None, "downloaded": False},
        {"repo": "allenai/OLMoE-1B-7B-0924", "role": "OLMOE", "revision": rev, "local_path": "/snap", "downloaded": True}]}))
    pin = EnginePin.resolve("allenai/OLMoE-1B-7B-0924", rev, world=2, checkpoints=ck)
    assert (pin.case, pin.local_path, pin.world) == ("OLMOE", "/snap", 2)


class _FakeVllm:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, **kw):
        self.calls.append((prompts, sampling_params, kw))
        return [SimpleNamespace(request_id=str(i), prompt_token_ids=[1, 2],
                                outputs=[SimpleNamespace(token_ids=[7, 8, 9], finish_reason="length")]) for i, _ in enumerate(prompts)]

    def get_tokenizer(self):
        return "tok"


@pytest.fixture
def built(monkeypatch):
    from verity_vllm.engine import vllm_adapter as va
    seen = {}

    def build_engine(checkpoints, **kw):
        seen.update(kw, checkpoints=checkpoints)
        return SimpleNamespace(llm=_FakeVllm(), kwargs={"model": "/snap", "max_num_seqs": kw["max_num_seqs"]}, env_flags={"F": "1"},
                               execution=kw["execution"], effective={"enforce_eager": True}, build_s=1.5)

    monkeypatch.setattr(va, "build_engine", build_engine)
    return seen


def test_the_engine_is_built_from_the_pin_with_only_the_options_set(built):
    llm = L.LLM(*LLAMA[:1], revision=LLAMA[1])
    assert built == {"checkpoints": config.CHECKPOINTS, "max_num_seqs": 8, "case": "LLAMA32_1B", "engine_args": {},
                     "execution": "bi-eager"}
    assert llm.engine_record["pin"]["case"] == "LLAMA32_1B"
    assert llm.engine_record["env_flags"] == {"F": "1"}
    assert llm.engine_record["engine_overrides"] == {}
    overridden = L.LLM(LLAMA[0], revision=LLAMA[1], tensor_parallel_size=2, max_num_seqs=4, gpu_memory_utilization=0.6, seed=3,
                       enforce_eager=False, dtype="float16", quantization="fp8", compilation_config={"mode": 0})
    assert built["engine_args"] == {"gpu_memory_utilization": 0.6, "dtype": "float16", "quantization": "fp8",
                                    "compilation_config": {"mode": 0}, "seed": 3, "tensor_parallel_size": 2}
    assert (built["max_num_seqs"], built["execution"]) == (4, "bi-compiled")
    assert overridden.engine_record["engine_overrides"] == {
        "dtype": "float16", "quantization": "fp8", "compilation_config": {"mode": 0},
    }


def test_prefix_caching_stays_off_unless_the_caller_turns_it_on(built, monkeypatch, tmp_path):
    L.LLM(LLAMA[0], revision=LLAMA[1])
    assert "enable_prefix_caching" not in built["engine_args"]
    L.LLM(LLAMA[0], revision=LLAMA[1], enable_prefix_caching=True)
    assert built["engine_args"] == {"enable_prefix_caching": True}
    from verity_vllm.engine import build as B
    from verity_vllm.engine import engine_profile as prof
    monkeypatch.setattr(prof, "apply_env", lambda: {})
    monkeypatch.setattr(B.served, "apply", lambda execution: {})
    cps = tmp_path / "checkpoints.json"
    cps.write_text(json.dumps({"checkpoints": [{"repo": "r/m", "role": "M", "downloaded": True, "local_path": str(tmp_path)}]}))
    assert B.engine_kwargs_for(cps, case="M")[0]["enable_prefix_caching"] is False
    assert B.engine_kwargs_for(cps, case="M", engine_args={"enable_prefix_caching": True})[0]["enable_prefix_caching"] is True


def test_an_execution_label_sets_enforce_eager_and_a_contradicting_one_raises(built):
    L.LLM(LLAMA[0], revision=LLAMA[1], execution="compiled")
    assert built["execution"] == "compiled"
    assert L.resolve_options({"execution": "compiled"})["enforce_eager"] is False
    assert L.resolve_options({"execution": "eager", "enforce_eager": True})["enforce_eager"] is True
    with pytest.raises(ValueError, match="builds with enforce_eager=False"):
        L.resolve_options({"execution": "compiled", "enforce_eager": True})
    with pytest.raises(ValueError, match="not modelled"):
        L.resolve_options({"execution": "fp4"})


def test_generate_returns_the_outputs_and_records_the_call(built):
    llm = L.LLM(LLAMA[0], revision=LLAMA[1])
    outs = llm.generate(["a", "b"], "SP", use_tqdm=False)
    assert outs[1].outputs[0].token_ids == [7, 8, 9]
    assert llm._engine.llm.calls == [(["a", "b"], "SP", {"use_tqdm": False})]
    rec = llm.records[-1]
    assert rec.sampling == ["'SP'"]
    assert rec.requests[0] == {"request_id": "0", "prompt_token_ids": [1, 2], "outputs": [{"token_ids": [7, 8, 9], "finish_reason": "length"}]}
    with pytest.raises(TypeError, match="lora_request"):
        llm.generate(["a"], lora_request=object())
    assert llm.get_tokenizer() == "tok"
    assert llm.vllm is llm._engine.llm


def test_without_protocols_nothing_is_installed(built):
    llm = L.LLM(LLAMA[0], revision=LLAMA[1])
    assert llm.protocols is None and llm.engine_record["protocols"] is None
    llm.close()
    with pytest.raises(ValueError, match="protocols_config without protocols"):
        L.LLM(LLAMA[0], revision=LLAMA[1], protocols_config={"pous": {}})


def test_a_refused_set_raises_before_the_engine_is_built(built):
    with pytest.raises(ValueError, match="sampled proofs runs in the Commit stage"):
        L.LLM(LLAMA[0], revision=LLAMA[1], protocols="sampled-proofs:rows-sha512")
    with pytest.raises(ValueError, match="unknown protocol"):
        L.LLM(LLAMA[0], revision=LLAMA[1], protocols={"zk": "v1"})
    assert built == {}


@pytest.mark.parametrize("option,value", [
    ("dtype", "float16"),
    ("quantization", "fp8"),
    ("compilation_config", {"mode": 0}),
])
def test_a_build_dependent_option_refuses_kernel_profile_overrides_before_build(built, option, value):
    with pytest.raises(ValueError, match=rf"sampled-proofs:rows-sha512: engine overrides {option} .* no longer match its Build"):
        L.LLM(
            LLAMA[0],
            revision=LLAMA[1],
            protocols="sampled-proofs:rows-sha512",
            **{option: value},
        )
    assert built == {}


def test_protocols_install_on_the_built_model_and_close_uninstalls_them(built, monkeypatch):
    from verity_vllm import protocols as PO
    from verity_vllm.engine import vllm_adapter as va
    model, seen = SimpleNamespace(name="model"), []

    def install(scheme, target):
        seen.append((scheme, target.model, target.context.stage, target.context.settings_of("pous")))
        h = target.hooks(f"pous:{scheme}")
        h.patch(target.model, "decoded", True)
        return PO.Option(name=f"pous:{scheme}", hooks=h, commits={"storage_root": "c"})

    fake = SimpleNamespace(NAME="pous", schemes=lambda: ("band-chain/d12/v1",), refusals=lambda scheme, ctx: (), install=install)
    monkeypatch.setitem(sys.modules, "verity_vllm.protocols.fake_pous", fake)
    monkeypatch.setitem(PO.ADAPTERS, "pous", "fake_pous")
    real = va.build_engine
    monkeypatch.setattr(va, "build_engine", lambda *a, **kw: SimpleNamespace(**vars(real(*a, **kw)), model=model))
    llm = L.LLM(LLAMA[0], revision=LLAMA[1], protocols="pous:band-chain/d12/v1", protocols_config={"pous": {"cpu": 3}})
    assert seen == [("band-chain/d12/v1", model, "serve", {"cpu": 3})] and model.decoded is True
    assert llm.model is model
    assert llm.engine_record["protocols"]["protocols"] == [{"entry": "pous:band-chain/d12/v1", "commitment": {"storage_root": "c"}}]
    llm.close()
    assert not hasattr(model, "decoded")


def test_importing_the_package_loads_neither_torch_nor_vllm():
    code = "import sys, verity_vllm; assert not {'torch', 'vllm'} & set(sys.modules), sorted({'torch', 'vllm'} & set(sys.modules))"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=config.ROOT)
