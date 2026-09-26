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
from verity_vllm.pipeline import llm as L

LLAMA = ("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da")


def test_the_package_exports_llm():
    assert verity_vllm.LLM is L.LLM
    assert "LLM" in verity_vllm.__all__


@pytest.mark.parametrize("opt", ["dtype", "tokenizer", "enable_prefix_caching", "quantization"])
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
    L.LLM(LLAMA[0], revision=LLAMA[1], tensor_parallel_size=2, max_num_seqs=4, gpu_memory_utilization=0.6, seed=3,
          enforce_eager=False)
    assert built["engine_args"] == {"gpu_memory_utilization": 0.6, "seed": 3, "tensor_parallel_size": 2}
    assert (built["max_num_seqs"], built["execution"]) == (4, "bi-compiled")


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


def test_importing_the_package_loads_neither_torch_nor_vllm():
    code = "import sys, verity_vllm; assert not {'torch', 'vllm'} & set(sys.modules), sorted({'torch', 'vllm'} & set(sys.modules))"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=config.ROOT)
