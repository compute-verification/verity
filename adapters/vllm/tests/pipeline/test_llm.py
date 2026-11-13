"""`verity_vllm.LLM`: the options it accepts, the pin it resolves, the engine it builds and what `generate` returns.

* an option outside the supported list, or a value of the wrong type, raises ``TypeError`` naming it and the list;
* every supported option is documented in the class docstring and the README;
* ``model`` / ``revision`` resolve to the pin in ``manifests/checkpoints.json`` (an unpinned model or another revision
  raises; a repo pinned under two roles resolves to the downloaded one);
* the engine is built by ``vllm_adapter.build_engine`` with the pin's role, and only the options the caller set reach vLLM;
* ``generate`` returns vLLM's outputs unchanged and records the call;
* ``engine_class="G"`` builds at class G's settings, which reach ``vllm.LLM``'s keyword arguments (its ``EngineArgs``) and are
  read back from the built engine's scheduler config, under its environment, which the build sets and records; an option, a
  checkpoint or a built engine the class contradicts is refused by name, and so is a launch that states no context limit; the
  class serves at the launch's ``max_model_len``, a length or the one vLLM fits (-1), and the built engine's admission refuses a
  prompt over the engine's limit and a window closed in flight;
* ``import verity_vllm`` imports neither torch nor vLLM, and neither does taking ``target_of`` from it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import verity_vllm
from verity_vllm import config
from verity_vllm import llm as L
from verity_vllm.config import EnginePin
from verity_vllm.engine import handle
from verity_vllm.engine.profiles import engine_classes as EC
from verity_vllm.errors import Refused

LLAMA = ("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da")
GLM53 = ("zai-org/GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42")
CLASS_G_ARGS = {"tensor_parallel_size": 8, "max_num_batched_tokens": 2048, "long_prefill_token_threshold": 256,
                "enable_prefix_caching": False, "moe_backend": "triton"}
#: a class-G launch's context limit, which the class leaves to it
CTX = 401408


def test_the_package_exports_llm_and_what_it_is_built_from():
    from verity_vllm.engine import spec
    from verity_vllm.pipeline import serve_seal
    from verity_vllm.program.frontend import target_profile

    assert verity_vllm.__all__ == ["EnginePin", "EngineSpec", "LLM", "TargetProfile", "target_of"]
    assert verity_vllm.LLM is L.LLM and verity_vllm.EnginePin is EnginePin
    assert verity_vllm.EngineSpec is spec.EngineSpec and verity_vllm.TargetProfile is target_profile.TargetProfile
    assert verity_vllm.target_of is serve_seal.target_of
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
    llm_engine: SimpleNamespace

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
    seen: dict[str, Any] = {}

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
    # unloaded for the test: one imported before the engine's pin, by another test in the same worker, is refused (`require_inductor`)
    monkeypatch.delitem(sys.modules, "torch._inductor.config", raising=False)
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
    with pytest.raises(ValueError, match="SPOT runs in the Commit stage"):
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


def test_an_engine_class_fills_its_settings():
    assert L.class_options({"engine_class": "G", "max_model_len": CTX}) == {
        "engine_class": "G", "execution": "eager", "max_num_seqs": 256, "max_model_len": CTX, **CLASS_G_ARGS}
    assert L.class_options({"engine_class": "G", "max_num_seqs": 4, "seed": 3, "max_model_len": 131072}) == {
        "engine_class": "G", "execution": "eager", "max_num_seqs": 4, "seed": 3, "max_model_len": 131072, **CLASS_G_ARGS}
    assert L.class_options({"seed": 3}) == {"seed": 3}
    assert EC.max_num_seqs("G") == EC.CLASS_G_LIMITS["max_num_seqs"] == 256
    assert all(0 < v <= EC.CLASS_G_LIMITS[k] for k, v in EC.CLASS_G_SERVED.items())


@pytest.mark.parametrize("limit", [4032, 131072, CTX, 1048576, -1])
def test_an_engine_class_serves_at_the_context_limit_its_launch_states(limit):
    """The class fixes no context limit: the launch's reaches the engine args as stated, a length or -1 (vLLM's
    ``--max-model-len auto``), and neither `CLASS_G_SERVED` nor the class's options hold one."""
    assert L.class_options({"engine_class": "G", "max_model_len": limit})["max_model_len"] == limit
    assert "max_model_len" not in {**EC.CLASS_G_SERVED, **EC.options("G")}


@pytest.mark.parametrize("options,stated", [({}, None), ({"max_model_len": 0}, 0), ({"max_model_len": -2}, -2)])
def test_an_engine_class_launch_that_states_no_context_limit_is_refused(options, stated):
    with pytest.raises(Refused, match=rf"engine class G refused: max_model_len={stated}: the class serves at its launch's context "
                                      r"limit, so state one \(a length, or -1 for the longest request the engine's KV pool holds, "
                                      r"vLLM's --max-model-len auto\)"):
        L.class_options({"engine_class": "G", **options})


def test_class_g_serves_compiled_at_its_breakable_graphs():
    """Class G's compiled execution is vLLM's breakable CUDA graphs at the class's capture sizes: its options fix them, and a built
    compiled engine that ran other graphs is refused by name."""
    sizes = [1, 2, 4, *range(8, 256, 8), *range(256, 513, 16)]
    graphs = {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": sizes}
    assert L.class_options({"engine_class": "G", "execution": "compiled", "max_model_len": CTX}) == {
        "engine_class": "G", "execution": "compiled", "max_num_seqs": 256, "max_model_len": CTX, **CLASS_G_ARGS,
        "compilation_config": graphs}
    assert EC.compilation("G", "compiled") == "breakable" and EC.compilation("G", None) is None
    with pytest.raises(Refused, match="execution 'bi-compiled' \\(class G serves eager, compiled\\)"):
        EC.compilation("G", "bi-compiled")
    with pytest.raises(Refused, match="compilation_config=.*where the class serves"):
        L.class_options({"engine_class": "G", "execution": "compiled", "max_model_len": CTX,
                         "compilation_config": {"cudagraph_mode": "PIECEWISE"}})
    eff = {"tensor_parallel_size": 8, "enable_prefix_caching": False, **EC.CLASS_G_SERVED, "max_model_len": CTX,
           "enforce_eager": False, "breakable_cudagraph": True, "capture_sizes": sizes}
    assert EC.effective_refusals("G", eff, EC.env("G"), CTX) == []
    assert EC.effective_refusals("G", {**eff, "breakable_cudagraph": False, "capture_sizes": [1, 2, 4]}, EC.env("G"), CTX) == [
        "the built engine's breakable_cudagraph is False, not the class's True",
        f"the built engine's capture_sizes is [1, 2, 4], not the class's {sizes}"]
    assert EC.effective_refusals("G", {**eff, "enforce_eager": True, "breakable_cudagraph": False}, EC.env("G"), CTX) == []


def test_a_built_engine_runs_at_its_launchs_context_limit_or_at_one_vllm_fitted():
    """A launch that states a length is served at it; one that leaves the limit to vLLM (-1) at whatever positive length the
    built engine reads back, and an engine still holding -1 wasn't fitted."""
    eff = {"tensor_parallel_size": 8, "enable_prefix_caching": False, **EC.CLASS_G_SERVED, "enforce_eager": True}
    for built, launch, bad in ((CTX, CTX, []), (4032, -1, []), (1048576, -1, []),
                               (4032, CTX, [f"the built engine's max_model_len is 4032, not the launch's {CTX}"]),
                               (-1, -1, ["the built engine's max_model_len is -1, not a length vLLM fitted"]),
                               (None, -1, ["the built engine's max_model_len is None, not a length vLLM fitted"]),
                               (None, CTX, [f"the built engine's max_model_len is None, not the launch's {CTX}"])):
        assert EC.effective_refusals("G", {**eff, "max_model_len": built}, EC.env("G"), launch) == bad, (built, launch)


def test_an_option_the_class_contradicts_is_refused_by_name():
    with pytest.raises(Refused) as e:
        L.class_options({"engine_class": "G", "tensor_parallel_size": 4, "long_prefill_token_threshold": 512,
                         "enable_prefix_caching": True, "execution": "bi-eager", "attention_backend": "FLASH_ATTN", "max_num_seqs": 257})
    msg = str(e.value)
    assert msg.startswith("engine class G refused: ")
    for part in ("tensor_parallel_size=4, where the class serves 8", "long_prefill_token_threshold=512, where the class serves 256",
                 "enable_prefix_caching=True, where the class serves False", "execution='bi-eager', where the class serves 'eager'",
                 "attention_backend='FLASH_ATTN': the class serves the engine profile's",
                 "max_num_seqs=257, over the class's 256", "max_model_len=None: the class serves at its launch's context limit"):
        assert part in msg, part
    with pytest.raises(Refused, match=r"engine class 'H' is not registered \(registered: G\)"):
        L.class_options({"engine_class": "H"})
    with pytest.raises(TypeError, match="'engine_class' must be str"):
        L.class_options({"engine_class": 7})


class _Scheduler:
    """vLLM V1's scheduler as the admission reads it: its ``max_model_len``, ``add_request`` and ``requests``, from which it
    deletes a freed request."""

    def __init__(self, max_model_len=None):
        self.max_model_len = max_model_len
        self.requests = {}

    def add_request(self, request):
        self.requests[request.request_id] = request


@pytest.fixture
def built_g(monkeypatch):
    """`build_engine` stood in, with the built engine's read-back (`effective`) what vLLM's config holds of the engine args,
    overridden by ``seen["effective"]``, its ``env_flags`` the ``env`` it was given (``seen["env_flags"]`` when set), and an
    in-process scheduler at the read-back's ``max_model_len``: the launch's, or ``seen["effective"]``'s, standing in for the
    length vLLM fits where the launch states -1."""
    from verity_vllm.engine import vllm_adapter as va
    seen: dict[str, Any] = {"effective": {}}

    def build_engine(checkpoints, **kw):
        seen.update(kw)
        args = kw["engine_args"]
        effective = {"tensor_parallel_size": args.get("tensor_parallel_size", 1),
                     "enable_prefix_caching": args.get("enable_prefix_caching", False),
                     **{k: args.get(k) for k in ("max_num_batched_tokens", "max_model_len", "long_prefill_token_threshold")},
                     **seen["effective"]}
        llm = _FakeVllm()
        llm.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(engine_core=SimpleNamespace(
            scheduler=_Scheduler(effective["max_model_len"]))))
        return SimpleNamespace(llm=llm, kwargs={"max_num_seqs": kw["max_num_seqs"], **args},
                               env_flags=seen.get("env_flags", dict(kw.get("env") or {})), execution=kw["execution"],
                               effective=effective, build_s=1.0)

    monkeypatch.setattr(va, "build_engine", build_engine)
    return seen


def test_class_g_builds_at_its_settings_and_its_window_refuses_past_its_limits(built_g):
    llm = L.LLM(GLM53[0], revision=GLM53[1], engine_class="G", max_model_len=CTX)
    assert built_g["engine_args"] == {**CLASS_G_ARGS, "max_model_len": CTX}
    assert (built_g["case"], built_g["max_num_seqs"], built_g["execution"]) == ("GLM53", 256, "eager")
    assert built_g["env"] == llm.engine_record["env_flags"] == {"VERITY_DSA_TOPK": "canonical"}
    assert llm.engine_record["engine_class"] == "G" and llm.engine_record["engine_overrides"] == {"moe_backend": "triton"}
    sched = handle.scheduler(llm)
    sched.add_request(SimpleNamespace(request_id="a", num_prompt_tokens=256))
    with pytest.raises(Refused, match=r"engine class G refused: 1 requests in flight at a Commit window's close, 'a' first"):
        llm.close_window()
    with pytest.raises(Refused, match="in flight"):
        sched.add_request(SimpleNamespace(request_id="b", num_prompt_tokens=8))


@pytest.mark.parametrize("launch,fitted,limit", [(CTX, None, CTX), (131072, None, 131072), (-1, 4032, 4032), (-1, 1048576, 1048576)])
def test_class_g_refuses_a_prompt_over_its_engines_limit(built_g, launch, fitted, limit):
    """The window's limit is the engine's: the length the launch states, or under -1 the one vLLM fitted to the engine's KV pool,
    a few thousand tokens on a small pool. The launch's -1 reaches vLLM as stated."""
    built_g["effective"] = {} if fitted is None else {"max_model_len": fitted}
    llm = L.LLM(GLM53[0], revision=GLM53[1], engine_class="G", max_model_len=launch)
    assert built_g["engine_args"]["max_model_len"] == launch and llm.admission.limit == limit
    sched = handle.scheduler(llm)
    sched.add_request(SimpleNamespace(request_id="long", num_prompt_tokens=limit))
    sched.requests.clear()
    with pytest.raises(Refused, match=f"request r0: a prompt of {limit + 1} tokens, over the {limit} a class-G window admits"):
        sched.add_request(SimpleNamespace(request_id="r0", num_prompt_tokens=limit + 1))
    assert sched.requests == {}
    with pytest.raises(Refused, match=f"a prompt of {limit + 1} tokens"):
        llm.close_window()


@pytest.mark.parametrize("launch,built,fact", [
    (CTX, 131072, f"the built engine's max_model_len is 131072, not the launch's {CTX}"),
    (-1, -1, "the built engine's max_model_len is -1, not a length vLLM fitted")])
def test_class_g_refuses_a_built_engine_that_doesnt_run_at_its_launchs_context_limit(built_g, launch, built, fact):
    built_g["effective"] = {"max_model_len": built}
    with pytest.raises(Refused) as e:
        L.LLM(GLM53[0], revision=GLM53[1], engine_class="G", max_model_len=launch)
    assert str(e.value) == f"engine class G refused: {fact}"


def test_without_an_engine_class_nothing_is_held(built):
    llm = L.LLM(LLAMA[0], revision=LLAMA[1])
    assert llm.admission is None and llm.engine_record["engine_class"] is None
    llm.close_window()


def test_class_g_refuses_another_model_before_the_build(built_g):
    with pytest.raises(Refused, match=rf"engine class G refused: {LLAMA[0]}@{LLAMA[1]} is a 'llama' checkpoint, not the class's "
                                      r"'glm_moe_dsa'"):
        L.LLM(LLAMA[0], revision=LLAMA[1], engine_class="G", max_model_len=CTX)
    assert "engine_args" not in built_g


def test_class_g_refuses_a_built_engine_whose_config_doesnt_carry_its_settings(built_g):
    built_g["effective"] = {"long_prefill_token_threshold": 0, "enable_prefix_caching": None}
    with pytest.raises(Refused) as e:
        L.LLM(GLM53[0], revision=GLM53[1], engine_class="G", max_model_len=CTX)
    assert str(e.value) == ("engine class G refused: the built engine's long_prefill_token_threshold is 0, not the class's 256; "
                            "the built engine's enable_prefix_caching is None, not the class's False")


def test_class_g_refuses_a_built_engine_without_its_environment(built_g):
    """The class's environment is the engine's, not the process's: a built engine whose build didn't set the DSA top-k's canonical
    row serves a row past INDEX_TOPK in the kernels' own order, which its registration doesn't read, and is refused by name."""
    built_g["env_flags"] = {"VERITY_DSA_TOPK": "0"}
    with pytest.raises(Refused) as e:
        L.LLM(GLM53[0], revision=GLM53[1], engine_class="G", max_model_len=CTX)
    assert str(e.value) == "engine class G refused: the built engine's environment sets VERITY_DSA_TOPK='0', not the class's 'canonical'"


def test_class_g_reaches_vllms_engine_args_and_is_read_back_from_its_scheduler_config(monkeypatch, tmp_path):
    """The real `build_engine` at class G's settings: they reach ``vllm.LLM(**kw)`` (which makes its ``EngineArgs`` of them), and
    the read-back takes them from the configs the pinned vLLM makes of those args (``EngineArgs.create_engine_config``: the
    scheduler config's ``long_prefill_token_threshold`` is the arg's), here stood in.  At TP8 no in-process model runner is read.
    The class's environment is set over an operator's export, so every rank vLLM spawns inherits it, and recorded.  Inductor's
    config is unloaded for the test: another test in the same worker may have imported it before the engine's pin, which
    `engine.env.require_inductor` refuses."""
    import os

    from verity_vllm.engine import build as B
    from verity_vllm.engine import engine_profile as prof
    from verity_vllm.engine import sparse_mla, topp_pin
    monkeypatch.setenv("VERITY_DSA_TOPK", "0")
    monkeypatch.setattr(prof, "apply_env", lambda: {})
    monkeypatch.setattr(B.served, "apply", lambda execution: {})
    monkeypatch.setattr(sparse_mla, "arm", lambda: None)
    monkeypatch.setattr(topp_pin, "arm", lambda: None)
    seen = {}

    class _VllmLLM:
        def __init__(self, **kw):
            seen.update(kw)
            sc = SimpleNamespace(max_num_batched_tokens=kw["max_num_batched_tokens"], max_num_seqs=kw["max_num_seqs"],
                                 long_prefill_token_threshold=kw.get("long_prefill_token_threshold", 0), enable_chunked_prefill=True)
            vc = SimpleNamespace(scheduler_config=sc, cache_config=SimpleNamespace(enable_prefix_caching=kw["enable_prefix_caching"]),
                                 model_config=SimpleNamespace(max_model_len=kw["max_model_len"], dtype="bfloat16"),
                                 parallel_config=SimpleNamespace(tensor_parallel_size=kw.get("tensor_parallel_size", 1)))
            self.llm_engine = SimpleNamespace(vllm_config=vc, engine_core=SimpleNamespace(engine_core=SimpleNamespace(
                scheduler=_Scheduler(), model_executor=SimpleNamespace())))

    vllm, envs = ModuleType("vllm"), ModuleType("vllm.envs")
    vllm.__dict__.update(LLM=_VllmLLM, envs=envs)
    envs.__dict__["VLLM_BATCH_INVARIANT"] = False
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    monkeypatch.delitem(sys.modules, "torch._inductor.config", raising=False)
    cps = tmp_path / "checkpoints.json"
    cps.write_text(json.dumps({"checkpoints": [{"repo": "r/m", "role": "M", "downloaded": True, "local_path": str(tmp_path)}]}))
    args = {k: v for k, v in L.class_options({"engine_class": "G", "max_model_len": CTX}).items()
            if k in (*L.ENGINE_ARGS, "tensor_parallel_size")}
    assert args == {**CLASS_G_ARGS, "max_model_len": CTX}
    eng = B.build_engine(cps, case="M", max_num_seqs=EC.max_num_seqs("G"), engine_args=args, execution="eager", env=EC.env("G"))
    assert {k: seen[k] for k in args} == args and (seen["enforce_eager"], seen["max_num_seqs"]) == (True, 256)
    assert eng.runner is None and eng.model is None
    assert eng.env_flags == {"VERITY_DSA_TOPK": "canonical"} == {"VERITY_DSA_TOPK": os.environ["VERITY_DSA_TOPK"]}
    assert EC.effective_refusals("G", eng.effective, eng.env_flags, CTX) == []
    assert eng.effective["long_prefill_token_threshold"] == 256 and eng.effective["batch_invariant"] is False
    target = SimpleNamespace(env=lambda: {}, engine_args=lambda: {})
    with pytest.raises(RuntimeError, match="at tensor_parallel_size 8 the model is in the ranks' processes: declare none"):
        B.build_engine(cps, case="M", engine_args=args, execution="eager", target=target)


@pytest.mark.parametrize("statement", ["import verity_vllm", "from verity_vllm import target_of"])
def test_importing_the_package_loads_neither_torch_nor_vllm(statement):
    code = f"import sys; {statement}; assert not {{'torch', 'vllm'}} & set(sys.modules), sorted({{'torch', 'vllm'}} & set(sys.modules))"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=config.ROOT)
