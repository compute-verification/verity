"""`tools/advice_compile.py --window K`'s inputs: the certified bodies are window K's own requests', so a user the run served
again in a later window keeps the prompt it had in K; K's carry is the close of the last window before it that ran steps; and the
default seed is the public one (`engine.served_seed`), whatever the run's server says it filled in. The registration's φ states
the checkpoint dtype its pinned config's `quantization_config` names, so an FP8 GLM-5.3 reaches `predict.dsa`'s decoder."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.pipeline.test_window_advice import W1, W2
from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
from verity_vllm.engine.served_seed import DEFAULT_SEED, derived
from verity_vllm.pipeline import window_advice as WA

SCRIPT = Path(__file__).resolve().parents[5] / "adapters" / "vllm" / "tools" / "advice_compile.py"
_spec = importlib.util.spec_from_file_location("advice_compile", SCRIPT)
assert _spec is not None and _spec.loader is not None
AC = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("advice_compile", AC)
_spec.loader.exec_module(AC)


def _run(tmp_path: Path, requests: list[dict]) -> Path:
    (tmp_path / "serve.json").write_text(json.dumps({"model": "m", "revision": "r", "target": {}, "default_seed": 7}))
    for k in (0, 3):
        (tmp_path / "advice" / f"w{k}").mkdir(parents=True)
        (tmp_path / "advice" / f"w{k}" / "registration.json").write_text(json.dumps({"window": k}))
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(q) + "\n" for q in requests))
    return tmp_path


def _q(user: str, windows: list[int], prompt: list[int], max_tokens: int) -> dict:
    return {"user": user, "http_id": f"h-{user}-{windows}", "windows": windows, "prompt_ids": prompt,
            "ingress": {"brought": {"max_tokens": max_tokens}}}


def test_a_window_reads_its_own_requests_when_a_user_is_served_again(tmp_path):
    run = _run(tmp_path, [_q("u", [0], [1, 2, 3], 8), _q("v", [0, 1], [9], 4), _q("u", [3], [4, 5], 16)])
    _, meta, bodies, reg = AC.served_inputs(run, 0)
    assert bodies == {"u": {"max_tokens": 8, "prompt": [1, 2, 3]}, "v": {"max_tokens": 4, "prompt": [9]}}
    assert reg == {"window": 0} and meta["default_seed"] == 7
    _, _, bodies, _ = AC.served_inputs(run, 3)
    assert bodies == {"u": {"max_tokens": 16, "prompt": [4, 5]}}


def _served(out: Path, **kw: Any) -> tuple[Path, dict, WA.Advice]:
    """One unseeded stochastic request from user ``q0`` served in serve-openai's window 0 (`test_serve_openai.Serving`, ``kw``
    its options), with the `_run` stubs beside its records: the run, the request's record and its window's advice, made from
    the rows the window committed, every output released in bucket 0."""
    from tests.pipeline.test_serve_openai import Serving, records

    with Serving(out, **kw) as s:
        assert s.post("/v1/completions", {"model": "m", "prompt": [1, 2, 3, 4, 5], "max_tokens": 3, "temperature": 0.8,
                                          "user": "q0", "stream": True})[0] == 200
    (rec,) = records(out, "requests.jsonl")
    _run(out, [])
    (out / "requests.jsonl").write_text(json.dumps(rec) + "\n")
    T = len(rec["streamed_ids"])
    steps = tuple(WA.StepAdvice((WA.Row(0, kind, start, hi - lo),)) for _w, _s, lo, hi, start, _lp, kind in rec["rows"])
    return out, rec, WA.Advice((WA.RequestAdvice("q0", len(rec["prompt_ids"]), T, "length"),), steps, (0,) * T)


@pytest.mark.usefixtures("code_versions_stood_in")
def test_an_unseeded_stochastic_request_served_in_a_window_compiles_under_its_windows_coin_to_the_seed_it_committed(tmp_path):
    """Serving derives the seed from the window's seed coin and the body's ``user``, the window's line names the coin, and
    the compile given that coin as the window's seed salt (`window_coin`) and the certified body derives the seed the window
    committed. Another coin gives another seed for the same id, the rest of the body none; a window whose line names no coin (a
    server that filled in `DEFAULT_SEED`) is refused by name, and a seed of the engine's own is not the one the compile derives.
    The compile is read at its plan (`plan_steps`: `compile_steps` up to building Programs), which settles each request's seed
    and refuses a request that has none; the test below builds the Program."""
    from tests.program.predict.test_steps import _registered
    from verity_vllm.program.predict.build import Unexpressible
    from verity_vllm.program.predict.steps import plan_steps

    (tmp_path / "ok").mkdir()
    run, rec, advice = _served(tmp_path / "ok")
    _, _, bodies, _ = AC.served_inputs(run, 0)
    coin = AC.window_coin(run, 0)
    assert "seed" not in bodies["q0"] and coin["source"] == "in-process" and rec["ingress"]["seed_rule"] == "derived"
    reg, key = _registered(), bytes.fromhex(coin["coin"])
    assert plan_steps(reg, advice, bodies, salt=key).requests[0].seed == rec["committed_seeds"][0] == derived(key, "q0")
    other = plan_steps(reg, advice, bodies, salt=bytes(64) + b"\x01").requests[0].seed
    assert other == derived(bytes(64) + b"\x01", "q0") != rec["committed_seeds"][0]
    body = {**bodies["q0"], "temperature": 0.5, "max_tokens": 3, "prompt": [9, 8, 7, 6, 5]}
    assert plan_steps(reg, advice, {"q0": body}, salt=key).requests[0].seed == rec["committed_seeds"][0]
    lines = [json.loads(x) for x in (run / "windows.jsonl").read_text().splitlines()]
    (run / "windows.jsonl").write_text("".join(json.dumps({k: v for k, v in w.items() if k != "seed_coin"}) + "\n" for w in lines))
    assert AC.window_coin(run, 0) is None
    with pytest.raises(Unexpressible, match=r"request 0 \('q0'\): stochastic with no seed in its certified body, and the compile has "
                                            r"no window seed salt to derive one from: a seed serving fills in "
                                            r"\(`served_seed.DEFAULT_SEED`\)"):
        plan_steps(reg, advice, bodies, salt=None)
    (tmp_path / "own").mkdir()
    run, rec, advice = _served(tmp_path / "own", seed_override=7)
    key = bytes.fromhex(AC.window_coin(run, 0)["coin"])
    assert rec["committed_seeds"] == [7] and not rec["check"]["passed"]
    assert plan_steps(reg, advice, AC.served_inputs(run, 0)[2], salt=key).requests[0].seed == derived(key, "q0") != 7


@pytest.mark.slow
@pytest.mark.usefixtures("code_versions_stood_in")
def test_the_session_compiled_under_the_windows_coin_carries_the_seed_the_window_committed(tmp_path):
    """`compile_steps` on each input the test above plans and accepts builds the request's Program, runs `check_compiled` on
    it and hands on its plan's seed: the window's coin, another coin, the body with another temperature, length and prompt,
    and a window whose engine used a seed of its own. Each Program is of the window's one request, whose `seed` Input takes
    the seed its plan derived."""
    from tests.program.predict.test_steps import _registered
    from verity_vllm.program.predict.steps import compile_steps

    (tmp_path / "ok").mkdir()
    run, rec, advice = _served(tmp_path / "ok")
    key, bodies, other = bytes.fromhex(AC.window_coin(run, 0)["coin"]), AC.served_inputs(run, 0)[2], bytes(64) + b"\x01"
    body = {**bodies["q0"], "temperature": 0.5, "max_tokens": 3, "prompt": [9, 8, 7, 6, 5]}
    (tmp_path / "own").mkdir()
    own, _, own_advice = _served(tmp_path / "own", seed_override=7)
    own_key = bytes.fromhex(AC.window_coin(own, 0)["coin"])
    reg = _registered()
    for adv, ingress, salt, seed in ((advice, bodies, key, rec["committed_seeds"][0]), (advice, bodies, other, derived(other, "q0")),
                                     (advice, {"q0": body}, key, rec["committed_seeds"][0]),
                                     (own_advice, AC.served_inputs(own, 0)[2], own_key, derived(own_key, "q0"))):
        compiled = compile_steps(reg, adv, ingress, salt=salt)
        (unit,) = compiled.requests
        assert unit.seed == seed and set(compiled.programs) == {unit.digest}
    assert rec["committed_seeds"][0] == derived(key, "q0") != 7


def test_the_registered_default_seed_is_the_public_one_and_a_run_that_used_another_is_refused():
    assert AC.registered_seed({"default_seed": DEFAULT_SEED}) == DEFAULT_SEED
    for seed in (7, 0, None, str(DEFAULT_SEED), float(DEFAULT_SEED)):
        why = f"filled in default seed {seed!r}, and the registered one is engine.served_seed."
        with pytest.raises(ValueError, match=re.escape(why)):
            AC.registered_seed({"default_seed": seed})
    with pytest.raises(ValueError, match="filled in default seed None"):
        AC.registered_seed({})


def test_the_registration_takes_the_engines_world_at_the_rank_and_the_engine_args_class_g_states():
    kw = {"tensor_parallel_size": 8, "max_model_len": 2048, "max_num_batched_tokens": 2048, "long_prefill_token_threshold": 256,
          "max_num_seqs": 256, "enable_prefix_caching": False, "gpu_memory_utilization": 0.9, "seed": 0}
    assert AC.engine_of({"engine_record": {"engine_kwargs": kw}}, 3) == (
        {"world": 8, "rank": 3}, {"max_model_len": 2048, "max_num_batched_tokens": 2048, "long_prefill_token_threshold": 256,
                                  "max_num_seqs": 256, "enable_prefix_caching": False})
    assert AC.engine_of({}, 0) == ({"world": 1, "rank": 0}, {})
    with pytest.raises(ValueError, match="rank 8: the run's engine has tensor-parallel world 8"):
        AC.engine_of({"engine_record": {"engine_kwargs": kw}}, 8)


def _launched(**engine_kwargs: Any) -> dict:
    return {"model": "m", "revision": "r", "target": {}, "execution": "eager",
            "engine_record": {"engine_kwargs": {"tensor_parallel_size": 8, "max_model_len": 2048, **engine_kwargs}}}


def _kv_refused(cache: str, model_type: str | None, cc: tuple[int, ...] | None) -> str:
    return (f"engine kv_cache_dtype={cache!r} for model_type {model_type!r} on cc {cc!r}: a request Program models the KV cache of a "
            "launch at kv_cache_dtype auto or bfloat16, and at kv_cache_dtype 'fp8' only glm_moe_dsa's on cc 9.0 (its layout "
            "fp8_ds_mla), so the compile of this launch would be Programs it did not run. Serve with kv_cache_dtype at vLLM's "
            "default to compile it")


@pytest.mark.parametrize("cache", ["fp8", "fp8_e4m3", "fp8_ds_mla"])
def test_a_launch_that_names_another_kv_cache_format_is_refused_by_its_setting(cache):
    """φ states an FP8 cache only as the layout of the one model and target it is stated for, so a launch at another format,
    or at that one with no model and target to state it for, would compile to the Programs of a launch at the default, which
    it did not run: the registration refuses it by the setting. The default, by either name or left out, compiles as before."""
    why = _kv_refused(cache, None, None)
    with pytest.raises(ValueError, match=re.escape(why)):
        AC.engine_of(_launched(kv_cache_dtype=cache), 0)
    with pytest.raises(ValueError, match=re.escape(why)):
        AC.phi_of(_launched(kv_cache_dtype=cache), {}, 0)
    for stated in ({}, {"kv_cache_dtype": None}, {"kv_cache_dtype": "auto"}, {"kv_cache_dtype": "bfloat16"}):
        assert AC.engine_of(_launched(**stated), 0) == ({"world": 8, "rank": 0}, {"max_model_len": 2048})
        assert AC.phi_of(_launched(**stated), {}, 0).engine == {"max_model_len": 2048}


MTP5 = {"method": "mtp", "num_speculative_tokens": 5}
_AT = "method='mtp' and num_speculative_tokens=5"


def _spec_refused(name: str, value: Any) -> str:
    return (f"engine {name}={value!r}: a request Program states speculative decoding only at {_AT}, so the compile of this launch "
            f"would be Programs it did not run. Serve at {_AT}, or with speculative decoding off, to compile it")


@pytest.mark.parametrize("setting", [{"speculative_config": dict(MTP5)}, {"speculative_config": {**MTP5, "model": None}},
                                     {"spec_method": "mtp", "spec_tokens": 5}, {"spec_method": "mtp", "spec_tokens": 5, "spec_model": None}])
def test_a_launch_that_drafts_five_tokens_by_mtp_states_it_in_phi_in_either_form(setting):
    """The model's own MTP module drafting 5 tokens a row, in the config's form or vLLM's short ones, is the one speculative
    decoding φ states, as its engine's ``speculative``; its digest is not the undrafted launch's, which is as before."""
    tp, engine = AC.engine_of(_launched(**setting), 0)
    assert (tp, engine) == ({"world": 8, "rank": 0}, {"max_model_len": 2048, "speculative": MTP5})
    drafted, plain = AC.phi_of(_launched(**setting), {}, 0), AC.phi_of(_launched(), {}, 0)
    assert drafted.engine == engine and plain.engine == {"max_model_len": 2048} and drafted.digest() != plain.digest()
    off = dict.fromkeys(("speculative_config", "spec_method", "spec_model", "spec_tokens"))
    assert AC.phi_of(_launched(**off), {}, 0).digest() == plain.digest()


@pytest.mark.parametrize("setting,why", [
    ({"speculative_config": {"method": "ngram", "num_speculative_tokens": 5}}, _spec_refused("speculative_config.method", "ngram")),
    ({"speculative_config": {"method": "eagle3", "num_speculative_tokens": 5}}, _spec_refused("speculative_config.method", "eagle3")),
    ({"speculative_config": {"method": "mtp", "num_speculative_tokens": 3}},
     _spec_refused("speculative_config.num_speculative_tokens", 3)),
    ({"speculative_config": {"method": "mtp", "num_speculative_tokens": 5.0}},
     _spec_refused("speculative_config.num_speculative_tokens", 5.0)),
    ({"speculative_config": {**MTP5, "model": "a/draft"}}, _spec_refused("speculative_config.model", "a/draft")),
    ({"speculative_config": {**MTP5, "draft_sample_method": "gumbel"}}, _spec_refused("speculative_config.draft_sample_method", "gumbel")),
    ({"speculative_config": {**MTP5, "disable_padded_drafter_batch": True}},
     _spec_refused("speculative_config.disable_padded_drafter_batch", True)),
    ({"speculative_config": {**MTP5, "num_speculative_tokens_per_batch_size": [[1, 8, 3]]}},
     _spec_refused("speculative_config.num_speculative_tokens_per_batch_size", [[1, 8, 3]])),
    ({"spec_method": "ngram", "spec_tokens": 5}, _spec_refused("spec_method", "ngram")),
    ({"spec_method": "mtp", "spec_tokens": 2}, _spec_refused("spec_tokens", 2)),
    ({"spec_method": "mtp", "spec_tokens": 5, "spec_model": "draft"}, _spec_refused("spec_model", "draft")),
    ({"spec_model": "draft"}, _spec_refused("spec_model", "draft")),
    ({"speculative_config": {}}, f"engine speculative decoding {{}} leaves method unset: a request Program states it only at {_AT}. "
                                 f"Serve at {_AT}, or with speculative decoding off, to compile it"),
    ({"spec_method": "mtp"}, f"engine speculative decoding {{'method': 'mtp'}} leaves num_speculative_tokens unset: a request Program "
                             f"states it only at {_AT}. Serve at {_AT}, or with speculative decoding off, to compile it"),
    ({"spec_tokens": 5}, f"engine speculative decoding {{'num_speculative_tokens': 5}} leaves method unset: a request Program states "
                         f"it only at {_AT}. Serve at {_AT}, or with speculative decoding off, to compile it"),
    ({"speculative_config": dict(MTP5), "spec_tokens": 5}, f"engine speculative_config={MTP5!r} beside spec_tokens=5: the launch states "
                                                           "its speculative decoding twice. Serve with one of the two to compile it"),
    ({"speculative_config": '{"method": "mtp", "num_speculative_tokens": 5}'},
     "engine speculative_config='{\"method\": \"mtp\", \"num_speculative_tokens\": 5}': φ reads a speculative config as a mapping of "
     "SpeculativeConfig's fields. Serve with it as one to compile it"),
])
def test_every_other_speculative_setting_is_refused_by_its_name(setting, why):
    """Another method, count or draft model, any other `SpeculativeConfig` field set, a setting that leaves the method or the
    count out, one stated in both forms, or a config that is no mapping would compile to Programs of a launch it did not run:
    the registration refuses each by the arg that names it."""
    with pytest.raises(ValueError, match=re.escape(why)):
        AC.engine_of(_launched(**setting), 0)
    with pytest.raises(ValueError, match=re.escape(why)):
        AC.phi_of(_launched(**setting), {}, 0)


def test_the_predictor_refuses_to_build_a_drafted_launchs_program_by_name():
    """φ admits the drafted launch, but no Program the predictor builds has its drafted rows, accept or draft forward yet, so
    its decoder (`domain.check_phi`, which every Program's build passes) refuses it by name, for a request and for a step: the
    admission does not let a drafted serve through as an undrafted Program. The undrafted launch's decoder builds as before."""
    from verity_vllm.program.predict import build
    from verity_vllm.program.predict.build import Unexpressible

    hf = json.loads(pinned_hf_config(*GLM53).read_text())
    drafted = AC.phi_of(_glm53_launch(speculative_config=dict(MTP5)), hf, 3)
    assert drafted.engine["speculative"] == MTP5
    why = re.escape(f"speculative decoding {MTP5!r} (a Program decodes one token a row; none builds a drafted request's rows, accept "
                    "and draft forward yet)")
    with pytest.raises(Unexpressible, match=why) as refused:
        build.decoder(drafted)
    assert "not modelled" not in str(refused.value)
    with pytest.raises(Unexpressible, match=why):
        build.build_program(drafted, "request", LP=4, T=2)
    with pytest.raises(Unexpressible, match=why):
        build.build_program(drafted, "step", M=6)
    assert build.decoder(AC.phi_of(_glm53_launch(), hf, 3)).kv_layout is None


def test_the_registration_states_the_dsa_top_k_environment_the_engines_build_set():
    """A run whose engine record has the environment its build set states the DSA top-k's, set or not, and nothing else of it:
    `predict.dsa` compiles a row past INDEX_TOPK only under the canonical row, and refuses it for an engine built without."""
    from verity_vllm.program.predict import build

    kw = {"tensor_parallel_size": 8, "max_model_len": 131072}
    flags = {"VLLM_USE_V2_MODEL_RUNNER": "1", "VERITY_DSA_TOPK": "canonical"}
    assert AC.engine_of({"engine_record": {"engine_kwargs": kw, "env_flags": flags}}, 0)[1] == {
        "max_model_len": 131072, "env": {"VERITY_DSA_TOPK": "canonical"}}
    hf = json.loads(pinned_hf_config(*GLM53).read_text())

    def decoder(env_flags: dict):
        meta = {"model": GLM53[0], "revision": GLM53[1], "target": H200, "execution": "eager",
                "engine_record": {"engine_kwargs": kw, "env_flags": env_flags}}
        return build.decoder(AC.phi_of(meta, hf, 0))

    assert decoder(flags).long_rows is None
    assert AC.engine_of({"engine_record": {"engine_kwargs": kw, "env_flags": {"VLLM_USE_V2_MODEL_RUNNER": "1"}}}, 0)[1] == {
        "max_model_len": 131072, "env": {}}
    assert "VERITY_DSA_TOPK=None, not {'VERITY_DSA_TOPK': 'canonical'}" in decoder({"VLLM_USE_V2_MODEL_RUNNER": "1"}).long_rows


GLM53 = ("zai-org/GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42")
H200 = {"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": False, "device_name": "NVIDIA H200", "moe_backend": "triton"}


def test_an_fp8_checkpoint_reaches_predict_dsa_as_float8_e4m3fn_and_a_config_stating_none_is_refused_as_before():
    from verity_vllm.program.predict import build, dsa
    from verity_vllm.program.predict.build import Unexpressible

    hf = json.loads(pinned_hf_config(*GLM53).read_text())
    assert hf["quantization_config"]["quant_method"] == "fp8" and hf["quantization_config"]["fmt"] == "e4m3"
    meta = {"model": GLM53[0], "revision": GLM53[1], "target": H200, "execution": "eager",
            "engine_record": {"engine_kwargs": {"tensor_parallel_size": 8, "max_model_len": 2048}}}
    phi = AC.phi_of(meta, hf, 3)
    assert phi.model["dtype"] == "float8_e4m3fn" and phi.tp == {"world": 8, "rank": 3} and phi.engine == {"max_model_len": 2048}
    assert isinstance(build.decoder(phi), dsa.Net)
    bare = AC.phi_of(meta, {k: v for k, v in hf.items() if k != "quantization_config"}, 3)
    assert "dtype" not in bare.model
    with pytest.raises(Unexpressible, match=re.escape("vocabulary cannot express: checkpoint dtype None (float8_e4m3fn only)")):
        build.decoder(bare)
    assert AC.checkpoint_dtype({"quantization_config": {"quant_method": "fp8", "fmt": "e5m2"}}) is None


def _glm53_launch(target: dict | None = None, **engine_kwargs: Any) -> dict:
    return {"model": GLM53[0], "revision": GLM53[1], "target": H200 if target is None else target, "execution": "eager",
            "engine_record": {"engine_kwargs": {"tensor_parallel_size": 8, "max_model_len": 2048, **engine_kwargs}}}


def test_an_fp8_launch_of_glm53_on_cc_9_0_states_the_fp8_ds_mla_layout_which_its_decoder_models():
    """GLM-5.3 at ``--kv-cache-dtype fp8`` on cc 9.0, the cache vLLM's FlashMLA sparse backend stores as ``fp8_ds_mla``: φ's
    engine states that layout, so its digest is not the default launch's, which is as before. Every other format, model or
    capability is refused by the setting, and so is the fp8 launch beside another attention backend or with layers skipped.
    GLM-5.3's decoder models the layout (`dsa.Net.kv_layout`), and the default launch's decoder has none."""
    from verity_vllm.program.predict import build

    hf = json.loads(pinned_hf_config(*GLM53).read_text())
    fp8, bf16 = AC.phi_of(_glm53_launch(kv_cache_dtype="fp8"), hf, 3), AC.phi_of(_glm53_launch(), hf, 3)
    assert fp8.engine == {"max_model_len": 2048, "kv_cache_dtype": "fp8_ds_mla"} and bf16.engine == {"max_model_len": 2048}
    assert AC.engine_of(_glm53_launch(kv_cache_dtype="fp8"), 3, hf) == ({"world": 8, "rank": 3}, fp8.engine)
    assert fp8.digest() != bf16.digest()
    assert {AC.phi_of(_glm53_launch(kv_cache_dtype=c), hf, 3).digest() for c in (None, "auto", "bfloat16")} == {bf16.digest()}
    assert AC.phi_of(_glm53_launch(kv_cache_dtype="fp8", attention_backend="FLASHMLA_SPARSE", kv_cache_dtype_skip_layers=[]),
                     hf, 3).digest() == fp8.digest()
    for cache in ("fp8_e4m3", "fp8_e5m2", "fp8_ds_mla"):
        with pytest.raises(ValueError, match=re.escape(_kv_refused(cache, "glm_moe_dsa", (9, 0)))):
            AC.phi_of(_glm53_launch(kv_cache_dtype=cache), hf, 3)
    with pytest.raises(ValueError, match=re.escape(_kv_refused("fp8", "glm_moe_dsa", (10, 0)))):
        AC.phi_of(_glm53_launch({**H200, "compute_capability": [10, 0]}, kv_cache_dtype="fp8"), hf, 3)
    with pytest.raises(ValueError, match=re.escape(_kv_refused("fp8", "glm_moe_dsa", None))):
        AC.phi_of(_glm53_launch({k: v for k, v in H200.items() if k != "compute_capability"}, kv_cache_dtype="fp8"), hf, 3)
    with pytest.raises(ValueError, match=re.escape(_kv_refused("fp8", "llama", (9, 0)))):
        AC.phi_of(_glm53_launch(kv_cache_dtype="fp8"), {**hf, "model_type": "llama"}, 3)
    with pytest.raises(ValueError, match=re.escape(_kv_refused("fp8", None, (9, 0)))):
        AC.engine_of(_glm53_launch(kv_cache_dtype="fp8"), 3)
    with pytest.raises(ValueError, match=re.escape(
            "engine attention_backend='FLASH_ATTN_MLA_SPARSE' at kv_cache_dtype='fp8': a request Program models that cache in "
            "FLASHMLA_SPARSE's layout fp8_ds_mla, which another backend does not store it in. Leave the backend to vLLM to compile it")):
        AC.phi_of(_glm53_launch(kv_cache_dtype="fp8", attention_backend="FLASH_ATTN_MLA_SPARSE"), hf, 3)
    with pytest.raises(ValueError, match=re.escape(
            "engine kv_cache_dtype_skip_layers=['0'] at kv_cache_dtype='fp8': a request Program models every layer's cache in the "
            "layout fp8_ds_mla, and those layers keep the default's. Serve with no layer skipped to compile it")):
        AC.phi_of(_glm53_launch(kv_cache_dtype="fp8", kv_cache_dtype_skip_layers=["0"]), hf, 3)
    assert (build.decoder(bf16).kv_layout, build.decoder(fp8).kv_layout) == (None, "fp8_ds_mla")


def test_a_user_with_two_requests_in_one_window_is_refused(tmp_path):
    run = _run(tmp_path, [_q("u", [3], [1], 8), _q("u", [3], [2], 8)])
    with pytest.raises(ValueError, match="user 'u' has two requests in window 3"):
        AC.served_inputs(run, 3)


def _windows(tmp_path: Path, windows: list[int], empty: list[int], advice: dict[int, WA.Advice]) -> Path:
    (tmp_path / "windows.jsonl").write_text("".join(json.dumps({"window": k, "empty": k in empty}) + "\n" for k in windows))
    for k, a in advice.items():
        (tmp_path / "advice" / f"w{k}" / "private").mkdir(parents=True)
        (tmp_path / "advice" / f"w{k}" / "private" / "advice.bin").write_bytes(WA.encode(a))
    return tmp_path


def test_a_window_carries_in_what_the_last_window_that_ran_steps_left(tmp_path):
    run = _windows(tmp_path, [0, 1, 2], [1], {0: W1})
    assert AC.carry_of(run, 0, WA.encode(W1)) == {"in": [], "out": ["a"], "follows": None}
    assert AC.carry_of(run, 2, WA.encode(W2)) == {"in": ["a"], "out": [], "follows": 0}


def test_a_carry_the_last_window_didnt_leave_is_refused(tmp_path):
    run = _windows(tmp_path, [0, 1], [], {0: W2})
    with pytest.raises(ValueError, match=r"window 1 doesn't open where window 0 closed: computed: carried request 0 \('a'\) .* not as "
                                         r"the last"):
        AC.carry_of(run, 1, WA.encode(W2))


def test_a_carry_with_no_advice_before_it_is_refused(tmp_path):
    run = _windows(tmp_path, [0, 1], [], {})
    with pytest.raises(ValueError, match="window 1 carries 1 requests in, 'a' first, and window 0, the one before it, has no advice"):
        AC.carry_of(run, 1, WA.encode(W2))
    with pytest.raises(ValueError, match="no window ran before it"):
        AC.carry_of(run, 0, WA.encode(W2))
