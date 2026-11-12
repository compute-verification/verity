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
from verity_vllm.pipeline import window_advice as WA

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "advice_compile.py"
_spec = importlib.util.spec_from_file_location("advice_compile", SCRIPT)
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
    `compile_steps` given that coin as the window's seed salt (`window_coin`) and the certified body derives the seed the window
    committed. Another coin gives another seed for the same id, the rest of the body none; a window whose line names no coin (a
    server that filled in `DEFAULT_SEED`) is refused by name, and a seed of the engine's own is not the one the compile derives."""
    from tests.program.predict.test_steps import _registered
    from verity_vllm.engine.served_seed import derived
    from verity_vllm.program.predict.build import Unexpressible
    from verity_vllm.program.predict.steps import compile_steps

    (tmp_path / "ok").mkdir()
    run, rec, advice = _served(tmp_path / "ok")
    _, _, bodies, _ = AC.served_inputs(run, 0)
    coin = AC.window_coin(run, 0)
    assert "seed" not in bodies["q0"] and coin["source"] == "in-process" and rec["ingress"]["seed_rule"] == "derived"
    reg, key = _registered(), bytes.fromhex(coin["coin"])
    assert compile_steps(reg, advice, bodies, salt=key).requests[0].seed == rec["committed_seeds"][0] == derived(key, "q0")
    other = compile_steps(reg, advice, bodies, salt=bytes(64) + b"\x01").requests[0].seed
    assert other == derived(bytes(64) + b"\x01", "q0") != rec["committed_seeds"][0]
    body = {**bodies["q0"], "temperature": 0.5, "max_tokens": 3, "prompt": [9, 8, 7, 6, 5]}
    assert compile_steps(reg, advice, {"q0": body}, salt=key).requests[0].seed == rec["committed_seeds"][0]
    lines = [json.loads(x) for x in (run / "windows.jsonl").read_text().splitlines()]
    (run / "windows.jsonl").write_text("".join(json.dumps({k: v for k, v in w.items() if k != "seed_coin"}) + "\n" for w in lines))
    assert AC.window_coin(run, 0) is None
    with pytest.raises(Unexpressible, match=r"request 0 \('q0'\): stochastic with no seed in its certified body, and the compile has "
                                            r"no window seed salt to derive one from: a seed serving fills in "
                                            r"\(`served_seed.DEFAULT_SEED`\)"):
        compile_steps(reg, advice, bodies, salt=None)
    (tmp_path / "own").mkdir()
    run, rec, advice = _served(tmp_path / "own", seed_override=7)
    key = bytes.fromhex(AC.window_coin(run, 0)["coin"])
    assert rec["committed_seeds"] == [7] and not rec["check"]["passed"]
    assert compile_steps(reg, advice, AC.served_inputs(run, 0)[2], salt=key).requests[0].seed == derived(key, "q0") != 7


def test_the_registered_default_seed_is_the_public_one_and_a_run_that_used_another_is_refused():
    from verity_vllm.engine.served_seed import DEFAULT_SEED
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


@pytest.mark.parametrize("cache", ["fp8", "fp8_e4m3", "fp8_ds_mla"])
def test_a_launch_that_names_another_kv_cache_format_is_refused_by_its_setting(cache):
    """φ carries no KV cache format, so such a launch would compile to the Programs of a launch at the default, which it did
    not run: the registration refuses it by the setting. The default, by either name or left out, compiles as before."""
    why = (f"engine kv_cache_dtype={cache!r}: a request Program models the KV cache of a launch at kv_cache_dtype auto or bfloat16 "
           "and no other format (an FP8 KV cache is not modelled), so the compile of this launch would be Programs it did not run. "
           "Serve with kv_cache_dtype at vLLM's default to compile it")
    with pytest.raises(ValueError, match=re.escape(why)):
        AC.engine_of(_launched(kv_cache_dtype=cache), 0)
    with pytest.raises(ValueError, match=re.escape(why)):
        AC.phi_of(_launched(kv_cache_dtype=cache), {}, 0)
    for stated in ({}, {"kv_cache_dtype": None}, {"kv_cache_dtype": "auto"}, {"kv_cache_dtype": "bfloat16"}):
        assert AC.engine_of(_launched(**stated), 0) == ({"world": 8, "rank": 0}, {"max_model_len": 2048})
        assert AC.phi_of(_launched(**stated), {}, 0).engine == {"max_model_len": 2048}


@pytest.mark.parametrize("setting", [{"speculative_config": {"method": "mtp", "num_speculative_tokens": 5}}, {"speculative_config": {}},
                                     {"spec_method": "mtp"}, {"spec_model": "draft"}, {"spec_tokens": 5}])
def test_a_launch_that_decodes_speculatively_is_refused_by_its_setting(setting):
    """φ carries no speculative config, so such a launch would compile to Programs whose decode rows are one token, which it
    did not run: the registration refuses it by the setting, in the config's form or vLLM's short ones. A launch that
    leaves them out, or at None, compiles as before."""
    (name, value), = setting.items()
    why = (f"engine {name}={value!r}: a request Program models no draft tokens (speculative decoding), its decode row is one "
           "token, so the compile of this launch would be Programs it did not run. Serve with speculative decoding off to compile it")
    with pytest.raises(ValueError, match=re.escape(why)):
        AC.engine_of(_launched(**setting), 0)
    with pytest.raises(ValueError, match=re.escape(why)):
        AC.phi_of(_launched(**setting), {}, 0)
    off = dict.fromkeys(("speculative_config", "spec_method", "spec_model", "spec_tokens"))
    assert AC.engine_of(_launched(**off), 0) == ({"world": 8, "rank": 0}, {"max_model_len": 2048})
    assert AC.phi_of(_launched(**off), {}, 0).engine == {"max_model_len": 2048}


def test_the_registration_states_the_dsa_top_k_environment_the_engines_build_set():
    """A run whose engine record has the environment its build set states the DSA top-k's, set or not, and nothing else of it:
    `predict.dsa` compiles a row past INDEX_TOPK only under the canonical row, and refuses it for an engine built without."""
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
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
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
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
