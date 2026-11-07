"""`tools/advice_run.py` under an engine class: the class sets the engine options the arguments leave unset, a `--max-prompt` over
the class's prompt limit is refused before any engine, every prompt fits `--max-prompt` in the model's chat template, and the
run's target states what the engine ran (its batch invariance read back, its experts, the device's name), as class G's
registration reads it."""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "advice_run.py"
_spec = importlib.util.spec_from_file_location("advice_run", SCRIPT)
AR = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("advice_run", AR)
_spec.loader.exec_module(AR)


def _args(**kw) -> argparse.Namespace:
    base = {"max_seqs": None, "max_batched_tokens": None, "kv_bytes": None, "execution": None, "engine_class": None}
    return argparse.Namespace(**{**base, **kw})


def test_the_engine_class_sets_what_the_arguments_leave_unset():
    assert AR.engine_options(_args(engine_class="G")) == {"engine_class": "G"}
    assert AR.engine_options(_args(engine_class="G", max_seqs=4)) == {"engine_class": "G", "max_num_seqs": 4}
    assert AR.engine_options(_args()) == {"max_num_seqs": 8, "max_num_batched_tokens": 256}
    assert AR.engine_options(_args(execution="eager", kv_bytes=1 << 30)) == {
        "max_num_seqs": 8, "max_num_batched_tokens": 256, "kv_cache_memory_bytes": 1 << 30, "execution": "eager"}


def test_a_max_prompt_over_the_class_limit_is_refused_before_any_engine(tmp_path):
    with pytest.raises(SystemExit, match="--max-prompt 3000 is over engine class G's prompt limit 2048"):
        AR.main(["--out", str(tmp_path / "run"), "--engine-class", "G", "--max-prompt", "3000"])
    assert not (tmp_path / "run").exists()


class _Tok:
    """A tokenizer of whitespace-separated words whose chat template adds 40 tokens."""

    chat_template = "t"

    def encode(self, text, add_special_tokens=True):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        return "<t> " * 40 + messages[0]["content"]


def test_every_prompt_fits_the_prompt_limit_in_the_chat_template():
    reqs = AR.traffic(_Tok(), 64, random.Random(0), 256, 0, [8, 32], sampled=0.0)
    lengths = [len(r.prompt_token_ids) for r in reqs]
    assert max(lengths) <= 256 and max(lengths) > 200
    assert all(r.temperature == 0.0 and r.seed is None for r in reqs)
    filler = _Tok().encode(AR.FILLER * 8)
    assert len(AR.prompt_ids(_Tok(), filler, 232, "Say hello.", 256)) == 256


def test_the_runs_target_states_what_the_engine_ran():
    device = {"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": True}
    record = {"effective": {"batch_invariant": False}, "engine_kwargs": {"moe_backend": "triton"}}
    assert AR.target_of(device, record, "NVIDIA H200") == {"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": False,
                                                           "moe_backend": "triton", "device_name": "NVIDIA H200"}
    assert AR.target_of(device, {}, "NVIDIA H200") == {**device, "device_name": "NVIDIA H200"}


def test_a_class_g_runs_registration_is_in_class_g():
    """What a class-G run records (its engine's keyword arguments, its target, its execution) registers, as
    `tools/advice_compile.py` reads it, a φ without batch invariance that class G admits."""
    import json

    from verity_vllm import llm as L
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
    from verity_vllm.program.predict.phi import Phi
    from verity_vllm.program.predict.steps import launch_class

    spec = importlib.util.spec_from_file_location("advice_compile", SCRIPT.parent / "advice_compile.py")
    AC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AC)
    model, revision = "zai-org/GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42"
    opts = L.class_options({"engine_class": "G"})
    kw = {k: v for k, v in opts.items() if k in (*L.ENGINE_ARGS, "tensor_parallel_size", "max_num_seqs")}
    record = {"engine_kwargs": {**kw, "enforce_eager": True}, "effective": {"batch_invariant": False}, "execution": opts["execution"]}
    target = AR.target_of({"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": True}, record, "NVIDIA H200")
    tp, engine = AC.engine_of({"engine_record": record}, 7)
    phi = Phi(model={"repo": model, "revision": revision, "hf_config": json.loads(pinned_hf_config(model, revision).read_text())},
              target=target, execution=record["execution"], tp=tp, engine=engine)
    assert tp == {"world": 8, "rank": 7} and phi.target_profile().batch_invariant is False
    launch_class(phi)
