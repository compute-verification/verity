"""`tools/advice_run.py` under an engine class: the class sets the engine options the arguments leave unset, the launch states
its context limit (`--max-model-len`), a `--max-prompt` over it is refused before any engine, every prompt fits `--max-prompt` in
the model's chat template, and the run's target states what the engine ran (its batch invariance read back, its experts, the
device's name) and its registration the context limit it ran at (the launch's, or the one vLLM fitted), as class G's
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
assert _spec is not None and _spec.loader is not None
AR = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("advice_run", AR)
_spec.loader.exec_module(AR)


def _args(**kw) -> argparse.Namespace:
    base = {"max_seqs": None, "max_batched_tokens": None, "kv_bytes": None, "execution": None, "engine_class": None,
            "max_model_len": None}
    return argparse.Namespace(**{**base, **kw})


def test_the_engine_class_sets_what_the_arguments_leave_unset():
    assert AR.engine_options(_args(engine_class="G")) == {"engine_class": "G"}
    assert AR.engine_options(_args(engine_class="G", max_seqs=4)) == {"engine_class": "G", "max_num_seqs": 4}
    assert AR.engine_options(_args(engine_class="G", max_model_len=-1)) == {"engine_class": "G", "max_model_len": -1}
    assert AR.engine_options(_args()) == {"max_num_seqs": 8, "max_num_batched_tokens": 256}
    assert AR.engine_options(_args(execution="eager", kv_bytes=1 << 30)) == {
        "max_num_seqs": 8, "max_num_batched_tokens": 256, "kv_cache_memory_bytes": 1 << 30, "execution": "eager"}


def test_a_max_prompt_over_the_launchs_context_limit_is_refused_before_any_engine(tmp_path):
    with pytest.raises(SystemExit, match="--max-prompt 401409 is over --max-model-len 401408, whose engine refuses a longer prompt"):
        AR.main(["--out", str(tmp_path / "run"), "--engine-class", "G", "--max-model-len", "401408", "--max-prompt", "401409"])
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
    compiled = {"effective": {"batch_invariant": False, "enforce_eager": False, "breakable_cudagraph": True}}
    assert AR.target_of(device, compiled, "NVIDIA H200")["compilation"] == "breakable"
    eager = {"effective": {"batch_invariant": False, "enforce_eager": True, "breakable_cudagraph": True}}
    assert "compilation" not in AR.target_of(device, eager, "NVIDIA H200")


def _class_g_phi(execution: str | None = None, effective: dict | None = None, max_model_len: int = 131072):
    """The φ a class-G run at ``execution`` and a launch of ``max_model_len`` registers, as `tools/advice_compile.py` reads what
    it records (its engine's keyword arguments, its target, its execution), the built engine reading back ``effective``."""
    import json

    from verity_vllm import llm as L
    from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
    from verity_vllm.program.predict.phi import Phi

    spec = importlib.util.spec_from_file_location("advice_compile", SCRIPT.parent / "advice_compile.py")
    assert spec is not None and spec.loader is not None
    AC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AC)
    model, revision = "zai-org/GLM-5.3", "aca966e4e02791568aa6a4ced368624b3d897f42"
    opts = L.class_options({"engine_class": "G", "max_model_len": max_model_len, **({"execution": execution} if execution else {})})
    kw = {k: v for k, v in opts.items() if k in (*L.ENGINE_ARGS, "tensor_parallel_size", "max_num_seqs")}
    eager = opts["execution"] == "eager"
    record = {"engine_kwargs": {**kw, "enforce_eager": eager}, "execution": opts["execution"],
              "effective": {"batch_invariant": False, "enforce_eager": eager, "breakable_cudagraph": True, **(effective or {})}}
    target = AR.target_of({"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": True}, record, "NVIDIA H200")
    tp, engine = AC.engine_of({"engine_record": record}, 7)
    phi = Phi(model={"repo": model, "revision": revision, "hf_config": json.loads(pinned_hf_config(model, revision).read_text())},
              target=target, execution=record["execution"], tp=tp, engine=engine)
    assert tp == {"world": 8, "rank": 7} and phi.target_profile().batch_invariant is False
    return phi


def test_a_class_g_runs_registration_is_in_class_g():
    """A class-G run's registration is a φ without batch invariance that class G admits, eager or compiled (at its breakable
    graphs, whose capture sizes its Build pads to); a compiled run without them is refused."""
    import dataclasses

    from verity_vllm.program.predict.common import Unexpressible, graph_capture_sizes
    from verity_vllm.program.predict.domain import check_phi
    from verity_vllm.program.predict.dsa import Net
    from verity_vllm.program.predict.steps import launch_class

    launch_class(_class_g_phi())
    assert _class_g_phi().engine["max_model_len"] == 131072
    assert graph_capture_sizes(_class_g_phi()) == ()
    phi = _class_g_phi("compiled")
    assert phi.target_profile().compilation == "breakable" and phi.target_profile().execution_label == "compiled"
    launch_class(phi)
    check_phi(dataclasses.replace(phi, model={**phi.model, "dtype": "float8_e4m3fn"}), Net.DOMAIN)
    assert graph_capture_sizes(phi) == phi.target_profile().graph_capture_sizes() and graph_capture_sizes(phi)[-1] == 512
    with pytest.raises(Unexpressible, match="target compilation None at execution 'compiled', not 'breakable'"):
        launch_class(_class_g_phi("compiled", {"breakable_cudagraph": False}))


def test_a_launch_that_leaves_the_limit_to_vllm_registers_the_length_the_engine_fitted():
    """Under ``max_model_len=-1`` the registration states the length the built engine reads back, never the launch's -1, which
    class G refuses; a record that reads none back is refused by name, and so is a length past the 131,072 positions a class-G
    registration may state, fitted or stated."""
    from verity_vllm.program.predict.common import Unexpressible
    from verity_vllm.program.predict.steps import launch_class

    phi = _class_g_phi(effective={"max_model_len": 4032}, max_model_len=-1)
    assert phi.engine["max_model_len"] == 4032
    launch_class(phi)
    with pytest.raises(Unexpressible, match="engine arg max_model_len None, not a stated value of at most 131072"):
        launch_class(_class_g_phi(max_model_len=-1))
    with pytest.raises(Unexpressible, match="engine arg max_model_len 383872, not a stated value of at most 131072"):
        launch_class(_class_g_phi(effective={"max_model_len": 383872}, max_model_len=-1))
    with pytest.raises(Unexpressible, match="engine arg max_model_len 262144, not a stated value of at most 131072"):
        launch_class(_class_g_phi(max_model_len=262144))
