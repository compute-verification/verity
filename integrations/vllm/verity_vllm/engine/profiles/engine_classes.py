"""The engine classes: by name, each engine a registration without batch invariance compiles in, and what serving it fixes.

Class G is the one registered: GLM-5.3 at TP8, eager or compiled (vLLM's breakable CUDA graphs over the eager kernels at fixed
capture sizes, `CLASS_G_EXECUTIONS`), on H200 (`predict.dsa`'s domain), with the engine limits that keep every
launch where `predict.dsa` writes it: steps of at most 2,048 tokens (its captured statics), at most 2,048 keys a row (DSA's top-k is
the identity there, and deterministic), and prefill chunks of at most 256 rows (its decode route at 8 heads a rank), with no prefix
cache.  The step compiler admits a registration in it (`predict.steps.launch_class`); `verity_vllm.LLM(engine_class="G")` serves it
with `options`, and its admission (`engine.class_admission`) refuses as it serves what the compiler would refuse of the window: a
prompt over the long-prefill threshold, which vLLM prefills in chunks, and a request in flight when a Commit window closes.
"""
from __future__ import annotations

import json
from typing import Any

from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
from verity_vllm.engine.profiles.devices import DEVICE_NAME
from verity_vllm.errors import Refused

#: class G's registration: the model type, tensor-parallel world, default execution label and device name (`predict.steps.launch_class`)
CLASS_G = {"model_type": "glm_moe_dsa", "world": 8, "execution": "eager", "device_name": DEVICE_NAME["h200"]}
#: the execution labels class G serves, each with the target compilation it runs at: compiled is vLLM's breakable CUDA graphs over
#: the eager kernels, the one compiled execution vLLM serves glm_moe_dsa at (`TargetProfile.COMPILATIONS`, `predict.dsa`)
CLASS_G_EXECUTIONS: dict[str, str | None] = {"eager": None, "compiled": "breakable"}
#: class G's engine limits: a registration states each, at most these, and a served class-G engine runs at exactly these
CLASS_G_LIMITS = {"max_num_batched_tokens": 2048, "max_model_len": 2048, "long_prefill_token_threshold": 256}
#: the experts class G's decoder writes (`predict.dsa` refuses another): vLLM's own selection for GLM-5.3's block-FP8 experts on
#: the H200, given to the served engine so it applies it and its target declares it
CLASS_G_MOE_BACKEND = "triton"

CLASSES: dict[str, tuple[dict[str, Any], dict[str, int], str]] = {"G": (CLASS_G, CLASS_G_LIMITS, CLASS_G_MOE_BACKEND)}
EXECUTIONS: dict[str, dict[str, str | None]] = {"G": CLASS_G_EXECUTIONS}


def of(name: str) -> tuple[dict[str, Any], dict[str, int], str]:
    """Engine class ``name``'s registration facts, engine limits and experts; `Refused` for a class not registered."""
    if name not in CLASSES:
        raise Refused(f"engine class {name!r} is not registered (registered: {', '.join(sorted(CLASSES))})")
    return CLASSES[name]


def compilation(name: str, execution: str | None) -> str | None:
    """The target compilation class ``name`` serves ``execution`` at (None: its default execution); `Refused` for an execution the
    class doesn't serve."""
    execution = execution or of(name)[0]["execution"]
    if execution not in EXECUTIONS[name]:
        raise Refused(f"execution {execution!r} (class {name} serves {', '.join(EXECUTIONS[name])})", what=f"engine class {name}")
    return EXECUTIONS[name][execution]


def options(name: str, execution: str | None = None) -> dict[str, Any]:
    """The `verity_vllm.LLM` options engine class ``name`` fixes at ``execution`` (default its own): its world, the execution, its
    engine limits at their values, the prefix cache off, its decoder's experts and, compiled, the CUDA graphs its compilation applies."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    facts, limits, moe = of(name)
    comp = compilation(name, execution)
    out = {"tensor_parallel_size": facts["world"], "execution": execution or facts["execution"], **limits, "enable_prefix_caching": False,
           "moe_backend": moe}
    if comp is not None:
        out["compilation_config"] = TargetProfile.default().with_overrides(compilation=comp).engine_args()["compilation_config"]
    return out


def prompt_limit(name: str) -> int:
    """The longest prompt a served window of class ``name`` admits: its long-prefill threshold, past which vLLM prefills a prompt in
    chunks, and the request wrapper prefills in one launch."""
    return of(name)[1]["long_prefill_token_threshold"]


def max_num_seqs(name: str) -> int:
    """The most requests a step of class ``name`` may run: as many prompts at the limit as one step's tokens hold, so no admitted
    prefill is cut by the step's budget either."""
    return of(name)[1]["max_num_batched_tokens"] // prompt_limit(name)


def effective_refusals(name: str, effective: dict[str, Any]) -> list[str]:
    """What of class ``name``'s world, engine limits and prefix cache the built engine doesn't run at, read back from its vLLM
    config (``effective``, `build.effective_execution_doc`); empty when it runs at all of them."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    facts, limits, _ = of(name)
    want = {"tensor_parallel_size": facts["world"], **limits, "enable_prefix_caching": False}
    if effective.get("enforce_eager") is False:
        c = TargetProfile.COMPILATIONS[compilation(name, "compiled")]
        want.update(breakable_cudagraph=True, capture_sizes=list(c["capture_sizes"]))
    return [f"the built engine's {k} is {effective.get(k)!r}, not the class's {v!r}" for k, v in want.items()
            if effective.get(k) != v]


def check_model(name: str, model: str, revision: str) -> None:
    """`Refused` unless the checkpoint pinned at ``model``@``revision`` is of class ``name``'s model type, read from its config of
    record (`dense_generic.pinned_hf_config`)."""
    path = pinned_hf_config(model, revision)
    got = None if path is None else json.loads(path.read_text()).get("model_type")
    want = of(name)[0]["model_type"]
    if got != want:
        raise Refused(f"{model}@{revision} is a {got!r} checkpoint, not the class's {want!r}", what=f"engine class {name}")


__all__ = ["CLASSES", "CLASS_G", "CLASS_G_EXECUTIONS", "CLASS_G_LIMITS", "CLASS_G_MOE_BACKEND", "EXECUTIONS", "check_model",
           "compilation", "effective_refusals", "max_num_seqs", "of", "options", "prompt_limit"]
