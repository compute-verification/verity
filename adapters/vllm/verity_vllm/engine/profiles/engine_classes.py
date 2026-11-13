"""The engine classes: by name, each engine a registration without batch invariance compiles in, and what serving it fixes.

Class G is the one registered: GLM-5.3 at TP8, eager or compiled (vLLM's breakable CUDA graphs over the eager kernels at fixed
capture sizes, `CLASS_G_EXECUTIONS`), on H200 (`predict.dsa`'s domain).  A registration states each of its engine limits, at most
`CLASS_G_LIMITS`: steps of up to 8,192 tokens and 256 sequences, positions up to 131,072, and a long-prefill threshold of at most 256
rows, so each prefill chunk stays on DSA's decode route at 8 heads a rank, and the prefix cache on or off.  The step compiler admits
such a registration (`predict.steps.launch_class`), compiles a long prompt as its prefill chunks, each at the launch it ran in, and
a prefix-cache hit or a request a Commit window carries as the request's Program after its history, and refuses by name, request by
request, a launch `predict.dsa` doesn't write: a step past the 2,048 tokens its captured all-reduce and cuBLAS routes cover.  A row
past INDEX_TOPK (2,048) keys compiles, its top-k computed from its logits in the Program and its rows read in ascending index
order, the canonical row an engine under `DSA_TOPK_ENV` serves, so the served `max_model_len` can pass INDEX_TOPK.
`verity_vllm.LLM(engine_class="G", max_model_len=N)` serves it at `CLASS_G_SERVED`, the limits within which every launch is
written, at the context limit its launch states (N positions, or -1: the longest request the engine's KV pool holds, which vLLM
fits when the engine starts), which the class doesn't fix: an engine serves past the 131,072 positions a registration may
state, and the step compiler refuses that engine's registration by name; and
up to its 256 sequences a step, with `options`, the prefix cache off, under `DSA_TOPK_ENV`, which its build sets over the process's
(`env`) and a built engine that doesn't carry is refused (`effective_refusals`); its admission (`engine.class_admission`) refuses as it serves
a prompt over the built engine's `max_model_len`, and a request in flight when a Commit window closes, since the served Commit doesn't
commit the window's edge ports (`workload.Edge`), which a carried request's history reads.
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
#: class G's engine limits: a registration states each, at most these.  Its `max_model_len` is the step compiler's alone, the
#: positions the Definitions a request's rows bind are circuit-checked to; a served engine's is its launch's (`CLASS_G_SERVED`)
CLASS_G_LIMITS = {"max_num_batched_tokens": 8192, "max_model_len": 131072, "long_prefill_token_threshold": 256, "max_num_seqs": 256}
#: the engine limits a served class-G engine runs at, its `max_num_seqs` at most the class's: within `CLASS_G_LIMITS`, where every
#: launch `predict.dsa` writes is captured (steps of at most 2,048 tokens), a long prompt prefilled in chunks of at most 256 rows;
#: its `max_model_len` is the launch's
CLASS_G_SERVED = {"max_num_batched_tokens": 2048, "long_prefill_token_threshold": 256}
#: the experts class G's decoder writes (`predict.dsa` refuses another): vLLM's own selection for GLM-5.3's block-FP8 experts on
#: the H200, given to the served engine so it applies it and its target declares it
CLASS_G_MOE_BACKEND = "triton"
#: the environment an engine serves a row past INDEX_TOPK keys under: the DSA top-k plugin's canonical row (the top-INDEX_TOPK
#: set in ascending index order), the order `DsaButterflyRows` reads it in; `predict.dsa` refuses such a row under an engine whose
#: stated environment lacks it
DSA_TOPK_ENV = {"VERITY_DSA_TOPK": "canonical"}

CLASSES: dict[str, tuple[dict[str, Any], dict[str, int], dict[str, int], str]] = {
    "G": (CLASS_G, CLASS_G_LIMITS, CLASS_G_SERVED, CLASS_G_MOE_BACKEND)}
EXECUTIONS: dict[str, dict[str, str | None]] = {"G": CLASS_G_EXECUTIONS}
#: the environment each class's engine is built under, over whatever the process inherited (`engine.build.build_engine`'s ``env``)
ENVS: dict[str, dict[str, str]] = {"G": DSA_TOPK_ENV}


def of(name: str) -> tuple[dict[str, Any], dict[str, int], dict[str, int], str]:
    """Engine class ``name``'s registration facts, engine limits, served limits and experts; `Refused` for a class not registered."""
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
    served limits at their values, the prefix cache off, its decoder's experts and, compiled, the CUDA graphs its compilation
    applies.  The context limit isn't one: the launch states its ``max_model_len``."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    facts, _, served, moe = of(name)
    comp = compilation(name, execution)
    out = {"tensor_parallel_size": facts["world"], "execution": execution or facts["execution"], **served, "enable_prefix_caching": False,
           "moe_backend": moe}
    if comp is not None:
        out["compilation_config"] = TargetProfile.default().with_overrides(compilation=comp).engine_args()["compilation_config"]
    return out


def env(name: str) -> dict[str, str]:
    """The environment class ``name``'s engine is built under (`ENVS`): its build sets it over the process's, so a served row past
    INDEX_TOPK is the canonical one its registration reads."""
    of(name)
    return dict(ENVS[name])


def max_num_seqs(name: str) -> int:
    """The most requests a step of class ``name`` runs: its limit, which a served engine runs at unless it asks for fewer."""
    return of(name)[1]["max_num_seqs"]


def effective_refusals(name: str, effective: dict[str, Any], env_flags: dict[str, str], max_model_len: int) -> list[str]:
    """What of class ``name``'s world, served limits, prefix cache and environment, and of its launch's context limit
    (``max_model_len``: a length, or -1 for one vLLM fitted to the KV pool), the built engine doesn't run at, read back from
    its vLLM config (``effective``, `build.effective_execution_doc`) and the environment its build set (``env_flags``); empty when
    it runs at all of them."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    facts, _, served, _ = of(name)
    want = {"tensor_parallel_size": facts["world"], **served, "enable_prefix_caching": False}
    if effective.get("enforce_eager") is False:
        c = TargetProfile.COMPILATIONS[compilation(name, "compiled")]
        want.update(breakable_cudagraph=True, capture_sizes=list(c["capture_sizes"]))
    got = effective.get("max_model_len")
    if max_model_len == -1:
        limit = [] if type(got) is int and got > 0 else [f"the built engine's max_model_len is {got!r}, not a length vLLM fitted"]
    else:
        limit = [] if got == max_model_len else [f"the built engine's max_model_len is {got!r}, not the launch's {max_model_len}"]
    return [f"the built engine's {k} is {effective.get(k)!r}, not the class's {v!r}" for k, v in want.items()
            if effective.get(k) != v] + limit + [f"the built engine's environment sets {k}={env_flags.get(k)!r}, not the class's {v!r}"
                                                 for k, v in env(name).items() if env_flags.get(k) != v]


def check_model(name: str, model: str, revision: str) -> None:
    """`Refused` unless the checkpoint pinned at ``model``@``revision`` is of class ``name``'s model type, read from its config of
    record (`dense_generic.pinned_hf_config`)."""
    path = pinned_hf_config(model, revision)
    got = None if path is None else json.loads(path.read_text()).get("model_type")
    want = of(name)[0]["model_type"]
    if got != want:
        raise Refused(f"{model}@{revision} is a {got!r} checkpoint, not the class's {want!r}", what=f"engine class {name}")


__all__ = ["CLASSES", "CLASS_G", "CLASS_G_EXECUTIONS", "CLASS_G_LIMITS", "CLASS_G_MOE_BACKEND", "CLASS_G_SERVED", "DSA_TOPK_ENV", "ENVS",
           "EXECUTIONS", "check_model", "compilation", "effective_refusals", "env", "max_num_seqs", "of", "options"]
