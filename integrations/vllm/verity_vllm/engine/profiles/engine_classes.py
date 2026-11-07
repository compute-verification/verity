"""The engine classes: by name, each engine a registration without batch invariance compiles in, and what serving it fixes.

Class G is the one registered: GLM-5.3 at TP8, eager, on H200 (`predict.dsa`'s domain).  A registration states each of its engine
limits, at most `CLASS_G_LIMITS`: steps of up to 8,192 tokens and 256 sequences, positions up to 32,768, and a long-prefill
threshold of at most 256 rows, so each prefill chunk stays on DSA's decode route at 8 heads a rank, and the prefix cache on or off.
The step compiler admits such a registration (`predict.steps.launch_class`), compiles a long prompt as its prefill chunks, each at
the launch it ran in, and a prefix-cache hit or a request a Commit window carries as the request's Program after its history, and
refuses by name, request by request, a launch `predict.dsa` doesn't write: a step past the 2,048 tokens its captured all-reduce and
cuBLAS routes cover.  A row past INDEX_TOPK (2,048) keys compiles, its top-k order a field of the Program's `run` input, checked to
be the canonical row an engine under `DSA_TOPK_ENV` serves; no committed value binds that input yet, so the served `max_model_len`
keeps every row within INDEX_TOPK.  `verity_vllm.LLM(engine_class="G")` serves it at `CLASS_G_SERVED`, the limits within which
every launch is written, and up to its 256 sequences a step, with `options`, the prefix cache off; its admission
(`engine.class_admission`) refuses as it serves a prompt over the served `max_model_len`, and a request in flight when a Commit
window closes, since the served Commit doesn't commit the window's edge ports (`workload.Edge`), which a carried request's history
reads.
"""
from __future__ import annotations

import json
from typing import Any

from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
from verity_vllm.engine.profiles.devices import DEVICE_NAME
from verity_vllm.errors import Refused

#: class G's registration: the model type, tensor-parallel world, execution label and device name (`predict.steps.launch_class`)
CLASS_G = {"model_type": "glm_moe_dsa", "world": 8, "execution": "eager", "device_name": DEVICE_NAME["h200"]}
#: class G's engine limits: a registration states each, at most these
CLASS_G_LIMITS = {"max_num_batched_tokens": 8192, "max_model_len": 32768, "long_prefill_token_threshold": 256, "max_num_seqs": 256}
#: the engine limits a served class-G engine runs at, its `max_num_seqs` at most the class's: within `CLASS_G_LIMITS`, where every
#: launch `predict.dsa` writes is captured
CLASS_G_SERVED = {"max_num_batched_tokens": 2048, "max_model_len": 2048, "long_prefill_token_threshold": 256}
#: the experts class G's decoder writes (`predict.dsa` refuses another): vLLM's own selection for GLM-5.3's block-FP8 experts on
#: the H200, given to the served engine so it applies it and its target declares it
CLASS_G_MOE_BACKEND = "triton"
#: the environment an engine serves a row past INDEX_TOPK keys under: the DSA top-k plugin's canonical row (the top-INDEX_TOPK
#: set in ascending index order), the one order `DsaTopKLongRow` passes; `predict.dsa` refuses such a row under an engine whose
#: stated environment lacks it
DSA_TOPK_ENV = {"VERITY_DSA_TOPK": "canonical"}

CLASSES: dict[str, tuple[dict[str, Any], dict[str, int], dict[str, int], str]] = {
    "G": (CLASS_G, CLASS_G_LIMITS, CLASS_G_SERVED, CLASS_G_MOE_BACKEND)}


def of(name: str) -> tuple[dict[str, Any], dict[str, int], dict[str, int], str]:
    """Engine class ``name``'s registration facts, engine limits, served limits and experts; `Refused` for a class not registered."""
    if name not in CLASSES:
        raise Refused(f"engine class {name!r} is not registered (registered: {', '.join(sorted(CLASSES))})")
    return CLASSES[name]


def options(name: str) -> dict[str, Any]:
    """The `verity_vllm.LLM` options engine class ``name`` fixes: its world, its execution, its served limits at their values, the
    prefix cache off and its decoder's experts."""
    facts, _, served, moe = of(name)
    return {"tensor_parallel_size": facts["world"], "execution": facts["execution"], **served, "enable_prefix_caching": False,
            "moe_backend": moe}


def prompt_limit(name: str) -> int:
    """The longest prompt a served window of class ``name`` admits: its served ``max_model_len`` (a longer prompt has no position
    for its first output), prefilled in chunks of at most its long-prefill threshold."""
    return of(name)[2]["max_model_len"]


def max_num_seqs(name: str) -> int:
    """The most requests a step of class ``name`` runs: its limit, which a served engine runs at unless it asks for fewer."""
    return of(name)[1]["max_num_seqs"]


def effective_refusals(name: str, effective: dict[str, Any]) -> list[str]:
    """What of class ``name``'s world, served limits and prefix cache the built engine doesn't run at, read back from its vLLM
    config (``effective``, `build.effective_execution_doc`); empty when it runs at all of them."""
    facts, _, served, _ = of(name)
    want = {"tensor_parallel_size": facts["world"], **served, "enable_prefix_caching": False}
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


__all__ = ["CLASSES", "CLASS_G", "CLASS_G_LIMITS", "CLASS_G_MOE_BACKEND", "CLASS_G_SERVED", "DSA_TOPK_ENV", "check_model", "effective_refusals",
           "max_num_seqs", "of", "options", "prompt_limit"]
