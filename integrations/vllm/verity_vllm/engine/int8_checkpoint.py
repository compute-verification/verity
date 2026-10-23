"""Int8 checkpoints for the exact-int8 linears, made offline from a pinned BF16 checkpoint: the recipes ``verity-int8-channel127/v1``
and ``/v2``.

    verity-vllm int8-checkpoint make --role ROLE [--hf-home DIR]        re-make a registered int8 pin (under its own recipe) and check
                                                                          every file's sha256
    verity-vllm int8-checkpoint pin --base-role ROLE --role ROLE --repo ID [--recipe ID] [--hf-home DIR]
                                                                          make one and print its new manifest entry

The weights the plain step reads (``engine.int8_exact``, ``TargetProfile.int8_exact``): ``benchmarks/ppl/steps.py``'s
``quant_rows(w.float(), 127)``, int8 with every entry in [-127, 127] and one float32 scale per output row.  Like
``engine.fp8_checkpoint``'s recipe, on its container (shard layout, header, index, copied files, pinned revision), it is
data-free and a pure function of the base checkpoint's bytes; the linear weights are the same tensors (every 2-D tensor none of
whose dimensions is ``vocab_size``).  For every such W [N, K]:

* W is read exactly as float32 (BF16, F16 and F32 widen exactly);
* for every row n: a = max |W[n, :]|, and s = max(a, RN32(1e-30)) / 127 in float32 (one round-to-nearest-even division);
* the stored weight is q = clamp(rint(W / s), -127, 127): the float32 quotient (round-to-nearest-even), rounded to the nearest
  integer, ties to even, as int8;
* stored as ``<name>`` I8 [N, K] and ``<module>.weight_scale`` F32 [N, 1]; the dequantized weight is q * s.

N and K must be multiples of 8 (``torch._int_mm``'s shape rule on CUDA), else the checkpoint is refused whole.  config.json gets
``quantization_config = {"modules_to_not_convert": ["lm_head"], "quant_method": "verity_int8", "weight_max": 127}``, the vLLM
method ``engine.int8_vllm`` registers.  The pin's ``int8_of`` names its BF16 base.  Int8 pins live in
``manifests/checkpoints-int8.json`` (`config.SIBLING_MANIFESTS`), which ``config.checkpoint_entries`` reads with ``checkpoints.json``.

A mixture of experts whose experts are stored as 2-D tensors (OLMoE's ``mlp.experts.<e>.{gate,up,down}_proj.weight``) is made too
(``fp8_checkpoint``'s ``moe``): each expert's projections are linears like any other, served by ``engine.int8_vllm``'s
``Int8ExpertsMethod``, and the routers, the [num_experts, hidden_size] tensors vLLM builds with no quantization config, are
copied as they are (bf16).

``verity-int8-channel127/v2`` is v1 plus the LM head: ``lm_head.weight`` [vocab_size, K] is made by the same row rule (an int8
``lm_head.weight`` and an F32 ``lm_head.weight_scale`` [vocab_size, 1]), and ``modules_to_not_convert`` is ``[]``, so vLLM's
``ParallelLMHead`` serves the int8 linear too.  The embedding stays as the base has it (a gather, not a product).  A config with
``tie_word_embeddings`` true or absent is refused, as is a checkpoint without an ``lm_head.weight`` of its own.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from verity_vllm import config as C
from verity_vllm.config import option
from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_exact as E
from verity_vllm.program.frontend.target_profile import TargetProfile

RECIPE = "verity-int8-channel127/v1"
RECIPE_V2 = "verity-int8-channel127/v2"
QUANTIZATION = {"quant_method": TargetProfile.INT8_EXACT_QUANTIZATION, "weight_max": E.W_MAX}


def channel_scales(w: np.ndarray) -> np.ndarray:
    """s [N, 1] float32 of a float32 [N, K]: max |W| per row, floored at 1e-30, / 127 (steps.py's `quant_rows`)."""
    amax = np.abs(w).max(axis=1, keepdims=True)
    if not np.isfinite(amax).all():
        raise ValueError("non-finite weight")
    return (np.maximum(amax, np.float32(E.SCALE_FLOOR)) / np.float32(E.W_MAX)).astype(np.float32)


def quantize(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """q [N, K] int8 of a float32 [N, K] under its row scales `s` [N, 1]."""
    return np.clip(np.rint(w / s), -E.W_MAX, E.W_MAX).astype(np.int8)


INT8 = F.Recipe(id=RECIPE, weight_dtype="I8", scale_suffix="weight_scale", rows_per_scale=1, scale_shape=lambda n, k: [n, 1],
                refusal=lambda n, k: "not a multiple of 8 on both axes" if n % 8 or k % 8 else None,
                scales=channel_scales, quantize=quantize, quantization=QUANTIZATION, of="int8_of", moe=True)
INT8_V2 = replace(INT8, id=RECIPE_V2, lm_head=True)
RECIPES = {r.id: r for r in (INT8, INT8_V2)}


def recipe_of(entry: dict) -> F.Recipe:
    """The int8 recipe a pin was made by (its `recipe.id`)."""
    rid = (entry.get("recipe") or {}).get("id")
    if rid not in RECIPES:
        raise C.UsageError(f"{entry.get('role')}: recipe {rid!r} is none of the int8 recipes {sorted(RECIPES)}")
    return RECIPES[rid]


@dataclass(frozen=True, kw_only=True)
class MakeOptions:
    """Re-make a registered int8 pin (`recipe` in checkpoints-int8.json) from its BF16 pin and check every file's sha256."""

    role: str = option("--role", help="the int8 pin's role in checkpoints-int8.json")
    checkpoints: str = option("--checkpoints", default=str(C.CHECKPOINTS))
    hf_home: str | None = option("--hf-home", env="HF_HOME", default=None,
                                 help="HF cache holding the base snapshot; the int8 snapshot goes beside it (default: the pins' local_path)")


@dataclass(frozen=True, kw_only=True)
class PinOptions:
    """Make an int8 checkpoint from a BF16 pin and print its new entry (to append to checkpoints-int8.json, append-only)."""

    base_role: str = option("--base-role", help="the BF16 pin's role, or its repo id when the role is prose")
    role: str = option("--role", help="the new int8 pin's role")
    repo: str = option("--repo", help="the new pin's repo id (a local namespace: no hub repo holds these bytes)")
    checkpoints: str = option("--checkpoints", default=str(C.CHECKPOINTS))
    hf_home: str | None = option("--hf-home", env="HF_HOME", default=None)
    by: str = option("--by", default="int8-checkpoint pin", help="resolved_by of the entry")
    at: str = option("--at", help="resolved_at of the entry (UTC, ISO 8601)")
    recipe: str = option("--recipe", default=RECIPE, choices=sorted(RECIPES), help="the int8 recipe (v2 also makes the LM head)")


COMMANDS = {"make": MakeOptions, "pin": PinOptions}


def main(a: MakeOptions | PinOptions) -> int:
    if isinstance(a, PinOptions):
        return F.run(a, RECIPES[a.recipe], "int8-checkpoint")
    entry = next((c for c in C.checkpoint_entries(a.checkpoints) if c.get("role") == a.role), None)
    return F.run(a, recipe_of(entry) if entry is not None else INT8, "int8-checkpoint")
