"""Int8 checkpoints for the exact-int8 linears, made offline from a pinned BF16 checkpoint: the recipe ``verity-int8-channel63/v1``.

    verity-vllm int8-checkpoint make --role ROLE [--hf-home DIR]        re-make a registered int8 pin and check every file's sha256
    verity-vllm int8-checkpoint pin --base-role ROLE --role ROLE --repo ID [--hf-home DIR]
                                                                          make one and print its new manifest entry

The weights the plain arm reads (``engine.int8_exact``, ``TargetProfile.int8_exact = "plain"``): int8 with every entry in
[-63, 63], one float32 scale per output channel.  Like ``engine.fp8_checkpoint``'s recipe, on its container (shard layout,
header, index, copied files, pinned revision), it is data-free and a pure function of the base checkpoint's bytes; the linear
weights are the same tensors (every 2-D tensor none of whose dimensions is ``vocab_size``).  For every such W [N, K]:

* W is read exactly as float32 (BF16, F16 and F32 widen exactly);
* for every row n: a = max |W[n, :]|, and s = a / 63 in float32 (IEEE round-to-nearest-even); s = 1 when a = 0;
* the stored weight is q = clamp(rint(W / s), -63, 63): the float32 quotient (round-to-nearest-even), rounded to the nearest
  integer, ties to even, as int8;
* stored as ``<name>`` I8 [N, K] and ``<module>.weight_scale`` F32 [N, 1]; the dequantized weight is q * s.

N and K must be multiples of 8 (``torch._int_mm``'s shape rule on CUDA), else the checkpoint is refused whole.  config.json gets
``quantization_config = {"modules_to_not_convert": ["lm_head"], "quant_method": "verity_int8", "weight_max": 63}``, the vLLM
method ``engine.int8_vllm`` registers.  The pin's ``int8_of`` names its BF16 base.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verity_vllm import config as C
from verity_vllm.config import option
from verity_vllm.engine import fp8_checkpoint as F
from verity_vllm.engine import int8_exact as E
from verity_vllm.program.frontend.target_profile import TargetProfile

RECIPE = "verity-int8-channel63/v1"
QUANTIZATION = {"quant_method": TargetProfile.INT8_EXACT_QUANTIZATION, "weight_max": E.W_MAX}


def channel_scales(w: np.ndarray) -> np.ndarray:
    """s [N, 1] float32 of a float32 [N, K]: max |W| per row / 63, 1 for an all-zero row."""
    amax = np.abs(w).max(axis=1, keepdims=True)
    if not np.isfinite(amax).all():
        raise ValueError("non-finite weight")
    s = (amax / np.float32(E.W_MAX)).astype(np.float32)
    return np.where(amax == 0, np.float32(1.0), s)


def quantize(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """q [N, K] int8 of a float32 [N, K] under its row scales `s` [N, 1]."""
    return np.clip(np.rint(w / s), -E.W_MAX, E.W_MAX).astype(np.int8)


INT8 = F.Recipe(id=RECIPE, weight_dtype="I8", scale_suffix="weight_scale", rows_per_scale=1, scale_shape=lambda n, k: [n, 1],
                refusal=lambda n, k: "not a multiple of 8 on both axes" if n % 8 or k % 8 else None,
                scales=channel_scales, quantize=quantize, quantization=QUANTIZATION, of="int8_of")


@dataclass(frozen=True, kw_only=True)
class MakeOptions:
    """Re-make a registered int8 pin (`recipe` in checkpoints.json) from its BF16 pin and check every file's sha256."""

    role: str = option("--role", help="the int8 pin's role in checkpoints.json")
    checkpoints: str = option("--checkpoints", default=str(C.CHECKPOINTS))
    hf_home: str | None = option("--hf-home", env="HF_HOME", default=None,
                                 help="HF cache holding the base snapshot; the int8 snapshot goes beside it (default: the pins' local_path)")


@dataclass(frozen=True, kw_only=True)
class PinOptions:
    """Make an int8 checkpoint from a BF16 pin and print its new checkpoints.json entry (to append; the manifest is append-only)."""

    base_role: str = option("--base-role", help="the BF16 pin's role, or its repo id when the role is prose")
    role: str = option("--role", help="the new int8 pin's role")
    repo: str = option("--repo", help="the new pin's repo id (a local namespace: no hub repo holds these bytes)")
    checkpoints: str = option("--checkpoints", default=str(C.CHECKPOINTS))
    hf_home: str | None = option("--hf-home", env="HF_HOME", default=None)
    by: str = option("--by", default="int8-checkpoint pin", help="resolved_by of the entry")
    at: str = option("--at", help="resolved_at of the entry (UTC, ISO 8601)")


COMMANDS = {"make": MakeOptions, "pin": PinOptions}


def main(a: MakeOptions | PinOptions) -> int:
    return F.run(a, INT8, "int8-checkpoint")
