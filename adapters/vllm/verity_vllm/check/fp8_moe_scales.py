"""The block scales of a TP rank's FP8 experts, composed from the checkpoint's: what `compose` records for `w13_weight_scale_inv` and
`w2_weight_scale_inv`.

vLLM d9105ea80 splits each expert's intermediate width I over the W ranks.  When the rank's width `I_r = I / W` is not a multiple of the
checkpoint's block G, the rank's scales live on a finer g-block grid (`refine_fp8_moe_block_shape`, fused_moe/oracle/fp8.py:272;
`program.model.fp8_moe_block`), and `RoutedExperts.weight_loader` (fused_moe/routed_experts.py:836-857) fills it by repeating each
checkpoint scale f = G / g times along both dimensions (`repeat_interleave`) before the rank's slice is cut as for any expert tensor
(`_load_w13`: rows of each of gate and up; `_load_w2`: columns of down).  So every served scale is one checkpoint word:

    w13, gate half (up likewise):  ws[j][k] = gate_ws[(r · I_r/g + j) // f][k // f]      j < I_r/g, k < H/g
    w2:                            ws[n][k] = down_ws[n // f][(r · I_r/g + k) // f]       n < H/g, k < I_r/g

The weights of record hold the refined tensor, not the checkpoint's words plus this map: `weights_of_record.check` compares each field's
digest with the served parameter's, and the served `w13_weight_scale_inv` / `w2_weight_scale_inv` are the refined tensors (the Triton MoE
backends, the only ones that take a refined grid, leave them as loaded; a backend that re-lays them out fails that comparison by name).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def on_rank_grid(parts: list[np.ndarray], srcs: list[str], which: str, ck: Any, world: int, note: dict[str, Any]) -> list[np.ndarray]:
    """One expert's checkpoint parts `srcs` (served width) for the `which` stack ("w13" or "w2"), with block scales
    (`<part>.weight_scale_inv`) moved onto the rank's grid; the caller then cuts the rank's slice.  Other parts, and scales already on
    the rank's grid (g = G), are returned unchanged."""
    stem, suffix = srcs[0].rsplit(".", 1) if srcs else ("", "")
    if suffix != "weight_scale_inv":
        return parts
    from verity_vllm.program.model import fp8_moe_block
    bs = (ck.config.get("quantization_config") or {}).get("weight_block_size")
    if not (isinstance(bs, (list, tuple)) and len(bs) == 2 and bs[0] == bs[1]):
        raise KeyError(f"{srcs[0]}: block scales, but config.json's quantization_config has no square weight_block_size (got {bs!r})")
    G = int(bs[0])
    weight = f"{stem}.weight"
    if not ck.has(weight):
        raise KeyError(f"{srcs[0]}: no {weight} in the checkpoint to size the expert's block grid from")
    N, K = ck.tensors[weight][2]
    I, H = (N, K) if which == "w13" else (K, N)
    if I % G or H % G or I % world:
        raise ValueError(f"{srcs[0]}: expert width {I} and hidden {H} must be multiples of the block {G}, and the width of the world {world}")
    for s, p in zip(srcs, parts):
        if list(p.shape) != [N // G, K // G]:
            raise ValueError(f"{s}: scale shape {list(p.shape)} is not the [{N // G}, {K // G}] blocks of {weight} [{N}, {K}]")
    g = fp8_moe_block(G, I // world, H)
    f = G // g
    if f == 1:
        return parts
    note["refine"] = f"block {G} -> {g}: each scale repeated {f}x along both dims before the rank slice (vLLM refine_fp8_moe_block_shape)"
    return [np.repeat(np.repeat(p, f, axis=0), f, axis=1) for p in parts]
