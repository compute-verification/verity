"""Weights vLLM makes at load from a checkpoint tensor by more than a copy, as the weights of record hold them, for the fields whose
Build role (derive-report `weight_roles`) names the rule:

  * `mhc.fn_broadcast` (DeepSeek-V4's first layer, `hc_attn_fn_broadcast`): `finalize_mhc_broadcast_weights` sums the layer's
    `hc_attn_fn` [M, HC*H] over its HC streams, `fn.view(-1, HC, H).sum(dim=1)`, in f32 on the GPU.  A reduction of at most 4 over a
    dimension that is not the innermost gives each output one thread, which adds the streams in order onto 0; a longer one is split
    across threads and has no rule here.
  * `dsv4.attn_sink`: the sink is `torch.full((NH,), -inf)` at the padded head count, and `load_weights` copies the rank's heads of the
    checkpoint's sink into its first slots.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verity_vllm.check.tp_shards import rows_of

NEG_INF_F32 = 0xFF800000


def compose(live: str, shape: list[int], bits: int, role: dict[str, Any] | None, ck: Any, world: int,
            rank: int) -> tuple[np.ndarray, dict[str, Any]] | None:
    """The served words of the field `live` (its served name) under its Build `role`, and the note; None when no rule here applies."""
    kind = (role or {}).get("role")
    if kind == "mhc.fn_broadcast":
        src, hc, h = live.removesuffix("_broadcast"), int(role["HC"]), int(role["H"])
        if bits != 32 or hc > 4 or not ck.has(src) or ck.dtype(src) != "F32" or ck.tensors[src][2] != [shape[0], hc * h] or shape[1:] != [h]:
            raise ValueError(f"{live}: the sum over {hc} streams of {src} has no rule for a {shape} x Value<{bits}> field")
        streams = ck.array(src).reshape(-1, hc, h)
        out = np.zeros(streams.shape[::2], dtype=np.float32)
        for i in range(hc):
            out = out + streams[:, i]
        return out.view(np.uint32), {"sources": [src], "cast": None, "shard": f"rank {rank}/{world}: replicated",
                                     "rule": f"vLLM's {src.rsplit('.', 1)[-1]}.view(-1, {hc}, {h}).sum(dim=1) in f32, the streams added in order"}
    if kind == "dsv4.attn_sink":
        nh, n = int(role["NH"]), ck.tensors[live][2][0] if ck.has(live) else 0
        lo, hi = rows_of(n, world, rank)
        if bits != 32 or ck.dtype(live) != "F32" or shape != [nh] or hi - lo > nh:
            raise ValueError(f"{live}: the rank's {hi - lo} sink heads padded to {nh} have no rule for a {shape} x Value<{bits}> field")
        out = np.full(nh, NEG_INF_F32, dtype=np.uint32)
        out[:hi - lo] = ck.array(live, rows=(lo, hi)).view(np.uint32)
        return out, {"sources": [live], "cast": None, "shard": f"rank {rank}/{world}: heads [{lo}, {hi}) of {n}",
                     "rule": f"the rank's heads, then -inf to the padded {nh} heads (vLLM's attn_sink)"}
    return None
