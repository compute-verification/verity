"""The training rules' attention (`rules.training`): at a compute capability whose flash attention forward is captured, torch's
`_scaled_dot_product_flash_attention` as that GPU's kernels compute it (`FlashSdpa`)."""

from __future__ import annotations

import operator

import torch

from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity.ml.definitions.registry import fa2_train as FT
from verity_vllm.program.frontend.rules.training_emit import BF16, _batch, _shape, _TrainRule, aten
from verity_vllm.program.frontend.rules.training_views import _info
from verity_vllm.program.frontend.strided_views import _tensor_type

#: the op's outputs past O and the LSE: the backward's state (cumulative sequence lengths, maxima, the RNG state, the debug mask)
_STATE = 7


class FlashSdpa(_TrainRule):
    """`_scaled_dot_product_flash_attention(q, k, v, 0, True)` of bf16 values [B, H, T, D] at the default scale -> one batch over
    the B * H heads of the head Definition torch's kernels compute at that shape on the capture's GPU (`fa2_train.train_sdpa`:
    `Fa2TrainHead_v1` unsplit, `Fa2TrainSplitHead_v1` in four splits), O (bf16) and the LSE (f32) of each head.  The rule claims
    every call at ``cc`` and refuses one it can't lower that way, so that no other rule lowers it with torch CPU's arithmetic: a
    shape the capture doesn't cover, a non-causal call, dropout, another scale, a debug mask, an operand that is not a bf16 value
    and a read of the backward's state (the backward is not lowered)."""

    def __init__(self, cc: tuple[int, int]):
        if tuple(cc) != CB.TRAIN_CC:
            raise ValueError(f"FlashSdpa: cc {tuple(cc)} has no captured flash attention (the captures are at {CB.TRAIN_CC})")
        self.cc = f"cc{cc[0]}.{cc[1]}"
        self.info = _info(f"FlashSdpa_{self.cc}", "aten._scaled_dot_product_flash_attention of bf16 values",
                          "q, k and v are bf16 values of one shape [B, H, T, D] that `fa2_train.TRAIN_SDPA` lists",
                          "causal, no dropout, the default scale, no debug mask; only O and the LSE read",
                          "one batch of heads")

    def targets(self):
        return {aten._scaled_dot_product_flash_attention.default}

    def applies(self, node, ctx):
        q, k, v = node.args[:3]
        dropout, causal, debug = (list(node.args[3:6]) + [0.0, False, False][len(node.args[3:6]):])
        for a in (q, k, v):
            if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll):
                return f"{getattr(a, 'name', a)} is not a value"
            if ctx.dtype_of(a) != torch.bfloat16:
                return f"numerics_unregistered: {a.name} has dtype {ctx.dtype_of(a)} (the captured kernels are bf16's)"
        shape = tuple(_shape(ctx, q))
        if tuple(_shape(ctx, k)) != shape or tuple(_shape(ctx, v)) != shape:
            return f"q, k and v of shapes {list(shape)}, {list(_shape(ctx, k))}, {list(_shape(ctx, v))} (only one shape)"
        if len(shape) != 4 or shape not in FT.TRAIN_SDPA:
            return (f"numerics_unregistered: no capture fits torch's flash attention forward at [B, H, T, D] = {list(shape)} "
                    f"(fa2_train.TRAIN_SDPA: {sorted(FT.TRAIN_SDPA)})")
        if not causal:
            return "numerics_unregistered: a non-causal call (the capture is causal)"
        if dropout:
            return "side_input: dropout reads the RNG (a random op)"
        if debug:
            return "numerics_unregistered: a debug mask"
        scale = node.kwargs.get("scale")
        if scale is not None and float(scale) != shape[3] ** -0.5:
            return f"numerics_unregistered: scale {scale} (the capture's is the default, D^-0.5)"
        for u in node.users:
            if u.target is not operator.getitem:
                return f"{u.name} reads the op's whole result"
            if u.args[1] >= 2 and u.users:
                return (f"numerics_unregistered: {u.name} reads output {u.args[1]}, the backward's state (the backward is not "
                        "lowered)")
        return True

    def emit(self, node, ctx):
        B, H, T, D = _shape(ctx, node.args[0])
        head = FT.train_sdpa(B, H, T, D)
        heads = Array(B * H, Array(T, Array(D, BF16)))
        q, k, v = (ctx.coll(a).reshape(heads) for a in node.args[:3])
        y = _batch(ctx, node, head, q, k, v, axes=(0, 0, 0))
        o = Coll(_tensor_type([B, H, T, D], 16), concat([y[i][0].refs for i in range(B * H)]))
        lse = Coll(_tensor_type([B, H, T], 32), concat([y[i][1].refs for i in range(B * H)]))
        ctx.define(node, [o, lse] + [None] * _STATE)
        ctx.record(self, node, {"B": B, "H": H, "T": T, "D": D, "route": FT.TRAIN_SDPA[(B, H, T, D)][0]},
                   emitted=f"{head.id} x{B * H}")


__all__ = ["FlashSdpa"]
