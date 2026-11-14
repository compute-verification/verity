"""The training rules' attention (`rules.training`): at a compute capability whose flash attention is captured, torch's
`_scaled_dot_product_flash_attention` and its backward as that GPU's kernels compute them (`FlashSdpa`, `FlashSdpaBackward`);
in a structure derivation (`structure`), each the head Definition at a stand-in block size instead."""

from __future__ import annotations

import operator

import numpy as np
import torch
from verity.circuits.defs import bind
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.definitions.registry import cublas_bf16 as CB
from verity.definitions.registry import fa2_train as FT
from verity.definitions.registry import fa2_train_bwd as FTB

from verity_vllm.program.frontend.rules.training_emit import (
    BF16,
    F32,
    _batch,
    _shape,
    _TrainRule,
    aten,
)
from verity_vllm.program.frontend.rules.training_views import _info
from verity_vllm.program.frontend.strided_views import _tensor_type

#: the op's outputs past O and the LSE: the backward's state (cumulative sequence lengths, maxima, the RNG state, the debug mask)
_STATE = 7
#: a structure derivation's block size, which no capture fits and no cell of which is read: the forward unsplit over key blocks of
#: 16, the backward over blocks of 16 rows and 16 keys and one dQ buffer (the smallest the heads bind at)
STAND_IN = 16


def _stand_in(defn, **statics):
    """`defn` bound at `statics`, the structure's stand-in head, and None; or None and why it refuses the shape."""
    try:
        return bind(defn, **statics), None
    except ValueError as e:
        return None, f"numerics_unregistered: the structure's stand-in head: {e} (derive the structure at another T)"


def _other_scale(node, D: int) -> str | None:
    """Why the op's scale is not the captures', D^-0.5 as the f32 the kernels read; None if it is."""
    scale = node.kwargs.get("scale")
    if scale is not None and np.float32(scale) != np.float32(D ** -0.5):
        return f"numerics_unregistered: scale {scale} (the capture's is the default, D^-0.5)"
    return None


class FlashSdpa(_TrainRule):
    """`_scaled_dot_product_flash_attention(q, k, v, 0, True)` of bf16 values [B, H, T, D] at the default scale -> one batch over
    the B * H heads of the head Definition torch's kernels compute at that shape on the capture's GPU (`fa2_train.train_sdpa`:
    `Fa2TrainHead_v1` unsplit, `Fa2TrainSplitHead_v1` in four splits), O (bf16) and the LSE (f32) of each head.  The rule claims
    every call at ``cc`` and refuses one it can't lower that way, so that no other rule lowers it with torch CPU's arithmetic: a
    shape the capture doesn't cover, a non-causal call, dropout, another scale, a debug mask, an operand that is not a bf16 value
    and a read of the backward's state by anything but the op's backward (which, without dropout, reads none of it).
    `structure`: a structure derivation's, whose cells nothing reads: the head at any shape it binds at with key blocks of
    `STAND_IN`, unsplit (`Fa2TrainHead_v1{BN=16}`), and no capture read."""

    def __init__(self, cc: tuple[int, int], structure: bool = False):
        if tuple(cc) != CB.TRAIN_CC:
            raise ValueError(f"FlashSdpa: cc {tuple(cc)} has no captured flash attention (the captures are at {CB.TRAIN_CC})")
        self.cc, self.structure = f"cc{cc[0]}.{cc[1]}", structure
        self.info = _info(f"FlashSdpa_{self.cc}" + ("_structure" if structure else ""),
                          "aten._scaled_dot_product_flash_attention of bf16 values",
                          "q, k and v are bf16 values of one shape [B, H, T, D] that " + (
                              f"the head binds at with key blocks of {STAND_IN}" if structure else "`fa2_train.TRAIN_SDPA` lists"),
                          "causal, no dropout, the default scale, no debug mask; only O and the LSE read, but by the op's backward",
                          "one batch of heads")

    def _head(self, B: int, H: int, T: int, D: int):
        if self.structure:
            return _stand_in(FT.Fa2TrainHead, T=T, D=D, BN=STAND_IN)
        if (B, H, T, D) not in FT.TRAIN_SDPA:
            return None, (f"numerics_unregistered: no capture fits torch's flash attention forward at [B, H, T, D] = "
                          f"{[B, H, T, D]} (fa2_train.TRAIN_SDPA: {sorted(FT.TRAIN_SDPA)})")
        return FT.train_sdpa(B, H, T, D), None

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
        if len(shape) != 4:
            return f"numerics_unregistered: q, k and v of shape {list(shape)} (the kernels' are [B, H, T, D])"
        if why := self._head(*shape)[1]:
            return why
        if not causal:
            return "numerics_unregistered: a non-causal call (the capture is causal)"
        if dropout:
            return "side_input: dropout reads the RNG (a random op)"
        if debug:
            return "numerics_unregistered: a debug mask"
        if why := _other_scale(node, shape[3]):
            return why
        for u in node.users:
            if u.target is not operator.getitem:
                return f"{u.name} reads the op's whole result"
            if u.args[1] >= 2 and any(w.target is not aten._scaled_dot_product_flash_attention_backward.default for w in u.users):
                return (f"numerics_unregistered: {u.name} reads output {u.args[1]}, the backward's state, outside the op's "
                        "backward")
        return True

    def emit(self, node, ctx):
        B, H, T, D = _shape(ctx, node.args[0])
        head = self._head(B, H, T, D)[0]
        heads = Array(B * H, Array(T, Array(D, BF16)))
        q, k, v = (ctx.coll(a).reshape(heads) for a in node.args[:3])
        y = _batch(ctx, node, head, q, k, v, axes=(0, 0, 0))
        o = Coll(_tensor_type([B, H, T, D], 16), concat([y[i][0].refs for i in range(B * H)]))
        lse = Coll(_tensor_type([B, H, T], 32), concat([y[i][1].refs for i in range(B * H)]))
        ctx.define(node, [o, lse] + [None] * _STATE)
        ctx.record(self, node, {"B": B, "H": H, "T": T, "D": D,
                                "route": "stand-in" if self.structure else FT.TRAIN_SDPA[(B, H, T, D)][0]},
                   emitted=f"{head.id} x{B * H}")


class FlashSdpaBackward(_TrainRule):
    """`_scaled_dot_product_flash_attention_backward(dO, q, k, v, O, LSE, None, None, T, T, 0, True, ..)` of bf16 values
    [B, H, T, D] and the f32 LSE [B, H, T] at the default scale, in a step under deterministic algorithms -> one batch over the
    B * H heads of the head Definition torch's backward kernels compute at that shape on the capture's GPU
    (`fa2_train_bwd.train_sdpa_backward`, `Fa2TrainBwdHead_v1`): dQ, dK and dV (bf16) of each head, from the op's own operands.
    The rule claims every call at ``cc`` and refuses one it can't lower that way: a step not under deterministic algorithms (the
    kernels then add a row's dQ parts in the order the card schedules its thread blocks, no function of the operands), a shape the
    capture doesn't cover, a variable-length call, a non-causal call, dropout, another scale, and an operand that is not a bf16
    value (the LSE an f32 one). `structure`: a structure derivation's, whose cells nothing reads: the head at any shape it binds
    at with blocks of `STAND_IN` rows and keys and one dQ buffer (`Fa2TrainBwdHead_v1{BM=16,BN=16,NS=1}`), no capture read and
    deterministic algorithms not required."""

    def __init__(self, cc: tuple[int, int], deterministic: bool = False, structure: bool = False):
        if tuple(cc) != CB.TRAIN_CC:
            raise ValueError(f"FlashSdpaBackward: cc {tuple(cc)} has no captured flash attention (the captures are at {CB.TRAIN_CC})")
        if deterministic and structure:
            raise ValueError("FlashSdpaBackward: a structure derivation reads no cell, so no setting of the kernels")
        self.cc, self.deterministic, self.structure = f"cc{cc[0]}.{cc[1]}", deterministic, structure
        self.info = _info(f"FlashSdpaBackward_{self.cc}" + ("_deterministic" if deterministic else "")
                          + ("_structure" if structure else ""),
                          "aten._scaled_dot_product_flash_attention_backward of bf16 values",
                          "dO, q, k, v and O are bf16 values of one shape [B, H, T, D] that " + (
                              f"the head binds at with blocks of {STAND_IN}" if structure else "`fa2_train_bwd.TRAIN_SDPA_BWD` lists")
                          + ", the LSE an f32 value [B, H, T]",
                          "a dense call: causal, no dropout, the default scale",
                          "a structure derivation" if structure else
                          "the step runs under deterministic algorithms; without them every call is refused",
                          "one batch of heads")

    def _head(self, B: int, H: int, T: int, D: int):
        if self.structure:
            return _stand_in(FTB.Fa2TrainBwdHead, T=T, D=D, BM=STAND_IN, BN=STAND_IN, NS=1)
        if (B, H, T, D) not in FTB.TRAIN_SDPA_BWD:
            return None, (f"numerics_unregistered: no capture fits torch's flash attention backward at [B, H, T, D] = "
                          f"{[B, H, T, D]} (fa2_train_bwd.TRAIN_SDPA_BWD: {sorted(FTB.TRAIN_SDPA_BWD)})")
        return FTB.train_sdpa_backward(B, H, T, D), None

    def targets(self):
        return {aten._scaled_dot_product_flash_attention_backward.default}

    def applies(self, node, ctx):
        do, q, k, v, o, lse, cum_q, cum_k, max_q, max_k, dropout, causal = node.args[:12]
        for a, dtype in ((do, torch.bfloat16), (q, torch.bfloat16), (k, torch.bfloat16), (v, torch.bfloat16), (o, torch.bfloat16),
                         (lse, torch.float32)):
            if not isinstance(a, torch.fx.Node) or not isinstance(ctx.val(a), Coll):
                return f"{getattr(a, 'name', a)} is not a value"
            if ctx.dtype_of(a) != dtype:
                return (f"numerics_unregistered: {a.name} has dtype {ctx.dtype_of(a)} (the captured kernels read bf16 operands and "
                        "an f32 LSE)")
        shape = tuple(_shape(ctx, q))
        if any(tuple(_shape(ctx, a)) != shape for a in (do, k, v, o)) or tuple(_shape(ctx, lse)) != shape[:3]:
            return (f"dO, q, k, v, O and the LSE of shapes {[list(_shape(ctx, a)) for a in (do, q, k, v, o, lse)]} (only one "
                    "[B, H, T, D], and its [B, H, T])")
        if len(shape) != 4:
            return f"numerics_unregistered: dO, q, k, v and O of shape {list(shape)} (the kernels' are [B, H, T, D])"
        if why := self._head(*shape)[1]:
            return why
        if cum_q is not None or cum_k is not None or (max_q, max_k) != (shape[2], shape[2]):
            return "numerics_unregistered: a variable-length call (the capture is dense)"
        if not causal:
            return "numerics_unregistered: a non-causal call (the capture is causal)"
        if dropout:
            return "side_input: dropout reads the RNG (a random op)"
        if why := _other_scale(node, shape[3]):
            return why
        if not (self.deterministic or self.structure):
            return ("side_input: the step does not run under deterministic algorithms, so the kernels add a row's dQ parts in the "
                    "order the card schedules its thread blocks: call torch.use_deterministic_algorithms(True) before the step")
        for u in node.users:
            if u.target is not operator.getitem:
                return f"{u.name} reads the op's whole result"
        return True

    def emit(self, node, ctx):
        B, H, T, D = _shape(ctx, node.args[1])
        head = self._head(B, H, T, D)[0]
        heads = Array(B * H, Array(T, Array(D, BF16)))
        do, q, k, v, o = (ctx.coll(a).reshape(heads) for a in node.args[:5])
        lse = ctx.coll(node.args[5]).reshape(Array(B * H, Array(T, F32)))
        y = _batch(ctx, node, head, do, q, k, v, o, lse, axes=(0,) * 6)
        ctx.define(node, [Coll(_tensor_type([B, H, T, D], 16), concat([y[i][j].refs for i in range(B * H)])) for j in range(3)])
        ctx.record(self, node, {"B": B, "H": H, "T": T, "D": D, **({"route": "stand-in"} if self.structure else {"deterministic": True})},
                   emitted=f"{head.id} x{B * H}")


__all__ = ["STAND_IN", "FlashSdpa", "FlashSdpaBackward"]
