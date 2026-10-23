"""The exact-int8 routed experts (`engine.int8_vllm.Int8ExpertsMethod`), with no vLLM: each expert GEMM is the plain arm's
linear (`engine.int8_exact`), the same two ops as a dense linear's (`PlainGemm`, which `Int8PlainLinearMethod` is too).

A MoE layer's experts are E (gate_up, down) pairs, each projection an `ExpertSite`: its int8 weight rows [N, K] and float32
scales [N, 1] (views of the layer's ``w13`` [E, 2I, H] and ``w2`` [E, H, I]), and its own `ExpertGemm`, whose ``observe`` a
protocol option wraps as it wraps a dense linear's.  `forward` of x [M, H] bf16 under the router's top-k ids and weights
[M, k] runs every expert in order e = 0 .. E - 1, its gate_up then its down:

    rows(e)  = the tokens t with e in ids[t], ascending (`routes`, a stable sort of the flat ids by expert)
    h        = gate_up(x[rows(e)])                       [m_e, 2I], the plain linear
    a        = RN_bf16(RN32(silu(f32(h[:, :I]))) * f32(h[:, I:]))   [m_e, I], `silu_and_mul`
    y_e      = down(a)                                   [m_e, H], the plain linear
    out[t]   = RN_bf16(sum_j RN32(f32(w[t, j]) * f32(y_{ids[t, j]}[t])), j = 0 .. k - 1 in order, f32 accumulation

A slot of id -1 (`UNROUTED`, vLLM's id for no expert) goes to no expert, its y zero.

Every (expert, projection) is called on every forward, an expert no token routed to on zero rows: an `ExpertGemm` hands a
zero-row call to ``observe`` too (s8 rows [0, K], scales [0], y [0, N]), so the calls of a step are the same E pairs in the
same order whatever the routing, the static layout H1 keeps (`protocol_options.pouw_h1`, one call per (step, linear)); its
zero rows commit as empty slots and no GEMM runs.  Each call's rows are its batch rows (``site.tokens``, set before the
call), so a listener reads which tokens an expert's rows are.  A row's value reads only that row, as a dense linear's, so
an expert's output for a token does not depend on which other tokens it got.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from verity_vllm.engine import int8_exact as E

GATE_UP, DOWN = "gate_up", "down"
#: an expert's projections in the order they run
PROJECTIONS = (GATE_UP, DOWN)
#: the top-k id of a slot routed to no expert (vLLM's, as its expert map has it; its profile run routes every slot so)
UNROUTED = -1


class PlainGemm:
    """The plain arm's call on one int8 weight (``layer.weight`` [N, K], ``layer.weight_scale`` [N, 1]): `int8_ops.linear`,
    or `int8_ops.linear_zy` over ``zy_rows`` slots, its served values then handed to ``observe``."""

    int8_arm = E.ARM
    #: T, the slots of each call's z/y records (`int8_ops.linear_zy`), or None to serve through ``scaled_mm``
    zy_rows: int | None = None
    #: whether a call of no rows still reaches ``observe`` (with no rows)
    observe_empty = False

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.calls = 0

    def apply(self, layer: Any, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        from verity_vllm.engine import int8_ops
        if bias is not None:
            raise ValueError(f"{self.prefix}: the int8 plain arm has no bias")
        if x.dtype != torch.bfloat16:
            raise ValueError(f"{self.prefix}: activations are {x.dtype}, the plain arm quantizes bf16")
        self.calls += 1
        lead, k = x.shape[:-1], x.shape[-1]
        n = layer.weight.shape[0]
        x2 = x.reshape(-1, k)
        if x2.shape[0] == 0:
            y = x.new_empty(*lead, n)
            if self.observe_empty:
                self.observe(layer, x2.new_empty((0, k), dtype=torch.int8), x2.new_empty(0, dtype=torch.float32),
                             y.reshape(0, n))
            return y
        rows = self.zy_rows
        if rows is not None and x2.shape[0] <= rows:
            xq, s_a, y, zy = int8_ops.linear_zy(x2, layer.weight, layer.weight_scale, rows)
            self.observe(layer, xq, s_a, y, zy)
        else:
            xq, s_a, y = int8_ops.linear(x2, layer.weight, layer.weight_scale)
            self.observe(layer, xq, s_a, y)
        return y.reshape(*lead, n)

    def observe(self, layer: Any, xq: torch.Tensor, s_a: torch.Tensor, y: torch.Tensor, z: torch.Tensor | None = None) -> None:
        """A served call's values, after both ops: its s8 rows xq [M, K], their f32 scales s_a [M], y [M, N] bf16 and, under
        ``zy_rows``, the z/y records [units, `int8_exact.ZY_BYTES`] ``scaled_mm_zy`` wrote. A protocol option wraps it; it
        must not change them."""


class ExpertGemm(PlainGemm):
    """One expert projection's call: the plain arm's, and on zero rows too (the static layout)."""

    observe_empty = True


@dataclass(eq=False)
class ExpertSite:
    """Expert ``expert``'s ``projection`` of the MoE layer ``prefix``: its int8 weight rows, their scales and its call."""

    prefix: str
    expert: int
    projection: str
    weight: torch.Tensor
    weight_scale: torch.Tensor
    quant_method: ExpertGemm = field(init=False)
    #: the batch rows (token indices, int64) of the latest call's rows, in order
    tokens: torch.Tensor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.quant_method = ExpertGemm(self.name)

    @property
    def name(self) -> str:
        return f"{self.prefix}.{self.expert}.{self.projection}"


def sites(prefix: str, w13: torch.Tensor, s13: torch.Tensor, w2: torch.Tensor, s2: torch.Tensor) -> list[ExpertSite]:
    """The layer's sites in the order `forward` calls them, (e, gate_up) then (e, down) for e = 0 .. E - 1: w13 [E, 2I, H]
    and w2 [E, H, I] int8, s13 [E, 2I, 1] and s2 [E, H, 1] float32."""
    if w13.shape[0] != w2.shape[0] or w13.shape[2] != w2.shape[1] or w13.shape[1] != 2 * w2.shape[2]:
        raise ValueError(f"{prefix}: w13 {tuple(w13.shape)} and w2 {tuple(w2.shape)} are not E gate_up [2I, H] and down [H, I]")
    out = []
    for e in range(w13.shape[0]):
        out += [ExpertSite(prefix, e, GATE_UP, w13[e], s13[e].reshape(-1, 1)),
                ExpertSite(prefix, e, DOWN, w2[e], s2[e].reshape(-1, 1))]
    return out


def routes(topk_ids: torch.Tensor, experts: int) -> list[torch.Tensor]:
    """Each expert's routed positions in the flat [M * k] top-k table, ascending (so its tokens ascend, position // k).
    A slot of id `UNROUTED` goes to no expert; a token routed twice to one expert, or to an id outside [-1, experts), is
    refused."""
    m, k = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int64)
    if flat.numel() and (int(flat.min()) < UNROUTED or int(flat.max()) >= experts):
        raise ValueError(f"a top-k id outside the layer's {experts} experts")
    if k > 1:
        s = topk_ids.sort(dim=1).values
        if bool(((s.diff(dim=1) == 0) & (s[:, 1:] != UNROUTED)).any()):
            raise ValueError("a token routed to one expert twice")
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat - UNROUTED, minlength=experts + 1).tolist()
    return list(order.split(counts))[1:]


def silu_and_mul(h: torch.Tensor) -> torch.Tensor:
    """a [m, I] bf16 of h [m, 2I] bf16: RN_bf16(RN32(silu(f32(gate))) * f32(up)), gate the first half, up the second."""
    i = h.shape[-1] // 2
    return (torch.nn.functional.silu(h[:, :i].float()) * h[:, i:].float()).to(torch.bfloat16)


def forward(layer_sites: list[ExpertSite], x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """The routed experts' output [M, H] bf16 of x [M, H] bf16 (see the module doc); every site is called once, in order."""
    m, k = topk_ids.shape
    if x.dim() != 2 or x.shape[0] != m or topk_weights.shape != topk_ids.shape:
        raise ValueError(f"x {tuple(x.shape)}, weights {tuple(topk_weights.shape)} and ids {tuple(topk_ids.shape)} are not "
                         "[M, H], [M, k] and [M, k]")
    experts = len(layer_sites) // 2
    hidden = layer_sites[1].weight.shape[0] if layer_sites else x.shape[1]
    slots = x.new_zeros((m * k, hidden))
    for e, pos in enumerate(routes(topk_ids, experts)):
        gate_up, down = layer_sites[2 * e], layer_sites[2 * e + 1]
        tokens = pos.to(x.device) // k
        gate_up.tokens = down.tokens = tokens
        h = gate_up.quant_method.apply(gate_up, x.index_select(0, tokens))
        y = down.quant_method.apply(down, silu_and_mul(h))
        slots[pos.to(x.device)] = y
    per = slots.view(m, k, hidden)
    w = topk_weights.float()
    acc = torch.zeros((m, hidden), dtype=torch.float32, device=x.device)
    for j in range(k):
        acc = acc + w[:, j:j + 1] * per[:, j].float()
    return acc.to(torch.bfloat16)


def reference(x, topk_weights, topk_ids, w13, s13, w2, s2):
    """`forward` in NumPy, from the module doc's definitions alone (x, the weights and scales as NumPy arrays, x and the
    result bf16 as float32 values): the test oracle."""
    import numpy as np

    def bf16(v):
        b = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32).astype(np.uint64)
        b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16 << 16
        return b.astype(np.uint32).view(np.float32)

    def linear(a, w, s):
        a = a.astype(np.float32)
        s_a = (np.maximum(np.abs(a).max(axis=1), np.float32(E.SCALE_FLOOR)) / np.float32(E.TOP)).astype(np.float32)
        q = np.clip(np.rint((a / s_a[:, None]).astype(np.float32)), -E.TOP, E.TOP).astype(np.int64)
        y = (q @ w.astype(np.int64).T).astype(np.float32)
        return bf16((y * s_a[:, None]).astype(np.float32) * s.reshape(1, -1).astype(np.float32))

    m, k = topk_ids.shape
    hidden = w2.shape[1]
    out = np.zeros((m, hidden), dtype=np.float32)
    ys = np.zeros((m, k, hidden), dtype=np.float32)
    for t in range(m):
        for j in range(k):
            e = int(topk_ids[t, j])
            if e == UNROUTED:
                continue
            h = linear(x[t:t + 1], w13[e], s13[e])
            i = h.shape[1] // 2
            g = h[:, :i].astype(np.float32)
            silu = (g / (np.float32(1) + np.exp(-g, dtype=np.float32))).astype(np.float32)
            ys[t, j] = linear(bf16(silu * h[:, i:]), w2[e], s2[e])[0]
    for j in range(k):
        out = (out + topk_weights[:, j:j + 1].astype(np.float32) * ys[:, j]).astype(np.float32)
    return bf16(out)


__all__ = ["DOWN", "ExpertGemm", "ExpertSite", "GATE_UP", "PROJECTIONS", "PlainGemm", "UNROUTED", "forward", "reference",
           "routes", "silu_and_mul", "sites"]
