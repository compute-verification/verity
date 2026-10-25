"""A minimal PyTorch serving loop on Verity's stack-neutral serving interface (`verity.ml.serving.interface`), with no vLLM.

* **The model** (:class:`Model`): a two-layer W8A8 MLP of width ``H``, ``model.layers.{0,1}.mlp.{up_proj,down_proj}``. Each
  linear's weights are s8 codes with one f32 scale per output channel; each token row is quantized per row to s8 codes and
  an f32 scale (its absmax / 127); and the linear is ``bf16((f32(x_q . w_q) * sx) * sw)`` in torch on CPU: an exact int32
  sum, two f32 multiplies (the token's scale first), then the bf16 cast (round to nearest even). That is
  `ScaledMmS8Coordinate_v1{K=H, DOT=BlackwellS8ImmaDot32_v1}`'s function (`verity.ml.circuits.ops.gemm`, the s8 binding of
  sm_120), the workload's one template: each output coordinate of each linear at each token row is one unit. The ReLU
  between the linears and each row's quantization are not units: their outputs are committed as the next linear's inputs.
* **Serving** (:class:`TorchLoop`, an `interface.Stack`): each request is one token, embedded and run through the MLP for
  ``steps`` decode steps, step s + 1 reading step s's output. Every linear's units are recorded into the window as they are
  computed (`ServingWindow.record`), with the request, step, layer and op path they serve. ``tamper``: a stack whose
  committed outputs are not its model's (bit 0 of every output word flipped before it is recorded), which the proof of any
  drawn unit must refuse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as G
from verity.ml.serving import interface as I

LINEARS = ("up_proj", "down_proj")


def template(k: int):
    """The workload's template: one output coordinate of a W8A8 linear over ``k`` inputs, sm_120's s8 binding."""
    return bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM120_S8)


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row s8 codes and f32 scales of ``x`` (rows, K): scale absmax / 127 (1 for a zero row), codes round(x / scale)."""
    x = x.to(torch.float32)
    amax = x.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 127, torch.ones_like(amax))
    return torch.round(x / scale[:, None]).clamp(-128, 127).to(torch.int8), scale


def linear(xq: torch.Tensor, sx: torch.Tensor, wq: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """The W8A8 linear: ``bf16((f32(xq . wq^T) * sx) * sw)``, (rows, N) bf16."""
    acc = xq.to(torch.int32) @ wq.to(torch.int32).T
    return ((acc.to(torch.float32) * sx[:, None]) * sw[None, :]).to(torch.bfloat16)


@dataclass(frozen=True)
class Layer:
    wq: torch.Tensor     # (H, H) s8
    sw: torch.Tensor     # (H,) f32


class Model:
    """The two-layer W8A8 MLP of width ``h``, its weights and a ``vocab``-row embedding drawn from ``seed``."""

    def __init__(self, h: int = 64, vocab: int = 16, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.h = h
        self.embed = torch.randn(vocab, h, generator=g)
        self.layers: list[Layer] = []
        for _ in range(2):
            w = torch.randn(h, h, generator=g) * 0.05
            wq, sw = quantize(w)
            self.layers.append(Layer(wq, sw))

    def op_path(self, layer: int) -> str:
        return f"model.layers.{layer}.mlp.{LINEARS[layer]}"


def bits(t: torch.Tensor) -> np.ndarray:
    """A tensor's elements as unsigned integers of their width (`evaluate_batch`'s convention)."""
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    a = t.contiguous().numpy()
    return a.view(f"u{a.dtype.itemsize}")


class TorchLoop:
    """The loop over ``requests`` (token ids) for ``steps`` decode steps of ``model`` (module docstring)."""

    def __init__(self, model: Model, requests: list[int], steps: int = 2, tamper: bool = False) -> None:
        self.model, self.requests, self.steps, self.tamper = model, list(requests), steps, tamper
        self.outputs: dict[tuple[int, int], torch.Tensor] = {}

    @property
    def units(self) -> int:
        return len(self.requests) * self.steps * 2 * self.model.h

    def workload(self) -> I.Workload:
        return I.Workload(f"torch-loop/mlp-w8a8-h{self.model.h}", ((template(self.model.h), self.units),))

    def serve(self, window: I.ServingWindow) -> None:
        coord, h = template(self.model.h), self.model.h
        for r, tok in enumerate(self.requests):
            x = self.model.embed[tok][None, :]
            for step in range(self.steps):
                for layer, lin in enumerate(self.model.layers):
                    xq, sx = quantize(x)
                    y = linear(xq, sx, lin.wq, lin.sw)
                    out = bits(y).reshape(h)
                    if self.tamper:
                        out = out ^ np.uint16(1)
                    window.record(coord, [np.repeat(bits(xq), h, 0), np.repeat(bits(sx), h), bits(lin.wq), bits(lin.sw)], [out],
                                  I.Served(f"r{r}", step, layer, self.model.op_path(layer), "y"))
                    x = torch.relu(y.to(torch.float32)) if layer == 0 else y.to(torch.float32)
                self.outputs[r, step] = x
