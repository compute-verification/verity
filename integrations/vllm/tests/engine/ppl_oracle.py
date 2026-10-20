"""The plain int8 step's oracle: `benchmarks/ppl/steps.py` itself, imported as the bench imports it (its directory on sys.path),
and run on the CPU through the bench's own `matmul.MM` and gates.

`int8_linear(x, w)` is what the bench's layer computes for a linear under the arm `[[linear]] step = "int8"` (defaults): the
chain `steps.int8_linear` -> `matmul.terminal` on x.float() and the bf16 weight's float(), then `.to(bf16)` as `model.Layer.linear`
does.  Every product runs on the check window, so the bench's gates (int8 range, s32 accumulator, float64 recomputation) run too.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

PPL = Path(__file__).resolve().parents[4] / "benchmarks" / "ppl"
if str(PPL) not in sys.path:
    sys.path.insert(0, str(PPL))

import gates as G  # noqa: E402
import matmul as MM  # noqa: E402
import methodology  # noqa: E402
import steps as S  # noqa: E402


def run(arm=None) -> MM.Run:
    return MM.Run(arm or types.SimpleNamespace(name="vllm-int8"), 0, methodology.CHECK_WINDOW, types.SimpleNamespace(li=0), G.Gates(),
                  {}, {}, None)


def _passed(r: MM.Run) -> None:
    bad = {n: g for n, g in r.gates.log.items() if g["failed"]}
    assert not bad, f"the bench's gates failed: {bad}"


def int8_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y [M, N] bf16 of x [M, K] bf16 and the bf16 weight w [N, K], by steps.int8_linear with its defaults."""
    r = run()
    mm = MM.MM("linear", "mlp.up_proj", r, x.reshape(-1, x.shape[-1]).float(), w, maps=(lambda t: t.float(),))
    y = S.int8_linear(mm, MM.terminal)
    _passed(r)
    return y.to(torch.bfloat16)


def int8_linear_scaled(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """steps.int8_linear's dequantization (bf16) of a given integer product y [M, N] (int64), on x's and w's scales: the step
    with the rest of its chain returning y."""
    mm = MM.MM("linear", "mlp.up_proj", run(), x.reshape(-1, x.shape[-1]).float(), w, maps=(lambda t: t.float(),))
    return S.int8_linear(mm, lambda _m: y).to(torch.bfloat16)


def int8_weights(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(int8 [N, K], scales [N]) steps.int8_linear quantizes the bf16 weight w to (`quant_rows(w.float(), 127)`)."""
    q, s = S.quant_rows(w.float(), 127)
    return q.to(torch.int8), s

