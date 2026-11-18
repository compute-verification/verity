"""Pearl+'s call as one dispatcher op at a Match capture: ``verity_pouw::pearl_plus_linear(x, weight, bias, entry, w) ->
y``, whose body is the live run's ``gemm`` (`RUNS`, by the attached scheme's build key ``entry``), in the form of the
first-stage linear's ``verity_pouw::ncp_linear``. An observer records the op with its operands and its served y, and
nothing of the device work inside it."""
from __future__ import annotations

from functools import cache
from typing import Any

#: the live Pearl+ runs at a capture, by the attached scheme's build key
RUNS: dict[str, Any] = {}


@cache
def linear_op() -> Any:
    import torch

    @torch.library.custom_op("verity_pouw::pearl_plus_linear", mutates_args=(),
                             schema="(Tensor x, Tensor weight, Tensor? bias, str entry, int w) -> Tensor")
    def pearl_plus_linear(x, weight, bias, entry, w):
        return RUNS[entry].gemm(w, x, weight, bias)

    @pearl_plus_linear.register_fake
    def _(x, weight, bias, entry, w):
        return x.new_empty((*x.shape[:-1], weight.shape[0]))

    return torch.ops.verity_pouw.pearl_plus_linear


__all__ = ["RUNS", "linear_op"]
