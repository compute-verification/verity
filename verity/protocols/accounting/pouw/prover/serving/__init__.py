"""PoUW's serving side: what runs beside a served model to commit its matrix multiplies, one directory a scheme (`h1`,
`lead`, `pearl_c`) and the NCP calls' operands, devices and circuits beside them. An adapter reaches it per call through
one function, `run_matmul`: run this matrix multiply under PoUW. The buffers are the caller's own (torch tensors on its
device), so this module loads no torch."""
from __future__ import annotations

from typing import Any, Protocol


class MatmulRun(Protocol):
    """One install's PoUW run of a scheme: ``gemm(site, x, weight, bias)`` serves y = x weightᵀ (+ bias) of the linear at
    weight id ``site`` and commits the call as its scheme does."""

    def gemm(self, site: int, x: Any, weight: Any, bias: Any, /) -> Any: ...


def run_matmul(run: MatmulRun, site: int, x: Any, weight: Any, bias: Any = None) -> Any:
    """Run this matrix multiply under PoUW: y = x weightᵀ (+ ``bias``) of the linear at weight id ``site``, on the caller's
    GPU buffers, served and committed by ``run``; y is a new buffer on x's device."""
    return run.gemm(site, x, weight, bias)


__all__ = ["MatmulRun", "run_matmul"]
