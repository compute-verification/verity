"""The served int8 arm PoUW's H1 (`h1`) records and checks: its exact GEMM and its dequantization, which the workload that serves the arm
hands in (`use`). H1 reads both only to redo what the arm served: a call's z when the arm kept none (`H1Run.record`), and
dq(z, s_a, s_w) when it checks the arm's z/y records (`h1.records.reason`, `h1.records.holds_on`).

`arm` raises `NotImplementedError` until a workload has handed its arm in."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

#: the goal that hands the arm in from the adapter's install
GOAL = "compute-accounting/pouw-serving-out-of-adapter"


class Int8Arm(Protocol):
    """The arm's two operations on the caller's tensors: ``int_mm(xq, w)``, X [M, K] int8 times W [N, K] int8 transposed,
    exact in int32; ``dequantize(z, s_a, s_w)``, RN_bf16(RN32(RN32(RN32(z) * s_a[i]) * s_w[n]))."""

    def int_mm(self, xq: Any, w: Any, /) -> Any: ...

    def dequantize(self, z: Any, s_a: Any, s_w: Any, /) -> Any: ...


_provider: Callable[[], Int8Arm] | None = None


def use(provider: Callable[[], Int8Arm]) -> None:
    """The workload's arm, as a callable that returns it when H1 first needs it, so naming it loads nothing."""
    global _provider
    _provider = provider


def arm() -> Int8Arm:
    if _provider is None:
        raise NotImplementedError(f"{GOAL}: no int8 arm handed in; the workload that serves it calls `int8_arm.use` first")
    return _provider()


__all__ = ["GOAL", "Int8Arm", "arm", "use"]
