"""The workload's buffers and Pearl-C's device run, which the workload that serves the model hands in (`use`,
`use_pearl_c`).

PoUW's serving code names no tensor library: where it reads, writes or allocates one of the workload's buffers it asks
`tensors()`, an object of the workload's (`Tensors`). A buffer is ``Any``: whatever the workload's linears take and return. An element type
is named as NumPy names it (``"uint8"``, ``"int16"``, ``"uint16"``, ``"int32"``, ``"int64"``, ``"float32"``), and
``"bfloat16"``. A ``like`` is a buffer, whose device (and, where a method says so, element type) the result takes, or a
device as the workload names one.

Each accessor raises `NotImplementedError` until a workload has handed its object in. A NumPy array on the host is read
without either."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

#: the goal under which the device adapter hands both in
GOAL = "compute-accounting/architecture-on-branch"


class Tensors(Protocol):
    """The workload's buffers as PoUW's serving code reads, writes and allocates them."""

    def is_buffer(self, x: Any, /) -> bool:
        """Whether ``x`` is one of the workload's buffers (a NumPy array is not)."""
        ...

    def host(self, t: Any, /) -> np.ndarray:
        """``t``'s values on the host, a NumPy array of its own element type (a uint16 buffer's as ``uint16``)."""
        ...

    def put(self, a: np.ndarray, like: Any = None, /, *, as_like: bool = False) -> Any:
        """``a`` as a buffer on ``like``'s device (the host when None), of ``a``'s element type, or of ``like``'s with
        ``as_like``; a copy to a device goes through page-locked memory and is queued on the device's current stream."""
        ...

    def zeros(self, shape: tuple[int, ...], dtype: str, like: Any = None, /) -> Any:
        """A zeroed buffer of ``shape`` and element type ``dtype`` on ``like``'s device; with ``like`` None, a NumPy array
        on the host, page-locked where a device is up, so a copy into it (`copy_to_host`) is queued, not waited on."""
        ...

    def view(self, t: Any, dtype: str, /) -> Any:
        """``t``'s bytes, where they are, read as element type ``dtype`` (its last dimension scaled by the size ratio)."""
        ...

    def pointer(self, t: Any, /) -> int:
        """``t``'s first element's address on its device."""
        ...

    def stream(self, like: Any = None, /) -> int:
        """The raw handle of ``like``'s device's current stream (the current device's when None)."""
        ...

    def capability(self, like: Any = None, /) -> tuple[int, int]:
        """``like``'s device's compute capability (the current device's when None), major and minor."""
        ...

    def copy_to_host(self, t: Any, out: np.ndarray, /) -> Callable[[], None]:
        """Queue a copy of ``t``'s bytes into ``out`` (a `zeros` host array's slice of as many bytes) on ``t``'s device's
        current stream, behind what it has enqueued, without waiting; the result waits for the copy, and ``out`` is read
        only after it has. ``t`` is kept alive for the copy."""
        ...


def dtype_name(x: Any) -> str:
    """``x``'s element type by its bare name (``"uint8"``, ``"float32"``), whichever library's buffer or array ``x`` is; ``""``
    when it has none."""
    return str(getattr(x, "dtype", "")).rpartition(".")[2]


class PearlCDevice(Protocol):
    """Pearl-C's run on the device, which a served Pearl+ opens (`compute.pouw.scheme.Attached.open`)."""

    PearlCDeviceRun: Callable[..., Any]
    """One install's run (`PearlCDeviceRun(scheme, entry, kernel, weights, roots, coins, opening, ..., weight_salts)`):
    its ``gemm``, ``end_pass``, ``finish``, ``commits`` and ``details``."""

    def weight_roots(self, kernel: Any, weights: dict[int, Any], scheme: str, *,
                     weight_salts: dict[int, bytes]) -> dict[int, bytes]:
        """Each weight's B-row tree root, by its block key, before any salt, its rows salted with ``weight_salts``."""
        ...

    def step_tokens(self, run: Any, calls: int) -> int:
        """The most tokens one step's forward can carry in one pass of ``run``."""
        ...

    runs: dict[str, Any]
    """The live runs `linear_op`'s body reads, by the attached scheme's build key."""

    def linear_op(self) -> Callable[..., Any]:
        """``verity_pouw::pearl_plus_linear(x, weight, bias, entry, w) -> y``: one call as one dispatcher op, whose body
        is ``runs[entry].gemm(w, x, weight, bias)``, what a Match capture records."""
        ...


_tensors: Callable[[], Tensors] | None = None
_pearl_c: Callable[[], PearlCDevice] | None = None


def use(provider: Callable[[], Tensors]) -> None:
    """The workload's `Tensors`, as a callable that returns it when PoUW first needs it, so naming it loads nothing."""
    global _tensors
    _tensors = provider


def use_pearl_c(provider: Callable[[], PearlCDevice]) -> None:
    """The workload's `PearlCDevice`, as `use` takes `Tensors`."""
    global _pearl_c
    _pearl_c = provider


def has_tensors() -> bool:
    """Whether a workload has handed its buffers in (`use`)."""
    return _tensors is not None


def tensors() -> Tensors:
    if _tensors is None:
        raise NotImplementedError(f"{GOAL}: no tensors handed in; the workload that serves the model calls `tensors.use` first")
    return _tensors()


def pearl_c_device() -> PearlCDevice:
    if _pearl_c is None:
        raise NotImplementedError(f"{GOAL}: no Pearl-C device run handed in; the workload that serves the model calls "
                                  "`tensors.use_pearl_c` first")
    return _pearl_c()


__all__ = ["GOAL", "PearlCDevice", "Tensors", "dtype_name", "has_tensors", "pearl_c_device", "tensors", "use",
           "use_pearl_c"]
