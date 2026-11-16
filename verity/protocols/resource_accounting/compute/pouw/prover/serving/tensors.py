"""The workload's buffers and H1's device half, which the workload that serves the model hands in (`use`, `use_h1`).

PoUW's serving code names no tensor library: where it reads, writes or allocates one of the workload's buffers it asks
`tensors()`, an object of the workload's (`Tensors`), and where H1 hashes, keeps its tables or traces a call on the device
it asks `h1_device()` (`H1Device`). A buffer is ``Any``: whatever the workload's linears take and return. An element type
is named as NumPy names it (``"uint8"``, ``"int16"``, ``"uint16"``, ``"int32"``, ``"int64"``, ``"float32"``), and
``"bfloat16"``. A ``like`` is a buffer, whose device (and, where a method says so, element type) the result takes, or a
device as the workload names one.

Each accessor raises `NotImplementedError` until a workload has handed its object in. A NumPy array on the host is read
without either."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
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


class H1Device(Protocol):
    """H1's work on the device: its registered rows' hashes (`h1.registered`), its call tables (`h1.H1Run`), the check of a
    call's z/y records where they are (`h1.records.holds_on`) and the credit gate's trace of a call (`credit_gate`)."""

    def available(self) -> bool:
        """Whether `device_hash` can run in this process (`h1.registered.path`'s ``"auto"``)."""
        ...

    def device_hash(self, words: Any, key: bytes, nonce: int, row_salts: bytes | None, domain_id: bytes,
                    public: tuple[bytes, int, bool] | None = None, also: tuple[bytes, int] | None = None) -> Any:
        """`h1.registered.commit`'s ``b || c`` strings and tree levels of ``words`` ``(n, W)`` u16, hashed on the current
        device: an object whose ``result()`` gives them."""
        ...

    def device_public_salts(self, head: bytes, first: int, n: int, ok: bool = False) -> bytes:
        """`h1.registered.public_salts` on the device."""
        ...

    def hashing(self) -> AbstractContextManager[Any]:
        """The device `device_hash` hashes on, its stream made current meanwhile."""
        ...

    def commit_tables(self, run: Any, call: Any, a: Any, s_a: Any, records: np.ndarray | None, zy: Any = None) -> None:
        """Queue ``call``'s A and z tables' hashes on the device, for ``run`` (an `h1.H1Run`) to settle."""
        ...

    def settle(self, run: Any, wait: bool) -> None:
        """Take ``run``'s finished table hashes in, waiting for every one with ``wait``."""
        ...

    def flush(self, run: Any) -> None:
        """Launch ``run``'s table hashes still held back."""
        ...

    def take_z(self, run: Any, call: Any, words: Any) -> tuple[bytes, int] | None:
        """The call's z table to commit beside its registered ``zy`` value (its domain id and nonce), when ``run`` waits
        for it and ``words`` are its records' own bytes; else None."""
        ...

    def row_words(self, t: Any, m: int, dtype: str) -> np.ndarray:
        """``t``'s first ``m`` rows on the host as ``dtype``."""
        ...

    def holds_on(self, zy: Any, rows: int, n: int, y: Any, s_a: Any, s_w: Any) -> bool:
        """`h1.records.holds_on` where the call's buffers are."""
        ...

    def trace(self, linear: str) -> AbstractContextManager[Any]:
        """The credit gate's trace of one call of ``linear`` (`credit_gate`): every op the call runs, which `credit_gate.verdict`
        reads through its ``linear``, ``invocations`` and ``producer(t)``."""
        ...

    def outside(self) -> AbstractContextManager[None]:
        """Ops run past the innermost trace, which doesn't see them."""
        ...


_tensors: Callable[[], Tensors] | None = None
_h1: Callable[[], H1Device] | None = None


def use(provider: Callable[[], Tensors]) -> None:
    """The workload's `Tensors`, as a callable that returns it when PoUW first needs it, so naming it loads nothing."""
    global _tensors
    _tensors = provider


def use_h1(provider: Callable[[], H1Device]) -> None:
    """The workload's `H1Device`, as `use` takes `Tensors`."""
    global _h1
    _h1 = provider


def has_tensors() -> bool:
    """Whether a workload has handed its buffers in (`use`)."""
    return _tensors is not None


def tensors() -> Tensors:
    if _tensors is None:
        raise NotImplementedError(f"{GOAL}: no tensors handed in; the workload that serves the model calls `tensors.use` first")
    return _tensors()


def h1_device() -> H1Device:
    if _h1 is None:
        raise NotImplementedError(f"{GOAL}: no H1 device handed in; the workload that serves the model calls `tensors.use_h1` "
                                  "first")
    return _h1()


__all__ = ["GOAL", "H1Device", "Tensors", "dtype_name", "h1_device", "has_tensors", "tensors", "use", "use_h1"]
