"""The integration's kernels as core `verity.evaluation` kernels, plus what core's kernel registry does not carry yet.

Every kernel here is registered with `verity.evaluation.register_kernel` against the Definition id it computes and is
checked against the reference by core `self_check`.  Core's kernel is a batch function returning a decline mask; the
integration's kernels are written per instance, and their callers record WHY an instance was declined.  `register` keeps
both forms:

    register(target, fn, kernel=..., domain=..., sample=..., check_targets=...)
    fn(*args, **statics) -> the output leaves in the Definition's leaf order  |  raises Declined(why) or ValueError

The core kernel runs `fn` per instance and declines the instances it raises on; a batch it declines entirely returns
NotImplemented, so `evaluate_batch` computes it with the reference.  To move upstream once core has them: the instance form
with its decline reason (`instance`), the specialisations a kernel's self-check binds when the served statics are too
wide for the gate reference (`check_targets`), and the list of registered (Definition, kernel) pairs (`registered`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from verity.evaluation import Kernel, register_kernel
from verity.evaluation.reference import key_of

__all__ = ["Declined", "Registered", "instance", "register", "registered"]


class Declined(Exception):
    """The kernel does not claim this instance (outside the domain it reproduces exactly); the message says why."""


@dataclass(frozen=True)
class Registered:
    key: str
    kernel: str
    fn: Callable[..., Any] = field(repr=False)
    check_targets: Callable[[], Sequence[Any]] = field(repr=False)
    core: Kernel = field(repr=False)


_REGISTERED: dict[tuple[str, str], Registered] = {}


def _batch(fn: Callable[..., Any]) -> Callable[..., Any]:
    def run(*arrays: np.ndarray, **statics: Any):
        n = arrays[0].shape[0] if arrays else 0
        rows: list[np.ndarray | None] = [None] * n
        for i in range(n):
            try:
                rows[i] = np.asarray(fn(*(a[i] for a in arrays), **statics)).reshape(-1)
            except (Declined, ValueError):
                continue
        done = [r for r in rows if r is not None]
        if not done:
            return NotImplemented
        out = np.zeros((n, done[0].size), dtype=np.uint64)
        for i, r in enumerate(rows):
            if r is not None:
                out[i] = r
        return out, np.array([r is None for r in rows], dtype=bool)
    return run


def register(target: Any, fn: Callable[..., Any], *, kernel: str, domain: str = "", version: int = 1,
             sample: Callable[..., Sequence[np.ndarray]] | None = None,
             check_targets: Callable[[], Sequence[Any]] = tuple) -> Registered:
    """Register the instance kernel `fn` of `target` (a Definition or its id) with core under `kernel`, and here with the
    specialisations its self-check binds.  Re-registering the same `fn` is a no-op."""
    prev = _REGISTERED.get((key_of(target), kernel))
    if prev is not None and prev.fn is fn:
        return prev
    core = register_kernel(target, _batch(fn), kernel=kernel, domain=domain, version=version, sample=sample)
    reg = _REGISTERED[(core.key, kernel)] = Registered(core.key, kernel, fn, check_targets, core)
    return reg


def instance(target: Any, kernel: str) -> Callable[..., Any]:
    """The instance form of kernel `kernel` of `target`: fn(*args, **statics) -> leaves, raising `Declined` with the reason."""
    return _REGISTERED[(key_of(target), kernel)].fn


def registered() -> list[Registered]:
    """Every kernel registered through `register`, in registration order."""
    return list(_REGISTERED.values())
