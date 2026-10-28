"""The integration's kernels as core `verity.ml.circuits.evaluation` kernels, plus what core's kernel registry does not carry yet.

Every kernel here is registered with `verity.ml.circuits.evaluation.register_kernel` against the Definition id it computes and is
checked against the reference by core `self_check` on its check specialisations (`check`: by tests/program/test_kernel_self_check.py,
and by the sampled replay before it grades a row with it, `check.replay.definitions`).  Core's kernel is a batch function returning a decline mask; the
integration's kernels are written per instance, and their callers record WHY an instance was declined.  `register` keeps
both forms:

    register(target, fn, kernel=..., domain=..., sample=..., check_targets=..., covers=...)
    fn(*args, **statics) -> the output leaves in the Definition's leaf order  |  raises Declined(why) or ValueError

The core kernel runs `fn` per instance and declines the instances it raises on; a batch it declines entirely returns
NotImplemented, so `evaluate_batch` computes it with the reference.  A kernel may also give its batch form,
`rows(*arrays, **statics) -> (out, declined) | None`: the same words as `fn` on every instance `fn` claims, declined where `fn`
raises, or None for a batch it leaves to `fn` per instance.  To move upstream once core has them: the instance form
with its decline reason (`instance`), the per-specialisation domain (`covers`: why a kernel does not claim a
specialisation at all, where core's `domain` is a description), the specialisations a kernel's self-check binds when the
served statics are too wide for the gate reference (`check_targets`), and the list of registered (Definition, kernel)
pairs (`registered`).
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from verity.ml.circuits.evaluation import Kernel, register_kernel, self_check
from verity.ml.circuits.evaluation.reference import key_of, statics_of

__all__ = ["Declined", "Registered", "check", "covers", "instance", "prim_static", "register", "registered"]


class Declined(Exception):
    """The kernel does not claim this instance (outside the domain it reproduces exactly); the message says why."""


_DOT_STATIC_FN = re.compile(r'^\{"fn":"([A-Za-z0-9_]+)"\}$')


def prim_static(statics: dict[str, Any], key: str, family: str) -> str:
    """A function-valued static (`DOT` of `Gemm_v2{K,N,DOT}` / `Attention_v2{..., DOT, INV}`, `INV` of the latter)
    -> the registered primitive id.  The spec text renders it as `DOT={"fn":"HopperBF16WgmmaDot16_v1"}` (verity.ml.circuits.codec._spec_id); a bare
    `<Name>_v<k>` id is accepted too.  Anything else is unresolved BY NAME: the evaluator never guesses the architecture from the target."""
    v = statics.get(key, "")
    raw = str(getattr(v, "id", v) or "")
    m = _DOT_STATIC_FN.match(raw)
    if m:
        return m.group(1)
    if re.match(r"^[A-Za-z0-9]+_v\d+$", raw):
        return raw
    raise Declined(f"{family} {key} static {raw!r} does not name a registered primitive")


def _claims_all(statics: dict) -> str | None:
    return None


def _no_targets(rng: np.random.Generator) -> tuple:
    return ()


@dataclass(frozen=True)
class Registered:
    key: str
    kernel: str
    fn: Callable[..., Any] = field(repr=False)
    check_targets: Callable[[np.random.Generator], Sequence[Any]] = field(repr=False)
    covers: Callable[[dict], str | None] = field(repr=False)
    core: Kernel = field(repr=False)


_REGISTERED: dict[tuple[str, str], Registered] = {}


def _batch(fn: Callable[..., Any], claims: Callable[[dict], str | None], batch: Callable[..., Any] | None = None) -> Callable[..., Any]:
    def run(*arrays: np.ndarray, **statics: Any):
        n = arrays[0].shape[0] if arrays else 0
        if claims(statics):
            return NotImplemented
        if batch is not None and n:
            got = batch(*arrays, **statics)
            if got is not None:
                return got
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
             check_targets: Callable[[np.random.Generator], Sequence[Any]] = _no_targets,
             covers: Callable[[dict], str | None] = _claims_all, batch: Callable[..., Any] | None = None) -> Registered:
    """Register the instance kernel `fn` of `target` (a Definition or its id) with core under `kernel`, and here with
    `check_targets(rng)`, the specialisations its self-check binds, and `covers(statics)`, the reason it does not claim a
    specialisation (None: it does); `batch`, its batch form (module docstring), runs first.  Re-registering the same `fn` is a
    no-op."""
    prev = _REGISTERED.get((key_of(target), kernel))
    if prev is not None and prev.fn is fn:
        return prev
    core = register_kernel(target, _batch(fn, covers, batch), kernel=kernel, domain=domain, version=version, sample=sample)
    reg = _REGISTERED[(core.key, kernel)] = Registered(core.key, kernel, fn, check_targets, covers, core)
    return reg


def instance(target: Any, kernel: str) -> Callable[..., Any]:
    """The instance form of kernel `kernel` of `target`: fn(*args, **statics) -> leaves, raising `Declined` with the reason."""
    return _REGISTERED[(key_of(target), kernel)].fn


def covers(target: Any, kernel: str) -> str | None:
    """None when kernel `kernel` claims the specialisation `target`, else why not."""
    reg = _REGISTERED.get((key_of(target), kernel))
    if reg is None:
        return f"no {kernel} kernel registered for {key_of(target)}"
    return reg.covers(statics_of(target))


def registered() -> list[Registered]:
    """Every kernel registered through `register`, in registration order."""
    return list(_REGISTERED.values())


def check(key: str, kernel: str, generator: Callable[[int], np.random.Generator], n: int = 6) -> list[str]:
    """Why kernel `kernel` of `key` is not its Definition, [] when it is: core `self_check` against the reference on `n` seeded
    instances of each of its check specialisations (`check_targets`, sampled by the caller's ``generator`` over a seed of the
    pair), each of which it must claim, and not decline every instance of."""
    reg = _REGISTERED.get((key, kernel))
    if reg is None:
        return [f"no {kernel} kernel registered for {key}"]
    targets = reg.check_targets(generator(zlib.crc32(f"{key}/{kernel}".encode())))
    if not targets:
        return [f"{key}: no check specialisations"]
    out = []
    for t in targets:
        why = covers(t, kernel)
        if why is not None:
            out.append(f"{t.id}: {why}")
            continue
        sc = self_check(t, kernel=kernel, n=n, seed=0)
        if not sc.ok:
            out.append(str(sc))
        elif sc.declined >= sc.n:
            out.append(f"{t.id}: {kernel} declined all {sc.n} instances")
    return out
