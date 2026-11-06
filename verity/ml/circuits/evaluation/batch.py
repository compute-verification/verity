"""Batch evaluation over numpy arrays, the kernel registry, and `self_check`.

A batch of inputs is one integer array per parameter with a leading batch axis: shape ``(N,)`` for a ``Value<w>``,
``(N, n, ...)`` for an ``Array``, ``(N, leaves)`` for a tuple or record.  `evaluate_batch` returns the output the same way.

A kernel is registered against a Definition id (without statics) or a `ScalarReference` id, under a name (``"numpy"``):

    fn(*arrays, **statics) -> (out, declined) | NotImplemented

``declined`` is a boolean ``(N,)`` mask (or ``None``) of inputs outside the kernel's exact domain; `evaluate_batch`
recomputes those with the reference, so a kernel never has to be total.  A kernel declines exactly the inputs the reference
rejects unless it is registered ``partial`` (its ``domain`` says which others it declines).  ``NotImplemented`` declines
the whole call (statics the kernel does not cover, e.g. an unknown ``DOT`` step).  Every registered kernel must pass
`self_check` (`SelfCheck.ok`): on seeded random inputs every input it accepts equals the reference bit for bit, every
input the reference rejects is one it declined, it computed at least one, and, unless it is partial, it declined
nothing else.

A Boolean Definition (`verity.ml.circuits.boolean`) is evaluated by its word Definition's kernel (`trace.word_of`), through
the word view (`boolean.word_type`: each element's bits are one word), when it has none of its own; a kernel registered
against ``Gemm_v2`` serves ``Gemm_v3``.  What that leaves, and what a kernel declines, is evaluated bit-sliced
(`bits.evaluate_bits`), never gate by gate.  The word Definition is held to the Boolean one by circuit-check's word view.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import SpecializedDefinition
from verity.ml.circuits.types import Array, Type, Value

from .bits import _to_bits, _to_words, evaluate_bits
from .reference import ScalarReference, Target, evaluate, key_of, refuse_registered_tables, resolve, signature_of, statics_of

__all__ = [
    "Kernel",
    "SelfCheck",
    "evaluate_batch",
    "kernels",
    "register_kernel",
    "sample_uniform",
    "self_check",
    "type_dtype",
    "type_shape",
    "word_kernel",
]

KernelFn = Callable[..., Any]
Sampler = Callable[..., Sequence[np.ndarray]]


@dataclass(frozen=True)
class Kernel:
    key: str
    name: str
    fn: KernelFn = field(repr=False)
    domain: str = ""
    version: int = 1
    sample: Sampler | None = field(default=None, repr=False)
    partial: bool = False


_KERNELS: dict[str, dict[str, Kernel]] = {}


def register_kernel(target: Target | str, fn: KernelFn, *, kernel: str = "numpy", domain: str = "", version: int = 1,
                    sample: Sampler | None = None, partial: bool = False) -> Kernel:
    """Register ``fn`` as kernel ``kernel`` of ``target`` (a Definition, a `ScalarReference`, or a Definition id).
    ``domain`` says in words which inputs it computes (the rest it declines); ``partial`` says it declines inputs the
    reference accepts (else it declines exactly those the reference rejects); ``sample(rng, n, params, **statics)``
    optionally draws the inputs `self_check` uses.  Re-registering the same function is a no-op; a different one
    under the same (target, kernel) is an error, as for Definitions."""
    key = key_of(target)
    k = Kernel(key, kernel, fn, domain, version, sample, partial)
    prev = _KERNELS.setdefault(key, {}).get(kernel)
    if prev is not None and prev.fn is not fn:
        raise ValueError(f"kernel {kernel!r} of {key} is already registered with a different function")
    _KERNELS[key][kernel] = k
    _WORD.clear()
    return k


def kernels(target: Target | str) -> dict[str, Kernel]:
    """The kernels registered against ``target`` (not those a Boolean Definition takes from its word Definition)."""
    return dict(_KERNELS.get(key_of(target), {}))


def _pick(target: Target, kernel: str) -> Kernel | None:
    if kernel == "reference":
        return None
    ks = _KERNELS.get(key_of(target), {})
    if kernel == "auto":
        return ks.get("numpy") or next(iter(ks.values()), None) or word_kernel(target)
    if kernel not in ks:
        raise KeyError(f"no kernel {kernel!r} registered for {key_of(target)} (registered: {sorted(ks)})")
    return ks[kernel]


_WORD: "weakref.WeakKeyDictionary[SpecializedDefinition, Kernel | None]" = weakref.WeakKeyDictionary()


def word_kernel(target: Target) -> Kernel | None:
    """The kernel a Boolean Definition takes from its word Definition: that Definition's ``"auto"`` kernel on the word view
    of the inputs, its output unpacked to bits (rows it declines stay declined).  None when ``target`` is not a Boolean
    composite, names no word Definition, has an element wider than 64 bits, or its word Definition has no kernel."""
    if not isinstance(target, SpecializedDefinition) or getattr(target.defn, "word", None) is None:
        return None
    if target in _WORD:
        return _WORD[target]
    k = None
    if BL.is_boolean(target):
        word = word_of(target)
        params, ret = signature_of(target)
        wparams, _ = signature_of(word)
        inner = _pick(resolve(word), "auto")
        if inner is not None and max((w for t in (*params, ret) for w in BL.elements(t)), default=1) <= 64:
            k = Kernel(key_of(target), inner.name, _through(inner, statics_of(word), params, wparams, BL.elements(ret)),
                       inner.domain, inner.version, partial=inner.partial)
    _WORD[target] = k
    return k


def _through(inner: Kernel, statics: dict, params: Sequence[Type], wparams: Sequence[Type], ret: list[int]) -> KernelFn:
    def fn(*arrays: np.ndarray, **_statics: Any) -> Any:
        n = arrays[0].shape[0]
        words = [_to_words(a.reshape(n, -1), BL.elements(t)).astype(type_dtype(wt)).reshape(n, *type_shape(wt))
                 for a, t, wt in zip(arrays, params, wparams)]
        res = inner.fn(*words, **statics)
        if res is NotImplemented:
            return res
        vals, declined = res
        return _to_bits(np.asarray(vals).reshape(n, -1), ret), declined

    return fn


# ---- types <-> arrays ---------------------------------------------------------------------------------------------------

def type_shape(t: Type) -> tuple[int, ...]:
    if isinstance(t, Value):
        return ()
    if isinstance(t, Array):
        return (t.n, *type_shape(t.elem))
    return (t.leaves,)


def _width_dtype(w: int) -> np.dtype:
    for bits, dt in ((8, np.uint8), (16, np.uint16), (32, np.uint32), (64, np.uint64)):
        if w <= bits:
            return np.dtype(dt)
    return np.dtype(object)


def type_dtype(t: Type) -> np.dtype:
    return _width_dtype(max(t.width_set(), default=1))


def _as_batch(a: Any, t: Type, name: str, n: int | None) -> np.ndarray:
    a = np.asarray(a)
    shape = type_shape(t)
    if a.ndim == 0:
        raise ValueError(f"{name}: a batch needs a leading batch axis")
    if a.shape[1:] != shape:
        if int(np.prod(a.shape[1:], dtype=np.int64)) != t.leaves:
            raise ValueError(f"{name}: shape {a.shape[1:]} per instance, expected {shape}")
        a = a.reshape((a.shape[0], *shape))
    if n is not None and a.shape[0] != n:
        raise ValueError(f"{name}: batch size {a.shape[0]}, expected {n}")
    if a.dtype.kind not in "uiO":
        raise TypeError(f"{name}: encodings must be integers, got {a.dtype}")
    dt = type_dtype(t)
    if dt.kind == "O" or a.dtype.kind == "O":
        return a.astype(object)
    flat = a.reshape(a.shape[0], t.leaves)
    if a.dtype.kind == "i" and flat.size and flat.min() < 0:
        raise ValueError(f"{name}: negative encoding")
    # widths from the type's structure, never one per leaf; a non-negative word of at most the narrowest width's bits fits every leaf
    ws = t.width_set()
    narrow = [w for w in ws if w < 64]
    if flat.size and narrow and a.dtype.itemsize * 8 > min(narrow):
        if len(ws) == 1:
            wide = np.any(flat >> flat.dtype.type(narrow[0]))
        else:
            widths = np.asarray(t.leaf_widths(), dtype=np.uint64)
            wide = np.any(flat[:, widths < 64].astype(np.uint64) >> widths[widths < 64])
        if wide:
            raise ValueError(f"{name}: encoding wider than its declared Value<w>")
    return a.astype(dt, copy=False)


def _reference_rows(target: Target, params: tuple[Type, ...], arrays: Sequence[np.ndarray], rows: np.ndarray):
    cols = [a.reshape(a.shape[0], pt.leaves) for a, pt in zip(arrays, params)]
    for i in rows:
        args = [int(c[i, 0]) if isinstance(pt, Value) else c[i].tolist() for c, pt in zip(cols, params)]
        yield int(i), args


def _store(out: np.ndarray, i: int, leaves: tuple[int, ...]) -> None:
    if out.ndim == 1:
        out[i] = leaves[0]
    else:
        out[i] = np.asarray(leaves, dtype=out.dtype).reshape(out.shape[1:])


# ---- evaluate_batch ----------------------------------------------------------------------------------------------------

def evaluate_batch(target: Target, *inputs: Any, kernel: str = "auto") -> np.ndarray:
    """Evaluate ``target`` on ``N`` inputs: one array per parameter with a leading batch axis, the output likewise.
    ``kernel="auto"`` uses the ``numpy`` kernel when one is registered (for a Boolean Definition without one, its word
    Definition's, `word_kernel`), ``"reference"`` forces the plain-integer reference, any other name selects that kernel.
    Inputs no kernel computes are computed by the reference, whose exceptions (inputs outside the function's domain)
    propagate, or, for a Boolean composite, bit-sliced (`bits.evaluate_bits`).  A `RegisteredTable` input is refused
    by name."""
    t = resolve(target)
    refuse_registered_tables(t.id, inputs, "evaluate_batch")
    params, ret = signature_of(t)
    if len(inputs) != len(params):
        raise TypeError(f"{t.id}: expected {len(params)} input arrays, got {len(inputs)}")
    arrays: list[np.ndarray] = []
    n: int | None = None
    for j, (a, pt) in enumerate(zip(inputs, params)):
        arrays.append(_as_batch(a, pt, f"{t.id} argument {j}", n))
        n = arrays[-1].shape[0]
    n = 0 if n is None else n
    out = np.zeros((n, *type_shape(ret)), dtype=type_dtype(ret))
    todo = np.ones(n, dtype=bool)
    k = _pick(t, kernel)
    if k is not None and n:
        res = k.fn(*arrays, **statics_of(t))
        if res is not NotImplemented:
            vals, declined = res
            out[...] = np.asarray(vals).reshape(out.shape)
            todo = np.zeros(n, dtype=bool) if declined is None else np.asarray(declined, dtype=bool).reshape(n)
    rows = np.flatnonzero(todo)
    if rows.size and kernel != "reference" and isinstance(t, SpecializedDefinition) and BL.is_boolean(t):
        out[rows] = evaluate_bits(t, *(a[rows] for a in arrays)).reshape(out[rows].shape)
        return out
    for i, args in _reference_rows(t, params, arrays, rows):
        _store(out, i, evaluate(t, *args))
    return out


# ---- self_check ----------------------------------------------------------------------------------------------------------

def sample_uniform(rng: np.random.Generator, n: int, params: Sequence[Type], **_statics: Any) -> list[np.ndarray]:
    """Uniform random encodings per leaf, one in eight replaced by a width-generic edge word (0, 1, the sign bit alone or
    with 1, all ones, all ones but the sign)."""
    out = []
    for pt in params:
        shape = (n, *type_shape(pt))
        widths = np.asarray(pt.leaf_widths(), dtype=np.uint64).reshape(type_shape(pt))
        dt = type_dtype(pt)
        if dt.kind == "O":
            raise TypeError("sample_uniform supports widths up to 64")
        top = np.uint64(1) << (widths - np.uint64(1))
        mask = np.where(widths >= 64, np.uint64(0xFFFFFFFFFFFFFFFF), (np.uint64(1) << np.minimum(widths, 63)) - np.uint64(1))
        words = rng.integers(0, np.iinfo(np.uint64).max, size=shape, dtype=np.uint64, endpoint=True) & mask
        edges = np.stack(np.broadcast_arrays(np.uint64(0), np.uint64(1), top, top | np.uint64(1), mask, mask ^ top))
        edges = np.broadcast_to(edges[:, None], (edges.shape[0], *shape))
        edge = np.take_along_axis(edges, rng.integers(0, edges.shape[0], size=(1, *shape)), 0)[0]
        words = np.where(rng.random(shape) < 0.125, edge, words)
        out.append(words.astype(dt))
    return out


@dataclass
class SelfCheck:
    target: str
    kernel: str
    n: int
    declined: int
    outside: int
    mismatches: list[tuple[int, list, Any, Any]]
    partial: bool = False

    @property
    def compared(self) -> int:
        """The inputs the kernel computed, each compared with the reference."""
        return self.n - self.declined

    @property
    def fault(self) -> str | None:
        """Why the kernel is not checked against the reference by this run, None when it is: ``"mismatch"`` (an accepted
        input differs, or an input the reference rejects was accepted), ``"compared-nothing"`` (it declined every input),
        or ``"declined-in-domain"`` (a kernel that is not `partial` declined an input the reference accepts)."""
        if self.mismatches:
            return "mismatch"
        if self.compared <= 0:
            return "compared-nothing"
        if not self.partial and self.declined != self.outside:
            return "declined-in-domain"
        return None

    @property
    def ok(self) -> bool:
        return self.fault is None

    def __str__(self) -> str:
        head = (f"{self.target} [{self.kernel}{', partial' if self.partial else ''}]: {self.n} instances, {self.declined} "
                f"declined, {self.outside} outside the reference's domain, {len(self.mismatches)} mismatches"
                + (f" ({self.fault})" if self.fault else ""))
        return "\n".join([head, *(f"  row {i}: kernel {kv!r} reference {rv!r} args {args!r}"[:400]
                                  for i, args, kv, rv in self.mismatches[:5])])


def self_check(target: Target, *, kernel: str = "numpy", n: int = 1024, seed: int = 0,
               inputs: Sequence[Any] | None = None) -> SelfCheck:
    """Compare kernel ``kernel`` of ``target`` with the reference on ``n`` seeded inputs (the kernel's own sampler if it
    registered one, else `sample_uniform`) or on ``inputs``.  An accepted input must equal the reference bit for bit;
    an input the reference rejects (raises on) must be declined; `SelfCheck.ok` also asks that some input was compared
    and, unless the kernel is ``partial``, that it declined nothing else."""
    t = resolve(target)
    params, ret = signature_of(t)
    k = _pick(t, kernel)
    if k is None:
        raise KeyError(f"no kernel {kernel!r} for {t.id}")
    statics = statics_of(t)
    if inputs is None:
        inputs = (k.sample or sample_uniform)(np.random.default_rng(seed), n, params, **statics)
    refuse_registered_tables(t.id, inputs, "self_check")
    arrays = [_as_batch(a, pt, f"{t.id} argument {j}", None) for j, (a, pt) in enumerate(zip(inputs, params))]
    n = arrays[0].shape[0] if arrays else 0
    res = k.fn(*arrays, **statics)
    if res is NotImplemented:
        raise ValueError(f"kernel {kernel!r} of {k.key} declines {t.id} entirely")
    vals, declined = res
    vals = np.asarray(vals).reshape((n, *type_shape(ret)))
    declined = np.zeros(n, dtype=bool) if declined is None else np.asarray(declined, dtype=bool).reshape(n)
    mismatches: list[tuple[int, list, Any, Any]] = []
    outside = 0
    for i, args in _reference_rows(t, params, arrays, np.arange(n)):
        try:
            ref = evaluate(t, *args)
        except Exception as e:  # noqa: BLE001 - the reference's own domain check: the kernel must have declined
            outside += 1
            if not declined[i]:
                mismatches.append((i, args, vals[i].tolist(), f"reference raised {type(e).__name__}: {e}"))
            continue
        if declined[i]:
            continue
        got = (int(vals[i]),) if vals.ndim == 1 else tuple(int(v) for v in vals[i].reshape(-1))
        if got != ref:
            mismatches.append((i, args, got, ref))
    return SelfCheck(k.key if isinstance(t, ScalarReference) else t.id, kernel, n, int(declined.sum()), outside, mismatches,
                     k.partial)
