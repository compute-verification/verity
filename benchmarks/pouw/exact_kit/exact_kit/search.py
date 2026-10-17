"""Exact kernel searches for small-width maps: exhaustive with pruning, meet in the middle, and 2-adic lifting.

The kit knows no map. The caller passes vectorized numpy functions over a packed domain (each point one unsigned int, its
fields at bit offsets of the caller's choosing), and every result is checked on every point of the domain it is given:
the probes only prune.
"""
from __future__ import annotations

import operator
from collections.abc import Callable

import numpy as np


def _probes(xs: np.ndarray, probe: int, seed: int) -> np.ndarray:
    probe = max(probe, 2)
    if len(xs) <= probe:
        return xs
    pick = np.random.default_rng(seed).choice(len(xs), size=probe - 2, replace=False)
    return xs[np.concatenate(([0, len(xs) - 1], pick))]


def _same(a, b) -> np.ndarray:
    if isinstance(a, tuple):
        out = np.ones(np.broadcast_shapes(*(x.shape for x in a)), dtype=bool)
        for x, y in zip(a, b, strict=True):
            out &= x == y
        return out
    return np.asarray(a) == np.asarray(b)


def kernel(f: Callable, xs, deltas, act: Callable = operator.xor, probe: int = 64, seed: int = 0) -> list[int]:
    """Every delta in `deltas` with f(act(x, delta)) == f(x) for every x in `xs`, sorted.

    f maps an array of points to an array (or a tuple of arrays) of outputs, elementwise; act(x, delta) moves points (XOR by
    default; for addition mod 2^n pass ``lambda x, d: (x + d) & mask``). Deltas are pruned on `probe` points of xs (the
    first, the last and a seeded sample) all at once, and each survivor is then checked on all of xs."""
    xs = np.asarray(xs, dtype=np.uint64)
    deltas = np.asarray(deltas, dtype=np.uint64)
    pts = _probes(xs, probe, seed)
    base = f(pts)
    base = tuple(np.asarray(b)[None, :] for b in base) if isinstance(base, tuple) else np.asarray(base)[None, :]
    alive = np.ones(len(deltas), dtype=bool)
    step = max(1, (1 << 22) // len(pts))
    for lo in range(0, len(deltas), step):
        d = deltas[lo:lo + step]
        alive[lo:lo + step] = _same(f(act(pts[None, :], d[:, None])), base).all(axis=1)
    full = f(xs)
    return sorted(int(d) for d in deltas[alive] if bool(_same(f(act(xs, d)), full).all()))


def meet(left, right) -> list[tuple[int, int]]:
    """Every (i, j) with left[i] == right[j], rows compared whole (1-D arrays: entries), sorted. Sort-merge, so the cost is
    that of sorting the two sides, not their product."""
    L = np.asarray(left)
    R = np.asarray(right)
    L = L.reshape(len(L), -1)
    R = R.reshape(len(R), -1)
    if L.shape[1] != R.shape[1]:
        raise ValueError("signatures of different lengths")
    _, inv = np.unique(np.concatenate([L, R]), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    li, ri = inv[:len(L)], inv[len(L):]
    order = np.argsort(ri, kind="stable")
    rs = ri[order]
    out = []
    for i, g in enumerate(li.tolist()):
        a, b = np.searchsorted(rs, g, "left"), np.searchsorted(rs, g, "right")
        out.extend((i, int(j)) for j in order[a:b].tolist())
    return sorted(out)


def meet_kernel(g: Callable, h: Callable, xs, A, B, probe: int = 64, seed: int = 0) -> list[tuple[int, int]]:
    """Every (a, b) in A x B with g(x, a) == h(x, b) for every x in xs, sorted: the pairs a perturbation splits into when
    one half's effect must be cancelled by the other's. Meets the halves' signatures on `probe` points of xs (`meet`), then
    checks each meeting pair on all of xs. g and h broadcast over (perturbation, point)."""
    xs = np.asarray(xs, dtype=np.uint64)
    A = np.asarray(A, dtype=np.uint64)
    B = np.asarray(B, dtype=np.uint64)
    pts = _probes(xs, probe, seed)
    sl = np.broadcast_to(g(pts[None, :], A[:, None]), (len(A), len(pts)))
    sr = np.broadcast_to(h(pts[None, :], B[:, None]), (len(B), len(pts)))
    out = []
    for i, j in meet(sl, sr):
        a, b = A[i], B[j]
        if bool(np.all(np.broadcast_to(g(xs, a), xs.shape) == np.broadcast_to(h(xs, b), xs.shape))):
            out.append((int(a), int(b)))
    return sorted(out)


def lift(ok: Callable[[int, int], bool], bits: int) -> list[int]:
    """Every c in [0, 2^bits) with ok(c mod 2^j, j) for j = 1 .. bits, sorted, by extending residues one bit at a time.

    It returns every solution of ok(., bits) when ok(., j) holds for the residue mod 2^j of each one, as it does for
    conditions on the low j bits of maps built from +, -, *, XOR, AND and OR (no rotation or right shift): low output bits
    read only low input bits."""
    level = [0]
    for j in range(bits):
        level = [c | b << j for c in level for b in (0, 1) if ok(c | b << j, j + 1)]
    return sorted(level)
