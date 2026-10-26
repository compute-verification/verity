"""Batch forms of a served tile's replay for `served_debit.py`: every place a Pearl-C v1 tile's check runs the device's E4M3
atom one word at a time, run over all of the tile's words at once on numpy (`verity.ml.kernels.tc.group_sum_total_batch`, the
atom's bit-exact batch form).

- `_pairs`: `pearl_c.chain` on every (a, b) pair of two lists of code rows: C̃'s words (`checked`), the fused columns of the peel factors
  (`peel_factors_a`, `peel_factors_b`) and G (`gram`).
- `form_v1`: `pearl_c.form_v1`, the per-element atom of v1's forming, with its scalar statistics, E′ and cast.
- `chain_debit`: `pearl_c_debit.chain_debit`, every word's masked group words, its totals, the backward bisection of the
  later promotions' intervals and its flags, over the tile's words together.

Each returns exactly what its scalar reference returns.  Wherever a word, an accumulator or a promotion leaves the finite
range, or an input is not the shape the batch takes, it calls the scalar reference instead, so a rejection (and any other
exception) is the reference's own.  FP32 adds and products are numpy float32's, IEEE nearest-even with subnormals kept, as
`verity.ml.circuits.ops.fp32` takes them; `install` refuses a process whose floating-point unit flushes subnormals.

`install(scheme)` puts them in place of the scalar functions for a Pearl-C v1 scheme on a promoted device, with the `blake3`
package's BLAKE3 (the same digests) for the plain-Python one, and returns the function that puts the scalar ones back:
`served_debit.py` and `verify_run.py --fast` replay every Nth tile both ways and stop on any difference (``--cross-check``).
"""
from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np
from verity.ml.kernels import tc as K
from verity.ml.circuits.silicon import fp32
from verity.ml.protocols.pouw.schemes import pearl_c as C
from verity.ml.protocols.pouw.schemes import pearl_c_debit as D
from verity.ml.protocols.pouw.schemes import pearl_kw as P

_TOP = 0x7F7FFFFF
_NEG_ZERO = np.uint32(0x80000000)
_SCALAR = {"chain": C.chain, "form_v1": C.form_v1, "chain_debit": D.chain_debit}


class _Scalar(Exception):
    """A lane left the finite range or an input is not the batch's shape: the scalar reference decides."""


def _finite(words: np.ndarray) -> np.ndarray:
    if ((words >> 23) & 0xFF == 0xFF).any():
        raise _Scalar
    return words


def _add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """FP32 nearest-even add on words (IEEE, subnormals kept): `fp32.add` on finite inputs."""
    with np.errstate(over="ignore", invalid="ignore"):
        return (x.view(np.float32) + y.view(np.float32)).view(np.uint32)


def _mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return (x.view(np.float32) * y.view(np.float32)).view(np.uint32)


def _plus_zero(words: np.ndarray) -> np.ndarray:
    return np.where(words == _NEG_ZERO, np.uint32(0), words)


_CHUNK = 1 << 15


def _step(model, acc: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The atom on every lane, in chunks so that the kernel's (lanes, k + 1) arrays stay small."""
    out = np.empty(acc.shape[0], np.uint32)
    for s in range(0, acc.shape[0], _CHUNK):
        out[s:s + _CHUNK] = K.group_sum_total_batch(model, acc[s:s + _CHUNK], a[s:s + _CHUNK], b[s:s + _CHUNK])[0]
    return _finite(out)


def _rows(rows: Sequence[Sequence[int]], k: int | None = None) -> np.ndarray:
    try:
        out = np.asarray(rows, dtype=np.int64)
    except (ValueError, TypeError, OverflowError):
        raise _Scalar from None
    if out.ndim != 2 or (k is not None and out.shape[1] != k):
        raise _Scalar
    return out


def _chain(a: np.ndarray, b: np.ndarray, ia: np.ndarray, ib: np.ndarray, device) -> np.ndarray:
    """`pearl_c.chain` on the pairs (a[ia[l]], b[ib[l]]), a and b rows of k E4M3 codes: per window of G atoms a fresh +0
    accumulator, promoted into the total by one FP32 add, a zero total +0."""
    k = a.shape[1]
    if b.shape[1] != k or k % C.SLICE or device.g is None:
        raise _Scalar
    total = np.zeros(len(ia), np.uint32)
    for ws in range(0, k, device.window):
        local = np.zeros(len(ia), np.uint32)
        for s in range(ws, min(ws + device.window, k), C.SLICE):
            local = _step(device.atom, local, a[ia, s:s + C.SLICE], b[ib, s:s + C.SLICE])
        total = _plus_zero(_finite(_add(total, local)))
    return total


def _pairs(a_rows, b_rows, device) -> list[list[int]]:
    """`[[chain(a, b) for b in b_rows] for a in a_rows]`."""
    na, nb = len(a_rows), len(b_rows)
    try:
        if not na or not nb:
            raise _Scalar
        flat = _chain(_rows(a_rows), _rows(b_rows), np.repeat(np.arange(na), nb), np.tile(np.arange(nb), na), device)
    except _Scalar:
        return [[_SCALAR["chain"](x, y, device) for y in b_rows] for x in a_rows]
    return [[int(w) for w in flat[i * nb:(i + 1) * nb]] for i in range(na)]


def peel_batch(c: Sequence[Sequence[int]], pa: Sequence[Sequence[int]], pb: Sequence[Sequence[int]], device) -> list[list[int]]:
    """`pearl_c.peel` on every word of C̃: U[i][j] = peel(c[i][j], pa[i], pb[j])."""
    na, nb = len(pa), len(pb)
    try:
        acc = np.asarray(c, dtype=np.uint32).reshape(-1)
        a, b = np.repeat(_rows(pa, C.PEEL_K), nb, axis=0), np.tile(_rows(pb, C.PEEL_K), (na, 1))
        if acc.shape[0] != na * nb:
            raise _Scalar
        _finite(acc)
        for s in range(0, C.PEEL_K, 16):
            acc = _step(device.peel, acc, a[:, s:s + 16], b[:, s:s + 16])
    except _Scalar:
        return [[C.peel(c[i][j], pa[i], pb[j], device) for j in range(nb)] for i in range(na)]
    return [[int(w) for w in acc[i * nb:(i + 1) * nb]] for i in range(na)]


def checked(scheme, ctx, ops) -> C.Checked:
    """`PearlC.checked`: C̃ and U over the tile's words."""
    d = scheme.device
    c = _pairs(ops.a.codes, ops.b.codes, d)
    return C.Checked(c, peel_batch(c, ops.pa, ops.pb, d))


def form_v1(x_rows: Sequence[Sequence[int]], e_rows: Sequence[Sequence[int]], f: Sequence[Sequence[int]], device=C.H100
            ) -> C.Formed:
    """`pearl_c.form_v1`: per element one atom from α·x with E′ and F's row, then Pearl's cast."""
    try:
        x = _rows(x_rows).astype(np.uint32)
        fr = _rows(f)
        r, k = x.shape
        if fr.shape != (k, C.SLICE) or len(e_rows) != r:
            raise _Scalar
        _finite(x)
        stats = [C.s5_stats(row) for row in x_rows]
        eps = [C.eprime(beta, e_row) for (_, _, _, beta), e_row in zip(stats, e_rows)]
        ep_rows = _rows(eps, C.SLICE)
        words = []
        for i in range(r):
            acc = _finite(_mul(np.full(k, stats[i][2], np.uint32), x[i]))
            words.append(_step(device.atom, acc, np.broadcast_to(ep_rows[i], (k, C.SLICE)), fr).tolist())
    except _Scalar:
        return _SCALAR["form_v1"](x_rows, e_rows, f, device)
    out = C.Formed([], [], [], [], [], [], [])
    cast, of = P.f32_to_fp8, P.f32_of_bits
    for row, (s, rho, al, beta), ep, ws in zip(x_rows, stats, eps, words):
        out.codes.append([cast(of(w)) for w in ws])
        for field, v in (("alpha", al), ("beta", beta), ("s", s), ("rho", rho), ("ep", ep), ("x", list(row))):
            getattr(out, field).append(v)
    return out


@functools.lru_cache(maxsize=48, typed=True)
def _gram(seed_b: bytes, k: int, norm: float, device) -> tuple[tuple[int, ...], ...]:
    fa, fb = C._basis(seed_b, 0, k, norm), C._basis(seed_b, 1, k, norm)
    fa_cols, fb_cols = [C._column(fa, q) for q in range(C.R)], [C._column(fb, p) for p in range(C.R)]
    g = _pairs(fb_cols, fa_cols, device)
    return tuple(tuple(row) for row in g)


def gram(seed_b: bytes, k: int, norm: float = P.NOISE_NORM, device=C.H100) -> tuple[tuple[int, ...], ...]:
    """`pearl_c.gram`, once per (seed_B, k, norm, device) in a process."""
    return _gram(seed_b, k, norm, device)


# ---- the debit --------------------------------------------------------------------------------------------------------

def _key(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.int64)
    return np.where(w >> 31 == 1, -(w & 0x7FFFFFFF), w)


def _word(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0, (-x) | 0x80000000, x).astype(np.uint32)


def _promote(total: np.ndarray, word: np.ndarray) -> np.ndarray:
    return _plus_zero(_finite(_add(total, word)))


def _least(pred, n: int) -> np.ndarray:
    """Per lane, the least key x in the finite FP32 range with pred(x) (monotone), or one past the top if there is none:
    `pearl_c_debit._least`'s bisection on every lane at once."""
    none = ~pred(np.full(n, _TOP, np.int64))
    lo, hi = np.full(n, -_TOP, np.int64), np.full(n, _TOP, np.int64)
    while True:
        live = lo < hi
        if not live.any():
            break
        mid = (lo + hi) // 2
        p = pred(mid)
        hi, lo = np.where(live & p, mid, hi), np.where(live & ~p, mid + 1, lo)
    return np.where(none, _TOP + 1, lo)


def _carried(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The key of RN(word(x) + w); an overflow keys past every finite value."""
    return _key(_plus_zero(_add(_word(x), w)))


def _debit(a: np.ndarray, b: np.ndarray, device) -> tuple[int, int]:
    g_, atom = device.g, device.atom.k
    ra, k = a.shape
    if g_ is None or b.shape[1] != k or k % (g_ * atom):
        raise _Scalar
    n = ra * b.shape[0]
    ia, ib = np.repeat(np.arange(ra), b.shape[0]), np.tile(np.arange(b.shape[0]), ra)
    groups = []
    for g0 in range(0, k // atom, g_):
        s = np.zeros((1 << g_, n), np.uint32)
        for j in range(g_):
            t, h = (g0 + j) * atom, 1 << j
            op_a, op_b = a[ia, t:t + atom], b[ib, t:t + atom]
            stepped = _step(device.atom, s[:h].reshape(-1), np.tile(op_a, (h, 1)), np.tile(op_b, (h, 1))).reshape(h, n)
            s[h:2 * h] = s[:h]
            s[:h] = stepped
        groups.append(s)
    totals = [np.zeros(n, np.uint32)]
    for masked in groups:
        totals.append(_promote(totals[-1], masked[0]))
    lo = hi = _key(totals[-1])
    bounds = [(lo, hi)]
    for g in range(len(groups) - 1, 0, -1):
        w = groups[g][0]
        lo, hi = (_least(lambda x, w=w, lo=lo: _carried(x, w) >= lo, n),
                  _least(lambda x, w=w, hi=hi: _carried(x, w) > hi, n) - 1)
        bounds.append((lo, hi))
    bounds.reverse()
    atoms = 0
    for g, masked in enumerate(groups):
        lo, hi = bounds[g]
        keep = np.zeros((1 << g_, n), bool)
        for mask in range(1, 1 << g_):
            v = _key(_promote(totals[g], masked[mask]))
            keep[mask] = (lo <= v) & (v <= hi)
        for j in range(g_):
            atoms += int(keep[[m for m in range(1 << g_) if m >> j & 1]].any(axis=0).sum())
    zeros = sum(int((t == 0).sum()) for t in totals[1:-1])
    return atoms, zeros


def chain_debit(a_rows: Sequence[Sequence[int]], b_rows: Sequence[Sequence[int]], device=D.H100) -> tuple[int, int]:
    """`pearl_c_debit.chain_debit`: (flagged atoms, later +0 promotions) over every word of the tile."""
    try:
        if not len(a_rows) or not len(b_rows):
            raise _Scalar
        return _debit(_rows(a_rows), _rows(b_rows), device)
    except _Scalar:
        return _SCALAR["chain_debit"](a_rows, b_rows, device)


# ---- installing them ----------------------------------------------------------------------------------------------------

def _ieee() -> bool:
    """Whether this process's float32 adds keep subnormals (no flush-to-zero, no denormals-are-zero)."""
    tiny = np.array([1, 0x00800000], np.uint32)
    return _add(tiny[:1], tiny[:1])[0] == 2 and _add(tiny[1:], np.array([0x80000001], np.uint32))[0] == 0x007FFFFF


def _native_blake3():
    """(hash, keyed) on the `blake3` package, or None when it is absent: the same digests as `verity.ml.crypto.blake3`."""
    try:
        import blake3 as native
    except ImportError:
        return None
    return (lambda message: native.blake3(message).digest()), (lambda key, message: native.blake3(message, key=key).digest())


def install(scheme, debit: bool = True) -> callable:
    """Put the batch forms in place for `scheme` (Pearl-C v1 on a promoted device), and the `blake3` package's BLAKE3 in place
    of the plain-Python one wherever a loaded module holds it; returns the undo.  ``debit=False`` leaves
    `pearl_c_debit.chain_debit` to the caller (`served_debit.py` records it through a wrapper of its own)."""
    import sys
    from verity.ml.crypto import blake3 as pure
    if scheme.forming != "v1" or scheme.device.g is None:
        raise ValueError(f"{scheme.name}: the batch forms take Pearl-C v1 on a promoted device")
    if not _ieee():
        raise RuntimeError("this process's float32 arithmetic flushes subnormals; the batch forms would not be exact")
    swaps: list[tuple[object, str, object]] = []

    def swap(owner, name, value):
        swaps.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def peel_factors_a(codes, be_rows, seed_b, k, norm=P.NOISE_NORM, device=C.H100):
        fb = C._basis(seed_b, 1, k, norm)
        cols = _pairs(codes, [C._column(fb, p) for p in range(C.R)], device)
        return [list(be) + [C._bf16_of_word(w) for w in row] for row, be in zip(cols, be_rows)]

    def peel_factors_b(codes, be_rows, seed_b, k, norm=P.NOISE_NORM, device=C.H100):
        fa = C._basis(seed_b, 0, k, norm)
        fa_b = _pairs(codes, [C._column(fa, q) for q in range(C.R)], device)
        g = C.gram(seed_b, k, norm, device)
        rows = []
        for row_fa, be in zip(fa_b, be_rows):
            mid = []
            for q in range(C.R):
                acc = 0
                for p in range(C.R):
                    acc = C._check_finite(fp32.fma(be[p] << 16, g[p][q], acc))
                mid.append(C._bf16_of_word(C._check_finite(fp32.add(acc, fp32.neg(row_fa[q])))))
            rows.append(mid + [x ^ 0x8000 for x in be])
        return rows

    native = _native_blake3()
    if native is not None:
        h, kd = native
        for mod in [m for m in list(sys.modules.values()) if m is not None]:
            for name, fn, new in (("blake3", pure.blake3, h), ("keyed", pure.keyed, kd)):
                if getattr(mod, name, None) is fn and (mod is pure or mod.__name__.startswith(("verity", "pearl_c"))):
                    swap(mod, name, new)
    swap(C, "_basis", functools.lru_cache(maxsize=48, typed=True)(C._basis.__wrapped__))
    swap(C, "form_v1", form_v1)
    swap(C, "gram", gram)
    swap(C, "peel_factors_a", peel_factors_a)
    swap(C, "peel_factors_b", peel_factors_b)
    if debit:
        swap(D, "chain_debit", chain_debit)
    swap(type(scheme), "checked", checked)

    def undo():
        for owner, name, value in reversed(swaps):
            setattr(owner, name, value)
        swaps.clear()
    return undo
