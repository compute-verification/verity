"""Input vectors for the correspondence checks: every edge word of a leaf's width first, then seeded random words.

A circuit is total by default (it matches the hardware on every input bit pattern), so the vectors read each leaf as every
float format its width can hold and put that format's edges in: NaN (quiet, signalling, negative), +-inf, +-0, the smallest
and largest subnormals, the smallest normal, +-max-finite and +-1.  Width 16 carries the BF16 and FP16 edges, 32 the FP32
ones, 8 the E4M3 and E5M2 ones, 64 the FP64 ones, and every width the integer edges (0, 1, the sign bit alone or with 1, all
ones, all ones but the sign).

Vector ``j < E`` sets every leaf to the ``j``-th edge of its width (all-NaN, all-inf, ...), so a specials rule is hit on every
operand at once; the random vectors then sample each leaf as an edge word (1 in 4), a word near 1.0 in magnitude (1 in 4,
floats only: arithmetic in the range serving sees) or a uniform word.  ``E`` is as many edges as fit, or fewer when the caller
keeps room for random vectors (`circuit_check.checks.Options.random_floor`).

Over many float leaves such a vector almost always holds a NaN or an infinity somewhere, so half the random vectors, those of
odd index (`finite_share`), are sampled finite instead: every leaf of width 8, 16, 32 or 64 at working scale (`_WORKING`), from
a second random stream, written over those rows once the first stream has sampled every row, so every other row is the
sample without the share (``finite=False``) bit for bit.  A leaf of another width keeps its first-stream word.  A leaf is finite when it
is finite in every float format of its width (`NONFINITE`): a 16-bit word only when BF16 and FP16 both read it finite, since
the vectors don't know which one a leaf holds.

A registered table a target reads (`TableProbe`: `tanh_rn` is 2^27 entries) is in no vector: every vector reads the program's
own table as one `verity.circuits.evaluation.RegisteredTable`, and the probes add vectors that read its first and last entries and the
first and last of each region (`probe_entries`), never all of them; the record says how many words the vectors read.

An operand over the input budget (a weight, `circuit_check.targets.LAZY_OPERANDS`) is sampled as `vectors` samples a parameter, its
j-th edge in every leaf for an edge vector and seeded random words for the rest (finite ones on the rows of odd index), but one
vector at a time (`wide_words`), and read word by word as the gates ask for it, never held as ``n`` vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from verity.circuits.evaluation import RegisteredTable
from verity.circuits.evaluation.batch import type_dtype, type_shape
from verity.circuits.types import Type

BF16 = (0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x3F80, 0xBF80)
FP16 = (0x7C00, 0xFC00, 0x7E00, 0x7D00, 0x7BFF, 0xFBFF, 0x03FF, 0x0400, 0x3C00)
FP32 = (0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
        0x7FC00000, 0xFFC00000, 0x7F800001, 0x3F800000, 0xBF800000)
E4M3 = (0x00, 0x80, 0x01, 0x81, 0x07, 0x08, 0x7E, 0xFE, 0x7F, 0xFF, 0x38, 0xB8)
E5M2 = (0x7C, 0xFC, 0x7D, 0x7E, 0x7B, 0xFB, 0x03, 0x04, 0x3C)
FP64 = (0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0x7FF0000000000001, 0x7FEFFFFFFFFFFFFF, 0x000FFFFFFFFFFFFF,
        0x0010000000000000, 0x3FF0000000000000, 0xBFF0000000000000)
FLOATS = {8: E4M3 + E5M2, 16: BF16 + FP16, 32: FP32, 64: FP64}
#: per width, the masks ``m`` of `FLOATS`' formats there: a word ``x`` is a NaN or an infinity in one of them when ``x & m == m``
#: (E4M3 has no infinity and one NaN, S.1111.111; E5M2, BF16, FP16, FP32 and FP64 are IEEE-like: an all-ones exponent)
NONFINITE = {8: (0x7F, 0x7C), 16: (0x7F80, 0x7C00), 32: (0x7F800000,), 64: (0x7FF0000000000000,)}
#: (exponent bits, bias) of the format a random "near 1.0" word is sampled in, per width
_NEAR_ONE = {16: (8, 127), 32: (8, 127)}
#: (exponent bits, bias, spread) of the finite share's words, per width (a random sign and mantissa, the exponent within the
#: spread of the bias): the near-1.0 words where the width has them, an E4M3 word within 2^6 of 1.0 either way at 8 bits (normal,
#: and finite read as E5M2) and an FP64 one within 2^8 at 64; each finite in every format of its width (`NONFINITE`)
_WORKING = {8: (4, 7, 6), 16: (*_NEAR_ONE[16], 8), 32: (*_NEAR_ONE[32], 8), 64: (11, 1023, 8)}
#: the second random stream's key beside the seed, from which the finite share is sampled
FINITE_STREAM = 0xF1


def integer_edges(w: int) -> tuple[int, ...]:
    mask = (1 << w) - 1
    top = 1 << (w - 1)
    return tuple(dict.fromkeys((0, 1, top, top | 1, mask, mask ^ top)))


def edges(w: int) -> tuple[int, ...]:
    """Every edge word of width ``w``, floats first, each once."""
    return tuple(dict.fromkeys(FLOATS.get(w, ()) + integer_edges(w)))


def _near(rng: np.random.Generator, w: int, shape: tuple[int, int], e_bits: int, bias: int, spread: int) -> np.ndarray:
    """Floats of width ``w`` with ``e_bits`` exponent bits: a random sign and mantissa, the exponent within ``spread`` of ``bias``."""
    m_bits = w - 1 - e_bits
    return ((rng.integers(0, 2, size=shape, dtype=np.uint64) << np.uint64(w - 1))
            | ((bias + rng.integers(-spread, spread + 1, size=shape)).astype(np.uint64) << np.uint64(m_bits))
            | rng.integers(0, 1 << m_bits, size=shape, dtype=np.uint64))


def _random(rng: np.random.Generator, w: int, shape: tuple[int, int]) -> np.ndarray:
    """Random words of width ``w <= 64``: an edge word (1 in 4), a float near 1.0 (1 in 4, where the width is a float), else uniform."""
    ew = np.array(edges(w), dtype=np.uint64)
    mask = np.uint64((1 << w) - 1)
    uni = rng.integers(0, np.iinfo(np.uint64).max, size=shape, dtype=np.uint64, endpoint=True) & mask
    r = rng.random(shape)
    out = np.where(r < 0.25, ew[rng.integers(0, len(ew), size=shape)], uni)
    if w in _NEAR_ONE:
        out = np.where((r >= 0.25) & (r < 0.5), _near(rng, w, shape, *_NEAR_ONE[w], 8), out)
    return out


def finite_words(words: np.ndarray, widths: Sequence[int]) -> np.ndarray:
    """Per row of ``words`` (``(rows, leaves)``, leaf ``j`` of width ``widths[j]``), whether every leaf is finite in every float
    format of its width (`NONFINITE`); a leaf of a width no float format has counts as finite."""
    words, widths = np.asarray(words), np.asarray(widths)
    ok = np.ones(words.shape[0], dtype=bool)
    for w, masks in NONFINITE.items():
        if len(cols := np.flatnonzero(widths == w)):
            x = words[:, cols].astype(np.uint64)
            for m in map(np.uint64, masks):
                ok &= ((x & m) != m).all(axis=1)
    return ok


def finite_rows(params: Sequence[Type], arrays: Sequence[np.ndarray], n: int) -> np.ndarray:
    """Per vector of the ``n`` in ``arrays`` (`vectors`' layout), whether every leaf of every parameter is finite (`finite_words`)."""
    ok = np.ones(n, dtype=bool)
    for pt, a in zip(params, arrays):
        ok &= finite_words(np.asarray(a).reshape(n, pt.leaves), pt.leaf_widths())
    return ok


def finite_args(params: Sequence[Type], args: Sequence[Any]) -> bool:
    """Whether every leaf of vector ``args`` (`instance_args`' form) is finite (`finite_words`), a `RegisteredTable` aside."""
    return all(isinstance(a, RegisteredTable)
               or bool(finite_words(np.asarray(a, dtype=object).reshape(1, pt.leaves), pt.leaf_widths())[0])
               for pt, a in zip(params, args))


def finite_share(n: int, edge: int) -> range:
    """The rows of ``n`` vectors, the first ``edge`` all-edge, that `vectors` samples finite: the random rows of odd index, half of
    them (to within one), spread through them so that every prefix of the random vectors holds its half, and a function of a
    row's index alone, so an operand sampled one vector at a time (`wide_words`) agrees with the rest."""
    return range(min(n, edge) | 1, n, 2)


def edge_count(params: Sequence[Type]) -> int:
    return max((len(edges(w)) for pt in params for w in set(pt.leaf_widths())), default=0)


def vectors(params: Sequence[Type], n: int, seed: int, edge: int | None = None, *, finite: bool = True) -> list[np.ndarray]:
    """``n`` vectors of the parameters ``params`` in ``evaluate_batch`` layout (one array per parameter, leading batch axis):
    the all-edge vectors first (``edge`` of them, default as many as fit in ``n``), then random ones, those `finite_share` names
    finite in every leaf of width 8, 16, 32 or 64 (`_WORKING`, from the second stream) unless ``finite`` is False."""
    rng, rng_finite = np.random.default_rng(seed), np.random.default_rng([seed, FINITE_STREAM])
    E = min(n, edge_count(params) if edge is None else edge)
    share = np.asarray(finite_share(n, E) if finite else (), dtype=np.intp)
    out = []
    for pt in params:
        widths = np.asarray(pt.leaf_widths())
        dt = type_dtype(pt)
        a = np.zeros((n, len(widths)), dtype=np.uint64 if dt.kind != "O" else object)
        for w in sorted(set(widths.tolist())):
            cols = np.flatnonzero(widths == w)
            ew = edges(w)
            if dt.kind == "O" or w > 64:
                for j in range(n):
                    for c in cols:
                        a[j, c] = ew[j % len(ew)] if j < E else int.from_bytes(rng.bytes((w + 7) // 8), "little") & ((1 << w) - 1)
                continue
            a[:E, cols] = np.array([ew[j % len(ew)] for j in range(E)], dtype=np.uint64)[:, None]
            if n > E:
                a[E:, cols] = _random(rng, w, (n - E, len(cols)))
        for w in sorted(set(widths.tolist()) & _WORKING.keys()) if len(share) else ():
            cols = np.flatnonzero(widths == w)
            words = _near(rng_finite, w, (len(share), len(cols)), *_WORKING[w])
            a[np.ix_(share, cols)] = words.astype(object) if dt.kind == "O" else words
        out.append((a if dt.kind == "O" else a.astype(dt)).reshape((n, *type_shape(pt))))
    return out


#: the leaves `wide_words` samples at once
WIDE_CHUNK = 1 << 22


def wide_words(pt: Type, i: int, edge_vectors: int, seed: int, param: int) -> Callable[[int], int]:
    """Vector ``i`` of parameter ``param``, of type ``pt`` and one width, as a function of the leaf index: the ``i``-th edge of its
    width in every leaf when ``i < edge_vectors``, else seeded random words (`_random`, seeded by ``seed``, ``param`` and ``i``),
    sampled into one array of the parameter's leaves; on an odd ``i`` (`finite_share`) of width 8, 16, 32 or 64, finite words
    (`_WORKING`, seeded by those and `FINITE_STREAM`).  Its ``finite`` says whether every word is finite (`finite_words`)."""
    w = pt.homogeneous_width()
    if w is None or w > 64:
        raise ValueError(f"{pt!r} is not of one width of at most 64 bits")
    if i < edge_vectors:
        ew = edges(w)
        word = ew[i % len(ew)]

        def edge(j: int) -> int:
            return word

        edge.finite = bool(finite_words(np.array([[word]], dtype=np.uint64), [w])[0])
        return edge
    working = i % 2 == 1 and w in _WORKING
    rng = np.random.default_rng([seed, param, i, FINITE_STREAM] if working else [seed, param, i])
    a = np.empty(pt.leaves, dtype=next(d for b, d in ((8, np.uint8), (16, np.uint16), (32, np.uint32), (64, np.uint64)) if w <= b))
    finite = True
    for lo in range(0, pt.leaves, WIDE_CHUNK):
        hi = min(lo + WIDE_CHUNK, pt.leaves)
        words = _near(rng, w, (1, hi - lo), *_WORKING[w]) if working else _random(rng, w, (1, hi - lo))
        finite = finite and bool(finite_words(words, np.full(hi - lo, w))[0])
        a[lo:hi] = words[0]

    def sampled(j: int) -> int:
        return int(a[j])

    sampled.finite = finite
    return sampled


def instance_args(params: Sequence[Type], arrays: Sequence[np.ndarray], i: int) -> list:
    """Vector ``i`` as `verity.circuits.evaluation.evaluate` takes it: an int per ``Value``, a flat int list per aggregate."""
    from verity.circuits.types import Value

    args = []
    for pt, a in zip(params, arrays):
        row = np.asarray(a[i]).reshape(-1)
        args.append(int(row[0]) if isinstance(pt, Value) else [int(v) for v in row])
    return args


def flat(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """``(n, leaves)`` uint64: every vector's input leaves in parameter order (what a Flock lowering evaluates)."""
    n = arrays[0].shape[0] if arrays else 0
    return np.concatenate([np.asarray(a, dtype=np.uint64).reshape(n, -1) for a in arrays], axis=1) if arrays else np.zeros((n, 0), np.uint64)


@dataclass(frozen=True)
class TableProbe:
    """The registered table a target takes as parameter ``param``: ``table`` holds the program's words, ``entry_words`` leaves to
    an entry; ``regions`` are [lo, hi) entry ranges (the parts the table's words come from); ``reads(i)`` gives vectors of the
    target's other parameters (in order, the table left out) that read entry ``i``."""

    param: int
    table: RegisteredTable
    entry_words: int
    regions: tuple[tuple[int, int], ...]
    reads: Callable[[int], Iterable[Sequence[Any]]]

    @property
    def entries(self) -> int:
        return self.table.leaves // self.entry_words


def probe_entries(entries: int, regions: Iterable[tuple[int, int]]) -> list[int]:
    """The entries probed: the first, the last, and each region's first and last."""
    out = {0, entries - 1}
    for lo, hi in regions:
        if not 0 <= lo < hi <= entries:
            raise ValueError(f"region [{lo}, {hi}) is not within the table's {entries} entries")
        out |= {lo, hi - 1}
    return sorted(out)


def table_instances(params: Sequence[Type], probe: TableProbe, n: int, seed: int) -> tuple[list[list], set[int], dict]:
    """The vectors of a target that reads a registered table: ``n`` edge and random ones of its other parameters (`vectors`),
    then those that read each probed entry (`probe_entries`, `probe.reads`), all reading one `RegisteredTable` that records the
    leaves read (the returned set, filled as the vectors are evaluated); and the probe's record."""
    other = [pt for j, pt in enumerate(params) if j != probe.param]
    arrays = vectors(other, n, seed)
    read: set[int] = set()
    word = probe.table.word

    def counted(j: int) -> int:
        read.add(j)
        return word(j)

    table = RegisteredTable(probe.table.name, counted, probe.table.leaves, probe.table.declared_by)

    def full(rest: Sequence[Any]) -> list:
        args = list(rest)
        args.insert(probe.param, table)
        return args

    out = [full(instance_args(other, arrays, i)) for i in range(n)]
    probed = probe_entries(probe.entries, probe.regions)
    for e in probed:
        out += [full(r) for r in probe.reads(e)]
    rec = {"name": probe.table.name, "param": probe.param, "entries": probe.entries, "leaves": probe.table.leaves,
           "regions": [list(r) for r in probe.regions], "probed_entries": probed, "probe_instances": len(out) - n}
    return out, read, rec


__all__ = ["BF16", "E4M3", "E5M2", "FINITE_STREAM", "FP16", "FP32", "FP64", "NONFINITE", "TableProbe", "edge_count", "edges",
           "finite_args", "finite_rows", "finite_share", "finite_words", "flat", "instance_args", "integer_edges", "probe_entries",
           "table_instances", "vectors", "wide_words"]
