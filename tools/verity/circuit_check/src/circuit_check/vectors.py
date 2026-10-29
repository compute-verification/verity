"""Input vectors for the correspondence checks: every edge word of a leaf's width first, then seeded random words.

A circuit is total by default (it matches the hardware on every input bit pattern), so the vectors read each leaf as every
float format its width can hold and put that format's edges in: NaN (quiet, signalling, negative), +-inf, +-0, the smallest
and largest subnormals, the smallest normal, +-max-finite and +-1.  Width 16 carries the BF16 and FP16 edges, 32 the FP32
ones, 8 the E4M3 and E5M2 ones, 64 the FP64 ones, and every width the integer edges (0, 1, the sign bit alone or with 1, all
ones, all ones but the sign).

Vector ``j < E`` sets every leaf to the ``j``-th edge of its width (all-NaN, all-inf, ...), so a specials rule is hit on every
operand at once; the random vectors then sample each leaf as an edge word (1 in 4), a word near 1.0 in magnitude (1 in 4,
floats only: arithmetic in the range serving sees) or a uniform word.

A registered table a target reads (`TableProbe`: `tanh_rn` is 2^27 entries) is in no vector: every vector reads the program's
own table as one `verity.ml.circuits.evaluation.RegisteredTable`, and the probes add vectors that read its first and last entries and the
first and last of each region (`probe_entries`), never all of them; the record says how many words the vectors read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from verity.ml.circuits.evaluation import RegisteredTable
from verity.ml.circuits.evaluation.batch import type_dtype, type_shape
from verity.ml.circuits.types import Type

BF16 = (0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x3F80, 0xBF80)
FP16 = (0x7C00, 0xFC00, 0x7E00, 0x7D00, 0x7BFF, 0xFBFF, 0x03FF, 0x0400, 0x3C00)
FP32 = (0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
        0x7FC00000, 0xFFC00000, 0x7F800001, 0x3F800000, 0xBF800000)
E4M3 = (0x00, 0x80, 0x01, 0x81, 0x07, 0x08, 0x7E, 0xFE, 0x7F, 0xFF, 0x38, 0xB8)
E5M2 = (0x7C, 0xFC, 0x7D, 0x7E, 0x7B, 0xFB, 0x03, 0x04, 0x3C)
FP64 = (0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0x7FF0000000000001, 0x7FEFFFFFFFFFFFFF, 0x000FFFFFFFFFFFFF,
        0x0010000000000000, 0x3FF0000000000000, 0xBFF0000000000000)
FLOATS = {8: E4M3 + E5M2, 16: BF16 + FP16, 32: FP32, 64: FP64}
#: (exponent bits, bias) of the format a random "near 1.0" word is sampled in, per width
_NEAR_ONE = {16: (8, 127), 32: (8, 127)}


def integer_edges(w: int) -> tuple[int, ...]:
    mask = (1 << w) - 1
    top = 1 << (w - 1)
    return tuple(dict.fromkeys((0, 1, top, top | 1, mask, mask ^ top)))


def edges(w: int) -> tuple[int, ...]:
    """Every edge word of width ``w``, floats first, each once."""
    return tuple(dict.fromkeys(FLOATS.get(w, ()) + integer_edges(w)))


def _random(rng: np.random.Generator, w: int, shape: tuple[int, int]) -> np.ndarray:
    """Random words of width ``w <= 64``: an edge word (1 in 4), a float near 1.0 (1 in 4, where the width is a float), else uniform."""
    ew = np.array(edges(w), dtype=np.uint64)
    mask = np.uint64((1 << w) - 1)
    uni = rng.integers(0, np.iinfo(np.uint64).max, size=shape, dtype=np.uint64, endpoint=True) & mask
    r = rng.random(shape)
    out = np.where(r < 0.25, ew[rng.integers(0, len(ew), size=shape)], uni)
    if w in _NEAR_ONE:
        e_bits, bias = _NEAR_ONE[w]
        m_bits = w - 1 - e_bits
        near = ((rng.integers(0, 2, size=shape, dtype=np.uint64) << np.uint64(w - 1))
                | ((bias + rng.integers(-8, 9, size=shape)).astype(np.uint64) << np.uint64(m_bits))
                | rng.integers(0, 1 << m_bits, size=shape, dtype=np.uint64))
        out = np.where((r >= 0.25) & (r < 0.5), near, out)
    return out


def edge_count(params: Sequence[Type]) -> int:
    return max((len(edges(w)) for pt in params for w in set(pt.leaf_widths())), default=0)


def vectors(params: Sequence[Type], n: int, seed: int) -> list[np.ndarray]:
    """``n`` vectors of the parameters ``params`` in ``evaluate_batch`` layout (one array per parameter, leading batch axis):
    the all-edge vectors first (as many as fit in ``n``), then random ones."""
    rng = np.random.default_rng(seed)
    E = min(n, edge_count(params))
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
        out.append((a if dt.kind == "O" else a.astype(dt)).reshape((n, *type_shape(pt))))
    return out


def instance_args(params: Sequence[Type], arrays: Sequence[np.ndarray], i: int) -> list:
    """Vector ``i`` as `verity.ml.circuits.evaluation.evaluate` takes it: an int per ``Value``, a flat int list per aggregate."""
    from verity.ml.circuits.types import Value

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


__all__ = ["BF16", "E4M3", "E5M2", "FP16", "FP32", "FP64", "TableProbe", "edge_count", "edges", "flat", "instance_args", "integer_edges",
           "probe_entries", "table_instances", "vectors"]
