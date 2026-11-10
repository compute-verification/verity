"""History tables: rows other proof blocks output, committed as one table that proof blocks read at hidden positions, as
registered values are read. The Lean verifier (`Spot.HistoryTable`) is the definition; this module mirrors it, held to it
by the spot suite's `test_history_difftest.py`, for the prover's committer (`HistoryTable`) and the verifier's side of a
window (`verity.ml.windows.history`).

A table's tree has a fixed capacity, `CAPACITY` leaves, whatever its length, so its shape says nothing of the request's.
Its domain is ``CommitmentDomain(B, -2, RangeIndexedDomain(0, capacity))``, ``B`` the identity digest of
``{"capacity", "program", "schema", "table", "words"}`` under `DOMAIN_TAG`. Leaf ``r`` below the table's length is its
writer's public ``b || c`` (the hm96 commit string of the writer's output row, a ``hm96-sha512/row-seg/v1`` row of
``words`` u16 words); every leaf past it is `placeholder`'s, the commit string of the zero row under the zero salt, which
anyone computes, so a read past the length returns that zero row. The verifier computes the root from those public
strings and holds the registered root to it (`check`): the prover's root is never trusted.

`order` gives each position's source from the program's runs, never from labels the prover supplies: a run is
``{"start", "count", "writer", "out", "radix", "base", "step"}``, position ``start + i`` the output ``out`` of unit
``base + sum(digit_k(i) * step[k])`` of writer `writer` (``i``'s mixed-radix digits over ``radix``, the first the
slowest), or ``{"start", "count", "held", "row"}``, position ``start + i`` row ``row + i`` of a value the verifier holds.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, row_leaf, sha512_row_seg_digest

CAPACITY = 1 << 17
DOMAIN_TAG = "verity/one-stage/history-table/v1"
SCHEMA = SCHEMA_HM96_SHA512_ROW_SEG
BC_BYTES = 128
DIGEST_BYTES = 64
OWNER = -2
_INST = hm96.SHA512
UNIT_KEYS = frozenset({"start", "count", "writer", "out", "radix", "base", "step"})
HELD_KEYS = frozenset({"start", "count", "held", "row"})


def _int(x: Any) -> bool:
    return type(x) is int


def _nat(x: Any, what: str) -> int:
    if not (_int(x) and x >= 0):
        raise ValueError(f"{what} is not a natural number")
    return x


def _str(x: Any, what: str) -> str:
    if type(x) is not str:
        raise ValueError(f"{what} is not a string")
    return x


def cap_ok(capacity: int) -> bool:
    return _int(capacity) and capacity >= 2 and capacity & (capacity - 1) == 0


def words_ok(words: int) -> bool:
    return _int(words) and 0 < words and 16 * words < 1 << 64


def domain(program: str, table: str, words: int, capacity: int = CAPACITY) -> CommitmentDomain:
    binding = identity_digest_sha512(DOMAIN_TAG, {"capacity": capacity, "program": program, "schema": SCHEMA,
                                                  "table": table, "words": words})
    return CommitmentDomain(binding, OWNER, RangeIndexedDomain(0, capacity), hash="sha512")


def placeholder(words: int) -> bytes:
    """The ``b || c`` of every leaf past a table's length: the zero row of ``words`` u16 words under the zero salt."""
    x = sha512_row_seg_digest(bytes(2 * words), 16 * words)
    return _INST.commit_string(_INST.default_key, x, bytes(_INST.salt_bytes))


def leaves(dom: CommitmentDomain, bcs: Sequence[bytes], ph: bytes) -> list[bytes]:
    pad = _INST.tree_leaf(_INST.default_key, ph)
    return [row_leaf(dom, r, _INST.tree_leaf(_INST.default_key, bcs[r]) if r < len(bcs) else pad, SCHEMA)
            for r in range(dom.count)]


def root(dom: CommitmentDomain, bcs: Sequence[bytes], ph: bytes) -> bytes:
    return tree_levels(dom, leaves(dom, bcs, ph))[-1][0]


def check(program: str, table: str, words: int, bcs: Sequence[bytes], registered: bytes,
          capacity: int = CAPACITY) -> None:
    """Hold a registered root to the table's root over its writers' ``b || c``, in `Spot.HistoryTable.checkAt`'s order:
    a capacity that isn't a power of two of at least 2, a row width out of range, more rows than the capacity, a
    ``b || c`` that isn't `BC_BYTES`, a root that isn't `DIGEST_BYTES`, then a root other than the table's."""
    if not cap_ok(capacity):
        raise ValueError(f"a table's capacity is a power of two of at least 2, not {capacity!r}")
    if not words_ok(words):
        raise ValueError(f"a table's rows are at least one word and under 2^64 bits, not {words!r} words")
    if capacity < len(bcs):
        raise ValueError(f"{len(bcs)} rows past the table's capacity of {capacity}")
    if any(len(bc) != BC_BYTES for bc in bcs):
        raise ValueError(f"a row's b || c is {BC_BYTES} bytes")
    if len(registered) != DIGEST_BYTES:
        raise ValueError(f"a registered root is {DIGEST_BYTES} bytes")
    if root(domain(program, table, words, capacity), bcs, placeholder(words)) != bytes(registered):
        raise ValueError("the registered root is not the table's root over its writers' rows")


def unit_at(radix: Sequence[int], base: int, step: Sequence[int], i: int) -> int:
    acc = 0
    for r, s in reversed(list(zip(radix, step))):
        acc += i % r * s
        i //= r
    return base + acc


def _run_why(run: Mapping[str, Any], writer: str, out: str, units: int, held: Mapping[str, int]) -> str | None:
    count = run["count"]
    if "held" in run:
        n = held.get(run["held"])
        if n is None:
            return f"a run reads {run['held']!r}, which the verifier holds no rows of"
        if n < run["row"] + count:
            return f"a run reads rows {run['row']}..{run['row'] + count} of {run['held']!r}, which has {n}"
        return None
    if run["writer"] != writer or run["out"] != out:
        return f"a run reads {run['writer']}.{run['out']}, not the table's writer {writer}.{out}"
    radix, step = run["radix"], run["step"]
    if len(radix) != len(step) or 0 in radix:
        return "a run's radix and step are one nonzero radix per step"
    if math.prod(radix) != count:
        return f"a run's radices multiply to {math.prod(radix)}, not its {count} positions"
    low = run["base"] + sum(min(0, s * (r - 1)) for r, s in zip(radix, step))
    high = run["base"] + sum(max(0, s * (r - 1)) for r, s in zip(radix, step))
    if low < 0 or units <= high:
        return f"a run reads units {low}..{high}, outside the writer's {units}"
    return None


def parse_run(j: Any) -> dict[str, Any]:
    if not isinstance(j, dict):
        raise ValueError("a run is not an object")
    _nat(j.get("start"), "start")
    _nat(j.get("count"), "count")
    if set(j) == UNIT_KEYS:
        if not _int(j["base"]):
            raise ValueError("base is not an integer")
        _str(j["writer"], "writer")
        _str(j["out"], "out")
        for k in ("radix", "step"):
            if not (isinstance(j[k], list) and all(_int(x) for x in j[k])):
                raise ValueError(f"{k} is not a list of integers")
        if any(x < 0 for x in j["radix"]):
            raise ValueError("radix: not a natural number")
        return dict(j)
    if set(j) == HELD_KEYS:
        _str(j["held"], "held")
        _nat(j["row"], "row")
        return dict(j)
    raise ValueError("a run is {start, count, writer, out, radix, base, step} or {start, count, held, row}")


def tile(runs: Sequence[Mapping[str, Any]], length: int, writer: str, out: str, units: int,
         held: Mapping[str, int]) -> list[dict[str, Any]]:
    """The runs in position order, refused as `order` refuses them, run by run and never position by position."""
    parsed = sorted((parse_run(r) for r in runs), key=lambda r: r["start"])
    at = 0
    for r in parsed:
        if r["count"] == 0:
            raise ValueError(f"a run at {r['start']} covers no position")
        if r["start"] != at:
            raise ValueError(f"a run starts at {r['start']}, not at {at}: the runs leave a gap or overlap")
        why = _run_why(r, writer, out, units, held)
        if why is not None:
            raise ValueError(why)
        at += r["count"]
    if at != length:
        raise ValueError(f"the runs cover {at} positions, not the table's {length}")
    return parsed


def order(runs: Sequence[Mapping[str, Any]], length: int, writer: str, out: str, units: int,
          held: Mapping[str, int]) -> list[tuple]:
    """Each position's source, ``("unit", u)`` or ``("held", value, row)``, from runs that tile ``[0, length)``: refuses
    an empty run, a gap or an overlap, a run of another writer or output, a run past its writer's units or its value's
    rows, and runs that end before or after ``length`` (`Spot.HistoryTable.order`)."""
    srcs: list[tuple] = []
    for r in tile(runs, length, writer, out, units, held):
        if "held" in r:
            srcs += [("held", r["held"], r["row"] + i) for i in range(r["count"])]
        else:
            srcs += [("unit", unit_at(r["radix"], r["base"], r["step"], i)) for i in range(r["count"])]
    return srcs


def string(row: Sequence[int] | np.ndarray, words: int, salt: bytes) -> bytes:
    """The ``b || c`` of a table row of ``words`` u16 words under ``salt``: its writer's output row's public string."""
    r = np.asarray(row, dtype="<u2")
    if r.shape != (words,):
        raise ValueError(f"a table row is {words} u16 words, not shape {r.shape}")
    return _INST.commit_string(_INST.default_key, sha512_row_seg_digest(r.tobytes(), 16 * words), bytes(salt))


class HistoryTable:
    """The prover's history table across windows: table ``value`` of ``program`` (`domain`), its rows of ``words`` u16 words
    appended in position order (`append`), each under its own salt, in a tree of ``capacity`` leaves with `placeholder`'s
    past its length. The tree is kept whole, so an append rehashes only the leaves and paths it changes: a window's rows
    cost about ``log2(capacity)`` hashes each, never the table's capacity again. It reads as a registered value does where
    a window commits one (`verity.ml.windows.interface.Window`'s ``registered``): ``rows``, ``salts``, ``words``,
    ``word_bits``, ``leaves`` (its capacity), ``root``, ``path`` and ``entry``; ``bcs`` are its rows' public strings."""

    word_bits = 16
    schema = SCHEMA

    def __init__(self, program: str, value: str, words: int, capacity: int = CAPACITY) -> None:
        if not cap_ok(capacity):
            raise ValueError(f"a table's capacity is a power of two of at least 2, not {capacity!r}")
        if not words_ok(words):
            raise ValueError(f"a table's rows are at least one word and under 2^64 bits, not {words!r} words")
        self.program, self.value, self.words, self.leaves = program, value, words, capacity
        self.domain = domain(program, value, words, capacity)
        self.rows: list[np.ndarray] = []
        self.salts: list[bytes] = []
        self.bcs: list[bytes] = []
        self._seen: set[bytes] = set()
        pad = _INST.tree_leaf(_INST.default_key, placeholder(words))
        self._levels = [bytearray(b"".join(row_leaf(self.domain, r, pad, SCHEMA) for r in range(capacity)))]
        while len(self._levels[-1]) > DIGEST_BYTES:
            self._levels.append(bytearray(DIGEST_BYTES * (len(self._levels[-1]) // (2 * DIGEST_BYTES))))
            self._rehash(len(self._levels) - 2, 0, len(self._levels[-1]) // DIGEST_BYTES)

    def _rehash(self, h: int, lo: int, hi: int) -> None:
        """Nodes ``[lo, hi)`` of the level above level ``h``, from their children."""
        below, here, n = memoryview(self._levels[h]), self._levels[h + 1], DIGEST_BYTES
        for j in range(lo, hi):
            here[n * j:n * j + n] = self.domain.node(h, j, bytes(below[2 * n * j:2 * n * j + n]),
                                                     bytes(below[2 * n * j + n:2 * n * j + 2 * n]))

    @property
    def root(self) -> bytes:
        return bytes(self._levels[-1])

    def append(self, rows: Any, salts: Sequence[bytes] | None = None) -> range:
        """Rows ``rows`` (``(k, words)`` u16) at the table's next ``k`` positions, each under its salt (default: fresh from
        the operating system), refused past the capacity or for a salt used before: equal salts on two rows reveal the XOR
        of their digests. -> their positions."""
        rows = np.asarray(rows, dtype=np.uint16)
        if rows.ndim != 2 or rows.shape[1] != self.words:
            raise ValueError(f"{self.value}: rows of {self.words} u16 words, not shape {rows.shape}")
        at, k = len(self.rows), len(rows)
        if at + k > self.leaves:
            raise ValueError(f"{self.value}: {at + k} rows past the table's capacity of {self.leaves}")
        salts = [_INST.fresh_salt() for _ in range(k)] if salts is None else [bytes(s) for s in salts]
        if len(salts) != k or any(len(s) != _INST.salt_bytes for s in salts):
            raise ValueError(f"{self.value}: one {_INST.salt_bytes}-byte salt per row")
        if len(set(salts)) != k or self._seen.intersection(salts):
            raise ValueError(f"{self.value}: a salt is never reused: equal salts on two rows reveal the XOR of their digests")
        if not k:
            return range(at, at)
        bcs = [string(r, self.words, s) for r, s in zip(rows, salts)]
        n, level = DIGEST_BYTES, self._levels[0]
        for i, bc in enumerate(bcs):
            level[n * (at + i):n * (at + i + 1)] = row_leaf(self.domain, at + i, _INST.tree_leaf(_INST.default_key, bc),
                                                            SCHEMA)
        lo, hi = at, at + k - 1
        for h in range(len(self._levels) - 1):
            lo, hi = lo >> 1, hi >> 1
            self._rehash(h, lo, hi + 1)
        self.rows += [r.copy() for r in rows]
        self.salts += salts
        self.bcs += bcs
        self._seen.update(salts)
        return range(at, at + k)

    def path(self, position: int) -> bytes:
        """Position ``position``'s path to the root: its siblings, leaf level first, 64 bytes each (`Opening`'s ``path``,
        `gates.embedding`'s)."""
        if not (_int(position) and 0 <= position < self.leaves):
            raise ValueError(f"{self.value}: position {position!r} outside the table's {self.leaves}")
        out, n = [], DIGEST_BYTES
        for level in self._levels[:-1]:
            j = position ^ 1
            out.append(bytes(level[n * j:n * j + n]))
            position >>= 1
        return b"".join(out)

    def entry(self) -> dict[str, Any]:
        """What a pin names, as a registered value's: ``{"value", "root", "leaves", "words", "domain"}``."""
        return {"value": self.value, "root": self.root.hex(), "leaves": self.leaves, "words": self.words,
                "domain": self.domain.domain_id.hex()}
