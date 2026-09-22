"""[E2, EOD 2026-09-11] The interior tree: leaves + Merkle tree + root + openings over an :class:`~e2e.interior.layout.InteriorLayout`,
built from E1's value rows (brief interface 2: JSONL ``{"g": <gate>, "k": "<kind>", "w": <bits>, "v": "<hex, LE>"}``, one row per gate,
one file per ``(request, step, root call)`` under ``interior/<req>/<step>/``).

Framing = the boundary trees' (``verity_vllm.hashing``): leaf ``H("verity-vllm/leaf/v1", id, dtype, "[n]", bytes)``, node
``verity-vllm/node/v1``, odd tail ``verity-vllm/lift/v1``, empty ``verity-vllm/empty/v1``, ``ids_digest = H("verity-vllm/ids/v1", ids)``.
Rank = layout order.  The tree is positional -- leaf hashes are built from bytes and ranks, never from parsed canonical ids (A2's
``o2-packed-path`` measured the adapter ``MerkleTree``'s id parsing at 8-9 us per leaf; this builder has no such term).

Openings are the existing single-leaf ``verity_vllm.commit.Opening`` (``index`` = rank; the word at ``flat`` is
``value[flat*B:(flat+1)*B]``) and a range opening over consecutive ranks (:class:`InteriorRangeOpening`); :func:`verify_opening` /
:func:`verify_range_opening` mirror ``verity_vllm.commit``'s without the id grammar.

Files (``write_tree``): ``index.json`` (schema ``verity-interior/index/v1``: tree, layout pin, root, leaf_count, ids_digest, per-leaf
rows with their sha256), ``gates.bin`` (every gate in layout order, int64 LE), ``leaf-<rank:06d>.bin`` (the leaf bytes).
``commitments.json#interior_roots["<req>/<step>"]`` gets :meth:`InteriorTree.record`; :func:`add_interior_roots` refuses to touch
``trees`` (the phase-0 / RU records) and asserts them equal to the phase-0 document.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import hashing
from verity_vllm.commit import Opening, fold_path, fold_range, path_shape, range_path_shape

from .layout import (
    INDEX_SCHEMA,
    INTERIOR_DUPLICATE_GATE,
    INTERIOR_INDEX_MALFORMED,
    INTERIOR_REGION_MISSING,
    INTERIOR_UNEXPECTED_GATE,
    INTERIOR_WIDTH_MISMATCH,
    ORIGIN,
    CallSpan,
    InteriorLayout,
    LeafSpec,
    layout_digest,
    layout_pin,
    rule_of,
)

__all__ = [
    "INTERIOR_VALUE_BYTES",
    "BuildReport",
    "InteriorRangeOpening",
    "InteriorTree",
    "Row",
    "RowArrays",
    "RuCommit",
    "add_interior_roots",
    "build_tree",
    "build_tree_arrays",
    "check_index_against_commitments",
    "commit_interiors",
    "commit_ru",
    "iter_rows",
    "iter_rows_dir",
    "leaf_hash_of_row",
    "load_index",
    "parse_rows_dir",
    "parse_rows_file",
    "recompute_root",
    "root_of_leaf_hashes",
    "load_tree",
    "read_word",
    "ru_dirs",
    "stage_e1",
    "verify_opening",
    "verify_range_opening",
    "write_commitments",
    "write_tree",
]

INTERIOR_VALUE_BYTES = "INTERIOR_VALUE_BYTES"
ROW_FORMAT = '{"g": <gate index>, "k": "<kind>", "w": <width bits>, "v": "<hex, LE words>"}'
INTERIOR_ROOTS_KEY = "interior_roots"
COMMITMENTS_FILE = "commitments.json"
GATES_FILE = "gates.bin"
INDEX_FILE = "index.json"


# --- E1's rows -------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    """One value row: gate index, width (bits), the word's little-endian bytes, the kind (informational)."""

    g: int
    w: int
    v: bytes
    k: str = ""


class RowError(ValueError):
    """A line of E1's JSONL that is not a row of the frozen format."""


def _parse_line(line: str, where: str) -> Row:
    try:
        d = json.loads(line)
        v = bytes.fromhex(str(d["v"]))
        return Row(int(d["g"]), int(d["w"]), v, str(d.get("k", "")))
    except (KeyError, TypeError, ValueError) as e:
        raise RowError(f"{where}: malformed row {line[:80]!r}: {type(e).__name__}: {e}; expected {ROW_FORMAT}") from e


def iter_rows(path: str | os.PathLike[str]) -> Iterator[Row]:
    """The rows of one JSONL file (blank lines skipped)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            s = line.strip()
            if s:
                yield _parse_line(s, f"{p.name}:{n}")


def iter_rows_dir(directory: str | os.PathLike[str]) -> Iterator[tuple[Path, Row]]:
    """Every row of every ``*.jsonl`` under ``directory`` (recursively; a call name's ``/`` may have become nested directories), files
    in sorted path order.  The gate index is the authority for a row's position; the file name is not consulted."""
    d = Path(directory)
    for p in sorted(d.rglob("*.jsonl")):
        for r in iter_rows(p):
            yield p, r


# --- rows as arrays (NIGHT 2026-09-12: the serial parts of the builder were the per-row Python objects) ------------------------------------

_ROW_RE = re.compile(rb'^\{"g":(\d+),"k":"[^"\\]*","w":(\d+),"v":"([0-9a-fA-F]*)"\}$', re.M)


@dataclass(slots=True)
class RowArrays:
    """The rows of one or more files as arrays: gate (int64), width (int64), the word as 8 little-endian bytes (``(n, 8)`` uint8; words
    wider than 64 bits are NOT here -- they are in ``wide`` as ``(g, w, bytes)``), the word's byte count (int64)."""

    g: np.ndarray
    w: np.ndarray
    v8: np.ndarray
    nbytes: np.ndarray
    wide: list[tuple[int, int, bytes]] = field(default_factory=list)
    rows: int = 0

    @classmethod
    def empty(cls) -> RowArrays:
        return cls(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 8), np.uint8), np.zeros(0, np.int64))

    @classmethod
    def concat(cls, parts: Sequence[RowArrays]) -> RowArrays:
        parts = [p for p in parts if p.rows]
        if not parts:
            return cls.empty()
        return cls(np.concatenate([p.g for p in parts]), np.concatenate([p.w for p in parts]), np.concatenate([p.v8 for p in parts]),
                   np.concatenate([p.nbytes for p in parts]), [x for p in parts for x in p.wide], sum(p.rows for p in parts))


def _rows_from_lists(gs: list[int], ws: list[int], vs: list[bytes], where: str) -> RowArrays:
    n = len(gs)
    g = np.asarray(gs, dtype=np.int64)
    w = np.asarray(ws, dtype=np.int64)
    nb = np.fromiter((len(v) for v in vs), dtype=np.int64, count=n)
    v8 = np.zeros((n, 8), dtype=np.uint8)
    wide: list[tuple[int, int, bytes]] = []
    for k in np.unique(nb).tolist():
        idx = np.flatnonzero(nb == k)
        if k == 0:
            continue
        if k > 8:
            wide.extend((int(g[i]), int(w[i]), vs[i]) for i in idx.tolist())
            continue
        v8[idx, :k] = np.frombuffer(b"".join(vs[i] for i in idx.tolist()), dtype=np.uint8).reshape(-1, k)
    return RowArrays(g, w, v8, nb, wide, n)


def parse_rows_file(path: str | os.PathLike[str]) -> RowArrays:
    """One ``*.jsonl`` file -> :class:`RowArrays`.  Fast path: the producer's exact compact spelling (``{"g":G,"k":"K","w":W,"v":"HEX"}``)
    is matched as ONE regex over the file; any line outside that spelling goes through ``json.loads`` (same acceptance as
    :func:`iter_rows`; a malformed line is :class:`RowError`).  No per-row Python object is created on the fast path."""
    p = Path(path)
    data = p.read_bytes()
    lines = [ln for ln in data.split(b"\n") if ln.strip()]
    n = len(lines)
    if n == 0:
        return RowArrays.empty()
    m = _ROW_RE.findall(data)
    if len(m) != n:
        # not every line is the compact spelling: json.loads per line (exactly `iter_rows`), arrays from the lists
        gs, ws, vs = [], [], []
        for i, ln in enumerate(lines, 1):
            r = _parse_line(ln.decode("utf-8"), f"{p.name}:{i}")
            gs.append(r.g)
            ws.append(r.w)
            vs.append(r.v)
        return _rows_from_lists(gs, ws, vs, p.name)
    g = np.array([t[0] for t in m], dtype="S20").astype(np.int64)
    w = np.array([t[1] for t in m], dtype="S6").astype(np.int64)
    hexs = [t[2] for t in m]
    hl = np.fromiter((len(h) for h in hexs), dtype=np.int64, count=n)
    if np.any(hl & 1):
        i = int(np.flatnonzero(hl & 1)[0])
        raise RowError(f"{p.name}:{i + 1}: malformed row {lines[i][:80]!r}: odd hex length; expected {ROW_FORMAT}")
    nb = hl // 2
    v8 = np.zeros((n, 8), dtype=np.uint8)
    wide: list[tuple[int, int, bytes]] = []
    for k in np.unique(nb).tolist():
        idx = np.flatnonzero(nb == k)
        if k == 0:
            continue
        if k > 8:
            wide.extend((int(g[i]), int(w[i]), bytes.fromhex(hexs[i].decode())) for i in idx.tolist())
            continue
        joined = b"".join(hexs[i] for i in idx.tolist())
        v8[idx, :k] = np.frombuffer(bytes.fromhex(joined.decode("ascii")), dtype=np.uint8).reshape(-1, k)
    return RowArrays(g, w, v8, nb, wide, n)


def _parse_rows_file_star(path: str) -> RowArrays:  # pool target
    return parse_rows_file(path)


def parse_rows_dir(directory: str | os.PathLike[str], *, workers: int | None = None, min_parallel_bytes: int = 4 << 20) -> Iterator[RowArrays]:
    """Stream the :class:`RowArrays` of every ``*.jsonl`` under ``directory`` (sorted path order).  With ``workers > 1`` and enough
    bytes the files are parsed by a process pool (chunk-parallel: one file = one root call; the pool is shared across calls -- one RU's
    decode step is ~1 M rows in ~90 files, a fresh pool per RU cost more than the parse); order is preserved."""
    files = sorted(str(p) for p in Path(directory).rglob("*.jsonl"))
    total = sum(os.path.getsize(f) for f in files)
    n_workers = int(workers if workers is not None else default_workers())
    if n_workers <= 1 or len(files) < 2 or total < min_parallel_bytes:
        for f in files:
            yield parse_rows_file(f)
        return
    yield from _shared_pool(n_workers).imap(_parse_rows_file_star, files, chunksize=1)


_POOLS: dict[int, Any] = {}


def _shared_pool(n_workers: int):
    """One process pool per worker count for the life of the process (``fork`` where available)."""
    pool = _POOLS.get(n_workers)
    if pool is None:
        import atexit
        import multiprocessing as mp
        import sys

        # fork is cheapest; inside a process that already holds a CUDA context / torch thread pools (the selective prover after its replay)
        # forking is unsafe, so use a fresh interpreter per worker there ($VERITOR_INTERIOR_MP overrides: fork | forkserver | spawn)
        method = os.environ.get("VERITOR_INTERIOR_MP")
        if not method:
            unsafe = "torch" in sys.modules or "cupy" in sys.modules
            method = ("forkserver" if unsafe else "fork") if hasattr(os, "fork") else "spawn"
        ctx = mp.get_context(method)
        pool = _POOLS[n_workers] = ctx.Pool(n_workers)
        atexit.register(pool.terminate)
    return pool


def default_workers() -> int:
    """``$VERITOR_INTERIOR_WORKERS`` or the machine's cores (≤ 32)."""
    env = os.environ.get("VERITOR_INTERIOR_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(os.cpu_count() or 1, 32))


# --- the tree --------------------------------------------------------------------------------------------------------------------------


def _eager(layout: Any) -> InteriorLayout:
    """The builder holds every word of the RU anyway: a verifier-side ``LazyInteriorLayout`` (E3's ``e2_layout`` / E9's required source)
    is materialised to the equal eager layout here; an ``InteriorLayout`` passes through."""
    m = getattr(layout, "materialize", None)
    return m() if m is not None and not isinstance(layout, InteriorLayout) else layout


def hash_leaves(layout: InteriorLayout, values: Sequence[bytes], *, workers: int | None = None) -> list[bytes]:
    """``leaf_hash(id, dtype, shape, bytes)`` of every leaf, in rank order -- hashed by a thread pool (``hashlib`` releases the GIL over
    the leaf bytes; 4096-word leaves are 8-32 KB), so the wall is the leaf payload divided by the cores, not one core."""
    n_workers = int(workers if workers is not None else default_workers())
    leaves = layout.leaves
    if n_workers <= 1 or len(leaves) < 64:
        return [hashing.leaf_hash(lf.canonical_id, lf.dtype, lf.shape, v) for lf, v in zip(leaves, values, strict=True)]

    def one(i: int) -> bytes:
        lf = leaves[i]
        return hashing.leaf_hash(lf.canonical_id, lf.dtype, lf.shape, values[i])

    with ThreadPoolExecutor(max_workers=min(n_workers, 16)) as ex:
        return list(ex.map(one, range(len(leaves)), chunksize=256))


def root_of_leaf_hashes(leaf_hashes: Iterable[bytes]) -> bytes:
    """The tree root over ``leaf_hashes`` in rank order -- the same tree as :func:`_levels` (pairs left to right, odd tail lifted), folded
    with O(log n) memory so a verifier can RECOMPUTE a shipped tree's root from its leaf files one leaf at a time (INTEGRATOR D35)."""
    pending: list[bytes | None] = []  # pending[k] = the unpaired node of level k, if any

    def push(k: int, h: bytes) -> None:
        while True:
            if k == len(pending):
                pending.append(h)
                return
            if pending[k] is None:
                pending[k] = h
                return
            left, pending[k] = pending[k], None
            h = hashing.node_hash(left, h)  # type: ignore[arg-type]
            k += 1

    n = 0
    for h in leaf_hashes:
        push(0, bytes(h))
        n += 1
    if n == 0:
        return hashing.empty_root()
    k = 0
    while k < len(pending) - 1:  # close the levels bottom-up: an odd tail is lifted into the level above
        if pending[k] is not None:
            h, pending[k] = pending[k], None
            push(k + 1, hashing.lift_hash(h))  # type: ignore[arg-type]
        k += 1
    top = pending[-1]
    assert top is not None and all(x is None for x in pending[:-1])
    return top


def _levels(leaf_hashes: Sequence[bytes]) -> list[list[bytes]]:
    level = list(leaf_hashes)
    levels = [level]
    while len(level) > 1:
        nxt = [hashing.node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(hashing.lift_hash(level[-1]))
        level = nxt
        levels.append(level)
    return levels


@dataclass(frozen=True, slots=True)
class InteriorRangeOpening:
    """Consecutive leaves ``[lo, hi)`` (ranks) of one interior tree with the boundary siblings per level; every leaf is rehashed from its
    own ``(id, dtype, shape, bytes)`` by :func:`verify_range_opening`."""

    tree: str
    root: bytes
    leaf_count: int
    lo: int
    hi: int
    leaves: tuple[tuple[str, str, tuple[int, ...], bytes], ...]
    path: tuple[tuple[bytes | None, bytes | None], ...]

    def replace(self, **changes: Any) -> InteriorRangeOpening:
        return dataclasses.replace(self, **changes)

    def to_json(self) -> dict[str, Any]:
        return {"kind": "range", "tree": self.tree, "root": self.root.hex(), "leaf_count": self.leaf_count, "lo": self.lo, "hi": self.hi,
                "leaves": [{"canonical_id": c, "dtype": d, "shape": list(s), "value_b64": base64.b64encode(b).decode("ascii")} for c, d, s, b in self.leaves],
                "path": [[None if s is None else s.hex() for s in pair] for pair in self.path]}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> InteriorRangeOpening:
        return cls(tree=str(doc["tree"]), root=bytes.fromhex(doc["root"]), leaf_count=int(doc["leaf_count"]), lo=int(doc["lo"]), hi=int(doc["hi"]),
                   leaves=tuple((str(x["canonical_id"]), str(x["dtype"]), tuple(int(d) for d in x["shape"]), base64.b64decode(x["value_b64"])) for x in doc["leaves"]),
                   path=tuple((None if a is None else bytes.fromhex(a), None if b is None else bytes.fromhex(b)) for a, b in doc["path"]))


class InteriorTree:
    """The Merkle tree over one RU's interior leaves: ``values[rank]`` are the leaf bytes of ``layout.leaves[rank]``."""

    __slots__ = ("layout", "leaf_hashes", "levels", "values")

    def __init__(self, layout: InteriorLayout, values: Sequence[bytes], *, leaf_hashes: Sequence[bytes] | None = None) -> None:
        layout = _eager(layout)
        if len(values) != layout.leaf_count:
            raise ValueError(f"{len(values)} leaf values for a layout of {layout.leaf_count} leaves")
        for lf, v in zip(layout.leaves, values, strict=True):
            if len(v) != lf.nbytes:
                raise ValueError(f"leaf {lf.canonical_id!r}: {len(v)} bytes, the layout says {lf.nbytes}")
        self.layout = layout
        self.values = [v if type(v) is bytes else bytes(v) for v in values]  # bytes pass through uncopied (payload held once)
        if leaf_hashes is None:
            leaf_hashes = hash_leaves(layout, self.values)
        self.leaf_hashes = [bytes(h) for h in leaf_hashes]
        self.levels = _levels(self.leaf_hashes) if self.leaf_hashes else []

    # -- identity ----------------------------------------------------------------------------------------------------------------------
    @property
    def tree_id(self) -> str:
        return self.layout.tree

    @property
    def leaf_count(self) -> int:
        return self.layout.leaf_count

    @property
    def root(self) -> bytes:
        return self.levels[-1][0] if self.leaf_hashes else hashing.empty_root()

    @property
    def ids_digest(self) -> bytes:
        return hashing.ids_digest(self.layout.ids)

    @property
    def depth(self) -> int:
        return len(self.levels) - 1 if self.leaf_hashes else 0

    def record(self) -> dict[str, Any]:
        """The ``commitments.json#interior_roots["<req>/<step>"]`` record."""
        origin = self.layout.leaves[0].origin if self.layout.leaves else ORIGIN
        return {"kind": origin, "origin": origin, "tree": self.tree_id, "root": self.root.hex(), "leaf_count": self.leaf_count, "ids_digest": self.ids_digest.hex(),
                "layout_digest": layout_digest(chunk_words=self.layout.chunk_words, rule=self.layout.rule), "words": self.layout.words, "nbytes": self.layout.nbytes}

    # -- openings ----------------------------------------------------------------------------------------------------------------------
    def path(self, rank: int) -> tuple[bytes | None, ...]:
        if not (0 <= rank < self.leaf_count):
            raise IndexError(rank)
        cursor, out = rank, []
        for level in self.levels[:-1]:
            if cursor == len(level) - 1 and len(level) % 2 == 1:
                out.append(None)
            else:
                out.append(level[cursor ^ 1])
            cursor //= 2
        return tuple(out)

    def open(self, rank: int) -> Opening:
        """The single-leaf opening of leaf ``rank`` (``verity_vllm.commit.Opening``; ``index`` = rank)."""
        lf = self.layout.leaves[rank]
        return Opening(tree=self.tree_id, root=self.root, leaf_count=self.leaf_count, index=rank, canonical_id=lf.canonical_id, dtype=lf.dtype,
                       shape=lf.shape, value=self.values[rank], path=self.path(rank))

    def open_gate(self, g: int) -> tuple[Opening, int]:
        """``(opening of the leaf holding gate g, flat)``; ``KeyError`` when the layout has no position for ``g``."""
        pos = self.layout.position(g)
        if pos is None:
            raise KeyError(f"gate {g} has no position in {self.tree_id}")
        rank, flat = pos
        return self.open(rank), flat

    def word(self, g: int) -> bytes:
        """The committed little-endian bytes of gate ``g``."""
        pos = self.layout.position(g)
        if pos is None:
            raise KeyError(f"gate {g} has no position in {self.tree_id}")
        rank, flat = pos
        b = self.layout.leaves[rank].word_nbytes
        return self.values[rank][flat * b:(flat + 1) * b]

    def range_path(self, lo: int, hi: int) -> tuple[tuple[bytes | None, bytes | None], ...]:
        out: list[tuple[bytes | None, bytes | None]] = []
        for (need_left, need_right), level in zip(range_path_shape(self.leaf_count, lo, hi), self.levels[:-1], strict=True):
            out.append((level[lo - 1] if need_left else None, level[hi] if need_right else None))
            lo, hi = (lo - int(need_left)) // 2, (hi + 1) // 2
        return tuple(out)

    def open_range(self, lo: int, hi: int) -> InteriorRangeOpening:
        """A range opening over ranks ``[lo, hi)``."""
        if not (0 <= lo < hi <= self.leaf_count):
            raise IndexError((lo, hi))
        leaves = tuple((lf.canonical_id, lf.dtype, lf.shape, self.values[r]) for r, lf in enumerate(self.layout.leaves) if lo <= r < hi)
        return InteriorRangeOpening(self.tree_id, self.root, self.leaf_count, lo, hi, leaves, self.range_path(lo, hi))

    # -- files ---------------------------------------------------------------------------------------------------------------------------
    def index_doc(self) -> dict[str, Any]:
        rows = []
        for lf, h in zip(self.layout.leaves, self.leaf_hashes, strict=True):
            r = lf.region()
            r["file"] = f"leaf-{lf.rank:06d}.bin"
            r["sha256"] = h.hex()
            rows.append(r)
        return {"schema": INDEX_SCHEMA, "tree": self.tree_id, "layout": layout_pin(self.layout.chunk_words, rule=self.layout.rule), "prefix": self.layout.prefix,
                "calls": [{"call": c.call, "lo": c.lo, "hi": c.hi} for c in self.layout.calls],
                "root": self.root.hex(), "leaf_count": self.leaf_count, "ids_digest": self.ids_digest.hex(), "words": self.layout.words,
                "nbytes": self.layout.nbytes, "gates_file": GATES_FILE, "leaves": rows,
                "hash": {"frame": hashing.FRAME_DESCRIPTION, "leaf": hashing.LEAF_TAG, "node": hashing.NODE_TAG, "lift": hashing.LIFT_TAG,
                         "ids": hashing.IDS_TAG, "empty": hashing.EMPTY_TAG, "rank": "layout order"}}


def leaf_hash_of_row(row: Mapping[str, Any], value: bytes) -> bytes:
    """The framed leaf hash of an ``index.json`` leaf row over ``value`` (what the row's ``sha256`` field holds)."""
    return hashing.leaf_hash(str(row["canonical_id"]), str(row["dtype"]), tuple(int(d) for d in row["shape"]), value)


def read_word(opening: Opening, flat: int) -> bytes:
    """The word at ``flat`` of an opened interior leaf (dtype ``uint<w>`` / ``u<w>`` -> ``ceil(w/8)`` bytes per word)."""
    n = int(opening.shape[0]) if opening.shape else 1
    if n <= 0 or len(opening.value) % n:
        raise ValueError("opened leaf bytes do not divide into its words")
    b = len(opening.value) // n
    if not 0 <= flat < n:
        raise IndexError(flat)
    return opening.value[flat * b:(flat + 1) * b]


def verify_opening(root: bytes, leaf_count: int, opening: Opening) -> bool:
    """Authenticate an interior leaf opening against ``root``: id, dtype, shape and bytes bind; the path shape is derived from
    ``(leaf_count, index)``; a self-describing opening must agree with the arguments.  Never raises."""
    try:
        if not isinstance(root, bytes | bytearray) or len(root) != hashing.DIGEST_BYTES:
            return False
        if opening.root != root or opening.leaf_count != leaf_count or leaf_count <= 0:
            return False
        if not isinstance(opening.canonical_id, str) or not opening.canonical_id:
            return False
        expected = path_shape(leaf_count, opening.index)
        if len(opening.path) != len(expected):
            return False
        for sibling, has_sibling in zip(opening.path, expected, strict=True):
            if has_sibling and (sibling is None or len(sibling) != hashing.DIGEST_BYTES):
                return False
            if not has_sibling and sibling is not None:
                return False
        return fold_path(opening.leaf(), opening.index, opening.path) == root
    except Exception:  # noqa: BLE001 -- a malformed opening is False, never an exception
        return False


def verify_range_opening(root: bytes, leaf_count: int, opening: InteriorRangeOpening) -> bool:
    """Authenticate a range opening: every leaf rehashed from its own fields, folded with the boundary siblings to ``root``."""
    try:
        if not isinstance(root, bytes | bytearray) or len(root) != hashing.DIGEST_BYTES:
            return False
        if opening.root != root or opening.leaf_count != leaf_count or leaf_count <= 0:
            return False
        if opening.hi - opening.lo != len(opening.leaves) or not opening.leaves:
            return False
        expected = range_path_shape(leaf_count, opening.lo, opening.hi)
        if len(opening.path) != len(expected):
            return False
        for (left, right), (need_left, need_right) in zip(opening.path, expected, strict=True):
            if need_left != (left is not None) or need_right != (right is not None):
                return False
            if any(s is not None and len(s) != hashing.DIGEST_BYTES for s in (left, right)):
                return False
        hashes = [hashing.leaf_hash(c, d, s, b) for c, d, s, b in opening.leaves]
        return fold_range(hashes, leaf_count, opening.lo, opening.path) == root
    except Exception:  # noqa: BLE001
        return False


# --- building from rows ----------------------------------------------------------------------------------------------------------------


@dataclass
class BuildReport:
    """What :func:`build_tree` saw: counts, refusals (problems -> no tree), timing."""

    rows: int = 0
    rows_placed: int = 0
    rows_unexpected: int = 0
    rows_boundary: int = 0
    rows_duplicate: int = 0
    words_missing: int = 0
    problems: list[str] = field(default_factory=list)
    seconds: float = 0.0
    nbytes: int = 0
    label: str = "MEASURED"

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["ok"] = self.ok
        d["rows_per_s"] = round(self.rows / self.seconds, 1) if self.seconds > 0 else None
        d["mb_per_s"] = round(self.nbytes / 1e6 / self.seconds, 3) if self.seconds > 0 else None
        return d

    def rows_per_s_str(self) -> str:
        return f"{self.rows} rows in {self.seconds:.2f}s = {self.rows / self.seconds:,.0f} rows/s MEASURED" if self.seconds > 0 else f"{self.rows} rows"


def build_tree(layout: InteriorLayout, rows: Iterable[Row], *, is_boundary: Callable[[int], bool] | None = None, strict_unexpected: bool = False,
               batch: int = 65536) -> tuple[InteriorTree | None, BuildReport]:
    """Place every row's word at its layout position and hash.  Refusals (``report.problems`` non-empty, tree ``None``):

    * ``INTERIOR_DUPLICATE_GATE`` -- a gate with a boundary position (``is_boundary(g)``) emitted as interior, or a gate seen twice;
    * ``INTERIOR_REGION_MISSING`` -- a layout word no row filled (named by leaf and gate);
    * ``INTERIOR_WIDTH_MISMATCH`` / ``INTERIOR_VALUE_BYTES`` -- a row's width differs from the layout's, or its bytes from ``ceil(w/8)``;
    * ``INTERIOR_UNEXPECTED_GATE`` -- a row for a gate the layout does not name: counted and ignored (the value is not required), refused
      with ``strict_unexpected``.
    """
    t0 = time.perf_counter()
    rep = BuildReport()
    layout = _eager(layout)
    bufs = [bytearray(lf.nbytes) for lf in layout.leaves]
    filled = np.zeros(layout.words, dtype=bool)
    leaf_start = layout._leaf_start  # noqa: SLF001 -- same module family
    widths = np.asarray([lf.width for lf in layout.leaves], dtype=np.int64)
    wb = np.asarray([lf.word_nbytes for lf in layout.leaves], dtype=np.int64)
    max_named = 16

    def flush(gs: list[int], ws: list[int], vs: list[bytes]) -> None:
        if not gs:
            return
        g = np.asarray(gs, dtype=np.int64)
        w = np.asarray(ws, dtype=np.int64)
        found, rank, flat = layout.positions(g)
        for i in np.flatnonzero(~found):
            gi = int(g[i])
            if is_boundary is not None and is_boundary(gi):
                rep.rows_boundary += 1
                if rep.rows_boundary <= max_named:
                    rep.problems.append(f"{INTERIOR_DUPLICATE_GATE}: gate {gi} has a boundary position (step-1 value) and is emitted as interior in {layout.tree}")
            else:
                rep.rows_unexpected += 1
                if strict_unexpected and rep.rows_unexpected <= max_named:
                    rep.problems.append(f"{INTERIOR_UNEXPECTED_GATE}: gate {gi} is not a required interior gate of {layout.tree}")
        idx = np.flatnonzero(found)
        if idx.shape[0] == 0:
            return
        r, f = rank[idx], flat[idx]
        pos = leaf_start[r] + f
        bad_w = w[idx] != widths[r]
        for i in np.flatnonzero(bad_w)[:max_named]:
            j = int(idx[i])
            rep.problems.append(f"{INTERIOR_WIDTH_MISMATCH}: gate {int(g[j])} row width {int(w[j])} != layout width {int(widths[r[i]])} (leaf {layout.leaves[int(r[i])].canonical_id!r})")
        dup = filled[pos].copy()  # already placed by an earlier batch ...
        _, first, counts = np.unique(pos, return_index=True, return_counts=True)
        if np.any(counts > 1):  # ... or twice within this batch: every occurrence after the first is a duplicate
            keep = np.zeros(pos.shape, dtype=bool)
            keep[first] = True
            dup |= ~keep
        for i in np.flatnonzero(dup)[:max_named]:
            rep.problems.append(f"{INTERIOR_DUPLICATE_GATE}: gate {int(g[idx[i]])} committed twice in {layout.tree}")
        rep.rows_duplicate += int(dup.sum())
        need = wb[r]
        bad_v = np.fromiter((len(vs[int(j)]) != int(b) for j, b in zip(idx, need, strict=True)), dtype=bool, count=idx.shape[0])
        for i in np.flatnonzero(bad_v)[:max_named]:
            j = int(idx[i])
            rep.problems.append(f"{INTERIOR_VALUE_BYTES}: gate {int(g[j])} carries {len(vs[j])} bytes, width {int(w[j])} needs {int(need[i])}")
        good = ~(bad_w | dup | bad_v)
        for i in np.flatnonzero(good):
            b = int(need[i])
            fl = int(f[i])
            bufs[int(r[i])][fl * b:(fl + 1) * b] = vs[int(idx[i])]
        rep.rows_placed += int(good.sum())
        filled[pos[good]] = True

    gs: list[int] = []
    ws: list[int] = []
    vs: list[bytes] = []
    for row in rows:
        rep.rows += 1
        gs.append(row.g)
        ws.append(row.w)
        vs.append(row.v)
        if len(gs) >= batch:
            flush(gs, ws, vs)
            gs, ws, vs = [], [], []
    flush(gs, ws, vs)
    missing = np.flatnonzero(~filled)
    rep.words_missing = int(missing.shape[0])
    if rep.words_missing:
        by_leaf: dict[int, int] = {}
        for k in missing:
            rnk = int(np.searchsorted(leaf_start, k, side="right")) - 1
            by_leaf[rnk] = by_leaf.get(rnk, 0) + 1
        for rnk, n in list(by_leaf.items())[:max_named]:
            lf = layout.leaves[rnk]
            first_missing = int(layout.gates[missing[np.searchsorted(missing, leaf_start[rnk])]])
            rep.problems.append(f"{INTERIOR_REGION_MISSING}: {layout.tree} leaf {lf.canonical_id!r} (rank {rnk}, gates {lf.g_first}..{lf.g_last}): {n} of {lf.words} words "
                                f"not supplied (first missing gate {first_missing})")
        if len(by_leaf) > max_named:
            rep.problems.append(f"{INTERIOR_REGION_MISSING}: ... {len(by_leaf) - max_named} more leaves with missing words")
    rep.nbytes = layout.nbytes
    if rep.problems:
        rep.seconds = time.perf_counter() - t0
        return None, rep
    tree = InteriorTree(layout, [bytes(b) for b in bufs])
    rep.seconds = time.perf_counter() - t0
    return tree, rep


class _WordStore:
    """The layout's words as one ``(words_w, nbytes_w)`` uint8 array per width (payload held once, vectorised placement); leaf ``rank``
    is the contiguous slice ``[wstart[rank], wstart[rank] + words)`` of its width's array."""

    __slots__ = ("arrays", "wstart", "widths", "wb")

    def __init__(self, layout: InteriorLayout) -> None:
        self.widths = np.asarray([lf.width for lf in layout.leaves], dtype=np.int64)
        self.wb = np.asarray([lf.word_nbytes for lf in layout.leaves], dtype=np.int64)
        words = np.asarray([lf.words for lf in layout.leaves], dtype=np.int64)
        self.wstart = np.zeros(len(layout.leaves), dtype=np.int64)
        self.arrays: dict[int, np.ndarray] = {}
        for w in np.unique(self.widths).tolist():
            ranks = np.flatnonzero(self.widths == w)
            starts = np.cumsum(np.concatenate(([0], words[ranks][:-1])))
            self.wstart[ranks] = starts
            self.arrays[int(w)] = np.zeros((int(words[ranks].sum()), int(self.wb[ranks[0]])), dtype=np.uint8)

    def place(self, rank: np.ndarray, flat: np.ndarray, v8: np.ndarray, wide: Sequence[tuple[int, bytes]] = ()) -> None:
        """Scatter rows (leaf rank, flat, 8 LE value bytes) into the store; ``wide`` = (row index, bytes) for words > 8 bytes."""
        pos = self.wstart[rank] + flat
        for w in np.unique(self.widths[rank]).tolist():
            sel = self.widths[rank] == w
            arr = self.arrays[int(w)]
            nb = arr.shape[1]
            if nb <= 8:
                arr[pos[sel]] = v8[sel, :nb]
        for i, b in wide:
            arr = self.arrays[int(self.widths[rank[i]])]
            arr[pos[i]] = np.frombuffer(b, dtype=np.uint8)

    def leaf_bytes(self, rank: int, words: int) -> bytes:
        a = self.arrays[int(self.widths[rank])]
        s = int(self.wstart[rank])
        return a[s:s + words].tobytes()


def build_tree_arrays(layout: InteriorLayout, batches: Iterable[RowArrays], *, is_boundary: Callable[[int], bool] | None = None,
                      strict_unexpected: bool = False, workers: int | None = None) -> tuple[InteriorTree | None, BuildReport]:
    """:func:`build_tree` over :class:`RowArrays` batches (one per file / pool result): the same refusals, by name, with no per-row Python
    object -- positions, width/duplicate/byte checks and the placement are numpy over each batch; leaves are hashed by threads."""
    t0 = time.perf_counter()
    rep = BuildReport()
    layout = _eager(layout)
    store = _WordStore(layout)
    filled = np.zeros(layout.words, dtype=bool)
    leaf_start = layout._leaf_start  # noqa: SLF001
    widths, wb = store.widths, store.wb
    max_named = 16
    for ra in batches:
        if not ra.rows:
            continue
        rep.rows += ra.rows
        found, rank, flat = layout.positions(ra.g)
        miss = np.flatnonzero(~found)
        if miss.shape[0]:
            if is_boundary is not None:
                bmask = np.fromiter((is_boundary(int(x)) for x in ra.g[miss]), dtype=bool, count=miss.shape[0])
            else:
                bmask = np.zeros(miss.shape, dtype=bool)
            nb_ = int(bmask.sum())
            for gi in ra.g[miss[bmask]][:max(0, max_named - rep.rows_boundary)].tolist():
                rep.problems.append(f"{INTERIOR_DUPLICATE_GATE}: gate {gi} has a boundary position (step-1 value) and is emitted as interior in {layout.tree}")
            rep.rows_boundary += nb_
            nu = int(miss.shape[0] - nb_)
            if strict_unexpected:
                for gi in ra.g[miss[~bmask]][:max(0, max_named - rep.rows_unexpected)].tolist():
                    rep.problems.append(f"{INTERIOR_UNEXPECTED_GATE}: gate {gi} is not a required interior gate of {layout.tree}")
            rep.rows_unexpected += nu
        idx = np.flatnonzero(found)
        if idx.shape[0] == 0:
            continue
        g, w, r, f = ra.g[idx], ra.w[idx], rank[idx], flat[idx]
        pos = leaf_start[r] + f
        bad_w = w != widths[r]
        for i in np.flatnonzero(bad_w)[:max_named].tolist():
            rep.problems.append(f"{INTERIOR_WIDTH_MISMATCH}: gate {int(g[i])} row width {int(w[i])} != layout width {int(widths[r[i]])} (leaf {layout.leaves[int(r[i])].canonical_id!r})")
        dup = filled[pos].copy()
        _, first, counts = np.unique(pos, return_index=True, return_counts=True)
        if np.any(counts > 1):
            keep = np.zeros(pos.shape, dtype=bool)
            keep[first] = True
            dup |= ~keep
        for i in np.flatnonzero(dup)[:max_named].tolist():
            rep.problems.append(f"{INTERIOR_DUPLICATE_GATE}: gate {int(g[i])} committed twice in {layout.tree}")
        rep.rows_duplicate += int(dup.sum())
        need = wb[r]
        bad_v = ra.nbytes[idx] != need
        for i in np.flatnonzero(bad_v)[:max_named].tolist():
            rep.problems.append(f"{INTERIOR_VALUE_BYTES}: gate {int(g[i])} carries {int(ra.nbytes[idx][i])} bytes, width {int(w[i])} needs {int(need[i])}")
        good = ~(bad_w | dup | bad_v)
        wide: list[tuple[int, bytes]] = []
        if ra.wide:  # words wider than 64 bits: their bytes are in `wide`, keyed by gate; k = the row's index among the placed rows
            wmap = {gg: b for gg, _w, b in ra.wide}
            wide = [(k, wmap[int(g[j])]) for k, j in enumerate(np.flatnonzero(good).tolist()) if int(need[j]) > 8]
        store.place(r[good], f[good], ra.v8[idx][good], wide)
        rep.rows_placed += int(good.sum())
        filled[pos[good]] = True
    missing = np.flatnonzero(~filled)
    rep.words_missing = int(missing.shape[0])
    if rep.words_missing:
        rnk = np.searchsorted(leaf_start, missing, side="right") - 1
        ranks, counts = np.unique(rnk, return_counts=True)
        for rk, n in list(zip(ranks.tolist(), counts.tolist(), strict=True))[:max_named]:
            lf = layout.leaves[rk]
            first_missing = int(layout.gates[missing[np.searchsorted(missing, leaf_start[rk])]])
            rep.problems.append(f"{INTERIOR_REGION_MISSING}: {layout.tree} leaf {lf.canonical_id!r} (rank {rk}, gates {lf.g_first}..{lf.g_last}): {n} of {lf.words} words "
                                f"not supplied (first missing gate {first_missing})")
        if ranks.shape[0] > max_named:
            rep.problems.append(f"{INTERIOR_REGION_MISSING}: ... {ranks.shape[0] - max_named} more leaves with missing words")
    rep.nbytes = layout.nbytes
    if rep.problems:
        rep.seconds = time.perf_counter() - t0
        return None, rep
    values = [store.leaf_bytes(lf.rank, lf.words) for lf in layout.leaves]
    del store
    tree = InteriorTree(layout, values, leaf_hashes=hash_leaves(layout, values, workers=workers))
    rep.seconds = time.perf_counter() - t0
    return tree, rep


# --- files -----------------------------------------------------------------------------------------------------------------------------


def _write_json(path: Path, doc: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_tree(tree: InteriorTree, directory: str | os.PathLike[str]) -> Path:
    """``index.json`` + ``gates.bin`` + ``leaf-<rank:06d>.bin`` under ``directory`` (created); returns the index path."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    leaves, values = tree.layout.leaves, tree.values

    def one(i: int) -> None:
        (d / f"leaf-{leaves[i].rank:06d}.bin").write_bytes(values[i])

    if len(leaves) < 64 or default_workers() <= 1:
        for i in range(len(leaves)):
            one(i)
    else:
        with ThreadPoolExecutor(max_workers=min(default_workers(), 16)) as ex:
            list(ex.map(one, range(len(leaves)), chunksize=64))
    (d / GATES_FILE).write_bytes(tree.layout.gates_bin())
    _write_json(d / INDEX_FILE, tree.index_doc())
    return d / INDEX_FILE


def load_index(directory: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads((Path(directory) / INDEX_FILE).read_text())


def _layout_from_index(index: Mapping[str, Any], gates: np.ndarray) -> InteriorLayout:
    """Reconstruct the layout the index claims (for re-hashing a stored tree; the verifier compares it with its own derivation)."""
    leaves: list[LeafSpec] = []
    cursor = 0
    for row in index.get("leaves") or []:
        n = int(row["words"])
        leaves.append(LeafSpec(int(row["rank"]), str(row["canonical_id"]), str(row["call"]), int(row["chunk"]), int(row["width"]), np.ascontiguousarray(gates[cursor:cursor + n]),
                              origin=str(row.get("origin", ORIGIN))))
        cursor += n
    if cursor != gates.shape[0]:
        raise ValueError(f"{INTERIOR_INDEX_MALFORMED}: gates.bin holds {gates.shape[0]} gates, the index {cursor} words")
    calls = [CallSpan(str(c["call"]), int(c["lo"]), int(c["hi"])) for c in index.get("calls") or []]
    framing = str((index.get("layout") or {}).get("framing", "chunks-4096"))
    chunk_words = int(framing.split("-", 1)[1]) if framing.startswith("chunks-") else 4096
    return InteriorLayout(str(index["tree"]), leaves, calls, prefix=str(index.get("prefix", "")), chunk_words=chunk_words, rule=rule_of(index.get("layout") or {}))


def load_tree(directory: str | os.PathLike[str], layout: InteriorLayout | None = None, *, recompute: bool = True) -> InteriorTree:
    """Rebuild the tree from a written directory.  With ``layout`` (the verifier's own derivation) the stored leaves are read into IT --
    a stored leaf set that differs is a ``ValueError`` (use :func:`~e2e.interior.layout.validate_supplied` first for the named refusal);
    without it the index's claimed layout is used.  ``recompute`` rehashes every leaf from its bytes (else the index's sha256 is trusted)."""
    d = Path(directory)
    index = load_index(d)
    gates = np.frombuffer((d / GATES_FILE).read_bytes(), dtype="<i8")
    lay = layout if layout is not None else _layout_from_index(index, gates)
    rows = list(index.get("leaves") or [])
    if len(rows) != lay.leaf_count:
        raise ValueError(f"index lists {len(rows)} leaves, the layout has {lay.leaf_count}")
    values = [(d / str(r.get("file", f"leaf-{i:06d}.bin"))).read_bytes() for i, r in enumerate(rows)]
    hashes = None if recompute else [bytes.fromhex(r["sha256"]) for r in rows]
    tree = InteriorTree(lay, values, leaf_hashes=hashes)
    if layout is not None and not np.array_equal(gates, lay.gates):
        raise ValueError("stored gates.bin differs from the derived layout's gate order")
    return tree


def recompute_root(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """[D35] The tree's ``root`` / ``leaf_count`` / ``ids_digest`` RECOMPUTED from the shipped files -- every leaf hashed from its bytes under the
    index's ``(canonical_id, dtype, shape)``, the root folded with O(log n) memory (:func:`root_of_leaf_hashes`), one leaf file in memory at a
    time.  Returns ``{"root", "leaf_count", "ids_digest", "index_root", "leaf_sha_mismatches", "ok"}``; ``ok`` = recomputed root ==
    ``index.json#root`` and no leaf whose bytes hash differently from its index row.  What E9's ``V3.trees_rehashed`` compares with
    ``commitments.json#interior_roots[key]`` -- :func:`load_tree` ``(recompute=True)`` does the same but holds the whole tree (values + layout,
    ~36 B/word: 2 GB at the 128x64 prefill RU); this holds one leaf."""
    d = Path(directory)
    index = load_index(d)
    rows = list(index.get("leaves") or [])
    mism = 0

    def hashes() -> Iterator[bytes]:
        nonlocal mism
        for i, r in enumerate(rows):
            h = leaf_hash_of_row(r, (d / str(r.get("file", f"leaf-{i:06d}.bin"))).read_bytes())
            if h.hex() != r.get("sha256"):
                mism += 1
            yield h

    root = root_of_leaf_hashes(hashes()).hex()
    ids = hashing.ids_digest([str(r.get("canonical_id")) for r in rows]).hex()
    return {"root": root, "leaf_count": len(rows), "ids_digest": ids, "index_root": index.get("root"), "leaf_sha_mismatches": mism,
            "ok": root == index.get("root") and mism == 0 and ids == index.get("ids_digest") and len(rows) == int(index.get("leaf_count", -1))}


def check_index_against_commitments(index: Mapping[str, Any], commitments: Mapping[str, Any], ru_key: str) -> list[str]:
    """The index's root / leaf_count / ids_digest must equal ``commitments.json#interior_roots[ru_key]`` (the committed record the
    chronology's interior digest binds); the ids_digest must be the digest of the index's ids in rank order."""
    p: list[str] = []
    rec = (commitments.get(INTERIOR_ROOTS_KEY) or {}).get(ru_key)
    if rec is None:
        return [f"{INTERIOR_REGION_MISSING}: commitments.json has no interior_roots[{ru_key!r}] (retained RU without a step-3 interior root)"]
    for k in ("root", "leaf_count", "ids_digest"):
        if rec.get(k) != index.get(k):
            p.append(f"interior_roots[{ru_key!r}].{k} = {str(rec.get(k))[:16]} != index.json {str(index.get(k))[:16]}")
    ids = [str(r.get("canonical_id")) for r in index.get("leaves") or []]
    if hashing.ids_digest(ids).hex() != index.get("ids_digest"):
        p.append("index.json ids_digest is not the digest of its ids in rank order")
    if rec.get("layout_digest") != (index.get("layout") or {}).get("digest"):
        p.append("interior_roots layout_digest != index.json layout.digest")
    return p


def add_interior_roots(commitments: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], *, phase0: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A copy of ``commitments`` with ``interior_roots`` = ``records`` (``"<req>/<step>" -> InteriorTree.record()``).  ``trees`` (the
    phase-0 boundary / weights records and the RU trees) is copied untouched and, when ``phase0`` (``commitments_phase0.json``) is
    given, every phase-0 record is asserted equal (``AssertionError`` otherwise -- initial values are immutable)."""
    out = dict(commitments)
    trees = out.get("trees") or {}
    if phase0 is not None:
        for tid, rec in (phase0.get("trees") or {}).items():
            if trees.get(tid) != rec:
                raise AssertionError(f"phase-0 record of {tid} would change: interior commitments never rewrite initial trees")
    if INTERIOR_ROOTS_KEY in out and out[INTERIOR_ROOTS_KEY]:
        for k in records:
            if k in out[INTERIOR_ROOTS_KEY] and out[INTERIOR_ROOTS_KEY][k] != dict(records[k]):
                raise AssertionError(f"interior root of {k} already recorded with another value: a second interior commitment for the same RU is refused")
    merged = dict(out.get(INTERIOR_ROOTS_KEY) or {})
    merged.update({k: dict(v) for k, v in records.items()})
    out[INTERIOR_ROOTS_KEY] = merged
    return out


def write_commitments(run_dir: str | os.PathLike[str], doc: Mapping[str, Any]) -> Path:
    """Write ``commitments.json`` in the capture's format (``indent=1``, key order kept; atomic replace)."""
    path = Path(run_dir) / COMMITMENTS_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    os.replace(tmp, path)
    return path


# --- the prover's interior phase: E1's directory -> trees -> roots ---------------------------------------------------------------------

CALLS_FILE = "calls.json"
REQUIRED_FILE = "required.json"
INTERIOR_CALLS_MISSING = "INTERIOR_CALLS_MISSING"
INTERIOR_NO_ROWS = "INTERIOR_NO_ROWS"
INTERIOR_ROW_MALFORMED = "INTERIOR_ROW_MALFORMED"


@dataclass
class RuCommit:
    """The outcome of committing one RU's interior: its key, the tree directory written (None when refused), the
    ``interior_roots`` record, the build report and the refusal lines (empty = committed)."""

    ru: str
    request_id: str
    step: str
    source: Path
    tree_dir: Path | None = None
    record: dict[str, Any] | None = None
    report: BuildReport | None = None
    problems: list[str] = field(default_factory=list)
    seconds_parse: float = 0.0
    seconds_write: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.problems and self.record is not None

    def to_json(self) -> dict[str, Any]:
        return {"ru": self.ru, "request_id": self.request_id, "step": self.step, "source": str(self.source), "tree_dir": str(self.tree_dir) if self.tree_dir else None,
                "ok": self.ok, "record": self.record, "report": self.report.to_json() if self.report else None, "problems": self.problems,
                "seconds_parse": round(self.seconds_parse, 3), "seconds_write": round(self.seconds_write, 3)}


def ru_dirs(values_root: str | os.PathLike[str]) -> dict[str, Path]:
    """E1's directory (brief interface 2: ``<root>/<req>/<step>/<call canonical id>.jsonl``) -> ``{"<req>/<step>": dir}`` for every RU
    directory holding at least one ``*.jsonl``, sorted.  ``<step>`` is taken as spelled (the capture spells it ``%06d``)."""
    root = Path(values_root)
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for req in sorted(p for p in root.iterdir() if p.is_dir()):
        for step in sorted(p for p in req.iterdir() if p.is_dir()):
            if any(step.rglob("*.jsonl")):
                out[f"{req.name}/{step.name}"] = step
    return out


def _read_calls(ru_dir: Path, spec: Mapping[str, Any] | None) -> tuple[list[CallSpan], str]:
    """The RU's pinned call order: ``spec`` (an entry of ``--interior-calls``) or ``<ru_dir>/calls.json``; either ``[{"call","lo","hi"}]``
    or ``{"calls": [...], "prefix": "..."}``.  Returns ``(calls, prefix)``; ``FileNotFoundError`` when neither exists."""
    doc: Any = spec
    if doc is None:
        p = ru_dir / CALLS_FILE
        if not p.exists():
            raise FileNotFoundError(2, "no call order", str(p))
        doc = json.loads(p.read_text())
    prefix = ""
    if isinstance(doc, Mapping):
        prefix = str(doc.get("prefix", ""))
        doc = doc.get("calls") or []
    calls = [CallSpan(str(c["call"]), int(c["lo"]), int(c["hi"])) if isinstance(c, Mapping) else CallSpan(str(c[0]), int(c[1]), int(c[2])) for c in doc]
    return calls, prefix


def _read_required(ru_dir: Path) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Optional ``<ru_dir>/required.json`` (E1's inventory / E3's oracle): ``{"gates": [...], "widths": [...]}``, ``{"gates": [...]}`` (widths
    taken from the rows), ``{"gates_bin": "<file>", "dtype": "<u4"}`` (a sorted binary gate list next to it) or ``[[g, w], ...]``.  With it the
    prover commits the *required* layout and a missing word is a refusal here; without it the rows define the layout."""
    p = ru_dir / REQUIRED_FILE
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    w: np.ndarray | None = None
    if isinstance(doc, Mapping):
        if "gates_bin" in doc:
            g = np.fromfile(ru_dir / str(doc["gates_bin"]), dtype=str(doc.get("dtype", "<u4"))).astype(np.int64)
        else:
            g = np.asarray(doc["gates"], dtype=np.int64)
        if doc.get("widths") is not None:
            w = np.asarray(doc["widths"], dtype=np.int64)
        elif "widths_bin" in doc:  # one width per gate of gates_bin, same order (uint8 unless widths_dtype says otherwise)
            w = np.fromfile(ru_dir / str(doc["widths_bin"]), dtype=str(doc.get("widths_dtype", "u1"))).astype(np.int64)
        if w is not None and w.shape != g.shape:
            raise ValueError(f"{REQUIRED_FILE}: {g.shape[0]} gates but {w.shape[0]} widths")
    else:
        arr = np.asarray(doc, dtype=np.int64).reshape(-1, 2)
        g, w = arr[:, 0], arr[:, 1]
    order = np.argsort(g, kind="stable")
    return g[order], (w[order] if w is not None else None)


def _widths_from_rows(source: Path, gates: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Widths of ``gates`` (sorted) from the rows under ``source``; returns ``(widths, gates without a row)``."""
    w = np.zeros(gates.shape, dtype=np.int64)
    seen = np.zeros(gates.shape, dtype=bool)
    gs: list[int] = []
    ws: list[int] = []

    def flush() -> None:
        if not gs:
            return
        g = np.asarray(gs, dtype=np.int64)
        k = np.searchsorted(gates, g)
        kk = np.minimum(k, gates.shape[0] - 1)
        hit = gates[kk] == g
        w[kk[hit]] = np.asarray(ws, dtype=np.int64)[hit]
        seen[kk[hit]] = True

    for _, r in iter_rows_dir(source):
        gs.append(r.g)
        ws.append(r.w)
        if len(gs) >= 65536:
            flush()
            gs, ws = [], []
    flush()
    return w, gates[~seen].tolist()


def stage_e1(e1_out: str | os.PathLike[str], values_root: str | os.PathLike[str], *, link: bool = True) -> list[str]:
    """Stage E1's producer output (``<e1_out>/inventory.json`` + ``required_gates.bin`` + ``interior/<req>/<step>/*.jsonl``) as the directory
    ``--interior-values`` consumes: per RU ``<values_root>/<req>/<step>/`` with the row files (symlinked, or copied with ``link=False``),
    ``calls.json`` (the inventory's root calls: ``name`` / ``lo`` / ``hi``, the canonical-id prefix) and ``required.json`` (the RU's slice of
    ``required_gates.bin``).  Returns the RU keys staged.  E10 runs this once E1's pod output is on the machine."""
    src, dst = Path(e1_out), Path(values_root)
    inv = json.loads((src / "inventory.json").read_text())
    req_id = str(inv["request_id"])
    rg = inv.get("required_gates_bin") or {}
    required = np.fromfile(src / str(rg.get("path", "required_gates.bin")), dtype=str(rg.get("dtype", "<u4"))).astype(np.int64)
    required.sort()
    by_step: dict[str, list[Mapping[str, Any]]] = {}
    for f in inv["files"]:
        by_step.setdefault(f"{int(f['step']):06d}", []).append(f)
    staged: list[str] = []
    for step, files in sorted(by_step.items()):
        files = sorted(files, key=lambda f: int(f["lo"]))
        d = dst / req_id / step
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = (src / str(f["path"])).resolve()
            out = d / Path(str(f["path"])).name
            if out.exists() or out.is_symlink():
                out.unlink()
            if link:
                out.symlink_to(target)
            else:
                shutil.copyfile(target, out)
        prefix = str(files[0]["canonical_id"]).split("/t=")[0] + "/"
        calls = [{"call": str(f["name"]), "lo": int(f["lo"]), "hi": int(f["hi"])} for f in files]
        (d / CALLS_FILE).write_text(json.dumps({"prefix": prefix, "calls": calls, "source": "E1 inventory.json"}, indent=1) + "\n")
        # [E1 SCALE 2026-09-13] `required` is sorted: mark each call's [lo, hi) slice by bisection instead of two full-array comparisons
        # per call -- 11,650 prefill calls x 97.8 M required gates (128x64) made this loop ~1 h (10 min at 100 % CPU with no file written,
        # 4090 pod 02:53Z); identical mask, identical `sel`.
        mask = np.zeros(required.shape, dtype=bool)
        for c in calls:
            a, b = np.searchsorted(required, int(c["lo"]), side="left"), np.searchsorted(required, int(c["hi"]), side="left")
            mask[a:b] = True
        sel = required[mask]
        sel.astype("<i8").tofile(d / "required_gates.bin")
        (d / REQUIRED_FILE).write_text(json.dumps({"gates_bin": "required_gates.bin", "dtype": "<i8", "count": int(sel.shape[0]),
                                                   "source": f"E1 {rg.get('path', 'required_gates.bin')} restricted to the RU's calls"}) + "\n")
        staged.append(f"{req_id}/{step}")
    return staged


def commit_ru(ru: str, source: Path, out_dir: Path, *, calls_spec: Mapping[str, Any] | None = None, is_boundary: Callable[[int], bool] | None = None,
              strict_unexpected: bool = False, workers: int | None = None) -> RuCommit:
    """Commit one RU's interior from E1's rows under ``source`` into ``out_dir`` (``adapter/selective/interior/<req>/<step>/``).  Never
    raises for the prover's data: a refusal is returned in ``problems`` (the run continues; the RU requests exemption).

    [NIGHT 2026-09-12] ONE pass over the row files, parsed file-parallel by ``workers`` processes into arrays (:func:`parse_rows_dir`);
    the widths the required list lacks come from those arrays (no second pass); placement, checks and leaf hashing are vectorised /
    threaded (:func:`build_tree_arrays`).  Same layout, same leaves, same roots, same refusals as the row-by-row path."""
    req, _, step = ru.partition("/")
    rc = RuCommit(ru, req, step, source)
    tree = f"interior/{ru}"
    try:
        calls, prefix = _read_calls(source, calls_spec)
    except FileNotFoundError as e:
        rc.problems.append(f"{INTERIOR_CALLS_MISSING}: {ru}: no pinned call order (--interior-calls entry or {e.filename}); the leaf ids need the program's call order")
        return rc
    except (KeyError, TypeError, ValueError, IndexError) as e:
        rc.problems.append(f"{INTERIOR_CALLS_MISSING}: {ru}: malformed call order: {type(e).__name__}: {str(e)[:120]}")
        return rc
    t_parse = time.perf_counter()
    try:
        req_gw = _read_required(source)
    except (OSError, ValueError, KeyError, TypeError) as e:
        rc.problems.append(f"{ru}: malformed {REQUIRED_FILE}: {type(e).__name__}: {str(e)[:160]}")
        return rc
    streaming = req_gw is not None and req_gw[1] is not None  # gates AND widths given: the layout needs no pass over the rows
    batches: Iterable[RowArrays] = parse_rows_dir(source, workers=workers)
    if not streaming:
        # the layout's widths come from the rows: hold the arrays (~26 B/row) for the placement pass
        try:
            batches = list(batches)
        except RowError as e:
            rc.problems.append(f"{INTERIOR_ROW_MALFORMED}: {ru}: {str(e)[:200]}")
            return rc
        rc.seconds_parse = time.perf_counter() - t_parse
    try:
        if req_gw is not None:
            g_req, w_req = req_gw
            if w_req is None:  # widths from the rows' arrays (the required list names gates only)
                w_req = np.zeros(g_req.shape, dtype=np.int64)
                seen = np.zeros(g_req.shape, dtype=bool)
                for ra in batches:
                    if not ra.rows:
                        continue
                    k = np.searchsorted(g_req, ra.g)
                    kk = np.minimum(k, g_req.shape[0] - 1)
                    hit = g_req[kk] == ra.g
                    w_req[kk[hit]] = ra.w[hit]
                    seen[kk[hit]] = True
                no_row = g_req[~seen]
                if no_row.shape[0]:
                    rc.problems.append(f"{INTERIOR_REGION_MISSING}: {ru}: {int(no_row.shape[0])} required gate(s) have no row (first {no_row[:8].tolist()})")
                    return rc
            layout = InteriorLayout.build(tree, calls, g_req, w_req, prefix=prefix)
        else:
            # the rows define the layout -- minus gates with a boundary position (interface 3): those stay outside the layout so
            # the build refuses them by name (INTERIOR_DUPLICATE_GATE) instead of committing a second value
            gs = [ra.g for ra in batches if ra.rows]
            ws = [ra.w for ra in batches if ra.rows]
            if not gs:
                rc.problems.append(f"{INTERIOR_NO_ROWS}: {ru}: no value rows under {source}")
                return rc
            g_all, w_all = np.concatenate(gs), np.concatenate(ws)
            if is_boundary is not None and g_all.shape[0]:
                keep = ~np.fromiter((is_boundary(int(x)) for x in g_all), dtype=bool, count=g_all.shape[0])
                g_all, w_all = g_all[keep], w_all[keep]
            pairs = np.unique(np.stack([g_all, w_all], axis=1), axis=0)
            if pairs.shape[0] > 1 and np.any(np.diff(pairs[:, 0]) == 0):
                dup = pairs[1:, 0][np.diff(pairs[:, 0]) == 0]
                raise ValueError(f"{INTERIOR_WIDTH_MISMATCH}: gate(s) {dup[:5].tolist()} carry two widths")
            layout = InteriorLayout.build(tree, calls, pairs[:, 0], pairs[:, 1], prefix=prefix)
    except ValueError as e:
        rc.problems.append(f"{ru}: layout refused: {str(e)[:200]}")
        return rc
    try:
        built, rep = build_tree_arrays(layout, batches, is_boundary=is_boundary, strict_unexpected=strict_unexpected, workers=workers)
    except RowError as e:  # streaming path: the parser raises inside the placement loop
        rc.problems.append(f"{INTERIOR_ROW_MALFORMED}: {ru}: {str(e)[:200]}")
        return rc
    del batches
    if streaming:
        rc.seconds_parse = 0.0  # folded into the placement pass (rep.seconds)
    rc.report = rep
    if built is None:
        rc.problems.extend(f"{ru}: {p}" for p in rep.problems)
        return rc
    t_w = time.perf_counter()
    write_tree(built, out_dir)
    rc.seconds_write = time.perf_counter() - t_w
    rc.tree_dir, rc.record = out_dir, built.record()
    return rc


def commit_interiors(values_root: str | os.PathLike[str], out_root: str | os.PathLike[str], *, request_ids: Iterable[str] | None = None,
                     calls: Mapping[str, Mapping[str, Any]] | None = None, is_boundary: Callable[[int], bool] | None = None,
                     strict_unexpected: bool = False, log: Callable[[str], None] | None = None, workers: int | None = None
                     ) -> tuple[dict[str, dict[str, Any]], dict[str, RuCommit]]:
    """The prover's interior phase over a directory of E1 rows: every RU directory under ``values_root`` (restricted to ``request_ids``
    when given -- J) is committed into ``out_root/<req>/<step>/``.  Returns ``(records for add_interior_roots, per-RU outcomes)``; RUs
    whose request is outside ``request_ids`` are skipped and reported as problems (interiors are committed for J only)."""
    root, out = Path(values_root), Path(out_root)
    allowed = set(request_ids) if request_ids is not None else None
    records: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, RuCommit] = {}
    for ru, src in ru_dirs(root).items():
        req = ru.partition("/")[0]
        if allowed is not None and req not in allowed:
            rc = RuCommit(ru, req, ru.partition("/")[2], src)
            rc.problems.append(f"{ru}: request outside J -- interior not committed")
            outcomes[ru] = rc
            if log is not None:
                log(f"interior {ru}: skipped (request outside J)")
            continue
        rc = commit_ru(ru, src, out / ru, calls_spec=(calls or {}).get(ru), is_boundary=is_boundary, strict_unexpected=strict_unexpected, workers=workers)
        outcomes[ru] = rc
        if rc.ok and rc.record is not None:
            records[ru] = rc.record
        if log is not None:
            if rc.ok and rc.report is not None:
                log(f"interior {ru}: {rc.record['leaf_count']} leaves, {rc.record['words']} words, root {rc.record['root'][:12]} "
                    f"(parse {rc.seconds_parse:.2f}s; place+hash {rc.report.rows_per_s_str()}; write {rc.seconds_write:.2f}s)")
            else:
                log(f"interior {ru}: REFUSED {rc.problems[:2]}")
    return records, outcomes


# --- CLI (INTEGRATOR D8: the builder runs on the pod over E1's output; E10 executes it) ----------------------------------------------------


def _cli(argv: Sequence[str] | None = None) -> int:
    """``python -m e2e.interior.tree <stage|commit|verify> ...``

    * ``stage --e1 <E1 out> --values <dir>``: :func:`stage_e1` (row files linked, ``calls.json`` + ``required.json`` per RU);
    * ``commit --values <dir> --out <selective dir> [--request R ...] [--commitments <commitments.json> [--phase0 <commitments_phase0.json>]]``:
      :func:`commit_interiors` -> ``<out>/interior/<req>/<step>/``, ``<out>/interior_commit.json`` (records + per-RU outcome) and, with
      ``--commitments``, the roots merged into it (phase-0 records asserted unchanged); ``--e1 <E1 out>`` stages first;
    * ``verify --out <selective dir> --commitments <commitments.json> --request R ... [--pin <protocol_instance.json>]``:
      :func:`~e2e.interior.layout.validate_step4` without ``required`` (index / commitments / rule self-consistency; rehash).
    Exit 0 = every RU committed / no problem; 1 otherwise."""
    import argparse

    from .layout import layout_pin, validate_step4

    ap = argparse.ArgumentParser(prog="python -m e2e.interior.tree")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--e1", required=True)
    s.add_argument("--values", required=True)
    s.add_argument("--copy", action="store_true", help="copy the row files instead of symlinking")
    c = sub.add_parser("commit")
    c.add_argument("--values", required=True, help="the --interior-values directory (<values>/<req>/<step>/*.jsonl [+ calls.json, required.json])")
    c.add_argument("--out", required=True, help="the selective bundle dir; trees go to <out>/interior/<req>/<step>/")
    c.add_argument("--e1", default=None, help="stage this E1 output into --values first")
    c.add_argument("--request", action="append", default=None, help="restrict to these request ids (J); default: every RU under --values")
    c.add_argument("--calls", default=None, help="JSON {ru: {calls, prefix}} overriding <ru dir>/calls.json")
    c.add_argument("--commitments", default=None, help="commitments.json to merge interior_roots into (rewritten in place)")
    c.add_argument("--phase0", default=None, help="commitments_phase0.json whose records must be unchanged")
    c.add_argument("--strict", action="store_true")
    c.add_argument("--workers", type=int, default=None, help="row-file parser processes (default: cores, or $VERITOR_INTERIOR_WORKERS)")
    v = sub.add_parser("verify")
    v.add_argument("--out", required=True)
    v.add_argument("--commitments", required=True)
    v.add_argument("--request", action="append", required=True)
    v.add_argument("--pin", default=None, help="protocol_instance.json (commitments.interior_layout); default: this rule's pin")
    a = ap.parse_args(argv)
    if a.cmd == "stage":
        staged = stage_e1(a.e1, a.values, link=not a.copy)
        print(json.dumps({"staged": staged}))
        return 0
    if a.cmd == "commit":
        t0 = time.perf_counter()
        staged = stage_e1(a.e1, a.values) if a.e1 else None
        calls = json.loads(Path(a.calls).read_text()) if a.calls else None
        records, outcomes = commit_interiors(a.values, Path(a.out) / "interior", request_ids=a.request, calls=calls, strict_unexpected=a.strict,
                                             log=lambda s: print(s, flush=True), workers=a.workers)
        doc: dict[str, Any] = {"schema": "verity-interior/commit/v1", "label": "MEASURED", "values_dir": str(a.values), "staged": staged, "layout": layout_pin(),
                               "committed": sorted(records), "records": records, "per_ru": {ru: rc.to_json() for ru, rc in outcomes.items()},
                               "seconds": round(time.perf_counter() - t0, 3)}
        if a.commitments:
            cpath = Path(a.commitments)
            commitments = json.loads(cpath.read_text())
            phase0 = json.loads(Path(a.phase0).read_text()) if a.phase0 else None
            merged = add_interior_roots(commitments, records, phase0=phase0)
            write_commitments(cpath.parent, merged) if cpath.name == COMMITMENTS_FILE else cpath.write_text(json.dumps(merged, indent=1) + "\n")
            doc["commitments"] = str(cpath)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "interior_commit.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(json.dumps({"committed": doc["committed"], "refused": [ru for ru, rc in outcomes.items() if not rc.ok], "seconds": doc["seconds"]}))
        return 0 if all(rc.ok for rc in outcomes.values()) else 1
    if a.cmd == "verify":
        commitments = json.loads(Path(a.commitments).read_text())
        pin = layout_pin()
        if a.pin:
            inst = json.loads(Path(a.pin).read_text())
            pin = ((inst.get("commitments") or {}).get("interior_layout")) or pin
        res = validate_step4(a.out, a.request, pin, commitments, rehash=True)
        print(json.dumps(res, indent=1, default=str))
        return 0 if not res["problems"] else 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_cli())
