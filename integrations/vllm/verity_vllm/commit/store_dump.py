"""The committed store of a GPU-tree committer on disk, and back as a CPU store that opens and verifies like the committer.

`dump(com, rc, d)` writes what the committer's range openings read (`native_ranges.NativeRangeOpenings`): per packed step its metas, its
retained bytes (the byte ranges the host copy holds) and its tree levels, plus the run commitment's roots (`RunCommitment.roots()`: step
roots, leaf counts, run root).  `load(d)` returns `(store, run)`: `run` is those roots read back (a `RunRoots`), and `store` a
`NativeRangeOpenings` over the dumped state with the committer's own leaf, layout and context rules, so an `OpenedReader(store, run)`
recomputes every leaf it returns and folds it to the step root and the run root of record, exactly as the committer does.  A dumped byte
that differs from the committed one does not verify.  Only packed GPU-tree steps with vllm-v1 leaves or under rows-sha512 (`step_rows`),
and no padding steps, are dumped; any other store is refused by name.

    store.json                  {schema, chunk, program_digest, geo_digest, run_id, layout_v2, gpu_tree_src, run_root, steps: [...]}
                                (rows-sha512: commit_scheme too)
    raw/s<step>.bin             the retained bytes of the step, the ranges `steps[i].ranges` ([lo, hi) of the step stream) back to back
    levels/s<step>.bin          the step tree's levels, leaves first, 32 bytes per node (rows-sha512: 64) (`steps[i].levels` = the widths)
    keys.bin                    rows-sha512, a full dump: each step's 64-byte salt key, in step order

A SLIM dump (`dump(..., touched=t)`, `t` recorded by `touched(com)` while a replay's reads were made) writes only the byte ranges and tree
nodes those reads opened: `steps[i].ranges` the touched ranges, and `steps[i].nodes` ([level, index] in file order) the Merkle siblings
their openings carried.  Under rows-sha512 it writes the salts of the leaves those ranges cover (`salts/s<step>.bin`, the leaf ranges
`steps[i].salted` back to back) and never a key, so no other leaf's salt can be derived from it.  A read outside them fails by name, as a
byte the committer no longer retains does.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.commit import leaf_rule, scheme, step_rows
from verity_vllm.commit.committer.native_host import NativeHostCommitter, TensorMeta
from verity_vllm.commit.native_ranges import NativeRangeOpenings
from verity_vllm.commit.run_roots import RunRoots, StepRoot

SCHEMA = "verity-vllm/committed-store/v1"
NODE = 32                                                       # a vllm-v1 tree node; rows-sha512's is step_rows.ROOT_BYTES (`_node`)
SCHEMES = (scheme.NAME, step_rows.NAME)
#: bytes written between flushes: the dirty pages of a dump are charged to the writer's memory cgroup, beside a pinned store
FLUSH = 1 << 30


def _ranges(blk: Any) -> list[tuple[int, int, Any]]:
    """The byte ranges of a step stream a retained block holds: (lo, hi, uint8 view of hi - lo bytes)."""
    if blk is None:
        return []
    if isinstance(blk, _Block):                                 # a dumped store's own block: (lo, hi, offset in its file)
        return [(lo, hi, blk.buf[off:off + hi - lo]) for lo, hi, off in blk.ranges]
    if hasattr(blk, "wins"):                                    # _HostWindows: window k covers [k W, k W + n)
        return [(k * blk.W, k * blk.W + n, hv[:n]) for k, (hv, n) in sorted(blk.wins.items())]
    if hasattr(blk, "ranges"):                                  # _HostRanges
        return [(lo, hi, hv) for lo, hi, hv in blk.ranges]
    return [(0, int(blk.numel()), blk)]


def _node(com: Any) -> int:
    return step_rows.ROOT_BYTES if getattr(com, "rows", False) else NODE


@dataclass
class Touched:
    """What openings read of a committer's retained copy: per step the byte ranges of its stream and the (level, index) tree nodes.
    `node`, `salt` and `chunk` size them: a tree node's bytes, a leaf's salt bytes (rows-sha512) and the bytes of a leaf."""
    ranges: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    nodes: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    node: int = NODE
    salt: int = 0
    chunk: int = 1

    def merged(self, step: int) -> list[tuple[int, int]]:
        out: list[list[int]] = []
        for lo, hi in sorted(self.ranges.get(step, ())):
            if out and lo <= out[-1][1]:
                out[-1][1] = max(out[-1][1], hi)
            else:
                out.append([lo, hi])
        return [(lo, hi) for lo, hi in out]

    def leaves(self, step: int) -> list[tuple[int, int]]:
        """The leaves [a, b) that hold the touched ranges of `step`, merged."""
        out: list[list[int]] = []
        for lo, hi in self.merged(step):
            a, b = lo // self.chunk, -(-hi // self.chunk)
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return [(a, b) for a, b in out]

    def nbytes(self) -> int:
        salts = self.salt * sum(b - a for s in self.ranges for a, b in self.leaves(s)) if self.salt else 0
        return sum(hi - lo for s in self.ranges for lo, hi in self.merged(s)) + self.node * sum(len(v) for v in self.nodes.values()) + salts


class _TouchedBlock:
    def __init__(self, blk: Any, rec: list[tuple[int, int]]) -> None:
        self._blk, self._rec = blk, rec

    def __getattr__(self, k: str) -> Any:
        return getattr(self._blk, k)

    def __getitem__(self, sl: slice) -> Any:
        self._rec.append((int(sl.start or 0), int(sl.stop)))
        return self._blk[sl]


class _TouchedLevel:
    def __init__(self, lv: Any, depth: int, rec: set[tuple[int, int]]) -> None:
        self._lv, self._d, self._rec = lv, depth, rec

    def __getattr__(self, k: str) -> Any:
        return getattr(self._lv, k)

    def __len__(self) -> int:
        return len(self._lv)

    def __getitem__(self, i: int) -> Any:
        self._rec.add((self._d, int(i)))
        return self._lv[i]


@contextmanager
def touched(com: Any) -> Iterator[Touched]:
    """Records, while open, every retained byte range and tree node `com`'s range openings read."""
    rows = getattr(com, "rows", False)
    t = Touched(node=_node(com), salt=step_rows.SALT_BYTES if rows else 0, chunk=int(com.chunk))
    blocks, levels = com._gpu_blocks, com._gpu_levels
    com._gpu_blocks = {s: (None if b is None else _TouchedBlock(b, t.ranges.setdefault(s, [])), u) for s, (b, u) in blocks.items()}
    com._gpu_levels = {s: [_TouchedLevel(lv, d, t.nodes.setdefault(s, set())) for d, lv in enumerate(lvs)] for s, lvs in levels.items()}
    try:
        yield t
    finally:
        com._gpu_blocks, com._gpu_levels = blocks, levels


def touch(com: Any, reads: Iterable[tuple[int, str, int, int]]) -> None:
    """Opens, without verifying, the leaves of every read (step, member name, byte lo, byte hi) as `OpenedReader` would: under `touched`
    the ranges and tree nodes a verified read of the same bytes opens are recorded, and no leaf is hashed."""
    from verity_vllm.commit.opened import meta_by_name

    for step, name, lo, hi in reads:
        m = meta_by_name(com._layouts.get(step) or [], name)
        lo, hi = max(0, int(lo)), min(int(hi), int(m.nbytes)) if m is not None else 0
        if m is not None and hi > lo:
            com.open_range(step, *com.leaf_span(step, m, lo, hi)[:2])


def _slim_ranges(blk: Any, want: list[tuple[int, int]]) -> list[tuple[int, int, Any]]:
    """The parts of the wanted byte ranges the retained block holds, as `_ranges` gives them."""
    out = []
    for rlo, rhi, hv in _ranges(blk):
        for lo, hi in want:
            a, b = max(lo, rlo), min(hi, rhi)
            if a < b:
                out.append((a, b, _as_bytes(hv)[a - rlo:b - rlo]))
    return sorted(out, key=lambda x: x[0])


def _nodes(lvs: list[Any], nodes: list[tuple[int, int]]) -> Iterator[memoryview]:
    for d, i in nodes:
        yield _as_bytes(lvs[d][i].contiguous() if hasattr(lvs[d][i], "contiguous") else lvs[d][i])


def _as_bytes(t: Any) -> memoryview:
    if getattr(t, "is_cuda", False):
        t = t.cpu()
    a = t.numpy() if hasattr(t, "numpy") else t
    return memoryview(a.reshape(-1) if hasattr(a, "reshape") else a).cast("B")


def _refusal(com: Any) -> str | None:
    if getattr(com, "hiding", None) is not None:
        return f"{com.leaf_scheme} leaves (salted): not dumped"
    if getattr(com, "commit_scheme", scheme.NAME) not in SCHEMES:
        return f"{com.commit_scheme} steps: not dumped"
    if getattr(com, "_padding", None):
        return "padding steps: not dumped"
    if getattr(com, "_streams", None):
        return "host-path step streams: not dumped (GPU-tree committers only)"
    if not getattr(com, "retain_levels", True):
        return "no tree levels kept (retain_levels=False): not dumped"
    missing = sorted(s for s in com._layouts if s not in com._gpu_blocks)
    if missing:
        return f"steps {missing[:8]} are not packed GPU-tree blocks: not dumped"
    return None


def write_file(p: Path, chunks: Iterable[memoryview]) -> str:
    """Writes ``chunks`` to ``p`` back to back, flushed and dropped from the page cache every `FLUSH` bytes; the sha256 of what was
    written."""
    h = hashlib.sha256()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        dirty = 0
        for mv in chunks:
            h.update(mv)
            dirty += len(mv)
            while mv:
                mv = mv[os.write(fd, mv):]
            if dirty >= FLUSH:
                os.fdatasync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                dirty = 0
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return h.hexdigest()


def _dump_step(com: Any, sc: Any, d: Path, t: Touched | None) -> tuple[dict[str, Any], dict[str, str]]:
    s = int(sc.step)
    blk, used = com._gpu_blocks[s]
    rs = _ranges(blk) if t is None else _slim_ranges(blk, t.merged(s))
    lvs = list(com._gpu_levels[s])
    nodes = None if t is None else sorted(t.nodes.get(s, ()))
    lv_bytes = (_as_bytes(lv.contiguous()) for lv in lvs) if nodes is None else _nodes(lvs, nodes)
    digests = {f"raw/s{s}.bin": write_file(d / "raw" / f"s{s}.bin", (_as_bytes(hv)[:hi - lo] for lo, hi, hv in rs)),
               f"levels/s{s}.bin": write_file(d / "levels" / f"s{s}.bin", lv_bytes)}
    salted = None
    if t is not None and getattr(com, "rows", False):
        salted = [(a, min(b, int(sc.n_leaves))) for a, b in t.leaves(s) if a < int(sc.n_leaves)]
        digests[f"salts/s{s}.bin"] = write_file(d / "salts" / f"s{s}.bin", (memoryview(com._range_salts(s, a, b)) for a, b in salted))
    return ({"step": s, "root": sc.root.hex(), "n_leaves": int(sc.n_leaves), "n_bytes": int(sc.n_bytes), "n_tensors": int(sc.n_tensors),
             "used": int(used), "ranges": [[int(lo), int(hi)] for lo, hi, _ in rs], "levels": [_width(lv) for lv in lvs],
             **({} if nodes is None else {"nodes": [[d_, i] for d_, i in nodes]}),
             **({} if salted is None else {"salted": [[a, b] for a, b in salted]})}, digests)


def metas_of(metas: Iterable[Any]) -> list[list]:
    """A step's metas as store.json lists them."""
    return [[m.ordinal, m.name, m.dtype, list(m.shape), m.nbytes, m.chunk, m.first_leaf, m.n_leaves, m.stream_off] for m in metas]


def _width(lv: Any) -> int:
    return int(lv.shape[0]) if hasattr(lv, "shape") else len(lv)


def nbytes(com: Any) -> int:
    """The bytes a full dump of `com` writes: every retained range and every tree node."""
    return sum(hi - lo for blk, _u in com._gpu_blocks.values() for lo, hi, _ in _ranges(blk)) + _node(com) * sum(
        _width(lv) for lvs in com._gpu_levels.values() for lv in lvs)


def not_retained(com: Any) -> dict[str, Any] | None:
    """What a committer that excludes value families from retention (``retain_exclude``, `VERITY_RETAIN_EXCLUDE`) committed and
    doesn't hold: by family the bytes of the chunk-aligned interiors (`host_retention._excluded_chunk_ranges`) its retained ranges
    leave out, beside the bytes its steps committed and those it retains; None when it excludes nothing.  A committed byte it
    doesn't hold outside those interiors is refused by name: its opening would fail with nothing to say why."""
    fams = set(getattr(com, "retain_exclude", None) or ())
    if not fams:
        return None
    return not_retained_of(fams, int(com.chunk), held_of(com), com._layouts)


def held_of(com: Any) -> dict[int, tuple[list[tuple[int, int]], int]]:
    """Each retained block's byte ranges of its step stream and its used bytes, by step."""
    return {s: ([(int(lo), int(hi)) for lo, hi, _ in _ranges(blk)], int(used)) for s, (blk, used) in com._gpu_blocks.items()}


def not_retained_of(fams: set[str], C: int, held: dict[int, tuple[list[tuple[int, int]], int]],
                    layouts: dict[int, Any]) -> dict[str, Any]:
    """`not_retained` over the blocks' retained ranges (`held_of`) and the steps' metas."""
    from verity_vllm.commit.committer.host_retention import _excluded_chunk_ranges, outside

    by, committed, retained, stray = dict.fromkeys(sorted(fams), 0), 0, 0, []
    for s, (ranges, used) in sorted(held.items()):
        held_s = sorted((lo, hi, "") for lo, hi in ranges)
        excl = _excluded_chunk_ranges(layouts[s], fams, C)
        committed, retained = committed + int(used), retained + sum(hi - lo for lo, hi, _ in held_s)
        for lo, hi in outside(held_s, 0, -(-int(used) // C) * C):
            stray += [f"step {s} [{a}, {b})" for a, b in outside(excl, lo, hi)]
            for elo, ehi, fam in excl:
                by[fam] += max(0, min(hi, ehi) - max(lo, elo))
    if stray:
        raise ValueError(f"{len(stray)} committed byte range(s) not retained outside the excluded families' interiors {sorted(fams)}: "
                         f"{stray[:4]}")
    return {"families": sorted(fams), "bytes": sum(by.values()), "by_family": by, "committed_bytes": committed,
            "retained_bytes": retained}


def dump(com: Any, rc: Any, d: str | os.PathLike, *, threads: int = 8, digests: dict[str, str] | None = None,
         touched: Touched | None = None) -> dict[str, Any]:
    """Writes the committed store of `com` (run commitment `rc`) under `d`, its steps ``threads`` at a time; returns the `store.json`
    document.  ``digests``, when given, receives the sha256 of every step file written (by its path under `d`).  ``touched``: a slim dump
    of only what those reads opened."""
    return write_doc(d, dump_files(com, rc, d, threads=threads, digests=digests, touched=touched), com._layouts)


def write_doc(d: str | os.PathLike, doc: dict[str, Any], layouts: dict[int, Any]) -> dict[str, Any]:
    """Writes `store.json` under `d` from `dump_files`'s document, each step's metas (`layouts`) added; returns it."""
    for e in doc["steps"]:
        e["metas"] = metas_of(layouts[e["step"]])
    (Path(d) / "store.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return doc


def dump_files(com: Any, rc: Any, d: str | os.PathLike, *, threads: int = 8, digests: dict[str, str] | None = None,
               touched: Touched | None = None) -> dict[str, Any]:
    """`dump`'s step files under `d`, and its `store.json` document without the steps' metas (`write_doc` adds them and writes it)."""
    why = _refusal(com)
    rows = getattr(com, "rows", False)
    keys = getattr(com, "row_keys", None) or {}
    if not why and rows and touched is None and any(int(sc.step) not in keys for sc in rc.steps):
        why = f"{step_rows.NAME} steps without their salt keys (a slim dumped store): no full dump"
    if why:
        raise ValueError(f"committed store not dumpable: {why}")
    d = Path(d)
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "levels").mkdir(parents=True, exist_ok=True)
    if rows and touched is not None:
        (d / "salts").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max(1, threads)) as pool:
        done = list(pool.map(lambda sc: _dump_step(com, sc, d, touched), rc.steps))
    steps = [e for e, _ in done]
    if rows and touched is None:
        done.append(({}, {"keys.bin": write_file(d / "keys.bin", (memoryview(keys[int(sc.step)]) for sc in rc.steps))}))
    if digests is not None:
        for _, dg in done:
            digests.update(dg)
    doc = {"schema": SCHEMA, "chunk": int(com.chunk), "program_digest": com.program_digest.hex(), "geo_digest": com._geo_digest.hex(),
           "run_id": com.run_id, "layout_v2": bool(getattr(com, "layout_v2", False)), "gpu_tree_src": int(com.GPU_TREE_SRC),
           **({"commit_scheme": step_rows.NAME} if rows else {}), "run_root": rc.run_root.hex(), "steps": steps}
    return doc


class _Seg:
    """A slice of retained bytes, in the shape `open_range` reads (`.numpy().tobytes()`)."""

    def __init__(self, b: bytes) -> None:
        self._b = b

    def numpy(self) -> _Seg:
        return self

    def tobytes(self) -> bytes:
        return self._b


class _Block:
    """A dumped step's retained bytes: `holds(lo, hi)` and `blk[lo:hi]` over the step stream, like the committer's host retainers."""

    is_cuda = False

    def __init__(self, buf: memoryview, ranges: list[list[int]], used: int, chunk: int) -> None:
        self.buf, self.used = buf, int(used)
        self.n = -(-self.used // chunk) * chunk
        self.ranges, off = [], 0
        for lo, hi in ranges:
            self.ranges.append((int(lo), int(hi), off))
            off += int(hi) - int(lo)

    def numel(self) -> int:
        return self.n

    def _cover(self, lo: int, hi: int) -> list[tuple[int, int, int]] | None:
        out, pos = [], lo
        for rlo, rhi, off in self.ranges:
            if rhi <= pos:
                continue
            if rlo > pos:
                return None
            out.append((rlo, rhi, off))
            pos = rhi
            if pos >= hi:
                return out
        return None

    def holds(self, lo: int, hi: int) -> bool:
        lo, hi = int(lo), int(min(hi, self.n))
        return hi <= lo or self._cover(lo, hi) is not None

    def __getitem__(self, sl: slice) -> _Seg:
        lo, hi = int(sl.start or 0), int(min(sl.stop if sl.stop is not None else self.n, self.n))
        if hi <= lo:
            return _Seg(b"")
        cov = self._cover(lo, hi)
        if cov is None:
            raise KeyError(f"bytes [{lo}, {hi}) of the step stream are not in the dumped store")
        return _Seg(b"".join(bytes(self.buf[off + max(lo, rlo) - rlo:off + min(hi, rhi) - rlo]) for rlo, rhi, off in cov))


class _SparseLevel:
    """One tree level of a slim dumped step: `len()` its full width, `lv[i]` a node the dump kept; any other node is refused by name."""

    def __init__(self, buf: memoryview, width: int, at: dict[int, int], where: str, node: int = NODE) -> None:
        self.buf, self.width, self.at, self.where, self.node = buf, width, at, where, node

    def __len__(self) -> int:
        return self.width

    def __getitem__(self, i: int) -> bytes:
        off = self.at.get(int(i))
        if off is None:
            raise KeyError(f"tree node {i} of {self.where} is not in the slim dumped store")
        return bytes(self.buf[off:off + self.node])


class _Level:
    """One tree level of a dumped step: `len()` nodes, `lv[i]` the i-th node (32 bytes, rows-sha512's 64)."""

    def __init__(self, buf: memoryview, off: int, width: int, node: int = NODE) -> None:
        self.buf, self.off, self.width, self.node = buf, off, width, node

    def __len__(self) -> int:
        return self.width

    def __getitem__(self, i: int) -> bytes:
        if not 0 <= i < self.width:
            raise IndexError(i)
        return bytes(self.buf[self.off + i * self.node:self.off + (i + 1) * self.node])


class DumpedStore(NativeRangeOpenings, leaf_rule.LeafSchemeRule):
    """The committer's openings over a dumped store (`load`)."""

    name = "dumped-store"
    hiding = None
    retain = "host"
    layout_digest = staticmethod(NativeHostCommitter.layout_digest)
    _block_holds = staticmethod(NativeHostCommitter._block_holds)
    _ctx_digest = NativeHostCommitter._ctx_digest
    _gpu_layout = NativeHostCommitter._gpu_layout
    _padding_leaf = NativeHostCommitter._padding_leaf

    def __init__(self, doc: dict[str, Any], d: Path) -> None:
        self.chunk = int(doc["chunk"])
        self.program_digest = bytes.fromhex(doc["program_digest"])
        self._geo_digest = bytes.fromhex(doc["geo_digest"])
        self.run_id = str(doc["run_id"])
        self.layout_v2 = bool(doc["layout_v2"])
        self.GPU_TREE_SRC = int(doc["gpu_tree_src"])
        self.commit_scheme = str(doc.get("commit_scheme", scheme.NAME))
        if self.commit_scheme not in SCHEMES:
            raise ValueError(f"{d / 'store.json'}: commit scheme {self.commit_scheme!r} is not one of {SCHEMES}")
        self.rows = self.commit_scheme == step_rows.NAME
        node = _node(self)
        self.row_keys: dict[int, bytes] = {}
        self._salted: dict[int, tuple[list[tuple[int, int, int]], memoryview]] = {}
        self._salts: dict[int, bytes] = {}
        self._padding: dict[int, Any] = {}
        self._streams: dict[int, Any] = {}
        self._levels: dict[int, Any] = {}
        self._layouts: dict[int, list[TensorMeta]] = {}
        self._gpu_blocks: dict[int, tuple[_Block, int]] = {}
        self._gpu_levels: dict[int, list[_Level]] = {}
        self._maps: list[mmap.mmap] = []
        for st in doc["steps"]:
            s = int(st["step"])
            self._layouts[s] = [TensorMeta(o, n, dt, tuple(sh), nb, c, fl, nl, stream_off=so) for o, n, dt, sh, nb, c, fl, nl, so in st["metas"]]
            self._gpu_blocks[s] = (_Block(self._map(d / "raw" / f"s{s}.bin"), st["ranges"], st["used"], self.chunk), int(st["used"]))
            lv_buf, off, lvs = self._map(d / "levels" / f"s{s}.bin"), 0, []
            if "salted" in st:
                at_leaf, k = [], 0
                for a, b in st["salted"]:
                    at_leaf.append((int(a), int(b), k))
                    k += (int(b) - int(a)) * step_rows.SALT_BYTES
                self._salted[s] = (at_leaf, self._map(d / "salts" / f"s{s}.bin"))
            if "nodes" in st:
                at: list[dict[int, int]] = [{} for _ in st["levels"]]
                for k, (dd, i) in enumerate(st["nodes"]):
                    at[int(dd)][int(i)] = k * node
                self._gpu_levels[s] = [_SparseLevel(lv_buf, int(w), at[dd], f"step {s} level {dd}", node) for dd, w in enumerate(st["levels"])]
                continue
            for w in st["levels"]:
                lvs.append(_Level(lv_buf, off, int(w), node))
                off += int(w) * node
            self._gpu_levels[s] = lvs
        if self.rows and (d / "keys.bin").exists():
            ks = bytes(self._map(d / "keys.bin"))
            if len(ks) != step_rows.KEY_BYTES * len(doc["steps"]):
                raise ValueError(f"{d / 'keys.bin'}: {len(ks)} bytes for {len(doc['steps'])} steps' {step_rows.KEY_BYTES}-byte salt keys")
            self.row_keys = {int(st["step"]): ks[i * step_rows.KEY_BYTES:(i + 1) * step_rows.KEY_BYTES] for i, st in enumerate(doc["steps"])}

    def _gpu_salt(self, step: int, index: int) -> bytes:
        """rows-sha512: leaf `index`'s salt, from the step's key (a full dump) or the slim dump's salts; any other leaf's is refused by name."""
        if not self.rows:
            return b""
        if step in self.row_keys:
            return step_rows.salt(self.row_keys[step], index)
        y = step_rows.SALT_BYTES
        at, buf = self._salted.get(step, ([], memoryview(b"")))
        for a, b, off in at:
            if a <= index < b:
                return bytes(buf[off + (index - a) * y:off + (index - a + 1) * y])
        raise KeyError(f"the salt of leaf {index} of step {step} is not in the slim dumped store")

    def _map(self, p: Path) -> memoryview:
        if p.stat().st_size == 0:
            return memoryview(b"")
        with open(p, "rb") as f:
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._maps.append(m)
        return memoryview(m)


def load(d: str | os.PathLike) -> tuple[DumpedStore, RunRoots]:
    d = Path(d)
    doc = json.loads((d / "store.json").read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{d / 'store.json'}: schema {doc.get('schema')!r} is not {SCHEMA}")
    run = RunRoots(bytes.fromhex(doc["run_root"]), [StepRoot(int(s["step"]), bytes.fromhex(s["root"]), int(s["n_leaves"]), int(s["n_bytes"]),
                                                             int(s["n_tensors"])) for s in doc["steps"]])
    if [s.step for s in run.steps] != list(range(len(run.steps))):
        raise ValueError(f"{d / 'store.json'}: steps are not numbered densely from 0")
    return DumpedStore(doc, d), run
