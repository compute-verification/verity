"""The committed store of a GPU-tree committer on disk, and back as a CPU store that opens and verifies like the committer.

`dump(com, rc, d)` writes what the committer's range openings read (`native_ranges.NativeRangeOpenings`): per packed step its metas, its
retained bytes (the byte ranges the host copy holds) and its tree levels, plus the run commitment (step roots, leaf counts, run root).
`load(d)` returns `(store, run)`: `store` is a `NativeRangeOpenings` over the dumped state with the committer's own leaf, layout and
context rules, so an `OpenedReader(store, run)` recomputes every leaf it returns and folds it to the step root and the run root of record,
exactly as the committer does.  A dumped byte that differs from the committed one does not verify.  Only packed GPU-tree steps with
vllm-v1 committer leaves and no padding steps are dumped; any other store is refused by name.

    store.json                  {schema, chunk, program_digest, geo_digest, run_id, layout_v2, gpu_tree_src, run_root, steps: [...], record?}
    raw/s<step>.bin             the retained bytes of the step, the ranges `steps[i].ranges` ([lo, hi) of the step stream) back to back
    levels/s<step>.bin          the step tree's levels, leaves first, 32 bytes per node (`steps[i].levels` = the widths)

A run of record (`commit.record`; its run root is the record's) also writes its record: ``store.json``'s ``record`` is the public record
(`RunRecord.doc`), and per step

    record/s<step>.bin          the record tree's levels, 64 bytes per node (`steps[i].record.levels` = the widths)
    record/salt_key             the run's salt key (a full dump holds every retained byte already)

A SLIM dump (`dump(..., touched=t)`, `t` recorded by `touched(com)` while a replay's reads were made) writes only the byte ranges and tree
nodes those reads opened: `steps[i].ranges` the touched ranges, and `steps[i].nodes` ([level, index] in file order) the Merkle siblings
their openings carried; of a record, `steps[i].record.nodes` its siblings and `steps[i].record.salts` ([lo, hi) leaves, their salts in
``record/salts-s<step>.bin``), and no salt key.  A read outside them fails by name, as a byte the committer no longer retains does.
"""
from __future__ import annotations

import dataclasses
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

from verity_vllm.commit import hiding
from verity_vllm.commit import record as R
from verity_vllm.commit.committer.native_host import NativeHostCommitter, TensorMeta
from verity_vllm.commit.native_ranges import NativeRangeOpenings

SCHEMA = "verity-vllm/committed-store/v1"
NODE = 32
RNODE = R.NODE
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


@dataclass
class Touched:
    """What openings read of a committer's retained copy: per step the byte ranges of its stream and the (level, index) tree nodes."""
    ranges: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    nodes: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    record_nodes: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    record_salts: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    def merged(self, step: int) -> list[tuple[int, int]]:
        return _merge(self.ranges.get(step, ()))

    def salts_merged(self, step: int) -> list[tuple[int, int]]:
        return _merge(self.record_salts.get(step, ()))

    def nbytes(self) -> int:
        return (sum(hi - lo for s in self.ranges for lo, hi in self.merged(s)) + NODE * sum(len(v) for v in self.nodes.values())
                + RNODE * sum(len(v) for v in self.record_nodes.values())
                + R.SALT_BYTES * sum(hi - lo for s in self.record_salts for lo, hi in self.salts_merged(s)))


def _merge(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


@dataclass
class _TouchedStep(R.StepRecord):
    """A record step whose salt reads are recorded (its levels are `_TouchedLevel`s)."""
    rec: list[tuple[int, int]] = field(default_factory=list, repr=False)

    def salt(self, lo: int, hi: int) -> bytes:
        self.rec.append((int(lo), int(hi)))
        return super().salt(lo, hi)


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
    t, blocks, levels, rec = Touched(), com._gpu_blocks, com._gpu_levels, getattr(com, "record", None)
    com._gpu_blocks = {s: (None if b is None else _TouchedBlock(b, t.ranges.setdefault(s, [])), u) for s, (b, u) in blocks.items()}
    com._gpu_levels = {s: [_TouchedLevel(lv, d, t.nodes.setdefault(s, set())) for d, lv in enumerate(lvs)] for s, lvs in levels.items()}
    if rec is not None:
        com.record = dataclasses.replace(rec, steps=[_TouchedStep(
            **{**{f.name: getattr(st, f.name) for f in dataclasses.fields(R.StepRecord)},
               "levels": [_TouchedLevel(lv, d, t.record_nodes.setdefault(st.step, set())) for d, lv in enumerate(st.levels)]},
            rec=t.record_salts.setdefault(st.step, [])) for st in rec.steps])
    try:
        yield t
    finally:
        com._gpu_blocks, com._gpu_levels = blocks, levels
        if rec is not None:
            com.record = rec


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
    if getattr(com, "_padding", None):
        return "padding steps: not dumped"
    if getattr(com, "_streams", None):
        return "host-path step streams: not dumped (GPU-tree committers only)"
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
    rec = _dump_record_step(com.record.steps[s], d, t, digests) if getattr(com, "record", None) is not None else None
    return ({"step": s, "root": sc.root.hex(), "n_leaves": int(sc.n_leaves), "n_bytes": int(sc.n_bytes), "n_tensors": int(sc.n_tensors),
             "used": int(used), "ranges": [[int(lo), int(hi)] for lo, hi, _ in rs], "levels": [_width(lv) for lv in lvs],
             **({} if nodes is None else {"nodes": [[d_, i] for d_, i in nodes]}), **({} if rec is None else {"record": rec}),
             "metas": [[m.ordinal, m.name, m.dtype, list(m.shape), m.nbytes, m.chunk, m.first_leaf, m.n_leaves, m.stream_off]
                       for m in com._layouts[s]]}, digests)


def _dump_record_step(st: R.StepRecord, d: Path, t: Touched | None, digests: dict[str, str]) -> dict[str, Any]:
    s, lvs = st.step, list(st.levels)
    if t is None:
        digests[f"record/s{s}.bin"] = write_file(d / "record" / f"s{s}.bin", (
            lv.buf[lv.off:lv.off + lv.width * lv.node] if isinstance(lv, _Level) else _as_bytes(lv) for lv in lvs))
        return {"levels": [_width(lv) for lv in lvs]}
    nodes, want = sorted(t.record_nodes.get(s, ())), t.salts_merged(s)
    digests[f"record/s{s}.bin"] = write_file(d / "record" / f"s{s}.bin", (memoryview(bytes(lvs[d_][i])) for d_, i in nodes))
    digests[f"record/salts-s{s}.bin"] = write_file(d / "record" / f"salts-s{s}.bin", (memoryview(st.salt(lo, hi)) for lo, hi in want))
    return {"levels": [_width(lv) for lv in lvs], "nodes": [[d_, i] for d_, i in nodes], "salts": [[lo, hi] for lo, hi in want]}


def _width(lv: Any) -> int:
    return int(lv.shape[0]) if hasattr(lv, "shape") else len(lv)


def nbytes(com: Any) -> int:
    """The bytes a full dump of `com` writes: every retained range and every tree node (the record's too)."""
    rec = getattr(com, "record", None)
    return sum(hi - lo for blk, _u in com._gpu_blocks.values() for lo, hi, _ in _ranges(blk)) + NODE * sum(
        _width(lv) for lvs in com._gpu_levels.values() for lv in lvs) + (
        0 if rec is None else RNODE * sum(_width(lv) for st in rec.steps for lv in st.levels))


def dump(com: Any, rc: Any, d: str | os.PathLike, *, threads: int = 8, digests: dict[str, str] | None = None,
         touched: Touched | None = None) -> dict[str, Any]:
    """Writes the committed store of `com` (run commitment `rc`) under `d`, its steps ``threads`` at a time; returns the `store.json`
    document.  ``digests``, when given, receives the sha256 of every step file written (by its path under `d`).  ``touched``: a slim dump
    of only what those reads opened."""
    why = _refusal(com)
    if why:
        raise ValueError(f"committed store not dumpable: {why}")
    rec = getattr(com, "record", None)
    if R.of_record(rc) and (rec is None or rec.root != rc.run_root):
        raise ValueError("committed store not dumpable: the run's root of record is not the committer's record")
    d = Path(d)
    for sub in ("raw", "levels") + (("record",) if rec is not None else ()):
        (d / sub).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max(1, threads)) as pool:
        done = list(pool.map(lambda sc: _dump_step(com, sc, d, touched), rc.steps))
    steps = [e for e, _ in done]
    if digests is not None:
        for _, dg in done:
            digests.update(dg)
    if rec is not None and touched is None:
        key = rec.steps[0].key if rec.steps else None
        if key is None:
            raise ValueError("committed store not dumpable: a full dump of a record needs its salt key")
        (d / "record" / "salt_key").write_bytes(key)
    doc = {"schema": SCHEMA, "chunk": int(com.chunk), "program_digest": com.program_digest.hex(), "geo_digest": com._geo_digest.hex(),
           "run_id": com.run_id, "layout_v2": bool(getattr(com, "layout_v2", False)), "gpu_tree_src": int(com.GPU_TREE_SRC),
           "run_root": rc.run_root.hex(), "steps": steps, **({} if rec is None else {"record": rec.doc()})}
    (d / "store.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
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
    """One tree level of a dumped step: `len()` nodes, `lv[i]` the i-th node (32 bytes; a record's 64)."""

    def __init__(self, buf: memoryview, off: int, width: int, node: int = NODE) -> None:
        self.buf, self.off, self.width, self.node = buf, off, width, node

    def __len__(self) -> int:
        return self.width

    def __getitem__(self, i: int) -> bytes:
        if not 0 <= i < self.width:
            raise IndexError(i)
        return bytes(self.buf[self.off + i * self.node:self.off + (i + 1) * self.node])


def _levels(buf: memoryview, widths: list[int], nodes: list[list[int]] | None, where: str, node: int) -> list[Any]:
    """A dumped step's levels over its file: every node (``nodes`` None) or only the slim dump's ``[level, index]`` nodes, in file order."""
    if nodes is not None:
        at: list[dict[int, int]] = [{} for _ in widths]
        for k, (dd, i) in enumerate(nodes):
            at[int(dd)][int(i)] = k * node
        return [_SparseLevel(buf, int(w), at[dd], f"{where} level {dd}", node) for dd, w in enumerate(widths)]
    out, off = [], 0
    for w in widths:
        out.append(_Level(buf, off, int(w), node))
        off += int(w) * node
    return out


class _Salts:
    """A slim dumped record step's salts: leaves ``[lo, hi)`` the dump kept, any other refused by name."""

    def __init__(self, buf: memoryview, ranges: list[list[int]], where: str) -> None:
        self.buf, self.where, self.ranges, off = buf, where, [], 0
        for lo, hi in ranges:
            self.ranges.append((int(lo), int(hi), off))
            off += (int(hi) - int(lo)) * R.SALT_BYTES

    def __call__(self, lo: int, hi: int) -> bytes:
        for rlo, rhi, off in self.ranges:
            if rlo <= lo and hi <= rhi:
                return bytes(self.buf[off + (lo - rlo) * R.SALT_BYTES:off + (hi - rlo) * R.SALT_BYTES])
        raise KeyError(f"the salts of leaves [{lo}, {hi}) of {self.where} are not in the slim dumped store")


@dataclass
class DumpedStep:
    step: int
    root: bytes
    n_leaves: int
    n_bytes: int
    n_tensors: int


@dataclass
class DumpedRun:
    run_root: bytes
    steps: list[DumpedStep]
    record: R.RunRecord | None = None


class DumpedStore(NativeRangeOpenings, hiding.LeafSchemeRule):
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
            self._gpu_levels[s] = _levels(self._map(d / "levels" / f"s{s}.bin"), st["levels"], st.get("nodes"), f"step {s}", NODE)
        self.record = self._load_record(doc, d)

    def _load_record(self, doc: dict[str, Any], d: Path) -> R.RunRecord | None:
        if "record" not in doc:
            return None
        by_step = {int(st["step"]): st["record"] for st in doc["steps"]}
        key_file = d / "record" / "salt_key"
        key = key_file.read_bytes() if key_file.exists() else None
        rec = R.from_doc(doc["record"], levels=lambda s: _levels(self._map(d / "record" / f"s{s}.bin"), by_step[s]["levels"],
                                                                  by_step[s].get("nodes"), f"record step {s}", RNODE), key=key)
        for st in rec.steps:
            if "salts" in by_step[st.step]:
                st.salt_of = _Salts(self._map(d / "record" / f"salts-s{st.step}.bin"), by_step[st.step]["salts"], f"record step {st.step}")
        return rec

    def _map(self, p: Path) -> memoryview:
        if p.stat().st_size == 0:
            return memoryview(b"")
        with open(p, "rb") as f:
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._maps.append(m)
        return memoryview(m)


def load(d: str | os.PathLike) -> tuple[DumpedStore, DumpedRun]:
    d = Path(d)
    doc = json.loads((d / "store.json").read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{d / 'store.json'}: schema {doc.get('schema')!r} is not {SCHEMA}")
    run = DumpedRun(bytes.fromhex(doc["run_root"]), [DumpedStep(int(s["step"]), bytes.fromhex(s["root"]), int(s["n_leaves"]), int(s["n_bytes"]),
                                                                int(s["n_tensors"])) for s in doc["steps"]])
    if [s.step for s in run.steps] != list(range(len(run.steps))):
        raise ValueError(f"{d / 'store.json'}: steps are not numbered densely from 0")
    store = DumpedStore(doc, d)
    if store.record is not None:
        if store.record.root != run.run_root or store.record.leaves_per_step() != [s.n_leaves for s in run.steps]:
            raise ValueError(f"{d / 'store.json'}: the record is not the run's root of record")
        run.record = store.record
    elif len(run.run_root) == RNODE:
        raise ValueError(f"{d / 'store.json'}: a {RNODE}-byte run root with no record")
    return store, run
