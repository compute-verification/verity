"""The Build's instance sequence, `instances.json.gz`: one row per root node of the Program (`instance_rows`), the comparator's unit.

Its readers parse its text, and no digest or published output reads its bytes (`research_outputs.NEVER`), so the text is the
contract: `json.dumps({**head, "rows": list(instance_rows(program))})`, whatever writes it.  Two writers give that text:

* `write_instances_file(path, head, rows)` encodes row dicts with `json.dumps` (the reference).
* `write_program_instances(path, head, program)` writes a Program's rows without building them (`RowTexts`).  A causal attention's
  key and value operands are cuts of one `PartLog` per head, each call's one row longer than the call before's, so their stored items
  are the items of the call before plus O(1) more (the greedy progressions' prefix property, `compact._Acc`).  `RowTexts` keeps each
  log's items as text and writes each cut as that text plus its own last items: O(1) Python work per operand where encoding the rows
  is O(rows) per operand and O(rows^2) per head (GLM-5.3 TP8 at 2048 input tokens: 1.3e9 items, 60 GB of text, 4 h single-threaded).

The text is compressed by `ParallelGzip`: one gzip member whose deflate blocks are compressed on threads.

`read_instances(path)` reads the whole document back, as `json.load` does, with equal argument items sharing one list (the rows
are a few times the file instead of 56 times).  `query.program_view.iter_instance_rows` streams the rows one at a time.
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

#: `instances.json.gz`'s gzip level.  A long context's text is mostly the attention Calls' key and value runs (B1 4096/512: 8.6 GB),
#: which level 6 compresses in under a third of level 9's CPU time, 2% larger, and which inflates as fast at either level.
INSTANCES_GZ_LEVEL = 6
#: text per deflate block of `ParallelGzip`: each block's last 32 KiB is the next one's dictionary, so the blocks compress as one stream
GZ_BLOCK = 1 << 24
_WINDOW = 1 << 15
_FINAL_EMPTY_BLOCK = b"\x03\x00"   # BFINAL=1, fixed Huffman, end of block


def instance_rows(program) -> Iterator[dict]:
    """The flat body as (spec, argument items) rows — the comparator's unit (compare.py), one at a time.  An argument is stored in the
    progressions form (instances v2, `compact.file_items_from_refs`, the fold writer's): O(rows) for a scattered Strided where `runs()`
    is O(leaves) — the GLM-5.3 MLA-absorb transposed-weight read is ~1e6 leaves per operand.  Every reader expands back to the
    per-leaf run lists, bit-identically (`compact.expanded_file_args` at `program_view.iter_instance_rows` and the raw doc loaders)."""
    from verity.ml.circuits.codec import descriptor_id
    from verity_vllm.program.compact import file_items_from_refs
    for k, n in enumerate(program.fn.body.nodes):
        yield {"i": k, "spec": descriptor_id(n.fn), "form": n.form, "name": n.name, "args": [file_items_from_refs(a.refs) for a in n.args]}


def _head_text(head: dict) -> str:
    return json.dumps(head)[:-1] + (', "rows": [' if head else '"rows": [')


def write_instances(f, head: dict, rows, chunk: int = 4096) -> None:
    """`json.dump({**head, "rows": list(rows)}, f)`, the same text, written a chunk of rows at a time with `json.dumps` (its C encoder;
    `json.dump` to a file always runs the Python one) and without the rows list."""
    f.write(_head_text(head))
    batch, first = [], True
    for r in rows:
        batch.append(r)
        if len(batch) == chunk:
            f.write(("" if first else ", ") + json.dumps(batch)[1:-1])
            batch, first = [], False
    if batch:
        f.write(("" if first else ", ") + json.dumps(batch)[1:-1])
    f.write("]}")


def write_instances_file(path: str, head: dict, rows, **gz) -> None:
    with ParallelGzip(path, **gz) as f:
        write_instances(f, head, rows)


def write_program_instances(path: str, head: dict, program, **gz) -> None:
    """`write_instances_file(path, head, instance_rows(program))`'s text, from `RowTexts`."""
    with ParallelGzip(path, **gz) as f:
        f.write(_head_text(head))
        for k, text in enumerate(RowTexts().rows(program)):
            f.write(text if k == 0 else ", " + text)
        f.write("]}")


def _items_text(progs) -> str:
    """The stored items of compact progressions as `json.dumps` writes them inside a list, brackets off."""
    from verity_vllm.program.compact import _file_items
    return json.dumps(_file_items(progs))[1:-1]


class _LogText:
    """One `PartLog`'s operand items so far: the greedy accumulator after `log.parts[:w]` is the final progressions (their stored
    items' text, `final`), its last progression `last` (which a further run may still extend) and its open run `open`."""

    __slots__ = ("log", "w", "final", "last", "open")

    def __init__(self, log) -> None:
        self.log, self.w, self.final, self.last, self.open = log, 0, "", None, None

    def _acc(self):
        from verity_vllm.program.compact import _Acc
        acc = _Acc()
        if self.last is not None:
            acc.out.append(self.last)
        acc.open = self.open
        return acc

    def advance(self, j: int) -> None:
        from verity_vllm.program.compact import _walk_refs
        if j <= self.w:
            return
        acc = self._acc()
        for p in self.log.parts[self.w:j]:
            _walk_refs(acc, p)
        out = acc.out
        if len(out) > 1:
            more = _items_text(out[:-1])
            self.final = f"{self.final}, {more}" if self.final else more
        self.w, self.last, self.open = j, (out[-1] if out else None), acc.open

    def cut(self, j: int, tail) -> str:
        """The stored items' text of `log.parts[:j] + tail` (`j` at least every cut before)."""
        from verity_vllm.program.compact import _walk_refs
        self.advance(j)
        acc = self._acc()
        for p in tail:
            _walk_refs(acc, p)
        rest = _items_text(acc.finish())
        if self.final and rest:
            return f"[{self.final}, {rest}]"
        return f"[{self.final or rest}]"


class RowTexts:
    """Each root node's row as the text `json.dumps` gives its `instance_rows` dict.  An operand cut from a `PartLog` (`Concat.prefix`)
    is the log's items so far plus its own (`_LogText`); a cut shorter than one already written for its log, and every other operand,
    is encoded whole (`compact.file_items_from_refs`)."""

    def __init__(self) -> None:
        self.logs: dict[int, _LogText] = {}
        self.specs: dict[int, str] = {}

    def operand(self, r) -> str:
        from verity.ml.circuits.refs import Concat
        from verity_vllm.program.compact import file_items_from_refs
        if type(r) is Concat and r.prefix is not None:
            log, j = r.prefix
            lt = self.logs.get(id(log))
            if lt is None:
                lt = self.logs[id(log)] = _LogText(log)
            if j >= lt.w:
                return lt.cut(j, r.parts[j:])
        return json.dumps(file_items_from_refs(r))

    def rows(self, program) -> Iterator[str]:
        from verity.ml.circuits.codec import descriptor_id
        dumps, specs = json.dumps, self.specs
        for k, n in enumerate(program.fn.body.nodes):
            spec = specs.get(id(n.fn))
            if spec is None:
                spec = specs[id(n.fn)] = dumps(descriptor_id(n.fn))
            args = ", ".join([self.operand(a.refs) for a in n.args])
            yield f'{{"i": {k}, "spec": {spec}, "form": {dumps(n.form)}, "name": {dumps(n.name)}, "args": [{args}]}}'


def _deflate(data: bytes, zdict: bytes, level: int) -> bytes:
    c = zlib.compressobj(level, zlib.DEFLATED, -15, 8, zlib.Z_DEFAULT_STRATEGY, zdict) if zdict else zlib.compressobj(level, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush(zlib.Z_SYNC_FLUSH)


def gz_threads() -> int:
    return max(1, min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1))


class ParallelGzip:
    """A text file written as one gzip member (RFC 1952) whose deflate stream is compressed `block` bytes at a time on threads, the
    layout pigz writes: each block is deflated with the 32 KiB of text before it as its dictionary and ends in a sync flush (a
    byte-aligned empty stored block), so the blocks concatenate into one deflate stream, closed by an empty final block.  zlib
    releases the GIL while it deflates.  The header carries no name and mtime 0: the same text gives the same bytes."""

    def __init__(self, path: str, *, level: int = INSTANCES_GZ_LEVEL, threads: int | None = None, block: int = GZ_BLOCK) -> None:
        self.level, self.block, self.threads = level, block, threads or gz_threads()
        self.f = open(path, "wb")
        self.f.write(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff")
        self.pool = ThreadPoolExecutor(self.threads)
        self.pending: collections.deque = collections.deque()
        self.parts: list[str] = []
        self.n = 0
        self.crc = self.size = 0
        self.prev = b""

    def write(self, s: str) -> None:
        self.parts.append(s)
        self.n += len(s)
        if self.n >= self.block:
            self._submit()

    def _submit(self) -> None:
        data = "".join(self.parts).encode()
        self.parts, self.n = [], 0
        self.crc, self.size = zlib.crc32(data, self.crc), self.size + len(data)
        zdict, self.prev = self.prev, data[-_WINDOW:]
        self.pending.append(self.pool.submit(_deflate, data, zdict, self.level))
        while len(self.pending) > 2 * self.threads:
            self.f.write(self.pending.popleft().result())

    def close(self) -> None:
        try:
            if self.parts:
                self._submit()
            while self.pending:
                self.f.write(self.pending.popleft().result())
            self.f.write(_FINAL_EMPTY_BLOCK + struct.pack("<II", self.crc & 0xFFFFFFFF, self.size & 0xFFFFFFFF))
        finally:
            self.pool.shutdown()
            self.f.close()

    def __enter__(self) -> "ParallelGzip":
        return self

    def __exit__(self, exc_type, *_) -> None:
        if exc_type is None:
            self.close()
        else:
            self.pool.shutdown(cancel_futures=True)
            self.f.close()


_CHUNK = 1 << 22


def _open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


class _Text:
    """A text file read `_CHUNK` at a time, for decoding one JSON value after another out of it."""

    __slots__ = ("f", "buf", "pos", "eof")

    def __init__(self, f) -> None:
        self.f, self.buf, self.pos, self.eof = f, "", 0, False

    def _more(self) -> None:
        more = self.f.read(max(_CHUNK, len(self.buf) - self.pos))   # a value longer than a chunk: double, so it is decoded O(1) times
        self.eof = not more
        self.buf, self.pos = self.buf[self.pos:] + more, 0

    def peek(self) -> str:
        """The next character that is not JSON whitespace ("" at the end), not consumed."""
        while True:
            buf, pos = self.buf, self.pos
            while pos < len(buf) and buf[pos] in " \t\n\r":
                pos += 1
            self.pos = pos
            if pos < len(buf) or self.eof:
                return buf[pos] if pos < len(buf) else ""
            self._more()

    def take(self, c: str) -> None:
        if self.peek() != c:
            raise json.JSONDecodeError(f"expected {c!r}", self.buf, self.pos)
        self.pos += 1

    def value(self, dec: json.JSONDecoder) -> Any:
        self.peek()
        while True:
            try:
                obj, end = dec.raw_decode(self.buf, self.pos)
                if end < len(self.buf) or self.eof:   # a number at the end of the buffer may go on in the next chunk
                    self.pos = end
                    if end > _CHUNK:
                        self.buf, self.pos = self.buf[end:], 0
                    return obj
            except json.JSONDecodeError:
                if self.eof:
                    raise
            self._more()


def read_instances(path: str, *, expand: bool = True) -> dict[str, Any]:
    """`json.load` of an instance-sequence file, the same document (with `expand`, each row's arguments `compact.expanded_file_args`),
    decoded a row at a time with every argument item that equals one already read being that item's list: a causal attention's key
    operand repeats the call before's items, so decoded one list per occurrence the rows were 56 times the file (GLM-5.3 TP8 B8 I256,
    8 request Programs of 2.5 GB: 139.5 GB).  The rows' keys, specs and forms are shared the same way.  A reader must not modify an
    item in place: equal items are one list."""
    from verity_vllm.program.compact import _FILE_SPACES, runs_from_file_items
    items: dict[tuple, list] = {}
    strs: dict[str, str] = {}

    def item(it):
        try:
            return items.setdefault(tuple(it), it)
        except TypeError:                     # an item that is not a flat list of scalars: as decoded
            return it

    def run(t):
        got = items.get(t)
        if got is None:
            got = items[t] = list(t)
        return got

    def operand(a):
        if not isinstance(a, list):
            return a
        if expand and any(len(it) == 8 and it[0] in _FILE_SPACES for it in a if isinstance(it, list)):
            return [run(t) for t in runs_from_file_items(a)]
        return [item(it) if isinstance(it, list) else it for it in a]

    def row(r):
        if not isinstance(r, dict):
            return r
        out = {}
        for k, v in r.items():
            k = strs.setdefault(k, k)
            if k == "args" and isinstance(v, list):
                v = [operand(a) for a in v]
            elif k in ("spec", "form") and isinstance(v, str):
                v = strs.setdefault(v, v)
            out[k] = v
        return out

    dec = json.JSONDecoder()
    doc: dict[str, Any] = {}
    with _open_text(path) as f:
        t = _Text(f)
        t.take("{")
        if t.peek() == "}":
            t.pos += 1
        else:
            while True:
                key = t.value(dec)
                if not isinstance(key, str):
                    raise json.JSONDecodeError("expected a key", t.buf, t.pos)
                t.take(":")
                if key == "rows" and t.peek() == "[":
                    t.pos += 1
                    rows = doc[key] = []
                    if t.peek() == "]":
                        t.pos += 1
                    else:
                        while True:
                            rows.append(row(t.value(dec)))
                            if t.peek() == "]":
                                t.pos += 1
                                break
                            t.take(",")
                else:
                    doc[key] = t.value(dec)
                if t.peek() == "}":
                    t.pos += 1
                    break
                t.take(",")
        if t.peek():
            raise json.JSONDecodeError("Extra data", t.buf, t.pos)
    return doc
