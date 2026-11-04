"""The served Program's weights registered at H200 scale: every tensor of `e2e_h200.served_weights` committed as
the registered value the Program reads (core's ``one_stage.registered``), in bounded memory, under the operating system's
salts, and every row opened later against its root.

* **The reader** (`reader`): the Program's leaf width of each tensor, from its root parameters' types (`program`'s
  ``_weight_type``: every weight parameter is an array of bits, so every tensor, an f32 scale too, is read at 16-bit
  leaves) and its digest, which the caller names (the H200 Program's is its registration run's). Neither the Program nor
  its gate lists are built: `weights.reader` takes them from ``registered.parameter_gates``, a list of every leaf's gate,
  ~86 GB a tensor at H200.
* **The commit** (`commit`): each tensor's rows in aligned chunks of 2^b rows (at most ``chunk_bytes``) over ``workers``
  processes. A chunk's bytes are generated from the seed, or read from a weights stream file (``source``: interface 1's
  ``plain`` stream, the file the H200's server loads); its rows' salts are sampled from the operating system by the worker
  that commits them (every hm96 salt is the operating system's) and written to the
  witness, or read from an existing witness (``salts="witness"``: the same registration again, or network's load's,
  `load_relay`'s witness). The witness holds every tensor's row salts in tensor order, each row's 192 bytes at
  ``192 * (rows of earlier tensors + row)`` (load_relay's layout); it is the prover's alone (0600). Each worker returns its
  chunk's subtree root; the main process keeps each tensor's chunk roots (``tops.bin``, 64 bytes a chunk) and folds them.
  A worker holds one chunk, the main process the chunk roots: at H200, 25,284,864 rows in 26,002 chunks, 1.7 MB.
* **The state** (``state``, a directory): ``commit.json`` (the shape, the seed, the source, the reader, each tensor's
  entry, chunk bits and offset in ``tops.bin``), ``tops.bin`` and, unless named elsewhere, the witness ``salts.bin``.
  ``commit.json``'s ``registered`` is the record's ``registered`` list (``registration.record``'s, `REG.Registered.entry`
  each); ``values_digest`` is interface 1's registration digest of it (`weights.registration_digest`).
* **Opening** (`open_state`, `Value`): a value as a sampled proof block reads it, standing in for a `REG.Registered` where a
  bundle takes one (``replay_bundle.build``'s ``weights``: its ``program``, ``bcs[pos]`` and ``path(pos)``): a row's
  bytes regenerated or read, its salt read from the witness, and its path the chunk's subtree recomputed, then the chunk
  roots above it.
* **The reads** (`reads`): a ``ScaledMmS8CoordinateRows_v2{K, R}`` proof block of linear ``op`` at output coordinate n reads
  ``wrow{r}`` as row ``n R + r`` of ``op`` and ``sw`` as row n of ``op.scale`` (`program`'s strided view of W: a registered
  row of W is a ``wrow`` part).

    python -m e2e_h200.weights_commit commit --shape h200 --program HEX --state DIR [--seed demo] [--source W]
        [--witness PATH] [--salts os|witness] [--workers N] [--out FILE]
    python -m e2e_h200.weights_commit open --state DIR --op l0.wup --coordinate N [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from e2e_h200 import served_weights as SW

from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import merkle_depth

from .program import _weight_type
from .shape import SHAPES, forms

FORMAT = "verity/e2e-h200/weights-registration/v1"
SALT = hm96.SHA512.salt_bytes
_INST = hm96.SHA512
SEEDS = {"test": SW.TEST_SEED, "demo": SW.DEMO_SEED}


def seed_of(text: str) -> bytes:
    """``test`` (TEST_SEED), ``demo`` (DEMO_SEED) or a text seed, as `run.py` reads ``--seed``."""
    return SEEDS.get(text, text.encode())


def reader(shape: SW.Shape, digest: str) -> SW.Reader:
    """The served Program's reader at `shape`: its digest (the caller's) and each tensor's leaf width, from the type the
    Program gives its root parameter (``registered.width`` and ``row_leaves`` on that type alone)."""
    digest, widths = SW.reader(shape, digest).digest, []
    for t in SW.tensors(shape):
        typ = _weight_type(t)
        w = typ.homogeneous_width()
        if w is None or not 0 < w <= max(REG.WIDTHS):
            raise ValueError(f"{t.name}: its leaves are not one width of at most {max(REG.WIDTHS)} bits")
        b = 16 * -(-w // 16)
        words = SW.registered_words(t, b)
        if t.schema != REG.SCHEMA and words * b % w:
            raise ValueError(f"{t.name}: a {t.schema} row of {words} {b}-bit leaves is no whole number of its {w}-bit leaves")
        k = words if t.schema == REG.SCHEMA else words * b // w
        if typ.leaves % k or typ.leaves // k != t.leaves:
            raise ValueError(f"{t.name}: the Program reads {typ.leaves} leaves of it, not {t.leaves} rows of {k}")
        widths.append((t.name, b))
    return SW.Reader(digest, tuple(widths))


def _bases(shape: SW.Shape) -> list[int]:
    """Each tensor's first row in the witness: the rows of the tensors before it."""
    out, at = [], 0
    for t in SW.tensors(shape):
        out.append(at)
        at += t.leaves
    return out


def _rows_bytes(shape: SW.Shape, seed: bytes, source: str | None, t: SW.Tensor, j0: int, n: int) -> bytes:
    rb = 2 * t.words
    if source is None:
        return SW.tensor_bytes(shape, seed, t.name, j0 * rb, (j0 + n) * rb)
    with open(source, "rb", buffering=0) as f:
        data = os.pread(f.fileno(), n * rb, t.offset + j0 * rb)
    if len(data) != n * rb:
        raise ValueError(f"{source}: {t.name} rows {j0}..{j0 + n - 1} are past its end")
    return data


def _salts(witness: str, fresh: bool, at: int, n: int) -> bytes:
    fd = os.open(witness, os.O_WRONLY if fresh else os.O_RDONLY)
    try:
        if fresh:
            ys = os.urandom(n * SALT)
            if os.pwrite(fd, ys, at * SALT) != len(ys):
                raise OSError(f"{witness}: a short write at row {at}")
            return ys
        ys = os.pread(fd, n * SALT, at * SALT)
        if len(ys) != n * SALT:
            raise ValueError(f"{witness}: no salts for rows {at}..{at + n - 1}")
        return ys
    finally:
        os.close(fd)


def _chunk(args: tuple) -> tuple[int, int, bytes]:
    shape, seed, source, witness, fresh, r, base, index, chunk, bits = args
    t = SW.tensors(shape)[index]
    j0 = chunk << bits
    n = max(0, min(1 << bits, t.leaves - j0))
    data = _rows_bytes(shape, seed, source, t, j0, n) if n else b""
    ys = _salts(witness, fresh, base + j0, n) if n else b""
    return index, chunk, SW.chunk_levels(t, SW.domain(t, r), data, ys, j0, bits)[-1][0]


def _fold(dom: Any, level: list[bytes], d: int) -> list[list[bytes]]:
    """The levels above the chunk roots `level` (at tree level `d`), up to the root."""
    levels = [level]
    while len(level) > 1:
        level = [dom.node(d, k // 2, level[k], level[k + 1]) for k in range(0, len(level), 2)]
        levels.append(level)
        d += 1
    return levels


def commit(shape: SW.Shape, seed: bytes, digest: str, state: str | Path, *, source: str | None = None,
           witness: str | None = None, salts: str = "os", workers: int = 1, chunk_bytes: int = SW.CHUNK_BYTES,
           progress: Any = None) -> dict[str, Any]:
    """Every tensor registered for the Program of `digest` at `shape` (module docstring), its state written to `state`:
    -> ``commit.json``'s document. ``salts``: ``os`` draws them into a new witness (default ``state/salts.bin``),
    ``witness`` reads an existing one."""
    if salts not in ("os", "witness"):
        raise ValueError(f"salts are the operating system's (os) or an existing witness's (witness), not {salts!r}")
    fresh = salts == "os"
    r, ts, bases = reader(shape, digest), SW.tensors(shape), _bases(shape)
    rows = sum(t.leaves for t in ts)
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    os.chmod(state, 0o700)
    wit = str(Path(witness) if witness else state / "salts.bin")
    if fresh:
        fd = os.open(wit, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.ftruncate(fd, rows * SALT)
        finally:
            os.close(fd)
    elif os.path.getsize(wit) != rows * SALT:
        raise ValueError(f"{wit}: {os.path.getsize(wit)} bytes, not {rows} rows' salts")
    if source is not None and os.path.getsize(source) < SW.stream_bytes(shape):
        raise ValueError(f"{source}: shorter than {shape.name}'s weights stream ({SW.stream_bytes(shape)} bytes)")
    plan, bits_of = [], {}
    for i, t in enumerate(ts):
        bits_of[i] = SW.chunk_bits(t, chunk_bytes)
        plan += [(shape, seed, source, wit, fresh, r, bases[i], i, c, bits_of[i])
                 for c in range((1 << merkle_depth(t.leaves)) >> bits_of[i])]
    got: dict[int, dict[int, bytes]] = {i: {} for i in range(len(ts))}
    t0 = time.time()
    pool = multiprocessing.get_context("fork").Pool(workers) if workers > 1 else None
    try:
        results = pool.imap_unordered(_chunk, plan, chunksize=1) if pool else map(_chunk, plan)
        for done, (i, c, sub) in enumerate(results, 1):
            got[i][c] = sub
            if progress is not None:
                progress(done, len(plan))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    tops, values, roots, at = bytearray(), {}, {}, 0
    for i, t in enumerate(ts):
        level = [got[i][c] for c in range(len(got[i]))]
        roots[t.name] = _fold(SW.domain(t, r), level, bits_of[i])[-1][0]
        values[t.name] = {"bits": bits_of[i], "chunks": len(level), "tops": at}
        tops += b"".join(level)
        at += len(level)
    (state / "tops.bin").write_bytes(bytes(tops))
    doc = {"format": FORMAT, "shape": shape.name, "seed_id": SW.model_id(shape, seed), "program": digest,
           "source": source, "witness": wit, "salts": "operating system" if fresh else "witness", "rows": rows,
           "chunks": len(plan), "widths": dict(r.widths), "values": values,
           "registered": SW.entries(shape, r, roots),
           "values_digest": SW.registration_digest(shape, r, roots),
           "seconds": round(time.time() - t0, 1),
           "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3),
           "workers_maxrss_gb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20, 3)}
    (state / "commit.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


class _Bcs:
    def __init__(self, value: Value):
        self.value = value

    def __getitem__(self, pos: int) -> bytes:
        v = self.value
        return _INST.commit_string(_INST.default_key, REG.row_digest(v.row(pos), v.word_bits, v.schema), v.salt(pos))

    def __len__(self) -> int:
        return self.value.leaves


class Value:
    """One registered tensor of a commit's state, as a sampled proof block reads it: what `REG.Registered` gives a bundle
    (``program``, ``value``, ``words``, ``word_bits``, ``leaves``, ``schema``, ``root``, ``domain``, ``entry()``, ``bcs[pos]``,
    ``path(pos)``, ``statement(positions)``), each row computed when asked."""

    def __init__(self, state: State, name: str):
        e = next((e for e in state.doc["registered"] if e["value"] == name), None)
        if e is None:
            raise ValueError(f"the commit registers no {name!r}")
        self.state, self._entry, self.tensor = state, e, SW.tensor(state.shape, name)
        self.index = next(i for i, t in enumerate(SW.tensors(state.shape)) if t.name == name)
        self.program, self.value = state.doc["program"], name
        self.words, self.leaves = int(e["words"]), int(e["leaves"])
        self.word_bits = int(e.get(REG.WIDTH_KEY, REG.WORD_BITS))
        self.schema, self.root = self.tensor.schema, bytes.fromhex(e["root"])
        self.window = self.produced = self.session = None
        self.bcs = _Bcs(self)
        v = state.doc["values"][name]
        self.bits, top = int(v["bits"]), int(v["tops"])
        self._tops = [state.tops[64 * (top + c):64 * (top + c + 1)] for c in range(int(v["chunks"]))]
        self._above: list[list[bytes]] | None = None
        self._last: tuple[int, list[list[bytes]]] = (-1, [])

    @property
    def domain(self):
        return REG.domain(self.program, self.value, self.words, self.leaves, self.word_bits)

    def entry(self) -> dict[str, Any]:
        return dict(self._entry)

    def _check(self, pos: int) -> int:
        if not 0 <= int(pos) < self.leaves:
            raise ValueError(f"{self.value} has rows [0, {self.leaves})")
        return int(pos)

    def row_bytes(self, pos: int) -> bytes:
        s = self.state
        return _rows_bytes(s.shape, s.seed, s.doc["source"], self.tensor, self._check(pos), 1)

    def row(self, pos: int) -> tuple[int, ...]:
        return tuple(int(x) for x in np.frombuffer(self.row_bytes(pos), "<u2" if self.word_bits == 16 else "<u4"))

    def salt(self, pos: int) -> bytes:
        return _salts(self.state.doc["witness"], False, self.state.bases[self.index] + self._check(pos), 1)

    def _chunk_levels(self, chunk: int) -> list[list[bytes]]:
        if self._last[0] != chunk:
            self._last = (chunk, self._levels(chunk))
        return self._last[1]

    def _levels(self, chunk: int) -> list[list[bytes]]:
        s, t, j0 = self.state, self.tensor, chunk << self.bits
        n = max(0, min(1 << self.bits, self.leaves - j0))
        data = _rows_bytes(s.shape, s.seed, s.doc["source"], t, j0, n)
        ys = _salts(s.doc["witness"], False, s.bases[self.index] + j0, n)
        levels = SW.chunk_levels(t, self.domain, data, ys, j0, self.bits)
        if levels[-1][0] != self._tops[chunk]:
            raise ValueError(f"{self.value} chunk {chunk}: its rows and salts are not the committed chunk root")
        return levels

    def path(self, pos: int) -> bytes:
        """The frame-v3 path of row `pos`: its siblings, leaf level first, 64 bytes each."""
        pos = self._check(pos)
        out, i = b"", pos - ((pos >> self.bits) << self.bits)
        for level in self._chunk_levels(pos >> self.bits)[:-1]:
            out += level[i ^ 1]
            i >>= 1
        if self._above is None:
            self._above = _fold(self.domain, self._tops, self.bits)
            if self._above[-1][0] != self.root:
                raise ValueError(f"{self.value}: its chunk roots are not its registered root")
        i = pos >> self.bits
        for level in self._above[:-1]:
            out += level[i ^ 1]
            i >>= 1
        return out

    def statement(self, positions: Sequence[int]) -> tuple[list[tuple[int, ...]], list[bytes], dict[str, Any]]:
        """As `REG.Registered.statement`: the rows at `positions`, their registered salts, and the header entry."""
        ps = [self._check(p) for p in positions]
        return ([self.row(p) for p in ps], [self.salt(p) for p in ps],
                {"value": self.value, "positions": ps, "paths": [self.path(p).hex() for p in ps]})


class State:
    """A commit's state directory (`commit`), opened: its document, chunk roots and witness."""

    def __init__(self, state: str | Path, seed: bytes):
        self.dir = Path(state)
        self.doc = json.loads((self.dir / "commit.json").read_text())
        if self.doc.get("format") != FORMAT:
            raise ValueError(f"{self.dir}: not a {FORMAT} state")
        self.shape, self.seed = SHAPES[self.doc["shape"]], seed
        if SW.model_id(self.shape, seed) != self.doc["seed_id"]:
            raise ValueError(f"{self.dir}: committed under another seed")
        self.tops = (self.dir / "tops.bin").read_bytes()
        self.bases = _bases(self.shape)
        self._values: dict[str, Value] = {}

    def value(self, name: str) -> Value:
        if name not in self._values:
            self._values[name] = Value(self, name)
        return self._values[name]

    def weights(self) -> Mapping[str, Value]:
        """Every registered value by name, opened when read: ``replay_bundle.build``'s ``weights``."""
        return _Lazy(self)


class _Lazy(dict):
    def __init__(self, state: State):
        super().__init__()
        self.state = state

    def __missing__(self, name: str) -> Value:
        self[name] = v = self.state.value(name)
        return v


def open_state(state: str | Path, seed: bytes) -> State:
    return State(state, seed)


def _parts(shape: SW.Shape, op: str) -> int:
    """R, the ``wrow`` parts of linear `op`'s proof block: K over its quantizer chunk, each part one registered row."""
    t = SW.tensor(shape, op)
    if t.kind != "s8":
        raise ValueError(f"{op} is no linear")
    q = forms(shape).qchunk(t.cols)
    if q != shape.row_codes:
        raise ValueError(f"{shape.name}: {op}'s wrow parts are {q} codes, its registered rows {shape.row_codes}: a part is "
                         "no one registered row (a shape whose row_codes is its quantizer chunk, as h200's and mini-rows' are)")
    return t.cols // q


def reads(shape: SW.Shape, op: str, n: int, port: str) -> tuple[str, int] | None:
    """The ``(value, row)`` port `port` of linear `op`'s ``ScaledMmS8CoordinateRows_v2`` proof block at output coordinate `n`
    reads (``wrow{r}``: W's row ``n R + r``; ``sw``: the scale's row n), None for a port that reads no registered value."""
    R, t = _parts(shape, op), SW.tensor(shape, op)
    if not 0 <= n < t.rows:
        raise ValueError(f"{op} has output coordinates [0, {t.rows}), not {n}")
    if port.startswith("wrow") and port[4:].isdigit() and int(port[4:]) < R:
        return op, n * R + int(port[4:])
    if port == "sw":
        return op + ".scale", n
    return None


def scaled_mm_ports(shape: SW.Shape, op: str) -> list[str]:
    return [f"wrow{r}" for r in range(_parts(shape, op))] + ["sw"]


def open_block(state: State, op: str, n: int) -> dict[str, Any]:
    """Every registered read of linear `op`'s proof block at output coordinate `n`, opened and checked as the verifier
    checks it (``REG.verify_read`` under the domain the entry's rule gives): ``{port: {value, position, string, path, ok}}``."""
    out = {}
    for port in scaled_mm_ports(state.shape, op):
        value, pos = reads(state.shape, op, n, port)
        v, e = state.value(value), state.value(value).entry()
        dom = REG.domain(state.doc["program"], value, int(e["words"]), int(e["leaves"]), int(e.get(REG.WIDTH_KEY, REG.WORD_BITS)))
        bc, path = v.bcs[pos], v.path(pos)
        out[port] = {"value": value, "position": pos, "string": bc.hex(), "path": path.hex(),
                     "ok": dom.domain_id.hex() == e["domain"] and
                     REG.verify_read(dom, bytes.fromhex(e["root"]), pos, bc, path, v.schema)}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e2e_h200.weights_commit", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commit")
    c.add_argument("--shape", default="h200")
    c.add_argument("--program", required=True, help="the reading Program's digest (128 hex)")
    c.add_argument("--state", required=True)
    c.add_argument("--seed", default="demo")
    c.add_argument("--source", help="a weights stream file to read rows from (default: generate them)")
    c.add_argument("--witness", help="the witness file (default STATE/salts.bin)")
    c.add_argument("--salts", default="os", choices=("os", "witness"))
    c.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    c.add_argument("--out", help="also write the registered entries and the summary here")
    o = sub.add_parser("open")
    o.add_argument("--state", required=True)
    o.add_argument("--seed", default="demo")
    o.add_argument("--op", required=True)
    o.add_argument("--coordinate", type=int, required=True)
    o.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "commit":
        last = [0.0]

        def progress(done: int, total: int) -> None:
            if time.time() - last[0] > 60 or done == total:
                last[0] = time.time()
                print(f"{time.strftime('%H:%M:%S')} {done}/{total} chunks", file=sys.stderr, flush=True)

        doc = commit(SHAPES[a.shape], seed_of(a.seed), a.program, a.state, source=a.source, witness=a.witness,
                     salts=a.salts, workers=a.workers, progress=progress)
        summary = {k: doc[k] for k in ("format", "shape", "seed_id", "program", "salts", "rows", "chunks", "values_digest",
                                       "seconds", "maxrss_gb", "workers_maxrss_gb")}
        if a.out:
            Path(a.out).write_text(json.dumps({**summary, "registered": doc["registered"]}, indent=1) + "\n")
        print(json.dumps(summary))
        return 0
    s = open_state(a.state, seed_of(a.seed))
    got = open_block(s, a.op, a.coordinate)
    ok = all(v["ok"] for v in got.values())
    verdict = {"verdict": "OPEN" if ok else "REFUSED", "program": s.doc["program"], "op": a.op, "coordinate": a.coordinate,
               "reads": {p: {"value": v["value"], "position": v["position"], "ok": v["ok"]} for p, v in got.items()}}
    if a.out:
        Path(a.out).write_text(json.dumps({**verdict, "opened": got}, indent=1) + "\n")
    print(json.dumps(verdict))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
