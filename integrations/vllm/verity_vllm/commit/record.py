"""The Commit's root of record: ``frame-v3-sha512`` trees of ``hm96-sha512/row/v1`` hiding rows, one row per committer leaf.

The committer (`committer.native_host` and its subclasses) lays each engine step out as leaves of ``chunk`` bytes: the positions
``(step, leaf)`` every opening, binding map and reader addresses.  The record commits those leaves again, each as one hiding row in the
format serving rows (`commit.serving_rows`) and M0 use, so a shared-row table (``GemmCoordinate_v1``'s) commits under the same leaf rule:

  row     the leaf's bytes, zero-padded to ``row_bytes`` (``chunk`` rounded up to whole SHA-512 blocks), as u16 little-endian words
  inner   ``x = SHA-512(sha512_row_prefix(role 1, 16, row_bytes / 2) || row)``                     (``sha512/row/v1``)
  salt    ``y`` = 192 bytes of the RFC 8439 ChaCha20 keystream under the run's salt key (32 fresh OS bytes the prover keeps), nonce the
          step, block counter ``3 * leaf`` (`StepRecord.salt`); an opening carries its leaves' salts
  leaf    the ``frame-v3-sha512`` leaf at rank ``leaf`` with schema ``hm96-sha512/row/v1`` over
          ``hm96.SHA512.tree_leaf(default_key, hm96.SHA512.commit_string(default_key, x, y))``

Step ``s`` is one tree under ``CommitmentDomain(binding_s, -1, [0, leaves), "sha512")``, ``binding_s = identity_digest_sha512(STEP_TAG,
{scheme, program, geo, step, layout, leaves, row_bytes})`` over the committer's Program digest, geometry digest and step layout digest,
so a verifier recomputes every domain from the run's public facts (`step_binding`).  The run root of record is ``identity_digest_sha512
(RUN_TAG, {scheme, program, geo, steps: [[step root, leaves], ...]})`` (`run_root`), 64 bytes.  A verifier checks an opening of
leaves ``[lo, hi)`` of a step from their bytes, their salts and the frame range path alone (`RunRecord.verify_leaves`).

The record replaces ``vllm-v1`` (SHA-256 position leaves) as what openings, binding draws, replay seeds, bundles and the serving rows'
window read.  The committer still hashes its own ``vllm-v1`` tree while serving; its root is ``committer_root``, a determinism diagnostic
(two runs over equal bytes have equal committer roots; a hiding root never repeats), not of record.  A run recorded before this module
names no record and reads its ``vllm-v1`` root as it always did.

Hashing is on the host here (`HostRows`), after every step is final and before any challenge.  A committer may carry a
``record_hasher`` with the same ``levels(jobs)`` method (the GPU committer's seam); the record is then the same bytes.  The host path
needs every leaf's bytes retained until the record is built: a committer that dropped some (``--retain none`` or ``sample``,
``--retain-exclude``) is refused by name (`NotRetained`).
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Protocol

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth
from verity.ml.commitments.rowleaf import (
    ROLE_X,
    SCHEMA_HM96_SHA512_ROW,
    row_leaf,
    sha512_row_prefix,
)

from .hiding import HidingLeaves
from .serving_rows import SCHEME, _copy_update, _Frame, _part, chacha20

FORMAT = "verity-vllm/commit-record/v1"
ROW_SCHEMA = SCHEMA_HM96_SHA512_ROW
STEP_TAG = "verity-vllm/commit-record/step/v1"
RUN_TAG = "verity-vllm/commit-record/run/v1"
#: the scheme a run recorded before the record names (``commit.scheme.NAME``): read only
LEGACY_SCHEME = "vllm-v1"
OWNER = -1
INST = hm96.SHA512
KEY = INST.default_key
SALT_BYTES = INST.salt_bytes
SALT_BLOCKS = SALT_BYTES // 64
NODE = 64
BLOCK = 128
#: below this many leaves in a batch, the host hasher runs in the calling process
POOL_MIN = 1 << 13
#: leaves per hashing task: one aligned subtree (a power of two)
TASK = 1 << 12
#: rows held at once while building (bytes)
BATCH_BYTES = 256 << 20


class NotRetained(RuntimeError):
    """A leaf the record must hash is not held by the committer."""


def row_bytes(chunk: int) -> int:
    """A record row's bytes for leaves of ``chunk`` bytes: whole SHA-512 blocks."""
    return max(BLOCK, -(-int(chunk) // BLOCK) * BLOCK)


def step_binding(program: bytes, geo: bytes, step: int, layout: bytes, leaves: int, row: int) -> bytes:
    return identity_digest_sha512(STEP_TAG, {"scheme": SCHEME, "program": bytes(program).hex(), "geo": bytes(geo).hex(), "step": int(step),
                                             "layout": bytes(layout).hex(), "leaves": int(leaves), "row_bytes": int(row)})


def step_domain(binding: bytes, leaves: int) -> CommitmentDomain:
    return CommitmentDomain(bytes(binding), OWNER, RangeIndexedDomain(0, int(leaves)), hash="sha512")


def run_root(program: bytes, geo: bytes, steps: Sequence[tuple[bytes, int]]) -> bytes:
    return identity_digest_sha512(RUN_TAG, {"scheme": SCHEME, "program": bytes(program).hex(), "geo": bytes(geo).hex(),
                                            "steps": [[bytes(r).hex(), int(n)] for r, n in steps]})


def salts(key: bytes, step: int, lo: int, hi: int) -> bytes:
    """Leaves ``[lo, hi)`` of ``step``'s salts under the run's salt key, concatenated."""
    return chacha20(key, int(step).to_bytes(12, "little"), (hi - lo) * SALT_BYTES, counter=lo * SALT_BLOCKS) if hi > lo else b""


def leaf_digest(domain: CommitmentDomain, index: int, value: bytes, salt: bytes, row: int) -> bytes:
    """The frame leaf of one row through core's reference (the verifier's path)."""
    if len(value) > row or len(salt) != SALT_BYTES:
        raise ValueError(f"a record leaf is at most {row} bytes with a {SALT_BYTES}-byte salt")
    x = hashlib.sha512(sha512_row_prefix(ROLE_X, 16, row // 2) + bytes(value) + bytes(row - len(value))).digest()
    return row_leaf(domain, int(index), INST.tree_leaf(KEY, INST.commit_string(KEY, x, bytes(salt))), ROW_SCHEMA)


def fold_range(domain: CommitmentDomain, digests: Sequence[bytes], leaves: int, lo: int, path: Sequence[tuple[bytes | None, bytes | None]]) -> bytes:
    """The root over the leaf digests of ranks ``[lo, lo + len(digests))`` and their boundary siblings (`StepRecord.range_path`'s shape)."""
    if not 0 <= lo < lo + len(digests) <= leaves or len(path) != merkle_depth(leaves):
        raise ValueError("the range or its path does not fit the tree")
    level, hi = list(digests), lo + len(digests)
    for d, (left, right) in enumerate(path):
        if (left is not None) != bool(lo % 2) or (right is not None) != bool(hi % 2):
            raise ValueError(f"level {d}: siblings do not match the range [{lo}, {hi})")
        if left is not None:
            level, lo = [bytes(left), *level], lo - 1
        if right is not None:
            level, hi = [*level, bytes(right)], hi + 1
        level = [domain.node(d, (lo >> 1) + j, level[2 * j], level[2 * j + 1]) for j in range(len(level) // 2)]
        lo, hi = lo >> 1, hi >> 1
    return level[0]


def path_pairs(index: int, path: Sequence[bytes]) -> tuple[tuple[bytes | None, bytes | None], ...]:
    """One leaf's sibling path as the range path of the leaf alone."""
    return tuple((sib, None) if (index >> d) & 1 else (None, sib) for d, sib in enumerate(path))


# --- the host hasher ----------------------------------------------------------------------------------------------------

@dataclass
class RowJob:
    """One step's rows to hash: ``rows`` its ``leaves`` rows of ``row`` bytes back to back, ``salts`` theirs."""
    domain: CommitmentDomain
    rows: bytes | memoryview
    salts: bytes
    row: int
    leaves: int


class RecordHasher(Protocol):
    def levels(self, jobs: Sequence[RowJob]) -> list[list[np.ndarray]]:
        """Each job's tree levels, leaves first, padded to a power of two (``frame-v3``), as ``(nodes, 64)`` uint8 arrays."""


_JOB: dict[str, Any] = {}


def _subtree(task: tuple[int, int, int]) -> tuple[int, int, list[bytes]]:
    """The levels of one aligned subtree ``[lo, lo + width)`` of job ``j``: its rows' leaves (pads past the job's leaves) folded up."""
    j, lo, width = task
    job, h = _JOB["jobs"][j], _JOB["hiding"]
    fr, row, n = _Frame(job.domain.domain_id), job.row, job.leaves
    hi = min(lo + width, n)
    leaves: list[bytes] = []
    if hi > lo:
        m, rows, ys = hi - lo, memoryview(job.rows), memoryview(job.salts)
        inner = hashlib.sha512(sha512_row_prefix(ROLE_X, 16, row // 2))
        x = b"".join(_copy_update(inner, rows[i * row:(i + 1) * row]) for i in range(lo, hi))
        b = (np.frombuffer(x, dtype=np.uint8).reshape(m, NODE) ^ h.masks(bytes(ys[lo * SALT_BYTES:hi * SALT_BYTES]), m)).tobytes()
        salt_h, leaf_h, schema = _JOB["salt_h"], _JOB["leaf_h"], _JOB["schema"]
        for i in range(m):
            s = b[i * NODE:(i + 1) * NODE] + _copy_update(salt_h, ys[(lo + i) * SALT_BYTES:(lo + i + 1) * SALT_BYTES])
            leaves.append(fr.leaf(lo + i, schema, _copy_update(leaf_h, s)))
    leaves += [fr.pad(r) for r in range(max(lo, n), lo + width)]
    out, level, first, d = [b"".join(leaves)], leaves, lo, 0
    while len(level) > 1:
        level = [fr.node(d, (first >> 1) + k, level[2 * k], level[2 * k + 1]) for k in range(len(level) // 2)]
        out.append(b"".join(level))
        first >>= 1
        d += 1
    return j, lo, out


class HostRows:
    """The record's trees on the host: numpy for ``M_key * y``, hashlib for the rest, one aligned subtree a task across ``workers``
    forked processes for a batch of `POOL_MIN` leaves or more.  Bit for bit with `leaf_digest` and core's ``FrameV3``."""

    def __init__(self, workers: int | None = None) -> None:
        self.workers = int(workers or os.environ.get("VERITY_RECORD_WORKERS") or min(16, os.cpu_count() or 1))

    def levels(self, jobs: Sequence[RowJob]) -> list[list[np.ndarray]]:
        if TASK & (TASK - 1):
            raise ValueError(f"TASK = {TASK}: a hashing task is an aligned subtree, a power of two of leaves")
        tasks, widths = [], []
        for j, job in enumerate(jobs):
            depth = merkle_depth(job.leaves) if job.leaves else 0
            w = min(TASK, 1 << depth) if job.leaves else 0
            widths.append((depth, w))
            tasks += [(j, lo, w) for lo in range(0, (1 << depth) if job.leaves else 0, w or 1)]
        _JOB.update(jobs=list(jobs), hiding=HidingLeaves(instance=INST), salt_h=hashlib.sha512(INST.salt_prefix),
                    leaf_h=hashlib.sha512(INST.leaf_prefix(KEY)), schema=_part(ROW_SCHEMA.encode()))
        try:
            if self.workers > 1 and len(tasks) > 1 and sum(job.leaves for job in jobs) >= POOL_MIN:
                with multiprocessing.get_context("fork").Pool(min(self.workers, len(tasks))) as pool:
                    res = pool.map(_subtree, tasks, chunksize=1)
            else:
                res = [_subtree(t) for t in tasks]
        finally:
            _JOB.clear()
        parts: list[dict[int, list[bytes]]] = [{} for _ in jobs]
        for j, lo, lv in res:
            parts[j][lo] = lv
        out = []
        for job, (depth, w), got in zip(jobs, widths, parts, strict=True):
            if not job.leaves:
                out.append([])
                continue
            low = [np.frombuffer(b"".join(got[lo][d] for lo in sorted(got)), dtype=np.uint8).reshape(-1, NODE) for d in range(w.bit_length())]
            fr, top, first = _Frame(job.domain.domain_id), [bytes(r) for r in low[-1]], w.bit_length() - 1
            while len(top) > 1:
                top = [fr.node(first, k, top[2 * k], top[2 * k + 1]) for k in range(len(top) // 2)]
                low.append(np.frombuffer(b"".join(top), dtype=np.uint8).reshape(-1, NODE))
                first += 1
            assert len(low) == depth + 1
            out.append(low)
        return out


# --- the record ---------------------------------------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One step's tree: its public facts (``binding`` recomputable from them), its levels and the run's salt key (the prover's)."""
    step: int
    leaves: int
    row: int
    layout: bytes
    binding: bytes
    levels: Sequence[Any]                        # (nodes, 64) per level, leaves first; [] for an empty step or a public-only record
    key: bytes | None = field(default=None, repr=False)
    root: bytes = b""
    #: without a key, the salts a store kept (a slim dump's): ``salt_of(lo, hi)``
    salt_of: Callable[[int, int], bytes] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.root:
            self.root = bytes(self.levels[-1][0]) if self.leaves else self.domain.empty_root()

    @cached_property
    def domain(self) -> CommitmentDomain:
        return step_domain(self.binding, self.leaves)

    def salt(self, lo: int, hi: int) -> bytes:
        if self.key is not None:
            return salts(self.key, self.step, lo, hi)
        if self.salt_of is not None:
            return self.salt_of(lo, hi)
        raise ValueError(f"step {self.step}: a public record holds no salts")

    def path(self, index: int) -> tuple[bytes, ...]:
        return tuple(bytes(self.levels[d][(index >> d) ^ 1]) for d in range(len(self.levels) - 1))

    def range_path(self, lo: int, hi: int) -> tuple[tuple[bytes | None, bytes | None], ...]:
        out = []
        for d in range(len(self.levels) - 1):
            out.append((bytes(self.levels[d][lo - 1]) if lo % 2 else None, bytes(self.levels[d][hi]) if hi % 2 else None))
            lo, hi = lo >> 1, (hi + 1) >> 1
        return tuple(out)

    def public(self) -> list[Any]:
        return [self.step, self.root.hex(), self.leaves, self.row, self.layout.hex(), self.binding.hex()]


@dataclass
class RunRecord:
    """The run's record: its steps' trees and the run root of record over them."""
    program: bytes
    geo: bytes
    steps: list[StepRecord]
    seconds: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def root(self) -> bytes:
        return run_root(self.program, self.geo, [(s.root, s.leaves) for s in self.steps])

    def leaves_per_step(self) -> list[int]:
        return [s.leaves for s in self.steps]

    def doc(self) -> dict[str, Any]:
        """The public record: what a verifier reads besides an opening (no salt, no node)."""
        return {"format": FORMAT, "scheme": SCHEME, "row_schema": ROW_SCHEMA, "root": self.root.hex(), "program": self.program.hex(),
                "geo": self.geo.hex(), "key_sha512": hashlib.sha512(KEY).hexdigest(), "step_tag": STEP_TAG, "run_tag": RUN_TAG,
                "columns": ["step", "root", "leaves", "row_bytes", "layout", "binding"], "steps": [s.public() for s in self.steps],
                "seconds": self.seconds}

    def verify_leaves(self, step: int, lo: int, values: Sequence[bytes], salts: bytes, path: Sequence[tuple[bytes | None, bytes | None]],
                      root: bytes, layout: bytes | None = None, steps_ok: dict[int, Any] | None = None) -> bool:
        """Leaves ``[lo, lo + len(values))`` of ``step`` against the run root ``root``: each recomputed from its bytes and salt, folded
        with the boundary siblings to the step's root; the step's binding recomputed from its public facts (``layout``: the step's
        layout digest as the caller holds it) and the step roots to ``root``.  ``steps_ok`` caches a step whose binding checked."""
        if bytes(root) != self.root or not 0 <= step < len(self.steps):
            return False
        st = self.steps[step]
        if st.step != step or not values or len(salts) != len(values) * SALT_BYTES or any(len(v) > st.row for v in values):
            return False
        if steps_ok is None or not steps_ok.get(step):
            if step_binding(self.program, self.geo, st.step, st.layout if layout is None else layout, st.leaves, st.row) != st.binding:
                return False
            if steps_ok is not None:
                steps_ok[step] = st.root
        try:
            digests = [leaf_digest(st.domain, lo + k, v, salts[k * SALT_BYTES:(k + 1) * SALT_BYTES], st.row) for k, v in enumerate(values)]
            return fold_range(st.domain, digests, st.leaves, lo, path) == st.root
        except (ValueError, TypeError, IndexError):
            return False


def from_doc(doc: dict[str, Any], levels: Callable[[int], Sequence[Any]] = lambda s: [], key: bytes | None = None) -> RunRecord:
    """A record from its public doc, and (the prover's store) each step's levels and the run's salt key; its root must be the doc's."""
    if doc.get("format") != FORMAT or doc.get("scheme") != SCHEME:
        raise ValueError(f"record {doc.get('format')!r} / {doc.get('scheme')!r} is not {FORMAT} / {SCHEME}")
    steps = [StepRecord(int(s), int(n), int(r), bytes.fromhex(lay), bytes.fromhex(b), levels(int(s)), key, bytes.fromhex(root))
             for s, root, n, r, lay, b in doc["steps"]]
    rec = RunRecord(bytes.fromhex(doc["program"]), bytes.fromhex(doc["geo"]), steps, dict(doc.get("seconds") or {}))
    if rec.root.hex() != doc["root"]:
        raise ValueError(f"record root {rec.root.hex()[:16]} is not the doc's {str(doc['root'])[:16]}")
    return rec


def build(program: bytes, geo: bytes, steps: Iterable[tuple[int, bytes, int, Callable[[], np.ndarray]]], *, chunk: int,
          hasher: RecordHasher | None = None, salt_key: bytes | None = None) -> RunRecord:
    """The record over ``steps`` = ``[(step, layout digest, leaves, rows)]``, ``rows()`` a ``(leaves, row_bytes(chunk))`` uint8 array read
    when its batch is hashed (at most `BATCH_BYTES` of rows held at once).  ``salt_key``: tests only; serving draws 32 OS bytes."""
    t0 = time.perf_counter()
    row, hasher = row_bytes(chunk), hasher or HostRows()
    key = os.urandom(32) if salt_key is None else bytes(salt_key)
    out: list[StepRecord] = []
    batch: list[tuple[int, bytes, bytes, RowJob]] = []
    held, t_hash = 0, 0.0

    def flush() -> None:
        nonlocal held, t_hash
        t = time.perf_counter()
        for (s, layout, b, job), lv in zip(batch, hasher.levels([j for *_m, j in batch]), strict=True):
            out.append(StepRecord(s, job.leaves, row, layout, b, lv, key))
        t_hash += time.perf_counter() - t
        batch.clear()
        held = 0

    for s, layout, n, rows_of in steps:
        rows = rows_of()
        if rows.shape != (n, row):
            raise ValueError(f"step {s}: rows {rows.shape} are not {n} x {row}")
        b = step_binding(program, geo, s, layout, n, row)
        batch.append((int(s), bytes(layout), b, RowJob(step_domain(b, n), memoryview(np.ascontiguousarray(rows)).cast("B"), salts(key, s, 0, n), row, n)))
        held += n * (row + SALT_BYTES)
        if held >= BATCH_BYTES:
            flush()
    flush()
    t1 = time.perf_counter()
    return RunRecord(bytes(program), bytes(geo), out, {"hash_s": round(t_hash, 3), "total_s": round(t1 - t0, 3),
                                                       "leaves": sum(s.leaves for s in out), "hasher": type(hasher).__name__})


# --- the committer's leaves ---------------------------------------------------------------------------------------------

def _np(blk: Any) -> np.ndarray:
    if getattr(blk, "is_cuda", False):
        blk = blk.cpu()
    if hasattr(blk, "numpy"):
        return blk.numpy().reshape(-1).view(np.uint8)
    return np.frombuffer(blk.tobytes() if hasattr(blk, "tobytes") else bytes(blk), dtype=np.uint8)


def step_rows(com: Any, step: int, leaves: int) -> np.ndarray:
    """Every leaf of ``step`` as the record's rows, read from the committer's retained copy (the same bytes `open` returns);
    `NotRetained` for a leaf the committer no longer holds."""
    C, row = int(com.chunk), row_bytes(com.chunk)
    out = np.zeros((leaves, row), dtype=np.uint8)
    ps = (getattr(com, "_padding", None) or {}).get(step)
    base = int(ps.leaf_base) if ps is not None else leaves
    if base:
        flat = None
        if step in com._gpu_blocks:
            blk, used = com._gpu_blocks[step]
            if blk is None or not com._block_holds(blk, 0, min(base * C, used)):
                raise NotRetained(f"step {step}: the committer does not hold every leaf's bytes (retain={getattr(com, 'retain', '?')})")
            flat = _np(blk[0:min(base * C, used)])
        elif step in com._streams:
            flat = np.frombuffer(bytes(com._streams[step][:base * C]), dtype=np.uint8)
        else:
            for m in com._layouts.get(step) or []:
                if m.n_leaves and m.first_leaf < base:
                    if m.host is None:
                        raise NotRetained(f"step {step}: {m.name} has no retained host copy (retain={getattr(com, 'retain', '?')})")
                    if int(m.chunk) > row:
                        raise ValueError(f"step {step}: {m.name}'s leaves of {m.chunk} bytes exceed the record's {row}-byte rows")
                    t = np.frombuffer(bytes(m.host), dtype=np.uint8)
                    for k in range(m.n_leaves):
                        piece = t[k * m.chunk:(k + 1) * m.chunk]
                        out[m.first_leaf + k, :len(piece)] = piece
        if flat is not None:
            n = min(len(flat), base * C)
            full, tail = divmod(n, C)
            if full:
                out[:full, :C] = flat[:full * C].reshape(full, C)
            if tail:
                out[full, :tail] = flat[full * C:n]
    if ps is not None and leaves > base:
        pad = np.frombuffer(bytes(ps.stream), dtype=np.uint8)
        for k in range(leaves - base):
            piece = pad[k * C:(k + 1) * C]
            out[base + k, :len(piece)] = piece
    return out


def of_committer(com: Any, steps: Sequence[Any]) -> RunRecord:
    """The record of a finalized run of ``com`` (``steps``: its `StepCommitment`s, padding applied), through the committer's
    ``record_hasher`` when it has one."""
    rows = [(int(s.step), bytes(s.layout_digest), int(s.n_leaves), lambda s=s: step_rows(com, int(s.step), int(s.n_leaves))) for s in steps]
    return build(com.program_digest, com._geo_digest, rows, hasher=getattr(com, "record_hasher", None), chunk=com.chunk)


def of_record(run: Any) -> bool:
    """Does ``run`` (a `RunCommitment` or a dumped run) read its root of record from a record?  Every run with a 64-byte root does; a
    run recorded before the record has a 32-byte ``vllm-v1`` root."""
    return getattr(run, "record", None) is not None or len(getattr(run, "run_root", b"") or b"") == NODE


def committer_root(run: Any) -> bytes:
    """The committer's ``vllm-v1`` root of ``run`` (a `RunCommitment` or a row's ``commit`` record): what two runs over equal bytes share
    (``--expect-root``, roots across arms, a replay's root), the run root itself for a run recorded before the record."""
    extra = run.extra if hasattr(run, "extra") else (run.get("extra") or {})
    root = bytes(run.run_root) if hasattr(run, "run_root") else bytes.fromhex(str(run.get("run_root")))
    if len(root) != NODE:
        return root
    if not (extra or {}).get("committer_root"):
        raise ValueError(f"run of record {root.hex()[:16]}: no committer root on its record (extra.committer_root)")
    return bytes.fromhex(extra["committer_root"])


class RecordRule:
    """The committer's root of record (a mixin of ``NativeHostCommitter`` and the dumped store, beside ``hiding.LeafSchemeRule``).
    ``of_record`` on: `finalize` builds the record over the finalized steps and its root is the run root; openings carry the record's
    path and salts; `verify` and `verify_range` check against it.  A run without a record reads ``vllm-v1`` as recorded."""

    of_record: bool = False
    record: RunRecord | None = None
    record_hasher: RecordHasher | None = None

    def _record_finalize(self, steps: Sequence[Any], committer_root: bytes) -> tuple[bytes, RunRecord | None]:
        """(run root, record): the record's when ``of_record`` and every step has a root, else the committer's ``vllm-v1`` root."""
        self.record = None
        if not self.of_record or not committer_root:
            return committer_root, None
        if getattr(self, "hiding", None) is not None:
            raise ValueError("the record's rows are hiding rows already: a record over hm96-sha256 committer leaves is refused")
        self.record = of_committer(self, steps)
        return self.record.root, self.record

    def _record_opened(self, step: int, lo: int, hi: int | None = None) -> tuple[Any, bytes]:
        """The record's (path, salts): of leaf ``lo`` alone (its sibling path), or of leaves ``[lo, hi)`` (their boundary siblings)."""
        st = self.record.steps[step]
        return (st.path(lo), st.salt(lo, lo + 1)) if hi is None else (st.range_path(lo, hi), st.salt(lo, hi))

    def _record_verify(self, run: Any, step: int, lo: int, values: Sequence[bytes], salts: bytes, path: Sequence[Any],
                       steps_ok: dict[int, Any] | None = None) -> bool:
        """Leaves ``[lo, lo + len(values))`` of ``step`` against ``run``'s root of record, the step's binding recomputed from THIS
        committer's Program digest, geometry digest and layout of the step."""
        rec = getattr(run, "record", None)
        if rec is None or rec.program != self.program_digest or rec.geo != self._geo_digest or not 0 <= step < len(rec.steps):
            return False
        if step >= len(run.steps) or rec.steps[step].leaves != run.steps[step].n_leaves or step not in self._layouts:
            return False
        return rec.verify_leaves(step, lo, values, salts, path, run.run_root, layout=self.layout_digest(self._layouts[step]), steps_ok=steps_ok)
