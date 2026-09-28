"""Capture format v1: what the adapter writes and everything else reads.

Layout under one run directory::

    <run>/manifest.json                 contract 2.1, canonical JSON; sha256[:12] = profile_id
    <run>/description.json              contract section 6 (verity_vllm.check.poc_description)
    <run>/required.json                 frozen required-interface set (verity_vllm.check.poc_required_interface)
    <run>/commitments.json              roots (verity_vllm.commit.merkle.build_commitments)
    <run>/capture.json                  run summary written by RunWriter.finish()
    <run>/weights/index.json            one entry per weight tensor (per shard)
    <run>/weights/values.bin            optional raw bytes (else the index carries sha256 + leaf)
    <run>/ru/<request_id>/boundary.json per request
    <run>/ru/<request_id>/<step:06d>/index.json   the replay unit's leaves
    <run>/ru/<request_id>/<step:06d>/values.bin   raw little-endian bytes, index order

Writer API (one :class:`CaptureWriter` per replay unit)::

    run = RunWriter(run_dir, manifest)
    ru = run.ru("req-0", step=0, phase="prefill", positions=[0, 1, 2, 3])
    ru.add(canonical_id, "bfloat16", [576], raw_bytes, origin="captured")
    ru.finish()
    run.write_boundary(boundary); run.write_description(desc); run.write_required(req)
    w = run.weights(store_values=True); w.add("model.embed_tokens.weight", "bfloat16", [V, H], b)
    w.finish(); run.finish()

The writer snapshots bytes at ``add`` time (it writes them to disk immediately) and refuses a
second capture of the same canonical id (N05): ``on_duplicate="error"`` raises
:class:`StaleCaptureError`; ``on_duplicate="flag"`` keeps the first value, records the conflict in
``index.json`` and marks the unit ``tainted`` so :mod:`verity_vllm.commit.merkle` refuses to commit it.

Reader API: :class:`CaptureReader` with lazy access by canonical id
(``reader.value(cid)``; ``reader.ru(request_id, step).read(cid)``; ``reader.weights.read(cid)``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import struct
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from verity_vllm.program import dtypes
from verity_vllm.commit import hashing, identity

RU_INDEX_SCHEMA = "verity-vllm/ru-index/v1"
RU_INDEX_SCHEMA_V2 = "verity-vllm/ru-index/v2"  # revision 2: compact derived slot families
RU_INDEX_SCHEMAS = (RU_INDEX_SCHEMA, RU_INDEX_SCHEMA_V2)
BOUNDARY_SCHEMA = "verity-vllm/boundary/v1"
# capture-format v1.1 (W10, state rows R3/R4): a replay unit whose leaves span prompt AND generated
# positions (a post-preemption recompute chunk) carries phase "mixed"; each leaf's ``ph`` is then
# the position-determined phase (prefill iff t < prompt_len).  Value ids never use "mixed".
MIXED_PHASE = "mixed"
RU_PHASES = tuple(identity.PHASES) + (MIXED_PHASE,)
WEIGHTS_INDEX_SCHEMA = "verity-vllm/weights-index/v1"
WEIGHTS_INDEX_SCHEMA_V2 = "verity-vllm/weights-index/v2"  # row-block weight leaves (W3c)
LEAVES_FILE = "leaves.bin"  # v2: 32-byte leaf hashes, entry by entry, block order
CAPTURE_SCHEMA = "verity-vllm/capture/v1"
MANIFEST_SCHEMA = "verity-vllm/profile/v1"

ORIGINS = ("captured", "derived")

RU_DIR = "ru"
WEIGHTS_DIR = "weights"
INDEX_FILE = "index.json"
VALUES_FILE = "values.bin"
BOUNDARY_FILE = "boundary.json"
MANIFEST_FILE = "manifest.json"
DESCRIPTION_FILE = "description.json"
REQUIRED_FILE = "required.json"
COMMITMENTS_FILE = "commitments.json"
CAPTURE_FILE = "capture.json"


class CaptureError(ValueError):
    pass


class StaleCaptureError(CaptureError):
    """The same canonical id was captured twice with different bytes (N05)."""


def step_dirname(step: int) -> str:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise CaptureError(f"step must be a non-negative int, got {step!r}")
    return f"{step:06d}"


def ru_tree_id(request_id: str, step: int) -> str:
    return f"ru/{request_id}/{step_dirname(step)}"


def boundary_tree_id(request_id: str) -> str:
    return f"boundary/{request_id}"


WEIGHTS_TREE_ID = "weights"


def _write_json(path: Path, document: Any, *, canonical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        data = hashing.canonical_json_bytes(document)
    else:
        data = (json.dumps(document, indent=1, sort_keys=False) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError as error:
        raise CaptureError(f"missing {path}") from error
    except json.JSONDecodeError as error:
        raise CaptureError(f"malformed JSON in {path}: {error}") from error


def as_bytes(data: Any) -> bytes:
    """Accept bytes-like or a numpy array; return an immutable ``bytes`` snapshot."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray | memoryview):
        return bytes(data)
    if isinstance(data, np.ndarray):
        return dtypes.array_to_bytes(data)
    raise CaptureError(
        f"value must be bytes-like or a numpy array, got {type(data).__name__}; for a torch "
        "tensor pass t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()"
    )


# --- records -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Entry:
    """One leaf of a replay unit as recorded in ``index.json``."""

    canonical_id: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    origin: str
    leaf: str  # hex of hashing.leaf_hash, computed by the writer at add() time
    # format revision 2 (capture-format.md section 11): a leaf whose bytes are prescribed, never
    # captured ("zeros" = all-zero bytes of dtype/shape). Such entries hold no bytes in values.bin
    # (offset == values_nbytes, nbytes = logical size) and are listed compactly in the index as
    # slot families; ``leaf`` may be "" and is computed on demand.
    derived: str | None = None

    def to_json(self) -> dict[str, Any]:
        doc = {
            "canonical_id": self.canonical_id,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset": self.offset,
            "nbytes": self.nbytes,
            "origin": self.origin,
            "leaf": self.leaf,
        }
        if self.derived is not None:
            doc["derived"] = self.derived
        return doc

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Entry:
        try:
            return cls(
                canonical_id=str(doc["canonical_id"]),
                dtype=dtypes.normalize_dtype(doc["dtype"]),
                shape=tuple(int(d) for d in doc["shape"]),
                offset=int(doc["offset"]),
                nbytes=int(doc["nbytes"]),
                origin=str(doc["origin"]),
                leaf=str(doc.get("leaf", "")),
                derived=doc.get("derived"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CaptureError(f"malformed index entry {doc!r}: {error}") from error

    def derived_bytes(self) -> bytes:
        if self.derived == "zeros":
            return bytes(self.nbytes)
        raise CaptureError(f"{self.canonical_id}: unknown derived fill {self.derived!r}")

    def leaf_bytes(self) -> bytes:
        """The leaf hash; computed for derived entries whose index record carries none."""
        if self.leaf:
            return bytes.fromhex(self.leaf)
        if self.derived is None:
            raise CaptureError(f"{self.canonical_id}: index entry has no leaf hash")
        return hashing.leaf_hash(self.canonical_id, self.dtype, self.shape, self.derived_bytes())


DERIVED_FILLS = ("zeros",)


def slot_family_id(canonical_id: str) -> str:
    """The family id of a per-slot value id: the id without its ``e=`` field."""
    cid = identity.parse(canonical_id)
    if cid.kind != "value" or cid.expert is None:
        raise CaptureError(f"{canonical_id}: not a per-expert-slot value id")
    return str(
        identity.value_id(
            cid.profile_id or "", cid.request_id or "", cid.t or 0, cid.phase or "", cid.layer or "",
            cid.op or "", cid.port or 0, shard=cid.shard, expert=None,
        )
    )  # fmt: skip


def slot_id(family_id: str, slot: int) -> str:
    """``family_id`` with ``e=<slot>`` appended (families never carry a ``c=`` coordinate)."""
    return f"{family_id}/e={slot}"


def expand_family(family: Mapping[str, Any], values_nbytes: int) -> list[Entry]:
    """The derived entries a compact slot-family record stands for (inactive slots only)."""
    try:
        fid = str(family["family"])
        slots = int(family["slots"])
        active = {int(e) for e in family["active"]}
        dtype = dtypes.normalize_dtype(family["dtype"])
        shape = tuple(int(d) for d in family["shape"])
        fill = str(family.get("fill", "zeros"))
    except (KeyError, TypeError, ValueError) as error:
        raise CaptureError(f"malformed slot family {family!r}: {error}") from error
    if fill not in DERIVED_FILLS:
        raise CaptureError(f"slot family {fid}: unknown fill {fill!r}")
    if any(e < 0 or e >= slots for e in active):
        raise CaptureError(f"slot family {fid}: active slot out of range")
    nbytes = dtypes.expected_nbytes(dtype, shape)
    if nbytes is None:
        raise CaptureError(f"slot family {fid}: dtype {dtype} has no fixed size")
    return [
        Entry(slot_id(fid, e), dtype, shape, values_nbytes, nbytes, "derived", "", fill)
        for e in range(slots)
        if e not in active
    ]


@dataclass(frozen=True, slots=True)
class Value:
    """A leaf with its bytes: what openings carry and what checkers receive."""

    canonical_id: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes
    origin: str = "captured"

    @property
    def nbytes(self) -> int:
        return len(self.data)

    def leaf(self) -> bytes:
        return hashing.leaf_hash(self.canonical_id, self.dtype, self.shape, self.data)

    def array(self) -> np.ndarray:
        return dtypes.bytes_to_array(self.dtype, self.shape, self.data)

    def float32(self) -> np.ndarray:
        return dtypes.to_float32(self.dtype, self.shape, self.data)

    def json(self) -> Any:
        if self.dtype != "json":
            raise CaptureError(f"{self.canonical_id} is {self.dtype}, not json")
        return json.loads(self.data.decode("utf-8"))

    def text(self) -> str:
        if self.dtype != "utf8":
            raise CaptureError(f"{self.canonical_id} is {self.dtype}, not utf8")
        return self.data.decode("utf-8")


def make_value(
    canonical_id: str, dtype: str, shape: Sequence[int], data: Any, origin: str = "captured"
) -> Value:
    """Validate and build a :class:`Value` (id canonical, dtype known, nbytes consistent)."""
    identity.parse(canonical_id)
    dtype = dtypes.normalize_dtype(dtype)
    shape_t = tuple(int(d) for d in shape)
    raw = as_bytes(data)
    if dtype in dtypes.VARIABLE_DTYPES and shape_t != ():
        raise CaptureError(f"{dtype} leaves must have shape [], got {list(shape_t)}")
    dtypes.check_nbytes(dtype, shape_t, len(raw))
    if origin not in ORIGINS:
        raise CaptureError(f"origin must be one of {ORIGINS}, got {origin!r}")
    return Value(canonical_id, dtype, shape_t, raw, origin)


def json_value(canonical_id: str, document: Any) -> Value:
    return Value(canonical_id, "json", (), hashing.canonical_json_bytes(document), "captured")


def utf8_value(canonical_id: str, text: str) -> Value:
    return Value(canonical_id, "utf8", (), text.encode("utf-8"), "captured")


def scalar_value(canonical_id: str, dtype: str, number: int | float) -> Value:
    arr = np.array(number, dtype=dtypes.view_dtype(dtype))
    return Value(canonical_id, dtypes.normalize_dtype(dtype), (), dtypes.array_to_bytes(arr))


# --- replay unit index ---------------------------------------------------------------------------


@dataclass(slots=True)
class RUIndex:
    request_id: str
    step: int
    phase: str
    positions: tuple[int, ...]
    entries: list[Entry]
    profile_id: str | None = None
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    tainted: bool = False
    values_nbytes: int = 0
    # revision 2: compact records of derived slot families; ``entries`` holds their expansion
    # (one derived Entry per inactive slot) after the captured entries
    families: list[dict[str, Any]] = field(default_factory=list)
    # speculative extension (lane SD): a speculative-decoding step
    # commits ``positions`` (the accepted slots) and additionally computed values at ``dead``
    # positions it did not commit — the rejected verification slots' target values and the draft
    # model's proposal rows.  Those leaves carry the ``x=<step>`` qualifier of THIS step and are the
    # only ids allowed at a dead position.  ``None`` for a plain step (format unchanged).
    speculative: dict[str, Any] | None = None

    @property
    def tree_id(self) -> str:
        return ru_tree_id(self.request_id, self.step)

    def dead_positions(self) -> tuple[int, ...]:
        return tuple(int(t) for t in (self.speculative or {}).get("dead", ()))

    def captured_entries(self) -> list[Entry]:
        return [e for e in self.entries if e.derived is None]

    def derived_entries(self) -> list[Entry]:
        return [e for e in self.entries if e.derived is not None]

    def to_json(self) -> dict[str, Any]:
        doc = {
            "schema": RU_INDEX_SCHEMA_V2 if self.families else RU_INDEX_SCHEMA,
            "request_id": self.request_id,
            "step": self.step,
            "phase": self.phase,
            "positions": list(self.positions),
            "profile_id": self.profile_id,
            "values_nbytes": self.values_nbytes,
            "tainted": self.tainted,
            "conflicts": self.conflicts,
            "entries": [e.to_json() for e in self.entries if e.derived is None],
        }
        if self.families:
            doc["families"] = [dict(f) for f in self.families]
        if self.speculative is not None:
            doc["speculative"] = dict(self.speculative)
        return doc

    @classmethod
    def from_json(
        cls, doc: Any, *, request_id: str | None = None, step: int | None = None
    ) -> RUIndex:
        if isinstance(doc, list):  # bare-list form: metadata comes from the path
            if request_id is None or step is None:
                raise CaptureError("bare-list index.json needs request_id and step from the path")
            entries = [Entry.from_json(e) for e in doc]
            phase = _infer_phase(entries)
            positions = tuple(sorted({identity.parse(e.canonical_id).t for e in entries}))
            return cls(request_id, step, phase, positions, entries)
        if not isinstance(doc, dict):
            raise CaptureError("index.json must be an object or a list")
        if doc.get("schema", RU_INDEX_SCHEMA) not in RU_INDEX_SCHEMAS:
            raise CaptureError(f"unexpected index schema {doc.get('schema')!r}")
        try:
            values_nbytes = int(doc.get("values_nbytes", 0))
            families = list(doc.get("families", []))
            entries = [Entry.from_json(e) for e in doc["entries"]]
            if not families and not values_nbytes:
                values_nbytes = max((e.offset + e.nbytes for e in entries), default=0)
            for family in families:
                entries.extend(expand_family(family, values_nbytes))
            index = cls(
                request_id=str(doc["request_id"]),
                step=int(doc["step"]),
                phase=str(doc["phase"]),
                positions=tuple(int(t) for t in doc["positions"]),
                entries=entries,
                profile_id=doc.get("profile_id"),
                conflicts=list(doc.get("conflicts", [])),
                tainted=bool(doc.get("tainted", False)),
                values_nbytes=values_nbytes,
                families=families,
                speculative=dict(doc["speculative"]) if doc.get("speculative") is not None else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CaptureError(f"malformed index.json: {error}") from error
        if request_id is not None and index.request_id != request_id:
            raise CaptureError(f"index request_id {index.request_id!r} != path {request_id!r}")
        if step is not None and index.step != step:
            raise CaptureError(f"index step {index.step} != path step {step}")
        return index

    def ids(self) -> list[str]:
        return identity.tree_sorted([e.canonical_id for e in self.entries])

    def validate(self) -> list[str]:
        """Structural problems (empty list when clean); see also RUReader.validate for bytes."""
        problems: list[str] = []
        if self.phase not in RU_PHASES:
            problems.append(f"phase {self.phase!r} not in {RU_PHASES}")
        seen: set[str] = set()
        cursor = 0
        for entry in self.entries:
            try:
                cid = identity.parse(entry.canonical_id)
            except identity.IdentityError as error:
                problems.append(str(error))
                continue
            if cid.kind != "value":
                problems.append(f"{entry.canonical_id}: RU leaves must be value ids")
                continue
            if cid.request_id != self.request_id:
                problems.append(f"{entry.canonical_id}: r != {self.request_id}")
            if self.phase != MIXED_PHASE and cid.phase != self.phase:
                problems.append(f"{entry.canonical_id}: ph != {self.phase}")
            if cid.speculative is None:
                if self.positions and cid.t not in self.positions:
                    problems.append(f"{entry.canonical_id}: t not in the unit's positions")
            else:
                # speculative extension: an uncommitted value of THIS step at a committed or a dead position
                if cid.speculative != self.step:
                    problems.append(f"{entry.canonical_id}: x != the unit's step {self.step}")
                if cid.t not in self.positions and cid.t not in self.dead_positions():
                    problems.append(f"{entry.canonical_id}: t neither a committed nor a dead position of the unit")
            if self.profile_id and cid.profile_id != self.profile_id:
                problems.append(f"{entry.canonical_id}: p != {self.profile_id}")
            if entry.canonical_id in seen:
                problems.append(f"{entry.canonical_id}: duplicate id in index")
            seen.add(entry.canonical_id)
            if entry.origin not in ORIGINS:
                problems.append(f"{entry.canonical_id}: origin {entry.origin!r}")
            try:
                dtypes.check_nbytes(entry.dtype, entry.shape, entry.nbytes)
            except dtypes.DtypeError as error:
                problems.append(f"{entry.canonical_id}: {error}")
            if entry.derived is not None:
                if entry.derived not in DERIVED_FILLS:
                    problems.append(f"{entry.canonical_id}: unknown derived fill {entry.derived!r}")
                if entry.origin != "derived":
                    problems.append(f"{entry.canonical_id}: derived entry must have origin derived")
                if cid.expert is None:
                    problems.append(f"{entry.canonical_id}: derived entries are expert slots (e=)")
                continue  # holds no bytes in values.bin
            if entry.offset != cursor:
                problems.append(f"{entry.canonical_id}: offset {entry.offset} != {cursor}")
            cursor = entry.offset + entry.nbytes
        for family in self.families:
            fid = str(family.get("family", ""))
            for e in family.get("active", []):
                if slot_id(fid, int(e)) not in seen:
                    problems.append(f"{fid}: active slot {e} has no captured entry")
        if self.values_nbytes and cursor != self.values_nbytes:
            problems.append(
                f"entries cover {cursor} bytes, values_nbytes says {self.values_nbytes}"
            )
        if self.tainted:
            problems.append("unit is tainted (duplicate captures with differing bytes)")
        return problems


def _infer_phase(entries: list[Entry]) -> str:
    phases = {identity.parse(e.canonical_id).phase for e in entries}
    if len(phases) != 1:
        raise CaptureError(f"cannot infer a single phase from {sorted(phases)}")
    return phases.pop()


# --- writer --------------------------------------------------------------------------------------


class CaptureWriter:
    """Writes one replay unit: ``add`` snapshots bytes immediately, ``finish`` writes the index."""

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        request_id: str,
        step: int,
        *,
        phase: str,
        positions: Sequence[int],
        profile_id: str | None = None,
        on_duplicate: str = "error",
        speculative: Mapping[str, Any] | None = None,
    ) -> None:
        if phase not in RU_PHASES:
            raise CaptureError(f"phase must be one of {RU_PHASES}, got {phase!r}")
        if on_duplicate not in ("error", "flag"):
            raise CaptureError("on_duplicate must be 'error' or 'flag'")
        identity._validate_request(request_id)
        self.run_dir = Path(run_dir)
        self.request_id = request_id
        self.step = step
        self.phase = phase
        self.positions = tuple(int(t) for t in positions)
        self.profile_id = profile_id
        self.on_duplicate = on_duplicate
        # speculative extension: the step's record (k, accepted, candidates, verdicts, dead positions)
        self.speculative = dict(speculative) if speculative is not None else None
        self.dead_positions = tuple(int(t) for t in (self.speculative or {}).get("dead", ()))
        if set(self.dead_positions) & set(self.positions):
            raise CaptureError("a dead position cannot also be a committed position of the unit")
        self.dir = self.run_dir / RU_DIR / request_id / step_dirname(step)
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / INDEX_FILE).exists():
            raise CaptureError(f"{self.dir} already holds a finished replay unit")
        self._entries: list[Entry] = []
        self._by_id: dict[str, Entry] = {}
        self._conflicts: list[dict[str, Any]] = []
        self._families: list[dict[str, Any]] = []
        self._offset = 0
        self._file: BinaryIO | None = open(self.dir / VALUES_FILE, "wb")  # noqa: SIM115
        self._finished = False

    def __enter__(self) -> CaptureWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        if exc[0] is None and not self._finished:
            self.finish()
        elif self._file is not None:
            self._file.close()
            self._file = None

    @property
    def tree_id(self) -> str:
        return ru_tree_id(self.request_id, self.step)

    def add(
        self,
        canonical_id: str,
        dtype: str,
        shape: Sequence[int],
        data: Any,
        origin: str = "captured",
    ) -> Entry:
        if self._finished or self._file is None:
            raise CaptureError("writer is finished")
        value = make_value(canonical_id, dtype, shape, data, origin)
        cid = identity.parse(canonical_id)
        if cid.kind != "value":
            raise CaptureError(f"{canonical_id}: replay units hold value ids only")
        if cid.request_id != self.request_id:
            raise CaptureError(f"{canonical_id}: r != {self.request_id}")
        # an uncommitted (``x=``) value's phase is determined by its position like every other value's:
        # a prefill unit's dead draft-proposal rows sit at decode positions
        if self.phase != MIXED_PHASE and cid.phase != self.phase and cid.speculative is None:
            raise CaptureError(f"{canonical_id}: ph != {self.phase}")
        if cid.speculative is None:
            if cid.t not in self.positions:
                raise CaptureError(f"{canonical_id}: t={cid.t} not in this unit's positions")
        else:
            if cid.speculative != self.step:
                raise CaptureError(f"{canonical_id}: x={cid.speculative} names another step than this unit's {self.step}")
            if cid.t not in self.positions and cid.t not in self.dead_positions:
                raise CaptureError(f"{canonical_id}: t={cid.t} is neither a committed nor a dead position of this unit")
        if self.profile_id is not None and cid.profile_id != self.profile_id:
            raise CaptureError(f"{canonical_id}: p != {self.profile_id}")
        leaf_hex = value.leaf().hex()
        previous = self._by_id.get(canonical_id)
        if previous is not None:
            same = previous.leaf == leaf_hex
            detail = {
                "canonical_id": canonical_id,
                "first_leaf": previous.leaf,
                "second_leaf": leaf_hex,
                "identical": same,
                "second_sha256": hashing.sha256_hex(value.data),
            }
            if self.on_duplicate == "error":
                raise StaleCaptureError(
                    f"{canonical_id} captured twice ("
                    f"{'identical' if same else 'DIFFERENT'} bytes); a second capture of one "
                    "identity is a stale or overwritten buffer and is never committed silently"
                )
            self._conflicts.append(detail)
            return previous
        entry = Entry(
            canonical_id=canonical_id,
            dtype=value.dtype,
            shape=value.shape,
            offset=self._offset,
            nbytes=value.nbytes,
            origin=origin,
            leaf=leaf_hex,
        )
        self._file.write(value.data)
        self._offset += value.nbytes
        self._entries.append(entry)
        self._by_id[canonical_id] = entry
        return entry

    def add_derived_family(
        self,
        family_id: str,
        *,
        slots: int,
        active: Sequence[int],
        dtype: str,
        shape: Sequence[int],
        fill: str = "zeros",
    ) -> dict[str, Any]:
        """Revision 2: declare the ``slots`` expert slots of one (position, op, port); the slots in
        ``active`` must be (or become) captured entries ``<family_id>/e=<slot>``; every other slot is
        a derived leaf with the prescribed ``fill`` bytes, never captured, holding no bytes in
        values.bin. The population is fixed by the description (layers x positions x slots); the
        family record only says which slots the run captured."""
        if self._finished:
            raise CaptureError("writer is finished")
        cid = identity.parse(family_id)
        if cid.kind != "value" or cid.expert is not None or cid.coordinate is not None:
            raise CaptureError(f"{family_id}: a slot family id is a value id without e=/c=")
        if (
            cid.request_id != self.request_id
            or (self.phase != MIXED_PHASE and cid.phase != self.phase)
            or cid.t not in self.positions
        ):
            raise CaptureError(f"{family_id}: not in this replay unit")
        if fill not in DERIVED_FILLS:
            raise CaptureError(f"unknown derived fill {fill!r}")
        active_l = sorted({int(e) for e in active})
        if any(e < 0 or e >= slots for e in active_l):
            raise CaptureError(f"{family_id}: active slot out of range")
        family = {
            "family": family_id,
            "slot_field": "e",
            "slots": int(slots),
            "active": active_l,
            "dtype": dtypes.normalize_dtype(dtype),
            "shape": [int(d) for d in shape],
            "fill": fill,
        }
        for entry in expand_family(family, 0):  # offsets are patched at finish()
            if entry.canonical_id in self._by_id:
                raise StaleCaptureError(
                    f"{entry.canonical_id}: captured as a value and declared as a derived slot"
                )
            self._entries.append(entry)
            self._by_id[entry.canonical_id] = entry
        self._families.append(family)
        return family

    def finish(self) -> RUIndex:
        if self._finished:
            raise CaptureError("writer already finished")
        assert self._file is not None
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None
        self._finished = True
        captured = [e for e in self._entries if e.derived is None]
        for family in self._families:
            for e in family["active"]:
                if slot_id(family["family"], e) not in self._by_id:
                    raise CaptureError(f"{family['family']}: active slot {e} was never captured")
        entries = captured + [
            e for family in self._families for e in expand_family(family, self._offset)
        ]
        index = RUIndex(
            request_id=self.request_id,
            step=self.step,
            phase=self.phase,
            positions=self.positions,
            entries=entries,
            profile_id=self.profile_id,
            conflicts=list(self._conflicts),
            tainted=any(not c["identical"] for c in self._conflicts),
            values_nbytes=self._offset,
            families=list(self._families),
            speculative=self.speculative,
        )
        _write_json(self.dir / INDEX_FILE, index.to_json())
        return index


# --- boundary ------------------------------------------------------------------------------------

# The chosen logprob's leaf encoding (WB2, audit-security-baseline finding 1): the API reports each
# chosen token's logprob from a float32 tensor; the leaf is that value's IEEE-754 binary32 word,
# little-endian (every tensor leaf of the format is little-endian), dtype string ``float32``, shape
# ``[]``.  The record's JSON double must *be* a binary32 value: a rounded or re-typed logprob is not
# the one the API reported and is refused by ``Boundary.validate`` -- it is never rounded to fit.
LOGPROB_DTYPE = "float32"


def binary32_exact(value: object) -> bool:
    """True when ``value`` is a float that round-trips through IEEE-754 binary32 unchanged (``-inf``
    included, ``nan`` and out-of-range excluded)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    x = float(value)
    if x != x:  # nan
        return False
    try:
        packed = struct.pack("<f", x)
    except OverflowError:
        return False
    return struct.unpack("<f", packed)[0] == x


@dataclass(frozen=True, slots=True)
class OutputRecord:
    """One protected output slot, indexed by the position ``t`` whose sampler produced it."""

    t: int
    present: bool
    token_id: int | None
    step: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {"t": self.t, "present": self.present, "token_id": self.token_id, "step": self.step}


@dataclass(frozen=True, slots=True)
class StepRecord:
    step: int
    phase: str
    positions: tuple[int, ...]
    # capture-format v1.1 (W10, R4): a step scheduled after a preemption recomputes positions the
    # request had already computed (and committed) in earlier steps.  Its leaves carry the SAME
    # canonical ids as the originals (identity is a function of (request, position), never of
    # the scheduler's decisions); the harness requires the recomputed bytes to equal the originals.
    recompute: bool = False
    # speculative extension (lane SD): a speculative-decoding step
    # verifies k candidates in k+1 slots and commits the accepted prefix — ``positions`` are the
    # committed slots (``accepted + 1`` of them, fewer only when the request terminates inside the
    # block); the record names ``k``, ``accepted``, the ``candidates`` verified, the per-slot
    # ``verdicts`` and the ``dead`` positions whose values the step computed and discarded.
    speculative: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"step": self.step, "phase": self.phase, "positions": list(self.positions)}
        if self.recompute:
            d["recompute"] = True
        if self.speculative is not None:
            d["speculative"] = dict(self.speculative)
        return d

    @classmethod
    def from_json(cls, s: Mapping[str, Any]) -> StepRecord:
        return cls(int(s["step"]), str(s["phase"]), tuple(int(t) for t in s["positions"]), bool(s.get("recompute", False)),
                   dict(s["speculative"]) if s.get("speculative") is not None else None)


@dataclass(frozen=True, slots=True)
class CachedPrefix:
    """capture-format v1.1 (W10, R3): positions of this request that were never computed in its
    own steps because the runtime served them from the prefix cache.  Their K/V identities are the
    *source* request's canonical ids at the same positions (``r=<source>/t=<pos>``); the binding
    condition, checked from both boundary trees, is that the two requests' token sequences agree
    on every position ``<= max(positions)``."""

    source_request_id: str
    source_prompt_len: int
    positions: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return {"source_request_id": self.source_request_id, "source_prompt_len": self.source_prompt_len, "positions": list(self.positions)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> CachedPrefix:
        return cls(str(d["source_request_id"]), int(d["source_prompt_len"]), tuple(int(t) for t in d["positions"]))


def phase_of_position(t: int, prompt_len: int) -> str:
    """The position-determined phase of a value id: ``prefill`` for prompt positions, ``decode``
    for generated ones.  Equal to the scheduling phase on every non-preempted trajectory."""
    return "prefill" if t < prompt_len else "decode"


@dataclass(slots=True)
class Boundary:
    """Per-request boundary: prompt, outputs, sampling params, seed, finish, presence, steps, and
    (since WB2) the chosen logprobs.

    ``chosen_logprobs`` is ``None`` for a boundary that does not bind them (every boundary written
    before the field existed): its tree has no ``output_logprob`` leaves and verifies exactly as
    before, and the row says ``logprobs: unbound``.  A list (one float per produced output token,
    each a binary32 value, see :data:`LOGPROB_DTYPE`) adds one leaf per present record.  The two
    trees have different ``ids_digest``s, so a bundle cannot be silently upgraded or downgraded.
    """

    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    sampling_params: dict[str, Any]
    seed: int | None
    finish_reason: str | None
    output_records: list[OutputRecord]
    steps: list[StepRecord]
    chosen_logprobs: list[float | None] | None = None
    cached_prefix: list[CachedPrefix] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        request_id: str,
        prompt_token_ids: Sequence[int],
        output_token_ids: Sequence[int],
        sampling_params: Mapping[str, Any],
        *,
        seed: int | None,
        finish_reason: str | None,
        steps: Sequence[StepRecord | Mapping[str, Any]],
        max_tokens: int | None = None,
        chosen_logprobs: Sequence[float | None] | None = None,
        cached_prefix: Sequence[CachedPrefix | Mapping[str, Any]] = (),
    ) -> Boundary:
        """Derive the presence records: ``max_tokens`` slots (default: the outputs produced).

        ``chosen_logprobs`` (one per produced output token, as the API reported them) makes the
        boundary bind them; values are kept as given and judged by :meth:`validate` (a value that is
        not exactly a binary32 number, or ``None``, is a structural problem, never rounded)."""
        prompt = [int(x) for x in prompt_token_ids]
        outputs = [int(x) for x in output_token_ids]
        if max_tokens is None:
            max_tokens = sampling_params.get("max_tokens")
        if max_tokens is None:
            max_tokens = len(outputs)
        step_records = [s if isinstance(s, StepRecord) else StepRecord.from_json(s) for s in steps]
        # the step that first computed a position (a recompute step never claims an output record)
        pos_to_step: dict[int, int] = {}
        for s in step_records:
            for t in s.positions:
                pos_to_step.setdefault(t, s.step)
        records = []
        for k in range(int(max_tokens)):
            t = len(prompt) - 1 + k
            present = k < len(outputs)
            records.append(
                OutputRecord(
                    t=t,
                    present=present,
                    token_id=outputs[k] if present else None,
                    step=pos_to_step.get(t),
                )
            )
        return cls(
            request_id=request_id,
            prompt_token_ids=prompt,
            output_token_ids=outputs,
            sampling_params=dict(sampling_params),
            seed=seed,
            finish_reason=finish_reason,
            output_records=records,
            steps=step_records,
            chosen_logprobs=None if chosen_logprobs is None else [
                None if x is None else (float(x) if binary32_exact(x) else x) for x in chosen_logprobs
            ],
            cached_prefix=[c if isinstance(c, CachedPrefix) else CachedPrefix.from_json(c) for c in cached_prefix],
        )

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def binds_logprobs(self) -> bool:
        """Whether this boundary's tree carries ``output_logprob`` leaves."""
        return self.chosen_logprobs is not None

    def logprob_at(self, t: int) -> float | None:
        """The bound logprob of the token the sampler produced at position ``t`` (``None`` when the
        boundary does not bind logprobs or ``t`` produced no token)."""
        if self.chosen_logprobs is None:
            return None
        k = t - self.prompt_len + 1
        if k < 0 or k >= len(self.chosen_logprobs):
            return None
        return self.chosen_logprobs[k]

    def phase_at(self, t: int) -> str:
        return phase_of_position(t, self.prompt_len)

    def cached_positions(self) -> set[int]:
        return {t for c in self.cached_prefix for t in c.positions}

    def cached_source(self, t: int) -> CachedPrefix | None:
        """The cached-prefix entry serving position ``t`` (None when the request computed it)."""
        for c in self.cached_prefix:
            if t in c.positions:
                return c
        return None

    def token_at(self, t: int) -> int | None:
        """The input token at position ``t`` (prompt, then outputs shifted by one)."""
        if t < 0:
            return None
        if t < self.prompt_len:
            return self.prompt_token_ids[t]
        k = t - self.prompt_len
        return self.output_token_ids[k] if k < len(self.output_token_ids) else None

    def sampled_token_at(self, t: int) -> int | None:
        """The token the sampler produced at position ``t`` (``output[t - prompt_len + 1]``)."""
        k = t - self.prompt_len + 1
        if k < 0 or k >= len(self.output_token_ids):
            return None
        return self.output_token_ids[k]

    def step_of(self, t: int) -> StepRecord | None:
        for s in self.steps:
            if t in s.positions:
                return s
        return None

    def sampled_positions(self) -> list[int]:
        return [r.t for r in self.output_records if r.present]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": BOUNDARY_SCHEMA,
            "request_id": self.request_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "output_token_ids": list(self.output_token_ids),
            "sampling_params": dict(self.sampling_params),
            "seed": self.seed,
            "finish_reason": self.finish_reason,
            "output_records": [r.to_json() for r in self.output_records],
            "steps": [s.to_json() for s in self.steps],
            # written only when bound, so a boundary.json that predates the field is unchanged
            **({"chosen_logprobs": list(self.chosen_logprobs)} if self.chosen_logprobs is not None else {}),
            **({"cached_prefix": [c.to_json() for c in self.cached_prefix]} if self.cached_prefix else {}),
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Boundary:
        if doc.get("schema", BOUNDARY_SCHEMA) != BOUNDARY_SCHEMA:
            raise CaptureError(f"unexpected boundary schema {doc.get('schema')!r}")
        lps = doc.get("chosen_logprobs")
        if lps is not None and not isinstance(lps, list):
            raise CaptureError("malformed boundary.json: chosen_logprobs must be a list or absent")
        try:
            return cls(
                request_id=str(doc["request_id"]),
                prompt_token_ids=[int(x) for x in doc["prompt_token_ids"]],
                output_token_ids=[int(x) for x in doc["output_token_ids"]],
                sampling_params=dict(doc["sampling_params"]),
                seed=None if doc.get("seed") is None else int(doc["seed"]),
                finish_reason=doc.get("finish_reason"),
                output_records=[
                    OutputRecord(
                        t=int(r["t"]),
                        present=bool(r["present"]),
                        token_id=None if r.get("token_id") is None else int(r["token_id"]),
                        step=None if r.get("step") is None else int(r["step"]),
                    )
                    for r in doc["output_records"]
                ],
                steps=[StepRecord.from_json(s) for s in doc.get("steps", [])],
                chosen_logprobs=None if lps is None else [
                    None if x is None else (float(x) if binary32_exact(x) else x) for x in lps
                ],
                cached_prefix=[CachedPrefix.from_json(c) for c in doc.get("cached_prefix", [])],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CaptureError(f"malformed boundary.json: {error}") from error

    def validate(self) -> list[str]:
        """Structural invariants of the protected output records (N14's first line of defence)."""
        problems: list[str] = []
        P = self.prompt_len
        if P == 0:
            problems.append("empty prompt")
        max_tokens = self.sampling_params.get("max_tokens")
        if max_tokens is not None and len(self.output_token_ids) > int(max_tokens):
            problems.append("more outputs than max_tokens")
        if max_tokens is not None and len(self.output_records) != int(max_tokens):
            problems.append(f"{len(self.output_records)} records for max_tokens={max_tokens}")
        if self.finish_reason == "length" and max_tokens is not None:
            if len(self.output_token_ids) != int(max_tokens):
                problems.append("finish_reason=length but outputs != max_tokens")
        expected_t = [P - 1 + k for k in range(len(self.output_records))]
        if [r.t for r in self.output_records] != expected_t:
            problems.append("output records are not the contiguous positions prompt_len-1 ...")
        present = [r.present for r in self.output_records]
        n_present = sum(present)
        if present != [True] * n_present + [False] * (len(present) - n_present):
            problems.append("presence is not a prefix (a present slot after a blank one)")
        if n_present != len(self.output_token_ids):
            problems.append("present records != outputs produced")
        for k, r in enumerate(self.output_records):
            if r.present and (
                k >= len(self.output_token_ids) or r.token_id != self.output_token_ids[k]
            ):
                problems.append(f"record t={r.t} token_id disagrees with output_token_ids[{k}]")
            if not r.present and r.token_id is not None:
                problems.append(f"blank record t={r.t} carries a token")
        if self.chosen_logprobs is not None:
            if len(self.chosen_logprobs) != len(self.output_token_ids):
                problems.append(
                    f"{len(self.chosen_logprobs)} chosen logprobs for {len(self.output_token_ids)} outputs"
                )
            for k, lp in enumerate(self.chosen_logprobs):
                if lp is None:
                    problems.append(f"chosen_logprobs[{k}] is null (the API reported no logprob for the chosen token)")
                elif not binary32_exact(lp):
                    problems.append(
                        f"chosen_logprobs[{k}] = {lp!r} is not exactly a binary32 value (a rounded or re-typed "
                        f"logprob is not the one the API reported; the boundary never rounds)"
                    )
        cached = sorted(self.cached_positions())
        if len(cached) != sum(len(c.positions) for c in self.cached_prefix):
            problems.append("cached_prefix entries overlap")
        for c in self.cached_prefix:
            if c.source_request_id == self.request_id:
                problems.append("cached_prefix names the request itself as its source")
            if list(c.positions) != sorted(set(c.positions)):
                problems.append("cached_prefix positions not sorted/unique")
        if self.steps:
            covered: list[int] = []
            recomputed: list[int] = []
            for s in self.steps:
                if s.phase not in RU_PHASES or (s.phase == MIXED_PHASE and not s.recompute):
                    problems.append(f"step {s.step}: phase {s.phase!r}")
                if s.recompute:
                    # a post-preemption chunk: the recomputed positions are a prefix of the
                    # chunk, already covered by earlier steps; the rest is new
                    seen = set(covered)
                    again = [t for t in s.positions if t in seen]
                    if again != list(s.positions[: len(again)]):
                        problems.append(f"step {s.step}: recomputed positions are not the chunk's prefix")
                    if any(t in cached for t in s.positions):
                        problems.append(f"step {s.step}: recompute of a cached-prefix position")
                    recomputed.extend(again)
                    covered.extend(t for t in s.positions if t not in seen)
                    continue
                if s.speculative is not None:
                    # speculative extension: the committed slots are a contiguous prefix of the block
                    sp = s.speculative
                    k, a = int(sp.get("k", 0)), int(sp.get("accepted", 0))
                    if not (0 <= a <= k):
                        problems.append(f"step {s.step}: accepted {a} outside 0..k={k}")
                    # a verify step (k > 0) commits the accepted prefix plus the bonus/recovered slot;
                    # a prefill step (k = 0) carries the record only for its dead (draft proposal) positions
                    if k > 0 and (len(s.positions) > a + 1 or len(s.positions) < 1):
                        problems.append(f"step {s.step}: {len(s.positions)} committed positions for accepted={a} (at most accepted+1)")
                    if s.positions and list(s.positions) != list(range(s.positions[0], s.positions[0] + len(s.positions))):
                        problems.append(f"step {s.step}: committed positions are not contiguous")
                    if any(t in s.positions for t in sp.get("dead", ())):
                        problems.append(f"step {s.step}: a dead position is also committed")
                elif s.phase == "decode" and len(s.positions) != 1:
                    problems.append(f"step {s.step}: a decode step has one position")
                if s.phase == "prefill" and any(t >= P for t in s.positions):
                    problems.append(f"step {s.step}: prefill position beyond the prompt")
                covered.extend(s.positions)
            processed = P + max(len(self.output_token_ids) - 1, 0)
            # cached-prefix positions come first and are never computed by this request
            if cached and (cached != list(range(len(cached))) or (covered and cached[-1] >= min(covered))):
                problems.append("cached_prefix positions are not the leading prefix of the sequence")
            if sorted(cached + covered) != list(range(processed)) or covered != sorted(covered):
                problems.append(
                    "steps do not cover positions 0..prompt_len+outputs-2 once, in order"
                    + (" (cached_prefix excluded)" if cached else "")
                )
        return problems

    def leaf_ids(self, profile_id: str, *, logprobs: bool | None = None) -> list[str]:
        """The canonical ids of the boundary tree's leaves, in :meth:`leaves` order; ``logprobs``
        overrides whether ``output_logprob`` ids are included (default: whether this boundary binds
        them), so a verifier can tell which leaf set a committed ``ids_digest`` describes."""
        rid = self.request_id
        with_lp = self.binds_logprobs if logprobs is None else logprobs
        ids = [str(identity.boundary_id(profile_id, rid, "prompt_token", t)) for t in range(len(self.prompt_token_ids))]
        for record in self.output_records:
            ids.append(str(identity.boundary_id(profile_id, rid, "output_presence", record.t)))
            if record.present:
                ids.append(str(identity.boundary_id(profile_id, rid, "output_token", record.t)))
                if with_lp:
                    ids.append(str(identity.boundary_id(profile_id, rid, "output_logprob", record.t)))
        ids += [str(identity.boundary_id(profile_id, rid, f)) for f in ("sampling_params", "seed", "finish_reason")]
        return ids

    def cached_prefix_problems(self, source_of: Callable[[str], Boundary | None]) -> list[str]:
        """Cross-request invariants of a prefix-cache binding (capture-format v1.1, R3): the source
        exists, computed every bound position in its own steps, and both requests fed the same
        tokens on ``0..t`` for every bound ``t`` -- the condition under which K/V at ``t`` is one
        value regardless of which request's forward produced it. Both token sequences are
        committed boundary leaves, so a verifier checks this from the two boundary trees."""
        problems: list[str] = []
        for c in self.cached_prefix:
            src = source_of(c.source_request_id)
            if src is None:
                problems.append(f"cached_prefix source {c.source_request_id} has no boundary")
                continue
            if src.prompt_len != c.source_prompt_len:
                problems.append(f"cached_prefix source {c.source_request_id}: prompt_len {src.prompt_len} != {c.source_prompt_len}")
            top = max(c.positions) if c.positions else -1
            for t in range(top + 1):
                if self.token_at(t) != src.token_at(t):
                    problems.append(f"cached_prefix source {c.source_request_id}: tokens differ at t={t} <= bound position {top}")
                    break
            for t in c.positions:
                if src.step_of(t) is None:
                    problems.append(f"cached_prefix source {c.source_request_id}: no step computed t={t}")
                    break
        return problems

    def covered_positions(self) -> list[int]:
        """The positions this request's own steps computed, in step order, each once: a recompute
        step's repeated positions (capture-format v1.1, R4) are dropped, every position of a
        non-recompute step is kept (a repeat there is the coverage error :meth:`validate` reports).
        ``sorted(cached_positions()) + covered_positions()`` must be ``0..prompt_len+outputs-2``."""
        covered: list[int] = []
        for s in self.steps:
            if s.recompute:
                seen = set(covered)
                covered.extend(t for t in s.positions if t not in seen)
            else:
                covered.extend(s.positions)
        return covered

    def recomputed_positions(self) -> dict[int, list[int]]:
        """step -> positions that step recomputed (empty for non-preempted trajectories)."""
        out: dict[int, list[int]] = {}
        seen: set[int] = set()
        for s in self.steps:
            again = [t for t in s.positions if t in seen]
            if s.recompute and again:
                out[s.step] = again
            seen.update(s.positions)
        return out

    def leaves(self, profile_id: str) -> list[Value]:
        """The boundary tree's leaves (contract section 5: prompt ids, output ids, params, seed,
        and -- when bound -- the chosen logprobs).

        Token ids are ``int32`` scalars; presence is a ``uint8`` scalar per record slot; a chosen
        logprob is a ``float32`` scalar (:data:`LOGPROB_DTYPE`: the binary32 word the API reported,
        little-endian) per present record, only when ``chosen_logprobs`` is set; ``sampling_params``
        and ``seed`` are canonical JSON; ``finish_reason`` is UTF-8 (``""`` while unfinished).
        """
        rid = self.request_id
        out: list[Value] = []
        for t, tok in enumerate(self.prompt_token_ids):
            cid = str(identity.boundary_id(profile_id, rid, "prompt_token", t))
            out.append(scalar_value(cid, "int32", tok))
        for k, record in enumerate(self.output_records):
            cid = str(identity.boundary_id(profile_id, rid, "output_presence", record.t))
            out.append(scalar_value(cid, "uint8", 1 if record.present else 0))
            if record.present:
                cid = str(identity.boundary_id(profile_id, rid, "output_token", record.t))
                out.append(scalar_value(cid, "int32", int(record.token_id)))
                if self.chosen_logprobs is not None:
                    lp = self.chosen_logprobs[k] if k < len(self.chosen_logprobs) else None
                    if lp is None or not binary32_exact(lp):
                        raise CaptureError(
                            f"{rid}: chosen_logprobs[{k}] = {lp!r} cannot be a float32 leaf (validate() first)"
                        )
                    cid = str(identity.boundary_id(profile_id, rid, "output_logprob", record.t))
                    out.append(scalar_value(cid, LOGPROB_DTYPE, float(lp)))
        out.append(
            json_value(
                str(identity.boundary_id(profile_id, rid, "sampling_params")), self.sampling_params
            )
        )
        out.append(json_value(str(identity.boundary_id(profile_id, rid, "seed")), self.seed))
        out.append(
            utf8_value(
                str(identity.boundary_id(profile_id, rid, "finish_reason")),
                self.finish_reason or "",
            )
        )
        return out


# --- weights -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeafLayout:
    """How one weight tensor is packed into leaves (capture format v2, W3c 2026-09-07).

    ``whole``: one leaf over the tensor's bytes, id ``w=<param>[/s=..]`` (the v1 layout).
    ``rows``: one leaf per block of ``rows_per_leaf`` leading-dimension rows, ids
    ``w=<param>[/s=..]/c=<first row>``; a leaf's shape is ``[K...]`` for a single row and
    ``[b, K...]`` for a block of ``b`` rows (the last block may be shorter).  ``leaf_count`` is
    ``ceil(N / rows_per_leaf)``; ``leaves_offset`` is where the leaf hashes start in
    ``weights/leaves.bin`` (32 bytes each, block order), or ``None`` when not stored.
    """

    kind: str = "whole"
    rows_per_leaf: int = 0
    leaf_count: int = 1
    leaves_offset: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | None) -> LeafLayout:
        if doc is None:
            return cls()
        kind = str(doc.get("kind", "whole"))
        if kind not in ("whole", "rows"):
            raise CaptureError(f"unknown leaf layout kind {kind!r}")
        return cls(
            kind=kind,
            rows_per_leaf=int(doc.get("rows_per_leaf", 0)),
            leaf_count=int(doc.get("leaf_count", 1)),
            leaves_offset=None if doc.get("leaves_offset") is None else int(doc["leaves_offset"]),
        )


@dataclass(frozen=True, slots=True)
class WeightEntry:
    canonical_id: str
    param: str
    shard: int | None
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    sha256: str
    leaf: str  # leaf hash over the whole tensor; the tree leaf under layout ``whole``, informational under ``rows``
    offset: int | None  # None when values are not stored
    source: dict[str, Any] | None = None  # e.g. {"file": "model.safetensors", "tensor": "..."}
    layout: LeafLayout = LeafLayout()

    def to_json(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["shape"] = list(self.shape)
        doc["layout"] = self.layout.to_json()
        return doc

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> WeightEntry:
        try:
            return cls(
                canonical_id=str(doc["canonical_id"]),
                param=str(doc["param"]),
                shard=None if doc.get("shard") is None else int(doc["shard"]),
                dtype=dtypes.normalize_dtype(doc["dtype"]),
                shape=tuple(int(d) for d in doc["shape"]),
                nbytes=int(doc["nbytes"]),
                sha256=str(doc["sha256"]),
                leaf=str(doc["leaf"]),
                offset=None if doc.get("offset") is None else int(doc["offset"]),
                source=doc.get("source"),
                layout=LeafLayout.from_json(doc.get("layout")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CaptureError(f"malformed weights entry {doc!r}: {error}") from error

    # --- row-block geometry (layout ``rows``) ---
    @property
    def rows(self) -> int:
        return int(self.shape[0]) if self.shape else 1

    @property
    def row_nbytes(self) -> int:
        return self.nbytes // self.rows if self.rows else self.nbytes

    def block_starts(self) -> range:
        if self.layout.kind != "rows":
            raise CaptureError(f"{self.canonical_id} has whole-tensor leaves")
        return range(0, self.rows, self.layout.rows_per_leaf)

    def block_of(self, row: int) -> tuple[int, int]:
        """``(first_row, end_row)`` of the block containing ``row``."""
        b = self.layout.rows_per_leaf
        if self.layout.kind != "rows" or b <= 0:
            raise CaptureError(f"{self.canonical_id} has whole-tensor leaves")
        if not 0 <= row < self.rows:
            raise CaptureError(f"{self.canonical_id}: row {row} outside {self.rows} rows")
        start = (row // b) * b
        return start, min(start + b, self.rows)

    def block_id(self, first_row: int) -> str:
        return str(identity.weight_id(self.param, self.shard, first_row))

    def block_shape(self, first_row: int, end_row: int) -> tuple[int, ...]:
        """Always ``(rows_in_block, *shape[1:])`` -- a one-row block is ``(1, N)``, not ``(N,)``."""
        return (end_row - first_row, *tuple(self.shape[1:]))

    def block_ids(self) -> list[str]:
        if self.layout.kind != "rows":
            return [self.canonical_id]
        return [self.block_id(r) for r in self.block_starts()]


@dataclass(slots=True)
class WeightsIndex:
    entries: list[WeightEntry]
    store_values: bool
    source_sha256: dict[str, str]  # file -> sha256, must equal manifest.model.weights_sha256
    values_nbytes: int = 0
    format_version: int = 1
    leaves_nbytes: int = 0

    def to_json(self) -> dict[str, Any]:
        v2 = self.format_version >= 2
        doc: dict[str, Any] = {
            "schema": WEIGHTS_INDEX_SCHEMA_V2 if v2 else WEIGHTS_INDEX_SCHEMA,
            "store_values": self.store_values,
            "source_sha256": dict(self.source_sha256),
            "values_nbytes": self.values_nbytes,
        }
        if v2:
            doc["leaves_nbytes"] = self.leaves_nbytes
            doc["entries"] = [e.to_json() for e in self.entries]
        else:
            doc["entries"] = [
                {k: v for k, v in e.to_json().items() if k != "layout"} for e in self.entries
            ]
        return doc

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> WeightsIndex:
        schema = doc.get("schema", WEIGHTS_INDEX_SCHEMA)
        if schema not in (WEIGHTS_INDEX_SCHEMA, WEIGHTS_INDEX_SCHEMA_V2):
            raise CaptureError(f"unexpected weights schema {schema!r}")
        version = 2 if schema == WEIGHTS_INDEX_SCHEMA_V2 else 1
        try:
            entries = [WeightEntry.from_json(e) for e in doc["entries"]]
            if version == 1 and any(e.layout.kind != "whole" for e in entries):
                raise CaptureError("a v1 weights index cannot carry row leaf layouts")
            return cls(
                entries=entries,
                store_values=bool(doc.get("store_values", False)),
                source_sha256=dict(doc.get("source_sha256", {})),
                values_nbytes=int(doc.get("values_nbytes", 0)),
                format_version=version,
                leaves_nbytes=int(doc.get("leaves_nbytes", 0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CaptureError(f"malformed weights index: {error}") from error

    def ids(self) -> list[str]:
        """Every tree leaf id (whole-tensor ids and row-block ids), in tree order."""
        out: list[str] = []
        for e in self.entries:
            out.extend(e.block_ids())
        return identity.tree_sorted(out)

    def leaf_count(self) -> int:
        return sum(e.layout.leaf_count if e.layout.kind == "rows" else 1 for e in self.entries)

    def leaves(self) -> dict[str, bytes]:
        """Whole-tensor leaf hashes as recorded (v1 semantics); row leaves need the reader."""
        if any(e.layout.kind == "rows" for e in self.entries):
            raise CaptureError("row leaf layouts: use WeightsReader.leaves()")
        return {e.canonical_id: bytes.fromhex(e.leaf) for e in self.entries}


def row_block_leaves(entry: WeightEntry, raw: bytes) -> list[bytes]:
    """Leaf hashes of every row block of ``entry`` from the tensor's bytes, block order."""
    out: list[bytes] = []
    rb = entry.row_nbytes
    for start in entry.block_starts():
        end = min(start + entry.layout.rows_per_leaf, entry.rows)
        out.append(
            hashing.leaf_hash(
                entry.block_id(start),
                entry.dtype,
                entry.block_shape(start, end),
                raw[start * rb : end * rb],
            )
        )
    return out


class WeightsWriter:
    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        store_values: bool = True,
        source_sha256: Mapping[str, str] | None = None,
        format_version: int = 1,
    ) -> None:
        self.dir = Path(run_dir) / WEIGHTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / INDEX_FILE).exists():
            raise CaptureError(f"{self.dir} already holds a finished weights index")
        if format_version not in (1, 2):
            raise CaptureError(f"weights index format {format_version} unknown")
        self.format_version = format_version
        self.store_values = store_values
        self.source_sha256 = dict(source_sha256 or {})
        self._entries: list[WeightEntry] = []
        self._ids: set[str] = set()
        self._offset = 0
        self._leaves_offset = 0
        self._file: BinaryIO | None = (
            open(self.dir / VALUES_FILE, "wb") if store_values else None  # noqa: SIM115
        )
        self._leaves_file: BinaryIO | None = (
            open(self.dir / LEAVES_FILE, "wb") if format_version >= 2 else None  # noqa: SIM115
        )
        self._finished = False

    def add(
        self,
        param: str,
        dtype: str,
        shape: Sequence[int],
        data: Any | None = None,
        *,
        shard: int | None = None,
        sha256: str | None = None,
        leaf: str | None = None,
        source: Mapping[str, Any] | None = None,
        store: bool | None = None,
        rows_per_leaf: int | None = None,
    ) -> WeightEntry:
        """Add one weight tensor.  With ``store_values`` the bytes are required and stored;
        otherwise pass the bytes (hashed, not stored) or a precomputed ``sha256`` and ``leaf``.

        ``store=False`` on a storing writer hashes this entry's bytes without writing them (the entry
        then has ``offset None`` and names its ``source``: a checkpoint tensor the reader's resolver
        rebuilds and re-hashes); ``store=True`` on a hash-only writer is an error.

        ``rows_per_leaf`` (format v2 only, needs the bytes) commits the tensor as one leaf per
        block of that many leading-dimension rows instead of one whole-tensor leaf."""
        if self._finished:
            raise CaptureError("weights writer is finished")
        if store is True and not self.store_values:
            raise CaptureError(f"{param}: store=True on a hash-only weights writer")
        store_this = self.store_values if store is None else bool(store)
        cid = str(identity.weight_id(param, shard))
        if cid in self._ids:
            raise StaleCaptureError(f"{cid} added twice")
        dtype = dtypes.normalize_dtype(dtype)
        shape_t = tuple(int(d) for d in shape)
        expected = dtypes.expected_nbytes(dtype, shape_t)
        if expected is None:
            raise CaptureError("weights must have a tensor dtype")
        layout = LeafLayout()
        if rows_per_leaf is not None:
            if self.format_version < 2:
                raise CaptureError("row leaf layouts need format_version=2")
            if data is None:
                raise CaptureError(f"{cid}: a row leaf layout needs the bytes")
            if len(shape_t) < 1 or shape_t[0] < 1 or rows_per_leaf < 1:
                raise CaptureError(f"{cid}: rows_per_leaf needs a non-empty leading dimension")
            n = shape_t[0]
            layout = LeafLayout(
                "rows", int(rows_per_leaf), (n + rows_per_leaf - 1) // rows_per_leaf, self._leaves_offset
            )
        offset: int | None = None
        if data is not None:
            raw = as_bytes(data)
            dtypes.check_nbytes(dtype, shape_t, len(raw))
            sha = hashing.sha256_hex(raw)
            leaf_hex = hashing.leaf_hash(cid, dtype, shape_t, raw).hex()
            if sha256 is not None and sha256 != sha:
                raise CaptureError(f"{cid}: given sha256 disagrees with the bytes")
            if leaf is not None and leaf != leaf_hex:
                raise CaptureError(f"{cid}: given leaf disagrees with the bytes")
            if self._file is not None and store_this:
                self._file.write(raw)
                offset = self._offset
                self._offset += len(raw)
        else:
            if self.store_values:
                raise CaptureError(f"{cid}: store_values=True needs the bytes")
            if sha256 is None or leaf is None:
                raise CaptureError(f"{cid}: without bytes both sha256 and leaf are required")
            sha, leaf_hex = sha256, leaf
        entry = WeightEntry(
            canonical_id=cid,
            param=param,
            shard=shard,
            dtype=dtype,
            shape=shape_t,
            nbytes=expected,
            sha256=sha,
            leaf=leaf_hex,
            offset=offset,
            source=dict(source) if source else None,
            layout=layout,
        )
        if self._leaves_file is not None:
            hashes = row_block_leaves(entry, raw) if layout.kind == "rows" else [bytes.fromhex(leaf_hex)]
            if layout.kind == "rows":
                assert len(hashes) == layout.leaf_count
            else:  # whole-tensor leaf: recorded at leaves_offset too, so leaves.bin covers every leaf
                entry = dataclasses.replace(entry, layout=LeafLayout("whole", 0, 1, self._leaves_offset))
            for h in hashes:
                self._leaves_file.write(h)
            self._leaves_offset += hashing.DIGEST_BYTES * len(hashes)
        self._entries.append(entry)
        self._ids.add(cid)
        return entry

    def finish(self) -> WeightsIndex:
        if self._finished:
            raise CaptureError("weights writer already finished")
        for f in (self._file, self._leaves_file):
            if f is not None:
                f.flush()
                os.fsync(f.fileno())
                f.close()
        self._file = None
        self._leaves_file = None
        self._finished = True
        index = WeightsIndex(
            entries=list(self._entries),
            store_values=self.store_values,
            source_sha256=dict(self.source_sha256),
            values_nbytes=self._offset,
            format_version=self.format_version,
            leaves_nbytes=self._leaves_offset,
        )
        _write_json(self.dir / INDEX_FILE, index.to_json())
        return index


# --- run writer ----------------------------------------------------------------------------------


class RunWriter:
    """Run-level files.  ``manifest.json`` is written as canonical JSON at construction."""

    def __init__(self, run_dir: str | os.PathLike[str], manifest: Mapping[str, Any]) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise CaptureError(f"manifest.schema must be {MANIFEST_SCHEMA!r}")
        self.manifest = dict(manifest)
        self.profile_id = hashing.profile_id(self.manifest)
        _write_json(self.run_dir / MANIFEST_FILE, self.manifest, canonical=True)
        self._rus: list[tuple[str, int]] = []
        self._requests: set[str] = set()

    def ru(
        self,
        request_id: str,
        step: int,
        *,
        phase: str,
        positions: Sequence[int],
        on_duplicate: str = "error",
        speculative: Mapping[str, Any] | None = None,
    ) -> CaptureWriter:
        self._rus.append((request_id, step))
        self._requests.add(request_id)
        return CaptureWriter(
            self.run_dir,
            request_id,
            step,
            phase=phase,
            positions=positions,
            speculative=speculative,
            profile_id=self.profile_id,
            on_duplicate=on_duplicate,
        )

    def write_boundary(self, boundary: Boundary) -> Path:
        problems = boundary.validate()
        if problems:
            raise CaptureError(f"boundary {boundary.request_id}: " + "; ".join(problems))
        self._requests.add(boundary.request_id)
        path = self.run_dir / RU_DIR / boundary.request_id / BOUNDARY_FILE
        _write_json(path, boundary.to_json())
        return path

    def write_description(self, description: Mapping[str, Any]) -> Path:
        path = self.run_dir / DESCRIPTION_FILE
        _write_json(path, description, canonical=True)
        return path

    def write_required(self, required: Mapping[str, Any]) -> Path:
        path = self.run_dir / REQUIRED_FILE
        _write_json(path, required, canonical=True)
        return path

    def weights(
        self,
        *,
        store_values: bool = True,
        source_sha256: Mapping[str, str] | None = None,
        format_version: int = 1,
    ) -> WeightsWriter:
        if source_sha256 is None:
            source_sha256 = self.manifest.get("model", {}).get("weights_sha256", {})
        return WeightsWriter(
            self.run_dir,
            store_values=store_values,
            source_sha256=source_sha256,
            format_version=format_version,
        )

    def finish(self) -> Path:
        summary = {
            "schema": CAPTURE_SCHEMA,
            "format_version": 1,
            "profile_id": self.profile_id,
            "request_ids": sorted(self._requests),
            "replay_units": [
                {"request_id": r, "step": s, "tree_id": ru_tree_id(r, s)}
                for r, s in sorted(self._rus)
            ],
        }
        path = self.run_dir / CAPTURE_FILE
        _write_json(path, summary)
        return path


# --- readers -------------------------------------------------------------------------------------


class RUReader:
    """Lazy access to one replay unit's bytes by canonical id."""

    def __init__(self, directory: Path, request_id: str, step: int) -> None:
        self.dir = directory
        self.request_id = request_id
        self.step = step
        self.index = RUIndex.from_json(
            _read_json(directory / INDEX_FILE), request_id=request_id, step=step
        )
        self._entries = {e.canonical_id: e for e in self.index.entries}
        self._values_path = directory / VALUES_FILE
        self._mm: np.memmap | None = None

    @property
    def tree_id(self) -> str:
        return ru_tree_id(self.request_id, self.step)

    def ids(self) -> list[str]:
        return self.index.ids()

    def __contains__(self, canonical_id: str) -> bool:
        return canonical_id in self._entries

    def entry(self, canonical_id: str) -> Entry:
        try:
            return self._entries[canonical_id]
        except KeyError as error:
            raise KeyError(f"{canonical_id} is not in {self.tree_id}") from error

    def _raw(self) -> np.memmap | np.ndarray:
        if self._mm is None:
            size = self._values_path.stat().st_size
            if size == 0:
                self._mm = np.zeros(0, dtype=np.uint8)  # type: ignore[assignment]
            else:
                self._mm = np.memmap(self._values_path, dtype=np.uint8, mode="r")
        return self._mm

    def read(self, canonical_id: str) -> bytes:
        entry = self.entry(canonical_id)
        if entry.derived is not None:
            return entry.derived_bytes()
        raw = self._raw()
        if entry.offset + entry.nbytes > len(raw):
            raise CaptureError(f"{canonical_id}: values.bin is shorter than the index says")
        return bytes(raw[entry.offset : entry.offset + entry.nbytes])

    def value(self, canonical_id: str) -> Value:
        entry = self.entry(canonical_id)
        return Value(canonical_id, entry.dtype, entry.shape, self.read(canonical_id), entry.origin)

    def array(self, canonical_id: str) -> np.ndarray:
        return self.value(canonical_id).array()

    def values(self) -> Iterator[Value]:
        for entry in self.index.entries:
            yield self.value(entry.canonical_id)

    def leaves(self, *, recompute: bool = True) -> dict[str, bytes]:
        """canonical_id -> leaf hash; ``recompute`` hashes the bytes, else trusts the index."""
        if not recompute:
            return {e.canonical_id: e.leaf_bytes() for e in self.index.entries}
        return {v.canonical_id: v.leaf() for v in self.values()}

    def validate(self, *, check_bytes: bool = True) -> list[str]:
        problems = self.index.validate()
        size = self._values_path.stat().st_size if self._values_path.exists() else -1
        if size < 0:
            problems.append("values.bin missing")
            return problems
        end = max((e.offset + e.nbytes for e in self.index.entries if e.derived is None), default=0)
        if end != size:
            problems.append(f"values.bin is {size} bytes, entries end at {end}")
        if check_bytes and end == size:
            for entry in self.index.entries:
                if entry.derived is not None:
                    continue  # prescribed bytes: nothing stored to compare against
                value = self.value(entry.canonical_id)
                if value.leaf().hex() != entry.leaf:
                    problems.append(f"{entry.canonical_id}: stored leaf != hash of the bytes")
        return problems


class WeightsReader:
    """Weights by canonical id.  Whole-tensor ids (``w=<param>``) always resolve to the whole
    tensor; under a ``rows`` layout (format v2) the tree leaves are the row blocks
    ``w=<param>/c=<row>`` and the whole tensor is authenticated by a range opening
    (:mod:`verity_vllm.commit.merkle`)."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.index = WeightsIndex.from_json(_read_json(directory / INDEX_FILE))
        self._entries = {e.canonical_id: e for e in self.index.entries}
        self._values_path = directory / VALUES_FILE
        self._leaves_path = directory / LEAVES_FILE
        self._mm: np.memmap | None = None
        self._leaves_mm: np.memmap | None = None
        self.resolver: Any = None  # optional callable(WeightEntry) -> bytes when values not stored

    def ids(self) -> list[str]:
        """Tree leaf ids in tree order."""
        return self.index.ids()

    def tensor_ids(self) -> list[str]:
        return identity.tree_sorted([e.canonical_id for e in self.index.entries])

    def locate(self, canonical_id: str) -> tuple[WeightEntry, tuple[int, int] | None]:
        """``(entry, (first_row, end_row) | None)``; ``None`` = the whole tensor."""
        cid = identity.parse(canonical_id)
        if cid.kind != "weight":
            raise KeyError(f"{canonical_id} is not a weight id")
        whole = str(identity.weight_id(cid.param or "", cid.shard))
        try:
            entry = self._entries[whole]
        except KeyError as error:
            raise KeyError(f"{canonical_id} is not in the weights index") from error
        if cid.coordinate is None:
            return entry, None
        if entry.layout.kind != "rows":
            raise KeyError(f"{canonical_id}: {whole} is committed as one whole-tensor leaf")
        start, end = entry.block_of(int(cid.coordinate))
        if start != int(cid.coordinate):
            raise KeyError(f"{canonical_id}: blocks start at multiples of {entry.layout.rows_per_leaf}")
        return entry, (start, end)

    def __contains__(self, canonical_id: str) -> bool:
        try:
            self.locate(canonical_id)
        except (KeyError, identity.IdentityError, CaptureError):
            return False
        return True

    def is_leaf(self, canonical_id: str) -> bool:
        """Is this id one tree leaf (as opposed to the whole tensor of a ``rows`` layout)?"""
        entry, block = self.locate(canonical_id)
        return block is not None or entry.layout.kind == "whole"

    def entry(self, canonical_id: str) -> WeightEntry:
        return self.locate(canonical_id)[0]

    def leaf_id_for_row(self, canonical_id: str, row: int) -> str:
        """The tree leaf holding row ``row`` of a tensor: its block id under a ``rows`` layout, the
        whole-tensor id otherwise."""
        entry = self.entry(canonical_id)
        if entry.layout.kind != "rows":
            return entry.canonical_id
        start, _end = entry.block_of(int(row))
        return entry.block_id(start)

    def _read_whole(self, entry: WeightEntry) -> bytes:
        if entry.offset is not None and self._values_path.exists():
            return bytes(self._read_rows_mm(entry, 0, entry.rows))
        if self.resolver is not None:
            raw = as_bytes(self.resolver(entry))
            if hashing.sha256_hex(raw) != entry.sha256:
                raise CaptureError(f"{entry.canonical_id}: resolved bytes do not match the index sha256")
            return raw
        raise CaptureError(f"{entry.canonical_id}: weight values are not stored and no resolver is set")

    def _read_rows_mm(self, entry: WeightEntry, start: int, end: int) -> bytes:
        if self._mm is None:
            self._mm = np.memmap(self._values_path, dtype=np.uint8, mode="r")
        assert entry.offset is not None
        rb = entry.row_nbytes
        return bytes(self._mm[entry.offset + start * rb : entry.offset + end * rb])

    def _read_rows(self, entry: WeightEntry, start: int, end: int) -> bytes:
        """Bytes of rows ``start .. end-1`` (subclasses route patched tensors through here)."""
        if entry.offset is not None and self._values_path.exists():
            return self._read_rows_mm(entry, start, end)
        rb = entry.row_nbytes
        return self._read_whole(entry)[start * rb : end * rb]

    def read(self, canonical_id: str) -> bytes:
        entry, block = self.locate(canonical_id)
        if block is None:
            return self._read_whole(entry)
        return self._read_rows(entry, *block)

    def value(self, canonical_id: str) -> Value:
        entry, block = self.locate(canonical_id)
        if block is None:
            return Value(canonical_id, entry.dtype, entry.shape, self._read_whole(entry), "captured")
        shape = entry.block_shape(*block)
        return Value(canonical_id, entry.dtype, shape, self._read_rows(entry, *block), "captured")

    def block_values(self, canonical_id: str) -> list[Value]:
        """The tree leaves covering ``canonical_id``: itself, or every row block of a whole id."""
        entry, block = self.locate(canonical_id)
        if block is not None or entry.layout.kind == "whole":
            return [self.value(canonical_id)]
        return [self.value(bid) for bid in identity.tree_sorted(entry.block_ids())]

    def _stored_leaves(self, entry: WeightEntry) -> list[bytes] | None:
        lay = entry.layout
        if lay.leaves_offset is None or not self._leaves_path.exists():
            return None
        if self._leaves_mm is None:
            self._leaves_mm = np.memmap(self._leaves_path, dtype=np.uint8, mode="r")
        n = lay.leaf_count if lay.kind == "rows" else 1
        raw = bytes(self._leaves_mm[lay.leaves_offset : lay.leaves_offset + n * hashing.DIGEST_BYTES])
        if len(raw) != n * hashing.DIGEST_BYTES:
            raise CaptureError(f"{entry.canonical_id}: leaves.bin is shorter than the index says")
        return [raw[i : i + hashing.DIGEST_BYTES] for i in range(0, len(raw), hashing.DIGEST_BYTES)]

    def _patched(self, entry: WeightEntry) -> bool:
        return False

    def _bytes_obtainable(self, entry: WeightEntry) -> bool:
        return (entry.offset is not None and self._values_path.exists()) or self.resolver is not None

    def entry_leaves(self, entry: WeightEntry, *, recompute: bool = False) -> dict[str, bytes]:
        """Leaf hashes of one tensor's tree leaves (``recompute`` hashes the bytes).

        A bundle may reference the weights by checkpoint sha and ship no ``values.bin`` (section 9,
        W2c review fix); without a resolver there are then no bytes to rehash and ``recompute`` falls
        back to the recorded leaf hashes, exactly as the pre-v2 ``weights_tree`` always did. Whether
        the weights were *actually* rebuilt from the registered checkpoint is the model binding's
        report (``verity_vllm.commit.poc_binding``), which installs a resolver and so never takes this
        fallback. A patched entry (negatives) has no recorded hash to fall back on."""
        if recompute and not self._patched(entry) and not self._bytes_obtainable(entry):
            recompute = False
        if entry.layout.kind == "whole":
            if recompute or self._patched(entry):
                raw = self._read_whole(entry)
                return {entry.canonical_id: hashing.leaf_hash(entry.canonical_id, entry.dtype, entry.shape, raw)}
            return {entry.canonical_id: bytes.fromhex(entry.leaf)}
        stored = None if (recompute or self._patched(entry)) else self._stored_leaves(entry)
        ids = entry.block_ids()
        if stored is not None:
            if len(stored) != len(ids):
                raise CaptureError(f"{entry.canonical_id}: leaves.bin has {len(stored)} leaves, layout says {len(ids)}")
            return dict(zip(ids, stored, strict=True))
        return dict(zip(ids, row_block_leaves(entry, self._read_whole(entry)), strict=True))

    def leaves(self, *, recompute: bool = False) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for entry in self.index.entries:
            out.update(self.entry_leaves(entry, recompute=recompute))
        return out

    def validate(self, *, check_bytes: bool = True) -> list[str]:
        problems: list[str] = []
        for entry in self.index.entries:
            try:
                cid = identity.parse(entry.canonical_id)
            except identity.IdentityError as error:
                problems.append(str(error))
                continue
            if cid.kind != "weight" or cid.param != entry.param or cid.shard != entry.shard or cid.coordinate is not None:
                problems.append(f"{entry.canonical_id}: id disagrees with param/shard")
            try:
                dtypes.check_nbytes(entry.dtype, entry.shape, entry.nbytes)
            except dtypes.DtypeError as error:
                problems.append(f"{entry.canonical_id}: {error}")
            if entry.layout.kind == "rows":
                if not entry.shape or entry.layout.rows_per_leaf < 1:
                    problems.append(f"{entry.canonical_id}: rows layout on a tensor without rows")
                    continue
                if entry.layout.leaf_count != len(entry.block_starts()):
                    problems.append(f"{entry.canonical_id}: leaf_count {entry.layout.leaf_count} != {len(entry.block_starts())} blocks")
            # review fix (W2c): a bundle may reference the weights by checkpoint sha and omit
            # values.bin (section 9); the index's leaf hashes still give the root, and weight
            # openings become `missing` in the checker instead of crashing validation here.
            if check_bytes and entry.offset is not None and self._values_path.exists():
                raw = self._read_whole(entry)
                if hashing.sha256_hex(raw) != entry.sha256:
                    problems.append(f"{entry.canonical_id}: sha256 != bytes")
                if (
                    hashing.leaf_hash(entry.canonical_id, entry.dtype, entry.shape, raw).hex()
                    != entry.leaf
                ):
                    problems.append(f"{entry.canonical_id}: leaf != hash of the bytes")
                if entry.layout.kind == "rows":
                    stored = self._stored_leaves(entry)
                    if stored is not None and stored != row_block_leaves(entry, raw):
                        problems.append(f"{entry.canonical_id}: stored row leaves != hash of the bytes")
        return problems


class CaptureReader:
    """Read-only view of a run directory; lazy, by canonical id."""

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise CaptureError(f"{self.run_dir} is not a directory")
        self._manifest: dict[str, Any] | None = None
        self._rus: dict[tuple[str, int], RUReader] = {}
        self._boundaries: dict[str, Boundary] = {}
        # lane OUT16: id -> leaf memo per request.  ``value()`` on a boundary id used to rebuild
        # and linearly scan the whole leaf list per lookup — O(leaves) per open, O(leaves^2) for
        # the every-audit output-linkage/domain checks that open one token leaf per designated
        # slot.  Memoised per request exactly like ``_boundaries`` (same instance lifetime; a
        # PatchedReader is constructed per mutation, so a patched boundary gets a fresh memo).
        self._boundary_leaves: dict[str, dict[str, Value]] = {}
        self._weights: WeightsReader | None = None

    # run-level documents
    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = _read_json(self.run_dir / MANIFEST_FILE)
        return self._manifest

    @property
    def profile_id(self) -> str:
        return hashing.profile_id(self.manifest)

    def manifest_bytes(self) -> bytes:
        return (self.run_dir / MANIFEST_FILE).read_bytes()

    @property
    def description(self) -> dict[str, Any]:
        return _read_json(self.run_dir / DESCRIPTION_FILE)

    @property
    def required(self) -> dict[str, Any]:
        return _read_json(self.run_dir / REQUIRED_FILE)

    @property
    def commitments(self) -> dict[str, Any] | None:
        path = self.run_dir / COMMITMENTS_FILE
        return _read_json(path) if path.exists() else None

    def has(self, name: str) -> bool:
        return (self.run_dir / name).exists()

    # requests and replay units
    def request_ids(self) -> list[str]:
        root = self.run_dir / RU_DIR
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir())

    def boundary(self, request_id: str) -> Boundary:
        if request_id not in self._boundaries:
            path = self.run_dir / RU_DIR / request_id / BOUNDARY_FILE
            self._boundaries[request_id] = Boundary.from_json(_read_json(path))
        return self._boundaries[request_id]

    def has_boundary(self, request_id: str) -> bool:
        return (self.run_dir / RU_DIR / request_id / BOUNDARY_FILE).exists()

    def ru_keys(self, request_id: str | None = None) -> list[tuple[str, int]]:
        keys: list[tuple[str, int]] = []
        for rid in [request_id] if request_id else self.request_ids():
            root = self.run_dir / RU_DIR / rid
            if not root.is_dir():
                continue
            for p in sorted(root.iterdir()):
                if p.is_dir() and p.name.isdigit() and (p / INDEX_FILE).exists():
                    keys.append((rid, int(p.name)))
        return keys

    def ru(self, request_id: str, step: int) -> RUReader:
        key = (request_id, step)
        if key not in self._rus:
            directory = self.run_dir / RU_DIR / request_id / step_dirname(step)
            self._rus[key] = RUReader(directory, request_id, step)
        return self._rus[key]

    def rus(self) -> Iterator[RUReader]:
        for rid, step in self.ru_keys():
            yield self.ru(rid, step)

    @property
    def weights(self) -> WeightsReader:
        if self._weights is None:
            self._weights = WeightsReader(self.run_dir / WEIGHTS_DIR)
        return self._weights

    def has_weights(self) -> bool:
        return (self.run_dir / WEIGHTS_DIR / INDEX_FILE).exists()

    def recompute_problems(self, request_id: str, *, check_bytes: bool = False) -> list[str]:
        """Capture-format v1.1 (R4): a replay unit flagged ``recompute`` repeats identities an earlier
        unit of the request committed; every repeated leaf must equal the first unit's leaf (index
        leaf hashes, or hashes of the bytes with ``check_bytes``; the audit ties index to bytes).
        The first unit is the one ``locate`` and the openings use, so a recompute unit that
        disagrees is a committed second value of one identity -- refused here, never shadowed."""
        if not self.has_boundary(request_id):
            return []
        boundary = self.boundary(request_id)
        problems: list[str] = []
        first_step: dict[int, int] = {}
        for s in boundary.steps:
            for t in s.positions:
                first_step.setdefault(t, s.step)

        def leaves_of(step: int) -> dict[str, str]:
            unit = self.ru(request_id, step)
            if check_bytes:
                return {cid: leaf.hex() for cid, leaf in unit.leaves(recompute=True).items()}
            return {e.canonical_id: e.leaf for e in unit.index.entries}

        for s in boundary.steps:
            if not s.recompute:
                continue
            primaries: dict[int, dict[str, str]] = {}
            for cid_text, leaf in leaves_of(s.step).items():
                cid = identity.parse(cid_text)
                assert cid.t is not None
                orig = first_step.get(cid.t)
                if orig is None or orig == s.step:
                    continue
                if orig not in primaries:
                    primaries[orig] = leaves_of(orig)
                first = primaries[orig].get(cid_text)
                if first is None:
                    problems.append(f"step {s.step}: {cid_text} recomputed but never committed by step {orig}")
                elif first != leaf:
                    problems.append(f"step {s.step}: {cid_text} recomputed with different bytes than step {orig}")
                if len(problems) >= 4:
                    return problems
        return problems

    # lookup by canonical id
    def locate(self, canonical_id: str) -> tuple[str, int] | None:
        """(request_id, step) of the replay unit holding a value id, via the boundary's steps,
        falling back to a scan of the request's units."""
        cid = identity.parse(canonical_id)
        if cid.kind != "value":
            return None
        rid = cid.request_id
        assert rid is not None and cid.t is not None
        if self.has_boundary(rid):
            step = self.boundary(rid).step_of(cid.t)
            if step is not None and canonical_id in self.ru(rid, step.step):
                return (rid, step.step)
        for key in self.ru_keys(rid):
            if canonical_id in self.ru(*key):
                return key
        return None

    def tree_of(self, canonical_id: str) -> str | None:
        cid = identity.parse(canonical_id)
        if cid.kind == "weight":
            return WEIGHTS_TREE_ID if canonical_id in self.weights else None
        if cid.kind == "boundary":
            return boundary_tree_id(cid.request_id or "")
        located = self.locate(canonical_id)
        return ru_tree_id(*located) if located else None

    def value(self, canonical_id: str) -> Value:
        """Any id kind: value ids from their replay unit, weights, boundary leaves."""
        cid = identity.parse(canonical_id)
        if cid.kind == "weight":
            return self.weights.value(canonical_id)
        if cid.kind == "boundary":
            rid = cid.request_id or ""
            leaves = self._boundary_leaves.get(rid)
            if leaves is None:
                leaves = self._boundary_leaves[rid] = {
                    leaf.canonical_id: leaf for leaf in self.boundary(rid).leaves(self.profile_id)
                }
            try:
                return leaves[canonical_id]
            except KeyError:
                raise KeyError(f"{canonical_id} is not a leaf of the boundary tree") from None
        located = self.locate(canonical_id)
        if located is None:
            raise KeyError(f"{canonical_id} is not in any replay unit")
        return self.ru(*located).value(canonical_id)

    def validate(self, *, check_bytes: bool = True) -> list[str]:
        problems: list[str] = []
        if not (self.run_dir / MANIFEST_FILE).exists():
            problems.append("manifest.json missing")
        for rid in self.request_ids():
            if self.has_boundary(rid):
                problems.extend(f"boundary {rid}: {p}" for p in self.boundary(rid).validate())
            else:
                problems.append(f"boundary {rid}: missing")
        for ru in self.rus():
            problems.extend(f"{ru.tree_id}: {p}" for p in ru.validate(check_bytes=check_bytes))
        if self.has_weights():
            problems.extend(f"weights: {p}" for p in self.weights.validate(check_bytes=check_bytes))
        return problems
