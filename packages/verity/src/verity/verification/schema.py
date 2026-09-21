"""The ``packed/v1`` leaf schema grammar, shared by the statement and the packed witness.

Split out of :mod:`veritor.protocol.proofs.packed` so that
:mod:`veritor.protocol.proofs.statement` can validate a range position's leaf
descriptor without importing the witness format (which imports the statement).
Everything here is re-exported by ``packed`` under its historical names.

::

    schema     = packed/v1:<framing>:u<w><le|be>:<count>:<index>     one word of a packed leaf
    descriptor = packed/v1:<framing>:u<w><le|be>:<count>             the leaf's framing and word layout

``(framing, u<w><order>, count)`` is the **leaf descriptor**: every position
that names one rank must agree on it (and on ``position``); the index is the
position's own.  A statement-format-3 range names a descriptor and a set of
indices ``start, start+stride, ...`` (:class:`~veritor.protocol.proofs.statement.PositionRange`).
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.errors import ProtocolError

__all__ = [
    "PACKED_SCHEMA_PREFIX",
    "VLLM_LEAF_FRAMING",
    "LeafDescriptor",
    "PackedSchema",
    "leaf_descriptor",
    "packed_schema",
    "parse_leaf_descriptor",
    "parse_packed_schema",
]

PACKED_SCHEMA_PREFIX = "packed/v1:"
#: The serving side's leaf framing (``verity_vllm.hashing.LEAF_TAG``); the tree
#: id in the interface's ``roots[].framing``.
VLLM_LEAF_FRAMING = "verity-vllm/leaf/v1"

_FRAMINGS = (VLLM_LEAF_FRAMING,)


@dataclass(frozen=True, slots=True)
class LeafDescriptor:
    """What every position naming one packed leaf must agree on: its framing and word layout."""

    framing: str
    width: int
    little_endian: bool
    count: int

    @property
    def word_bytes(self) -> int:
        return (self.width + 7) // 8

    @property
    def leaf_bytes(self) -> int:
        return self.count * self.word_bytes

    def text(self) -> str:
        """The canonical descriptor text ``packed/v1:<framing>:u<w><order>:<count>``."""

        return leaf_descriptor(self.framing, self.width, self.little_endian, self.count)

    def schema(self, index: int) -> str:
        """The schema of word ``index`` of a leaf with this descriptor."""

        return packed_schema(self.framing, self.width, self.little_endian, self.count, index)


@dataclass(frozen=True, slots=True)
class PackedSchema:
    """A parsed ``packed/v1`` schema: the leaf descriptor and this position's word index."""

    framing: str
    width: int
    little_endian: bool
    count: int
    index: int

    @property
    def descriptor(self) -> LeafDescriptor:
        return LeafDescriptor(self.framing, self.width, self.little_endian, self.count)

    @property
    def word_bytes(self) -> int:
        return (self.width + 7) // 8

    def text(self) -> str:
        return packed_schema(self.framing, self.width, self.little_endian, self.count, self.index)


def leaf_descriptor(framing: str, width: int, little_endian: bool, count: int) -> str:
    """The canonical text of a leaf descriptor (a packed schema without its index)."""

    if framing not in _FRAMINGS:
        raise ProtocolError(f"unknown packed-leaf framing {framing!r}")
    if not 1 <= width <= 64:
        raise ProtocolError("word width must be in [1, 64]")
    if not 1 <= count < 1 << 32:
        raise ProtocolError(f"a packed leaf holds between 1 and 2^32-1 words, not {count}")
    order = "le" if little_endian else "be"
    return f"{PACKED_SCHEMA_PREFIX}{framing}:u{width}{order}:{count}"


def packed_schema(framing: str, width: int, little_endian: bool, count: int, index: int) -> str:
    """The canonical text of a packed schema (see the module docstring)."""

    if not 0 <= index < count:
        raise ProtocolError(f"word index {index} outside a leaf of {count} words")
    return f"{leaf_descriptor(framing, width, little_endian, count)}:{index}"


def _decimal(text: str, bound: int) -> int | None:
    if not text.isdigit() or (len(text) > 1 and text[0] == "0"):
        return None
    value = int(text)
    return value if value < bound else None


def parse_leaf_descriptor(text: str) -> LeafDescriptor | None:
    """The :class:`LeafDescriptor` of ``text``, or ``None`` if it is not a well-formed descriptor.

    Strict: exactly three ``:``-separated fields after the prefix (the framing
    contains no ``:``), decimal fields without leading zeros, ``1 <= w <= 64``,
    ``1 <= count < 2^32``.
    """

    if not text.startswith(PACKED_SCHEMA_PREFIX):
        return None
    parts = text[len(PACKED_SCHEMA_PREFIX) :].split(":")
    if len(parts) != 3:
        return None
    framing, word, count_text = parts
    if framing not in _FRAMINGS:
        return None
    if len(word) < 4 or word[0] != "u" or word[-2:] not in ("le", "be"):
        return None
    width = _decimal(word[1:-2], 65)
    count = _decimal(count_text, 1 << 32)
    if width is None or width == 0 or count is None or count == 0:
        return None
    return LeafDescriptor(framing, width, word.endswith("le"), count)


def parse_packed_schema(schema: str) -> PackedSchema | None:
    """The :class:`PackedSchema` of ``schema``, or ``None`` if it is not a well-formed packed schema.

    Strict: exactly four ``:``-separated fields after the prefix (the framing
    contains no ``:``), decimal fields without leading zeros, ``1 <= w <=
    64``, ``1 <= count < 2^32``, ``index < count``.
    """

    head, sep, index_text = schema.rpartition(":")
    if not sep:
        return None
    descriptor = parse_leaf_descriptor(head)
    index = _decimal(index_text, 1 << 32)
    if descriptor is None or index is None or index >= descriptor.count:
        return None
    return PackedSchema(descriptor.framing, descriptor.width, descriptor.little_endian, descriptor.count, index)
