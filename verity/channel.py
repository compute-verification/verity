"""What a channel's messages declare: messages and their schemas, and refusals by name. Stdlib only, and it imports no
other unit, so an auditor party and a developer party read one declaration.

A message type is a tag and a version, and its `Schema` states its head's fields, its body's bounds and its kind: one of
the four public kinds (`structure`, `output`, `coins`, `commitment`) or `private`, which never enters a record. A
`MessageTypes` is the types one end of a channel accepts. A rule broken is refused by name (`Refused.name`, one of
`REFUSALS`): a type the registry doesn't hold (`unknown-type`), a version it doesn't (`unknown-version`), and fields other
than the schema's (`schema`); `Phase` refuses a message that isn't one the party expects next (`out-of-phase`). A head
is canonical JSON (`canonical`). The bytes on a channel, a message's frame and the parse of one (`malformed`,
`oversized`), are the runtime's (`verity.runtime.framing`). `Record` is a record entry: a message as it enters a run's
transcript, numbered, chained and signed by its sender.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

MAX_HEAD = 1 << 16
"""A head's default bound: 64 KiB. A message with more to say says it in its body."""
KINDS = ("structure", "output", "coins", "commitment", "private")
PUBLIC = frozenset(KINDS[:4])
REFUSALS = ("malformed", "oversized", "unknown-type", "unknown-version", "schema", "out-of-phase")
NAME = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,63}")
"""A message type, a party or a channel: a letter, then at most 63 letters, digits, '.', '_' and '-'."""
RESERVED = frozenset({"type", "v"})


class Refused(Exception):
    """A refusal by name: `name` says which rule, `detail` what broke it."""

    def __init__(self, name: str, detail: str) -> None:
        super().__init__(f"{name}: {detail}")
        self.name, self.detail = name, detail


def _plain(value: Any, path: str = "$") -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, Mapping):
        if any(type(k) is not str for k in value):
            raise TypeError(f"{path} has a key that isn't a string")
        return {k: _plain(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{path}[{i}]") for i, v in enumerate(value)]
    raise TypeError(f"{path} is a {type(value).__name__}, which canonical JSON doesn't hold (no floats: exact values only)")


def canonical(value: Any) -> bytes:
    """`value` as canonical JSON: the commitments' rules (`verity.commitments.identity.canonical_json_bytes`)."""
    return json.dumps(_plain(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


# --- schemas -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Int:
    lo: int = 0
    hi: int = (1 << 63) - 1

    def why(self, v: Any) -> str | None:
        return None if type(v) is int and self.lo <= v <= self.hi else f"an integer in [{self.lo}, {self.hi}]"


@dataclass(frozen=True)
class Bool:
    def why(self, v: Any) -> str | None:
        return None if type(v) is bool else "a boolean"


@dataclass(frozen=True)
class Str:
    max: int = 256
    pattern: str | None = None

    def why(self, v: Any) -> str | None:
        if type(v) is not str or len(v) > self.max or (self.pattern and not re.fullmatch(self.pattern, v)):
            return f"a string of at most {self.max} characters" + (f" matching {self.pattern}" if self.pattern else "")
        return None


@dataclass(frozen=True)
class Hex:
    """Bytes in lowercase hex: exactly `size` bytes, or at most `max`."""

    size: int | None = None
    max: int = 1 << 12

    def why(self, v: Any) -> str | None:
        n = self.size
        ok = (type(v) is str and len(v) % 2 == 0 and re.fullmatch(r"[0-9a-f]*", v) is not None
              and (len(v) // 2 == n if n is not None else len(v) // 2 <= self.max))
        return None if ok else (f"{n} bytes in lowercase hex" if n is not None else f"at most {self.max} bytes in hex")


@dataclass(frozen=True)
class List:
    item: Any
    max: int = 1 << 10

    def why(self, v: Any) -> str | None:
        if type(v) is not list or len(v) > self.max:
            return f"a list of at most {self.max}"
        bad = next((w for w in map(self.item.why, v) if w is not None), None)
        return None if bad is None else f"a list of {bad}"


@dataclass(frozen=True)
class Json:
    """Any canonical JSON value of at most `max` bytes: for a field whose shape its own module checks."""

    max: int = 1 << 12

    def why(self, v: Any) -> str | None:
        try:
            return None if len(canonical(v)) <= self.max else f"JSON of at most {self.max} bytes"
        except TypeError as e:
            return str(e)


@dataclass(frozen=True)
class Body:
    """A body's bounds: at most `max` bytes, a whole number of `unit`s, exactly `size` when set."""

    max: int = 0
    unit: int = 1
    size: int | None = None


@dataclass(frozen=True)
class Schema:
    """One message type at one version: its head's fields (beyond `type` and `v`), its body's bounds and its kind."""

    type: str
    version: int
    kind: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    body: Body = Body()

    def __post_init__(self) -> None:
        if not NAME.fullmatch(self.type) or type(self.version) is not int or self.version < 1:
            raise ValueError(f"a message type is a name ({NAME.pattern}) at a version from 1, not {self.type!r} v{self.version}")
        if self.kind not in KINDS:
            raise ValueError(f"{self.type}'s kind is one of {KINDS}, not {self.kind!r}")
        if RESERVED & set(self.fields):
            raise ValueError(f"{self.type} names a field {sorted(RESERVED & set(self.fields))}, which the head reserves")

    def why(self, head: Mapping[str, Any], body: bytes) -> tuple[str, str] | None:
        """(refusal, detail) when `head` and `body` aren't this schema's message, else None."""
        names = set(head) - RESERVED
        if names != set(self.fields):
            return "schema", f"{self.type} v{self.version} has fields {sorted(self.fields)}, not {sorted(names)}"
        for k, f in self.fields.items():
            if (w := f.why(head[k])) is not None:
                return "schema", f"{self.type}.{k} is {w}"
        b = self.body
        if len(body) > b.max:
            return "oversized", f"{self.type}'s body is at most {b.max} bytes, not {len(body)}"
        if len(body) % b.unit or (b.size is not None and len(body) != b.size):
            return "schema", f"{self.type}'s body is " + (f"{b.size} bytes" if b.size is not None else f"whole {b.unit}-byte units")
        return None


@dataclass(frozen=True)
class Message:
    """A message: its type and version, its head's fields, and its body."""

    type: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    body: bytes = b""
    version: int = 1

    def head(self) -> dict[str, Any]:
        return {**self.fields, "type": self.type, "v": self.version}


class MessageTypes:
    """The message types one end of a channel accepts, by type and version, and its bounds on a frame's head and body
    (`verity.runtime.framing` encodes and decodes under them)."""

    def __init__(self, schemas: Iterable[Schema] = (), max_head: int = MAX_HEAD) -> None:
        self.schemas: dict[tuple[str, int], Schema] = {}
        for s in schemas:
            if (s.type, s.version) in self.schemas and self.schemas[(s.type, s.version)] != s:
                raise ValueError(f"{s.type} v{s.version} is declared twice, differently")
            self.schemas[(s.type, s.version)] = s
        self.max_head = max_head
        self.max_body = max((s.body.max for s in self.schemas.values()), default=0)

    def __or__(self, other: MessageTypes) -> MessageTypes:
        return MessageTypes([*self.schemas.values(), *other.schemas.values()], max(self.max_head, other.max_head))

    def only(self, types: Iterable[str]) -> MessageTypes:
        """The schemas of `types`, every version."""
        keep = set(types)
        return MessageTypes([s for (t, _), s in self.schemas.items() if t in keep], self.max_head)

    @property
    def types(self) -> frozenset[str]:
        return frozenset(t for t, _ in self.schemas)

    def schema(self, type_: str, version: int) -> Schema:
        s = self.schemas.get((type_, version))
        if s is None:
            if type_ not in self.types:
                raise Refused("unknown-type", f"{type_!r} is none of {sorted(self.types)}")
            raise Refused("unknown-version", f"{type_} has versions {sorted(v for t, v in self.schemas if t == type_)}, "
                                             f"not {version!r}")
        return s

    def kind(self, type_: str, version: int = 1) -> str:
        return self.schema(type_, version).kind


class Phase:
    """The message types a party expects next: `check` refuses any other (`out-of-phase`)."""

    def __init__(self, *expected: str) -> None:
        self.expected = frozenset(expected)

    def expect(self, *types: str) -> None:
        self.expected = frozenset(types)

    def check(self, msg: Message) -> Message:
        if msg.type not in self.expected:
            raise Refused("out-of-phase", f"{msg.type} where {sorted(self.expected)} is expected")
        return msg


def names(values: Sequence[str]) -> tuple[str, ...]:
    """`values` as a tuple of names, each checked (`NAME`)."""
    bad = [v for v in values if type(v) is not str or not NAME.fullmatch(v)]
    if bad:
        raise ValueError(f"{bad} are not names ({NAME.pattern})")
    return tuple(values)


# --- the record --------------------------------------------------------------------------------------------------------


class Record:
    """`Core.Record` in Python: a record entry, one message of a run's transcript, numbered, chained to the entry before
    it and signed by the party that sent it (`verity/protocols/interface`, `CoreProtocol/Record.lean`).

        entry   = {body, kind, prev, schema, seq, session, sig}
        message = SHA-512([role] || I || [d])              d: the schema's tagged SHA-512 of the entry without its sig

    each `[x]` a u64 big-endian length then `x`, and `I` the signer's 16-byte LMS identifier.  `body` is the message, an
    object of the schema's form; `kind` its kind under the schema; `schema` the protocol's entry schema and version,
    which names the tag `d` is under; `session` the session it belongs to, None for an entry of none.  Entry `i` of a
    record has `seq = i` and `prev` the SHA-512 of entry `i - 1`'s line (`ZERO` for entry 0, `prev_of`), the line being
    the entry's `line`: sorted keys, no whitespace, every character past ASCII escaped.

    * `of_json`: `Record.ofJson`, the form: the seven keys, each of its type.  Everything a protocol accepts beyond the
      form (its schemas, kinds and bodies, which key signs, its sessions) is its own check; the auditor's ledger
      (`spot.challenger`) is the first.
    * `line`: an entry's line, as the auditor's ledger writes it (Lean `Spot.PyJson.canonical`).
    * `unsigned`, `message`, `prev_of`: `Record.unsigned`, `Record.message`, `Record.prevOf`.
    """

    FIELDS = ("body", "kind", "prev", "schema", "seq", "session", "sig")
    """A record entry's keys, sorted."""
    ZERO = "0" * 128
    """The `prev` of a record's first entry."""
    FORM = ("a record entry is {body, kind, prev, schema, seq, session, sig}: an object, a string, a SHA-512 in "
            "lowercase hex, a string, a natural number, a string or null, and lowercase hex")
    _PREV = re.compile(r"[0-9a-f]{128}")
    _HEX = re.compile(r"(?:[0-9a-f]{2})*")

    @staticmethod
    def of_json(e: Any) -> dict:
        """`e`, a parsed JSON value, when it is a record entry: its seven keys, each of its type; ValueError(`FORM`)
        otherwise."""
        if not (isinstance(e, dict) and sorted(e) == list(Record.FIELDS) and isinstance(e["body"], dict)
                and type(e["kind"]) is str and type(e["prev"]) is str and Record._PREV.fullmatch(e["prev"]) is not None
                and type(e["schema"]) is str and type(e["seq"]) is int and e["seq"] >= 0
                and (e["session"] is None or type(e["session"]) is str)
                and type(e["sig"]) is str and Record._HEX.fullmatch(e["sig"]) is not None):
            raise ValueError(Record.FORM)
        return e

    @staticmethod
    def line(entry: dict) -> str:
        """An entry's line: sorted keys, no whitespace, every character past ASCII escaped."""
        return json.dumps(entry, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def unsigned(entry: dict) -> dict:
        """An entry without its `sig`: what its signature's digest is over."""
        return {k: v for k, v in entry.items() if k != "sig"}

    @staticmethod
    def _field(b: bytes) -> bytes:
        return len(b).to_bytes(8, "big") + b

    @staticmethod
    def message(role: str, I: bytes, d: bytes) -> bytes:
        """What an entry's `sig` signs under the protocol's `role`, the signer's identifier `I` and `d`, the digest of
        the entry without its sig under the tag its schema names."""
        return hashlib.sha512(Record._field(role.encode()) + I + Record._field(d)).digest()

    @staticmethod
    def prev_of(line: str | bytes) -> str:
        """The `prev` of the entry after the one whose line is `line`."""
        return hashlib.sha512(line.encode() if isinstance(line, str) else line).hexdigest()


__all__ = ["KINDS", "MAX_HEAD", "NAME", "PUBLIC", "REFUSALS", "Body", "Bool", "Hex", "Int", "Json", "List", "Message",
           "Phase", "Record", "Refused", "MessageTypes", "Schema", "Str", "canonical", "names"]
