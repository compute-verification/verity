"""The wire every party speaks: frames, messages and their schemas, and refusals by name. Stdlib only, and it imports no
other unit, so an auditor party and a developer party parse with the same code (`note:memory-accounting/20261026T2305Z-
draft-deployment-api-v3` §2).

~~~text
frame     u32be len(head) ‖ u32be len(body) ‖ head ‖ body
head      canonical JSON (the commitments' rules: sorted keys, no whitespace, UTF-8, no floats) of one object,
          {"type": T, "v": V, ...fields}
body      the message's bytes, never base64
~~~

The frame is `sampled_proofs.remote`'s, with a body. A message type is a tag and a version, and its `Schema` states its
head's fields, its body's bounds and its kind: one of the four public kinds (`structure`, `output`, `coins`,
`commitment`) or `private`, which never enters a record. `Registry.decode` parses every frame as adversarial and refuses
by name (`Refused.name`, one of `REFUSALS`): a frame that isn't one (`malformed`), a head or body past its bound
(`oversized`), a type the registry doesn't hold (`unknown-type`), a version it doesn't (`unknown-version`), and fields
other than the schema's (`schema`). `Phase` refuses a message that isn't one the party expects next (`out-of-phase`).
The head must be the canonical encoding of what it says, so one message has one frame. `handshake` is the exchange that
opens every channel between parties.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

WIRE_VERSION = 1
HEADER = struct.Struct(">II")
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
    """`value` as canonical JSON: the commitments' rules (`verity.ml.commitments.identity.canonical_json_bytes`)."""
    return json.dumps(_plain(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError("an object repeats a key")
    return dict(pairs)


def _no_float(s: str) -> Any:
    raise ValueError(f"a number with a fraction or an exponent ({s[:32]})")


def _no_constant(s: str) -> Any:
    raise ValueError(f"{s} is not JSON")


def parse(data: bytes) -> Any:
    """The canonical JSON `data` encodes; anything else (another encoding of it included) raises `Refused('malformed')`."""
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_float=_no_float,
                           parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise Refused("malformed", f"the head is not JSON ({e})") from None
    if canonical(value) != data:
        raise Refused("malformed", "the head is not in canonical form")
    return value


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


def frame(head: bytes, body: bytes = b"") -> bytes:
    return HEADER.pack(len(head), len(body)) + head + body


def split(data: bytes, max_head: int = MAX_HEAD, max_body: int = 1 << 32) -> tuple[bytes, bytes]:
    """A frame's head and body, its header checked against the bounds before anything else."""
    if len(data) < HEADER.size:
        raise Refused("malformed", f"a frame is at least {HEADER.size} bytes, not {len(data)}")
    nh, nb = HEADER.unpack_from(data)
    if nh > max_head or nb > max_body:
        raise Refused("oversized", f"a frame is a head of at most {max_head} bytes and a body of at most {max_body}, not "
                                   f"{nh} + {nb}")
    if len(data) != HEADER.size + nh + nb:
        raise Refused("malformed", f"a frame of {nh} + {nb} bytes is {len(data) - HEADER.size} bytes long")
    return data[HEADER.size:HEADER.size + nh], data[HEADER.size + nh:]


class Registry:
    """The message types one end of a channel accepts, by type and version."""

    def __init__(self, schemas: Iterable[Schema] = (), max_head: int = MAX_HEAD) -> None:
        self.schemas: dict[tuple[str, int], Schema] = {}
        for s in schemas:
            if (s.type, s.version) in self.schemas and self.schemas[(s.type, s.version)] != s:
                raise ValueError(f"{s.type} v{s.version} is declared twice, differently")
            self.schemas[(s.type, s.version)] = s
        self.max_head = max_head
        self.max_body = max((s.body.max for s in self.schemas.values()), default=0)

    def __or__(self, other: Registry) -> Registry:
        return Registry([*self.schemas.values(), *other.schemas.values()], max(self.max_head, other.max_head))

    def only(self, types: Iterable[str]) -> Registry:
        """The schemas of `types`, every version."""
        keep = set(types)
        return Registry([s for (t, _), s in self.schemas.items() if t in keep], self.max_head)

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

    def encode(self, msg: Message) -> bytes:
        """`msg`'s frame, checked against its schema as `decode` checks it."""
        head = msg.head()
        s = self.schema(msg.type, msg.version)
        if (why := s.why(head, msg.body)) is not None:
            raise Refused(*why)
        h = canonical(head)
        if len(h) > self.max_head:
            raise Refused("oversized", f"{msg.type}'s head is {len(h)} bytes, past {self.max_head}")
        return frame(h, bytes(msg.body))

    def decode(self, data: bytes) -> Message:
        """The message `data` frames, parsed as adversarial: each rule broken is refused by name (`Refused`)."""
        h, body = split(data, self.max_head, self.max_body)
        head = parse(h)
        if type(head) is not dict or type(head.get("type")) is not str:
            raise Refused("malformed", "a head is an object with a string type")
        t, v = head["type"], head.get("v")
        if t not in self.types:
            raise Refused("unknown-type", f"{t!r} is none of {sorted(self.types)}")
        if type(v) is not int:
            raise Refused("unknown-version", f"{t}'s version is an integer, not {v!r}")
        s = self.schema(t, v)
        if (why := s.why(head, body)) is not None:
            raise Refused(*why)
        return Message(t, {k: head[k] for k in s.fields}, bytes(body), v)


class FrameReader:
    """Frames out of a byte stream: `feed` the bytes as they arrive, `next` each whole frame. A header past the bounds
    is refused as soon as it arrives (`oversized`), before its body is buffered."""

    def __init__(self, max_head: int = MAX_HEAD, max_body: int = 1 << 32) -> None:
        self.max_head, self.max_body, self._buf = max_head, max_body, bytearray()

    def feed(self, data: bytes) -> None:
        self._buf += data

    @property
    def pending(self) -> int:
        return len(self._buf)

    def next(self) -> bytes | None:
        if len(self._buf) < HEADER.size:
            return None
        nh, nb = HEADER.unpack_from(self._buf)
        if nh > self.max_head or nb > self.max_body:
            raise Refused("oversized", f"a frame is a head of at most {self.max_head} bytes and a body of at most "
                                       f"{self.max_body}, not {nh} + {nb}")
        n = HEADER.size + nh + nb
        if len(self._buf) < n:
            return None
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out


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


def frames(data: bytes, max_head: int = MAX_HEAD, max_body: int = 1 << 32) -> list[bytes]:
    """Every frame of `data`, which holds whole frames back to back."""
    r = FrameReader(max_head, max_body)
    r.feed(data)
    out = []
    while (f := r.next()) is not None:
        out.append(f)
    if r.pending:
        raise Refused("malformed", f"{r.pending} bytes after the last whole frame")
    return out


def names(values: Sequence[str]) -> tuple[str, ...]:
    """`values` as a tuple of names, each checked (`NAME`)."""
    bad = [v for v in values if type(v) is not str or not NAME.fullmatch(v)]
    if bad:
        raise ValueError(f"{bad} are not names ({NAME.pattern})")
    return tuple(values)


__all__ = ["HEADER", "KINDS", "MAX_HEAD", "NAME", "PUBLIC", "REFUSALS", "WIRE_VERSION", "Body", "Bool", "FrameReader",
           "Hex", "Int", "Json", "List", "Message", "Phase", "Refused", "Registry", "Schema", "Str", "canonical", "frame",
           "frames", "names", "parse", "split"]
