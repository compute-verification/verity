"""The messages the adapter hands the prover and takes from it, by their protocols' declarations (each protocol's
`prover/workload.py`): a body is checked against its declared type as it is written and as it is read, and its bytes
are left as they are. `tests/workload_messages.toml` lists each message's writers and readers.
"""

from __future__ import annotations

from pathlib import Path

from verity.channel import Message, MessageTypes, Refused
from verity.protocols.proofs.spot.prover import workload as spot
from verity.protocols.resource_accounting.compute.pouw.prover import workload as pouw
from verity.protocols.resource_accounting.memory.pous.prover import workload as pous
from verity.protocols.resource_accounting.network.prover import workload as certifier

__all__ = ["TYPES", "append", "body", "line", "lines", "read", "write"]

TYPES = MessageTypes((*spot.MESSAGES, *pouw.MESSAGES, *pous.MESSAGES, *certifier.MESSAGES))


def body(type_: str, data: bytes) -> bytes:
    """``data`` as the body of a ``type_`` message, unchanged; `verity.channel.Refused` when its declaration refuses
    it."""
    msg = Message(type_, body=data)
    if (why := TYPES.schema(msg.type, msg.version).why(msg.head(), msg.body)) is not None:
        raise Refused(*why)
    return data


def write(path: Path, type_: str, data: bytes) -> None:
    """Write ``data`` to ``path`` as a ``type_`` message's body."""
    path.write_bytes(body(type_, data))


def read(path: Path, type_: str) -> bytes:
    """``path``'s bytes, a ``type_`` message's body."""
    return body(type_, path.read_bytes())


def line(type_: str, data: bytes) -> bytes:
    """``data`` as one line of a JSON-lines file, a ``type_`` message's body."""
    return body(type_, data) + b"\n"


def append(path: Path, type_: str, data: bytes) -> None:
    """Append ``data`` to ``path`` as one line, a ``type_`` message's body."""
    with open(path, "ab") as fh:
        fh.write(line(type_, data))


def lines(path: Path, type_: str) -> list[bytes]:
    """``path``'s lines but blank ones, each a ``type_`` message's body."""
    return [body(type_, line) for line in path.read_bytes().splitlines() if line.strip()]
