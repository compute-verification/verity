"""The network certifier's content rows: the formats of three rows of a connection it released, which hold what the
connection carried where its slots of the lengths and fails tables (`connection_rows`) hold how much. Only the formats
are here, with each row's size. No table, head or registration holds such a row and the network certifier writes
none: `WardenParams` has no form for them, and a schedule's records, heads and certificates are the ones it had.

**The three rows** of one released connection, every field little-endian:

- `request`: u32 bytes ‖ the request's bytes as the network certifier cut them, then zero bytes to the row's size.
- `response`: u32 m ‖ O u32 ids. m is the `DATA` cells the network certifier released on the connection before its
  `END` or its `FAIL`, one id each (`io_tie.parse_cell`), word p is cell p's id for p below m, and words m to O − 1
  are 0. The count is in the row because an id may be 0: the ids alone don't say where they end.
- `end`: u16 class ‖ u16 at, in a fails slot's layout (`connection_rows.SLOT`): (0, n) where the connection ended in
  its `END` after n `DATA` cells and (1, n) where it ended in a `FAIL` after n. `at` is the `DATA` cells the network
  certifier released, on an `END` too, where the fails table's own slot counts them before a `FAIL` alone.

**A row's size is its class's**, so which class a connection is in is the one thing its rows' sizes say. A class
is a size of prompt ids P and a size of ids released O, each a power of two of at least `MIN_SIZE`:

- a response row is 4·(O + 1) bytes (`response_bytes`);
- a request row is 4 bytes and the longest request of at most P prompt ids, to a whole 16-bit word
  (`request_bytes`): the canonical id request of P ids in the schedule's B-byte cells (`active.id_request` at offset
  0), whose length is a function of P and B alone and grows with P (`request_room`);
- an end row is 4 bytes (`END_BYTES`).

**A response row's ids** (`check_ids`): the response tie's proof block
(`verity.protocols.certifier.prover.response_tie`) returns the whole row after its verdict, and a proof block's
outputs fit one public size class, so O is at most the greatest power of two with 16 + 32·(O + 1) bits in the widest
class. It is the one statement of that bound, which the proof block's shape is held to. A count past it, `NO_COUNT`
among them, is no response row's (`response_of` refuses it).

A decoder (`request_of`, `response_of`, `end_of`) gives back what its encoder was given and refuses, naming the
field, bytes that are not a row of the form: another size, a count past the row, anything past the count, an end
row whose class is neither 0 nor 1.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from functools import lru_cache

from verity.circuits.partition_v2 import UNIT_CLASS_MAX
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier.active import id_request
from verity.protocols.certifier.verifier.warden_rows import WORD_BITS

REQUEST, RESPONSE, END = "request", "response", "end"
#: a class's least size, of prompt ids or of ids released
MIN_SIZE = 2
#: a count's and an id's bytes: a u32
U32_BYTES = 4
#: an end row: a fails slot's layout, u16 class ‖ u16 at
END_SLOT = CR.SLOT[CR.FAILS]
END_BYTES = END_SLOT.size
#: how a connection ended, an end row's class: in its `END`, or in a `FAIL`
ENDED, FAILED = 0, 1
#: a count no response row has, since a row's count is at most its ids: what the response tie's proof block returns
#: as the count of a row it refuses
NO_COUNT = 0xFFFF_FFFF
#: the words the response tie's proof block returns before the row: its verdict
TIE_WORDS = 1


def check_size(size: object, what: str) -> None:
    """ValueError unless `size` is a class's size of `what`: a power of two of at least `MIN_SIZE`."""
    if type(size) is not int or size < MIN_SIZE or size & (size - 1):
        raise ValueError(f"a class's size of {what} is a power of two of at least {MIN_SIZE}, not {size!r}")


def most_ids() -> int:
    """The most ids a response row has (the module docstring's "A response row's ids")."""
    fit = (UNIT_CLASS_MAX - TIE_WORDS * WORD_BITS) // (8 * U32_BYTES) - 1
    return 1 << fit.bit_length() - 1 if fit >= 1 else 0


def check_ids(O: object) -> None:
    """ValueError unless a response row of O ids is one its tie's proof block returns: a class's size of at most
    `most_ids`."""
    check_size(O, "ids released")
    if O > most_ids():
        raise ValueError(f"a response row is at most {most_ids()} ids, not O={O!r}: its tie's proof block returns the "
                         f"whole row, {8 * U32_BYTES} bits an id after its count, behind {TIE_WORDS * WORD_BITS} bits "
                         f"of verdict in a public size class of at most {UNIT_CLASS_MAX} bits; a schedule whose "
                         f"responses are longer has no response row (fewer egress_cells or a lower max_tokens_cap)")


def response_bytes(O: int) -> int:
    """A response row's bytes in a class of O ids released: its count and its ids."""
    check_ids(O)
    return U32_BYTES * (O + 1)


@lru_cache(maxsize=None)
def request_room(B: int, P: int) -> int:
    """The bytes of the longest request a class of P prompt ids holds in B-byte cells: the canonical id request of P
    ids (`active.id_request` at offset 0), whatever its other fields. ValueError for a cell an id request doesn't
    fit."""
    check_size(P, "prompt ids")
    if type(B) is not int:
        raise ValueError(f"a cell's size is an integer, not {B!r}")
    return len(id_request(model="", user="", prompt=[0] * P, max_tokens=0, frame_bytes=B, offset=0))


def request_bytes(B: int, P: int) -> int:
    """A request row's bytes in a class of P prompt ids at B-byte cells: its count and `request_room` bytes, to a
    whole 16-bit word."""
    size = U32_BYTES + request_room(B, P)
    return size + -size % (WORD_BITS // 8)


def request_row(request: bytes, B: int, P: int) -> bytes:
    """A request's row in a class of P prompt ids at B-byte cells. ValueError for a request of no byte or of more
    than the class holds."""
    room = request_room(B, P)
    if not isinstance(request, (bytes, bytearray)) or not 0 < len(request) <= room:
        raise ValueError(f"a request of a class of {P} prompt ids at {B}-byte cells is 1 to {room} bytes, not "
                         f"{len(request) if isinstance(request, (bytes, bytearray)) else request!r}")
    return (len(request).to_bytes(U32_BYTES, "little") + bytes(request)).ljust(request_bytes(B, P), b"\0")


def request_of(row: bytes, B: int, P: int) -> bytes:
    """The request a request row of the class holds, or ValueError naming how the bytes are no such row."""
    size, room = request_bytes(B, P), request_room(B, P)
    if len(row) != size:
        raise ValueError(f"a request row of a class of {P} prompt ids at {B}-byte cells is {size} bytes, not "
                         f"{len(row)}")
    n = int.from_bytes(row[:U32_BYTES], "little")
    if not 0 < n <= room:
        raise ValueError(f"the request row's bytes is {n}, not 1 to the class's {room}")
    if row[U32_BYTES + n:].strip(b"\0"):
        raise ValueError(f"the request row is not zero past its {n} bytes")
    return row[U32_BYTES:U32_BYTES + n]


def response_row(ids: Sequence[int], O: int) -> bytes:
    """The response row, in a class of O ids released, of a connection whose released `DATA` cells held `ids`, in
    order. ValueError for more ids than the class holds, or an id that is no u32."""
    check_ids(O)
    ids = list(ids)
    if len(ids) > O or any(type(t) is not int or not 0 <= t < 1 << 32 for t in ids):
        raise ValueError(f"a response row of a class of {O} ids holds at most {O} ids, each a u32, not "
                         f"{len(ids)}: {ids[:4]}")
    return struct.pack(f"<{O + 1}I", len(ids), *ids, *[0] * (O - len(ids)))


def response_of(row: bytes, O: int) -> tuple[int, ...]:
    """The ids a response row of the class holds, in order, or ValueError naming how the bytes are no such row."""
    if len(row) != response_bytes(O):
        raise ValueError(f"a response row of a class of {O} ids is {response_bytes(O)} bytes, not {len(row)}")
    m, *ids = struct.unpack(f"<{O + 1}I", row)
    if m > O:
        raise ValueError(f"the response row's count is {m}, past its {O} ids")
    if any(ids[m:]):
        raise ValueError(f"the response row is not zero past its {m} ids")
    return tuple(ids[:m])


def end_row(failed: bool, cells: int) -> bytes:
    """The end row of a connection that released `cells` `DATA` cells and then ended: in a `FAIL` (`failed`), (1,
    `cells`), or in its `END`, (0, `cells`). ValueError for a count that is no u16."""
    if type(failed) is not bool or type(cells) is not int or not 0 <= cells < 1 << 16:
        raise ValueError(f"a connection ended in a FAIL or not, after a u16 of DATA cells, not {failed!r}, {cells!r}")
    return END_SLOT.pack(FAILED if failed else ENDED, cells)


def end_of(row: bytes) -> tuple[bool, int]:
    """An end row's reading: whether the connection ended in a `FAIL`, and the `DATA` cells released before it or
    before its `END`. ValueError naming how the bytes are no end row."""
    if len(row) != END_BYTES:
        raise ValueError(f"an end row is {END_BYTES} bytes, not {len(row)}")
    cls, at = END_SLOT.unpack(row)
    if cls not in (ENDED, FAILED):
        raise ValueError(f"the end row's class is {cls}, not {ENDED} or {FAILED}")
    return cls == FAILED, at


__all__ = ["END", "END_BYTES", "END_SLOT", "ENDED", "FAILED", "MIN_SIZE", "NO_COUNT", "REQUEST", "RESPONSE",
           "TIE_WORDS", "U32_BYTES", "check_ids", "check_size", "end_of", "end_row", "most_ids", "request_bytes",
           "request_of", "request_room", "request_row", "response_bytes", "response_of", "response_row"]
