"""The network certifier's content rows: three rows of a connection it released, which hold what the connection
carried where its slots of the lengths and fails tables (`connection_rows`) hold how much, and the tables a
schedule's classes lay them in. A schedule has them only where its parameters say so (`WardenParams.content`, in a
schedule with lengths rows): a schedule without them has the tables, the head and the digest it had. The tables are
listed in the connection rows' one head, after `lengths` and `fails`, and signed with them, once
(`connection_rows`' "The signed head"): nothing here commits or signs.

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

**The classes** (`classes`): a schedule lists its classes in order. A class is its place in the list: two classes
may have the same sizes, and class i's tables are `request.i`, `response.i` and `end.i`, i from 0 (`names`). What a
class is beside its two sizes is the lengths rows' form's (`counted`):

- under form 1 a class is (P, O, rows): its two sizes and the rows of each of its three tables, at most the lengths
  table's slots, since an execution phase releases no more;
- under form 2 a class is (P, O) and its tables have no size of their own: each holds a row a release filed in the
  class and no other, at most R = `WardenParams.phase_releases` of them, a bound on an execution phase's releases.

A class of the other form's is refused where the schedule is built, with what to change (`sized` is each class with
the most rows its tables have under either form).

**The tables** (`tables`) of an execution phase are made from its releases in release order, each with its request's
bytes and its ids as the network certifier cut and released them (`warden.Released`), and from each release's class,
which is an argument. Row r of class i's three tables is the r-th of the execution phase's releases filed in class
i. A release filed in no class, or in a class whose rows are all held, is unfiled: it has no content row, and its
slots of the lengths and fails tables are the ones it had. A release filed in a class that doesn't hold it, its
request longer than `request_room` or its ids more than O, is refused, never cut short.

- Under form 1 every row after the last held one is zero bytes. So a table's size is the schedule's whatever the
  load, a class's held rows are its first, and how many are held is a value.
- Under form 2 a table is its held rows and nothing after them. A class's three tables have one count of rows, the
  releases filed in it, and that count is public wherever the tables are committed (`connection_rows`' "The signed
  head"); a class in which none is filed has three tables of no row. A class's rows are all held at R of them.

**The filing** (`file`): a release's class is the first the schedule lists that holds it. The rule reads the
release as the network certifier cut it and nothing a server says. It has one caller, `connection_rows.tables`, and
`tables` lays whatever class that caller gives.

**A row of zero bytes** is what every table of form 1 holds where no release is filed (`empty_rows`, the one
statement of it), and each table's decoder reads it its own way. `request_of` refuses it: its count is 0, and a
released request has a byte. `response_of` reads a response of no id, and `end_of` an `END` after no `DATA` cell. So
a held connection that ended before any id has the response and end rows of an unheld row, and of the three only
the request row says whether a row is held (`held`). Under form 2 no table has an unheld row.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from functools import lru_cache

from verity.circuits.partition_v2 import UNIT_CLASS_MAX
from verity.protocols.certifier.verifier.active import id_request
from verity.protocols.certifier.verifier.warden import Released, WardenParams
from verity.protocols.certifier.verifier.warden_rows import WORD_BITS

REQUEST, RESPONSE, END = "request", "response", "end"
#: the most classes a schedule lists: the head counts its tables in a u16, three a class after `lengths` and `fails`
MOST_CLASSES = ((1 << 16) - 3) // 3
#: a class's least size, of prompt ids or of ids released
MIN_SIZE = 2
#: a count's and an id's bytes: a u32
U32_BYTES = 4
#: an end row: a fails slot's layout (`connection_rows.SLOT`), u16 class ‖ u16 at
END_SLOT = struct.Struct("<HH")
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


def empty_rows(B: int, P: int, O: int) -> tuple[bytes, bytes, bytes]:
    """What a class's request, response and end tables hold where no release is filed: zero bytes, each of its
    table's row size (the module docstring's "A row of zero bytes")."""
    return bytes(request_bytes(B, P)), bytes(response_bytes(O)), bytes(END_BYTES)


def held(request: bytes) -> bool:
    """Whether a request row is a filed release's: its count is not 0, as an empty row's is."""
    return any(request[:U32_BYTES])


def counted(params: WardenParams) -> bool:
    """Whether a class's tables hold a row a release filed in it and no other, so that their rows are a count and
    no size of the schedule's (the module docstring's "The classes"): under the lengths rows' form 2."""
    return params.closes


def classes(params: WardenParams) -> tuple[tuple[int, ...], ...]:
    """A schedule's classes as its parameters hold them (the module docstring's "The classes"), each (P, O, rows)
    under the lengths rows' form 1 and (P, O) under form 2, or ValueError naming what is none. `WardenParams` calls
    it where it is built and keeps what it returns, so no schedule holds classes of another form."""
    if params.lengths is None:
        raise ValueError("the content rows' tables are listed in the connection rows' head, so a schedule with "
                         "content has lengths rows (lengths is 1 or 2)")
    got = params.content
    form, numbers = ("(P, O)", 2) if counted(params) else ("(P, O, rows)", 3)
    if not isinstance(got, (tuple, list)) or not 0 < len(got) <= MOST_CLASSES:
        raise ValueError(f"content is 1 to {MOST_CLASSES} classes, each {form}, or unset, not {got!r}")
    slots, most = params.window // params.every * params.stride_releases, params.ingress_cells * params.payload
    out = []
    for i, c in enumerate(got):
        if not isinstance(c, (tuple, list)) or len(c) not in (2, 3):
            raise ValueError(f"content class {i} is {form}, not {c!r}")
        if len(c) > numbers:
            raise ValueError(f"content class {i} is (P, O) under the lengths rows' form 2, not {c!r}: drop its third "
                             f"number, since each of its tables holds a row a release filed in it and no other "
                             f"(or lengths is 1, under which a class has rows of its own)")
        if len(c) < numbers:
            raise ValueError(f"content class {i} is (P, O, rows) under the lengths rows' form 1, not {c!r}: give "
                             f"the rows of each of its tables, 1 to {slots} (or lengths is 2, under which a class "
                             f"is (P, O) and its tables hold a row a release filed in it and no other)")
        P, O = c[:2]
        check_size(P, "prompt ids")
        check_ids(O)
        if P > most:
            raise ValueError(f"content class {i} has P={P} prompt ids, and a request of the schedule is at most "
                             f"{most} bytes (ingress_cells · a cell's payload), each id at least one")
        request_room(params.cell_bytes, P)
        if not counted(params) and (type(c[2]) is not int or not 0 < c[2] <= slots):
            raise ValueError(f"content class {i} has 1 to {slots} rows, the lengths table's slots, since an "
                             f"execution phase releases no more, not {c[2]!r}")
        out.append(tuple(c))
    return tuple(out)


def sized(params: WardenParams) -> tuple[tuple[int, int, int], ...]:
    """Each class of a schedule as (P, O, the most rows each of its tables has): the class's own rows under form 1
    and, where a table's rows are a count (`counted`), R = `WardenParams.phase_releases`. None for a schedule
    without content rows."""
    if params.content is None or not counted(params):
        return params.content or ()
    return tuple((P, O, params.phase_releases) for P, O in params.content)


def names(i: int) -> tuple[str, str, str]:
    """Class i's three tables as the head names them: `request.i`, `response.i` and `end.i`."""
    return f"{REQUEST}.{i}", f"{RESPONSE}.{i}", f"{END}.{i}"


def tables_of(params: WardenParams) -> tuple[tuple[str, int, int], ...]:
    """The content tables a schedule's head lists after the lengths and fails tables, class by class in the
    schedule's order: each one's (name, rows, a row's bytes), the rows the table's own under form 1 and the most it
    has where they are a count (`counted`). None for a schedule without content rows."""
    B = params.cell_bytes
    return tuple(t for i, (P, O, rows) in enumerate(sized(params))
                 for t in zip(names(i), (rows,) * 3, (request_bytes(B, P), response_bytes(O), END_BYTES)))


def layout(params: WardenParams, name: str) -> dict[str, int]:
    """What a content table's registration says of its rows' order: its class's place in the schedule's list and
    the class's two sizes. Row r is the r-th release of the execution phase filed in that class. ValueError for a
    name that is no content table of the schedule."""
    for i, (P, O, _) in enumerate(sized(params)):
        if name in names(i):
            return {"class": i, "prompt_ids": P, "ids": O}
    raise ValueError(f"the schedule has no content table {name!r}")


def ids_of(packed: bytes) -> tuple[int, ...]:
    """The ids a release carries (`warden.Released.ids`, a little-endian u32 each), in order."""
    if len(packed) % U32_BYTES:
        raise ValueError(f"a release's ids are {U32_BYTES} bytes each, not {len(packed)} bytes")
    return struct.unpack(f"<{len(packed) // U32_BYTES}I", packed)


def file(params: WardenParams, release: Released) -> int | None:
    """The class the network certifier files a release in (the module docstring's "The filing"): the first of the
    schedule's that holds its request's bytes and its ids, or None where none does."""
    B = params.cell_bytes
    for i, (P, O, _) in enumerate(sized(params)):
        if len(release.request) <= request_room(B, P) and len(release.ids) <= U32_BYTES * O:
            return i
    return None


def tables(params: WardenParams, released: Sequence[Released],
           filing: Sequence[int | None]) -> dict[str, list[bytes]]:
    """An execution phase's content tables (the module docstring's "The tables") from its releases in release order
    and each one's class in `filing`, None for a release filed in none: each table its held rows and, under form 1,
    zero rows to the class's rows. ValueError for a schedule without content rows, releases out of order or without
    the content of their lengths slots, a class the schedule doesn't list, or a release filed in a class that
    doesn't hold it."""
    if params.content is None:
        raise ValueError("the schedule has no content rows (WardenParams.content)")
    if len(filing) != len(released):
        raise ValueError(f"a class, or None, a release: {len(filing)} for {len(released)} releases")
    B, last, listed = params.cell_bytes, -1, sized(params)
    filed: list[list[Released]] = [[] for _ in listed]
    for r, i in zip(released, filing):
        if r.order <= last:
            raise ValueError(f"the releases are in release order, and {r.order} follows {last}")
        last = r.order
        if len(r.request) != r.request_bytes or len(r.ids) != U32_BYTES * r.outputs:
            raise ValueError(f"release {r.order} has {len(r.request)} bytes of its request's {r.request_bytes} and "
                             f"{len(r.ids)} bytes of ids for its {r.outputs} outputs, {U32_BYTES} each: not the "
                             f"content of its lengths slot")
        if i is None:
            continue
        if type(i) is not int or not 0 <= i < len(filed):
            raise ValueError(f"release {r.order} is filed in class {i!r}, and the schedule lists {len(filed)}")
        P, O, rows = listed[i]
        if len(r.request) > request_room(B, P) or r.outputs > O:
            raise ValueError(f"release {r.order}, of {len(r.request)} bytes and {r.outputs} ids, is filed in class "
                             f"{i}, which holds {request_room(B, P)} bytes and {O} ids")
        if len(filed[i]) < rows:
            filed[i].append(r)
    out: dict[str, list[bytes]] = {}
    for i, ((P, O, rows), mine) in enumerate(zip(listed, filed)):
        made = ([request_row(r.request, B, P) for r in mine], [response_row(ids_of(r.ids), O) for r in mine],
                [end_row(r.failed, r.outputs) for r in mine])
        for name, held_, empty in zip(names(i), made, empty_rows(B, P, O)):
            out[name] = held_ if counted(params) else held_ + [empty] * (rows - len(mine))
    return out


__all__ = ["END", "END_BYTES", "END_SLOT", "ENDED", "FAILED", "MIN_SIZE", "MOST_CLASSES", "NO_COUNT", "REQUEST",
           "RESPONSE", "TIE_WORDS", "U32_BYTES", "check_ids", "check_size", "classes", "counted", "empty_rows",
           "end_of", "end_row", "file", "held", "ids_of", "layout", "most_ids", "names", "request_bytes",
           "request_of", "request_room", "request_row", "response_bytes", "response_of", "response_row", "sized",
           "tables", "tables_of"]
