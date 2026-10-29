"""``verity/rec-links/v1``: the link table of a recursive session's inner population, as the prover builds and commits it, and
the mirror of its reference check (``Flock/RecLinks.lean``, `Flock.RecLinks.check`).

Every row of the inner population is committed under its own salt, so no two commitments show that their rows are equal,
and which unit reads which is private. A link says that input row ``port`` of unit ``reader`` holds, word for word, output
row ``out`` of unit ``writer``. The table holds one link a row of `ROW_BYTES` bytes (`KIND_LINK`, ``reader``, ``port``,
``writer``, ``out``, each a u32 little-endian, then zeros), the links first, then all-zero filler rows, ``grid(L)`` rows in
all (`table`; `verity.ml.circuits.partition_v2.grid` at `GRID_BITS`). `register` commits them as advice before any coin
(`verity.ml.commitments.advice`: ``hm96-sha512/row/v2`` hiding leaves in one ``frame-v3-sha512`` tree under `BINDING`), so
what is public of the table is ``grid(L)`` and the root.

`check` is the Lean check's mirror, refusal for refusal: it raises `Refused` with the Lean's code, in the Lean's order
(`CODES`). `vectors` writes the cases ``flock-rec-links`` holds the two to.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Callable, Sequence

from verity.ml.circuits.partition_v2 import grid as _grid
from verity.ml.commitments import advice

FORMAT = "verity/rec-links/v1"
BINDING = hashlib.sha512(FORMAT.encode()).digest()
ROW_BYTES = 128
GRID_BITS = 4
KIND_LINK = 1
FIELD_BYTES = 20
#: `Flock.RecLinks.check`'s refusals, in the order it checks them
CODES = ("row-size", "row-kind", "row-nonzero", "order", "empty", "count", "range", "width", "twice", "unequal")
VECTORS_FORMAT = "verity/rec-links-vectors/v1"


class Refused(ValueError):
    """A table `check` refuses, with the Lean's code."""

    def __init__(self, code: str, why: str):
        assert code in CODES, code
        super().__init__(f"{code}: {why}")
        self.code = code


@dataclass(frozen=True)
class Link:
    reader: int
    port: int
    writer: int
    out: int


@dataclass(frozen=True)
class Shape:
    """The inner population's public shape: its units, and each input row's and each output row's words."""

    units: int
    ins: tuple[int, ...]
    outs: tuple[int, ...]


def grid(x: int) -> int:
    return _grid(x, GRID_BITS)


def encode(link: Link) -> bytes:
    fields = (KIND_LINK, link.reader, link.port, link.writer, link.out)
    if any(not 0 <= f < 2 ** 32 for f in fields):
        raise ValueError(f"a link's fields are u32s: {link}")
    return struct.pack("<5I", *fields) + bytes(ROW_BYTES - FIELD_BYTES)


def table(links: Sequence[Link]) -> list[bytes]:
    """The table's rows: the links, then fillers up to ``grid(L)``."""
    if not links:
        raise ValueError("a table without links is no table")
    return [encode(link) for link in links] + [bytes(ROW_BYTES)] * (grid(len(links)) - len(links))


def register(links: Sequence[Link], salts: Sequence[bytes] | None = None) -> advice.Registration:
    """The table committed as advice; its root and row count are what goes public."""
    return advice.commit(table(links), 8 * ROW_BYTES, BINDING, salts)


def decode(row: bytes) -> Link | None:
    if len(row) != ROW_BYTES:
        raise Refused("row-size", f"a row of {len(row)} bytes, not {ROW_BYTES}")
    kind, reader, port, writer, out = struct.unpack_from("<5I", row)
    if kind == 0:
        if any(row):
            raise Refused("row-nonzero", "a filler row is not all zero")
        return None
    if kind == KIND_LINK:
        if any(row[FIELD_BYTES:]):
            raise Refused("row-nonzero", "a link row's bytes past its fields are not zero")
        return Link(reader, port, writer, out)
    raise Refused("row-kind", f"a row of kind {kind}, neither a link ({KIND_LINK}) nor a filler (0)")


Rows = Callable[[int, int], Sequence[int]]


def check(sh: Shape, in_row: Rows, out_row: Rows, rows: Sequence[bytes]) -> int:
    """`Flock.RecLinks.check`: the link count, or `Refused`."""
    xs = [decode(r) for r in rows]
    first_filler = next((i for i, x in enumerate(xs) if x is None), len(xs))
    if any(x is not None for x in xs[first_filler:]):
        raise Refused("order", "a link row after a filler row")
    links = [x for x in xs if x is not None]
    if not links:
        raise Refused("empty", "a table without links is no table")
    if len(rows) != grid(len(links)):
        raise Refused("count", f"{len(rows)} rows, not the {len(links)} links' grid {grid(len(links))}")
    seen: set[tuple[int, int]] = set()
    for k, link in enumerate(links):
        if not (link.reader < sh.units and link.writer < sh.units and link.port < len(sh.ins) and link.out < len(sh.outs)):
            raise Refused("range", f"link {k} is outside the population")
        if sh.ins[link.port] != sh.outs[link.out]:
            raise Refused("width", f"link {k}'s rows have {sh.ins[link.port]} and {sh.outs[link.out]} words")
        if (link.reader, link.port) in seen:
            raise Refused("twice", f"link {k} reads an input row an earlier link reads")
        if list(in_row(link.reader, link.port)) != list(out_row(link.writer, link.out)):
            raise Refused("unequal", f"link {k}'s reader row is not its writer row")
        seen.add((link.reader, link.port))
    return len(links)


def _population(sh: Shape, seed: int) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    """Every unit's input and output rows, words from a fixed LCG."""
    state = seed
    def word() -> int:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2 ** 64
        return state >> 48
    outs = [[[word() for _ in range(w)] for w in sh.outs] for _ in range(sh.units)]
    ins = [[[word() for _ in range(w)] for w in sh.ins] for _ in range(sh.units)]
    return ins, outs


def _chain(sh: Shape, seed: int) -> tuple[list[Link], list[list[list[int]]], list[list[list[int]]]]:
    """A chain: unit u + 1's input row 0 reads unit u's output row 0, its values copied, so every link holds."""
    ins, outs = _population(sh, seed)
    links = [Link(u + 1, 0, u, 0) for u in range(sh.units - 1)]
    for link in links:
        ins[link.reader][link.port] = list(outs[link.writer][link.out])
    return links, ins, outs


def vectors() -> dict:
    """The difftest's cases: honest tables at counts on and off the grid, and one table for each refusal."""
    sh = Shape(units=40, ins=(4, 3), outs=(4, 2))
    links, ins, outs = _chain(sh, 1)
    cases: list[tuple[str, Shape, list[list[list[int]]], list[list[list[int]]], list[bytes]]] = []

    def add(name: str, rows: list[bytes], shape: Shape = sh, i=None) -> None:
        cases.append((name, shape, i if i is not None else ins, outs, rows))

    for n in (1, 2, 16, 17, 33, 39):
        add(f"honest-{n}", table(links[:n]))
    fan = [Link(u, 0, 0, 0) for u in range(1, 18)]
    fan_ins = [[list(r) for r in unit] for unit in ins]
    for link in fan:
        fan_ins[link.reader][link.port] = list(outs[0][0])
    add("honest-fan-out", table(fan), i=fan_ins)
    rows = table(links[:17])
    add("row-size", rows[:3] + [rows[3][:-1]] + rows[4:])
    add("row-kind", rows[:2] + [struct.pack("<I", 2) + rows[2][4:]] + rows[3:])
    add("row-nonzero-link", rows[:5] + [rows[5][:-1] + b"\1"] + rows[6:])
    add("row-nonzero-filler", rows[:-1] + [bytes(ROW_BYTES - 1) + b"\1"])
    add("order", rows[:3] + [bytes(ROW_BYTES)] + rows[4:17] + [rows[3]])
    add("empty", [bytes(ROW_BYTES)] * 16)
    add("count", rows + [bytes(ROW_BYTES)] * 2)
    add("count-short", rows[:17])
    add("range-unit", table(links[:3] + [Link(40, 0, 1, 0)]))
    add("range-port", table(links[:3] + [Link(5, 2, 4, 0)]))
    add("range-out", table(links[:3] + [Link(5, 0, 4, 2)]))
    add("width", table(links[:3] + [Link(5, 1, 4, 0)]))
    add("twice", table(links[:3] + [links[1]]))
    add("unequal", table(links[:3] + [Link(5, 0, 7, 0)]))
    bent = [[list(r) for r in unit] for unit in ins]
    bent[9][0][3] ^= 1
    add("unequal-word", table(links[:12]), i=bent)
    return {"format": VECTORS_FORMAT, "cases": [
        {"name": name, "units": shape.units, "ins": list(shape.ins), "outs": list(shape.outs), "in_rows": i, "out_rows": o,
         "rows": [r.hex() for r in rs]} for name, shape, i, o, rs in cases]}


def verdict(case: dict) -> str:
    """A vector's verdict as ``flock-rec-links`` prints it: ``NAME ok L`` or ``NAME refused CODE``."""
    sh = Shape(case["units"], tuple(case["ins"]), tuple(case["outs"]))
    ins, outs = case["in_rows"], case["out_rows"]
    try:
        n = check(sh, lambda u, p: ins[u][p], lambda u, p: outs[u][p], [bytes.fromhex(r) for r in case["rows"]])
    except Refused as e:
        return f"{case['name']} refused {e.code}"
    return f"{case['name']} ok {n}"
