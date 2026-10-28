"""A live statement's tables and reads (`verity.ml.protocols.pouw.circuit.live`'s ``live@<s>``), with no Program
built: the registered values it reads, their rows' word layout, the known ``m@<s>``, ``round@<s>``, ``caps`` and
``live-ok`` rows, the statement's port rows and its verifier ``--registered`` file. Credit reads a live statement
through these, and the served path registers its tables in their layout; `circuit.live` holds the unit's Definition and
the live Program."""
from __future__ import annotations

import functools
from typing import Any, Sequence

from verity.core.protocols.sampled_proofs.one_stage import registered as G

#: a request header's u32 fields: index, round, first slot, cap and prompt length
HEADER = 5
#: the live Program's parameters after its ``PREFIX``, and the v2 Program's (`live.program_v2`)
ROLES = ("slots", "requests", "m", "round")
ROLES_V2 = (*ROLES, "caps")
#: a slot leaf's and a request header's u16 words a row (u32 little-endian, low half first)
SLOT_WORDS, HEADER_WORDS = 2, 2 * HEADER
#: the statement's output row of the unit's ok, which reads the known ``<prefix>live-ok``
OK_PORT = "o0"
OK_VALUE = "live-ok"
#: the statement's output rows, as C-Flock's unit stager takes them: the unit's one output, its ok
STATEMENT_OUT_ROWS = ((G.SCHEMA_V2, (0,)),)
#: ``requests@0``'s row format: a statement needs a row whose commitment compresses before its last SHA-512 block (C-Flock's
#: and Lean's ``sha512x3`` range holds no slot otherwise), and a ``row/v2`` row of at most 111 bytes has none, so the headers
#: are ``row-seg/v1`` rows; ``slots@<s>``, ``m@<s>``, ``round@<s>``, ``caps`` and ``live-ok`` are ``row/v2`` rows, as known
#: content is
REQUEST_SCHEMA = G.SCHEMA_SEG
SLOTS, REQUESTS, M_ROWS, ROUND_ROWS = "slots", "requests", "m", "round"
#: the v2 unit's public caps, one table every round reads (``<prefix>caps``, its port ``cap_<i>`` row i)
CAPS = "caps"


def _cc() -> Any:
    """`core_check`, imported only when a known table is built: `circuit` (which circuit-check loads) imports this module
    for `HEADER`, and core_check brings the verifier's challenger with it."""
    from verity.core.protocols.pouw import core_check

    return core_check


def table(prefix: str, name: str, s: int = 0) -> str:
    """A live statement's registered value: round s's slot table ``<prefix>slots@<s>`` (`SLOTS`), round 0's request table
    ``<prefix>requests@0`` (`REQUESTS`), round s's public m ``<prefix>m@<s>`` (`M_ROWS`), its public round
    ``<prefix>round@<s>`` (`ROUND_ROWS`), the v2 unit's public caps ``<prefix>caps`` (`CAPS`) or the ok rows
    ``<prefix>live-ok`` (`OK_VALUE`)."""
    if name in (OK_VALUE, CAPS):
        return f"{prefix}{name}"
    if name not in (SLOTS, REQUESTS, M_ROWS, ROUND_ROWS) or (name == REQUESTS and s != 0):
        raise ValueError(f"a live statement's tables are {SLOTS}@<s>, {REQUESTS}@0, {M_ROWS}@<s>, {ROUND_ROWS}@<s>, "
                         f"{CAPS} and {OK_VALUE}, not {name}@{s}")
    return f"{prefix}{name}@{int(s)}"


def slot_words(leaves: Sequence[int]) -> list[list[int]]:
    """Slot leaves (each a u32: q + 1, or 0) as ``slots@<s>``'s rows of u16 words."""
    return [[v & 0xFFFF, v >> 16] for v in leaves]


def header_words(headers: Sequence[Sequence[int]]) -> list[list[int]]:
    """Request headers (each `HEADER` u32) as ``requests@0``'s rows of u16 words."""
    return [[w for v in h for w in (v & 0xFFFF, v >> 16)] for h in headers]


def m_table(program: str, prefix: str, window: str, s: int, m: int) -> tuple[list[list[int]], list[bytes], G.Registered]:
    """Round s's public m ``<prefix>m@<s>``: one row, m's two u16 words, under `core_check.public_row_salt` of the window's
    open root `window` and s, registered under `program` binding `window`. Every input is public."""
    if not (isinstance(m, int) and 0 <= m < 1 << 32):
        raise ValueError(f"m is a u32, not {m!r}")
    rows, salts = slot_words([m]), [_cc().public_row_salt(window, int(s))]
    return rows, salts, G.Registered.commit(program, table(prefix, M_ROWS, s), rows, salts, G.WORD_BITS, G.SCHEMA_V2,
                                            window=window)


def round_table(program: str, prefix: str, window: str, s: int) -> tuple[list[list[int]], list[bytes], G.Registered]:
    """Round s's public round ``<prefix>round@<s>``, the unit's ``s``: one row, s's two u16 words, under
    `core_check.public_row_salt` of the window's open root `window` and s, registered under `program` binding `window`.
    Every input is public."""
    if not (isinstance(s, int) and not isinstance(s, bool) and 0 <= s < 1 << 32):
        raise ValueError(f"s is a u32, not {s!r}")
    rows, salts = slot_words([s]), [_cc().public_row_salt(window, s)]
    return rows, salts, G.Registered.commit(program, table(prefix, ROUND_ROWS, s), rows, salts, G.WORD_BITS,
                                            G.SCHEMA_V2, window=window)


def caps(max_tokens: Sequence[int], T: int) -> list[int]:
    """The v2 unit's public caps at T leaves: leaf q's the public max_tokens of request q (`credit.committed_length`, in
    admission order), 0 past them; a list longer than T is cut at T."""
    return [int(max_tokens[q]) if q < len(max_tokens) else 0 for q in range(T)]


def caps_table(program: str, prefix: str, window: str, T: int, max_tokens: Sequence[int]
               ) -> tuple[list[list[int]], list[bytes], G.Registered]:
    """The v2 unit's public caps ``<prefix>caps``: T rows, row q `caps`' leaf q as two u16 words, under
    `core_check.public_row_salt` of the window's open root `window` and q, registered under `program` binding `window`.
    Every input is public: the caps are the max_tokens the period's open committed."""
    cs = caps(max_tokens, T)
    if not all(isinstance(c, int) and 0 <= c < 1 << 32 for c in cs):
        raise ValueError(f"each public cap is a u32, not {cs!r}")
    rows, salts = slot_words(cs), [_cc().public_row_salt(window, q) for q in range(T)]
    return rows, salts, G.Registered.commit(program, table(prefix, CAPS), rows, salts, G.WORD_BITS, G.SCHEMA_V2,
                                            window=window)


@functools.lru_cache(maxsize=8)
def ok_table(program: str, prefix: str, window: str, rounds: int) -> tuple[list[list[int]], list[bytes], G.Registered]:
    """The live units' ok rows ``<prefix>live-ok``: row s the one word 1 (`core_check.OK_ROW`) under `core_check.ok_salt` of
    the window's open root `window` and s, registered under `program` binding `window`. Every input is public."""
    rows = [list(_cc().OK_ROW) for _s in range(rounds)]
    salts = [_cc().ok_salt(window, s) for s in range(rounds)]
    return rows, salts, G.Registered.commit(program, table(prefix, OK_VALUE), rows, salts, G.WORD_BITS, G.SCHEMA_V2,
                                            window=window)


def statement_rows(T: int, v2: bool = False) -> list[tuple[str, str, int, list[str]]]:
    """The input rows of a live statement, each ``(port, schema, bits, params)`` as C-Flock's unit stager takes them: each
    slot leaf its own row (``slot_<i>``), then each request header (``req_<i>``, a ``row-seg/v1`` row, `REQUEST_SCHEMA`),
    then ``m`` and ``s``, and under `v2` each public cap its own row (``cap_<i>``), the others ``row/v2`` rows. Its one
    output row is ``o0`` (`OK_PORT`), the unit's ok."""
    def key(name: str, i: int) -> str:
        return name if T == 1 else f"{name}[{i}]"
    return ([(f"slot_{i}", G.SCHEMA_V2, 32, [key("slot", i)]) for i in range(T)]
            + [(f"req_{i}", REQUEST_SCHEMA, 32 * HEADER, [key("req", i)]) for i in range(T)]
            + [("m", G.SCHEMA_V2, 32, ["m"]), ("s", G.SCHEMA_V2, 32, ["s"])]
            + ([(f"cap_{i}", G.SCHEMA_V2, 32, [key("cap", i)]) for i in range(T)] if v2 else []))


def reads_file(program: str, T: int, S: int, s: int, roots: dict[str, str], window: str, session: str, m: int,
               prefix: str = "", max_tokens: Sequence[int] | None = None) -> dict[str, Any]:
    """The verifier's ``--registered`` file (`registered.READS`) of round s's statement ``live@<s>``, unit s of the live
    Program of S rounds at T: its slot ports at round s's ``<prefix>slots@<s>`` and its header ports at
    ``<prefix>requests@0``, at the roots `roots` (``{value: root}``, the window's entries) gives them, each under
    `registered.domain` of `program` (the window Program's digest, which the tables are registered under), the slot table
    bound to the window's open root `window` and the request table to its session `session`; ``m`` and ``s`` at their known
    content (`m_table`, `round_table`) and the output row ``o0`` at the ok rows (`ok_table`). With `max_tokens` it is the v2
    unit's: each ``cap_<i>`` reads row i of the known public caps (`caps_table` of them). Units other than s read nothing.
    Refused (ValueError) when `roots` lacks a table or s is no round."""
    if not 0 <= s < S:
        raise ValueError(f"round {s} is none of the window's {S}")
    sv, rv = table(prefix, SLOTS, s), table(prefix, REQUESTS)
    if missing := [v for v in (sv, rv) if v not in roots]:
        raise ValueError(f"live@{s} reads {missing}, whose roots the window doesn't register")
    before = [None] * s

    def port(value: str, words: int, leaves: int, row: int, schema: str = G.SCHEMA_V2, **bind: str) -> dict[str, Any]:
        return {"value": value, "root": roots[value], "leaves": leaves, "words": words, "word_bits": G.WORD_BITS,
                "domain": G.domain(program, value, words, leaves, **bind).domain_id.hex(), **G.schema_entry(schema),
                "reads": before + [row]}

    def known(value: str, rows: list, salts: list, reg: G.Registered, row: int) -> dict[str, Any]:
        return {"value": value, "root": reg.root.hex(), "leaves": len(rows), "words": len(rows[0]),
                "word_bits": G.WORD_BITS, "domain": reg.domain.domain_id.hex(), **G.schema_entry(G.SCHEMA_V2),
                "reads": before + [row], "known": {"rows": rows, "salts": [x.hex() for x in salts]}}

    ports = {f"slot_{i}": port(sv, SLOT_WORDS, T, i, window=window) for i in range(T)}
    ports |= {f"req_{i}": port(rv, HEADER_WORDS, T, i, REQUEST_SCHEMA, session=session) for i in range(T)}
    ports["m"] = known(table(prefix, M_ROWS, s), *m_table(program, prefix, window, s, m), 0)
    ports["s"] = known(table(prefix, ROUND_ROWS, s), *round_table(program, prefix, window, s), 0)
    if max_tokens is not None:
        rows, salts, reg = caps_table(program, prefix, window, T, max_tokens)
        ports |= {f"cap_{i}": known(table(prefix, CAPS), rows, salts, reg, i) for i in range(T)}
    ports[OK_PORT] = known(table(prefix, OK_VALUE), *ok_table(program, prefix, window, S), s)
    return {"format": G.READS, "ports": ports}


__all__ = ["CAPS", "HEADER", "HEADER_WORDS", "M_ROWS", "OK_PORT", "OK_VALUE", "REQUESTS", "REQUEST_SCHEMA", "ROLES", "ROLES_V2",
           "ROUND_ROWS", "SLOTS", "SLOT_WORDS", "STATEMENT_OUT_ROWS", "caps", "caps_table", "header_words", "m_table", "ok_table",
           "reads_file", "round_table", "slot_words", "statement_rows", "table"]
