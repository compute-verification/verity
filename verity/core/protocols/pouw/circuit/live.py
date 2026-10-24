"""`PouwLiveSlots_v1{T}`: a round's served rows under rule A, the unit that makes dense credit's per-call row count m_s a proved
value (its interface: note:compute-accounting/20261024T0212Z-draft-liveunit-interface).

Rule A: a credited row serves an admitted request, a slot the round's slot table assigns to a request in the gate-signed
request table under the window-open root. Round s's request table is the requests admitted before round s's root, one table
a round. Under the default, round-0-only admission, it is round 0's (`pouw_h1_requests.request_table`, whose root is in
`credit.requests_root` and so in `salt.window_open_root`). Round s's slot table is under round s's root before its coin.
Both are hiding leaves, so credit, which sees their roots only, can't count m_s itself: this unit counts it in a proof, one
unit a round, proved every round (never sampled).

    PouwLiveSlots_v1{T}   (slot: Array<T, U32>, req: Array<T, Array<5, U32>>, m: U32) -> Value<1>

T is the tables' leaf count, the pouw setting ``rows`` (H1's T, a multiple of 32: `pouw_h1_requests.slot_table` and
`request_table` write T leaves each), so m ≤ T. The window's slot M = `h1window.slot`, the least power of two at or above T
(and at least R), is credit's tile grid, not the unit's: M - T slots have no leaf, and their rows are dead. ``slot`` is
round s's slot leaves: q + 1 for a row of request q, 0 (``EMPTY_SLOT``) past the batch. The step's rows are its requests'
rows in order, row r spanning n_r slots of computed tokens (a cache hit is no row, nor is a CUDA-graph pad row past Σ_r n_r,
whose slot is ``EMPTY_SLOT``, so an m that counts one is refused), so m_s = Σ_r n_r. ``req`` is the headers of the round's
request table, leaf q request q in admission
order (index, round, slot, cap, prompt length, each a u32; ``EMPTY_REQUEST`` is all zero, and every entry has a prompt
token). ``m``, the public row, is the step's served rows. These are the network advice's I1 request index and step rows. ok = 1
iff all three hold:

* m ≤ T;
* the live requests (prompt length ≠ 0) are a prefix [0, N), each one's index its leaf;
* every slot i < m names a live request (1 ≤ slot_i ≤ N), and every slot i ≥ m is ``EMPTY_SLOT``.

So ok = 1 means exactly the slots [0, m) are assigned to admitted requests: m is rule A's count (`admitted`). A tile of the
round whose rows all lie at or past m (R·ti ≥ m) is dead. Its admit bits, [i < m], are all 0, so its receipt is `h1tile`'s
canonical dead-tile receipt (`PouwH1Tile_v4`, `h1tile.reference`'s gated branch). Dense credit's population is the live
tiles, so a dead tile is never drawn. The Lean statement is `Pouw.SecurityProofs.LiveCredit`
(``Security/Proofs/Pouw/LiveCredit.lean``), whose `liveOk` is `live_ok`.

`live_ok` is the unit on words, and `traced(T)` its gate circuit.

**The live Program** (`program`) is a window's S rounds' units: one template, `PouwLiveSlots_v1{T}`, called once a round,
unit s round s's. It is a Program of its own beside the window's (`h1window.program`), so the window's tile Program, its
partition and its draw are untouched. Its parameters are named `PREFIX` and their role: ``slots`` (S·T slot leaves, round
s's from row s·T), ``requests`` (T headers, read by every round) and ``m`` (S rows). Its circuit file is the period-open's
``circuit_files["live"]``, and its partition is ``Q_template_instances`` over its template (`query`).

**Round s's statement** ``live@<s>`` reads registered values (`reads_file`), each under the window Program's digest:

* port ``slot_<i>`` reads row i of round s's slot table ``<prefix>slots@<s>`` (`SLOTS`: T rows of `SLOT_WORDS` u16 words,
  the leaf's u32 little-endian), bound to the window's open root, its root round s's ``slots``;
* port ``req_<i>`` reads row i of round 0's request table ``<prefix>requests@0`` (`REQUESTS`: T rows of `HEADER_WORDS`
  words, the header's five u32 little-endian), bound to the window's session (its root, round 0's ``request_table``, is
  inside the open root, as ``ids@0``'s is);
* port ``m`` reads ``<prefix>m@<s>`` (`M_ROWS`) at its known content (`m_table`): one row, m's two words, under
  `core_check.public_row_salt` of the open root and s;
* the output row ``o0`` (`OK_PORT`) reads row s of ``<prefix>live-ok`` (`OK_VALUE`) at its known content (`ok_table`):
  S rows of `core_check.OK_ROW`, row s under `core_check.ok_salt` of the open root and s.

So a round's statement that proves ok = 1 at public m on the tables its round root and the open root commit fixes m as rule
A's count, and one whose m isn't the round's served rows opens no row of ``live-ok``.
"""
from __future__ import annotations

import functools
from typing import Any, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, OR, ZERO, add, and_reduce, const_bits, or_reduce, sub
from verity.ml.circuits.boolean.trace import Traced, emit, trace
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.partition_object import template_instances_query
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Type

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.sampled_proofs.one_stage import registered as G

U32 = Array(32, BL.BIT)
#: a request leaf's header words: index, round, slot, cap and prompt length
HEADER = 5
INDEX, PROMPT = 0, 4
PORTS = ("slot", "req", "m")


def _check(slot: Sequence[int], req: Sequence[Sequence[int]]) -> None:
    if len(req) != len(slot) or any(len(h) != HEADER for h in req):
        raise ValueError(f"the unit reads T slots and T {HEADER}-word headers, not {len(slot)} slots and headers of "
                         f"{[len(h) for h in req]} words")


def live_ok(slot: Sequence[int], req: Sequence[Sequence[int]], m: int) -> int:
    """The unit on words: 1 iff m ≤ T, the live requests are a prefix [0, N) with each one's index its leaf, and exactly the
    slots [0, m) name a live request (1 ≤ slot ≤ N), every later slot 0."""
    _check(slot, req)
    live = [h[PROMPT] != 0 for h in req]
    n = sum(live)
    return int(m <= len(slot)
               and all(live[q] or not live[q + 1] for q in range(len(req) - 1))
               and all(h[INDEX] == q for q, h in enumerate(req) if live[q])
               and all(1 <= s <= n if i < m else s == 0 for i, s in enumerate(slot)))


def admitted(slot: Sequence[int], req: Sequence[Sequence[int]]) -> int:
    """Rule A's count: the slots that name (as q + 1) a request leaf q that is an entry, prompt length ≠ 0. ``live_ok`` at m
    implies it is m (`Pouw.SecurityProofs.LiveCredit.liveOk_admitted`)."""
    _check(slot, req)
    return sum(1 for s in slot if 1 <= s <= len(req) and req[s - 1][PROMPT] != 0)


def _types(T: int) -> tuple[list[Type], Type]:
    return [Array(T, U32), Array(T, Array(HEADER, U32)), U32], BL.BIT


def _build(T: int) -> Any:
    w = T.bit_length()  # N ≤ T < 2^w, and m too once m ≤ T

    def build(C: Any, *e: Any) -> list:
        slot, m = e[:T], e[(HEADER + 1) * T]
        req = [e[T + HEADER * q:T + HEADER * (q + 1)] for q in range(T)]
        live = [or_reduce(C, h[PROMPT]) for h in req]
        conds = [NOT(sub(C, const_bits(T, 32), m, 32)[1])]
        for q in range(T):
            index = and_reduce(C, [b if q >> j & 1 else NOT(b) for j, b in enumerate(req[q][INDEX])])
            conds.append(NOT(C.AND(live[q], NOT(index))))
            if q + 1 < T:
                conds.append(NOT(C.AND(live[q + 1], NOT(live[q]))))
        n = [ZERO] * w
        for b in live:
            n = add(C, n, [b], width=w)
        for i, s in enumerate(slot):
            lo, hi = or_reduce(C, s[:w]), or_reduce(C, s[w:])
            named = C.AND(C.AND(lo, NOT(hi)), NOT(sub(C, n, s[:w], w)[1]))
            empty = NOT(OR(C, lo, hi))
            before = sub(C, const_bits(i, w), m[:w], w)[1]
            conds.append(empty ^ C.AND(before, named ^ empty))
        return [and_reduce(C, conds)]

    return build


@functools.cache
def traced(T: int) -> Traced:
    """The unit's gate circuit at T slots."""
    return trace(_build(T), *_types(T))


def _refuse(b: dict) -> str | None:
    t = b.get("T")
    if not (isinstance(t, int) and not isinstance(t, bool) and t > 0):
        return f"T is the window's slot count, a positive integer, not {t!r}"
    return None


PouwLiveSlots = CompositeDefinition(
    "PouwLiveSlots", 1, ("T",), lambda S: (tuple(zip(PORTS, _types(S.T)[0])), _types(S.T)[1]),
    lambda B, S, *c: emit(traced(S.T), *_types(S.T), B, *c), conformance="protocol-defined", refuse=_refuse,
    doc="Round s's served rows under rule A (`circuit.live`): T slot leaves (q + 1, or 0; T the tables' leaf count, the "
        "pouw setting rows), the T headers of the requests admitted before its root (index, round, slot, cap, prompt length) and the public m, to 1 iff m <= T, the live "
        "requests (prompt length nonzero) are a prefix [0, N) each at its own index, every slot below m is in [1, N] and "
        "every slot from m on is 0.")

#: the live Program's parameters after its ``PREFIX``
ROLES = ("slots", "requests", "m")
#: a slot leaf's and a request header's u16 words a row (u32 little-endian, low half first)
SLOT_WORDS, HEADER_WORDS = 2, 2 * HEADER
#: the statement's output row of the unit's ok, which reads the known ``<prefix>live-ok``
OK_PORT = "o0"
OK_VALUE = "live-ok"
#: the statement's output rows, as C-Flock's unit stager takes them: the unit's one output, its ok
STATEMENT_OUT_ROWS = ((G.SCHEMA_V2, (0,)),)


def template(T: int) -> Any:
    """The live Program's one template: `PouwLiveSlots_v1` at T slots."""
    return bind(PouwLiveSlots, T=int(T))


def _window_signature(S: Any) -> tuple:
    slot, req, m = _types(S.T)[0]
    params = ((S.PREFIX + "slots", Array(S.S * S.T, slot.elem)), (S.PREFIX + "requests", req),
              (S.PREFIX + "m", Array(S.S, m)))
    return params, Array(S.S, BL.BIT)


def _window_body(B: Any, S: Any, slots: Any, requests: Any, m: Any) -> Any:
    unit = template(S.T)
    outs = [B.call(unit, slots[s * S.T:(s + 1) * S.T], requests, m[s]) for s in range(S.S)]
    return coll_concat(_window_signature(S)[1], outs)


PouwLiveWindow = CompositeDefinition(
    "PouwLiveWindow", 1, ("T", "S", "PREFIX"), _window_signature, _window_body, register=False,
    doc="A window's S rounds' live units: round s's PouwLiveSlots_v1{T} on rows [s T, (s + 1) T) of PREFIX slots, the T "
        "request headers PREFIX requests and row s of PREFIX m; every parameter named PREFIX and its role.")


def program(T: int, S: int, prefix: str = "") -> Program:
    """The live Program of a window of S rounds whose tables have T leaves, its parameters named ``prefix`` and their roles
    (`ROLES`). Refused (ValueError) when T or S isn't a positive integer."""
    if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in (T, S)):
        raise ValueError(f"the live Program's T and S are positive integers, not {T!r} and {S!r}")
    return Program(bind(PouwLiveWindow, T=T, S=S, PREFIX=str(prefix)))


def query(T: int) -> dict[str, Any]:
    """The live Program's partition query: every instance of its template one unit, unit s round s's."""
    return template_instances_query([template(T)])


def table(prefix: str, name: str, s: int = 0) -> str:
    """A live statement's registered value: round s's slot table ``<prefix>slots@<s>`` (`SLOTS`), round 0's request table
    ``<prefix>requests@0`` (`REQUESTS`), round s's public m ``<prefix>m@<s>`` (`M_ROWS`) or the ok rows
    ``<prefix>live-ok`` (`OK_VALUE`)."""
    if name == OK_VALUE:
        return f"{prefix}{OK_VALUE}"
    if name not in (SLOTS, REQUESTS, M_ROWS) or (name == REQUESTS and s != 0):
        raise ValueError(f"a live statement's tables are {SLOTS}@<s>, {REQUESTS}@0, {M_ROWS}@<s> and {OK_VALUE}, not "
                         f"{name}@{s}")
    return f"{prefix}{name}@{int(s)}"


SLOTS, REQUESTS, M_ROWS = "slots", "requests", "m"


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
    rows, salts = slot_words([m]), [CC.public_row_salt(window, int(s))]
    return rows, salts, G.Registered.commit(program, table(prefix, M_ROWS, s), rows, salts, G.WORD_BITS, G.SCHEMA_V2,
                                            window=window)


@functools.lru_cache(maxsize=8)
def ok_table(program: str, prefix: str, window: str, rounds: int) -> tuple[list[list[int]], list[bytes], G.Registered]:
    """The live units' ok rows ``<prefix>live-ok``: row s the one word 1 (`core_check.OK_ROW`) under `core_check.ok_salt` of
    the window's open root `window` and s, registered under `program` binding `window`. Every input is public."""
    rows = [list(CC.OK_ROW) for _s in range(rounds)]
    salts = [CC.ok_salt(window, s) for s in range(rounds)]
    return rows, salts, G.Registered.commit(program, table(prefix, OK_VALUE), rows, salts, G.WORD_BITS, G.SCHEMA_V2,
                                            window=window)


def statement_rows(T: int) -> list[tuple[str, str, int, list[str]]]:
    """The input rows of a live statement, each ``(port, schema, bits, params)`` as C-Flock's unit stager takes them: each
    slot leaf its own row (``slot_<i>``), then each request header (``req_<i>``), then ``m``, every one a ``row/v2`` row.
    Its one output row is ``o0`` (`OK_PORT`), the unit's ok."""
    def key(name: str, i: int) -> str:
        return name if T == 1 else f"{name}[{i}]"
    return ([(f"slot_{i}", G.SCHEMA_V2, 32, [key("slot", i)]) for i in range(T)]
            + [(f"req_{i}", G.SCHEMA_V2, 32 * HEADER, [key("req", i)]) for i in range(T)]
            + [("m", G.SCHEMA_V2, 32, ["m"])])


def reads_file(program: str, T: int, S: int, s: int, roots: dict[str, str], window: str, session: str, m: int,
               prefix: str = "") -> dict[str, Any]:
    """The verifier's ``--registered`` file (`registered.READS`) of round s's statement ``live@<s>``, unit s of the live
    Program of S rounds at T: its slot ports at round s's ``<prefix>slots@<s>`` and its header ports at
    ``<prefix>requests@0``, at the roots `roots` (``{value: root}``, the window's entries) gives them, each under
    `registered.domain` of `program` (the window Program's digest, which the tables are registered under), the slot table
    bound to the window's open root `window` and the request table to its session `session`; ``m`` at its known content
    (`m_table`) and the output row ``o0`` at the ok rows (`ok_table`). Units other than s read nothing. Refused
    (ValueError) when `roots` lacks a table or s is no round."""
    if not 0 <= s < S:
        raise ValueError(f"round {s} is none of the window's {S}")
    sv, rv = table(prefix, SLOTS, s), table(prefix, REQUESTS)
    if missing := [v for v in (sv, rv) if v not in roots]:
        raise ValueError(f"live@{s} reads {missing}, whose roots the window doesn't register")
    before = [None] * s

    def port(value: str, words: int, leaves: int, row: int, **bind: str) -> dict[str, Any]:
        return {"value": value, "root": roots[value], "leaves": leaves, "words": words, "word_bits": G.WORD_BITS,
                "domain": G.domain(program, value, words, leaves, **bind).domain_id.hex(), **G.schema_entry(G.SCHEMA_V2),
                "reads": before + [row]}

    def known(value: str, rows: list, salts: list, reg: G.Registered, row: int) -> dict[str, Any]:
        return {"value": value, "root": reg.root.hex(), "leaves": len(rows), "words": len(rows[0]),
                "word_bits": G.WORD_BITS, "domain": reg.domain.domain_id.hex(), **G.schema_entry(G.SCHEMA_V2),
                "reads": before + [row], "known": {"rows": rows, "salts": [x.hex() for x in salts]}}

    ports = {f"slot_{i}": port(sv, SLOT_WORDS, T, i, window=window) for i in range(T)}
    ports |= {f"req_{i}": port(rv, HEADER_WORDS, T, i, session=session) for i in range(T)}
    ports["m"] = known(table(prefix, M_ROWS, s), *m_table(program, prefix, window, s, m), 0)
    ports[OK_PORT] = known(table(prefix, OK_VALUE), *ok_table(program, prefix, window, S), s)
    return {"format": G.READS, "ports": ports}
