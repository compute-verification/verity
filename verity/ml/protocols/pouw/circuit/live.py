"""`PouwLiveSlots_v1{T}`: a round's served rows under rule A, the unit that makes dense credit's per-call row count m_s a proved
value.

Rule A: a credited row serves an admitted request, a slot the round's slot table assigns to a request in the gate-signed
request table under the window-open root. Round s's request table is the requests admitted before round s's root, one table
a round. Under the default, round-0-only admission, it is round 0's (`pouw_h1_requests.request_table`, whose root is in
`credit.requests_root` and so in `salt.window_open_root`). Round s's slot table is under round s's root before its coin.
Both are hiding leaves, so credit, which sees their roots only, can't count m_s itself: this unit counts it in a proof, one
unit a round, proved every round (never sampled).

    PouwLiveSlots_v1{T}   (slot: Array<T, U32>, req: Array<T, Array<5, U32>>, m: U32, s: U32) -> Value<1>

T is the tables' leaf count, the pouw setting ``rows`` (H1's T, a multiple of 32: `pouw_h1_requests.slot_table` and
`request_table` write T leaves each), so m ≤ T. The window's slot M = `h1window.slot`, the least power of two at or above T
(and at least R), is credit's tile grid, not the unit's: M - T slots have no leaf, and their rows are dead. ``slot`` is
round s's slot leaves: q + 1 for a row of request q, 0 (``EMPTY_SLOT``) past the batch. The step's rows are its requests'
rows in order, row r spanning n_r slots of computed tokens (a cache hit is no row, nor is a CUDA-graph pad row past Σ_r n_r,
whose slot is ``EMPTY_SLOT``, so an m that counts one is refused), so m_s = Σ_r n_r. ``req`` is the headers of the round's
request table, leaf q request q in admission
order (index, round, slot, cap, prompt length, each a u32; ``EMPTY_REQUEST`` is all zero, and every entry has a prompt
token). ``m``, the public row, is the step's served rows, and ``s``, public too, the round (protocol structure: the statement
is round s's). These are the network advice's I1 request index and step rows. ok = 1 iff all four hold:

* m ≤ T;
* the live requests (prompt length ≠ 0) are a prefix [0, N), each one's index its leaf;
* every slot i < m names a live request (1 ≤ slot_i ≤ N), and every slot i ≥ m is ``EMPTY_SLOT``;
* each request q is named by at most `takes` (its header, s) slots: its prompt length in the round that admitted it (s =
  its round: its prefill, the whole prompt in one step as H1's static schedule has it), one in each of the cap - 1 rounds
  after (round < s < round + cap: a decode row each, the request running at most its cap's steps), none in any other.

So ok = 1 means exactly the slots [0, m) are assigned to admitted requests, none past what its request can take that step:
m is rule A's count (`admitted`), at most Σ_q takes. Without the last check one admitted request named in all T slots
would credit m_s = T rows, the padded ones among them. A step that splits a prompt's prefill (chunked prefill), resumes a
preempted request or verifies draft tokens is refused (its rows are more than `takes`); a static window (the vLLM
integration's `pouw_h1_static.Static`) has none. A tile of the
round whose rows all lie at or past m (R·ti ≥ m) is dead. Its admit bits, [i < m], are all 0, so its receipt is `h1tile`'s
canonical dead-tile receipt (`PouwH1Tile_v4`, `h1tile.reference`'s gated branch). Dense credit's population is the live
tiles, so a dead tile is never drawn. The Lean statement is `Pouw.SecurityProofs.LiveCredit`
(``Security/Proofs/Pouw/LiveCredit.lean``), whose `liveOk` is `live_ok`.

`live_ok` is the unit on words, and `traced(T)` its gate circuit.

**The live Program** (`program`) is a window's S rounds' units: one template, `PouwLiveSlots_v1{T}`, called once a round,
unit s round s's. It is a Program of its own beside the window's (`h1window.program`), so the window's tile Program, its
partition and its draw are untouched. Its parameters are named `PREFIX` and their role: ``slots`` (S·T slot leaves, round
s's from row s·T), ``requests`` (T headers, read by every round), ``m`` (S rows) and ``round`` (S rows, row s the unit's
``s``). Its circuit file is the period-open's
``circuit_files["live"]``, and its partition is ``Q_template_instances`` over its template (`query`).

**Round s's statement** ``live@<s>`` reads registered values (`live_reads.reads_file`; the names below are
`live_reads`', which builds no Program), each under the window Program's digest:

* port ``slot_<i>`` reads row i of round s's slot table ``<prefix>slots@<s>`` (`SLOTS`: T rows of `SLOT_WORDS` u16 words,
  the leaf's u32 little-endian), bound to the window's open root, its root round s's ``slots``;
* port ``req_<i>`` reads row i of round 0's request table ``<prefix>requests@0`` (`REQUESTS`: T rows of `HEADER_WORDS`
  words, the header's five u32 little-endian), bound to the window's session (its root, round 0's ``request_table``, is
  inside the open root, as ``ids@0``'s is);
* port ``m`` reads ``<prefix>m@<s>`` (`M_ROWS`) at its known content (`m_table`): one row, m's two words, under
  `core_check.public_row_salt` of the open root and s;
* port ``s`` reads ``<prefix>round@<s>`` (`ROUND_ROWS`) at its known content (`round_table`): one row, s's two words,
  under `core_check.public_row_salt` of the open root and s;
* the output row ``o0`` (`OK_PORT`) reads row s of ``<prefix>live-ok`` (`OK_VALUE`) at its known content (`ok_table`):
  S rows of `core_check.OK_ROW`, row s under `core_check.ok_salt` of the open root and s.

So a round's statement that proves ok = 1 at public m on the tables its round root and the open root commit fixes m as rule
A's count, and one whose m isn't the round's served rows opens no row of ``live-ok``.

**`PouwLiveSlots_v2{T}`** is v1 with one more public port, ``cap``, each request leaf's public cap: the max_tokens its
period's open commits for request q (credit's fixed length, ``length.max_tokens``), 0 past them.

    PouwLiveSlots_v2{T}   (slot, req, m, s, cap: Array<T, U32>) -> Value<1>

ok = 1 iff v1's ok and, for every leaf q, the leaf is live iff its public cap is nonzero, and a live leaf's header has
round 0 and cap the public cap. The request table it reads is round 0's, so its requests were admitted in round 0. Under
v1 the headers' caps and rounds are the prover's hidden choice: with each round's count m_s held to the public schedule,
a prover could still give request q another's cap, or a later round, and so move a decode row from one request to
another. Under v2 each request takes a decode row in round s ≥ 1 only when s < its public max_tokens. Its Program
(`program_v2`, ``PouwLiveWindow_v2``) has a fifth parameter ``caps``, T rows read by every round. Round s's statement reads
port ``cap_<i>`` at row i of the known ``<prefix>caps`` (`live_reads.caps_table`: T rows, each one u32's two words, under
`core_check.public_row_salt` of the open root and i), which credit builds from the period's committed max_tokens. Credit
reads only v2 (`credit.LIVE_FN`). v1 stays unchanged.
"""
from __future__ import annotations

import functools
from typing import Any, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import (NOT, OR, ZERO, add, and_reduce, compress_columns, const_bits, or_reduce,
                                              sub)
from verity.ml.circuits.boolean.trace import Traced, emit, trace
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.partition_object import template_instances_query
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Type

from verity.core.protocols.pouw.live_reads import HEADER

U32 = Array(32, BL.BIT)
#: a request leaf's header words: index, round, slot, cap and prompt length
INDEX, ROUND, CAP, PROMPT = 0, 1, 3, 4
PORTS = ("slot", "req", "m", "s")
PORTS_V2 = (*PORTS, "cap")


def _check(slot: Sequence[int], req: Sequence[Sequence[int]]) -> None:
    if len(req) != len(slot) or any(len(h) != HEADER for h in req):
        raise ValueError(f"the unit reads T slots and T {HEADER}-word headers, not {len(slot)} slots and headers of "
                         f"{[len(h) for h in req]} words")


def takes(h: Sequence[int], s: int) -> int:
    """How many of round s's slots the request with header `h` can take: its prompt length in the round that admitted it (its
    prefill), one in each of the cap - 1 rounds after (a decode row each), none in any other round."""
    r = h[ROUND]
    return h[PROMPT] if s == r else int(r < s < r + h[CAP])


def live_ok(slot: Sequence[int], req: Sequence[Sequence[int]], m: int, s: int) -> int:
    """The unit on words at round s: 1 iff m ≤ T, the live requests are a prefix [0, N) with each one's index its leaf,
    exactly the slots [0, m) name a live request (1 ≤ slot ≤ N), every later slot 0, and no request q is named by more than
    `takes` (its header, s) slots."""
    _check(slot, req)
    live = [h[PROMPT] != 0 for h in req]
    n = sum(live)
    return int(m <= len(slot)
               and all(live[q] or not live[q + 1] for q in range(len(req) - 1))
               and all(h[INDEX] == q for q, h in enumerate(req) if live[q])
               and all(1 <= x <= n if i < m else x == 0 for i, x in enumerate(slot))
               and all(slot.count(q + 1) <= takes(h, s) for q, h in enumerate(req)))


def caps_bound(req: Sequence[Sequence[int]], cap: Sequence[int]) -> bool:
    """v2's binding: leaf q is live iff its public cap ``cap[q]`` is nonzero, and a live leaf's header has round 0 and
    cap ``cap[q]``."""
    if len(cap) != len(req):
        raise ValueError(f"the unit reads T public caps, one a request leaf, not {len(cap)} for {len(req)} leaves")
    return all((h[PROMPT] != 0) == (c != 0) and (h[PROMPT] == 0 or (h[ROUND] == 0 and h[CAP] == c))
               for h, c in zip(req, cap))


def live_ok_v2(slot: Sequence[int], req: Sequence[Sequence[int]], m: int, s: int, cap: Sequence[int]) -> int:
    """`PouwLiveSlots_v2` on words: `live_ok` and `caps_bound` at the public caps `cap`."""
    return int(caps_bound(req, cap) and live_ok(slot, req, m, s) == 1)


def admitted(slot: Sequence[int], req: Sequence[Sequence[int]]) -> int:
    """Rule A's count: the slots that name (as q + 1) a request leaf q that is an entry, prompt length ≠ 0. ``live_ok`` at m
    implies it is m (`Pouw.SecurityProofs.LiveCredit.liveOk_admitted`)."""
    _check(slot, req)
    return sum(1 for s in slot if 1 <= s <= len(req) and req[s - 1][PROMPT] != 0)


def _types(T: int) -> tuple[list[Type], Type]:
    return [Array(T, U32), Array(T, Array(HEADER, U32)), U32, U32], BL.BIT


def _types_v2(T: int) -> tuple[list[Type], Type]:
    ps, ret = _types(T)
    return [*ps, Array(T, U32)], ret


def _onehot(C: Any, bits: Sequence[Any], values: range) -> list:
    """[bits == v for v in values], over the low and high halves' full decodes."""
    k = len(bits) // 2

    def decode(b: Sequence[Any]) -> list:
        return [and_reduce(C, [x if v >> j & 1 else NOT(x) for j, x in enumerate(b)]) for v in range(1 << len(b))]

    lo, hi = decode(bits[:k]), decode(bits[k:])
    return [C.AND(lo[v & ((1 << k) - 1)], hi[v >> k]) for v in values]


def _build(T: int, v2: bool = False) -> Any:
    w = T.bit_length()  # N ≤ T < 2^w, and m too once m ≤ T

    def build(C: Any, *e: Any) -> list:
        slot, m, s = e[:T], e[(HEADER + 1) * T], e[(HEADER + 1) * T + 1]
        cap = e[(HEADER + 1) * T + 2:(HEADER + 2) * T + 2]
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
        for i, x in enumerate(slot):
            lo, hi = or_reduce(C, x[:w]), or_reduce(C, x[w:])
            named = C.AND(C.AND(lo, NOT(hi)), NOT(sub(C, n, x[:w], w)[1]))
            empty = NOT(OR(C, lo, hi))
            before = sub(C, const_bits(i, w), m[:w], w)[1]
            conds.append(empty ^ C.AND(before, named ^ empty))
        # slot i names q iff its low w bits are q + 1: a slot with a high bit fails the test above
        names = [_onehot(C, x[:w], range(1, T + 1)) for x in slot]
        for q, h in enumerate(req):
            count = compress_columns(C, [[names[i][q] for i in range(T)]])
            prefill = and_reduce(C, [NOT(a ^ b) for a, b in zip(h[ROUND], s)])
            since, early = sub(C, s, h[ROUND], 32)
            decode = C.AND(C.AND(NOT(early), NOT(prefill)), sub(C, since, h[CAP], 32)[1])
            within = NOT(sub(C, h[PROMPT], count, 32)[1])
            rest = or_reduce(C, count[1:])
            none, one = NOT(OR(C, count[0], rest)), C.AND(count[0], NOT(rest))
            later = none ^ C.AND(decode, one)
            conds.append(later ^ C.AND(prefill, within ^ later))
        if v2:
            for q, (h, c) in enumerate(zip(req, cap)):
                public = or_reduce(C, c)
                same = and_reduce(C, [NOT(a ^ b) for a, b in zip(h[CAP], c)] + [NOT(b) for b in h[ROUND]])
                conds.append(NOT(live[q] ^ public))
                conds.append(NOT(C.AND(live[q], NOT(same))))
        return [and_reduce(C, conds)]

    return build


@functools.cache
def traced(T: int) -> Traced:
    """The unit's gate circuit at T slots."""
    return trace(_build(T), *_types(T))


@functools.cache
def traced_v2(T: int) -> Traced:
    """`PouwLiveSlots_v2`'s gate circuit at T slots."""
    return trace(_build(T, v2=True), *_types_v2(T))


def _refuse(b: dict) -> str | None:
    t = b.get("T")
    if not (isinstance(t, int) and not isinstance(t, bool) and t > 0):
        return f"T is the window's slot count, a positive integer, not {t!r}"
    return None


PouwLiveSlots = CompositeDefinition(
    "PouwLiveSlots", 1, ("T",), lambda S: (tuple(zip(PORTS, _types(S.T)[0])), _types(S.T)[1]),
    lambda B, S, *c: emit(traced(S.T), *_types(S.T), B, *c), conformance="protocol-defined", refuse=_refuse,
    doc="Round s's served rows under rule A (`circuit.live`): T slot leaves (q + 1, or 0; T the tables' leaf count, the "
        "pouw setting rows), the T headers of the requests admitted before its root (index, round, slot, cap, prompt "
        "length), the public m and the public round s, to 1 iff m <= T, the live requests (prompt length nonzero) are a "
        "prefix [0, N) each at its own index, every slot below m is in [1, N], every slot from m on is 0, and each request "
        "is named by at most its prompt length's slots in its own round, one in each of the cap - 1 rounds after, and none "
        "in any other.")


def template(T: int) -> Any:
    """The live Program's one template: `PouwLiveSlots_v1` at T slots."""
    return bind(PouwLiveSlots, T=int(T))


def _window_signature(S: Any) -> tuple:
    slot, req, m, s = _types(S.T)[0]
    params = ((S.PREFIX + "slots", Array(S.S * S.T, slot.elem)), (S.PREFIX + "requests", req),
              (S.PREFIX + "m", Array(S.S, m)), (S.PREFIX + "round", Array(S.S, s)))
    return params, Array(S.S, BL.BIT)


def _window_body(B: Any, S: Any, slots: Any, requests: Any, m: Any, rounds: Any) -> Any:
    unit = template(S.T)
    outs = [B.call(unit, slots[s * S.T:(s + 1) * S.T], requests, m[s], rounds[s]) for s in range(S.S)]
    return coll_concat(_window_signature(S)[1], outs)


PouwLiveWindow = CompositeDefinition(
    "PouwLiveWindow", 1, ("T", "S", "PREFIX"), _window_signature, _window_body, register=False,
    doc="A window's S rounds' live units: round s's PouwLiveSlots_v1{T} on rows [s T, (s + 1) T) of PREFIX slots, the T "
        "request headers PREFIX requests, row s of PREFIX m and row s of PREFIX round; every parameter named PREFIX and its "
        "role.")


def program(T: int, S: int, prefix: str = "") -> Program:
    """The live Program of a window of S rounds whose tables have T leaves, its parameters named ``prefix`` and their roles
    (`ROLES`). Refused (ValueError) when T or S isn't a positive integer."""
    if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in (T, S)):
        raise ValueError(f"the live Program's T and S are positive integers, not {T!r} and {S!r}")
    return Program(bind(PouwLiveWindow, T=T, S=S, PREFIX=str(prefix)))


def query(T: int) -> dict[str, Any]:
    """The live Program's partition query: every instance of its template one unit, unit s round s's."""
    return template_instances_query([template(T)])


PouwLiveSlotsV2 = CompositeDefinition(
    "PouwLiveSlots", 2, ("T",), lambda S: (tuple(zip(PORTS_V2, _types_v2(S.T)[0])), _types_v2(S.T)[1]),
    lambda B, S, *c: emit(traced_v2(S.T), *_types_v2(S.T), B, *c), conformance="protocol-defined", refuse=_refuse,
    doc="PouwLiveSlots_v1 with each request leaf's public cap (`circuit.live`): T slot leaves, the T headers of round 0's "
        "request table (index, round, slot, cap, prompt length), the public m, the public round s and T public caps (the "
        "max_tokens the period's open commits for request q, 0 past them), to 1 iff v1's ok holds and every leaf is live "
        "(prompt length nonzero) iff its public cap is nonzero, a live leaf's header having round 0 and cap its public "
        "cap.")


def template_v2(T: int) -> Any:
    """The v2 live Program's one template: `PouwLiveSlots_v2` at T slots."""
    return bind(PouwLiveSlotsV2, T=int(T))


def _window_signature_v2(S: Any) -> tuple:
    params, ret = _window_signature(S)
    return (*params, (S.PREFIX + "caps", _types_v2(S.T)[0][4])), ret


def _window_body_v2(B: Any, S: Any, slots: Any, requests: Any, m: Any, rounds: Any, caps: Any) -> Any:
    unit = template_v2(S.T)
    outs = [B.call(unit, slots[s * S.T:(s + 1) * S.T], requests, m[s], rounds[s], caps) for s in range(S.S)]
    return coll_concat(_window_signature_v2(S)[1], outs)


PouwLiveWindowV2 = CompositeDefinition(
    "PouwLiveWindow", 2, ("T", "S", "PREFIX"), _window_signature_v2, _window_body_v2, register=False,
    doc="A window's S rounds' v2 live units: round s's PouwLiveSlots_v2{T} on rows [s T, (s + 1) T) of PREFIX slots, the T "
        "request headers PREFIX requests, row s of PREFIX m, row s of PREFIX round and the T public caps PREFIX caps; every "
        "parameter named PREFIX and its role.")


def program_v2(T: int, S: int, prefix: str = "") -> Program:
    """The v2 live Program, credit's (`program` with each round's unit `PouwLiveSlots_v2` and the parameter ``caps``,
    `live_reads.ROLES_V2`). Refused (ValueError) when T or S isn't a positive integer."""
    if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in (T, S)):
        raise ValueError(f"the live Program's T and S are positive integers, not {T!r} and {S!r}")
    return Program(bind(PouwLiveWindowV2, T=T, S=S, PREFIX=str(prefix)))


def query_v2(T: int) -> dict[str, Any]:
    """The v2 live Program's partition query: every instance of its template one unit, unit s round s's."""
    return template_instances_query([template_v2(T)])
