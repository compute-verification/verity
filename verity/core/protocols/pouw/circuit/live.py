"""`PouwLiveSlots_v1{T}`: a round's served rows under rule A, the unit that makes dense credit's per-call row count m_s a proved
value (its interface: note:compute-accounting/20261024T0212Z-draft-liveunit-interface).

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
"""
from __future__ import annotations

import functools
from typing import Any, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import (NOT, OR, ZERO, add, and_reduce, compress_columns, const_bits, or_reduce,
                                              sub)
from verity.ml.circuits.boolean.trace import Traced, emit, trace
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.types import Array, Type

U32 = Array(32, BL.BIT)
#: a request leaf's header words: index, round, slot, cap and prompt length
HEADER = 5
INDEX, ROUND, CAP, PROMPT = 0, 1, 3, 4
PORTS = ("slot", "req", "m", "s")


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


def admitted(slot: Sequence[int], req: Sequence[Sequence[int]]) -> int:
    """Rule A's count: the slots that name (as q + 1) a request leaf q that is an entry, prompt length ≠ 0. ``live_ok`` at m
    implies it is m (`Pouw.SecurityProofs.LiveCredit.liveOk_admitted`)."""
    _check(slot, req)
    return sum(1 for s in slot if 1 <= s <= len(req) and req[s - 1][PROMPT] != 0)


def _types(T: int) -> tuple[list[Type], Type]:
    return [Array(T, U32), Array(T, Array(HEADER, U32)), U32, U32], BL.BIT


def _onehot(C: Any, bits: Sequence[Any], values: range) -> list:
    """[bits == v for v in values], over the low and high halves' full decodes."""
    k = len(bits) // 2

    def decode(b: Sequence[Any]) -> list:
        return [and_reduce(C, [x if v >> j & 1 else NOT(x) for j, x in enumerate(b)]) for v in range(1 << len(b))]

    lo, hi = decode(bits[:k]), decode(bits[k:])
    return [C.AND(lo[v & ((1 << k) - 1)], hi[v >> k]) for v in values]


def _build(T: int) -> Any:
    w = T.bit_length()  # N ≤ T < 2^w, and m too once m ≤ T

    def build(C: Any, *e: Any) -> list:
        slot, m, s = e[:T], e[(HEADER + 1) * T], e[(HEADER + 1) * T + 1]
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
        "pouw setting rows), the T headers of the requests admitted before its root (index, round, slot, cap, prompt "
        "length), the public m and the public round s, to 1 iff m <= T, the live requests (prompt length nonzero) are a "
        "prefix [0, N) each at its own index, every slot below m is in [1, N], every slot from m on is 0, and each request "
        "is named by at most its prompt length's slots in its own round, one in each of the cap - 1 rounds after, and none "
        "in any other.")
