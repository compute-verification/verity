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
"""
from __future__ import annotations

import functools
from typing import Any, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, OR, ZERO, add, and_reduce, const_bits, or_reduce, sub
from verity.ml.circuits.boolean.trace import Traced, emit, trace
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.types import Array, Type

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
