"""The response tie's proof block: one response row of the network certifier's content rows (`content_rows`)
computed, as the proof block's gates, from the compute side's committed row of the same connection's ids and the
network certifier's end row of it. Nothing here is in a Spec or proved, and no table holds the rows it reads and
returns: `content_rows` has their formats alone.

**One proof block a connection** (`ResponseTie`), in one of two forms by its class (the static STOP). It reads
``egress``, the compute side's committed row of the O ids it served the connection, O its class's size; in a stop
class ``stop``, the compute side's committed stop word of the same connection; and ``end``, the network certifier's
end row of it, a certified input. Its outputs are ``o0``, a u16 whose bit 0 is 1 (`ACCEPT`) exactly when every check
of its form holds and whose other 15 bits are 0, and ``o1``, the response row. The row is gates of the circuit, never
an input: a statement that commits o1 under the salt of the network certifier's response row and registers it as a
read of that row opens the signed root only when the ids the compute side committed, and their count, are the ids
the network certifier released, word for word. Every switch is computed from the rows read and none is witness, so
the proof block has no free input of its own.

**The rows it reads**, each an array of bits, least significant first:

- ``egress``: O u32 ids, bit 32·p + i bit i of id p. Whatever it holds past the count is not read into the output.
- ``stop``: one u16, the registry's `StopRule16_v1` word n + finish·2^13: n, in its low `COUNT_BITS` bits, the ids
  the stop rule ends the response after, and finish why (1 a stop id, 2 the request's ``max_tokens``, 3 the model's
  length; 0 where the rule doesn't end it within the row's ids).
- ``end``: `content_rows`' end row, u16 class ‖ u16 at, little-endian: (0, 0) where the connection ended in its
  `END`, (1, n) where it ended in a `FAIL` after n `DATA` cells.

**The response row** it returns is `content_rows`': u32 m ‖ O u32 ids, little-endian, bit 8b + i of the row bit i of
its byte b. The ids are ``egress``'s id p for p below the count its form reads and 0 from there to O; m is that
count where every check holds.

**The two forms** (`refusals`):

- a stop class (STOP = 1), whose connections ended in their `END` where the stop rule ends them. The checks: the
  end row is (0, 0); the stop word's finish is not 0; its n is at most O. The count is n. No other value of the
  finish is refused: the stop word is its own proof block's output, and this one reads where it ends, not why.
- a failed class (STOP = 0), whose connections ended in a `FAIL`; it has no stop word. The checks: the end row's
  class is 1; its `at` is at most O. The count is `at`.

So in either form an end row whose class is neither 0 nor 1 is refused.

**Where a check fails** o0 is 0 and the count word of o1 is all ones (`content_rows.NO_COUNT`), a count no response
row has, since a row's count is at most its O ids. So no row a refused block returns is one the network certifier
signs, and the block is sound where nothing holds o0 to `ACCEPT`: a statement that registers o1 as a read of the
signed row opens its root on no refused input. That rests on the network certifier signing only rows of the form
(`trusted-host`) and on the commitment binding its row. The ids of such a row are still what the gates compute
(`outputs` gives the same).

**The gates**, with k = log2 O. The count's low k bits and whether it is O or more give the comparison of every
position p below O with the count (`_prefix`): the upper half of p's bits against the count's and the lower half
against the count's (`_below`, 2^w − w − 1 ANDs for a half of w bits), and one AND a position to join the two.
One AND an id bit then keeps the ids below the count, and the count word takes one OR a bit. The checks are an OR
over the end row's bits, the count's comparison with O, and in a stop class the finish's three bits.

    ResponseTie{O, STOP=1}(egress: Array<32·O>, stop: Array<16>, end: Array<32>) -> (u16, Array<32·(O + 1)>)
    ResponseTie{O, STOP=0}(egress: Array<32·O>, end: Array<32>) -> (u16, Array<32·(O + 1)>)
        ANDs = 33·O + 2^⌊k/2⌋ + 2^⌈k/2⌉ − k + c       (`Shape.ands`)
        c = 48 in a stop class and 32 in a failed class; one fewer at O = 2, whose count has one bit below O's,
        and in a stop class at k = 12, whose stop word has no bit past the count's bit k

Every array is of bits, least significant first. O is a class's size of ids released, a power of two from
`content_rows.MIN_SIZE` to `content_rows.most_ids`, so the outputs fit the widest public size class (16 + 32·(O + 1)
bits of at most `UNIT_CLASS_MAX`): `content_rows.check_ids` is that rule, which the shape is held to where it is
built. It is one Definition with a static for the form, since the two forms return the same row from the same ids
under the same comparison and differ in the word the count is read from and in three checks; the static adds the
stop word's parameter, as `compile_slice_v6`'s AS adds its cap bits. Bound at a form it is that form's own circuit,
so the two forms are two circuit files and two sessions: a failed class's circuit has no ``stop`` port, and every
port of either is a row each connection of its class has, as a statement's registered reads need
(`Flock.Registered`'s `checkPort` and `complete`).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from verity.circuits.boolean.forms import ONE, OR, ZERO, or_reduce
from verity.circuits.boolean.trace import part
from verity.circuits.defs import bind
from verity.circuits.evaluation.bits import evaluate_bits
from verity.circuits.types import Array, Tuple
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier import head_check as HC

BIT = HC.BIT
U16 = Array(16, BIT)
#: o0 of a row whose checks all hold
ACCEPT = 1
#: the Definition's statics, in order
STATICS = ("O", "STOP")
#: an id's and a count's bits, and an end row's
ID_BITS, END_BITS = 8 * CT.U32_BYTES, 8 * CT.END_BYTES
#: the stop word: n in its low `COUNT_BITS` bits and the finish in the `FINISH_BITS` above. `COUNT_BITS` is the stop
#: rule's `N_BITS`, which this unit may not import: the certifier's suite holds the two equal
STOP_BITS, COUNT_BITS = 16, 13
FINISH_BITS = STOP_BITS - COUNT_BITS
#: an end row's `at`, the count a failed class's form reads
AT_BITS = END_BITS // 2


@dataclass(frozen=True)
class Shape:
    """A response tie's proof block: a row of O ids, in a stop class (STOP = 1), whose rows have a stop word, or in
    a failed class (STOP = 0)."""

    O: int
    STOP: int

    def __post_init__(self) -> None:
        CT.check_ids(self.O)
        if type(self.STOP) is not int or self.STOP not in (0, 1):
            raise ValueError(f"a response tie's STOP is 1, a stop class's form, or 0, a failed class's, not "
                             f"{self.STOP!r}")
        if self.k >= self.count_bits:
            raise ValueError(f"a row of O = {self.O} ids has a count its form's count field, of {self.count_bits} "
                             f"bits, doesn't hold: at most {1 << self.count_bits - 1} ids")

    @property
    def count_bits(self) -> int:
        """The bits of the field the form reads its count from: the stop word's n, or the end row's `at`."""
        return COUNT_BITS if self.STOP else AT_BITS

    @property
    def k(self) -> int:
        """log2 O."""
        return self.O.bit_length() - 1

    @property
    def egress_bits(self) -> int:
        return ID_BITS * self.O

    @property
    def row_bits(self) -> int:
        """The response row's bits: its count and its O ids."""
        return 8 * CT.response_bytes(self.O)

    @property
    def input_bits(self) -> int:
        return self.egress_bits + STOP_BITS * self.STOP + END_BITS

    @property
    def output_bits(self) -> int:
        return 16 + self.row_bits

    @property
    def ands(self) -> int:
        """The proof block's AND gates (the module docstring's formula)."""
        k = self.k
        checks = 48 - (k == COUNT_BITS - 1) if self.STOP else 32
        return 33 * self.O + (1 << k // 2) + (1 << k - k // 2) - k + checks - (k == 1)

    def statics(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in STATICS}


def _shape(S: Any) -> Shape:
    return Shape(S.O, S.STOP)


# --- the reference ---------------------------------------------------------------------------------------------------

def stop_word(n: int, finish: int) -> int:
    """The stop word of a response the stop rule ends after n ids for the reason `finish`: n + finish·2^13."""
    if type(n) is not int or type(finish) is not int or not 0 <= n < 1 << COUNT_BITS or \
            not 0 <= finish < 1 << FINISH_BITS:
        raise ValueError(f"a stop word is a count of {COUNT_BITS} bits and a finish of {FINISH_BITS}, not {n!r}, "
                         f"{finish!r}")
    return n | finish << COUNT_BITS


def _ids(sh: Shape, egress: Sequence[int]) -> list[int]:
    ids = list(egress)
    if len(ids) > sh.O or any(type(t) is not int or not 0 <= t < 1 << ID_BITS for t in ids):
        raise ValueError(f"an egress row is at most O = {sh.O} ids, each a u32, not {len(ids)}: {ids[:4]}")
    return ids + [0] * (sh.O - len(ids))


def _end(end: bytes) -> tuple[int, int]:
    if len(end) != CT.END_BYTES:
        raise ValueError(f"an end row is {CT.END_BYTES} bytes, not {len(end)}")
    return CT.END_SLOT.unpack(end)


def _word(sh: Shape, stop: int | None) -> tuple[int, int]:
    """The stop word's (n, finish) in a stop class; a failed class has none."""
    if not sh.STOP:
        if stop is not None:
            raise ValueError("a failed class's row has no stop word")
        return 0, 0
    if type(stop) is not int or not 0 <= stop < 1 << STOP_BITS:
        raise ValueError(f"a stop class's row has a stop word, a u16, not {stop!r}")
    return stop & (1 << COUNT_BITS) - 1, stop >> COUNT_BITS


def refusals(sh: Shape, egress: Sequence[int], end: bytes, stop: int | None = None) -> list[str]:
    """Why the proof block refuses the rows, a line a failed check: none where o0 is `ACCEPT`."""
    _ids(sh, egress)
    (cls, at), (n, finish) = _end(end), _word(sh, stop)
    out = []
    if cls > CT.FAILED:
        out.append(f"the end row's class is {cls}, not {CT.ENDED} or {CT.FAILED}")
    if sh.STOP:
        if cls == CT.FAILED:
            out.append("the end row has a failure, and a stop class's connection ended in its END")
        if cls == CT.ENDED and at:
            out.append(f"the end row has no failure and at = {at}, not 0")
        if not finish:
            out.append("the stop word's finish is 0: the stop rule doesn't end the response within the row's ids")
        if n > sh.O:
            out.append(f"the stop word counts {n} ids and the row holds {sh.O}")
    else:
        if cls == CT.ENDED:
            out.append("the end row has no failure, and a failed class's connection ended in a FAIL")
        if at > sh.O:
            out.append(f"the connection failed after {at} cells and the row holds {sh.O} ids")
    return out


def outputs(sh: Shape, egress: Sequence[int], end: bytes, stop: int | None = None) -> tuple[int, bytes]:
    """`ResponseTie`'s outputs on an egress row (its first ids, the rest 0), an end row's bytes and, in a stop
    class, a stop word: o0, `ACCEPT` or 0, and o1, the response row's bytes, its count `content_rows.NO_COUNT`
    where o0 is 0."""
    ids, (_, at), (n, _) = _ids(sh, egress), _end(end), _word(sh, stop)
    count = n if sh.STOP else at
    refused = bool(refusals(sh, egress, end, stop))
    row = struct.pack(f"<{sh.O + 1}I", CT.NO_COUNT if refused else count, *ids[:count], *[0] * (sh.O - min(count, sh.O)))
    return (0 if refused else ACCEPT), row


def arguments(sh: Shape, egress: Sequence[int], end: bytes, stop: int | None = None) -> list[list[int]]:
    """`ResponseTie`'s arguments in its signature's order, each its leaves' bits."""
    ids, _, _ = _ids(sh, egress), _end(end), _word(sh, stop)
    word = [HC._int_bits(stop, STOP_BITS)] if sh.STOP else []
    return [[b for t in ids for b in HC._int_bits(t, ID_BITS)], *word, HC._bytes_bits(end)]


def output_bits(sh: Shape, egress: Sequence[int], end: bytes, stop: int | None = None) -> list[int]:
    """`ResponseTie`'s outputs as its leaves' bits: o0's 16, then the row's, bit 8b + i bit i of byte b."""
    verdict, row = outputs(sh, egress, end, stop)
    return HC._int_bits(verdict, 16) + HC._bytes_bits(row)


def block(sh: Shape):
    """`ResponseTie` bound at a shape."""
    return bind(ResponseTie, **sh.statics())


def evaluate(sh: Shape, cases: Sequence[tuple]) -> list[tuple[int, bytes]]:
    """The circuit's outputs on each case, (egress row, end row) and in a stop class the stop word after them, read
    as `outputs` gives them."""
    args = [arguments(sh, *case) for case in cases]
    got = np.asarray(evaluate_bits(block(sh), *[np.asarray(col, dtype=np.uint8) for col in zip(*args)]))
    out = []
    for bits in got.reshape(len(cases), -1):
        verdict = sum(int(b) << i for i, b in enumerate(bits[:16]))
        out.append((verdict, np.packbits(bits[16:], bitorder="little").tobytes()))
    return out


# --- the circuit -----------------------------------------------------------------------------------------------------

def _sig(S):
    sh = _shape(S)
    word = (("stop", U16),) if sh.STOP else ()
    return ((("egress", Array(sh.egress_bits, BIT)),) + word + (("end", Array(END_BITS, BIT)),),
            Tuple(U16, Array(sh.row_bits, BIT)))


def _below(C, x: Sequence) -> list:
    """[p < x] for each p below 2^len(x), x least significant bit first. Bit by bit from the least: a bit of x is
    ORed into each comparison so far and ANDed into its copy for the positions that have the bit, one AND for
    both, so w bits take 2^w − w − 1 ANDs."""
    out = [ZERO]
    for b in x:
        both = [C.AND(b, t) for t in out]
        out = [b ^ t ^ a for t, a in zip(out, both)] + both
    return out


def _prefix(C, low: Sequence, top) -> tuple[list, Any]:
    """[p < min(count, O)] for each p below O = 2^len(low), `low` the count's low bits and `top` whether the count
    is O or more: p is below the count where its high half is below the count's (`top` among its bits), or is the
    count's with its low half below the count's. The two don't hold at once, so their OR is an XOR. With it,
    whether `low` has a bit set: position 0 of either half is below it."""
    j = len(low) // 2
    lo, hi = _below(C, low[:j]), _below(C, low[j:])
    some = OR(C, lo[0], hi[0])
    hi = [OR(C, top, t) for t in hi]
    out = []
    for ph, under in enumerate(hi):
        here = (hi[ph - 1] if ph else ONE) ^ under
        out += [under ^ C.AND(here, t) for t in lo]
    return out, some


def _build(C, S, egress, *rest):
    sh = _shape(S)
    O, k, end = sh.O, sh.k, rest[-1]
    cls, at = end[:AT_BITS], end[AT_BITS:]
    if sh.STOP:
        count, finish = rest[0][:COUNT_BITS], rest[0][COUNT_BITS:]
        bad = [or_reduce(C, end), or_reduce(C, finish).inv()]
    else:
        count = at
        bad = [cls[0].inv(), or_reduce(C, cls[1:])]
    past = or_reduce(C, count[k + 1:])
    below, some = _prefix(C, count[:k], OR(C, count[k], past))
    refused = or_reduce(C, bad + [OR(C, past, C.AND(count[k], some))])
    ids = [C.AND(below[p], b) for p in range(O) for b in egress[ID_BITS * p:ID_BITS * (p + 1)]]
    m = [OR(C, refused, b) for b in count[:k + 1]] + [refused] * (ID_BITS - k - 1)
    return [[refused.inv()] + [ZERO] * 15, m + ids]


ResponseTie = part(
    "ResponseTie", STATICS, _sig, _build,
    "one response row of the network certifier's content rows computed from the egress row of O u32 ids, the end row "
    "(u16 class, u16 at) and, at STOP = 1, the stop word (u16 n + finish * 8192): the second output is u32 m then O "
    "u32 ids, id p of the egress row for p below the count and 0 from there, the count the stop word's n at STOP = 1 "
    "and the end row's at at STOP = 0; the first has 1 in bit 0 exactly when, at STOP = 1, the end row is (0, 0), "
    "the finish is not 0 and n <= O, and at STOP = 0, the end row's class is 1 and at <= O; m is the count where "
    "that bit is 1 and 0xFFFFFFFF where it is 0. Every field little-endian, the row's bit 8b + i bit i of byte b")


__all__ = ["ACCEPT", "AT_BITS", "COUNT_BITS", "END_BITS", "FINISH_BITS", "ID_BITS", "ResponseTie", "STATICS",
           "STOP_BITS", "Shape", "arguments", "block", "evaluate", "output_bits", "outputs", "refusals", "stop_word"]
