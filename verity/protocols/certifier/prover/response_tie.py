"""The response tie's proof block: one response row of the network certifier's content rows (`content_rows`)
computed, as the proof block's gates, from the compute side's committed row of the same connection's ids and the
network certifier's end row of it. Nothing here is in a Spec or proved, and no table holds the rows it reads and
returns: `content_rows` has their formats alone.

**One proof block a connection**, of one of two Definitions by the kind of its class: `ResponseTieStop` in a stop
class, whose rows have a stop word, and `ResponseTieNoStop` in a no-stop class, whose rows have none. Each takes
every connection its kind of class holds, however the connection ended. It reads ``egress``, the compute side's
committed row of the connection's O ids, O its class's size; in a stop class ``stop``, the compute
side's committed stop word of the same connection; and ``end``, the network certifier's end row of it, a certified
input. Its outputs are ``o0``, a u16 whose bit 0 is 1 (`ACCEPT`) exactly when every check
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
- ``end``: `content_rows`' end row, u16 class ‖ u16 at, little-endian: the class is 0 where the connection ended in
  its `END` and 1 where it ended in a `FAIL`, and `at` is the `DATA` cells the network certifier released before
  either.

**The response row** it returns is `content_rows`': u32 m ‖ O u32 ids, little-endian, bit 8b + i of the row bit i of
its byte b. The ids are ``egress``'s id p for p below the count its form reads and 0 from there to O; m is that
count where every check holds.

**The two forms** (`refusals`), one for each kind of class the compute side files a connection in:

- a stop class (`ResponseTieStop`), whose connections the stop rule ends as the engine served them, after the stop
  word's n ids. Three checks hold however the connection ended: the stop word's finish is not 0; its n is at most
  O; the end row's class is 0 or 1. Then, by the end row's class:

  - 0, the connection ended in its `END`: the count is n. `at` is in no check: the row's count is n, and the
    signed row holds it to the network certifier's.
  - 1, the network certifier failed the connection after the engine finished it: the count is `at`, and the check
    is that `at` is at most n. Soundness needs the check, and the check alone does not give it. The stop word is
    the same whatever the egress row holds past its n-th id, so the ids past n in the committed row are held by
    nothing, and without the check a server could commit ids it chose there, send them, and have the connection
    failed. With it the row carries no id past n; what holds the ids below n is the premise below.

  No other value of the finish is refused: the stop word is its own proof block's output, and this one reads where
  it ends, not why.
- a no-stop class (`ResponseTieNoStop`), every other connection's: one the engine aborted, and one that ended in
  its `END` where the stop rule doesn't end it as it was served (past the rule's most ids, a body's own stop
  strings, a ``min_tokens``). It has no stop word. The checks: the end row's class is 0 or 1; its `at` is at most
  O. The count is `at` either way, so no gate reads the class's low bit.

So in either form an end row whose class is neither 0 nor 1 is refused.

**The premise of both forms, which is open.** The block computes the network certifier's row from the compute
side's committed row, and holds nothing about where the committed ids came from. So either form is sound only where
the compute audit holds the committed ids below the count to be the Program's outputs, a premise no gate here can
check. Nothing in the tree holds it today, for either kind of class. In a stop class the stop rule's proof block
holds the stop word to the rule over the committed ids: it holds none of those ids to the Program. A no-stop
class's form has, beside that, no committed count: its count is the network certifier's `at`, and the compute side
commits none.

**Where a check fails** o0 is 0 and the count word of o1 is all ones (`content_rows.NO_COUNT`), a count no response
row has, since a row's count is at most its O ids. So no row a refused block returns is one the network certifier
signs, and a refusal needs nothing to hold o0 to `ACCEPT`: a statement that registers o1 as a read of the
signed row opens its root on no refused input. That rests on the network certifier signing only rows of the form
(`trusted-host`) and on the commitment binding its row. The ids of such a row are still what the gates compute
(`outputs` gives the same): those below `at` where the end row's class has its low bit, and below n otherwise.

**A statement of it is taken only from O = 32.** Flock's composer takes a statement only where one of its rows has
a compression before its last block, a row of 112 bytes or more: `verity_flock.circuit`'s `compose` refuses any
other ("no row has a compression before its last block"), by its `Port.comps`
(`verity/protocols/flock/prover/python/verity_flock/circuit.py`; the 111 bytes a last block holds are
`windows.interface.LAST_BLOCK_BYTES`). The longest row here is o1, 4·(O + 1) bytes, so a tied class under 32 ids
has no statement today. That is a fact of the composer and no rule of the classes: the gates are right from O =
`content_rows.MIN_SIZE`, and circuit-check, which composes no statement, doesn't see it.

**The gates**, with k = log2 O. The count's low k bits and whether it is O or more give the comparison of every
position p below O with the count (`_prefix`): the upper half of p's bits against the count's and the lower half
against the count's (`_below`, 2^w − w − 1 ANDs for a half of w bits), and one AND a position to join the two.
One AND an id bit then keeps the ids below the count, and the count word takes one OR a bit. The checks are an OR
over the end row's class above its low bit and the count's comparison with O. A stop class's form also has an OR
over the finish's three bits, the comparison of `at` with n (`_above` on their low k + 1 bits, n being at most O
where it is read, with an OR over `at`'s bits above them), and one AND a bit of the count to take it from `at` or
from n by the end row's class.

    ResponseTieStop{O}(egress: Array<32·O>, stop: Array<16>, end: Array<32>) -> (u16, Array<32·(O + 1)>)
        ANDs = 33·O + 2^⌊k/2⌋ + 2^⌈k/2⌉ + k + 49       (`Shape.ands`)
    ResponseTieNoStop{O}(egress: Array<32·O>, end: Array<32>) -> (u16, Array<32·(O + 1)>)
        ANDs = 33·O + 2^⌊k/2⌋ + 2^⌈k/2⌉ − k + 31       one fewer at O = 2, whose count has one bit below O's

Every array is of bits, least significant first. O is a class's size of ids released, a power of two from
`content_rows.MIN_SIZE` to `content_rows.most_ids`, so the outputs fit the widest public size class (16 + 32·(O + 1)
bits of at most `UNIT_CLASS_MAX`): `content_rows.check_ids` is that rule, which the shape is held to where it is
built. The two forms are two Definitions, each with the one static O and no optional port: a statement's every
port is a row each of its instances reads (`Flock.Registered`'s `checkPort` and `complete`), and a no-stop class
has no stop word, so each form is a circuit file and a session of its own. They share their gates (`_build`): the
same row from the same ids under the same comparison, and differ in the word the count is read from and in their
checks.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from verity.circuits.boolean.forms import ONE, OR, ZERO, maj, mux, or_reduce
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
#: each Definition's one static: the ids its class's rows hold
STATICS = ("O",)
#: an id's and a count's bits, and an end row's
ID_BITS, END_BITS = 8 * CT.U32_BYTES, 8 * CT.END_BYTES
#: the stop word: n in its low `COUNT_BITS` bits and the finish in the `FINISH_BITS` above. `COUNT_BITS` is the stop
#: rule's `N_BITS`, which this unit may not import: the certifier's suite holds the two equal
STOP_BITS, COUNT_BITS = 16, 13
FINISH_BITS = STOP_BITS - COUNT_BITS
#: an end row's `at`: the count a no-stop class's form reads, and a stop class's of a connection that failed
AT_BITS = END_BITS // 2


@dataclass(frozen=True)
class Shape:
    """A response tie's proof block: a row of O ids, in a stop class (`stop`), whose rows have a stop word and
    whose Definition is `ResponseTieStop`, or in a no-stop class, whose Definition is `ResponseTieNoStop`."""

    O: int
    stop: bool

    def __post_init__(self) -> None:
        CT.check_ids(self.O)
        if type(self.stop) is not bool:
            raise ValueError(f"a response tie's class is a stop class (True) or a no-stop class (False), not "
                             f"{self.stop!r}")
        if self.k >= self.count_bits:
            raise ValueError(f"a row of O = {self.O} ids has a count its form's count field, of {self.count_bits} "
                             f"bits, doesn't hold: at most {1 << self.count_bits - 1} ids")

    @property
    def count_bits(self) -> int:
        """The bits of the field no count of the form is past: the stop word's n, or the end row's `at`."""
        return COUNT_BITS if self.stop else AT_BITS

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
        return self.egress_bits + STOP_BITS * self.stop + END_BITS

    @property
    def output_bits(self) -> int:
        return 16 + self.row_bits

    @property
    def ands(self) -> int:
        """The proof block's AND gates (the module docstring's formulas)."""
        k = self.k
        checks = k + 49 if self.stop else 31 - k - (k == 1)
        return 33 * self.O + (1 << k // 2) + (1 << k - k // 2) + checks

    def statics(self) -> dict[str, int]:
        """Its Definition's statics."""
        return {k: getattr(self, k) for k in STATICS}


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
    """The stop word's (n, finish) in a stop class; a no-stop class has none."""
    if not sh.stop:
        if stop is not None:
            raise ValueError("a no-stop class's row has no stop word")
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
    if sh.stop:
        if not finish:
            out.append("the stop word's finish is 0: the stop rule doesn't end the response within the row's ids")
        if n > sh.O:
            out.append(f"the stop word counts {n} ids and the row holds {sh.O}")
        if cls == CT.FAILED and at > n:
            out.append(f"the connection failed after {at} cells and the stop rule ends the response after {n} ids")
    elif at > sh.O:
        out.append(f"the connection ended after {at} cells and the row holds {sh.O} ids")
    return out


def outputs(sh: Shape, egress: Sequence[int], end: bytes, stop: int | None = None) -> tuple[int, bytes]:
    """The shape's Definition's outputs on an egress row (its first ids, the rest 0), an end row's bytes and, in a
    stop class, a stop word: o0, `ACCEPT` or 0, and o1, the response row's bytes, its count `content_rows.NO_COUNT`
    where o0 is 0. The count its ids are below is `at` in a no-stop class and, in a stop class, where the end row's
    class has its low bit; a stop class's is the stop word's n otherwise."""
    ids, (cls, at), (n, _) = _ids(sh, egress), _end(end), _word(sh, stop)
    count = n if sh.stop and not cls & 1 else at
    refused = bool(refusals(sh, egress, end, stop))
    row = struct.pack(f"<{sh.O + 1}I", CT.NO_COUNT if refused else count, *ids[:count], *[0] * (sh.O - min(count, sh.O)))
    return (0 if refused else ACCEPT), row


def arguments(sh: Shape, egress: Sequence[int], end: bytes, stop: int | None = None) -> list[list[int]]:
    """The shape's Definition's arguments in its signature's order, each its leaves' bits."""
    ids, _, _ = _ids(sh, egress), _end(end), _word(sh, stop)
    word = [HC._int_bits(stop, STOP_BITS)] if sh.stop else []
    return [[b for t in ids for b in HC._int_bits(t, ID_BITS)], *word, HC._bytes_bits(end)]


def output_bits(sh: Shape, egress: Sequence[int], end: bytes, stop: int | None = None) -> list[int]:
    """The shape's Definition's outputs as its leaves' bits: o0's 16, then the row's, bit 8b + i bit i of byte b."""
    verdict, row = outputs(sh, egress, end, stop)
    return HC._int_bits(verdict, 16) + HC._bytes_bits(row)


def block(sh: Shape):
    """The shape's Definition, `ResponseTieStop` or `ResponseTieNoStop`, bound at its O."""
    return bind(ResponseTieStop if sh.stop else ResponseTieNoStop, **sh.statics())


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

def _sig(stop: bool):
    """A form's signature: the egress row, in a stop class the stop word, and the end row."""

    def sig(S):
        sh = Shape(S.O, stop)
        word = (("stop", U16),) if stop else ()
        return ((("egress", Array(sh.egress_bits, BIT)),) + word + (("end", Array(END_BITS, BIT)),),
                Tuple(U16, Array(sh.row_bits, BIT)))

    return sig


def _below(C, x: Sequence) -> list:
    """[p < x] for each p below 2^len(x), x least significant bit first. Bit by bit from the least: a bit of x is
    ORed into each comparison so far and ANDed into its copy for the positions that have the bit, one AND for
    both, so w bits take 2^w − w − 1 ANDs."""
    out = [ZERO]
    for b in x:
        both = [C.AND(b, t) for t in out]
        out = [b ^ t ^ a for t, a in zip(out, both)] + both
    return out


def _prefix(C, low: Sequence, top) -> tuple[list, tuple[Any, Any]]:
    """[p < min(count, O)] for each p below O = 2^len(low), `low` the count's low bits and `top` whether the count
    is O or more: p is below the count where its high half is below the count's (`top` among its bits), or is the
    count's with its low half below the count's. The two don't hold at once, so their OR is an XOR. With it, each
    half's comparison of position 0: `low` has a bit set where either holds."""
    j = len(low) // 2
    lo, hi = _below(C, low[:j]), _below(C, low[j:])
    zero = lo[0], hi[0]
    hi = [OR(C, top, t) for t in hi]
    out = []
    for ph, under in enumerate(hi):
        here = (hi[ph - 1] if ph else ONE) ^ under
        out += [under ^ C.AND(here, t) for t in lo]
    return out, zero


def _above(C, a: Sequence, b: Sequence):
    """[a > b] for two words of one length, least significant bit first: the carry out of a + ~b, one AND a bit."""
    carry = ZERO
    for x, y in zip(a, b, strict=True):
        carry = maj(C, x, y.inv(), carry)
    return carry


def _build(stop: bool):
    """A form's gates: its checks and the word its count is read from, then the row, which both forms build alike."""

    def build(C, S, egress, *rest):
        sh = Shape(S.O, stop)
        O, k, end = sh.O, sh.k, rest[-1]
        cls, at = end[:AT_BITS], end[AT_BITS:]
        bad, past = [or_reduce(C, cls[1:])], or_reduce(C, at[k + 1:])
        if stop:
            n, finish = rest[0][:COUNT_BITS], rest[0][COUNT_BITS:]
            beyond = or_reduce(C, n[k + 1:])
            # n above O; and, read where n is at most O, `at` above n in a connection that failed
            bad += [or_reduce(C, finish).inv(), OR(C, beyond, C.AND(n[k], or_reduce(C, n[:k]))),
                    C.AND(cls[0], OR(C, past, _above(C, at[:k + 1], n[:k + 1])))]
            count = mux(C, cls[0], at[:k + 1], n[:k + 1])
            (past,) = mux(C, cls[0], [past], [beyond])
            below, _ = _prefix(C, count[:k], OR(C, count[k], past))
        else:
            count = at[:k + 1]
            below, zero = _prefix(C, count[:k], OR(C, count[k], past))
            bad.append(OR(C, past, C.AND(count[k], OR(C, *zero))))
        refused = or_reduce(C, bad)
        ids = [C.AND(below[p], b) for p in range(O) for b in egress[ID_BITS * p:ID_BITS * (p + 1)]]
        m = [OR(C, refused, b) for b in count] + [refused] * (ID_BITS - k - 1)
        return [[refused.inv()] + [ZERO] * 15, m + ids]

    return build


ResponseTieStop = part(
    "ResponseTieStop", STATICS, _sig(True), _build(True),
    "one response row of the network certifier's content rows computed from a stop class's egress row of O u32 ids, "
    "its stop word (u16 n + finish * 8192) and the end row (u16 class, u16 at): the second output is u32 m then O "
    "u32 ids, id p of the egress row for p below the count and 0 from there, the count at where the class has its "
    "low bit and n otherwise; the first has 1 in bit 0 exactly when the finish is not 0, n <= O, the class is 0 or "
    "1 and, where it is 1, at <= n; m is the count where that bit is 1 and 0xFFFFFFFF where it is 0. Every field "
    "little-endian, the row's bit 8b + i bit i of byte b")

ResponseTieNoStop = part(
    "ResponseTieNoStop", STATICS, _sig(False), _build(False),
    "one response row of the network certifier's content rows computed from a no-stop class's egress row of O u32 "
    "ids and the end row (u16 class, u16 at): the second output is u32 m then O u32 ids, id p of the egress row for "
    "p below at and 0 from there; the first has 1 in bit 0 exactly when the end row's class is 0 or 1 and at <= O; "
    "m is at where that bit is 1 and 0xFFFFFFFF where it is 0. Every field little-endian, the row's bit 8b + i bit "
    "i of byte b")


__all__ = ["ACCEPT", "AT_BITS", "COUNT_BITS", "END_BITS", "FINISH_BITS", "ID_BITS", "ResponseTieNoStop",
           "ResponseTieStop", "STATICS", "STOP_BITS", "Shape", "arguments", "block", "evaluate", "output_bits",
           "outputs", "refusals", "stop_word"]
