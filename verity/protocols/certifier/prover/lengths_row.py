"""The lengths rows' proof block: one row of the network certifier's lengths table (`connection_rows`) computed, as the
proof block's gates, from the compute side's committed row of the same requests and the network certifier's fails row.
Nothing here is in a Spec.

**One proof block a link, execution phase and slot stride** (`LengthsRow`). It reads two rows and no cell:
``requests``, the compute side's committed row of the stride's K request slots, and ``fails``, row j of the network
certifier's fails table, a registered read (a certified input). Its outputs are ``o0``, a u16 whose bit 0 is 1
(`ACCEPT`) exactly when the fails row is of its form against the request row and whose other 15 bits are 0, ``o1``, a
u16, the row's failures F, and ``o2``, the lengths row. The row is gates of the circuit, never an input: the proof
block's statement commits o2 under the salt of the network certifier's lengths row j and registers it as a read of that
row (`connection_rows.port`: proof block `reads_from + j` reads row j), so the proof opens the network certifier's
signed root only when the row the request row gives is the signed one, bit for bit. Every switch is computed from the
two rows and none is witness, so the proof block has no free input of its own: a request's length on the wire is held
to the output count its request row states, the count a tied audit already counts among its free inputs.

**The request row** (`request_fields`) is K slots as a struct of arrays, each field's elements in turn and each element
least significant bit first, then zero bits to a whole 16-bit word (unread): `valid` K×1, `id` K×IW, `P` K×LW and
`O` K×LW, the layout of `replay_chain.arrival_fields` for K arrival slots without a cap bit. A slot's `id` is the
request's place N in the network certifier's release order, `P` its prompt's ids and `O` its outputs. A slot whose
`valid` is 0 reads as empty whatever its other fields hold.

**The fails row** is `connection_rows`' (`SLOT`): slot k is u16 class ‖ u16 at, little-endian, (0, 0) where the
connection ended in its `END` and in an empty slot, (1, n) where it ended in a `FAIL` after n `DATA` cells.

**The lengths row** the block returns is `connection_rows`': slot k is u32 N + 1 ‖ u32 bytes ‖ u32 n, little-endian,
bit 8b + i of the row bit i of its byte b. On a valid slot:

- N + 1 is `id` + 1;
- bytes is `request_bytes(B, P)`, the bytes of the canonical id request of P prompt ids in B-byte cells
  (`active.id_request` at offset 0), computed from P alone: no field straddles a cell, so the request is its head and
  first prompt field's offset A0 (`io_request.layout`), 11 bytes a prompt id, the gap before the first id that doesn't
  fit the first cell and before every m-th id after it, the two closing bytes, and one more byte where those two would
  straddle. It is increasing in P, so the signed bytes hold P;
- n is `at` where the fails slot's class has bit 0 set, else `O`.

On a slot that is not valid all three are 0, an empty slot of the table. A valid slot of P = 0 gets bytes A0 + 2,
which no id request has (its prompt is at least one id), so no signed row holds it.

**The verdict** (`refusals`) holds, at every slot: the class is 0 or 1; class 0 has `at` 0; class 1 is on a valid slot
and has `at` at most `O` (the cells sent before a `FAIL` are outputs the request row counts); and `id` + 1 fits its
u32. F counts the slots whose class has bit 0 set. Where a check fails, o1 and o2 are still what the gates compute
(`outputs` gives the same), and o0 refuses them.

**The gates**, a slot: `valid` gates `id`, `P` and `O` (IW + 2·LW ANDs), so an empty slot's fields come out zero with
no gate on the outputs; an increment for N + 1; for bytes the quotient of P − f0 − 1 by m, a non-restoring division by
the constant (one add of bitlen(m − 1) + 1 bits a quotient bit), then one carry-save sum of 11·P, the constants and
the gaps; a mux of `at` and `O`; and the verdict's comparison of `at` with `O`. F is one carry-save count over the row.

    LengthsRow{K, B, IW, LW}(requests: Array<RB>, fails: Array<32·K>) -> (u16, u16, Array<96·K>)
        RB = `Shape.request_bits`: K·(1 + IW + 2·LW) rounded up to a multiple of 16

Every array is of bits, least significant first. K is at least 1 and at most 2,730, so the outputs fit the widest
public size class (32 + 96·K bits of at most `UNIT_CLASS_MAX`): `connection_rows.check_slots` is that rule, which a
schedule with lengths rows is held to when its parameters are built (`WardenParams.stride_releases`). B is a cell
size an id request's head and first prompt field fit (`io_request.layout`); 1 <= IW <= 32; 1 <= LW <= 32 with the
longest prompt's request below 2^32 bytes.

**The prover's side** (`blocks`): from the network certifier's opening of an execution phase's connection rows
(`connection_rows.Rows`, witness) and the compute side's request rows, one a slot stride, each row's proof block: its
arguments, its outputs and the two registered reads its statement makes (`reads`). It refuses, naming the row, the slot
and the field, request rows whose proof block would not accept or would return another row than the signed one, which
no statement opens.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, NamedTuple, Sequence

import numpy as np

from verity.circuits.boolean.forms import ONE, ZERO, add, compress_columns, const_bits, or_reduce, sub
from verity.circuits.boolean.trace import part
from verity.circuits.defs import bind
from verity.circuits.evaluation.bits import evaluate_bits
from verity.circuits.types import Array, Tuple
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier import head_check as HC
from verity.protocols.certifier.verifier import io_request as Q
from verity.protocols.certifier.verifier import warden_rows as WR
from verity.protocols.certifier.verifier.active import ID_FIELD_BYTES
from verity.protocols.certifier.verifier.connection_rows import Rows, Table
from verity.protocols.certifier.verifier.warden import WardenParams

BIT = HC.BIT
U16 = Array(16, BIT)
WORD_BITS = 16
#: o0 of a row whose fails row is of its form against its request row
ACCEPT = 1
#: the Definition's statics, in order
STATICS = ("K", "B", "IW", "LW")
#: a lengths slot's and a fails slot's bits (`connection_rows.SLOT`: three u32s, two u16s)
LENGTHS_BITS, FAILS_BITS = 8 * CR.SLOT[CR.LENGTHS].size, 8 * CR.SLOT[CR.FAILS].size
#: an id request's two closing bytes, after its last prompt id
TAIL_BYTES = 2


class Form(NamedTuple):
    """Where a canonical id request's prompt ids fall in B-byte cells: `room` payload bytes a cell, the first prompt
    id's offset `start`, the `first` ids that fit the first cell and the `left` bytes they leave in it, and the `each`
    ids a later cell holds and the `over` bytes they leave."""

    room: int
    start: int
    first: int
    left: int
    each: int
    over: int


@lru_cache(maxsize=None)
def form(B: int) -> Form:
    """`Form` at cell size B. ValueError where an id request's head and first prompt id don't fit the first cell
    (`io_request.layout`)."""
    lay = Q.layout(B)
    return Form(lay.room, lay.start, *divmod(lay.room - lay.start, ID_FIELD_BYTES), *divmod(lay.room, ID_FIELD_BYTES))


def request_bytes(B: int, P: int) -> int:
    """The bytes of the canonical id request of P >= 1 prompt ids in B-byte cells (`active.id_request` at offset 0),
    whatever its other fields: a function of P and B alone, increasing in P. At P = 0 it is the head and the closing
    bytes, which no id request is."""
    if type(P) is not int or P < 0:
        raise ValueError(f"a prompt is a number of ids, not {P!r}")
    f = form(B)
    if P <= f.first:
        return f.start + ID_FIELD_BYTES * P + TAIL_BYTES + (f.left == 1 and P == f.first)
    cells, last = divmod(P - f.first - 1, f.each)
    return f.start + ID_FIELD_BYTES * P + f.left + f.over * cells + TAIL_BYTES + (f.over == 1 and last == f.each - 1)


@dataclass(frozen=True)
class Shape:
    """A lengths row's proof block: K slots a row, B-byte cells, a release order place of IW bits and prompt and output
    counts of LW bits."""

    K: int
    B: int
    IW: int
    LW: int

    def __post_init__(self) -> None:
        K, IW, LW = self.K, self.IW, self.LW
        if any(type(v) is not int for v in (K, self.B, IW, LW)):
            raise ValueError(f"a lengths row's K, B, IW and LW are integers, not {(K, self.B, IW, LW)!r}")
        CR.check_slots(K)
        if not 1 <= IW <= 32 or not 1 <= LW <= 32:
            raise ValueError(f"a release order place is 1 <= IW <= 32 bits and a count 1 <= LW <= 32, not IW={IW!r}, "
                             f"LW={LW!r}")
        if request_bytes(self.B, (1 << LW) - 1) >= 1 << 32:
            raise ValueError(f"at {self.B}-byte cells a request of 2^{LW} − 1 prompt ids is 2^32 bytes or more, which "
                             f"no lengths slot holds: a smaller LW")

    @classmethod
    def of(cls, params: WardenParams, IW: int, LW: int) -> Shape:
        """The shape that reads a schedule's rows: its K and B. ValueError for a schedule without lengths rows."""
        if params.lengths is None:
            raise ValueError("the schedule has no lengths rows (WardenParams.lengths)")
        return cls(params.stride_releases, params.cell_bytes, IW, LW)

    @cached_property
    def form(self) -> Form:
        return form(self.B)

    @property
    def request_bits(self) -> int:
        """The request row's bits, padded to whole 16-bit words."""
        return -(-sum(n * w for _, n, w in request_fields(self)) // WORD_BITS) * WORD_BITS

    @property
    def fails_bits(self) -> int:
        return FAILS_BITS * self.K

    @property
    def row_bits(self) -> int:
        return LENGTHS_BITS * self.K

    def statics(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in STATICS}


def _shape(S: Any) -> Shape:
    return Shape(S.K, S.B, S.IW, S.LW)


def request_fields(sh: Shape) -> list[tuple[str, int, int]]:
    """The request row's fields in order, each (name, elements, bits an element)."""
    return [("valid", sh.K, 1), ("id", sh.K, sh.IW), ("P", sh.K, sh.LW), ("O", sh.K, sh.LW)]


# --- the reference ---------------------------------------------------------------------------------------------------

class Request(NamedTuple):
    """A slot of the request row: `valid` (0 or 1), the request's place in the network certifier's release order, its
    prompt's ids and its outputs."""

    valid: int
    order: int
    prompt: int
    outputs: int


#: a slot no request holds
EMPTY = Request(0, 0, 0, 0)


def _slots(sh: Shape, requests: Sequence[Request]) -> list[Request]:
    if len(requests) > sh.K:
        raise ValueError(f"a request row has K = {sh.K} slots, not {len(requests)}")
    out = [Request(*r) for r in requests] + [EMPTY] * (sh.K - len(requests))
    for r in out:
        if r.valid not in (0, 1) or type(r.order) is not int or not 0 <= r.order < 1 << sh.IW or \
                any(type(v) is not int or not 0 <= v < 1 << sh.LW for v in (r.prompt, r.outputs)):
            raise ValueError(f"{r} is no slot of the request row: valid 0 or 1, a place of IW = {sh.IW} bits and "
                             f"counts of LW = {sh.LW} bits")
    return out


def _cells(sh: Shape, fails: bytes) -> list[tuple[int, int]]:
    if len(fails) != sh.fails_bits // 8:
        raise ValueError(f"a fails row of K = {sh.K} slots is {sh.fails_bits // 8} bytes, not {len(fails)}")
    return [(int.from_bytes(fails[i:i + 2], "little"), int.from_bytes(fails[i + 2:i + 4], "little"))
            for i in range(0, len(fails), 4)]


def refusals(sh: Shape, requests: Sequence[Request], fails: bytes) -> list[str]:
    """Why the proof block refuses the two rows, a line a failed check, in slot order: none where o0 is `ACCEPT`."""
    out = []
    for k, (r, (cls, at)) in enumerate(zip(_slots(sh, requests), _cells(sh, fails))):
        outputs = r.outputs if r.valid else 0
        if cls > 1:
            out.append(f"slot {k}: the fails slot's class is {cls}, not 0 or 1")
        if not cls & 1 and at:
            out.append(f"slot {k}: the fails slot has no failure and at = {at}, not 0")
        if cls & 1 and not r.valid:
            out.append(f"slot {k}: the fails slot has a failure and the request row no request")
        if cls & 1 and at > outputs:
            out.append(f"slot {k}: the connection failed after {at} outputs and its request has {outputs}")
        if r.valid and r.order + 1 >= 1 << 32:
            out.append(f"slot {k}: the place {r.order} + 1 is no u32")
    return out


def outputs(sh: Shape, requests: Sequence[Request], fails: bytes) -> tuple[int, int, bytes]:
    """`LengthsRow`'s outputs on a request row (its first slots, the rest empty) and a fails row's bytes: o0, `ACCEPT`
    or 0, o1, the failures, and o2, the lengths row's bytes."""
    row, failures = b"", 0
    for r, (cls, at) in zip(_slots(sh, requests), _cells(sh, fails)):
        place = (r.order + 1) % (1 << 32) if r.valid else 0
        size = request_bytes(sh.B, r.prompt) if r.valid else 0
        n = at if cls & 1 else r.outputs if r.valid else 0
        row += place.to_bytes(4, "little") + size.to_bytes(4, "little") + n.to_bytes(4, "little")
        failures += cls & 1
    return (0 if refusals(sh, requests, fails) else ACCEPT), failures, row


def request_row(sh: Shape, requests: Sequence[Request]) -> list[int]:
    """The request row's bits (the module docstring's layout)."""
    slots = _slots(sh, requests)
    bits = [x for i, w in enumerate((1, sh.IW, sh.LW, sh.LW)) for r in slots for x in HC._int_bits(r[i], w)]
    return bits + [0] * (sh.request_bits - len(bits))


def arguments(sh: Shape, requests: Sequence[Request], fails: bytes) -> list[list[int]]:
    """`LengthsRow`'s arguments in its signature's order, each its leaves' bits."""
    _cells(sh, fails)
    return [request_row(sh, requests), HC._bytes_bits(fails)]


def output_bits(sh: Shape, requests: Sequence[Request], fails: bytes) -> list[int]:
    """`LengthsRow`'s outputs as its leaves' bits: o0's 16, o1's 16, then the row's, bit 8b + i bit i of byte b."""
    verdict, failures, row = outputs(sh, requests, fails)
    return HC._int_bits(verdict, 16) + HC._int_bits(failures, 16) + HC._bytes_bits(row)


def block(sh: Shape):
    """`LengthsRow` bound at a shape."""
    return bind(LengthsRow, **sh.statics())


def evaluate(sh: Shape, cases: Sequence[tuple[Sequence[Request], bytes]]) -> list[tuple[int, int, bytes]]:
    """The circuit's outputs on each (request row, fails row) case, read as `outputs` gives them."""
    args = [arguments(sh, requests, fails) for requests, fails in cases]
    got = np.asarray(evaluate_bits(block(sh), *[np.asarray(col, dtype=np.uint8) for col in zip(*args)]))
    out = []
    for bits in got.reshape(len(cases), -1):
        word = [sum(int(b) << i for i, b in enumerate(bits[at:at + 16])) for at in (0, 16)]
        row = bytes(sum(int(b) << i for i, b in enumerate(bits[at:at + 8])) for at in range(32, len(bits), 8))
        out.append((word[0], word[1], row))
    return out


# --- the circuit -----------------------------------------------------------------------------------------------------

def _sig(S):
    sh = _shape(S)
    return ((("requests", Array(sh.request_bits, BIT)), ("fails", Array(sh.fails_bits, BIT))),
            Tuple(U16, U16, Array(sh.row_bits, BIT)))


def _quotient(C, x: Sequence, m: int, top: int, remainder: bool) -> tuple[list, list | None]:
    """(⌊x / m⌋, x mod m) of x <= `top` for a constant m >= 2, least significant bit first, by non-restoring division:
    each quotient bit one add of ±m, of w = bitlen(m − 1) + 1 bits, to the shifted partial remainder, which is that
    step's remainder where it is not negative and m below it otherwise, so a quotient bit is the complement of its
    sign. The leading bits whose prefix of `top` is below m are 0 and take no gate. The remainder, when asked for,
    takes one more add."""
    w = (m - 1).bit_length() + 1
    rem: list = []
    neg = None
    q = []
    for k in reversed(range(len(x))):
        if neg is None and top >> k < m:
            rem = [x[k], *rem]
            q.append(ZERO)
            continue
        # a prefix below m is fewer than w bits, and from here the partial remainder is w bits, two's complement
        assert neg is not None or len(rem) < w
        back = ZERO if neg is None else neg
        cur = ([x[k], *rem] + [ZERO] * w)[:w]
        rem = add(C, cur, [back if m >> i & 1 else back.inv() for i in range(w)], cin=back.inv(), width=w)
        neg = rem[w - 1]
        q.append(neg.inv())
    if not remainder:
        return q[::-1], None
    if neg is not None:
        rem = add(C, rem, [neg if m >> i & 1 else ZERO for i in range(w)], width=w)
    return q[::-1], (list(rem) + [ZERO] * w)[:w - 1]


def _is(bits: Sequence, value: int):
    """The literals whose AND is 1 iff `bits` is `value`, or None where it never is."""
    return [b if value >> i & 1 else b.inv() for i, b in enumerate(bits)] if value < 1 << len(bits) else None


def _and(C, bits: Sequence):
    out = ONE
    for b in bits:
        out = C.AND(out, b)
    return out


def _request_bytes(C, sh: Shape, valid, prompt: Sequence) -> list:
    """`request_bytes(B, P)` of a valid slot's prompt, 0 of an empty one's, as a u32: `prompt` is already gated by
    `valid`, so the constants are `valid` and the gaps follow from P."""
    f, LW = sh.form, sh.LW
    width = max(1, request_bytes(sh.B, (1 << LW) - 1).bit_length())
    cols: list[list] = [[] for _ in range(width)]

    def put(bits: Sequence, times: int) -> None:
        """`times` · `bits` into the sum's columns."""
        for s in range(times.bit_length()):
            if times >> s & 1:
                for i, b in enumerate(bits):
                    if b.key() != ZERO.key():
                        cols[s + i].append(b)

    put(prompt, ID_FIELD_BYTES)
    put([valid], f.start + TAIL_BYTES)
    if f.left == 1 and (at_first := _is(prompt, f.first)) is not None:
        cols[0].append(_and(C, at_first))
    if f.first + 1 < 1 << LW and (f.left or f.over):
        past, inside = sub(C, prompt, const_bits(f.first + 1, LW), LW)
        later = inside.inv()
        put([later], f.left)
        if f.over:
            cells, last = _quotient(C, past, f.each, (1 << LW) - 2 - f.first, remainder=f.over == 1)
            put([C.AND(later, b) for b in cells], f.over)
            if last is not None:
                cols[0].append(C.AND(later, _and(C, _is(last, f.each - 1))))
    return (compress_columns(C, cols)[:width] + [ZERO] * 32)[:32]


def _build(C, S, requests, fails):
    sh = _shape(S)
    K, IW, LW = sh.K, sh.IW, sh.LW
    at_, fields = 0, {}
    for name, n, w in request_fields(sh):
        fields[name] = [list(requests[at_ + i * w:at_ + (i + 1) * w]) for i in range(n)]
        at_ += n * w
    wide = max(16, LW)
    row, bad, failed = [], [], []
    for k in range(K):
        valid = fields["valid"][k][0]
        order, prompt, count = ([C.AND(valid, b) for b in fields[name][k]] for name in ("id", "P", "O"))
        cls, at = fails[FAILS_BITS * k:FAILS_BITS * k + 16], fails[FAILS_BITS * k + 16:FAILS_BITS * (k + 1)]
        fail = cls[0]
        # N + 1: the carry into bit i is valid and every lower bit of id
        place, carry = [], valid
        for i, (raw, b) in enumerate(zip(fields["id"][k], order)):
            place.append(b ^ carry)
            carry = C.AND(carry, raw) if i else b
        if IW == 32:
            bad.append(carry)
        else:
            place += [carry] + [ZERO] * (31 - IW)
        size = _request_bytes(C, sh, valid, prompt)
        sent, said = list(at) + [ZERO] * (wide - 16), count + [ZERO] * (wide - LW)
        n = [o ^ C.AND(fail, a ^ o) for a, o in zip(sent, said)] + [ZERO] * (32 - wide)
        _, short = sub(C, said, sent, wide)
        bad += [or_reduce(C, cls[1:]), C.AND(fail.inv(), or_reduce(C, at)), C.AND(fail, valid.inv()),
                C.AND(fail, short)]
        failed.append(fail)
        row += place + size + n
    failures = (compress_columns(C, [failed])[:K.bit_length()] + [ZERO] * 16)[:16]
    return [[or_reduce(C, bad).inv()] + [ZERO] * 15, failures, row]


LengthsRow = part(
    "LengthsRow", STATICS, _sig, _build,
    "one row of the network certifier's lengths table computed from the request row of its K slots (valid K×1, id "
    "K×IW, P K×LW, O K×LW, each element LSB first, zero-padded to whole u16s) and the fails row (a slot u16 class, u16 "
    "at): the third output's slot k is u32 id + 1, u32 the bytes of the canonical id request of P prompt ids in "
    "B-byte cells and u32 at where the class has bit 0 set, else O, all 0 where valid is 0; the second output counts "
    "the slots whose class has bit 0 set; the first has 1 in bit 0 exactly when at every slot the class is 0 or 1, "
    "class 0 has at 0, class 1 is on a valid slot with at <= O, and id + 1 fits a u32. Every field little-endian, the "
    "row's bit 8b + i bit i of byte b")


# --- the prover's side -----------------------------------------------------------------------------------------------

class Refused(ValueError):
    """Request rows no lengths row's proof block proves against the network certifier's signed rows."""


class Read(NamedTuple):
    """One registered read of a statement: the row as u16 words, its salt (the prover's) and the header's `registered`
    entry, `{"value", "positions", "paths"}` (`warden_rows.Cells.statement`'s form)."""

    words: list[int]
    salt: bytes
    entry: dict[str, Any]


class RowBlock(NamedTuple):
    """One lengths row's proof block as the prover builds it: the row j it reads and returns, its request row, the
    fails row it reads, and its outputs o1 and o2 (o0 is `ACCEPT`)."""

    position: int
    requests: tuple[Request, ...]
    fails: bytes
    failures: int
    lengths: bytes

    def arguments(self, sh: Shape) -> list[list[int]]:
        return arguments(sh, self.requests, self.fails)

    def output_bits(self, sh: Shape) -> list[int]:
        return output_bits(sh, self.requests, self.fails)


def reads(table: Table, position: int) -> Read:
    """What a statement that reads row `position` of one of the network certifier's connection-rows tables commits
    and names: the fails row a block reads, or the lengths row it returns, under the network certifier's own salt."""
    if not 0 <= position < len(table.rows):
        raise ValueError(f"row {position} is outside the table's {len(table.rows)} rows")
    return Read(WR.words(table.rows[position]), table.salts[position],
                {"value": table.value, "positions": [position], "paths": [table.path(position).hex()]})


def blocks(sh: Shape, opened: Rows, requests: Sequence[Sequence[Request]]) -> list[RowBlock]:
    """Each lengths row's proof block of an execution phase, row by row, from the network certifier's opening of its
    connection rows and the compute side's request rows, one a slot stride in the table's order. `Refused`, naming the
    row, the slot and what differs, where a block would refuse its rows or return another lengths row than the signed
    one: the request rows are then not the ones the network certifier released, and no statement of them opens its
    root."""
    if sh != Shape.of(opened.params, sh.IW, sh.LW):
        raise ValueError(f"{sh} doesn't read the schedule's rows of {opened.params.stride_releases} slots in "
                         f"{opened.params.cell_bytes}-byte cells")
    lengths, fails = opened.table(CR.LENGTHS).rows, opened.table(CR.FAILS).rows
    if len(requests) != len(lengths):
        raise ValueError(f"an execution phase's request rows are one a slot stride, {len(lengths)}, not "
                         f"{len(requests)}")
    slot = CR.SLOT[CR.LENGTHS]
    out = []
    for j, (mine, signed, failed) in enumerate(zip(requests, lengths, fails)):
        why = refusals(sh, mine, failed)
        if why:
            raise Refused(f"row {j}: {why[0]}")
        _, failures, row = outputs(sh, mine, failed)
        for k, (got, want) in enumerate(zip(slot.iter_unpack(row), slot.iter_unpack(signed))):
            for name, a, b in zip(("N + 1", "bytes", "n"), got, want):
                if a != b:
                    raise Refused(f"row {j}: slot {k}: the request row gives {name} = {a} and the network certifier "
                                  f"signed {b}")
        out.append(RowBlock(j, tuple(_slots(sh, mine)), failed, failures, row))
    return out


__all__ = ["ACCEPT", "EMPTY", "FAILS_BITS", "Form", "LENGTHS_BITS", "LengthsRow", "Read", "Refused", "Request",
           "RowBlock", "STATICS", "Shape", "arguments", "block", "blocks", "evaluate", "form", "output_bits",
           "outputs", "reads", "refusals", "request_bytes", "request_fields", "request_row"]
