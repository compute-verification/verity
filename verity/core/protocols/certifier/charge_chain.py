"""The charge's carry chain: `ChargeCount_v1{N}`'s carry handed from one window to the next under hiding commitments.

    ChargeLink_v1{N}  (book, carry_in[96], carry_out[96], en[N], bit[N]) -> (ok, bits[CW])

`code_v1` carries the decisions' KT counts and their context min(run, 2) from window to window (`arith_encode.Carry`),
so a window's charge depends on every decision before it. The session's first carry is the book's priors and run 0,
public (`arith_encode.first_carry`). Every later one is a function of the decisions so far, a value, so it is committed
as `replay_chain` commits the scan's state: each window has a produced value `VALUE` of two rows, its carry-in and its
carry-out, and `ChargeLink_v1{N}` is `ChargeCount_v1{N}` reading its carry from row 0 and holding its carry' to row 1.
This module is that chain: the row (`encode`, `carry_of`), the commitment and reads (`commit_window`, `reads`), the
verifier's check of how the windows join (`check_chain`), and the Boolean part with its specification
(`ChargeLink`, `reference_link`).

**The row**, `ROW_BITS` = 96 bits, six u16 words: n0 3×15 | n1 3×15 | run 1×2, each element least significant bit
first, then four zero bits (`arith_encode._CARRY`'s fields in order).

**The carry.** Window w's row 0 is window w − 1's row 1 under the same salt (`commit_window(carry=…)`), so its opening
b‖c is the same, and the verifier checks they're equal (`check_chain`). Window 0's row 0 is the priors' row under its
public salt (`start_salt`), whose opening the verifier computes (`start_opening`). Row 1 is under a fresh salt.

**Why a restart is refused.** The verifier holds window 0's carry-in to the priors (`CH-first`) and each later
window's to the last one's carry-out (`CH-carry`), on every session, and the unit holds each window's carry-out to its
charge's carry' from its carry-in (ok). So, window by window, the carry each window is charged from is the one `code_v1`
carries over the whole session. A prover that restarts the counts mid-session, at the priors or at counts fitted to the
window, either breaks the chain, which `check_chain` refuses, or claims a unit whose ok is 0, which fails that unit's
proof as any false unit does. A carry row's padding needs no check: window 0's is the priors' row, and every later one
a proved unit's carry-out.

**Why a value of its own.** `replay_chain.ReplayLoad` writes the window row and checks it in full at the window's
start, before any decision is coded, so a carry-out there would need ReplayLoad to leave a field unchecked and change
its pins. A sibling value under the same carry rule gives the same guarantee and leaves the scan's chain as it is.

**Cost** (`size`): `ChargeCount_v1{N}` and 96 ANDs, the equality of carry' with row 1 and the ok: 19,632 at N = 2 and
50,760 at N = 14, and in a lowering ChargeCount's copies of its carry, about 175 rows a decision. Per window two rows
and the charging unit's two reads, each a frame-v3 path of depth 1.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from types import SimpleNamespace

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ZERO, Circuit, and_reduce
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.core.protocols.certifier import arith_encode as AE
from verity.core.protocols.certifier.arith_encode import Carry, ChargeBook
from verity.core.protocols.certifier.head_check import BIT, _part
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.core.protocols.sampled_proofs.one_stage import registered as G

VALUE = "replay-charge"
WORD_BITS = 16
_CARRY_BITS = sum(n * w for _, n, w in AE._CARRY)
ROW_BITS = WORD_BITS * -(-_CARRY_BITS // WORD_BITS)
_INST = hm96.SHA512


# --- the row ----------------------------------------------------------------------------------------------------------

def encode(carry: Carry) -> tuple[int, ...]:
    """The carry's `ROW_BITS` bits; ValueError for a count that doesn't fit 15 bits or a run that doesn't fit 2."""
    if len(carry.counts) != AE.RUNS:
        raise ValueError(f"a carry is {AE.RUNS} contexts' counts")
    vals = {"n0": [c[0] for c in carry.counts], "n1": [c[1] for c in carry.counts], "run": [carry.run]}
    bits = [b for f in AE._args(AE._CARRY, vals) for b in f]
    return tuple(bits + [0] * (ROW_BITS - len(bits)))


def carry_of(bits: Sequence[int]) -> Carry:
    """The carry a row's bits hold, its padding ignored, as `ChargeLink_v1` reads row 0."""
    if len(bits) != ROW_BITS or any(b not in (0, 1) for b in bits):
        raise ValueError(f"a carry row is {ROW_BITS} bits")
    f = AE._fields(AE._CARRY, bits)
    return Carry(tuple(zip(f["n0"], f["n1"])), f["run"][0])


def row(bits: Sequence[int]) -> tuple[int, ...]:
    """A row's bits as the produced row's u16 words, bit i of the row bit i % 16 of word i // 16."""
    if len(bits) % WORD_BITS:
        raise ValueError("a row is whole 16-bit words")
    return tuple(sum(int(bits[w + i]) << i for i in range(WORD_BITS)) for w in range(0, len(bits), WORD_BITS))


def unrow(words: Sequence[int]) -> tuple[int, ...]:
    return tuple((int(w) >> i) & 1 for w in words for i in range(WORD_BITS))


# --- the specification ------------------------------------------------------------------------------------------------

def reference_link(N: int, book: ChargeBook, row_in: Sequence[int], row_out: Sequence[int], en: Sequence[int],
                   bits: Sequence[int]) -> tuple[int, int]:
    """(ok, bits) of one window's charge from its carry rows: `arith_encode.reference_charge` from `row_in`'s carry,
    and ok only if `row_out` is its carry' (`encode`). A refusal returns ok 0, and then only ok is specified."""
    ok, length, nxt = AE.reference_charge(N, book, carry_of(row_in), en, bits)
    return int(bool(ok) and tuple(row_out) == encode(nxt)), length


def session(N: int, book: dict, windows: Sequence[tuple[Sequence[int], Sequence[int]]]
            ) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    """The honest chain over a session's windows, each (en, bits) in N slots, under the registered book (its JSON):
    each window's (bits, row_in, row_out), the first row_in the priors' row and each later one the last row_out.
    ValueError for a window the charge refuses, which has no carry'."""
    cb, carry = AE.charge_book(book, N)
    out = []
    for w, (en, bits) in enumerate(windows):
        ok, length, nxt = AE.reference_charge(N, cb, carry, en, bits)
        if not ok:
            raise ValueError(f"window {w}'s charge is refused")
        out.append((length, encode(carry), encode(nxt)))
        carry = nxt
    return out


# --- the chain --------------------------------------------------------------------------------------------------------

def start_salt(program: str, value: str = VALUE) -> bytes:
    """The public salt of the session's first carry row (`registered.supplied_salts`' row 0 of `value`)."""
    return G.supplied_salts(program, value, 1)[0]


def _opening(words: Sequence[int], salt: bytes) -> bytes:
    return _INST.commit_string(_INST.default_key, G.row_digest(words, WORD_BITS, SCHEMA), salt)


def start_opening(program: str, book: dict) -> bytes:
    """b‖c of the priors' row under `start_salt`, which the verifier computes from the registered book."""
    return _opening(row(encode(AE.first_carry(book))), start_salt(program))


def commit_rows(program: str, value: str, first: Sequence[int], out: Sequence[int], registration: str,
                carry: G.Registered | None = None, salt: bytes | None = None) -> G.Registered:
    """A window's produced `value` of two carry rows, given as bits: row 0 `first` under `start_salt` for the session's
    first window (`carry` None), else the previous window's row 1 under its salt; row 1 `out` under `salt` (default
    fresh, `hm96`'s `fresh_salt`)."""
    if carry is None:
        rows0, salt0 = row(first), start_salt(program, value)
    else:
        rows0, salt0 = carry.rows[-1], carry.salts[-1]
    return produced.commit(program, value, [rows0, row(out)], registration,
                           [salt0, salt if salt is not None else _INST.fresh_salt()], WORD_BITS, SCHEMA)


def commit_window(program: str, book: dict, carry_out: Carry, registration: str, carry: G.Registered | None = None,
                  salt: bytes | None = None) -> G.Registered:
    """A window's produced value: its carry-in row and `carry_out`'s. `carry`: the previous window's value, whose row 1
    is this window's row 0 under the same salt; for the session's first window (None), row 0 is the priors' row under
    `start_salt`. `salt`: row 1's (default fresh, `hm96`'s `fresh_salt`)."""
    return commit_rows(program, VALUE, encode(AE.first_carry(book)), encode(carry_out), registration, carry, salt)


def reads() -> dict[str, dict[str, int]]:
    """The charging unit's rows: `carry_in` 0 and `carry_out` 1."""
    return {VALUE: {"carry_in": 0, "carry_out": 1}}


def openings(value: G.Registered) -> list[bytes]:
    """Each carry row's b‖c, as the produced record's `openings` give them."""
    return list(value.bcs)


def check_chain(windows: Sequence[Sequence[bytes]], program: str, book: dict) -> str | None:
    """The verifier's check of a session's carry rows, given each window's two openings b‖c in order: None, or why not.
    `CH-rows`: a window isn't 2 carry rows; `CH-first`: the session doesn't start at the book's priors under their
    public salt; `CH-carry`: a window's carry-in isn't the previous window's carry-out."""
    return check_rows(windows, start_opening(program, book))


def check_rows(windows: Sequence[Sequence[bytes]], start: bytes) -> str | None:
    """`check_chain` given the session's first carry row's opening `start`."""
    for w, rows in enumerate(windows):
        if len(rows) != 2:
            return f"CH-rows: window {w} has {len(rows)} carry rows, not 2"
    if windows and bytes(windows[0][0]) != start:
        return "CH-first: the session's first carry isn't the book's priors under their public salt"
    for w in range(1, len(windows)):
        if bytes(windows[w][0]) != bytes(windows[w - 1][1]):
            return f"CH-carry: window {w}'s carry-in isn't window {w - 1}'s carry-out"
    return None


# --- the circuit ------------------------------------------------------------------------------------------------------

def book_fields(N: int) -> list[tuple[str, int, int]]:
    """`ChargeLink_v1{N}`'s public inputs, (name, elements, width) as a row's fields: `arith_encode.charge_fields(N)`
    without the carry."""
    return AE.charge_fields(N)[:-len(AE._CARRY)]


def book_values(book: ChargeBook) -> dict[str, list[int]]:
    """`book_fields`' values, by name."""
    vals = AE.charge_values(book, Carry())
    return {name: vals[name] for name, _n, _w in book_fields(1)}


def _link_in(S) -> list[tuple[str, int, int]]:
    return book_fields(S.N) + [("carry_in", 1, ROW_BITS), ("carry_out", 1, ROW_BITS), ("en", S.N, 1), ("bit", S.N, 1)]


_LINK_OUT = [("ok", 1, 1), ("bits", 1, AE.CW)]


def _carry_elements(bits) -> list[list]:
    out, at = [], 0
    for _name, n, w in AE._CARRY:
        out += [list(bits[at + e * w:at + (e + 1) * w]) for e in range(n)]
        at += n * w
    return out


def _build_link(C, S, *elements):
    f = AE._split(_link_in(S), elements)
    book = [e for name, _n, _w in book_fields(S.N) for e in f[name]]
    ok, bits, nxt = AE.build_charge(C, S.N, book + _carry_elements(f["carry_in"][0]), [e[0] for e in f["en"]],
                                    [e[0] for e in f["bit"]])
    C.phase = "link"
    out = (list(b for e in nxt for b in e) + [ZERO] * ROW_BITS)[:ROW_BITS]
    same = and_reduce(C, [(x ^ y).inv() for x, y in zip(out, f["carry_out"][0])])
    return [[C.AND(ok, same)], list(bits)]


def _sig_link(S):
    if not 1 <= S.N < 1 << 16:
        raise ValueError(f"statics {dict(S.__dict__)}: 1 ≤ N < 2^16")
    rows = {"carry_in", "carry_out"}
    return (tuple((name, Array(w, BIT) if name in rows else Array(n, Array(w, BIT))) for name, n, w in _link_in(S)),
            AE._type(_LINK_OUT))


ChargeLink = _part(
    "ChargeLink", ("N",), _sig_link, _build_link,
    "one window's charge under advice v1 with its carry chained: ChargeCount_v1{N} from the carry in the window's "
    "committed carry-in row, ok only if its carry' is the window's carry-out row, the next window's carry-in "
    "(`charge_chain.reference_link`)")


def link_unit(N: int):
    return bind(ChargeLink, N=N)


def build_link(C, N: int, book: Sequence, row_in: Sequence, row_out: Sequence, en: Sequence, bits: Sequence) -> tuple:
    """`ChargeLink_v1{N}` inline in a caller's circuit: `book`, the elements of `book_fields(N)` in order; `row_in` and
    `row_out`, the window's carry rows' `ROW_BITS` wires each (its reads of `VALUE`'s rows 0 and 1); `en` and `bits`,
    one wire per slot. Returns (ok, bits[CW]), bits the window's charge, which the caller holds to c_w."""
    out = _build_link(C, SimpleNamespace(N=N), *book, list(row_in), list(row_out), *[[e] for e in en],
                      *[[b] for b in bits])
    return out[0][0], out[1]


# --- arguments and evaluation -----------------------------------------------------------------------------------------

def link_arguments(N: int, book: ChargeBook, row_in: Sequence[int], row_out: Sequence[int], en: Sequence[int],
                   bits: Sequence[int]) -> list[list[int]]:
    """`ChargeLink_v1{N}`'s arguments, each a flat bit list."""
    if len(row_in) != ROW_BITS or len(row_out) != ROW_BITS or len(en) != N or len(bits) != N:
        raise ValueError(f"two {ROW_BITS}-bit rows and N = {N} slots")
    return AE._args(book_fields(N), book_values(book)) + [list(row_in), list(row_out), list(en), list(bits)]


def evaluate_links(N: int, cases: Sequence[tuple[ChargeBook, Sequence[int], Sequence[int], Sequence[int],
                                                 Sequence[int]]]) -> list[tuple[int, int]]:
    """`ChargeLink_v1{N}` on each (book, row_in, row_out, en, bits), in one evaluation: (ok, bits) each."""
    args = [list(col) for col in zip(*(link_arguments(N, *c) for c in cases))]
    out = evaluate_bits(link_unit(N), *args).reshape(len(cases), -1).tolist()
    return [(int(o[0]), sum(int(b) << j for j, b in enumerate(o[1:1 + AE.CW]))) for o in out]


# --- cost -------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _staged(N: int) -> dict[str, int]:
    S = SimpleNamespace(N=N)
    C = Circuit(cse=True)
    params, _ = _sig_link(S)
    ins = [C.input(f"p{i}e{j}", w) for i, (_n, t) in enumerate(params) for j, w in enumerate(BL.elements(t))]
    C.output("out", [b for e in _build_link(C, S, *ins) for b in e])
    return C.phase_counts()


def size(N: int) -> dict[str, int]:
    """`ChargeLink_v1{N}`'s ANDs by stage (`arith_encode.size("charge")`'s and `link`, the carry-out's equality and the
    ok), traced on a counting circuit, its ``carry`` copies (rows of a lowering, no ANDs), and the chain's rows and
    reads a window."""
    out = dict(_staged(N))
    out["ands"] = sum(v for k, v in out.items() if k != "carry")
    return {**out, "row_bits": ROW_BITS, "rows_per_window": 2, "reads_per_window": 2, "path_depth": 1,
            "record_bytes_per_window": 2 * produced.BC_BYTES}


__all__ = ["ChargeLink", "ROW_BITS", "VALUE", "WORD_BITS", "book_fields", "book_values", "build_link", "carry_of",
           "check_chain", "check_rows", "commit_rows", "commit_window", "encode", "evaluate_links", "link_arguments",
           "link_unit", "openings", "reads", "reference_link", "row", "session", "size", "start_opening", "start_salt",
           "unrow"]
