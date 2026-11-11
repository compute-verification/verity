"""The compile slice that charges advice v1's length terms: `CompileSlice_v1` (`compile_slice`) with its charge
`window_charge.ChargeLink_v2{D, 0, 0, K}` in place of `charge_chain.ChargeLink_v1{D}`, so a window's charge covers
`code_v1`'s segment and its length terms' tail, which v1 sends while its registration sends lengths
(`codec.Registered.send_lengths`).

    CompileSlice_v7{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW, Q, D, P, K, CN, CTW, CU}
        (pub, adv, cert, srv, cin, cout, wit) -> verdict[16]
    CompileSliceRowGroups_v2{R, …, CU, G}
        (pub, adv, cert, srv0, …, srv{P/G − 1}, cin, cout, wit) -> verdict[16]

Everything but the charge is CompileSlice_v1's (and CompileSliceRowGroups_v1's at G < P): the load, the steps, the
cache, the served rows and the verdict's bits (load, steps, code, charge, rows, cache; `ACCEPT` = 0x3F), on the same
rows but three:
- ``pub``: the declared charge c, then the registered book's tables at the bins a window of D decisions reaches
  (`window_charge.book_fields(D, 0, 0)`, `window_charge.window_book`);
- ``cert``: after the certified arrivals and their tokens, ``a_mt``, each arrival slot's certified max_tokens
  (`window_charge.MW` bits, a u32 of the certified body; 0 in an empty slot);
- ``cin`` and ``cout``: `window_charge`'s carry rows (`window_charge.ROW_BITS`), the window's reads of rows 0 and 1 of
  `window_charge.VALUE`, which the verifier holds by their b‖c to the session's chain (`window_charge.check_chain`).

**The charge** is `window_charge.build_link(C, D, 0, 0, K, …)` on ``pub``'s book from ``cin`` to ``cout``: the
decisions as CompileSlice_v1 charges them (en_i = i < ``count``), and a length slot for each arrival slot of ``cert``,
qen_k its ``a_valid``, mt_k its ``a_mt`` and olen_k its ``a_O`` (the certified output length, zero-extended to 32
bits). So the window charges each request whose body lands in it, the window's certified arrivals, bitlen(max_tokens)
+ 1 bits, as `code_v1` sends its term (`lengths.tails`), whether or not its output ends in the window (the slice loads
each window from the empty state, so a window that opens with a request running is no slice), and ``code`` refuses an
output length past its max_tokens' width. No input of the charge is free witness: qen, mt and olen are ``cert``'s, and en
is ``adv``'s count. What holds ``cert`` to the certified record is a stand-in: the honest cert row, opened by its b‖c
under the certifier's salt (`compile_slice_zk`), until max_tokens is read from the certified body. The circuit bounds
max_tokens only from above, so until then a lower max_tokens the output still fits passes at a lower charge. H = O = 0: the
slice's class runs inside the pool bound with the network certifier's ingress, so its windows send no hits and no order
entries (`protocol/charge-chain.md`, "The slice"), and a window that sends either is CompileSlice_v1's stand-in, the
still-sent fields, as before.

**The reference** is `check` and `case`: `compile_slice`'s with `CODE`, on a `Slice` that carries each arrival's
max_tokens. **Cost** (`size`): CompileSlice_v1's with ChargeLink_v2{D, 0, 0, K} in place of ChargeLink_v1{D}: its rank
check, about 109 ANDs a length slot and 12 for the tail's sum and its bound, and the link's equality over 160 bits in
place of 96.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ZERO, Circuit
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.protocols.certifier import window_charge as WC
from verity.core.protocols.certifier import charge_chain as CC, compile_slice as CS, replay_chain as RC
from verity.core.protocols.certifier.head_check import _part

STATICS, BLOCK_STATICS, FLAGS, ACCEPT = CS.STATICS, CS.BLOCK_STATICS, CS.FLAGS, CS.ACCEPT


@dataclass(frozen=True)
class Slice(CS.Slice):
    """`compile_slice.Slice` with each arrival's certified max_tokens, in the arrivals' order."""

    max_tokens: tuple[int, ...] = ()


def _max_tokens(sl: CS.Slice, cert: Sequence[RC.Arrival]) -> list[int]:
    """Each of ``cert``'s arrivals' max_tokens, by its id, from `sl`'s."""
    by = dict(zip((a[0] for a in sl.arrivals), getattr(sl, "max_tokens", ())))
    if missing := [a[0] for a in cert if a[0] not in by]:
        raise ValueError(f"arrivals {missing} have no max_tokens in the slice")
    return [by[a[0]] for a in cert]


class LengthCharge(CC.SliceCharge):
    """`window_charge.ChargeLink_v2{D, 0, 0, K}` over ``adv``'s decisions and ``cert``'s length slots (the module's
    docstring), in `charge_chain.SliceCharge`'s form."""

    carry_bits = WC.ROW_BITS

    def pub_fields(self, S) -> list[tuple[str, int, int]]:
        return WC.book_fields(S.D, 0, 0)

    def cert_fields(self, S) -> list[tuple[str, int, int]]:
        return [("a_mt", S.K, WC.MW)]

    def link(self, C, S, rows: dict, en: Sequence) -> tuple:
        c = rows["cert"]
        book = [e for name, _n, _w in WC.book_fields(S.D, 0, 0) for e in rows["pub"][name]]
        return WC.build_link(C, S.D, 0, 0, S.K, book, rows["cin"]["carry"][0], rows["cout"]["carry"][0], en,
                             [e[0] for e in rows["adv"]["dec"]], qen=[e[0] for e in c["a_valid"]], mt=c["a_mt"],
                             olen=[list(o) + [ZERO] * (WC.MW - len(o)) for o in c["a_O"]])

    def reference(self, S, rows: dict, count: int, dec: Sequence[int]) -> tuple:
        F = CS.fields(S, self)
        p, c = CC.ints(rows["pub"], F["pub"]), CC.ints(rows["cert"], F["cert"])
        book = WC.WindowBook(tuple((tuple(p[name]), p[name + "_total"][0]) for name in WC.SYMBOLS))
        ok, length, nxt = WC.reference_window(S.D, 0, 0, S.K, book, WC.carry_of([int(b) for b in rows["cin"]]),
                                              [int(k < count) for k in range(S.D)], dec, qen=c["a_valid"],
                                              mt=c["a_mt"], olen=c["a_O"])
        return ok, length, WC.encode(nxt)

    def book(self, S, book: dict) -> tuple[dict, WC.WindowCarry]:
        wb, first = WC.window_book(book, S.D, 0, 0)
        return WC.book_values(wb), first

    def charge(self, S, book: dict, carry: WC.WindowCarry, sl: CS.Slice, dec: Sequence[int],
               cert: Sequence[RC.Arrival]) -> tuple:
        pad = [0] * (S.K - len(cert))
        return WC.reference_window(S.D, 0, 0, S.K, WC.window_book(book, S.D, 0, 0)[0], carry,
                                   [int(k < len(dec)) for k in range(S.D)], list(dec) + [0] * (S.D - len(dec)),
                                   qen=[1] * len(cert) + pad, mt=_max_tokens(sl, cert) + pad,
                                   olen=[a[2] for a in cert] + pad)

    def encode(self, carry: WC.WindowCarry) -> tuple[int, ...]:
        return WC.encode(carry)

    def cert(self, S, sl: CS.Slice, cert: Sequence[RC.Arrival]) -> list[int]:
        return [(m >> j) & 1 for m in _max_tokens(sl, cert) + [0] * (S.K - len(cert)) for j in range(WC.MW)]


CODE = LengthCharge()


def _sig(S):
    CC.check_length_slots(vars(S))
    return CS._sig(S, CODE)


_build = partial(CS._build, code=CODE)
_DOC = ("a slice of the compile unit, one window whole, as CompileSlice_v1 with its charge ChargeLink_v2{D, 0, 0, K}: "
        "the committed decisions and a length term for each certified arrival (its certified max_tokens and output "
        "length, in the certified arrivals' row) charged from the window's carry-in to its carry-out and held to the "
        "declared charge (verdict bits load, steps, code, charge, rows, cache)")
CompileSlice = _part("CompileSlice", STATICS, _sig, _build, _DOC, version=7)
CompileSliceRowGroups = _part("CompileSliceRowGroups", BLOCK_STATICS, _sig, _build,
                              _DOC + "; its served rows are P / G rows of G steps each, step u held to row u // G",
                              version=2)
CompileSliceRowGroups.refuse = CS._refuse_blocks


def statics(*args, **kw) -> dict:
    """`compile_slice.statics`, CompileSlice_v1's, refused where an output length is wider than a length slot."""
    return CC.check_length_slots(CS.statics(*args, **kw))


def unit(**st):
    """The unit at statics `st`: CompileSlice_v7 at G = P (`st` without G), else CompileSliceRowGroups_v2{…, G}."""
    return bind(CompileSliceRowGroups if "G" in st else CompileSlice, **st)


def demo() -> tuple[dict, Slice]:
    """`compile_slice.demo` with max_tokens 8, 16 and 255: request 0 runs to its cap, the others stop before it."""
    st, sl = CS.demo()
    return st, Slice(sl.arrivals, sl.decisions, sl.steps, sl.tokens, (8, 16, 255))


def demo_async() -> tuple[dict, Slice]:
    """`compile_slice.demo_async` with max_tokens 4, 64 and 2: request 2's cap is its output length (its ``cap`` 1)."""
    st, sl = CS.demo_async()
    return st, Slice(sl.arrivals, sl.decisions, sl.steps, sl.tokens, (4, 64, 2))


def check(S, rows: dict) -> dict[str, bool | None]:
    return CS.check(S, rows, CODE)


def case(S, sl: CS.Slice, book: dict, **kw) -> CS.Case:
    """`compile_slice.case` under this charge: `kw`'s carries are `window_charge.WindowCarry`s."""
    return CS.case(S, sl, book, code=CODE, **kw)


def pub(S, charge: int, book: dict) -> list[int]:
    return CS.pub(S, charge, book, CODE)


def evaluate(st: dict, cases: Sequence[CS.Case]) -> list[int]:
    """The unit's verdict word on each case, in one evaluation."""
    args = [[c.rows[r] for c in cases] for r in CS.row_names(SimpleNamespace(**st))]
    out = evaluate_bits(unit(**st), *args).reshape(len(cases), -1).tolist()
    return [sum(int(b) << j for j, b in enumerate(o)) for o in out]


@lru_cache(maxsize=None)
def _counted(key: tuple) -> dict[str, int]:
    S = SimpleNamespace(**dict(key))
    C = Circuit(cse=True)
    ins = [C.input(name, sum(BL.elements(t))) for name, t in _sig(S)[0]]
    C.output("verdict", _build(C, S, *ins)[0])
    return C.counts()


def size(**st) -> dict[str, int]:
    """The unit's ANDs and its rows' widths at statics `st`."""
    key = tuple((k, st[k]) for k in BLOCK_STATICS if k in st)
    return {"ands": _counted(key)["ands"], **{f"{r}_bits": b for r, b in CS.row_bits(SimpleNamespace(**st), CODE).items()}}


__all__ = ["ACCEPT", "BLOCK_STATICS", "CODE", "CompileSlice", "CompileSliceRowGroups", "FLAGS", "LengthCharge",
           "STATICS", "Slice", "case", "check", "demo", "demo_async", "evaluate", "pub", "size", "statics", "unit"]
