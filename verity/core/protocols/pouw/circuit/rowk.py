"""Row units under C-Flock's block (``Shape.row_k``): each row unit of Pearl-C's (`pc8`) and Pearl-C4's (`pc4`) tile
check cut along k into a head, one form per segment of row_k positions and a tail.  It is opt-in: a shape without
``row_k`` builds today's Programs, which this module leaves alone.

C-Flock proves one instance in at most a 2^27-bit block (`verity_flock.circuit.K_MAX`), and a whole row unit outgrows it
long before the served k (Pearl-C's row of A on sm_120 past k = 2,560).  Cut at row_k = 2,048 (`ROW_K`), no piece grows
with k but through the row of A it reads.  The pieces of row i of A (row j of B's are the same with F_B and F_A swapped,
E_B, B's rows and, in the tail, G):

    template                    one instance per    reads                                      computes and returns
    Pc8RowAHead{K}              row i               A_i, E_A,i                                 α, E′, θ, finite, floor
    Pc8RowAFirst{K,KS,DEV}      row i, segment 0    A_i, the head's row, segment 0 of F_A      the segment's codes, the
                                                    and of F_B                                 carry
    Pc8RowAForm{K,KS,J,DEV}     row i, segment J    the same at segment J, and segment J−1's   the same
                                                    carry
    Pc8RowATail{K,KS}           row i               the head's row, every segment's codes,     A′_i's codes, P_A,i, ok
                                                    the last carry
    Pc8RowBTail{K,KS}           row j of B          the same, and G                            B̃_j's codes, P_B,j, ok

The head is the row's statistics (`pc8._moments`, `_bounds`) and E′ = e4m3(β·E).  Form J computes its segment's codes, the
dead count over it, and the clean-up's 32 chains (A′_i·F_B, B̃_j·F_A) over it, each started from the carry: the dead count
so far, and each chain's promoted total so far (its one accumulator, unpromoted).  A segment is whole windows of the device
and whole 32-position blocks (row_k a multiple of both), so each form adds the same atoms and promotions in the same order
as the whole row, and the carry is exact.  The tail concatenates the codes (wiring only), computes P from E′ and the
chains, and the row's admission from the head's bits and the dead count.  So every value is the whole row's, each computed
in exactly one unit, and a tile reads the tail's fp8-p and ok rows as it reads today's row unit's.

Pearl-C4's pieces (`Pc4Row{A,B}{Head,First,Form,Tail}`) are the same cut: its head is `pc4._stats` (its admission is
complete there, so the head writes the row's ok row), its forms `pc4._form` on the segment with the NVF4 chains carried,
and its tail Ẽ and P.  Each form decodes its segment's BF16 words again (`Bf16ToF32`, core's wiring), the only gate that
two units hold.

Each piece's rows (`PORTS8`, `PORTS4`; `leaves`' pc8-head, pc8-carry, fp8-codes, pc4-head, pc4-carry and nvfp4-codes) are
C-Flock-committed ports in the row's session, and a form reads its segment of F_A and F_B as a public input region under
the anchor's slice rule (`DERIVATIONS8`, `DERIVATIONS4`, `anchors.Derivation.per_segment`).  Every form opens the row of A
whole, so a form's SHA work still grows with k.  C-Flock pads each output port to whole words and reads an input row as
its parameters' bits run together, so the head returns its row in whole-word ports (`HEAD8_OUT`, `HEAD4_OUT`: a_E with
its pad, and the finite and floor bits with theirs, one port each) and commits the very words the forms and tail read.
"""
from __future__ import annotations

import math
from functools import lru_cache

from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.fp32 import F32AddV2, F32FmaV2
from verity.ml.circuits.ops.prims import Bf16ToF32, E4m3ToF32, F2fpBf16
from verity.ml.circuits.ops.scalar import BitAnd, F32Neg, I32Le

from ..protocol import DomainError
from ..schemes.pearl_c_device import DEVICES
from . import anchors as A
from . import leaves as L
from . import pc4, pc8
from . import words as W

#: the deployed segment length: a multiple of every device's window, of 32 and of Pearl-C4's 64-position atom
ROW_K = 2048
R, F32, OK = pc8.R, W.F32, L.OK
E4M3, E2M1, U8, BF16 = pc8.E4M3, W.E2M1, W.U8, W.BF16

#: the head's row, as each form and the tail read it (`leaves`' pc8-head and pc4-head rows)
HEAD8 = (("alpha", F32), ("ep", pc8.LINE), ("th", Value(64)), ("finite", Value(1)), ("floor", Value(1)), ("pad", Value(14)))
HEAD4 = (("alpha", F32), ("ae", U8), ("pad", Value(8)))
#: the head's output ports, each a run of its row's fields: C-Flock pads each output port to whole u16 words
#: (`class_statement.unit_lowering`), so a_E and its pad, and the finite and floor bits and theirs, are one port each
HEAD8_OUT = (("alpha",), ("ep",), ("th",), ("finite", "floor", "pad"))
HEAD4_OUT = (("alpha",), ("ae", "pad"))
#: a form's carry, as the next form and the tail read it (pc8-carry, pc4-carry); pc8-carry's zero fill makes the row whole
#: SHA-512 blocks (128 words), a row shape C-Flock's widened scope takes, where 66 words would be neither short nor whole
FILL8 = 62
CARRY8 = (("dead", Value(32)), ("tot", Array(R, F32)), ("fill", Array(FILL8, Value(16))))
CARRY4 = (("tot", Array(R, F32)),)
_NAMES8, _NAMES4 = tuple(n for n, _t in HEAD8), tuple(n for n, _t in HEAD4)


def segment(fn: SpecializedDefinition) -> int:
    """A form template's segment of k: J, or 0 for a row's first form."""
    return fn.bindings.get("J", 0)


def _head_out(head: tuple, ports: tuple) -> tuple:
    """The head's output types, a field per port: its row field, or a Tuple of the port's fields."""
    types_ = dict(head)
    return tuple(types_[p[0]] if len(p) == 1 else Tuple(*(types_[n] for n in p)) for p in ports)


# ---- Pearl-C's pieces ----------------------------------------------------------------------------------------------------

def _head8_body(B, S_, x, line):
    e = pc4._Emit(B)
    s, rho, alpha, beta, th = pc8._moments(e, x, S_.K)
    floor, finite = pc8._bounds(e, s, rho, alpha)
    return tuple_of(alpha, array_of(pc8._ep(e, beta, line)), th, tuple_of(finite, floor, e.c(14, 0)))


def _seg8_sig(S_, own: str, other: str, carry: bool):
    n = S_.KS
    params = ((("x", Array(S_.K, F32)),) + HEAD8 + (CARRY8 if carry else ())
              + ((own, pc8.BASIS(n)), (other, pc8.BASIS(n))))
    return params, Tuple(Array(n, E4M3), *(t for _n, t in CARRY8))


def _seg8(B, S_, j: int, x, alpha, ep, th, _finite, _floor, _pad, *rest):
    e, n, dev = pc4._Emit(B), S_.KS, S_.DEV
    dead, tot, _fill, own, other = rest if len(rest) == 5 else (0, [0] * R, None, *rest)
    lo = j * n
    codes = array_of(pc8._codes(e, x, alpha, ep, own, lo, lo + n, dev))
    dead = pc8._dead(e, x, th, lo, lo + n, dead)
    return tuple_of(codes, dead, array_of([pc8._chain(e, codes, other.take(1, p), n, dev, tot[p]) for p in range(R)]),
                    array_of([e.c(16, 0)] * FILL8))


def _tail8_sig(S_, extra: tuple = ()):
    k, n = S_.K, S_.KS
    return HEAD8 + (("cs", Array(k // n, Array(n, E4M3))),) + CARRY8 + extra, Tuple(Array(k, E4M3), pc8.P_ROW, OK)


def _ok8(e: pc4._Emit, finite, floor, dead, k: int):
    return pc4.ok_word(e, e.call(BitAnd, finite, e.call(BitAnd, floor, e.call(I32Le, dead, k // 64))))


def _tail8a_body(B, S_, alpha, ep, th, finite, floor, pad, cs, dead, tot, fill):
    e, k = pc4._Emit(B), S_.K
    ew = [e.call(E4m3ToF32, ep[p]) for p in range(R)]
    p_row = [e.call(F2fpBf16, w) for w in ew] + [e.call(F2fpBf16, tot[p]) for p in range(R)]
    return tuple_of(cs.reshape(Array(k, E4M3)), array_of(p_row), _ok8(e, finite, floor, dead, k))


def _tail8b_body(B, S_, alpha, ep, th, finite, floor, pad, cs, dead, tot, fill, g):
    e, k = pc4._Emit(B), S_.K
    ew = [e.call(E4m3ToF32, ep[p]) for p in range(R)]
    mid = []
    for q in range(R):
        acc = 0
        for p in range(R):
            acc = e.call(F32FmaV2, ew[p], g[p][q], acc)
        mid.append(e.call(F2fpBf16, e.call(F32AddV2, acc, e.call(F32Neg, tot[q]))))
    p_row = mid + [e.call(W.PouwBf16Neg, e.call(F2fpBf16, w)) for w in ew]
    return tuple_of(cs.reshape(Array(k, E4M3)), array_of(p_row), _ok8(e, finite, floor, dead, k))


def _pieces8(side: str):
    own, other = ("fa", "fb") if side == "A" else ("fb", "fa")
    head = CompositeDefinition(
        f"Pc8Row{side}Head", 1, ("K",),
        lambda S_: ((("x", Array(S_.K, F32)), ("e", pc8.LINE)), Tuple(*_head_out(HEAD8, HEAD8_OUT))),
        _head8_body, doc=f"Row {side}'s head: its statistics (α, θ, the finite and floor bits) and E′ = e4m3(β·E).",
        conformance="protocol-defined")
    first = CompositeDefinition(
        f"Pc8Row{side}First", 1, ("K", "KS", "DEV"), lambda S_: _seg8_sig(S_, own, other, False),
        lambda B, S_, *a: _seg8(B, S_, 0, *a),
        doc=f"Row {side}'s first segment: its KS codes, and the dead count and the clean-up's chains over it from +0.",
        conformance="protocol-defined")
    form = CompositeDefinition(
        f"Pc8Row{side}Form", 1, ("K", "KS", "J", "DEV"), lambda S_: _seg8_sig(S_, own, other, True),
        lambda B, S_, *a: _seg8(B, S_, S_.J, *a),
        doc=f"Row {side}'s segment J >= 1: its KS codes, and the dead count and the clean-up's chains carried over it.",
        conformance="protocol-defined")
    return head, first, form


Pc8RowAHeadDef, Pc8RowAFirstDef, Pc8RowAFormDef = _pieces8("A")
Pc8RowBHeadDef, Pc8RowBFirstDef, Pc8RowBFormDef = _pieces8("B")
Pc8RowATailDef = CompositeDefinition(
    "Pc8RowATail", 1, ("K", "KS"), _tail8_sig, _tail8a_body,
    doc="Row A's tail: A′'s codes (its segments' in order), P_A = [bf16(E′) ‖ RNE_bf16(A′·F_B)] and its admission.",
    conformance="protocol-defined")
Pc8RowBTailDef = CompositeDefinition(
    "Pc8RowBTail", 1, ("K", "KS"), lambda S_: _tail8_sig(S_, (("g", pc8.GRAM),)), _tail8b_body,
    doc="Row B's tail: B̃'s codes (its segments' in order), P_B = [RNE_bf16(E′·G − B̃·F_A) ‖ −bf16(E′)] and its admission.",
    conformance="protocol-defined")


# ---- Pearl-C4's pieces ---------------------------------------------------------------------------------------------------

def _head4_body(B, S_, x):
    e = pc4._Emit(B)
    _w, alpha, ae, ok = pc4._stats(e, x, S_.K)
    return tuple_of(alpha, tuple_of(ae, e.c(8, 0)), pc4.ok_word(e, ok))


def _seg4_sig(S_, own: str, other: str, carry: bool):
    n = S_.KS
    params = ((("x", Array(S_.K, BF16)),) + HEAD4 + (CARRY4 if carry else ())
              + (("e", pc4.LINE), (own, pc4.BASIS(n)), (other, pc4.BASIS(n))))
    return params, Tuple(Array(n, E2M1), Array(n // pc4.BLOCK, U8), Array(R, F32))


def _seg4(B, S_, j: int, x, alpha, ae, _pad, *rest):
    e, n = pc4._Emit(B), S_.KS
    tot, line, own, other = rest if len(rest) == 4 else ([0] * R, *rest)
    lo = j * n
    w = [e.call(Bf16ToF32, x[t]) for t in range(lo, lo + n)]
    codes, scales = pc4._form(e, w, alpha, ae, line, own, n)
    ones = pc4._ones(e, n)
    return tuple_of(codes, scales, array_of([pc4._chain(e, codes, other.take(1, p), scales, ones, n, tot[p])
                                             for p in range(R)]))


def _tail4_sig(S_, extra: tuple = ()):
    k, n = S_.K, S_.KS
    params = HEAD4 + (("cs", Array(k // n, Array(n, E2M1))), ("ss", Array(k // n, Array(n // pc4.BLOCK, U8)))) + CARRY4
    return params + (("e", pc4.LINE),) + extra, Tuple(Array(k, E2M1), Array(k // pc4.BLOCK, U8), pc4.P_ROW)


def _tail4a_body(B, S_, alpha, ae, pad, cs, ss, tot, line):
    e, k = pc4._Emit(B), S_.K
    af = [e.call(F2fpBf16, tot[p]) for p in range(R)]
    return tuple_of(cs.reshape(Array(k, E2M1)), ss.reshape(Array(k // pc4.BLOCK, U8)),
                    array_of(pc4._e_tilde(e, line, ae) + af))


def _tail4b_body(B, S_, alpha, ae, pad, cs, ss, tot, line, g):
    e, k = pc4._Emit(B), S_.K
    et = pc4._e_tilde(e, line, ae)
    ew = [e.call(Bf16ToF32, v) for v in et]
    mid = []
    for q in range(R):
        acc = 0
        for p in range(R):
            acc = e.call(F32FmaV2, ew[p], g[p][q], acc)
        mid.append(e.call(F2fpBf16, e.call(F32AddV2, acc, e.call(F32Neg, tot[q]))))
    return tuple_of(cs.reshape(Array(k, E2M1)), ss.reshape(Array(k // pc4.BLOCK, U8)),
                    array_of(mid + [e.call(W.PouwBf16Neg, v) for v in et]))


def _pieces4(side: str):
    own, other = ("fa", "fb") if side == "A" else ("fb", "fa")
    head = CompositeDefinition(
        f"Pc4Row{side}Head", 1, ("K",), lambda S_: ((("x", Array(S_.K, BF16)),), Tuple(*_head_out(HEAD4, HEAD4_OUT), OK)),
        _head4_body, doc=f"Row {side}'s head: its statistics (α, a_E) and its admission (`pc4._stats`).",
        conformance="protocol-defined")
    first = CompositeDefinition(
        f"Pc4Row{side}First", 1, ("K", "KS"), lambda S_: _seg4_sig(S_, own, other, False),
        lambda B, S_, *a: _seg4(B, S_, 0, *a),
        doc=f"Row {side}'s first segment: its KS codes and scale bytes, and the clean-up's NVF4 chains over it from +0.",
        conformance="protocol-defined")
    form = CompositeDefinition(
        f"Pc4Row{side}Form", 1, ("K", "KS", "J"), lambda S_: _seg4_sig(S_, own, other, True),
        lambda B, S_, *a: _seg4(B, S_, S_.J, *a),
        doc=f"Row {side}'s segment J >= 1: its KS codes and scale bytes, and the clean-up's NVF4 chains carried over it.",
        conformance="protocol-defined")
    return head, first, form


Pc4RowAHeadDef, Pc4RowAFirstDef, Pc4RowAFormDef = _pieces4("A")
Pc4RowBHeadDef, Pc4RowBFirstDef, Pc4RowBFormDef = _pieces4("B")
Pc4RowATailDef = CompositeDefinition(
    "Pc4RowATail", 1, ("K", "KS"), _tail4_sig, _tail4a_body,
    doc="Row A's tail: A′'s codes and scale bytes (its segments' in order) and P_A = [Ẽ ‖ RNE_bf16(A′·F_B)].",
    conformance="protocol-defined")
Pc4RowBTailDef = CompositeDefinition(
    "Pc4RowBTail", 1, ("K", "KS"), lambda S_: _tail4_sig(S_, (("g", pc4.GRAM),)), _tail4b_body,
    doc="Row B's tail: B̃'s codes and scale bytes (its segments' in order) and P_B = [RNE_bf16(Ẽ·G − B̃·F_A) ‖ −Ẽ].",
    conformance="protocol-defined")


@lru_cache(maxsize=32)
def pieces8(k: int, ks: int, dev: str, side: str) -> tuple:
    """Pearl-C's pieces of a row of ``side`` ("a" or "b") at depth ``k`` cut every ``ks``: (head, first form, the forms
    of segments 1.., tail)."""
    head, first, form, tail = {"a": (Pc8RowAHeadDef, Pc8RowAFirstDef, Pc8RowAFormDef, Pc8RowATailDef),
                               "b": (Pc8RowBHeadDef, Pc8RowBFirstDef, Pc8RowBFormDef, Pc8RowBTailDef)}[side]
    return (bind(head, K=k), bind(first, K=k, KS=ks, DEV=dev),
            tuple(bind(form, K=k, KS=ks, J=j, DEV=dev) for j in range(1, k // ks)), bind(tail, K=k, KS=ks))


@lru_cache(maxsize=32)
def pieces4(k: int, ks: int, side: str) -> tuple:
    """Pearl-C4's pieces of a row of ``side``, as `pieces8`."""
    head, first, form, tail = {"a": (Pc4RowAHeadDef, Pc4RowAFirstDef, Pc4RowAFormDef, Pc4RowATailDef),
                               "b": (Pc4RowBHeadDef, Pc4RowBFirstDef, Pc4RowBFormDef, Pc4RowBTailDef)}[side]
    return (bind(head, K=k), bind(first, K=k, KS=ks), tuple(bind(form, K=k, KS=ks, J=j) for j in range(1, k // ks)),
            bind(tail, K=k, KS=ks))


def _col(out, m: int, f: int):
    return array_of([out[i][f] for i in range(m)])


def _rows(B, pieces, ks: int, x, before: tuple, after: tuple, head_out: tuple, n_codes: int, extra: tuple):
    """One side's row units in a root body: the heads, each segment's forms in order, then the tails, every batch over the
    side's rows.  ``before`` are the heads' per-row inputs after the row (Pearl-C's E line), ``after`` the forms' and
    tails' after the carry (Pearl-C4's); ``head_out`` the head's output ports (`HEAD8_OUT`), whose fields the forms and
    tail read in order; ``extra`` the tails' shared inputs (G)."""
    head, first, forms, tail = pieces
    m = len(x)
    heads = B.batch(head, x, *before, axes=(0,) * (1 + len(before)))
    h = [array_of([heads[i][f] if len(p) == 1 else heads[i][f][s] for i in range(m)])
         for f, p in enumerate(head_out) for s in range(len(p))]
    segs, carry = [], []
    for j, fn in enumerate((first, *forms)):
        own, other = (a[j * ks:(j + 1) * ks] for a in extra[:2])
        mapped = (x, *h, *carry, *after)
        out = B.batch(fn, *mapped, own, other, axes=(0,) * len(mapped) + (None, None))
        segs.append([_col(out, m, f) for f in range(n_codes)])
        carry = [_col(out, m, f) for f in range(n_codes, len(fn.signature()[1].fields))]
    cs = [array_of([array_of([segs[j][f][i] for j in range(len(segs))]) for i in range(m)]) for f in range(n_codes)]
    mapped = (*h, *cs, *carry, *after)
    tails = B.batch(tail, *mapped, *extra[2:], axes=(0,) * len(mapped) + (None,) * len(extra[2:]))
    return heads, [_col(tails, m, f) for f in range(len(tail.signature()[1].fields))]


def rows8(B, k: int, ks: int, dev: str, side: str, x, e, own, other, *extra):
    """Pearl-C's rows of ``side`` cut every ``ks``, in a root body: [codes, P, ok] by row, as `pc8`'s row units return
    them; ``own`` and ``other`` the bases it forms and chains against, ``extra`` the tails' G."""
    _heads, outs = _rows(B, pieces8(k, ks, dev, side), ks, x, (e,), (), HEAD8_OUT, 1, (own, other, *extra))
    return outs


def rows4(B, k: int, ks: int, side: str, x, e, own, other, *extra):
    """Pearl-C4's rows of ``side`` cut every ``ks``, in a root body: [codes, scale bytes, P, ok] by row, as `pc4`'s row
    units return them (ok the head's)."""
    heads, outs = _rows(B, pieces4(k, ks, side), ks, x, (), (e,), HEAD4_OUT, 2, (own, other, *extra))
    return outs + [_col(heads, len(x), len(HEAD4_OUT))]


def check8(shape) -> None:
    step = math.lcm(pc8.SLICE, DEVICES[shape.dev].window)
    if not (type(shape.row_k) is int and shape.row_k > 0 and shape.row_k % step == 0 and shape.k % shape.row_k == 0):
        raise DomainError(f"Pearl-C's row cut needs row_k a multiple of {step} (whole windows and 32-position blocks) "
                          f"that divides k: {shape}")


def check4(shape) -> None:
    if not (type(shape.row_k) is int and shape.row_k > 0 and shape.row_k % pc4.ATOM == 0 and shape.k % shape.row_k == 0):
        raise DomainError(f"Pearl-C4's row cut needs row_k a multiple of {pc4.ATOM} that divides k: {shape}")


# ---- the roots: today's signatures, each row unit cut -------------------------------------------------------------------

def _check8_body(B, S_, a, ea, b, eb, fa, fb, g, hidden: bool):
    bc, pb, ok_b = pc8._weight_units(B, S_.K, S_.DEV, b, eb, fa, fb, g, row_k=S_.KS)
    return tuple_of(*pc8._call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, a, ea, fa, fb, bc, pb, hidden, S_.KS), ok_b)


def _check4_body(B, S_, a, ea, b, eb, fa, fb, g, hidden: bool):
    bc, bs, pb, ok_b = pc4._weight_units(B, S_.K, b, eb, fa, fb, g, row_k=S_.KS)
    return tuple_of(*pc4._call_units(B, S_.K, (S_.TM, S_.TN), a, ea, fa, fb, bc, bs, pb, hidden, S_.KS), ok_b)


_BLOCK8, _BLOCK4 = ("M", "K", "N", "TM", "TN", "DEV", "KS"), ("M", "K", "N", "TM", "TN", "KS")
_HIDDEN = "with each row unit cut every KS positions (`rowk`)."
Pc8WeightRowKDef = CompositeDefinition(
    "Pc8WeightRowK", 1, ("K", "N", "DEV", "KS"), pc8._weight_sig,
    lambda B, S_, *a: tuple_of(*pc8._weight_units(B, S_.K, S_.DEV, *a, row_k=S_.KS)),
    doc="`Pc8Weight` " + _HIDDEN, conformance="protocol-defined")
Pc8CallRowKDef = CompositeDefinition(
    "Pc8CallRowK", 1, _BLOCK8, pc8._call_sig,
    lambda B, S_, *a: tuple_of(*pc8._call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, *a, row_k=S_.KS)),
    doc="`Pc8Call` " + _HIDDEN, conformance="protocol-defined")
Pc8CheckRowKDef = CompositeDefinition(
    "Pc8CheckRowK", 1, _BLOCK8, pc8._check_sig, lambda B, S_, *a: _check8_body(B, S_, *a, False),
    doc="`Pc8Check` " + _HIDDEN, conformance="protocol-defined")
Pc8CallHiddenRowKDef = CompositeDefinition(
    "Pc8CallHiddenRowK", 1, _BLOCK8, lambda S_: (pc8._call_sig(S_)[0], Tuple(pc4.DIGESTS(S_), Array(S_.M, OK))),
    lambda B, S_, *a: tuple_of(*pc8._call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, *a, hidden=True, row_k=S_.KS)),
    doc="`Pc8CallHidden` " + _HIDDEN, conformance="protocol-defined")
Pc8CheckHiddenRowKDef = CompositeDefinition(
    "Pc8CheckHiddenRowK", 1, _BLOCK8,
    lambda S_: (pc8._check_sig(S_)[0], Tuple(pc4.DIGESTS(S_), Array(S_.M, OK), Array(S_.N, OK))),
    lambda B, S_, *a: _check8_body(B, S_, *a, True), doc="`Pc8CheckHidden` " + _HIDDEN, conformance="protocol-defined")
Pc4WeightRowKDef = CompositeDefinition(
    "Pc4WeightRowK", 1, ("K", "N", "KS"), pc4._weight_sig,
    lambda B, S_, *a: tuple_of(*pc4._weight_units(B, S_.K, *a, row_k=S_.KS)),
    doc="`Pc4Weight` " + _HIDDEN, conformance="protocol-defined")
Pc4CallRowKDef = CompositeDefinition(
    "Pc4CallRowK", 1, _BLOCK4, pc4._call_sig,
    lambda B, S_, *a: tuple_of(*pc4._call_units(B, S_.K, (S_.TM, S_.TN), *a, row_k=S_.KS)),
    doc="`Pc4Call` " + _HIDDEN, conformance="protocol-defined")
Pc4CheckRowKDef = CompositeDefinition(
    "Pc4CheckRowK", 1, _BLOCK4, pc4._check_sig, lambda B, S_, *a: _check4_body(B, S_, *a, False),
    doc="`Pc4Check` " + _HIDDEN, conformance="protocol-defined")
Pc4CallHiddenRowKDef = CompositeDefinition(
    "Pc4CallHiddenRowK", 1, _BLOCK4, lambda S_: (pc4._call_sig(S_)[0], Tuple(pc4.DIGESTS(S_), Array(S_.M, OK))),
    lambda B, S_, *a: tuple_of(*pc4._call_units(B, S_.K, (S_.TM, S_.TN), *a, hidden=True, row_k=S_.KS)),
    doc="`Pc4CallHidden` " + _HIDDEN, conformance="protocol-defined")
Pc4CheckHiddenRowKDef = CompositeDefinition(
    "Pc4CheckHiddenRowK", 1, _BLOCK4,
    lambda S_: (pc4._check_sig(S_)[0], Tuple(pc4.DIGESTS(S_), Array(S_.M, OK), Array(S_.N, OK))),
    lambda B, S_, *a: _check4_body(B, S_, *a, True), doc="`Pc4CheckHidden` " + _HIDDEN, conformance="protocol-defined")

_ROOTS8 = {("weight", False): Pc8WeightRowKDef, ("weight", True): Pc8WeightRowKDef, ("call", False): Pc8CallRowKDef,
           ("call", True): Pc8CallHiddenRowKDef, ("check", False): Pc8CheckRowKDef, ("check", True): Pc8CheckHiddenRowKDef}
_ROOTS4 = {("weight", False): Pc4WeightRowKDef, ("weight", True): Pc4WeightRowKDef, ("call", False): Pc4CallRowKDef,
           ("call", True): Pc4CallHiddenRowKDef, ("check", False): Pc4CheckRowKDef, ("check", True): Pc4CheckHiddenRowKDef}


def root8(shape, side: str) -> SpecializedDefinition:
    """``pc8.weight``, ``call`` or ``check`` (``side``) of a shape with ``row_k``."""
    defn = _ROOTS8[side, shape.hidden]
    block = {} if side == "weight" else {"M": shape.m, "TM": shape.tile[0], "TN": shape.tile[1]}
    return bind(defn, K=shape.k, N=shape.n, DEV=shape.dev, KS=shape.row_k, **block)


def root4(shape, side: str) -> SpecializedDefinition:
    """``pc4.weight``, ``call`` or ``check`` (``side``) of a shape with ``row_k``."""
    defn = _ROOTS4[side, shape.hidden]
    block = {} if side == "weight" else {"M": shape.m, "TM": shape.tile[0], "TN": shape.tile[1]}
    return bind(defn, K=shape.k, N=shape.n, KS=shape.row_k, **block)


# ---- the partition -------------------------------------------------------------------------------------------------------

#: each side's roles in the order its root body first uses them (`pc4.ROLES` under the cut)
ROLES = {"weight": ("row_b_head", "row_b_first", "row_b_form", "row_b_tail"),
         "call": ("row_a_head", "row_a_first", "row_a_form", "row_a_tail", "tile")}
ROLES["check"] = ROLES["weight"] + ROLES["call"]
#: the roles whose units are one segment of a row, `pc4.ProofBlock.where` (row, segment)
FORMS = ("row_a_first", "row_a_form", "row_b_first", "row_b_form")
ROLE_OF = {id(d): f"row_{side.lower()}_{piece}"
           for scheme in (
               {"A": (Pc8RowAHeadDef, Pc8RowAFirstDef, Pc8RowAFormDef, Pc8RowATailDef),
                "B": (Pc8RowBHeadDef, Pc8RowBFirstDef, Pc8RowBFormDef, Pc8RowBTailDef)},
               {"A": (Pc4RowAHeadDef, Pc4RowAFirstDef, Pc4RowAFormDef, Pc4RowATailDef),
                "B": (Pc4RowBHeadDef, Pc4RowBFirstDef, Pc4RowBFormDef, Pc4RowBTailDef)})
           for side, defs in scheme.items() for piece, d in zip(("head", "first", "form", "tail"), defs)}


def _flat(pieces: tuple) -> list:
    head, first, forms, tail = pieces
    return [head, first, *forms, tail]


def unit_templates8(shape, side: str) -> list[SpecializedDefinition]:
    """The side's unit templates under the cut, in the order its root body first uses them (`pc8.query`)."""
    b, a = (_flat(pieces8(shape.k, shape.row_k, shape.dev, s)) for s in ("b", "a"))
    tile = pc8.templates(shape.k, shape.tile, shape.dev, shape.hidden)["tile"]
    return {"weight": b, "call": a + [tile], "check": b + a + [tile]}[side]


def unit_templates4(shape, side: str) -> list[SpecializedDefinition]:
    """The side's unit templates under the cut, in the order its root body first uses them (`pc4.query`)."""
    b, a = (_flat(pieces4(shape.k, shape.row_k, s)) for s in ("b", "a"))
    tile = pc4.templates(shape.k, shape.tile, shape.hidden)["tile"]
    return {"weight": b, "call": a + [tile], "check": b + a + [tile]}[side]


# ---- the ports and the slice anchors ------------------------------------------------------------------------------------

def _ports(head_row: str, carry_row: str, codes_row: str, data_row: str, input_row: str, scheme: str) -> dict:
    """One scheme's ports under the cut, as `pc4.PORTS`: each piece's committed rows and public input regions."""
    out = {}
    head8 = scheme == "pc8"
    names, carry = (_NAMES8, tuple(n for n, _t in CARRY8)) if head8 else (_NAMES4, ("tot",))
    head_outs = tuple(range(len(HEAD8_OUT if head8 else HEAD4_OUT)))
    codes_out = (0,) if head8 else (0, 1)
    carry_out = (1, 2, 3) if head8 else (2,)
    tail_codes = ("cs",) if head8 else ("cs", "ss")
    for side, role in (("b", ROLE_W), ("a", ROLE_X)):
        own, other = ("fa", "fb") if side == "a" else ("fb", "fa")
        e = {"e": "ea" if side == "a" else "eb"}
        slices = {own: f"{own}-slice", other: f"{other}-slice"}
        out[f"row_{side}_head"] = {
            "rows": ((input_row, role, ("x",)), (head_row, role, head_outs))
            + (() if head8 else (("ok", role, (len(head_outs),)),)), "regions": e if head8 else {}}
        seg_rows = ((input_row, role, ("x",)), (head_row, role, names))
        seg_out = ((codes_row, role, codes_out), (carry_row, role, carry_out))
        seg_regions = slices if head8 else e | slices
        out[f"row_{side}_first"] = {"rows": seg_rows + seg_out, "regions": seg_regions}
        out[f"row_{side}_form"] = {"rows": seg_rows + ((carry_row, role, carry),) + seg_out, "regions": seg_regions}
        tail_out = ((data_row, role, (0, 1)), ("ok", role, (2,))) if head8 else ((data_row, role, (0, 1, 2)),)
        out[f"row_{side}_tail"] = {
            "rows": ((head_row, role, names), (codes_row, role, tail_codes), (carry_row, role, carry)) + tail_out,
            "regions": ({} if head8 else e) | ({"g": "g"} if side == "b" else {})}
    return out


#: each piece's ports under hiding, by role (`pc8.PORTS`'s form; the tile's are today's)
PORTS8 = _ports("pc8-head", "pc8-carry", "fp8-codes", "fp8-p", "fp32", "pc8") | {"tile": pc8.PORTS["tile"]}
PORTS4 = _ports("pc4-head", "pc4-carry", "nvfp4-codes", "nvfp4-p", "bf16", "pc4") | {"tile": pc4.PORTS["tile"]}


def _segment_of(basis):
    """Segment j of a basis's lines, positions j·row_k <= t < (j+1)·row_k, flat in its region's leaf order."""
    return lambda shape, seed, j: basis(shape, seed)[R * shape.row_k * j:R * shape.row_k * (j + 1)]


def _slice_rule(rule: str) -> str:
    """The slice rule of an anchor's rule, at its version: ``…/<anchor>/v<n>`` -> ``…/<anchor>/slice/v<n>``."""
    head, version = rule.rsplit("/", 1)
    return f"{head}/slice/{version}"


def _slices(today: dict) -> dict:
    return {f"{a}-slice": A.Derivation(_slice_rule(today[a].rule), today[a].seed,
                                       today[a].domain + "; segment j of it: its lines for j·row_k <= t < (j+1)·row_k",
                                       _segment_of(today[a].fn), today[a].bits, per_segment=True) for a in ("fa", "fb")}


#: each anchor's recomputation under the cut: today's, and each form's segment of F_A and F_B under its slice rule
DERIVATIONS8 = pc8.DERIVATIONS | _slices(pc8.DERIVATIONS)
DERIVATIONS4 = pc4.DERIVATIONS | _slices(pc4.DERIVATIONS)


__all__ = ["CARRY4", "CARRY8", "DERIVATIONS4", "DERIVATIONS8", "FILL8", "FORMS", "HEAD4", "HEAD4_OUT", "HEAD8", "HEAD8_OUT",
           "PORTS4", "PORTS8", "ROLES", "ROLE_OF", "ROW_K", "check4", "check8", "pieces4", "pieces8", "root4", "root8", "rows4",
           "rows8", "segment", "unit_templates4", "unit_templates8"]
