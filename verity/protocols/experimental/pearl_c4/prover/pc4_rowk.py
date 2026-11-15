"""Pearl-C4's row units under Flock's block (``Shape.row_k``): `pc4`'s row units cut along k into a head, one form per
segment of row_k positions and a tail, as `rowk` cuts Pearl-C's (`rowk`'s docstring states the cut).

Pearl-C4's pieces (`Pc4Row{A,B}{Head,First,Form,Tail}`): its head is `pc4._stats` (its admission is complete there, so the
head writes the row's ok row), its forms `pc4._form` on the segment with the NVF4 chains carried, and its tail Ẽ and P.
Each form decodes its segment's BF16 words again (`Bf16ToF32`, core's wiring), the only gate that two units hold.  Each
piece's rows (`PORTS4`; `leaves`' pc4-head, pc4-carry and nvfp4-codes) are Flock-committed ports in the row's session,
and a form reads its segment of F_A and F_B as a public input region under the anchor's slice rule (`DERIVATIONS4`).
"""
from __future__ import annotations

from functools import lru_cache

from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple
from verity.circuits.ops.fp32 import F32AddV2, F32FmaV2
from verity.circuits.ops.prims import Bf16ToF32, F2fpBf16
from verity.circuits.ops.scalar import F32Neg

from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import DomainError
from . import pc4
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import words as W
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.rowk import (BF16, CARRY4, E2M1, F32, HEAD4, HEAD4_OUT, OK, R, U8, _HIDDEN, _col, _flat, _head_out, _ports, _rows,
                   _slices)


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
def pieces4(k: int, ks: int, side: str) -> tuple:
    """Pearl-C4's pieces of a row of ``side``, as `pieces8`."""
    head, first, form, tail = {"a": (Pc4RowAHeadDef, Pc4RowAFirstDef, Pc4RowAFormDef, Pc4RowATailDef),
                               "b": (Pc4RowBHeadDef, Pc4RowBFirstDef, Pc4RowBFormDef, Pc4RowBTailDef)}[side]
    return (bind(head, K=k), bind(first, K=k, KS=ks), tuple(bind(form, K=k, KS=ks, J=j) for j in range(1, k // ks)),
            bind(tail, K=k, KS=ks))


def rows4(B, k: int, ks: int, side: str, x, e, own, other, *extra):
    """Pearl-C4's rows of ``side`` cut every ``ks``, in a root body: [codes, scale bytes, P, ok] by row, as `pc4`'s row
    units return them (ok the head's)."""
    heads, outs = _rows(B, pieces4(k, ks, side), ks, x, (), (e,), HEAD4_OUT, 2, (own, other, *extra))
    return outs + [_col(heads, len(x), len(HEAD4_OUT))]


def check4(shape) -> None:
    if not (type(shape.row_k) is int and shape.row_k > 0 and shape.row_k % pc4.ATOM == 0 and shape.k % shape.row_k == 0):
        raise DomainError(f"Pearl-C4's row cut needs row_k a multiple of {pc4.ATOM} that divides k: {shape}")


# ---- the roots: today's signatures, each row unit cut -------------------------------------------------------------------

def _check4_body(B, S_, a, ea, b, eb, fa, fb, g, hidden: bool):
    bc, bs, pb, ok_b = pc4._weight_units(B, S_.K, b, eb, fa, fb, g, row_k=S_.KS)
    return tuple_of(*pc4._call_units(B, S_.K, (S_.TM, S_.TN), a, ea, fa, fb, bc, bs, pb, hidden, S_.KS), ok_b)


_BLOCK4 = ("M", "K", "N", "TM", "TN", "KS")
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

_ROOTS4 = {("weight", False): Pc4WeightRowKDef, ("weight", True): Pc4WeightRowKDef, ("call", False): Pc4CallRowKDef,
           ("call", True): Pc4CallHiddenRowKDef, ("check", False): Pc4CheckRowKDef, ("check", True): Pc4CheckHiddenRowKDef}


def root4(shape, side: str) -> SpecializedDefinition:
    """``pc4.weight``, ``call`` or ``check`` (``side``) of a shape with ``row_k``."""
    defn = _ROOTS4[side, shape.hidden]
    block = {} if side == "weight" else {"M": shape.m, "TM": shape.tile[0], "TN": shape.tile[1]}
    return bind(defn, K=shape.k, N=shape.n, KS=shape.row_k, **block)


# ---- the partition -------------------------------------------------------------------------------------------------------

#: each piece's role, by its Definition's id (`rowk.ROLE_OF`'s form)
ROLE_OF = {id(d): f"row_{side.lower()}_{piece}"
           for side, defs in {"A": (Pc4RowAHeadDef, Pc4RowAFirstDef, Pc4RowAFormDef, Pc4RowATailDef),
                              "B": (Pc4RowBHeadDef, Pc4RowBFirstDef, Pc4RowBFormDef, Pc4RowBTailDef)}.items()
           for piece, d in zip(("head", "first", "form", "tail"), defs)}


def unit_templates4(shape, side: str) -> list[SpecializedDefinition]:
    """The side's unit templates under the cut, in the order its root body first uses them (`pc4.query`)."""
    b, a = (_flat(pieces4(shape.k, shape.row_k, s)) for s in ("b", "a"))
    tile = pc4.templates(shape.k, shape.tile, shape.hidden)["tile"]
    return {"weight": b, "call": a + [tile], "check": b + a + [tile]}[side]


# ---- the ports and the slice anchors ------------------------------------------------------------------------------------

#: each piece's ports under hiding, by role (`pc4.PORTS`'s form; the tile's are today's)
PORTS4 = _ports("pc4-head", "pc4-carry", "nvfp4-codes", "nvfp4-p", "bf16", "pc4") | {"tile": pc4.PORTS["tile"]}


#: each anchor's recomputation under the cut: today's, and each form's segment of F_A and F_B under its slice rule
DERIVATIONS4 = pc4.DERIVATIONS | _slices(pc4.DERIVATIONS)


__all__ = ["DERIVATIONS4", "PORTS4", "ROLE_OF", "check4", "pieces4", "root4", "rows4", "unit_templates4"]
