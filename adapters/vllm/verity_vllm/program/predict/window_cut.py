"""A window circuit (`window_circuit`) cut at its linears, given Pearl+'s tables (`PearlTables`, `window_class`): PoUW
credits each linear's product, so the circuit computes none but the LM head's. Each `Gemm_v2{K, N}` but an LM head (a GEMM
a token select reads, which stays computed) is round s's call c, the c-th cut linear of its position's forward, and is one
unit `LinearARow_v1{K, R, ROUND, CALL}` (`registry.linear_cut`), reading its x as its writers' R rows, its output held to
its row of the call's A rows, and one unit `LinearOut_v1{N, W, ROUND, CALL}` reading its row of the call's y and handing it
on as N / W rows of W words (`linears`), which its readers read as they would its slices (`cut_linear`). Each call's
readers of each of its tables are a template of their own (ROUND and CALL bound), the circuit's program takes the tables
as its values (`cut_values`), and `census` counts the cut linears by shape.

Refused by name (`Unexpressible`): a class Program with an int8 linear or no linear but its LM heads, a position whose
linears aren't the first position's, and a linear whose y no rows of ``words`` doubled hand on in one unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity.circuits.defs import bind
from verity.circuits.refs import runs
from verity.definitions.registry import linear_cut as LC
from verity.commitments.windows import interface as IF

from .build import Unexpressible
from .window_builder import Row
from .window_class import PearlTables, WindowClass


@dataclass(frozen=True)
class Linears:
    """A class Program's linears as a cut at them takes them (`PearlTables`): ``heads`` its LM heads' nodes (GEMMs a token
    select reads, which stay computed) and ``calls`` each position's other GEMMs in order, the same at every position, each
    ``(K, N, W)``: its shape and the words of each row it hands its y on in."""
    heads: frozenset[int]
    calls: tuple[tuple[int, int, int], ...]


def _hands_on(N: int, W: int, rows: int, provable: bool) -> bool:
    """Whether `LinearOut_v1{N, W}` is within ``rows`` rows and, unless not ``provable``, a proof block C-Flock proves."""
    if N // W + 1 > rows:
        return False
    if provable:
        try:
            IF.check_proof_block(bind(LC.LinearOut, N=N, W=W, ROUND=0, CALL=0))
        except ValueError:
            return False
    return True


def linears(fn: Any, words: int, rows: int, provable: bool = True) -> Linears:
    """The class Program ``fn``'s `Linears`, each call's W the least of ``words`` doubled that divides its N and leaves its
    `LinearOut_v1` within ``rows`` rows and, unless not ``provable``, a proof block C-Flock's composer proves
    (`interface.check_proof_block`; a call's y row is read whole). Refused where it has an int8 linear, a position's
    linears aren't the first position's, an N has no such W, or there is none to cut."""
    nodes = fn.body.nodes
    if (q := next((n for n in nodes if n.fn.name == "Int8RowQuant"), None)) is not None:
        raise Unexpressible(f"{q.fn.id}: an int8 linear, and Pearl+'s tables cut BF16 linears (`Gemm_v2`) alone")
    heads = frozenset(idx for n in nodes if n.fn.name == "TokenSelect"
                      for space, idx, _lo, _hi in runs(n.args[0].refs) if space == "n")
    by: list[list[tuple[int, int]]] = []
    for j, n in enumerate(nodes):
        if n.fn.name == "Embedding":
            by.append([])
        elif n.fn.name == "Gemm" and j not in heads and by:
            by[-1].append((n.fn.bindings["K"], n.fn.bindings["N"]))
    if not by or not by[0]:
        raise Unexpressible(f"{fn.id}: no BF16 linear (`Gemm_v2`) but its LM heads, so nothing for Pearl+'s tables to cut")
    if (p := next((p for p, d in enumerate(by) if d != by[0]), None)) is not None:
        raise Unexpressible(f"{fn.id}: position {p}'s linears (K, N) {by[p]}, not position 0's {by[0]}: a cut reads every "
                            "position's call c at the same rows of its round's call c")
    calls = []
    for K, N in by[0]:
        W = words
        while N % W == 0 and not _hands_on(N, W, rows, provable):
            W *= 2
        calls.append((K, N, W))
    if bad := sorted({(K, N) for K, N, W in calls if N % W}):
        raise Unexpressible(f"{fn.id}: linears (K, N) {bad}, whose y no rows of Pearl+'s {words} words doubled hand on in "
                            f"one unit within {rows} rows" + (" that C-Flock proves" if provable else ""))
    return Linears(heads, tuple(calls))


def cut_linear(P: Any, j: int, node: Any) -> None:
    """Node ``j`` of the slot's lowered Program ``P`` (`window_circuit`'s, its ``cut`` the `Linears`), a linear
    `Gemm_v2{K, N}` cut at: round s's call c, c its index among its position's cut linears (``P.linears[-1]``), one unit
    `LinearARow_v1{K, R, ROUND, CALL}` reading its x as the R rows its writers hold it in, its A row (``P.arows[j]``, x
    widened, which no reader reads) held to row i of the call's A rows, and one unit `LinearOut_v1{N, W, ROUND, CALL}`
    reading row i of the call's y (``P.ys[j]``, y widened, which the window commits by reference from the call's y), which
    its readers read as its N / W output rows of W words: i its position's row (`FixedLoad.at`). Its weights are read
    nowhere."""
    t, calls, K, N = P.tables, P.cut.calls, node.fn.bindings["K"], node.fn.bindings["N"]
    xs, c = P.rows(node.args[0].refs), P.linears[-1]
    if K % len(xs):
        raise Unexpressible(f"{node.fn.id}: its x is {len(xs)} writer rows, which don't divide its {K} values")
    if c >= len(calls) or calls[c][:2] != (K, N):
        raise Unexpressible(f"{P.name}: node {j} ({node.fn.id}) is its position's cut linear {c}, not the layout's "
                            f"(K, N) {[call[:2] for call in calls]}")
    P.linears[-1], W = c + 1, calls[c][2]
    s, i = t.at(P.cls, P.slot_index, P.position - 1)
    tid, u = P.b.unit(bind(LC.LinearARow, K=K, R=len(xs), ROUND=s, CALL=c))
    P.b.feed(tid, "x", u, 1, xs)
    P.b.value(tid, "o0", u, t.a(s, c), i, t.a_rows(P.cls, s))
    P.arows[j] = Row(tid, u, "o0", 2 * K)
    ytid, yu = P.b.unit(bind(LC.LinearOut, N=N, W=W, ROUND=s, CALL=c))
    P.b.value(ytid, "y", yu, t.y(s, c), i, t.rows(P.cls, s))
    P.ys[j] = Row(ytid, yu, "y", 2 * N)
    P.outs[j] = [(k * W, (k + 1) * W, Row(ytid, yu, f"o{k}", W)) for k in range(N // W)]


#: the rows of a call's A rows and y as Pearl+'s trees commit them, which their readers' ports read
#: (`interface.Registered.formats`): FP32 words, 32-bit leaves, in ``hm96-sha512/row-seg/v1`` rows
ROW_FORMAT = {"row_schema": IF.SEG, "word_bits": 32}


def cut_values(t: PearlTables, cls: WindowClass, layout: Linears) -> tuple[str, ...]:
    """The values the cut circuit's program takes (`interface.Registered.values`, each of `ROW_FORMAT`): round by round and
    call by call, the call's A rows and then its y."""
    return tuple(v for s in range(cls.outputs) for c in range(len(layout.calls)) for v in (t.a(s, c), t.y(s, c)))


#: why a cut linear's y is untied: the prover's registered value, held to nothing PoUW credits
UNTIED = ("Pearl+'s verifier holds its own trees of each call's A rows and y (row-seg rows of FP32 words under PoUW's "
          "bindings) to its credited product, and the circuit reads the window's registered values of them in that "
          "format; unless the window's registration reuses those trees' roots, nothing holds the two to each other, so "
          "the y the circuit reads is the prover's")


def census(lowered: Any) -> list[dict[str, Any]]:
    """The cut linears of ``lowered``'s class (`window_circuit.Lowered`), a row a round's call: its round and call (its
    index among a position's cut linears), its K, N and W, its rows (the round's, `PearlTables.rows`) and multiply-adds, its
    A rows' value and rows (`PearlTables.a_rows`) and its y's, PoUW's scheme whose credited product a tie would hold y to,
    and whether one does (none yet: `UNTIED`). Empty for a circuit not cut. A circuit's own description gives the same
    multiply-adds: each call's `LinearARow_v1{K, R, ROUND, CALL}` units times K and its `LinearOut_v1{N, W, ROUND, CALL}`'s
    N."""
    P = lowered.programs[0]
    if P.cut is None:
        return []
    t, cls, out = P.tables, P.cls, []
    for s in range(cls.outputs):
        m = t.rows(cls, s)
        for c, (k, n, w) in enumerate(P.cut.calls):
            out.append({"round": s, "call": c, "k": k, "n": n, "w": w, "rows": m, "macs": m * n * k,
                        "a": {"value": t.a(s, c), "rows": t.a_rows(cls, s)}, "y": {"value": t.y(s, c), "rows": m},
                        "scheme": t.scheme, "tied": False, "untied": UNTIED})
    return out


__all__ = ["Linears", "ROW_FORMAT", "UNTIED", "census", "cut_linear", "cut_values", "linears"]
