"""PoUS's session verdict as one proof block: whether a memory challenger's session passes, from its round records and the
expected digests of the labels they challenged.  Its one output bit is the session's public verdict.

    PousVerdict_v1{K, M, LIMIT}
        (records: Array<K, Array<736, Value<1>>>, expected: Array<K, Array<512, Value<1>>>) -> Value<1>

Every array is of `Value<1>`, and a byte string is bits in byte order, bit t being bit t mod 8 of byte ⌊t/8⌋.
`records[j]` is round j's record as the memory challenger writes it (`memory_challenger.Round.value`, 92 bytes):
``u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)``, integers big-endian.  `expected[j]` is the
``sha512/row/v2`` digest of label ``index`` of C, what `memory_challenger.verdict`'s ``expected(index)`` returns.

**What it computes.**  Round j *hits* when its record names round j, its receipt is not before its send, ``receipt − send
≤ LIMIT`` (Δ + RTT in ns) and its ``H(block)`` is ``expected[j]``; every other round misses (`round_verdict`).  The output
is 1 iff at most M of the K rounds miss: `Pous.contAccepts M` over the rounds' verdicts
(`Security/Definitions/Pous/Model/Continuous.lean`), the continuous audit's ``max_misses = M`` over ``n = K`` challenges.
At M = 0 it is the per-round part of `memory_challenger.verdict`: no round wrong, none late, every record its own round.
`PousVerdict.verdictProg_accepts` (`Security/Proofs/Pous/Continuous/Verdict.lean`) proves it of the gates.

**The gates** (`gates`, which `PousVerdict.verdictProg` restates, and the body is written from gate for gate):
the K·1248 input bits, the constants 0 and 1, then per round

- one ``Xor3(a, b, 1)`` (``a = b``) for each of the 32 bits of j against round j's constant and each of the 512 digest
  bits against ``expected[j]``;
- ``receipt + ¬send + 1`` as a carry chain (`chain`): ``p_i = a_i ⊕ s_i ⊕ 1``, then per bit ``c_{i+1} = a_i ⊕ p_i·(a_i ⊕
  c_i)``, a majority with one AND; its last carry is ``send ≤ receipt``, and ``d_i = p_i ⊕ c_i`` is ``receipt − send``;
- ``LIMIT + ¬d + 1`` as another chain, its last carry ``d ≤ LIMIT``;
- the round's hit, one ``And546`` over the 544 equalities and the two carries;

then a W-bit miss counter, W = bit_length(max(K, M)), incremented per round by the chain ``cnt + 0 + ¬hit`` (``s = 1``),
and the verdict, the last carry of ``M + ¬cnt + 1``.  Consecutive gates of one kind and width that read only earlier gates
are one batch.  ANDs: 673 per round (545 in the hit, 64 per chain), W per round in the counter and W in the verdict.

**Why hash-free.**  The record holds ``H(block)``, the memory challenger's own digest of the answer, so the verdict needs
no SHA-512: it compares digests, and its gates are small enough to prove in Lean gate by gate.  What ties its inputs to
the session is the rest of the statement: ``records`` are the rows the signed records root commits, and ``expected`` the
digests of C's labels at the opened coins.
"""
from __future__ import annotations

from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array
from verity.ml.commitments.scheme import DIGEST_BYTES

__all__ = ["DIGEST_BITS", "RECORD_BITS", "PousVerdict", "chain", "counter_width", "gates", "round_hit", "round_verdict",
           "verdict_value"]

BIT = BL.BIT

RECORD_BITS = 8 * 92
"""A round record's bits: ``u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)``."""

DIGEST_BITS = 8 * DIGEST_BYTES["sha512"]

J_AT, INDEX_AT, SEND_AT, RECEIPT_AT, DIGEST_AT = 0, 4, 12, 20, 28
"""The record's fields, by byte."""

LIMIT_BITS = 64


def _statics(K: int, M: int, LIMIT: int) -> None:
    for name, v, lo in (("K", K, 1), ("M", M, 0), ("LIMIT", LIMIT, 1)):
        if isinstance(v, bool) or not isinstance(v, int) or v < lo:
            raise ValueError(f"PousVerdict_v1's {name} is an int >= {lo}, not {v!r}")
    if LIMIT >> LIMIT_BITS:
        raise ValueError(f"PousVerdict_v1's LIMIT is below 2^{LIMIT_BITS} ns, not {LIMIT}")


def counter_width(K: int, M: int) -> int:
    """The miss counter's width: every count up to K, and M, fit."""
    return max(K, M).bit_length()


def be_wires(base: int, at: int, nbytes: int) -> list[int]:
    """The wires of the big-endian integer at bytes ``at .. at + nbytes − 1`` of the byte string at wire ``base``, least
    significant bit first."""
    return [base + 8 * (at + nbytes - 1 - k // 8) + k % 8 for k in range(8 * nbytes)]


def const_wires(zero: int, one: int, width: int, value: int) -> list[int]:
    return [one if value >> i & 1 else zero for i in range(width)]


Gate = tuple


def chain(gs: list[Gate], one: int, a: Sequence[int], s: Sequence[int], c0: int) -> tuple[list[int], list[int]]:
    """``a + ¬s + c0`` after ``gs``: the ``p_i = a_i ⊕ s_i ⊕ 1`` wires and the carries ``c_0 = c0, …, c_n``."""
    p0 = len(gs)
    gs.extend(("xor", (x, y, one)) for x, y in zip(a, s))
    ps = list(range(p0, len(gs)))
    cs = [c0]
    for x, p in zip(a, ps):
        q = len(gs)
        gs.append(("xor", (x, cs[-1])))
        gs.append(("and", (p, q)))
        gs.append(("xor", (x, q + 1)))
        cs.append(q + 2)
    return ps, cs


def _sums(gs: list[Gate], ps: Sequence[int], cs: Sequence[int]) -> list[int]:
    d0 = len(gs)
    gs.extend(("xor", (p, c)) for p, c in zip(ps, cs))
    return list(range(d0, len(gs)))


def gates(K: int, M: int, LIMIT: int) -> tuple[list[Gate], list[int]]:
    """The proof block's gates in layout order and its output wire: ``("input",)``, ``("const", b)``, ``("not", (i,))``,
    ``("and", wires)``, ``("xor", wires)``, wire i being gate i.  `PousVerdict.verdictProg` restates it builder for
    builder."""
    _statics(K, M, LIMIT)
    n = (RECORD_BITS + DIGEST_BITS) * K
    gs: list[Gate] = [("input",)] * n + [("const", 0), ("const", 1)]
    zero, one = n, n + 1
    limit = const_wires(zero, one, LIMIT_BITS, min(LIMIT, (1 << LIMIT_BITS) - 1))
    hits = []
    for j in range(K):
        row, exp = RECORD_BITS * j, RECORD_BITS * K + DIGEST_BITS * j
        l0 = len(gs)
        gs.extend(("xor", (w, c, one)) for w, c in zip(be_wires(row, J_AT, 4), const_wires(zero, one, 32, j)))
        gs.extend(("xor", (row + 8 * DIGEST_AT + t, exp + t, one)) for t in range(DIGEST_BITS))
        lits = list(range(l0, len(gs)))
        ps, cs = chain(gs, one, be_wires(row, RECEIPT_AT, 8), be_wires(row, SEND_AT, 8), one)
        d = _sums(gs, ps, cs)
        _, cs2 = chain(gs, one, limit, d, one)
        gs.append(("and", (*lits, cs[-1], cs2[-1])))
        hits.append(len(gs) - 1)
    w = counter_width(K, M)
    cnt = [zero] * w
    for h in hits:
        gs.append(("not", (h,)))
        ps, cs = chain(gs, one, cnt, [one] * w, len(gs) - 1)
        cnt = _sums(gs, ps, cs)
    _, cs = chain(gs, one, const_wires(zero, one, w, M), cnt, one)
    return gs, [cs[-1]]


def round_verdict(LIMIT: int, j: int, record: bytes, expected: bytes) -> str:
    """Round j's verdict as `PousVerdict.roundVerdict` states it: ``hit``, ``wrong``, ``late`` or ``reject``."""
    rj = int.from_bytes(record[J_AT:J_AT + 4], "big")
    send = int.from_bytes(record[SEND_AT:SEND_AT + 8], "big")
    receipt = int.from_bytes(record[RECEIPT_AT:RECEIPT_AT + 8], "big")
    if rj != j or receipt < send:
        return "reject"
    if receipt - send > LIMIT:
        return "late"
    return "hit" if record[DIGEST_AT:DIGEST_AT + DIGEST_BITS // 8] == expected else "wrong"


def round_hit(LIMIT: int, j: int, record: bytes, expected: bytes) -> bool:
    return round_verdict(LIMIT, j, record, expected) == "hit"


def verdict_value(K: int, M: int, LIMIT: int, records: Sequence[bytes], expected: Sequence[bytes]) -> int:
    """The reference: 1 iff at most M of the K rounds miss."""
    _statics(K, M, LIMIT)
    if len(records) != K or len(expected) != K:
        raise ValueError(f"a session of {K} rounds has {K} records and {K} expected digests")
    return int(sum(not round_hit(LIMIT, j, r, e) for j, (r, e) in enumerate(zip(records, expected))) <= M)


def _sig(S):
    return (("records", Array(S.K, Array(RECORD_BITS, BIT))), ("expected", Array(S.K, Array(DIGEST_BITS, BIT)))), BIT


def _shape(g: Gate) -> tuple:
    return (g[0],) if g[0] == "not" else (g[0], len(g[1]))


def _prim(shape: tuple):
    if shape[0] == "not":
        return BL.NOT
    return BL.and_gate(shape[1]) if shape[0] == "and" else BL.xor_gate(shape[1])


def _coll(t, refs: list) -> Coll:
    return Coll(t, TR._refs(list(refs)))


def _body(B, S, records, expected):
    gs, outs = gates(S.K, S.M, S.LIMIT)
    ref = [records.refs[k] for k in range(records.refs.count)] + [expected.refs[k] for k in range(expected.refs.count)]
    i = len(ref)
    while i < len(gs):
        g = gs[i]
        if g[0] == "const":
            ref.append(B.call(BL.constant(g[1])).refs[0])
            i += 1
            continue
        shape, j = _shape(g), i + 1
        while j < len(gs) and gs[j][0] != "const" and _shape(gs[j]) == shape and max(gs[j][1]) < i:
            j += 1
        ops = [gs[k][1] for k in range(i, j)]
        if j == i + 1:
            arity = len(ops[0])
            out = B.call(_prim(shape), _coll(BIT if shape[0] == "not" else Array(arity, BIT), [ref[o] for o in ops[0]]))
        elif shape[0] == "not":
            out = B.batch(BL.NOT, _coll(Array(j - i, BIT), [ref[o[0]] for o in ops]), axes=(0,))
        else:
            # operand-major (`Array<n, Array<m, Value<1>>>` over axis 1), as `trace.emit` writes a batch
            arity = len(ops[0])
            out = B.batch(_prim(shape), _coll(Array(arity, Array(j - i, BIT)), [ref[o[a]] for a in range(arity) for o in ops]),
                          axes=(1,))
        ref.extend(out.refs[m] for m in range(j - i))
        i = j
    return _coll(BIT, [ref[o] for o in outs])


def _validated_sig(S):
    _statics(S.K, S.M, S.LIMIT)
    return _sig(S)


PousVerdict = CompositeDefinition(
    "PousVerdict", 1, ("K", "M", "LIMIT"), _validated_sig, _body, conformance="exact-model-tested",
    doc="1 iff at most M of the K rounds miss: a round hits when its record names it, its receipt is not before its send, "
        "receipt − send ≤ LIMIT and its H(block) is the expected digest (module docstring)")
