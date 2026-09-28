"""`PartLog`: a cut is `concat` of the same parts, and the consumers that reuse a log's prefix (runs, the descriptor encoder,
liveness) give what they give on the same sequence built part by part."""
from __future__ import annotations

import json
import random

from verity.ml.circuits.codec import decode_program, encode_program
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, Registry
from verity.ml.circuits.liveness import dead_gates
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, PartLog, Strided, broadcast, concat, prefix_runs, runs
from verity.ml.circuits.types import Array, Value

V16 = Value(16)


def _flat(r):
    if isinstance(r, Concat):
        return [x for p in r.parts for x in _flat(p)]
    return [r]


def _same(a, b) -> None:
    assert type(a) is type(b), (a, b)
    assert a.count == b.count
    assert a.to_json() == b.to_json()
    if isinstance(a, Concat):
        assert a.starts == b.starts
        assert [type(p) for p in a.parts] == [type(p) for p in b.parts]
        assert [p.to_json() for p in a.parts] == [p.to_json() for p in b.parts]


def _part(rng: random.Random, prev):
    k = rng.randrange(10)
    if k == 0 and prev is not None and isinstance(prev, Affine) and prev.count:
        return Affine(prev.space, prev.idx, prev.base + prev.count, rng.randrange(1, 4))  # continues the run
    if k == 1:
        return Affine("n", rng.randrange(2), rng.randrange(4), 1)  # a single leaf: a broadcast when repeated
    if k == 2:
        return broadcast("n", rng.randrange(2), rng.randrange(4), rng.randrange(1, 4))
    if k == 3:
        return Affine("n", rng.randrange(3), rng.randrange(8), 0)
    if k == 4:
        return Explicit(tuple(("p", 0, rng.randrange(6)) for _ in range(rng.randrange(1, 4))))
    if k == 5:
        return Strided("n", rng.randrange(3), rng.randrange(4), 2, 5, rng.choice([1, 2]), 4)
    if k == 6:
        return Concat([Affine("p", 1, rng.randrange(6), 2), Affine("n", 0, 7, 1)])
    return Affine(rng.choice("np"), rng.randrange(3), rng.randrange(20), rng.randrange(1, 70))


def test_a_cut_is_concat_of_its_parts_and_shares_its_runs():
    rng = random.Random(7)
    for trial in range(400):
        parts: list = []
        for _ in range(rng.randrange(1, 14)):
            parts.append(_part(rng, parts[-1] if parts else None))
        if trial % 3 == 0:  # a clean log: rows of distinct produced values
            parts = [Affine("n", i, 0, 3) for i in range(len(parts))]
        log = PartLog(parts)
        shared: dict[int, list] = {}
        for j in range(len(parts) + 1):
            extras = [[], [_part(rng, parts[j - 1] if j else None)], [_part(rng, None), _part(rng, None)]]
            for extra in extras:
                got = log.cut(j, extra)
                want = concat(parts[:j] + extra)
                _same(got, want)
                assert runs(got) == runs(want)
                rs, lg, s = prefix_runs(got)
                assert rs == runs(want)
                if lg is not None:
                    assert lg is log and got.prefix == (log, j)
                    shared[j] = rs[:s]
        for j, pre in shared.items():  # the shared runs of a cut are the first runs of every longer cut
            for k in range(j, len(parts) + 1):
                assert runs(concat(parts[:k]))[:len(pre)] == pre


def test_rows_of_distinct_values_are_cut_without_copying_part_by_part():
    rows = [Affine("n", i, 0, 4) for i in range(6)]
    log = PartLog(rows)
    c = log.cut(5)
    assert isinstance(c, Concat) and c.prefix == (log, 5) and c.parts == rows[:5] and c.starts == [0, 4, 8, 12, 16]
    assert log.cut(1) is rows[0]
    merged = PartLog([Affine("n", 0, 4 * i, 4) for i in range(3)]).cut(3)  # one contiguous run: one Affine, as `concat` gives
    _same(merged, Affine("n", 0, 0, 12))


Neg16 = PrimitiveDefinition("TPLNeg16", 1, (("a", V16),), V16, lambda a: -a & 0xFFFF, register=False)
Add16 = PrimitiveDefinition("TPLAdd16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a + b) & 0xFFFF, register=False)


def _comp(name, params, ret, body):
    return CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False)


def _pair_body(B, S, a):
    return Coll(Array(2, V16), concat([a.refs, B.call(Neg16, a).refs]))


def _sum_body(B, S, x):
    acc = x[0]
    for i in range(1, x.type.n):
        acc = B.call(Add16, acc, x[i])
    return B.call(Neg16, acc)


def _attention_like(N: int, use_log: bool, dead_row: int | None = None) -> Program:
    """Row t of the history is one call's output; call t reads rows 0..t (a causal attention's key sequence)."""
    pair = _comp("TPLPair", [("a", V16)], Array(2, V16), _pair_body)
    sums = [_comp(f"TPLSum{P}", [("x", Array(2 * P, V16))], V16, _sum_body) for P in range(1, N + 1)]

    def body(B, S, x):
        rows = [B.call(pair, x[t]) for t in range(N)]
        log = PartLog([r.refs for r in rows]) if use_log else None
        outs = []
        for P in range(1, N + 1):
            refs = log.cut(P) if use_log else concat([r.refs for r in rows[:P]])
            outs.append(B.call(sums[P - 1], Coll(Array(2 * P, V16), refs)))
        if dead_row is not None:
            outs[dead_row] = outs[0]
        return Coll(Array(N, V16), concat([o.refs for o in outs]))

    return Program(_comp(f"TPLCausal{N}", [("x", Array(N, V16))], Array(N, V16), body))


def test_encoding_and_liveness_of_log_cuts_are_those_of_concat():
    for N in (1, 2, 3, 12, 40):
        for dead in (None, N - 1):
            a, b = _attention_like(N, True, dead), _attention_like(N, False, dead)
            assert encode_program(a) == encode_program(b)
            assert dead_gates(a).to_json() == dead_gates(b).to_json()


def test_a_decoded_alias_chain_is_one_log_and_decodes_to_the_flat_parts():
    """Call P's key sequence is written as an alias to call P-1's plus one part; the decoder extends one log along the chain,
    and every decoded sequence is `Concat` of the flat parts the writer had."""
    reg = Registry()
    reg.add(Neg16)
    reg.add(Add16)
    N = 30
    p = _attention_like(N, True)
    desc = json.loads(json.dumps(encode_program(p)))
    d = decode_program(desc, reg)
    assert encode_program(d) == desc
    logs = set()
    for want, got in zip(p.root.body.nodes, d.root.body.nodes):
        for a, b in zip(want.args, got.args):
            if isinstance(a.refs, Concat):
                flat = _flat(a.refs)
                assert isinstance(b.refs, Concat) and b.refs.starts == Concat(flat).starts and b.refs.count == a.refs.count
                assert [x.to_json() for x in b.refs.parts] == [x.to_json() for x in flat]
                if b.refs.prefix is not None:
                    logs.add(id(b.refs.prefix[0]))
    assert len(logs) == 1
