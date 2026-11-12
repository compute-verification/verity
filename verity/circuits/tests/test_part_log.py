"""`PartLog`: a cut is `concat` of the same parts, and the consumers that reuse a log's prefix (runs, the descriptor encoder,
liveness) give what they give on the same sequence built part by part. The descriptor writer interns part lists, and writes
the bytes the writer before it did, which kept every argument's whole tuple of part keys (`_KeyTupleWriter`)."""
from __future__ import annotations

import json
import random

import pytest
import test_codec_stream as CS

from verity.circuits import bind, codec
from verity.circuits.codec import (ALIAS_WINDOW, SCHEMA_V0, SCHEMA_V1, _BodyEncoder, _flat_parts, _part_key, canonical_json,
                                      decode_program, descriptor_chunks, encode_program)
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, DefinitionTable
from verity.circuits.liveness import dead_gates
from verity.circuits.program import Program
from verity.circuits.refs import Affine, Coll, Concat, Explicit, PartLog, Strided, broadcast, concat, prefix_runs, runs
from verity.circuits.types import Array, Value
from verity.definitions.registry.dsa_topk_mask import DsaButterflyRows, DsaButterflyRowsPad, butterfly_switches
from verity.definitions.registry.prims import BIT

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


def test_a_cut_is_the_same_sequence_whatever_is_appended_to_its_log_later():
    """A cut's `parts` and `starts` read as the lists `concat` builds (index, slice, iterate, reverse, compare), and parts appended
    to the log afterwards are not in them."""
    rng = random.Random(3)
    rows = [Affine("n", i, 0, 4) for i in range(9)]
    log = PartLog(rows[:6])
    cuts = [(j, extra, log.raw_cut(j, extra)) for j in range(7) for extra in ([], [Affine("p", 0, j, 2)], [Affine("p", 1, 0, 3)] * 2)]
    for r in rows[6:]:
        log.append(r)
    for j, extra, c in cuts:
        parts, starts = rows[:j] + extra, Concat(rows[:j] + extra).starts
        for got, want in ((c.parts, parts), (c.starts, starts)):
            assert len(got) == len(want) and list(got) == want and got == want and want == got and list(reversed(got)) == want[::-1]
            assert got != want + [0] and [got[i] for i in range(-len(want), len(want))] == want + want
            for _ in range(8):
                lo, hi = rng.randrange(-2, len(want) + 2), rng.randrange(-2, len(want) + 2)
                assert got[lo:hi] == want[lo:hi] and got[lo:] == want[lo:] and got[::2] == want[::2]
        assert c.count == sum(p.count for p in parts) and [c[i] for i in range(c.count)] == [x for p in parts for x in p]
        assert c.slice(3, c.count - 2).to_json() == Concat(parts).slice(3, c.count - 2).to_json()


def test_slicing_and_indexing_a_cut_give_what_the_same_parts_give():
    """`slice` and `[j]` search a cut's shared prefix in C rather than item by item; every slice and leaf is the plain `Concat`'s,
    at part boundaries, across empty parts and into the cut's own tail, after the log has grown."""
    rng = random.Random(11)
    for _ in range(60):
        parts: list = []
        for _ in range(rng.randrange(1, 12)):
            parts.append(_part(rng, parts[-1] if parts else None))
        log = PartLog(parts)
        cuts = []
        for j in range(len(parts) + 1):
            extra = [_part(rng, None) for _ in range(rng.randrange(3))]
            cuts.append((log.raw_cut(j, extra), Concat(parts[:j] + extra)))
        for _ in range(3):
            log.append(_part(rng, None))
        for c, want in cuts:
            n = want.count
            assert [c[i] for i in range(n)] == [want[i] for i in range(n)]
            edges = sorted({e for s in want.starts for e in (s - 1, s, s + 1)} | {-1, 0, n, n + 1})
            pairs = [(rng.choice(edges), rng.choice(edges)) for _ in range(40)]
            pairs += [(lo, lo + rng.randrange(-1, n + 2)) for lo in (rng.randrange(-1, n + 2) for _ in range(20))]
            for lo, hi in pairs:
                _same(c.slice(lo, hi), want.slice(lo, hi))


def test_a_cut_costs_the_same_memory_at_any_length():
    """A cut shares its log's prefix: cutting at row 2000 allocates what cutting at row 20 does (copied, it was 100 times more)."""
    import tracemalloc
    log = PartLog([Affine("n", i, 0, 4) for i in range(2001)])

    def cost(j):
        tracemalloc.start()
        keep = [log.cut(j, [Affine("p", 0, 0, 1)]) for _ in range(50)]
        size = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        assert len(keep) == 50
        return size

    small, large = cost(20), cost(2000)
    assert large < 1.2 * small, (small, large)


def test_the_writer_holds_one_entry_a_cut_of_a_history():
    """Each cut of a history one row longer than the one before is written as an alias to it plus the new row, and the writer
    keeps one interned part list a cut, never a cut's whole list of part keys (O(rows^2) a history, holding each)."""
    log, enc = PartLog(), _BodyEncoder(SCHEMA_V1, tables=object())
    for t in range(300):
        log.append(Affine("n", t, 0, 4))
        out = enc.refs(log.cut(t + 1), (t, 0))
        if t >= 2:
            assert out == {"concat": [{"alias": [t - 1, 0]}, Affine("n", t, 0, 4).to_json()]}
    assert len(enc.trie) == 300 and len(enc.seqs) == 299


class _KeyTupleWriter(_BodyEncoder):
    """The v1 writer before part lists were interned: an argument's whole tuple of `_part_key`s keys `seqs`. The writer is held
    to its bytes."""

    def refs(self, r, at=None):
        if self.schema == SCHEMA_V0 or not isinstance(r, Concat):
            return r.to_json()
        cut = r.prefix is not None and r.prefix[0].leaf_parts(r.prefix[1])
        parts = list(r.parts) if cut and not any(isinstance(p, Concat) for p in r.parts[r.prefix[1]:]) else _flat_parts(r)
        keys = tuple(_part_key(p) for p in parts)
        n, out = len(keys), None
        for m in range(n, max(1, n - ALIAS_WINDOW - 1), -1):
            hit = self.seqs.get(keys[:m])
            if hit is not None and (at is None or hit[0] < at[0]):
                out = [{"alias": list(hit)}] + [p.to_json() for p in parts[m:]]
                break
        if out is None:
            out = [p.to_json() for p in parts]
        if at is not None and n > 1:
            self.seqs.setdefault(keys, at)
        return out[0] if len(out) == 1 else {"concat": out}


def _arguments(rng: random.Random, logs: list[PartLog], n: int = 400) -> list:
    """`n` (refs, at) of one body's arguments in body order: cuts of the histories ``logs`` at any length, most at their newest
    rows, again and out of order, some with parts past the cut; the same parts as a cut with no log; flat and nested
    sequences; single references; and a few with no `at` (the body's return). The first two histories grow rows of distinct
    values, the same rows for their first 12; the others grow any part, nested ones among them, which a cut can't read from
    its log."""
    out, k = [], 0
    for _ in range(n):
        i, c = rng.randrange(len(logs)), rng.randrange(10)
        log, t = logs[i], len(logs[i])
        if c < 3 or t < 2:
            row = Affine("n", 10 + t + 1000 * i * (t >= 12), 0, 1 + t % 4)
            log.append(row if i < 2 else _part(rng, log.parts[-1] if t else None))
            continue
        j = rng.randrange(2, t + 1) if rng.random() < 0.5 else max(2, t - rng.randrange(2))
        if c < 6:
            r = log.cut(j)
        elif c == 6:
            r = log.cut(j, [_part(rng, None) for _ in range(rng.randrange(1, 4))])
        elif c == 7:
            r = concat(list(log.parts[:j]))
        elif c == 8:
            r = Concat([_part(rng, None) for _ in range(rng.randrange(2, 12))])
        else:
            r = _part(rng, None)
        k += rng.randrange(2)
        out.append((r, None if rng.random() < 0.05 else (k, rng.randrange(3))))
    return out


@pytest.mark.parametrize("seed", range(40))
def test_the_writer_writes_each_argument_as_the_key_tuple_writer_did(seed):
    """Call for call over one body's arguments, the writer's output is the key-tuple writer's: the same alias, to the same
    argument, with the same suffix. On odd seeds a second body's writer, over arguments of its own, takes turns with the
    first over the same histories (each keeps its own ids in a log's memo, so each takes the log back from the other), and
    each writes what its own key-tuple writer does."""
    rng = random.Random(seed)
    logs = [PartLog() for _ in range(3)]
    bodies = [_arguments(rng, logs) for _ in range(1 + seed % 2)]
    new = [_BodyEncoder(SCHEMA_V1, tables=object()) for _ in bodies]
    old = [_KeyTupleWriter(SCHEMA_V1, tables=object()) for _ in bodies]
    aliased = 0
    for i in range(max(map(len, bodies))):
        for w, args in enumerate(bodies):
            if i < len(args):
                r, at = args[i]
                got = new[w].refs(r, at)
                assert got == old[w].refs(r, at), (seed, w, i)
                aliased += isinstance(got, dict) and "alias" in json.dumps(got)
    assert aliased >= 20, aliased


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


def _last_body(B, S, x):
    return B.call(Neg16, x[x.type.n - 1])


def _attention_like(N: int, use_log: bool, dead_row: int | None = None, read=_sum_body) -> Program:
    """Row t of the history is one call's output; call t reads rows 0..t (a causal attention's key sequence) and sums them, or,
    with ``read=_last_body``, negates the last word alone, so a long history's descriptor is its root's references."""
    pair = _comp("TPLPair", [("a", V16)], Array(2, V16), _pair_body)
    sums = [_comp(f"TPL{read.__name__}{P}", [("x", Array(2 * P, V16))], V16, read) for P in range(1, N + 1)]

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
    reg = DefinitionTable()
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


def _long_rows(K: int, W: int, n_max: int, nested: bool) -> Program:
    """Rows K+1..n_max of a causal history read as DSA's long rows read it, each at its class N (n rounded up to a multiple of
    K): `DsaButterflyRows_v1{N}` over the history's first n rows when N = n, else `DsaButterflyRowsPad_v1{T=n,N}` over them and
    N - n pad rows. The history is cut from one log, a row one call's output, or two calls' when ``nested``."""
    row = Array(W, V16)
    classes = [(n, -(-n // K) * K) for n in range(K + 1, n_max + 1)]
    bits = [butterfly_switches(N, K) for _, N in classes]
    neg = _comp(f"TPLNegRow{W}", [("a", row)], row, lambda B, S, a: Coll(row, concat([B.call(Neg16, a[i]).refs for i in range(W)])))

    def body(B, S, sw, x, pad):
        log = PartLog()
        for t in range(n_max):
            r = B.call(neg, x[t]).refs
            log.append(concat([r.slice(0, W - 1), B.call(Neg16, x[t][0]).refs]) if nested else r)
        outs, at = [], 0
        for (n, N), b in zip(classes, bits):
            s, at = Coll(Array(b, BIT), sw.refs.slice(at, at + b)), at + b
            hist = Coll(Array(n, row), log.cut(n))
            if N > n:
                rest = Coll(Array(N - n, row), pad.refs.slice(0, (N - n) * W))
                outs.append(B.call(bind(DsaButterflyRowsPad, T=n, N=N, K=K, W=W, E=16), s, hist, rest))
            else:
                outs.append(B.call(bind(DsaButterflyRows, N=N, K=K, W=W, E=16), s, hist))
        return Coll(Array(len(outs) * K, row), concat([o.refs for o in outs]))

    params = [("sw", Array(sum(bits), BIT)), ("x", Array(n_max, row)), ("pad", Array(K - 1, row))]
    return Program(_comp(f"TPLLongRows{K}x{W}x{n_max}{'n' * nested}", params, Array(len(classes) * K, row), body))


def test_a_descriptor_is_the_key_tuple_writers_byte_for_byte(monkeypatch):
    """A v1 descriptor's bytes, whole or streamed, are those the key-tuple writer wrote, over the circuits that alias most:
    causal histories cut from a log or concatenated, with a dead row; DSA's long rows at their classes, Rows and RowsPad, the
    Definitions' own bodies among them, a history row one part or two; and a decoded circuit, whose histories are the decoder's
    logs."""
    reg = DefinitionTable()
    reg.add(Neg16)
    reg.add(Add16)
    progs = [_attention_like(N, log, dead) for N in (3, 12, 40) for log in (True, False) for dead in (None, N - 1)]
    progs += [_long_rows(2, 3, 12, False), _long_rows(2, 3, 12, True), _long_rows(3, 2, 10, False)]
    progs.append(decode_program(json.loads(json.dumps(encode_program(_attention_like(30, True)))), reg))
    progs += [_attention_like(400, log, read=_last_body) for log in (True, False)]

    def written():
        return [(canonical_json(encode_program(p)), b"".join(descriptor_chunks(p))) for p in progs]

    new = written()
    monkeypatch.setattr(codec, "_BodyEncoder", _KeyTupleWriter)
    assert written() == new
    assert all(b'"alias"' in whole for whole, _ in new)
    assert [whole.count(b'"alias"') for whole, _ in new[-2:]] == [398, 398], "every cut past the second aliases the one before"


recorded = CS.recorded


def test_the_codec_streams_programs_are_the_key_tuple_writers_byte_for_byte(monkeypatch, recorded):
    """`test_codec_stream`'s Programs, its random ones (`_Gen`) and every one the circuits tests it runs build (`MODULES`), are
    written as the key-tuple writer wrote them, whole and streamed at node batches of 1, 3 and the default (an alias reaching
    back across a batch boundary), or refused with its refusal. That suite holds them to the codec before the streamed digest,
    which calls today's writer."""
    CS.run_the_other_tests()
    others = list(recorded)
    progs = [CS._Gen(seed).program() for seed in range(CS.RANDOM)] + others
    batches = (1, 3, codec._NODE_BATCH)

    def written():
        out = []
        for p in progs:
            out.append(CS.outcome(lambda q: canonical_json(encode_program(q)), p))
            for b in batches:
                monkeypatch.setattr(codec, "_NODE_BATCH", b)
                out.append(CS.outcome(lambda q: b"".join(descriptor_chunks(q)), p))
        return out

    new = written()
    monkeypatch.setattr(codec, "_BodyEncoder", _KeyTupleWriter)
    assert written() == new
    whole = new[::len(batches) + 1]
    assert len(others) > 50 and sum(o == "ok" and b'"alias"' in w for o, w in whole[:CS.RANDOM]) > CS.RANDOM // 10
