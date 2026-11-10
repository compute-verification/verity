"""The columnar `ProgramView` reads exactly what the materialized reader read.

`from_instances` streams the rows of `instances.json.gz` into flat int columns and hands out `CallView`s / `ValueRef`s lazily; the
oracle below is the reader it replaced (`json.load` of the document, a `CallView` per row with a tuple of `ValueRef`s), kept here
verbatim so every Value, its order within a Call, the dedup of a producer's two runs (an attention row's K and V of one token), the
multi-port overlap resolution and the record-field resolution are checked Call for Call -- on a document, on a streamed file (small
chunks: buffer refills mid-row), on a file whose rows are not in body order (the fallback), and through `boundary()`."""
from __future__ import annotations

import gzip
import json
import random

import pytest
from verity_vllm.query.units import PartitionError, ValueRef, VerificationUnit, boundary, partition_by
from verity_vllm.query import module_body as mb
from verity_vllm.query import program_view as pv
from verity_vllm.query.module_body import boundary_without_literals, literal_calls, without_literals
from verity_vllm.query.program_view import CallView, ProgramView, from_instances, param_reads, reads_any_of

EPS = '{"f64":"0x1p-17"}'
PARAMS = [["prompt", "Array<4,Value<32>>"],
          ["weights", "{m.a.weight: Array<4,Array<8,Value<16>>>, m.b.weight: Array<8,Array<4,Value<16>>>, m.n.weight: Array<4,Value<16>>}"]]
# weights leaves: m.a.weight [0, 32), m.b.weight [32, 64), m.n.weight [64, 68)
ROWS = [
    {"i": 0, "spec": "Embedding_v1{V=8,H=4}", "form": "call", "name": "m.embed/embedding[0]", "args": [[["p", 0, 0, 1]], [["p", 1, 0, 32]]]},
    {"i": 1, "spec": "Gemm_v1{K=4,N=8}", "form": "call", "name": "m.a/linear[0]", "args": [[["n", 0, 0, 4]], [["p", 1, 0, 32]]]},
    # a fused norm: TWO ports ("0" = normed, "1" = residual); reads Call 1 and the norm weight (a range inside the record)
    {"i": 2, "spec": f"RMSNormFusedCuda_v2{{N=4,EPS={EPS}}}", "form": "call", "name": "m.n/fused[0]", "args": [[["n", 1, 0, 4]], [["n", 0, 0, 4]], [["p", 1, 64, 68]]]},
    # reads port "1" of the fused norm only (leaves [4, 8)), then port "0"; and Call 1's K and V halves as two runs -> ONE Value
    {"i": 3, "spec": "Attention_v3{T=1,NH=2,KVH=1,D=4,BN=16}", "form": "call", "name": "m.attn/attn[t=0]", "args": [[["n", 2, 4, 8]], [["n", 1, 0, 4], ["n", 1, 4, 8]], [["n", 2, 0, 4]]]},
    # a run spanning both ports of the norm (leaves [2, 6)) -> both Values, in port order
    {"i": 4, "spec": "Gemm_v1{K=8,N=4}", "form": "call", "name": "m.b/linear[0]", "args": [[["n", 2, 2, 6]], [["p", 1, 32, 64]]]},
    {"i": 5, "spec": "SiluMul_v1{I=4}", "form": "call", "name": "m.act/silu[0]", "args": [[["n", 4, 0, 4]]]},
    {"i": 6, "spec": "TokenSelect_v1{V=4}", "form": "call", "name": "argmax[0]", "args": [[["n", 5, 0, 4]]]},
    {"i": 7, "spec": "Embedding_v1{V=8,H=4}", "form": "call", "name": "m.embed/embedding[1]", "args": [[["n", 6, 0, 1]], [["p", 1, 0, 32]]]},
]


def doc(rows=ROWS) -> dict:
    return {"schema": "cb-a/instance-sequence/v1", "digest": "d" * 64, "params": PARAMS, "rows": rows}


# --- the reader this replaced (verbatim), as the oracle ----------------------------------------------------------------------------

def _resolve_reads_ref(args, port_of, params, forms, out):
    seen = set()
    for operand in args:
        for run in operand:
            space, idx, lo, hi = run[0], int(run[1]), int(run[2]), int(run[3])
            if space == "n":
                table = port_of[idx]
                if len(table) == 1:
                    vs = (ValueRef(idx, table[0].name),)
                else:
                    vs = tuple(ValueRef(idx, p.name) for p in table if p.lo < hi and lo < p.hi)
                    if not vs and forms[idx] == "batch":
                        vs = (ValueRef(idx, table[0].name),)
            elif space == "p":
                p_ = params[idx]
                table = p_.port_table
                if len(table) == 1:
                    vs = (ValueRef(p_.id, table[0].name),)
                else:
                    vs = tuple(ValueRef(p_.id, p.name) for p in table if p.lo < hi and lo < p.hi)
            else:
                continue
            for v in vs:
                if v not in seen:
                    seen.add(v)
                    out.append(v)


def reference_from_instances(d: dict) -> ProgramView:
    params = pv._params_view(d.get("params") or [])
    rows = d["rows"]
    port_of = [()] * len(rows)
    forms = [""] * len(rows)
    for r in rows:
        i = int(r["i"])
        port_of[i] = pv.definition_ports(r["spec"])
        forms[i] = str(r.get("form") or "call")
    calls = []
    for r in rows:
        i = int(r["i"])
        reads: list[ValueRef] = []
        _resolve_reads_ref(r.get("args") or (), port_of, params, forms, reads)
        spec = r["spec"]
        calls.append(CallView(i, spec, pv.spec_base(spec), forms[i], str(r.get("name") or ""), port_of[i], tuple(reads)))
    calls.sort(key=lambda c: c.id)
    return ProgramView(tuple(calls), params, (), d.get("digest"), "instances")


def assert_same_view(P: ProgramView, R: ProgramView) -> None:
    assert len(P.calls) == len(R.calls) and P.params == R.params and P.outputs == R.outputs and P.digest == R.digest and P.source == R.source
    for c, r in zip(P.calls, R.calls):
        assert (c.id, c.definition, c.family, c.form, c.name, c.port_table) == (r.id, r.definition, r.family, r.form, r.name, r.port_table)
        assert tuple(c.reads) == r.reads, c.id
        assert c.reads == r.reads and c == r                # the lazy sequence compares equal to the tuple; the CallViews compare equal
        assert tuple(param_reads(c)) == tuple(v for v in r.reads if v.call < 0)


def test_columnar_view_equals_the_materialized_reader_on_a_document():
    P, R = from_instances(doc()), reference_from_instances(doc())
    assert P.columns is not None and R.columns is None
    assert_same_view(P, R)
    # the cases the rows were written for
    assert tuple(P.call(3).reads) == (ValueRef(2, "1"), ValueRef(1, "out"), ValueRef(2, "0"))       # K and V runs of Call 1 -> one Value
    assert tuple(P.call(4).reads) == (ValueRef(2, "0"), ValueRef(2, "1"), ValueRef(-2, "m.b.weight"))
    assert tuple(P.call(2).reads) == (ValueRef(1, "out"), ValueRef(0, "out"), ValueRef(-2, "m.n.weight"))
    assert tuple(P.call(0).reads) == (ValueRef(-1, "out"), ValueRef(-2, "m.a.weight"))
    assert P.columns.nbytes() < 1000


def test_streamed_file_with_small_chunks_equals_the_document(tmp_path, monkeypatch):
    p = tmp_path / "instances.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump(doc(), f)
    monkeypatch.setattr(pv, "_CHUNK", 64)                   # every row straddles chunk boundaries
    P = from_instances(str(p))
    assert_same_view(P, reference_from_instances(doc()))
    assert [r["i"] for r in pv.iter_instance_rows(str(p))] == list(range(len(ROWS)))
    monkeypatch.setattr(pv, "_CHUNK", 1 << 22)
    assert_same_view(from_instances(str(p)), reference_from_instances(doc()))


def test_rows_out_of_body_order_are_read_as_a_document(tmp_path):
    rows = [ROWS[3], ROWS[0], ROWS[7], ROWS[1], ROWS[2], ROWS[5], ROWS[4], ROWS[6]]
    P = from_instances(doc(rows))                             # a document: sorted
    assert_same_view(P, reference_from_instances(doc(rows)))
    p = tmp_path / "instances.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump(doc(rows), f)
    assert_same_view(from_instances(str(p)), reference_from_instances(doc(rows)))     # a file: the fallback
    bad = [dict(r, i=r["i"] + 1) for r in ROWS]              # ids are not body positions at all
    with pytest.raises(ValueError):
        from_instances(doc(bad))


def test_lazy_sequences_behave_as_tuples():
    P = from_instances(doc())
    calls = P.calls
    assert len(calls) == 8 and calls[-1].id == 7 and calls[2:4] == (calls[2], calls[3]) and isinstance(calls[2:4], tuple)
    assert calls[1] in calls and list(reversed(calls))[0].id == 7 and calls.index(calls[5]) == 5
    with pytest.raises(IndexError):
        calls[8]
    assert (calls[:2] + (calls[7],)) == (calls[0], calls[1], calls[7])
    r = P.call(3).reads
    assert len(r) == 3 and r[0] == ValueRef(2, "1") and r[-1] == ValueRef(2, "0") and r[1:] == (ValueRef(1, "out"), ValueRef(2, "0"))
    assert ValueRef(1, "out") in r and hash(r) == hash(tuple(r))
    with pytest.raises(IndexError):
        r[3]
    assert P.call(7).id == 7 and P.calls[7].name == "m.embed/embedding[1]"
    assert reads_any_of(P.call(7), {6}) and not reads_any_of(P.call(7), {5}) and not reads_any_of(P.call(0), {6})
    assert tuple(param_reads(P.call(3))) == () and tuple(param_reads(P.call(7))) == (ValueRef(-2, "m.a.weight"),)


def test_boundary_over_the_columnar_view_equals_the_materialized_one():
    P, R = from_instances(doc()), reference_from_instances(doc())
    Q = partition_by(lambda c: c.name.split("/")[0] if "/" in c.name else "")
    a, b = boundary(P, Q), boundary(R, Q)
    assert a.values == b.values and a.produced_in == b.produced_in and a.read_by == b.read_by
    assert [(u.id, u.calls) for u in a.units] == [(u.id, u.calls) for u in b.units]
    assert a.values == {ValueRef(-1, "out"), ValueRef(-2, "m.a.weight"), ValueRef(-2, "m.b.weight"), ValueRef(-2, "m.n.weight"),
                        ValueRef(0, "out"), ValueRef(1, "out"), ValueRef(2, "0"), ValueRef(2, "1"), ValueRef(4, "out"), ValueRef(5, "out"), ValueRef(6, "out")}


def attention_like_rows(n: int, seed: int) -> list[dict]:
    """ROWS, then `n` rows that each read a random set of earlier Calls (either port of the fused norm, Call 2), a weight, or
    nothing (a literal), under five module prefixes."""
    rnd = random.Random(seed)
    rows = list(ROWS)
    for i in range(len(ROWS), len(ROWS) + n):
        if rnd.random() < 0.25:
            args = []
        else:
            args = [[["n", j, 0, 4]] if j != 2 else [["n", 2, *rnd.choice([(0, 4), (4, 8), (2, 6)])]] for j in rnd.sample(range(i), min(i, rnd.randint(1, 12)))]
            if rnd.random() < 0.3:
                args.append([["p", rnd.choice([0, 1]), 0, 1]])
        rows.append({"i": i, "spec": "SiluMul_v1{I=4}", "form": "call", "name": f"m.l{rnd.randrange(5)}/x[{i}]", "args": args})
    return rows


QUERIES = {"word": partition_by(lambda c: c.id), "module": partition_by(lambda c: c.name.split("/")[0]), "one": partition_by(lambda c: ""),
           "parity": partition_by(lambda c: str(c.id % 2), order=lambda k: -int(k))}


@pytest.mark.parametrize("step", [1, 2, 7, 1 << 24])
@pytest.mark.parametrize("q", sorted(QUERIES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_columnar_boundary_without_literals_is_boundarys(monkeypatch, step, q, seed):
    """Values, `produced_in` in insertion order, `read_by`, units and owner, whatever the numpy step (a Value first read in one step and
    again in a later one keeps its first read), on Programs with outputs among the crossing Values, beside them and made by a literal."""
    monkeypatch.setattr(mb, "_READS_PER_STEP", step)
    P0 = from_instances(doc(attention_like_rows(60, seed)))
    lit = literal_calls(P0)
    outs = (ValueRef(len(P0.calls) - 1, "out"), ValueRef(2, "1"), ValueRef(-1, "out"), *(ValueRef(k, "out") for k in sorted(lit)[:1]))
    for P in (P0, ProgramView(P0.calls, P0.params, outs, P0.digest, P0.source)):
        for literals in (lit, frozenset()):
            a, b = boundary_without_literals(P, QUERIES[q], literals), without_literals(boundary(P, QUERIES[q]), literals)
            assert a.read_by._d is None
            assert list(a.produced_in.items()) == list(b.produced_in.items()) and a.values == b.values
            assert [(u.id, u.calls) for u in a.units] == [(u.id, u.calls) for u in b.units] and a.owner == b.owner
            assert dict(a.read_by) == b.read_by and a.read_by == b.read_by
    assert lit and any(v.call in lit for v in boundary(P0, QUERIES["word"]).values), "the rows give the literal filter something to drop"


def test_a_body_that_is_not_columnar_or_units_that_are_not_a_partition_go_to_boundary():
    P, R = from_instances(doc()), reference_from_instances(doc())
    Q = QUERIES["module"]
    a, b = boundary_without_literals(R, Q, frozenset({6})), without_literals(boundary(R, Q), frozenset({6}))
    assert list(a.produced_in.items()) == list(b.produced_in.items()) and a.read_by == b.read_by and isinstance(a.read_by, dict)
    for units in ((VerificationUnit("a", frozenset(range(8))), VerificationUnit("b", frozenset({3}))),      # Call 3 in two units
                  (VerificationUnit("a", frozenset(range(7))),),                                            # Call 7 in none
                  (VerificationUnit("a", frozenset(range(1, 9))),)):                                        # 8 is not a Call
        with pytest.raises(PartitionError):
            boundary_without_literals(P, lambda _P, units=units: units, frozenset())


def test_a_wide_record_resolves_by_bisection_to_the_linear_scans_ports():
    """A weights struct of hundreds of fields: each parameter run's ports, found by bisection, are the linear scan's, in table order;
    a table whose ports overlap keeps the linear scan."""
    rng = random.Random(5)
    edges = sorted(rng.sample(range(1, 50_000), 299))
    table = tuple(pv.Port(f"f{k}", lo, hi, hi - lo, 16) for k, (lo, hi) in enumerate(zip([0] + edges, edges + [50_000])))
    runs = [("p", 0, a, a + rng.choice((1, 7, 400, 9_000))) for a in (rng.randrange(0, 49_000) for _ in range(500))]
    for t in (table, table[:3] + (pv.Port("over", 0, 50_000, 50_000, 16),)):
        b = pv._Builder((pv.InputView(-1, "weights", "x", t),))
        assert (b._param_lo[0] is None) == (t is not table)
        b.add(0, "Gemm_v1{K=4,N=8}", "call", "x", pv.definition_ports("Gemm_v1{K=4,N=8}"), [runs])
        got = [b.c.port_names[p] for p in b.c.read_port]
        want, seen = [], set()
        for _s, _i, lo, hi in runs:
            for p in t:
                if p.lo < hi and lo < p.hi and p.name not in seen:
                    seen.add(p.name)
                    want.append(p.name)
        assert got == want


def test_stored_progressions_resolve_to_the_columns_of_their_runs():
    """v2 rows (node and parameter progressions, `compact.file_items_from_runs`) build the same columns as their per-leaf v1 runs:
    every Value in the same order, for strided weight reads of a wide record, gathers over producer Calls and overlapping runs."""
    from verity_vllm.program import compact as C
    rng = random.Random(9)
    edges = sorted(rng.sample(range(1, 40_000), 199))
    wide = tuple(pv.Port(f"f{k}", lo, hi, hi - lo, 16) for k, (lo, hi) in enumerate(zip([0] + edges, edges + [40_000])))
    ins = (pv.InputView(-1, "weights", "x", wide), pv.InputView(-2, "prompt", "y", (pv.Port("0", 0, 4, 4, 32),)))
    spec = f"RMSNormFusedCuda_v2{{N=4,EPS={EPS}}}"
    rows = []
    for i in range(40):
        ops = []
        for _ in range(3):
            kind = rng.randrange(4)
            if kind == 0:                        # a strided weight read: one parameter progression
                w, st, b = rng.choice((1, 3, 64)), rng.choice((64, 200, 7_000)), rng.randrange(0, 1_000)
                runs = [("p", 0, b + k * st, b + k * st + w) for k in range(rng.randrange(2, 30)) if b + k * st + w <= 40_000]
            elif kind == 1 and i:                # a gather over earlier Calls
                runs = [("n", j, rng.choice((0, 2, 4)), 8) for j in range(0, i, max(1, i // 5))]
            elif kind == 2:
                runs = [("p", 1, 0, 4)] * 2
            else:
                runs = [("p", 0, a, a + rng.randrange(1, 900)) for a in (rng.randrange(0, 39_000) for _ in range(5))]
            ops.append(runs or [("p", 1, 0, 4)])
        rows.append((i, ops))
    cols = []
    for v2 in (False, True):
        b = pv._Builder(ins)
        for i, ops in rows:
            args = [C.file_items_from_runs(o) if v2 else [list(r) for r in o] for o in ops]
            b.add(i, spec, "call", f"r{i}", pv.definition_ports(spec), args)
        cols.append((list(b.c.read_call), list(b.c.read_port), list(b.c.pread_call), list(b.c.pread_port), list(b.c.read_off)))
    assert any(len(it) == 8 for _i, ops in rows for o in ops for it in C.file_items_from_runs(o))
    assert cols[0] == cols[1]


def test_weight_read_progressions_resolve_to_the_columns_and_reads_of_their_runs():
    """The weight-read shape (each progression inside one field of the weights record, most in the field the one before it read)
    builds the columns its per-leaf runs build and the reads the reader this replaced gives them, beside progressions that cross
    fields, sit in a gap between fields, are empty, have one run or none, step backwards or over parameters, on a single-port and
    an overlapping table."""
    from verity_vllm.program import compact as C
    rng = random.Random(11)
    edges = sorted(rng.sample(range(1, 30_000), 149))
    gapped = []
    for k, (lo, hi) in enumerate(zip([0] + edges, edges + [30_000])):
        hi = max(lo + 1, hi - rng.choice((0, 0, 3)))
        gapped.append(pv.Port(f"f{k}", lo, hi, hi - lo, 16))
    over = (pv.Port("a", 0, 100, 100, 16), pv.Port("b", 50, 150, 100, 16))
    ins = (pv.InputView(-1, "weights", "x", tuple(gapped)), pv.InputView(-2, "prompt", "y", (pv.Port("0", 0, 4, 4, 32),)),
           pv.InputView(-3, "kv", "z", over))

    def inside(f):                       # one progression inside field f
        p = gapped[f]
        w = rng.randint(1, max(1, min(4, p.elems)))
        st = rng.randint(w, w + 6)
        lo = rng.randrange(p.lo, p.hi - w + 1)
        cnt = max(1, min(rng.randint(2, 40), (p.hi - w - lo) // st + 1))
        return ["P", 0, lo, lo + w, 0, st, st, cnt]

    def spill(f):                        # one inside field f but for one leaf of the field after it (or before it)
        p = gapped[f]
        if f + 1 < len(gapped) and (f == 0 or rng.random() < 0.5):
            lo = max(p.lo, p.hi - rng.randint(1, 6))
            return ["P", 0, lo, lo + 1, 0, 1, 1, gapped[f + 1].lo + 1 - lo]
        lo = gapped[f - 1].hi - 1
        return ["P", 0, lo, lo + 1, 0, 1, 1, rng.randint(p.lo - lo + 1, p.hi - lo)]

    def other():
        lo = rng.randrange(0, 29_900)
        return rng.choice([
            ["P", 0, lo, lo + 2, 0, 40, 40, rng.randint(2, 9)],                  # crosses fields (or a gap)
            ["P", 0, lo, lo, 0, 3, 3, rng.randint(1, 5)],                        # empty runs
            ["P", 0, lo, lo + 5, 0, 0, 0, 1], ["P", 0, lo, lo + 5, 0, 7, 7, 0],  # one run; none
            ["P", 0, lo + 50, lo + 52, 0, -9, -9, 4],                            # backwards
            ["p", 0, lo, lo + rng.randint(1, 300)], ["p", 1, 0, 4], ["P", 1, 0, 4, 0, 0, 0, 1],
            ["P", 1, 0, 4, 1, 0, 0, 2],                                          # over parameters 1 and 2
            ["P", 2, rng.randrange(0, 140), 145, 0, 1, 1, 3], ["p", 2, 40, 60],
        ])

    gaps = [(a.hi, b.lo) for a, b in zip(gapped, gapped[1:]) if b.lo - a.hi >= 2]
    assert gaps
    rows = []
    for i in range(60):
        ops = []
        for _ in range(rng.randint(1, 4)):
            items, fields = [], [rng.randrange(len(gapped)) for _ in range(rng.randint(1, 3))]
            for _ in range(rng.randint(1, 60)):
                r = rng.random()
                items.append(inside(rng.choice(fields)) if r < 0.6 else spill(rng.choice(fields)) if r < 0.7 else other() if r < 0.95 else
                             ["P", 0, gaps[0][0], gaps[0][0] + 1, 0, 1, 1, gaps[0][1] - gaps[0][0] - 1])   # inside a gap
            ops.append(items)
        rows.append((i, ops))
    assert sum(1 for _i, ops in rows for o in ops for a, b in zip(o, o[1:]) if a[0] == b[0] == "P" and a[1] == b[1] == 0
               and any(p.lo <= a[2] and a[3] + (a[7] - 1) * a[5] <= p.hi and p.lo <= b[2] and b[3] + (b[7] - 1) * b[5] <= p.hi
                       for p in gapped)) > 300
    spec = f"RMSNormFusedCuda_v2{{N=4,EPS={EPS}}}"
    cols = []
    for v2 in (True, False):
        b = pv._Builder(ins)
        for i, ops in rows:
            b.add(i, spec, "call", f"r{i}", pv.definition_ports(spec), [o if v2 else [list(t) for t in C.runs_from_file_items(o)] for o in ops])
        cols.append((list(b.c.read_call), list(b.c.read_port), list(b.c.pread_call), list(b.c.pread_port), list(b.c.read_off)))
        if v2:
            P = b.view((), None, "instances")
    assert cols[0] == cols[1]
    for i, ops in rows:
        want: list[ValueRef] = []
        _resolve_reads_ref([C.runs_from_file_items(o) for o in ops], [], ins, [], want)
        assert tuple(P.call(i).reads) == tuple(want), i


@pytest.mark.parametrize("cap", [0, 40, 1 << 25])
@pytest.mark.parametrize("least", [1, 3, 8])
def test_a_repeated_parameter_operand_resolves_once_to_the_columns_and_reads_of_its_runs(monkeypatch, cap, least):
    """The memo of a document's parameter operands builds the columns the Builder without it builds and the reads the reader this
    replaced gives, for operands that are one list again, a copy of it, equal runs that are other objects, more operands sharing a
    first run than the memo keeps, parameter runs then a Call's, and a Call reading a weight port before the repeat; under any cap."""
    from verity_vllm.program import compact as C
    monkeypatch.setattr(pv, "RESOLVE_MEMO_RUNS", cap)
    monkeypatch.setattr(pv, "RESOLVE_MEMO_MIN", least)
    rng = random.Random(13)
    edges = sorted(rng.sample(range(1, 20_000), 99))
    table = tuple(pv.Port(f"f{k}", lo, hi, hi - lo, 16) for k, (lo, hi) in enumerate(zip([0] + edges, edges + [20_000])))
    ins = (pv.InputView(-1, "weights", "x", table), pv.InputView(-2, "prompt", "y", (pv.Port("0", 0, 4, 4, 32),)))

    def weight(k: int) -> list[list]:
        out = []
        for _ in range(k):
            p, r = table[rng.randrange(len(table))], rng.random()
            lo = rng.randrange(p.lo, p.hi)
            out.append(["P", 0, lo, lo + 1, 0, 1, 1, rng.randint(1, p.hi - lo)] if r < 0.7 else
                       ["P", 0, lo, lo + 2, 0, 40, 40, rng.randint(1, 9)] if r < 0.8 else
                       ["p", 0, lo, min(20_000, lo + rng.randint(1, 900))] if r < 0.9 else ["p", 1, 0, 4])
        return out

    pool = [weight(rng.randint(8 if k == 0 else 1, 14)) for k in range(6)]
    first = pool[0][0]
    shared = [[first, *weight(rng.randint(8, 12))] for _ in range(pv._MEMO_WAYS + 2)]
    spec = f"RMSNormFusedCuda_v2{{N=4,EPS={EPS}}}"
    rows = []
    for i in range(80):
        ops = []
        for _ in range(rng.randint(1, 4)):
            o, r = rng.choice(pool), rng.random()
            ops.append(o if r < 0.35 else o.copy() if r < 0.5 else rng.choice(shared) if r < 0.65 else [list(t) for t in o] if r < 0.75 else
                       [*o, ["n", rng.randrange(i), 0, 8]] if r < 0.85 and i else [["n", rng.randrange(i), 2, 6]] if i else weight(3))
        rows.append((i, ops))
    cols, memos = [], []
    for memo in (False, True):
        b = pv._Builder(ins, memo)
        for i, ops in rows:
            b.add(i, spec, "call", f"r{i}", pv.definition_ports(spec), ops)
        cols.append((list(b.c.read_call), list(b.c.read_port), list(b.c.pread_call), list(b.c.pread_port), list(b.c.read_off)))
        memos.append(b._memo)
        if memo:
            P = b.view((), None, "instances")
    assert cols[0] == cols[1]
    assert memos[0] is None and b._memo_runs <= cap and all(len(w) <= pv._MEMO_WAYS for w in memos[1].values())
    if cap > 40:
        assert len(memos[1][id(first)]) == pv._MEMO_WAYS and b._memo_runs < sum(len(o) for _i, ops in rows for o in ops if len(o) >= least)
    port_of = [pv.definition_ports(spec)] * len(rows)
    for i, ops in rows:
        want: list[ValueRef] = []
        _resolve_reads_ref([C.runs_from_file_items(o) for o in ops], port_of, ins, ["call"] * len(rows), want)
        assert tuple(P.call(i).reads) == tuple(want), i
