"""`verity.ml.circuits.census`: census/v1 against the lowered circuit (each value's bits, readers, candidates and output from the
gates' operands, its classes from the activations), served MACs and family rules, steps, and the digest."""
from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace

import pytest

import test_ports as TP
from test_ir_liveness import Add16, Add32, Coord, Embedding, Gemm, Mul16, comp, prim, programs as liveness_programs
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import census as CE, ports
from verity.ml.circuits.census import NoBooleanWork, census, seal, silicon_step, step_ends
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.codec import canonical_json, descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.layout import resolve
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, array_of, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

V8, V16, V32, V64 = Value(8), Value(16), Value(32), Value(64)
A4 = Array(4, V16)

Lookup = prim("TLookup16", [("i", V32), ("xs", Array(16, V16))], V16, lambda i, *xs: xs[i % 16])
Lookup.evaluate_lazy = lambda get: get(1 + get(0) % 16)  # type: ignore[attr-defined]
Argmax = prim("TArgmax", [("a", A4)], V32, lambda *a: max(range(4), key=lambda j: a[j]))
Dup = comp("TDup", [("x", V32)], Tuple(V32, V32), lambda Bd, S, x: tuple_of(x, Bd.call(Add32, x, x)))

Embed = comp("TEmbed", [("tok", V32), ("table", Array(16, A4))], A4,
             lambda Bd, S, tok, table: Bd.batch(Lookup, tok, Coll(Array(4, Array(16, V16)), strided_view(table, 0, 16, 1, 4, 64)),
                                                axes=(None, 0)))


def _layer(Bd, S, h, w):
    k = Bd.batch(Mul16, h, w, axes=(0, 0))
    return tuple_of(k, Bd.batch(Add16, h, k, axes=(0, 0)))


Layer = comp("TLayer", [("h", A4), ("w", A4)], Tuple(A4, A4), _layer)


def _attn(Bd, S, q, kv):
    acc = q
    for t in range(S.T):
        acc = Bd.batch(Add16, acc, kv[t], axes=(0, 0))
    return acc


Attn = CompositeDefinition("TAttn", 1, ("T",), lambda S: ((("q", A4), ("kv", Array(S.T, A4))), A4), _attn, register=False)
Select = comp("TSelect", [("a", A4)], V32, lambda Bd, S, a: Bd.call(Argmax, a))


def _request(Bd, S, toks, table, w):
    tok, kvs, out = toks[0], [], []
    for s in range(S.T):
        kq = Bd.call(Layer, Bd.call(Embed, tok, table), w)
        kvs.append(kq[0])
        tok = Bd.call(Select, Bd.call(bind(Attn, T=s + 1), kq[1], array_of(kvs)))
        out.append(tok)
    return array_of(out)


Request = CompositeDefinition("TRequest", 1, ("T",), lambda S: ((("toks", Array(1, V32)), ("table", Array(16, A4)), ("w", A4)),
                                                                Array(S.T, V32)), _request, register=False)


def request(T: int = 3) -> Program:
    """Steps of an embedding gather (a selection among the registered table), a layer that writes a KV row, attention over every
    row so far and a sampling event (`TSelect_v1`)."""
    return Program(bind(Request, T=T), registered=("table", "w"))


def _passes(Bd, S, x):
    d = Bd.call(Dup, x[0])
    return Bd.call(Add32, d[0], Bd.call(Add32, d[1], x[1]))


Col = comp("TCol", [("x", Array(3, V32))], V32, lambda Bd, S, x: Bd.call(Add32, x[0], x[2]))


def programs():
    yield from liveness_programs()
    yield "request", request()
    yield "pass-through", Program(comp("TPasses", [("x", Array(2, V32))], V32, _passes))
    yield "columns", Program(comp("TColumns", [("M", Array(3, Array(4, V32)))], Array(4, V32),
                                  lambda Bd, S, M: Bd.batch(Col, M, axes=(1,))))


def brute(P: Program) -> dict:
    """Per leaf a root node produces (the first naming of each gate): its width, the Calls whose gates take it as a direct operand,
    those that have it among a selection's candidates, and whether the root returns it."""
    C, nodes, off = P.circuit, P.root.body.nodes, P.root.body.offsets
    first: dict[int, tuple[int, int]] = {}
    for j, nd in enumerate(nodes):
        for t in range(nd.out.leaves):
            first.setdefault(resolve(C.root_scope, ("n", j, t)), (j, t))
    leaves = {leaf: {"bits": C.gate(g).prim.ret.w, "reads": set(), "cands": set(), "output": False} for g, leaf in first.items()}
    for k, nd in enumerate(nodes):
        for g in range(off[k], off[k + 1]):
            gate = C.gate(g)
            lazy = hasattr(gate.prim, "evaluate_lazy")
            pos = 0
            for _, t in gate.prim.params:
                for o in gate.operands[pos:pos + t.leaves]:
                    if not off[k] <= o < off[k + 1]:
                        leaves[first[o]]["cands" if lazy and not isinstance(t, Value) else "reads"].add(k)
                pos += t.leaves
    for g in C.root_outputs():
        leaves[first[g]]["output"] = True
    return leaves


def signatures(c: dict) -> list[dict]:
    """Per value: its readers, the Calls it is a candidate of, and whether it is an output."""
    cands: dict[int, set] = defaultdict(set)
    for k, vs in zip(c["selects"]["call"], c["selects"]["values"]):
        for v in vs:
            cands[v].add(k)
    v = c["values"]
    return [{"reads": set(r), "cands": cands[i], "output": o, "fact": f} for i, (r, o, f) in enumerate(zip(v["reads"], v["output"], v["fact"]))]


def per_leaf(c: dict) -> dict:
    v, sig = c["values"], signatures(c)
    return {(j, t): sig[i] for i, (j, lo, hi) in enumerate(zip(v["producer"], v["lo"], v["hi"])) for t in range(lo, hi)}


@pytest.mark.parametrize("label,P", list(programs()), ids=lambda x: x if isinstance(x, str) else "")
def test_values_are_the_gates_operands(label, P):
    """Every leaf a root node produces is in exactly one value; its readers, candidacy and output are the lowered circuit's, and a
    value's bits are its gates' widths."""
    c = census(P)
    want = brute(P)
    got = per_leaf(c)
    assert set(got) == set(want), label
    for leaf, w in want.items():
        assert {k: got[leaf][k] for k in ("reads", "cands", "output")} == {k: w[k] for k in ("reads", "cands", "output")}, (label, leaf)
    v, sig = c["values"], signatures(c)
    for j, lo, hi, bits in zip(v["producer"], v["lo"], v["hi"], v["bits"]):
        assert bits == sum(want[(j, t)]["bits"] for t in range(lo, hi)), label
    for i in range(1, len(sig)):  # maximal runs: adjacent values of one producer differ
        if v["producer"][i] == v["producer"][i - 1] and v["lo"][i] == v["hi"][i - 1]:
            assert sig[i] != sig[i - 1], label
    assert not c["approximate"]


def _activations(P: Program) -> Counter:
    got: Counter = Counter()
    todo = list(P.circuit.root_calls(primitive_batches=None))
    while todo:
        fc = todo.pop()
        if fc.is_primitive:
            continue
        got[descriptor_id(fc.fn)] += 1
        todo.extend(fc.children())
    return got


@pytest.mark.parametrize("label,P", list(programs()), ids=lambda x: x if isinstance(x, str) else "")
def test_classes_count_the_activations_and_gates(label, P):
    c = census(P)
    C = P.circuit
    prims = Counter(C.gate(g).prim.id for g in range(C.size))
    cl = c["classes"]
    assert {k: e["instances"] for k, e in cl.items() if e["kind"] != "composite"} == dict(prims), label
    assert {k: e["instances"] for k, e in cl.items() if e["kind"] == "composite"} == dict(_activations(P)), label
    for e in cl.values():
        assert e["gates"] == sum(e["primitives"].values())
    assert sum(e["instances"] for e in cl.values() if e["kind"] != "composite") == c["totals"]["gates"] == P.gates
    inputs = c["totals"]["inputs"]
    assert inputs["run"]["gates"] + inputs["registration"]["gates"] == P.input_gates
    assert c["totals"]["computed_gates"] == sum(c["steps"]["gates"]) == P.gates - P.input_gates


def test_the_request_census():
    P = request()
    c = census(P, step_ends(P, {"TSelect_v1"}))
    assert c["inputs"] == 3 and c["steps"]["ends"] == [7, 11, 15]
    assert [c["callees"][i] for i in c["nodes"]["callee"][3:7]] == ["TEmbed_v1", "TLayer_v1", "TAttn_v1{T=1}", "TSelect_v1"]
    assert c["classes"]["TLookup16_v1"] == {"family": "TLookup16_v1", "statics": {}, "kind": "primitive", "binding": None,
                                            "instances": 12, "gates": 1, "primitives": {"TLookup16_v1": 1}, "served_macs": 0,
                                            "tail_atoms": 0, "receipt_folds": 0, "served_macs_by_precision": {}}
    assert c["classes"]["TAttn_v1{T=2}"]["statics"] == {"T": 2} and c["classes"]["TAttn_v1{T=2}"]["gates"] == 8
    assert c["classes"]["RegisteredInput16_v1"]["binding"] == "registration" and c["classes"]["Input32_v1"]["binding"] == "run"
    assert c["totals"]["inputs"] == {"run": {"gates": 1, "bits": 32}, "registration": {"gates": 68, "bits": 1088}}
    assert c["steps"]["select_gates"] == [4, 4, 4] and c["steps"]["select_bits"] == [64, 64, 64]
    # each Embed selects among the whole table (input node 1), which no gate reads directly
    table = [i for i, j in enumerate(c["values"]["producer"]) if j == 1]
    assert c["selects"]["values"] == [table] * 3 and all(c["values"]["reads"][i] == [] for i in table)
    # a KV row (a TLayer's k) is read by its own step's attention and every later one's
    kv = [i for i, (j, lo) in enumerate(zip(c["values"]["producer"], c["values"]["lo"])) if j in (4, 8, 12) and lo == 0]
    assert [c["values"]["reads"][i] for i in kv] == [[5, 9, 13], [9, 13], [13]]


def _adds_up(c: dict) -> bool:
    """Every class's, step's and the totals' served MACs by precision add up to its served MACs."""
    return (all(sum(e["served_macs_by_precision"].values()) == e["served_macs"] for e in c["classes"].values())
            and [sum(m.values()) for m in c["steps"]["served_macs_by_precision"]] == c["steps"]["served_macs"]
            and sum(c["totals"]["served_macs_by_precision"].values()) == c["totals"]["served_macs"])


def test_served_macs_and_family_rules(monkeypatch):
    P = Program(bind(Gemm, K=32, N=6))
    assert census(P)["totals"]["served_macs"] == 0 and census(P)["totals"]["served_macs_by_precision"] == {}
    monkeypatch.setitem(CE.PRODUCTS, "TDot16_v1", 16)
    monkeypatch.setitem(CE.PRECISION, "TDot16_v1", "bf16")
    c = census(P)
    assert c["classes"]['TGemmCoordinate_v1{K=32}']["served_macs"] == 32 and c["totals"]["served_macs"] == 6 * 32
    assert c["steps"]["served_macs"] == [192] and c["steps"]["served_macs_by_precision"] == [{"mac:bf16": 192}] and _adds_up(c)
    monkeypatch.setitem(CE.FAMILY_COUNTS, "TGemmCoordinate_v1", lambda S: {"served_macs": S["K"] - 2, "tail_atoms": 1})
    c = census(P)
    assert {k: c["totals"][k] for k in CE.COUNTERS} == {"served_macs": 180, "tail_atoms": 6, "receipt_folds": 0}
    assert c["classes"]['TGemmCoordinate_v1{K=32}']["tail_atoms"] == 1 and c["steps"]["tail_atoms"] == [6]
    assert c["classes"]['TDot_v1{K=32}']["served_macs"] == 32  # under the rule's class, the bottom-up count stays the class's own
    assert c["totals"]["served_macs_by_precision"] == {"mac:bf16": 180} and _adds_up(c), "a rule's served MACs at its steps' precision"
    monkeypatch.setitem(CE.FAMILY_COUNTS, "TGemmCoordinate_v1", lambda S: {"served": 1})
    with pytest.raises(ValueError, match="non-negative ints"):
        census(P)


def test_a_rule_family_counts_lacks_is_refused(monkeypatch):
    """A family `RULED` names, or a Definition with a rule (`.counts`), is refused while `FAMILY_COUNTS` has no rule for it, never
    counted by `PRODUCTS`."""
    P = Program(bind(Gemm, K=32, N=6))
    monkeypatch.setitem(CE.PRODUCTS, "TDot16_v1", 16)
    monkeypatch.setitem(CE.PRECISION, "TDot16_v1", "bf16")
    monkeypatch.setitem(CE.RULED, "TGemmCoordinate_v1", "toy.module")
    with pytest.raises(ValueError, match=r"^TGemmCoordinate_v1\{K=32\} counts by its family's rule, which census.FAMILY_COUNTS "
                                         r"lacks: import toy.module, which registers it"):
        census(P)
    monkeypatch.delitem(CE.RULED, "TGemmCoordinate_v1")
    monkeypatch.setattr(Coord, "counts", lambda S: {"served_macs": S["K"] - 2}, raising=False)
    with pytest.raises(ValueError, match=r"^TGemmCoordinate_v1 has a census rule \(`.counts`\) that census.FAMILY_COUNTS and "
                                         r"census.RULED don't name"):
        census(P)
    monkeypatch.setitem(CE.FAMILY_COUNTS, "TGemmCoordinate_v1", Coord.counts)
    assert census(P)["totals"]["served_macs"] == 6 * 30


def test_served_macs_split_by_precision(monkeypatch):
    """A class over steps at two precisions carries both; a step with no precision, and a rule's served MACs over steps at two,
    are refused by name."""
    P = Program(bind(Gemm, K=32, N=6))
    monkeypatch.setitem(CE.PRODUCTS, "TDot16_v1", 16)
    monkeypatch.setitem(CE.PRODUCTS, "TCvt16_v1", 1)
    monkeypatch.setitem(CE.PRECISION, "TDot16_v1", "bf16")
    with pytest.raises(ValueError, match="names TCvt16_v1 with no census.PRECISION"):
        census(P)
    monkeypatch.setitem(CE.PRECISION, "TCvt16_v1", "fp8")
    c = census(P)
    assert c["classes"]['TGemmCoordinate_v1{K=32}']["served_macs_by_precision"] == {"mac:bf16": 32, "mac:fp8": 1}
    assert c["totals"]["served_macs_by_precision"] == {"mac:bf16": 192, "mac:fp8": 6} and _adds_up(c)
    monkeypatch.setitem(CE.FAMILY_COUNTS, "TGemmCoordinate_v1", lambda S: {"served_macs": S["K"]})
    with pytest.raises(ValueError, match=r"over steps at \['mac:bf16', 'mac:fp8'\]: a rule's served MACs take the one precision"):
        census(P)


def test_step_ends_and_their_refusals():
    P = request()
    assert step_ends(P, ()) == [len(P.root.body.nodes)] == [15] and step_ends(P, {"TSelect_v1"}) == [7, 11, 15]
    assert census(P)["steps"]["ends"] == [15]
    for bad in ([7, 11], [7, 7, 15], [2, 15], []):
        with pytest.raises(ValueError, match="step ends"):
            census(P, bad)


def test_the_digest_is_of_the_canonical_json():
    P = request()
    a, b = census(P, digest="ab" * 64), census(request())
    assert a["digest"] == seal({k: v for k, v in a.items() if k != "digest"})["digest"]
    assert a["program"] == "ab" * 64 and b["program"] is None and a["digest"] != b["digest"]
    assert census(request())["digest"] == b["digest"] != census(request(2))["digest"]
    with pytest.raises(ValueError, match="canonical JSON domain"):
        seal({"schema": "x", "n": 2 ** 53})


def test_a_batch_too_large_to_enumerate_is_marked_approximate(monkeypatch):
    monkeypatch.setattr(CE, "_MEMBER_LIMIT", 0)
    c = census(Program(comp("TColumns2", [("M", Array(3, Array(4, V32)))], Array(4, V32),
                            lambda Bd, S, M: Bd.batch(Col, M, axes=(1,)))))
    assert c["approximate"] and c["values"]["reads"][0] == [1]  # every leaf of M taken as read


def test_embedding_without_evaluate_lazy_reads_its_table():
    c = census(Program(bind(Embedding)))
    assert c["selects"] == {"call": [], "gates": [], "bits": [], "values": []}
    assert c["values"]["reads"][:2] == [[2], [2]]


Wide64 = prim("TWide64", [], V64, lambda: 1 << 60)


def typed_programs():
    """Programs whose Definitions declare ports (`test_ports`): declared intervals, a coin, a constant and a transfer; input
    bounds on part of an input; and constants at the root, one wider than the canonical JSON's integers."""
    yield "codec", TP._typed_program()
    yield "chain", Program(TP.comp("TChainUse", [("x", Array(8, V16))], Array(2, V32),
                                   lambda Bd, S, x: Bd.call(TP.chain("TChainT", TP.T_S8), Bd.batch(TP.Quant, x, axes=(0,)),
                                                            Bd.batch(TP.Quant, x, axes=(0,)))))
    yield "input-bounds", Program(TP.comp("TInB", [("x", Array(4, V16))], Array(4, V32),
                                          lambda Bd, S, x: array_of([Bd.call(TP.Mul, x[t], x[t]) for t in range(2)] +
                                                                    [Bd.call(TP.Mul, Bd.call(TP.Quant, x[t]), x[0]) for t in (2, 3)]),
                                          ports=TP.Ports(params={"x": TP.Leaves(((0, 2, TP.In(bound=(-128, 127))),))})))
    yield "constants", Program(comp("TConsts", [("x", V64)], Tuple(V64, V64, V8),
                                    lambda Bd, S, x: tuple_of(Bd.call(Wide64), x, Bd.call(TP.Tag))))



@pytest.mark.parametrize("label,P", list(programs()) + list(typed_programs()), ids=lambda x: x if isinstance(x, str) else "")
def test_a_values_fact_is_its_leaves(label, P):
    """Each value's fact is `ports.facts` of each of its leaves at its producer, and each class carries its declared ports."""
    c, fx = census(P), ports.facts(P.root)
    v = c["values"]
    for j, lo, hi, f in zip(v["producer"], v["lo"], v["hi"], v["fact"]):
        want = c["facts"][f]
        for _a, _b, got in fx.node(j).segments(lo, hi):
            iv = got.interval
            assert want == got.to_json() | ({"interval": None} if iv and max(-iv[0], iv[1]) > 2 ** 53 - 1 else {}), (label, j, lo)
    assert len(c["facts"]) == len({canonical_json(f) for f in c["facts"]})
    for fn in _fns(P.root):
        assert c["classes"][descriptor_id(fn)].get("ports") == ports.declared(fn), (label, fn.id)


def _fns(root):
    seen, todo = {}, [root]
    while todo:
        for nd in todo.pop().body.nodes:
            if id(nd.fn) not in seen:
                seen[id(nd.fn)] = nd.fn
                if not isinstance(nd.fn, PrimitiveDefinition):
                    todo.append(nd.fn)
    return list(seen.values())


def test_input_bounds_declared_intervals_and_constants_are_facts():
    c = census(dict(typed_programs())["input-bounds"])
    v, facts = c["values"], c["facts"]
    inp = [facts[f]["interval"] for j, lo, hi, f in zip(v["producer"], v["lo"], v["hi"], v["fact"]) if j == 0 for _ in range(lo, hi)]
    assert inp == [[-128, 127]] * 2 + [None] * 2
    assert {facts[f]["origin"][0] for j, f in zip(v["producer"], v["fact"]) if j == 0} == {"input"}
    assert c["classes"]["PortsTestQuant_v1"]["ports"] == {"ret": [[0, 1, {"interval": [-128, 127]}]]}
    assert "ports" not in c["classes"]["Input16_v1"]
    c = census(dict(typed_programs())["constants"])
    k = {c["callees"][i]: j for j, i in enumerate(c["nodes"]["callee"])}
    by_producer = {j: c["facts"][f] for j, f in zip(c["values"]["producer"], c["values"]["fact"])}
    assert by_producer[k["PortsTestTag_v1"]] == {"interval": [7, 7], "origin": ["static"], "sink": False, "word": None}
    assert by_producer[k["TWide64_v1"]] == {"interval": None, "origin": ["static"], "sink": False, "word": None}
    chain = census(dict(typed_programs())["chain"])
    words = [chain["facts"][f]["interval"] for j, f in zip(chain["values"]["producer"], chain["values"]["fact"])
             if chain["callees"][chain["nodes"]["callee"][j]].startswith("TChainT")]
    assert words == [list(iv) for iv in TP.T_S8]


# -- Boolean work ----------------------------------------------------------------------------------------------------

BIT, B2 = BL.BIT, Array(2, BL.BIT)
Half = comp("TBHalf", [("a", BIT), ("b", BIT)], B2,
            lambda Bd, S, a, b: array_of([Bd.call(BL.XOR, BL.operands(a, b)), Bd.call(BL.AND, BL.operands(a, b))]))


def _bstep(Bd, S, acc, x):
    lo = Bd.call(Half, acc[0], x[0])
    carry = Bd.call(BL.and_gate(3), BL.operands(lo[1], acc[1], x[1]))
    hi = Bd.call(BL.xor_gate(3), BL.operands(acc[1], x[1], lo[1]))
    return array_of([lo[0], Bd.call(BL.NOT, Bd.call(BL.XOR, BL.operands(hi, carry)))])


#: a toy accumulator step on bits, which a test binds as a tensor-core step
BStep = comp("TBStep", [("acc", B2), ("x", B2)], B2, _bstep)


def _bcoord(Bd, S, x):
    zero = Bd.call(BL.constant(0))
    acc = array_of([zero, zero])
    for k in range(S.K):
        acc = Bd.call(BStep, acc, x[k])
    return Bd.call(BL.AND, BL.operands(acc[0], Bd.call(BL.NOT, acc[1])))


BCoord = CompositeDefinition("TBCoord", 1, ("K",), lambda S: ((("x", Array(S.K, B2)),), BIT), _bcoord, register=False)
BScan = comp("TBScanStep", [("c", BIT), ("x", BIT)], Tuple(BIT, BIT),
             lambda Bd, S, c, x: tuple_of(Bd.call(BL.XOR, BL.operands(c, x)), Bd.call(BL.AND, BL.operands(c, x))))


def _broot(Bd, S, x, y):
    rows = Bd.batch(bind(BCoord, K=3), x, axes=(0,))
    carry = Bd.scan(BScan, y[0], xs=[rows], n=4)
    return tuple_of(Bd.call(BL.NOT, carry[0]), Bd.call(Half, y[0], y[1]))


#: a tiny Boolean Program: a batch of four coordinates of three steps each, a scan over their outputs and a Call of `TBHalf`
BRoot = comp("TBRoot", [("x", Array(4, Array(3, B2))), ("y", B2)], Tuple(BIT, B2), _broot)


def _walk(fn, tc: dict[str, str], under: str | None = None) -> Counter:
    """Every gate under `fn`, expanded node by node: its two-input gates and ANDs by facet, a `tc` family's (and everything
    under it) at its precision."""
    got: Counter = Counter()
    for nd in fn.body.nodes:
        n = nd.n if nd.form in ("batch", "scan") else 1
        if isinstance(nd.fn, PrimitiveDefinition):
            if (gk := BL.gate_kind(nd.fn)) is not None:
                facet = under or "cuda:gate"
                got[facet, "gates"] += n * max(1, gk[1] - 1)
                got[facet, "and"] += n * (gk[1] - 1 if gk[0] == "and" else 0)
            continue
        fam = CE.family(nd.fn)
        for key, v in _walk(nd.fn, tc, under or (f"tc:{tc[fam]}" if fam in tc else None)).items():
            got[key] += n * v
    return got


def _as_work(c: Counter) -> dict:
    return {f: {"gates": c[f, "gates"], "and": c[f, "and"]} for f in sorted({f for f, _ in c}) if c[f, "gates"]}


def _bstep_bound(monkeypatch, dtype: str = "e4m3") -> None:
    monkeypatch.setitem(CE.PRODUCTS, "TBStep_v1", 2)
    monkeypatch.setitem(CE.PRECISION, "TBStep_v1", "fp8")
    monkeypatch.setattr(CE, "SILICON", {})
    monkeypatch.setattr(BStep, "silicon", None, raising=False)
    silicon_step(BStep, SimpleNamespace(dtype=dtype))


def test_a_boolean_programs_work_is_its_gates(monkeypatch):
    """Each class's, step's and the totals' work is a direct expansion of the gates (an n-input gate n - 1 two-input gates) and
    the compiled circuit's two-input operations (`evaluation.bits`); a bound step's gates are its precision's, all others
    `cuda:gate`. Without `boolean` the census has no work."""
    P = Program(BRoot)
    plain = census(P)
    assert "work" not in plain["totals"] and "work" not in plain["steps"] and all("work" not in e for e in plain["classes"].values())
    ops = BS.compile(BRoot).counts
    for tc in ({}, {"TBStep_v1": "fp8"}):
        if tc:
            _bstep_bound(monkeypatch)
        c = census(P, boolean=True)
        want = _as_work(_walk(P.root, tc))
        assert c["totals"]["work"] == want and c["steps"]["work"] == [want]
        assert sum(w["gates"] for w in want.values()) == sum(ops.values()) and sum(w["and"] for w in want.values()) == ops["and"]
        assert c["classes"]["TBCoord_v1{K=3}"]["work"] == _as_work(_walk(bind(BCoord, K=3), tc)), tc
        assert c["classes"]["And3_v1"]["work"] == {"cuda:gate": {"gates": 2, "and": 2}}
        assert c["classes"]["Xor3_v1"]["work"] == {"cuda:gate": {"gates": 2, "and": 0}}
        assert c["classes"]["Const1[0x0]_v1"]["work"] == {} and c["classes"]["Input1_v1"]["work"] == {}
    assert set(c["totals"]["work"]) == {"tc:fp8", "cuda:gate"}
    assert c["classes"]["TBStep_v1"]["work"] == {"tc:fp8": {"gates": 8, "and": 3}}
    assert c["classes"]["TBHalf_v1"]["work"] == {"cuda:gate": {"gates": 2, "and": 1}}


def test_silicon_step_refusals(monkeypatch):
    """A binding is refused unless `PRODUCTS` counts the family at the model's precision; a `PRODUCTS` step bound to no model
    is refused when work is counted, and only then."""
    P = Program(BRoot)
    with pytest.raises(ValueError, match="census.PRODUCTS has no k for it"):
        silicon_step(BStep, SimpleNamespace(dtype="e4m3"))
    monkeypatch.setitem(CE.PRODUCTS, "TBStep_v1", 2)
    monkeypatch.setitem(CE.PRECISION, "TBStep_v1", "fp8")
    assert census(P)["totals"]["served_macs_by_precision"] == {"mac:fp8": 24}
    with pytest.raises(ValueError, match=r"^TBStep_v1 is a tensor-core step \(census.PRODUCTS\) that census.SILICON binds to no "
                                         r"silicon model: import the module defining TBStep_v1"):
        census(P, boolean=True)
    for model, why in ((SimpleNamespace(dtype="bf16"), "computes bf16 products; census.PRECISION gives fp8"),
                       (SimpleNamespace(dtype="u8", b_dtype="e4m3"), r"operands \('u8', 'e4m3'\) are not one precision"),
                       (SimpleNamespace(dtype="f16"), "are not one precision")):
        with pytest.raises(ValueError, match=why):
            silicon_step(BStep, model)
    _bstep_bound(monkeypatch)
    assert CE.SILICON == {"TBStep_v1": "fp8"} and BStep.silicon.dtype == "e4m3"


#: a word Program over a word step, and its Boolean twin (`TBCoord_v1`) as a resolver's Boolean version
WStep = prim("TWStep", [("acc", Value(2)), ("x", Value(2))], Value(2), lambda acc, x: (acc + x) & 3)
WZero = prim("TWZero2", [], Value(2), lambda: 0)
WOut = prim("TWOut", [("acc", Value(2))], Value(1), lambda acc: acc & 1)


def _wcoord(Bd, S, x):
    acc = Bd.call(WZero)
    for k in range(S.K):
        acc = Bd.call(WStep, acc, x[k])
    return Bd.call(WOut, acc)


WCoord = CompositeDefinition("TWCoord", 1, ("K",), lambda S: ((("x", Array(S.K, Value(2))),), Value(1)), _wcoord, register=False)
WRoot = comp("TWRoot", [("x", Array(4, Array(3, Value(2))))], Array(4, Value(1)),
             lambda Bd, S, x: Bd.batch(bind(WCoord, K=3), x, axes=(0,)))
BTwin = comp("TBTwin", [("x", Array(4, Array(3, B2)))], Array(4, BIT), lambda Bd, S, x: Bd.batch(bind(BCoord, K=3), x, axes=(0,)))


def test_a_word_program_counts_through_its_boolean_versions(monkeypatch):
    _bstep_bound(monkeypatch)
    twins = {"TWCoord_v1{K=3}": bind(BCoord, K=3)}
    c = census(Program(WRoot), boolean=lambda fn: twins[descriptor_id(fn)])
    b = census(Program(BTwin), boolean=True)
    assert c["totals"]["work"] == b["totals"]["work"] == _as_work(_walk(Program(BTwin).root, {"TBStep_v1": "fp8"}))
    assert c["steps"]["work"] == b["steps"]["work"]
    w = c["classes"]["TWCoord_v1{K=3}"]
    assert w["boolean_version"] == "TBCoord_v1{K=3}" and w["work"] == b["classes"]["TBCoord_v1{K=3}"]["work"]
    assert "work" not in c["classes"]["TWStep_v1"] and "boolean_version" not in b["classes"]["TBCoord_v1{K=3}"]


def test_a_call_without_a_boolean_version_is_refused_by_name():
    with pytest.raises(NoBooleanWork, match=r"^census work: no Boolean gate count for TWCoord_v1: TWCoord_v1\{K=3\}: it reaches "
                                            r"TW\w+_v1, which is not a Boolean gate"):
        census(Program(WRoot), boolean=True)

    def nothing(fn):
        raise LookupError(f"no Boolean version of {fn.base_id}")

    with pytest.raises(NoBooleanWork) as e:
        census(Program(WRoot), boolean=nothing)
    assert e.value.missing == {"TWCoord_v1{K=3}": "no Boolean version of TWCoord_v1"}
    with pytest.raises(NoBooleanWork, match=r"its Boolean version TWCoord_v1\{K=3\} reaches TW\w+_v1"):
        census(Program(WRoot), boolean=lambda fn: fn)
    with pytest.raises(NoBooleanWork, match=r"^census work: no Boolean gate count for TGemmCoordinate_v1: "):
        census(Program(bind(Gemm, K=32, N=6)), boolean=True)
