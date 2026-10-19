"""`verity.ml.circuits.census`: census/v1 against the lowered circuit (each value's bits, readers, candidates and output from the
gates' operands, its classes from the activations), served MACs and family rules, steps, and the digest."""
from __future__ import annotations

from collections import Counter, defaultdict

import pytest

import test_ports as TP
from test_ir_liveness import Add16, Add32, Embedding, Gemm, Mul16, comp, prim, programs as liveness_programs
from verity.ml.circuits import census as CE, ports
from verity.ml.circuits.census import census, seal, step_ends
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
                                            "tail_atoms": 0, "receipt_folds": 0}
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


def test_served_macs_and_family_rules(monkeypatch):
    P = Program(bind(Gemm, K=32, N=6))
    assert census(P)["totals"]["served_macs"] == 0
    monkeypatch.setitem(CE.PRODUCTS, "TDot16_v1", 16)
    c = census(P)
    assert c["classes"]['TGemmCoordinate_v1{K=32}']["served_macs"] == 32 and c["totals"]["served_macs"] == 6 * 32
    assert c["steps"]["served_macs"] == [192]
    monkeypatch.setitem(CE.FAMILY_COUNTS, "TGemmCoordinate_v1", lambda S: {"served_macs": S["K"] - 2, "tail_atoms": 1})
    c = census(P)
    assert {k: c["totals"][k] for k in CE.COUNTERS} == {"served_macs": 180, "tail_atoms": 6, "receipt_folds": 0}
    assert c["classes"]['TGemmCoordinate_v1{K=32}']["tail_atoms"] == 1 and c["steps"]["tail_atoms"] == [6]
    assert c["classes"]['TDot_v1{K=32}']["served_macs"] == 32  # under the rule's class, the bottom-up count stays the class's own
    monkeypatch.setitem(CE.FAMILY_COUNTS, "TGemmCoordinate_v1", lambda S: {"served": 1})
    with pytest.raises(ValueError, match="non-negative ints"):
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
    bounds on part of an input (a harness's, the one Program whose inputs may declare one: `ports.harness`); and constants at
    the root, one wider than the canonical JSON's integers."""
    yield "codec", TP._typed_program()
    yield "chain", Program(TP.comp("TChainUse", [("x", Array(8, V16))], Array(2, V32),
                                   lambda Bd, S, x: Bd.call(TP.chain("TChainT", TP.T_S8), Bd.batch(TP.Quant, x, axes=(0,)),
                                                            Bd.batch(TP.Quant, x, axes=(0,)))))
    yield "input-bounds", TP.standalone_call(bind(TP.comp(
        "TInB", [("x", Array(4, V16))], Array(4, V32),
        lambda Bd, S, x: array_of([Bd.call(TP.Mul, x[t], x[t]) for t in range(2)] +
                                  [Bd.call(TP.Mul, Bd.call(TP.Quant, x[t]), x[0]) for t in (2, 3)]),
        ports=TP.Ports(params={"x": TP.Leaves(((0, 2, TP.In(bound=(-128, 127))),))}))))[0]
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
