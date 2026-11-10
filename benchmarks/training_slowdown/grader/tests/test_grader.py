"""The grader (`grader.py`) on the lf model's tiny training step (conftest's `steps`), f32, with bf16 matrix products, and with
those on sm_120's cuBLAS routes, beside a stand-in for training's slowdown code (`CODE`): the hand-over it reads, in a fresh
process too, and each one it refuses; the step's matrix products as the step's ATen graph has them (each mm and bmm with its
dims, phase, operand and output widths and token dims, N and O, and its cells' |C|); the GPU step's linears as its tensor
core's dots; the products it refuses to read; the comparison with a trace in the slowdown code's row format (`against`); and
|C|'s pricing."""
from __future__ import annotations

import collections
import dataclasses
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from conftest import CC, LF, STEP, STRUCTURE, dot_alone, k16_steps, route_factor

import grader as G
from verity.definitions.circuits.census import census
from verity.circuits.codec import canonical_json
from verity.circuits.program import REGISTRATION, input_prim
from verity_vllm.program.frontend.train_step import primitives

aten = torch.ops.aten
#: training's slowdown code as the grader calls it (`load_slowdown_code`'s namespace), standing in: the ops it says compute
#: products are the step's mm and bmm, as its `opnames.gemm_like` says of the lf step, and `HE_BITS` gives two Definitions'
#: operand and output widths by their gates
CODE = SimpleNamespace(opnames=SimpleNamespace(gemm_like=lambda name: name in G.PRODUCTS),
                       formalism=SimpleNamespace(HE_BITS={1.0: (16, 16), 2.0: (32, 32)}),
                       version="stand-in", commit="0" * 40, checked=False)
#: the forward's matrix products of the tiny model: each layer's seven linears and attention's two, and the logits
FORWARD = 9 * LF.SIZES["tiny"].n_layer + 1
AGREE = ("count", "N", "O", "widths", "token_dims")
#: the tiny steps: f32, bf16 products on torch CPU's arithmetic, and on sm_120's cuBLAS
STEPS = {"f32": (None, None), "bf16": (torch.bfloat16, None), "bf16-cublas": (torch.bfloat16, CC)}


def test_a_handover_reads_back(steps) -> None:
    s = steps()
    doc = json.loads(s.path.read_text())
    assert (doc["format"], doc["step"]["batch"], doc["structure"]["batch"]) == (G.FORMAT, list(STEP), list(STRUCTURE))
    assert sorted(doc["registered"]) == sorted(s.program.theta)
    for where in (s.path, s.path.parent):
        h = G.load_handover(where)
        assert h.path == s.path and sorted(h.step.root.theta) == sorted(s.program.root.theta)


@pytest.mark.slow
def test_a_handover_reads_in_a_fresh_process(steps) -> None:
    """A process that derived no step decodes a hand-over: the grader registers the Definitions the frontend lowers to, the
    GPU's routes among them, before it decodes."""
    s = steps(torch.bfloat16, CC)
    script = (f"import sys; sys.path.insert(0, {str(Path(G.__file__).parent)!r}); import grader as G; "
              f"print(len(G.load_handover({str(s.path)!r}).step.root.body.nodes))")
    got = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert got.returncode == 0 and got.stdout.split() == [str(len(s.program.root.body.nodes))], got.stderr[-2000:]


def _record(d, edit) -> None:
    p = d / "handover.json"
    doc = json.loads(p.read_text())
    edit(doc)
    p.write_text(json.dumps(doc))


def _program(d, key: str, edit) -> dict:
    p = d / f"{key}.json.gz"
    desc = json.loads(gzip.decompress(p.read_bytes()))
    edit(desc)
    p.write_bytes(gzip.compress(json.dumps(desc).encode()))
    return desc


def _annotated(edit):
    """An edit of the step Program's annotations, the record naming the annotations as edited."""
    def go(d) -> None:
        desc = _program(d, "step", lambda desc: edit(desc["annotations"]))
        _record(d, lambda doc: doc["step"].update(annotations=hashlib.sha512(canonical_json(desc["annotations"])).hexdigest()))
    return go


#: an edit of a hand-over, and what the grader's refusal of it says
REFUSALS = {
    "format": (lambda d: _record(d, lambda doc: doc.update(format="grader-handover/v0")), "isn't a grader hand-over"),
    "run": (lambda d: _record(d, lambda doc: doc["run"].pop("id")), "the hand-over's run has no 'id'"),
    "checkpoint": (lambda d: _record(d, lambda doc: doc["run"]["checkpoints"][0].update(steps=0)), "with at least one step"),
    "frontend": (lambda d: _record(d, lambda doc: doc["frontend"].update(commit="")), "the hand-over's frontend has no 'commit'"),
    "registered": (lambda d: _record(d, lambda doc: doc["registered"].pop()), "theta as registered isn't the step Program's"),
    "batch": (lambda d: _record(d, lambda doc: doc["step"].update(batch=[2])), r"batch shape \[2\] isn't two sizes"),
    "tokens": (lambda d: _record(d, lambda doc: doc["structure"].update(batch=[2, 5])), "share a token-indexed size"),
    "file": (lambda d: (d / "step.json.gz").unlink(), "cannot read the step Program"),
    "digest": (lambda d: _program(d, "step", lambda desc: desc.update(extra=1)), "the step Program's digest is"),
    "annotations": (lambda d: _program(d, "structure", lambda desc: desc["annotations"].update(extra=1)),
                    "the structure Program's annotations aren't the ones the hand-over names"),
    "derived": (_annotated(lambda ann: ann.pop("train_step")), "the step Program isn't a training step the frontend derived"),
    "inputs": (_annotated(lambda ann: ann["correspondence"]["inputs"].pop()),
               "the step Program's correspondence doesn't name its \\d+ input nodes"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_handover_is_refused_with_its_fix(case, steps, tmp_path) -> None:
    edit, said = REFUSALS[case]
    d = tmp_path / "handover"
    shutil.copytree(steps().path.parent, d)
    edit(d)
    with pytest.raises(G.Refused, match=said):
        G.load_handover(d)


def _graph_products(gm, gm2) -> dict:
    """Each mm and bmm of a step's ATen graph by node: its op, dims, operand widths, output width and token dims (those that
    differ in the structure's graph `gm2`)."""
    def dims(n) -> dict:
        a, b = (x.meta["val"].shape for x in n.args[:2])
        return ({"batch": 1, "m": a[0], "k": a[1], "n": b[1]} if n.target == aten.mm.default
                else {"batch": a[0], "m": a[1], "k": a[2], "n": b[2]})

    def width(n) -> int:
        return n.meta["val"].dtype.itemsize * 8

    other = {n.name: n for n in gm2.graph.nodes}
    return {n.name: (n.target.overloadpacket.__name__, dims(n), [(width(n.args[0]), width(n.args[1]))], [width(n)],
                     sorted(d for d, v in dims(n).items() if dims(other[n.name])[d] != v))
            for n in gm.graph.nodes if n.op == "call_function" and n.target in (aten.mm.default, aten.bmm.default)}


def _cell_gates(h) -> collections.Counter:
    """Each occurrence's multiply-add cells' word gates, off the decoded step Program's own nodes."""
    nodes, calls = h.step.root.body.nodes, h.step.annotations["correspondence"]["calls"]
    got: collections.Counter = collections.Counter()
    for nd, c in zip(nodes[len(nodes) - len(calls):], calls):
        if getattr(nd.fn, "base_id", None) in G.MAC_CELLS:
            got[c["occ"]] += (nd.n if nd.form in ("batch", "scan") else 1) * nd.fn.gates
    return got


@pytest.mark.parametrize("step", sorted(STEPS))
def test_the_products_are_the_steps_matrix_products(step, steps, unit_prices) -> None:
    """Each mm and bmm of the step's ATen graph is one product row, with the graph's dims, operand and output widths (the bf16
    linears' 16 bits; attention's 32, as torch's math path computes it from upcast operands) and the token dims the structure's
    graph varies, in its phase: the forward's products, and two in the backward for each. Every operand depends on θ′, so N
    is every product's MACs and O every output's wires. Each row's |C| is its cells' (at one gate a primitive, their word
    gates)."""
    dtype, cc = STEPS[step]
    s = steps(dtype, cc)
    h = G.load_handover(s.path)
    out = G.read(h, CODE)
    assert out["format"] == G.READ and out["slowdown_code"] == {"version": "stand-in", "commit": "0" * 40, "checked": False,
                                                                "capture": G.CAPTURE}
    rows = [r for r in out["products"] if r["op"] in G.PRODUCTS]
    graph = _graph_products(s.gm, s.gm2)
    assert {r["node"]: (r["op"], r["dims"], r["widths"], r["out_widths"], r["token_dims"]) for r in rows} == graph
    assert collections.Counter(r["phase"] for r in rows) == {"forward": FORWARD, "backward": 2 * FORWARD}
    assert all(r["macs"] == r["macs_both"] == r["dims_macs"] and r["dep"] == [True, True] for r in rows)
    t = out["totals"]
    assert t["products"] == len(graph) == 3 * FORWARD
    assert t["N"] == t["macs"] == sum(math.prod(d.values()) for _, d, *_ in graph.values())
    assert t["O"] == sum(d["batch"] * d["m"] * d["n"] * w for _, d, _, (w,), _ in graph.values())
    assert (t["widths"], t["out_widths"]) == (([(16, 16), (32, 32)], [16, 32]) if dtype else ([(32, 32)], [32]))
    assert {r["occ"]: r["gates"] for r in out["products"]} == _cell_gates(h) and not t["unpriced"]
    assert t["gates"] == sum(t["gates_by_op"].values()) == sum(t["gates_by_phase"].values()) == sum(r["gates"] for r in rows)


def test_the_gpu_steps_linears_are_its_tensor_cores_dots(steps, monkeypatch) -> None:
    """On sm_120 each linear's GEMM is its route's row cells (conftest's `ROUTES`: unsplit, split-K over a bf16 workspace,
    serial split-K and the three over grid.z), each k16 step a call of the tensor-core dot whose Boolean Definition the
    slowdown code prices a bf16 MAC at (`HopperBF16WgmmaDot16_v2`, its gates over 16): priced at that dot alone, a linear's
    |C| is that price for each k16 step its route takes, a trace's at its gates a MAC but for the zeros a grid.z slice pads
    its residue tile with, and what their full |C| adds rounds and reduces. Attention's products stay `LinearRef_v1` rows at
    the 32 bits torch's math path computes them in, which call no dot."""
    s = steps(torch.bfloat16, CC)
    price = dot_alone(monkeypatch, s.program)
    assert price["boolean"] == "HopperBF16WgmmaDot16_v2"
    h = G.load_handover(s.path)
    out, k16 = G.read(h, CODE), k16_steps(h)
    mm, bmm = ([r for r in out["products"] if r["op"] == op] for op in G.PRODUCTS)
    assert {c.split("{")[0] for r in mm for c in r["cells"]} == set(G.ROWS) - {G.CELL}
    assert all(r["gates"] == k16[r["occ"]] * price["gates"] for r in mm)
    assert all(r["gates"] == 0 and r["widths"] == [(32, 32)] and {c.split("{")[0] for c in r["cells"]} == {G.CELL} for r in bmm)
    padded = [r for r in mm if 16 * k16[r["occ"]] != r["macs"]]
    assert padded and all(16 * k16[r["occ"]] > r["macs"] and "Aligned" in r["cells"][0] for r in padded)
    C = G.against(out, _trace(out["products"], 1.0, price["gates"] / 16), CODE)["C"]
    assert C["mm"]["circuit"] == out["totals"]["gates_by_op"]["mm"]
    assert C["mm"]["ratio"] == pytest.approx(route_factor(k16, out["products"])) and C["mm"]["ratio"] > 1


def test_a_structure_of_another_step_is_refused(steps) -> None:
    h = G.load_handover(steps().path)
    other = G.load_handover(steps(torch.bfloat16).path)
    with pytest.raises(G.Refused, match="derive both with step_program from the same model"):
        G.read(dataclasses.replace(h, structure=other.structure), CODE)


def test_a_product_it_doesnt_read_is_refused(steps) -> None:
    """An op the slowdown code says computes products that isn't an mm or bmm (here the stand-in's mul), and an mm whose rows
    aren't its row cells, are refused: the grader reads a step's products whole or not at all."""
    h = G.load_handover(steps().path)
    code = SimpleNamespace(**{**vars(CODE), "opnames": SimpleNamespace(gemm_like=lambda name: name in (*G.PRODUCTS, "mul"))})
    with pytest.raises(G.Refused, match=r"\(aten\.mul\.\w+\) computes matrix products the grader doesn't read"):
        G.read(h, code)
    p = G._Program(h.step, "step")
    o = p.ops.index("mm")
    p.calls = [{**c, "occ": None} if c["occ"] == o else c for c in p.calls]
    d = G._dims(p, o)
    with pytest.raises(G.Refused, match=rf"one for each of its {d['batch'] * d['m']} rows \(0 rows\)"):
        G._check_products(p, CODE)


#: a stand-in trace's |C| besides its products'
REST = 500


def _trace(rows: list[dict], cell: float, gates_per_mac: float = 2.0) -> dict:
    """The circuit's product `rows` as training's slowdown code writes a step's (its capture's .products.json): identical rows
    one row with their count, each at the Definition whose gates are `cell`, a MAC at `gates_per_mac`, and the step's |C|
    its products' and `REST`."""
    merged: dict = {}
    for r in rows:
        if r["op"] in G.PRODUCTS:
            d = r["dims"]
            row = {"phase": r["phase"], "op": r["op"], "dims": d, "token_dims": r["token_dims"], "macs": math.prod(d.values()),
                   "outputs": d["batch"] * d["m"] * d["n"], "affected": [True, True], "widths": list(r["widths"][0]),
                   "cell_gates": cell, "gates_per_mac": gates_per_mac}
            key = json.dumps(row, sort_keys=True)
            merged[key] = {"count": merged.get(key, {}).get("count", 0) + 1, **row}
    rows = list(merged.values())
    return {"capture": G.CAPTURE, "gates": sum(r["count"] * r["macs"] * gates_per_mac for r in rows) + REST, "rows": rows}


def test_against_a_trace(steps, unit_prices) -> None:
    """The circuit's products beside a trace of the same step in the slowdown code's row format agree at every (phase, op,
    dims) on count, N, O, widths and token dims; each difference shows in its own column; both sides' |C| are the step's,
    each op's products', all products' and the rest's, the trace's products at its gates a MAC; a trace of another capture
    is refused."""
    out = G.read(G.load_handover(steps().path), CODE)
    a = G.against(out, _trace(out["products"], 2.0), CODE)
    assert a["agree"] == dict.fromkeys(AGREE, True)
    t = out["totals"]
    base = {"count": 3 * FORWARD, "N": t["N"], "O": t["O"]}
    assert (a["totals"]["circuit"], a["totals"]["trace"]) == ({**base, "gates": t["gates"]}, {**base, "gates": 2 * t["macs"]})
    assert len(a["rows"]) < 3 * FORWARD
    macs = {op: sum(r["macs"] for r in out["products"] if r["op"] == op) for op in G.PRODUCTS}
    circuit = {"step": out["C"]["gates"], **t["gates_by_op"], "products": t["gates"], "rest": out["C"]["gates"] - t["gates"]}
    trace = {"step": 2 * t["macs"] + REST, **{op: 2 * m for op, m in macs.items()}, "products": 2 * t["macs"], "rest": REST}
    assert a["C"] == {k: {"circuit": circuit[k], "trace": trace[k], "ratio": circuit[k] / trace[k]} for k in circuit}

    def agree(edit=None, cell: float = 2.0) -> list[str]:
        trace = _trace(out["products"], cell)
        if edit:
            edit(trace)
        return [k for k, v in G.against(out, trace, CODE)["agree"].items() if not v]

    assert agree(cell=1.0) == ["O"]          # a Definition with 16-bit outputs: the trace's O is half the circuit's
    assert agree(cell=3.0) == []             # one HE_BITS doesn't list: its outputs at the narrower operand's width
    assert agree(lambda tr: tr["rows"][0].update(affected=[True, False])) == ["N", "O"]
    assert agree(lambda tr: tr["rows"][0].update(widths=[16, 32])) == ["widths"]
    assert agree(lambda tr: tr["rows"][0].update(token_dims=["seq"])) == ["token_dims"]
    assert agree(lambda tr: tr["rows"].pop()) == ["count", "N", "O", "widths", "token_dims"]
    with pytest.raises(G.Refused, match=f"the trace is capture 4.12.8's, and the grader reads beside {G.CAPTURE}'s"):
        G.against(out, {**_trace(out["products"], 2.0), "capture": "4.12.8"}, CODE)


def test_a_trace_is_its_product_rows(tmp_path) -> None:
    (tmp_path / "a.json").write_text("{")
    with pytest.raises(G.Refused, match="cannot read a trace"):
        G.load_trace(tmp_path / "a.json")
    (tmp_path / "b.json").write_text(json.dumps({"rows": []}))
    with pytest.raises(G.Refused, match="isn't training's slowdown code's product rows"):
        G.load_trace(tmp_path / "b.json")
    good = {"capture": G.CAPTURE, "gates": 1, "rows": []}
    (tmp_path / "c.json").write_text(json.dumps(good))
    assert G.load_trace(tmp_path / "c.json") == good


def test_each_class_is_priced_at_its_primitives_boolean_prices() -> None:
    cen = {"callees": ["x", "A", "B"],
           "classes": {"x": {"kind": "input"}, "A": {"kind": "call", "gates": 3, "primitives": {"p": 2, "q": 1}},
                       "B": {"kind": "call", "gates": 1, "primitives": {"q": 1}}},
           "nodes": {"callee": [0, 1, 1, 2], "n": [1, 4, 1, 5]}}
    prices = {"p": {"boolean": "P_v1", "gates": 10, "and": 3}, "q": {"boolean": None, "why": "no Boolean version"}}
    assert G.priced(cen, prices) == {
        "classes": [{"class": "A", "instances": 5, "word_gates": 15, "gates": 100, "ands": 30, "unpriced": ["q"]},
                    {"class": "B", "instances": 5, "word_gates": 5, "gates": 0, "ands": 0, "unpriced": ["q"]}],
        "gates": 100, "ands": 30, "word_gates": 20, "unpriced_word_gates": {"q": 10}}


def test_inputs_have_no_boolean_price() -> None:
    """A step's inputs, a bool one among them (whose Boolean Definition is itself, no composite to compile), are bound, not
    computed: `boolean_prices` leaves them out."""
    assert G.boolean_prices({p.id: p for p in (input_prim(1), input_prim(32), input_prim(64, REGISTRATION))}) == {}


@pytest.mark.slow
def test_circuit_size_on_the_tiny_step(steps) -> None:
    """|C| of the tiny f32 step, read off its hand-over as `read` reads it: each root class at its primitives' Boolean prices,
    the matrix products' LinearRef_v1 rows among them, the primitives with no Boolean version left out by name (but the
    inputs', which |C| doesn't count), and the same |C| as the step Program's before it was handed over."""
    s = steps()
    C = G.circuit_size(G.load_handover(s.path).step)
    assert C["gates"] == sum(r["gates"] for r in C["classes"]) > 0 and C["ands"] == sum(r["ands"] for r in C["classes"])
    assert all(r["gates"] > 0 for r in C["classes"] if r["class"].startswith(G.CELL))
    prices = G.boolean_prices(primitives(s.program))
    assert set(C["unpriced_word_gates"]) == {pid for pid, p in prices.items() if p["boolean"] is None}
    assert G.priced(census(s.program), prices) == C
