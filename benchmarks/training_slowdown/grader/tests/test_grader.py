"""The grader (`grader.py`) on the lf model's tiny training step (conftest's `steps`), f32, with bf16 matrix products, and with
those on sm_120's cuBLAS routes and its attention sm_120's flash attention, beside a stand-in for training's slowdown code
(`CODE`): the hand-over it reads, in a fresh process too, and each one it refuses; the step's matrix products as the step's
ATen graph has them (each mm and bmm, and each of a flash attention's products, with its dims, phase, operand and output
widths and token dims, N and O, and its cells' |C|); the GPU step's linears and attention as its tensor core's dots; the
fused attention's table (`ATTENTION`) held to its head Definitions' census; the products it refuses to read; the comparison
with a trace in the slowdown code's row format (`against`); and |C|'s pricing."""
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
from verity.circuits.defs import bind
from verity.definitions.circuits.census import census
from verity.circuits.codec import canonical_json, descriptor_id
from verity.circuits.program import REGISTRATION, Program, input_prim
from verity.definitions.ops.gemm import DotBf16
from verity.definitions.ops.prims import HopperBF16WgmmaDot16
from verity.definitions.registry import fa2_train as FT
from verity.definitions.registry import fa2_train_bwd as FTB
from verity_vllm.program.frontend.train_step import primitives

aten = torch.ops.aten
#: training's slowdown code as the grader calls it (`load_slowdown_code`'s namespace), standing in: the ops it says compute
#: products are the step's mm and bmm and torch's flash attention forward and backward, as its `opnames.gemm_like` says of
#: the lf step, and `HE_BITS` gives two Definitions' operand and output widths by their gates
CODE = SimpleNamespace(opnames=SimpleNamespace(gemm_like=lambda name: name in (*G.PRODUCTS, *G.ATTENTION)),
                       work=SimpleNamespace(HE_BITS={1.0: (16, 16), 2.0: (32, 32)}),
                       version="stand-in", commit="0" * 40, checked=False)
LAYERS = LF.SIZES["tiny"].n_layer
#: the forward's matrix products of the tiny model: each layer's seven linears and attention's two, and the logits
FORWARD = 9 * LAYERS + 1
AGREE = ("count", "N", "O", "widths", "token_dims")
#: the tiny steps: f32, bf16 products on torch CPU's arithmetic, and on sm_120's cuBLAS and flash attention
STEPS = {"f32": (None, None), "bf16": (torch.bfloat16, None), "bf16-cublas": (torch.bfloat16, CC)}
#: torch's flash attention ops: the argument their query is, and the bmm of torch's math path each product of theirs stands
#: for, its m, k and n of the head's T and D; the backward computes the scores again, which the math path saves
FUSED = {aten._scaled_dot_product_flash_attention.default: (0, {"scores": "TDT", "pv": "TTD"}),
         aten._scaled_dot_product_flash_attention_backward.default: (
             1, {"scores": "TDT", "dp": "TDT", "dv": "TTD", "dk": "DTT", "dq": "TTD"})}
#: the product of a fused attention the math path doesn't compute
RECOMPUTED = ("_scaled_dot_product_flash_attention_backward", "scores")


@pytest.mark.slow
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
    """Each mm and bmm of a step's ATen graph by node, and each product of a flash attention op by node and product (`FUSED`):
    its op, dims, operand widths, output width and token dims (those that differ in the structure's graph `gm2`). A flash
    attention's products are the bmm's each stands for at its query's [B, H, T, D], on bf16 operands into the f32
    accumulators of the card's tensor cores."""
    def dims(n, product=None) -> dict:
        if product is not None:
            B, H, T, D = n.args[FUSED[n.target][0]].meta["val"].shape
            return {"batch": B * H, **{d: {"T": T, "D": D}[s] for d, s in zip("mkn", FUSED[n.target][1][product])}}
        a, b = (x.meta["val"].shape for x in n.args[:2])
        return ({"batch": 1, "m": a[0], "k": a[1], "n": b[1]} if n.target == aten.mm.default
                else {"batch": a[0], "m": a[1], "k": a[2], "n": b[2]})

    def width(n) -> int:
        return n.meta["val"].dtype.itemsize * 8

    other = {n.name: n for n in gm2.graph.nodes}
    got = {}
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target not in (aten.mm.default, aten.bmm.default, *FUSED):
            continue
        ws = ([(16, 16)], [32]) if n.target in FUSED else ([(width(n.args[0]), width(n.args[1]))], [width(n)])
        for product in FUSED[n.target][1] if n.target in FUSED else (None,):
            d, d2 = dims(n, product), dims(other[n.name], product)
            got[n.name, product] = (n.target.overloadpacket.__name__, d, *ws, sorted(x for x in d if d2[x] != d[x]))
    return got


def _cell_gates(h) -> collections.Counter:
    """Each occurrence's multiply-add cells' word gates, off the decoded step Program's own nodes."""
    nodes, calls = h.step.root.body.nodes, h.step.annotations["correspondence"]["calls"]
    got: collections.Counter = collections.Counter()
    for nd, c in zip(nodes[len(nodes) - len(calls):], calls):
        if getattr(nd.fn, "base_id", None) in G.MAC_CELLS:
            got[c["occ"]] += (nd.n if nd.form in ("batch", "scan") else 1) * nd.fn.gates
    return got


def _head_steps(h) -> dict:
    """Each fused attention occurrence's k16 steps, off the step's census: its head classes' calls of the tensor-core dot
    times their instances."""
    cen, calls = census(h.step), h.step.annotations["correspondence"]["calls"]
    nodes, k0 = h.step.root.body.nodes, len(h.step.root.body.nodes) - len(calls)
    got: collections.Counter = collections.Counter()
    for i, c in enumerate(calls, k0):
        if getattr(nodes[i].fn, "base_id", None) in G.HEADS:
            cl = cen["classes"][cen["callees"][cen["nodes"]["callee"][i]]]
            got[c["occ"]] += cen["nodes"]["n"][i] * cl["primitives"][HopperBF16WgmmaDot16.id]
    return got


@pytest.mark.parametrize("step", sorted(STEPS))
def test_the_products_are_the_steps_matrix_products(step, steps, unit_prices) -> None:
    """Each mm and bmm of the step's ATen graph is one product row, and so is each product of a flash attention op (two of the
    forward's, five of the backward's), with the graph's dims (a flash attention's the bmm's it stands for), operand and
    output widths (the bf16 linears' 16 bits; attention's 32 on torch's math path, which computes it from upcast operands, and
    on the card bf16 operands into f32 accumulators) and the token dims the structure's graph varies, in its phase: the
    forward's products, and two in the backward for each but a flash attention's, which has five for two. Every operand
    depends on θ′, so N is every product's MACs and O every output's wires: an mm's and a bmm's their dims', a flash
    attention's 16 for each k16 step its heads take in the step's census. Each row's |C| is its cells' (at one gate a
    primitive, their word gates), a flash attention's its heads', on its occurrence."""
    dtype, cc = STEPS[step]
    s = steps(dtype, cc)
    h = G.load_handover(s.path)
    out = G.read(h, CODE)
    assert out["format"] == G.READ and out["slowdown_code"] == {"version": "stand-in", "commit": "0" * 40, "checked": False,
                                                                "capture": G.CAPTURE}
    rows = [r for r in out["products"] if r["op"] in (*G.PRODUCTS, *G.ATTENTION)]
    fused = [r for r in rows if r["product"]]
    graph = _graph_products(s.gm, s.gm2)
    assert {(r["node"], r["product"]): (r["op"], r["dims"], r["widths"], r["out_widths"], r["token_dims"]) for r in rows} == graph
    assert len(fused) == (7 * LAYERS if cc else 0) and all(r["gates"] is None for r in fused)
    assert collections.Counter(r["phase"] for r in rows) == {"forward": FORWARD, "backward": 2 * FORWARD + len(fused) // 7}
    assert all(r["macs"] == r["macs_both"] and r["dep"] == [True, True] for r in rows)
    assert all(r["macs"] == r["dims_macs"] for r in rows if not r["product"])
    steps16 = _head_steps(h)
    assert {o: sum(r["macs"] for r in fused if r["occ"] == o) for o in steps16} == {o: 16 * k for o, k in steps16.items()}
    t = out["totals"]
    assert t["products"] == len(graph) == 3 * FORWARD + len(fused) // 7
    assert t["fused"] == {"ops": 2 * len(fused) // 7, "products": len(fused)}
    assert t["N"] == t["macs"] == (sum(math.prod(d.values()) for (_, p), (_, d, *_) in graph.items() if not p)
                                   + 16 * sum(steps16.values()))
    assert t["O"] == (sum(d["batch"] * d["m"] * d["n"] * w for (_, p), (_, d, _, (w,), _) in graph.items() if not p)
                      + sum(32 * r["outputs"] for r in fused))
    assert (t["widths"], t["out_widths"]) == (([(16, 16)], [16, 32]) if cc else ([(16, 16), (32, 32)], [16, 32]) if dtype
                                              else ([(32, 32)], [32]))
    gates = {**{r["occ"]: r["gates"] for r in out["products"] if not r["product"]}, **{a["occ"]: a["gates"] for a in out["attention"]}}
    assert gates == _cell_gates(h) and not t["unpriced"]
    products = sum(gates[o] for o in {r["occ"] for r in rows})
    assert t["gates"] == sum(t["gates_by_op"].values()) == sum(t["gates_by_phase"].values()) == products


def test_the_gpu_steps_linears_are_its_tensor_cores_dots(steps, monkeypatch) -> None:
    """On sm_120 each linear's GEMM is its route's row cells (conftest's `ROUTES`: unsplit, split-K over a bf16 workspace,
    serial split-K and the three over grid.z), each k16 step a call of the tensor-core dot whose Boolean Definition the
    slowdown code prices a bf16 MAC at (`HopperBF16WgmmaDot16_v2`, its gates over 16): priced at that dot alone, a linear's
    |C| is that price for each k16 step its route takes, a trace's at its gates a MAC but for the zeros a grid.z slice pads
    its residue tile with, and what their full |C| adds rounds and reduces. Attention is the card's flash attention, no bmm:
    each occurrence's heads (`Fa2TrainHead_v1`, `Fa2TrainBwdHead_v1`) are that price for each k16 step, 1/16 of their
    products' MACs, and a trace's six whole rectangles at the same price a MAC come to the Definitions' MACs over the
    rectangles'."""
    s = steps(torch.bfloat16, CC)
    price = dot_alone(monkeypatch, s.program)
    assert price["boolean"] == "HopperBF16WgmmaDot16_v2"
    h = G.load_handover(s.path)
    out, k16 = G.read(h, CODE), k16_steps(h)
    mm, bmm = ([r for r in out["products"] if r["op"] == op] for op in G.PRODUCTS)
    assert {c.split("{")[0] for r in mm for c in r["cells"]} == set(G.ROWS) - {G.CELL}
    assert all(r["gates"] == k16[r["occ"]] * price["gates"] for r in mm)
    assert not bmm
    padded = [r for r in mm if 16 * k16[r["occ"]] != r["macs"]]
    assert padded and all(16 * k16[r["occ"]] > r["macs"] and "Aligned" in r["cells"][0] for r in padded)
    fused = [r for r in out["products"] if r["product"]]
    assert {c.split("{")[0] for a in out["attention"] for c in a["cells"]} == {"Fa2TrainHead_v1", "Fa2TrainBwdHead_v1"}
    assert all(a["gates"] == sum(r["macs"] for r in fused if r["occ"] == a["occ"]) // 16 * price["gates"]
               for a in out["attention"])
    C = G.against(out, _trace(out["products"], 1.0, price["gates"] / 16), CODE)["C"]
    assert C["mm"]["circuit"] == out["totals"]["gates_by_op"]["mm"]
    assert C["mm"]["ratio"] == pytest.approx(route_factor(k16, out["products"])) and C["mm"]["ratio"] > 1
    six = [r for r in fused if (r["op"], r["product"]) != RECOMPUTED]
    assert C["attention"]["circuit"] == out["totals"]["gates_by_op"]["attention"] > 0
    assert C["attention"]["ratio"] == pytest.approx(sum(r["macs"] for r in fused) / sum(r["dims_macs"] for r in six))


def test_a_structure_of_another_step_is_refused(steps) -> None:
    """A structure Program of another step is refused, and so is a math-path structure of a step that keeps the card's flash
    attention: its attention is another graph's occurrences."""
    h = G.load_handover(steps().path)
    other = G.load_handover(steps(torch.bfloat16).path)
    with pytest.raises(G.Refused, match="derive both with step_program from the same model"):
        G.read(dataclasses.replace(h, structure=other.structure), CODE)
    fused = G.load_handover(steps(torch.bfloat16, CC).path)
    with pytest.raises(G.Refused, match=r"the step has \d+ occurrences and the structure Program \d+: derive both"):
        G.read(dataclasses.replace(fused, structure=other.structure), CODE)


def test_a_structure_derivation_is_refused_as_the_step(steps, tmp_path) -> None:
    """A Program whose train_step annotation marks it the frontend's structure derivation is refused as a hand-over's step."""
    d = tmp_path / "handover"
    shutil.copytree(steps().path.parent, d)
    _annotated(lambda ann: ann["train_step"].update({G.STRUCTURE_MARK: True}))(d)
    with pytest.raises(G.Refused, match="the step Program is a structure derivation"):
        G.load_handover(d)


def test_a_product_it_doesnt_read_is_refused(steps) -> None:
    """An op the slowdown code says computes products that isn't an mm, a bmm or a flash attention (here the stand-in's mul),
    an mm whose rows aren't its row cells, a flash attention whose heads aren't one for each of its heads, and a head outside
    its op's occurrences, are refused: the grader reads a step's products whole or not at all."""
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
    fused = G.load_handover(steps(torch.bfloat16, CC).path)
    p = G._Program(fused.step, "step")
    first, second = [o for o, op in enumerate(p.ops) if op == "_scaled_dot_product_flash_attention"][:2]
    p.calls = [{**c, "occ": second} if c["occ"] == first else c for c in p.calls]
    x = G._dims(p, first)
    with pytest.raises(G.Refused, match=rf"one for each of its {x['B'] * x['H']} heads \(0 heads\)"):
        G._check_products(p, CODE)
    p = G._Program(fused.step, "step")
    p.calls = [{**c, "occ": None} if c["occ"] == first else c for c in p.calls]
    with pytest.raises(G.Refused, match="computes a fused attention's products on no occurrence"):
        G._check_products(p, CODE)


#: a stand-in trace's |C| besides its products'
REST = 500


def _trace(rows: list[dict], cell, gates_per_mac: float = 2.0) -> dict:
    """The circuit's product `rows` as training's slowdown code writes a step's (its capture's .products.json): identical rows
    one row with their count, each at the Definition whose gates are `cell` (or `cell(row)`), a MAC at `gates_per_mac`, and
    the step's |C| its products' and `REST`; a flash attention's products as torch's math path has them, each the whole rectangle of the bmm
    it stands for, and the backward's scores, which the math path saves, none."""
    merged: dict = {}
    for r in rows:
        if r["op"] in G.PRODUCTS or (r["product"] and (r["op"], r["product"]) != RECOMPUTED):
            d = r["dims"]
            row = {"phase": r["phase"], "op": "bmm" if r["product"] else r["op"], "dims": d, "token_dims": r["token_dims"],
                   "macs": math.prod(d.values()),
                   "outputs": d["batch"] * d["m"] * d["n"], "affected": [True, True], "widths": list(r["widths"][0]),
                   "cell_gates": cell(r) if callable(cell) else cell, "gates_per_mac": gates_per_mac}
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
    trace = {"step": 2 * t["macs"] + REST, **{op: 2 * m for op, m in macs.items()}, "attention": 0, "products": 2 * t["macs"],
             "rest": REST}
    assert a["C"] == {k: {"circuit": circuit[k], "trace": trace[k], "ratio": circuit[k] / trace[k] if trace[k] else None}
                      for k in circuit}

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


#: a flash attention's products with an operand of the softmax's, P or dS, which the card's kernels round to bf16 and torch's
#: math path keeps in f32
SOFTMAX = ("pv", "dv", "dk", "dq")


def fused_keys(out: dict, products) -> set:
    """The (phase, dims) of the circuit's flash attention `products` in `read`'s `out`."""
    return {(r["phase"], tuple(r["dims"][d] for d in G.DIMS)) for r in out["products"] if r["product"] in products}


def differences(a: dict, out: dict) -> dict[str, set]:
    """Where the circuit's products and a trace of the same step differ (`against`'s `a`, `read`'s `out`), by column
    (`AGREE`): the (phase, dims) of each row that differs in it, every one attention's; a count differs by the backward's
    scores alone, one a layer, which the math path saves."""
    def key(r) -> tuple:
        return r["phase"], tuple(r["dims"][d] for d in G.DIMS)

    rows = [r for r in a["rows"] if any(r["circuit"][x] != r["trace"][x] for x in AGREE)]
    assert all(r["op"] == "attention" for r in rows), [r for r in rows if r["op"] != "attention"]
    backward = sum(1 for x in out["attention"] if x["phase"] == "backward")
    assert all(r["circuit"]["count"] - r["trace"]["count"] == backward for r in rows if r["circuit"]["count"] != r["trace"]["count"])
    return {x: got for x in AGREE if (got := {key(r) for r in rows if r["circuit"][x] != r["trace"][x]})}


def test_against_a_trace_of_the_cards_attention(steps, unit_prices) -> None:
    """The step that keeps sm_120's flash attention beside a trace in the slowdown code's row format, its attention as torch's
    math path computes it: they agree everywhere but at attention's (phase, dims), where a flash attention's products are op
    "attention", and there they differ in three columns alone: the count, by the backward's scores, which the math path saves;
    N, the head Definitions' multiply-adds against whole rectangles, and O, the outputs they compute against the rectangles',
    both only where the scores are (at T = BN the blocks a row visits are the whole rectangle). A head's |C| isn't split by
    product: each attention key's circuit gates are None, and `C`'s attention is the heads' |C|. Each trace row is at the
    stand-in Definition of its circuit row's output words (`HE_BITS`)."""
    out = G.read(G.load_handover(steps(torch.bfloat16, CC).path), CODE)
    by_width = {b_y: gates for gates, (_, b_y) in CODE.work.HE_BITS.items()}
    a = G.against(out, _trace(out["products"], lambda r: by_width[max(r["out_widths"])]), CODE)
    scores = fused_keys(out, ("scores",))
    assert differences(a, out) == {"count": {k for k in scores if k[0] == "backward"}, "N": scores, "O": scores}
    assert all(r["circuit"]["gates"] is None for r in a["rows"] if r["op"] == "attention")
    assert a["C"]["attention"]["circuit"] == out["totals"]["gates_by_op"]["attention"] == sum(x["gates"] for x in out["attention"])
    assert a["totals"]["circuit"]["gates"] == out["totals"]["gates"]


#: each head Definition `ATTENTION` names at shapes it binds at: one block of keys and several, a split head's four, and a
#: backward's key blocks one and two of its query blocks
HEAD_SHAPES = [(FT.Fa2TrainHead, {"T": 32, "D": 32, "BN": 16}), (FT.Fa2TrainHead, {"T": 48, "D": 64, "BN": 16}),
               (FT.Fa2TrainSplitHead, {"T": 64, "D": 32, "BN": 16}),
               (FTB.Fa2TrainBwdHead, {"T": 32, "D": 32, "BM": 16, "BN": 16, "NS": 1}),
               (FTB.Fa2TrainBwdHead, {"T": 64, "D": 32, "BM": 16, "BN": 32, "NS": 3})]


@pytest.mark.parametrize(("head", "b"), HEAD_SHAPES, ids=lambda x: getattr(x, "name", None) or "-".join(map(str, x.values())))
def test_the_attention_table_is_its_heads_census(head, b) -> None:
    """The multiply-adds `ATTENTION` gives a head's products are 16 for each k16 step of the tensor-core dot in the head's
    census, and those of its products whose k is D (the scores, and the backward's dP) D for each `DotBf16_v2` dot, with one
    output a dot: the grader's table can't drift from the Definitions it reads."""
    fn = bind(head, **b)
    classes = census(Program(fn))["classes"]
    products = G.HEADS[fn.base_id].products
    assert sum(pr.macs(b) for pr in products) == G.MAC_CELLS[fn.base_id](b)[0] == 16 * classes[HopperBF16WgmmaDot16.id]["instances"]
    dots = classes[descriptor_id(bind(DotBf16, K=b["D"], DOT=HopperBF16WgmmaDot16))]["instances"]
    by_dot = [pr for pr in products if pr.mkn[1] == "D"]
    assert (sum(pr.macs(b) for pr in by_dot), sum(pr.outputs(b) for pr in by_dot)) == (b["D"] * dots, dots)


def test_every_head_is_held_to_its_census() -> None:
    assert {bind(head, **b).base_id for head, b in HEAD_SHAPES} == set(G.HEADS)


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
