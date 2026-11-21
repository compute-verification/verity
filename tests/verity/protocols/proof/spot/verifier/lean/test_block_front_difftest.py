"""The front of a block's circuit (`Spot.BlockFront`, the ``block_front`` piece): the block's gate ranges, its input and
output ports and its lift to bits, held on the one-circuit chain's blocks to a reference that walks every gate of C and
computes Core's ``Circuit.into`` and ``Circuit.exits`` as they are defined (the gates outside S that a gate of S reads; the
gates of S that a gate outside S reads or that are C's outputs), and lists the word primitives the block's Definitions
reach. Lean's front reads the descriptor and the block's boundary only, never the block's interior: that the two agree is
what the ports' closed form owes Core's definitions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from functools import cache
from pathlib import Path

import pytest

import verity_difftest as D
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks
from verity.circuits.evaluation.reference import signature_of
from verity.circuits.layout import resolve

HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[7] / "verity" / "protocols" / "proof" / "spot" / "verifier" / "lean"
PIECES = "spot-difftest"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@cache
def chain_module():
    path = HERE.parents[1] / "test_one_circuit_chain.py"
    spec = importlib.util.spec_from_file_location("one_circuit_chain_for_block_front", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@cache
def chain():
    """The one-circuit chain's wired Program (`test_one_circuit_chain.circuit`) and its descriptor's text."""
    program, _query = chain_module().circuit().program()
    return program, b"".join(descriptor_chunks(program)).decode()


def unit_ranges(program) -> list[tuple[int, int]]:
    """Each unit's gates ``[lo, hi)``: the root's Calls in node order, a call one unit, a batch one a member."""
    body, out = program.root.body, []
    for k, fn in PO.calls(program):
        node = body.nodes[k]
        if node.form not in ("call", "batch"):
            raise D.Reject(f"root node {k} is a {node.form}")
        for m in range(node.n if node.form == "batch" else 1):
            out.append((body.offsets[k] + m * fn.gates, body.offsets[k] + (m + 1) * fn.gates))
    return out


def word_primitives(program, ks: list[int]) -> list[str]:
    """The primitives the root nodes ``ks``' Definitions reach that are neither inputs nor one-bit gates."""
    seen, out, todo = set(), set(), [program.root.body.nodes[k].fn for k in ks]
    while todo:
        fn = todo.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        if hasattr(fn, "body"):
            todo.extend(n.fn for n in fn.body.nodes)
            continue
        params, _ = signature_of(fn)
        bit = fn.ret.w == 1 and all(w == 1 for p in params for w in p.leaf_widths())
        if not (PO.is_input(fn) or bit):
            out.add(fn.id)
    return sorted(out)


def reference(j) -> dict:
    program, text = chain()
    if j.get("descriptor") != text:
        raise D.Reject("not the chain's descriptor")
    units, boolean = j["units"], json.loads(text)["boolean"]
    if not units or len(set(units)) != len(units):
        raise D.Reject("a block names each of its units, at least one, once")
    ranges = unit_ranges(program)
    if any(u >= len(ranges) for u in units):
        raise D.Reject("a unit past C's units")
    S = sorted(ranges[u] for u in units)
    inside = lambda g: any(lo <= g < hi for lo, hi in S)  # noqa: E731
    lay = program.circuit
    gates = list(lay.gates())
    width = {g.index: g.prim.ret.w for g in gates}
    outputs = {resolve(lay.root_scope, r) for r in program.root.body.ret}
    read_outside = {o for g in gates if not inside(g.index) for o in g.operands}
    into = sorted({o for g in gates if inside(g.index) for o in g.operands if not inside(o)})
    exits = sorted(g.index for g in gates if inside(g.index) and (g.index in outputs or g.index in read_outside))
    calls = [k for k, _ in PO.calls(program)]
    members = [k for k in calls for _ in range(program.root.body.nodes[k].n if program.root.body.nodes[k].form == "batch" else 1)]
    words = word_primitives(program, sorted({members[u] for u in units}))
    return {"ranges": [list(r) for r in S], "ins": [[g, width[g]] for g in into], "outs": [[g, width[g]] for g in exits],
            "lift": [[w, boolean[w]] for w in words]}


FRONT = D.Port(PIECES, "block_front", reference, None, package=PACKAGE, of_record="lean")
pieces = D.server_fixture(FRONT)

def blocks() -> list[list[int]]:
    """The chain's eight units, each a block; its four members of two (the window's cap); and the whole chain."""
    return [[u] for u in range(8)] + [[2 * m, 2 * m + 1] for m in range(4)] + [list(range(8))]


@pytest.mark.parametrize("units", blocks(), ids=lambda us: "units-" + "-".join(map(str, us)))
def test_the_chain_block_fronts_agree(pieces, units):
    _program, text = chain()
    j = {"descriptor": text, "units": units}
    want = reference(j)
    assert want["ins"] and want["outs"], want
    assert pieces.call("block_front", j) == ("output", D.canonical(want))


@pytest.mark.parametrize("name, change", [
    ("no unit", lambda j: {**j, "units": []}),
    ("a unit twice", lambda j: {**j, "units": [3, 3]}),
    ("a unit past C's", lambda j: {**j, "units": [8]}),
], ids=lambda x: x if isinstance(x, str) else "")
def test_the_front_refuses(pieces, name, change):
    _program, text = chain()
    j = change({"descriptor": text, "units": [0]})
    with pytest.raises(D.Reject):
        reference(j)
    assert pieces.call("block_front", j)[0] == "reject", name


def test_the_order_is_increasing_c_index_and_the_inputs_precede_every_unit(pieces):
    """MERGE's `Partition.pos` and the file's numbering agree on the chain: every input gate precedes every unit's."""
    program, text = chain()
    first = min(lo for lo, _ in unit_ranges(program))
    out = json.loads(pieces.call("block_front", {"descriptor": text, "units": [5, 2]})[1])
    gates = [g for g, _ in out["ins"]]
    assert gates == sorted(gates)
    assert any(g < first for g in gates) and out["ranges"] == sorted(out["ranges"])


#: circuits' vector: the chain's first block (units 0 and 1), in(S) and out(S) as C's word-level descriptor gates
FIRST_IN, FIRST_OUT = list(range(256)) + list(range(512, 544)), [642 + 3 * k for k in range(32)]


@cache
def restricted(member: int) -> R.Block:
    """`restrict`'s block of member ``member`` (units ``2m`` and ``2m + 1``) on the chain."""
    _program, text = chain()
    c = text.encode()
    query = {"name": "Q_template_instances", "version": 0, "params": {"templates": [chain_module().TI.GID]}}
    return R.restrict(c, {"query": query}, [(2 * member, 2 * member + 2)], R.Bits(R.program(c)))


def writer(program, g: int) -> tuple[int, int]:
    """Gate ``g``'s root node and, for a node of a composite, its member."""
    body = program.root.body
    k = next(k for k in range(len(body.nodes)) if body.offsets[k] <= g < body.offsets[k + 1])
    fn = body.nodes[k].fn
    return k, (g - body.offsets[k]) // fn.gates if hasattr(fn, "body") else 0


def rows(program, ports) -> list:
    """``ports`` cut where the writer changes: ``[g<first gate>, bits, gates]``."""
    out, last = [], None
    for g, w in ports:
        if writer(program, g) == last:
            out[-1][1] += w
            out[-1][2].append(g)
        else:
            out.append([f"g{g}", w, [g]])
        last = writer(program, g)
    return out


@pytest.mark.parametrize("member", range(4))
def test_the_block_files_ports_are_restricts_and_it_reads_back(pieces, member):
    program, text = chain()
    b = restricted(member)
    j = {"descriptor": text, "units": [2 * member, 2 * member + 1], "partition": "a" * 128}
    kind, body = pieces.call("block_file", j)
    assert kind == "output", body
    out = json.loads(body)
    assert [g for g, _ in out["ins"]] == list(b.inputs) and [g for g, _ in out["outs"]] == list(b.outputs)
    if member == 0:
        assert list(b.inputs) == FIRST_IN and list(b.outputs) == FIRST_OUT
    assert out["rows"] == rows(program, out["ins"]) and out["out_rows"] == rows(program, out["outs"])
    meta, words = out["meta"], [[n, (bits + 15) // 16] for n, bits, _ in out["out_rows"]]
    assert meta["statement"] == "verity/flock-circuit" and meta["input_ports"] == len(out["rows"])
    assert [p[0] for p in meta["ports"]] == [r[0] for r in out["rows"]] + [n for n, _ in words]
    assert meta["out_ports"] == words
    assert [s[1] for s in meta["row_schemas"][:len(out["rows"])]] == [r[1] for r in out["rows"]]
    assert meta["program_sha512"] == hashlib.sha512(text.encode()).hexdigest()
    assert len(meta["dummy"]["outputs"]) == sum(w for _, w in words)
