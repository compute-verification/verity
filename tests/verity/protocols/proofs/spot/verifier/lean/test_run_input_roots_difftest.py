"""A member's ports at C's run input nodes' trees on drawn programs: `members._run_input_roots` held to
`OneStageZk.runInputRootsAt` (`Spot.HeaderHeldAt.runInputPiece`), unit by unit. A drawn program is a population of
one to three templates, its parameters each a run input node, under a record holding a drawn subset of the nodes'
trees ``input/n<w>``; Lean's side reads each node the way the door's `OneCircuit.pos` does (`regRow`), which
`Spot.HeaderHeldAt`'s guards hold."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.circuits import codec, population as PN
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.types import Array, Value
from verity.protocols.proofs.spot.verifier.one_stage import members as M, partition as P

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "lean")
PIECES = "spot-difftest"
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

XOR = PrimitiveDefinition("RunInputXor16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: a ^ b,
                          register=False)
PAIR = bind(CompositeDefinition("RunInputPair", 0, (), lambda S: ((("x", Array(4, Value(16))),), Value(16)),
                                lambda B, S, x: B.call(XOR, x[0], x[1]), register=False))
TWO = bind(CompositeDefinition("RunInputTwo", 0, (), lambda S: ((("x", Array(2, Value(16))), ("y", Value(16))),
                                                                 Value(16)),
                               lambda B, S, x, y: B.call(XOR, B.call(XOR, x[0], x[1]), y), register=False))
TEMPLATES = (PAIR, TWO)
#: a drawn program's text, by its text, with its Program and query
PROGRAMS: dict[str, tuple] = {}


def program_case(members, keep) -> dict:
    """A piece's input: the population of ``members`` (template index, count), every unit, and the record's roots of
    the run input nodes ``keep`` picks."""
    seen, ms = set(), []
    for i, n in members:
        if i not in seen:
            seen.add(i)
            ms.append((TEMPLATES[i], n))
    program = PN.mixed_population_program(ms)
    query = P.template_instances_query([t.id for t, _ in ms])
    text = codec.canonical_json(codec.encode_program(program)).decode()
    PROGRAMS[text] = (program, query)
    nodes = len(program.root.body.nodes)
    roots = [{"domain": f"d{w}", "leaves": 4, "port": f"input/n{w}", "root": f"r{w}"} for w in range(nodes)
             if keep >> w & 1]
    return {"program": text, "units": list(range(sum(n for _, n in ms))), "roots": roots}


def reference(j):
    program, query = PROGRAMS[j["program"]]
    out = []
    for u in j["units"]:
        ms = {"m": {"units": [[u, u + 1]]}}
        try:
            got = M._run_input_roots(program, query, ms, {"roots": j["roots"]}, {"m": []})["m"]
        except M._Refused as e:
            out.append(str(e))
            continue
        out.append([[r["port"], r["tree"]] for r in got])
    return out


INPUTS = st.builds(program_case,
                   st.lists(st.tuples(st.integers(0, len(TEMPLATES) - 1), st.integers(1, 3)), min_size=1, max_size=3),
                   st.integers(0, 255))
ROOTS = D.Port(PIECES, "run_input_roots", reference, INPUTS, package=PACKAGE, of_record="lean", need_rejects=False)

pieces = D.server_fixture(ROOTS)


@needs_lake
def test_a_two_template_population_reads_each_port_at_its_own_nodes_tree(pieces):
    j = program_case([(0, 2), (1, 1)], 255)
    want = reference(j)
    assert want[0] == [["x", "input/n0"]] and want[2] == [["x", "input/n1"], ["y", "input/n2"]], want
    assert pieces.call("run_input_roots", j) == ("output", D.canonical(want))


@needs_lake
def test_the_run_input_roots_piece_agrees(pieces):
    stats = D.run(ROOTS, examples=200, server=pieces)
    assert stats.outputs > 50, stats
