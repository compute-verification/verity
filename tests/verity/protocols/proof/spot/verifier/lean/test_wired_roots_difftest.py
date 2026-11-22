"""Each member's wired rows' roots at their writers' output field trees, and the reads its registered file may open at
them, both from C (`Spot.Pieces.wiredRoots`: `OneStageZk.fieldRoots`, `OneStageZk.wiredRows`), held to the Python
mirror (`members._wired_roots`, `members._wired_rows`) on a writer of many members read by a plain wire (the periods
tests' `chain`) and by a wire cut in two (their `sliced`): every cut row ``x#j`` gets its writer's tree, once."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.circuits import partition_object as PO
from verity.protocols.proof.spot.verifier.one_stage import members as M, partition as P

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_difftest import Setup

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proof" / "spot" / "verifier" / "lean")
PIECES = "spot-difftest"
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
_TI = importlib.util.spec_from_file_location(
    "periods_test_interface_wired", Path(__file__).resolve().parents[7] / "tests/verity/commitments/periods/test_interface.py")
TI = importlib.util.module_from_spec(_TI)
_TI.loader.exec_module(TI)


def root(tree: str, leaves: int) -> dict:
    return {"port": tree, "root": hashlib.sha512(tree.encode()).hexdigest(),
            "domain": hashlib.sha512(b"d" + tree.encode()).hexdigest(), "leaves": leaves}


CHAIN, SLICED = Setup(*TI.chain().program()), Setup(*TI.sliced().program())
SETUPS = {s.sha512: s for s in (CHAIN, SLICED)}
#: the chain's four members of two units each; unit u >= 2 reads unit u - 2's ``o0`` at ``x``
CHAIN_MS = [[f"{TI.GID}/c{k}", [[2 * k, 2 * k + 2]]] for k in range(4)]
#: three ``G8`` slices' members and two of the ``Gemm`` reading its ``x`` cut in two, a row from each slice
SLICED_MS = [[f"{TI.G8ID}/c{k}", [[2 * k, 2 * k + 2]]] for k in range(3)] + [[f"{TI.GID}/c0", [[6, 8]]],
                                                                             [f"{TI.GID}/c1", [[8, 9]]]]


def case(s: Setup, ms: list, roots: list) -> dict:
    return {"program": s.desc, "query": s.query, "members": ms, "roots": roots}


def reference(j):
    s = SETUPS.get(PO.program_sha512(j["program"])) if isinstance(j.get("program"), dict) else None
    if s is None:
        raise RuntimeError("the strategy drew a program this file has no setup for")
    ms = {n: {"units": us, "registered": {}} for n, us in j["members"]}
    try:
        ranges = P._template(s.program, j["query"], "the members' partition").ranges
        rec = {"roots": j["roots"]}
        roots = M._wired_roots(s.desc, ranges, ms, M._field_trees(M._templates(ms), rec), {})
    except (M.Outside, M._Refused, ValueError, P.QueryRefused) as e:
        raise D.Reject(str(e)) from None
    rows = M._wired_rows(s.program, j["query"], s.desc, ms)
    return {"roots": [[n, [[r["port"], r["tree"], r["root"]] for r in roots.get(n, [])]] for n in ms],
            "rows": [[n, [list(x) for x in sorted(rows.get(n, ()))]] for n in ms]}


CASES = [case(CHAIN, CHAIN_MS, [root(f"{TI.GID}/o0", 8)]),
         case(CHAIN, CHAIN_MS, []),
         case(CHAIN, CHAIN_MS[1:3], [root(f"{TI.GID}/o0", 8)]),
         case(SLICED, SLICED_MS, [root(f"{TI.G8ID}/o0", 6), root(f"{TI.GID}/o0", 3)]),
         case(SLICED, SLICED_MS, [root(f"{TI.GID}/o0", 3)])]
WIRED_ROOTS = D.Port(PIECES, "wired_roots", reference, st.sampled_from(CASES), package=PACKAGE, of_record="lean",
                     need_rejects=False)

pieces = D.server_fixture(WIRED_ROOTS)


def test_a_writer_of_many_members_wires_each_reader_at_its_one_field_tree():
    t, ref = f"{TI.GID}/o0", root(f"{TI.GID}/o0", 8)["root"]
    got = reference(CASES[0])
    assert got["roots"] == [[CHAIN_MS[0][0], []]] + [[n, [["x", t, ref]]] for n, _ in CHAIN_MS[1:]]
    assert got["rows"] == [[CHAIN_MS[0][0], []]] + [[n, [["x", t]]] for n, _ in CHAIN_MS[1:]]
    assert reference(CASES[1])["roots"] == [[n, []] for n, _ in CHAIN_MS]


def test_each_cut_row_of_a_reader_opens_its_writers_tree():
    t, ref = f"{TI.G8ID}/o0", root(f"{TI.G8ID}/o0", 6)["root"]
    got = reference(CASES[3])
    want = [[n, []] for n, _ in SLICED_MS[:3]] + [[n, [["x#0", t, ref], ["x#1", t, ref]]] for n, _ in SLICED_MS[3:]]
    assert got["roots"] == want
    assert got["rows"] == [[n, []] for n, _ in SLICED_MS[:3]] + [[n, [["x#0", t], ["x#1", t]]] for n, _ in SLICED_MS[3:]]


@needs_lake
@pytest.mark.parametrize("k", range(len(CASES)))
def test_the_wired_roots_piece_agrees_on_each_circuit(k, pieces):
    assert pieces.call("wired_roots", CASES[k]) == ("output", D.canonical(reference(CASES[k])))
