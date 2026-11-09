"""`circuit_check.structural_slots` (``circuit-check --structural DIR TARGET``): a split's nets, composed by its wires through
its lookup slots with each slot word uninterpreted, are accepted as the Definition's flat lowering by construction, and a
split whose wires compose anything else is refused by name."""
from __future__ import annotations

import json

import pytest
from circuit_check import cli
from circuit_check import structural as ST
from circuit_check import structural_slots as SSC
from circuit_check import targets as TG
from verity_flock import struct_slots as SS

from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.evaluation.reference import resolve
from verity.ml.circuits.ops.fp32 import F32AddFtz
from verity.ml.circuits.ops.mufu import MufuEx2Ftz, MufuRcpFtz
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array, Value

F32 = Value(32)


def _chain(B, _S, x, y):
    r = B.call(MufuRcpFtz, B.call(F32AddFtz, B.call(MufuEx2Ftz, x), y))
    return array_of([r, B.call(F32AddFtz, r, x), B.call(MufuEx2Ftz, y)])


CHAIN = CompositeDefinition("StructuralSlotsChain", 1, (), lambda _S: ((("x", F32), ("y", F32)), Array(3, F32)), _chain,
                            register=False)


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    d = tmp_path_factory.mktemp("split")
    SS.write(SS.lower(CHAIN)[0], d)
    return d


def _forged(split, tmp_path, edit):
    d = tmp_path / "forged"
    d.mkdir()
    for f in split.iterdir():
        (d / f.name).write_text(f.read_text())
    m = json.loads((d / "slots.json").read_text())
    edit(m)
    (d / "slots.json").write_text(json.dumps(m))
    return SSC.check(CHAIN, d)


def _wire(m, dst, port):
    return next(w for w in m["wires"] if w[0] == dst and w[1] == port[0] and w[2] == port[1])


def test_a_split_is_its_definitions_lowering_by_construction(split):
    """Three nets and three slot words (two ex2, one rcp of an ex2's value): each slot word is one uninterpreted node per
    table and index on both sides, so the forms agree without reading a table."""
    r = ST.check_file(CHAIN, split)
    rec = r["structural"]
    assert (rec["verdict"], r["failures"], rec["nets"], rec["lowering"]["slot_reads"]) == ("syntactic", [], 3, 3)
    assert rec["template"]["slot_words"] == 3 and rec["error_bound"]["log2"] < -400


def test_the_command_holds_a_registered_definitions_split(tmp_path):
    """``--structural DIR TARGET`` on a served Definition at small statics: HeadEnd's one rcp read in a slot."""
    spec = "SparseMlaHeadEnd_v1{DV=16,INV=Fa3InvSum_v1}"
    SS.write(SS.lower(resolve(TG.definition(spec)))[0], tmp_path / "s")
    out = tmp_path / "r.json"
    assert cli.main(["--structural", str(tmp_path / "s"), "--out", str(out), spec]) == 0
    rec = json.loads(out.read_text())["targets"][0]["structural"]
    assert (rec["verdict"], rec["nets"], rec["lookups"]) == ("syntactic", 2, 1)


@pytest.mark.parametrize(("edit", "code"), [
    (lambda m: m["wires"].pop(), "wires"),
    (lambda m: m["wires"].append(list(m["wires"][0])), "wires"),
    (lambda m: m["wires"].append(["unit", 0, 0, "unit", 0, 0]), "wires"),
    (lambda m: _wire(m, "lookup:rcp", (0, 0)).__setitem__(3, "stage1"), "wires"),
    (lambda m: _wire(m, "stage0", (0, 0)).__setitem__(5, 1), "wires"),
    (lambda m: m["outs"][-1].__setitem__(0, ["to", 9]), "ports"),
], ids=["a port no wire fills", "a port wired twice", "a wire into the unit", "an index from a later net",
        "a width that disagrees", "no outputs"])
def test_a_split_its_wires_dont_compose_is_refused_by_name(split, tmp_path, edit, code):
    r = _forged(split, tmp_path, edit)
    assert r["structural"]["verdict"] == "refused" and [f["code"] for f in r["failures"]] == [code]


def test_a_split_reading_the_wrong_slot_is_refused_on_the_tables(split, tmp_path):
    """The two ex2 slots' indices swapped: each word is still a read of a well-formed index, so only the semantic pass,
    which reads the tables, tells the outputs apart."""
    def swap(m):
        a, b = _wire(m, "lookup:ex2", (0, 0)), _wire(m, "lookup:ex2", (1, 0))
        a[3:], b[3:] = b[3:], a[3:]

    r = _forged(split, tmp_path, swap)
    assert r["structural"]["verdict"] == "refused" and [f["code"] for f in r["failures"]] == ["mismatch"]


def test_a_directory_that_is_no_split_is_refused(tmp_path):
    r = SSC.check(CHAIN, tmp_path)
    assert r["structural"]["verdict"] == "refused" and [f["code"] for f in r["failures"]] == ["malformed"]
