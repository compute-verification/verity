"""Round 2's wired key (`SecondStage.keyFailure`, `openedRow`) held to its Python (`second_stage._key_failure`) through
the ``second_statements`` piece of two executables: `sampled-proofs-difftest`, the build of record's, whose switch is
off (`SecondStage.WIRED_KEY`), and `sampled-proofs-wired-key-dev`, the development executable with it on. One MoE
layer's read of its expert's row, over the two-template interior (`SECOND_MIXED`): its router, PAIR, writes each
token's expert index at `out`, and the read, TRIPLE, a read of the table round 1's record registers, takes its key
`tok` from it, unit for unit. And the verifier of record (`sampled-proofs-verifier`) serves no piece that takes the
switch from its input.
"""

from __future__ import annotations

import copy
import dataclasses
import shutil

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.protocols.proofs.spot.verifier import second_stage as SS

from test_difftest import PACKAGE, PIECES, SECOND_MIXED, _as_stated, _second_case, _second_own

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

DEV = "sampled-proofs-wired-key-dev"
VERIFIER = "sampled-proofs-verifier"
PAIR, TRIPLE = SECOND_MIXED.templates
SAMPLE, DRAWN = [0, 2], [1, 4, 6, 9]
#: The read's table in round 1's record (`t`) of `V` rows, its root and domain id, which the verifier fills as bytes.
V, T_ROOT, T_DOMAIN = 8, "c1" * 64, "c2" * 64
READ_VALUES = ("root", "domain", "row", "salt", "path", "one")


def reference(on: bool):
    """`second_statements`' Python with the wired key's switch `on` (the development executable's) or as the code holds
    it (the build of record's)."""
    def check(j):
        allow, own, statements = _second_own(j)
        return SS.check(allow, dataclasses.replace(own, wired_key=True) if on else own, statements).to_json()
    return check


OF_RECORD = D.Port(PIECES, "second_statements", reference(False), st.nothing(), package=PACKAGE)
ON = D.Port(DEV, "second_statements", reference(True), st.nothing(), package=PACKAGE)
VERIFIER_PORT = D.Port(VERIFIER, "second_statements", reference(False), st.nothing(), package=PACKAGE)

of_record = D.server_fixture(OF_RECORD)
dev = D.server_fixture(ON)
verifier = D.server_fixture(VERIFIER_PORT)


def _moe(statements: bool = False) -> dict:
    """The MoE layer's run as round 2 reads it: the interior registers the read's opened row with its salt and path,
    the table's root and domain id and the verdict's value, all but the row, salt and path the verifier's own; the
    design links the read's ports to them, wires its key to the router's `out` (each of its two units a block its
    router unit's row, of the router's three), and derives the rows of two words both sides; the read's reference
    statement reads its verdict at the verifier's 1. Without `statements`, round 2 reads none (``second-statements``)."""
    c = _second_case(SECOND_MIXED, True, SAMPLE, DRAWN, allow=True)
    entry = {"root": "dd" * 64, "domain": "de" * 64, "leaves": 8}
    c["interior"] = {**c["interior"], "registered": [{"value": v, **entry} for v in READ_VALUES]}
    c["boundary"] = {**c["boundary"], "roots": [*c["boundary"]["roots"], {"port": "t", "root": T_ROOT,
                                                                          "domain": T_DOMAIN, "leaves": V}]}
    c["links"] = [{"template": TRIPLE, "port": p, "value": p} for p in ("root", "domain", "row", "salt", "path")]
    c["wires"] = [{"template": TRIPLE, "port": "tok", "writer": PAIR, "wport": "out"}]
    c["rows"] = [*({"template": TRIPLE, "port": p, "per": 1, "runs": [[0, 2, 0, 0]]}
                   for p in ("root", "domain", "row", "salt", "path")),
                 {"template": TRIPLE, "port": "tok", "per": 3, "runs": [[0, 2, 0, 1]]}]
    c["widths"] = [{"template": TRIPLE, "port": "tok", "reads": 2, "writes": 2}]
    c["supplied"] = {"root": [[{"bytes": T_ROOT}]], "domain": [[{"bytes": T_DOMAIN}]], "one": [[{"bytes": "00000001"}]]}
    c["reads"] = {TRIPLE: V}
    c["ref_reads"] = {**c["ref_reads"], TRIPLE: {"ports": {"o1": {"value": "one"}}}}
    if not statements:
        c["statements"] = None
    return c


def _with(c: dict, **fields) -> dict:
    return {**copy.deepcopy(c), **fields}


MOE = _moe()
WIRE, KEY_ROWS = MOE["wires"][0], MOE["rows"][-1]
#: Each wired key round 2 refuses with the switch on, with the reason it gives.
REFUSED = [
    ("two wires", _with(MOE, wires=[WIRE, {**WIRE, "wport": "x"}]), SS.KEY_WIRES_WHY),
    ("linked too", _with(MOE, links=[*MOE["links"], {"template": TRIPLE, "port": "tok", "value": "root"}]),
     SS.KEY_LINKED_WHY),
    ("other words", _with(MOE, widths=[{**MOE["widths"][0], "writes": 1}]), SS.KEY_WIDTH_WHY),
    ("no width", _with(MOE, widths=None), SS.KEY_WIDTH_WHY),
    ("writer unit out of bounds", _with(MOE, rows=[*MOE["rows"][:-1], {**KEY_ROWS, "runs": [[0, 2, 2, 1]]}]),
     SS.KEY_BOUNDS_WHY),
]


def _agree(port, server, case):
    got = port.reference(copy.deepcopy(case))
    assert D.replay(port, case, server) == ("output", D.canonical(got)), got
    return got


def test_the_moe_layers_wired_key_passes_only_with_the_switch_on(of_record, dev):
    """The honest MoE layer: with the switch on its row, salt and path pass as a read's opening at a key one wire fills
    whole, and round 2 asks only for the statements; in the build of record its row is refused, as before the switch,
    with `KEY_WHY`."""
    on = _agree(ON, dev, MOE)
    assert on["codes"] == ["second-statements"], on
    off = _agree(OF_RECORD, of_record, MOE)
    assert off["codes"] == ["second-inputs", "second-statements"], off
    assert off["detail"]["second-inputs"] == {"value": "row", "why": SS.KEY_WHY}


@pytest.mark.parametrize("name,case,why", REFUSED, ids=[n for n, _c, _w in REFUSED])
def test_each_wired_key_that_is_not_whole_is_refused_with_its_reason(of_record, dev, name, case, why):
    """With the switch on, a wired key with two wires, linked too, wired to a row of other words or of no width, or
    whose rows give a unit a writer row past the router's three, has its row refused (`second-inputs`) with its own
    reason; in the build of record each is refused with `KEY_WHY`."""
    on = _agree(ON, dev, case)
    assert on["detail"]["second-inputs"] == {"value": "row", "why": why}, (name, on)
    off = _agree(OF_RECORD, of_record, case)
    assert off["detail"]["second-inputs"] == {"value": "row", "why": SS.KEY_WHY}, (name, off)


def test_no_input_turns_the_switch_on(of_record):
    """The build of record reads the switch from no input: the MoE layer with `wired_key` or `wiredKey` set in it is
    refused as without them."""
    for extra in ({"wired_key": True}, {"wiredKey": True}, {"wired_key": True, "wiredKey": True, "allow": True}):
        got = _agree(OF_RECORD, of_record, {**MOE, **extra})
        assert got["detail"]["second-inputs"] == {"value": "row", "why": SS.KEY_WHY}, (extra, got)


def test_the_verifier_of_record_serves_no_piece_that_takes_the_switch(verifier):
    """The verifier of record's executable answers no ``second_statements`` request, whatever it carries: round 2 runs
    in it only inside the stream's run (`Run.run`), whose `Own` holds the switch as the build does."""
    for extra in ({}, {"wired_key": True}, {"wiredKey": True}):
        with pytest.raises(D.ProtocolError, match="unknown fn: second_statements"):
            verifier.call("second_statements", {**MOE, **extra})


def test_a_sampled_unit_whose_key_is_not_its_writers_row_is_refused(dev):
    """The wired key's read in a statement (`second-read`, `second_stage._read_ok`'s wired arm, which the switch doesn't
    touch): TRIPLE's session reads `tok` at its writer units' committed rows, each slot its writer's first unit in its
    cell, and passes; a prover that opens another row, another unit's or another cell's, is refused ``second-read``.
    This run registers no opened row, so the key clause (`openedRow`) is not read here: the MoE layer above is that
    clause's run."""
    c = _key_statements()
    s1 = c["statements"][1]
    tok = s1["registered"]["ports"]["tok"]
    honest = _agree(ON, dev, c)
    assert honest["codes"] == [], honest
    for reads in ([1, 1, 4, 4], [3, 3, 0, 0]):
        case = _as_stated({**c, "statements": [c["statements"][0],
                                               {**s1, "registered": {**s1["registered"], "ports": {
                                                   "tok": {**tok, "reads": reads}}}}]})
        got = _agree(ON, dev, case)
        assert "second-read" in got["codes"], (reads, got)


def _key_statements() -> dict:
    """The two-template interior with TRIPLE's key `tok` wired to PAIR's `out` and read in its statement, each slot at
    its writer's first unit in its cell: `_second_case`'s wired run with TRIPLE's port named `tok`."""
    c = _second_case(SECOND_MIXED, True, SAMPLE, DRAWN, wires=[(TRIPLE, PAIR)], allow=True)

    def renamed(x):
        if isinstance(x, dict):
            return {("tok" if k == "x" else k): renamed(v) for k, v in x.items()}
        if isinstance(x, list):
            return [renamed(v) for v in x]
        return x
    c["statements"] = [c["statements"][0], renamed(c["statements"][1])]
    c["refs"] = {**c["refs"], TRIPLE: renamed(c["refs"][TRIPLE])}
    c["ref_reads"] = {s["template"]: s["registered"] for s in c["statements"]}
    c["wires"] = [{**w, "port": "tok"} for w in c["wires"]]
    c["rows"] = [{**r, "port": "tok"} if r["template"] == TRIPLE else r for r in c["rows"]]
    lean = c["verdicts"]["lean"]
    lean = {**lean, "registered": {(f"{TRIPLE}/tok" if k == f"{TRIPLE}/x" else k): v
                                   for k, v in lean.get("registered", {}).items()}}
    c["widths"] = [{"template": TRIPLE, "port": "tok", "reads": 2, "writes": 2}]
    return _as_stated({**c, "verdicts": {**c["verdicts"], "lean": lean}})
