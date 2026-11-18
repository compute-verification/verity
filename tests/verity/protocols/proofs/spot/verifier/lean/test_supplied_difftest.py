"""A program's supplied inputs (`Spot.Supplied.piece`, `supplied_marks`): the marks of its top-level `inputs`, each
marked input's nodes all `SuppliedInput<w>_v1`, its function one of the registry's with a body, its value under the
verifier's coins read by its layout, held to the Python mirror (`one_stage.registered.marks`, `of_mark`); and the
registry's closed list in Lean equal to the format's (`verity.circuits.supplied.FUNCTIONS`)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.circuits import supplied as SU
from verity.protocols.proofs.spot.verifier.one_stage import registered as G

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "lean")
PIECES = "spot-difftest"
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

SALT = {"supplied": "pous.setup.salt/v1", "args": {}, "layout": "le", "bits": 192}
ONE = {"supplied": "const.u16/v1", "args": {"value": 1}, "layout": "le", "bits": 16}
CALLEES = ["Input16_v1", "RegisteredInput16_v1", "SuppliedInput16_v1", "SuppliedInput1_v1"]


def desc(inputs, fns) -> dict:
    """A descriptor whose root leads with one node a callee index of ``fns`` and whose top-level `inputs` is ``inputs``."""
    return {"root": "R", "callees": CALLEES, "inputs": inputs,
            "definitions": {"R": {"body": {"nodes": [{"fn": f} for f in fns]}}}}


def reference(j):
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    try:
        ms = G.marks(j.get("program"))
        coins = j.get("coins")
        if not isinstance(coins, dict) or not all(isinstance(h, str) and re.fullmatch(r"(?:[0-9a-f]{2})*", h)
                                                  for h in coins.values()):
            raise ValueError("coins")
        widths = j.get("widths")
        if not isinstance(widths, dict):
            raise ValueError("widths")
        out = {}
        for v, m in ms.items():
            w = G._nat(widths.get(v))
            if w is None:
                raise ValueError("width")
            rows = G.of_mark({f: bytes.fromhex(h) for f, h in coins.items()}, v, m)
            out[v] = [G.le_leaves(bytes.fromhex(h), w) if k == G.LE else
                      G.digest_leaves(bytes.fromhex(h)) if m["layout"] == "digest" else G.leaves_of(bytes.fromhex(h), w)
                      for r in rows for k, h in r]
        return out
    except ValueError as e:
        raise D.Reject(str(e)) from e


JUNK = st.one_of(st.none(), st.booleans(), st.text(max_size=3), st.integers(-3, 3), st.lists(st.integers(-1, 3), max_size=2))
MARKS = st.one_of(st.just(SALT), st.just(ONE), st.builds(lambda v: {**ONE, "args": {"value": v}}, st.integers(0, 70000)),
                  st.builds(lambda f, lay, b: {"supplied": f, "args": {}, "layout": lay, "bits": b},
                            st.sampled_from([*SU.FUNCTIONS, "no.such/v1"]), st.sampled_from([*SU.LAYOUTS, "be"]),
                            st.sampled_from([0, 16, 192, 1 << 16, (1 << 16) + 8])))


@st.composite
def instances(draw) -> dict:
    """One to three inputs, each run, registered or supplied with a drawn mark, its nodes perhaps of the wrong kind; a
    coin perhaps missing, short or present; widths perhaps missing."""
    inputs, fns = [], []
    for i in range(draw(st.integers(1, 3))):
        n = draw(st.integers(0, 2))
        kind = draw(st.sampled_from(["run", "registered", "supplied", "supplied"]))
        fn = {"run": 0, "registered": 1, "supplied": draw(st.sampled_from([2, 2, 3]))}[kind]
        if draw(st.integers(0, 9)) == 0:
            fn = draw(st.sampled_from([0, 1, 2, 7, -1]))
        fns += [fn] * n
        inputs.append([f"x{i}", n, draw(MARKS)] if kind == "supplied" else [f"x{i}", n])
    if draw(st.integers(0, 9)) == 0:
        inputs.append(draw(JUNK))
    coins = draw(st.sampled_from([{}, {"pous.setup.salt/v1": "ab" * 24}, {"pous.setup.salt/v1": "ab" * 23},
                                  {"pous.setup.salt/v1": 5}]))
    widths = {x[0]: draw(st.sampled_from([16, 16, 1, 3, -1])) for x in inputs if isinstance(x, list) and x
              and isinstance(x[0], str)}
    if draw(st.integers(0, 9)) == 0:
        widths = draw(JUNK)
    return {"program": desc(inputs, fns), "coins": coins, "widths": widths}


INPUTS = st.one_of(*[instances() for _ in range(6)], JUNK)
MARKED = D.Port(PIECES, "supplied_marks", reference, INPUTS, package=PACKAGE, of_record="lean")

pieces = D.server_fixture(MARKED)


@needs_lake
def test_the_supplied_marks_piece_agrees(pieces):
    stats = D.run(MARKED, examples=1500, server=pieces)
    assert stats.outputs > 100 and stats.rejects > 0, stats


SETUP = {"program": desc([["store", 1], ["salt", 1, SALT], ["one", 1, ONE]], [1, 2, 2]),
         "coins": {"pous.setup.salt/v1": bytes(range(24)).hex()}, "widths": {"salt": 16, "one": 16}}


@needs_lake
@pytest.mark.parametrize("name, change, want", [
    ("the setup's salt and the held 1", lambda j: j, {"salt": [[0x0100 + 0x0202 * i for i in range(12)]], "one": [[1]]}),
    ("no coin for the salt: no statement", lambda j: {**j, "coins": {}}, None),
    ("a salt of 23 bytes: no statement", lambda j: {**j, "coins": {"pous.setup.salt/v1": "00" * 23}}, None),
    ("a mark on registered nodes", lambda j: {**j, "program": desc([["store", 1, SALT]], [1])}, None),
    ("supplied nodes with no mark", lambda j: {**j, "program": desc([["salt", 1]], [2])}, None),
    ("a function the registry lacks", lambda j: {**j, "program": desc([["one", 1, {**ONE, "supplied": "x/v1"}]], [2])},
     None),
    ("a value that grows with the population", lambda j: {**j, "program": desc(
        [["pub", 1, {"supplied": "poub.read.pub/v1", "args": {"draw": 0, "K": 4, "masked": 0, "N": 1 << 20},
                     "layout": "le", "bits": 4320 + 6 * (1 << 20)}]], [2])}, None),
    ("a function with no body yet: no statement", lambda j: {**j, "program": desc(
        [["base", 1, {"supplied": "log.base/v1", "args": {}, "layout": "halves", "bits": 64}]], [2])}, None),
], ids=lambda x: x if isinstance(x, str) else "")
def test_named_cases(pieces, name, change, want):
    j = change(SETUP)
    if want is None:
        with pytest.raises(D.Reject):
            reference(j)
        assert pieces.call("supplied_marks", j)[0] != "output"
    else:
        assert reference(j) == want
        assert pieces.call("supplied_marks", j) == ("output", D.canonical(want))


def test_the_registrys_list_is_the_formats():
    text = (PACKAGE / "Spot" / "Supplied.lean").read_text()
    body = re.search(r"def FUNCTIONS : List String :=\s*\[(.*?)\]", text, re.S)
    assert body is not None
    assert re.findall(r'"([^"]+)"', body[1]) == list(SU.FUNCTIONS)
