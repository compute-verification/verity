"""An output held to a supplied input (`Spot.Supplied.heldPiece`, `supplied_held`): the root's held return rows, each
with the value the verifier computes for its supplied input, and why the return's leaves break them, held to the Python
mirror (`one_stage.registered.held`, `held_why`)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.protocols.proofs.spot.verifier.one_stage import registered as G

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "lean")
PIECES = "spot-difftest"
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ONE = {"supplied": "const.uint/v1", "args": {"value": 1, "bits": 16}, "layout": "le", "bits": 16}
SALT = {"supplied": "pous.setup.salt/v1", "args": {}, "layout": "le", "bits": 192}
ROOT = {"supplied": "registered.root/v1", "args": {"value": "w"}, "layout": "digest", "bits": 512}
CALLEES = ["Input1_v1", "SuppliedInput1_v1", "SuppliedInput16_v1", "SuppliedInput8_v1"]


def desc(mark, rows, fn=1, n=16) -> dict:
    """A root of input ``a`` (one plain node) and ``one`` (``n`` nodes of callee ``fn`` under ``mark``), its return's
    held rows ``rows``."""
    return {"root": "R", "callees": CALLEES, "inputs": [["a", 1], ["one", n, mark]],
            "definitions": {"R": {"body": {"nodes": [{"fn": 0}] + [{"fn": fn}] * n}, "ports": {"ret": rows}}}}


def reference(j):
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    try:
        coins = j.get("coins")
        if not isinstance(coins, dict) or not all(isinstance(h, str) and re.fullmatch(r"(?:[0-9a-f]{2})*", h)
                                                  for h in coins.values()):
            raise ValueError("coins")
        hs = G.held({f: bytes.fromhex(h) for f, h in coins.items()}, j.get("program"))
        outs = j.get("outputs")
        if outs is not None and not (isinstance(outs, list) and all(G._nat(x) is not None for x in outs)):
            raise ValueError("outputs")
        why = None if outs is None else G.held_why(hs, outs) or ""
        return {"held": [[h.lo, h.hi, h.name, h.start, h.n, h.leaves] for h in hs], "why": why}
    except ValueError as e:
        raise D.Reject(str(e)) from e


VERDICT = [[0, 16, {"equals": ["one", 0, 16]}]]
GOOD = {"program": desc(ONE, VERDICT), "coins": {}, "outputs": [1] + [0] * 15}
MARKS = st.one_of(st.just(ONE), st.just(SALT), st.just(ROOT),
                  st.builds(lambda v, b: {**ONE, "args": {"value": v, "bits": b}, "bits": b},
                            st.integers(0, 70000), st.sampled_from([8, 16, 24])),
                  st.builds(lambda f: {**ONE, "supplied": f}, st.sampled_from(["const.u16/v1", "no.such/v1"])))
ROWS = st.lists(st.one_of(
    st.builds(lambda lo, k, name, at, n: [lo, lo + k, {"equals": [name, at, n]}], st.integers(0, 20), st.integers(0, 17),
              st.sampled_from(["one", "a", "zz"]), st.integers(0, 17), st.integers(0, 17)),
    st.builds(lambda lo, hi, eq: [lo, hi, {"equals": eq}], st.integers(0, 3), st.integers(0, 20),
              st.one_of(st.none(), st.integers(), st.lists(st.integers(0, 2), max_size=3))),
    st.builds(lambda lo, hi: [lo, hi, {"interval": [0, 1]}], st.integers(0, 3), st.integers(0, 20)),
    st.lists(st.integers(0, 3), max_size=4)), max_size=3)


@st.composite
def instances(draw) -> dict:
    """A drawn mark on ``one``, its nodes of a drawn width, drawn held rows, a coin perhaps present and drawn return
    leaves near the held one, or none."""
    fn, n = draw(st.sampled_from([(1, 16), (2, 1), (3, 2), (1, 8), (0, 16)]))
    outs = draw(st.one_of(st.none(), st.lists(st.integers(0, 2), max_size=20),
                          st.lists(st.integers(0, 1), min_size=16, max_size=16).map(lambda xs: [1] + xs[1:])))
    return {"program": desc(draw(MARKS), draw(ROWS), fn, n), "outputs": outs,
            "coins": draw(st.sampled_from([{}, {"pous.setup.salt/v1": bytes(range(24)).hex()}]))}


INPUTS = st.one_of(*[instances() for _ in range(4)], st.just(GOOD))
HELD = D.Port(PIECES, "supplied_held", reference, INPUTS, package=PACKAGE, of_record="lean")

pieces = D.server_fixture(HELD)


@needs_lake
def test_the_supplied_held_piece_agrees(pieces):
    stats = D.run(HELD, examples=1500, server=pieces)
    assert stats.outputs > 100 and stats.rejects > 0, stats


@needs_lake
@pytest.mark.parametrize("name, j, why", [
    ("a verdict of 1 held to the supplied 1", GOOD, ""),
    ("a verdict word of 0 held to 1", {**GOOD, "outputs": [0] * 16},
     "held output: return leaf 0 is 0, held to leaf 0 of one, which is 1"),
    ("a verdict of 1 held to a constant of another value", {**GOOD, "program": desc(
        {**ONE, "args": {"value": 2, "bits": 16}}, VERDICT)},
     "held output: return leaf 0 is 1, held to leaf 0 of one, which is 0"),
    ("a return shorter than the held row", {**GOOD, "outputs": [1] * 15},
     "held output: the return has 15 leaves, and a held row ends at 16"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_held_verdicts(pieces, name, j, why):
    want = reference(j)
    assert want["why"] == why
    assert pieces.call("supplied_held", j) == ("output", D.canonical(want))


@needs_lake
@pytest.mark.parametrize("name, j, text", [
    ("a function the closed list does not hold", {**GOOD, "program": desc({**ONE, "supplied": "const.u16/v1"}, VERDICT)},
     "supplied input one: const.u16/v1 is no supplied function"),
    ("held to an input with no mark", {**GOOD, "program": desc(ONE, [[0, 16, {"equals": ["a", 0, 16]}]])},
     "held output: a is no supplied input of the program"),
    ("held to a value the verifier computes from the record", {**GOOD, "program": desc(ROOT, VERDICT, 1, 512)},
     "held output: one is registered.root/v1, whose value is no bytes the verifier holds: an output is held to a "
     "constant or a coin"),
    ("a run past the input's leaves", {**GOOD, "program": desc(ONE, [[0, 16, {"equals": ["one", 8, 16]}]])},
     "held output: [0, 16) held to leaves [8, 24) of one's 16, no run of them whose length divides the row's"),
    ("a run whose length does not divide the row's", {**GOOD, "program": desc(ONE, [[0, 16, {"equals": ["one", 0, 5]}]])},
     "held output: [0, 16) held to leaves [0, 5) of one's 16, no run of them whose length divides the row's"),
    ("not the held form", {**GOOD, "program": desc(ONE, [[0, 16, {"equals": ["one", 0]}]])},
     "held output: a held row is [lo, hi, {equals: [input, at, n]}]"),
    ("nodes of two widths", {**GOOD, "program": {**desc(ONE, VERDICT), "definitions": {"R": {
        "body": {"nodes": [{"fn": 0}, {"fn": 2}] + [{"fn": 1}] * 15}, "ports": {"ret": VERDICT}}}}},
     "held output: supplied input one's nodes are no SuppliedInput<w>_v1 of one width"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_refusals(pieces, name, j, text):
    with pytest.raises(D.Reject, match=f"^{re.escape(text)}$"):
        reference(j)
    assert pieces.call("supplied_held", j) == ("reject", text)
