"""A member's header held to its slice at its registered reads (`Spot.HeaderHeldAt.piece`, `OneStageZk.headerHeldAt`),
held to the Python mirror (`one_stage.members.population_why` past its units): a port the slice holds no root of that
the reads name at a registered value is not compared, and every other port is, as before."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.protocols.proofs.spot.verifier.one_stage import members as M

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "lean")
PIECES = "spot-difftest"
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

SCHEME = "frame-v3-sha512/hm96-sha512"
SLICE = {"population": 2, "scheme": SCHEME,
         "roots": [{"domain": "daa", "leaves": 4, "port": "x", "root": "aa"},
                   {"domain": "dee", "leaves": 4, "port": "o0", "root": "ee", "tree": "t/o0"}]}
READS = [["w", "h/w"], ["v", "h/w"], ["x", "h/w"], ["o0", "t/o0"], ["w", "t/o0"], ["w", "u/o1"], ["w", "input/n3"],
         ["w", "committed/w"]]


def header(n: int, roots: dict[str, str]) -> dict:
    return {"frame_v3": {"domain_ids": {p: "d" + r for p, r in roots.items()}, "roots": roots}, "instances": n,
            "scheme": SCHEME}


def reference(j):
    reads = {(p, v) for p, v in j["reads"]}
    n = j["slice"]["population"]
    h = j["header"] | {"units": {"indices": list(range(n))}}
    return M.population_why([[0, n]], [("t", [[0, n]])], h, j["slice"], reads) or ""


INPUTS = st.builds(lambda reads, roots, n: {"reads": reads, "slice": SLICE, "header": header(n, roots)},
                   st.lists(st.sampled_from(READS), max_size=3),
                   st.dictionaries(st.sampled_from(["x", "o0", "w", "v"]), st.sampled_from(["aa", "ee", "77", "ab"]),
                                   min_size=1),
                   st.sampled_from([2, 5]))
HELD = D.Port(PIECES, "header_held_at", reference, INPUTS, package=PACKAGE, of_record="lean", need_rejects=False)

pieces = D.server_fixture(HELD)

HONEST = {"x": "aa", "o0": "ee"}
CASES = [
    ([["w", "h/w"]], HONEST | {"w": "77"}, 2, ""),
    ([], HONEST | {"w": "77"}, 2, M.unwired_why("w")),
    ([["v", "h/w"]], HONEST | {"w": "77"}, 2, M.unwired_why("w")),
    ([["w", "t/o0"]], HONEST | {"w": "77"}, 2, M.unwired_why("w")),
    ([["w", "u/o1"]], HONEST | {"w": "77"}, 2, M.unwired_why("w")),
    ([["w", "input/n3"]], HONEST | {"w": "77"}, 2, M.unwired_why("w")),
    ([["w", "committed/w"]], HONEST | {"w": "77"}, 2, M.unwired_why("w")),
    ([["x", "h/w"]], {"x": "ab", "o0": "ee"}, 2, "public file: not the registration's: roots, domains"),
    ([["w", "h/w"]], {"x": "ab", "o0": "ee", "w": "77"}, 2, "public file: not the registration's: roots, domains"),
    ([["w", "h/w"]], HONEST | {"w": "77"}, 5, M.POPULATION_WHY),
]


@needs_lake
@pytest.mark.parametrize("reads, roots, n, want", CASES)
def test_a_port_read_at_a_registered_value_is_held_by_the_reads_and_no_other(reads, roots, n, want, pieces):
    j = {"reads": reads, "slice": SLICE, "header": header(n, roots)}
    assert reference(j) == want
    assert pieces.call("header_held_at", j) == ("output", D.canonical(want))


@needs_lake
def test_the_header_held_at_piece_agrees(pieces):
    stats = D.run(HELD, examples=600, server=pieces)
    assert stats.outputs > 100, stats
