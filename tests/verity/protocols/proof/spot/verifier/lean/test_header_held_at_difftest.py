"""A member's header held to its slice at its registered reads and the verifier's registered file
(`Spot.HeaderHeldAt.piece`, `OneStageZk.headerHeldAt`), held to the Python mirror (`one_stage.members.population_why`
past its units): a port the slice holds no root of that the reads name at a value the file holds there as a registered
value is not compared, one read at any other value is refused by name, and every other port is held as before."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.protocols.proof.spot.verifier.one_stage import members as M

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proof" / "spot" / "verifier" / "lean")
PIECES = "spot-difftest"
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

SCHEME = "frame-v3-sha512/hm96-sha512"
SLICE = {"population": 2, "scheme": SCHEME,
         "roots": [{"domain": "daa", "leaves": 4, "port": "x", "root": "aa"},
                   {"domain": "dee", "leaves": 4, "port": "o0", "root": "ee", "tree": "t/o0"}]}
READS = [["w", "h/w"], ["w", "h/z"], ["v", "h/w"], ["x", "h/w"], ["z", "h/w"], ["o0", "t/o0"], ["w", "t/o0"],
         ["w", "u/o1"], ["w", "input/n3"], ["w", "committed/w"]]


def header(n: int, roots: dict[str, str | None]) -> dict:
    """A header of the ports given, each its root (a null root a null domain too)."""
    return {"frame_v3": {"domain_ids": {p: None if r is None else "d" + r for p, r in roots.items()}, "roots": roots},
            "instances": n, "scheme": SCHEME}


def registered(ports: dict[str, str] | None) -> dict | None:
    """The verifier's `verity/registered-reads/v2` file of the ports given, each its name and value; None for none (a
    file names at least one port)."""
    if not ports:
        return None
    return {"format": "verity/registered-reads/v2",
            "ports": {p: {"domain": "dd" * 64, "leaves": 2, "root": "aa" * 64, "value": v} for p, v in ports.items()}}


def reference(j):
    reads = {(p, v) for p, v in j["reads"]}
    n = j["slice"]["population"]
    h = j["header"] | {"units": {"indices": list(range(n))}}
    held = j["registered"]["ports"] if j["registered"] is not None else None
    return M.population_why([[0, n]], [("t", [[0, n]])], h, j["slice"], reads, held) or ""


def case(reads, ports, roots, n, slice_=SLICE) -> dict:
    """A piece's input: ``reads`` sorted, the order `population_why` reads its set in."""
    return {"reads": sorted(reads), "registered": registered(ports), "slice": slice_, "header": header(n, roots)}


INPUTS = st.builds(case,
                   st.lists(st.sampled_from(READS), max_size=3, unique_by=tuple),
                   st.one_of(st.none(), st.dictionaries(st.sampled_from(["w", "v", "x"]),
                                                        st.sampled_from(["h/w", "h/z", "t/o0", "committed/w"]))),
                   st.dictionaries(st.sampled_from(["x", "o0", "w", "v"]), st.sampled_from(["aa", "ee", "77", "ab", None]),
                                   min_size=1),
                   st.sampled_from([2, 5]))
HELD = D.Port(PIECES, "header_held_at", reference, INPUTS, package=PACKAGE, of_record="lean", need_rejects=False)

pieces = D.server_fixture(HELD)

HONEST = {"x": "aa", "o0": "ee"}
W = HONEST | {"w": "77"}
CASES = [
    ([["w", "h/w"]], {"w": "h/w"}, W, 2, ""),
    ([["w", "h/z"]], {"w": "h/w"}, W, 2, M.unregistered_why("w", "h/z")),
    ([["w", "h/w"]], {"v": "h/w"}, W, 2, M.unregistered_why("w", "h/w")),
    ([["w", "h/w"]], None, W, 2, M.unregistered_why("w", "h/w")),
    ([["w", "t/o0"]], {"w": "t/o0"}, W, 2, M.unregistered_why("w", "t/o0")),
    ([["w", "u/o1"]], {"w": "u/o1"}, W, 2, M.unregistered_why("w", "u/o1")),
    ([["w", "input/n3"]], {"w": "input/n3"}, W, 2, M.unregistered_why("w", "input/n3")),
    ([["w", "committed/w"]], {"w": "committed/w"}, W, 2, M.unregistered_why("w", "committed/w")),
    ([], {"w": "h/w"}, W, 2, M.unwired_why("w")),
    ([["v", "h/w"]], {"v": "h/w"}, W, 2, M.unwired_why("w")),
    ([["x", "h/w"]], {"x": "h/w"}, {"x": "ab", "o0": "ee"}, 2, "public file: not the registration's: roots, domains"),
    ([["w", "h/w"]], {"w": "h/w"}, {"x": "ab", "o0": "ee", "w": "77"}, 2,
     "public file: not the registration's: roots, domains"),
    ([["w", "h/w"]], {"w": "h/w"}, W, 5, M.POPULATION_WHY),
    # a port in the header's roots at a null root is no port of the header's: neither held by the reads nor compared
    ([["w", "h/w"]], {"w": "h/w"}, HONEST | {"w": None}, 2, M.unwired_why("w")),
    ([["w", "h/z"]], {"w": "h/w"}, HONEST | {"w": None}, 2, M.unwired_why("w")),
    ([], None, {"x": "aa", "o0": None}, 2, "public file: not the registration's: roots, domains"),
    # a port the slice holds and the reads name: compared to the slice, as with no read
    ([["x", "h/w"]], {"x": "h/w"}, HONEST, 2, ""),
    ([["o0", "t/o0"]], {"o0": "h/w"}, {"x": "aa", "o0": "ab"}, 2, ""),
    # a read of a port the header does not name: nothing
    ([["z", "h/z"]], {"w": "h/w"}, HONEST, 2, ""),
    # one port read at two values: each read held, the one at a value the file does not hold there refused
    ([["w", "h/w"], ["w", "h/z"]], {"w": "h/w"}, W, 2, M.unregistered_why("w", "h/z")),
]


@needs_lake
@pytest.mark.parametrize("reads, ports, roots, n, want", CASES)
def test_a_port_is_held_by_the_reads_only_at_a_registered_value_the_file_holds(reads, ports, roots, n, want, pieces):
    j = case(reads, ports, roots, n)
    assert reference(j) == want
    assert pieces.call("header_held_at", j) == ("output", D.canonical(want))


#: a member's port ``w`` read from C's run input node 0 (`OneStageZk.runInputRootsAt`): held at ``input/n0``'s root
RUN_SLICE = {"population": 2, "scheme": SCHEME,
             "roots": [{"domain": "dee", "leaves": 4, "port": "w", "root": "ee", "tree": "input/n0"}]}
RUN_CASES = [
    ([["w", "input/n0"]], ""),
    ([], M.shared_unread("w", "input/n0")),
    ([["w", "input/n1"]], M.shared_unread("w", "input/n0")),
    ([["w", "committed/w"]], M.shared_unread("w", "input/n0")),
]


@needs_lake
@pytest.mark.parametrize("reads, want", RUN_CASES)
def test_a_run_input_port_is_held_only_by_a_read_of_its_nodes_tree(reads, want, pieces):
    """`Spot.HeaderHeldAt`'s guards on the run input slice, replayed: a header root at ``w`` that is not the
    registration's is held only by a read of ``w`` at ``input/n0``; no read, another run input's tree and the member's
    own committed tree each refuse by `SHARED_UNREAD`."""
    j = case(reads, None, {"w": "77"}, 2, RUN_SLICE)
    assert reference(j) == want
    assert pieces.call("header_held_at", j) == ("output", D.canonical(want))


@needs_lake
def test_a_run_input_port_at_its_nodes_own_root_is_refused_without_a_read(pieces):
    """`Spot.HeaderHeldAt`'s guard on the run input slice at the node's own root and domain: with no read the session
    would hold the member's rows to the tree's leaves in order, not the rows C gives, so both refuse by
    `SHARED_UNREAD`; a read of ``w`` at ``input/n0`` holds it."""
    for reads, want in (([], M.shared_unread("w", "input/n0")), ([["w", "input/n0"]], "")):
        j = case(reads, None, {"w": "ee"}, 2, RUN_SLICE)
        assert reference(j) == want
        assert pieces.call("header_held_at", j) == ("output", D.canonical(want))


def test_a_wired_row_named_as_its_writers_field_is_refused_by_name():
    assert M.wired_as_field_why("R/c0", "y") == (
        "member R/c0: its row y of in(S) is named as the writer's output field it reads, and a member's slice names a "
        "field tree's field only at the member's own output")


@needs_lake
def test_the_header_held_at_piece_agrees(pieces):
    stats = D.run(HELD, examples=600, server=pieces)
    assert stats.outputs > 100, stats
