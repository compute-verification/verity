"""`members_run` (`test_members_run`'s window of two members) on a C whose first template reads a slice of its run input
node: `MembersRunPair`'s two units read rows 1 and 3 of a node of 4, its one `Read` a first row above zero and a stride
of two rows; `MembersRunTriple`'s three units read their node whole. Each member's circuit file is lowered by Lean, its
public file is held to its slice at the record's one tree of the whole node, and the run credits the sessions; a tree
of only the rows read is refused by name."""
from __future__ import annotations

import pytest

import test_members_run as MR
from verity.circuits import partition_object as PO
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import array_of, coll_concat
from verity.circuits.types import Array, Tuple, Value
from verity.commitments import registered_value as RV
from verity.commitments.periods import interface as I
from verity.protocols.proof.spot.verifier.one_stage import members as M, partition as P

ROW = Array(64, Value(16))
RET = Tuple(Array(2, Value(16)), Array(3, Value(16)))
SLICED = Program(bind(CompositeDefinition(
    "MembersRunSliced", 0, (), lambda S: ((("0.x", Array(4, ROW)), ("1.x", Array(3, ROW))), RET),
    lambda B, S, x0, x1: coll_concat(RET, [B.batch(MR.PAIR, array_of([x0[1], x0[3]]), axes=(0,)),
                                           B.batch(MR.TRIPLE, x1, axes=(0,))]), register=False)))


def inputs(rows_a: int = 4) -> dict:
    return {m: (f"input/n{w}", n, r, RV.domain(P.program_digest(SLICED), f"input/n{w}", 64, n, 16,
                                                window=MR.WINDOW_ROOT).domain_id.hex())
            for m, w, n, r in ((MR.A, 0, rows_a, "33" * 64), (MR.B, 1, 3, "44" * 64))}


def record(ins: dict) -> dict:
    return {**MR.RECORD, "roots": [r for r in MR.RECORD["roots"] if not r["port"].startswith("input/")] +
            [{"domain": d, "leaves": n, "port": t, "root": r} for t, n, r, d in ins.values()]}


@pytest.fixture(autouse=True)
def ledger(monkeypatch):
    monkeypatch.setattr(MR.C, "trusted_session_of", lambda *a: None)


@pytest.fixture
def sliced(monkeypatch):
    partition = PO.build(SLICED, MR.QUERY, [[MR.UNITS[MR.A][0], MR.UNITS[MR.B][0]]])
    descriptor = MR.json.dumps(MR.json.loads(MR.json.dumps(MR.codec.encode_program(SLICED))), sort_keys=True,
                               separators=(",", ":")).encode()
    for k, v in (("PROGRAM", SLICED), ("PARTITION", partition), ("DESCRIPTOR", descriptor), ("INPUTS", inputs())):
        monkeypatch.setattr(MR, k, v)
    MR.lowered.cache_clear()
    yield
    MR.lowered.cache_clear()


def test_the_sliced_reads_are_one_read_at_an_offset_and_a_stride():
    words = dict(I.layout(MR.PAIR).rows)["x"]
    got = I.run_input_reads([(MR.PAIR, 2), (MR.TRIPLE, 3)], SLICED, MR.QUERY, ())
    assert [(r.reader, r.port, r.param, r.at, r.radix, r.base, r.step) for r in got] == [
        (MR.PAIR.id, "x", "input/n0", 0, (2,), words, (2 * words,)),
        (MR.TRIPLE.id, "x", "input/n1", 0, (3,), 0, (words,))]


@MR.lean
def test_a_member_reading_a_slice_of_its_run_input_node_is_held_at_the_whole_nodes_tree(tmp_path, sliced):
    """The run holds each member's public file to its slice, the first member's port ``x`` at the record's tree of all
    four rows of node 0, lowers each member's C_S in Lean, and credits both sessions; with the record's tree of only the
    two rows read, the slice is refused by name before any session."""
    w = MR.Window(tmp_path)
    assert not w.underived, w.underived
    run, refused = w.run(record=record(inputs()), program=SLICED)
    assert refused is None and run.all and run.proved == (1, 2, 4), refused
    sl = M.slices(MR.UNITS, record(inputs()), program=SLICED, query=MR.QUERY, window_root=MR.WINDOW_ROOT)
    assert [r for r in sl[MR.A]["roots"] if r["port"] == "x"] == [MR.input_entry(MR.A)]

    MR.INPUTS = inputs(rows_a=2)
    (tmp_path / "read-rows").mkdir()
    w = MR.Window(tmp_path / "read-rows")
    assert w.run(MR.never, record=record(MR.INPUTS), program=SLICED) == (None, M.run_tree_why(0, "leaf count"))
