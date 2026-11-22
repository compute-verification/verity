from verity.circuits.replay_links import Link, Part, Rows, Run
from verity.protocols.proof.spot.verifier import second_stage as SS

ROOT1, DOM1 = "aa" * 64, "bb" * 64
PART = Part("Pick_v0", 0, 1)


def _own(positions: tuple[SS.Pos, ...] | None) -> SS.Own:
    boundary = {"population": 4, "roots": [{"port": "logits", "root": ROOT1, "domain": DOM1, "leaves": 4}]}
    return SS.Own(sample=(1, 3), parts=(PART,), interior={}, boundary=boundary, drawn=(),
                  links=(Link("Pick_v0", "x", "logits"),), rows=(Rows("Pick_v0", "x", 1, (Run(0, 1, 0, 0),)),),
                  positions=positions)


def _entry(**rows) -> dict:
    return {"value": "logits", "root": ROOT1, "domain": DOM1, "leaves": 4, **rows}


def test_a_read_at_a_row_other_than_the_circuits_is_refused():
    """With positions held, a linked read's rows are the ones C gives (`Own.positions`), never the file's: C's rows
    (slot 0 at replay block 1's row, slot 1 at block 3's) pass whatever the file states, none included; a row C gives
    other than its unit's in its cell's replay block is refused, and so is a port C gives no rows."""
    honest = (SS.Pos("Pick_v0", "x", (1, 3), 0),)
    assert SS._read_ok(_own(honest), False, PART, "x", _entry())
    assert SS._read_ok(_own(honest), False, PART, "x", _entry(reads=[0, 0], reads_from=0))
    assert not SS._read_ok(_own((SS.Pos("Pick_v0", "x", (1, 2), 0),)), False, PART, "x", _entry(reads=[1, 3]))
    assert not SS._read_ok(_own(()), False, PART, "x", _entry(reads=[1, 3], reads_from=0))
    assert SS._read_ok(_own(None), False, PART, "x", _entry(reads=[1, 3], reads_from=0))


def test_a_port_linked_to_round_1_with_a_tree_of_its_own_is_refused():
    """A port a link names to round 1's tree is placed there alone; the second stage committing a tree of its own for it
    gives it two placings, refused by name."""
    own = _own(None)
    assert SS.placings(own, False, "Pick_v0", "x") == [(SS.ROUND1_TREE, "logits")]
    mine = {"roots": [{"port": "x", "root": "cc" * 64, "domain": "dd" * 64, "leaves": 2}]}
    assert [r for r, _ in SS.placings(SS.Own(**{**own.__dict__, "interior": mine}), False, "Pick_v0", "x")] == \
        [SS.OWN_TREE, SS.ROUND1_TREE]
