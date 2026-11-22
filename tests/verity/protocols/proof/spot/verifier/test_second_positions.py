from verity.circuits import partition_object as PO
from verity.circuits.boolean import NOT
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.population import mixed_population_program
from verity.circuits.replay_links import Link, Part, Rows, Run
from verity.circuits.types import Array, Value
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
    other than its unit's in its cell's replay block is refused."""
    honest = (SS.Pos("Pick_v0", "x", (1, 3), 0),)
    assert SS._read_ok(_own(honest), False, PART, "x", _entry())
    assert SS._read_ok(_own(honest), False, PART, "x", _entry(reads=[0, 0], reads_from=0))
    assert not SS._read_ok(_own((SS.Pos("Pick_v0", "x", (1, 2), 0),)), False, PART, "x", _entry(reads=[1, 3]))
    assert SS._read_ok(_own(None), False, PART, "x", _entry(reads=[1, 3], reads_from=0))


def test_a_linked_port_c_gives_no_rows_is_read_at_round_ones_rows_and_tree():
    """A link's port C gives no position (a return round 1's record commits) is read at each slot's unit's row of its
    replay block in round 1's port, whatever the file states; at a tree other than round 1's it is refused."""
    assert SS._read_ok(_own(()), False, PART, "x", _entry(reads=[0, 0], reads_from=0))
    assert SS._at_pos(_own(()), "Pick_v0", "x", _entry(), SS._linked_rows(_own(()), PART, Rows("Pick_v0", "x", 1,
                      (Run(0, 1, 0, 0),)), _entry()))["reads"] == [1, 3]
    assert not SS._read_ok(_own(()), False, PART, "x", {**_entry(), "leaves": 16})


PICK = bind(CompositeDefinition("TSecondPositionsPick", 1, (), lambda S: ((("x", Array(16, Value(1))),), Array(16, Value(1))),
                                lambda B, S, x: B.batch(NOT, x, axes=(0,)), register=False))


def test_the_twin_takes_the_rows_c_gives_each_port():
    """`positions_of`, Lean's `positionsOf`: the rows C gives a registered read's port at the statement's units, here
    an earlier value of 6 rows that unit i reads at row (1, 4)[i]; none for a port C reads no input at (a return,
    ``o0``), and none where the read's rows are not the input node's."""
    program = mixed_population_program([(PICK, 2)], reads=[("logits", 6, 0, "x", (1, 4))])
    t = PO.template_id(PICK)

    def of(x: dict) -> tuple[SS.Pos, ...]:
        stmt = {"template": t, "header": {"units": {"indices": [0, 1]}}, "draw": {},
                "registered": {"ports": {"x": x, "o0": {"value": "o0", "leaves": 6}}}}
        return SS.positions_of(program, PO.template_instances_query([PICK]), [stmt])

    assert of({"value": "logits", "leaves": 6}) == (SS.Pos(t, "x", (1, 4), 0),)
    assert of({"value": "logits", "leaves": 5}) == () == of({"value": "input/n0", "leaves": 6})
