"""A two-stage sampling design on the proof system's `Stream`: what the stream's spec, params and rule state and its
first stage samples; and its replay grain's fan-out."""

from __future__ import annotations

import dataclasses
import os
import random
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pytest

from verity.circuits import population as PN
from verity.circuits import codec, partition_object as PO
from verity.circuits.cut import member_refs
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.population import population
from verity.circuits.program import Program
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple, Value as Bits
from verity.protocols.proof.spot.prover import optimize as O, plancheck as PC
from verity.protocols.proof.spot.verifier import second_stage as SS, service as S
from verity.protocols.proof.spot.verifier.one_stage import draw as D, partition as P, docket as R, verifier as V
from verity.protocols.proof.spot.prover import docket as DK

import fake_challenger as FC

XOR8 = PrimitiveDefinition("TwoStageTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("TwoStageTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
#: A replay block: three proof blocks, instances of PAIR.
BLOCK = bind(CompositeDefinition("TwoStageTestBlock", 0, (), lambda S_: ((("x", Array(3, Array(2, Bits(8)))),),
                                                                         Array(3, Bits(8))),
                                 lambda B, S_, x: B.batch(PAIR, x, axes=(0,)), register=False))
QUERY = P.template_instance_query(PAIR)
GRAIN = P.template_instance_query(BLOCK)
N_RHO = 3
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-two-stage-test"
SESSION = "5e" * 64
DELTA = 2.0 ** -40
N, Q, K = 4, 2, 1
#: The boundary docket's ports, the replay block's `x` and output named by its template, which the design's links
#: (`second_stage.replay_links`) name.
BOUNDARY = BX, BOUT = tuple(f"{codec.descriptor_id(BLOCK)}/{p}" for p in ("x", "o0"))
INTERIOR = ("x", "out")


def _public(boundary: tuple[str, ...]) -> tuple[S.Public, ...]:
    return (S.Public("population", "structure", "N, the sample's range"),
            S.Public("window", "structure", "provenance: served, on test prompts (step 0)"),
            *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k} (step 0)") for v in boundary + INTERIOR
              for k in ("leaves", "domain")))


def _values(boundary: tuple[str, ...]) -> tuple[S.Value, ...]:
    return (*(S.Value(v, SCHEMA, "serve") for v in boundary), *(S.Value(v, SCHEMA, "interior") for v in INTERIOR))


PUBLIC, VALUES = _public(BOUNDARY), _values(BOUNDARY)
RULE = S.TwoStage(GRAIN, "rate:1/2", K)
TEMPLATE, = (t for t, _ in P.strata(PN.population_program(PAIR, Q * N_RHO), QUERY))
#: The reference header of an honest statement in these streams, fixed before the run: PAIR's part's, its ports' schemas
#: and the set it was staged in, none of the run's fields (`fake_challenger.reference_header`).
REFERENCE = FC.reference_header(TEMPLATE, 0, Q * N_RHO, dict.fromkeys(INTERIOR, SCHEMA))
#: The registered reads of an honest statement's session in these streams, fixed before the run: PAIR's linked `x` and
#: `o0`, each the boundary port its link names, and none of the run's fields (`second_stage.blank_reads`).
READS = {"ports": {"x": {"value": BX}, "o0": {"value": BOUT}}}


def _spec(selection=RULE, values=VALUES, proof_deadline=60.0) -> S.Spec:
    return S.Spec(values, selection, S.Check(P.program_digest(Program(PAIR)), QUERY, proof_deadline=proof_deadline),
                  PUBLIC, b"two-stage-test")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _stream(spec=None, clock=None, **kwargs) -> S.Stream:
    return FC.Stream(spec or _spec(), PAIR, **{"replay": BLOCK, "budget": 2, "delta": DELTA,
                                               "challenger": "challenger-test", "clock": clock or Clock(), **kwargs})


def _ports(program: Program, query, names, n: int | Mapping[str, int]) -> dict:
    """The ports `names` of a docket over `program`, port `v` of ``n[v]`` leaves (each of `n` for a number)."""
    part = P.partition(program, query)
    leaves = n if isinstance(n, Mapping) else dict.fromkeys(names, n)
    return {v: {"leaves": leaves[v],
                "domain": R.window_domain(part["program"], P.digest(part), v, SCHEMA, leaves[v], RUN)} for v in names}


def _record(program: Program, query, names, n: int | Mapping[str, int], sampling_design, window, previous) -> dict:
    ports = _ports(program, query, names, n)
    roots = [{"port": v, "root": f"{i + 1:02x}" * 64, **ports[v]} for i, v in enumerate(names)]
    return DK.record(program, query, sampling_design, S.SCHEME, roots, window, previous=previous)


def _boundary(n=N, row=0, previous=None, q=Q, **window) -> tuple[dict, dict]:
    """The boundary docket over `n` replay blocks: BLOCK's `x` and output, each of N_RHO rows a block, one for
    each of its PAIR instances' `x` and `o0` (`second_stage.replay_rows`); its window names `window`'s items too."""
    program, leaves = PN.population_program(BLOCK, n), dict.fromkeys(BOUNDARY, n * N_RHO)
    return (_record(program, GRAIN, BOUNDARY, leaves, {"law": "subset", "k": q},
                    {"kind": "served", "run": RUN, "row": row, **window}, previous),
            _ports(program, GRAIN, BOUNDARY, leaves))


def _interior(m: int, previous: str, sample, row=0, **change) -> tuple[dict, dict]:
    """The interior docket's record over `m` proof blocks, its window naming the replay sample `sample` it lays out
    (`window.replay_sample`, the first stage's proved blocks in order where honest), and its ports."""
    program = PN.population_program(PAIR, m)
    rec = _record(program, QUERY, INTERIOR, m, {"law": "subset", "k": m},
                  {"kind": "served", "run": RUN, "row": row, "interior": True, "replay_sample": sample}, previous)
    return {**rec, **change}, _ports(program, QUERY, INTERIOR, m)


def _lean(accepted=True, reads=None, checked=None, tables=1) -> dict:
    """The verdict of record, accepted or not, reporting the sessions' registered reads `reads`, and naming what
    `verity-verify-proof verify` names of the session of statement `checked` it checked: its header's and registered reads'
    digests, its sample and the `tables` it set it up as (None: no statement's). For a list of statements, one such
    verdict of record for each."""
    v = {"of_record": True, "accepted": accepted, "digest": "dd" * 64, "sessions": [SESSION]}
    if reads is not None:
        v["registered"] = reads
    if isinstance(checked, list):
        return {f"lean-verify/{i}": v | FC.checked(s, tables) for i, s in enumerate(checked)}
    if isinstance(checked, dict):
        v |= FC.checked(checked, tables)
    return {"lean": v}


def _refused(f) -> tuple[str, ...]:
    with pytest.raises(S.Refused) as e:
        f()
    return e.value.verdict.codes


def test_a_two_stage_outcome_is_complete_only_on_the_request_definition_and_its_records():
    """`second_stage.complete_twin`: true only when the replay block is the registered request Definition and each
    audit record accepted and is itself complete. One missing, it is false: a stand-in replay block; a record that is
    missing, did not accept, is not complete or does not say; no record at all. An argument that is not the boolean
    true counts as missing."""
    whole = {"accepted": True, "complete": True}
    assert SS.complete_twin(True, [whole, whole]) is True and SS.complete_twin(True, [whole]) is True
    assert not SS.complete_twin(False, [whole, whole])
    for record in (None, {"accepted": True, "complete": False}, {"accepted": False, "complete": True},
                   {"accepted": True}, {"complete": True}, {"accepted": 1, "complete": True}):
        assert not SS.complete_twin(True, [whole, record])
    assert not SS.complete_twin(True, [])
    assert not SS.complete_twin("yes", [whole])


def test_only_an_accepted_verdict_of_record_names_what_was_checked():
    v = {"of_record": True, "accepted": True, "public_header_sha512": "ab" * 64, "unit_draw": {"units": [1]}}
    assert SS.verified_of({"lean": v, "upstream": {"of_record": False, "accepted": True}}) == (
        SS.Checked("ab" * 64, None, {"units": [1]}, None),)
    assert SS.verified_of({"lean": {**v, "session_tables": 2}}) == (SS.Checked("ab" * 64, None, {"units": [1]}, 2),)
    assert SS.verified_of({"a": {**v, "accepted": False}, "b": {**v, "of_record": False},
                           "c": {**v, "public_header_sha512": None}}) == ()


ID18 = PrimitiveDefinition("TwoStageTestId18", 1, (("a", Bits(8)),), Bits(18), lambda a: a, register=False)
#: A unit returning one 18-bit field, and four of them a replay block.
WIDE = bind(CompositeDefinition("TwoStageTestWide", 0, (), lambda S_: ((("x", Bits(8)),), Bits(18)),
                                lambda B, S_, x: B.call(ID18, x), register=False))
#: A unit returning eight 18-bit values as one field, and two of them a replay block.
PACK = bind(CompositeDefinition("TwoStageTestPack", 0, (), lambda S_: ((("x", Array(8, Bits(8))),), Array(8, Bits(18))),
                                lambda B, S_, x: array_of([B.call(ID18, x[i]) for i in range(8)]), register=False))


def _returned(block) -> int | None:
    return SS.replay_returned(PN.population_program(block, N), codec.descriptor_id(block))


ID32 = PrimitiveDefinition("TwoStageTestId32", 1, (("a", Bits(8)),), Bits(32), lambda a: a, register=False)
#: A unit returning one 32-bit id, as a token select returns it.
WORD = bind(CompositeDefinition("TwoStageTestWord", 0, (), lambda S_: ((("x", Bits(8)),), Bits(32)),
                                lambda B, S_, x: B.call(ID32, x), register=False))


def test_a_cap_of_875_ids_is_refused_for_an_output_class_of_1024():
    """A request's replay block in an output class of O ids, one unit an id as a serve's stand-in lays it out
    (`population`), returns every id of the class as its unit's 32-bit word, served or not: 32,768 bits at O = 1,024
    and 16,384 at 512. So a served cap of 875 ids at 32 bits, 28,000, is refused for the class of 1,024 and priced for
    the class of 512 (`plancheck.two`'s `returned`): a cap holds a class only when the class is no larger than it."""
    top, half = (_returned(population(WORD, o)) for o in (1024, 512))
    assert (top, half) == (32 * 1024, 32 * 512)
    cells, k, cap = [O.Cell(3, 8.0)], {0: 1}, 875 * 32
    assert PC.two(cells, k, 2, .5, .01, cap, fanout=1, returned=half)["S"] == str(cap)
    with pytest.raises(ValueError, match="served-cover: a replay block of the registered Program returns 32768 bits"):
        PC.two(cells, k, 2, .5, .01, cap, fanout=1, returned=top)
    assert PC.two(cells, k, 2, .5, .01, top, fanout=1, returned=top)["S"] == "32768"


#: A read's template, which the auditor's params record as a read of 8 rows (`READ_SIZES`), and its ports' links,
#: each at a value of its own name.
READ_T = "EmbeddingRead_v2{V=8,H=2}"
READ_SIZES = ((READ_T, 8),)
READ_LINKS = tuple(SS.Link(READ_T, p, p) for p in ("tok", "root", "domain", "row", "salt", "path"))
#: The roots and domain ids of tables ``t`` and ``u``, and round 1's record registering both.
T_ROOT, T_DOMAIN, U_ROOT, U_DOMAIN = ("a1" * 64, "a2" * 64, "b1" * 64, "b2" * 64)
READ_RECORD = {"roots": [{"port": "t", "root": T_ROOT, "domain": T_DOMAIN, "leaves": 8},
                         {"port": "u", "root": U_ROOT, "domain": U_DOMAIN, "leaves": 8}]}
#: What the verifier fills of a read: its table's root and domain id, as the bytes it computed for the table, its key,
#: and the 1 its verdict port reads.
READ_FILLED = {"root": [[{"bytes": T_ROOT}]], "domain": [[{"bytes": T_DOMAIN}]], "tok": [[{"bytes": "00000002"}]],
               "one": [[{"bytes": "0001"}], [{"bytes": "0001"}]]}
READ_REF = ((READ_T, {"ports": {"o1": {"value": "one"}}}),)
#: The rows of the read's root and domain ports: one row a block, which every unit reads.
READ_ROWS = tuple(SS.Rows(READ_T, p, 1, (SS.Run(0, 1, 0, 0),)) for p in ("root", "domain"))
#: `READ_FILLED` with the root and the domain of two rows each, of tables ``t`` and ``u``.
TWO_TABLES = {**READ_FILLED, "root": [[{"bytes": T_ROOT}], [{"bytes": U_ROOT}]],
              "domain": [[{"bytes": T_DOMAIN}], [{"bytes": U_DOMAIN}]]}


def _holding(registered, links=READ_LINKS, wires=(), supplied=None, fixed=(), ref_reads=READ_REF,
             rows=READ_ROWS, boundary=READ_RECORD, reads=READ_SIZES, tables=()) -> SS.Own:
    """What round 2 holds (`second_stage.Own`) of an interior docket registering values `registered`, with the
    design's `links`, `wires` and `rows`, the verifier's own `supplied` values and `fixed` names, the reference
    reads, round 1's record `boundary`, the templates the params record as `reads`, and the tables they pin."""
    return SS.Own((0,), (), {"registered": [{"value": v} for v in registered]}, boundary, (), tuple(links),
                  wires=tuple(wires), supplied=SS.supplied_of(READ_FILLED if supplied is None else supplied),
                  fixed=tuple(fixed), ref_reads=tuple(ref_reads), rows=tuple(rows), reads=tuple(reads),
                  tables=tuple(tables))


READ_VALUES = ("tok", "root", "domain", "row", "salt", "path", "one")


@pytest.mark.parametrize("own,value", [
    (_holding(READ_VALUES), None),
    (_holding(()), None),
    (_holding(("logits",), [SS.Link("T", "logits", "logits")], supplied={}), "logits"),
    (_holding(("i",), [SS.Link("T", "i", "i")], supplied={"i": [[{"bytes": "00000003"}]]}), None),
    (_holding(("s",), [SS.Link("T", "s", "s")], supplied={}, fixed=("s",)), None),
    (_holding(("p",), [SS.Link("T", "p", "p")], supplied={"p": [[{"port": "T/o0"}]]}), "p"),
    (_holding(READ_VALUES, supplied={k: v for k, v in READ_FILLED.items() if k != "tok"}), "tok"),
    (_holding(READ_VALUES[1:], [*READ_LINKS[1:], SS.Link(READ_T, "tok", "b.ingress")]), None),
    (_holding(READ_VALUES[1:], READ_LINKS[1:]), "row"),
    (_holding(READ_VALUES, supplied={**READ_FILLED, "domain": [[{"bytes": U_DOMAIN}]]}), "row"),
    (_holding(READ_VALUES, supplied={**READ_FILLED, "domain": [[{"bytes": T_ROOT}]]}), "row"),
    (_holding(READ_VALUES, supplied={**READ_FILLED, "root": [[{"bytes": "00"}]]}), "row"),
    (_holding(READ_VALUES, supplied={**READ_FILLED, "root": [[{"bytes": T_ROOT}, {"bytes": T_ROOT}]]}), "row"),
    (_holding(READ_VALUES, supplied={**READ_FILLED, "root": [[{"history": "t/root"}]]}), "root"),
    (_holding(READ_VALUES, boundary={"roots": []}), "row"),
    (_holding(READ_VALUES, boundary={"roots": READ_RECORD["roots"][1:]}), "row"),
    (_holding(READ_VALUES, boundary={"roots": [{**READ_RECORD["roots"][0], "domain": U_DOMAIN}]}), "row"),
    (_holding(READ_VALUES, boundary={"roots": [{"port": "t", "root": T_ROOT}, {"port": "v", "domain": T_DOMAIN}]}), "row"),
    (_holding(READ_VALUES, supplied={**READ_FILLED, "root": TWO_TABLES["root"]}), "row"),
    (_holding(READ_VALUES, supplied={**READ_FILLED, "root": [], "domain": []}), "row"),
    (_holding(READ_VALUES, supplied=TWO_TABLES), None),
    (_holding(READ_VALUES, rows=()), "row"),
    (_holding(READ_VALUES, rows=READ_ROWS[:1]), "row"),
    (_holding(READ_VALUES, rows=(READ_ROWS[0], SS.Rows(READ_T, "domain", 2, READ_ROWS[1].runs))), "row"),
    (_holding(READ_VALUES, rows=(READ_ROWS[0], SS.Rows(READ_T, "domain", 1, (SS.Run(0, 1, 0, 1),)))), "row"),
    (_holding(READ_VALUES, supplied={k: v for k, v in READ_FILLED.items() if k != "root"}), "root"),
    (_holding(READ_VALUES, supplied={**READ_FILLED, "one": [[{"bytes": "0001"}], [{"bytes": "0000"}]]}), "row"),
    (_holding(READ_VALUES, supplied={**READ_FILLED, "one": [[{"bytes": "00000001"}], [{"bytes": "00000001"}]]}), "row"),
    (_holding(READ_VALUES, supplied={**READ_FILLED, "one": []}), "row"),
    (_holding(READ_VALUES, supplied={k: v for k, v in READ_FILLED.items() if k != "one"}, fixed=("one",)), "row"),
    (_holding(READ_VALUES, ref_reads=((READ_T, {"ports": {"o1": {"value": f"{READ_T}/o1"}}}),)), "row"),
    (_holding(READ_VALUES, ref_reads=()), "row"),
    (_holding(READ_VALUES, [*READ_LINKS, SS.Link("Other_v1", "x", "row")]), "row"),
    (_holding(READ_VALUES, [SS.Link("Other_v1{V=8}", x.port, x.value) for x in READ_LINKS],
              ref_reads=(("Other_v1{V=8}", READ_REF[0][1]),)), "row"),
    (_holding(READ_VALUES, reads=()), "row"),
    (_holding(READ_VALUES, reads=(("Other_v1{V=8}", 8),)), "row"),
    (_holding(READ_VALUES, reads=((READ_T, 16),)), "row"),
    (_holding(READ_VALUES, reads=((READ_T, 16), (READ_T, 8))), "row"),
    (_holding(READ_VALUES, reads=((READ_T, 8), (READ_T, 16))), None),
    (_holding(READ_VALUES, boundary={"roots": [{**e, "leaves": 16} for e in READ_RECORD["roots"]]}), "row"),
    (_holding(READ_VALUES, boundary={"roots": [{**e, "leaves": "8"} for e in READ_RECORD["roots"]]}), "row"),
    (_holding(READ_VALUES, boundary={"roots": [{k: v for k, v in e.items() if k != "leaves"}
                                               for e in READ_RECORD["roots"]]}), "row"),
    (_holding(READ_VALUES, boundary={"roots": [READ_RECORD["roots"][0], {**READ_RECORD["roots"][0], "leaves": 16}]}),
     "row"),
    (_holding(READ_VALUES, boundary={"roots": [{**READ_RECORD["roots"][0], "leaves": 16}, READ_RECORD["roots"][0]]}),
     None),
    (_holding((*READ_VALUES, "w")), "w"),
    (_holding(READ_VALUES, ref_reads=((READ_T, {"ports": {"o1": {"value": "one"}, "w": {"value": "row"}}}),)), "row"),
    (_holding(READ_VALUES, ref_reads=(*READ_REF, ("Other_v1", {"ports": {"x": {"value": "salt"}}}))), "salt"),
    (_holding(READ_VALUES, ref_reads=((READ_T, {"ports": {"o1": {"value": "one"}, "row": {"value": "row"},
                                                          "tok": {"value": "tok"}, "x": {"value": "one"}}}),)), None),
])
def test_the_interior_registration_gives_a_replay_block_only_a_reads_opened_row(own, value):
    """`second_stage.inputs_failure`, the first value of the interior docket that a replay block's input may not
    read. Held without the prover, and passing: a value the verifier fills from its own bytes, or pinned or holds to an
    earlier record. Refused: one the prover registers, or one filled from a root of the interior record. A read's row,
    salt and path pass where the auditor's params record the template as a read, the verifier fills its root and domain
    with the root and the domain id round 1's record registers for one table of the read's rows, its key is filled or
    a port of round 1's record, and its verdict port reads a value filled with 1 at every row; and are refused, the
    row named, where the key is unlinked, the verdict's value has a row that is not 1, no row, is pinned and not
    filled, is the interior's own port or is not named, where another template reads the row too, or the params record
    no read of the template, another template's alone, or one of another count of rows (a template's first record is
    its record). The root and the domain are one table's of round 1's record, by their bytes: refused where the
    domain's row is another table's or a root, where the root's row is bytes round 1's record registers nowhere or two
    sources, where the two have other counts of rows or none, where round 1's record registers no table, another one,
    the root under another domain, the root and the domain at two ports, or the table at another count of leaves, a
    count that is no number or none (the port's last entry is its entry), and where the two ports' rows are missing or
    place a unit differently; two tables filled alike row by row pass. A root left as a table's name, which the
    verifier resolved to no bytes, is none of its values, and is refused by its own name. A key or a root the prover
    registers is refused by its own name, and so is a value no link names. A row, a salt or a path that a reference
    statement reads at a port no link names it at is refused too: only its opening may read it; a reference reading it
    at its link's port, or reading a value the verifier fills anywhere, changes nothing."""
    assert SS.inputs_failure(own) == value
    found = dict(SS._interior_failures(own)).get("second-inputs")
    assert found == (None if value is None else {"value": value, "why": SS.INPUTS_WHY})


#: The tables' registration's entries for tables ``t`` and ``u``, as the auditor's params pin them (`service.pinned_tables`).
PINNED = (("t", {"value": "t", "root": T_ROOT, "leaves": 8, "words": 2, "domain": T_DOMAIN}),
          ("u", {"value": "u", "root": U_ROOT, "leaves": 8, "words": 2, "domain": U_DOMAIN}))
NO_RECORD = {"roots": []}


@pytest.mark.parametrize("own,value", [
    (_holding(READ_VALUES, boundary=NO_RECORD, tables=PINNED), None),
    (_holding(READ_VALUES, boundary=NO_RECORD, tables=PINNED[:1]), None),
    (_holding(READ_VALUES, tables=PINNED), None),
    (_holding(READ_VALUES, boundary=NO_RECORD, supplied=TWO_TABLES, tables=PINNED), None),
    (_holding(READ_VALUES, boundary={"roots": READ_RECORD["roots"][1:]}, supplied=TWO_TABLES, tables=PINNED[:1]), None),
    (_holding(READ_VALUES, boundary=NO_RECORD), "row"),
    (_holding(READ_VALUES, boundary=NO_RECORD, tables=PINNED[1:]), "row"),
    (_holding(READ_VALUES, boundary=NO_RECORD, supplied=TWO_TABLES, tables=PINNED[:1]), "row"),
    (_holding(READ_VALUES, boundary=NO_RECORD, tables=(("t", {**PINNED[0][1], "root": "c3" * 64}),)), "row"),
    (_holding(READ_VALUES, boundary=NO_RECORD, tables=(("t", {**PINNED[0][1], "domain": U_DOMAIN}),)), "row"),
    (_holding(READ_VALUES, boundary=NO_RECORD, tables=(("t", {**PINNED[0][1], "leaves": 16}),)), "row"),
    (_holding(READ_VALUES, boundary=NO_RECORD, tables=(("t", {**PINNED[0][1], "leaves": "8"}),)), "row"),
    (_holding(READ_VALUES, boundary=NO_RECORD, tables=PINNED, rows=READ_ROWS[:1]), "row"),
    (_holding(READ_VALUES, boundary=NO_RECORD, tables=PINNED, reads=()), "row"),
    (_holding(READ_VALUES[1:], READ_LINKS[1:], boundary=NO_RECORD, tables=PINNED), "row"),
])
def test_round_two_takes_a_reads_table_from_the_tables_the_auditors_params_pin(own, value):
    """`second_stage._one_table` beside round 1's record: a read's root and domain may be one table's of the tables
    the auditor's params pin before any coin (`Own.tables`, the tables' registration's entries), so a read's row,
    salt and path pass with round 1's record registering no table, or with it registering one of two tables filled
    row by row and the params pinning the other. Refused, the row named, with `second_stage.INPUTS_WHY`: a table
    neither pinned nor registered, a pin whose root, domain id or count of leaves is another than the one the verifier
    filled, and a pin that leaves the read's other conditions unmet (its rows, its record as a read, its key)."""
    assert SS.inputs_failure(own) == value
    found = dict(SS._interior_failures(own)).get("second-inputs")
    assert found == (None if value is None else {"value": value, "why": SS.INPUTS_WHY})


def test_the_params_pin_registered_tables_beside_the_circuit_files():
    """`template_circuits` and `pinned_tables`: the auditor's params may add ``"tables"``, the tables' registration's
    entries by parameter, which it pinned before any coin; the verifier takes them in the parameters' order and the
    circuit files as without them, and refuses an entry not exactly the docket's form (`service._table_entry`)
    and tables that are not an object."""
    entry = dict(PINNED[0][1])
    pinned = {**_circuits(), "tables": {"u": {**entry, "value": "u"}, "t": entry}}
    assert S.template_circuits(pinned, WINDOW_QUERY) == S.template_circuits(_circuits(), WINDOW_QUERY)
    assert S.pinned_tables(pinned) == (("t", entry), ("u", {**entry, "value": "u"}))
    assert S.pinned_tables(_circuits()) == () == S.pinned_tables(None)
    assert S.template_circuits({**_circuits(), "tables": {"t": {**entry, "word_bits": 8}}}, WINDOW_QUERY)
    for bad in ({**entry, "value": "u"}, {**entry, "root": "AB" * 64}, {**entry, "domain": "ab" * 32},
                {**entry, "leaves": 0}, {**entry, "words": True}, {**entry, "word_bits": 0}, {**entry, "x": 1},
                {k: v for k, v in entry.items() if k != "words"}, [entry]):
        with pytest.raises(ValueError, match="a pinned table's entry is exactly"):
            S.template_circuits({**_circuits(), "tables": {"t": bad}}, WINDOW_QUERY)
    for bad in ([entry], "t", None):
        with pytest.raises(ValueError, match="the params' proofs are"):
            S.template_circuits({**_circuits(), "tables": bad}, WINDOW_QUERY)


def test_a_read_at_a_key_the_block_computes_is_refused_and_says_so():
    """A read whose key is wired to another template's output reads a key the block computes, which no statement
    covers: its row is refused (`second-inputs`), with a reason of its own (`second_stage.KEY_WHY`), where the same read
    at a key of round 1's record passes, and one with no key at all is refused for the general reason."""
    wired = _holding(READ_VALUES[1:], READ_LINKS[1:], wires=[SS.Wire(READ_T, "tok", "W", "o0")])
    assert SS.inputs_failure(wired) == "row"
    assert dict(SS._interior_failures(wired))["second-inputs"] == {"value": "row", "why": SS.KEY_WHY}
    both = _holding(READ_VALUES, wires=[SS.Wire(READ_T, "tok", "W", "o0")])
    assert dict(SS._interior_failures(both))["second-inputs"] == {"value": "row", "why": SS.KEY_WHY}
    linked = _holding(READ_VALUES[1:], [*READ_LINKS[1:], SS.Link(READ_T, "tok", "b.ingress")])
    assert SS.inputs_failure(linked) is None
    unkeyed = _holding(READ_VALUES[1:], READ_LINKS[1:])
    assert dict(SS._interior_failures(unkeyed))["second-inputs"] == {"value": "row", "why": SS.INPUTS_WHY}
    assert SS.KEY_WHY != SS.INPUTS_WHY and "no statement covers it" in SS.KEY_WHY


#: One MoE layer's read of its expert's row: the router (`ROUTER_T`), the window's first member, writes each token's
#: expert index at its output `o0`, a row of two u16 words, and the read's key, of two words too, is wired to it, each
#: of the read's three units a block reading its router unit's row.
ROUTER_T = "Router_v1"
KEY_WIRE = SS.Wire(READ_T, "tok", ROUTER_T, "o0")
KEY_ROWS = SS.Rows(READ_T, "tok", 3, (SS.Run(0, 3, 0, 1),))
KEY_WIDTH = SS.Width(READ_T, "tok", 2, 2)


def _keyed(on: bool = True, **change) -> SS.Own:
    """What round 2 holds of the MoE layer's read, its row registered and its key wired (`KEY_WIRE`), with the wired
    key's switch `on`, and `change` made."""
    own = _holding(READ_VALUES[1:], READ_LINKS[1:], wires=[KEY_WIRE], rows=(*READ_ROWS, KEY_ROWS))
    return dataclasses.replace(own, **{"parts": (SS.Part(ROUTER_T, 0, 3), SS.Part(READ_T, 3, 3)),
                                       "widths": (KEY_WIDTH,), "wired_key": on, **change})


def test_the_wired_keys_switch_is_off_and_refuses_as_before():
    """The switch is a constant of the code, off (`second_stage.WIRED_KEY`): round 2's `Own` holds it by default, so
    the service's run, which never sets it, refuses the MoE layer's wired key exactly as before, with `KEY_WHY`."""
    assert SS.WIRED_KEY is False
    own = dataclasses.replace(_keyed(), wired_key=SS.Own((), None, None, None, ()).wired_key)
    assert own.wired_key is False
    assert SS.inputs_failure(own) == "row"
    assert dict(SS._interior_failures(own))["second-inputs"] == {"value": "row", "why": SS.KEY_WHY}


@pytest.mark.parametrize("change,why", [
    ({}, None),
    ({"wires": (KEY_WIRE, SS.Wire(READ_T, "tok", ROUTER_T, "o1"))}, SS.KEY_WIRES_WHY),
    ({"links": (*READ_LINKS[1:], SS.Link(READ_T, "tok", "b.ingress"))}, SS.KEY_LINKED_WHY),
    ({"widths": (SS.Width(READ_T, "tok", 2, 1),)}, SS.KEY_WIDTH_WHY),
    ({"widths": ()}, SS.KEY_WIDTH_WHY),
    ({"parts": (SS.Part(READ_T, 0, 3), SS.Part(ROUTER_T, 3, 3))}, SS.KEY_ORDER_WHY),
    ({"parts": (SS.Part(READ_T, 0, 3),)}, SS.KEY_ORDER_WHY),
    ({"rows": (*READ_ROWS, SS.Rows(READ_T, "tok", 3, (SS.Run(0, 3, 1, 1),)))}, SS.KEY_BOUNDS_WHY),
    ({"rows": READ_ROWS}, SS.KEY_BOUNDS_WHY),
    ({"parts": (SS.Part(ROUTER_T, 0, 2), SS.Part(READ_T, 2, 3))}, SS.KEY_BOUNDS_WHY),
])
def test_with_the_switch_on_a_wired_key_passes_only_whole(change, why):
    """With the wired key's switch on, the MoE layer's read, its key one wire from the router's output, passes
    (`second_stage._key_failure`); and its row is refused (`second-inputs`), each with a reason of its own, where its key
    has two wires, is linked too, is wired to a row of other words or of no width the design derives, to a writer not
    before it in the window or of no part, or where its rows give a unit a writer row past the writer's units, or give
    the key no rows. Switched off, each is refused for the one reason, `KEY_WHY`."""
    own = _keyed(**change)
    assert SS.inputs_failure(own) == (None if why is None else "row")
    assert dict(SS._interior_failures(own)).get("second-inputs") == (None if why is None else {"value": "row",
                                                                                               "why": why})
    assert len({SS.KEY_WIRES_WHY, SS.KEY_LINKED_WHY, SS.KEY_WIDTH_WHY, SS.KEY_ORDER_WHY, SS.KEY_BOUNDS_WHY,
                SS.KEY_WHY, SS.INPUTS_WHY}) == 7
    off = dataclasses.replace(own, wired_key=False)
    assert dict(SS._interior_failures(off))["second-inputs"] == {"value": "row", "why": SS.KEY_WHY}


def test_a_read_whose_key_is_wired_lists_no_table_rows():
    """A read whose key is wired lists no ``table_rows`` at a port of its opening (its key, row, salt, path or verdict):
    the rows it reads are a function of a committed value, so listing them would disclose it. Round 2 refuses such an
    entry with ``second-table-rows`` and `KEY_ROWS_WHY`, the switch on or off. It reads no ``table_rows`` yet, so one
    listed anywhere else is refused too, with `TABLE_ROWS_WHY`; an entry that lists none, or lists null, passes."""
    listed = {"value": "row", "table_rows": [0, 2]}

    def stmt(port: str, e: dict) -> dict:
        return {"template": READ_T, "header": {}, "draw": {}, "registered": {"ports": {port: e}}}

    for own in (_keyed(), _keyed(on=False)):
        assert all(SS.table_rows_why(own, READ_T, p, listed) == SS.KEY_ROWS_WHY for p in SS.KEYED_PORTS)
        assert SS.table_rows_why(own, READ_T, "root", listed) == SS.TABLE_ROWS_WHY
        assert SS.table_rows_why(own, READ_T, "row", {"value": "row"}) is None
        assert SS.table_rows_why(own, READ_T, "row", {"value": "row", "table_rows": None}) is None
        assert ("second-table-rows", {"statement": 0, "template": READ_T, "port": "row", "why": SS.KEY_ROWS_WHY}) in \
            SS._failures(own, own.parts, [stmt("row", listed)])
        assert not any(c == "second-table-rows" for c, _ in SS._failures(own, own.parts, [stmt("row", {})]))
    unwired = dataclasses.replace(_keyed(), wires=())
    assert SS.table_rows_why(unwired, READ_T, "row", listed) == SS.TABLE_ROWS_WHY
    assert SS.KEYED_PORTS == ("tok", "row", "salt", "path", "o1")


def test_nothing_of_record_turns_the_wired_key_on():
    """No file under `verity/` sets the wired key's switch but round 2's own code and its examples: the service's run,
    the verifier, every executable and every script take the switch as the code holds it, off."""
    root = next(p for p in Path(__file__).resolve().parents if (p / "verity" / "__init__.py").is_file())
    spot = root / "verity/protocols/proof/spot/verifier"
    allowed = {spot / "second_stage.py", spot / "lean/Spot/SecondStage.lean", spot / "lean/Spot/SecondStageExamples.lean"}
    sets = []
    skip = {".lake", ".git", "__pycache__", "target", "node_modules", ".venv", "fixtures"}
    for top, dirs, files in os.walk(root / "verity"):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            path = Path(top) / f
            if path.suffix not in {".py", ".lean", ".sh", ".toml"}:
                continue
            text = path.read_text(errors="replace")
            if any(s in text for s in ("wired_key=", "wired_key = ", "wiredKey :=")) and path not in allowed:
                sets.append(path)
    assert sets == [], sets


def test_a_one_stage_stream_has_no_interior_registration():
    spec = S.Spec(VALUES[:2], S.SamplingDesign("rate:1/2"), S.Check(P.program_digest(Program(BLOCK)), GRAIN, proof_deadline=60.0),
                  PUBLIC, b"two-stage-test")
    stream = FC.Stream(spec, BLOCK, budget=2, delta=DELTA, challenger="challenger-test", clock=Clock())
    reg = stream.register(*_boundary())
    first = stream.select(reg)
    assert _refused(lambda: stream.register_interiors(first, *_interior(Q * N_RHO, reg.digest, D.proved(first.draw)))) == \
        ("rule",)
    assert _refused(lambda: stream.outcome(first, _lean(), {"statement": "stub"}, statements=[])) == ("statements",)


def test_the_stream_holds_the_design_to_its_replay_block():
    one = bind(CompositeDefinition("TwoStageTestOne", 0, (), lambda S_: ((("x", Array(1, Array(2, Bits(8)))),),
                                                                       Array(1, Bits(8))),
                                   lambda B, S_, x: B.batch(PAIR, x, axes=(0,)), register=False))
    with pytest.raises(ValueError, match="replay="):
        _stream(replay=None)
    with pytest.raises(ValueError, match="its rule is 'law'"):
        _stream(spec=_spec(S.TwoStage(P.template_instance_query(one), "rate:1/2", K)), replay=one)
    with pytest.raises(ValueError, match="grain is not the template query"):
        _stream(spec=_spec(S.TwoStage(QUERY, "rate:1/2", K)))
    for k in (N_RHO, N_RHO + 1, {PAIR.id: N_RHO}):
        with pytest.raises(ValueError, match="whole replay blocks"):
            _stream(spec=_spec(S.TwoStage(GRAIN, "rate:1/2", k)))
    with pytest.raises(ValueError, match="name each of the check's templates once"):
        _stream(spec=_spec(S.TwoStage(GRAIN, "rate:1/2", {BLOCK.id: 1})))
    with pytest.raises(ValueError, match="one round"):
        _stream(rounds=2)
    with pytest.raises(ValueError, match="only a two-stage stream"):
        FC.Stream(S.Spec(VALUES[:2], S.SamplingDesign("rate:1/2"), _spec().check, PUBLIC, b"t"), PAIR, replay=BLOCK, budget=2,
                  delta=DELTA, challenger="challenger-test", clock=Clock())


def test_the_rule_and_the_spec_state_one_form():
    assert RULE.to_json() == {"rule": "two-stage", "grain": GRAIN, "replay": "rate:1/2", "k": K}
    assert S.TwoStage(GRAIN, "rate:1/3:2", 2).replay == "rate:1/3:2"
    assert S.TwoStage(GRAIN, "rate:1/2", {PAIR.id: 2}).to_json()["k"] == {PAIR.id: 2}
    for bad, match in (((dict(GRAIN, version=1), "rate:1/2", 1), "grain"), ((GRAIN, "stratified:2", 1), "replay law"),
                       ((GRAIN, "subset:2", 1), "replay law"), ((GRAIN, "stratified-rate:1/2", 1), "replay law"),
                       ((GRAIN, "work-rate:1/2", 1), "replay law"), ((GRAIN, "rate:1/2", 0), "k >= 1"),
                       ((GRAIN, "rate:1/2", True), "k >= 1"), ((GRAIN, "rate:1/2", {}), "k >= 1"),
                       ((GRAIN, "rate:1/2", {PAIR.id: 0}), "k >= 1"), ((GRAIN, "rate:1/2", {PAIR.id: True}), "k >= 1")):
        with pytest.raises(ValueError, match=match):
            S.TwoStage(*bad)
    for retired in (RULE.to_json(), {"rule": "two-stage", "replay": "subset:4", "check": "all"}, "two-stage:subset:4"):
        with pytest.raises(ValueError, match="a selection rule is All, Law or TwoStage"):
            _spec(selection=retired)
    assert _spec(values=VALUES[:2]).values == VALUES[:2]
    with pytest.raises(ValueError, match="boundary values"):
        _spec(values=VALUES[2:])
    with pytest.raises(ValueError, match="one-stage selection has none"):
        _spec(selection=S.SamplingDesign("rate:1/2"))
    assert S.selection_sampling_design(RULE, N) == {"law": "subset", "k": Q}
    assert S.boundary_values(VALUES) == VALUES[:2] and S.interior_values(VALUES) == VALUES[2:]


def test_the_verifier_samples_both_stages():
    """The verifier's executable (`Pieces.verifier`, `verifier.EXE`) samples a two-stage rule's first stage, `Q`
    of the `N` replay blocks, and its second, `k` of each sampled replay block's proof blocks, as the service does."""
    assert V.EXE == "spot-verifier" and _spec().selection == RULE
    source = random.Random(1).randbytes(4096)
    first = {"rule": RULE.to_json(), "program": codec.encode_program(PN.population_program(BLOCK, N)), "query": GRAIN,
             "work_table": None, "closure": None, "source": source.hex()}
    assert len(V.lean("select", first)["draw"]["units"]) == Q
    for ns, ks, cell in (((N_RHO,), (K,), lambda u: u // N_RHO), ((2, 1), (1, 1), lambda u: (u // 2, 0) if u < 4 else
                                                                   (u - 4, 1))):
        source = random.Random(2).randbytes(4096)
        drawn = V.lean("select_second", {"sample": [0, 2], "ns": list(ns), "ks": list(ks), "source": source.hex()})
        read = iter(source)
        assert drawn["draw"] == S.select_second([0, 2], ns, ks, lambda m: bytes(next(read) for _ in range(m)))
        assert sorted({cell(u) for u in drawn["draw"]["units"]}) == sorted({cell(u) for u in range(2 * sum(ns))})


def test_the_second_stage_samples_k_in_each_cell():
    """One stratum per cell, template by template and in each by sampled replay block, tiling the interior in order,
    each of the work sampling design's floor its count (cells' counts aren't in proportion to their sizes, as the stratified sampling design's
    are); a cell smaller than the rule's count is proved whole, and counts proving every cell whole are refused."""
    sampling_design = S.second_sampling_design([1, 3], (4,), (2,))
    assert sampling_design == {"law": "work", "k": 1,
                   "strata": [{"template": "1/0", "units": [[0, 4]], "work": 1, "floor": 2, "k": 2},
                              {"template": "3/0", "units": [[4, 8]], "work": 1, "floor": 2, "k": 2}]}
    assert D.strata_problem(sampling_design, 8) is None
    obj = S.select_second([1, 3], (4,), (2,), random.Random(5).randbytes)
    assert obj["population"] == 8 and len(obj["units"]) == 4
    assert sorted(u // 4 for u in obj["units"]) == [0, 0, 1, 1]
    sampling_design = S.second_sampling_design([1, 3], (3, 1), (2, 1))
    assert [(s["template"], s["units"], s["k"]) for s in sampling_design["strata"]] == [
        ("1/0", [[0, 3]], 2), ("3/0", [[3, 6]], 2), ("1/1", [[6, 7]], 1), ("3/1", [[7, 8]], 1)]
    assert D.strata_problem(sampling_design, 8) is None
    obj = S.select_second([1, 3], (3, 1), (2, 1), random.Random(6).randbytes)
    assert obj["population"] == 8 and {6, 7} <= set(obj["units"]) and len(obj["units"]) == 6
    for ns, ks in (((10, 1), (1, 1)), ((1, 9, 2), (1, 3, 2)), ((5, 5), (1, 4))):
        sampling_design = S.second_sampling_design([0, 2, 5], ns, ks)
        assert D.strata_problem(sampling_design, 3 * sum(ns)) is None
        stratified = {"law": "stratified", "k": 3 * sum(ks),
                      "strata": [{v: s[v] for v in ("template", "units", "k")} for s in sampling_design["strata"]]}
        assert "k is not the rule's" in D.strata_problem(stratified, 3 * sum(ns))
        obj = S.select_second([0, 2, 5], ns, ks, random.Random(7).randbytes)
        assert len(obj["units"]) == 3 * sum(ks)
    assert S.second_counts(2, ["a", "b"], (3, 1)) == (2, 1) and S.second_counts({"b": 1, "a": 3}, ["a", "b"], (3, 2)) == (3, 1)
    with pytest.raises(ValueError, match="name each"):
        S.second_counts({"a": 1}, ["a", "b"], (3, 2))
    for sample, ns, ks, match in (([], (4,), (1,), "ascending"), ([2, 1], (4,), (1,), "ascending"),
                                  ([0], (1,), (1,), "'law'"), ([0], (), (), "one or more templates'"),
                                  ([0], (2, 0), (1, 1), "at least one proof block"), ([0], (2, 1), (1,), "a count each"),
                                  ([0], (4,), (4,), "whole replay blocks"), ([0], (3, 1), (3, 1), "whole replay blocks"),
                                  ([0], (4,), (0,), "1 to n_t"), ([0], (4, 2), (1, 3), "1 to n_t")):
        with pytest.raises(ValueError, match=match):
            S.select_second(sample, ns, ks, lambda m: bytes(m))


def _over(r, x):
    """`x`, a replay block's input, with its first pair's second byte taken from replay block result `r`'s first byte."""
    return array_of([array_of([x[0][0], r[0]]), x[1], x[2]])


def _shared(B, S_, x):
    a = B.call(BLOCK, x[0])
    b = B.call(BLOCK, _over(a, x[1]))
    c = B.call(BLOCK, x[2])
    d = B.batch(BLOCK, array_of([_over(b, x[3]), x[4]]), axes=(0,))
    return tuple_of(a, b, c, d, B.call(BLOCK, _over(d[1], x[5])))


def _fanout_walk(program: Program, query) -> int:
    """The fan-out walked a proof block at a time, every batch member's own share: the most proof blocks one group joined
    by reads of each other's results holds.  The reference `partition.fanout`'s summary a root node at a time is held to;
    a verifier never runs it."""
    part = PO.evaluate(program, query)
    nodes = program.root.body.nodes
    first = {k: lo for k, _t, lo, _n in part.ranges}
    parent = list(range(part.population))

    def group(u: int) -> int:
        while parent[u] != u:
            u = parent[u]
        return u

    for k, _t, lo, n in part.ranges:
        node = nodes[k]
        axes = node.axes if node.form == "batch" else (None,) * len(node.args)
        for m in range(n):
            for a, ax in zip(node.args, axes):
                for i, a_lo, a_hi, _once in P._runs(member_refs(a, ax, m) if node.form == "batch" else a.refs):
                    if i in first:
                        w = nodes[i].fn.out_leaves
                        for v in range(first[i] + a_lo // w, first[i] + (a_hi - 1) // w + 1):
                            parent[group(v)] = group(lo + m)
    return max(Counter(group(u) for u in range(part.population)).values())


def _chain(B, S_, x):
    a = B.call(BLOCK, x[0])
    b = B.call(BLOCK, _over(a, x[1]))
    return tuple_of(a, b, B.call(BLOCK, x[2]), B.call(BLOCK, _over(b, x[3])))


def test_a_replay_block_reading_anothers_values_joins_its_group():
    """Fan-out (`partition.fanout`): replay blocks are joined when one reads part of another's result, directly or
    through others, and F is at least the most one group holds, summarized a root node at a time.  A population of
    closed replay blocks has F = 1.  Of calls alone, F is the walk's exactly: here the second and the fourth read the
    first's chain, groups {0, 1, 3} and {2}, F = 3.  Where a batch's members read other replay blocks in parts, the
    summary counts the whole batch: here the second reads the first's result and the fourth, a batch member, the
    second's, while the last reads the batch's other member, groups {0, 1, 3}, {2} and {4, 5} walked a member at a
    time (F = 3), and the summary's joined nodes hold 5.  F is a template query's."""
    assert P.fanout(PN.population_program(BLOCK, 4), GRAIN) == _fanout_walk(PN.population_program(BLOCK, 4), GRAIN) == 1
    o = Array(3, Bits(8))
    calls = Program(bind(CompositeDefinition("TwoStageTestChain", 0, (),
                                             lambda S_: ((("x", Array(4, Array(3, Array(2, Bits(8))))),),
                                                         Tuple(o, o, o, o)), _chain, register=False)))
    assert P.fanout(calls, GRAIN) == _fanout_walk(calls, GRAIN) == 3
    prog = Program(bind(CompositeDefinition("TwoStageTestShared", 0, (),
                                            lambda S_: ((("x", Array(6, Array(3, Array(2, Bits(8))))),),
                                                        Tuple(o, o, o, Array(2, o), o)), _shared, register=False)))
    assert P.units(prog, GRAIN) == 6 and _fanout_walk(prog, GRAIN) == 3 and P.fanout(prog, GRAIN) == 5
    cells, k, f = [O.Cell(3, 8.0)], {0: 1}, P.fanout(prog, GRAIN)
    charge = O.two_stage_exfil(cells, k, 6, .5, DELTA, 8.0, fanout=f)
    assert charge == O.two_stage_exfil(cells, k, 6, .5, DELTA, f * 8.0) > O.two_stage_exfil(cells, k, 6, .5, DELTA, 8.0)
    with pytest.raises(P.QueryRefused, match="template query's"):
        P.fanout(Program(PAIR), PO.call_query(16))


TRIPLE = bind(CompositeDefinition("TwoStageTestTriple", 0, (), lambda S_: ((("x", Array(3, Bits(8))),), Bits(8)),
                                  lambda B, S_, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
#: A replay block of two templates: two instances of PAIR, then one of TRIPLE.
MIXED = bind(CompositeDefinition("TwoStageTestMixed", 0, (),
                                 lambda S_: ((("x", Array(2, Array(2, Bits(8)))), ("y", Array(3, Bits(8)))),
                                             Tuple(Array(2, Bits(8)), Bits(8))),
                                 lambda B, S_, x, y: tuple_of(B.batch(PAIR, x, axes=(0,)), B.call(TRIPLE, y)),
                                 register=False))
WINDOW = (PAIR, TRIPLE)
WINDOW_QUERY = P.template_instances_query(WINDOW)
MIXED_GRAIN = P.template_instance_query(MIXED)
#: MIXED's boundary ports, its `x` and `y` and its two outputs named by its template: PAIR's `x` and `o0` are its `x`
#: and `o0`, TRIPLE's its `y` and `o1` (`second_stage.replay_links`).
MIXED_BOUNDARY = tuple(f"{codec.descriptor_id(MIXED)}/{p}" for p in ("x", "y", "o0", "o1"))


def _window_spec(k=K, query=WINDOW_QUERY, window=WINDOW) -> S.Spec:
    return S.Spec(_values(MIXED_BOUNDARY), S.TwoStage(MIXED_GRAIN, "rate:1/2", k),
                  S.Check(tuple(P.program_digest(Program(t)) for t in window), query, proof_deadline=60.0),
                  _public(MIXED_BOUNDARY), b"two-stage-window-test")


def _window_stream(spec=None, window=WINDOW, replay=MIXED) -> S.Stream:
    return FC.Stream(spec or _window_spec(), window, replay=replay, budget=2, delta=DELTA, challenger="challenger-test",
                     clock=Clock())


def test_round_two_names_what_a_replay_block_returns_and_refuses_an_interior_value_of_it():
    """What a replay block returns is committed before the first stage's sample. Its names in the boundary record are
    the block's output ports, one for each field of what it returns, named as the record names its ports
    (`second_stage.replay_returns`): each port a link names for the return is one of them. An interior docket
    that registers a value of one is refused by name (`second-return`, `returns_ok`), with the interior record's
    order; one that registers any other name, or a run whose return has no such name, is not."""
    for block, fields in ((BLOCK, 1), (MIXED, 2)):
        program, key = PN.population_program(block, N), codec.descriptor_id(block)
        assert SS.replay_returns(program, key, None) == tuple(f"o{j}" for j in range(fields))
        named = SS.replay_returns(program, key, {"roots": [{"port": f"{key}/x"}]})
        assert named == tuple(f"{key}/o{j}" for j in range(fields))
        assert {x.value for x in SS.replay_links(program, key, None) if x.port.startswith("o")} == {
            f"o{j}" for j in range(fields)}
        assert SS.replay_returns(program, "Other_v0", None) == ()
    assert SS.replay_returns(PN.population_program(MIXED, N), codec.descriptor_id(MIXED),
                             {"roots": [{"port": p} for p in MIXED_BOUNDARY]}) == MIXED_BOUNDARY[2:]

    def own(registered, returns=(BOUT,)) -> SS.Own:
        return SS.Own((0, 1), None, {"registered": registered}, {}, (), returns=returns)
    for registered, returns in (([{"value": "w"}, {"value": 5}, {}], (BOUT,)), (5, (BOUT,)), ([{"value": BOUT}], ()),
                                ([{"value": BX}], (BOUT,))):
        assert SS.returns_ok(own(registered, returns))
        assert "second-return" not in SS.check_twin(own(registered, returns), None).codes
    refused = own([{"value": "w"}, {"value": BOUT}])
    v = SS.check_twin(refused, None)
    assert not SS.returns_ok(refused) and "second-return" in v.codes and not v.ok, v
    assert v.to_json()["detail"]["second-return"] == SS.RETURN_WHY


def test_a_replay_block_of_two_templates_takes_a_count_per_template():
    """`k` per template: k_t of each cell of template t, capped at the cell's size; counts naming another template are
    refused, and so are counts proving every cell whole."""
    ids = [P.template_id(t) for t in WINDOW]
    stream = _window_stream(_window_spec({ids[0]: 1, ids[1]: 5}))
    assert stream.counts == (1, 1) and stream.spec.selection.to_json()["k"] == {ids[0]: 1, ids[1]: 5}
    with pytest.raises(ValueError, match="whole replay blocks"):
        _window_stream(_window_spec({ids[0]: 2, ids[1]: 1}))
    with pytest.raises(ValueError, match="name each of the check's templates once"):
        _window_stream(_window_spec({ids[0]: 1}))


def _circuits(window=WINDOW, files=("ab" * 64, "cd" * 64), header=REFERENCE, reads=(READS, READS)) -> dict:
    return {"templates": [{"template": P.template_id(t), "circuit": c, "header": header, "registered": r}
                          for t, c, r in zip(window, files, reads)]}


def test_the_params_name_each_templates_circuit_file_in_the_checks_order():
    """`template_circuits`: the auditor's params name each of the check's templates' circuit files by SHA-512 and its
    statements' reference header and registered reads, in the check's order, as (template, circuit, header,
    registered), for one template's check and a window's, the reads null for none; none names no file; any other form,
    a header that isn't an object, registered reads neither an object nor null or none given, and a list that isn't
    the check's templates once each in its order, is refused."""
    ids = [P.template_id(t) for t in WINDOW]
    assert S.template_circuits(_circuits(reads=(READS, None)), WINDOW_QUERY) == (
        (ids[0], "ab" * 64, REFERENCE, READS), (ids[1], "cd" * 64, REFERENCE, None))
    assert S.template_circuits(_circuits((PAIR,)), QUERY) == ((ids[0], "ab" * 64, REFERENCE, READS),)
    assert S.template_circuits(None, WINDOW_QUERY) == ()
    entries = _circuits()["templates"]
    for proofs, match in (({"templates": entries[::-1]}, "not the check's templates"),
                          ({"templates": entries[:1]}, "not the check's templates"),
                          ({"templates": [*entries, entries[0]]}, "not the check's templates"),
                          ({"templates": [{**entries[0], "circuit": "AB" * 64}, entries[1]]}, "lowercase hex"),
                          ({"templates": [{**entries[0], "circuit": "ab" * 64 + "\n"}, entries[1]]}, "lowercase hex"),
                          ({"templates": [{**entries[0], "kind": "flat"}, entries[1]]}, "lowercase hex"),
                          ({"templates": [{**entries[0], "header": None}, entries[1]]}, "each header an object"),
                          ({"templates": [{**entries[0], "header": []}, entries[1]]}, "each header an object"),
                          ({"templates": [{k: v for k, v in entries[0].items() if k != "header"}, entries[1]]},
                           "each header an object"),
                          ({"templates": [{**entries[0], "registered": []}, entries[1]]},
                           "each registered an object or null"),
                          ({"templates": [{**entries[0], "registered": "r"}, entries[1]]},
                           "each registered an object or null"),
                          ({"templates": [{k: v for k, v in entries[0].items() if k != "registered"}, entries[1]]},
                           "each registered an object or null"),
                          ({"templates": entries, "circuit": "ab" * 64}, "proofs are"), ({}, "proofs are")):
        with pytest.raises(ValueError, match=match):
            S.template_circuits(proofs, WINDOW_QUERY)
    with pytest.raises(ValueError, match="a template query"):
        S.template_circuits(_circuits(), {"name": "Q_word", "version": 1, "params": {"X": 16, "W": 32}})


def test_the_params_record_a_template_as_a_table_read_beside_its_circuit_file():
    """`template_reads`: an entry of the auditor's params may record its template as a table read, `"read": {"V", "H"}`,
    beside the circuit file's SHA-512: the record that the template check accepted that file as the read of a table of
    V rows of H words. The verifier takes the templates so recorded, each with its rows, and a template with no record
    as no read; `template_circuits` gives the same entries with or without the record, and refuses a record that is
    not two positive counts."""
    ids = [P.template_id(t) for t in WINDOW]
    entries = _circuits()["templates"]
    read = {"templates": [{**entries[0], "read": {"V": 8, "H": 2}}, entries[1]]}
    assert S.template_circuits(read, WINDOW_QUERY) == S.template_circuits(_circuits(), WINDOW_QUERY)
    assert S.template_reads(read) == ((ids[0], 8),)
    both = {"templates": [{**entries[0], "read": {"V": 8, "H": 2}}, {**entries[1], "read": {"V": 131072, "H": 576}}]}
    assert S.template_reads(both) == ((ids[0], 8), (ids[1], 131072))
    assert S.template_reads(_circuits()) == () and S.template_reads(None) == ()
    for bad in (None, {}, {"V": 8}, {"V": 0, "H": 2}, {"V": 8, "H": 0}, {"V": True, "H": 2}, {"V": "8", "H": 2},
                {"V": 8.0, "H": 2}, {"V": -1, "H": 2}, {"V": 8, "H": 2, "x": 1}, [8, 2]):
        with pytest.raises(ValueError, match="each read two positive counts"):
            S.template_circuits({"templates": [{**entries[0], "read": bad}, entries[1]]}, WINDOW_QUERY)


@pytest.mark.parametrize("partition", [None, {"blocks": [[[0, 3]]], "format": "verity/partition/v1"}])
def test_the_interiors_partition_reaches_the_leans_run(monkeypatch, partition):
    """`verifier.run` hands the Lean's run the partition object the stream held the interior docket's digest to
    (`register_interiors`' ``partition``), None for the blockless one, so its R2 holds the docket to the same object."""
    seen = []
    monkeypatch.setattr(V, "lean", lambda f, value: seen.append(value) or {"refused": "stub"})
    program = PN.population_program(PAIR, N_RHO)
    interior = {"program": program, "query": QUERY, "law": {"law": "subset", "k": N_RHO}, "copies": ({}, None, [], [])
                + (None,) * 7, "replay": program, "values": [], "record": {}, "received": "0", "read": b"", "coins": None,
                "proved_units": None, "instances": None, "served": None, "session_roots": None, "partition": partition}
    V.run({}, {}, program, QUERY, {"law": "subset", "k": 1}, "s", {}, None, [], [], None, None, {}, [], [], "c", "0", b"",
          None, [], {}, [], {}, 0.5, interior=interior)
    assert seen[0]["interior"]["params"]["partition"] == partition
