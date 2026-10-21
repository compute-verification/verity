"""Produced values (`one_stage.produced`): a field one unit outputs and another reads, committed after the draw under a domain
bound to the registration's digest; the verifier's reads from its own Program's cut between units; the post-draw record's
check; and the opened roots held to the record."""
from __future__ import annotations

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, coll_concat
from verity.ml.circuits.types import Array, Record, Tuple, Value
from verity.core.protocols.sampled_proofs.one_stage import audit, partition as P, produced as PR, registered as G

XOR16 = PrimitiveDefinition("ProducedTestXor16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: a ^ b,
                            register=False)
EQ16 = PrimitiveDefinition("ProducedTestEq16", 1, (("a", Value(16)), ("b", Value(16))), Value(1), lambda a, b: int(a == b),
                           register=False)
AND1 = PrimitiveDefinition("ProducedTestAnd1", 1, (("a", Value(1)), ("b", Value(1))), Value(1), lambda a, b: a & b,
                           register=False)
ROW = Array(2, Value(16))
N = 3
LANE = bind(CompositeDefinition(
    "ProducedTestLane", 0, (), lambda S: ((("x", ROW), ("k", ROW)), Record(("partial", ROW), ("ok", Value(1)))),
    lambda B, S, x, k: coll_concat(Record(("partial", ROW), ("ok", Value(1))),
                                   [B.call(XOR16, x[0], k[0]), B.call(XOR16, x[1], k[1]), B.call(EQ16, x[0], x[0])]),
    register=False))


def _combine_body(B, S, p, receipt):
    acc = [p[0][0], p[0][1]]
    for i in range(1, S.N):
        acc = [B.call(XOR16, acc[j], p[i][j]) for j in range(2)]
    return B.call(AND1, B.call(EQ16, acc[0], receipt[0]), B.call(EQ16, acc[1], receipt[1]))


COMBINE = CompositeDefinition("ProducedTestCombine", 0, ("N",), lambda S: ((("p", Array(S.N, ROW)), ("receipt", ROW)), Value(1)),
                              _combine_body, register=False)
REG = "ab" * 64
OTHER_REG = "cd" * 64
DRAW = "ef" * 64


def _program(order=None, partial_read=False) -> Program:
    order = list(order or range(N))

    def sig(S):
        return ((("x", Array(N, ROW)), ("k", Array(N, ROW)), ("receipt", ROW)),
                Tuple(*[LANE.signature()[1]] * N, Value(1)))

    def body(B, S, x, k, receipt):
        outs = [B.call(LANE, x[i], k[i]) for i in range(N)]
        ps = [outs[i].field("partial") for i in order]
        if partial_read:
            ps[0] = array_of([outs[0].field("partial")[0], outs[1].field("partial")[1]])
        ok = B.call(bind(COMBINE, N=N), array_of(ps), receipt)
        return coll_concat(sig(S)[1], [*outs, ok])

    return Program(bind(CompositeDefinition("ProducedTestLeaf", 0, (), sig, body, register=False)))


def _query():
    return P.template_instances_query([LANE.id, bind(COMBINE, N=N).id])


def test_the_reads_are_the_programs_cut_between_the_lanes_and_the_combine():
    prog = _program()
    reads = PR.reads(prog, _query(), {"partial": 2})
    assert reads == {"partial": {"value": "partial", "rows": [0, 1, 2, None]},
                     **{f"p[{i}]": {"value": "partial", "rows": [None, None, None, i]} for i in range(N)}}
    swapped = PR.reads(_program(order=[1, 0, 2]), _query(), {"partial": 2})
    assert [swapped[f"p[{i}]"]["rows"][N] for i in range(N)] == [1, 0, 2], "the combine reads the row its wire is"


def test_a_read_of_part_of_a_row_or_of_two_rows_is_refused():
    with pytest.raises(ValueError, match="no public read|not one whole produced row"):
        PR.reads(_program(partial_read=True), _query(), {"partial": 2})
    with pytest.raises(ValueError, match="no unit returns"):
        PR.reads(_program(), _query(), {"other": 2})
    with pytest.raises(ValueError, match="one row of"):
        PR.reads(_program(), _query(), {"partial": 3})


def test_the_produced_tree_opens_under_its_registrations_domain_only():
    prog = _program()
    pd = P.program_digest(prog)
    rows = [[1, 2], [3, 4], [5, 6]]
    tree = PR.commit(pd, "partial", rows, REG)
    mine = PR.entries(prog, _query(), {"partial": 2}, REG)
    assert tree.entry() == {"value": "partial", "root": tree.root.hex(), **{k: mine["partial"][k] for k in ("leaves", "words",
                                                                                                             "domain")}}
    dom = PR.domain(pd, "partial", 2, 3, 16, REG)
    for r in range(3):
        assert G.verify_read(dom, tree.root, r, tree.bcs[r], tree.path(r))
        assert not G.verify_read(PR.domain(pd, "partial", 2, 3, 16, OTHER_REG), tree.root, r, tree.bcs[r], tree.path(r))
        assert not G.verify_read(G.domain(pd, "partial", 2, 3), tree.root, r, tree.bcs[r], tree.path(r))
    assert not G.verify_read(dom, tree.root, 0, tree.bcs[1], tree.path(0)), "a swapped row opens at no other position"
    with pytest.raises(ValueError, match="registration digest"):
        PR.domain(pd, "partial", 2, 3, 16, "AB" * 64)
    with pytest.raises(ValueError, match="not both"):
        G.domain(pd, "partial", 2, 3, 16, window="00" * 64, produced=REG)


def test_the_reads_ports_name_the_lanes_output_row_and_the_combines_parameters():
    prog = _program()
    tree = PR.commit(P.program_digest(prog), "partial", [[1, 2], [3, 4], [5, 6]], REG)
    ports = PR.reads_ports(prog, _query(), {"partial": 2}, {"partial": tree.root.hex()}, REG,
                           ports={"partial": "o0", **{f"p[{i}]": f"p{i}" for i in range(N)}})
    assert sorted(ports) == ["o0", "p0", "p1", "p2"]
    assert all(e["root"] == tree.root.hex() and e["domain"] == tree.domain.domain_id.hex() and e["leaves"] == 3 for e in ports.values())
    assert ports["o0"]["reads"] == [0, 1, 2, None] and ports["p2"]["reads"] == [None, None, None, 2]


def _rec(prog, **kw):
    tree = PR.commit(P.program_digest(prog), "partial", [[1, 2], [3, 4], [5, 6]], kw.pop("registration", REG))
    return PR.record(REG, DRAW, [tree.entry()]), tree


def test_the_record_is_held_to_the_registration_the_draw_and_the_verifiers_entries():
    prog = _program()
    mine = PR.entries(prog, _query(), {"partial": 2}, REG)
    rec, _tree = _rec(prog)
    assert PR.check(rec, REG, DRAW, mine) == ()
    e = rec["produced"][0]
    other, _ = _rec(prog, registration=OTHER_REG)
    cases = [
        ({**rec, "x": 1}, "P-keys"), ({**rec, "format": "v1"}, "P-format"), ({**rec, "registration": OTHER_REG}, "P-registration"),
        ({**rec, "draw": REG}, "P-draw"), ({**rec, "produced": {}}, "P-produced"), ({**rec, "produced": [{**e, "x": 1}]}, "P-produced"),
        ({**rec, "produced": []}, "P-values"), ({**rec, "produced": [e, e]}, "P-values"),
        ({**rec, "produced": [{**e, "value": "q"}]}, "P-values"), ({**rec, "produced": [{**e, "root": "AB" * 64}]}, "P-root"),
        ({**rec, "produced": [{**e, "leaves": 4}]}, "P-leaves"), ({**rec, "produced": [{**e, "leaves": True}]}, "P-leaves"),
        ({**rec, "produced": [{**e, "words": 3}]}, "P-words"), ({**rec, "produced": [{**e, "word_bits": 16}]}, "P-word_bits"),
        ({**rec, "produced": [{**e, "word_bits": 32}]}, "P-word_bits"),
        ({**rec, "produced": other["produced"]}, "P-domain"), (None, "P-keys")]
    for bad, code in cases:
        assert code in PR.check(bad, REG, DRAW, mine), (bad, code)


def test_the_opened_roots_are_the_records():
    prog = _program()
    rec, tree = _rec(prog)
    v = lambda root: {"lean": {"of_record": True, "accepted": True,  # noqa: E731
                               "registered": {"o0": {"value": "partial", "root": root}, "p0": {"value": "partial", "root": root}}}}
    assert PR.opened(rec, audit.session_registered(v(tree.root.hex()))) == ()
    assert PR.opened(rec, audit.session_registered(v("00" * 64))) == ("P-opened",)
    assert PR.opened(rec, audit.session_registered({})) == ("P-opened",), "a produced value no verifier of record opened"
    two = {**v(tree.root.hex()), "lean2": {**v("00" * 64)["lean"]}}
    assert PR.opened(rec, audit.session_registered(two)) == ("P-opened",), "two roots for one value"
