"""Served trees as registered ports (`one_stage.served.register_served`): the registration it builds passes R7 and the file
is `reads_file`'s, an earlier output keeps its port, and every malformed tree is refused."""
from __future__ import annotations

import dataclasses
import json

import pytest

import verity.ml.circuits.ops.scalar  # noqa: F401
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Value
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G, registration as R
from verity.core.protocols.sampled_proofs.one_stage.served import ServedTree, register_served

XOR8 = PrimitiveDefinition("ServedTestXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)
MIX = bind(CompositeDefinition("ServedTestMix", 0, (), lambda S: ((("x", Array(2, Value(8))), ("w", Array(2, Value(8)))), Value(8)),
                               lambda B, S, x, w: B.call(XOR8, B.call(XOR8, x[0], w[0]), B.call(XOR8, x[1], w[1])), register=False))
#: the row of the served tree W each unit reads
POS = (3, 3, 0, 2, 1, 3)
TABLE = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
SCHEME = "frame-v3-sha512/hm96-sha512"
LAW = {"law": "subset", "k": 2}
SERVED = {"kind": "served", "row": 7, "run": "r-served"}
Q = P.template_instance_query(MIX.id)


def _program(pos=POS, rows: int = len(TABLE), unread: bool = False) -> Program:
    params = (("x", Array(len(pos), Array(2, Value(8)))), ("W", Array(rows, Array(2, Value(8)))))
    params += (("V", Array(1, Array(2, Value(8)))),) if unread else ()

    def sig(S):
        return params, Array(len(pos), Value(8))

    def body(B, S, x, W, *_):
        return coll_concat(sig(S)[1], [B.call(MIX, x[i], W[p]) for i, p in enumerate(pos)])

    return Program(bind(CompositeDefinition("ServedTestProgram", 0, (), sig, body, register=False)))


def _ports(prog: Program) -> dict:
    part = P.partition(prog, Q)
    n = P.units(prog, Q)
    return {p: {"leaves": leaves, "domain": R.served_domain(part["program"], P.digest(part), p, schema, leaves, "r-served")}
            for p, schema, leaves in (("x", "hm96-sha512/row/v1", n), ("out", "u16", n))}


def _record(prog: Program, entries, window=SERVED, previous=None, roots=None):
    ports = _ports(prog)
    roots = roots or [{"port": p, "root": "ab" * 64, **ports[p]} for p in ports]
    return R.record(prog, Q, LAW, SCHEME, roots, window, previous=previous, registered=entries), ports


def _tree(prog: Program) -> tuple[G.Registered, ServedTree]:
    reg = G.Registered.commit(P.program_digest(prog), "W", TABLE)
    return reg, ServedTree.of(reg)


def test_the_registration_it_builds_passes_r7_and_the_file_is_the_verifiers():
    """Round trip: the entries go into a v2 record that R7 accepts on the verifier's own copies (`values`, `outputs`), and
    the file is `reads_file` on the same program, query and roots, byte for byte as canonical JSON."""
    prog = _program()
    reg, tree = _tree(prog)
    sr = register_served(prog, Q, [tree])
    assert sr.registered == [reg.entry()] and sr.values == {"W": 2} and sr.outputs is None
    assert sr.roots == {"W": reg.root.hex()}
    rec, ports = _record(prog, sr.registered)
    assert R.check(rec, prog, Q, LAW, SCHEME, ports, registered=sr.values, outputs=sr.outputs).ok
    assert sr.file == G.reads_file(prog, Q, {"W": 2}, {"W": reg.root.hex()})
    assert sr.ports == {"w": {"value": "W", "root": reg.root.hex(), "leaves": 5, "words": 2, "word_bits": 16,
                              "domain": reg.domain.domain_id.hex(), "reads": list(POS)}}
    assert sr.file_bytes() == canonical_json(sr.file) and json.loads(sr.file_bytes()) == sr.file
    assert sr.partition == P.partition(prog, Q) and tree.row_schema == G.SCHEMA and "row_schema" not in tree.entry()
    assert register_served(prog, Q, [tree], ports={"w": "cs"}).ports.keys() == {"cs"}


def test_each_registered_read_opens_the_served_root(tmp_path):
    """The file's port is what flock-verify opens a statement's rows against: every row the program reads opens the served
    root under the file's domain, and `write` writes the bytes `flock-verify --registered` reads."""
    prog = _program()
    reg, tree = _tree(prog)
    sr = register_served(prog, Q, [tree])
    port = sr.ports["w"]
    dom = G.domain(P.program_digest(prog), "W", 2, 5)
    assert port["domain"] == dom.domain_id.hex()
    for r in port["reads"]:
        assert G.verify_read(dom, bytes.fromhex(port["root"]), r, reg.bcs[r], reg.path(r))
    assert sr.write(tmp_path / "registered.json").read_bytes() == sr.file_bytes()


MIX32 = bind(CompositeDefinition(
    "ServedTestMix32", 0, (), lambda S: ((("x", Array(2, Value(8))), ("w", Array(2, Value(32)))), Value(8)),
    lambda B, S, x, w: B.call(XOR8, x[0], x[1]), register=False))


def test_a_32_bit_tree_names_its_width():
    """A tree of 32-bit leaves carries `word_bits` in its entry and its port, and R7 holds it to the program's width."""
    def sig(S):
        return (("x", Array(len(POS), Array(2, Value(8)))), ("W", Array(len(TABLE), Array(2, Value(32))))), Array(len(POS), Value(8))

    prog = Program(bind(CompositeDefinition("ServedTestWide", 0, (), sig,
                                            lambda B, S, x, W: coll_concat(sig(S)[1], [B.call(MIX32, x[i], W[p]) for i, p in enumerate(POS)]),
                                            register=False)))
    q = P.template_instance_query(MIX32.id)
    reg = G.Registered.commit(P.program_digest(prog), "W", [[r * 0x10001, r] for r in range(5)], word_bits=32)
    tree = ServedTree.of(reg)
    sr = register_served(prog, q, [tree])
    assert sr.registered == [reg.entry()] and reg.entry()["word_bits"] == 32
    assert {(e["word_bits"], e["reads"] == list(POS)) for e in sr.ports.values()} == {(32, True)}
    part = P.partition(prog, q)
    ports = {"x": {"leaves": len(POS), "domain": R.served_domain(part["program"], P.digest(part), "x", "hm96-sha512/row/v1",
                                                                  len(POS), "r-served")}}
    rec = R.record(prog, q, LAW, SCHEME, [{"port": "x", "root": "ab" * 64, **ports["x"]}], SERVED, registered=sr.registered)
    assert R.check(rec, prog, q, LAW, SCHEME, ports, registered=sr.values).ok
    with pytest.raises(ValueError, match="16-bit leaves, and the program's parameter is 5 rows of 2 32-bit"):
        register_served(prog, q, [dataclasses.replace(tree, word_bits=16)])


def test_an_earlier_outputs_tree_keeps_its_port():
    """#2929: a tree that is an earlier record's port registers under that port's domain and root, R7 accepts it with the
    returned `outputs`, and the file opens the reads against the port's domain."""
    prog = _program()
    reg, _ = _tree(prog)
    part = P.partition(prog, Q)
    dom = R.served_domain(part["program"], P.digest(part), "prev.out", "hm96-sha512/row/v1", len(TABLE), "r-earlier")
    first, ports = _record(prog, [reg.entry()])
    earlier = R.record(prog, Q, LAW, SCHEME, [*first["roots"], {"port": "prev.out", "root": "cd" * 64, "leaves": len(TABLE),
                                                               "domain": dom}], {**SERVED, "row": 6}, registered=[reg.entry()])
    out = ServedTree("W", "cd" * 64, len(TABLE), 2, dom, output=(R.digest(earlier), "prev.out"))
    sr = register_served(prog, Q, [out], log=[earlier])
    assert sr.outputs == {"W": {"record": R.digest(earlier), "port": "prev.out"}}
    assert sr.registered == [{**reg.entry(), "root": "cd" * 64, "domain": dom}]
    assert {e["domain"] for e in sr.ports.values()} == {dom} and {e["root"] for e in sr.ports.values()} == {"cd" * 64}
    rec = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 8}, previous=R.digest(earlier),
                   registered=sr.registered)
    assert R.check(rec, prog, Q, LAW, SCHEME, ports, registered=sr.values, outputs=sr.outputs, log=[earlier]).ok
    for bad, why in ((dataclasses.replace(out, root="ce" * 64), "not port 'prev.out'"),
                     (dataclasses.replace(out, output=("ee" * 64, "prev.out")), "the log holds no record"),
                     (dataclasses.replace(out, output=(R.digest(earlier), "nope")), "the log holds no record")):
        with pytest.raises(ValueError, match=why):
            register_served(prog, Q, [bad], log=[earlier])
    with pytest.raises(ValueError, match="no earlier output"):
        register_served(prog, Q, [dataclasses.replace(out, output=None)], log=[earlier])


@pytest.mark.parametrize("change, why", [
    ({"root": "AB" * 64}, "root is not lowercase hex of 64 bytes"),
    ({"root": "ab" * 63}, "root is not lowercase hex of 64 bytes"),
    ({"root": "ab" * 63 + "a\n"}, "root is not lowercase hex of 64 bytes"),
    ({"root": b"\xab" * 64}, "root is not lowercase hex of 64 bytes"),
    ({"domain": "00" * 32}, "domain is not lowercase hex of 64 bytes"),
    ({"domain": "00" * 64}, "registered-value domain"),
    ({"leaves": 0}, "a tree has at least one"),
    ({"leaves": True}, "a tree has at least one"),
    ({"leaves": 4}, "the program's parameter is 5 rows"),
    ({"words": 0}, "a leaf has at least one"),
    ({"word_bits": 8}, "not 16 or 32"),
    ({"word_bits": 32}, "the program's parameter is 5 rows of 2 16-bit"),
    ({"value": ""}, "names the parameter"),
    ({"value": "nope"}, "not a parameter of the program"),
    ({"output": ("ab" * 64,)}, "an output is"),
    ({"row_schema": "hm96-sha512/row/v2"}, "the registered form reads hm96-sha512/row/v1 rows alone"),
    ({"row_schema": "hm96-sha512/row-seg/v1"}, "the registered form reads hm96-sha512/row/v1 rows alone"),
    ({"row_schema": "u16"}, "not one of"),
])
def test_a_malformed_tree_is_refused(change, why):
    prog = _program()
    _, tree = _tree(prog)
    with pytest.raises(ValueError, match=why):
        register_served(prog, Q, [dataclasses.replace(tree, **change)])


def test_a_value_twice_no_tree_or_one_no_unit_reads_is_refused():
    prog = _program()
    _, tree = _tree(prog)
    with pytest.raises(ValueError, match="a value twice"):
        register_served(prog, Q, [tree, dataclasses.replace(tree, root="cd" * 64)])
    with pytest.raises(ValueError, match="no served tree"):
        register_served(prog, Q, [])
    with pytest.raises(ValueError, match="a ServedTree"):
        register_served(prog, Q, [tree.entry()])
    both = _program(unread=True)
    w = ServedTree.of(G.Registered.commit(P.program_digest(both), "W", TABLE))
    v = ServedTree.of(G.Registered.commit(P.program_digest(both), "V", TABLE[:1]))
    with pytest.raises(ValueError, match=r"served trees \['V'\]: no unit of the population reads them"):
        register_served(both, Q, [w, v])


def test_a_tree_the_log_registered_under_another_root_is_refused():
    """R7-held: a value an earlier record of the log registered keeps that root, so a later window's tree under another root
    is refused here, as R7 would refuse its record."""
    prog = _program()
    reg, tree = _tree(prog)
    first, _ = _record(prog, [reg.entry()])
    assert register_served(prog, Q, [tree], log=[first]).roots == {"W": reg.root.hex()}
    _, other = _tree(prog)
    with pytest.raises(ValueError, match="R7-held"):
        register_served(prog, Q, [other], log=[first])
