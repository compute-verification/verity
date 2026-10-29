"""Served trees as registered ports (`one_stage.served.register_served`): the registration it builds passes R7 and the file
is `reads_file`'s, an earlier output keeps its port, a reused tree its earlier entry, and every malformed tree is
refused."""
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
#: the row of the served tree W each proof block reads
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
    return {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, schema, leaves, "r-served")}
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
    ports = {"x": {"leaves": len(POS), "domain": R.window_domain(part["program"], P.digest(part), "x", "hm96-sha512/row/v1",
                                                                  len(POS), "r-served")}}
    rec = R.record(prog, q, LAW, SCHEME, [{"port": "x", "root": "ab" * 64, **ports["x"]}], SERVED, registered=sr.registered)
    assert R.check(rec, prog, q, LAW, SCHEME, ports, registered=sr.values).ok
    with pytest.raises(ValueError, match="16-bit leaves, and the program's parameter is 5 rows of 2 32-bit"):
        register_served(prog, q, [dataclasses.replace(tree, word_bits=16)])


def test_an_earlier_outputs_tree_keeps_its_port():
    """A tree that is an earlier record's port registers under that port's domain and root, R7 accepts it with the
    returned `outputs`, and the file opens the reads against the port's domain."""
    prog = _program()
    reg, _ = _tree(prog)
    part = P.partition(prog, Q)
    dom = R.window_domain(part["program"], P.digest(part), "prev.out", "hm96-sha512/row/v1", len(TABLE), "r-earlier")
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


def test_a_reused_tree_takes_an_earlier_records_registered_value():
    """A tree an earlier record registered as one of its values (H1's `a/<K>@<s>` as a Commit's AROW, say) registers under
    that entry's domain and root with no tree of its own; R7 accepts it with the returned `outputs`, and a tree whose root,
    domain or rows aren't the entry's, or which the log doesn't hold, is refused."""
    prog = _program()
    reg, _ = _tree(prog)
    dom = "da" * 64
    theirs = {**reg.entry(), "value": "a/16@0", "root": "cd" * 64, "domain": dom}
    first, ports = _record(prog, [reg.entry()])
    earlier = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 6}, registered=[theirs])
    tree = ServedTree("W", "cd" * 64, len(TABLE), 2, dom, reused=(R.digest(earlier), "a/16@0"))
    sr = register_served(prog, Q, [tree], log=[earlier], window_root="a7" * 64)
    assert sr.outputs == {"W": {"record": R.digest(earlier), "value": "a/16@0"}}
    assert sr.registered == [{**reg.entry(), "root": "cd" * 64, "domain": dom}]
    assert {e["domain"] for e in sr.ports.values()} == {dom} and {e["root"] for e in sr.ports.values()} == {"cd" * 64}
    rec = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 8}, previous=R.digest(earlier),
                   registered=sr.registered)
    assert R.check(rec, prog, Q, LAW, SCHEME, ports, registered=sr.values, outputs=sr.outputs, log=[earlier],
                   window_root="a7" * 64).ok
    for bad, why in ((dataclasses.replace(tree, root="ce" * 64), "not value 'a/16@0'"),
                     (dataclasses.replace(tree, domain="db" * 64), "not value 'a/16@0'"),
                     (dataclasses.replace(tree, reused=("ee" * 64, "a/16@0")), "the log holds no record"),
                     (dataclasses.replace(tree, reused=(R.digest(earlier), "nope")), "the log holds no record"),
                     (dataclasses.replace(tree, reused=(R.digest(earlier), "a/16@0"), output=(R.digest(earlier), "x")),
                      "not both"),
                     (dataclasses.replace(tree, reused=("ab" * 64,)), "reused is")):
        with pytest.raises(ValueError, match=why):
            register_served(prog, Q, [bad], log=[earlier])
    short = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 6}, registered=[{**theirs, "words": 1}])
    with pytest.raises(ValueError, match="the log holds no record"):
        register_served(prog, Q, [dataclasses.replace(tree, reused=(R.digest(short), "a/16@0"))], log=[short])


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
    ({"row_schema": "hm96-sha512/row/v2"}, r"10 leaves are not rows of 2 \(4 of its leaves in a hm96-sha512/row/v2 row\)"),
    ({"row_schema": "hm96-sha512/row-seg/v1"}, r"not rows of 2 \(4 of its leaves in a hm96-sha512/row-seg/v1 row\)"),
    ({"row_schema": "u16"}, "not one of"),
])
def test_a_malformed_tree_is_refused(change, why):
    prog = _program()
    _, tree = _tree(prog)
    with pytest.raises(ValueError, match=why):
        register_served(prog, Q, [dataclasses.replace(tree, **change)])


def test_the_row_formats_are_the_commitment_layers():
    """`row_schema`'s values are the frame-v3-sha512 leaf schemas Flock's ports name (`Flock.HmRow.ROW_SCHEMA*`)."""
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2
    from verity.core.protocols.sampled_proofs.one_stage.served import SCHEMAS

    assert SCHEMAS == G.SCHEMAS == (SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_V2, SCHEMA_HM96_SHA512_ROW_SEG)


#: TABLE's rows packed as bit rows: each row's two 8-bit leaves in one u16 word, leaf 0 low (H1's A: K int8 codes, K/2 words)
PACKED = [[a | b << 8] for a, b in TABLE]


@pytest.mark.parametrize("schema", [G.SCHEMA_V2, G.SCHEMA_SEG])
def test_a_bit_row_tree_registers_in_its_format(schema):
    """A `row/v2` or `row-seg/v1` tree is a row of `words` u16 words holding as many of the parameter's leaves as their bits
    do: W's rows of two 8-bit leaves are one word each, so the tree is 5 rows of 1 word, its port names the format, and
    every read opens the served root under the format's leaves (`verify_read`), where the v1 file opens none."""
    prog = _program()
    reg = G.Registered.commit(P.program_digest(prog), "W", PACKED, schema=schema)
    tree = ServedTree.of(reg)
    assert (tree.row_schema, tree.leaves, tree.words) == (schema, 5, 1) and "row_schema" not in tree.entry()
    sr = register_served(prog, Q, [tree])
    assert sr.registered == [reg.entry()] and sr.values == {"W": 1} and sr.roots == {"W": reg.root.hex()}
    assert sr.file == G.reads_file(prog, Q, {"W": 1}, {"W": reg.root.hex()}, schemas={"W": schema})
    assert sr.ports == {"w": {"value": "W", "root": reg.root.hex(), "leaves": 5, "words": 1, "word_bits": 16,
                              "domain": reg.domain.domain_id.hex(), "row_schema": schema, "reads": list(POS)}}
    dom = G.domain(P.program_digest(prog), "W", 1, 5)
    assert reg.domain.domain_id == dom.domain_id
    for r in POS:
        assert G.verify_read(dom, reg.root, r, reg.bcs[r], reg.path(r), schema)
        assert not G.verify_read(dom, reg.root, r, reg.bcs[r], reg.path(r))


@pytest.mark.parametrize("schema", [G.SCHEMA_V2, G.SCHEMA_SEG])
@pytest.mark.parametrize("change, why", [
    ({"words": 2}, r"10 leaves are not rows of 2 \(4 of its leaves"),
    ({"words": 3}, r"10 leaves are not rows of 3 \(6 of its leaves"),
    ({"word_bits": 32}, "5 rows of 1 32-bit leaves, and the program's parameter is 5 rows of 1 16-bit leaves"),
    ({"leaves": 10}, "10 rows of 1 16-bit leaves, and the program's parameter is 5 rows"),
])
def test_a_bit_row_tree_whose_bits_are_not_the_parameters_rows_is_refused(schema, change, why):
    """A bit row is exactly its `words * word_bits` bits: a tree whose words or width make its rows other than W's
    two-leaf rows, or that counts its rows as v1 would (one 8-bit leaf a word), is refused."""
    prog = _program()
    tree = ServedTree.of(G.Registered.commit(P.program_digest(prog), "W", PACKED, schema=schema))
    with pytest.raises(ValueError, match=why):
        register_served(prog, Q, [dataclasses.replace(tree, **change)])


def test_r7_counts_a_bit_row_trees_rows_in_its_format():
    """The record's entry names no format: R7 counts a row's leaves in the format the verifier holds for the value (the
    returned `schemas`). It holds W's packed `row/v2` and `row-seg/v1` rows with them, and without them counts W as 10
    rows of one leaf (R7-leaves); a `row/v2` tree whose leaves are its words' width (32-bit leaves in 32-bit words) has
    the same rows either way."""
    prog = _program()
    for schema in (G.SCHEMA_V2, G.SCHEMA_SEG):
        sr = register_served(prog, Q, [ServedTree.of(G.Registered.commit(P.program_digest(prog), "W", PACKED, schema=schema))])
        assert sr.schemas == {"W": schema}
        rec, ports = _record(prog, sr.registered)
        assert R.check(rec, prog, Q, LAW, SCHEME, ports, registered=sr.values, schemas=sr.schemas).ok
        v = R.check(rec, prog, Q, LAW, SCHEME, ports, registered=sr.values)
        assert not v.ok and v.detail["R7-leaves"] == {"value": "W", "registered": 5, "verifier": 10}
        other = {G.SCHEMA_V2: G.SCHEMA_SEG, G.SCHEMA_SEG: G.SCHEMA_V2}[schema]
        assert R.check(rec, prog, Q, LAW, SCHEME, ports, registered=sr.values, schemas={"W": other}).ok
    _, tree = _tree(prog)
    assert register_served(prog, Q, [tree]).schemas == {}

    def sig(S):
        return (("x", Array(len(POS), Array(2, Value(8)))), ("W", Array(len(TABLE), Array(2, Value(32))))), Array(len(POS), Value(8))

    wide = Program(bind(CompositeDefinition("ServedTestWideV2", 0, (), sig,
                                            lambda B, S, x, W: coll_concat(sig(S)[1], [B.call(MIX32, x[i], W[p]) for i, p in enumerate(POS)]),
                                            register=False)))
    q = P.template_instance_query(MIX32.id)
    reg = G.Registered.commit(P.program_digest(wide), "W", [[r * 0x10001, r] for r in range(5)], word_bits=32, schema=G.SCHEMA_V2)
    sr = register_served(wide, q, [ServedTree.of(reg)])
    assert {e["row_schema"] for e in sr.ports.values()} == {G.SCHEMA_V2}
    part = P.partition(wide, q)
    ports = {"x": {"leaves": len(POS), "domain": R.window_domain(part["program"], P.digest(part), "x", "hm96-sha512/row/v1",
                                                                  len(POS), "r-served")}}
    rec = R.record(wide, q, LAW, SCHEME, [{"port": "x", "root": "ab" * 64, **ports["x"]}], SERVED, registered=sr.registered)
    assert R.check(rec, wide, q, LAW, SCHEME, ports, registered=sr.values).ok
    assert R.check(rec, wide, q, LAW, SCHEME, ports, registered=sr.values, schemas=sr.schemas).ok


@pytest.mark.parametrize("schemas, registered", [
    ({"W": "u16"}, {"W": 1}),
    ({"W": None}, {"W": 1}),
    ({"V": G.SCHEMA_V2}, {"W": 1}),
    ({"W": G.SCHEMA_V2}, None),
    ({"W": G.SCHEMA_V2}, {}),
    ([("W", G.SCHEMA_V2)], {"W": 1}),
])
def test_r7_refuses_a_row_format_of_another_form(schemas, registered):
    """A row format names a value the verifier expects registered and doesn't fill, in one of `registered.SCHEMAS`, as
    `OneStage.paramsOfJson` reads it."""
    prog = _program()
    rec, ports = _record(prog, [])
    with pytest.raises(ValueError, match="row format"):
        R.check(rec, prog, Q, LAW, SCHEME, ports, registered=registered, schemas=schemas)


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
    """R7-held: a value the verifier holds across windows that an earlier record of the log registered keeps that root, so
    a later window's tree under another root is refused here, as R7 would refuse its record."""
    prog = _program()
    reg, tree = _tree(prog)
    first, _ = _record(prog, [reg.entry()])
    assert register_served(prog, Q, [tree], log=[first], held=["W"]).roots == {"W": reg.root.hex()}
    _, other = _tree(prog)
    with pytest.raises(ValueError, match="R7-held"):
        register_served(prog, Q, [other], log=[first], held=["W"])


def test_a_fixed_name_served_again_without_the_windows_root_is_refused():
    """Fail closed: without the window's open root every window's domain is the same, so a value the log
    registered that the verifier doesn't hold is a fixed-name per-window value from its second window on, and both
    `register_served` and R7's verifier refuse their inputs, naming `window_root` and `held`, whatever the root: the first
    window's (a replay) or a fresh one. The first window, and a window whose values take fresh names, need no root."""
    prog = _program()
    reg, tree = _tree(prog)
    sr = register_served(prog, Q, [tree])
    first, ports = _record(prog, sr.registered)
    assert R.check(first, prog, Q, LAW, SCHEME, ports, registered=sr.values).ok
    _, fresh = _tree(prog)
    for again in (tree, fresh):
        with pytest.raises(ValueError, match=r"registered values \['W'\].*`window_root`.*`held`"):
            register_served(prog, Q, [again], log=[first])
    second, ports = _record(prog, sr.registered, window={**SERVED, "row": 8}, previous=R.digest(first))
    with pytest.raises(ValueError, match=r"registered values \['W'\].*`window_root`"):
        R.check(second, prog, Q, LAW, SCHEME, ports, log=[first], registered=sr.values)
    assert R.check(second, prog, Q, LAW, SCHEME, ports, log=[first], registered=sr.values, held=["W"]).ok


WINDOWS = ("a1" * 64, "b2" * 64)


def test_each_window_registers_its_own_trees_under_its_root_and_the_weights_once():
    """Under the window's open root a window-independent program's per-window tree (x) is served afresh every
    window under the same name, R7 accepts each window's record with the returned copies, and the weights (W, `held`)
    keep their window-independent domain and first root. A second window that serves the first window's x tree is
    refused: its domain is the first window's."""
    prog = _program()
    d = P.program_digest(prog)
    w = G.Registered.commit(d, "W", TABLE)
    xs = [G.Registered.commit(d, "x", [[i, i] for i in range(len(POS))], window=r) for r in WINDOWS]
    log: list = []
    for r, x in zip(WINDOWS, xs):
        sr = register_served(prog, Q, [ServedTree.of(x), ServedTree.of(w)], log=log, window_root=r, held=["W"])
        assert sr.window_root == r and sr.held == ["W"]
        assert sr.ports["x"]["domain"] == x.entry()["domain"] and sr.ports["w"]["domain"] == w.entry()["domain"]
        rec, ports = _record(prog, sr.registered, window={**SERVED, "row": 7 + len(log)},
                             previous=R.digest(log[-1]) if log else None)
        assert R.check(rec, prog, Q, LAW, SCHEME, ports, log=log, registered=sr.values, window_root=sr.window_root,
                       held=sr.held).ok
        log.append(rec)
    with pytest.raises(ValueError, match="registered-value domain"):
        register_served(prog, Q, [ServedTree.of(xs[0]), ServedTree.of(w)], log=log[:1], window_root=WINDOWS[1],
                        held=["W"])
    with pytest.raises(ValueError, match="R7-held"):
        register_served(prog, Q, [ServedTree.of(xs[1]), ServedTree.of(G.Registered.commit(d, "W", TABLE))], log=log[:1],
                        window_root=WINDOWS[1], held=["W"])
    with pytest.raises(ValueError, match="held values"):
        register_served(prog, Q, [ServedTree.of(xs[1])], window_root=WINDOWS[1], held=["W"])


def test_a_served_tree_cannot_hold_itself_across_windows():
    """Which values are held across windows is the verifier's (`held`), never the served tree's. A per-window tree
    committed under the window-independent domain, as if held, is refused under the window's root, and the `--registered`
    file names each port's domain as the verifier derived it, with no window root or held values of its own."""
    prog = _program()
    d = P.program_digest(prog)
    w = G.Registered.commit(d, "W", TABLE)
    unwindowed = G.Registered.commit(d, "x", [[i, i] for i in range(len(POS))])
    with pytest.raises(ValueError, match="registered-value domain"):
        register_served(prog, Q, [ServedTree.of(unwindowed), ServedTree.of(w)], window_root=WINDOWS[0], held=["W"])
    x = G.Registered.commit(d, "x", [[i, i] for i in range(len(POS))], window=WINDOWS[0])
    sr = register_served(prog, Q, [ServedTree.of(x), ServedTree.of(w)], window_root=WINDOWS[0], held=["W"])
    assert set(sr.file) == {"format", "ports"}
    assert all("held" not in e and "window_root" not in e for e in sr.ports.values())
