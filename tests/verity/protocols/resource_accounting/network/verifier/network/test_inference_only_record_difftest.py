"""The inference-only policy's check of an execution phase's circuit record (`InferenceOnlyCheck.Record`) against the Python
reference: the Python half.

`verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts/record_difftest_vectors.py` writes 714 cases (600 random circuits,
two in five broken in one way, 94 shaped ones, most in the shape of a served class, and 20 toys of edges) and, on each,
`scripts/inference_only_record_reference.py`'s results, which visit every row of every unit. `check` runs the Lean half in its
Lean audit (`runs` in that package's `lean-audit.json`): `InferenceOnlyCheck/RecordDifftestMain.lean` fails unless
`Record`, `RecordCount`, `RecordRows` and `RecordChosen` give Python's results on every case, and unless SPOT's reader reads the
same wiring from the form's instance and from the one Python builds from its own layout of each template. The audit
checks the data against the record's SHA-256 and caches its pass on the Lean package's files alone, so the first test
is what says the data is still what the generator and the reference write. The second checks that the generator
reaches what its docstring says, that the published result is the judgment alone, and that a text the form reads
nothing of is refused: a record that lists its members, and one that names no Program. The third states the verdicts on
the served class's shape, the negative vectors among them, and the fourth the refusals of a pinned reader's layout:
one negative vector a role, each refused, and the same records under no pin. The fifth is that an edge's writer
output has its row's leaves: an edge from an output of other leaves, which a wire's leaves were refused for
(`leaf-widths`), is no Program's, refused with the Program (`program`). The sixth is the refusal that makes a row
what the form counts it as: a root that hands a unit a row that is not input gates of its own (`own-rows`), with
what the form would count without the refusal. The seventh is what the
form exports of a row: its unit's slot, and why it is held. The eighth is the check that no wire crosses slots: none
does at the served class's shape, and one that does is counted. The ninth sets the form beside
`Desc.evaluate`'s reference on the same Program: one block for the whole phase there and one a slot here,
and input bits apart by exactly the rows no gate reads. The tenth holds the reference to
`verity.commitments.periods.interface`, the rules a lowering states a record by: the stage that refuses a record, each unit's
slot, and `input_count`'s totals. The eleventh is that a row is declared once: the served shape's record with one
declaration more into a row it already declares is refused, whatever the two declarations are, and the second alone
is taken. The twelfth is that an output has a bit: a template that returns no field, a Tuple's or a Record's, or a field
of no bit, is refused with its templates, and the interface lays out no unit of it. The thirteenth is that a root node
of a composite is no input, whatever its key reads as: a template keyed as an input id is refused with its templates,
its record alone and a unit of it at the root that the record does not list. The fourteenth is a block's rows: on the
served shape the rows the reference adds up unit by unit are what the record's listing gives, a template's units in a
block times its rows of one unit, with the rate rule's two failures in rows and the refusal of a table that lacks a
template. The fifteenth is a history table: a table over no registered parameter, so every row read of one is fresh,
whatever pins and registers the rest. The sixteenth is what the prover chose of an execution phase's input bits: the
rows no declaration fills, the reads of a parameter that is not θ and the ports of an opening that is not held, over
every slot's units, with an opening's ports off the count only where the opening is held. The seventeenth is that a row
format is held to the registration log: a value read in leaves of another width than 16 bits is registered in leaves of
that width. The eighteenth is the row an opening may declare: the first of its reader's two outputs, the second
its verdict, of a reader of six input rows, by which nothing is filled or counted. The nineteenth is the opening whose
row is its verdict's own output: refused with the declarations, under its reader's pin and under none. The twentieth is
the edges (`EDGED`): each toy taken or refused at the stage it states, and a taken one's every unit counted in a block.

The tenth holds to `input_count` only where every opening it counts registered is one the form holds (`held`: the
reader's Definition pinned, with table reads of the table's root and domain id over the opening's units). Where one is
not, `input_count` counts that opening's ports registered and the form counts them fresh, and the test holds the two
apart by exactly those bits. The interface states no layout and no leaf's width: a record the form refuses for a
pinned reader's roles (`reader-layout`, `reader-roles`) or for a wire's leaves (`leaf-widths`) is one the interface
takes. It states no row format and does not read the registration log: it refuses a record that states row formats,
which the tenth leaves out (`_formatted`), and takes one the form refuses for its formats against the log (`formats`).
The form reads a history table's keys and its name and nothing of its writer, rows and runs: a record
whose history the interface refuses for those (`LOOSE`) is one the form takes.

The interface refuses a record with a `ValueError`, and the helper that asks it (`_interface`) takes nothing else for
its refusal: an error of any other type, the helper's own or the interface's, fails the test as an error. A test of
the helper holds it to that, on a helper that unpacks a pair from `Blocks.of`. It also states two records the form
refuses and the interface does not: one that states no blocks (`blocks`), since the interface reads a stated record's
blocks where it has them, and one with an opening of a reader that is no template of the circuit (`inputs`), which the
interface refuses or fails on with a `KeyError` of the reader's name.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from verity.circuits.program import input_id, is_input
from verity.circuits.types import Array, Value

LEAN = Path(__file__).resolve().parents[7] / "verity" / "protocols" / "workload_classification" / "training_detection" / "inference_only" / "verifier" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("record_difftest_vectors")
ref, IF, codec = vectors.ref, vectors.IF, vectors.codec
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["InferenceOnlyVerifier.RecordDifftestMain"]
#: the random cases the quick tests read, enough of them to reach every way of `BROKEN`
SEEDS = range(500)
#: the form's own refusals of a record `verity.commitments.periods.interface` takes: a wire's leaves, a pinned reader's
#: layout, and the row formats held to the registration log, which the interface does not read
OWN = ("leaf-widths", "formats", "reader-layout", "reader-roles")
#: the cases on a text the form reads nothing of, each with the stage that refuses it: a record that lists its
#: members, and the text of no wire and nothing else
LISTED, BARE = "2x3 a record that lists its members", "2x3 a text of no wire that names no Program"
TEXT = {LISTED: "wires", BARE: "templates"}
#: the cases on a row declared twice, each with the stage that refuses it
TWICE = {f"2x3 {what}": stage for what, (stage, *_) in vectors.TWICE.items()}
#: the cases on a template with an output that has no bit, each with the stage that refuses it
BITLESS = {f"2x3 {what}": "templates" for what in vectors.BITLESS}
#: the cases on a template keyed as an input id, its record alone and a unit of it the record does not list, each
#: with the stage that refuses it
KEYED = {f"2x3 {what}, {how}": "templates" for what in vectors.KEYED for how in ("alone", "unlisted")}
#: the cases on `registered.histories`, each with the stage that refuses it, None where the form takes it
HISTORIES = {f"2x3 {what}": stage for what, (stage, _) in vectors.HISTORIES.items()}
#: of those, the one the form takes that the interface refuses, and the one whose opening is against a history
LOOSE, OPENED = f"2x3 {vectors.LOOSE}", "2x3 the opening against a history"
#: the cases on a row format, each with the stage that refuses it, or None
FORMATS = {(f"2x3 {what}" if shape is None else what): stage for what, (stage, shape, _) in vectors.FORMATS.items()}
#: the cases on an opening that declares its row, each with the stage that refuses it, None where the form takes it
ROW_ROLE = {f"2x3 {what}": stage for what, (stage, _) in vectors.ROW_ROLE.items()} | {
    f"2x3 {vectors.ROW_UNPINNED}": vectors.ROW_ROLE[vectors.ROW_VERDICT][0]}
#: the cases of one template in two root nodes, each with the stage that refuses it, None where the form takes it
SPLITS = {f"2x3 {what}": ("templates" if short else None) for what, (_, short) in vectors.SPLITS.items()}
#: the case that reads an input node at another width, refused before the record
RECUT = {"a read at another width": "recut-unread"}


@pytest.fixture(scope="module")
def random_cases() -> list[dict]:
    return [vectors.case(s) for s in SEEDS]


@pytest.fixture(scope="module")
def shaped() -> dict[str, dict]:
    return {m["shape"]: m for m in vectors.shaped()}


@pytest.mark.slow
def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 714 cases and the reference's results, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{record_difftest/cases.jsonl}", "{record_difftest/python.jsonl}", str(vectors.total())]


def test_the_generator_covers_what_it_says(random_cases, shaped):
    made = [*random_cases, *shaped.values()]
    results = [vectors.results(m["case"]) for m in made]
    for m, r in zip(made, results, strict=True):
        detail = r["detail"]
        assert r["public"] == (detail if "error" in detail else {"complies": detail["complies"]})
        assert detail.get("error") == m["broken"], (m["shape"], m.get("how"))
        assert (m["case"]["spot"] is None) == (m["broken"] in vectors.EARLY)
        assert isinstance(r["kinds"], list) == isinstance(r["exported"], list) == ("error" not in detail)
        assert ("message" in detail) == (m["broken"] in ("own-rows", "leaf-widths", "recut-unread"))
    assert {m["broken"] for m in made} == set(vectors.BROKEN) | {None}
    assert {m["how"] for m in random_cases if m["broken"]} == {how for hows in vectors.BROKEN.values() for how in hows}
    good = [r for r in results if "error" not in r["detail"]]
    assert {r["detail"]["failure"] for r in good} == {None, "size", "nci"}
    assert {r["count"].get("failure") for r in good if "error" not in r["count"]} == {None, "gates", "fresh", "count"}
    assert {r["count"].get("error") for r in good} >= {None, "instance", "params", "gate-table"}
    assert {r["block_rows"].get("failure") for r in good if "error" not in r["block_rows"]} == {None, "size", "nci"}
    assert {r["block_rows"].get("error") for r in good} == {None, "instance", "params", "rows-table"}
    assert all(r["block_rows"]["rows"] > 0 for r in good if "error" not in r["block_rows"])
    zero = [r["block_rows"] for m, r in zip(made, results, strict=True) if "error" not in r["detail"]
            and isinstance(m["case"]["block_rows"].get("table"), dict) and 0 in m["case"]["block_rows"]["table"].values()]
    assert {z.get("error") for z in zero} == {"params"}
    assert {k for r in good for k in r["kinds"]} == set(ref.KINDS)
    for k in ("fresh", *ref.FRESH):
        assert any(r["detail"][k] > 0 for r in good), k
    assert any(r["detail"]["fresh"] == 0 for r in good)
    assert any(r["detail"]["blocks"] > 1 for r in good)
    for what, stage in TEXT.items():
        r = vectors.results(shaped[what]["case"])
        assert r["detail"] == r["public"] == r["kinds"] == r["exported"] == r["count"] == {"error": stage}, what
        assert r["block_rows"] == {"error": stage}, what
    # a record that lists its members, as the interface reads them back, states no wires: refused, not read as none
    listed = json.loads(shaped[LISTED]["case"]["instance"]["record"])
    assert "wires" not in listed and len(IF._listed_members(listed)) == len(listed["templates"])
    assert shaped[LISTED]["case"]["spot"]["wires"] is None
    # the text of no wire and nothing else names no Program: refused before its wires are read
    assert json.loads(shaped[BARE]["case"]["instance"]["record"]) == {"wires": []}


def test_the_served_shape(shaped):
    """The verdicts on a served class's shape: accepted with the weights registered at the start of the deployment, a
    block's fresh bits then its scan state's alone; refused by the rate rule, and by the count on its fresh bits,
    with none registered or committed, under another root, with an opening that reads no root, with an opening whose
    reader is another Definition under the pinned one's name, and with one wire from the next slot."""
    got = {what: vectors.results(m["case"]) for what, m in shaped.items()
           if what not in LAYOUT | ROWS | TEXT | TWICE | BITLESS | KEYED | HISTORIES | FORMATS | ROW_ROLE | SPLITS}
    verdicts = {what: (r["detail"]["failure"], r["count"]["failure"]) for what, r in got.items()}
    refused = ("nci", "fresh")
    assert verdicts == {
        "2x3 registered": (None, None), "4x35 registered": (None, None),
        "2x3 none registered": refused, "4x35 none registered": refused,
        "2x3 none committed": refused, "4x35 none committed": refused,
        "2x3 another root": refused, "2x3 no root read": refused, "2x3 a constant verdict": refused,
        "2x3 one wire from the next slot": refused,
        "2x3 the order beside the record": (None, None), "2x3 the run's record first": (None, None),
        "2x3 the key a 32-bit word, its writers' too": (None, None)}
    # in rows the rate rule decides each the same way: a size's cases list the same units, so a block's rows are
    # those of the accepted case, which the policy's `Γ` and `α` in rows hold exactly
    for what, r in got.items():
        held = got[f"{what.split()[0]} registered"]["block_rows"]
        assert r["block_rows"]["rows"] == held["rows"] > 0 and held["complies"], what
        assert r["block_rows"]["failure"] == r["detail"]["failure"], what
    for size, blocks in (("2x3", 2), ("4x35", 4)):
        held = got[f"{size} registered"]["detail"]
        assert held["blocks"] == blocks
        assert held["fresh"] == held["unregistered"] > 0
        assert held["cross_wires"] == held["free"] == held["ingress"] == held["openings"] == 0
        for what in ("none registered", "none committed"):
            none = got[f"{size} {what}"]["detail"]
            assert (none["gates"], none["inputs"]) == (held["gates"], held["inputs"])
            assert none["unregistered"] > held["unregistered"] and none["openings"] > 0
    held = got["2x3 registered"]["detail"]
    # the opening's ports are fresh, and nothing else moves, where its reader is not the pinned Definition or reads no root
    forged, rootless = got["2x3 a constant verdict"]["detail"], got["2x3 no root read"]["detail"]
    assert forged["openings"] > 0 and forged["unregistered"] == held["unregistered"]
    assert forged["fresh"] == held["fresh"] + forged["openings"]
    assert rootless["openings"] == forged["openings"]
    assert got["2x3 another root"]["detail"]["unregistered"] > held["unregistered"]
    forged_case, held_case = shaped["2x3 a constant verdict"]["case"], shaped["2x3 registered"]["case"]
    assert forged_case["instance"]["pins"] == held_case["instance"]["pins"]
    assert json.loads(forged_case["instance"]["record"])["registered"] == json.loads(
        held_case["instance"]["record"])["registered"]


#: the cases on a pinned reader's layout, at two slots and three positions: the stage that refuses each, or the rate
#: rule's and the count's verdicts
LAYOUT = {
    "2x3 the key not wired": ("nci", "fresh"),
    "2x3 the key among the ports": "reader-roles",
    "2x3 the key among the ports, no pin": ("nci", "fresh"),
    "2x3 another row the key": "reader-roles",
    "2x3 the root read in the domain id's row": "reader-roles",
    "2x3 the domain id read in the root's row": "reader-roles",
    "2x3 the root's and the domain id's rows swapped": "reader-roles",
    "2x3 an opened row left out": "reader-roles",
    "2x3 the opened rows in another order": "reader-roles",
    "2x3 another output the verdict": "reader-roles",
    "2x3 the key the second row, pinned": "reader-layout",
    "2x3 the key the second row, no pin": ("nci", "fresh"),
    "2x3 a reader of one output, pinned": "reader-layout",
    "2x3 a reader of one output, no pin": ("nci", "fresh"),
}


def test_a_pinned_readers_layout(shaped, monkeypatch):
    """A pin is a digest, and one layout by position says which rows of a pinned reader are its key, its table's root
    and domain id, its opened rows and its verdict (the form's `layoutOf`). The form refuses a record that names
    any other row or output of a pinned reader, one negative vector a role, and a pinned reader in no layout; it
    counts none of them. The first three cases are why: with the token of the embedding read left unwired, the 48 bits
    a block reads of it are free and the rate refuses; a record that then lists the token among the opening's ports
    would have them counted with the opened rows, registered, and it is refused; under no pin the same record is taken
    and every port of the opening is fresh. Whatever of these the form refuses, `verity.commitments.periods.interface` takes.
    The layout's stage alone refuses them: without it the reference refuses none, and accepts the token among the
    ports with a block's fresh bits those of the case that wires it."""
    assert {what for what, m in shaped.items() if m["broken"]} == {
            what for what, v in (LAYOUT | ROWS | TEXT | TWICE | BITLESS | KEYED | HISTORIES | FORMATS | ROW_ROLE | SPLITS | RECUT).items()
            if isinstance(v, str)}
    got = {what: vectors.results(shaped[what]["case"]) for what in LAYOUT}
    for what, want in LAYOUT.items():
        detail, count, m = got[what]["detail"], got[what]["count"], shaped[what]
        if isinstance(want, str):
            assert detail == count == got[what]["public"] == got[what]["kinds"] == {"error": want}, what
            assert shaped[what]["broken"] == want and _interface(m)[0] is None, what
        else:
            assert (detail["failure"], count["failure"]) == want, what
    held = vectors.results(shaped["2x3 registered"]["case"])["detail"]
    unwired, no_pin = got["2x3 the key not wired"]["detail"], got["2x3 the key among the ports, no pin"]["detail"]
    assert (unwired["free"], unwired["openings"]) == (48, 0) and unwired["fresh"] == held["fresh"] + 48
    assert (no_pin["free"], no_pin["fresh"]) == (0, held["fresh"] + no_pin["openings"])
    # the refused record differs from the one taken under no pin by its pins alone, and from the unwired one by the
    # token listed among its opening's ports
    among, loose, bare = (shaped[f"2x3 the key {what}"]["case"]["instance"]
                          for what in ("among the ports", "among the ports, no pin", "not wired"))
    assert among["record"] == loose["record"] and among["pins"] == shaped["2x3 registered"]["case"]["instance"]["pins"]
    assert loose["pins"] == []
    listed, plain = json.loads(among["record"]), json.loads(bare["record"])
    [o], [p] = listed["registered"]["openings"], plain["registered"]["openings"]
    assert o == {**p, "ports": [p["key"], *p["ports"]]}
    assert {k: v for k, v in listed.items() if k != "registered"} == {k: v for k, v in plain.items() if k != "registered"}
    # a reader in no layout is refused exactly where it is pinned
    for what in ("the key the second row", "a reader of one output"):
        pinned, free = (shaped[f"2x3 {what}, {how}"]["case"]["instance"] for how in ("pinned", "no pin"))
        assert pinned["program"] == free["program"] and len(pinned["pins"]) == 1 and free["pins"] == []
        assert json.loads(pinned["record"])["registered"] == json.loads(free["record"])["registered"]
    # no other stage refuses a negative vector, so each is refused for its roles
    monkeypatch.setattr(ref, "roles", lambda *_: None)
    off = {what: ref.check_detail(shaped[what]["case"]["instance"], shaped[what]["case"]["trusted"]) for what in LAYOUT}
    assert not any("error" in detail for detail in off.values())
    taken = off["2x3 the key among the ports"]
    assert taken["complies"] and (taken["fresh"], taken["free"], taken["openings"]) == (held["fresh"], 0, 0)


#: the cases on a unit's rows, each with the stage that refuses it
ROWS = {
    "2x3 one input run handed to two rows": "own-rows",
    "2x3 a row every unit shares": "own-rows",
    "2x3 a row of computed gates": "own-rows",
}
#: the toys of an edge from an output of leaves other than its row's (`vectors.EDGED`)
LEAVES = ("an edge from an output of two widths into one 32-bit leaf", "an edge from an output of 8 bits into a 16-bit leaf",
          "an edge from an output of 31 bits into 32 one-bit leaves", "an edge from a 16-bit output into 16 one-bit leaves")


def test_an_edges_writer_output_has_its_rows_leaves():
    """A wire's leaves were refused (`leaf-widths`) where its writer's leaves straddle the row's, or the row holds
    bits its writer's output lacks: a leaf the reader passes on would carry part of a writer's gate further. The
    record's wiring is the Program's edges, and an edge hands a row an output of the row's own type: an edge from an
    output of two widths into one 32-bit leaf, of 8 bits into a 16-bit leaf, of 31 bits into 32 one-bit leaves, or of
    a 16-bit leaf into 16 one-bit leaves is no Program's. The Program's builder refuses each as a type error, and the
    descriptor that states it by its references is refused with the Program (`program`), by the record, its count
    and its rows rule alike."""
    for what in LEAVES:
        *_, stage = vectors.EDGED[what]
        assert stage == "program", what
        r = vectors.results(vectors.edged_case(what))
        assert r["detail"] == r["count"] == r["block_rows"] == {"error": "program"}, what
    W = vectors.template("LeavesW", [("a", Value(16))], [Value(8)], [[("op", 8, [(0, 0)])]])
    R = vectors.template("LeavesR", [("v", Value(16))], [Value(16)], [[("op", 16, [(0, 0)])]])
    with pytest.raises(TypeError):
        vectors.Edged(1, [("a", Array(2, Value(16)))], Array(2, Value(16)),
                      lambda B, S, a: B.batch(R, B.batch(W, a, axes=(0,)), axes=(0,)),
                      [(W, 2, vectors._units(1, 2)), (R, 2, vectors._units(1, 2))]).program()


def test_a_units_rows_are_input_gates_of_its_own(shaped, monkeypatch):
    """The form counts a unit's row by what the record says fills it, which says something of the circuit only where
    the row is input gates of that unit alone. The root says which gates a unit's rows are, and the form refuses a
    root that hands a unit anything else (`own-rows`): the input gates of another row, the first unit's for every
    unit, or computed gates. Each case here is the accepted one with its root so edited, the record the same but for
    the Program it names; without the refusal the reference reads the same of each as of the accepted one, and
    accepts it."""
    held = shaped["2x3 registered"]
    want = vectors.results(held["case"])
    doc = json.loads(held["case"]["instance"]["record"])
    made = {what: shaped[what] for what, stage in ROWS.items() if stage == "own-rows"}
    assert len(made) == 3
    for what, m in made.items():
        r, inst = vectors.results(m["case"]), m["case"]["instance"]
        assert r["detail"] == r["public"] == {"error": "own-rows", "message": ref.OWN_ROWS}, what
        assert r["count"] == r["kinds"] == r["exported"] == {"error": "own-rows"}, what
        assert inst["program"] != held["case"]["instance"]["program"]
        assert {**json.loads(inst["record"]), "program": doc["program"]} == doc
    # what the root hands, gate by gate: the same gates to two rows, to every unit, or gates that are no input
    from verity.circuits.layout import Scope, resolve_gate
    from verity.circuits.parts import n_members
    from verity.circuits.program import is_input
    honest, twice, shared, computed = (ref.reference.desc_program(m["case"]["instance"]["program"])
                                       for m in (held, *made.values()))

    def gates(program, node: int, param: int) -> list[list[tuple]]:
        """Per unit of root node `node`, the gates its parameter `param`'s leaves are, each with its primitive."""
        root = Scope(program.root, 0, None, 0, 0)
        n = program.root.body.nodes[node]
        return [[resolve_gate(root.child(node, m), ("p", param, leaf))
                 for leaf in range(n.fn.signature()[0][param][1].leaves)] for m in range(n_members(n))]

    def distinct(rows: list[list[tuple]]) -> bool:
        """Whether the rows are input gates, none twice."""
        flat = [g for row in rows for g in row]
        return all(is_input(prim) for _, prim in flat) and len({g for g, _ in flat}) == len(flat)
    units = [k for k, n in enumerate(honest.root.body.nodes) if n.args]
    norm, select, feed = units[2], units[-2], units[-1]
    assert distinct(gates(honest, norm, 0) + gates(honest, norm, 1))
    assert distinct(gates(honest, feed, 0)) and distinct(gates(honest, select, 0))
    assert gates(twice, norm, 0) == gates(twice, norm, 1)
    assert len({tuple(g for g, _ in row) for row in gates(shared, feed, 0)}) == 1 < len(gates(shared, feed, 0))
    assert not any(is_input(prim) for row in gates(computed, select, 0) for _, prim in row)
    monkeypatch.setattr(ref, "own_rows", lambda *_: None)
    for what, m in made.items():
        off = vectors.results(m["case"] | {"rows": held["case"]["rows"]})
        assert off == want and off["detail"]["complies"], what


def test_what_the_form_exports_of_a_row(shaped):
    """Of each row the form exports its unit's slot, and why the row is held: a read of a θ parameter, a table read
    of a θ table's root or domain id, or a row opened from a θ table by a pinned reader, with the pin. A row has a
    reason exactly where its class is θ. At the served class's shape with the weights registered at the start of the
    deployment: every weight row a read, the embedding read's root and domain id rows table reads, its opened rows
    opened under the one pin, and the scan state, which no deployment registers, with no reason."""
    m = shaped["2x3 registered"]
    case, c = m["case"], m["circuit"]
    inst, tr = case["instance"], case["trusted"]
    rows = [[codec.descriptor_id(fn), p, u, r] for fn, n in c.templates for p, _ in fn.signature()[0]
            for u in range(n) for r in range(c.cuts.get(codec.descriptor_id(fn), {}).get(p, 1))]
    got, kinds, slot = ref.exported(inst, tr, rows), ref.kinds(inst, tr, rows), c.slot()
    assert [s for s, _ in got] == [slot[t][u] for t, _, u, _ in rows]
    assert [why is not None for _, why in got] == [k == "theta" for k in kinds]
    scan, embed, norm, gemm, *_ = c.units
    [pin] = inst["pins"]
    table, param = "table/weights/embed", "weights/embed/table"
    why = {(t, p): {tuple(w) if w else None for (t2, p2, _, _), (_, w) in zip(rows, got) if (t2, p2) == (t, p)}
           for t, p, _, _ in rows}
    assert why[scan, "state"] == {None} and why[embed, "tok"] == {None}
    assert why[norm, "w"] == {("read", "weights/norm")}
    assert why[gemm, "w"] == {("read", "weights/query"), ("read", "weights/mixing")}
    assert why[embed, "root"] == {("table", table, param, True)}
    assert why[embed, "domain"] == {("table", table, param, False)}
    assert why[embed, "row"] == why[embed, "salt"] == why[embed, "path"] == {("opened", table, param, pin)}
    # with none registered at the start of the deployment nothing is held, and the slots are the same
    none = shaped["2x3 none registered"]["case"]
    assert ref.exported(none["instance"], none["trusted"], rows) == [[s, None] for s, _ in got]


def test_no_wire_crosses_slots_at_the_served_shape(random_cases, shaped):
    """A wire from another slot's unit is the reading block's fresh input, so the form is a check that none crosses.
    At the served class's shape none does and the count is 0. One wire of one unit from the next slot is counted, its
    reader rows' bits and nothing more, and refuses the block. On any record the count is 0 exactly where every wire
    reads its reader's slot."""
    crossed = "2x3 one wire from the next slot"
    for what, m in shaped.items():
        if m["broken"] or not m["circuit"].name.startswith("lowered"):
            continue
        detail = vectors.results(m["case"])["detail"]
        assert detail["wired_rows"] > 0
        if what != crossed:
            assert all(m["inside"]) and detail["cross_wires"] == 0, what
    m, c = shaped[crossed], shaped[crossed]["circuit"]
    [w] = [w for w, inside in zip(c.wires, m["inside"], strict=True) if not inside]
    writer = {codec.descriptor_id(fn): fn for fn, _ in c.templates}[w.writer]
    rows = list(vectors._slots(c, w))
    assert len({reader for reader, _ in rows}) == 1 and all(reader != writer_slot for reader, writer_slot in rows)
    got, held = vectors.results(m["case"])["detail"], vectors.results(shaped["2x3 registered"]["case"])["detail"]
    assert got["cross_wires"] == 16 * dict(IF.layout(writer).outs)[w.out] * len(rows) > 0
    assert got["fresh"] == held["fresh"] + got["cross_wires"]
    assert (got["wired_rows"], got["wired_bits"]) == (held["wired_rows"], held["wired_bits"])
    assert (got["failure"], held["failure"]) == ("nci", None)
    some = 0
    for m in random_cases:
        detail = vectors.results(m["case"])["detail"]
        if "error" not in detail:
            assert (detail["cross_wires"] == 0) == all(m["inside"]), m["shape"]
            some += not all(m["inside"])
    assert some >= 20


def test_every_row_is_counted_and_a_block_is_a_slot(shaped):
    """`Desc.evaluate`'s reference on the Program of the served class's shape, every root node that computes
    in one block (a root node holds every slot's units of its template, so no partition of them separates two slots)
    and θ empty: that block's gates are the form's gates a block times its blocks, and its input bits are the leaves
    some gate reads. Every row that is input gates is a row the form counts that no edge fills (`inside`), so over the
    phase those rows' bits are the Program's input bits, more than the block's by exactly the rows no gate reads."""
    reference = ref.reference
    for size, blocks in (("2x3", 2), ("4x35", 4)):
        case = shaped[f"{size} registered"]["case"]
        text = case["instance"]["program"]
        form = vectors.results(case)["detail"]
        inside = ref.expand(case["instance"], case["trusted"])["inside"]
        inp, _, _ = reference.root_nodes(text)
        whole = reference.check_desc({"program": text, "theta": [], "owner": ["input" if i else 0 for i in inp],
                                      "gamma": form["gates"] * blocks, "alpha": [vectors.BIG, 1]})
        width, is_input, _, reads = reference.circuit(text)
        read = set().union(*reads)
        unread = sum(w for g, w in enumerate(width) if is_input[g] and g not in read)
        own = form["inputs"] * blocks - sum(inside)
        assert form["blocks"] == blocks
        assert whole["ius"] == [[form["gates"] * blocks, own - unread]] and whole["complies"]
        assert sum(w for g, w in enumerate(width) if is_input[g]) == own
        assert unread > 0 and sum(inside) > 0


def test_a_row_is_declared_once(shaped):
    """What a row holds is read from the one declaration that fills it, so a record that declares a row twice says
    two things of it: the first declaration found would decide, and a row that ingress fills could count as a
    registered read. The form refuses such a record, whichever two of a wire, a registered read, a table read, an
    opening's port and ingress declare the row: two wires by the wires' own check (`wires`), any other pair with the
    declarations (`inputs`). Each case is the accepted record with one declaration more and nothing else changed. The
    second declaration is well formed: with the row's first declaration dropped the stage takes the record, so each
    is refused for the row it shares and for nothing else."""
    held = shaped["2x3 registered"]["case"]
    *_, templates = ref.front(held["instance"], held["trusted"])
    built = json.loads(held["instance"]["record"])

    def declarations(rec: dict) -> None:
        """The wires' check and the declarations' on the record `rec`, refused (`Malformed`) as the reference refuses."""
        wires, cuts = ref.wires_of(templates, rec, None)
        ref.Inputs(templates, cuts, wires, rec)

    declarations(built)
    assert {stage for stage, *_ in vectors.TWICE.values()} == {"wires", "inputs"}
    for what, (stage, declare, drop) in vectors.TWICE.items():
        m = shaped[f"2x3 {what}"]
        r = vectors.results(m["case"])
        assert r["detail"] == r["public"] == r["kinds"] == r["exported"] == r["count"] == {"error": stage}, what
        twice, alone = copy.deepcopy(built), copy.deepcopy(built)
        declare(twice)
        assert json.loads(m["case"]["instance"]["record"]) == twice != built, what
        with pytest.raises(ref.Malformed) as refusal:
            declarations(twice)
        assert refusal.value.code == stage, what
        drop(alone)
        assert alone != built, what
        declare(alone)
        declarations(alone)


def test_an_output_has_a_bit(shaped):
    """A template whose return has no field, a Tuple's or a Record's, or has an output of no bit is refused with its
    templates. A return of no field has no output, and a field of no bit is an output of no bit beside one of 16. Such
    an output has no word, so no wire could read it. The interface lays a return of no field out as one output of no
    bit, and lays out no unit of either template, so no record it writes holds one."""
    returns = {"a template that returns no field": (["Tuple", []], []),
               "a template that returns a Record of no field": (["Record", []], []),
               "a return field of no bit": (["Tuple", [["Value", 16], ["Tuple", []]]], [16, 0])}
    assert set(returns) == set(vectors.BITLESS)
    for what, (name, outs, plan) in vectors.BITLESS.items():
        m = shaped[f"2x3 {what}"]
        r = vectors.results(m["case"])
        assert r["detail"] == r["public"] == r["kinds"] == r["exported"] == r["count"] == {"error": "templates"}, what
        fn = vectors.template(name, [("p", vectors.Value(16))], outs, plan)
        tid = codec.descriptor_id(fn)
        assert json.loads(m["case"]["instance"]["record"])["templates"] == [[tid, 2]], what
        program = json.loads(m["case"]["instance"]["program"])
        ret, bits = returns[what]
        assert program["definitions"][tid]["body"]["ret_type"] == ret, what
        assert [sum(t.leaf_widths()) for t in ref.fields_of(fn.signature()[1])] == bits, what
        with pytest.raises(ValueError, match=f"o{max(len(bits), 1) - 1}: its leaves are .* 0 in all"):
            IF.layout(fn)


def test_a_root_node_of_a_composite_is_no_input(shaped):
    """An input node is a root node of an input primitive. A composite's gates are computed whatever its key reads
    as, so a root node of a composite keyed as an input id is no input, and a record that left it out would count
    none of its gates. Such a Program is refused with its templates: the record of that one template, which lists
    it, and the accepted record beside a Program whose root has a unit of it more, which the record does not list.
    The key alone decides the first: the same template under a key that is no input id is read past its
    templates, and its record's registered reads still name templates it no longer lists: a declaration naming a
    template the Program lacks is refused at the first stage that reads a declaration (`inputs`). Unlisted it is
    refused under either key."""
    held = shaped["2x3 registered"]
    gate = [[("op", 16, [(0, 0)])]]
    for what, name in vectors.KEYED.items():
        fn = vectors.template(name, [("p", vectors.Value(16))], [vectors.Value(16)], gate)
        tid = codec.descriptor_id(fn)
        assert input_id(tid) is not None and not is_input(fn), what
        for how in ("alone", "unlisted"):
            m = shaped[f"2x3 {what}, {how}"]
            r = vectors.results(m["case"])
            assert r["detail"] == r["public"] == r["kinds"] == r["exported"] == r["count"] == {"error": "templates"}, what
            program = json.loads(m["case"]["instance"]["program"])
            assert program["definitions"][tid]["form"] == "composite", what
            root = program["definitions"][program["root"]]["body"]["nodes"]
            assert [n["form"] for n in root if program["callees"][n["fn"]] == tid] == ["batch"], what
            listed = [t for t, _ in json.loads(m["case"]["instance"]["record"])["templates"]]
            assert (tid in listed) == (how == "alone"), what
    plain = vectors.template("Plain7", [("p", vectors.Value(16))], [vectors.Value(16)], gate)
    assert input_id(codec.descriptor_id(plain)) is None
    for edit, stage in ((vectors._alone, "inputs"), (vectors._unlisted, "templates")):
        case = copy.deepcopy(held["case"])
        edit(case, held["circuit"], plain)
        assert ref.check_detail(case["instance"], case["trusted"]) == {"error": stage}


def test_a_row_format_is_held_to_the_registration_log(shaped):
    """A record states the leaf width of a registered value whose rows are not 16-bit leaves (`registered.formats`),
    and the form holds it to the registration log: the run's own record registers the value in leaves of that width,
    or the log's first record does (θ, at that width). Held, the scan state in 32-bit leaves, of either schema, and a
    weight registered in 32-bit leaves are what they are in 16-bit ones: the verdict and every count of the accepted
    case. The same records with another width, with the format dropped, with a format the log does not register, or
    with the value's entry stating 16 bits over the same rows and words are refused (`formats`); a format the form's
    shape refuses is refused with the declarations (`inputs`). A weight the first record registers in 32-bit leaves
    with no format in the record is no θ: its rows are fresh."""
    got = {what: vectors.results(shaped[what]["case"]) for what in FORMATS}
    held = vectors.results(shaped["2x3 registered"]["case"])
    for what, stage in FORMATS.items():
        r = got[what]
        if stage is not None:
            assert r["detail"] == r["public"] == r["kinds"] == r["exported"] == r["count"] == {"error": stage}, what
        elif what != "2x3 a weight in 32-bit leaves, no format":
            assert (r["detail"], r["count"]) == (held["detail"], held["count"]), what
    loose = got["2x3 a weight in 32-bit leaves, no format"]
    assert (loose["detail"]["failure"], loose["count"]["failure"]) == ("nci", "fresh")
    assert loose["detail"]["unregistered"] > held["detail"]["unregistered"]


def _formatted(made: dict) -> bool:
    """Whether the case's record states row formats (`registered.formats`). `verity.commitments.periods.interface`
    states no row format and refuses such a record, which the form reads, so the tenth test leaves it out."""
    doc = json.loads(made["case"]["instance"]["record"])
    return isinstance(doc, dict) and isinstance(doc.get("registered"), dict) and "formats" in doc["registered"]


def _interface(made: dict):
    """The stage at which `verity.commitments.periods.interface` refuses the case's record (`wires`, `inputs` or
    `blocks`), else None with what it reads of it: the calls `window_of_record` makes on a stated record, which
    refuses wires that are no list and reads the record's blocks where it has them. The interface refuses with a
    `ValueError`, so that alone, from one of its own calls, is a refusal here. An error of any other type is this
    helper's or the interface's, and is raised."""
    case = made["case"]
    doc = json.loads(case["instance"]["record"])
    lays = {codec.descriptor_id(fn): (IF.layout(fn), n) for fn, n in made["circuit"].templates}
    units = {t: n for t, (_, n) in lays.items()}
    spot = case["spot"]
    stated, cut = spot["wires"], spot["cuts"]
    if not isinstance(stated, list):
        return "wires", None
    try:
        wires, cuts = tuple(IF.Wire.of(w) for w in stated), IF._cuts(cut)
        IF.check_wires(lays, wires, cuts, IF.order_of(spot))
    except ValueError:
        return "wires", None
    try:
        reg = IF.Registered.of(doc["registered"]) if "registered" in doc else None
        IF.check_inputs(lays, cuts, wires, reg, IF._ingress_of(doc, "ingress"))
    except ValueError:
        return "inputs", None
    try:
        blocks = IF.Blocks.of(doc["blocks"], units, "") if "blocks" in doc else None
    except ValueError:
        return "blocks", None
    return None, (lays, units, blocks, doc)


def _strays(made: dict) -> set[str]:
    """The readers the case's record names in its openings that are no template of its circuit."""
    reg = json.loads(made["case"]["instance"]["record"]).get("registered")
    openings = reg.get("openings") if isinstance(reg, dict) else None
    readers = {o["reader"] for o in openings if isinstance(o, dict) and isinstance(o.get("reader"), str)} \
        if isinstance(openings, list) else set()
    return readers - {codec.descriptor_id(fn) for fn, _ in made["circuit"].templates}


def test_a_fault_of_the_helper_is_an_error_and_no_refusal(shaped, monkeypatch):
    """`_interface` takes a `ValueError` of one of the interface's calls for the interface's refusal, and nothing else.
    A helper that unpacks a pair from `Blocks.of`, which returns the blocks alone, raises a `TypeError` at that call:
    it is raised, where a refusal at `blocks` would say of every record that the interface refuses it. So is the
    `KeyError` of the helper's own lookup on a case that states no cuts. The interface's `ValueError` at the same call
    is the refusal. Two records the form refuses are none the interface refuses: one that states no blocks is
    taken, and on an opening of a reader that is no template the interface refuses or fails with a `KeyError` of that
    reader's name, which is raised."""
    held = shaped["2x3 registered"]
    assert _interface(held)[0] is None

    def edited(edit) -> dict:
        made = {**held, "case": copy.deepcopy(held["case"])}
        vectors._rewrite(made, edit)
        return made

    blocks_of = IF.Blocks.of

    def pair(d, units, partition):
        blocks, _ = blocks_of(d, units, partition)
        return blocks

    with monkeypatch.context() as patch:
        patch.setattr(IF.Blocks, "of", pair)
        with pytest.raises(TypeError, match="unpack"):
            _interface(held)
    uncut = {**held, "case": {**held["case"], "spot": {k: v for k, v in held["case"]["spot"].items() if k != "cuts"}}}
    with pytest.raises(KeyError, match="cuts"):
        _interface(uncut)
    refused = edited(lambda d: d["blocks"]["inference"].update(slots=0))
    assert _interface(refused) == ("blocks", None)
    assert ref.check_detail(refused["case"]["instance"], refused["case"]["trusted"]) == {"error": "blocks"}

    bare = edited(lambda d: d.pop("blocks"))
    stage, (*_, blocks, doc) = _interface(bare)
    assert stage is None and blocks is None and "blocks" not in doc
    assert ref.check_detail(bare["case"]["instance"], bare["case"]["trusted"]) == {"error": "blocks"}
    stray = edited(lambda d: d["registered"]["openings"][0].update(reader="Nope_v1"))
    assert _strays(held) == set() and _strays(stray) == {"Nope_v1"}
    assert ref.check_detail(stray["case"]["instance"], stray["case"]["trusted"]) == {"error": "inputs"}
    try:
        assert _interface(stray) == ("inputs", None)
    except KeyError as fault:
        assert fault.args == ("Nope_v1",)


def test_the_reference_reads_a_record_as_the_interface_does(random_cases, shaped):
    made = [m for m in (*random_cases, *shaped.values()) if m["case"]["spot"] is not None and not _formatted(m)]
    exact = apart = unpinned = 0
    for m in made:
        case = m["case"]
        inst, tr = case["instance"], case["trusted"]
        refusal = ref.check_detail(inst, tr).get("error")
        try:
            stage, read = _interface(m)
        except KeyError as fault:
            assert refusal == "inputs" and fault.args[0] in _strays(m), (m["shape"], m.get("how"))
            continue
        if m["shape"] == LOOSE:
            assert (stage, refusal) == ("inputs", None)
            continue
        bare = refusal == "blocks" and "blocks" not in json.loads(inst["record"])
        assert stage == (None if refusal in OWN or bare else refusal), (m["shape"], m.get("how"))
        if read is None or refusal is not None:
            continue
        lays, units, blocks, doc = read
        e = ref.expand(inst, tr)
        x = e["x"]
        for t, n in units.items():
            assert list(blocks.slots_steps(t, 0, n)[0]) == e["slot"][t]
        total = {k: sum(e[k]) for k in ("inputs", "fresh", *ref.KINDS)}
        assert total["inputs"] == sum(total[k] for k in ref.KINDS)
        assert total["fresh"] == sum(total[k] for k in ref.FRESH)
        # θ, as the interface holds a parameter: the log's first record's root for it is the run's own
        T = ref.reference.trusted_of(tr)
        theta = sorted(p for p in x.params if ref.theta(T, x.params, p, ref.width_of(doc, p)))
        assert theta == m["theta"]
        roots = T["log"][T["run"]]["window"].get("registered_roots", {})
        pins = {p: T["log"][0]["registered"][k]["root"] for p in theta
                for k in [max(k for k, en in enumerate(T["log"][0]["registered"]) if en["value"] == p)]}
        assert sorted(IF.held_params(pins, roots)) == theta
        if "order" in inst and "order" not in doc:
            doc = doc | {"order": inst["order"]}
        count = IF.input_count(doc, resolve={codec.descriptor_id(fn): fn for fn, _ in m["circuit"].templates}.__getitem__,
                               pins=pins, roots=roots)
        assert count["units"] == sum(units.values())
        assert count["inputs"] == total["inputs"]
        assert count["wired"] == e["wired_bits"] == total["inside"] + total["cross_wires"]
        assert count["ingress"] == total["ingress"]
        # the openings `input_count` counts registered that the form does not hold
        desc = json.loads(inst["program"])
        is_pinned = {t: ref.closure_digest(desc, t) in inst["pins"] for t in units}.__getitem__
        loose = [o for o in x.openings
                 if x.tables[o["table"]]["param"] in theta and not ref.held(x, theta.__contains__, is_pinned, o)]
        bits = sum(o["units"] * 16 * sum(dict(lays[o["reader"]][0].rows)[p] for p in o["ports"]) for o in loose)
        assert count["registered"] == total["theta"] + bits
        assert count["free"] == total["free"] + total["unregistered"] + total["openings"] - bits
        assert total["openings"] >= bits
        exact += not loose and any(x.tables[o["table"]]["param"] in theta for o in x.openings)
        apart += bool(loose)
        unpinned += any(not is_pinned(o["reader"]) for o in loose)
    assert exact >= 10 and apart >= 5 and unpinned >= 3, (exact, apart, unpinned)


def test_a_blocks_rows(shaped):
    """A block's rows are, over the record's templates, the template's units in a block times its rows of one unit.
    The reference adds them up unit by unit into each unit's slot and takes the most a slot holds; on the served
    shape, at both sizes, that is the sum the record's listing gives. `Γ` and `α` are the limits' own, in rows,
    whatever the instance's are in the Program's gates: a row more than `Γ` is `size`, and a fresh bit more than `α`
    times the rows is `inference_only`. An entry of the table for no template of the record is not counted, and a table that
    lacks a template of the record is refused (`rows-table`), after the limits' own refusals and the record's. An
    entry of 0 is one of the limits' own refusals (`params`), for a template of the record or for none, so no table
    counts a template as holding no rows."""
    for size, slots in (("2x3", 2), ("4x35", 4)):
        case = shaped[f"{size} registered"]["case"]
        inst, tr, limits = case["instance"], case["trusted"], case["block_rows"]
        table, listed = limits["table"], json.loads(inst["record"])["templates"]
        assert sorted(t for t, _ in listed) == sorted(table) and all(n % slots == 0 for _, n in listed)
        rows = sum(n // slots * table[t] for t, n in listed)
        fresh = ref.check_detail(inst, tr)["fresh"]
        held = ref.block_rows(inst, tr, limits)
        assert held == {"rows": rows, "fresh": fresh, "complies": True, "failure": None}
        assert (limits["gamma"], limits["alpha"]) == (rows, [fresh, rows]) and 0 < fresh < rows
        refused = {**inst, "gamma": 0, "alpha": [0, 1]}
        assert ref.check_detail(refused, tr)["failure"] == "size" and ref.block_rows(refused, tr, limits) == held
        assert ref.block_rows(inst, tr, {**limits, "gamma": rows - 1})["failure"] == "size"
        assert ref.block_rows(inst, tr, {**limits, "alpha": [fresh - 1, rows]})["failure"] == "nci"
        assert ref.block_rows(inst, tr, {**limits, "table": {**table, "Other_v1": 7}}) == held
        for t, n in listed if slots == 2 else listed[:1]:
            more = ref.block_rows(inst, tr, {**limits, "table": {**table, t: table[t] + 1}})
            assert (more["rows"], more["failure"]) == (rows + n // slots, "size"), t
            lacking = {k: v for k, v in table.items() if k != t}
            assert ref.block_rows(inst, tr, {**limits, "table": lacking}) == {"error": "rows-table"}, t
            assert ref.block_rows(inst, tr, {**limits, "table": {**table, t: 0}}) == {"error": "params"}, t
            assert ref.block_rows(inst, tr, {**limits, "table": {**lacking, "Other_v1": 0}}) == {"error": "params"}, t
    case = shaped[LISTED]["case"]
    inst, tr, limits = case["instance"], case["trusted"], case["block_rows"]
    assert ref.block_rows(inst, tr, {**limits, "table": {}}) == {"error": TEXT[LISTED]}
    for bad in (-1, "3", 1.5, True, None):
        assert ref.block_rows(inst, tr, {**limits, "gamma": bad}) == {"error": "params"}, bad
        assert ref.block_rows(inst, tr, {**limits, "table": {"Other_v1": bad}}) == {"error": "params"}, bad
    assert ref.block_rows(inst, tr, {**limits, "table": {"Other_v1": 0}}) == {"error": "params"}
    for bad in ([1], [1, 0], [1, 2, 3], [1.5, 1], [-1, 1], [1, "2"], "1/2", None):
        assert ref.block_rows(inst, tr, {**limits, "alpha": bad}) == {"error": "params"}, bad
    for bad in ([], None, *({k: v for k, v in limits.items() if k != gone} for gone in ref.ROWS_LIMITS),
                {**limits, "table": []}, {**limits, "table": [], "gamma": -1}):
        assert ref.block_rows(inst, tr, bad) == {"error": "instance"}, bad


def test_a_history_is_a_table_over_no_registered_parameter(shaped):
    """A history table is a table of its own name over a parameter of that name, which is no registered parameter's:
    the log's first record registers no such value, so no row read of it is θ. The served shape's record with one that
    no row reads counts as the accepted case does. With its table reads and its opening naming the history in place of
    the weights' table, under the same pin and the same registered weights, every row they fill is fresh (the table
    reads' `unregistered`, the opening's ports `openings`) and the rate refuses, in gates, in rows and by the count.
    The form reads a history's keys and its name and nothing else of it: one of a writer the circuit has no template of
    is taken and counts the same, where the interface refuses it. A history under a name a table, a registered
    parameter or another history has, one with a key more or a key less, one named by no string, and histories that
    are no list are refused with the declarations, as the interface refuses each."""
    accepted = shaped["2x3 registered"]
    held = vectors.results(accepted["case"])
    built = json.loads(accepted["case"]["instance"]["record"])
    assert set(HISTORIES.values()) == {None, "inputs"} and {LOOSE, OPENED} <= set(HISTORIES)
    for what, (stage, declare) in vectors.HISTORIES.items():
        m = shaped[f"2x3 {what}"]
        r = vectors.results(m["case"])
        doc = copy.deepcopy(built)
        declare(doc)
        assert json.loads(m["case"]["instance"]["record"]) == doc != built, what
        assert m["case"]["instance"]["pins"] == accepted["case"]["instance"]["pins"] != [], what
        if stage is not None:
            assert r["detail"] == r["public"] == r["kinds"] == r["exported"] == r["count"] == r["block_rows"] == {"error": stage}, what
        elif m["shape"] != OPENED:
            assert (r["detail"], r["count"], r["block_rows"]) == (held["detail"], held["count"], held["block_rows"]), what
        assert (_interface(m)[0] is None) == (stage is None and m["shape"] != LOOSE), what
    # the opening against a history: the record is the accepted one but for the table its reads and its opening name
    m = shaped[OPENED]
    inst, tr = m["case"]["instance"], m["case"]["trusted"]
    doc = json.loads(inst["record"])
    reg, was = doc["registered"], built["registered"]
    assert [h["name"] for h in reg.pop("histories")] == [vectors.HISTORY]
    assert vectors.HISTORY not in reg["params"] and vectors.HISTORY not in {t["name"] for t in reg["tables"]}
    for key in ("roots", "openings"):
        assert reg[key] == [{**x, "table": vectors.HISTORY} for x in was[key]] != was[key]
    assert {k: v for k, v in reg.items() if k not in ("roots", "openings")} == {
        k: v for k, v in was.items() if k not in ("roots", "openings")}
    assert {k: v for k, v in doc.items() if k != "registered"} == {k: v for k, v in built.items() if k != "registered"}
    x = ref.expand(inst, tr)["x"]
    assert x.tables[vectors.HISTORY] == {"name": vectors.HISTORY, "param": vectors.HISTORY}
    # each of the reader's rows they fill is θ in the accepted case and fresh here, with no reason it is held
    embed = next(fn for fn, _ in m["circuit"].templates if codec.descriptor_id(fn) == vectors.EMBED)
    bits = {p: 16 * words for p, words in IF.layout(embed).rows}
    units = dict(built["templates"])[vectors.EMBED]
    reads, ports = ("root", "domain"), ("row", "salt", "path")
    rows = [[vectors.EMBED, p, u, 0] for p in (*reads, *ports) for u in range(units)]
    before = accepted["case"]["instance"], accepted["case"]["trusted"]
    assert ref.kinds(*before, rows) == ["theta"] * len(rows)
    assert all(reason is not None for _, reason in ref.exported(*before, rows))
    assert ref.kinds(inst, tr, rows) == ["unregistered"] * (2 * units) + ["openings"] * (3 * units)
    assert all(reason is None for _, reason in ref.exported(inst, tr, rows))
    r = vectors.results(m["case"])
    d, d0 = r["detail"], held["detail"]
    a_block = units // d["blocks"]
    assert d["unregistered"] == d0["unregistered"] + a_block * sum(bits[p] for p in reads)
    assert d["openings"] == a_block * sum(bits[p] for p in ports) > 0 == d0["openings"]
    assert d["fresh"] == d0["fresh"] + a_block * sum(bits[p] for p in (*reads, *ports))
    assert {k: d[k] for k in ("blocks", "gates", "inputs", "wired_rows", "wired_bits")} == {
        k: d0[k] for k in ("blocks", "gates", "inputs", "wired_rows", "wired_bits")}
    assert (d["failure"], r["count"]["failure"], r["block_rows"]["failure"]) == ("nci", "fresh", "nci")
    assert r["block_rows"]["rows"] == held["block_rows"]["rows"] and r["block_rows"]["fresh"] == d["fresh"]


def test_what_the_prover_chose_of_an_execution_phase(random_cases, shaped):
    """The whole execution phase's input bits by kind, every slot's units, and of them the ones the prover chose: the
    rows no declaration fills (`free`), the reads of a parameter that is not θ (`unregistered`) and the ports of an
    opening that is not held (`openings`). Every row has one kind, so the kinds' bits are the phase's input bits, a
    block's times the blocks. A fresh kind's bits are at least the most a block holds and at most that many in every
    block, and fewer wherever the slots hold it unevenly, so the count is no multiple of the summary's. It is refused
    exactly where the form is, with the form's code.

    On the served shape, with the weights registered and the opening held, the prover chose the scan's state alone. An
    opening takes its ports off the count only where it is held: under no pin, with another Definition under the
    pinned one's name, or with no table read of the root, every row it opens is on the count at every unit it covers;
    with no root read the root's rows, which nothing fills then, are on it too. Against a history its ports and its
    table reads' rows are all on it. With nothing registered every row no wire fills is. A wire from the next slot is a
    block's fresh input and none of the prover's choosing: the summary's fresh bits move and the count does not."""
    uneven = none = 0
    for m in (*random_cases, *shaped.values()):
        inst, tr = m["case"]["instance"], m["case"]["trusted"]
        detail, c = ref.check_detail(inst, tr), ref.chosen(inst, tr)
        if "error" in detail:
            assert c == {"error": detail["error"]}, m["shape"]
            continue
        blocks = detail["blocks"]
        assert c["inputs"] == detail["inputs"] * blocks == sum(c[k] for k in ref.KINDS), m["shape"]
        assert c["chosen"] == c["free"] + c["unregistered"] + c["openings"], m["shape"]
        assert all(detail[k] <= c[k] <= detail[k] * blocks for k in ref.FRESH), m["shape"]
        uneven += any(c[k] < detail[k] * blocks for k in ref.FRESH)
        none += c["chosen"] == 0
    assert uneven >= 50 and none >= 3, (uneven, none)

    def chosen(what: str, **over) -> dict:
        case = shaped[what]["case"]
        return ref.chosen({**case["instance"], **over}, case["trusted"])

    def moved(c: dict, held: dict, **by: int) -> bool:
        """Whether `c` is `held` but for the kinds `by` names, each more by its bits, and θ, less by all of them."""
        want = {**held, **{k: held[k] + n for k, n in by.items()}, "theta": held["theta"] - sum(by.values())}
        return c == {**want, "chosen": want["free"] + want["unregistered"] + want["openings"]}

    for size, blocks in (("2x3", 2), ("4x35", 4)):
        case = shaped[f"{size} registered"]["case"]
        held, detail = chosen(f"{size} registered"), ref.check_detail(case["instance"], case["trusted"])
        assert vectors.results(case)["chosen"] == held
        assert held["chosen"] == held["unregistered"] == blocks * detail["unregistered"] > 0
        assert held["free"] == held["openings"] == held["cross_wires"] == held["ingress"] == 0
        for what in ("none registered", "none committed"):
            none = chosen(f"{size} {what}")
            assert none["theta"] == 0 and none["chosen"] == none["inputs"] - none["inside"] > held["chosen"]
    accepted = shaped["2x3 registered"]
    held = chosen("2x3 registered")
    embed = next(fn for fn, _ in accepted["circuit"].templates if codec.descriptor_id(fn) == vectors.EMBED)
    bits = {p: 16 * words for p, words in IF.layout(embed).rows}
    (opening,) = json.loads(accepted["case"]["instance"]["record"])["registered"]["openings"]
    units, ports = opening["units"], sum(bits[p] for p in opening["ports"])
    assert accepted["case"]["instance"]["pins"] != [] and units * ports > 0
    assert moved(chosen("2x3 registered", pins=[]), held, openings=units * ports)
    assert moved(chosen("2x3 a constant verdict"), held, openings=units * ports)
    assert moved(chosen("2x3 no root read"), held, openings=units * ports, free=units * bits["root"])
    assert moved(chosen(OPENED), held, openings=units * ports, unregistered=units * (bits["root"] + bits["domain"]))
    # a wire from the next slot: computed in the phase, so not chosen, and fresh to the block that reads it
    crossing, case = chosen("2x3 one wire from the next slot"), shaped["2x3 one wire from the next slot"]["case"]
    assert crossing["cross_wires"] > 0 and crossing["chosen"] == held["chosen"]
    wired = crossing["cross_wires"]
    assert crossing == {**held, "cross_wires": wired, "inside": held["inside"] - wired}
    assert (ref.check_detail(case["instance"], case["trusted"])["fresh"]
            > ref.check_detail(accepted["case"]["instance"], accepted["case"]["trusted"])["fresh"])


def test_an_opening_may_declare_its_row(shaped):
    """An opening may declare its `row`, the output of its reader that returns the row opened. The form holds it to
    the row a table read returns, the first of its reader's two outputs, the second its verdict, of a reader of six
    input rows, and reads nothing else of it: the served shape's record with its opening declaring that output counts
    as the accepted record does, row by row, and no row is of another kind for it. One whose row is the verdict's own
    output, no output of the reader, an input row of it or no string is refused with the declarations. A record that declares no row is read as before: the accepted
    record is each of these with the key dropped. `verity.commitments.periods.interface` takes and refuses the same,
    and reads the row back as the record states it."""
    accepted = shaped["2x3 registered"]
    held = vectors.results(accepted["case"])
    built = json.loads(accepted["case"]["instance"]["record"])
    assert all("row" not in o for o in built["registered"]["openings"])
    embed = next(fn for fn, _ in accepted["circuit"].templates if codec.descriptor_id(fn) == vectors.EMBED)
    lay = IF.layout(embed)
    assert set(ROW_ROLE.values()) == {None, "inputs"}
    for what, (stage, row) in vectors.ROW_ROLE.items():
        m = shaped[f"2x3 {what}"]
        r = vectors.results(m["case"])
        doc = json.loads(m["case"]["instance"]["record"])
        opening, = doc["registered"]["openings"]
        assert opening.pop("row") == row and doc == built, what
        assert m["case"]["instance"]["pins"] == accepted["case"]["instance"]["pins"] != [], what
        returned = len(lay.rows) == 6 and [q for q, _ in lay.outs] == [row, opening["verdict"]]
        assert (stage is None) == (isinstance(row, str) and returned), what
        assert _interface(m)[0] == stage, what
        if stage is not None:
            assert r["detail"] == r["public"] == r["kinds"] == r["exported"] == r["count"] == r["block_rows"] == {"error": stage}, what
            continue
        assert (r["detail"], r["count"], r["block_rows"]) == (held["detail"], held["count"], held["block_rows"]), what
        inst, tr = m["case"]["instance"], m["case"]["trusted"]
        units = dict(built["templates"])[vectors.EMBED]
        rows = [[vectors.EMBED, p, u, 0] for p, _ in lay.rows for u in range(units)]
        before = accepted["case"]["instance"], accepted["case"]["trusted"]
        assert ref.kinds(inst, tr, rows) == ref.kinds(*before, rows), what
        assert ref.exported(inst, tr, rows) == ref.exported(*before, rows), what
        read, = IF.Registered.of(json.loads(inst["record"])["registered"]).openings
        assert read.row == row and read.doc()["row"] == row, what


def test_an_opening_whose_row_is_its_verdicts_output_is_refused(shaped):
    """An opening whose `row` is its verdict's own output is refused with the declarations, under its reader's pin and
    under none: the row an opening declares is the one a table read returns, the first of its reader's two outputs,
    and the verdict is the second. `verity.commitments.periods.interface` refuses it too. The record is the accepted
    one with the key added, so the refusal is the row's alone."""
    accepted, pinned = shaped["2x3 registered"], shaped[f"2x3 {vectors.ROW_VERDICT}"]
    loose = shaped[f"2x3 {vectors.ROW_UNPINNED}"]
    built = json.loads(accepted["case"]["instance"]["record"])
    stage = vectors.ROW_ROLE[vectors.ROW_VERDICT][0]
    assert stage == "inputs"
    for m, pins in ((pinned, accepted["case"]["instance"]["pins"]), (loose, [])):
        doc = json.loads(m["case"]["instance"]["record"])
        opening, = doc["registered"]["openings"]
        assert opening["row"] == opening["verdict"] == "o1" and tuple(opening["ports"]) == ("row", "salt", "path")
        assert opening.pop("row") and doc == built
        assert m["case"]["instance"]["pins"] == pins
        r = vectors.results(m["case"])
        assert m["broken"] == stage
        assert r["detail"] == r["public"] == r["kinds"] == r["exported"] == r["count"] == r["block_rows"] == {"error": stage}
        assert _interface(m)[0] == stage


def test_a_template_in_two_root_nodes(shaped):
    """One template in two root nodes (`SPLITS`): its units numbered across them, the record's count their sum, it
    is taken with the served shape's verdict, side by side or with another template's node between; a count short of
    the sum is refused with its templates."""
    base = vectors.results(shaped["2x3 registered"]["case"])
    for what, (_, short) in vectors.SPLITS.items():
        got = vectors.results(shaped[f"2x3 {what}"]["case"])
        if short:
            assert got["detail"] == {"error": "templates"}, what
        else:
            assert got["detail"] == base["detail"] and got["chosen"] == base["chosen"], what


def test_a_units_row_may_be_an_earlier_units_output():
    """The edges (`EDGED`): each toy is taken, or refused at the stage it states, by the record, its count and its
    rows rule alike; a taken one counts every unit of the circuit in its blocks."""
    for what, (make, _, _, stage) in vectors.EDGED.items():
        case = vectors.edged_case(what)
        r = vectors.results(case)
        assert r["detail"].get("error") == stage, (what, r["detail"])
        if stage is None:
            assert r["detail"]["blocks"] >= 1 and isinstance(r["kinds"], list), what
            assert "error" not in r["count"] and "error" not in r["block_rows"], what
        else:
            assert r["count"] == r["block_rows"] == {"error": stage}, what
