"""The inference-only policy's check of an execution phase's circuit record (`NciCheck.Record`) against the Python
reference: the Python half.

`verity/protocols/inference_only/verifier/nci/lean/scripts/record_difftest_vectors.py` writes 687 cases (600 random circuits,
two in five broken in one way, and 87 shaped ones, most in the shape of a served class) and, on each,
`scripts/nci_record_reference.py`'s results, which visit every row of every unit. `check` runs the Lean half in its
Lean audit (`runs` in that package's `lean-audit.json`): `NciCheck/RecordDifftestMain.lean` fails unless
`Record` and `RecordCount` give Python's results on every case, and unless SPOT's reader reads the
same wiring from the form's instance and from the one Python builds from its own layout of each template. The audit
checks the data against the record's SHA-256 and caches its pass on the Lean package's files alone, so the first test
is what says the data is still what the generator and the reference write. The second checks that the generator
reaches what its docstring says, that the published result is the judgment alone, and that a text the form reads
nothing of is refused: a record that lists its members, and one that names no Program. The third states the verdicts on
the served class's shape, the negative vectors among them, and the fourth the refusals of a pinned reader's layout:
one negative vector a role, each refused, and the same records under no pin. The fifth and the sixth are the two
refusals that make a row what the form counts it as: a wire whose writer's leaf is not inside one leaf of the row
that reads it, into a row that is not closed (`leaf-widths`), and a root that hands a unit a row that is not input
gates of its own (`own-rows`), each with what the form would count without the refusal. The seventh is what the
form exports of a row: its unit's slot, and why it is held. The eighth is the check that no wire crosses slots: none
does at the served class's shape, and one that does is counted. The ninth sets the form beside
`Desc.evaluate`'s reference on the same Program: one block for the whole phase there and one a slot here,
and input bits apart by exactly the rows no gate reads. The tenth holds the reference to
`verity.commitments.windows.interface`, the rules a lowering states a record by: the stage that refuses a record, each unit's
slot, and `input_count`'s totals. The eleventh is that a row is declared once: the served shape's record with one
declaration more into a row it already declares is refused, whatever the two declarations are, and the second alone
is taken. The twelfth is that an output has a bit: a template that returns no field, or a field of no bit, is refused
with its templates, and the interface lays out no unit of it. The thirteenth is that a root node of a composite is no
input, whatever its key reads as: a template keyed as an input id is refused with its templates, its record alone and
a unit of it at the root that the record does not list. The fourteenth is that a row format is held to the
registration log: a value read in leaves of another width than 16 bits is registered in leaves of that width.

The tenth holds to `input_count` only where every opening it counts registered is one the form holds (`held`: the
reader's Definition pinned, with table reads of the table's root and domain id over the opening's units). Where one is
not, `input_count` counts that opening's ports registered and the form counts them fresh, and the test holds the two
apart by exactly those bits. The interface states no layout and no leaf's width: a record the form refuses for a
pinned reader's roles (`reader-layout`, `reader-roles`) or for a wire's leaves (`leaf-widths`) is one the interface
takes. It states no row format and does not read the registration log: it refuses a record that states row formats,
which the tenth leaves out (`_formatted`), and takes one the form refuses for its formats against the log (`formats`).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from verity.circuits.program import input_id, is_input

LEAN = Path(__file__).resolve().parents[4] / "inference_only" / "verifier" / "nci" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("record_difftest_vectors")
ref, IF, codec = vectors.ref, vectors.IF, vectors.codec
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NciCheck.RecordDifftestMain"]
#: the random cases the quick tests read, enough of them to reach every way of `BROKEN`
SEEDS = range(500)
#: the form's own refusals of a record `verity.commitments.windows.interface` takes: a wire's leaves, a pinned reader's
#: layout, and the row formats held to the registration log, which the interface does not read
OWN = ("leaf-widths", "formats", "reader-layout", "reader-roles")
#: the cases whose wires Python's layout has no rows for
TWO_WIDTHS = "a wire from an output of two widths"
TWO_CLOSED = "an output of two widths into a closed row"
#: `short_output`: an output of 8 bits into a 16-bit row, one leaf or sixteen that are closed
SHORT, SHORT_CLOSED = "an output that lacks bits of its row", "an output that lacks bits of a closed row"
#: `odd_output`: an output of 31 bits, two words, into a row of two words and wired into a row of one
ODD, ODD_ONE = "an output of 31 bits into a row of two words", "an output of 31 bits wired into a row of one word"
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
#: the cases on a row format, each with the stage that refuses it, or None
FORMATS = {(f"2x3 {what}" if shape is None else what): stage for what, (stage, shape, _) in vectors.FORMATS.items()}


@pytest.fixture(scope="module")
def random_cases() -> list[dict]:
    return [vectors.case(s) for s in SEEDS]


@pytest.fixture(scope="module")
def shaped() -> dict[str, dict]:
    return {m["shape"]: m for m in vectors.shaped()}


@pytest.mark.slow
def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 687 cases and the reference's results, byte for byte as the record pins them."""
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
        assert (m["case"]["spot"] is None) == (m["broken"] in vectors.EARLY or m["shape"] in (TWO_WIDTHS, TWO_CLOSED, ODD, ODD_ONE))
        assert isinstance(r["kinds"], list) == isinstance(r["exported"], list) == ("error" not in detail)
        assert ("message" in detail) == (m["broken"] in ("own-rows", "leaf-widths"))
    assert {m["broken"] for m in made} == set(vectors.BROKEN) | {None}
    assert {m["how"] for m in random_cases if m["broken"]} == {how for hows in vectors.BROKEN.values() for how in hows}
    good = [r for r in results if "error" not in r["detail"]]
    assert {r["detail"]["failure"] for r in good} == {None, "size", "nci"}
    assert {r["count"].get("failure") for r in good if "error" not in r["count"]} == {None, "gates", "fresh", "count"}
    assert {r["count"].get("error") for r in good} >= {None, "instance", "params", "gate-table"}
    assert {k for r in good for k in r["kinds"]} == set(ref.KINDS)
    for k in ("fresh", *ref.FRESH):
        assert any(r["detail"][k] > 0 for r in good), k
    assert any(r["detail"]["fresh"] == 0 for r in good)
    assert any(r["detail"]["blocks"] > 1 for r in good)
    for what, stage in TEXT.items():
        r = vectors.results(shaped[what]["case"])
        assert r["detail"] == r["public"] == r["kinds"] == r["exported"] == r["count"] == {"error": stage}, what
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
           if what not in LAYOUT | ROWS | TEXT | TWICE | BITLESS | KEYED | FORMATS}
    verdicts = {what: (r["detail"]["failure"], r["count"]["failure"]) for what, r in got.items()}
    refused = ("nci", "fresh")
    assert verdicts == {
        "2x3 registered": (None, None), "4x35 registered": (None, None),
        "2x3 none registered": refused, "4x35 none registered": refused,
        "2x3 none committed": refused, "4x35 none committed": refused,
        "2x3 another root": refused, "2x3 no root read": refused, "2x3 a constant verdict": refused,
        "2x3 one wire from the next slot": refused,
        "2x3 the order beside the record": (None, None), "2x3 the run's record first": (None, None)}
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
    and every port of the opening is fresh. Whatever of these the form refuses, `verity.commitments.windows.interface` takes.
    The layout's stage alone refuses them: without it the reference refuses none, and accepts the token among the
    ports with a block's fresh bits those of the case that wires it."""
    assert {what for what, m in shaped.items() if m["broken"]} == {
        what for what, v in (LAYOUT | ROWS | TEXT | TWICE | BITLESS | KEYED | FORMATS).items() if isinstance(v, str)}
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


#: the cases on a wire's leaves and on a unit's rows: the stage that refuses each, or the rate rule's and the count's
#: verdicts
CARRIED = "a leaf that reads sixteen gates through one row"
#: `passed_on`, its leaves passed on another way: by a Definition of no gate, through a call, a callee inside a
#: callee, a batch member, a scan
PASSED = {f"sixteen gates through one row, passed on {how}": shape for how, shape in (
    ("by a Definition of no gate", {"gate": False}), ("through a call", {"how": "call"}),
    ("through a callee inside a callee", {"how": "nested"}), ("through a batch member", {"how": "batch"}),
    ("through a scan", {"how": "scan"}))}
#: `passed_on`, one of its leaves passed on through a scan's carry, and the other fifteen as a scan's outputs: per
#: case, which of the sixteen leaves of its return are leaves of its row
PART = {"a gate through one row, passed on through a scan's carry": [0],
        "fifteen gates through one row, passed on as a scan's outputs": list(range(1, 16))}
#: `passed_on`, its first output gates of its own and its leaves passed on to a second output, which no wire reads
UNREAD = "sixteen gates through one row, passed on to an output no wire reads"
#: `passed_on`, its leaves handed to a callee that returns none of them, and its output gates of its own over them
BESIDE, GATES = "a row a callee is handed and returns no leaf of", "a closed row: sixteen gates read, none passed on"
ROWS = {
    "2x3 the key's reader bits, the other side a word": (None, None),
    "2x3 the key's writer bits, the other side a word": (None, None),
    CARRIED: "leaf-widths",
    **dict.fromkeys(PASSED, "leaf-widths"),
    **dict.fromkeys(PART, "leaf-widths"),
    UNREAD: "leaf-widths",
    BESIDE: "leaf-widths",
    GATES: (None, None),
    TWO_WIDTHS: "leaf-widths",
    TWO_CLOSED: (None, None),
    SHORT: "leaf-widths",
    SHORT_CLOSED: "leaf-widths",
    ODD: "leaf-widths",
    ODD_ONE: "wires",
    "2x3 one input run handed to two rows": "own-rows",
    "2x3 a row every unit shares": "own-rows",
    "2x3 a row of computed gates": "own-rows",
}
WHOLE = ("Every leaf of a writer's output lies inside one leaf of the row that reads it, each side's leaves of one "
         "width and the writer's dividing the reader's; or the reader's row is one-bit leaves and its Definition "
         "passes none of them on: its return refers to no part of that row, itself or through a call, a batch or a "
         "scan. Fix the Definitions, not the record: give the reader's row leaves of the writer's width or a multiple "
         "of it, or the writer's output leaves of a width that divides the reader's, or make the reader's row "
         "one-bit leaves and have its Definition return gates of its own in their place.")
FILLED = ("Every bit of a row a wire fills is a bit of its writer's output: a row is whole 16-bit words, and a bit of "
          "it past its writer's output is no gate's. Fix the Definitions, not the record: give the writer's output "
          "whole 16-bit words, each bit a gate of its own.")


def test_a_writers_leaf_lies_inside_one_leaf_of_the_row_that_reads_it(shaped, monkeypatch):
    """A wire's two rows have the same words, and that alone does not make the row the writer's gates: where a
    writer's leaf straddles leaves of the row that reads it, a leaf the reader passes on carries part of a gate of
    the writer further. The form takes a wire whose writer's leaves each lie inside one leaf of the row, and a wire
    into a closed row, one-bit leaves its Definition passes none of on, which only gates of its own unit read; it
    refuses any other (`leaf-widths`), and its message names the wire, the two widths and the fix. At the served
    class's shape, the token a word (one 32-bit leaf) into a key of 32 one-bit leaves is taken, the key a closed
    row, and so is the token 32 one-bit leaves into a key of one word; each is counted as any wire. An output whose
    leaves have two widths is refused into a 32-bit leaf and taken into a closed row.

    Of either way, every bit of the row is a bit of its writer's output. A wire's two rows have the same words, and
    an output's words are its bits rounded up: an output of 8 bits fills a row of 16, whose other 8 bits are no
    gate's. It is refused into one 16-bit leaf and into a closed row alike, the message naming the two rows' bits.
    An output of 31 bits is two words, so it is refused the same way into a row of two words, and a record that
    wires it into a row of one word is refused before that, by the wires' own check: with its words rounded down
    that check takes the wire, and sixteen one-bit leaves read a gate of 31 bits.

    What the refusal is for is `passed_on`: a template that passes on, with no gate, one bit of each of sixteen
    16-bit gates, and a unit of another slot that reads those sixteen bits as one 16-bit leaf. Without the refusal the
    reference takes the record and counts 16 fresh bits a block, the one row that crosses slots, and the block is
    accepted under a policy of 16 bits; followed leaf by leaf, that row reads sixteen 16-bit gates of the other
    slot. It is refused however the template passes the bits on: itself, with no gate of its own at all, through a
    call, through a callee inside a callee, through a batch member and through a scan, each followed leaf by leaf
    through the root to the row's own gates (`hands_on`); where it passes on one of them alone, through every
    iteration of a scan's carry, or the other fifteen alone, as a scan's outputs; and where it passes them on to an
    output no wire reads, the output the row is wired to gates of its own. Where the template's output is sixteen gates of its own over those bits, the row is closed
    and the record is taken: the 16-bit leaf then reads sixteen one-bit gates. The rule reads a Definition's
    reference parts, a parameter at a time, so it is narrower than what it is for: a template that hands the bits to
    a callee beside gates of its own, and gets the gates back, passes no leaf on and is refused."""
    got = {what: vectors.results(shaped[what]["case"]) for what in ROWS}
    into_on = ("PassWide_v1.o0 -> PassOn_v1.x", "16 bits", "1 bits")
    wires = {**dict.fromkeys((CARRIED, *PASSED, *PART, UNREAD, BESIDE), into_on),
             TWO_WIDTHS: ("WidthsTwo_v1.o0 -> WidthsWhole_v1.v", "more than one width", "32 bits")}
    eight = ("ShortEight_v1.o0 -> ShortWhole_v1.v", 8, 1)
    short = {SHORT: (*eight, []), SHORT_CLOSED: (*eight, ["v"]), ODD: ("OddOut_v1.o0 -> OddWhole_v1.v", 31, 2, ["v"])}
    assert set(wires) | set(short) == {what for what, want in ROWS.items() if want == "leaf-widths"}
    for what, (wire, bits, words, closed) in short.items():
        said = {"error": "leaf-widths", "message": f"wire {wire}: the writer's output is {bits} bits and the row it "
                                                   f"fills {16 * words}. {FILLED}"}
        assert got[what]["detail"] == got[what]["public"] == said, what
        assert got[what]["count"] == got[what]["kinds"] == got[what]["exported"] == {"error": "leaf-widths"}, what
        writer, whole = (ref.Template(codec.descriptor_id(fn), fn, n) for fn, n in shaped[what]["circuit"].templates)
        # the two rows have the same words, which SPOT's check holds, and the row has bits the output has not
        assert writer.port("o0", outs=True) == whole.port("v") == words
        assert (sum(writer.widths["o0"]), sum(whole.widths["v"])) == (bits, 16 * words)
        assert whole.closed == closed
    # the 31 bits wired into a row of one word: two words against one, refused by the wires' own check; with the
    # output's words rounded down that check takes the same wires
    one = shaped[ODD_ONE]
    assert got[ODD_ONE]["detail"] == got[ODD_ONE]["public"] == got[ODD_ONE]["count"] == {"error": "wires"}
    odd, whole = (ref.Template(codec.descriptor_id(fn), fn, n) for fn, n in one["circuit"].templates)
    assert (sum(odd.widths["o0"]), odd.port("o0", outs=True), whole.port("v"), whole.closed) == (31, 2, 1, ["v"])
    rec = json.loads(one["case"]["instance"]["record"])
    assert rec["wires"] == [vectors.ODD_WIRE]

    def lays(down: bool) -> dict:
        return {t.tid: (IF.RowLayout(tuple((p, w) for p, w, _ in t.rows),
                                     tuple((o, sum(t.widths[o]) // 16 if down else w) for o, w in t.outs), (),
                                     tuple(n for *_, n in t.rows)), t.units) for t in (odd, whole)}
    wired = tuple(IF.Wire.of(w) for w in rec["wires"])
    IF.check_wires(lays(True), wired, IF._cuts(rec.get("cuts", {})), IF.order_of(rec))
    with pytest.raises(ValueError):
        IF.check_wires(lays(False), wired, IF._cuts(rec.get("cuts", {})), IF.order_of(rec))
    for what, (wire, writer, reader) in wires.items():
        said = {"error": "leaf-widths", "message": f"wire {wire}: the writer's leaves are of {writer} and the reader's "
                                                   f"of {reader}. {WHOLE}"}
        assert got[what]["detail"] == got[what]["public"] == said, what
        assert got[what]["count"] == got[what]["kinds"] == got[what]["exported"] == {"error": "leaf-widths"}, what
    # the served class's shape: the two sides of the token's wires, a word and its 32 bits, either way round
    for side, (key, tok) in (("reader", ((1,) * 32, (32,))), ("writer", ((32,), (1,) * 32))):
        m = shaped[f"2x3 the key's {side} bits, the other side a word"]
        scan, embed, *_, feed = (ref.Template(codec.descriptor_id(fn), fn, n) for fn, n in m["circuit"].templates)
        assert (embed.widths["tok"], scan.widths["o0"], feed.widths["o0"]) == (key, tok, tok)
        into = [(w.writer, w.out) for w in m["circuit"].wires if (w.reader, w.port) == (embed.tid, "tok")]
        assert sorted(into) == sorted([(scan.tid, "o0"), (feed.tid, "o0")])
        # the key of bits is a closed row: gates of `Embed` read it and its return refers to no part of it
        assert embed.closed == (["tok"] if side == "reader" else []) and not ref.passed_on(embed.fn)
        taken = got[f"2x3 the key's {side} bits, the other side a word"]
        assert (taken["detail"]["failure"], taken["count"]["failure"]) == (None, None) and taken["detail"]["complies"]
        assert taken["detail"]["fresh"] == taken["detail"]["unregistered"] > 0 == taken["detail"]["cross_wires"]
    # the row that carries sixteen gates: what the wires say, leaf by leaf
    m = shaped[CARRIED]
    c, case = m["circuit"], m["case"]
    (wide, _), (on, _), (take, _) = c.templates
    wide_id, on_id, take_id = map(codec.descriptor_id, (wide, on, take))
    passed = [on.body.ret[j] for j in range(16)]
    assert passed == [("p", 0, 17 * j) for j in range(16)] and wide.signature()[1].leaf_widths() == [16]
    rows = {(w.reader, w.port, u, w.cut[0] + r): (w.writer, v)
            for w in c.wires for u, r, v in ref.entries({**w.doc(), "c0": w.cut[0], "C": w.cut[1]})}
    slot = c.slot()
    for s in range(c.slots):
        writer, v = rows[take_id, "v", s, 0]
        # the leaf is the 16 bits `On` passes on: bit j is its row j's bit j, and its row j is one 16-bit gate
        gates = {rows[on_id, "x", v, leaf // 16] for _, _, leaf in passed}
        assert writer == on_id and len(gates) == 16 and {t for t, _ in gates} == {wide_id}
        assert {slot[wide_id][u] for _, u in gates} == {slot[on_id][v]} != {slot[take_id][s]}
    assert case["count"]["k"] == 16 < 16 * 16
    # however `On` passes them on, the same sixteen leaves of its row are leaves of its return: through the root, the
    # same gates. Each of those Definitions has the one row, not closed; none but the first refers to it in its return.
    from verity.circuits.layout import Scope, resolve_gate

    def returned(m: dict) -> list[list[int]]:
        """Per unit of `On`, the gates the sixteen leaves of its first output are."""
        program = ref.reference.desc_program(m["case"]["instance"]["program"])
        [k] = [k for k, n in enumerate(program.root.body.nodes) if codec.descriptor_id(n.fn) == on_id]
        fn, root = program.root.body.nodes[k].fn, Scope(program.root, 0, None, 0, 0)
        return [[resolve_gate(root.child(k, u), fn.body.ret[j])[0] for j in range(16)] for u in range(c.slots)]

    def row(m: dict) -> list[list[int]]:
        """Per unit of `On`, the gates the sixteen leaves `17 j` of its row are."""
        program = ref.reference.desc_program(m["case"]["instance"]["program"])
        [k] = [k for k, n in enumerate(program.root.body.nodes) if codec.descriptor_id(n.fn) == on_id]
        root = Scope(program.root, 0, None, 0, 0)
        return [[resolve_gate(root.child(k, u), ("p", 0, 17 * j))[0] for j in range(16)] for u in range(c.slots)]

    def of(m: dict) -> tuple:
        """`On` as the reference reads it: what it passes on, its closed rows, whether a unit hands a leaf on."""
        program = ref.reference.desc_program(m["case"]["instance"]["program"])
        [fn] = [n.fn for n in program.root.body.nodes if codec.descriptor_id(n.fn) == on_id]
        return sorted(ref.passed_on(fn)), ref.Template(on_id, fn, c.slots).closed, ref.hands_on(program, on_id, "x")

    assert returned(m) == row(m) and of(m) == ([0], [], True)
    for what, shape in PASSED.items():
        other = shaped[what]
        assert returned(other) == row(other) and of(other) == ([0], [], True), what
        [on_fn] = [fn for fn, _ in other["circuit"].templates if codec.descriptor_id(fn) == on_id]
        direct = any(space == "p" for space, _, _ in (on_fn.body.ret[j] for j in range(16)))
        assert direct == ("how" not in shape) and ("gate" in shape) == (not ref.template_gates(on_fn)), what
    # one leaf passed on through a scan's carry, and fifteen as a scan's outputs: those leaves of the return are the
    # row's gates, leaf for leaf, and the others are gates of `On`
    for what, kept in PART.items():
        some, ret, own = shaped[what], returned(shaped[what]), row(shaped[what])
        assert of(some) == ([0], [], True), what
        for u in range(c.slots):
            assert [j for j in range(16) if ret[u][j] in own[u]] == kept == [j for j in range(16) if ret[u][j] == own[u][j]]
        [on_fn] = [fn for fn, _ in some["circuit"].templates if codec.descriptor_id(fn) == on_id]
        [scan] = [n for n in on_fn.body.nodes if n.form == "scan"]
        assert scan.n == 15 and not any(space == "p" for space, _, _ in (on_fn.body.ret[j] for j in range(16))), what
    # passed on to an output no wire reads: the leaves of the output the row is wired to are gates of `On`, none of
    # them the row's, and the row's leaves are leaves of its second output all the same
    unread = shaped[UNREAD]
    assert not set(sum(returned(unread), [])) & set(sum(row(unread), [])) and of(unread) == ([0], [], True)
    [on_fn] = [fn for fn, _ in unread["circuit"].templates if codec.descriptor_id(fn) == on_id]
    assert [on_fn.body.ret[j] for j in range(16, 32)] == passed and len(ref.fields_of(on_fn.signature()[1])) == 2
    assert not [w for w in unread["circuit"].wires if (w.writer, w.out) == (on_id, "o1")]
    # handed to a callee beside gates and none returned: no leaf of the row is a leaf of the return, and the rule,
    # which reads parts, still counts the row passed on
    beside, gates = shaped[BESIDE], shaped[GATES]
    assert not set(sum(returned(beside), [])) & set(sum(row(beside), [])) and of(beside) == ([0], [], False)
    # the output gates of its own: a closed row, taken, and the 16 fresh bits a block are the one row across slots
    assert not set(sum(returned(gates), [])) & set(sum(row(gates), [])) and of(gates) == ([], ["x"], False)
    closed = got[GATES]
    assert closed["detail"]["complies"] and closed["count"]["complies"]
    assert closed["detail"]["fresh"] == closed["detail"]["cross_wires"] == 16
    # an output of two widths into a closed row
    two = got[TWO_CLOSED]
    assert two["detail"]["complies"] and two["count"]["complies"] and two["detail"]["cross_wires"] == 0
    [whole] = [ref.Template(codec.descriptor_id(fn), fn, n) for fn, n in shaped[TWO_CLOSED]["circuit"].templates
               if codec.descriptor_id(fn) == "WidthsWhole_v1"]
    assert whole.closed == ["v"] and whole.widths["v"] == (1,) * 32
    # without the stage the reference refuses none of them, and counts the carried row as the one row across slots
    monkeypatch.setattr(ref, "leaf_widths", lambda *_: None)
    off = {what: vectors.results(shaped[what]["case"]) for what in (*wires, *short)}
    assert not any("error" in r["detail"] for r in off.values())
    assert all(off[what]["detail"]["complies"] and off[what]["count"]["complies"] for what in short)
    for what in (CARRIED, *PASSED, *PART, UNREAD, BESIDE):
        carried = off[what]
        assert carried["detail"]["complies"] and carried["count"]["complies"], what
        assert carried["detail"]["fresh"] == carried["detail"]["cross_wires"] == 16, what


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
    some gate reads. The form counts every row of every unit, so over the phase its input bits are more by exactly the
    rows no gate reads."""
    reference = ref.reference
    for size, blocks in (("2x3", 2), ("4x35", 4)):
        case = shaped[f"{size} registered"]["case"]
        text = case["instance"]["program"]
        form = vectors.results(case)["detail"]
        inp, _, _ = reference.root_nodes(text)
        whole = reference.check_desc({"program": text, "theta": [], "owner": ["input" if i else 0 for i in inp],
                                      "gamma": form["gates"] * blocks, "alpha": [vectors.BIG, 1]})
        width, is_input, _, reads = reference.circuit(text)
        read = set().union(*reads)
        unread = sum(w for g, w in enumerate(width) if is_input[g] and g not in read)
        assert form["blocks"] == blocks
        assert whole["ius"] == [[form["gates"] * blocks, form["inputs"] * blocks - unread]] and whole["complies"]
        assert sum(w for g, w in enumerate(width) if is_input[g]) == form["inputs"] * blocks
        assert unread > 0


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
    """A template with an output that has no bit is refused with its templates. A return of no field is itself its
    one output, of no bit, as the interface lays a return out, and a field of no bit is an output of no bit beside
    one of 16. Such an output has no word, so no wire could read it, and the interface lays out no unit of such a
    template, so no record it writes holds one."""
    returns = {"a template that returns no field": ["Tuple", []],
               "a return field of no bit": ["Tuple", [["Value", 16], ["Tuple", []]]]}
    assert set(returns) == set(vectors.BITLESS)
    for what, (name, outs, plan) in vectors.BITLESS.items():
        m = shaped[f"2x3 {what}"]
        r = vectors.results(m["case"])
        assert r["detail"] == r["public"] == r["kinds"] == r["exported"] == r["count"] == {"error": "templates"}, what
        fn = vectors.template(name, [("p", vectors.Value(16))], outs, plan)
        tid = codec.descriptor_id(fn)
        assert json.loads(m["case"]["instance"]["record"])["templates"] == [[tid, 2]], what
        program = json.loads(m["case"]["instance"]["program"])
        assert program["definitions"][tid]["body"]["ret_type"] == returns[what], what
        fields = ref.fields_of(fn.signature()[1])
        assert [sum(t.leaf_widths()) for t in fields] == [*(16 for _ in outs[:-1]), 0], what
        with pytest.raises(ValueError, match=f"o{len(fields) - 1}: its leaves are .* 0 in all"):
            IF.layout(fn)


def test_a_root_node_of_a_composite_is_no_input(shaped):
    """An input node is a root node of an input primitive. A composite's gates are computed whatever its key reads
    as, so a root node of a composite keyed as an input id is no input, and a record that left it out would count
    none of its gates. Such a Program is refused with its templates: the record of that one template, which lists
    it, and the accepted record beside a Program whose root has a unit of it more, which the record does not list.
    The key alone decides the first: the same template under a key that is no input id is read past its
    templates. Unlisted it is refused under either key."""
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
    for edit, stage in ((vectors._alone, "wires"), (vectors._unlisted, "templates")):
        case = copy.deepcopy(held["case"])
        edit(case, held["circuit"], plain)
        assert ref.check_detail(case["instance"], case["trusted"]) == {"error": stage}


def test_a_row_format_is_held_to_the_registration_log(shaped):
    """A record states the leaf width of a registered value whose rows are not 16-bit leaves (`registered.formats`),
    and the form holds it to the registration log: the run's own record registers the value in leaves of that width,
    or the log's first record does (θ, at that width). Held, the scan state in 32-bit leaves, of either schema, and a
    weight registered in 32-bit leaves are what they are in 16-bit ones: the verdict and every count of the accepted
    case. The same records with another width, with the format dropped, or with a format the log does not register
    are refused (`formats`); a format the form's shape refuses is refused with the declarations (`inputs`). A weight
    the first record registers in 32-bit leaves with no format in the record is no θ: its rows are fresh."""
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
    """Whether the case's record states row formats (`registered.formats`). `verity.commitments.windows.interface`
    states no row format and refuses such a record, which the form reads, so the tenth test leaves it out."""
    doc = json.loads(made["case"]["instance"]["record"])
    return isinstance(doc, dict) and isinstance(doc.get("registered"), dict) and "formats" in doc["registered"]


def _interface(made: dict):
    """The stage at which `verity.commitments.windows.interface` refuses the case's record (`wires`, `inputs` or `blocks`), else
    None with what it reads of it."""
    case = made["case"]
    doc = json.loads(case["instance"]["record"])
    lays = {codec.descriptor_id(fn): (IF.layout(fn), n) for fn, n in made["circuit"].templates}
    units = {t: n for t, (_, n) in lays.items()}
    spot = case["spot"]
    malformed = (ValueError, KeyError, TypeError, AttributeError)
    try:
        if not isinstance(spot["wires"], list):
            raise ValueError("the wires are a list")
        wires, cuts = tuple(IF.Wire.of(w) for w in spot["wires"]), IF._cuts(spot["cuts"])
        IF.check_wires(lays, wires, cuts, IF.order_of(spot))
    except malformed:
        return "wires", None
    try:
        reg = IF.Registered.of(doc["registered"]) if "registered" in doc else None
        IF.check_inputs(lays, cuts, wires, reg, IF._ingress_of(doc, "ingress"))
    except malformed:
        return "inputs", None
    try:
        blocks, _ = IF.Blocks.of(doc["blocks"], units, "")
    except malformed:
        return "blocks", None
    return None, (lays, units, blocks, doc)


def test_the_reference_reads_a_record_as_the_interface_does(random_cases, shaped):
    made = [m for m in (*random_cases, *shaped.values()) if m["case"]["spot"] is not None and not _formatted(m)]
    exact = apart = unpinned = 0
    for m in made:
        case = m["case"]
        inst, tr = case["instance"], case["trusted"]
        refusal = ref.check_detail(inst, tr).get("error")
        stage, read = _interface(m)
        assert stage == (None if refusal in OWN else refusal), (m["shape"], m.get("how"))
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
