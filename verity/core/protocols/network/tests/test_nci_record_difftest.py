"""The inference-only policy's check of an execution phase's circuit record (`NciCheck.Record`) against the Python
reference: the Python half.

`verity/core/protocols/network/nci/lean/scripts/record_difftest_vectors.py` writes 612 cases (600 random circuits,
two in five broken in one way, and 12 in the shape of a served class) and, on each,
`scripts/nci_record_reference.py`'s results, which visit every row of every unit. `check` runs the Lean half in its
Lean audit (`runs` in that package's `lean-audit.json`): `NciCheck/RecordDifftestMain.lean` fails unless
`NciCheck.Record` and `NciCheck.RecordCount` give Python's results on every case, and unless SPOT's reader reads the
same wiring from the form's instance and from the one Python builds from its own layout of each template. The audit
checks the data against the record's SHA-256 and caches its pass on the Lean package's files alone, so the first test
is what says the data is still what the generator and the reference write. The second checks that the generator
reaches what its docstring says and that the published result is the judgment alone. The third states the verdicts on
the served class's shape, the negative vectors among them. The fourth is the check that no wire crosses slots: none
does at the served class's shape, and one that does is counted. The fifth sets the form beside
`NciCheck.Desc.evaluate`'s reference on the same Program: one block for the whole phase there and one a slot here,
and input bits apart by exactly the rows no gate reads. The sixth holds the reference to
`verity.ml.windows.interface`, the rules a lowering states a record by: the stage that refuses a record, each unit's
slot, and `input_count`'s totals.

The sixth holds to `input_count` only where every opening it counts registered is one the form holds (`held`: the
reader's Definition pinned, with table reads of the table's root and domain id over the opening's units). Where one is
not, `input_count` counts that opening's ports registered and the form counts them fresh, and the test holds the two
apart by exactly those bits.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[1] / "nci" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("record_difftest_vectors")
ref, IF, codec = vectors.ref, vectors.IF, vectors.codec
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NciCheck.RecordDifftestMain"]
#: the random cases the quick tests read
SEEDS = range(300)


@pytest.fixture(scope="module")
def random_cases() -> list[dict]:
    return [vectors.case(s) for s in SEEDS]


@pytest.fixture(scope="module")
def shaped() -> dict[str, dict]:
    return {m["shape"]: m for m in vectors.shaped()}


@pytest.mark.slow
def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 612 cases and the reference's results, byte for byte as the record pins them."""
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
        assert (m["case"]["spot"] is None) == (m["broken"] in ("instance", "params", "program", "not-topological",
                                                               "record", "trusted", "run-program", "run-circuit",
                                                               "templates"))
        assert isinstance(r["kinds"], list) == ("error" not in detail)
    assert {m["broken"] for m in made} == set(vectors.BROKEN) | {None}
    assert {m.get("how") for m in made if m["broken"]} == {how for hows in vectors.BROKEN.values() for how in hows}
    good = [r for r in results if "error" not in r["detail"]]
    assert {r["detail"]["failure"] for r in good} == {None, "size", "nci"}
    assert {r["count"].get("failure") for r in good if "error" not in r["count"]} == {None, "gates", "fresh", "count"}
    assert {r["count"].get("error") for r in good} >= {None, "instance", "params", "gate-table"}
    assert {k for r in good for k in r["kinds"]} == set(ref.KINDS)
    for k in ("fresh", *ref.FRESH):
        assert any(r["detail"][k] > 0 for r in good), k
    assert any(r["detail"]["fresh"] == 0 for r in good)
    assert any(r["detail"]["blocks"] > 1 for r in good)


def test_the_served_shape(shaped):
    """The verdicts on a served class's shape: accepted with the weights registered at the start of the deployment, a
    block's fresh bits then its scan state's alone; refused by the rate rule, and by the count on its fresh bits,
    with none registered or committed, under another root, with an opening that reads no root, with an opening whose
    reader is another Definition under the pinned one's name, and with one wire from the next slot."""
    got = {what: vectors.results(m["case"]) for what, m in shaped.items()}
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


def test_no_wire_crosses_slots_at_the_served_shape(random_cases, shaped):
    """A wire from another slot's unit is the reading block's fresh input, so the form is a check that none crosses.
    At the served class's shape none does and the count is 0. One wire of one unit from the next slot is counted, its
    reader rows' bits and nothing more, and refuses the block. On any record the count is 0 exactly where every wire
    reads its reader's slot."""
    crossed = "2x3 one wire from the next slot"
    for what, m in shaped.items():
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
    """`NciCheck.Desc.evaluate`'s reference on the Program of the served class's shape, every root node that computes
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


def _interface(made: dict):
    """The stage at which `verity.ml.windows.interface` refuses the case's record (`wires`, `inputs` or `blocks`), else
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
    made = [m for m in (*random_cases, *shaped.values()) if m["case"]["spot"] is not None]
    exact = apart = unpinned = 0
    for m in made:
        case = m["case"]
        inst, tr = case["instance"], case["trusted"]
        detail = ref.check_detail(inst, tr)
        stage, read = _interface(m)
        assert stage == detail.get("error"), (m["shape"], m.get("how"))
        if read is None:
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
        theta = sorted(p for p in x.params if ref.theta(T, x.params, p))
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
