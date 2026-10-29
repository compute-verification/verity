"""What an integration hands Verity (`verity.ml.windows.interface`), against the interface alone: a circuit takes only
Verity's Definitions, a unit's rows are its Definition's bit rows as C-Flock stages them, a window tiles the partition and
commits under the window-domain rule, a label in a circuit with blocks is its run's class structure (its bounds where
the units change inference block, its step and slot and a form of the window's), an integration's outputs are checked
against the Definitions before it commits, a wire pattern wires whole members unit by unit under its writers' salts, a
cut port's rows are each wired from their own writer, a registered read's ports are tables of the registered rows under
the value's salts, every other input row is
declared registered, ingress or free and counted from the description, and the verifier's window (and its registered
reads) is rebuilt from the circuit record alone, its members stated by cap, wires, cuts and reads. A window written under
core's registration, and `of_records` against it, are the records audit's tests
(`benchmarks/prove_path/tests/test_records_audit.py`), since these may not import core."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops.boolean import cast as BC, gemm as BG
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.commitments import row_table as RT
from verity.ml.commitments.gates import embedding as E
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, bit_row
from verity.ml.windows import flock_inputs as FI, interface as I, rows as CS, wiring as WI

K = 64
COORD = bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM120_S8)
ACC = bind(G.GemmAccumulatorS8, K=K, **G.GEMM_SM120_S8)
TID, AID = descriptor_id(COORD), descriptor_id(ACC)
BY_ID = {TID: COORD, AID: ACC}
LAW = {"law": "stratified", "k": 2}
RUN = "ab" * 64


def values(n: int, seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``n`` coordinates' inputs (x, sx, wrow, sw as `evaluate_batch` takes them) and the Definition's outputs on them."""
    rng = np.random.default_rng(seed)
    x, w = (rng.integers(-128, 128, (n, K)).astype(np.int8) for _ in range(2))
    sx, sw = ((np.abs(rng.standard_normal(n)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
    return ins, [np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16)]


def window(n: int = 10, cap: int = 4) -> tuple[I.Window, list[np.ndarray], list[np.ndarray]]:
    win = I.Window(I.Circuit("test", ((COORD, n),)), RUN, LAW, cap)
    ins, outs = values(n)
    win.record(COORD, [a[:6] for a in ins], [a[:6] for a in outs], {"run": 0})
    win.record(COORD, [a[6:] for a in ins], [a[6:] for a in outs], {"run": 1})
    return win, ins, outs


def test_a_circuit_takes_only_verity_definitions():
    I.Circuit("ok", ((COORD, 3),))
    own = CompositeDefinition("IntegrationOwnIdentity", 1, (), lambda S: ((("x", Array(4, Value(16))),), Array(4, Value(16))),
                              lambda B, S, x: x, register=False)
    with pytest.raises(ValueError, match="Verity registers"):
        I.Circuit("own", ((bind(own), 3),))
    with pytest.raises(ValueError, match="complete binding"):
        I.Circuit("unbound", ((G.ScaledMmS8Coordinate.bind(K=K), 3),))
    with pytest.raises(ValueError, match="each once"):
        I.Circuit("twice", ((COORD, 3), (COORD, 2)))
    assert I.Circuit("io", ((COORD, 3),), external=(f"{TID}/o0",)).external == (f"{TID}/o0",)
    with pytest.raises(ValueError, match="no template's port"):
        I.Circuit("io", ((COORD, 3),), external=(f"{TID}/frames",))


def test_a_units_rows_are_its_definitions_bit_rows():
    lay = I.layout(COORD)
    assert lay.rows == (("x", K // 2), ("sx", 2), ("wrow", K // 2), ("sw", 2)) and lay.outs == (("o0", 1),)
    assert I.unit_rows(COORD) == ([("x", V2, 8 * K, ["x"]), ("sx", V2, 32, ["sx"]), ("wrow", V2, 8 * K, ["wrow"]),
                                   ("sw", V2, 32, ["sw"])], [(V2, (0,))])
    ins, _ = values(3)
    assert np.array_equal(I.words(ins[0], 8, 3, K // 2)[:, 0], ins[0][:, 0].astype(np.uint16) | ins[0][:, 1].astype(np.uint16) << 8)
    assert np.array_equal(I.words(ins[1], 32, 3, 2)[:, 1], (ins[1] >> 16).astype(np.uint16))
    with pytest.raises(ValueError, match="16-bit"):
        I.words(ins[1], 16, 3, 1)


def test_a_boolean_definitions_rows_are_its_ports_words():
    """A Boolean-profile Definition (every leaf one bit, as the network certifier's egress units) has the bit rows of its
    `boolean.word_type` view; a port whose words aren't whole u16 words of one width is still refused."""
    bit = Value(1)

    def boolean(name, params, ret):
        return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), lambda B, S, *xs: xs[-1], register=False))

    lay = I.layout(boolean("BooleanRows", (("b", Array(2, Array(8, bit))), ("w", Array(32, bit))), Array(32, bit)))
    assert lay.rows == (("b", 1), ("w", 2)) and lay.outs == (("o0", 2),) and lay.widths == (8, 32, 32)
    for bad in (bit, Array(8, bit), Tuple(Array(8, bit), Array(16, bit))):
        with pytest.raises(ValueError, match="a row is whole u16"):
            I.layout(boolean("BooleanBad", (("f", bad),), bad))


def test_the_window_tiles_the_partition_in_chunks():
    win, *_ = window(10, 4)
    assert [(m.base, m.n, m.name) for m in win.window.members] == [(0, 4, f"{TID}/c0"), (4, 4, f"{TID}/c1"),
                                                                   (8, 2, f"{TID}/c2")]
    assert all(m.row_schemas == [[V2, 16 * w] for _, w in m.all_rows] for m in win.window.members)
    program, query = I.Circuit("test", ((COORD, 10),)).program()
    assert CS.template_partition(program, query).population == win.window.population == 10


def test_close_commits_every_unit_under_the_window_domain_rule():
    win, ins, outs = window()
    with pytest.raises(ValueError, match="past"):
        win.record(COORD, [a[:1] for a in ins], [a[:1] for a in outs])
    c = win.close(salt_key=bytes(32))
    want = CS.expected_ports(win.window)
    assert set(c.roots) == set(want) and all(c.domains[p].domain_id.hex() == want[p]["domain"] for p in want)
    m1 = win.window.members[1]
    assert np.array_equal(c.rows[m1.port("x")], I.words(ins[0][4:8], 8, 4, K // 2))
    assert np.array_equal(c.rows[m1.port("o0")], outs[0][4:8, None])
    again = win.close(salt_key=bytes(32))
    assert again.roots == c.roots
    changed, _, _ = window()
    changed._recorded[TID][1].words[0][0, 0] ^= 1
    assert changed.close(salt_key=bytes(32)).roots[win.window.members[1].port("x")] != c.roots[m1.port("x")]


def test_close_concatenates_each_templates_words_once(monkeypatch):
    """Every member's rows are views of one array per port, not each a copy of its whole template's words: close's memory
    was members × units (249 GB at the decoder's 353,092 units)."""
    seen = []
    monkeypatch.setattr(CS, "commit", lambda window, data, **kw: seen.extend(data))
    win, *_ = window(10, 2)
    win.close()
    for p in ("x", "o0"):
        group = "outs" if p == "o0" else "rows"
        assert len({id(d[group][p].base) for d in seen}) == 1


def test_a_window_closes_over_every_unit():
    win = I.Window(I.Circuit("test", ((COORD, 10),)), RUN, LAW, 4)
    ins, outs = values(10)
    win.record(COORD, [a[:6] for a in ins], [a[:6] for a in outs])
    with pytest.raises(ValueError, match="closes over all"):
        win.close()


def test_labels_are_opaque_runs_of_each_members_units():
    win, ins, outs = window()
    assert [m["labels"] for m in win.labels()["members"]] == [[[0, 4, {"run": 0}]], [[4, 6, {"run": 0}], [6, 8, {"run": 1}]],
                                                             [[8, 10, {"run": 1}]]]
    with pytest.raises(TypeError):
        I.Window(I.Circuit("test", ((COORD, 10),)), RUN, LAW, 4).record(COORD, ins, outs, object())


#: twelve coordinates at 3 steps of 2 slots, two units an inference block: [0, 2) slot 0 step 0, [2, 4) slot 1 step 0,
#: [4, 6) slot 0 step 1, and so on
STEPS_SLOTS = I.Blocks(2, 3, (I.InferenceEntry(TID, 0, (3, 2, 2), ("step", "slot", "-"), 0),))
FORMS = ("node", "evaluated")
#: each inference block's units a run, labelled by its step, slot and form
HONEST = [(2, {"step": t, "slot": s, "form": FORMS[t % 2]}) for t in range(3) for s in range(2)]


def labelled(runs: list[tuple[int, object]], forms: tuple[str, ...] = FORMS) -> I.Window:
    """A window of `STEPS_SLOTS`' twelve coordinates in members of 4, each run ``(units, label)`` recorded in turn."""
    win = I.Window(I.Circuit("test", ((COORD, 12),), blocks=STEPS_SLOTS), RUN, LAW, 4, forms=forms)
    (ins, outs), at = values(12), 0
    for n, label in runs:
        win.record(COORD, [a[at:at + n] for a in ins], [a[at:at + n] for a in outs], label)
        at += n
    return win


def test_a_label_in_a_circuit_with_blocks_is_its_runs_class_structure():
    """A labelled run starts and ends where its template's units change inference block, labelled by its step and slot and
    a form of the window's: the honest labelling commits what the unlabelled window does, and the index shows only those
    bounds and the members', an unlabelled stretch of runs one run."""
    win = labelled(HONEST)
    assert win.close(salt_key=bytes(32)).roots == labelled([(12, None)]).close(salt_key=bytes(32)).roots
    assert [m["labels"] for m in win.labels()["members"]] == [[[4 * m + 2 * j, 4 * m + 2 * j + 2, HONEST[2 * m + j][1]]
                                                               for j in range(2)] for m in range(3)]
    whole = labelled([(4, {"step": 0}), (2, {"slot": 0, "form": "evaluated"}), (1, None), (1, None), (4, {})])
    assert [m["labels"] for m in whole.labels()["members"]] == [[[0, 4, {"step": 0}]],
                                                                [[4, 6, {"slot": 0, "form": "evaluated"}], [6, 8, None]],
                                                                [[8, 12, {}]]]
    with pytest.raises(ValueError, match="a window's forms are names"):
        labelled([], (0,))


@pytest.mark.parametrize("runs, forms, why", [
    ([(3, {"step": 0})], FORMS, "starts and ends where its template's units change inference block"),
    ([(1, None), (2, {"step": 0})], FORMS, "starts and ends where its template's units change inference block"),
    ([(4, {"step": 0, "slot": 0})], FORMS, "its units are at slot [0, 1], not its label's 0"),
    ([(2, {"step": 1, "slot": 0})], FORMS, "its units are at step [0], not its label's 1"),
    ([(2, {"step": 0, "slot": 0, "request": 3})], FORMS, "class structure, never a value"),
    ([(2, {"step": 0, "form": "served"})], FORMS, "class structure, never a value"),
    ([(2, {"step": 0, "form": "node"})], (), "class structure, never a value"),
    ([(2, {"step": "0"})], FORMS, "class structure, never a value"),
    ([(2, {"slot": True})], FORMS, "class structure, never a value"),
    ([(2, [0, 0])], FORMS, "class structure, never a value"),
], ids=["ends-inside-a-block", "starts-inside-a-block", "two-slots-as-one", "another-step", "extra-key", "form-outside-forms",
        "window-without-forms", "step-a-string", "slot-a-bool", "not-a-mapping"])
def test_a_label_that_is_not_its_runs_class_structure_is_refused(runs, forms, why):
    with pytest.raises(ValueError, match=re.escape(why)):
        labelled(runs, forms)


def test_mismatches_are_the_units_whose_outputs_are_not_the_definitions():
    win, ins, outs = window()
    assert win.mismatches() == {}
    bad = I.Window(I.Circuit("test", ((COORD, 10),)), RUN, LAW, 4)
    wrong = outs[0].copy()
    wrong[7] ^= 1
    bad.record(COORD, ins, [wrong])
    assert bad.mismatches() == {TID: [7]}


ROUND, WIDEN = bind(BG.F2fpBf16), bind(BC.Bf16ToF32)
RID, WID = descriptor_id(ROUND), descriptor_id(WIDEN)


def rounded(n: int = 6) -> tuple[I.Circuit, np.ndarray, np.ndarray, np.ndarray]:
    """``n`` f32 words rounded to bf16 (``F2fpBf16_v2``) and widened back (``Bf16ToF32_v2``), each widening's ``a`` wired from
    its rounding's output: Boolean-profile Definitions, every leaf one bit and every value its `boolean.word_type` word, as
    the evaluator's word view computes it. -> the circuit, the f32, bf16 and widened words."""
    f32 = np.random.default_rng(5).standard_normal(n).astype(np.float32).view(np.uint32)
    bf16 = evaluate_words(ROUND, f32).astype(np.uint16)
    wire = I.Wire(WID, "a", RID, "o0", at=0, radix=(n,), base=0, step=(1,))
    return I.Circuit("bits", ((ROUND, n), (WIDEN, n)), wires=(wire,)), f32, bf16, evaluate_words(WIDEN, bf16).astype(np.uint32)


@pytest.mark.parametrize("given", [True, False], ids=["in-full", "by-reference"])
def test_a_boolean_profile_ports_rows_are_read_bit_by_bit(given):
    """`Window.mismatches` on Boolean-profile templates: the Definition reads each port's rows bit by bit, the widening's
    ``a`` recorded in full or read through its wire, so honest values have no mismatch, and one bit flipped in a rounding's
    output (which its widening reads), an f32's sign or a widened word is that unit's mismatch."""
    circuit, f32, bf16, wide = rounded()

    def mismatches(f: np.ndarray, b: np.ndarray, w: np.ndarray) -> dict[str, list[int]]:
        win = I.Window(circuit, RUN, LAW, 4)
        win.record(ROUND, [f], [b])
        win.record(WIDEN, [b if given else None], [w])
        return win.mismatches()

    def flip(a: np.ndarray, unit: int, bit: int) -> np.ndarray:
        a = a.copy()
        a[unit] ^= a.dtype.type(1) << a.dtype.type(bit)
        return a

    assert mismatches(f32, bf16, wide) == {}
    assert mismatches(f32, flip(bf16, 2, 5), wide) == {RID: [2], WID: [2]}
    assert mismatches(flip(f32, 4, 31), bf16, wide) == {RID: [4]}
    assert mismatches(f32, bf16, flip(wide, 1, 16)) == {WID: [1]}


def test_several_templates_are_one_population():
    w = I.Circuit("two", ((COORD, 3), (ACC, 2)))
    win = I.Window(w, RUN, LAW, 2)
    assert [(m.template, m.base, m.n) for m in win.window.members] == [(TID, 0, 2), (TID, 2, 1), (AID, 3, 2)]
    assert I.layout(ACC).outs == (("o0", 2),)


@pytest.mark.parametrize("templates", [((COORD, 10),), ((COORD, 3), (ACC, 2))])
def test_the_verifiers_window_is_the_circuit_records(templates):
    win = I.Window(I.Circuit("rec", templates), RUN, LAW, 4)
    doc = json.loads(I.circuit_bytes(win.circuit_record()))
    mine = I.window_of_record(doc, LAW, BY_ID.__getitem__)
    same = ("query", "law", "run", "members", "row_tables", "context", "work_table", "partition", "partition_digest")
    assert all(getattr(mine, x) == getattr(win.window, x) for x in same)
    assert CS.expected_ports(mine) == CS.expected_ports(win.window)
    assert doc["statements"][TID] == {"in_rows": [list(r[:3]) + [r[3]] for r in I.unit_rows(COORD)[0]],
                                      "out_rows": [[V2, [0]]], "reads": "inline"}


def test_a_circuit_record_its_templates_dont_make_is_refused():
    win = I.Window(I.Circuit("rec", ((COORD, 10),)), RUN, LAW, 4)
    doc = json.loads(I.circuit_bytes(win.circuit_record()))
    listed = json.loads(I.circuit_bytes(I.circuit_record(win.window, "rec", [(TID, 10)])))
    with pytest.raises(ValueError, match="not the partition's units"):
        I.window_of_record({**listed, "templates": [[TID, 11]]}, LAW, BY_ID.__getitem__)
    for bad in ({**doc, "templates": [[TID, 11]]}, {**doc, "partition": "00" * 64}, {**listed, "partition": "00" * 64}):
        with pytest.raises(ValueError, match="not the ones its templates make"):
            I.window_of_record(bad, LAW, BY_ID.__getitem__)
    with pytest.raises(ValueError, match="resolves"):
        I.window_of_record(doc, LAW, lambda t: ACC)
    with pytest.raises(ValueError, match="window-circuit"):
        I.window_of_record({**doc, "format": "x"}, LAW, BY_ID.__getitem__)


GEMM = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
GID = descriptor_id(GEMM)
REQUESTS, STEPS, LAYERS = 2, 2, 2


def chain() -> I.Circuit:
    """Two requests' two decode steps through two layers, a ``Gemm_v2{K=16, N=16}`` unit each, in (layer, step, request)
    order: layer 1 reads layer 0's output at its step and request, layer 0 at step s + 1 reads layer 1's output at step s,
    and layer 0's input at step 0 (the embedding) is external."""
    layer = REQUESTS * STEPS
    return I.Circuit("chain", ((GEMM, LAYERS * layer),), external=(f"{GID}/x",),
                     wires=(I.Wire(GID, "x", GID, "o0", at=layer, radix=(STEPS, REQUESTS), base=0, step=(REQUESTS, 1)),
                            I.Wire(GID, "x", GID, "o0", at=REQUESTS, radix=(STEPS - 1, REQUESTS), base=layer,
                                   step=(REQUESTS, 1))))


def chain_values(seed: int = 0, first: np.ndarray | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """`chain`'s units' inputs (x, w) and outputs in its order, each unit's input its writer's output; ``first``: each
    request's input at layer 0, step 0 (else drawn)."""
    rng = np.random.default_rng(seed)

    def bf16(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    w = [bf16(rng.standard_normal((16, 16)) / 4) for _ in range(LAYERS)]
    x = {(0, 0, r): bf16(rng.standard_normal(16)) if first is None else first[r] for r in range(REQUESTS)}
    y = {}
    for s in range(STEPS):
        for layer in range(LAYERS):
            for r in range(REQUESTS):
                if (layer, s) != (0, 0):
                    x[layer, s, r] = y[0, s, r] if layer else y[1, s - 1, r]
                y[layer, s, r] = np.asarray(evaluate_batch(GEMM, x[layer, s, r][None], w[layer][None], kernel="reference"),
                                            dtype=np.uint16).reshape(16)
    order = [(layer, s, r) for layer in range(LAYERS) for s in range(STEPS) for r in range(REQUESTS)]
    return [np.stack([x[k] for k in order]), np.stack([w[k[0]] for k in order])], [np.stack([y[k] for k in order])]


def test_a_wire_pattern_wires_whole_members_unit_by_unit_under_their_writers_salts():
    """Each member's wired port reads, unit by unit, the writer the pattern gives it; the committed rows' public strings are
    the writers' (a5's wiring check), a wire moved to other writers is refused there (on the drawn units alone, the units
    it draws), and a reader row that isn't its writer's output doesn't commit."""
    win = I.Window(chain(), RUN, LAW, 2)
    assert [(m.base, m.n, m.wired) for m in win.window.members] == [
        (0, 2, ()), (2, 2, (("x", "o0", (4, 5)),)), (4, 2, (("x", "o0", (0, 1)),)), (6, 2, (("x", "o0", (2, 3)),))]
    ins, outs = chain_values()
    win.record(GEMM, ins, outs, {"chain": 0})
    assert win.mismatches() == {}
    c = win.close(salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(win.window.members)}
    assert WI.check(win.window.members, publics) == {}
    moved = [replace(m, wired=(("x", "o0", m.wired[0][2][::-1]),)) if m.wired else m for m in win.window.members]
    assert set(WI.check(moved, publics)) == {m.key for m in moved if m.wired}
    everyone = range(sum(m.n for m in win.window.members))
    assert WI.check_drawn(win.window.members, publics, everyone) == {}
    assert set(WI.check_drawn(moved, publics, everyone)) == set(WI.check(moved, publics))
    assert WI.check_drawn(moved, publics, []) == {}
    assert list(WI.check_drawn(moved, publics, [3])) == [moved[1].key]
    lacking = [replace(m, wired=(("x", "o1", m.wired[0][2]),)) if m.wired else m for m in win.window.members]
    assert list(WI.check_drawn(lacking, publics, [3])) == [moved[1].key]
    bad = I.Window(chain(), RUN, LAW, 2)
    x = ins[0].copy()
    x[2, 0] ^= 1
    bad.record(GEMM, [x, ins[1]], outs)
    with pytest.raises(ValueError, match="is not the o0 row"):
        bad.close(salt_key=bytes(32))


def test_the_record_states_members_by_cap_and_wires_and_the_verifier_derives_them():
    win = I.Window(chain(), RUN, LAW, 2)
    doc = json.loads(I.circuit_bytes(win.circuit_record()))
    by_id = {GID: GEMM}.__getitem__
    assert "members" not in doc and doc["cap"] == 2 and tuple(I.Wire.of(w) for w in doc["wires"]) == chain().wires
    mine = I.window_of_record(doc, LAW, by_id)
    assert mine.members == win.window.members and mine.partition_digest == win.window.partition_digest
    assert I.window_of_record({**doc, "wires": [{**doc["wires"][0], "base": 1}, doc["wires"][1]]}, LAW, by_id).members \
        != win.window.members
    for bad, why in [({k: v for k, v in doc.items() if k != "cap"}, "one or the other"), ({**doc, "members": []}, "one or the other"),
                     ({**doc, "wires": [{**doc["wires"][0], "radix": 2}]}, "radix and step lists"),
                     ({**doc, "wires": [{**doc["wires"][0], "extra": 0}]}, "a wire's record"),
                     ({**doc, "cap": 0}, "at least one unit")]:
        with pytest.raises(ValueError, match=why):
            I.window_of_record(bad, LAW, by_id)
    split = I.window_of_record({**doc, "cap": 3}, LAW, by_id).members
    assert [(m.base, m.n, m.wired) for m in split] == [
        (0, 2, ()), (2, 2, (("x", "o0", (4, 5)),)), (4, 3, (("x", "o0", (0, 1, 2)),)), (7, 1, (("x", "o0", (3,)),))]
    listed = json.loads(I.circuit_bytes(I.circuit_record(win.window, "chain", [(GID, 8)])))
    assert "cap" not in listed and I.window_of_record(listed, LAW, by_id).members == win.window.members
    with pytest.raises(ValueError, match="stated with a cap"):
        I.circuit_record(win.window, "chain", [(GID, 8)], wires=chain().wires)


def test_a_wire_reads_whole_rows_among_its_templates_units_each_reader_port_once():
    """A wire's checks are per wire: its ends are its templates' ports, its reader and writer units are among theirs, it reads
    a row of its writer's words (a gathered row is refused), and no reader unit's port is wired twice. A step of 0 is a
    fan-out."""
    w = {"reader": GID, "port": "x", "writer": GID, "out": "o0", "at": 4, "radix": (2, 2), "base": 0, "step": (2, 1)}
    I.Circuit("ok", ((GEMM, 8),), wires=(I.Wire(**w),))
    for edit, why in [({"writer": TID}, "templates of the circuit"), ({"port": "o0"}, "parameters"), ({"out": "x"}, "outputs"),
                      ({"port": "w"}, "gathered from several writers"), ({"at": 5}, "reader units"),
                      ({"base": 5}, "writer units"), ({"step": (-2, 1)}, "writer units"), ({"radix": ()}, "at least one digit"),
                      ({"step": (2,)}, "one integer step a digit"), ({"at": True}, "a first reader unit")]:
        with pytest.raises(ValueError, match=why):
            I.Circuit("bad", ((GEMM, 8),), wires=(I.Wire(**{**w, **edit}),))
    with pytest.raises(ValueError, match="both fill rows of units from 5"):
        I.Circuit("twice", ((GEMM, 8),), wires=(I.Wire(**w), I.Wire(**{**w, "at": 5, "radix": (1, 2)})))
    fan = I.Wire(GID, "x", GID, "o0", at=4, radix=(2, 2), base=1, step=(0, 2))
    assert fan.writers(4, 4).tolist() == [1, 3, 1, 3] and fan.span() == (1, 3)


EMB = I.resolve("EmbeddingRead_v1{V=4,H=16}")
G8 = bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
EID, G8ID = descriptor_id(EMB), descriptor_id(G8)
WEIGHTS = 512 + 4 * 16
#: the verifier's pin of ``weights``, and the root a window whose registration records the same root records
PIN = {"weights": "ab" * 64}


def declared(**edit) -> I.Circuit:
    """Three positions' embedding reads (ingress at step 0 for the first two, the third's token free), each read's row two
    ``Gemm_v2{K=16, N=8}`` slices of one fan-out, and a ``Gemm_v2{K=16, N=16}`` a position reading its ``x`` cut in two, a
    row from each slice; every weight a registered read and the table's root, domain and row the registration's."""
    parts = dict(
        wires=(I.Wire(G8ID, "x", EID, "o0", at=0, radix=(3, 2), base=0, step=(1, 0)),
               I.Wire(GID, "x", G8ID, "o0", at=0, radix=(3, 2), base=0, step=(2, 1), cut=(0, 2))),
        cuts={GID: {"x": 2}},
        registered=I.Registered(
            {"weights": WEIGHTS}, (I.Table("embed", "weights", 512, 16, 4),),
            (I.Read(G8ID, "w", "weights", at=0, radix=(3, 2), base=0, step=(0, 128)),
             I.Read(GID, "w", "weights", at=0, radix=(3,), base=256, step=(0,))),
            (I.TableRead(EID, "root", "embed", "root", 0, 3), I.TableRead(EID, "domain", "embed", "domain", 0, 3)),
            (I.Opening(EID, "embed", "tok", ("row", "salt", "path"), "o1", 0, 3),)),
        ingress=(I.Ingress(EID, "tok", 0, 2, 0),))
    return I.Circuit("declared", ((EMB, 3), (G8, 6), (GEMM, 3)), **{**parts, **edit})


def sliced(**edit) -> I.Circuit:
    """Three positions' rows, each two ``Gemm_v2{K=16, N=8}`` slices, and a ``Gemm_v2{K=16, N=16}`` a position reading its
    ``x`` cut in two, a row from each slice."""
    parts = dict(wires=(I.Wire(GID, "x", G8ID, "o0", at=0, radix=(3, 2), base=0, step=(2, 1), cut=(0, 2)),),
                 cuts={GID: {"x": 2}})
    return I.Circuit("sliced", ((G8, 6), (GEMM, 3)), **{**parts, **edit})


def test_a_cut_port_is_rows_each_wired_from_its_own_writer():
    """A port cut ``k`` ways is ``k`` rows of its outer elements, and a cut wire's last digit walks a reader unit's rows: each
    member's ``x#j`` rows are the slices' outputs, and the record states the cuts, from which the verifier derives them."""
    c = sliced()
    assert I.unit_rows(GEMM, {"x": 2})[0][:2] == [("x#0", V2, 128, [f"x[{r}]" for r in range(8)]),
                                                  ("x#1", V2, 128, [f"x[{r}]" for r in range(8, 16)])]
    ms = I.members(c, 2)
    assert [(m.template, m.base, m.n, m.wired) for m in ms] == [
        (G8ID, 0, 2, ()), (G8ID, 2, 2, ()), (G8ID, 4, 2, ()),
        (GID, 6, 2, (("x#0", "o0", (0, 2)), ("x#1", "o0", (1, 3)))), (GID, 8, 1, (("x#0", "o0", (4,)), ("x#1", "o0", (5,))))]
    assert ms[-1].rows == (("x#0", 8), ("x#1", 8), ("w", 256))
    rec = I.descriptor_record(c, 2)
    assert rec["cuts"] == {GID: {"x": 2}} and rec["statements"][GID]["in_rows"][1][:3] == ["x#1", V2, 128]
    assert "run" not in rec and I.window_of_record({**rec, "run": RUN, "context": ""}, LAW, I.resolve).members == ms
    for cuts, why in [({GID: {"x": 3}}, "divides its outer"), ({GID: {"y": 2}}, "names a template's parameter"),
                      ({}, r"rows \[0, 2\) of x's 1")]:
        with pytest.raises(ValueError, match=why):
            sliced(cuts=cuts)
    for edit, why in [({"radix": (2, 3)}, "its last digit"), ({"cut": (1, 2)}, r"rows \[1, 3\)")]:
        with pytest.raises(ValueError, match=why):
            sliced(wires=(replace(c.wires[0], **edit),))
    with pytest.raises(ValueError, match="both fill rows"):
        sliced(wires=(*c.wires, I.Wire(GID, "x", G8ID, "o0", at=2, radix=(1,), base=0, step=(0,), cut=(1, 1))))


def test_a_proof_block_c_flock_would_refuse_is_refused_at_lowering():
    """A template passes with the k_log its rows alone need, and is refused, its message naming it, its cuts, its rows and
    the fix, when its rows are more than C-Flock's composer takes, none is past 111 bytes, or they alone need a C-Flock block
    past 2^27 bits."""
    assert I.check_proof_block(GEMM) == I.check_proof_block(GEMM, {"x": 2}) == 21
    wide = bind(G.Gemm, K=16, N=256, **G.GEMM_AMPERE)
    with pytest.raises(ValueError, match=re.escape(f"{descriptor_id(wide)} cut {{'w': 256}}: x 32 B, w 256 rows of 32 B, "
                                                   "o0 512 B: 258 rows, past a proof block's 256")):
        I.check_proof_block(wide, {"w": 256})
    with pytest.raises(ValueError, match=f"{re.escape(GID)} cut .*: no row past 111 B.*bind the Definition larger"):
        I.check_proof_block(GEMM, {"w": 16})
    with pytest.raises(ValueError, match=r"need a C-Flock block of 2\^28 bits, past 2\^27: bind the Definition smaller"):
        I.check_proof_block(bind(G.Gemm, K=2048, N=48, **G.GEMM_AMPERE))


def test_the_proof_block_limits_are_c_flocks_composers():
    """`check_proof_block`'s sizes are C-Flock's composer's, and `rows_k_log` is the k_log its ranges give a unit's rows,
    holding every gate range's k_log at or above it: a row has a compression before its last block exactly past
    ``LAST_BLOCK_BYTES``, and `MAX_ROWS` is the most rows whose hm96 range leaves a 2^K_MAX-bit C-Flock block room."""
    from verity_flock import circuit as FC

    assert (I.K_MAX, I.SLOT_LOG, I.PER_SLOT, I.MASK_LOG) == (FC.K_MAX, FC.SHA_LOG, FC.PER_SLOT, FC.MASK_LOG)
    assert FC.HM_LOG == FC.SHA_LOG and FC._ranges([("mask", FC.MASK_LOG, 1, {})])[1] == I.K_MIN
    assert all((FC.Port("r", b // 2, 8 * b).comps > 0) == (b > I.LAST_BLOCK_BYTES) for b in range(2, 2048, 2))
    rng = np.random.default_rng(0)
    for n in (*range(1, 10), 16, 17, 100, 255, 256, 257):
        for _ in range(4):
            rb = [int(b) for b in 2 * rng.integers(1, rng.choice([56, 1 << 9, 1 << 17]), n)]
            ports = [FC.Port(f"r{j}", b // 2, 8 * b) for j, b in enumerate(rb)]
            if (n_sha := FC._row_slots(ports)[0]) == 0:
                continue
            rows = [("unit", FC.SHA_LOG, n_sha, {"packed": True}), ("unit", FC.HM_LOG, FC._pow2(n), {}),
                    ("mask", FC.MASK_LOG, 1, {})]
            assert I.rows_k_log(rb) == FC._ranges(rows)[1], rb
            gates = ("unit", int(rng.integers(7, 26)), int(rng.integers(1, 64)), {"packed": bool(rng.integers(2))})
            assert I.rows_k_log(rb) <= FC._ranges([*rows, gates])[1]
    assert I.rows_k_log([112] * I.MAX_ROWS) == I.K_MAX < I.rows_k_log([112] * (I.MAX_ROWS + 1))


def sliced_values(seed: int = 0) -> tuple[list[np.ndarray], ...]:
    """`sliced`'s units' values: the six slices' (x, w) and outputs, then the three positions' (x, w) and outputs, a
    position's ``x`` its two slices' outputs."""
    rng = np.random.default_rng(seed)

    def bf16(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    x8, w8 = bf16(rng.standard_normal((6, 16))), bf16(rng.standard_normal((6, 8, 16)) / 4)
    y8 = np.asarray(evaluate_batch(G8, x8, w8, kernel="reference"), dtype=np.uint16).reshape(6, 8)
    x, w = y8.reshape(3, 16), bf16(rng.standard_normal((3, 16, 16)) / 4)
    return [x8, w8], [y8], [x, w], [np.asarray(evaluate_batch(GEMM, x, w, kernel="reference"), dtype=np.uint16).reshape(3, 16)]


def test_a_window_commits_a_cut_ports_rows_each_under_its_writers_salt():
    """A port cut in two is committed as two rows, each its words' half: a position's ``x#0`` and ``x#1`` are its slices'
    outputs, the wiring check holds on the committed bodies, the verifier derives the same members from the record, which
    states the cut rows, and a cut row that isn't its writer's output doesn't commit."""
    c = sliced()
    ins8, outs8, ins, outs = sliced_values()
    win = I.Window(c, RUN, LAW, 2)
    win.record(G8, ins8, outs8)
    win.record(GEMM, ins, outs)
    assert win.mismatches() == {}
    got = win.close(salt_key=bytes(32))
    ms, y8 = win.window.members, outs8[0]
    assert np.array_equal(got.rows[ms[3].port("x#0")], y8[[0, 2]]) and np.array_equal(got.rows[ms[3].port("x#1")], y8[[1, 3]])
    publics = {m.key: got.public_body(k) for k, m in enumerate(ms)}
    assert WI.check(ms, publics) == {} and WI.check_drawn(ms, publics, range(9)) == {}
    doc = json.loads(I.circuit_bytes(win.circuit_record()))
    assert doc["cuts"] == {GID: {"x": 2}} and doc["statements"][GID]["in_rows"][1][:3] == ["x#1", V2, 128]
    assert I.window_of_record(doc, LAW, {GID: GEMM, G8ID: G8}.__getitem__).members == ms
    x = ins[0].copy()
    x[1, 8] ^= 1
    bad = I.Window(c, RUN, LAW, 2)
    bad.record(G8, ins8, outs8)
    bad.record(GEMM, [x, ins[1]], [np.asarray(evaluate_batch(GEMM, x, ins[1], kernel="reference"), dtype=np.uint16).reshape(3, 16)])
    with pytest.raises(ValueError, match="x#1 row is not the o0 row"):
        bad.close(salt_key=bytes(32))


def test_every_input_row_is_wired_registered_ingress_or_free_and_counted_from_the_description():
    """`input_count` counts each kind of input from the circuit's description alone, never its program; the one free row is
    the third token. A declaration outside its port, units, parameter or table, or on a row another fills, is refused."""
    rec = declared().description()
    words = 3 * (2 + 32 + 32 + 16 + 96 + 64) + 6 * (16 + 128) + 3 * (16 + 256)
    assert I.input_count(rec, pins=PIN, roots=PIN) == {"units": 12, "inputs": 16 * words, "wired": 16 * (6 * 16 + 6 * 8),
                                                       "registered": 16 * (6 * 128 + 3 * 256 + 3 * 64 + 3 * 176),
                                                       "ingress": 16 * 2 * 2, "free": 32}
    assert I.input_count({**rec, "ingress": [{**rec["ingress"][0], "units": 3}]}, pins=PIN, roots=PIN)["free"] == 0
    c = declared()
    reg = c.registered
    for edit, why in [
        (dict(ingress=(I.Ingress(EID, "tok", 0, 4, 0),)), r"units \[0, 4\)"),
        (dict(ingress=(I.Ingress(EID, "row", 0, 1, 0),)), "both fill rows"),
        (dict(ingress=(I.Ingress(EID, "tok", 0, 1, -1),)), "at a step"),
        (dict(registered=replace(reg, reads=(replace(reg.reads[1], base=WEIGHTS - 255),))), "leaves"),
        (dict(registered=replace(reg, roots=(I.TableRead(EID, "salt", "embed", "root", 0, 3),))), "64-byte port"),
        (dict(registered=replace(reg, roots=(I.TableRead(EID, "root", "other", "root", 0, 3),))), "declared table"),
        (dict(registered=replace(reg, openings=(replace(reg.openings[0], verdict="o2"),))), "verdict"),
        (dict(registered=replace(reg, tables=(I.Table("embed", "weights", 512, 16, 5),))), "registered parameter's leaves"),
        (dict(registered=replace(reg, reads=(*reg.reads, I.Read(G8ID, "x", "weights", 0, (1,), 0, (0,))))), "both fill rows"),
    ]:
        with pytest.raises(ValueError, match=why):
            declared(**edit)
    with pytest.raises(ValueError, match="both fill rows"):
        I.input_count({**rec, "ingress": [*rec["ingress"], {**rec["ingress"][0], "at": 1}]})


def test_a_registered_read_is_registered_only_where_the_windows_root_is_the_one_the_verifier_pinned_before_the_run():
    """The third token read from ``acts``, a parameter whose root the prover commits in the window, is free; the same read
    at a pinned root the window's registration records is registered. A window whose ``weights`` root is another than the
    pin, or records none, has every read, table read and opening of ``weights`` free, and so has a window with no pins
    (a bare string is none). A set of names is no pins: refused, as is a root not hex."""
    reg = declared().registered
    rec = declared(registered=replace(reg, params={**reg.params, "acts": 2}, reads=(
        *reg.reads, I.Read(EID, "tok", "acts", at=2, radix=(1,), base=0, step=(0,))))).description()
    pinned = I.input_count(rec, pins=PIN, roots=PIN)
    weights = 16 * (6 * 128 + 3 * 256 + 3 * 64 + 3 * 176)
    assert (pinned["registered"], pinned["free"]) == (weights, 32)
    both = {**PIN, "acts": "12" * 64}
    assert I.input_count(rec, pins=both, roots=both) == {**pinned, "registered": weights + 32, "free": 0}
    assert I.input_count(rec, pins=both, roots={**both, "acts": "13" * 64}) == pinned
    assert I.input_count(rec, pins=PIN, roots={"weights": PIN["weights"].upper()}) == pinned
    free = {**pinned, "registered": 0, "free": weights + 32}
    assert (I.input_count(rec) == I.input_count(rec, pins="weights") == I.input_count(rec, pins=PIN)
            == I.input_count(rec, pins=PIN, roots={"weights": "cd" * 64}) == I.input_count(rec, pins="weights", roots=PIN)
            == free)
    assert I.held_params(PIN, {"weights": "cd" * 64, "acts": "ab" * 64}) == frozenset()
    for pins, roots in [({"weights"}, PIN), (["weights"], PIN), (PIN, {"weights"}), ({"weights": "zz"}, PIN),
                        ({"weights": bytes(64)}, PIN), (PIN, {"weights": None})]:
        with pytest.raises(ValueError, match=r"\{param: root\}"):
            I.input_count(rec, pins=pins, roots=roots)
    assert I.registered_roots({"run": RUN, "registered_roots": PIN}) == PIN and I.registered_roots({"run": RUN}) == {}
    with pytest.raises(ValueError, match=r"\{param: root\}"):
        I.registered_roots({"registered_roots": {"weights": {"root": PIN["weights"]}}})


def test_coalesce_merges_runs_of_like_patterns():
    """Per-unit wires and reads coalesce to the patterns a reader's runs make, and read the same rows."""
    c = declared()
    one = [I.Wire(G8ID, "x", EID, "o0", at=u, radix=(1,), base=u // 2, step=(0,)) for u in range(6)]
    rows = [I.Wire(GID, "x", G8ID, "o0", at=p, radix=(2,), base=2 * p, step=(1,), cut=(0, 2)) for p in range(3)]
    reads = [I.Read(G8ID, "w", "weights", at=u, radix=(1,), base=128 * (u % 2), step=(0,)) for u in range(6)]
    assert set(I.coalesce(one + rows)) == set(c.wires)
    assert I.coalesce(reads) == [c.registered.reads[0]]

    def sources(ps):
        return {(p.port, p.at + i // p.cut[1], p.cut[0] + i % p.cut[1]): int(p.source(i)) for p in ps for i in range(p.units)}

    gapped = one[:2] + one[3:]
    assert len(I.coalesce(gapped)) == 3 and sources(I.coalesce(gapped)) == sources(gapped)


class Held:
    """A registered value as the committer reads core's ``one_stage.registered.Registered``: its rows of ``words`` u16 words,
    one salt a row, a root and each row's path (a stand-in's), and a statement's table of the rows at some positions."""

    def __init__(self, name: str, rows: np.ndarray) -> None:
        self.name, self.rows, self.words, self.leaves = name, rows, rows.shape[1], len(rows)
        self.value = name
        self.salts = [hashlib.shake_256(b"salt %d" % i).digest(CS.SALT_BYTES) for i in range(len(rows))]
        self.root = hashlib.sha512(b"root " + name.encode()).digest()

    def path(self, position: int) -> bytes:
        return hashlib.sha512(b"path %d " % position + self.name.encode()).digest()

    def statement(self, positions):
        return ([self.rows[p] for p in positions], [self.salts[p] for p in positions],
                {"value": self.name, "positions": list(positions), "paths": [self.path(p).hex() for p in positions]})


class Supplied(Held):
    """A supplied value as core's ``Registered.commit`` makes it for `interface.supplied`: its rows under the salts given,
    its entry's root and domain digests of what it was committed from."""

    @classmethod
    def commit(cls, program, value, rows, salts, schema):
        v = cls(value, np.array(rows, dtype=np.uint16))
        v.salts, v.program, v.schema = list(salts), program, schema
        return v

    def entry(self):
        return {"value": self.name, "leaves": self.leaves, "words": self.words,
                "root": hashlib.sha512(repr((self.program, self.rows.tolist(), self.salts, self.schema)).encode()).hexdigest(),
                "domain": hashlib.sha512(f"{self.program} {self.name}".encode()).hexdigest()}


#: core's ``one_stage.registered`` names `interface.registered_reads` and `interface.supplied` read, as the caller passes
#: the module
REG = type("REG", (), {"WORD_BITS": 16, "SCHEMA_KEY": "row_schema", "READS_FROM": "reads_from",
                       "READS_V1": "verity/registered-reads/v1", "Registered": Supplied,
                       "supplied_salts": staticmethod(lambda program, value, n: [
                           hashlib.shake_256(f"{program} {value} {i}".encode()).digest(CS.SALT_BYTES) for i in range(n)])})
#: 12 units, unit u reading row u // 2 of six weight rows
READ = I.Read(TID, "wrow", "w", 0, (6, 2), 0, (K // 2, 0))


def registered(n: int = 12, cap: int = 4) -> tuple[I.Window, Held]:
    w = Held("w", np.random.default_rng(1).integers(0, 1 << 16, (6, K // 2), dtype=np.uint16))
    circuit = I.Circuit("reads", ((COORD, n),), registered=I.Registered({"w": 6 * K // 2}, reads=(READ,)))
    win = I.Window(circuit, RUN, LAW, cap, registered={"w": w})
    ins, _ = values(n)
    ins[2] = w.rows[np.arange(n) // 2].view(np.uint8)
    win.record(COORD, ins, [np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16)])
    return win, w


def test_a_circuit_declares_its_registered_reads_and_the_record_counts_its_free_inputs():
    """The record format: the reads in the record's ``registered``, its other fields only when declared, and the free
    input bits counted from the description alone. A pin names its parameter alone: ``w``'s holds no ``w/a@8``, whose
    reads count registered only at its own pin and the window's same root for it."""
    win, _ = registered()
    doc = json.loads(I.circuit_bytes(win.circuit_record()))
    assert I.Registered.of(doc["registered"]) == win.circuit.registered and "ingress" not in doc
    assert doc["registered"]["reads"] == [{"reader": TID, "port": "wrow", "param": "w", "at": 0, "radix": [6, 2], "base": 0,
                                           "step": [K // 2, 0]}]
    bits = 16 * sum(w for _, w in I.layout(COORD).rows) * 12
    w = {"w": "ab" * 64}
    assert I.input_count(doc, BY_ID.__getitem__, w, w) == {"units": 12, "inputs": bits, "wired": 0, "registered": 12 * 8 * K,
                                                           "ingress": 0, "free": bits - 12 * 8 * K}
    assert I.input_count(doc, BY_ID.__getitem__, "weights")["registered"] == 0
    assert I.input_count(doc, BY_ID.__getitem__, w, {"w": "cd" * 64})["registered"] == 0
    nested = {**doc, "registered": {**doc["registered"], "params": {"w/a@8": 6 * K // 2},
                                    "reads": [{**doc["registered"]["reads"][0], "param": "w/a@8"}]}}
    own, other = {"w/a@8": "ef" * 64}, {"w/a@8": "12" * 64}
    for pins in (own, {**w, **own}):
        assert I.input_count(nested, BY_ID.__getitem__, pins, {**w, **own})["registered"] == 12 * 8 * K
    for pins, roots in [(w, w), (w, {**w, **own}), (w, {"w/a@8": "ab" * 64}), ({"w/b@8": "ab" * 64}, {"w/b@8": "ab" * 64}),
                        (own, other), (own, w), ({**w, **own}, {**w, **other})]:
        assert I.input_count(nested, BY_ID.__getitem__, pins, roots)["registered"] == 0, (pins, roots)
    assert "registered" not in json.loads(I.circuit_bytes(window()[0].circuit_record()))
    for bad, why in [({**doc["registered"], "params": {"w": 5 * K // 2}}, "leaves"),
                     ({**doc["registered"], "reads": [{**doc["registered"]["reads"][0], "port": "frames"}]}, "names no parameter"),
                     ({**doc["registered"], "reads": doc["registered"]["reads"] * 2}, "both fill rows"),
                     ({**doc["registered"], "reads": [{**doc["registered"]["reads"][0], "x": 0}]}, "a read's record")]:
        with pytest.raises(ValueError, match=why):
            I.input_count({**doc, "registered": bad}, BY_ID.__getitem__)


def test_a_registered_values_shape_is_the_width_of_the_port_its_reads_name():
    """`read_shape`: ``w``'s rows are ``wrow``'s width, its leaves over that, and a value no read names, one read at two
    widths or one whose leaves aren't whole rows is refused."""
    win, _ = registered()
    wrow = dict(I.layout(COORD).rows)["wrow"]
    assert I.read_shape(win.circuit, "w") == {"value": "w", "words": wrow, "rows": 6 * K // 2 // wrow, "word_bits": 16,
                                              "row_schema": I.V2}
    with pytest.raises(ValueError, match="no registered read of 'v'"):
        I.read_shape(win.circuit, "v")
    other = next(p for p, n in I.layout(COORD).rows if n != wrow)
    two = I.Circuit("two", ((COORD, 12),), registered=I.Registered(
        {"w": 6 * K // 2}, reads=(READ, I.Read(TID, other, "w", 0, (12,), 0, (0,)))))
    with pytest.raises(ValueError, match="not one width"):
        I.read_shape(two, "w")
    odd = I.Circuit("odd", ((COORD, 12),), registered=I.Registered({"w": 6 * K // 2 + 1}, reads=(READ,)))
    with pytest.raises(ValueError, match="aren't whole rows"):
        I.read_shape(odd, "w")


def test_a_read_covers_whole_members_and_reads_whole_rows():
    """Members are cut where a read's units begin or end, as where a wire's do, so a read covers whole members: units 2..11
    read the weight's rows as tables and units 0 and 1 hold their own rows."""
    reg = lambda r: I.Registered({"w": 6 * K // 2}, reads=(r,))  # noqa: E731
    win = I.Window(I.Circuit("part", ((COORD, 12),), registered=reg(I.Read(TID, "wrow", "w", 2, (5, 2), 0, (K // 2, 0)))),
                   RUN, LAW, 4)
    assert [(m.base, m.n, bool(m.tables)) for m in win.window.members] == [(0, 2, False), (2, 4, True), (6, 4, True),
                                                                           (10, 2, True)]
    with pytest.raises(ValueError, match="whole registered rows"):
        I.Window(I.Circuit("half", ((COORD, 12),), registered=reg(I.Read(TID, "wrow", "w", 0, (12,), 0, (K // 8,)))),
                 RUN, LAW, 4)
    with pytest.raises(ValueError, match="uncut registered reads"):
        I.Window(I.Circuit("cut", ((COORD, 12),), cuts={TID: {"x": 2}},
                           registered=reg(I.Read(TID, "x", "w", 0, (12, 2), 0, (0, K // 4), cut=(0, 2)))), RUN, LAW, 4)


def test_the_window_commits_each_read_port_as_a_table_of_the_registered_rows_under_their_salts():
    """A member a read covers is a shared-row member: its read port's table the distinct rows its units read, under the
    registered value's salts, each unit's ref its row; its other ports its units' own rows. The header names the rows'
    positions (`write`), the verifier derives the same members from the record, and its ``--registered`` file names each
    unit's row from the record and the pin alone."""
    win, w = registered()
    c = win.close(salt_key=bytes(32))
    m1 = win.window.members[1]
    assert m1.tables == (("x", 4), ("sx", 4), ("wrow", 2), ("sw", 4))
    assert np.array_equal(c.rows[m1.port("wrow")], w.rows[[2, 3]]) and c.salts[m1.port("wrow")] == w.salts[2] + w.salts[3]
    assert c.refs[1][:, 2].tolist() == [0, 0, 1, 1] and c.refs[1][:, 0].tolist() == [0, 1, 2, 3]
    assert win._entries[m1.key] == {"wrow": w.statement([2, 3])[2]} and w.statement([2, 3])[2]["paths"][1] == w.path(3).hex()
    doc = json.loads(I.circuit_bytes(win.circuit_record()))
    mine = I.window_of_record(doc, LAW, BY_ID.__getitem__)
    assert mine.members == win.window.members
    pin = {"w": {"value": "w", "root": "cd" * 64, "leaves": 6, "words": K // 2, "domain": "ef" * 64}}
    files = I.registered_reads(doc, mine, pin, registered=REG)
    assert sorted(files) == [m.key for m in mine.members]
    assert files[m1.key] == {"format": REG.READS_V1, "ports": {"wrow": {
        "value": "w", "root": "cd" * 64, "leaves": 6, "words": K // 2, "word_bits": 16, "domain": "ef" * 64,
        "row_schema": V2, "reads": [2, 2, 3, 3], "reads_from": 4}}}
    with pytest.raises(ValueError, match="didn't pin"):
        I.registered_reads(doc, mine, {}, registered=REG)
    with pytest.raises(ValueError, match="not the port's"):
        I.registered_reads(doc, mine, {"w": {**pin["w"], "words": 2}}, registered=REG)
    unread = I.Window(win.circuit, RUN, LAW, 4)
    unread._recorded = win._recorded
    with pytest.raises(ValueError, match="give its committed value"):
        unread.close()
    twice, _ = registered()
    twice._recorded[TID][0].words[2][1, 0] ^= 1
    with pytest.raises(ValueError, match="two rows for one row"):
        twice.close()


def test_a_wired_member_reading_a_registered_row_is_checked_through_its_refs():
    """`chain` with each layer's ``w`` a registered row: every member is a shared-row member, its wired ``x`` a table of its
    units' own rows, and the wiring check reads a unit's string through its ref, the row its statement reads. The verifier
    derives the same members, a wire moved to other writers is refused, and so is a ref moved to another unit's row."""
    ins, outs = chain_values()
    layer = REQUESTS * STEPS
    w = Held("w", ins[1][::layer].reshape(LAYERS, 256))
    read = I.Read(GID, "w", "w", at=0, radix=(LAYERS, layer), base=0, step=(256, 0))
    circuit = replace(chain(), registered=I.Registered({"w": LAYERS * 256}, reads=(read,)))
    win = I.Window(circuit, RUN, LAW, 2, registered={"w": w})
    ms = win.window.members
    assert [(m.base, m.tables, m.wired) for m in ms] == [
        (0, (("x", 2), ("w", 1)), ()), (2, (("x", 2), ("w", 1)), (("x", "o0", (4, 5)),)),
        (4, (("x", 2), ("w", 1)), (("x", "o0", (0, 1)),)), (6, (("x", 2), ("w", 1)), (("x", "o0", (2, 3)),))]
    win.record(GEMM, ins, outs)
    c = win.close(salt_key=bytes(32))
    publics = {m.key: c.public_body(k) for k, m in enumerate(ms)}
    assert WI.check(ms, publics) == {} and WI.check_drawn(ms, publics, range(8)) == {}
    doc = json.loads(I.circuit_bytes(win.circuit_record()))
    assert I.window_of_record(doc, LAW, {GID: GEMM}.__getitem__).members == ms
    moved = [replace(m, wired=(("x", "o0", m.wired[0][2][::-1]),)) if m.wired else m for m in ms]
    assert set(WI.check(moved, publics)) == {m.key for m in ms if m.wired}
    body, at = bytearray(publics[ms[1].key]), WI.BC_BYTES * sum(CS.port_leaves(ms[1]).values())
    body[at:at + 4] = (1).to_bytes(4, "little")
    assert list(WI.check_drawn(ms, {**publics, ms[1].key: bytes(body)}, [2])) == [ms[1].key]
    with pytest.raises(ValueError, match="own rows"):
        CS.Window(win.window.program, win.window.query, LAW, RUN,
                  (*ms[:1], replace(ms[1], tables=(("x", 1), ("w", 1))), *ms[2:]))


class Bound(Held):
    """`Held` with what core's ``Registered`` carries for a commit by reference (`rows.ValueRows`): 16-bit leaves, its row
    schema and each row's ``b || c`` under its salt."""

    def __init__(self, name: str, rows: np.ndarray) -> None:
        super().__init__(name, rows)
        self.word_bits, self.schema, inner = 16, V2, CS._inner(V2, self.words)
        self.bcs = [CS.INST.commit_string(CS.INST.default_key, inner(np.ascontiguousarray(r, dtype="<u2").tobytes()), s)
                    for r, s in zip(rows, self.salts)]


def _same_commit(full: I.Window, refd: I.Window, tmp_path) -> None:
    """``refd`` (some inputs recorded by reference) commits what ``full`` does: every root, string, salt and row, the files
    `write` writes byte for byte, and each member's prover body as the files make it (`rows.read_private`), where ``refd``'s
    leave a registered value's rows and salts out of the member's own file (``value_rows``); both leave a wired port's
    (``wired_rows``)."""
    a, b = full.close(salt_key=bytes(32)), refd.close(salt_key=bytes(32))
    assert list(a.roots.items()) == list(b.roots.items()) and a.bc == b.bc and a.salts == b.salts
    assert list(a.rows) == list(b.rows) and all(np.array_equal(a.rows[p], b.rows[p]) for p in a.rows)
    def record(_program, *x, **k):
        return {"x": json.dumps(x, sort_keys=True), **k}

    reg = type("R", (), {"KINDS": ("served",), "FORMAT": "f", "record": staticmethod(record), "digest": staticmethod(lambda rec: "d")})
    files = []
    for win, c, d in ((full, a, tmp_path / "full"), (refd, b, tmp_path / "refd")):
        win.write(c, d, {"by": "test"}, registrar=reg)
        doc = json.loads((d / "window.json").read_text())
        got = {f.name: f.read_bytes() for f in d.iterdir() if not re.fullmatch(r"[mv][0-9]+-private\.bin", f.name)}
        got["window.json"] = {k: v for k, v in doc.items() if k not in ("seconds", "value_rows", "members")}
        got["members"] = [{f: v for f, v in m.items() if f != "value_rows"} for m in doc["members"]]
        got["bodies"] = [CS.read_private(d, doc, k) for k in range(len(doc["members"]))]
        assert got["bodies"] == [c.private_body(k) for k in range(len(doc["members"]))]
        files.append(got)
    assert files[0] == files[1]


SRC = bind(G.Gemm, K=32, N=16, **G.GEMM_AMPERE)
SID = descriptor_id(SRC)


def fed_chain() -> tuple[I.Circuit, Bound, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """`chain` fed by a ``Gemm_v2{K=32, N=16}`` unit a request (layer 0's ``x`` at step 0 its output), each layer's ``w`` a
    registered row: the circuit, the registered value, the feed's inputs (x, w) and outputs, and the chain's."""
    rng = np.random.default_rng(3)
    sx = (rng.standard_normal((REQUESTS, 32)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    sw = ((rng.standard_normal((REQUESTS, 16, 32)) / 4).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    sy = np.asarray(evaluate_batch(SRC, sx, sw, kernel="reference"), dtype=np.uint16).reshape(REQUESTS, 16)
    ins, outs = chain_values(first=sy)
    layer = REQUESTS * STEPS
    w = Bound("w", ins[1][::layer].reshape(LAYERS, 256))
    read = I.Read(GID, "w", "w", at=0, radix=(LAYERS, layer), base=0, step=(256, 0))
    circuit = replace(chain(), templates=((SRC, REQUESTS), (GEMM, LAYERS * layer)), external=(f"{SID}/x", f"{SID}/w"),
                      wires=(I.Wire(GID, "x", SID, "o0", at=0, radix=(REQUESTS,), base=0, step=(1,)), *chain().wires),
                      registered=I.Registered({"w": LAYERS * 256}, reads=(read,)))
    return circuit, w, [sx, sw], [sy], ins, outs


def _loop_turns(f) -> int:
    """The jumps Python makes in `verity.ml.windows`' modules while ``f()`` runs (`sys.monitoring`'s JUMP events: every
    turn of a loop, a comprehension's included, is one): the work a window does in Python, counted, not timed."""
    mon, here, ours = sys.monitoring, str(Path(I.__file__).parent), {}
    tool, turns = next(t for t in range(6) if mon.get_tool(t) is None), 0

    def jump(code, _at, _to):
        nonlocal turns
        if (mine := ours.get(code)) is None:
            mine = ours[code] = str(Path(code.co_filename).parent) == here
        if not mine:
            return mon.DISABLE
        turns += 1

    mon.use_tool_id(tool, "loop turns")
    try:
        mon.register_callback(tool, mon.events.JUMP, jump)
        mon.set_events(tool, mon.events.JUMP)
        f()
    finally:
        mon.set_events(tool, mon.events.NO_EVENTS)
        mon.register_callback(tool, mon.events.JUMP, None)
        mon.free_tool_id(tool)
        mon.restart_events()
    return turns


def _long_chain(layers: int, d: Path) -> None:
    """A ``Gemm_v2{K=16, N=16}`` chain of ``layers`` layers of two units, a member a unit (cap 1), layer 0's ``x`` a
    ``Gemm_v2{K=32, N=16}`` feed's output and layer l's layer l - 1's, a wire each, and each layer's ``w`` its own row of
    a registered value, a read each, both recorded by reference: the window built, recorded, closed and written, and the
    verifier's window, wiring check and ``--registered`` files made from its record."""
    rng = np.random.default_rng(5)

    def bf16(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    sx, sw, w = bf16(rng.standard_normal((2, 32))), bf16(rng.standard_normal((2, 16, 32)) / 4), bf16(
        rng.standard_normal((layers, 16, 16)) / 4)
    ys = [np.asarray(evaluate_batch(SRC, sx, sw, kernel="reference"), dtype=np.uint16).reshape(2, 16)]
    for j in range(layers):
        ys.append(np.asarray(evaluate_batch(GEMM, ys[-1], np.repeat(w[j][None], 2, axis=0), kernel="reference"),
                             dtype=np.uint16).reshape(2, 16))
    wires = [I.Wire(GID, "x", SID if j == 0 else GID, "o0", at=2 * j, radix=(2,), base=2 * max(j - 1, 0), step=(1,))
             for j in range(layers)]
    reads = tuple(I.Read(GID, "w", "w", at=2 * j, radix=(2,), base=256 * j, step=(0,)) for j in range(layers))
    circuit = I.Circuit("long", ((SRC, 2), (GEMM, 2 * layers)), external=(f"{SID}/x", f"{SID}/w"), wires=tuple(wires),
                        registered=I.Registered({"w": 256 * layers}, reads=reads))
    win = I.Window(circuit, RUN, LAW, 1, registered={"w": Bound("w", w.reshape(layers, 256))})
    win.record(SRC, [sx, sw], ys[:1])
    win.record(GEMM, [None, None], [np.concatenate(ys[1:])])
    c = win.close(salt_key=bytes(32))
    reg = type("R", (), {"KINDS": ("served",), "FORMAT": "f", "record": staticmethod(lambda *x, **k: {"leaf_layer": "l"}),
                         "digest": staticmethod(lambda rec: "d")})
    win.write(c, d, {"by": "test"}, registrar=reg)
    doc = json.loads((d / "circuit.json").read_bytes())
    mine = I.window_of_record(doc, LAW, {SID: SRC, GID: GEMM}.__getitem__)
    assert WI.check(mine.members, {m.key: c.public_body(k) for k, m in enumerate(mine.members)}) == {}
    pin = {"w": {"value": "w", "root": "cd" * 64, "leaves": layers, "words": 256, "domain": "ef" * 64}}
    assert len(I.registered_reads(doc, mine, pin, registered=REG)) == 2 * layers


def test_a_windows_commit_is_linear_in_its_members(tmp_path):
    """A window's commit and the verifier's rebuild of it from its record (`_long_chain`) run loops whose turns grow with
    its members, wires and reads, not with their products: from 32 layers to 64 the turns grow at most about twice what
    they grow from 16 to 32 (the difference drops what a window does once, such as its hiding tables). The 4,32,4 window
    class's commit (58,719 members, 166,038 wired member ports, 6,727 reads) ran each port's hashes past every port's, each
    member past every read and each wired port past every member, hours on one core; a timing can't say which, a count
    can."""
    a, b, c = (_loop_turns(lambda n=n: _long_chain(n, tmp_path / str(n))) for n in (16, 32, 64))
    assert c - b <= 2.2 * (b - a), (a, b, c)


def test_wired_and_registered_rows_recorded_by_reference_commit_as_their_rows_given_in_full(tmp_path):
    """`fed_chain`'s units recorded with no inputs at all: a wired ``x`` row is its writer's output row and a ``w`` the
    value's, so the commitment, the rows and every file are the full record's, and the mismatch check gathers the inputs it
    evaluates on (`Window.input_words`)."""
    circuit, w, (sx, sw), feed, ins, outs = fed_chain()
    full, refd = (I.Window(circuit, RUN, LAW, 2, registered={"w": w}) for _ in range(2))
    for win, given in ((full, ins), (refd, [None, None])):
        win.record(SRC, [sx, sw], feed)
        win.record(GEMM, given, outs)
    assert all(np.array_equal(a, b.reshape(8, -1)) for a, b in zip(refd.input_words(GID, 0, 8), ins))
    assert refd.mismatches() == full.mismatches() == {}
    _same_commit(full, refd, tmp_path)
    with pytest.raises(ValueError, match="no wire or registered read fills it"):
        free = I.Window(circuit, RUN, LAW, 2, registered={"w": w})
        free.record(SRC, [None, sw], feed)
        free.record(GEMM, ins, outs)
        free.close(salt_key=bytes(32))


class _Counted(list):
    """A value's salts, each read counted."""

    reads = 0

    def __getitem__(self, i):
        self.reads += 1
        return super().__getitem__(i)


def test_a_windows_registered_rows_are_each_drawn_once(monkeypatch, tmp_path):
    """`fed_chain`'s chain members, two a layer, each read their layer's one row of ``w``: committing and writing the window
    reads each row's salt once, however many members read it, and asks for no statement. Core's ``Registered.statement``
    draws a member's salts again (a pure-Python ChaCha20 call each) and makes its rows' ints, and the rows' commit drew
    them a third time: 10,565,136 draws at the 4,32,4 class, whose members read 37,788 distinct rows."""
    circuit, w, feed_ins, feed, _ins, outs = fed_chain()
    w.salts = _Counted(w.salts)
    monkeypatch.setattr(Bound, "statement", lambda *_: pytest.fail("a statement of the reads drew their salts again"))
    win = I.Window(circuit, RUN, LAW, 2, registered={"w": w})
    win.record(SRC, feed_ins, feed)
    win.record(GEMM, [None, None], outs)
    c = win.close(salt_key=bytes(32))
    reads = [e["w"]["positions"] for e in win._entries.values()]
    assert w.salts.reads == LAYERS and len(reads) == 2 * LAYERS and sorted(reads) == [[0], [0], [1], [1]]
    reg = type("R", (), {"KINDS": ("served",), "FORMAT": "f", "record": staticmethod(lambda *x, **k: {"leaf_layer": "l"}),
                         "digest": staticmethod(lambda rec: "d")})
    win.write(c, tmp_path, {"by": "test"}, registrar=reg)
    assert w.salts.reads == LAYERS


def test_a_windows_value_rows_and_paths_are_written_once_and_a_drawn_members_body_is_made_from_them(tmp_path):
    """`fed_chain` recorded by reference: each chain member's ``m<k>-private.bin`` leaves out its ``w`` rows and salts and
    its wired ``x``'s, ``v0-private.bin`` holds each row of ``w`` the window reads once, then its salt, and ``window.json``
    each row's path once; each member's prover body as the files make it (`rows.read_private`: a wired row and its salt off
    its writer's own file; `rows.Bodies`, with as few as one file mapped at a time) and its header (`rows.registered_header`) are the committed ones, and so is the circuit prover's
    file staged from them (`flock_inputs.flock_input_files`). The 4,32,4 class's bodies held 3,200,960 rows of 32,768
    words, 22,864 of them distinct, 9,657,548 wired rows of 3,750,680 output rows, and its ``window.json`` 5,282,638 paths,
    37,788 of them distinct. A file short of its ports, a row its value's file doesn't hold, a row with no path, a row
    given two paths, a wire to a writer that is no member or not of the port's words and a row of a value committed under
    two salts are refused."""
    circuit, w, feed_ins, feed, _ins, outs = fed_chain()
    win = I.Window(circuit, RUN, LAW, 2, registered={"w": w})
    win.record(SRC, feed_ins, feed)
    win.record(GEMM, [None, None], outs)
    c = win.close(salt_key=bytes(32))
    reg = type("R", (), {"KINDS": ("served",), "FORMAT": "f", "digest": staticmethod(lambda rec: "d"),
                         "record": staticmethod(lambda *x, **k: {"leaf_layer": "l", "window": x[5]})})
    win.write(c, tmp_path, {"by": "test"}, registrar=reg)
    doc, ms = json.loads((tmp_path / "window.json").read_text()), win.window.members
    chained = [k for k, m in enumerate(ms) if m.template == GID]
    assert len(chained) == 2 * LAYERS and [doc["members"][k].get("value_rows") for k in range(len(ms))] == [
        ["w"] if k in chained else None for k in range(len(ms))]
    assert [doc["members"][k].get("wired_rows") for k in range(len(ms))] == [["x"] if m.wired else None for m in ms]
    assert all(m.wired for k, m in enumerate(ms) if k in chained)
    assert doc["value_rows"] == {"w": {"words": 256, "positions": [0, 1], "file": "v0-private.bin"}}
    assert (tmp_path / "v0-private.bin").read_bytes() == w.rows[[0, 1]].astype("<u2").tobytes() + w.salts[0] + w.salts[1]
    assert doc["registered_paths"] == {"w": {"0": w.path(0).hex(), "1": w.path(1).hex()}}
    files = CS.Bodies(tmp_path, doc)
    for k, m in enumerate(ms):
        body, leaves = c.private_body(k), CS.port_leaves(m)
        left = {"w", "x"} if k in chained else {p for p, _o, _ws in m.wired}
        assert CS.read_private(tmp_path, doc, k) == body
        out = io.BytesIO()
        files.copy(k, out)
        assert out.getvalue() == body
        assert (tmp_path / f"m{k}-private.bin").stat().st_size == len(body) - sum(
            leaves[p] * (2 * n + CS.SALT_BYTES) for p, n in m.all_rows if p in left)
        assert all("paths" not in e for e in doc["members"][k].get("registered", {}).values())
        assert CS.registered_header(doc, k) == win._entries.get(m.key)
    files = CS.Bodies(tmp_path, doc)
    files.OPEN = 1
    for k in range(len(ms)):
        out = io.BytesIO()
        files.copy(k, out)
        assert out.getvalue() == c.private_body(k) and len(files._own) + len(files._values) <= 1
    k, mem = chained[1], doc["members"][chained[1]]
    meta = {"scheme": CS.SCHEME, "ports": [[p, n, 1] for p, n in mem["ports"]], "input_ports": mem["input_ports"],
            "out_ports": mem["out_ports"], "row_schemas": mem["row_schemas"], "vus_per_block": 1, "units_per_vu": 1,
            "subcircuit": "gemm"}
    got = FI.flock_input_files(tmp_path, k, "flock-circuit v2\nMETA " + json.dumps(meta) + "\n", tmp_path / "staged")
    head, body = Path(got["inst"]).read_bytes().split(b"\n", 1)
    assert json.loads(head)["registered"] == win._entries[ms[k].key] and body == c.private_body(k) + c.public_body(k)
    assert json.loads(Path(got["pub"]).read_bytes().split(b"\n", 1)[0])["registered"] == win._entries[ms[k].key]
    row = mem["registered"]["w"]["positions"][0]
    feed = _first(win.window, SID)
    for edit, read, why in [
            (lambda d: d["value_rows"]["w"].update(positions=[2, 3]), CS.read_private, "rows of 256 words the window's"),
            (lambda d: d["registered_paths"]["w"].pop(str(row)), CS.registered_header, "no path for rows"),
            (lambda d: d["members"][k]["registered"]["w"].update(positions=[0, 1]), CS.read_private, "rows of 256 words"),
            (lambda d: d["members"][k]["wired"][0].update(writers=[10 ** 6] * mem["n"]), CS.read_private, "not output rows"),
            (lambda d: d["members"][k]["wired"][0].update(writers=[ms[feed].base] * mem["n"], out="o1"), CS.read_private,
             "not output rows"),
            (lambda d: d["members"][k].pop("wired"), CS.read_private, "no wire gives")]:
        other = copy.deepcopy(doc)
        edit(other)
        with pytest.raises(ValueError, match=why):
            read(tmp_path, other, k) if read is CS.read_private else read(other, k)
    own = tmp_path / f"m{k}-private.bin"
    own.write_bytes(own.read_bytes()[:-1])
    with pytest.raises(ValueError, match="bytes, not the"):
        CS.read_private(tmp_path, doc, k)
    other = next(j for j in chained if j != k and win._entries[ms[j].key]["w"]["positions"] == [row])
    salts = dict(c.salts)
    c.salts[ms[other].port("w")] = bytes(CS.SALT_BYTES)
    with pytest.raises(ValueError, match="committed under two salts"):
        win.write(c, tmp_path / "salted", {"by": "test"}, registrar=reg)
    c.salts = salts
    win._entries[ms[k].key] = {"w": {**win._entries[ms[k].key]["w"], "paths": ["00" * 64]}}
    with pytest.raises(ValueError, match="given two paths"):
        win.write(c, tmp_path / "again", {"by": "test"}, registrar=reg)


def test_a_cut_ports_rows_recorded_by_reference_are_each_its_writers_output_row(tmp_path):
    """`sliced`'s position units recorded with ``x`` by reference and ``w`` in full: each cut row ``x#j`` is its slice's
    output row, committed as the full record commits it; one template's parameter is by reference for all its units or
    none."""
    c = sliced()
    ins8, outs8, ins, outs = sliced_values()
    full, refd = (I.Window(c, RUN, LAW, 2) for _ in range(2))
    for win, x in ((full, ins[0]), (refd, None)):
        win.record(G8, ins8, outs8)
        win.record(GEMM, [x, ins[1]], outs)
    assert np.array_equal(refd.input_words(GID, 1, 2)[0], ins[0][1:])
    assert refd.mismatches() == {}
    _same_commit(full, refd, tmp_path)
    mixed = I.Window(c, RUN, LAW, 2)
    mixed.record(G8, ins8, outs8)
    mixed.record(GEMM, [None, ins[1][:1]], [outs[0][:1]])
    mixed.record(GEMM, [ins[0][1:], ins[1][1:]], [outs[0][1:]])
    with pytest.raises(ValueError, match="by reference for some units"):
        mixed.close(salt_key=bytes(32))


def given_by_reference(monkeypatch) -> tuple[CS.Window, list[dict], Bound]:
    """`fed_chain` with every chain input recorded by reference: the window and each member's rows as `Window.close` hands
    them to `rows.commit` (each wired ``x`` `rows.WIRED`, each ``w`` a `rows.ValueRows` of the registered value)."""
    circuit, w, feed_ins, feed, _ins, outs = fed_chain()
    win = I.Window(circuit, RUN, LAW, 2, registered={"w": w})
    win.record(SRC, feed_ins, feed)
    win.record(GEMM, [None, None], outs)
    given = []
    with monkeypatch.context() as mp:
        mp.setattr(CS, "commit", lambda window, data, **_: given.append((window, data)))
        win.close(salt_key=bytes(32))
    ((window, data),) = given
    return window, data, w


def _first(window: CS.Window, tid: str) -> int:
    return next(k for k, m in enumerate(window.members) if m.template == tid)


def _like(value: Bound, **fields) -> Bound:
    out = copy.copy(value)
    vars(out).update(fields)
    return out


def _value_rows(value=None, positions=None, salts=None):
    """An edit giving the first chain member's ``w`` as ``positions`` of ``value(registered value)`` under ``salts(the salts
    `Window.close` drew)``, each as given when None (the salts drawn again by `rows.commit` when the positions change)."""
    def edit(window: CS.Window, data: list[dict]) -> CS.Window:
        k = _first(window, GID)
        src = data[k]["rows"]["w"]
        data[k]["rows"]["w"] = CS.ValueRows(src.value if value is None else value(src.value),
                                            src.positions if positions is None else positions,
                                            (src.salts if positions is None else None) if salts is None else salts(src.salts))
        return window
    return edit


def _feed_x_wired(window: CS.Window, data: list[dict]) -> CS.Window:
    """The feed's external ``x``, which no wire writes, given as `rows.WIRED`."""
    data[_first(window, SID)]["rows"]["x"] = CS.WIRED
    return window


def _beside_a_row_table(window: CS.Window, data: list[dict]) -> CS.Window:
    """The feed's ``x`` made the one row of a row table both its units read, the chain's rows still by reference."""
    k = _first(window, SID)
    m = window.members[k]
    data[k]["rows"][CS.LINKED] = data[k]["rows"][CS.LINKED][:1]
    data[k]["refs"] = np.stack([np.zeros(m.n, dtype=np.int64) if p == CS.LINKED else np.arange(m.n) for p, _ in m.rows], axis=1)
    linked = replace(m, tables=tuple((p, 1 if p == CS.LINKED else m.n) for p, _ in m.rows), links=((0, 0),),
                     link_units=(m.n,))
    table = CS.RowTable(0, "feed", m.template, dict(m.rows)[CS.LINKED], ((0,),))
    return replace(window, members=(*window.members[:k], linked, *window.members[k + 1:]), row_tables=(table,),
                   context="cd" * RT.DIGEST_BYTES)


REFUSED = {
    "a value of another row width": (_value_rows(value=lambda v: Bound("w", v.rows.reshape(-1, 128))),
                                     "rows of 128 words, not 256"),
    "a value of another schema": (_value_rows(value=lambda v: _like(v, schema=CS.TABLE_SCHEMA)),
                                  f"{CS.TABLE_SCHEMA} rows, not {V2}"),
    "a value of 8-bit leaves": (_value_rows(value=lambda v: _like(v, word_bits=8)), "8-bit leaves, not 16"),
    "a position past the value's rows": (_value_rows(positions=(LAYERS,)), f"positions outside its {LAYERS} rows"),
    "a negative position": (_value_rows(positions=(-1,)), f"positions outside its {LAYERS} rows"),
    "two rows for the port's one": (_value_rows(positions=(0, 1)), "2 rows, not 1"),
    "salts short of its rows": (_value_rows(salts=lambda s: s[:-1]), f"salts not one {CS.SALT_BYTES}-byte salt a row"),
    "wired rows on a port no wire writes": (_feed_x_wired, f"given as {CS.WIRED} rows, but no wire writes it"),
    "rows by reference in a window with row tables": (_beside_a_row_table, "in a window with row tables"),
}


@pytest.mark.parametrize("edit, why", REFUSED.values(), ids=REFUSED)
def test_rows_given_by_reference_are_refused_unless_their_source_is_the_ports(monkeypatch, edit, why):
    """`rows.commit` refuses rows given by reference, naming why: a registered value's rows of another row width, schema or
    leaf width, at a position outside the value (a negative one included, which indexing would wrap) or of another count
    than the port's rows, or salts given that aren't one a row; `rows.WIRED` rows on a port no wire writes; and a port by
    reference in a window with row tables,
    whose linked members' rows are read off their ``x`` tables. The rows as `Window.close` gives them commit; each one edit
    is refused."""
    window, data, _w = given_by_reference(monkeypatch)
    CS.commit(window, data, salt_key=bytes(32))
    with pytest.raises(ValueError, match=re.escape(why)):
        CS.commit(edit(window, data), data, salt_key=bytes(32))


class Table(Held):
    """`EMB`'s table, one token a row under ``hm96-sha512/row-seg/v1`` (`gates.embedding`), as core registers it: its rows,
    salts, tree and entry."""

    def __init__(self) -> None:
        rng = np.random.default_rng(7)
        super().__init__("embed", rng.integers(0, 1 << 16, (4, 16), dtype=np.uint16))
        self.salts = [bytes(rng.integers(0, 256, E.SALT_BYTES, dtype=np.uint8)) for _ in range(4)]
        self.dom = CommitmentDomain(hashlib.sha512(b"embed").digest(), -2, RangeIndexedDomain(0, 4), hash="sha512")
        self.tree = RT.commit(self.dom, [bit_row((r.tolist(), 16)) for r in self.rows], self.salts)

    def entry(self):
        return {"value": "embed", "root": self.tree.root.hex(), "leaves": 4, "words": 16, "domain": self.dom.domain_id.hex()}

    def path(self, tok: int) -> list[bytes]:
        p = self.tree.path(tok)
        return [p[k:k + 64] for k in range(0, len(p), 64)]


def embedding(**edit) -> I.Circuit:
    """Three embedding reads of `Table`, their tokens free."""
    reg = I.Registered({"embed": 64}, (I.Table("embed", "embed", 0, 16, 4),), (),
                       (I.TableRead(EID, "root", "embed", "root", 0, 3), I.TableRead(EID, "domain", "embed", "domain", 0, 3)),
                       (I.Opening(EID, "embed", "tok", ("row", "salt", "path"), "o1", 0, 3),))
    return I.Circuit("embedding", ((EMB, 3),), **{"registered": reg, **edit})


def embedding_window() -> tuple[I.Window, Table]:
    """`embedding`'s window in members of at most 2 units, its units' values `gates.embedding`'s reference on tokens
    (3, 3, 1)."""
    table = Table()
    win = I.Window(embedding(), RUN, LAW, 2, registered={"embed": table})
    toks = [3, 3, 1]
    got = [E.reference(4, t, table.tree.root, table.dom.domain_id, table.rows[t].tolist(), table.salts[t], table.path(t))
           for t in toks]
    assert [ok for _, ok in got] == [1, 1, 1]

    def digest(d: bytes) -> np.ndarray:
        return np.frombuffer(d, ">u8").astype(np.uint64)

    ins = [np.array(toks, dtype=np.uint32), np.stack([digest(table.tree.root)] * 3), np.stack([digest(table.dom.domain_id)] * 3),
           table.rows[toks], np.stack([np.frombuffer(table.salts[t], ">u8").astype(np.uint64) for t in toks]),
           np.stack([np.stack([digest(s) for s in table.path(t)]) for t in toks])]
    win.record(EMB, ins, [np.array([h for h, _ in got], dtype=np.uint16), np.array([ok for _, ok in got], dtype=np.uint16)])
    return win, table


def test_a_tables_reads_and_verdicts_take_its_supplied_values():
    """A table as registered reads, no window built: a table read takes the table's supplied root or domain id and an
    opening's verdict its supplied 1, each one row for all its units, made from the entry of the table's parameter alone
    under public salts. A table that isn't its parameter's every row, and an ingress, are refused by the committer."""
    c, table = embedding(), Table()
    assert [(r.port, r.param, r.at, r.reader_units, r.base, r.step) for r in I.committed_reads(c.registered)] == [
        ("root", "embed/root", 0, 3, 0, (0,)), ("domain", "embed/domain", 0, 3, 0, (0,)), ("o1", "embed/opened", 0, 3, 0, (0,))]
    got = I.supplied(c.registered, {"embed": table.entry()}, "p", module=REG)
    assert {v: (x.rows.tolist(), x.salts, x.program) for v, x in got.items()} == {
        "embed/root": ([I.digest_words(table.tree.root)], REG.supplied_salts("p", "embed/root", 1), "p"),
        "embed/domain": ([I.digest_words(table.dom.domain_id)], REG.supplied_salts("p", "embed/domain", 1), "p"),
        "embed/opened": ([[1]], REG.supplied_salts("p", "embed/opened", 1), "p")}
    assert I.supplied(c.registered, {}, "p", module=REG) == {}
    words = I.digest_words(bytes(range(64)))
    assert len(words) == 32 and words[:4] == [0x0607, 0x0405, 0x0203, 0x0001]
    with pytest.raises(ValueError, match="a registered parameter of its own"):
        I.Window(embedding(registered=replace(c.registered, params={"embed": 80})), RUN, LAW, 3)
    with pytest.raises(ValueError, match="ingress are recorded by their description"):
        I.Window(embedding(ingress=(I.Ingress(EID, "tok", 0, 2, 0),)), RUN, LAW, 3)


@pytest.mark.slow
def test_a_window_commits_its_table_reads_and_verdicts_as_registered_reads():
    """A table's registered reads committed: each table read and verdict a registered read of its supplied value, the opening's row,
    salt and path the units' own; and the verifier derives the same members, and each member's ``--registered`` file, from
    the record and its pin of the table's parameter alone, which no cheaper test reaches. Slow (about 140 s on one core)
    for the two population Programs of `EmbeddingRead_v1` it builds, the prover's and the verifier's, its Merkle path
    traced in each; the cheap check of the reads themselves is `test_a_tables_reads_and_verdicts_take_its_supplied_values`."""
    win, table = embedding_window()
    with pytest.raises(ValueError, match="Window.supply"):
        win.close(salt_key=bytes(32))
    win.supply(REG)
    c = win.close(salt_key=bytes(32))
    m0, m1 = win.window.members
    assert [(m.base, m.n, m.tables) for m in (m0, m1)] == [
        (0, 2, (("tok", 2), ("root", 1), ("domain", 1), ("row", 2), ("salt", 2), ("path", 2))),
        (2, 1, (("tok", 1), ("root", 1), ("domain", 1), ("row", 1), ("salt", 1), ("path", 1)))]
    root, opened = win.registered["embed/root"], win.registered["embed/opened"]
    assert root.rows.tolist() == [I.digest_words(table.tree.root)] and opened.rows.tolist() == [[1]]
    assert np.array_equal(c.rows[m0.port("root")], root.rows) and c.salts[m0.port("root")] == root.salts[0]
    assert c.salts[m0.port("o1")] == opened.salts[0] * 2 and c.salts[m1.port("o1")] == opened.salts[0]
    assert table.salts[3] not in c.salts[m0.port("row")]
    assert win._entries[m0.key]["o1"]["positions"] == [0, 0] and sorted(win._entries[m1.key]) == ["domain", "o1", "root"]

    doc = json.loads(I.circuit_bytes(win.circuit_record()))
    mine = I.window_of_record(doc, LAW, {EID: EMB}.__getitem__)
    assert mine.members == win.window.members
    files = I.registered_reads(doc, mine, {"embed": table.entry()}, registered=REG)
    ports = files[m0.key]["ports"]
    assert sorted(ports) == sorted(files[m1.key]["ports"]) == ["domain", "o1", "root"]
    assert {p: (e["value"], e["root"], e["reads"]) for p, e in ports.items()} == {
        "root": ("embed/root", root.entry()["root"], [0, 0]),
        "domain": ("embed/domain", win.registered["embed/domain"].entry()["root"], [0, 0]),
        "o1": ("embed/opened", opened.entry()["root"], [0, 0])}
    with pytest.raises(ValueError, match="didn't pin"):
        I.registered_reads(doc, mine, {}, registered=REG)
    win.registered = {**win.registered, "embed/opened": Held("embed/opened", np.ones((1, 3), dtype=np.uint16))}
    with pytest.raises(ValueError, match="rows of 3 words"):
        win.close(salt_key=bytes(32))


def test_the_boundary_is_carried_as_given():
    b = {"format": "verity/boundary/v1", "egress": {"template": GID}}
    win = I.Window(I.Circuit("b", ((GEMM, 2),), boundary=b), RUN, LAW, 2)
    assert json.loads(I.circuit_bytes(win.circuit_record()))["boundary"] == b
    assert "boundary" not in I.Window(I.Circuit("b", ((GEMM, 2),)), RUN, LAW, 2).circuit_record()
    with pytest.raises(ValueError, match="JSON object"):
        I.Circuit("b", ((GEMM, 2),), boundary=["egress"])


def test_an_advice_binding_is_well_formed():
    c = {"name": "compiler", "digest": "cd" * 64}
    assert I.advice_binding(c, "ef" * 64, 12) == {"compiler": c, "advice": {"root": "ef" * 64, "bits": 12}}
    at = {**c, "commit": "0a" * 20}
    assert I.advice_binding(at, "ef" * 64, 12)["compiler"] == at
    for bad in [({"name": "c"}, "ef" * 64, 1), (c, "ef" * 32, 1), (c, "ef" * 64, -1), (c, "ef" * 64, True),
                ({**c, "commit": "0a" * 19}, "ef" * 64, 1), ({**c, "commit": "main"}, "ef" * 64, 1), ({**c, "other": 1}, "ef" * 64, 1)]:
        with pytest.raises(ValueError):
            I.advice_binding(*bad)


def test_a_compilers_digest_is_its_name_and_the_code_git_lists_under_its_paths(tmp_path):
    repo, plain = tmp_path / "repo", tmp_path / "plain"

    def git(*args: str) -> str:
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()

    (repo / "pkg").mkdir(parents=True)
    git("init", "-q")
    (repo / "pkg" / "c.py").write_text("x = 1\n")
    (repo / "other.py").write_text("y = 1\n")
    (repo / ".gitignore").write_text("build/\n")
    git("add", "-A")
    git("commit", "-q", "-m", "a")
    d = I.compiler_digest("c", repo, ["pkg"])
    assert I.code_files(repo, ["pkg"]) == {"pkg/c.py": hashlib.sha512(b"x = 1\n").hexdigest()}
    assert I.commit_of(repo) == git("rev-parse", "HEAD") and I.compiler_digest("other", repo, ["pkg"]) != d
    (repo / "other.py").write_text("y = 2\n")
    (repo / "pkg" / "README.md").write_text("docs\n")
    (repo / "pkg" / "build").mkdir()
    (repo / "pkg" / "build" / "out.bin").write_bytes(b"\0")
    assert I.compiler_digest("c", repo, ["pkg"]) == d
    (repo / "pkg" / "new.py").write_text("z = 1\n")
    assert I.compiler_digest("c", repo, ["pkg"]) != d
    (repo / "pkg" / "new.py").unlink()
    (repo / "pkg" / "c.py").write_text("x = 2\n")
    assert I.compiler_digest("c", repo, ["pkg"]) != d
    with pytest.raises(ValueError, match="no file"):
        I.code_files(repo, ["nothing"])
    plain.mkdir()
    (plain / "c.py").write_text("x = 1\n")
    for f in (lambda: I.code_files(plain, ["."]), lambda: I.commit_of(plain)):
        with pytest.raises(ValueError, match="not a git work tree"):
            f()


def test_the_catalog_resolves_a_descriptor_id_to_verity_s_definition():
    assert descriptor_id(I.resolve(TID)) == TID and descriptor_id(I.resolve(AID)) == AID
    assert [descriptor_id(fn) for fn, _ in I.Circuit("r", ((I.resolve(TID), 3),)).templates] == [TID]
    own = CompositeDefinition("IntegrationOwnResolve", 1, (), lambda S: ((("x", Array(4, Value(16))),), Array(4, Value(16))),
                              lambda B, S, x: x)
    for bad, why in [("NoSuchDefinition_v1", "catalog"), (descriptor_id(bind(own)), "catalog"), (TID + "x", "catalog"), (TID[:-1], "JSONDecodeError"),
                     (TID.replace("K=64", "K=64,K=64"), "new static"), (TID.replace("K=64", "K=6x"), "no "),
                     (TID.replace("K=64", "K=65"), "refuses this binding")]:
        with pytest.raises(ValueError, match=why):
            I.resolve(bad)


def test_a_template_whose_static_is_a_composite_definition_resolves():
    from verity.ml.circuits.ops.boolean import gemm as BG
    fn = bind(BG.ScaledMmS8CoordinateRows, K=128, R=2, DOT=BG.GEMM_SM120_S8["DOT"])
    assert descriptor_id(I.resolve(descriptor_id(fn))) == descriptor_id(fn)


def test_the_advice_opens_to_the_root_its_compiler_s_tree_commits():
    c, data = {"name": "compiler", "digest": "cd" * 64}, bytes(range(70))
    reg = I.commit_advice(c, data, [bytes([i]) * 192 for i in range(3)])
    assert [len(r) for r in I.advice_rows(data)] == [32] * 3 and I.advice_rows(data)[-1][6:] == bytes(26)
    doc = I.advice_opening(data, reg)
    assert I.opened_advice(c, doc) == (data, reg.root.hex())
    assert I.opened_advice({**c, "digest": "ce" * 64}, doc)[1] != reg.root.hex()
    assert I.opened_advice(c, {**doc, "salts": [doc["salts"][1]] + doc["salts"][1:]})[1] != reg.root.hex()
    assert I.commit_advice(c, data).root != reg.root
    for bad in [{**doc, "scheme": "x"}, {**doc, "advice": "zz"}, {**doc, "salts": doc["salts"][:2]}, None]:
        with pytest.raises(ValueError):
            I.opened_advice(c, bad)
