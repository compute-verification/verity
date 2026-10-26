"""What an integration hands Verity (`verity.ml.windows.interface`), against the interface alone: a circuit takes only
Verity's Definitions, a unit's rows are its Definition's bit rows as C-Flock stages them, a window tiles the partition and
commits under the window-domain rule, an integration's outputs are checked against the Definitions before it commits, a
wire pattern wires whole members unit by unit under its writers' salts, a cut port's rows are each wired from their own
writer, a registered read's ports are tables of the registered rows under the value's salts, every other input row is
declared registered, ingress or free and counted from the description, and the verifier's window (and its registered
reads) is rebuilt from the circuit record alone, its members stated by cap, wires, cuts and reads. A window written under
core's registration, and `of_records` against it, are the records audit's tests
(`benchmarks/prove_path/tests/test_records_audit.py`), since these may not import core."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.ml.windows import interface as I, rows as CS, wiring as WI

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
    """A Boolean-profile Definition (every leaf one bit, as network-accounting's egress units) has the bit rows of its
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


def test_mismatches_are_the_units_whose_outputs_are_not_the_definitions():
    win, ins, outs = window()
    assert win.mismatches() == {}
    bad = I.Window(I.Circuit("test", ((COORD, 10),)), RUN, LAW, 4)
    wrong = outs[0].copy()
    wrong[7] ^= 1
    bad.record(COORD, ins, [wrong])
    assert bad.mismatches() == {TID: [7]}


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


def chain_values(seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """`chain`'s units' inputs (x, w) and outputs in its order, each unit's input its writer's output."""
    rng = np.random.default_rng(seed)

    def bf16(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    w = [bf16(rng.standard_normal((16, 16)) / 4) for _ in range(LAYERS)]
    x = {(0, 0, r): bf16(rng.standard_normal(16)) for r in range(REQUESTS)}
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
    with pytest.raises(ValueError, match="cut rows"):
        I.Window(c, RUN, LAW, 2)


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
    one salt a row, and a statement's table of the rows at some positions (the paths are core's; none here)."""

    def __init__(self, name: str, rows: np.ndarray) -> None:
        self.name, self.rows, self.words, self.leaves = name, rows, rows.shape[1], len(rows)
        self.salts = [hashlib.shake_256(b"salt %d" % i).digest(CS.SALT_BYTES) for i in range(len(rows))]

    def statement(self, positions):
        return ([self.rows[p] for p in positions], [self.salts[p] for p in positions],
                {"value": self.name, "positions": list(positions), "paths": []})


#: core's ``one_stage.registered`` names `interface.registered_reads` reads, as the caller passes the module
REG = type("REG", (), {"WORD_BITS": 16, "SCHEMA_KEY": "row_schema", "READS_FROM": "reads_from",
                       "READS_V1": "verity/registered-reads/v1"})
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
    """#4317's record format: the reads in the record's ``registered``, its other fields only when declared, and the free
    input bits counted from the description alone."""
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
    assert "registered" not in json.loads(I.circuit_bytes(window()[0].circuit_record()))
    for bad, why in [({**doc["registered"], "params": {"w": 5 * K // 2}}, "leaves"),
                     ({**doc["registered"], "reads": [{**doc["registered"]["reads"][0], "port": "frames"}]}, "names no parameter"),
                     ({**doc["registered"], "reads": doc["registered"]["reads"] * 2}, "both fill rows"),
                     ({**doc["registered"], "reads": [{**doc["registered"]["reads"][0], "x": 0}]}, "a read's record")]:
        with pytest.raises(ValueError, match=why):
            I.input_count({**doc, "registered": bad}, BY_ID.__getitem__)


def test_a_read_covers_whole_members_and_reads_whole_rows():
    reg = lambda r: I.Registered({"w": 6 * K // 2}, reads=(r,))  # noqa: E731
    with pytest.raises(ValueError, match="covers part of"):
        I.Window(I.Circuit("part", ((COORD, 12),), registered=reg(I.Read(TID, "wrow", "w", 2, (5, 2), 0, (K // 2, 0)))),
                 RUN, LAW, 4)
    with pytest.raises(ValueError, match="whole registered rows"):
        I.Window(I.Circuit("half", ((COORD, 12),), registered=reg(I.Read(TID, "wrow", "w", 0, (12,), 0, (K // 8,)))),
                 RUN, LAW, 4)
    with pytest.raises(ValueError, match="uncut registered reads"):
        I.Window(I.Circuit("ingress", ((COORD, 12),), ingress=(I.Ingress(TID, "x", 0, 12, 0),)), RUN, LAW, 4)


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
    assert win._entries[m1.key] == {"wrow": {"value": "w", "positions": [2, 3], "paths": []}}
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
