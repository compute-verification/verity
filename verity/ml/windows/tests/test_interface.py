"""What an integration hands Verity (`verity.ml.windows.interface`), against the interface alone: a circuit takes only
Verity's Definitions, a unit's rows are its Definition's bit rows as C-Flock stages them, a window tiles the partition and
commits under the window-domain rule, an integration's outputs are checked against the Definitions before it commits, and
the verifier's window is rebuilt from the circuit record alone. A window written under core's registration, and
`of_records` against it, are the records audit's tests (`benchmarks/prove_path/tests/test_records_audit.py`), since these
may not import core."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.types import Array, Value
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.ml.windows import interface as I, rows as CS

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
    doc = json.loads(I.circuit_bytes(I.Window(I.Circuit("rec", ((COORD, 10),)), RUN, LAW, 4).circuit_record()))
    with pytest.raises(ValueError, match="not the partition's units"):
        I.window_of_record({**doc, "templates": [[TID, 11]]}, LAW, BY_ID.__getitem__)
    with pytest.raises(ValueError, match="not the ones its templates make"):
        I.window_of_record({**doc, "partition": "00" * 64}, LAW, BY_ID.__getitem__)
    with pytest.raises(ValueError, match="resolves"):
        I.window_of_record(doc, LAW, lambda t: ACC)
    with pytest.raises(ValueError, match="window-circuit"):
        I.window_of_record({**doc, "format": "x"}, LAW, BY_ID.__getitem__)


def test_an_advice_binding_is_well_formed():
    c = {"name": "compiler", "digest": "cd" * 64}
    assert I.advice_binding(c, "ef" * 64, 12) == {"compiler": c, "advice": {"root": "ef" * 64, "bits": 12}}
    for bad in [({"name": "c"}, "ef" * 64, 1), (c, "ef" * 32, 1), (c, "ef" * 64, -1), (c, "ef" * 64, True)]:
        with pytest.raises(ValueError):
            I.advice_binding(*bad)


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
