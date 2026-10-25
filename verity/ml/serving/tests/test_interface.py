"""The stack-neutral serving interface (`verity.ml.serving.interface`), against the interface alone: workloads take only
Verity's Definitions, a unit's rows are its Definition's bit rows as C-Flock stages them, a window tiles the partition and
commits under the served-domain rule, and a stack's outputs are checked against the Definitions before it commits. Core's
acceptance of the written registration is the torch loop's test (`integrations/torch_loop/tests`), since these may not
import core."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.types import Array, Value
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.ml.serving import interface as I, rows as CS

K = 64
COORD = bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM120_S8)
ACC = bind(G.GemmAccumulatorS8, K=K, **G.GEMM_SM120_S8)
TID, AID = descriptor_id(COORD), descriptor_id(ACC)
LAW = {"law": "stratified", "k": 2}
RUN = "ab" * 64


def values(n: int, seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``n`` coordinates' inputs (x, sx, wrow, sw as `evaluate_batch` takes them) and the Definition's outputs on them."""
    rng = np.random.default_rng(seed)
    x, w = (rng.integers(-128, 128, (n, K)).astype(np.int8) for _ in range(2))
    sx, sw = ((np.abs(rng.standard_normal(n)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
    return ins, [np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16)]


def window(n: int = 10, cap: int = 4) -> tuple[I.ServingWindow, list[np.ndarray], list[np.ndarray]]:
    win = I.ServingWindow(I.Workload("test", ((COORD, n),)), RUN, LAW, cap)
    ins, outs = values(n)
    win.record(COORD, [a[:6] for a in ins], [a[:6] for a in outs], I.Served("r0", op_path="model.layers.0.mlp.up_proj"))
    win.record(COORD, [a[6:] for a in ins], [a[6:] for a in outs], I.Served("r1", op_path="model.layers.0.mlp.up_proj"))
    return win, ins, outs


def test_a_workload_takes_only_verity_definitions():
    I.Workload("ok", ((COORD, 3),))
    own = CompositeDefinition("StackOwnIdentity", 1, (), lambda S: ((("x", Array(4, Value(16))),), Array(4, Value(16))),
                              lambda B, S, x: x, register=False)
    with pytest.raises(ValueError, match="Verity registers"):
        I.Workload("own", ((bind(own), 3),))
    with pytest.raises(ValueError, match="complete binding"):
        I.Workload("unbound", ((G.ScaledMmS8Coordinate.bind(K=K), 3),))
    with pytest.raises(ValueError, match="each once"):
        I.Workload("twice", ((COORD, 3), (COORD, 2)))


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
    program, query = I.Workload("test", ((COORD, 10),)).program()
    assert CS.template_partition(program, query).population == win.window.population == 10


def test_close_commits_every_unit_under_the_served_domain_rule():
    win, ins, outs = window()
    with pytest.raises(ValueError, match="past"):
        win.record(COORD, [a[:1] for a in ins], [a[:1] for a in outs], I.Served("r2"))
    c = win.close(salt_key=bytes(32))
    want = CS.expected_ports(win.window)
    assert set(c.roots) == set(want) and all(c.domains[p].domain_id.hex() == want[p]["domain"] for p in want)
    m1 = win.window.members[1]
    assert np.array_equal(c.rows[m1.port("x")], I.words(ins[0][4:8], 8, 4, K // 2))
    assert np.array_equal(c.rows[m1.port("o0")], outs[0][4:8, None])
    again = win.close(salt_key=bytes(32))
    assert again.roots == c.roots
    changed, _, _ = window()
    changed._batches[TID][1].words[0][0, 0] ^= 1
    assert changed.close(salt_key=bytes(32)).roots[win.window.members[1].port("x")] != c.roots[m1.port("x")]


def test_a_window_closes_over_every_unit():
    win = I.ServingWindow(I.Workload("test", ((COORD, 10),)), RUN, LAW, 4)
    ins, outs = values(10)
    win.record(COORD, [a[:6] for a in ins], [a[:6] for a in outs], I.Served("r0"))
    with pytest.raises(ValueError, match="closes over all"):
        win.close()


def test_the_index_names_each_members_served_computations():
    win, *_ = window()
    rows = [m["rows"] for m in win.index()["members"]]
    assert [[e[:3] for e in r] for r in rows] == [[[0, 6, "r0"]], [[0, 6, "r0"], [6, 10, "r1"]], [[6, 10, "r1"]]]


def test_mismatches_are_the_units_whose_outputs_are_not_the_definitions():
    win, ins, outs = window()
    assert win.mismatches() == {}
    bad = I.ServingWindow(I.Workload("test", ((COORD, 10),)), RUN, LAW, 4)
    wrong = outs[0].copy()
    wrong[7] ^= 1
    bad.record(COORD, ins, [wrong], I.Served("r0"))
    assert bad.mismatches() == {TID: [7]}


def test_several_templates_are_one_population():
    w = I.Workload("two", ((COORD, 3), (ACC, 2)))
    win = I.ServingWindow(w, RUN, LAW, 2)
    assert [(m.template, m.base, m.n) for m in win.window.members] == [(TID, 0, 2), (TID, 2, 1), (AID, 3, 2)]
    assert I.layout(ACC).outs == (("o0", 2),)
