"""Per-unit circuits over a partition (`verity_flock.partition_units`): on circuit-check's tiny Serve_v1 Program, the units of
Q_word v1 and of a template query cover every computing gate once, read only committed values, and the lowered classes agree
with the reference evaluator on synthetic Call inputs; a class circuit run on another class's lanes disagrees."""
from __future__ import annotations

import pytest

pytest.importorskip("verity_vllm.program.registry")
from circuit_check import targets as TG  # noqa: E402
from verity.ir import partition_object as PO  # noqa: E402
from verity.ir.codec import descriptor_id  # noqa: E402
from verity.ir.cut import derived_committed  # noqa: E402
from verity.ir.defs import bind  # noqa: E402
from verity.ir.program import Program  # noqa: E402

from verity_flock import partition_units as PU  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    TG.load_registries()
    from verity_vllm.program.registry import b1

    return Program(bind(b1.Serve, C=TG.TINY, LP=3, STEPS=2))


def test_q_word_units_cover_each_call_once(tiny):
    units = PU.PartitionUnits(tiny, PO.query(16, 32))
    placements = list(units)
    assert len(placements) == units.partition.population
    for p, (k, cut) in enumerate(units.partition.calls):
        act = units.activation(cut.fn)
        owner = units.partition.owner(p)
        committed = {act.graph.keep[c] for c in derived_committed(act.graph, owner)}
        mine = [pl.circuit for pl in placements if pl.node == k]
        owned = [j for uc in mine for j in uc.gates if j in set(act.graph.keep)]
        assert sorted(owned) == sorted(act.graph.keep)                    # every computing gate in exactly one unit
        for uc in mine:
            assert set(uc.outputs) <= set(uc.gates) and set(uc.outputs) <= committed
            for x in uc.inputs:                                           # a parameter leaf, or another unit's committed gate
                assert x < 0 or (x in committed and x not in uc.gates)


def test_q_word_lowered_classes_match_the_evaluator(tiny):
    r = PU.check(PU.PartitionUnits(tiny, PO.query(16, 32)), T=8)
    assert r["units"] == 3595 and r["shapes_lowered"] >= 1
    assert r["units_checked"] == r["units_lowered"] > 0
    assert r["mismatches"] == 0


def test_template_instances_make_each_call_one_unit(tiny):
    ids = sorted({descriptor_id(n.fn) for k, n in enumerate(tiny.root.body.nodes) if k in dict(PO.calls(tiny))})
    units = PU.PartitionUnits(tiny, PO.template_instances_query(ids))
    placements = list(units)
    assert len(placements) == len(PO.calls(tiny))
    for pl in placements:
        act = units.activation(tiny.root.body.nodes[pl.node].fn)
        assert pl.circuit.computing == act.graph.n
        assert all(x < 0 for x in pl.circuit.inputs)                      # a whole instance reads only its Call's parameters


def test_a_class_circuit_rejects_another_class(tiny):
    units = PU.PartitionUnits(tiny, PO.query(16, 32))
    by_shape = {}
    for pl in units:
        by_shape.setdefault(pl.circuit.shape, pl)
    lowered = [(pl, PU.lower_class(units, pl.circuit)) for pl in by_shape.values()]
    lowered = [(pl, cc) for pl, cc in lowered if cc.circuit is not None]
    a, b = next((x, y) for x in lowered for y in lowered
                if x[0].circuit.shape != y[0].circuit.shape and x[0].circuit.in_widths == y[0].circuit.in_widths
                and x[0].circuit.out_widths == y[0].circuit.out_widths)
    act = units.activation(tiny.root.body.nodes[b[0].node].fn)
    widths = [act.circuit.gate(p).prim.ret.w for p in range(act.lo)]
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": "test", "seed": 1})
    runs = [act.evaluate(w) for w in PU.synthetic_words(key, b[0].node, 0, widths, 8)]
    ins = [[r[x] for r in runs] for x in b[0].circuit.inputs]
    want = [[r[j] for r in runs] for j in b[0].circuit.outputs]
    assert PU.class_mismatches(b[1].circuit, b[0].circuit, ins, want) == 0
    assert PU.class_mismatches(a[1].circuit, a[0].circuit, ins, want) > 0
