"""Flock over an arbitrary partition (`verity_flock.partition_units`): every lowered unit class agrees with the reference
evaluator on synthetic inputs lifted through its flat Definition, under `Q_word` on circuit-check's tiny Serve_v1 and under two
deliberately odd owner functions on a RoPE Call; a class circuit run on another class's lanes disagrees."""
from __future__ import annotations

import pytest

pytest.importorskip("verity.ml.definitions.registry")
from circuit_check import targets as TG  # noqa: E402
from verity.ml.circuits import partition_object as PO  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402

from verity_flock import partition_units as PU  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    TG.load_registries()
    from verity.ml.definitions.registry import b1

    return Program(bind(b1.Serve, C=TG.TINY, LP=3, STEPS=2))


@pytest.fixture(scope="module")
def rope():
    TG.load_registries()
    return PU.one_call(TG.definition("RoPE_v1{NHEADS=2,D=16}"))


@pytest.mark.slow
def test_q_word_on_the_tiny_model(tiny):
    r = PU.check(PU.partition_units(tiny, "q-word"), T=8)
    assert r["units"] == 3595 and r["shapes"] == 31 and r["shapes_lowered"] >= 11
    assert r["distinct_circuits_checked"] > 0 and r["mismatches"] == 0


@pytest.mark.parametrize("spec", ["blocks:3", "scatter:5/1"])
def test_odd_partitions(rope, spec):
    units = PU.partition_units(rope, spec)
    assert not PU.check_owners(units)                                          # the partition invariant holds
    r = PU.check(units, T=8)
    assert r["shapes_lowered"] == r["shapes"] and r["mismatches"] == 0
    assert r["units"] != PU.check(PU.partition_units(rope, "q-word"), T=1)["units"]


def test_a_registered_input_is_an_input():
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.circuits.types import Value
    from verity.ml.circuits.ops.scalar import I32Add

    I32 = Value(32)
    top = CompositeDefinition("PuRegistered", 1, (), lambda S: ((("x", I32), ("eps", I32)), I32),
                              lambda B, S, x, eps: B.call(I32Add, B.call(I32Add, x, eps), eps), register=False)
    runs = [PU.check(PU.partition_units(Program(top, registered=reg), "q-word"), T=4) for reg in ((), ("eps",))]
    for r in runs:
        assert r["shapes_lowered"] == r["shapes"] == 1 and r["units"] == 2 and r["mismatches"] == 0
    assert runs[0]["classes"] == runs[1]["classes"] and runs[0]["partition"] != runs[1]["partition"]


@pytest.mark.slow
def test_a_class_circuit_rejects_another_class(tiny):
    units = PU.partition_units(tiny, "q-word")
    lowered = []
    for cl in units.classes().values():
        cc = PU.lower_class(units.activation(cl.definition), cl.representative)
        if cc.circuit is not None:
            lowered.append((cl, cc))
    (a, ca), (b, cb) = next((x, y) for x in lowered for y in lowered if x[0].shape != y[0].shape
                            and x[0].representative.in_widths == y[0].representative.in_widths
                            and x[0].representative.out_widths == y[0].representative.out_widths)
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": "test", "seed": 1})
    ins, want = PU.lanes(units.activation(b.definition), [b.representative], key, 8)[(b.representative.definition, b.representative.unit)]
    assert PU.class_mismatches(cb.circuit, b.representative, ins, want) == 0
    assert PU.class_mismatches(ca.circuit, a.representative, ins, want) > 0
