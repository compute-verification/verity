"""R2's other partition (`r2_population`), which every arm's registration-partition-mismatch negative registers, on toy
windows: a one-unit window, whose one-short population would be empty and crashed the arm after its verdict, a mixed window
of single instances, and a window of several. Each builds, differs from the window's own partition, and the registration
check refuses the window's record against it with an R2 code."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.types import Value as Bits
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import r2_population as R2  # noqa: E402

XOR8 = PrimitiveDefinition("R2PopulationTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
AND8 = PrimitiveDefinition("R2PopulationTestAnd8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a & b,
                           register=False)
SCHEME = "frame-v3-sha512/hm96-sha512"
LAW = {"law": "subset", "k": 1}
WINDOW = {"kind": "served", "row": 1, "run": "r-test"}


def _ports(program, query) -> dict:
    part, n = P.partition(program, query), P.units(program, query)
    return {"x": {"leaves": n, "domain": R.window_domain(part["program"], P.digest(part), "x", "hm96-sha512/row/v1", n,
                                                         "r-test")}}


def _own(population, mixed: bool):
    if mixed:
        return P.mixed_population_program(population), P.template_instances_query([d for d, _ in population])
    return P.population_program(*population[0]), P.template_instance_query(population[0][0])


@pytest.mark.parametrize(("population", "mixed", "counts"), [
    ([(XOR8, 1)], False, [2]),
    ([(XOR8, 1)], True, [2]),
    ([(XOR8, 1), (AND8, 1)], True, [2, 1]),
    ([(XOR8, 3)], False, [2]),
    ([(XOR8, 3), (AND8, 1)], True, [2, 1]),
], ids=["one-unit", "one-unit-mixed", "single-instances", "three", "three-and-one"])
def test_r2_registers_another_nonempty_partition(population, mixed, counts):
    assert [n for _, n in R2.resized(population)] == counts
    own, query = _own(population, mixed)
    other = R2.program(population, mixed=mixed)
    assert P.units(other, query) == sum(counts)
    assert P.digest(P.partition(other, query)) != P.digest(P.partition(own, query))
    ports = _ports(own, query)
    rec = R.record(own, query, LAW, SCHEME, [{"port": p, "root": "ab" * 64, **w} for p, w in ports.items()], WINDOW,
                   leaf_layer="ee" * 64)
    assert R.check(rec, own, query, LAW, SCHEME, ports, leaf_layer="ee" * 64).ok
    v = R.check(rec, other, query, LAW, SCHEME, ports, leaf_layer="ee" * 64)
    assert not v.ok and any(c.startswith("R2-") for c in v.codes)


def test_one_template_is_not_a_mixed_population():
    with pytest.raises(ValueError):
        R2.program([(XOR8, 1), (AND8, 1)], mixed=False)
