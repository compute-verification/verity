"""`regions.required_size` against `upstream.required` on the explicit wiring: random regions made of whole activations of
circuits that cover every node form (`synthetic.net`: calls, batches on axes None, 0 and 1, a scan, a pass-through) and of a
real GEMM's dot steps, with random classes among the gates the region's activations return, swept gate by gate and node by
node."""

from __future__ import annotations

import random

import pytest
import synthetic as SY
from circuit_properties import references as R
from circuit_properties import regions as RG
from circuit_properties import upstream as U

from verity.ml.circuits.layout import Circuit


def _wiring(program):
    C = Circuit(program.root)
    return C, {g: tuple(C.gate(g).operands) for g in range(program.root.gates)}


def _disjoint(rng, acts, k):
    chosen, taken = [], []
    for s in rng.sample(acts, len(acts)):
        lo, hi = s.offset, s.offset + s.spec.gates
        if hi > lo and all(hi <= a or lo >= b for a, b in taken):
            chosen.append(s)
            taken.append((lo, hi))
            if len(chosen) == k:
                break
    return chosen


def _agree(program, trials: int, seed: int, flat_max: int, min_gates: int = 1) -> int:
    """Trials whose required set is nonempty, every trial having agreed."""
    rng = random.Random(seed)
    C, reads = _wiring(program)
    acts = [s for s in RG.activations(C.root_scope) if s.spec.gates >= min_gates]
    checked = 0
    for _ in range(trials):
        region = _disjoint(rng, acts, rng.randint(1, 6))
        A = frozenset(g for s in region for g in range(s.offset, s.offset + s.spec.gates))
        named = sorted({g for s in region for g in RG.returns(s) if g in A})
        outside = [g for g in range(program.root.gates) if g not in A]
        classes = []
        for _c in range(rng.randint(1, 4)):
            T = set(rng.sample(named, rng.randint(1, min(3, len(named)))))
            if outside and rng.random() < 0.3:
                T.add(rng.choice(outside))
            classes.append(frozenset(T))
        got = RG.required_size(region, classes, RG.Pull(flat_max=flat_max))
        want = U.required(reads, A, classes)
        assert got.gates == len(A)
        assert got.required == len(want), ([s.path() for s in region], classes)
        checked += len(want) >= 8
    return checked


@pytest.mark.parametrize("flat_max", [0, 10 ** 9])
def test_synthetic_net(flat_max):
    assert _agree(SY.net(), 300, 11, flat_max) > 10
    assert _agree(SY.net(), 100, 12, flat_max, min_gates=8) > 10


@pytest.mark.parametrize("flat_max,trials", [(0, 6), (10 ** 9, 30)])
def test_a_gemm_of_dot_steps(flat_max, trials):
    assert _agree(R.boolean_gemm(K=32, N=2), trials, 5, flat_max, min_gates=200) >= trials // 3


def test_a_class_gate_must_be_returned():
    C, _reads = _wiring(SY.net())
    big = max(RG.activations(C.root_scope), key=lambda s: s.spec.gates)
    rets = set(RG.returns(big))
    inner = next(g for g in range(big.offset, big.offset + big.spec.gates) if g not in rets)
    with pytest.raises(ValueError, match="no activation of it returns it"):
        RG.required_size([big], [frozenset({inner})])


def test_canonical_forgets_the_class_names():
    a, _ = RG.canonical([0b0011, 0b0001, 0b0100])
    b, atoms = RG.canonical([0b1100, 0b0100, 0b0001])
    assert a == b == (0b011, 0b010, 0b100) and atoms == [0b1000, 0b0100, 0b0001]
