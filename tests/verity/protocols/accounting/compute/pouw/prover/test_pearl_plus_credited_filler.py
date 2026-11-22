"""A partial tile on C's words: `PearlPlusTileCoreWords_v5` over a linear of one row at TM = 2, so its one unit has one
real row (TR = 1) and one filler row. Honest, its ``ok`` is 1, its digest `pearl_plus_tile.reference`'s and its y the real
row's served y, and it is credited its real rows; a nonzero word in the filler row makes ``ok`` 0. The core's own filler rule (on its word body and
its σ) is the verifier suite's `test_circuit_pearl_plus_tile`; this is the same rule through the word ports C reads."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.circuits.evaluation import evaluate_batch
from verity.protocols.accounting.compute.pouw import scheme as PS
from verity.protocols.accounting.compute.pouw.prover import schemes as pouw_schemes
from verity.protocols.accounting.compute.pouw.prover.circuit import inert_bound
from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c_u as CU

SCHEME = "pearl-c-h100-v1-h3"
K = 256


def _flat(x) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    return [w for v in x for w in _flat(v)]


def _bf16_row(rng: random.Random) -> list[int]:
    return [int(np.float32(rng.uniform(-2, 2)).view(np.uint32)) & 0xFFFF0000 for _ in range(K)]


@pytest.fixture(scope="module")
def evaluated():
    """The unit, and C's evaluation of it on its honest words and on the same words with filler row 1's word 3 set to 1.0,
    beside `reference`'s (digest, y, ok) of each, and the served y of the honest tile."""
    rng = random.Random("credited/filler")
    (t,) = PC.tiles(SCHEME, K, 1, 1, tile=(2, 1))
    s = pouw_schemes.get(SCHEME, k_min=K, tile=(2, 1))
    salt, root, unit = rng.randbytes(32), rng.randbytes(64), 7
    a, b = [_bf16_row(rng), [0] * K], [_bf16_row(rng)]
    ctx = C.Context(salt, C.Matmul(unit, C.Shape(len(a), K, 1), 0), bytes(64), root)
    ops = s.encode(ctx, s.noise(ctx), [C.f32_row(r) for r in a], [C.f32_row(r) for r in b], C.Tile(t.rows, t.cols))
    assert ops is not None
    checked = s.checked(ctx, ops)
    dev = s.device.name
    _d, ng = inert_bound._promoted(dev, K)
    cert = [[[list(inert_bound.certificates(ops.a.codes[i], ops.b.codes[0], dev))[2 * z:2 * z + 2] for z in range(ng - 1)]]
            for i in range(2)]
    y = [[CU.y(checked.u[0][0], ops.a.alpha[0], ops.b.alpha[0])]]
    pub = PC.public(SCHEME, K, t, unit, salt, root)
    bad = [a[0], [*a[1][:3], 0x3F800000, *a[1][4:]]]
    rho, rhor = t.core.bindings["RHO"], t.core.bindings["RHOR"]
    want = [PT.reference(dev, K, 1, x, b, cert, salt, root, unit, list(t.rows), list(t.cols), rho, rhor, t.real)
            for x in (a, bad)]

    def args(x):
        return [x, [[w >> 16 for w in r] for r in b], cert, *(pub[p] for p in PC.PUBLIC)]

    out = evaluate_batch(t.core, *(np.asarray([_flat(h), _flat(z)], dtype=np.int64) for h, z in zip(args(a), args(bad))))
    got = []
    for g in out.reshape(2, -1).tolist():
        got.append((bytes(sum(g[8 * i + j] << j for j in range(8)) for i in range(64)), [g[512:513]], g[513]))
    return t, got, want, y


@pytest.mark.slow
def test_a_partial_tile_passes_and_is_credited_its_real_rows(evaluated) -> None:
    """Slow: C's word ports evaluate through the σ core, the only check that a partial tile passes on the words C reads."""
    t, got, want, y = evaluated
    assert (t.core.bindings["TM"], t.real) == (2, 1)
    assert t.core.signature()[1].fields[1].n == 1
    assert want[0][1:] == (y, 1) and got[0] == (want[0][0], y, 1)
    (block,) = PS.PEARL_PLUS.blocks
    assert block.cost(t.core.bindings) == PS.tile_macs(t.core.bindings) == 1 * 1 * K


@pytest.mark.slow
def test_a_nonzero_filler_row_fails(evaluated) -> None:
    """Slow: as above; a filler row is held to zero word for word, so one nonzero word sets ``ok`` to 0."""
    _t, got, want, _y = evaluated
    assert want[1][2] == 0 and got[1] == want[1]
