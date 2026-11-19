"""A credited linear of C as Pearl+'s product (`pearl_plus_credited`): its grid's units of `PearlPlusTileCoreWords_v2`,
each one proof block whose amount is its credited MACs, and each unit's public row. That the core on these ports evaluates and
one drawn unit goes through the door is ``benchmarks/pouw/credited_in_c.py``'s run."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.protocols.resource_accounting.compute.pouw import scheme as PS
from verity.protocols.resource_accounting.compute.pouw.prover import schemes as pouw_schemes
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C

SCHEME = "pearl-c-h100-v1-h3"
K = 256


def test_a_linears_units_are_its_grid_row_major_its_last_row_tile_part_filler() -> None:
    got = PC.tiles(SCHEME, K, 3, 2, tile=(2, 1))
    assert [(t.rows, t.cols, t.col, t.real) for t in got] == [
        ((0, 1), (0,), 0, 2), ((0, 1), (1,), 1, 2), ((2, 3), (0,), 0, 1), ((2, 3), (1,), 1, 1)]
    assert {t.core.defn.id for t in got} == {PC.PearlPlusTileCoreWords.id}
    assert all({s: t.core.bindings[s] for s in ("K", "TM", "TN", "N")} == {"K": K, "TM": 2, "TN": 1, "N": 2} for t in got)
    with pytest.raises(ValueError, match="whole column tiles"):
        PC.tiles(SCHEME, K, 3, 3, tile=(1, 2))


def test_each_unit_is_one_block_of_the_scheme_whose_amount_is_its_credited_macs() -> None:
    (block,) = PS.PEARL_PLUS.blocks
    for t in PC.tiles(SCHEME, K, 3, 2, tile=(2, 1)):
        assert block.definition == t.core.defn.id
        assert block.cost(t.core.bindings) == t.real * 1 * K


def test_a_units_public_row_is_the_noise_pearl_plus_forms_its_tile_with() -> None:
    """E_B's lines and F_A, F_B and G are the ones `PearlC.encode` and `PearlC.checked` use at the tile, from the salt and
    the weights' root alone, and the rest is the unit's place: its call, rows and column tile."""
    rng = random.Random(0)
    salt, root, unit = rng.randbytes(32), rng.randbytes(64), 5
    t = PC.tiles(SCHEME, K, 2, 2, tile=(1, 1))[3]
    got = PC.public(SCHEME, K, t, unit, salt, root)
    assert tuple(got) == PC.PUBLIC
    s = pouw_schemes.get(SCHEME, k_min=K, tile=(1, 1))
    ctx = C.Context(salt, C.Matmul(unit, C.Shape(2, K, 2), 0), bytes(64), root)

    def rows(n: int) -> list:
        return [C.f32_row([int(np.float32(rng.uniform(-2, 2)).view(np.uint32)) for _ in range(K)]) for _ in range(n)]

    ops = s.encode(ctx, s.noise(ctx), rows(1), rows(1), C.Tile((1,), (1,)))
    assert got["eb"] == [list(ops.e_b[0])]
    seed_b = s.noise(ctx).seed_b
    assert got["fa"] == [list(x) for x in C._basis(seed_b, 0, K, C.LINE_NORM_V1)]
    assert got["fb"] == [list(x) for x in C._basis(seed_b, 1, K, C.LINE_NORM_V1)]
    assert got["g"] == [list(r) for r in C.gram(seed_b, K, C.LINE_NORM_V1, s.device)]
    assert (got["salt"], got["root"], got["unit"], got["rows"]) == (salt, root, unit, [1])
