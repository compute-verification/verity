"""The limb-form v2 epilogue: the same function as ``Pack.check`` over Z, a BabyBear-verifiable witness, and the
circuit's own rejection of a wrong word / every bumped hint (``gpu/v2/epilogue_limbs.py``).  CPU only."""
import random

import pytest
from verity_numerical.checker import REAL
from verity_numerical.gkr_export import v2 as V2

from gpu.v2 import epilogue_limbs as EL

P = 2**31 - 2**27 + 1
p = REAL
G = len(p.groups)


def _states(n, seed=1):
    rng = random.Random(seed)
    for _ in range(n):
        z = rng.randint(0, 1)
        st = (rng.randint(0, 1), 0 if z else rng.randint(p.e_min, p.e_min + 250), 0 if z else rng.randint(2**23, 2**24 - 1), z)
        flags = [(o := rng.choice([0, 0, 0, 1]), rng.randint(0, 1) if o else 0) for _ in range(G)]
        yield st, flags


def test_pack_limbs_matches_pack():
    for st, flags in _states(5000):
        hi, lo, _ = EL.pack_word_limbs(p, st, flags)
        assert hi * 2**16 + lo == EL.pack_check_word(p, st, flags)
        assert hi < 2**16 and lo < 2**16


def test_witness_verifies_mod_babybear():
    ctx, paths = EL.build_epilogue_circuit_limbs(p)
    for st, flags in _states(2000, seed=2):
        y16 = EL.pack_word_limbs(p, st, flags)[2]
        assert V2.verify_row(ctx, EL.epilogue_witness_limbs(ctx, paths, st, flags, y16, p), P) is None


@pytest.mark.parametrize("bump", [None, *EL.HINTS])
def test_circuit_rejects(bump):
    ctx, paths = EL.build_epilogue_circuit_limbs(p)
    st, flags = next(_states(1, seed=3))
    y16 = EL.pack_word_limbs(p, st, flags)[2]
    off = 5 + 2 * G
    if bump is None:
        row = EL.epilogue_witness_limbs(ctx, paths, st, flags, (y16 + 1) % 2**16, p)
    else:
        base = EL.epilogue_witness_limbs(ctx, paths, st, flags, y16, p)[: off + 1 + len(paths)]
        base[off + 1 + paths.index(f"h.{bump}")] += 1
        row = V2.derive(ctx, base + [None] * (len(ctx.cols) - len(base)))
    assert V2.verify_row(ctx, row, P) is not None
