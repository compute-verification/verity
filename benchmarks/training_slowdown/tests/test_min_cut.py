"""`training_slowdown.formalism`'s copy of the min-cut bound's arithmetic against the slowdown calculator's module, the
one place that computes it (`benchmarks/rho/rho/min_cut.py`, loaded by path): `BETA_K` and `beta_k`, `he_rho` and
`SHARE`."""
import importlib.util
import itertools
import math
from pathlib import Path

import pytest

from training_slowdown import formalism as F

MIN_CUT = Path(__file__).resolve().parents[3] / "benchmarks/rho/rho/min_cut.py"


def _min_cut():
    spec = importlib.util.spec_from_file_location("training_slowdown_tests_min_cut", MIN_CUT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _min_cut()

#: (N, O, β_w, β_x, b_y, c, k, |C|): a small work, a traced step's, one whose outputs' term outweighs the first, and
#: one of cheap weights
POINTS = ((2.0e9, 3.0e7, 15.9, 15.9, 16.0, 1.0e9, 6.5e4, 5.0e12),
          (7.2986e14, 3.5673e11, 15.9, 15.9, 16.0, 4.6566e14, 65_536, 1.2017e19),
          (3.6e10, 2.8e8, 16.0, 1.02, 16.0, 1.4e13, 4.2e8, 1.4e13),
          (1.0e18, 2.7e14, 2.5, 4.0, 4.0, 2.45e19, 2.0e7, 3.0e21))


def test_the_rows_of_beta_are_the_calculators():
    assert F.BETA_K == M.BETA
    ks = [k for k, _ in F.BETA_K]
    between = [x for a, b in itertools.pairwise(ks) for x in (math.sqrt(a * b), a + (b - a) / 3)]
    for k in (1, 640, ks[0] - 1, *ks, *between, 422_197_408):
        assert F.beta_k(k) == M.beta(k) is not None, k
    for k in (ks[-1] + 1, 1e9, 1.7e9):
        assert F.beta_k(k) is None and M.beta(k) is None


def test_he_rho_is_the_calculators_slowdown_conjunct():
    """`he_rho` is the left side of the statement's second conjunct, at `F = 1` and at `SHARE`, below 0 included."""
    assert F.SHARE == pytest.approx(M.share(3), rel=1e-15) and M.share(2) == 1
    signs = set()
    for (n, o, bw, bx, b, c, k, size), share in itertools.product(POINTS, (1.0, F.SHARE)):
        mine = F.he_rho(n, o, bw, bx, b, share, c, k, size)
        assert mine == pytest.approx(M.slowdown_at_least(n, o, size, c, k, bw * bx, b, share), rel=1e-12)
        assert mine <= M.blocks_at_least(n, o, k, bw * bx, b, share) * c / size
        signs.add(mine > 0)
    assert signs == {True, False}
