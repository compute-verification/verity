"""fp4_census.py's vectorised atom is bit-exact against verity.ml.tc, and its quantizers match fp4_quality.py's."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fp4_census as C  # noqa: E402


def test_atom_matches_the_pinned_models():
    rep = C.selftest(cases=600, seed=11)
    for name, r in rep.items():
        assert r["step_mismatches"] == 0 and r["chain_mismatches"] == 0, (name, r)


def test_e2m1_rounds_to_nearest_even():
    ties = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, 7.0, -100.0])
    got = C.E2M1_VALUES[C.e2m1_codes(ties) & 7] * np.where(C.e2m1_codes(ties) & 8, -1, 1)
    assert got.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, 6.0, -6.0]


@pytest.mark.parametrize("fmt", ["nvf4", "mxf4"])
def test_quantizer_is_lossless_on_its_own_grid(fmt):
    rng = np.random.default_rng(3)
    x = rng.standard_t(3, (16, 256))
    codes, sc, g = C.quantize(x, fmt)
    d = C.dequantize(codes, sc, g, fmt)
    codes2, sc2, _ = C.quantize(d, fmt)
    assert np.array_equal(C.dequantize(codes2, sc2, g, fmt), d)


@pytest.mark.parametrize("fmt", ["nvf4", "mxf4"])
def test_quality_quantizers_agree(fmt):
    torch = pytest.importorskip("torch")
    import fp4_quality as Q
    rng = np.random.default_rng(4)
    x = rng.standard_t(3, (32, 512))
    codes, sc, g = C.quantize(x, fmt)
    want = C.dequantize(codes, sc, g, fmt)
    got = Q.QUANT[fmt](torch.tensor(x, dtype=torch.float32)).double().numpy()
    assert np.allclose(got, want, rtol=0, atol=1e-6 * np.abs(x).max())


def test_uniform_scale_chains_are_exact():
    """Equal scales everywhere: every step sums integers far below 2^24, so the whole chain is exact."""
    from verity.ml.tc.models import BLACKWELL_SM120_MXF4
    atom = C.Atom(BLACKWELL_SM120_MXF4)
    rng = np.random.default_rng(5)
    A, B = C.family("grid-aligned", 16, 2048, rng)
    r = C.census_matrix(atom, "mxf4", A, B, "none", 1.0, rng, pairs=16)
    assert r["words"]["exact_chains"] == 1.0


def test_breakeven_has_no_paying_dense_rewrite():
    doc = C.breakeven(C.PRICES)
    dense = [r for r in doc["strassen"] if "sparse" not in r["sub_products"] and "hypothetical" not in r["sub_products"]]
    assert dense and not any(r["pays"] for r in dense)
