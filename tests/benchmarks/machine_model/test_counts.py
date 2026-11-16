"""`machine-model counts`: an honest kernel's closed-form instruction counts against every ncu observation of it."""
from __future__ import annotations

import pytest

from verity_machine_model import counts as cf
from verity_machine_model import eps as E
from verity_machine_model.cli import main

PI = E.machine("VF3l-16")   # the vector these counts were checked at
C = cf.load("e-l7qs")


def variant(v: str) -> list[cf.Obs]:
    return [o for o in C.obs if C.variant(o.shape[2]) == v]


def test_every_observation_within_tolerance():
    assert C.check() == []
    assert C.check_bound(PI) == []


def test_the_coefficients_are_the_fit_of_the_observations():
    """k2's four terms are fitted; k3 runs the same place_step loop, so it takes k2's cross and fits the other three."""
    assert cf.fit(variant("k2"), C.tile, C.ctas) == C.variants["k2"]
    assert cf.fit(variant("k3"), C.tile, C.ctas, cross=C.variants["k2"]["cross"]) == C.variants["k3"]


def test_the_crossings_are_the_rasters():
    """At m = 2048 every CTA crosses into the second group once; the 28672-row gate_up's 28 groups take 27 crossings a
    CTA."""
    assert cf.crossings(2048, 4096, C.tile, C.ctas) == C.ctas
    assert cf.crossings(28672, 6016, C.tile, C.ctas) == 27 * C.ctas
    assert cf.crossings(8192, 8192, C.tile, C.ctas) == 1316


BASE = {"k2": [(2048, 4096, 1024), (2048, 4096, 4096), (2048, 28672, 4096), (28672, 6016, 4096)],
        "k3": [(2048, 4096, 8192), (2048, 4096, 14336), (8192, 8192, 8192)]}


def test_a_few_shapes_predict_the_rest():
    """Each variant fitted on one observation at each of a few shapes (its terms apart: k2's four, k3's three with k2's
    cross) predicts every other observation within tolerance, its model bound too: k2 the crossings at m = 4096 and 6144
    from m = 2048 and 28672."""
    for v, shapes in BASE.items():
        base = [next(o for o in variant(v) if o.shape == s) for s in shapes]
        cross = None if v == "k2" else C.variants["k2"]["cross"]
        held = cf.Counts(**{**C.__dict__, "variants": {**C.variants, v: cf.fit(base, C.tile, C.ctas, cross)},
                            "obs": tuple(o for o in variant(v) if o not in base)})
        assert len(held.obs) >= 4 and held.check() == [] and held.check_bound(PI) == [], v


def test_the_exact_classes_are_the_kernels_structure():
    """A stage is 4 atoms of 8 MMAs and 2 LDSM.x4 a warp over 16 warps, and two TMA loads; the peel is P atoms (2, or 3
    for k > 4096) and its stage's two loads."""
    for v, peel in (("k2", 2), ("k3", 3)):
        s, t = C.variants[v]["stage"], C.variants[v]["tile"]
        assert (s["MMA"], t["MMA"]) == (16 * 32, 16 * 8 * peel)
        assert (s["LDSM_X4"], t["LDSM_X4"]) == (16 * 16, 16 * 4 * peel)
        assert (s["UTMA16K"], t["UTMA16K"]) == (2, 2)
        assert not any(C.variants[v]["launch"].get(c) for c in C.exact)


def test_the_bound_at_8192_is_the_measured_profiles():
    h = E.honest("e-l7qs")
    load = PI.load(h.per_mma)
    t, rows = C.bound(PI, 8192, 8192, 8192)
    assert t == pytest.approx(max(load.values()), rel=C.bound_tolerance)
    assert rows == ("JI3e_0",)


def test_refuses_what_the_kernel_does_not_run():
    for shape in ((1000, 4096, 4096), (2048, 4100, 4096), (2048, 4096, 4000), (2048, 4096, 16384), (1024, 2048, 4096)):
        with pytest.raises(SystemExit):
            C.n(*shape)


def test_cli(capsys):
    assert main(["counts", "e-l7qs", "--check", "--vector", "VF3l-16"]) == 0
    assert "pass: 22 observations" in capsys.readouterr().out
    assert main(["counts", "e-l7qs", "2048", "28672", "4096", "--vector", "VF3l-16"]) == 0
    assert "e_l7qs at 2048 x 28672 x 4096: bound 22.5157" in capsys.readouterr().out
