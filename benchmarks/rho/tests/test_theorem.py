import math

from rho.theorem import (
    Blocks,
    Budget,
    Cut,
    Partition,
    Region,
    least,
    rho,
    share,
    slowdown,
)

KAPPA = 0.5


def region(*cuts, layout=None, compute=1000.0):
    return Region("r", compute, 1.0, cuts, layout)


def test_a_region_contributes_its_least_cut_and_never_more_than_its_compute():
    a = Cut("a", (Blocks("x", Budget(200, 10), 4),), {"free": 200})
    b = Cut("b", (Blocks("y", Budget(1000, 50), 1),))
    s = share(region(a, b), lambda bud: bud.inputs, KAPPA)
    # a: 4 blocks of min(10, κ·200/2 = 50) plus κ·200/2 free = 90; b: min(50, 250) = 50; full is κ·1000/2 = 250
    assert s.cut == "b" and math.isclose(s.value, 50 / 250)
    assert share(region(a, b), lambda bud: math.inf, KAPPA).value == 1.0


def test_an_attack_takes_the_layout_and_ignores_the_cuts():
    cut = Cut("tight", (Blocks("x", Budget(1000, 1), 1),))
    layout = Cut("layout", (Blocks("x", Budget(1000, 1), 1),), {"free": 200})
    s = share(region(cut, layout=layout), lambda bud: 0.0, KAPPA, side="attack")
    assert s.cut == "layout" and math.isclose(s.value, 0.2)
    assert share(region(cut), lambda bud: 0.0, KAPPA, side="attack").value == 1.0


def test_rho_is_the_dark_pool_plus_the_monitored_rate():
    assert rho(0.0, 1.0) == 1.0
    assert math.isclose(rho(0.01, 0.0), 100.0)
    assert math.isclose(rho(0.01, 0.0, efficiency=3.0), 300.0)
    assert rho(0.0, 0.0) == math.inf


def test_slowdown_weights_regions_by_their_compute():
    free = Region("free", 3000.0, 1.0, ())
    locked = Region("locked", 1000.0, 1.0, (Cut("c", (Blocks("x", Budget(1000, 0), 1),)),))
    s = slowdown(Partition((free, locked), 0.0), lambda bud: bud.inputs, KAPPA)
    assert math.isclose(s.g, 0.75) and math.isclose(s.rho, 4 / 3)
    assert math.isclose(sum(s.terms.values()), s.g)


def test_least_takes_the_smallest_cap():
    a = least(lambda b: 5.0, lambda b: 3.0, lambda b: math.inf)
    assert a(Budget(1, 1)) == 3.0 and least()(Budget(1, 1)) == math.inf
