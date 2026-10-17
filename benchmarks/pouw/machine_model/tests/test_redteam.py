"""RT-DC's red-team audit (campaign cmp-13): the binding and near facets of each headline design, decision 76's margin on
their measured lane coefficients (the driver, and the least filler that holds it), and the uniform-datapath gap a device
without a UNI class leaves open."""
from __future__ import annotations

import pytest

from verity_machine_model import eps as E
from verity_machine_model import redteam as RT
from verity_machine_model.model import device

# device -> its headline design (decision 93 / #2322 / JI89 / JIM): the rows these audits read must exist
HEAD = {"sm_90": "dither-2s-li-l32", "sm_100": "dither-2s-li-l32", "sm_89": "dither-2s-l16", "sm_80": "dither-2s-l16"}


@pytest.mark.parametrize("dev,design", sorted(HEAD.items()))
def test_audit_is_well_formed(dev: str, design: str) -> None:
    data = device(dev)
    pi = E.machine("", data)
    a = RT.audit(design, data)
    assert 0.0 <= a.core < 1.0
    assert a.binding, "the cheater's LP must bind some row"
    assert set(a.binding) <= set(pi.rows)
    assert a.near and a.near[0].load >= RT.NEAR
    for r in a.near:
        assert r.name in pi.rows
        if r.ceiling:
            assert r.name == RT.ISSUE_CEILING and r.flip_pct is None


def test_margin_moves_only_the_lane_coefficient() -> None:
    """Decision 76 moves a facet's small (lane) coefficient, not the tensor-pipe MMA value it also carries. On sm_90
    JPA90_0 (ALU beside the IGMMA tensor pipe), the move must scale the ALU coefficient and leave the MMA one exactly
    as measured. Scaling the whole column — the round-1 bug — mis-prices the tensor route and is not this margin."""
    pi = E.machine("", device("sm_90"))
    small, hit = RT._lane_coef(pi, "JPA90_0")
    assert hit == ("ALU",) and small == pytest.approx(pi.c["ALU"]["JPA90_0"])
    tensor = next(k for k in pi.c if "JPA90_0" in pi.c[k] and k != "ALU")
    assert pi.c[tensor]["JPA90_0"] > small                      # the MMA coefficient is the large one
    moved = RT._single(pi, "JPA90_0", 0.98)
    assert moved.c["ALU"]["JPA90_0"] == pytest.approx(pi.c["ALU"]["JPA90_0"] * 0.98)
    assert moved.c[tensor]["JPA90_0"] == pi.c[tensor]["JPA90_0"], "the tensor-pipe coefficient must be left alone"


def test_h200_margin_driver_is_the_alu_tensor_joint() -> None:
    """On the H200 the single facet that drives decision 76's margin is JPA90_0 (the ALU pipe beside the tensor), not
    the razor-thin generic row JI90_0: moving JI90_0's lane coefficient 2% does not raise the core. This is the round-1
    retraction (note:20261017T2016Z-draft-rt-dc): the earlier JI90_0 'flips at 0.5%' was whole-column scaling."""
    mg = RT.margin_audit("dither-2s-li-l32", device("sm_90"))
    assert mg.driver == "JPA90_0"
    singles = dict(mg.rows)
    assert "JI90_0" in singles and singles["JI90_0"] == pytest.approx(mg.base, abs=1e-6)
    assert singles["JPA90_0"] > mg.base + 1e-3
    assert mg.worst >= max(s for _, s in mg.rows) - 1e-9, "the joint margin is at least as bad as any single row"


def test_b200_issue_bound_is_robust() -> None:
    """B200's headline cheater binds the hardware issue width, which silicon cannot beat; no near measured row flips at
    decision 76's 2% margin, so the joint margin leaves the core where it is and names no driver. A refit that makes one
    flip there means B200 turned thin — the guard should fire."""
    data = device("sm_100")
    a = RT.audit("dither-2s-li-l32", data)
    assert RT.ISSUE_CEILING in a.binding
    for r in a.near:
        if not r.ceiling:
            assert r.flip_pct is None or r.flip_pct > 2.0
    mg = RT.margin_audit("dither-2s-li-l32", data)
    assert mg.driver == "" and mg.worst == pytest.approx(mg.base, abs=1e-6)


def test_least_filler_holds_the_margin_on_the_h200() -> None:
    """The H200 LI design leaks above its floor under decision 76's 2% margin (the lane route opens); the cheapest
    verified FFMA filler a word brings the joint worst back to the floor. least_filler finds that add-on; a design that
    already holds the margin needs none."""
    data = device("sm_90")
    lf = RT.least_filler("dither-2s-li-l32", data)
    assert lf.base_worst > lf.core + 1e-3, "the design as built leaks above the floor"
    assert lf.per_word is not None and lf.per_word > 0.0
    assert lf.core <= RT.margin_audit("dither-2s-li-l32", data).base + 1e-6, "the filler restores the margin to the floor"
    assert lf.mix == {"FFMA": 1}


def test_a100_uniform_datapath_is_an_open_gap() -> None:
    """sm_80's census prices no uniform datapath; crediting one at sm_90's measured rates raises the headline core
    materially, because the cheater runs the fold off the bound ALU pipe. The guard holds until sm_80 measures it."""
    data = device("sm_80")
    assert not RT.audit("dither-2s-l16", data).uniform_route, "sm_80 has no UNI class yet"
    base, opened, binding = RT.uniform_gap("dither-2s-l16", data)
    assert opened > base + 1e-3, "a uniform datapath opens a material gap on A100"
    assert binding, "the opened route still binds some row"


def test_measured_uniform_route_reports_no_gap() -> None:
    """A device whose census already prices the uniform datapath (sm_90) reports no gap to bound."""
    base, opened, _ = RT.uniform_gap("dither-2s-li-l32", device("sm_90"))
    assert opened == base
