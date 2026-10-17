"""RT-DC's red-team audit (campaign cmp-13): the binding and near facets of each headline design, the decision-76
flip, and the uniform-datapath gap a device without a UNI class leaves open."""
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


def test_b200_issue_bound_is_robust() -> None:
    """B200's headline cheater binds the hardware issue width, which silicon cannot beat; no near measured row flips
    at decision 76's 2% margin. A refit that makes one flip there means B200 turned thin — the guard should fire."""
    a = RT.audit("dither-2s-li-l32", device("sm_100"))
    assert RT.ISSUE_CEILING in a.binding
    for r in a.near:
        if not r.ceiling:
            assert r.flip_pct is None or r.flip_pct > 2.0


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
