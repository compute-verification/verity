"""IDP.4A-made words pay on checked_strip's facets once the fold drops below where JI3e_0 binds honest's tile."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

import zero_routes_check as zr


def test_idp4a_route_opens_below_the_duo_fold():
    """On Llama-3.1-8B's linears under co32c, T and D words made by 8 IDP.4A each gain nothing at f = 3 and about 1.5% at
    f = 2, where checked_strip's x = 12 pricing shows no gain."""
    model = "Llama-3.1-8B"
    base3, base2 = zr.linears(3, model), zr.linears(2, model)
    assert abs(zr.linears(3, model, zr.lin_routes(zr.IDP4A)) - base3) < 1e-12
    assert abs(zr.linears(2, model, zr.lin_x(12)) - base2) < 1e-12
    assert 0.012 < zr.linears(2, model, zr.lin_routes(zr.IDP4A)) < 0.018
