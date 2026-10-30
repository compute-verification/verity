"""The pinned H1 tile's population (`population_program`, `TemplateInstances_v1{PouwH1Tile_v1{K=64,R=4,C=8}, N}`, a window's
program over the tile): each instance's salt is a coin, its position a position and its receipt a sink, so the port rules
accept it, and a Program that feeds an instance's salt from anything but a coin is refused."""
from __future__ import annotations

import pytest

from verity.ml.protocols.pouw.circuit import h1tile as Ht
from verity.ml.circuits import ports as P
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.population import population_program
from verity.ml.circuits.program import Program

TILE = bind(Ht.PouwH1Tile, EXACT=bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8), R=4, C=8,
            PAIRS=tuple((i, j) for i in range(4) for j in range(8)))


def test_the_population_of_two_tiles_passes_the_port_rules():
    prog = population_program(TILE, 2)
    assert P.check(prog.root) is not None
    pt, (params, _ret) = P.port_types(prog.fn), prog.fn.signature()
    kinds = {name: sorted({p.kind for _lo, _hi, p, _w in rs if p.kind}) for (name, _t), rs in zip(params, pt.params)}
    assert kinds["salt"] == ["coin"] and kinds["position"] == ["position"]
    salt_w = len(dict(params)["salt"].leaf_widths())
    assert [(lo, hi) for lo, hi, _p, _w in pt.params[[n for n, _ in params].index("salt")]] == [(0, salt_w)]
    assert pt.ret and all(o.sink for _lo, _hi, o, _w in pt.ret)


def test_a_salt_fed_from_a_run_input_is_refused():
    """The population called by a root that declares no ports: each instance's salt is then an ordinary run input, not a coin."""
    pop = population_program(TILE, 2).fn
    params, ret = pop.signature()
    bare = CompositeDefinition("H1PopulationNoCoin", 1, (), lambda S: (tuple(params), ret),
                               lambda B, S, *a: B.call(pop, *a), register=False)
    with pytest.raises(P.PortError, match=r"E_PORT_KIND: .* parameter 'salt'"):
        Program(bind(bare))
