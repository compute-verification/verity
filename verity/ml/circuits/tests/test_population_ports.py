"""`population_program` of a template that takes a coin and a position (a salted tile, PoUW's `PouwH1Tile`): the population
keeps each instance's coin and position, so the port rules accept it; any other template's population declares no ports."""
from __future__ import annotations

import pytest

from verity.ml.circuits import ports as P
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.population import population_program
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Value

V32 = Value(32)
TAdd = PrimitiveDefinition("TPopAdd32", 1, (("a", V32), ("b", V32)), V32, lambda a, b: (a + b) & 0xFFFFFFFF,
                           register=False)
#: a seed: a coin and a position in, no sink out, so no rule follows the coin into its body
Seed = bind(CompositeDefinition("TPopSeed", 1, (), lambda S: ((("salt", V32), ("position", V32)), V32),
                                lambda B, S, salt, position: B.call(TAdd, salt, position), register=False,
                                ports=lambda S: P.Ports(params={"salt": P.In(kind="coin"),
                                                                "position": P.In(kind="position")})))
Plain = bind(CompositeDefinition("TPopPlain", 1, (), lambda S: ((("a", V32), ("b", V32)), V32),
                                 lambda B, S, a, b: B.call(TAdd, a, b), register=False))


def test_a_coin_template_population_keeps_each_instance_coin_and_position():
    prog = population_program(Seed, 3)
    kinds = [[(lo, hi, p.kind) for lo, hi, p, _w in rs] for rs in P.port_types(prog.fn).params]
    assert kinds == [[(0, 3, "coin")], [(0, 3, "position")]]


def test_the_same_seed_with_no_ports_on_its_population_is_refused():
    params, ret = Seed.signature()
    bare = CompositeDefinition("TemplateInstances", 1, ("N",), lambda S: (tuple((n, t) for n, t in params), ret),
                               lambda B, S, *a: B.call(Seed, *a), register=False)
    with pytest.raises(P.PortError, match="E_PORT_KIND"):
        Program(bind(bare, N=1))


def test_a_template_with_no_coin_has_a_population_without_ports():
    assert P.declared(population_program(Plain, 3).fn) is None
