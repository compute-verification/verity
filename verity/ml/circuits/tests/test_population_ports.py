"""`population_program` of a unit (a template that takes a coin and a position and returns a sink, a salted tile such as
PoUW's `PouwH1Tile`): the population keeps each instance's coin, position and sink, so the port rules accept it as a unit.
A template that takes a position alone (a key count) keeps it too, and a mixed population keeps each member's ports. A
seed's population (a coin and no sink) and any other template's declare no ports."""
from __future__ import annotations

import pytest

from verity.ml.circuits import ports as P
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.population import mixed_population_program, population_program
from verity.ml.circuits.types import Value

V32 = Value(32)
TAdd = PrimitiveDefinition("TPopAdd32", 1, (("a", V32), ("b", V32)), V32, lambda a, b: (a + b) & 0xFFFFFFFF,
                           register=False)
TKeyFold = PrimitiveDefinition("TPopKeyFold32", 1, (("x", V32), ("key", V32)), V32,
                               lambda x, key: ((x ^ key) * 0x9E3779B1) & 0xFFFFFFFF, register=False,
                               ports=P.Ports(params={"key": P.In(kind="coin")}, ret=P.Out(sink=True)))
#: a unit: a coin and a position in, its receipt (a fold keyed by the coin) a sink
Unit = bind(CompositeDefinition("TPopUnit", 1, (), lambda S: ((("salt", V32), ("position", V32)), V32),
                                lambda B, S, salt, position: B.call(TKeyFold, position, salt), register=False,
                                ports=lambda S: P.Ports(params={"salt": P.In(kind="coin"),
                                                                "position": P.In(kind="position")},
                                                        ret=P.Out(sink=True))))
#: a seed: a coin and a position in, no sink out, so no rule follows the coin into its body
Seed = bind(CompositeDefinition("TPopSeed", 1, (), lambda S: ((("salt", V32), ("position", V32)), V32),
                                lambda B, S, salt, position: B.call(TAdd, salt, position), register=False,
                                ports=lambda S: P.Ports(params={"salt": P.In(kind="coin"),
                                                                "position": P.In(kind="position")})))
Plain = bind(CompositeDefinition("TPopPlain", 1, (), lambda S: ((("a", V32), ("b", V32)), V32),
                                 lambda B, S, a, b: B.call(TAdd, a, b), register=False))
#: a position alone in, no coin and no sink, as `ops.gemm.PvRowPad`'s key count `n`
Counted = bind(CompositeDefinition("TPopCounted", 1, (), lambda S: ((("x", V32), ("n", V32)), V32),
                                   lambda B, S, x, n: B.call(TAdd, x, n), register=False,
                                   ports=lambda S: P.Ports(params={"n": P.In(kind="position")})))


def _kinds(pt):
    return [[(lo, hi, p.kind) for lo, hi, p, _w in rs if p.kind in ("coin", "position")] for rs in pt.params]


def test_a_unit_population_keeps_each_instance_coin_position_and_sink():
    pt = P.port_types(population_program(Unit, 3).fn)
    assert [[(lo, hi, p.kind) for lo, hi, p, _w in rs] for rs in pt.params] == [[(0, 3, "coin")], [(0, 3, "position")]]
    assert [(lo, hi, o.sink) for lo, hi, o, _w in pt.ret] == [(0, 3, True)]


def test_a_seed_population_declares_no_ports_and_is_refused():
    with pytest.raises(P.PortError, match="E_PORT_KIND"):
        population_program(Seed, 1)


def test_a_template_with_no_coin_has_a_population_without_ports():
    assert P.declared(population_program(Plain, 3).fn) is None


def test_a_population_of_a_position_alone_keeps_each_instance_position():
    pt = P.port_types(population_program(Counted, 3).fn)
    assert _kinds(pt) == [[], [(0, 3, "position")]]
    assert not any(o.sink for _lo, _hi, o, _w in pt.ret)


def test_a_mixed_population_keeps_each_member_ports_at_its_offset():
    pt = P.port_types(mixed_population_program([(Plain, 2), (Counted, 3), (Unit, 2)]).fn)
    assert _kinds(pt) == [[], [], [], [(0, 3, "position")], [(0, 2, "coin")], [(0, 2, "position")]]
    assert [(lo, hi) for lo, hi, o, _w in pt.ret if o.sink] == [(5, 7)]
    assert P.declared(mixed_population_program([(Plain, 2), (Plain, 1)]).fn) is None
