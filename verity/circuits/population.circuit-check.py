"""circuit-check's bindings of `verity.circuits.population`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`). Populations of other units' templates are bound beside them: the GEMMs' in
`../definitions/ops/boolean/gemm.circuit-check.py`, a coin-taking tile's (PoUW's H1) in `pouw/circuit/h1tile.circuit-check.py`. The v0
programs are not bound: their ids name the counts alone, so one pin key would stand for every template's
(`tests/test_population_recorded.py` holds them)."""
from __future__ import annotations

from typing import Any


def _population_roots() -> list[Any]:
    """`TemplateInstances_v1` of the sharded plan unit (S = 32, two gaps) at N = 2, and `TemplatePopulations_v1` of it and the
    plan unit (P = 4, 16 registered blocks) at counts 1 and 2, without values and with two (a round's A table and input-ids
    table, in 16-bit words, as a window registers them): populations of templates with no coin, whose programs declare no
    ports."""
    from verity.circuits.defs import bind
    from verity.circuits.population import mixed_population_program, population_program
    from verity.definitions.circuits.requirement_units import PlanMatch, PlanShard

    shard, match = bind(PlanShard, S=32, GAPS=2), bind(PlanMatch, P=4, NR=16, V=16, G=2, TB=8, RPB=2)
    values = (("row-table/0/k2048", 6, 16), ("input-ids/0", 4, 16))
    return [population_program(shard, 2).fn, mixed_population_program([(shard, 1), (match, 2)]).fn,
            mixed_population_program([(shard, 1), (match, 2)], values=values).fn]
