"""The inference-only rate rule's weighing held to the gate table: a call weighs nothing in a block's allowance of fresh bits
(`inference_only_reference.weighs_nothing`, the Lean's `InferenceOnlyCheck.weighsNothing`, which the difftest holds it to)
exactly where its primitive holds no gate. The rule names those primitives, a constant by having no parameter and the
rest by id; the gate table (`verity.circuits.ops.boolean.versions.gate_table`) prices each primitive through its Boolean
version. A primitive the table prices at nothing that the rule weighs would buy fresh bits with no gate, and one of
`GATELESS` the table prices at a gate would weigh nothing that computes.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[7] / "verity" / "protocols" / "workload_classification" / "training_detection" / "inference_only" / "verifier" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load("inference_only_reference")

#: the modules whose import registers every primitive a served circuit calls: the catalog, the served Definitions, and
#: PoUW's segment pieces
REGISTRIES = ("verity.circuits.ops", "verity_vllm.definitions")
MODULES = ("verity.protocols.resource_accounting.compute.pouw.prover.circuit.segment",)


def _primitives() -> list:
    from verity.circuits.defs import REGISTRY, PrimitiveDefinition
    from verity.circuits.population import bit_of, word_of
    for name in REGISTRIES:
        pkg = importlib.import_module(name)
        for m in pkgutil.walk_packages(pkg.__path__, name + "."):
            importlib.import_module(m.name)
    for name in MODULES:
        importlib.import_module(name)
    wiring = [bit_of(16, 0), bit_of(16, 15), bit_of(32, 7), word_of(16), word_of(32)]
    return [*(d for d in REGISTRY.defs.values() if isinstance(d, PrimitiveDefinition)), *wiring]


@pytest.mark.slow
def test_a_primitive_weighs_nothing_exactly_where_the_gate_table_prices_it_at_nothing():
    """Every registered primitive the gate table prices at no Boolean gate weighs nothing, and every one of `GATELESS`
    is registered and priced at nothing. About 40 s, importing the registries and building each primitive's Boolean
    version; a smaller set of primitives would not catch a newly registered one the rule's names miss."""
    from verity.circuits.ops.boolean.versions import gate_table
    prims = _primitives()
    table, _ = gate_table(prims)
    by_id = {p.id: p for p in prims}
    priced_at_nothing = {k for k, v in table.items() if v == 0}
    assert sorted(k for k in priced_at_nothing if not reference.weighs_nothing(by_id[k])) == []
    assert sorted(reference.GATELESS - priced_at_nothing) == []
    assert {"BitOf16x0_v1", "WordOf32_v1"} <= priced_at_nothing
