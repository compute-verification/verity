"""The network attester's declaration as a resource scheme (`verity.protocols.resource_accounting.network.scheme`)."""

from __future__ import annotations

from verity import hooks as H
from verity.circuits.defs import REGISTRY, CompositeDefinition
from verity.protocols.interface.charge import charge_chain, window_charge
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.network import scheme as NS
from verity.protocols.resource_accounting.network.prover.response_tie import ResponseTieStop


def test_registered_and_found_by_name() -> None:
    assert S.get("certifier") is NS.NETWORK_ATTESTER
    assert NS.NETWORK_ATTESTER.unit == "bits"


def test_hooks_are_a_served_runs() -> None:
    assert set(NS.NETWORK_ATTESTER.hooks) <= set(H.HOOKS)


def test_every_block_is_a_registered_definition() -> None:
    ids = [b.definition for b in NS.NETWORK_ATTESTER.blocks]
    assert len(set(ids)) == len(ids)
    for d, tid in zip(NS.DEFINITIONS, ids, strict=True):
        assert isinstance(d, CompositeDefinition) and REGISTRY.defs[tid] is d


def test_cost_is_output_bits() -> None:
    cost = NS.NETWORK_ATTESTER.blocks[NS.DEFINITIONS.index(ResponseTieStop)].cost
    assert cost({"O": 4}) == 176


def test_the_charge_circuits_are_the_interfaces() -> None:
    ids = {b.definition for b in NS.NETWORK_ATTESTER.blocks}
    assert ids.isdisjoint({charge_chain.ChargeLink.id, window_charge.ChargeCount.id, window_charge.ChargeLink.id})
    assert [c.name for c in NS.NETWORK_ATTESTER.charges][0] == "advice"
