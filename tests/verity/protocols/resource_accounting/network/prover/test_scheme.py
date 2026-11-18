"""The network attester's declaration as a resource scheme (`verity.protocols.resource_accounting.network.scheme`)."""

from __future__ import annotations

from verity import hooks as H
from verity.circuits.defs import REGISTRY, CompositeDefinition
from verity.protocols.compose import schemes
from verity.protocols.interface.charge import charge_chain, window_charge
from verity.protocols.resource_accounting.network import scheme as NS
from verity.protocols.resource_accounting.network.prover.response_tie import ResponseTieStop


def test_registered_and_found_by_name() -> None:
    assert schemes.get("network-attester") is NS.NETWORK_ATTESTER
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


def test_ports_are_its_blocks() -> None:
    ids = {b.definition for b in NS.NETWORK_ATTESTER.blocks}
    assert {(p.direction, p.definition) for p in NS.NETWORK_ATTESTER.ports} == {
        ("input", "IngressScanStart_v1"), ("output", "EgressBucket_v1")}
    assert {p.definition for p in NS.NETWORK_ATTESTER.ports} <= ids


class _Exchange:
    def __init__(self) -> None:
        self.filed: list[tuple[str, object]] = []

    def request(self, name: str, request: object) -> None:
        self.filed.append((name, request))

    def subscribe(self, name: str, hook: str, fn: object):
        self.filed.append((name, hook))
        return lambda: None


def test_attach_files_the_id_relay_and_installs_admission_when_configured() -> None:
    ex = _Exchange()
    NS.NETWORK_ATTESTER.attach(ex, {})
    assert ex.filed == [("network-attester", NS.START_ID_RELAY)]
    assert NS.NETWORK_ATTESTER.requests({}) == (NS.START_ID_RELAY,)
    assert set(NS.NETWORK_ATTESTER.processes) == {"id-relay"}
    ex = _Exchange()
    NS.NETWORK_ATTESTER.attach(ex, {"admission": {"order": "release-order"}, "submit": print, "abort": print})
    assert ex.filed == [("network-attester", NS.START_ID_RELAY)]
