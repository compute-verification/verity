"""The network attester as a resource scheme (`verity.protocols.resource_accounting.scheme`): what it bounds is egress,
the bits that leave beyond what the registered circuit explains."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from verity.circuits.defs import bind
from verity.protocols.resource_accounting.network.prover.egress import EgressBucket
from verity.protocols.resource_accounting.network.prover.ingress_scan import IngressScanBucket, IngressScanStart
from verity.protocols.resource_accounting.network.prover.lengths_row import LengthsRow
from verity.protocols.resource_accounting.network.prover.response_tie import ResponseTieNoStop, ResponseTieStop
from verity.protocols.resource_accounting.network.verifier.io_request import IoTieRequest
from verity.protocols.resource_accounting.network.verifier.io_tie import IoTieBucket, IoTieContent, IoTieRoute
from verity.protocols.resource_accounting.scheme import Block, Charge, Order, Port, Scheme, register
from verity import hooks as H
from verity.protocols.resource_accounting.network.prover import id_relay
from verity.protocols.resource_accounting.network.prover.admission import Admitter

DEFINITIONS = (IngressScanStart, IngressScanBucket, EgressBucket, IoTieRoute, IoTieContent, IoTieBucket, IoTieRequest,
               ResponseTieStop, ResponseTieNoStop, LengthsRow)


def attach(exchange: Any, ctx: Mapping[str, Any]) -> Any:
    """The scheme on a served run's ``exchange``. With ``admit`` in ``ctx``, its admission (`prover.admission.Admitter`,
    the configuration ``admit`` with ``ctx``'s ``source``, ``submit``, ``abort``, ``take``, ``pull``, ``log``,
    ``clock`` and ``schedule``), installed and returned; without, the id relay beside the workload (`START_ID_RELAY`)."""
    if "admit" not in ctx:
        for r in requests(ctx):
            exchange.request(NETWORK_ATTESTER.name, r)
        return None
    kw = {k: ctx[k] for k in ("source", "submit", "abort", "take", "pull", "log", "clock", "schedule") if k in ctx}
    return Admitter(ctx["admit"], **kw).install(exchange, NETWORK_ATTESTER.name)


def output_bits(definition: Any) -> Callable[[Mapping[str, int]], int]:
    """A block's cost: its return bits at the public shape (its statics), all of which a wrong block may set."""
    return lambda shape: sum(bind(definition, **shape).signature()[1].leaf_widths())


#: the scheme's request that starts the id relay beside the workload
START_ID_RELAY = H.StartBeside("id-relay", id_relay.READERS)


def requests(ctx: Mapping[str, Any]) -> tuple[Any, ...]:
    """The id relay beside the workload where ``ctx`` configures no admission (``admit``), else nothing."""
    return () if "admit" in ctx else (START_ID_RELAY,)


NETWORK_ATTESTER = register(Scheme(
    name="network-attester",
    protocol="verity/certifier/v0",
    resource="egress",
    blocks=tuple(Block(d.id, output_bits(d)) for d in DEFINITIONS),
    assumptions=("trusted-device",),
    log=(Order("window", "sample"), Order("start", "first-tick")),
    charges=(
        Charge("advice", "scans[].free"),
        Charge("jitter", "scans[].jitter"),
        Charge("schedules", "schedules"),
        Charge("cells", "cells"),
        Charge("start", "start"),
        Charge("tied-entries", "scans[].faulty_entries"),
    ),
    hooks=("declared", "steps"),
    requires={"one-labelling": "the network attester's rows (`verifier.network_attester_rows`)",
              "one-design": "each scan's own sample at its δ (`prover.scan_zk`)"},
    ports=(Port("scan-state", "input", IngressScanStart.id, "sin"), Port("released", "output", EgressBucket.id, "o1")),
    requests=requests,
    attach=attach,
    processes={"id-relay": id_relay.process},
))

__all__ = ["DEFINITIONS", "NETWORK_ATTESTER", "START_ID_RELAY", "attach", "output_bits", "requests"]
