"""The messages between the workload and the prover are one list: each protocol declares its own in its
`prover/workload.py`, and every message is declared once."""

from __future__ import annotations

import importlib

MODULES = {
    "spot": "verity.protocols.proof.spot.prover.workload",
    "pouw": "verity.protocols.accounting.compute.pouw.prover.workload",
    "pous": "verity.protocols.accounting.memory.pous.prover.workload",
    "certifier": "verity.protocols.accounting.network.prover.workload",
}


def test_every_message_is_declared_once():
    seen: dict[str, str] = {}
    for p, m in MODULES.items():
        for s in importlib.import_module(m).MESSAGES:
            assert s.type not in seen, f"{s.type} is declared by both {MODULES[seen[s.type]]} and {m}"
            seen[s.type] = p
