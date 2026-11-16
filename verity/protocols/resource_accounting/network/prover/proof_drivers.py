"""What network certification's zero-knowledge drivers take from code `verity` does not load.

The drivers (`scan_zk`, `egress_zk`, `network_attester_zk`, `n2_zk`, `n2_tie_zk`, `leak_scan`) stage, prove and verify through the
one-stage audit's benchmark drivers under `benchmarks/one_stage/`, Flock's own distribution and Flock's proof-format
reader, and read a network attester's run through the chain's reader under `benchmarks/network_traces/`. `verity` loads
none of them, so they come in as one `ProofDrivers`, installed by the command line that runs a driver
(`benchmarks/network_traces/zk_drivers.py`) before the driver reads it:

    proof_drivers.install(drivers)   # once, by the caller
    proof_drivers.drivers().a8.session(...)
"""
from __future__ import annotations

from typing import Any, Protocol


class ProofDrivers(Protocol):
    #: the one-stage audit's driver: ``SCHEME``, ``run``, ``lean_verify``, ``proved_units``, ``challenger_sample``, ...
    a0: Any
    #: the requirement audit's driver: ``M_MAX``, ``statement_checks``, ``frame_ports``, ``opens``, ``public_bcs``
    a7: Any
    #: the ZK server's session driver: ``session``, ``received``, ``PIECE``, ``needles``, ``find``, ``cost``
    a8: Any
    #: the auditor's challenger's sessions: ``start``, ``RankCoins``, ``evidence``, ``negatives``, ``Replay``
    a5_coins: Any
    #: the development challenger on this machine, which ``start`` takes for ``--challenger local``: ``Local``
    local_challenger: Any
    #: the proof subprotocol: ``frame_ports`` over a statement's tables
    prove_subprotocol: Any
    #: Flock's circuit writer (``verity_flock.circuit``)
    circuit: Any
    #: Flock's unit stager (``verity_flock.class_statement``)
    class_statement: Any
    #: Flock's proof-format reader (``redigest``: ``layout``, and ``tc.R``, the reads it records)
    redigest: Any
    #: the chain's reader of a network attester's run: ``pairs``, ``link``
    network_attester_chain: Any


_installed: ProofDrivers | None = None


def install(drivers: ProofDrivers) -> None:
    global _installed
    _installed = drivers


def drivers() -> ProofDrivers:
    if _installed is None:
        raise RuntimeError("no proof drivers installed: run the driver through benchmarks/network_traces/zk_drivers.py, "
                           "or call proof_drivers.install first")
    return _installed
