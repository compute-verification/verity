"""zk-drivers: the proof system's benchmark drivers as one `proof_drivers.ProofDrivers`, which network attestation's
prover takes (`verity/protocols/accounting/network/prover/`). This builds it from SPOT's template audit, challenger session and development challenger
(`a0`, `a5_coins`, `local_challenger`),
`benchmarks/one_stage/` (`a7`, `a8`, `prove_subprotocol`), Flock's distribution (`verity_flock.circuit`,
`verity_flock.class_statement`), Flock's proof-format reader (`redigest`) and the chain's reader (`network_attester_chain`),
and installs it (`install`), which `leak_scan` reads its proof-format reader from.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def proof_drivers() -> SimpleNamespace:
    for d in (ROOT / "benchmarks/one_stage", ROOT / "verity/protocols/proof/flock/prover/agreement"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    import a7
    import a8
    import prove_subprotocol
    import redigest
    from verity_flock import circuit, class_statement
    from verity.protocols.proof.spot.verifier.challenger import challenger_session as a5_coins
    from verity.protocols.proof.spot.prover.audit import local_challenger
    import a0
    from benchmarks.network_traces import network_attester_chain

    return SimpleNamespace(a0=a0, a5_coins=a5_coins, local_challenger=local_challenger, a7=a7, a8=a8, prove_subprotocol=prove_subprotocol, circuit=circuit, class_statement=class_statement,
                           redigest=redigest, network_attester_chain=network_attester_chain)


def install() -> None:
    from verity.protocols.accounting.network.prover import proof_drivers as PD

    PD.install(proof_drivers())
