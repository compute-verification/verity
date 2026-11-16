"""zk-drivers: the command line of the network certifier's zero-knowledge drivers, which live in the certifier's prover
(`verity/protocols/resource_accounting/network/prover/`) and take the proof system's benchmark drivers as one
`proof_drivers.ProofDrivers`. This builds it from SPOT's template audit, challenger session and development challenger
(`a0`, `a5_coins`, `local_challenger`),
`benchmarks/one_stage/` (`a7`, `a8`, `prove_subprotocol`), Flock's distribution (`verity_flock.circuit`,
`verity_flock.class_statement`), Flock's proof-format reader (`redigest`) and the chain's reader (`network_certifier_chain`),
installs it, and runs the driver's `main`:

    python benchmarks/network_traces/zk_drivers.py scan_zk prove --bin flock-circuit --lean verity-verify-proof --out DIR
    python benchmarks/network_traces/zk_drivers.py egress_zk|network_certifier_zk|n2_zk|n2_tie_zk ARGS...
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DRIVERS = ("scan_zk", "egress_zk", "network_certifier_zk", "n2_zk", "n2_tie_zk")


def proof_drivers() -> SimpleNamespace:
    for d in (ROOT / "benchmarks/one_stage", ROOT / "verity/protocols/proofs/flock/prover/agreement"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    import a7
    import a8
    import prove_subprotocol
    import redigest
    from verity_flock import circuit, class_statement
    from verity.protocols.proofs.spot.verifier.challenger import challenger_session as a5_coins
    from verity.protocols.proofs.spot.prover.audit import local_challenger
    import a0
    from benchmarks.network_traces import network_certifier_chain

    return SimpleNamespace(a0=a0, a5_coins=a5_coins, local_challenger=local_challenger, a7=a7, a8=a8, prove_subprotocol=prove_subprotocol, circuit=circuit, class_statement=class_statement,
                           redigest=redigest, network_certifier_chain=network_certifier_chain)


def install() -> None:
    from verity.protocols.resource_accounting.network.prover import proof_drivers as PD

    PD.install(proof_drivers())


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in DRIVERS:
        raise SystemExit(f"usage: zk_drivers.py {{{'|'.join(DRIVERS)}}} ARGS...")
    install()
    driver = importlib.import_module(f"verity.protocols.resource_accounting.network.prover.{argv[0]}")
    return driver.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
