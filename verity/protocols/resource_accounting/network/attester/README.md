# The network attester's device (`verity/protocols/resource_accounting/network/attester`)

## Interface

The network attester's device: its party on the deployment layer, what it needs of that layer's runtime, its end of
the challenges' link, its host programs, and the network attester's program in executable Lean (the executable forms of the
network attester's definitions). It has no console script.

- **Python**: the package `verity.protocols.resource_accounting.network.attester`. `device_party.NetworkAttester` is the
  manifest's `network-certifier` (`../verifier/parties.py`) as a party, and `device_party.PROTOCOL` the manifest with
  it; the parts of its life the runtime has no form for yet raise `NotImplementedError` naming their goal.
  `device_needs` holds those needs, one proposed type each. `coin_link.CoinLink` is the link for an execution phase's challenges,
  which `verity/protocols/compose/verifier/live_challenger/coin_link.py` runs (`../verifier/protocol/coin-link.md`).
  Their tests are in `../verifier/tests`. The relay (`../verifier/network_attester.py`) and its pacer stay in `../verifier`,
  whose modules read the schedule and the cells' layout from them.
- **Host programs**, which run the network attester on a host in front of a server: `network_attester_serve.py` (the
  relay in front of any TCP server, and the offline check of what it published), `load_relay.py` (the relay of the
  weights' load and its commit), `connection_rows_run.py` (a run's lengths and content rows), with `runner_cli.py` and
  `runner_parts.py`, which they start a run with; `stall_probe.py` (what blocks a tick) and `certifier_key.py` (a
  device's LMS key on disk). Their tests are in `tests/`.

- **Lean**: `program/lean/` is the Lake package `certifier_program`, whose library `NetworkAttester` the packages at
  `../verifier/lean` and `security/Proofs` require by path.
- **Commands** (`lake exe`, in `program/lean/`): `certifier-difftest`, `certifier-charge-difftest` and
  `certifier-window-difftest`, each `INSTANCES PYTHON N`.
- **Scripts**: `program/lean/scripts/` writes the difftests' instances and the Python's results on them,
  `python scripts/NAME.py CHECKOUT --out DIR [--count N]`. The Lean check runs each script, then its difftest on what
  it wrote, as `program/lean/lean-audit.json` says (`runs`). No code outside this directory imports the scripts; their
  tests are in `../verifier/tests`, which load them by path and hold what they write to those files' pins.
- **Documents**: none but those instance and result files, which the difftest executables read.
- **Loads from**: `program/lean/`, no package. The Python, `verity.party`, `verity.channel`, `verity.runtime`,
  `verity.commitments`, `verity.protocols.proofs.spot.verifier.challenger` and `../verifier`. The scripts, `verity.protocols.resource_accounting.network.verifier`;
  `window_vectors.py` also the prover side, `verity.protocols.resource_accounting.network.prover`.
- **Not here**: the programs that drive a run's proofs and chains (`chain_record.py`, `network_attester_chain.py` and the
  `*_zk.sh` jobs), which are in `benchmarks/network_traces`.
