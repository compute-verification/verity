# The network certifier's device (`verity/protocols/accounting/certifier/device`)

## Interface

The network certifier's device: its party on the deployment layer, what it needs of that layer's runtime, its end of
the challenges' link, and the certifier program in executable Lean (the executable forms of the certifier's
definitions). It has no console script.

- **Python**: the package `verity.protocols.accounting.certifier.device`. `device_party.NetworkCertifier` is the
  manifest's `network-certifier` (`../verifier/parties.py`) as a party, and `device_party.PROTOCOL` the manifest with
  it; the parts of its life the runtime has no form for yet raise `NotImplementedError` naming their goal.
  `device_needs` holds those needs, one proposed type each. `coin_link.CoinLink` is the link for a window's challenges,
  which `verity/protocols/compose/device/live_challenger/coin_link.py` runs (`../verifier/protocol/coin-link.md`).
  Their tests are in `../verifier/tests`. The relay (`../verifier/warden.py`) and its pacer stay in `../verifier`,
  whose modules read the schedule and the cells' layout from them.

- **Lean**: `program/lean/` is the Lake package `certifier_program`, whose library `NetworkCertifier` the packages at
  `../verifier/lean` and `security/Proofs` require by path.
- **Commands** (`lake exe`, in `program/lean/`): `certifier-difftest`, `certifier-charge-difftest` and
  `certifier-window-difftest`, each `INSTANCES PYTHON N`.
- **Scripts**: `program/lean/scripts/` writes the difftests' instances and the Python's results on them,
  `python scripts/NAME.py CHECKOUT --out DIR [--count N]`. The Lean check runs each script, then its difftest on what
  it wrote, as `program/lean/lean-audit.json` says (`runs`). No code outside this directory imports the scripts; their
  tests are in `../verifier/tests`, which load them by path and hold what they write to those files' pins.
- **Documents**: none but those instance and result files, which the difftest executables read.
- **Loads from**: `program/lean/`, no package. The Python, `verity.party`, `verity.channel`, `verity.runtime`,
  `verity.commitments`, `verity.protocols.proofs.spot.device` and the certifier's `verifier/`. The scripts, `verity.protocols.accounting.certifier.verifier`;
  `window_vectors.py` also the prover side, `verity.protocols.accounting.certifier.prover`.
- **Not here**: the certifier's host programs, which run the certifier on a host in front of a server: they are in
  `benchmarks/network_traces`.
