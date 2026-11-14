# The network certifier's device (`verity/protocols/accounting/certifier/device`)

## Interface

The certifier program in executable Lean: the executable forms of the certifier's definitions. This directory is
`program/lean/` and this file: it has no Python package and no console script.

- **Lean**: `program/lean/` is the Lake package `certifier_program`, whose library `NetworkCertifier` the packages at
  `../verifier/lean` and `security/Proofs` require by path.
- **Commands** (`lake exe`, in `program/lean/`): `certifier-difftest`, `certifier-charge-difftest` and
  `certifier-window-difftest`, each `INSTANCES PYTHON N`.
- **Scripts**: `program/lean/scripts/` writes the difftests' instances and the Python's results on them,
  `python scripts/NAME.py CHECKOUT --out DIR [--count N]`. The Lean check runs each script, then its difftest on what
  it wrote, as `program/lean/lean-audit.json` says (`runs`). No code outside this directory imports the scripts; their
  tests are in `../verifier/tests`, which load them by path and hold what they write to those files' pins.
- **Documents**: none but those instance and result files, which the difftest executables read.
- **Loads from**: `program/lean/`, no package. The scripts, `verity.protocols.accounting.certifier.verifier`;
  `window_vectors.py` also the prover side, `verity.protocols.accounting.certifier.prover`.
- **Not here**: the certifier's host programs, which run the certifier on a host in front of a server: they are in
  `benchmarks/network_traces`.
