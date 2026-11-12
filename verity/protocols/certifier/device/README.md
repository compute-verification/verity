# The network certifier's device (`verity/protocols/certifier/device`)

## Interface

The network certifier's device in executable Lean: the certifier's proxy as a state machine, the driver that runs it
and the issuer that signs its records. This directory is `lean/`, the certifier program in `program/lean/` and this
file: it has no Python package and no console script.

- **Lean**: `lean/` is the Lake package `certifier_device`, whose library `CertifierDevice` `security/Properties`
  requires by path. `program/lean/` is the certifier program, the executable forms of the certifier's definitions:
  the Lake package `certifier_program`, whose library `NetworkCertifier` the packages at `lean`, `../verifier/lean`
  and `security/Proofs` require by path.
- **Commands** (`lake exe`, in `lean/`): `certifier-device-difftest` and `certifier-issuer-difftest`, each
  `INSTANCES PYTHON N`, and `certifier-device-smoke`, which takes no argument and which no test runs, since it reads
  the wall clock. In `program/lean/`: `certifier-difftest`, `certifier-charge-difftest` and
  `certifier-window-difftest`, each `INSTANCES PYTHON N`.
- **Scripts**: `lean/scripts/` writes the difftests' instances and the Python's results on them,
  `python scripts/NAME.py CHECKOUT --out DIR [--count N]`. The Lean check runs each script, then its difftest on what
  it wrote, as `lean/lean-audit.json` says (`runs`). `program/lean/scripts/` does the same for the program's
  difftests, as `program/lean/lean-audit.json` says. No code outside this directory imports the scripts; their tests
  are in `../verifier/tests`, which load them by path and hold what they write to those files' pins.
- **Documents**: none but those instance and result files, which the difftest executables read.
- **Loads from**: `lean/`, the packages at `program/lean`, `../verifier/lean`, `../../../hardware/world` and
  `../../../commitments/lms/lean`; `program/lean/`, no package. The scripts, `verity.protocols.certifier.verifier`;
  `issuer_vectors.py` also `verity.commitments.crypto`, and `program/lean/scripts/window_vectors.py` the prover side,
  `verity.protocols.certifier.prover`.
- **Not here**: the certifier's host programs, which run the certifier on a host in front of a server: they are in
  `benchmarks/network_traces`.
