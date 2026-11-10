# The network certifier's device (`verity/protocols/certifier/device`)

## Interface

The network certifier in executable Lean: the certifier program, and the device around it. This directory has no
Python package and no console script.

- **`program/lean/`**, the certifier program (the executable forms of the certifier's definitions): the Lake package
  `certifier_program`, library `NetworkCertifier`, Lean core only. The packages at `../verifier/lean`, `lean/` and
  `security/Proofs` require it by path. Its executables (`lake exe`): `certifier-difftest`,
  `certifier-charge-difftest` and `certifier-window-difftest`, each `INSTANCES PYTHON N`.
- **`lean/`**, the device (the certifier's proxy as a state machine, the driver that runs it and the issuer that signs
  its records): the Lake package `certifier_device`, library `CertifierDevice`, which `security/Properties` requires
  by path. Its executables: `certifier-device-difftest` and `certifier-issuer-difftest`, each `INSTANCES PYTHON N`,
  and `certifier-device-smoke`, which takes no argument and which no test runs, since it reads the wall clock.
- **Scripts**: each package's `scripts/` writes its difftests' instances and the Python's results on them,
  `python scripts/NAME.py CHECKOUT --out DIR [--count N]`, as the package's `lean-audit.json` runs them (`runs`). No
  code outside this directory imports them; their tests are in `../verifier/tests`, which loads them by path and
  runs the executables.
- **Documents**: none but those instance and result files, which the difftest executables read.
- **Loads from**: `program/lean/`, nothing. `lean/`, the packages at `program/lean`, `../verifier/lean`,
  `verity/hardware/world` and `verity/commitments/lms/lean`. The scripts, `verity.protocols.certifier.verifier`;
  `window_vectors.py` also `verity.protocols.certifier.prover`, and `issuer_vectors.py` `verity.commitments.crypto`.
- **Not here yet**: the certifier's host programs, which run the certifier on a host in front of a server. They are
  in `benchmarks/network_traces`.
