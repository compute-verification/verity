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
- **Loads from**: `lean/`, the packages at `program/lean`, `../verifier/lean`, `verity/hardware/world` and
  `../../../commitments/lms/lean`; `program/lean/`, no package. The scripts, `verity.protocols.certifier.verifier`;
  `issuer_vectors.py` also `verity.commitments.crypto`, and `program/lean/scripts/window_vectors.py` the prover side,
  `verity.protocols.certifier.prover`.
- **Not here**: the certifier's host programs, which run the certifier on a host in front of a server: they are in
  `benchmarks/network_traces`.

## Overview

`lean/` is the Lake package `certifier_device` (library `CertifierDevice`). It is the executable Lean of the proxy form
of the network certifier, `verity/protocols/certifier/verifier/active.py`'s `Proxy`, as a pure state machine. It also holds
the issuer that signs each closed link-window's record.

The network certifier is an auditor-trusted device on a node's uplink (`trusted-host`). It carries the node's traffic
across the link on a public, constant-rate schedule and signs a network certificate for each sampling block of transfer
blocks it carried (the code's *link-window*). Its protocol, its Python and its README are in
`verity/protocols/certifier/`. The auditor's checks of what it signs are in `verity/protocols/`.

This package models the proxy. The relay form, `verity/protocols/certifier/verifier/warden.py`, is the one
`benchmarks/network_traces/` runs, and it has no executable Lean yet. Its analysis is in `security/`
(`security/Proofs/NetworkCertifier/Warden.lean`).

## What is in it

The package requires `certifier_program` (`verity/protocols/certifier/device/program/lean`), `network_check`
(`verity/protocols/certifier/verifier/lean`), `core_world` (`verity/hardware/world`) and `lms`
(`verity/commitments/lms/lean`). It requires no Mathlib and nothing of `security/`.

| module | what it is | its Python reference (in `verity/protocols/certifier/`) |
| --- | --- | --- |
| `CertifierDevice.Device` | `step : State → Event → State × List Action`: the proxy on one boundary link with its I/O taken out | `active.py`'s `Proxy` |
| `CertifierDevice.Driver` | `run`, which feeds `step` from a `Core.World`: its clock, the link, one server connection per stream, `publish` and the signer's leaves | `active.py`'s I/O shell (clock, tickers and sockets) |
| `CertifierDevice.Certifier` | one link's certifier: constant-rate, left-packed, FIFO and fail-closed | `grid.py`'s `Certifier` and `BlockGrid` |
| `CertifierDevice.Shaper` | the egress shaper's handing rule and the ingress bucketer | `active.py`'s `Shaper` and `Bucketer` |
| `CertifierDevice.Online` | the online prover: each window's sync chosen `lag` buckets ahead | `online.py`'s `Prover` |
| `CertifierDevice.Served` | a served link's admission rule | `served.py` |
| `CertifierDevice.Frame` | the link's frames and the server's events | `active.py`'s framing |
| `CertifierDevice.Issuer` | one network certificate per closed link-window, and each egress window's clock certificate | `certificate.py`'s `Issuer` |

Each module's docstring states its rules exactly.

## Executables

| executable | what it does |
| --- | --- |
| `certifier-device-difftest` | `CertifierDevice.Difftest`: replays generated traces through `step` and fails unless every action, in order, and the run's end results are `active.Proxy`'s |
| `certifier-issuer-difftest` | `CertifierDevice.IssuerDifftest`: runs generated calls through `Issuer.certify` and `Issuer.clock` and fails unless every result is `certificate.Issuer`'s |
| `certifier-device-smoke` | `run` in a `Core.World` on the operating system's clock, with an in-memory link and server, then certifies its records |

The two difftests are the Lean check's `runs` (`lean/lean-audit.json`). Each run's data is pinned by SHA-256 and
written by a script in `lean/scripts/`:

- `difftest_vectors.py`: the traces, and what `active.Proxy` does on each;
- `issuer_vectors.py`: the LMS keys and issuer calls, and what `certificate.Issuer` returns.

Both are compiled evaluation: tests, not proofs. The smoke run reads the wall clock, so no test runs it.

## Tests

The Python halves are in the certifier's suite, `verity/protocols/certifier/verifier/tests/`:

- `test_network_certifier_device_lean_difftest.py` checks that `difftest_vectors.py` still writes the pinned data;
- `test_network_certifier_issuer_lean_difftest.py` does the same for `issuer_vectors.py`.

When the proxy, its certifiers, its prover or the issuer change, the data changes and these tests fail. The Lean check's
`--update` on this package records the new data, and the Lean check then runs the Lean half on it.

## How to run it

~~~sh
cd verity/protocols/certifier/device/lean && lake build      # the library and the three executables, under .lake/build/bin/
lake exe certifier-device-smoke                          # the smoke run, by hand
~~~

From the repository root:

~~~sh
python3 tools/verity/lean/check.py --build verity/protocols/certifier/device/lean    # the Lean check, with both difftests
python3 tools/verity/lean/check.py --update verity/protocols/certifier/device/lean   # record new difftest data
uv run tools/verity/check/suites.py verity/protocols/certifier             # the certifier's suite (Python halves)
~~~
