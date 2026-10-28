# The network certifier's executable Lean (`verity/core/devices/certifier`)

`lean/` is the Lake package `certifier_device` (library `CertifierDevice`). It is the executable Lean of the proxy form
of the network certifier, `verity/core/protocols/certifier/active.py`'s `Proxy`, as a pure state machine. It also holds
the issuer that signs each closed link-window's record.

The network certifier is an auditor-trusted device on a node's uplink (`trusted-host`). It carries the node's traffic
across the link on a public, constant-rate schedule and signs a network certificate for each sampling block of transfer
blocks it carried (the code's *link-window*). Its protocol, its Python and its README are in
`verity/core/protocols/certifier/`. The auditor's checks of what it signs are in `verity/core/protocols/network/`.

This package models the proxy. The relay form, `verity/core/protocols/certifier/warden.py`, is the one
`benchmarks/network_traces/` runs, and it has no executable Lean yet. Its analysis is in `Security/`
(`Security/Proofs/NetworkCertifier/Warden.lean`).

## What is in it

The package requires `certifier_program` (`verity/core/protocols/certifier/lean`), `network_check`
(`verity/core/protocols/network/lean`), `core_world` (`verity/core/primitives/world`) and `lms`
(`verity/core/primitives/lms/lean`). It requires no Mathlib and nothing of `Security/`.

| module | what it is | its Python reference (in `verity/core/protocols/certifier/`) |
| --- | --- | --- |
| `CertifierDevice.Device` | `step : State → Event → State × List Action`: the proxy on one boundary link with its I/O taken out | `active.py`'s `Proxy` |
| `CertifierDevice.Driver` | `run`, which feeds `step` from a `Core.World`: its clock, the link, one server connection per stream, `publish` and the signer's leaves | `active.py`'s I/O shell (clock, tickers and sockets) |
| `CertifierDevice.Certifier` | one link's certifier: constant-rate, left-packed, FIFO and fail-closed | `grid.py`'s `Certifier` and `Record` |
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

The Python halves are in the certifier's suite, `verity/core/protocols/certifier/tests/`:

- `test_network_certifier_device_lean_difftest.py` checks that `difftest_vectors.py` still writes the pinned data;
- `test_network_certifier_issuer_lean_difftest.py` does the same for `issuer_vectors.py`.

When the proxy, its certifiers, its prover or the issuer change, the data changes and these tests fail. The Lean check's
`--update` on this package records the new data, and the Lean check then runs the Lean half on it.

## How to run it

~~~sh
cd verity/core/devices/certifier/lean && lake build      # the library and the three executables, under .lake/build/bin/
lake exe certifier-device-smoke                          # the smoke run, by hand
~~~

From the repository root:

~~~sh
python3 tools/verity/lean/check.py --build verity/core/devices/certifier/lean    # the Lean check, with both difftests
python3 tools/verity/lean/check.py --update verity/core/devices/certifier/lean   # record new difftest data
uv run tools/verity/check/suites.py verity/core/protocols/certifier             # the certifier's suite (Python halves)
~~~
