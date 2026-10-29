# Network certification's Lean checks (`verity/core/protocols/network`)

`NetworkCheck` is the Lean verifier of the network certifiers' records and of the certification period's check, with
the suite that holds each check to its Python reference in `verity/core/protocols/certifier/`.

A network certifier is an auditor-trusted device on a node's uplink. For each sampling block of transfer blocks it
carries on a link (the code's *link-window*, over one sampling period) it signs a network certificate. A certification
period is m consecutive link-windows on one link under one signed root (`NetworkCheck.Period`). The identifiers keep the
names `window` and `LinkWindow` until a format version (the Glossary's older names).

- `verity/core/protocols/certifier/`: the network certifier itself (its relay, records, certificates and the Python
  audit), and in its README the protocol.
- `nci/`: the inference-only policy's check, a separate package with its own README.
- `Security/Proofs/NetworkCertifier/README.md`: the guarantees and proofs about the certifier's timing channel, which
  bound `K` (below).

## What runs

`lean/` is the Lake package `network_check` (library `NetworkCheck`). It requires five packages by path and no
Mathlib: C-Flock's verifier (`verity/core/service/flock`), the core protocol interface
(`verity/core/protocols/interface`), the certifier program (`verity/core/protocols/certifier/lean`), the LMS port
(`verity/core/primitives/lms/lean`) and core's identity hash (`verity/core/primitives/identity`).

Each check is a `Core.Protocol`, so each executable has the interface's three commands: `id`,
`check PARAMS COMMITMENT EVIDENCE` and `serve` (`verity/core/protocols/interface/README.md`). The module docstring of
each check states its parameters, commitment and evidence byte for byte.

The Python references below are in `verity/core/protocols/certifier/` except `tests/period_case.py`, which is this
package's.

| executable | protocol id | module | what it decides | its Python reference |
| --- | --- | --- | --- | --- |
| `network-check` | `verity/network-certificate/v4` | `NetworkCheck.Check` | one spatial unit's certified link-windows: every egress link in order, then every ingress link, under the certifier's LMS key in its session | `audit.py`'s `Audit` (`check_link`, `check_ingress_link`) |
| `network-records` | `verity/network-records/v3` | `NetworkCheck.Records` | the same audit over raw, committed and certified records, with the egress grid compared or not | `audit.py`'s `Audit` |
| `network-period` | `verity/certifier/v0` | `NetworkCheck.Period` | a certification period, the onsite composite protocol's communication component (below) | `tests/period_case.py` |
| `network-check-difftest` | none | `NetworkCheck.Difftest` | that `certificate.py`'s encodings are `NetworkCheck.Certificate`'s, byte for byte | `certificate.py` |

The other modules:

- `NetworkCheck.Certificate`: the canonical formats every check reads. These are the frame tree, the parameters
  digest, the record head and its digest, and the message a certificate signs.
- `NetworkCheck.RecordsGrid` and `NetworkCheck.RecordsBytes`: the package's guarantees (`lean-audit.json`).
  `RecordsGrid` proves that the records check's linear Program grid is `NetworkCheck.Check`'s. `RecordsBytes` proves
  that its evidence, with payloads kept as bytes, is v4's evidence read as bits.
- `NetworkCheck.LinkWindow`: the verdict of record on one link-window from its proof (`service.LinkWindows.verify`).

`network-records verdicts` serves three functions on the difftest harness's protocol (`Difftest.serve`): `verdicts`
(every link-window's verdict, with the grid compared), `verdicts-no-grid`, and `link-window`.

## The period check

`network-period` checks one node's link over m link-windows, beside the period's compute audits (sampled proofs'
one-stage, one answer per registered sampling block of the workload). Its parameters name one of two forms:

- **the clock form**: each link-window proved, both its ends certified and its clock certified (`service.LinkWindows`);
- **the rows form**: the network certifier's own run, each link-window's cells committed as rows under one fixed
  schedule a direction, its records certified, and the scan's one-stage answer for it.

It accepts when every link-window passes its form's check and every compute audit accepted. It then publishes `K`, the
bits a covert sender in the server can move through the network certifier in the period, as
`NetTiming.Guarantees.PeriodLeakAccepted` counts them (`Security/Proofs/NetworkCertifier/README.md`).

## Who runs it

| executable | run by |
| --- | --- |
| `network-records` | `verity/core/protocols/certifier/lean_audit.py` (`lean_audit.Audit`) and `service.py`'s link-window verdict (`verdicts`' `link-window`); the records' runners (`benchmarks/network_e2e/run.sh`, `benchmarks/network_traces/chain_record.sh`) build it and pass it on |
| `network-period` | `benchmarks/network_traces/certifier_piece.py`, `block_certifier.py` and `scan_period.sh` |
| `network-check-difftest` | the Lean check, on the vectors below |
| `network-check` | only its own test, `tests/test_network_check_protocol.py`; the records run `network-records` |

- `lean_audit.Audit` is `audit.Audit` with every verdict taken from one `network-records verdicts` process.
  `VERITY_NETWORK_RECORDS_EXE` names a built executable; without it, the first request runs `lake build
  network-records` here.
- The onsite composite protocol (`verity/core/protocols/onsite`) composes `verity/certifier/v0` as its communication
  component (`Core.Onsite.components`). The composition's difftest takes the period's mirror and cases from
  `tests/period_case.py`.

## Tests

`tests/` is one suite, run after `check`'s Lean steps since it needs `lake` (it skips without it):

- `test_network_check_protocol.py`: `network-check serve` against `audit.py`, on generated spatial units, each with
  one change aimed at one path of the check (`MUTATIONS`).
- `test_network_records_protocol.py`: `network-records serve` and `verdicts`, with the grid compared and without it, on
  the same units' records published raw, committed, certified or mixed.
- `test_network_period_protocol.py`: `network-period serve` against `period_case.py`'s mirror, in both forms.
- `test_lean_audit.py`: `lean_audit.Audit` against `audit.Audit`, which must give the same verdicts, clock, `accepted`
  and `violations`.
- `test_nci_*.py`: the Python halves of the inference-only policy check's difftests, whose Lean is in `nci/`.

`lean/scripts/difftest_vectors.py` writes `network-check-difftest`'s data, 2,000 random instances and
`certificate.py`'s results on each. The Lean check runs the difftest on them (`runs` in `lean-audit.json`, which pins
both files' SHA-256), and `verity/core/protocols/certifier/tests/test_network_check_lean_difftest.py` checks that the
script still writes them.

## How to run it

~~~sh
cd verity/core/protocols/network/lean && lake build      # the four executables, under .lake/build/bin/
.lake/build/bin/network-check id                          # verity/network-certificate/v4
~~~

From the repository root:

~~~sh
uv run tools/verity/check/suites.py verity/core/protocols/network    # the difftests and the period mirror
python3 tools/verity/lean/check.py --build verity/core/protocols/network/lean    # this package's Lean check
~~~
