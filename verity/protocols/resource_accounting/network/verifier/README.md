# Network attestation's verifier (`verity.protocols.resource_accounting.network.verifier`)

Network attestation bounds what a covert sender in the serving node can tell a client through the network attester
beyond what the registered Program computes from its committed values. The network attester is the auditor's device on
a boundary link: it relays every connection in fixed-size cells on a fixed schedule of buckets, records exactly the
cells it sent, and once per execution phase signs a network attestation over a salted digest of each record. Its host
programs are `../attester`, and what the prover builds is `../prover`.

## The guarantee

`NetTiming.Guarantees.MonitoringPeriodLeakAccepted` (`security/Definitions/Certifier/Guarantees/MonitoringPeriod.lean`)
bounds, per execution phase, the probability that an accepted audit leaves the attested traffic outside what the
registered Program accepts, and over a monitoring period the bits of a covert sender's secrets a decoder recovers from
the client's view. `NetTiming.Guarantees.PooledLeakAccepted` (`.../Guarantees/PooledLeak.lean`) is the pooled count,
with the incorrect proof blocks paid for once over the monitoring period. Both are stated over the network attester's
code as `security/Definitions/Certifier` models it, not over the Lean verifier here. Their premises are their records in
`security/lean-audit.json`; the model's trusted layer and modelling choices are `security/Proofs/Certifier/README.md`.

The network attester is auditor-trusted (`trusted-device`): it runs on a host the developer doesn't control, with the
custody of its keys and its LMS leaf state. The challenger's live coins reach the server through the network attester's
coin link (`../attester/coin_link.py`) on two more premises, `CoinsConfidential` and `OneClock`, named in
`verity/protocols/proofs/spot/verifier/README.md`.

## Parts

Each module's docstring is its statement.

- The grid and its records: `grid` (constant-rate, left-packed, FIFO, fail-closed), `commitment`, `certificate` (record
  v2, the signed digest and the device state) and `audit` (`Audit`, the exact audit of every link, one clock per
  spatial unit).
- One execution phase through the proof system (N2): `service` and `head_check`. The verdict of record is the Lean's
  (`service.OF_RECORD`).
- The active network attester: `network_attester` (cells, admission, schedule, egress form, fail closed, clock,
  records), `pacer`, `release`, `network_attester_rows`, `connection_rows` and `content_rows`.
- The ties of cells to the served computation: `io_tie`, `io_request`, `tie_service` and `id_form`.
- The schedule's advice and its charge: `schedule` (the clock-sync advice), `advice`, `arith_encode` and `charge_chain`.

## Lean

`lean/` is the Lake package `network_check` (library `NetworkCheck`). Each executable but the difftest is a core
protocol, `id | check PARAMS COMMITMENT EVIDENCE | serve`, and its module's docstring has its formats:

- `network-check`: `verity/network-certificate/v4` (`NetworkCheck.Check`), held to `audit.Audit`;
- `network-records`: `verity/network-records/v3` (`NetworkCheck.Records`), held to `audit.Audit`. Its `verdicts` verb is
  N2's verdict of record (`NetworkCheck.TransferBlocks`), the process `service` asks; `VERITY_NETWORK_RECORDS_EXE`
  names a built one;
- `network-period`: `verity/certifier/v0` (`NetworkCheck.Period`), an attestation period, onsite's network attestation
  subprotocol. It reads the inference-only policy's Lean verifier, which the public tree withholds, so the public tree
  doesn't build it;
- `network-span`: `verity/certifier/span/v0` (`NetworkCheck.Span`);
- `network-rows`: `verity/network-rows/v1` (`NetworkCheck.RowsAudit`), an active network attester's run as rows, held
  to `network_attester.audit`;
- `network-check-difftest INSTANCES PYTHON N`: `certificate.py`'s encodings held to `NetworkCheck.Certificate`.

`../attester/program/lean/` is the Lake package `certifier_program` (library `NetworkAttester`): the network attester's
program, with `certifier-difftest`, `certifier-charge-difftest` and `certifier-window-difftest` (`INSTANCES PYTHON N`)
on what its `scripts/` write. The Lean check runs both packages' scripts and difftests (each `lean-audit.json`'s `runs`).

## Use

~~~python
from verity.protocols.resource_accounting.network.verifier.audit import Audit
from verity.protocols.resource_accounting.network.verifier.grid import GridParams, Certifier
from verity.protocols.resource_accounting.network.verifier.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

params = GridParams(buckets=600, allocation=1024, frame_bytes=4096)   # 1-min execution phases of 100 ms buckets
sync_set = SyncSet(tuple(range(16)), tuple(range(1, 9)))              # 7 bits per execution phase
frames = [LogicalFrame(b"\0" * 4096, anchor=3, index=i) for i in range(5)]
syncs = [ClockSync(delta=2, rho=4)]
records = run_shaper(frames, ready=[4] * 5, syncs=syncs, certifier=Certifier("responses", params))
audit = Audit()
audit.check_link("responses", records, frames, syncs, sync_set, params)
assert audit.accepted
~~~

Tests: `uv run tools/check/suites.py tests/verity/protocols/resource_accounting/network/verifier`.
