# Network attestation

Every byte into and out of a served node passes the auditor's network attester, which relays each connection, releases
its bytes on a grid of buckets and signs each window's records. The scan's Definitions recompute, bucket by bucket, what
the registered circuit lets leave, and the period check in Lean decides a monitoring period from the signed records and
the scan's sampled blocks.

Not yet: on the tip the network audit is not run: the attester's piece writes NOT RUN, nothing proves a scan block, and
no window's verdict reads the scan or the period check. A period whose schedule has connection rows has no Lean verdict
at all: the Lean period check refuses it, and only a Python check reads those rows.
Even when run, under a uniform sample the scan's few blocks may all go unsampled, so the bound holds for a window only at
the sample's chance of hitting them, until accounting blocks can always be proved.

The guarantee, stated: `NetTiming.Guarantees.MonitoringPeriodLeakAccepted` (`security/Definitions/Certifier/Guarantees/MonitoringPeriod.lean`),
on what a period the check accepts can leak; the lock lists it as deployed, but its honest witness is a toy code that
accepts no run, and it rests on four open premises (`Compiled.KS`, `CertifierRegisters`, `ScopeAcc`, `TrustedDevice`).
The network attester is the auditor's hardware on a machine the developer does not control; trust in it and its host is
the named assumption `trusted-device`.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the network attester | relays each connection, releases bytes on the grid's buckets, signs each window's records | `verifier/network_attester.py`, `verifier/pacer.py` |
| its host programs | keys, the serve and rehearsal, the load relay, the connection rows | `attester/` |
| the attester's program | the records' bytes and grid as Lean the relay is held to | `attester/program/lean/` |
| the scan | the Definitions that carry the state between buckets and compute what may leave | `prover/ingress_scan.py`, `prover/egress.py` |
| the engine's admission | when the engine may take a request: by arrival, release order or tick | `prover/admission.py`, `prover/id_relay.py` |
| the Lean verifier | the signatures, the openings of the signed rows, the period's check | `verifier/lean/` (`network-period`, `network-records`, `network-rows`) |
| the scheme | what the network attester attaches to a serve | `scheme.py` |

## What varies

| what varies | on `onsite_full.toml`'s road, the one application that lists the attester |
| --- | --- |
| admission order | `arrival` served; `release-order` exists; `tick` exists |
| framing | `ids` served (`--protocol ids`, the id relay); plain HTTP exists |
| ports | the scan's own state port exists; the compiler's named external ports never written |
| schedule form | form 1 exists; form 2 exists |
| connection rows | exists |
| the check of the connection rows | Python only (`verifier/connection_rows_check.py`); the Lean period check refuses a schedule with them |

A new implementation supplies: its framing of a request and its admission hook in the engine.
Fixed for every implementation: the grid, the signed records, the scan's Definitions and the Lean check.

## Interfaces

- From the workload: the served bytes at the attester's ports and, under `ids`, requests as token ids.
- From `run`: the session and its key, the period's execution phases, the coins that sample the scan's blocks.
- It gives: today, the attester's piece's file naming NOT RUN, which the application's verdict reads as pending; when
  run, the Lean period check's verdict on the period, which `run`'s verdict reads.

## Try it

~~~sh
uv run python verity/protocols/accounting/network/attester/network_attester_serve.py rehearse --fake --out /tmp/net --requests 4
~~~

It relays four requests to a stand-in server and signs each execution phase, then prints the relay check's verdict, in
Python: `ACCEPTED: 4/4 ended, …; N/N execution phases certified; cells in their bucket …`, with the grid's timing.
