# Network attestation

It bounds, in bits a monitoring period, what leaves a served node beyond what its registered circuit explains: every
byte in and out passes the auditor's network attester, which releases and times it on a grid and signs what it carried.
The bound rests on those signed records, the scan's proof blocks sampled on live coins, and the period check in Lean.

The guarantee: `NetTiming.Guarantees.MonitoringPeriodLeakAccepted` (`security/Definitions/Certifier/Guarantees/MonitoringPeriod.lean`).

Not yet: the served road runs the attester's front but its piece writes NOT RUN, so no served window's verdict reads the period check.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the network attester | relays each connection, releases bytes on the grid's buckets, signs each window's records | `verifier/network_attester.py`, `verifier/pacer.py` |
| its host programs | keys, the serve and rehearsal, the load relay, the connection rows | `attester/` |
| the attester's program | the records' bytes and grid as Lean the relay is held to | `attester/program/lean/` |
| the scan | the proof blocks that carry the state between buckets and compute what may leave | `prover/ingress_scan.py`, `prover/egress.py` |
| the engine's admission | when the engine may take a request: by arrival, release order or tick | `prover/admission.py`, `prover/id_relay.py` |
| the Lean verifier | the signatures, the openings of the signed rows, the period's check | `verifier/lean/` (`network-period`, `network-records`, `network-rows`) |
| the scheme | what the network attester attaches to a serve | `scheme.py` |

## What varies

| what varies | choices |
| --- | --- |
| admission order | `arrival` served; `release-order` exists; `tick` exists |
| framing | `ids` served (`--protocol ids`, the id relay); plain HTTP exists |
| ports | the scan's own state port exists; the compiler's named external ports never written |
| schedule form | form 1 exists; form 2 exists |
| connection rows | on exists |

A new implementation supplies: its framing of a request and its admission hook in the engine.
Fixed for every implementation: the grid, the signed records, the scan's Definitions and the Lean check.

## Interfaces

- From the workload: the served bytes at the attester's ports and, under `ids`, requests as token ids.
- From `run`: the session and its key, the period's execution phases, the coins that sample the scan's blocks.
- It gives: the period's verdict and its leak bound in bits, which compose's verdict reads.

## Try it

~~~sh
uv run python verity/protocols/accounting/network/attester/network_attester_serve.py rehearse --fake --out /tmp/net --requests 4
~~~

It relays four requests to a stand-in server, signs each execution phase and checks the run, then prints one line:
`ACCEPTED: 4/4 ended, …; N/N execution phases certified; cells in their bucket …`, with the grid's timing beside it.
