module

public import NetworkCheck.Certificate
public import NetworkCheck.Check
public import NetworkCheck.Registered
public import NetworkCheck.Difftest
public import NetworkCheck.Records
public import NetworkCheck.RecordsGrid
public import NetworkCheck.RecordsBytes
public import NetworkCheck.Period
public import NetworkCheck.LinkWindow
public import NetworkCheck.ConnectionRows
public import NetworkCheck.ConnectionRowsDifftest

/-!
# The network certificate's checks

Executable Lean that reads the network certificate's canonical formats (`NetworkCheck.Certificate`: the frame tree, the
parameters digest, the record head and its digest, the message a certificate signs), and its difftest against
`verity/core/protocols/certifier/certificate.py` (`NetworkCheck.Difftest`). `NetworkCheck.Check` is the auditor's
check of a spatial unit's certified execution phases as a core protocol (`verity/network-certificate/v4`, the
`network-check` executable), held to `audit.py` by `tests/test_network_check_protocol.py`. `NetworkCheck.Records` is the
same audit over raw, committed and certified records (`verity/network-records/v3`, the `network-records` executable,
which `lean_audit.Audit` runs, with the egress grid compared or, for `check_link(grid=False)`, not), held to `audit.py`
by `tests/test_network_records_protocol.py`; `NetworkCheck.RecordsGrid` proves that its linear Program grid is
`NetworkCheck.Check`'s, and `NetworkCheck.RecordsBytes` that its evidence, payloads kept
as bytes, is v4's read as bits. `NetworkCheck.Period` is onsite's communication component
(`verity/certifier/v0`, the `network-period` executable): a period's link-windows, either proved by N2 and certified
with their clocks (the clock form) or the network certifier's own rows, registered under its one schedule and read by
the scan (the rows form), beside its compute audits, and the bound `K` on what left the site, held to its Python mirror
by `tests/test_network_period_protocol.py`. Code: it imports only Lean, Flock's verifier (`Flock.Hash`, `Flock.Bytes`,
`Flock.Canon`, and `Flock.Public`'s frame-v3 hash), the core protocol interface, the certifier program and the LMS port.
-/

@[expose] public section

/-!
`NetworkCheck.LinkWindow` is N2's verdict of record on one execution phase from its proof (`service.LinkWindows.verify`,
the `link-window` function of `network-records verdicts`). Beyond the above it imports Flock's `Flock.Hm96` and core's
identity hash (`Identity.Tagged`).

`NetworkCheck.ConnectionRows` is the head the network certifier signs for an execution phase's connection rows and the
verifier's read of its registration (`connection_rows.py`): the tables of a schedule with lengths rows and content
classes, each one's domain, the head's encoding and digest, and under the lengths rows' form 2 the rules on a class's
count of held rows; `NetworkCheck.ConnectionRowsDifftest` is its difftest against the Python. It imports the period
check and what that imports.
-/
