module

public import NetworkCheck.Certificate
public import NetworkCheck.Check
public import NetworkCheck.Difftest
public import NetworkCheck.Records
public import NetworkCheck.RecordsGrid
public import NetworkCheck.RecordsBytes
public import NetworkCheck.Circuit
public import NetworkCheck.Period
public import NetworkCheck.RowsAudit
public import NetworkCheck.LinkWindow

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
by `tests/test_network_period_protocol.py`. `NetworkCheck.RowsAudit` audits a run of the active network certifier over
the records it signed (`verity/network-rows/v1`, the `network-rows` executable, which the certifier chain's steps run):
each record held to its certified head, then the certifier program's rows audit (`NetworkCertifier.Rows.audit`), which
refuses cells cut, clocked or ordered otherwise than the schedule. Code: it imports only Lean, Flock's verifier
(`Flock.Hash`, `Flock.Bytes`, `Flock.Canon`, and `Flock.Public`'s frame-v3 hash), the core protocol interface, the
certifier program and the LMS port.
-/

@[expose] public section

/-!
`NetworkCheck.LinkWindow` is N2's verdict of record on one execution phase from its proof (`service.LinkWindows.verify`,
the `link-window` function of `network-records verdicts`). Beyond the above it imports Flock's `Flock.Hm96` and core's
identity hash (`Identity.Tagged`).
-/

/-!
`NetworkCheck.Circuit` checks a compute audit's circuit record against its Program at the size of their description:
the population program, the wires, the registered reads, the wire order and the count of free input bits. Beyond the
above it imports NCI's layout checks (`NciCheck.Desc`) and the sampled audit's wires (`Spot.Wires`).
-/
