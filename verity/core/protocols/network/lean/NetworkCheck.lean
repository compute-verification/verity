import NetworkCheck.Certificate
import NetworkCheck.Check
import NetworkCheck.Difftest
import NetworkCheck.Records
import NetworkCheck.RecordsGrid
import NetworkCheck.RecordsBytes
import NetworkCheck.Period
import NetworkCheck.LinkWindow

/-!
# The network certificate's checks

Executable Lean that reads the network certificate's canonical formats (`NetworkCheck.Certificate`: the frame tree, the
parameters digest, the record head and its digest, the message a certificate signs), and its difftest against
`verity/core/protocols/certifier/certificate.py` (`NetworkCheck.Difftest`). `NetworkCheck.Check` is the auditor's
check of a spatial unit's certified link-windows as a core protocol (`verity/network-certificate/v4`, the `network-check`
executable), held to `audit.py` by `tests/test_network_check_protocol.py`. `NetworkCheck.Records` is the same audit over
raw, committed and certified records (`verity/network-records/v3`, the `network-records` executable, which
`lean_audit.Audit` runs, with the egress grid compared or, for `check_link(grid=False)`, not), held to `audit.py` by `tests/test_network_records_protocol.py`; `NetworkCheck.RecordsGrid` proves
that its linear Program grid is `NetworkCheck.Check`'s, and `NetworkCheck.RecordsBytes` that its evidence, payloads kept
as bytes, is v4's read as bits. `NetworkCheck.Period` is onsite's communication component
(`verity/certifier/v0`, the `network-period` executable): a period's link-windows, either proved by N2, each proof
checked by one-stage v2 run by the check itself, and certified with their clocks (the clock form), or the network
certifier's own rows, registered under its one schedule and read by the scan (the rows form, refused at the scan's
answer until one-stage runs the scan), beside its compute audits, and the bound `K` on what left the site, held to its
Python mirror by `tests/test_network_period_protocol.py`. Code: it imports only Lean, C-Flock's verifier (`Flock.Hash`,
`Flock.Bytes`, `Flock.Canon`, and `Flock.Public`'s frame-v3 hash), the core protocol interface, sampled proofs'
one-stage v2 (`SampledProofs.OneStageZk`), the certifier program and the LMS port.
-/

/-!
`NetworkCheck.LinkWindow` is N2's verdict of record on one link-window from its proof (`service.LinkWindows.verify`, the
`link-window` function of `network-records verdicts`). Beyond the above it imports C-Flock's `Flock.Hm96` and core's
identity hash (`Identity.Tagged`).
-/
