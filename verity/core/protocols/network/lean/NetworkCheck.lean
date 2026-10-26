import NetworkCheck.Certificate
import NetworkCheck.Check
import NetworkCheck.Difftest
import NetworkCheck.Records
import NetworkCheck.RecordsGrid
import NetworkCheck.RecordsBytes
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
as bytes, is v4's read as bits. `NetworkCheck.LinkWindow` is N2's verdict of record on one link-window from its proof
(`service.LinkWindows.verify`, the `link-window` function of `network-records verdicts`). Code: it imports only Lean,
C-Flock's verifier (`Flock.Hash`, `Flock.Bytes`, `Flock.Canon`, `Flock.Hm96`), the core protocol interface, the
certifier program, the identity hash and the LMS port.
-/
