import NetworkCheck.Certificate
import NetworkCheck.Check
import NetworkCheck.Difftest

/-!
# The network certificate's checks

Executable Lean that reads the network certificate's canonical formats (`NetworkCheck.Certificate`: the frame tree, the
parameters digest, the record head and its digest, the message a certificate signs), and its difftest against
`verity/core/protocols/certifier/certificate.py` (`NetworkCheck.Difftest`). `NetworkCheck.Check` is the auditor's
check of a spatial unit's certified link-windows as a core protocol (`verity/network-certificate/v2`, the `network-check`
executable), held to `audit.py` by `tests/test_network_check_protocol.py`. Code: it imports only Lean, C-Flock's verifier
(`Flock.Hash`, `Flock.Bytes`, `Flock.Canon`), the core protocol interface, the certifier program and the LMS port.
-/
