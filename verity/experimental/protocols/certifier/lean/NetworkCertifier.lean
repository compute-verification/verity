import NetworkCertifier.Grid
import NetworkCertifier.Run
import NetworkCertifier.Ingress
import NetworkCertifier.Schedule
import NetworkCertifier.Difftest

/-!
# The network certifier's program

The executable forms of the reference definitions in `verity/Security/Definitions/NetworkCertifier/` that the difftest runs
(`NetworkCertifier`), and the difftest itself (`NetworkCertifier.Difftest`). Code: it imports only Lean, and never
`verity/Security/`; `security_proofs` requires it and proves the agreement (`Proofs.NetworkCertifier.Code`).
-/
