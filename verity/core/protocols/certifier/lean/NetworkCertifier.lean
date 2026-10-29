import NetworkCertifier.Grid
import NetworkCertifier.Run
import NetworkCertifier.Ingress
import NetworkCertifier.Schedule
import NetworkCertifier.Difftest
import NetworkCertifier.AdviceV1
import NetworkCertifier.ChargeDifftest
import NetworkCertifier.WindowCharge
import NetworkCertifier.WindowDifftest
import NetworkCertifier.Calls
import NetworkCertifier.CallsDifftest

/-!
# The network certifier's program

The executable forms of the reference definitions in `Security/Definitions/NetworkCertifier/` that the difftest runs
(`NetworkCertifier`, advice v1's charge, `NetworkCertifier.AdviceV1`, and its whole window's,
`NetworkCertifier.WindowCharge`), and the difftests themselves (`NetworkCertifier.Difftest`,
`NetworkCertifier.AdviceV1.ChargeDifftest`, `NetworkCertifier.WindowCharge.WindowDifftest`). Code: it imports only Lean,
and never `Security/`; `security_proofs` requires it and proves the agreement (`Proofs.NetworkCertifier.Code`).

The certifier as the lead's coin source and per-call count, `NetworkCertifier.Calls`, is the definition itself, with
its difftest against `calls.Calls` (`NetworkCertifier.Calls.CallsDifftest`).
-/
