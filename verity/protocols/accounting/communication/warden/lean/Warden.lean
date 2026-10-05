import Warden.Grid
import Warden.Run
import Warden.Ingress
import Warden.Schedule
import Warden.Difftest

/-!
# The network warden's program

The executable forms of the reference definitions in `verity/Security/Definitions/Warden/` that the difftest runs
(`NetTiming.Code`), and the difftest itself (`NetTiming.Code.Difftest`). Code: it imports only Lean, and never
`verity/Security/`; `security_proofs` requires it and proves the agreement (`Proofs.Warden.Code`).
-/
