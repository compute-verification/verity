import Attest.FrameV3Sha512
import Attest.Period

/-!
# An auditor party's signed period roots and liveness records, as executable Lean

`verity.core.attest`'s checks: the frame-v3-sha512 tree over a range of positions (`Attest.FrameV3Sha512`), and a
party's period tree, the period check its proof runs and the liveness check the verifier runs in the clear
(`Attest.Period`). Every auditor party's protocol requires this package: the memory challenger's and the network
certifier's. `MemoryChallenger.Difftest` holds it to the Python. Code: it imports only Lean, C-Flock's verifier, core's
randomness and LMS.
-/
