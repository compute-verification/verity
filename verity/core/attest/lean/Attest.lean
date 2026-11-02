import Attest.Period

/-!
# An auditor party's signed period roots and liveness records, as executable Lean

`verity.core.attest`'s checks: a party's period tree, a frame-v3-sha512 tree over a range of positions
(`Flock.FV3.Sha512`), the period check its proof runs and the liveness check the verifier runs in the clear
(`Attest.Period`). Every auditor party's protocol requires this package: the memory challenger's and the network
certifier's. `MemoryChallenger.Difftest` holds it to the Python. Code: it imports only Lean, C-Flock's verifier and LMS.
-/
