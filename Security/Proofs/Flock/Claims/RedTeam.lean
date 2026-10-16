import Proofs.Flock.Claims.RedTeam.ZK
import Proofs.Flock.Claims.RedTeam.Complete
import Proofs.Flock.Claims.RedTeam.Efficient
import Proofs.Flock.Claims.RedTeam.Deployed

/-! Red team of Flock's claims: claims that their own arguments discharge. `ZK`: `ZKmalR` at `ε_zk = 0` through the
simulator's own collision. `Complete`: an honest prover with an empty tape. `Efficient`: a step count of `0`.
`Deployed`: each at SHA-512 and the deployed records. -/
