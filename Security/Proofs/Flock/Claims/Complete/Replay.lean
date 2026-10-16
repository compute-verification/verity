import Proofs.Flock.Claims.Complete.Replay.Steps
import Proofs.Flock.Claims.Complete.Replay.Zerocheck
import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra

/-! The verifier's phases replayed on the honest prover's rounds: the steps of both sides (`Steps`: the prover's in the
pure session, `squeezeP_go`, `forIn_go`; the verifier's on rounds that carry the prover's, `squeezeWith_sent`, `Agrees`),
and the zerocheck (`Zerocheck`: the prover's in closed form, `zerocheckP_go`, and the verifier's on its rounds,
`zerocheckWith_of`; `ZerocheckAlgebra`: on an R1CS-valid trace whose outer coins avoid 1 the residual claim ends at
final_a · final_b, `fold_final`). -/
