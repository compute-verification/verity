import Proofs.Flock.Claims.Complete.Replay.Steps
import Proofs.Flock.Claims.Complete.Replay.Zerocheck

/-! The verifier's phases replayed on the honest prover's rounds: the steps of both sides (`Steps`: the prover's in the
pure session, `squeezeP_go`, `forIn_go`; the verifier's on rounds that carry the prover's, `squeezeWith_sent`, `Agrees`),
and the zerocheck (`Zerocheck`: the prover's in closed form, `zerocheckP_go`, and the verifier's on its rounds,
`zerocheckWith_of`). -/
