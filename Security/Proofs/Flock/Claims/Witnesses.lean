import Proofs.Flock.Claims.Witnesses.Shared
import Proofs.Flock.Claims.Witnesses.Flock
import Proofs.Flock.Claims.Witnesses.Sampling

/-!
# Refutation witnesses

Every claim definition in `Definitions.Claims*` carries a theorem `<Claim>_fails`: a concrete toy instance at which
the claim is false, so its definition constrains something. `Shared` has the shared forms' witnesses and the toys they
use, `Flock` Flock's and `Sampling` the sampling compiler's. A claim still met by a trivial instance has a theorem
`<Claim>_vacuous` that exhibits it.
-/
