import Proofs.Flock.Claims.Witnesses.Shared
import Proofs.Flock.Claims.Witnesses.Flock
import Proofs.Flock.Claims.Witnesses.Sampling
import Proofs.Flock.Claims.Witnesses.Recursion
import Proofs.Flock.Claims.Witnesses.Stated
import Proofs.Flock.Claims.Witnesses.Meets

/-!
# Refutation witnesses

Every claim definition in `Definitions.Claims*` carries a theorem `<Claim>_fails`: a concrete toy instance at which
the claim is false, so its definition constrains something. `Shared` has the shared forms' witnesses and the toys they
use, `Flock` Flock's, `Sampling` the sampling compiler's and `Recursion` the ZK-recursion compiler's. A claim still met
by a trivial instance has a theorem `<Claim>_vacuous` that exhibits it.

The lock's `claims` section (`Security/lean-audit.json`) lists, per claim, its `fails` and `meets` witnesses as
guarantees: `Stated` states each claim's failure over the spec (`Claims.<Claim>_fails`), and `Meets` states the claims'
checkable antecedents at Flock's deployed records (`deployPlain`, `deployZk`).
-/
