import Proofs.Flock.Claims.Witnesses.Shared
import Proofs.Flock.Claims.Witnesses.Flock
import Proofs.Flock.Claims.Witnesses.Sampling
import Proofs.Flock.Claims.Witnesses.SamplingGeneric
import Proofs.Flock.Claims.Witnesses.SamplingGenericCost
import Proofs.Flock.Claims.Witnesses.SamplingGenericZK
import Proofs.Flock.Claims.Witnesses.SamplingGenericCompiler
import Proofs.Flock.Claims.Witnesses.SamplingGenericComplete
import Proofs.Flock.Claims.Witnesses.ZKOutLeak
import Proofs.Flock.Claims.Witnesses.Recursion
import Proofs.Flock.Claims.Witnesses.RecursionMeets
import Proofs.Flock.Claims.Witnesses.RecursionLin
import Proofs.Flock.Claims.Witnesses.RecursionLin2
import Proofs.Flock.Claims.Witnesses.Stated
import Proofs.Flock.Claims.Witnesses.Meets
import Proofs.Flock.Claims.Witnesses.BindingMeets

/-!
# Refutation witnesses

Every claim definition in `Definitions.Core.Interaction` and `Definitions.Claims*` carries a theorem `<Claim>_fails`: a
concrete toy instance at which the claim is false, so its definition constrains something. `Shared` has the shared
forms' witnesses and the toys they use, `Flock` Flock's, `Sampling` the sampling compiler's, `SamplingGeneric` the
generic sampled compiler's (with their statements over the spec; `SamplingGenericCost` its cost claims', and
`SamplingGenericZK` its zero-knowledge claims' and `ZKOut`'s) and `Recursion` the ZK-recursion compiler's. A claim still met
by a trivial instance has a theorem `<Claim>_vacuous` that exhibits it.

The lock's `claims` section (`Security/lean-audit.json`) lists, per claim, its `fails` and `meets` witnesses as
guarantees: `Stated` states the shared, Flock and sampling claims' failures over the spec (`Claims.<Claim>_fails`), and
`Meets` their checkable antecedents at Flock's deployed records (`deployPlain`, `deployZk`), and `BindingMeets` the
sampling `Binding`'s at the deployed sampling record (`SamplingDeploy.deploy`). The ZK-recursion compiler's claims carry
`Claims.Witness.Recursion.<Claim>_fails` and `<Claim>_meets`, their checkable antecedents met at the deployed record
(`RecursionMeets`). A closed statement proved of every instance (the generic sampled compiler's, and `ZKOutLeak`) is
under `claims.generic` with the guarantee that states it, which the lock's `witnesses` holds to an honest and a
nontrivial lemma (`SamplingGenericCompiler`, `SamplingGenericComplete`, `ZKOutLeak`).
-/
