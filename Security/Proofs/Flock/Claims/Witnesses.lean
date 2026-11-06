module

public import Proofs.Flock.Claims.Witnesses.Shared
public import Proofs.Flock.Claims.Witnesses.Flock
public import Proofs.Flock.Claims.Witnesses.Sampling
public import Proofs.Flock.Claims.Witnesses.Recursion
public import Proofs.Flock.Claims.Witnesses.RecursionMeets
public import Proofs.Flock.Claims.Witnesses.RecursionLin
public import Proofs.Flock.Claims.Witnesses.RecursionLin2
public import Proofs.Flock.Claims.Witnesses.Stated
public import Proofs.Flock.Claims.Witnesses.Meets
public import Proofs.Flock.Claims.Witnesses.BindingMeets

/-!
# Refutation witnesses

Every claim definition in `Definitions.Core.Interaction` and `Definitions.Claims*` carries a theorem `<Claim>_fails`: a
concrete toy instance at which the claim is false, so its definition constrains something. `Shared` has the shared
forms' witnesses and the toys they use, `Flock` Flock's, `Sampling` the sampling compiler's and `Recursion` the ZK-recursion compiler's. A claim still met
by a trivial instance has a theorem `<Claim>_vacuous` that exhibits it.

The lock's `claims` section (`Security/lean-audit.json`) lists, per claim, its `fails` and `meets` witnesses as
guarantees: `Stated` states the shared, Flock and sampling claims' failures over the spec (`Claims.<Claim>_fails`), and
`Meets` their checkable antecedents at Flock's deployed records (`deployPlain`, `deployZk`), and `BindingMeets` the
sampling `Binding`'s at the deployed sampling record (`SamplingDeploy.deploy`). The ZK-recursion compiler's claims carry
`Claims.Witness.Recursion.<Claim>_fails` and `<Claim>_meets`, their checkable antecedents met at the deployed record
(`RecursionMeets`).
-/

@[expose] public section
