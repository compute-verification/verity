module

public import Proofs.Flock.Claims.Witnesses.Shared
public import Proofs.Flock.Claims.Witnesses.Flock
public import Proofs.Flock.Claims.Witnesses.Sampling
public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.Claims.Witnesses.SamplingGenericCost
public import Proofs.Flock.Claims.Witnesses.SamplingGenericZK
public import Proofs.Flock.Claims.Witnesses.SamplingGenericCompiler
public import Proofs.Flock.Claims.Witnesses.SamplingGenericKSAt
public import Proofs.Flock.Claims.Witnesses.SamplingGenericKSRegAt
public import Proofs.Flock.Claims.Witnesses.SamplingTwo
public import Proofs.Flock.Claims.Witnesses.SamplingTwoKSAt
public import Proofs.Flock.Claims.Witnesses.SamplingTwoWork
public import Proofs.Flock.Claims.Witnesses.SamplingTwoCredit
public import Proofs.Flock.Claims.Witnesses.SamplingGenericComplete
public import Proofs.Flock.Claims.Witnesses.SamplingTwoRankin
public import Proofs.Flock.Claims.Witnesses.SamplingTwoReg
public import Proofs.Flock.Claims.Witnesses.SamplingTwoRegMiss
public import Proofs.Flock.Claims.Witnesses.SamplingTwoRegX
public import Proofs.Flock.Claims.Witnesses.SamplingTwoFirstHit
public import Proofs.Flock.Claims.Witnesses.Encodings
public import Proofs.Flock.Claims.Witnesses.EncodingsGeneric
public import Proofs.Flock.Claims.Witnesses.ZKOutLeak
public import Proofs.Flock.Claims.Witnesses.Recursion
public import Proofs.Flock.Claims.Witnesses.RecursionMeets
public import Proofs.Flock.Claims.Witnesses.RecursionLin
public import Proofs.Flock.Claims.Witnesses.RecursionLin2
public import Proofs.Flock.Claims.Witnesses.Stated
public import Proofs.Flock.Claims.Witnesses.KSTable
public import Proofs.Flock.Claims.Witnesses.Meets
public import Proofs.Flock.Claims.Witnesses.BindingMeets

/-!
# Refutation witnesses

Every claim definition in `Definitions.Core.Interaction` and `Definitions.Claims*` carries a theorem `<Claim>_fails`: a
concrete toy instance at which the claim is false, so its definition constrains something. `Shared` has the shared
forms' witnesses and the toys they use, `Flock` Flock's, `Sampling` the sampling compiler's, `SamplingGeneric` the
generic sampled compiler's (with their statements over the spec; `SamplingGenericCost` its cost claims',
`SamplingGenericKSAt` its KS read at a time budget, with `CompilerKSAt`'s nontrivial lemma, `SamplingGenericKSRegAt`
its committed form read at a time budget, with `CompilerKSRegAt`'s, and `SamplingGenericZK` its
zero-knowledge claims' and `ZKOut`'s) and `Recursion` the ZK-recursion compiler's. A claim still met by a trivial
instance has a theorem `<Claim>_vacuous` that exhibits it.

The lock's `claims` section (`Security/lean-audit.json`) lists, per claim, its `fails` and `meets` witnesses as
guarantees: `Stated` states the shared, Flock and sampling claims' failures over the spec (`Claims.<Claim>_fails`), and
`Meets` their checkable antecedents at Flock's deployed records (`deployPlain`, `deployZk`), and `BindingMeets` the
sampling `Binding`'s at the deployed sampling record (`SamplingDeploy.deploy`). The ZK-recursion compiler's claims carry
`Claims.Witness.Recursion.<Claim>_fails` and `<Claim>_meets`, their checkable antecedents met at the deployed record
(`RecursionMeets`). A closed statement proved of every instance (the generic sampled compiler's, and `ZKOutLeak`) is
under `claims.generic` with the guarantee that states it, which the lock's `witnesses` holds to an honest and a
nontrivial lemma (`SamplingGenericCompiler`, `SamplingGenericComplete`, `SamplingTwo` for the two-stage compiler,
`SamplingTwoKSAt` for its extractor's cost and its KS read at a time budget, `SamplingTwoRankin` for Rankin's bound on
its count, `SamplingTwoReg` for its committed form and headline, `SamplingTwoRegMiss` for its miss bound and the
headline's honest lemma, `SamplingTwoRegX` for the three at the boundary trace read at the statement, `ZKOutLeak`, and
the registered encodings' `EncodingsGeneric` at the toys of `Encodings`).
-/

@[expose] public section
