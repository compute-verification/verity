module

public import Proofs.Flock.Claims.Witnesses.Shared
public import Proofs.Flock.Claims.Witnesses.Flock
public import Proofs.Flock.Claims.Witnesses.Sampling
public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.Claims.Witnesses.SamplingGenericCost
public import Proofs.Flock.Claims.Witnesses.SamplingGenericZK
public import Proofs.Flock.Claims.Witnesses.SamplingGenericCompiler
public import Proofs.Flock.Claims.Witnesses.ZKOutLeak
public import Proofs.Flock.Claims.Witnesses.Recursion
public import Proofs.Flock.Claims.Witnesses.RecursionMeets
public import Proofs.Flock.Claims.Witnesses.RecursionLin
public import Proofs.Flock.Claims.Witnesses.RecursionLin2
public import Proofs.Flock.Claims.Witnesses.Stated
public import Proofs.Flock.Claims.Witnesses.Meets
public import Proofs.Flock.Claims.Witnesses.BindingMeets
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

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
nontrivial lemma (`SamplingGenericCompiler`, `ZKOutLeak`).
-/

@[expose] public section
