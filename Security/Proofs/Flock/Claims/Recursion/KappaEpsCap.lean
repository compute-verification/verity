module

public import Proofs.Flock.Claims.Recursion.KappaEpsDeployed
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
# Route 2C's split bound at the audit's call caps: `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}`

`kappa_eps_deployed` evaluates `κ + t·ε` where the inner setup `|S(x_I)|` and the call bounds `n` and `v` are each at
most `2^32`. The audit reads it at the bounds every accepted statement meets: the inner setup's calls at most
`innerB ≤ 2^32` (`innerSetupIns_le_acc`), the inner verifier's `2^28 + (innerB + 2^24) ≤ 2^32` (`nR_le`), and V\*'s
setup calls `nS x · 138543104` (`setupCalls_deployedB_le_cap`), at most `16 · 138543104 ≤ 2^32` at the verifier's cap
of `16` parts (`RecBudget.MAX_STATEMENTS`, `nS_le_sixteen`). So at the deployment's tables (`OneTableAt`: `SizesAt` but
for its inner-setup bound), `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}` (`kappa_eps_oneTable_cap`).
-/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **The deployment's tables at a statement `x`**, `SizesAt` but for its inner-setup bound (which every accepted
statement meets, `innerSetupIns_le_acc`): its inner statement is one plain table, and each V\* part it stages is one
`--zk` table. The parts are at most `16`, the verifier's cap (`nS_le_sixteen`). -/
def OneTableAt (b : Brings) (x : Msg) : Prop :=
  (∃ s spec sts, FlockDeploy.PlainIn (xIOf x) s spec sts ∧ sts.size = 1) ∧
    ∀ j < nS x, ∃ i spec' sts', PartIn (staged b.st x (pick b.st x j) j) i spec' sts' ∧ sts'.size = 1

theorem SizesAt.oneTableAt {b : Brings} {x : Msg} (h : SizesAt b x) : OneTableAt b x :=
  ⟨h.1, h.2.1⟩

/-- **Route 2C's split bound at the audit's call caps: `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}`**, at a statement at the
deployment's tables (`OneTableAt b x`), an inner setup of at most `2^32` calls (`hS`) and call bounds `n ≤ 2^32` and
`v ≤ nS x · 138543104` (`hU`, `hV`): what `innerSetupIns_le_acc` proves of the inner setup and `nR_le` of the inner
verifier's calls at every accepted statement, and what `setupCalls_deployedB_le_cap` proves of V\*'s setup calls. The
parts are at most `16` (`nS_le_sixteen`), so `v ≤ 16 · 138543104 ≤ 2^32`, and the bound is `kappa_eps_deployed`'s.
With `bitsLin2KOn_of_ksLin2On` it makes `Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}`. -/
theorem kappa_eps_oneTable_cap (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) {x : Msg} (h : OneTableAt b x)
    (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 32) {n : ℕ} (hU : n ≤ 2 ^ 32) {v : ℕ} (hV : v ≤ nS x * 138543104) :
    Claims.Recursion.κ ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI) x ≤
        2 ^ (-(167.7 : ℝ)) ∧
      Claims.Recursion.epsLin2
        ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
        (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
        FlockDeploy.KS.CpDep x n v ≤ 2 ^ (-(174.95 : ℝ)) := by
  obtain ⟨⟨s, spec, sts, hin, h1⟩, hO⟩ := h
  exact kappa_eps_deployed rel b L nextE colsE wE join rowIn final w decI hin h1 hO hS hU
    (hV.trans ((Nat.mul_le_mul_right _ (nS_le_sixteen hO)).trans (by norm_num)))

end RecDeploy
