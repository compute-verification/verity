module

public import Definitions.Pouw.TS.Routes
public import Definitions.Pouw.RelativeSecurity
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

/-! EPS1-SPEC §2.7's two plays over PoUW's rounds, full checking and sampling. -/

@[expose] public section

namespace ResourceAccounting.TS

open Pouw.RelativeSecurity

/-- **A play** of a window of `S` rounds on coins `Ω`: on each path the protocol's acceptance and the cheater's
labelled run, and the ideal game. -/
structure Play (Ω : Type) (S : ℕ) (K T L U : Type) where
  game : RoundGame Ω S
  accepts : Paths Ω S → Prop
  run : Paths Ω S → Anatomy K T L U

namespace RoundGame

variable {Ω : Type} [Fintype Ω]

/-- **(L) at `p`, one bet a round**: every round's share lies in `[0, 1]`, and at every round, whatever the past, a
deficient round passes with probability at most `p`: `Pr[pass ∧ d_w > 0] ≤ p·Pr[d_w > 0]` over the round's coins
(when `d_w` is fixed before them, as a committed share is, this is `d_w > 0 → Pr[pass] ≤ p`). -/
def LuckAt (p : ℝ) : {n : ℕ} → RoundGame Ω n → Prop
  | _, .done => True
  | _, .round d pass next =>
      (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧ prob (fun ω => pass ω ∧ 0 < d ω) ≤ p * prob (fun ω => 0 < d ω) ∧
        ∀ ω, LuckAt p (next ω)

/-- **`Λ` bounds the saving per unit of detection**: at every round, whatever the past, the shares are `≥ 0` and
`E[exp(d_w / Λ) · 1{pass}] ≤ 1` over the round's coins. -/
noncomputable def BudgetAt (Λ : ℝ) : {n : ℕ} → RoundGame Ω n → Prop
  | _, .done => True
  | _, .round d pass next =>
      (∀ ω, 0 ≤ d ω) ∧
        (∑ ω, (by classical exact if pass ω then Real.exp (d ω / Λ) else 0)) ≤ (Fintype.card Ω : ℝ) ∧
        ∀ ω, BudgetAt Λ (next ω)

noncomputable def deficientRounds : {n : ℕ} → Pouw.RelativeSecurity.RoundGame Ω n → Paths Ω n → ℕ
  | _, .done, _ => 0
  | _, .round d _ next, x => (by classical exact if 0 < d x.1 then 1 else 0) + deficientRounds (next x.1) x.2

end RoundGame

end ResourceAccounting.TS
