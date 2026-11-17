module

public import Definitions.Pouw.Assumptions.CreditedWords
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

/-! The joint period's PoUW guarantee from a credit in words, at any price. -/

@[expose] public section

namespace Properties.Certifier.JointBridge

/-- **PoUW's per-period guarantee in the joint period from a credit in words, at any price.** On any device and any
period of a node's run whose PoUW audit is a source game's on PoUW's coordinate of the coins, with the work certificate
within the credit at `m` multiply-adds a word, less the `K / b_unit` units the network attester's bits can carry at `u`
multiply-adds a unit, all at `p .tc` (`ResourceAccounting.RealizesCreditedWords`), each right word of a unit the device didn't
outsource costing `m` multiply-adds of priced work (`ResourceAccounting.Assumptions.WordCostsMacs`) and at most `K / b_unit`
units outsourced (`ResourceAccounting.Assumptions.HelpWithinCertifier`): if, at each value `ξ` of the coins, the source game
accepts while its words fixed right fall below its credit with probability at most `δ`, then PoUW's per-period
guarantee holds at certificates `c`, price `p` and error `δ`. -/
def WorkOfCreditedWords : Prop :=
  ∀ (ι α I : Type) [Fintype α] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    (d : ResourceAccounting.Device ι) (C : Verity.Circuit) (r : ResourceAccounting.PeriodRun d α C Ω) (c : ResourceAccounting.Certificates)
    (p : ResourceAccounting.Price) (m u bUnit : ℕ) (X : Type) [Fintype X] (acc : ((i : I) → Ω i) → X → Prop)
    (fixed credit help : ((i : I) → Ω i) → X → ℝ) (δ : ℝ)
    (real : ResourceAccounting.RealizesCreditedWords r c p m u bUnit X acc credit),
    ResourceAccounting.Assumptions.WordCostsMacs r p m u real.coin acc fixed help →
    ResourceAccounting.Assumptions.HelpWithinCertifier r c bUnit real.coin acc help →
    (∀ ξ, Pouw.pr (fun x : X => acc ξ x ∧ fixed ξ x < credit ξ x) ≤ δ) →
    r.Sound c p .pouw δ

end Properties.Certifier.JointBridge
