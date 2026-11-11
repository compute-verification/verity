module

public import Properties.Accounting.JointBridge.WorkOfCreditedWords
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
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# Proof of the joint period's PoUW guarantee from a credit in words

At each value `ξ` of the other audits' coins, PoUW's coordinate of the coins is the source game's (`coin`), so the
joint event, PoUW accepting while the compute account did less than `W`, maps into the source's (`pr_le_pr_of_equiv`):
the verdict is the source's acceptance (`accepts_iff`), and on an accepted period whose fixed words reach the credit,
`W ≤ m·p(tc)·credit − u·p(tc)·K/b_unit ≤ m·p(tc)·fixed − u·p(tc)·help ≤ work` (`cert_le`, `HelpWithinCertifier`,
`WordCostsMacs`), since `m·p(tc) ≥ 0` and `u·p(tc) ≥ 0`.
-/

@[expose] public section

namespace Accounting.SecurityProofs.Bridge

/-- The uniform probability of `E` on `A` is at most that of `F` on `X` when `e : A ≃ X` maps `E` into `F`. -/
theorem pr_le_pr_of_equiv {A X : Type} [Fintype A] [Fintype X] (e : A ≃ X) {E : A → Prop} {F : X → Prop}
    (h : ∀ a, E a → F (e a)) : Pouw.pr E ≤ Pouw.pr F := by
  classical
  unfold Pouw.pr Claims.Pr Claims.prUniform
  rw [Fintype.card_congr e]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have hmap : ((Finset.univ.filter E).map e.toEmbedding) ⊆ Finset.univ.filter F := by
    intro x hx
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Equiv.coe_toEmbedding] at hx ⊢
    obtain ⟨a, ha, rfl⟩ := hx
    exact h a ha
  have := Finset.card_le_card hmap
  rw [Finset.card_map] at this
  convert (Nat.cast_le (α := ℝ)).2 this using 3

/-- **The joint period's PoUW guarantee from a credit in words, at any price**
(`Properties.Accounting.JointBridge.WorkOfCreditedWords`). -/
theorem WorkOfCreditedWords : Properties.Accounting.JointBridge.WorkOfCreditedWords := by
  intro ι α I _ _ Ω _ d C r c p m u bUnit X _ acc fixed credit help δ real hcost hhelp hb ξ
  refine le_trans (pr_le_pr_of_equiv (F := fun x => acc ξ x ∧ fixed ξ x < credit ξ x) real.coin
    fun y ⟨hacc, hnot⟩ => ?_) (hb ξ)
  have hacc' := (real.accepts_iff ξ y).1 hacc
  refine ⟨hacc', lt_of_not_ge fun hle => hnot ?_⟩
  show c.W ≤ (r.work p (Function.update ξ (r.own .pouw) y) : ℝ)
  have hmp : (0 : ℝ) ≤ m * p .tc := mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have hup : (0 : ℝ) ≤ u * p .tc := mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  have hcert := real.cert_le ξ y hacc'
  have hwork := hcost ξ y hacc'
  have hcred : (m : ℝ) * p .tc * credit ξ (real.coin y) ≤ m * p .tc * fixed ξ (real.coin y) :=
    mul_le_mul_of_nonneg_left hle hmp
  have hunits : (u : ℝ) * p .tc * help ξ (real.coin y) ≤ u * p .tc * c.helpUnits bUnit :=
    mul_le_mul_of_nonneg_left (hhelp ξ y hacc') hup
  linarith

end Accounting.SecurityProofs.Bridge
