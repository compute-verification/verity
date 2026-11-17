module

public import Definitions.Pouw.HMu
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
import Mathlib.RingTheory.WittVector.IsPoly
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Pouw.HMu.Luck
import Std.Tactic.BVDecide.Normalize.Prop

/-! Why the coins don't give the rest of (H-μ⁺): a view that reads the coin is lucky on every deficient round, and
two words in one unit cancel under a step injective in both arguments. -/

@[expose] public section

namespace ResourceAccounting.HMuProofs

open ResourceAccounting.HMu Pouw.RelativeSecurity Finset

noncomputable def readsCoinRound : SkipRound Bool Bool ℤ ℤ where
  view := id
  skipped := fun e => if e then 1 else 0
  d := fun o => if o then 1 else 0
  committed := fun _ => 1
  honest := fun _ x => x

theorem readsCoin :
    prob readsCoinRound.lucky = prob (fun ω => 0 < readsCoinRound.d (readsCoinRound.view ω)) ∧
      prob (fun ω => 0 < readsCoinRound.d (readsCoinRound.view ω)) = 1 / 2 := by
  have hd : ∀ ω, 0 < readsCoinRound.d (readsCoinRound.view ω) ↔ ω = true := by
    intro ω; cases ω <;> simp [readsCoinRound]
  refine ⟨prob_congr fun ω => ?_, ?_⟩
  · cases ω <;> simp [SkipRound.lucky, readsCoinRound]
  · rw [prob_congr hd, prob_eq_card, show (univ.filter fun ω : Bool => ω = true).card = 1 by decide]
    norm_num

theorem twoCopies :
    (∀ n x, Function.Injective (fun s : ZMod 2 => (fun _ s x => s + x : ℕ → ZMod 2 → ZMod 2 → ZMod 2) n s x)) ∧
      (∀ n s, Function.Injective ((fun _ s x => s + x : ℕ → ZMod 2 → ZMod 2 → ZMod 2) n s)) ∧
      ¬Function.Injective (fun X : ZMod 2 => chain (fun _ s x => s + x) 0 (persist (fun _ => 0) {0, 1} X) 2) := by
  refine ⟨fun n x a b h => add_right_cancel h, fun n s a b h => add_left_cancel h, fun h => ?_⟩
  have := @h 0 1 (by simp [chain, persist]; decide)
  exact absurd this (by decide)

end ResourceAccounting.HMuProofs
