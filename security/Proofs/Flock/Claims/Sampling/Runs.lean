module

public import Proofs.Flock.Claims.Sampling.KS
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
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.Sampling.Cost

/-! The opening checks' count, and the reruns of a list's programs. -/

@[expose] public section

namespace Claims.Sampling

open Claims

theorem flatMapRP_runs_le {R β γ : Type} (H : Msg → Msg) (run : R → Transcript) (f : β → RP R (List γ)) (B : ℕ)
    (l : List β) (h : ∀ b ∈ l, ((f b).eval H run).2.2 ≤ B) : ((flatMapRP f l).eval H run).2.2 ≤ l.length * B := by
  rw [flatMapRP_runs]
  refine (List.sum_le_length_nsmul _ B fun n hn => ?_).trans_eq (by simp)
  obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hn
  exact h b hb

theorem ownLog_flatMapRP_le {R β γ : Type} (H : Msg → Msg) (run : R → Transcript) (f : β → RP R (List γ)) (B : ℕ)
    (l : List β) (h : ∀ b ∈ l, (ownLog H run (f b)).length ≤ B) :
    (ownLog H run (flatMapRP f l)).length ≤ l.length * B := by
  rw [ownLog_flatMapRP]
  refine (List.sum_le_length_nsmul _ B fun n hn => ?_).trans_eq (by simp)
  obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hn
  exact h b hb

theorem length_zip_finRange_le {α : Type} (k : ℕ) (l : List α) : ((List.finRange k).zip l).length ≤ k := by
  rw [List.length_zip, List.length_finRange]
  exact min_le_left _ _

theorem ownLog_checkRevealed_length {R : Type} (H : Msg → Msg) (run : R → Transcript) (Cm : CommitmentScheme Bool)
    (rs : List (List Msg × Opening Bool)) :
    (ownLog H run (checkRevealed (R := R) Cm rs)).length = (rs.map fun p => ((Cm.opens p.1 p.2).eval H).2.length).sum := by
  unfold checkRevealed
  rw [ownLog_flatMapRP]
  congr 1
  refine List.map_congr_left fun p _ => ?_
  rw [ownLog_bind, ownLog_ofOC, List.length_append]
  simp [ownLog]

section Runs

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

def OpenCallsLe (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x s, D.parse x = some s → ∀ u v wF,
    ((D.reveal s u v wF).map fun p => ((D.commit.opens p.1 p.2).eval H).2.length).sum ≤ D.openCalls x

/-- Stated at every `k`, so that the kernel never meets the deployed `k`'s `List.finRange`. -/
theorem outer_runs_le {R γ β : Type} (H : Msg → Msg) (run : R → Transcript) (k K B : ℕ) (rs : List R)
    (hrs : rs.length = K) (body : Fin k × (R × Seen) → RP R (List γ)) (fin : List γ → β)
    (hB : ∀ jo, ((body jo).eval H run).2.2 ≤ B) :
    (((waitAccC k rs []).bind fun outer =>
        (flatMapRP body ((List.finRange k).zip outer)).bind fun found => RP.ret (fin found)).eval H run).2.2 ≤
      K + k * B := by
  rw [(RP.eval_bind _ _ _ _).2, (RP.eval_bind _ _ _ _).2, waitAccC_runs_eq]
  have h1 := waitRuns_le_length run k rs 0
  have h2 := (flatMapRP_runs_le H run body B _ fun jo _ => hB jo).trans
    (Nat.mul_le_mul_right B (length_zip_finRange_le k ((waitAccC k rs []).eval H run).1))
  simp only [RP.eval, List.length_nil, Nat.add_zero]
  rw [hrs] at h1
  exact Nat.add_le_add h1 h2

end Runs

end Claims.Sampling
