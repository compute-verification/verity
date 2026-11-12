module

public import Definitions.Claims.SamplingGeneric
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
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.KS

/-!
# The sampled compiler's count form

`KSCount` from `KS` at any compiled sampled verifier (`Compiled.ksCount_of_ks`): `KS` at `δ = max(0, miss(t))`
(`missAt`), an incorrect set in `M_δ` having fewer than `t` proof blocks (`Sampling.Pa_le_miss`).
So `CompilerKS` gives `CompilerKSCount` (`compilerKSCount_of`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

theorem ksCount_of_ks {H : Msg → Msg} {L : Compiled Coin A} (h : L.KS H) : L.KSCount H := by
  intro t x hx Ω _ _ P
  have h0 : 0 ≤ L.missAt t x := by
    unfold Compiled.missAt
    split
    · exact le_max_left _ _
    · exact le_rfl
  have hk := h (L.missAt t x) h0 x hx Ω P
  unfold PrBad at hk ⊢
  refine le_trans (Pr_mono fun s hs => ⟨hs.1, fun hR => hs.2 ?_⟩) hk
  revert hR
  cases (L.E.eval H P x s.1 s.2.1 s.2.2).1 with
  | none => exact id
  | some w =>
    rintro ⟨s', Pt, hs, hPt, hM⟩
    refine ⟨s', Pt, hs, hPt, lt_of_not_ge fun ht => ?_⟩
    have hmx : L.missAt t x = max 0 (s'.design.miss Pt t) := by
      simp only [Compiled.missAt, hs, Compiled.part, hPt, Option.getD_some]
    have hM' : L.missAt t x < s'.design.Pa Pt (Sampling.faulty H L.commit s' Pt w) := hM
    rw [hmx] at hM'
    exact absurd ((Sampling.Pa_le_miss s'.design Pt (by classical exact Finset.filter_subset _ _) ht).trans
      (le_max_right _ _)) (not_le.2 hM')

end Compiled

theorem compilerKSCount_of (h : CompilerKS) : CompilerKSCount := by
  intro Coin _ _ _ A L H hA hB hS
  exact Compiled.ksCount_of_ks (h Coin A L H hA hB hS)

end Claims.SamplingGeneric
