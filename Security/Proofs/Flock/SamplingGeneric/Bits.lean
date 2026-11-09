module

public import Definitions.Claims.SamplingGeneric
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Star.Real
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
import Proofs.Flock.SamplingGeneric.Fails

/-!
# The sampled compiler's bits

`Bits` from `KS` at any compiled sampled verifier whose extractor's expected reruns and own SHA-512 calls are
`Efficient`'s (`Compiled.bits_of_ks`, `Sampling.Bits_of_KS` generic, through `Sampling.Bits_of_KSLin`). Those two
bounds are `CompilerEfficient`'s second and third conjuncts, and `CompilerBits` takes `CompilerEfficient`'s premises,
so `CompilerKS` and `CompilerEfficient` give `CompilerBits` (`compilerBits_of`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem κ_pos (L : Compiled Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (x : Msg) : 0 < L.κ δ x := by
  have hs : 0 < Sampling.slack := by unfold Sampling.slack; positivity
  have h1 : 0 ≤ Sampling.slack ^ 2 * (δ + L.κs x) :=
    mul_nonneg (pow_nonneg hs.le 2) (add_nonneg hδ (maxBelow_nonneg _ _))
  have h2 : 0 < 2 * (L.kOut x : ℝ) / L.capOut x + Real.exp (-(L.kOut x : ℝ) / 8) := by positivity
  have h3 : 0 ≤ Sampling.slack * (2 * (L.kIn x : ℝ) / L.capIn x + Real.exp (-(L.kIn x : ℝ) / 8)) :=
    mul_nonneg hs.le (by positivity)
  unfold Compiled.κ
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem γ_nonneg (L : Compiled Coin A) (x : Msg) : 0 ≤ L.γ x := by
  have hs : 0 < Sampling.slack := by unfold Sampling.slack; positivity
  unfold Compiled.γ
  exact mul_nonneg (pow_nonneg hs.le 4) (mul_nonneg (Nat.cast_nonneg _) (maxBelow_nonneg _ _))

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem work_pos (L : Compiled Coin A) (x : Msg) : 0 < L.work x := by
  have h1 : 0 ≤ (L.kOut x : ℝ) * L.kIn x * (1 + L.nPU x * L.es x) :=
    mul_nonneg (by positivity) (add_nonneg zero_le_one (mul_nonneg (Nat.cast_nonneg _) (maxBelow_nonneg _ _)))
  have h2 : 0 ≤ (L.kOut x : ℝ) * L.kIn x * L.nPU x * L.openCalls x := by positivity
  have h3 : (0 : ℝ) ≤ L.kOut x := Nat.cast_nonneg _
  unfold Compiled.work
  linarith

theorem bits_of_ks {H : Msg → Msg} {L : Compiled Coin A} (hks : L.KS H)
    (hruns : ExtractRunsExpLe H L.E (fun x => 1 + L.kOut x + (L.kOut x : ℝ) * L.kIn x * (1 + L.nPU x * L.es x)))
    (hown : Sampling.ExtractOwnLe H L.E (fun x => L.kOut x * L.kIn x * L.nPU x * L.openCalls x)) :
    L.Bits H := fun δ hδ =>
  Sampling.Bits_of_KSLin (hks δ hδ) hruns hown (fun x => L.κ_pos hδ x) L.γ_nonneg L.work_pos
    fun x => le_of_eq (by unfold Compiled.work; push_cast; ring)

end Compiled

theorem compilerBits_of (hks : CompilerKS) (heff : CompilerEfficient) : CompilerBits := by
  intro Coin _ _ _ A L H hA hB hS hopen
  obtain ⟨-, hr, ho⟩ := heff Coin A L H hS hopen
  exact Compiled.bits_of_ks (hks Coin A L H hA hB hS) hr ho

end Claims.SamplingGeneric
