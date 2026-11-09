module

public import Definitions.Accounting.Guarantees.Actual.Dampener
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
import Proofs.Accounting.Actual.Threaded
import Proofs.Nci.Fit
import Proofs.Nci.Threads

/-!
# The compute dampener's windows from the memory cap over threads

`ThreadedRuntimeSlots` at `r = αF`, with `M_r + M_d ≤ |T.M|`, `⌈t/ω⌉₊ = (t + ω − 1)/ω` (`ceil_div_eq_pred_div`), and
`Λ`, which is `Threading.bound` by definition; `SlowdownWindowsCap` is its instance at one slot per weight gate.
-/

@[expose] public section

namespace Accounting.SecurityProofs.Actual

open Verity Nci Finset Nci.Dampener

/-- **The windows' count is the ceiling**: `⌈t/ω⌉₊ = (t + ω − 1)/ω` for a whole-tick window `ω > 0`. -/
theorem ceil_div_eq_pred_div {t ω : ℕ} (hω : 0 < ω) : ⌈(t : ℝ) / ω⌉₊ = (t + ω - 1) / ω := by
  have hωr : (0 : ℝ) < ω := Nat.cast_pos.2 hω
  refine le_antisymm (Nat.ceil_le.2 ?_) ?_
  · have h1 : t ≤ (t + ω - 1) / ω * ω := by
      have := Nat.lt_div_mul_add (a := t + ω - 1) hω
      omega
    rw [div_le_iff₀ hωr]
    exact_mod_cast h1
  · have h1 : (t : ℝ) ≤ ⌈(t : ℝ) / ω⌉₊ * ω := (div_le_iff₀ hωr).1 (Nat.le_ceil _)
    have h2 : t ≤ ⌈(t : ℝ) / ω⌉₊ * ω := by exact_mod_cast h1
    have h3 : (t + ω - 1) / ω < ⌈(t : ℝ) / ω⌉₊ + 1 := by
      rw [Nat.div_lt_iff_lt_mul hω, Nat.add_mul, Nat.one_mul]
      omega
    omega

theorem SlowdownWindowsCapSlots : Guarantees.Actual.SlowdownWindowsCapSlots := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω Γ R F α b_w b_a c T σ d e Mr Md hω hX hC hΦ hL hbw hba hd hΓ hr hR
    hcont hT hW hM
  obtain ⟨h, -⟩ := ThreadedRuntimeSlots A ℓ Φ ends β link n X θ hθ t ω Γ R (α * F) b_w b_a c T σ d e hω hX hC hΦ hL
    hbw hba hd hΓ hr hR hcont hT hW
  have hb : T.bound (R + α * F * ω) b_w d e = Lambda T.reuse e d b_w (R + α * F * ω) := rfl
  calc b_a * ((Mr : ℝ) + Md) ≤ b_a * T.M.card := mul_le_mul_of_nonneg_left (by exact_mod_cast hM) hba.le
    _ ≤ ((n * ((t + ω - 1) / ω) : ℕ) : ℝ) * ((R + α * F * ω) * T.bound (R + α * F * ω) b_w d e) := h
    _ = n * ⌈(t : ℝ) / ω⌉₊ * ((R + α * F * ω) * Lambda T.reuse e d b_w (R + α * F * ω)) := by
        rw [ceil_div_eq_pred_div hω, Nat.cast_mul, hb]

/-- `SlowdownWindowsCapSlots` at one slot per weight gate. -/
theorem SlowdownWindowsCap : Guarantees.Actual.SlowdownWindowsCap := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω Γ R F α b_w b_a c T d e Mr Md hω hX hC hΦ hL hbw hba hd hΓ hr hR
    hcont hT hW hM
  exact SlowdownWindowsCapSlots A ℓ Φ ends β link n X θ hθ t ω Γ R F α b_w b_a c T (Nci.SecurityProofs.Fit.idSlots T)
    d e Mr Md hω hX hC hΦ hL hbw hba hd hΓ hr hR hcont hT
    (Nci.SecurityProofs.weightsIncompressibleSlots_of (Nci.SecurityProofs.Fit.idSlots T) hbw.le hW) hM

end Accounting.SecurityProofs.Actual
