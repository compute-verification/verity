module

public import Proofs.Flock.Claims.Witnesses.Flock
public import Definitions.Claims.SamplingTable
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Algebra.Order.Star.Real
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Data.NNRat.Floor
import Mathlib.Tactic.NormNum.RealSqrt

/-!
# Refutation witness: Flock's knowledge soundness at one table of a session

`Claims.Flock.KSTable` fails at the toy record of `KS` (`ksD`: the verifier accepts after one round, no statement has a
table, `m′ = 4` waits and `K = 1024` reruns), at tables whose words have length `1` (`ksN`) and whose errors are `0`:
there `1 ≤ k_0 ≤ 2`, so `κ_0 ≤ 1/16 + 4/1024 + e^{−1/8} < 1`, while the silent prover is accepted surely and no
extraction returns a witness of table `0`, which no statement has. `Claims.Flock.KSTable_fails` states it over the spec,
quantifying over the record, the lengths and the errors it fails at.
-/

@[expose] public section

namespace Claims.Witness.Flock

open Claims

/-- The toy lengths: every table's word has length `1`. -/
def ksN : Msg → ℕ → ℕ := fun _ _ => 1

theorem ceil_four_div_e_pos : 1 ≤ ⌈4 * ((1 : ℕ) : ℝ) / Real.exp 1⌉₊ :=
  Nat.one_le_iff_ne_zero.2 (Nat.ceil_pos.2 (div_pos (by norm_num) (Real.exp_pos 1))).ne'

theorem ceil_four_div_e_le : ⌈4 * ((1 : ℕ) : ℝ) / Real.exp 1⌉₊ ≤ 2 := by
  have h2 : (1 : ℝ) + 1 ≤ Real.exp 1 := Real.add_one_le_exp 1
  refine Nat.ceil_le.2 ?_
  rw [div_le_iff₀ (Real.exp_pos 1)]
  push_cast
  linarith

theorem ksN_kTable_pos (j : ℕ) (x : Msg) : 1 ≤ Claims.Flock.kTable ksN j x := ceil_four_div_e_pos

theorem ksN_kTable_le (j : ℕ) (x : Msg) : Claims.Flock.kTable ksN j x ≤ 2 := ceil_four_div_e_le

theorem exp_neg_eighth_le : Real.exp (-(1 / 8 : ℝ)) ≤ 8 / 9 := by
  have h8 : (1 / 8 : ℝ) + 1 ≤ Real.exp (1 / 8) := Real.add_one_le_exp (1 / 8)
  rw [Real.exp_neg]
  calc (Real.exp (1 / 8))⁻¹ ≤ ((1 / 8 : ℝ) + 1)⁻¹ := inv_anti₀ (by norm_num) h8
    _ = 8 / 9 := by norm_num

section Numbers

variable {Coin : Type}

theorem ksD_κTable (j : ℕ) (x : Msg) : Claims.Flock.κTable (ksD Coin) ksN (fun _ _ => 0) j x < 1 := by
  have h1 : (1 : ℝ) ≤ (Claims.Flock.kTable ksN j x : ℝ) := by exact_mod_cast ksN_kTable_pos j x
  have h2 : (Claims.Flock.kTable ksN j x : ℝ) ≤ 2 := by exact_mod_cast ksN_kTable_le j x
  have he : Real.exp (-(Claims.Flock.kTable ksN j x : ℝ) / 8) ≤ 8 / 9 :=
    (Real.exp_le_exp.2 (by linarith)).trans exp_neg_eighth_le
  have hw : (2 : ℝ) ^ (-((4 : ℕ) : ℝ)) = 1 / 16 := by
    rw [Real.rpow_neg (by norm_num), Real.rpow_natCast]; norm_num
  unfold Claims.Flock.κTable
  simp only [ksD] at hw ⊢
  rw [hw]
  push_cast at he ⊢
  linarith

theorem ksD_noRTable (j : ℕ) (x : Msg) (w : Claims.Flock.Wit) : ¬ Claims.Flock.RTable (ksD Coin).table j x w := by
  rintro ⟨t, ht, -⟩
  simp [ksD] at ht

end Numbers

section Claims

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **Flock's `KSTable` fails** at `ksD`, at words of length `1` and errors `0`, at table `0`. -/
theorem KSTable_fails (H : Msg → Msg) : ¬ Claims.Flock.KSTable H (ksD Coin) ksN (fun _ _ => 0) := fun h =>
  Witness.KS_fails H _ _ _ _ [] (ksD_noRTable 0 []) (ksD_κTable (Coin := Coin) 0 [])
    (h _ (listWithin_unit _ _ _) 0)

end Claims

end Claims.Witness.Flock

namespace Claims

/-- **Flock's `KSTable` fails** at some record, lengths and errors. -/
theorem Flock.KSTable_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (D : Claims.Flock.Deploy Coin Unit Unit) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ),
      ¬ Claims.Flock.KSTable H D N ε :=
  ⟨_, _, _, Witness.Flock.KSTable_fails H⟩

end Claims
