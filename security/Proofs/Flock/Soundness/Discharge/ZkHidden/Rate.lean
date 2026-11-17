module

public import Proofs.Flock.Soundness.Discharge.ZkBind.Rate
public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
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

/-! The planned rate of the hidden-output `--zk` plan of `J` tables, at every sample. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open ZkLink ZkBind Composed
open FlockSoundness.Discharge.Exec (Inputs)

variable {n mPts : ℕ} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) {Reg : Type}

theorem logLen_tableZHJ_le {L : ℕ} (hL : ZkLogLenCap L) (S : Finset (Fin n)) (j : Fin (nTab I)) :
    (lvOf (tableZHJ dj y₀ S j).sch 0).logLen ≤ L := by
  unfold tableZHJ
  split_ifs with h
  · exact hL _ _ _ _ (Classical.choice h).hz
  · exact hL _ _ _ _ y₀.hz

theorem planZHJ_get (S : Finset (Fin n)) (R : Reg) (i : Fin (planZHJ (Reg := Reg) dj y₀ S R).length) :
    (planZHJ (Reg := Reg) dj y₀ S R).get i = tableZHJ dj y₀ S (Fin.cast (planZHJ_length dj y₀ S R) i) :=
  List.get_ofFn _ _

theorem rateZ_planZHJ_le {L : ℕ} (hL : ZkLogLenCap L) (S : Finset (Fin n)) (R : Reg)
    (i : Fin (planZHJ (Reg := Reg) dj y₀ S R).length) (k : ℕ) :
    rateZ (planZHJ (Reg := Reg) dj y₀ S R) i k ≤ 2 ^ L / (Real.exp 1 * k) := by
  unfold rateZ
  rw [planZHJ_get dj y₀ S R i]
  exact div_le_div_of_nonneg_right (pow_le_pow_right₀ (by norm_num) (logLen_tableZHJ_le dj y₀ hL S _))
    (by positivity)

theorem hr_planZHJ {L k : ℕ} {ρ : ℝ} (hL : ZkLogLenCap L) (hkρ : (2 : ℝ) ^ L ≤ ρ * (Real.exp 1 * k))
    (S : Finset (Fin n)) (R : Reg) (i : Fin (planZHJ (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZHJ (Reg := Reg) dj y₀ S R) i k ≤ ρ :=
  (rateZ_planZHJ_le dj y₀ hL S R i k).trans (cap_div_le hkρ)

theorem hr_planZHJ_23 {k : ℕ} {ρ : ℝ} (hkρ : (2 : ℝ) ^ 23 ≤ ρ * (Real.exp 1 * k))
    (S : Finset (Fin n)) (R : Reg) (i : Fin (planZHJ (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZHJ (Reg := Reg) dj y₀ S R) i k ≤ ρ :=
  hr_planZHJ dj y₀ zkLogLenCap_23 hkρ S R i

end FlockSoundness.Discharge.ZkHidden
