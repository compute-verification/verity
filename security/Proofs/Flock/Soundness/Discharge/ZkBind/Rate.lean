module

public import Proofs.Flock.Soundness.Discharge.ZkBind.At
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
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
import Proofs.Flock.Soundness.Discharge.ZkLink.Sched

/-! The planned rate of the `--zk` plan, bounded at every sample by a cap on level 0's codeword length
(`hr_planZAt`). -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkBind

open Flock Accounting Refine ZkLink ZkExec
open FlockSoundness.Discharge.Exec (Inputs)

local notation "𝔽" => FlockLevel3.GF128

variable {wideLeaf : Bool}

def ZkLogLenCap (L : ℕ) : Prop :=
  ∀ wideLeaf m (s : Flock.Schedule) (t : ℕ), Zk.schedule wideLeaf m = .ok (s, t) → (lvOf (schedOf s) 0).logLen ≤ L

def zkLogLen₀ (wideLeaf : Bool) (m : ℕ) : ℕ :=
  match Zk.schedule wideLeaf m with
  | .ok (s, _) => (lvOf (schedOf s) 0).logLen
  | .error _ => 0

theorem zkLogLen₀_le : ∀ m, 25 ≤ m → m ≤ 35 → zkLogLen₀ wideLeaf m ≤ 23 := by
  intro m h1 h2; cases wideLeaf <;> interval_cases m <;> decide +kernel

theorem zkLogLenCap_23 : ZkLogLenCap 23 := by
  intro wideLeaf m s t h
  obtain ⟨h1, h2, -⟩ := schedule_spec h
  have hle := zkLogLen₀_le (wideLeaf := wideLeaf) m h1 h2
  unfold zkLogLen₀ at hle
  rw [h] at hle
  exact hle

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : SampleSetupZK I mPts S₀ (dj S₀)) {Reg : Type}

theorem logLen_tableAtZK_le {L : ℕ} (hL : ZkLogLenCap L) (S : Finset (Fin n)) :
    (lvOf (tableAtZK dj y₀ S).sch 0).logLen ≤ L := by
  unfold tableAtZK
  split_ifs with h
  · exact hL _ _ _ _ (Classical.choice h).hz
  · exact hL _ _ _ _ y₀.hz

theorem planZAt_get (S : Finset (Fin n)) (R : Reg) (i : Fin (planZAt (Reg := Reg) dj y₀ S R).length) :
    (planZAt (Reg := Reg) dj y₀ S R).get i = tableAtZK dj y₀ S := by
  obtain ⟨i, hi⟩ := i
  have h0 : i = 0 := by simpa using hi
  subst h0
  rfl

theorem rateZ_planZAt_le {L : ℕ} (hL : ZkLogLenCap L) (S : Finset (Fin n)) (R : Reg)
    (i : Fin (planZAt (Reg := Reg) dj y₀ S R).length) (k : ℕ) :
    rateZ (planZAt (Reg := Reg) dj y₀ S R) i k ≤ 2 ^ L / (Real.exp 1 * k) := by
  unfold rateZ
  rw [planZAt_get dj y₀ S R i]
  exact div_le_div_of_nonneg_right (pow_le_pow_right₀ (by norm_num) (logLen_tableAtZK_le dj y₀ hL S))
    (by positivity)

theorem cap_div_le {L k : ℕ} {ρ : ℝ} (hkρ : (2 : ℝ) ^ L ≤ ρ * (Real.exp 1 * k)) :
    (2 : ℝ) ^ L / (Real.exp 1 * k) ≤ ρ := by
  have hpos : 0 < Real.exp 1 * k := by
    rcases Nat.eq_zero_or_pos k with rfl | hk
    · have h2 : (0 : ℝ) < 2 ^ L := by positivity
      have h0 : ρ * (Real.exp 1 * ((0 : ℕ) : ℝ)) = 0 := by simp
      linarith
    · positivity
  rw [div_le_iff₀ hpos]
  exact hkρ

theorem hr_planZAt {L k : ℕ} {ρ : ℝ} (hL : ZkLogLenCap L) (hkρ : (2 : ℝ) ^ L ≤ ρ * (Real.exp 1 * k))
    (S : Finset (Fin n)) (R : Reg) (i : Fin (planZAt (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZAt (Reg := Reg) dj y₀ S R) i k ≤ ρ :=
  (rateZ_planZAt_le dj y₀ hL S R i k).trans (cap_div_le hkρ)

theorem hr_planZAt_23 {k : ℕ} {ρ : ℝ} (hkρ : (2 : ℝ) ^ 23 ≤ ρ * (Real.exp 1 * k))
    (S : Finset (Fin n)) (R : Reg) (i : Fin (planZAt (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZAt (Reg := Reg) dj y₀ S R) i k ≤ ρ :=
  hr_planZAt dj y₀ zkLogLenCap_23 hkρ S R i

end FlockSoundness.Discharge.ZkBind
