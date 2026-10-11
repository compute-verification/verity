import Proofs.Flock.Soundness.Discharge.ZkBind.At
import Proofs.Flock.Soundness.Discharge.ZkLink.Sched
import Proofs.Flock.Soundness.Discharge.ZkLink.Compiled

/-!
# The planned rate of the `--zk` plan, at every draw

`rateZ ts j k = 2^{logLen₀}/(e·k)` reads only level 0's codeword length of table `j`'s schedule. Every table the one-table
`--zk` plan `planZAt dj y₀ S R` holds, at every draw `S` (a `DrawSetupZK`'s table, or `y₀`'s refused one), carries a
schedule `Zk.schedule` returned (`DrawSetupZK.hz`). So a cap `L` on level 0's `logLen` over everything `Zk.schedule`
returns (`ZkLogLenCap L`) bounds the rate at every draw, and the count curve's `∀ S' R' i, rateZ … ≤ ρ` follows from
`2^L ≤ ρ·e·k`:

* `logLen_tableAtZK_le`, `rateZ_planZAt_le`: the table's `logLen₀ ≤ L`, and its rate `≤ 2^L/(e·k)`, at every draw.
* `hr_planZAt`: `2^L ≤ ρ·(e·k)` gives `rateZ (planZAt dj y₀ S R) i k ≤ ρ` at every `dj`, `y₀`, `S`, `R`, `i`.
* `zkLogLenCap_23`: `Zk.schedule` accepts `m` from 25 to 35 only (`schedule_spec`), and fast100's level 0 there has rate
  1 and `m − 7 − k0` columns, so `logLen₀ = m − 6 − k0 ≤ 23` (checked by `decide +kernel` at each `m`).
-/

namespace FlockSoundness.Discharge.ZkBind

open Flock Accounting Refine ZkLink ZkExec
open FlockSoundness.Discharge.Exec (Inputs)

local notation "𝔽" => FlockLevel3.GF128

/-- **A cap on level 0's codeword length**: every schedule `Zk.schedule` returns has level 0's `logLen` at most `L`. -/
def ZkLogLenCap (L : ℕ) : Prop :=
  ∀ m (s : Flock.Schedule) (t : ℕ), Zk.schedule m = .ok (s, t) → (lvOf (schedOf s) 0).logLen ≤ L

/-- Level 0's codeword length of `Zk.schedule m`'s schedule (0 where it refuses `m`). -/
def zkLogLen₀ (m : ℕ) : ℕ :=
  match Zk.schedule m with
  | .ok (s, _) => (lvOf (schedOf s) 0).logLen
  | .error _ => 0

/-- At every `m` `Zk.schedule` accepts (25 to 35), level 0's `logLen` is at most 23, by the kernel's `decide`. -/
theorem zkLogLen₀_le : ∀ m, 25 ≤ m → m ≤ 35 → zkLogLen₀ m ≤ 23 := by
  intro m h1 h2; interval_cases m <;> decide +kernel

/-- **The existing cap**: `Zk.schedule`'s m range (25 to 35) caps level 0's `logLen` at 23. -/
theorem zkLogLenCap_23 : ZkLogLenCap 23 := by
  intro m s t h
  obtain ⟨h1, h2, -⟩ := schedule_spec h
  have hle := zkLogLen₀_le m h1 h2
  unfold zkLogLen₀ at hle
  rw [h] at hle
  exact hle

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) {Reg : Type}

/-- **The `--zk` table's `logLen₀` is under the cap at every draw**: the draw's `DrawSetupZK`'s schedule, or `y₀`'s,
is one `Zk.schedule` returned. -/
theorem logLen_tableAtZK_le {L : ℕ} (hL : ZkLogLenCap L) (S : Finset (Fin n)) :
    (lvOf (tableAtZK dj y₀ S).sch 0).logLen ≤ L := by
  unfold tableAtZK
  split_ifs with h
  · exact hL _ _ _ (Classical.choice h).hz
  · exact hL _ _ _ y₀.hz

/-- The one-table plan's table at every index is `tableAtZK`. -/
theorem planZAt_get (S : Finset (Fin n)) (R : Reg) (i : Fin (planZAt (Reg := Reg) dj y₀ S R).length) :
    (planZAt (Reg := Reg) dj y₀ S R).get i = tableAtZK dj y₀ S := by
  obtain ⟨i, hi⟩ := i
  have h0 : i = 0 := by simpa using hi
  subst h0
  rfl

/-- **The planned rate at every draw is at most `2^L/(e·k)`.** -/
theorem rateZ_planZAt_le {L : ℕ} (hL : ZkLogLenCap L) (S : Finset (Fin n)) (R : Reg)
    (i : Fin (planZAt (Reg := Reg) dj y₀ S R).length) (k : ℕ) :
    rateZ (planZAt (Reg := Reg) dj y₀ S R) i k ≤ 2 ^ L / (Real.exp 1 * k) := by
  unfold rateZ
  rw [planZAt_get dj y₀ S R i]
  exact div_le_div_of_nonneg_right (pow_le_pow_right₀ (by norm_num) (logLen_tableAtZK_le dj y₀ hL S))
    (by positivity)

/-- `2^L ≤ ρ·(e·k)` gives `2^L/(e·k) ≤ ρ`: the step from a cap's rate bound to the count curve's `ρ`. -/
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

/-- **The count curve bounds every planned rate**: under a cap `L`, `2^L ≤ ρ·(e·k)` gives `rateZ … ≤ ρ` at every draw
`S`, registration `R` and table `i`. `ZkHidden.hr_planZHJ` is the same at the hidden-output plan of `J` tables, which
the recursive audit's `_hr` reads. -/
theorem hr_planZAt {L k : ℕ} {ρ : ℝ} (hL : ZkLogLenCap L) (hkρ : (2 : ℝ) ^ L ≤ ρ * (Real.exp 1 * k))
    (S : Finset (Fin n)) (R : Reg) (i : Fin (planZAt (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZAt (Reg := Reg) dj y₀ S R) i k ≤ ρ :=
  (rateZ_planZAt_le dj y₀ hL S R i k).trans (cap_div_le hkρ)

/-- `hr_planZAt` at the existing cap: `2^23 ≤ ρ·(e·k)` bounds every planned rate by `ρ`. -/
theorem hr_planZAt_23 {k : ℕ} {ρ : ℝ} (hkρ : (2 : ℝ) ^ 23 ≤ ρ * (Real.exp 1 * k))
    (S : Finset (Fin n)) (R : Reg) (i : Fin (planZAt (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZAt (Reg := Reg) dj y₀ S R) i k ≤ ρ :=
  hr_planZAt dj y₀ zkLogLenCap_23 hkρ S R i

end FlockSoundness.Discharge.ZkBind
