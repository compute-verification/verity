import Proofs.Flock.Soundness.Discharge.ZkBind.Rate
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ

/-!
# The planned rate of the hidden-output `--zk` plan of `J` tables, at every draw

`ZkBind/Rate.lean` at `planZHJ dj y₀ S R = List.ofFn (tableZHJ dj y₀ S)`, the plan `PublicAudit`'s rate premise `_hr`
reads. Each of its tables, at every draw `S` (table `j`'s `DrawSetupZHJ`'s table, or `y₀`'s refused one), carries a
schedule `Zk.schedule` returned (`DrawSetupZHJ.hz`), so a cap `L` (`ZkBind.ZkLogLenCap L`) bounds every table's rate:

* `logLen_tableZHJ_le`, `rateZ_planZHJ_le`: table `j`'s `logLen₀ ≤ L`, and the plan's rate `≤ 2^L/(e·k)` at every table.
* `hr_planZHJ`: `2^L ≤ ρ·(e·k)` gives `rateZ (planZHJ dj y₀ S R) i k ≤ ρ` at every `dj`, `y₀`, `S`, `R`, `i`.
* `hr_planZHJ_23`: the same at `ZkBind.zkLogLenCap_23`, the cap `Zk.schedule`'s m range puts on level 0's `logLen`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open ZkLink ZkBind Composed
open FlockSoundness.Discharge.Exec (Inputs)

variable {n mPts : ℕ} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {Reg : Type}

/-- **Table `j`'s `logLen₀` is under the cap at every draw**: its `DrawSetupZHJ`'s schedule, or `y₀`'s, is one
`Zk.schedule` returned. -/
theorem logLen_tableZHJ_le {L : ℕ} (hL : ZkLogLenCap L) (S : Finset (Fin n)) (j : Fin (nTab I)) :
    (lvOf (tableZHJ dj y₀ S j).sch 0).logLen ≤ L := by
  unfold tableZHJ
  split_ifs with h
  · exact hL _ _ _ (Classical.choice h).hz
  · exact hL _ _ _ y₀.hz

/-- The plan's table at index `i` is `tableZHJ` at `i`. -/
theorem planZHJ_get (S : Finset (Fin n)) (R : Reg) (i : Fin (planZHJ (Reg := Reg) dj y₀ S R).length) :
    (planZHJ (Reg := Reg) dj y₀ S R).get i = tableZHJ dj y₀ S (Fin.cast (planZHJ_length dj y₀ S R) i) :=
  List.get_ofFn _ _

/-- **The planned rate at every draw and table is at most `2^L/(e·k)`.** -/
theorem rateZ_planZHJ_le {L : ℕ} (hL : ZkLogLenCap L) (S : Finset (Fin n)) (R : Reg)
    (i : Fin (planZHJ (Reg := Reg) dj y₀ S R).length) (k : ℕ) :
    rateZ (planZHJ (Reg := Reg) dj y₀ S R) i k ≤ 2 ^ L / (Real.exp 1 * k) := by
  unfold rateZ
  rw [planZHJ_get dj y₀ S R i]
  exact div_le_div_of_nonneg_right (pow_le_pow_right₀ (by norm_num) (logLen_tableZHJ_le dj y₀ hL S _))
    (by positivity)

/-- **The count curve bounds every planned rate**: under a cap `L`, `2^L ≤ ρ·(e·k)` gives `rateZ … ≤ ρ` at every draw
`S`, registration `R` and table `i`. -/
theorem hr_planZHJ {L k : ℕ} {ρ : ℝ} (hL : ZkLogLenCap L) (hkρ : (2 : ℝ) ^ L ≤ ρ * (Real.exp 1 * k))
    (S : Finset (Fin n)) (R : Reg) (i : Fin (planZHJ (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZHJ (Reg := Reg) dj y₀ S R) i k ≤ ρ :=
  (rateZ_planZHJ_le dj y₀ hL S R i k).trans (cap_div_le hkρ)

/-- `hr_planZHJ` at the existing cap: `2^23 ≤ ρ·(e·k)` bounds every planned rate by `ρ`. -/
theorem hr_planZHJ_23 {k : ℕ} {ρ : ℝ} (hkρ : (2 : ℝ) ^ 23 ≤ ρ * (Real.exp 1 * k))
    (S : Finset (Fin n)) (R : Reg) (i : Fin (planZHJ (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZHJ (Reg := Reg) dj y₀ S R) i k ≤ ρ :=
  hr_planZHJ dj y₀ zkLogLenCap_23 hkρ S R i

end FlockSoundness.Discharge.ZkHidden
