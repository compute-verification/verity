import Proofs.Flock.Claims.ZK.SaltProver

/-!
# Levels ≥ 1's salts at the executable's indices

A Ligerito prover of levels ≥ 1 (`Composed.LigProver`) gets rep `r`'s levels ≥ 1 salts as one array `HiOf Tz`, and the
masked prover reads them by level and position (`ligeritoPZk`'s `salt l q`, `1 ≤ l < |levels|`, `q < 2^d`). Here the
array is read at `hiIdx` (`hiSalt`), the order `TapeLayout.exe` lays them out in, so the later rounds' prover reads the
tape's salts.

* `hiSalt_exe`: at the executable's layout, the salts the model hands table `j`'s rep `r` (`Base`'s last field, as
  `Coined.randOf` and the view read it) are, at every level `l ≥ 1` and position `p` of its tree, the masked prover's
  tape's `salt j r l p` (`exeTape`); `hiSalt_exe_eq` as functions, empty off the schedule.
* `ofSalts_exe_on`: so at the executable's layout, `ofSalts Pc` sends what `Pc` sends on the tape's salts on the
  schedule, empty off it; on the tape's salts once `Pc` reads only the schedule's levels ≥ 1 and their positions
  (`ofSalts_exe`, `SaltsOn`).
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Masked

theorem hiOff_length (ls : List FlockSoundness.Accounting.Level) :
    hiOff ls ls.length = (ls.tail.map fun lv => 2 ^ lv.logLen).sum := by
  unfold hiOff
  rw [List.take_of_length_le (by simp)]

theorem length_le_sum (ls : List FlockSoundness.Accounting.Level) :
    ls.length ≤ (ls.map fun lv => 2 ^ lv.logLen).sum := by
  induction ls with
  | nil => simp
  | cons a l ih =>
    simp only [List.length_cons, List.map_cons, List.sum_cons]
    have := Nat.two_pow_pos a.logLen
    omega

/-- **Level `l`'s salt at position `p` is in the levels ≥ 1's array**: before the levels after it. -/
theorem hiOff_add_lt {ls : List FlockSoundness.Accounting.Level} {l p : ℕ} (h : HiAt ls l p) :
    hiOff ls l + p < (ls.tail.map fun lv => 2 ^ lv.logLen).sum := by
  obtain ⟨h1, hl, hp⟩ := h
  have e := hiOff_succ ls h1 hl
  have m := hiOff_mono ls (show l + 1 ≤ ls.length by omega)
  rw [hiOff_length] at m
  omega

theorem le_sum_tail {ls : List FlockSoundness.Accounting.Level} {l p : ℕ} (h : HiAt ls l p) :
    l ≤ (ls.tail.map fun lv => 2 ^ lv.logLen).sum := by
  have := length_le_sum ls.tail
  rw [List.length_tail] at this
  have := h.2.1
  omega

/-- **Table `j`'s rep `r` levels ≥ 1 salts at the executable's layout**: `Base`'s last field of its randomness, what
`Coined.randOf` hands `lig`. -/
noncomputable abbrev hiTape (P : Composed.LigProver) (x : Msg) (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (j : Fin (jOf x)) (r : Fin 2) : Fin (Composed.hiCount (zkTab x j).table) → ZkView.Salt :=
  (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j).2.2.2.2.2.2.2.2.2 r

/-- `exeRead`'s levels ≥ 1 salts: the cell's, by level and position or at a leftover place, at `ιH`. -/
theorem exeRead_hi {z : ZkView.Sizes} (hz : Fits z zMax) {h : ℕ}
    (ιH : Fin h ↪ (Fin (schMax + 1) × Fin schMax) ⊕ Fin schMax) (c : ExeCell) (r : Fin 2) :
    (exeRead hz ιH c).1.2.2.2.2.2.2.2.2.2 r = Sum.elim (c.1.2.2.2.2.2.2.2.2.2 r).1 (c.1.2.2.2.2.2.2.2.2.2 r).2 ∘ ιH :=
  rfl

/-- **Table `j`'s rep `r` levels ≥ 1 salts at the executable's layout** are cell `j`'s at `hiEmb`. Each side unfolds to
the other by the definitions alone, so the kernel never compares `schMax` with another size. -/
theorem hiTape_eq (P : Composed.LigProver) (x : Msg) (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (j : Fin (jOf x)) (r : Fin 2) :
    hiTape P x t j r = Sum.elim ((t (Fin.castLE (jOf_le_cap x) j)).1.2.2.2.2.2.2.2.2.2 r).1
      ((t (Fin.castLE (jOf_le_cap x) j)).1.2.2.2.2.2.2.2.2.2 r).2 ∘ hiEmb x j :=
  exeRead_hi (fits_zkTab P (zkTab x j)) (hiEmb x j) (t (Fin.castLE (jOf_le_cap x) j)) r

/-- **The salts the model hands levels ≥ 1 are the executable's**: at the executable's layout, table `j`'s rep `r`
salt at level `l ≥ 1` and position `p` of its tree, read at `hiIdx`, is the masked prover's tape's `salt j r l p`. -/
theorem hiSalt_exe (P : Composed.LigProver) (x : Msg) (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (j : Fin (jOf x)) (r : Fin 2) {l p : ℕ} (h : HiAt (zkTab x j).table.sch.levels l p) :
    hiSalt (zkTab x j).table.sch.levels (hiTape P x t j r) l p = (exeTape t).salt j.val r.val l p := by
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hn : Composed.hiCount (zkTab x j).table ≤ schMax := hi_le (zkTab x j)
  have hk : hiOff (zkTab x j).table.sch.levels l + p < Composed.hiCount (zkTab x j).table := by
    have := hiOff_add_lt h
    unfold Composed.hiCount
    omega
  have hl' : l < schMax + 1 := by
    have := le_sum_tail h
    unfold Composed.hiCount at hn
    omega
  have hp' : p < schMax := by omega
  have hq : HiV (zkTab x j).table.sch.levels (⟨l, hl'⟩, ⟨p, hp'⟩) := h
  have hl0 : l ≠ 0 := by have := h.1; omega
  have he : hiEmb x j ⟨hiIdx (zkTab x j).table.sch.levels (⟨l, hl'⟩, ⟨p, hp'⟩), hk⟩ = .inl (⟨l, hl'⟩, ⟨p, hp'⟩) :=
    hiEmbOf_hiIdx _ hn hq hk
  have hR : saltAt (cellAt t j.val) r.val l p =
      ZkView.saltBytes (((t (Fin.castLE (jOf_le_cap x) j)).1.2.2.2.2.2.2.2.2.2 r).1 (⟨l, hl'⟩, ⟨p, hp'⟩)) := by
    rw [cellAt_lt t hc]
    simp only [saltAt, fAt, hl0, r.isLt, hl', hp', ↓reduceIte, ↓reduceDIte]
    rfl
  unfold hiSalt
  rw [ite_eq_left h, fAt_lt _ _ hk]
  show ZkView.saltBytes (hiTape P x t j r ⟨hiIdx (zkTab x j).table.sch.levels (⟨l, hl'⟩, ⟨p, hp'⟩), hk⟩) = _
  rw [hiTape_eq, Function.comp_apply, he, Sum.elim_inl]
  exact hR.symm

/-- `hiSalt_exe` as functions: the tape's salts on the schedule's levels ≥ 1 and their positions, empty off them. -/
theorem hiSalt_exe_eq (P : Composed.LigProver) (x : Msg) (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (j : Fin (jOf x)) (r : Fin 2) :
    hiSalt (zkTab x j).table.sch.levels (hiTape P x t j r) = fun l p =>
      if HiAt (zkTab x j).table.sch.levels l p then (exeTape t).salt j.val r.val l p else ByteArray.empty := by
  funext l p
  split_ifs with h
  · exact hiSalt_exe P x t j r h
  · unfold hiSalt
    rw [ite_eq_right h]

/-- **At the executable's layout the later rounds' prover reads the tape's salts on the schedule**: `ofSalts Pc` on table
`j`'s rep `r` salts sends what `Pc` sends on the masked prover's tape's (`exeTape`'s `salt j r`) at the schedule's levels
≥ 1 and their positions, empty off them. -/
theorem ofSalts_exe_on (Pc : SaltProver) (x : Msg) (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (j : Fin (jOf x)) (c : Composed.CoinsZK (Composed.RestOf (zkTab x j).table (stAt x j))) (r : Fin 2) (L : ℕ)
    (y : Fin L → 𝕂) :
    ofSalts Pc (zkTab x j).table (stAt x j) c r L y (hiTape (ofSalts Pc) x t j r) =
      Pc (zkTab x j).table (stAt x j) c r L y fun l p =>
        if HiAt (zkTab x j).table.sch.levels l p then (exeTape t).salt j.val r.val l p else ByteArray.empty :=
  congrArg (Pc (zkTab x j).table (stAt x j) c r L y) (hiSalt_exe_eq (ofSalts Pc) x t j r)

/-- **At the executable's layout the later rounds' prover reads the tape's salts**: `ofSalts Pc` on table `j`'s rep `r`
salts sends what `Pc` sends on the masked prover's tape's (`exeTape`'s `salt j r`), once `Pc` reads only the schedule's
levels ≥ 1 and their positions. -/
theorem ofSalts_exe (Pc : SaltProver) (hPc : SaltsOn Pc) (x : Msg) (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (j : Fin (jOf x)) (c : Composed.CoinsZK (Composed.RestOf (zkTab x j).table (stAt x j))) (r : Fin 2) (L : ℕ)
    (y : Fin L → 𝕂) :
    ofSalts Pc (zkTab x j).table (stAt x j) c r L y (hiTape (ofSalts Pc) x t j r) =
      Pc (zkTab x j).table (stAt x j) c r L y ((exeTape t).salt j.val r.val) := by
  rw [ofSalts_exe_on]
  exact hPc _ _ c r L y _ _ fun l p h => ite_eq_left h

end Masked

end Claims.ZKMal
