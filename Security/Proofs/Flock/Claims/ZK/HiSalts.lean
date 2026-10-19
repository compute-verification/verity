import Proofs.Flock.Claims.ZK.ExeLayout

/-!
# Levels ≥ 1's salts at the executable's indices

A Ligerito prover of levels ≥ 1 (`Composed.LigProver`) gets rep `r`'s levels ≥ 1 salts as one array `HiOf Tz`, and the
masked prover reads them by level and position (`ligeritoPZk`'s `salt l q`, `1 ≤ l < |levels|`, `q < 2^d`). Here the
array is read at `hiIdx`, the order `TapeLayout.exe` lays them out in, so the later rounds' prover reads the tape's
salts.

* `hiSalt`: level `l`'s salt at position `p` of an array of levels ≥ 1's salts, at `hiIdx`; empty off the schedule.
* `hiSalt_exe`: at the executable's layout, the salts the model hands table `j`'s rep `r` (`Base`'s last field, as
  `Coined.randOf` and the view read it) are, at every level `l ≥ 1` and position `p` of its tree, the masked prover's
  tape's `salt j r l p` (`exeTape`); `hiSalt_exe_eq` as functions, empty off the schedule.
* `ofSalts`: the Ligerito prover of a prover `Pc` that reads salts by level and position, reading them at `hiIdx`.
  At the executable's layout it sends what `Pc` sends on the tape's salts on the schedule, empty off it
  (`ofSalts_exe_on`), so on the tape's salts once `Pc` reads only the schedule's levels ≥ 1 and their positions
  (`ofSalts_exe`, `SaltsOn`).
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

namespace Masked

/-- A level `l ≥ 1` of the schedule and a position `p` of its tree. -/
def HiAt (ls : List FlockSoundness.Accounting.Level) (l p : ℕ) : Prop :=
  1 ≤ l ∧ l < ls.length ∧ p < 2 ^ (ls.getD l default).logLen

instance (ls : List FlockSoundness.Accounting.Level) (l p : ℕ) : Decidable (HiAt ls l p) := by
  unfold HiAt; infer_instance

/-- **Level `l`'s salt at position `p`** of the array `hi` of levels ≥ 1's salts in level order (`hiIdx`), empty off the
schedule. -/
def hiSalt (ls : List FlockSoundness.Accounting.Level) {n : ℕ} (hi : Fin n → ZkView.Salt) (l p : ℕ) : ByteArray :=
  if HiAt ls l p then fAt ByteArray.empty (fun k => ZkView.saltBytes (hi k)) (hiOff ls l + p) else ByteArray.empty

theorem hiOff_length (ls : List FlockSoundness.Accounting.Level) :
    hiOff ls ls.length = (ls.tail.map fun lv => 2 ^ lv.logLen).sum := by
  unfold hiOff
  have e : ls.length - 1 = ls.tail.length := by rw [List.length_tail]
  rw [e]
  simp

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

/-- **The salts the model hands levels ≥ 1 are the executable's**: at the executable's layout, table `j`'s rep `r`
salt at level `l ≥ 1` and position `p` of its tree, read at `hiIdx`, is the masked prover's tape's `salt j r l p`. -/
theorem hiSalt_exe (P : Composed.LigProver) (x : Msg) (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (j : Fin (jOf x)) (r : Fin 2) {l p : ℕ} (h : HiAt (zkTab x j).table.sch.levels l p) :
    hiSalt (zkTab x j).table.sch.levels (hiTape P x t j r) l p = (exeTape t).salt j.val r.val l p := by
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hn : Composed.hiCount (zkTab x j).table ≤ schMax := hi_le (zkTab x j)
  have hcount : Composed.hiCount (zkTab x j).table =
      2 * (((zkTab x j).table.sch.levels).tail.map fun lv => 2 ^ lv.logLen).sum := rfl
  have hk : hiOff (zkTab x j).table.sch.levels l + p < Composed.hiCount (zkTab x j).table := by
    have := hiOff_add_lt h
    omega
  have hl' : l < schMax + 1 := by
    have := le_sum_tail h
    omega
  have hp' : p < schMax := by omega
  have hq : HiV (zkTab x j).table.sch.levels (⟨l, hl'⟩, ⟨p, hp'⟩) := h
  have hl0 : l ≠ 0 := by have := h.1; omega
  have hR : saltAt (cellAt t j.val) r.val l p =
      ZkView.saltBytes (((t ⟨j.val, hc⟩).1.2.2.2.2.2.2.2.2.2 r).1 (⟨l, hl'⟩, ⟨p, hp'⟩)) := by
    rw [cellAt_lt t hc]
    simp only [saltAt, fAt, hl0, r.isLt, hl', hp', ↓reduceIte, ↓reduceDIte] <;> rfl
  unfold hiSalt
  rw [if_pos h, fAt_lt _ _ hk]
  change ZkView.saltBytes (Sum.elim ((t ⟨j.val, hc⟩).1.2.2.2.2.2.2.2.2.2 r).1 ((t ⟨j.val, hc⟩).1.2.2.2.2.2.2.2.2.2 r).2
      (hiEmbOf (zkTab x j).table.sch.levels hn ⟨hiIdx (zkTab x j).table.sch.levels (⟨l, hl'⟩, ⟨p, hp'⟩), hk⟩)) =
    saltAt (cellAt t j.val) r.val l p
  rw [hiEmbOf_hiIdx _ hn hq hk, hR] <;> rfl

/-- `hiSalt_exe` as functions: the tape's salts on the schedule's levels ≥ 1 and their positions, empty off them. -/
theorem hiSalt_exe_eq (P : Composed.LigProver) (x : Msg) (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (j : Fin (jOf x)) (r : Fin 2) :
    hiSalt (zkTab x j).table.sch.levels (hiTape P x t j r) = fun l p =>
      if HiAt (zkTab x j).table.sch.levels l p then (exeTape t).salt j.val r.val l p else ByteArray.empty := by
  funext l p
  split_ifs with h
  · exact hiSalt_exe P x t j r h
  · unfold hiSalt
    rw [if_neg h]

/-- **A prover of Ligerito's levels ≥ 1 reading salts by level and position**: rep `r`'s bytes after `T'` from the
table, the statement, the coins, the restart `y₁'` and the salts, level `l`'s at position `p` (`ligeritoPZk`'s
`salt l p`). -/
abbrev SaltProver : Type :=
  ∀ {mPts : ℕ} (Tz : ZkSession.TabZK 𝔽 mPts) (st : _root_.Flock.Stmt), Composed.CoinsZK (Composed.RestOf Tz st) →
    Fin 2 → (L : ℕ) → (Fin L → 𝕂) → (ℕ → ℕ → ByteArray) → ByteArray

/-- **`Pc` reads only the schedule's levels ≥ 1 and their positions**: salts that agree there give the same bytes. -/
def SaltsOn (Pc : SaltProver) : Prop :=
  ∀ {mPts : ℕ} (Tz : ZkSession.TabZK 𝔽 mPts) (st : _root_.Flock.Stmt) (c : Composed.CoinsZK (Composed.RestOf Tz st))
    (r : Fin 2) (L : ℕ) (y : Fin L → 𝕂) (s s' : ℕ → ℕ → ByteArray),
    (∀ l p, HiAt Tz.sch.levels l p → s l p = s' l p) → Pc Tz st c r L y s = Pc Tz st c r L y s'

/-- **The Ligerito prover of `Pc`**: levels ≥ 1's salts read at `hiIdx` (`hiSalt`). -/
def ofSalts (Pc : SaltProver) : Composed.LigProver :=
  fun Tz st c r L y hi => Pc Tz st c r L y (hiSalt Tz.sch.levels hi)

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
  exact hPc _ _ c r L y _ _ fun l p h => if_pos h

end Masked

end Claims.ZKMal
