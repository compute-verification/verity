module

public import Proofs.Flock.Soundness.Discharge.PinnedSat
public import Proofs.Flock.Soundness.Discharge.Hidden.Defs
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
import Proofs.Flock.Soundness.Discharge.Hidden.Delta

/-! A hidden-output statement without Δ's pinned copies (`unpin`), which every witness of the accepted statement satisfies, carrying 0 on every pinned bit (`sat_unpin`). -/

@[expose] public section

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.Hidden

open Refine

/-- **`st` without Δ's pinned copies**: Δ is `HmRow.delta`, then `outPairs`. Reducible, as `flatTwin` is, so that its
circuit, pin and regions are `st`'s to `rw` and `simp`. -/
abbrev unpin (st : Stmt) : Stmt :=
  { st with da := (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray,
            db := (HmRow.delta st.c st.pin).2 ++ (outPairs st.c).toArray }

/-- **The pinned bits' columns** (`HmOut.pinPairs`'s): per template instance `g`, pinned output word `o` and bit `t < 16`,
bit `t` of the word's column. -/
def pinnedCols (c : Circuit) : List ℕ :=
  (List.range c.g).flatMap fun g => (List.range (HmOut.pinnedWords c)).flatMap fun o =>
    (List.range 16).map fun t => HmOut.outCol c g (HmOut.pinnedFrom c + o) + t

theorem pinList_eq (c : Circuit) : pinList c = (pinnedCols c).flatMap fun i => [(i, i), (i, Hm.zeroCol c)] := by
  unfold pinList pinnedCols
  simp only [List.flatMap_assoc, List.flatMap_map, Function.comp_def]

theorem pinnedCols_nil {c : Circuit} (h : HmRow.pinnedCount c.metaJ = 0) : pinnedCols c = [] := by
  unfold pinnedCols
  rw [pinnedWords_zero h]
  simp

theorem count_pinPairs (L : List ℕ) (zero r j : ℕ) :
    (L.flatMap fun i => [(i, i), (i, zero)]).count (r, j) =
      L.count r * ((if j = r then 1 else 0) + (if j = zero then 1 else 0)) := by
  induction L with
  | nil => simp
  | cons x L ih =>
    rw [List.flatMap_cons, List.count_append, ih]
    simp only [List.count_cons, List.count_nil, beq_iff_eq, Prod.mk.injEq]
    split_ifs <;> omega

theorem dCnt_append (x y : Array (ℕ × ℕ)) (i j : ℕ) : dCnt (x ++ y) i j = dCnt x i j + dCnt y i j := by
  unfold dCnt
  rw [Array.toList_append, List.count_append]

theorem dCnt_toArray (l : List (ℕ × ℕ)) (i j : ℕ) : dCnt l.toArray i j = l.count (i, j) := rfl

/-- Δ's pinned copies at row `i`, column `j`: `pinnedCols`'s count of `i` copies of `e_i + e_zero`. -/
def pinCnt (c : Circuit) (i j : ℕ) : ℕ :=
  (pinnedCols c).count i * ((if j = i then 1 else 0) + (if j = Hm.zeroCol c then 1 else 0))

theorem dCnt_pin (c : Circuit) (pin i j : ℕ) :
    dCnt ((HmRow.delta c pin).1 ++ (outPairs c).toArray ++ (pinList c).toArray) i j =
      dCnt ((HmRow.delta c pin).1 ++ (outPairs c).toArray) i j + pinCnt c i j := by
  rw [dCnt_append, pinList_eq, dCnt_toArray (List.flatMap _ _), count_pinPairs]
  rfl

theorem dCnt_pinB (c : Circuit) (pin i j : ℕ) :
    dCnt ((HmRow.delta c pin).2 ++ (outPairs c).toArray ++ (pinList c).toArray) i j =
      dCnt ((HmRow.delta c pin).2 ++ (outPairs c).toArray) i j + pinCnt c i j := by
  rw [dCnt_append, pinList_eq, dCnt_toArray (List.flatMap _ _), count_pinPairs]
  rfl

theorem matA_unpin {st : Stmt} (hda : st.da = (HmOut.delta st.c st.pin).1) (i j : ℕ) :
    matA st i j = matA (unpin st) i j + (pinCnt st.c i j : ZMod 2) := by
  have e : matA (unpin st) i j = ((rangeCnt (slotA st) st.c.ranges.toList i j +
      dCnt ((HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) i j : ℕ) : ZMod 2) := rfl
  rw [e]
  unfold matA
  rw [hda, hidden_delta' st.c st.pin]
  dsimp only
  rw [dCnt_pin]
  push_cast
  ring

theorem matB_unpin {st : Stmt} (hdb : st.db = (HmOut.delta st.c st.pin).2) (i j : ℕ) :
    matB st i j = matB (unpin st) i j + (pinCnt st.c i j : ZMod 2) := by
  have e : matB (unpin st) i j = ((rangeCnt (slotB st) st.c.ranges.toList i j +
      dCnt ((HmRow.delta st.c st.pin).2 ++ (outPairs st.c).toArray) i j : ℕ) : ZMod 2) := rfl
  rw [e]
  unfold matB
  rw [hdb, hidden_delta' st.c st.pin]
  dsimp only
  rw [dCnt_pinB]
  push_cast
  ring

theorem unpin_wf {st : Stmt} (h : StmtWF st) : StmtWF (unpin st) := ⟨h.hk6, h.hk, h.pin, h.slots⟩

theorem pinRows_unpin {st : Stmt} (hda : st.da = (HmOut.delta st.c st.pin).1)
    (hdb : st.db = (HmOut.delta st.c st.pin).2) (hwf : StmtWF st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) (zero : Fin (2 ^ st.c.kLog))
    (hzero : zero.val = Hm.zeroCol st.c) :
    Pinned.PinRows (stmtOf st hwf regs) (stmtOf (unpin st) (unpin_wf hwf) regs).A₀
      (stmtOf (unpin st) (unpin_wf hwf) regs).B₀ zero (fun r => ((pinnedCols st.c).count r.val : ZMod 2)) := by
  intro r j
  let z : Fin (2 ^ (stmtOf st hwf regs).kLog) := zero
  have h1 : (if j.val = r.val then 1 else 0 : ℕ) = if j = r then 1 else 0 := by
    by_cases h : j = r
    · rw [if_pos h, if_pos (congrArg Fin.val h)]
    · rw [if_neg h, if_neg (fun e => h (Fin.ext e))]
  have h2 : (if j.val = Hm.zeroCol st.c then 1 else 0 : ℕ) = if j = z then 1 else 0 := by
    by_cases h : j = z
    · rw [if_pos h, if_pos (by rw [← hzero]; exact congrArg Fin.val h)]
    · rw [if_neg h, if_neg (fun e => h (Fin.ext (e.trans hzero.symm)))]
  have hc : (pinCnt st.c r.val j.val : ZMod 2) =
      ((pinnedCols st.c).count r.val : ZMod 2) * ((if j = r then 1 else 0) + (if j = z then 1 else 0)) := by
    unfold pinCnt
    rw [h1, h2, Nat.cast_mul, Nat.cast_add, Nat.cast_ite, Nat.cast_ite, Nat.cast_one, Nat.cast_zero]
  exact ⟨(matA_unpin hda r.val j.val).trans (congrArg (matA (unpin st) r.val j.val + ·) hc),
    (matB_unpin hdb r.val j.val).trans (congrArg (matB (unpin st) r.val j.val + ·) hc)⟩

theorem sat_unpin {st : Stmt} (hda : st.da = (HmOut.delta st.c st.pin).1)
    (hdb : st.db = (HmOut.delta st.c st.pin).2) (hwf : StmtWF st)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {zero : Fin (2 ^ st.c.kLog)}
    (hzero : zero.val = Hm.zeroCol st.c) {z : Model.Witness st.m} (hz : (stmtOf st hwf regs).Satisfies z)
    (hz0 : ∀ o (r : Fin (2 ^ (stmtOf st hwf regs).kLog)), ((pinnedCols st.c).count r.val : ZMod 2) = 1 → (stmtOf st hwf regs).block z o zero = 0)
    (hone : ∀ o (r : Fin (2 ^ (stmtOf st hwf regs).kLog)), ((pinnedCols st.c).count r.val : ZMod 2) = 1 →
      (stmtOf (unpin st) (unpin_wf hwf) regs).B₀.mulVec ((stmtOf st hwf regs).block z o) r = 1) :
    (stmtOf (unpin st) (unpin_wf hwf) regs).Satisfies z ∧
      ∀ o (r : Fin (2 ^ (stmtOf st hwf regs).kLog)), ((pinnedCols st.c).count r.val : ZMod 2) = 1 → (stmtOf st hwf regs).block z o r = 0 :=
  Pinned.sat_unpin (pinRows_unpin hda hdb hwf regs zero hzero) hz hz0 hone

end FlockSoundness.Discharge.Hidden
