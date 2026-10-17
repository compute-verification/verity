import Proofs.Flock.Claims.Bridge.CustodyPlain
import Proofs.Flock.Soundness.Discharge.DecodeZL.Assemble
import Proofs.Flock.Soundness.Discharge.DecodeZL.OP
import Proofs.Flock.Soundness.Discharge.DecodeZL.LIG

/-!
# The deployed plain decoders at `wrP`, at one table

The plain verifier is in scope at one table (`InScope`: `zk ∨ tables ≤ 1`). There the old discharge of the decoder
(`DecodeZL.decodesOn_exec`) is stated at one table's live outcome (`LiveOut`, `Decodes`), and the transfer the bridge
takes (`live_le_sessionBLJ`, `LivePlain.Pr_acc_plain_le`) at the outcome of a session of `J` tables (`LiveOutJ`,
`DecodesTable`). At `J = 1` the two meet:

* `oneOf`, `jOf`: a one-table session's outcome as one table's, and back. `RecordsLiveAtP` reads an outcome only through
  `oneOf` (`recordsLiveAtP_oneOf`).
* `wrP1`, `recordsLive_wrP1`: `wrP` at one table, read at one table's outcome, records the live session
  (`DecodeZL.RecordsLive`); where `verify` accepts what `wrP` writes, it accepts what `wrP1` writes for the outcome's
  `oneOf` (`accepts_wrP1_oneOf`).
* `decodesTable_wrP`: at a draw `S` that sets the table up at `mPts` link points (`DrawOk`), under the verify command's
  tags (`TagsOk`), the deployed decoders are right at `wrP` (`DecodesTable`, `Pr_acc_plain_le`'s `hdec`), from
  `decodesOn_exec` at the law of the one draw `S`.
-/

namespace Claims.Bridge.DecodePlain

open _root_.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec Claims.Bridge.CustodyPlain
open Lean (Json)

/-- A one-table session's live outcome, as one table's. -/
def oneOf {mPts : ℕ} (o : Tables.LiveOutJ 1 mPts) : LiveOut mPts :=
  ⟨o.rootB[0]!, o.pts, o.reps 0 0, o.reps 0 1, o.proofs⟩

/-- One table's live outcome, as a one-table session's. -/
def jOf {mPts : ℕ} (o : LiveOut mPts) : Tables.LiveOutJ 1 mPts :=
  ⟨#[o.rootB], o.pts, fun _ r => pick (o.rep0, o.rep1) r, o.proofs⟩

/-- **`RecordsLiveAtP` at one table reads the outcome through `oneOf`.** -/
theorem recordsLiveAtP_oneOf (I : Exec.Inputs) (d : Json) {mPts : ℕ} (o : Tables.LiveOutJ 1 mPts)
    (t : Exec.Transcript) : RecordsLiveAtP I d (jOf (oneOf o)) t ↔ RecordsLiveAtP I d o t := by
  have hr : ∀ j : Fin 1, (jOf (oneOf o)).rootB[j.val]! = o.rootB[j.val]! := by
    intro j
    obtain rfl := Subsingleton.elim j 0
    rfl
  have hp : ∀ (j : Fin 1) (r : Fin 2), (jOf (oneOf o)).reps j r = o.reps j r := by
    intro j r
    obtain rfl := Subsingleton.elim j 0
    fin_cases r <;> rfl
  unfold RecordsLiveAtP
  simp only [hr, hp]
  rfl

/-- **`wrP` at one table**, read at one table's live outcome. -/
noncomputable def wrP1 {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json) (S : Finset (Fin n))
    (R : Reg) (o : LiveOut mPts) : Exec.Transcript :=
  wrP I dj (fun _ _ => 1) S R (jOf o)

/-- **`wrP1` records one table's live session** (`DecodeZL.RecordsLive`, the record half of the old one-table
custody). -/
theorem recordsLive_wrP1 {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json) :
    DecodeZL.RecordsLive (Reg := Reg) (mPts := mPts) I dj (wrP1 I dj) := by
  intro S R o hacc
  obtain ⟨hd, hpr, hrec⟩ := recordsLiveJ_wrP (mPts := mPts) (Reg := Reg) I dj (fun _ _ => 1) S R (jOf o) hacc
  refine ⟨hd, hpr, fun spec sts sess hset hdec => ?_⟩
  obtain ⟨hpts, hj⟩ := hrec spec sts sess hset hdec
  obtain ⟨hroot, hstr⟩ := hj 0
  exact ⟨hroot, hpts, hstr⟩

/-- **Where `verify` accepts what `wrP` writes at one table, it accepts what `wrP1` writes for `oneOf`**: the two
choose among the same transcripts (`recordsLiveAtP_oneOf`). -/
theorem accepts_wrP1_oneOf {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (S : Finset (Fin n)) (R : Reg) (o : Tables.LiveOutJ 1 mPts)
    (h : Exec.Accepts I (wrP I dj (fun _ _ => 1) S R o)) : Exec.Accepts I (wrP1 I dj S R (oneOf o)) := by
  unfold wrP1 wrP at *
  split at h
  · rename_i hex
    have hex' : ∃ t, Exec.Accepts I t ∧ RecordsLiveAtP I (dj S) (jOf (oneOf o)) t :=
      let ⟨t, ha, hr⟩ := hex
      ⟨t, ha, (recordsLiveAtP_oneOf I (dj S) o t).2 hr⟩
    rw [dite_eq_left_of_eq_true (eq_true hex')]
    exact (Classical.choose_spec hex').1
  · exact absurd h (not_accepts_nil _ _)

/-- **The deployed plain decoders are right at `wrP`, at one table** (`DecodesTable`: `LivePlain.Pr_acc_plain_le`'s
`hdec` at the table `tablesAt` sets up): at a draw `S` that sets the table up at `mPts` link points (`DrawOk`), under
the verify command's tags (`TagsOk`), on a circuit whose outputs are its units', from `decodesOn_exec` at the law of the
one draw `S`. -/
theorem decodesTable_wrP {c : _root_.Flock.Circuit} {n : ℕ} {I : Exec.Inputs} (ht : I.tags.typed = false)
    (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet) {mPts : ℕ}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} (x₀ : Integrate.DrawSetup I mPts S₀ (dj S₀)) {Reg : Type}
    (hT : DecodeZL.TagsOk I) {S : Finset (Fin n)} (hS : DecodeZL.DrawOk I dj mPts S) (R : Reg)
    (j : Fin [Integrate.tablesAt (Reg := Reg) ht dj x₀ S R].length) :
    Tables.DecodesTable (fun _ cap pend os => tableOutL H512 enc512 leaf512 cap pend os)
      (fun _ o => o.accepted = true) (fun o => Exec.Accepts I (wrP I dj (fun _ _ => 1) S R o))
      (fun ps _ => DecodeZL.opnAt (Integrate.tablesAt ht dj x₀) I dj S R ps)
      (fun _ => DecodeZL.decAt (Integrate.tablesAt ht dj x₀) DecodeZL.wOP DecodeZL.wLIG S R) (fun _ _ => true)
      (fun _ _ _ => []) (fun rs _ => capOfRoot rs[0]!)
      (fun j => repsOf (D := Dig) execArith ([Integrate.tablesAt (Reg := Reg) ht dj x₀ S R].get j)) j := by
  have hj : j.val = 0 := by
    have h := j.isLt
    simp only [List.length_cons, List.length_nil] at h
    omega
  obtain rfl : j = ⟨0, Nat.one_pos⟩ := Fin.ext hj
  have h := DecodeZL.decodesOn_exec ht hc ho dj x₀ { Ω := Unit, draw := fun _ => S }
    (wrP1 (Reg := Reg) (mPts := mPts) I dj) DecodeZL.wOP DecodeZL.wLIG DecodeZL.op_carries DecodeZL.lig_carries hT
    (recordsLive_wrP1 I dj) (fun _ => hS) () R
  intro o hacc
  obtain ⟨pend, h0, h1, hok⟩ := h (oneOf o) (accepts_wrP1_oneOf I dj S R o hacc)
  exact ⟨pend, h0, h1, hok⟩

end Claims.Bridge.DecodePlain
