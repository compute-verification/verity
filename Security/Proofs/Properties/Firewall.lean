module

public import Properties.Firewall

/-!
# The ZK server's record, proved

`admitNext`, `stopEnds`, `outOfTurnStops` and `afterOnTime` of `Properties.Firewall`, by unfolding
`Core.Firewall.admit` (its last step, `admit.served`, apart: `served_ok`, `admit_ok`), `Core.Firewall.stop` and
`Core.Firewall.Decision.after`. `stopOf_push_stop` and `stopOf_push_refused` say what appending a stop or a refusal does
to a recursive session's stop. The honest lemmas are at `finishedRecord` and `scheduledAdmitted`, the nontrivial ones at
the empty record and at a missed release time.
-/

@[expose] public section

namespace Properties.Firewall

open Core.Firewall

/-- A statement `nextOf` names is one of the recursive session's statements. -/
theorem nextOf_lt {evs : Events} {o : Open} {j : Nat} (h : nextOf evs o = some j) : j < o.parts.size := by
  simp only [nextOf] at h
  obtain ⟨hj, -⟩ := Array.findIdx?_eq_some_iff_getElem.mp h
  exact hj

/-- A stop of the recursive session `s` appended to the record: its stop is the first one, else this one. -/
theorem stopOf_push_stop (evs : Events) (s p why : String) :
    stopOf (evs.push (.stop s p why)) s = (stopOf evs s).or (some p) := by
  simp [stopOf, Array.findSome?_push]

/-- A refusal appended to the record leaves every recursive session's stop as it was. -/
theorem stopOf_push_refused (evs : Events) (s s' p : String) :
    stopOf (evs.push (.refused s' p)) s = stopOf evs s := by
  simp [stopOf, Array.findSome?_push]

/-- `admit`'s last step admits only statement `j`, asked for as `part`, once the inner session finished, recording
nothing. -/
theorem served_ok {evs : Events} {s part : String} {o : Open} {j : Nat} {a : Admitted} {evs' : Events}
    (h : admit.served evs s part o j = (.ok a, evs')) :
    finished evs s = true ∧ o.parts[j]! = part ∧
      a = { stream := o.stream, index := j, releaseMs := releaseMs o.release j } ∧ evs' = evs := by
  unfold admit.served admit.refuse at h
  split at h
  · simp at h
  · rename_i hf
    dsimp only at h
    split at h
    · simp at h
    · rename_i hp
      simp only [Prod.mk.injEq, Except.ok.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨by simpa using hf, by simpa using hp, rfl, rfl⟩

/-- `admit` admits only through its last step, at the recursive session's open, before any stop and at its next
statement. -/
theorem admit_ok {evs : Events} {s part : String} {a : Admitted} {evs' : Events}
    (h : admit evs s part = (.ok a, evs')) :
    ∃ o j, openOf evs s = some o ∧ stopOf evs s = none ∧ nextOf evs o = some j ∧
      admit.served evs s part o j = (.ok a, evs') := by
  unfold admit at h
  split at h
  · simp [admit.refuse] at h
  · rename_i o ho
    split at h
    · simp [admit.refuse] at h
    · rename_i hs
      split at h
      · simp [admit.refuse] at h
      · rename_i j hn
        refine ⟨o, j, ho, hs, hn, ?_⟩
        split at h
        · dsimp only at h
          split at h
          · simp [admit.refuse] at h
          · exact h
        · exact h

theorem admitNext : AdmitNext := by
  intro evs s part a evs' h
  obtain ⟨o, j, ho, hs, hn, hserved⟩ := admit_ok h
  obtain ⟨hf, hp, rfl, rfl⟩ := served_ok hserved
  have hj := nextOf_lt hn
  refine ⟨rfl, o, ho, hs, hf, hn, ?_, rfl, rfl⟩
  rw [Array.getElem?_eq_getElem hj, ← hp, getElem!_pos o.parts j hj]

theorem stopEnds : StopEnds := by
  intro evs s hs
  refine ⟨fun part => ?_, fun part why => ?_⟩
  · unfold admit
    cases ho : openOf evs s with
    | none => exact ⟨_, rfl⟩
    | some o =>
      obtain ⟨p, hp⟩ := Option.isSome_iff_exists.mp hs
      simp only [hp]
      exact ⟨_, rfl⟩
  · simp [stop, hs]

theorem outOfTurnStops : OutOfTurnStops := by
  intro evs s part o j ho hs hf hb hn hp
  have hj := nextOf_lt hn
  have hne : (o.parts[j]! != part) = true := by
    rw [getElem!_pos o.parts j hj]
    rw [Array.getElem?_eq_getElem hj] at hp
    simpa using hp
  have hserved : admit.served evs s part o j =
      (.error s!"recursive session {s} stopped at {o.parts[j]!}: {part} is not its next statement",
        (evs.push (.stop s o.parts[j]! s!"never opened: {part} was asked in its place")).push (.refused s part)) := by
    unfold admit.served admit.refuse
    simp [hf, hne]
  have hadmit : admit evs s part = admit.served evs s part o j := by
    unfold admit
    simp only [ho, hs, hn]
    cases hbo : budgetOf evs o.stream with
    | none => rfl
    | some b =>
      simp [Nat.not_le.mpr (hb b hbo)]
  refine ⟨_, _, hadmit.trans hserved, ?_⟩
  rw [stopOf_push_refused, stopOf_push_stop, hs, Option.none_or, Array.getElem?_eq_getElem hj,
    getElem!_pos o.parts j hj]

theorem afterOnTime : AfterOnTime := by
  intro a now prove ready a' h
  unfold Decision.after at h
  split at h
  · cases h
  · cases h
  · split at h
    · cases h
    · cases h
    · split at h
      · rename_i t ht
        split at h
        · cases h
        · rename_i hle
          cases h
          refine ⟨rfl, rfl, rfl, fun t' ht' => ?_⟩
          rw [ht] at ht'
          cases ht'
          exact Nat.not_lt.mp hle
      · rename_i ht
        cases h
        refine ⟨rfl, rfl, rfl, fun t' ht' => ?_⟩
        rw [ht] at ht'
        cases ht'

/-- **`AdmitNext`'s honest lemma**: `admit` admits `r0000`'s statement `p` once its inner session finished. -/
theorem admitNext_honest : admit finishedRecord "r0000" "p" = (.ok firstAdmitted, finishedRecord) := by
  simp [admit, admit.served, finishedRecord, oneStatement, firstAdmitted, openOf, stopOf, nextOf, finished, budgetOf,
    releaseMs]

/-- **`AdmitNext`'s nontrivial lemma**: on the empty record no recursive session was opened, so no admission there is
`AdmittedNext`, while a check that admits everything admits it. -/
theorem admitNext_nontrivial : ¬ AdmittedNext #[] "r0000" "p" firstAdmitted #[] := by
  rintro ⟨-, o, ho, -⟩
  simp [openOf] at ho

/-- **`AfterOnTime`'s honest lemma**: the proofs of a session scheduled for 10 ms leave at 10 ms, the prover and the send
check having returned `ok`. -/
theorem afterOnTime_honest : Decision.after scheduledAdmitted 10 .ok .ok = .go scheduledAdmitted := by
  rfl

/-- **`AfterOnTime`'s nontrivial lemma**: at 11 ms a session scheduled for 10 ms missed its release time, so its proofs
leaving then is not `SentOnTime`, while a decision that lets everything leave lets them. -/
theorem afterOnTime_nontrivial : ¬ SentOnTime scheduledAdmitted 11 .ok .ok scheduledAdmitted := by
  rintro ⟨-, -, -, h⟩
  exact absurd (h 10 rfl) (by decide)

end Properties.Firewall
