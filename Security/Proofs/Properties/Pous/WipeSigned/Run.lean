import Properties.Pous.WipeSigned

/-!
# What the challenger's record of a wipe keeps

Read by read (`WipeRun.step`), the challenger's record keeps an answer for each batch, each batch's wrong count is 0
exactly when its answer is its read's label, and a session that has ended has a batch (`Inv`, `inv_recorded`).
-/

namespace MemoryChallenger.Properties.SecurityProofs.WipeRuns

open MemoryChallenger Flock

/-- What the record keeps through the reads. -/
def Inv (r : WipeRun) (st : WipeRecorded) : Prop :=
  st.answers.size = st.batches.size ∧
    (∀ j (hj : j < r.commitment.params.k), j < st.batches.size →
      st.batches[j]!.wrong = if st.answers[j]! = wipeLabel r.key (r.reads ⟨j, hj⟩) then 0 else 1) ∧
    (st.ended ≠ none → 0 < st.batches.size)

theorem push_get! {α : Type} [Inhabited α] (a : Array α) (x : α) {j : Nat} (hj : j < a.size + 1) :
    (a.push x)[j]! = if j < a.size then a[j]! else x := by
  rw [getElem!_pos (a.push x) j (by rw [Array.size_push]; exact hj), Array.getElem_push]
  by_cases h : j < a.size
  · simp [h]
  · simp [h]

theorem inv_step (r : WipeRun) (st : WipeRecorded) (o : WipeObserved) (h : Inv r st) : Inv r (r.step st o) := by
  unfold WipeRun.step
  dsimp only
  split
  · rename_i hc
    obtain ⟨h1, h2, -⟩ := h
    refine ⟨by simp [h1], ?_, fun _ => by simp⟩
    intro j hj hjs
    simp only [Array.size_push] at hjs
    rw [push_get! _ _ hjs, push_get! _ _ (by rw [h1]; exact hjs)]
    by_cases hlt : j < st.batches.size
    · have ha : j < st.answers.size := by rw [h1]; exact hlt
      simp only [hlt, ha, ↓reduceIte]
      exact h2 j hj hlt
    · have ha : ¬ j < st.answers.size := by rw [h1]; exact hlt
      simp only [hlt, ha, ↓reduceIte]
      have hj' : j = st.batches.size := by omega
      subst hj'
      rfl
  · exact h

theorem inv_foldl (r : WipeRun) : ∀ (l : List WipeObserved) (st : WipeRecorded), Inv r st → Inv r (l.foldl r.step st)
  | [], _, h => h
  | o :: l, st, h => inv_foldl r l _ (inv_step r st o h)

theorem inv_recorded (r : WipeRun) : Inv r r.recorded :=
  inv_foldl r r.observed _ ⟨rfl, fun _ _ h => absurd h (Nat.not_lt_zero _), fun h => absurd rfl h⟩

end MemoryChallenger.Properties.SecurityProofs.WipeRuns
