module

public import Proofs.Properties.Verity.Sampled.Walk

/-! The order of a window's members' sessions in the device's log. -/

@[expose] public section

namespace Verity.Sampled

def OpensInOrder (tree : ℕ → ℕ) (m : ℕ) (log : List Release) : Prop :=
  ∀ pre r post, log = pre ++ r :: post → ∀ j, j < m → ∀ s n, r.frame = .opening (tree j) s n →
    ∀ q ∈ pre, ∀ j', j' < m → ∀ s' n', q.frame = .opening (tree j') s' n' → j' ≤ j

def EvidenceRecorded (tree : ℕ → ℕ) (entry : ℕ → ByteArray) (m : ℕ) (log : List Release) : Prop :=
  ∀ pre r post, log = pre ++ r :: post → ∀ j, j < m → ∀ s n, r.frame = .opening (tree j) s n →
    ∀ j', j' < j → ∃ q ∈ pre, q.answers = Flock.Sha512.hash (entry j')

structure SessionsInOrder (tree : ℕ → ℕ) (entry : ℕ → ByteArray) (m : ℕ) (log : List Release) : Prop where
  opens : OpensInOrder tree m log
  recorded : EvidenceRecorded tree entry m log

theorem OpensInOrder.of_append {tree : ℕ → ℕ} {m : ℕ} {l₁ l₂ : List Release}
    (h : OpensInOrder tree m (l₁ ++ l₂)) : OpensInOrder tree m l₁ :=
  fun pre r post e => h pre r (post ++ l₂) (by rw [e, List.append_assoc, List.cons_append])

theorem EvidenceRecorded.of_append {tree : ℕ → ℕ} {entry : ℕ → ByteArray} {m : ℕ} {l₁ l₂ : List Release}
    (h : EvidenceRecorded tree entry m (l₁ ++ l₂)) : EvidenceRecorded tree entry m l₁ :=
  fun pre r post e => h pre r (post ++ l₂) (by rw [e, List.append_assoc, List.cons_append])

theorem SessionsInOrder.of_append {tree : ℕ → ℕ} {entry : ℕ → ByteArray} {m : ℕ} {l₁ l₂ : List Release}
    (h : SessionsInOrder tree entry m (l₁ ++ l₂)) : SessionsInOrder tree entry m l₁ :=
  ⟨h.opens.of_append, h.recorded.of_append⟩

theorem sessionsInOrder_one (tree : ℕ → ℕ) (entry : ℕ → ByteArray) {m : ℕ} (hm : m ≤ 1) (log : List Release) :
    SessionsInOrder tree entry m log :=
  ⟨fun _ _ _ _ j hj _ _ _ _ _ j' hj' _ _ _ => by omega, fun _ _ _ _ j hj _ _ _ j' hj' => by omega⟩

theorem EvidenceRecorded.recorded_before {tree : ℕ → ℕ} {entry : ℕ → ByteArray} {m : ℕ} {pre post : List Release}
    {r : Release} (h : EvidenceRecorded tree entry m (pre ++ r :: post)) {j s n : ℕ} (hj : j < m)
    (hr : r.frame = .opening (tree j) s n) {j' : ℕ} (hj' : j' < j) :
    ∃ pre₁ q pre₂, pre = pre₁ ++ q :: pre₂ ∧ q.answers = Flock.Sha512.hash (entry j') := by
  obtain ⟨q, hq, hqa⟩ := h pre r post rfl j hj s n hr j' hj'
  obtain ⟨pre₁, pre₂, e⟩ := List.append_of_mem hq
  exact ⟨pre₁, q, pre₂, e, hqa⟩

theorem not_evidenceRecorded_of_unrecorded (tree : ℕ → ℕ) (entry : ℕ → ByteArray) (s n : ℕ) (b : ByteArray)
    (cs : List Bool) : ¬ EvidenceRecorded tree entry 2 [⟨.opening (tree 1) s n, b, cs⟩] := fun h => by
  obtain ⟨q, hq, -⟩ := h [] _ [] rfl 1 (by omega) s n rfl 0 (by omega)
  cases hq

end Verity.Sampled
