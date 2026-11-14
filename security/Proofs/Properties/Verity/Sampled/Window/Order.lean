module

public import Proofs.Properties.Verity.Sampled.Walk

/-!
# The order of a window's members' sessions in the device's log

A window's members' sessions run in one session of the challenger. Of a member's session its log holds the coin
tree's `Hello` (`Frame.hello`, the release log's `coin-tree` line) and each round it opened (`Frame.opening`, a
`tree-open` line), after the bytes the round answers: the SHA-512 of the message the round follows. A member's
evidence, its session's record and proof files, is none of them. It reaches the verifier with the play's evidence,
after the device's close (`Verity.finish`), so the log of itself holds no member's evidence before a later member's
coins, and a strategy may choose a member's proof files after it has seen every member's coins.

What a log shows where the members' sessions are one after another and each member's evidence entry is fixed, up to a
SHA-512 collision, before the next member's coins, over the members' coin trees (`tree j`, the index of member `j`'s
among the trees the log shows sampled) and their evidence entries (`entry j`):

* **The sessions one after another** (`OpensInOrder`): every round opened before a round of member `j`'s coin tree is
  of that tree or an earlier member's.
* **A member's evidence recorded before a later member's coins** (`EvidenceRecorded`): before a round of member `j`'s
  coin tree is opened, the log holds, for each earlier member, a release that answers the SHA-512 of that member's
  evidence entry. The device releases coins only after the bytes they answer, so those bytes were sent before any coin
  of member `j`'s rounds was released. Any release may carry them: a family's coin (`Frame.fresh`, the release log's
  `root` line, 64 bytes) answers a root, and a round's opening 64 bytes.
* `SessionsInOrder`: both. It holds of a log's prefixes (`SessionsInOrder.of_append`) and, of one member, of every log
  (`sessionsInOrder_one`). `EvidenceRecorded` gives the release (`EvidenceRecorded.recorded_before`) and fails at a
  log that opens a second member's round after nothing (`not_evidenceRecorded_of_unrecorded`).
-/

@[expose] public section

namespace Verity.Sampled

/-- **The sessions one after another**, in the device's log `log`, at `m` members whose coin trees are the `tree j`-th
the log shows sampled: every round opened before a round of member `j`'s tree is of that tree or an earlier member's. -/
def OpensInOrder (tree : ℕ → ℕ) (m : ℕ) (log : List Release) : Prop :=
  ∀ pre r post, log = pre ++ r :: post → ∀ j, j < m → ∀ s n, r.frame = .opening (tree j) s n →
    ∀ q ∈ pre, ∀ j', j' < m → ∀ s' n', q.frame = .opening (tree j') s' n' → j' ≤ j

/-- **Each member's evidence is recorded before a later member's coins**, in the device's log `log`: before a round of
member `j`'s coin tree is opened, the log holds, for each earlier member `j'`, a release that answers the SHA-512 of
that member's evidence entry `entry j'`. -/
def EvidenceRecorded (tree : ℕ → ℕ) (entry : ℕ → ByteArray) (m : ℕ) (log : List Release) : Prop :=
  ∀ pre r post, log = pre ++ r :: post → ∀ j, j < m → ∀ s n, r.frame = .opening (tree j) s n →
    ∀ j', j' < j → ∃ q ∈ pre, q.answers = Flock.Sha512.hash (entry j')

/-- **The members' sessions are one after another, each member's evidence recorded before the next member's coins**
(`OpensInOrder`, `EvidenceRecorded`). -/
structure SessionsInOrder (tree : ℕ → ℕ) (entry : ℕ → ByteArray) (m : ℕ) (log : List Release) : Prop where
  opens : OpensInOrder tree m log
  recorded : EvidenceRecorded tree entry m log

theorem OpensInOrder.of_append {tree : ℕ → ℕ} {m : ℕ} {l₁ l₂ : List Release}
    (h : OpensInOrder tree m (l₁ ++ l₂)) : OpensInOrder tree m l₁ :=
  fun pre r post e => h pre r (post ++ l₂) (by rw [e, List.append_assoc, List.cons_append])

theorem EvidenceRecorded.of_append {tree : ℕ → ℕ} {entry : ℕ → ByteArray} {m : ℕ} {l₁ l₂ : List Release}
    (h : EvidenceRecorded tree entry m (l₁ ++ l₂)) : EvidenceRecorded tree entry m l₁ :=
  fun pre r post e => h pre r (post ++ l₂) (by rw [e, List.append_assoc, List.cons_append])

/-- **The order holds of a log's prefixes.** -/
theorem SessionsInOrder.of_append {tree : ℕ → ℕ} {entry : ℕ → ByteArray} {m : ℕ} {l₁ l₂ : List Release}
    (h : SessionsInOrder tree entry m (l₁ ++ l₂)) : SessionsInOrder tree entry m l₁ :=
  ⟨h.opens.of_append, h.recorded.of_append⟩

/-- **At one member the order holds of every log**: no member comes before it. -/
theorem sessionsInOrder_one (tree : ℕ → ℕ) (entry : ℕ → ByteArray) {m : ℕ} (hm : m ≤ 1) (log : List Release) :
    SessionsInOrder tree entry m log :=
  ⟨fun _ _ _ _ j hj _ _ _ _ _ j' hj' _ _ _ => by omega, fun _ _ _ _ j hj _ _ _ j' hj' => by omega⟩

/-- **The release that records an earlier member's evidence**: where a round of member `j`'s coin tree is opened after
the log `pre`, `pre` holds a release that answers the SHA-512 of each earlier member's evidence entry. -/
theorem EvidenceRecorded.recorded_before {tree : ℕ → ℕ} {entry : ℕ → ByteArray} {m : ℕ} {pre post : List Release}
    {r : Release} (h : EvidenceRecorded tree entry m (pre ++ r :: post)) {j s n : ℕ} (hj : j < m)
    (hr : r.frame = .opening (tree j) s n) {j' : ℕ} (hj' : j' < j) :
    ∃ pre₁ q pre₂, pre = pre₁ ++ q :: pre₂ ∧ q.answers = Flock.Sha512.hash (entry j') := by
  obtain ⟨q, hq, hqa⟩ := h pre r post rfl j hj s n hr j' hj'
  obtain ⟨pre₁, pre₂, e⟩ := List.append_of_mem hq
  exact ⟨pre₁, q, pre₂, e, hqa⟩

/-- **A log that opens a round of the second member's coin tree after nothing records no evidence**: at two members
`EvidenceRecorded` fails there. -/
theorem not_evidenceRecorded_of_unrecorded (tree : ℕ → ℕ) (entry : ℕ → ByteArray) (s n : ℕ) (b : ByteArray)
    (cs : List Bool) : ¬ EvidenceRecorded tree entry 2 [⟨.opening (tree 1) s n, b, cs⟩] := fun h => by
  obtain ⟨q, hq, -⟩ := h [] _ [] rfl 1 (by omega) s n rfl 0 (by omega)
  cases hq

end Verity.Sampled
