module

public import Definitions.Core.Interaction

/-!
# A run's sessions in sequence

A sampled verifier that runs its sessions one after another gives each its slot of rounds: after a prefix of `pre`
rounds, session `i`'s rounds start at `sessOff pre kS i`, session `j` holding `kS j` of them. Inside its slot the
verifier answers as the session's Flock verifier does, with an acceptance shown as a message `mark` (`markOut`), so the
run goes on to the next session; and a session's transcript is read back off the run's from its slot's first round
(`cutSeen`), `mark` read as the acceptance it shows.
-/

@[expose] public section

namespace Claims.Sampling

/-- `o` with an acceptance shown as the message `mark`. -/
def markOut (mark : Msg) : Out → Out
  | .msg m => .msg m
  | .acc => .msg mark
  | .rej => .rej

/-- A transcript entry with its acceptance shown as `mark`. -/
def markE (mark : Msg) (e : Msg × Out) : Msg × Out := (e.1, markOut mark e.2)

/-- **A session's transcript, read off the run's from the session's first round**: at most `k` entries, the message
`mark` read as an acceptance, up to the first acceptance or refusal; refused where `k` entries show neither. -/
def cutSeen (mark : Msg) : ℕ → List (Msg × Out) → Seen
  | 0, _ => ⟨[], .rej⟩
  | _ + 1, [] => ⟨[], .rej⟩
  | k + 1, (a, .msg m) :: rest =>
    if m = mark then ⟨[(a, .acc)], .acc⟩ else ⟨(a, .msg m) :: (cutSeen mark k rest).msgs, (cutSeen mark k rest).out⟩
  | _ + 1, (a, .acc) :: _ => ⟨[(a, .acc)], .acc⟩
  | _ + 1, (a, .rej) :: _ => ⟨[(a, .rej)], .rej⟩

/-- **Session `i`'s first round**: after a prefix of `pre` rounds and sessions `0, …, i − 1` of `kS j` rounds each. -/
def sessOff (pre : ℕ) (kS : ℕ → ℕ) : ℕ → ℕ
  | 0 => pre
  | i + 1 => sessOff pre kS i + kS i

theorem le_sessOff (pre : ℕ) (kS : ℕ → ℕ) : ∀ i, pre ≤ sessOff pre kS i
  | 0 => le_rfl
  | i + 1 => (le_sessOff pre kS i).trans (Nat.le_add_right _ _)

theorem sessOff_mono (pre : ℕ) (kS : ℕ → ℕ) {i j : ℕ} (h : i ≤ j) : sessOff pre kS i ≤ sessOff pre kS j := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact ih.trans (Nat.le_add_right _ _)

/-- **A session ends by the next one's first round.** -/
theorem sessOff_add_le (pre : ℕ) (kS : ℕ → ℕ) {i j : ℕ} (h : i < j) : sessOff pre kS i + kS i ≤ sessOff pre kS j :=
  sessOff_mono pre kS (i := i + 1) h

end Claims.Sampling
