/-
Level 3, L3-S (PROTOCOL.md §6.2, §17.4 stage 1c's slot): session binding at the transcript. A squeeze that succeeds
returns exactly the recorded coins of the next round, and only after the framed bytes it answers are that round's retained
bytes (or hash to its recorded digest when the server kept none). So every coin the verifier reads came after, and is bound
to, the prover's messages before it.
-/
import Flock.Transcript

namespace Flock

open Transcript

theorem squeeze_answers (kind : UInt8) (n : Nat) (t t' : Transcript) (coins : Array F128) (hn : n ≠ 0)
    (h : squeeze kind n t = .ok (coins, t')) :
    ∃ r, t.rounds[t.k]? = some r ∧ r.n = n ∧ r.answers (framed t kind n) ∧ coins = r.coins ∧
      t' = { t with pending := ByteArray.empty, k := t.k + 1 } := by
  unfold squeeze at h
  simp only [hn, if_false] at h
  split at h
  · exact absurd h (by simp)
  · rename_i r hr
    by_cases h1 : r.n ≠ n
    · simp [h1] at h
    · by_cases h2 : ¬ r.answers (framed t kind n)
      · simp [h1, h2] at h
      · simp only [h1, h2, if_false] at h
        injection h with h
        obtain ⟨hc, ht⟩ := Prod.mk.inj h
        exact ⟨r, hr, Decidable.of_not_not h1, Decidable.of_not_not h2, hc.symm, ht.symm⟩

/-- With retained bytes, the answered bytes are the retained bytes exactly: no hash assumption. -/
theorem squeeze_retained (kind : UInt8) (n : Nat) (t t' : Transcript) (coins : Array F128) (hn : n ≠ 0)
    (h : squeeze kind n t = .ok (coins, t')) (r : Round) (hr : t.rounds[t.k]? = some r) (b : ByteArray)
    (hb : r.bytes = some b) : framed t kind n = b ∧ coins = r.coins := by
  obtain ⟨r', hr', _, ha, hc, _⟩ := squeeze_answers kind n t t' coins hn h
  rw [hr] at hr'
  cases hr'
  unfold Round.answers at ha
  rw [hb] at ha
  exact ⟨ha, hc⟩

end Flock
