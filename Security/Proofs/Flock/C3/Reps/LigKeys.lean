import Proofs.Flock.C3.Reps.SentKeys
import Proofs.Flock.C3.Reps.LigNames

/-!
# The Bool check at Ligerito's keys: every key list of `ligKeys` passes `wKey`

`ligKeys` lists, in `ligMsgs`' order, each message's keys (`msgK`), each out-of-domain value's (`oodK`) and the final
`yr`'s (`yrK`), with `recKeys` for the levels after the first. `ligKeys_wKey` puts each of those keys in w below the
schedule's counts, so each list passes `wKey` (`ligKeys_all`).
-/

namespace FlockC3.Reps

open Flock.RecAlgebra (Key)

theorem msgK_all {P : Key → Bool} {j : ℕ} (h : ∀ a < 2, ∀ b < 2, P (.msg j a b) = true) :
    (msgK j).all P = true := by
  simp [msgK, h 0 (by decide) 0 (by decide), h 0 (by decide) 1 (by decide), h 1 (by decide) 0 (by decide),
    h 1 (by decide) 1 (by decide)]

/-- **`recKeys`' lists pass `P`** at `n + 1` levels from message `msg` and out-of-domain value `ood`, given `P` at the
next `7n + 4` messages, the next `2n` out-of-domain values and the final `yr`. -/
theorem recKeys_all {P : Key → Bool} {ext : ℕ} (hy : ∀ j < 2 ^ (ext + 1), P (.yr j) = true) : ∀ (n msg ood : ℕ),
    (∀ j, msg ≤ j → j < msg + (7 * n + 4) → ∀ a < 2, ∀ b < 2, P (.msg j a b) = true) →
    (∀ j, ood ≤ j → j < ood + 2 * n → P (.ood j) = true) →
    ∀ ks ∈ recKeys ext (n + 1) msg ood, ks.all P = true := by
  intro n
  induction n with
  | zero =>
    intro msg ood hm _ ks hks
    simp only [recKeys, List.mem_append, List.mem_map, List.mem_range', List.mem_singleton] at hks
    rcases hks with ⟨i, ⟨i', hi', rfl⟩, rfl⟩ | rfl
    · exact msgK_all (hm _ (by omega) (by omega))
    · exact List.all_eq_true.2 fun x hx => by
        obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hx
        exact hy j (List.mem_range.1 hj)
  | succ n ih =>
    intro msg ood hm ho ks hks
    simp only [recKeys, List.mem_append, List.mem_map, List.mem_range', List.mem_cons, List.not_mem_nil,
      or_false] at hks
    rcases hks with ((⟨i, ⟨i', hi', rfl⟩, rfl⟩ | rfl | rfl | rfl | rfl | rfl) | rfl) | hks
    · exact msgK_all (hm _ (by omega) (by omega))
    · rfl
    · simp [oodK, ho ood (by omega) (by omega)]
    · exact msgK_all (hm _ (by omega) (by omega))
    · simp [oodK, ho (ood + 1) (by omega) (by omega)]
    · exact msgK_all (hm _ (by omega) (by omega))
    · exact msgK_all (hm _ (by omega) (by omega))
    · exact ih (msg + 7) (ood + 2) (fun j h1 h2 => hm j (by omega) (by omega))
        (fun j h1 h2 => ho j (by omega) (by omega)) ks hks

/-- **Ligerito's keys pass `wKey`**: every list of `ligKeys` at the shape's schedule. -/
theorem ligKeys_all (sh : Flock.Shape) : ∀ ks ∈ ligKeys sh.sched, ks.all (wKey sh) = true := by
  obtain ⟨m, o, l⟩ := ligKeys_wKey sh
  intro ks hks
  simp only [ligKeys, List.mem_append, List.mem_map, List.mem_range', List.mem_cons, List.not_mem_nil,
    or_false] at hks
  rcases hks with (((rfl | rfl | ⟨i, ⟨i', hi', rfl⟩, rfl⟩) | rfl | rfl | rfl | rfl | rfl) | rfl) | hks
  · simp [oodK, o 0 (by omega)]
  · exact msgK_all (m 0 (by omega))
  · exact msgK_all (m _ (by omega))
  · rfl
  · simp [oodK, o 1 (by omega)]
  · exact msgK_all (m _ (by omega))
  · simp [oodK, o 2 (by omega)]
  · exact msgK_all (m _ (by omega))
  · exact msgK_all (m _ (by omega))
  · generalize hR : sh.sched.levels.size - 1 = R at hks
    rcases R with _ | n
    · simp [recKeys] at hks
    · obtain ⟨m', o', y⟩ := l n hR
      exact recKeys_all y n _ _ (fun j _ h2 => m' j (by omega)) (fun j _ h2 => o' j (by omega)) ks hks

end FlockC3.Reps
