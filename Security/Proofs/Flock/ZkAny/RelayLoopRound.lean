module

public import Proofs.Flock.ZkAny.RelayLoop

/-!
# A frame round's slot at the deployed loop

At a frame round's slot `envPre_inv`'s step reads the worker's value at the model's history (`LoopRead`'s round clause).
Here the session's side of it:

* `slot_round_open`: a round's slot is for a stream whose `Open` left. `request_round_slot`: before a stop the session
  lets a round at its slot through, the coin check (`coinsOf`) and the place (`State.place`) both passing.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session
open _root_.Flock.Firewall hiding State
open _root_.Flock (SessionSpec Stmt)

/-- **A round's slot is for a stream whose `Open` left.** -/
theorem slot_round_open {S : State} {P : Placed} {s : String} {n : Nat} (h : slot S P = some (.round s n)) :
    (S.labels.extract 0 P.opens).any (·.1 == s) = true := by
  unfold slot at h
  split at h; · simp at h
  split at h; · simp at h
  split at h; · simp at h
  split at h; · simp at h
  split at h; · simp at h
  split at h
  · split at h
    · rename_i hany
      simp only [Option.some.injEq, Session.Kind.round.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact hany
    · simp at h
  · repeat' split at h
    all_goals simp at h

/-- **Before a stop the session lets a round at its slot through**: the round is the schedule's next opening, so the
coin check passes, and its stream's `Open` left, so its place does. -/
theorem request_round_slot {S : State} {P : Placed} {s : String} {n : Nat} (hR : Reach S P) (h : S.stop = none)
    (hsl : slot S P = some (.round s n)) : S.request P (.round s n) = none := by
  obtain ⟨hsent, hj, hlink, hq⟩ := slot_round hsl
  obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hR.sess hsent)
  rw [opened_of hC] at hj hq
  have hlt : C.opened.length - 1 < S.plan.size := (Array.getElem?_eq_some_iff.mp hq).1
  have hg : S.plan.getD (C.opened.length - 1) ("", 0) = (s, n) := by
    rw [Array.getD_eq_getD_getElem?, hq, Option.getD_some]
  have hc : coinsOf S (.round s n) = none := by
    have hgt : ¬ C.opened.length > S.plan.size := by omega
    simp [coinsOf, hC, reqAt, hj, hg, hgt, show (Session.Kind.round s n == .finish) = false from rfl,
      show (Session.Kind.round s n == .commit) = false from rfl]
  have hp : S.place P (.round s n) = none := by
    simp only [State.place, hlink, slot_round_open hsl, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  have e : S.request P (.round s n) = match coinsOf S (.round s n) with
      | some r => some r
      | none => (S.place P (.round s n)).map fun why => S.refuse s!"{Coins.STOP}{why}" := by
    unfold State.request; rw [h]; rfl
  rw [e]
  simp only [hc, hp, Option.map_none]

end Claims.ZKServer.Relay
