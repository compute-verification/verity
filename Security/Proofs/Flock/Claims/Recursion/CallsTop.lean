import Proofs.Flock.Claims.Recursion.CallsRep

/-!
# A verdict's SHA-512 calls on every run

A rep's verification calls SHA-512 for its Ligerito queries (`ligC`) and for at most each recorded round of its stream,
the transcript's potential at the start (`cl_verifyRep`).
-/

namespace FlockDeploy.KS

open Flock

theorem cl_throw_bind {α β : Type} (e : String) (f : α → LogV β) (k : ℕ) : CL (throw e >>= f) k := fun l => by
  have : (throw e >>= f : LogV β).go l = (.error e, l) := rfl
  rw [this]
  exact Nat.le_add_right _ _

theorem potT_init (rounds : Array Round) (d : ByteArray) : potT (Transcript.init rounds d) = rounds.size := rfl

/-- One step of a walk over a logged program, at any budget. -/
macro "cl_step" : tactic => `(tactic| first
  | with_reducible exact cl_throw_bind _ _ _
  | with_reducible exact calcCL.pure' _ _
  | with_reducible exact calcCL.mono (cl_throw _) (Nat.zero_le _)
  | with_reducible exact calcCL.mono (cl_liftM _) (Nat.zero_le _)
  | with_reducible refine calcCL.bind0 ?_ fun _ => ?_
  | with_reducible refine calcCL.ite_k ?_ ?_
  | split
  | dsimp only)

/-- A schedule's Ligerito calls. -/
abbrev ligC (sch : Schedule) : ℕ := (1 + (sch.levels.size - 1)) * qAll sch

/-- **A rep's verification makes at most `ligC sch + B` calls**, `B` bounding its session's streams' rounds, whether it
accepts or throws. -/
theorem cl_verifyRep (h : ByteArray → ByteArray) (spec : SessionSpec) (st : Setup) (sch : Schedule) (sess : Session)
    (i r : ℕ) (bytes : ByteArray) (B : ℕ) (hB : ∀ s ∈ sess.streams, s.rounds.size ≤ B) :
    CL (verifyRepWith (Hashes.logged h) spec st sch sess i r bytes) (ligC sch + B) := by
  unfold verifyRepWith
  repeat' (first
    | with_reducible refine calcCL.bind_le (cl_run (pot := potT) (n := ligC sch) ?_ _) (fun _ => ?_) ?_
    | with_reducible exact cs_ligerito h _ _ _ _ _ _
    | with_reducible exact (calcCS potT).mono (cs_zerocheck h _ _ _ _) (Nat.zero_le _)
    | with_reducible exact (calcCS potT).mono (cs_lincheck h _ _ _) (Nat.zero_le _)
    | with_reducible exact (calcCS potT).mono (cs_ringSwitch h _ _ _) (Nat.zero_le _)
    | tw_step h
    | cl_step)
  all_goals
    rw [potT_init]
    have := hB _ (Array.mem_of_find?_eq_some (by assumption))
    omega

end FlockDeploy.KS
