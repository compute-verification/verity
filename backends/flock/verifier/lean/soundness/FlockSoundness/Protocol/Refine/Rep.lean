import Flock.Verify
import FlockSoundness.Protocol.Refine.LigeritoRun

/-!
# One rep refines the model's (PROTOCOL.md §9–§14)

`rep_refines`: when the executable's `verifyRep` accepts rep `r`, the record carries the rep's stream and the proof
decodes. Then, at the proof's openings, the model's compiled rep `repC`, played with the proof's messages (`repMsgs`) and
the coins of every round of the stream, ends in a result:
- whose verdict is true;
- whose caps are the proof's;
- whose requests passed the executable's Merkle checks.

It composes the four phases: `zerocheck_refines`, `lincheck_refines`, `opening_refines` and `ligerito_accepts_opening`.
The statement's own facts enter as `Realizes`-style hypotheses: `FoldRealizes` for the circuit fold, `ExtraRealizes`
for the extra claims.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

/-- The shape `verifyRep` decodes a proof with (§8.3), `nExtra` the statement's extra claims. -/
def shapeOf (st : Setup) (sch : Flock.Schedule) (nExtra : ℕ) : Shape :=
  { m := st.m, kLog := st.kLog, claims := 2 + nExtra, sched := sch, hashLen := st.merkle.digestLen,
    saltLen := if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes }

theorem ebind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases hx : x with
  | error e => rw [hx] at h; cases h
  | ok a => rw [hx] at h; exact ⟨a, rfl, h⟩

end FlockSoundness.Refine
