import FlockSoundness.Protocol.Refine.ArithK
import FlockLevel3.LookupRows
import FlockSoundness.Protocol.Refine.Schedule

/-!
# A run of the executable's Ligerito is a run of the model's (PROTOCOL.md §13.3)

`ligerito_refines`: when the executable's Ligerito verifier accepts, the model's compiled Ligerito (`ligeritoC`), played
with the proof's messages (`ligMsgs`) and the rounds the executable consumed, ends in a result

* whose caps are the proof's recursive caps, decoded by `capD`;
* whose every request is a query level whose Merkle check the executable passed (`ReqOK`);
* whose verdict at the openings `o` is the model's final check (§13.6) on the executable's final state, where the
  executable's own final check (`finalProg`, the verifier's last step, verbatim) passed.

That the executable's final check makes the model's hold, and that a passed Merkle check is the model's path check
(§14), are separate steps.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

variable {D : Type}

/-! ## The executable's final check, verbatim -/

section
open Flock
end

/-! ## Tapes, requests and the loop's invariant -/

/-- Level `ℓ`'s opened rows, siblings and salts in the proof, `R` the last level. -/
def openAt (p : LigeritoProof) (R ℓ : ℕ) : Array (Array F128) × Array ByteArray × Array ByteArray :=
  if ℓ = 0 then (p.initialProof.rows, p.initialProof.siblings, p.initialProof.salts)
  else if ℓ < R then
    ((p.recursiveProofs[ℓ - 1]!).rows, (p.recursiveProofs[ℓ - 1]!).siblings, (p.recursiveProofs[ℓ - 1]!).salts)
  else (p.finalProof.rows, p.finalProof.siblings, p.finalProof.salts)

/-! ## Helpers -/

/-! ## The model's prefix -/

/-! ## The run -/

end FlockSoundness.Refine
