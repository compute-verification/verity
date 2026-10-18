import PouwWindow.Rounds
import PouwCheck.Inputs
import PouwCheck.Pass

/-!
# A live window's round coins (`core_check.py`, "a live window's round coins")

`roundsCheck` is `core_check.rounds`: under the proof service's challenger's live coin commitment
(`SampledProofs.Challenger.LiveCommitment`), round `r`'s coin opened once, after the `r`-th round root the verifier
recorded, and every recorded round's coin opened (`PouwWindow.roundsReason`). A refusal is `coin` (the coin rule's, or a
malformed commitment or opening, in the challenger's words) or `rounds` (the roots or the openings not lists of their
kind), with Python's words.
-/

namespace PouwCheck

open Lean (Json)
open SampledProofs.Challenger (LiveCommitment LiveOpening)

/-- `core_check.rounds`: none when `openings` are the window's round coins over `roots`, else the refusal. -/
def roundsCheck (coins roots openings : Json) : R Unit := do
  let c ← (LiveCommitment.ofJson coins).mapError fun e => ("coin", s!"the verifier's coins are malformed: {e}")
  let some rs := (match roots with
      | .arr rs => rs.toList.mapM fun r => if (str? r).any isHexDigest then hex? r else none
      | _ => none) | throw ("rounds", "the round roots are not a list of SHA-512 digests")
  let .arr os := openings | throw ("rounds", "the round coins are not a list")
  let os ← os.toList.zipIdx.mapM fun (o, i) => (LiveOpening.ofJson o).mapError fun e => ("coin", s!"round coin {i}: {e}")
  if let some why := (← port (PouwWindow.roundsReason c rs os)) then coin why

end PouwCheck
