import PouwWindow.Rounds
import PouwCheck.Inputs
import PouwCheck.Pass
import SampledProofs.ChallengerSession

/-!
# A live window's round coins (`core_check.py`, "a live window's round coins")

`roundsCheck` is `core_check.rounds`. It reads the challenger's coins only through the challenger's own session, never a
commitment the prover hands in: the one session the auditor's signed v2 ledger opens and ends closed for the window
(`{root}`) under the window's pinned challenger key, its record on the host the ledger opened it on and signed at the
leaf and over the commitment the ledger's end names, its close signed under that key, and its release log and refused
hellos the ones the ledger's end sealed (`SampledProofs.Challenger.trustedSessionOf`). Under the commitment that session
signed, round `r`'s coin opened once,
after the `r`-th round root, every recorded round's coin opened (`PouwWindow.roundsReason`), and the round roots are the
ones the session's log shows the challenger received, in order. A refusal is `coin` (the key, the session or the coin
rule's, in the challenger's words) or `rounds` (the roots or the openings not lists of their kind, or roots the
challenger didn't receive), with Python's words.
-/

namespace PouwCheck

open Lean (Json)
open SampledProofs.Challenger (ChallengerKey LiveOpening trustedSessionOf)

/-- `core_check.rounds`: none when `openings` are the window's round coins over `roots`, read through the challenger's
session, else the refusal. -/
def roundsCheck (key ledger window session roots openings : Json) : R Unit := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let (c, received) ← (trustedSessionOf ledger window k session).mapError fun e =>
    ("coin", s!"the challenger's session is not trusted: {e}")
  let some rs := (match roots with
      | .arr rs => rs.toList.mapM fun r => if (str? r).any isHexDigest then hex? r else none
      | _ => none) | throw ("rounds", "the round roots are not a list of SHA-512 digests")
  let .arr os := openings | throw ("rounds", "the round coins are not a list")
  let os ← os.toList.zipIdx.mapM fun (o, i) => (LiveOpening.ofJson o).mapError fun e => ("coin", s!"round coin {i}: {e}")
  if let some why := (← port (PouwWindow.roundsReason c rs os)) then coin why
  unless (received.lookup PouwWindow.ROUND).getD [] == rs do
    throw ("rounds", "the round roots are not the roots the challenger's log shows it received, in order")

end PouwCheck
