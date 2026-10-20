import PouwWindow.Rounds
import PouwCheck.Inputs
import PouwCheck.Pass
import SampledProofs.ChallengerSession

/-!
# A live window's salt and round coins (`core_check.py`, "a live window's salt and round coins")

`saltCheck` is `core_check.salt`: under the proof service's challenger's live coin commitment, a live epoch's, the root the
auditor's signed open entry opened the session on is `windowOpenRoot` of the window's weights' root, its round-0 request
table's root and the session its signed reserve entry names; the session's log shows that root, alone, as what its salt
coin followed; and the salt coin is the epoch's `pouw/salt` coin 0, opened after it
(`SampledProofs.Challenger.liveReason`). A refusal is `coin` (the coin rule's, or a malformed commitment or opening, in
the challenger's words) or `salt` (the roots or the session not of their form, the open root another window's, or a log
that shows another root), with Python's words.

`roundsCheck` is `core_check.rounds`. It reads the challenger's coins only through the challenger's own session, never a
commitment the prover hands in: the one session the auditor's signed ledger opens and ends closed for the window under
the window's pinned challenger key, its record on the host the ledger opened it on and its close signed under that key,
and its release log the one the close signs and the ledger's end sealed (`SampledProofs.Challenger.trustedSessionOf`). Under the commitment that session signed, round `r`'s coin opened once,
after the `r`-th round root, every recorded round's coin opened (`PouwWindow.roundsReason`), and the round roots are the
ones the session's log shows the challenger received, in order. A refusal is `coin` (the key, the session or the coin
rule's, in the challenger's words) or `rounds` (the roots or the openings not lists of their kind, or roots the
challenger didn't receive), with Python's words.
-/

namespace PouwCheck

open Lean (Json)
open SampledProofs.Challenger (ChallengerKey LiveCommitment LiveOpening trustedSessionOf)

/-- `core_check.SALT_FORM`, `OTHER_WINDOW` and `NOT_RECEIVED`: the reasons of a `salt` refusal. -/
def SALT_FORM : String :=
  "the session's open root, the window's weights' root and its round-0 request table's root are not SHA-512 " ++
    "digests, or the session is not a gate's session id"
def OTHER_WINDOW : String :=
  "the session's open root is not the window's: salt.window_open_root of its weights' root, its round-0 " ++
    "request table's root and the session"
def NOT_RECEIVED : String :=
  "the challenger's log does not show the window's open root as the one root its salt coin followed"

/-- `core_check.salt`: none when `opening` is the window's salt coin, opened after the window's own open root `root`,
else the refusal. -/
def saltCheck (coins session root weights requests received opening : Json) : R Unit := do
  let c ← (LiveCommitment.ofJson coins).mapError fun e => ("coin", s!"the verifier's coins are malformed: {e}")
  if let some why := PouwWindow.liveEpochReason c then coin why
  let digest := fun (j : Json) => if (str? j).any isHexDigest then hex? j else none
  let (some r, some w, some t, some s) := (digest root, digest weights, digest requests, (str? session).filter sessionId?)
    | throw ("salt", SALT_FORM)
  if windowOpenRoot w t s != r then throw ("salt", OTHER_WINDOW)
  if received != .arr #[root] then throw ("salt", NOT_RECEIVED)
  let o ← (LiveOpening.ofJson opening).mapError fun e => ("coin", s!"the salt's {e}")
  if let some why ← port (SampledProofs.Challenger.liveReason c o PouwWindow.SALT 0 r) then coin s!"the salt's {why}"

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
