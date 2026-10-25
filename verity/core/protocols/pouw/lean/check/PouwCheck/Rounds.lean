import PouwWindow.Rounds
import PouwCheck.Inputs
import PouwCheck.Pass
import SampledProofs.ChallengerSession

/-!
# A live window's salt and round coins (`core_check.py`, "a live window's salt and round coins")

`saltCheck` is `core_check.salt`. It reads the window's coins and roots through the challenger's own session as
`roundsCheck` does, never ones the prover hands in: the one session the auditor's signed v2 ledger opens and ends closed
for the window under the window's pinned challenger key, run from a commit the host's pins log records and the
verifier's copy of main's first-parent history holds (`saltSession`, over `SampledProofs.Challenger.pinnedSession` and
`sessionNamed`).
S is the session the ledger's signed reserve entry names (a session opened without one is refused) and R the root its
signed open entry opened S on; R is `windowOpenRoot` of the window's weights' root, its round-0 request table's root and
S; the session's log shows R, alone, as what its salt coin followed; and under the commitment the session signed, a live
epoch's, the salt coin is the epoch's `pouw/salt` coin 0, opened after R (`SampledProofs.Challenger.liveReason`). A
refusal is `coin` (the key, the session or the coin rule's, in the challenger's words) or `salt` (a session opened
without a reserve, the roots not of their form, the open root another window's, or a log that shows another root), with
Python's words.

`roundsCheck` is `core_check.rounds`. It reads the challenger's coins only through the challenger's own session, never a
commitment the prover hands in: the one session the auditor's signed v2 ledger opens and ends closed for the window
(`{root}`) under the window's pinned challenger key, its record on the host the ledger opened it on and signed at the
leaf and over the commitment the ledger's end names, its close signed under that key, and its release log and refused
hellos the ones the ledger's end sealed, the host having run a commit its pins log records on main's first-parent line
(`SampledProofs.Challenger.trustedSessionOf`, over `pinnedSession`). Under the commitment that session
signed, round `r`'s coin opened once,
after the `r`-th round root, every recorded round's coin opened (`PouwWindow.roundsReason`), and the round roots are the
ones the session's log shows the challenger received, in order. A refusal is `coin` (the key, the session or the coin
rule's, in the challenger's words) or `rounds` (the roots or the openings not lists of their kind, or roots the
challenger didn't receive), with Python's words.
-/

namespace PouwCheck

open Lean (Json)
open SampledProofs.Challenger (ChallengerKey LiveOpening trustedSessionOf)

/-- `core_check.SALT_FORM`, `OTHER_WINDOW` and `NOT_RECEIVED`: with `NO_RESERVE`, the reasons of a `salt` refusal. -/
def SALT_FORM : String := "the window's weights' root and its round-0 request table's root are not SHA-512 digests"
def OTHER_WINDOW : String :=
  "the session's open root is not the window's: salt.window_open_root of its weights' root, its round-0 " ++
    "request table's root and the session"
def NOT_RECEIVED : String :=
  "the challenger's log does not show the window's open root as the one root its salt coin followed"

/-- `core_check.salt`: none when `opening` is the window's salt coin, opened after the window's own open root, its
session S and open root R the ones the auditor's signed reserve and open entries name and its coins and log the
challenger's session's, else the refusal. -/
def saltCheck (key ledger pins main window session weights requests opening : Json) : R Unit := do
  let (a, c, received) ← saltSession key ledger pins main window session
  if let some why := PouwWindow.liveEpochReason c then coin why
  let named := sessionEntries ledger a.session
  let (some _, some opened) := (named.find? (·.kind == "reserve"), named.find? (·.kind == "open"))
    | throw ("salt", NO_RESERVE)
  let some r := hex? (get (get opened.body "window") "root") | throw ("salt", NO_RESERVE)
  let digest := fun (j : Json) => if (str? j).any isHexDigest then hex? j else none
  let (some w, some t) := (digest weights, digest requests) | throw ("salt", SALT_FORM)
  if windowOpenRoot w t a.session != r then throw ("salt", OTHER_WINDOW)
  unless (received.lookup PouwWindow.SALT).getD [] == [r] do throw ("salt", NOT_RECEIVED)
  let o ← (LiveOpening.ofJson opening).mapError fun e => ("coin", s!"the salt's {e}")
  if let some why ← port (SampledProofs.Challenger.liveReason c o PouwWindow.SALT 0 r) then coin s!"the salt's {why}"

/-- `core_check.rounds`: none when `openings` are the window's round coins over `roots`, read through the challenger's
session, else the refusal. -/
def roundsCheck (key ledger pins main window session roots openings : Json) : R Unit := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let (c, received) ← (trustedSessionOf ledger window k pins main session).mapError fun e =>
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
