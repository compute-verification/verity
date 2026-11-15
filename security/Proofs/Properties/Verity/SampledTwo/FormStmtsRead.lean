module

public import Proofs.Properties.Verity.SampledTwo.FormCount
public import Proofs.Properties.Verity.SampledTwo.FormVerdicts
public import Proofs.Properties.Verity.SampledTwo.FormStmts
public import Proofs.Properties.Verity.SampledTwo.FormSetup
public import Proofs.Properties.Verity.SampledTwo.HeldOfAtDesc
public import Proofs.Properties.Verity.SampledTwo.AcceptsOfSteps
public import Proofs.Properties.Verity.SampledTwo.SessionLive

/-!
# Towards `StmtsRead` as a theorem (`stmtsRead_of`)

`StmtsRead` (`OfRun.lean`) says an accepted play's sessions' reading has a statements' message `m₂` whose statements are
the ones the run of record's second stage checked (`StmtsAre`). The reading states one statement a session
(`SessionMsgs.stmtsMsg`), and the run's second stage reads one at each position of the evidence's `stmts`
(`VStar.Recursive.SampledTwo.stmtsOwn`). The two read the same files: the session statement the reading prints
(`SessionMsgs.stmtAtCase`, a `stmtRaw`) and the statement the second stage makes
(`VStar.Recursive.SampledTwo.stmtOwnAt`, a `stmtOf`) are the same member's, at the same public file, the same record's
sample and the same registered reads, and the template name each gives agrees because the held copy's names are the
interior's circuits' (`heldOf`, `fields_of_accept`). The two JSON readers are one.

These are the building blocks the reconciliation stands on: the two JSON readers are one (`jsonOf_toOption`,
`headerOf_eq`), and a held copy's names are the interior's circuits' (`names_of_heldOf`). The statement's existence comes
from the second stage's count check (`stmtsSessions`, `accepts_iff_steps`) and its fields from what the accept verified
(`verified_at`); `PinHeldAt` is not needed here.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-- **`verity-verify-proof verify`'s JSON reader is the one `stmtOf`/`stmtOwnAt` use**: `FlockDeploy.jsonOf`'s
`.toOption` (dropping its error message) is `VStar.Recursive.SampledTwo.jsonOfBytes`. -/
theorem jsonOf_toOption (what : String) (b : ByteArray) :
    (FlockDeploy.jsonOf what b).toOption = VStar.Recursive.SampledTwo.jsonOfBytes b := by
  unfold FlockDeploy.jsonOf VStar.Recursive.SampledTwo.jsonOfBytes
  cases h : String.fromUTF8? b with
  | none => simp [h]
  | some s =>
    simp only [h, Option.bind_some]
    cases Flock.JsonParse.parse s <;> rfl

/-- **The second stage reads a public file's header the way the reading does**: `SpotFlock.headerOf` and
`VStar.Recursive.SampledTwo.headerLine` are the same reader. -/
theorem headerOf_eq (file : ByteArray) :
    SpotFlock.headerOf file = VStar.Recursive.SampledTwo.headerLine file := rfl

/-- **A held copy's template names are the interior's circuits'**: `heldOf` sets `names` to the circuits' names, and the
frame leaves them untouched. -/
theorem names_of_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : SpotFlock.HeldOf.Unread}
    {h : SpotFlock.Held} (hh : SpotFlock.HeldOf.heldOf i it u = some h) :
    h.names = (it.circuits.map (·.1)).toList := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := SpotFlock.HeldOf.heldOf_spec hh
  rfl

end Verity.SampledTwo
