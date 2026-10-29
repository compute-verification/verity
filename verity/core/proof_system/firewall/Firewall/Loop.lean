/-
# The release side around the session loop (`core-firewall session`, live PROTOCOL.md §10.5)

`Flock.Firewall.Session`'s loop, each of its sessions one outer session of the recursive session `--session` at the
statement `--part`, held to the record's rules (`Firewall.Record`, `Firewall.Release`) on the firewall's state directory
`--state`, the files `verity_flock.firewall` keeps: `record.jsonl`, read whole and appended under `record.lock` at each
op, and `slot.lock`, the outer slot. Both are `flock(2)` locks (`IO.FS.Handle.lock`), so Python's processes and these
exclude each other.

* At a session's first request other than `Ping`, before anything of it reaches the auditor, it takes the slot (a
  session that finds it taken records `wait`, then waits for it) and, on one read of the record, records its slot and
  its admission (`seat`). A refused session reaches nothing of the auditor.
* Its proofs and `Finish` are held. At `Finish`, `Decision.after` decides at the wall clock's time: the prover's run
  is the loop's own refusal of the session if it made one (`State.finish`), and the send check is whether the loop holds
  the session's proofs, two a table. A stop is recorded, the prover is told of it in a verdict that does not accept,
  and the auditor's connection is cut without `Finish`. A session that goes waits for its release time; then its proofs
  and `Finish` leave, and the auditor's verdict is its release or its stop.
* An admitted session cut before its `Finish` was decided, or a fault of the loop's, is its stop.
-/
import Firewall.Release
import Flock.Firewall.Session
import Std.Time

namespace Core.Firewall.Loop

open Flock (V)

/-- The lock file `name` in the state directory `dir`, created if absent. -/
def lockFile (dir : System.FilePath) (name : String) : IO IO.FS.Handle :=
  IO.FS.Handle.mk (dir / name) .append

/-- `f` on the record as one op of the firewall's: under `record.lock`, `record.jsonl` read whole (`Record.op` reads it
again and refuses a line the firewall doesn't write), and the lines `f` adds appended. -/
def withRecord {α : Type} (dir : System.FilePath) (f : Record → V (α × Record)) : IO α := do
  let lock ← lockFile dir "record.lock"
  lock.lock
  try
    let path := dir / "record.jsonl"
    let text ← if ← path.pathExists then IO.FS.readFile path else pure ""
    match f { text, events := #[] } with
    | .error e => throw (IO.userError s!"{path}: {e}")
    | .ok (a, r) =>
      let add := r.text.toUTF8.extract text.utf8ByteSize r.text.utf8ByteSize
      unless add.isEmpty do
        let out ← IO.FS.Handle.mk path .append
        out.write add
        out.flush
      pure a
  finally
    lock.unlock

/-- The wall clock, in milliseconds since the Unix epoch. -/
def nowMs : IO Nat := return (← Std.Time.Timestamp.now).toMillisecondsSinceUnixEpoch.val.toNat

/-- Returns once the wall clock reads `t` ms or later. -/
def sleepUntilMs (t : Nat) : IO Unit := do
  repeat
    let now ← nowMs
    if now ≥ t then break
    IO.sleep (min (t - now) 60000).toUInt32

/-- **One outer session of `s` at `part`** on the firewall whose state directory is `dir`, as the loop's hold. -/
def hold (dir : System.FilePath) (s part : String) : IO Flock.Firewall.Session.Hold := do
  let slot ← IO.mkRef (none : Option IO.FS.Handle)
  let seated ← IO.mkRef (none : Option Admitted)
  let recordStop := fun (why : String) => withRecord dir (stopAt · s part why)
  let told := fun (why : String) => s!"{why}: recursive session {s} stopped at {part}"
  pure {
    admit := do
      let h ← lockFile dir "slot.lock"
      unless ← h.tryLock do
        withRecord dir fun r => r.op fun evs => ((), evs.push (.wait s part))
        IO.eprintln s!"firewall: {s} at {part} waits for the outer slot"
        (← IO.getStderr).flush
        h.lock
      slot.set (some h)
      let admitted ← (withRecord dir (seat · s part)).tryCatch fun e => do
        h.unlock
        slot.set none
        throw e
      match admitted with
      | .ok a =>
        seated.set (some a)
        pure none
      | .error why => pure (some why)
    barrier := fun stopped proofs need => do
      let some a ← seated.get | throw (IO.userError "a barrier before the session's admission")
      let prove : Run := match stopped with
        | some why => .err why
        | none => .ok
      let ready : Run :=
        if proofs == need then .ok else .err s!"the prover sent Finish with {proofs} of the session's {need} proofs"
      match Decision.after a (← nowMs) prove ready with
      | .go a =>
        if let some t := a.releaseMs then sleepUntilMs t
        pure none
      | .stop why | .refused why =>
        recordStop why
        pure (some (told why))
    sent := fun v => discard <| withRecord dir (settle · s part (v.mapError (s!"the firewall's fault ({·})")))
    cut := recordStop
    leave := do
      if let some h ← slot.get then h.unlock
      slot.set none
  }

/-- `core-firewall session --state DIR --session ID --part NAME` and `flock-firewall session`'s flags: the session loop
(`Session.run`), each of its sessions an outer session of the recursive session `ID` at `NAME`, held to the record in
`DIR` (`hold`). -/
def main (args : List String) : IO UInt32 := do
  let usage := "usage: core-firewall session --state DIR --session ID --part NAME --verifier HOST:PORT [--listen PORT] " ++
    Flock.Firewall.Hello.FLAGS
  let (some dir, some s, some part) := (Flock.Witness.argOf args "--state", Flock.Witness.argOf args "--session",
      Flock.Witness.argOf args "--part")
    | IO.eprintln usage; return 2
  IO.FS.createDirAll dir
  Flock.Firewall.Session.run usage args (some (hold dir s part))

end Core.Firewall.Loop
