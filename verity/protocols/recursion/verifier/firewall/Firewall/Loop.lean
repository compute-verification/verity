/-
# The release side around the session loop (`core-firewall session`, live PROTOCOL.md §10.5)

`Flock.Firewall.Session`'s loop, each of its sessions one outer session of the recursive session `--session` at the
statement `--part`, held to the record's rules (`Firewall.Record`, `Firewall.Release`) on the ZK server's state directory
`--state`, the files `verity_flock.firewall` keeps: `record.jsonl`, read whole and appended under `record.lock` at each
op, and `slot.lock`, the outer slot. Both are `flock(2)` locks (`IO.FS.Handle.lock`), so Python's processes and these
exclude each other.

* At a session's first request other than `Ping`, before anything of it reaches the auditor, it takes the slot (a
  session that finds it taken records `wait`, then waits for it) and records its slot and its admission (`seat`). A
  session refused there reaches nothing of the auditor.
* With `--salt-key FILE` its statement's registrations are admitted too (rule 5), once the auditor has answered its
  `Register` and before the prover receives the answer, so before `Hello` leaves (`seatReads`): of the session only
  `Register`, one constant byte, has left. What follows the draw in that answer is the verifier's list of the statement's
  known trees (`Salts.parseKnown`; no byte is the list of none), which the loop cuts off the answer. A tree the list
  names reads no registration and is recorded (`known`); every other hm96 tree of the public file must be a table of
  a registration on the record whose record tag is the server's under the key in `FILE`, or, for an interior
  session's, under the session's own key on the record, which the key in `FILE` vouches for there
  (`Salts.readsKnown`), and the registrations it reads are admitted, oldest first, those it passes over skipped
  (`admitReads`). A session refused there has sent the auditor its `Register` and sends nothing more. The list is the
  session's alone: the loop reads one nowhere else and keeps none for a later session.
* Its proofs and `Finish` are held. At `Finish`, `Decision.after` decides at the wall clock's time: the prover's run
  is the loop's own refusal of the session if it made one (`State.finish`), and the send check is whether the loop holds
  the session's proofs, two a table. A stop is recorded, the prover is told of it in a verdict that does not accept,
  and the auditor's connection is cut without `Finish`. A session that goes waits for its release time; then its proofs
  and `Finish` leave, and the auditor's verdict is its release or its stop. Either way the loop's stderr gets one
  `FIREWALL-TIMES` line: the time it decided (`ready_ms`), the release time and the decision, which is what a run times
  its release windows by. Nothing of it reaches the auditor.
* An admitted session cut before its `Finish` was decided, or a fault of the loop's, is its stop.

`core-firewall values --state DIR --salt-key FILE` is the ZK server's value service (`Flock.Firewall.Values`) on the same
state directory: each registration takes the next number on the record and is appended there whole, under
`record.lock`, in the step that numbers it (`register`), an interior session's with the session's own key. That key is
witness, so before it serves, the service makes the state directory its own OS user's alone.
-/
module

public import Firewall.Release
public import Flock.Firewall.Session
public import Flock.Firewall.Values
public import Std.Time

@[expose] public section

namespace Core.Firewall.Loop

open Flock (V)

/-- The lock file `name` in the state directory `dir`, created if absent. -/
def lockFile (dir : System.FilePath) (name : String) : IO IO.FS.Handle :=
  IO.FS.Handle.mk (dir / name) .append

/-- `f` on the record as one op of the ZK server's: under `record.lock`, `record.jsonl` read whole (`Record.op` reads it
again and refuses a line the ZK server doesn't write), and the lines `f` adds appended. -/
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

/-- **One outer session of `s` at `part`** on the ZK server whose state directory is `dir`, as the loop's hold. If the
loop admits its statement's registrations, `reads` gives them, with its known trees, on the record's events and the
verifier's list (`seatReads`). -/
def hold (dir : System.FilePath) (s part : String)
    (reads : Option (Events → ByteArray → Except String (Array Nat × Array ByteArray))) :
    IO Flock.Firewall.Session.Hold := do
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
    known := fun list => do
      let some f := reads | pure none
      match ← withRecord dir (seatReads · s part (f · list)) with
      | .ok () => pure none
      | .error why => pure (some why)
    barrier := fun stopped proofs need => do
      let some a ← seated.get | throw (IO.userError "a barrier before the session's admission")
      let prove : Run := match stopped with
        | some why => .err why
        | none => .ok
      let ready : Run :=
        if proofs == need then .ok else .err s!"the prover sent Finish with {proofs} of the session's {need} proofs"
      let now ← nowMs
      let times := fun (out : String) => do
        IO.eprintln s!"FIREWALL-TIMES {Lean.Json.compress (Lean.Json.mkObj [("session", .str s), ("part", .str part),
          ("ready_ms", Lean.toJson now), ("release_ms", match a.releaseMs with | some t => Lean.toJson t | none => .null),
          ("outcome", .str out)])}"
        (← IO.getStderr).flush
      match Decision.after a now prove ready with
      | .go a =>
        times "go"
        if let some t := a.releaseMs then sleepUntilMs t
        pure none
      | .stop why | .refused why =>
        times "stop"
        recordStop why
        pure (some (told why))
    sent := fun v => discard <| withRecord dir (settle · s part (v.mapError (s!"the firewall's fault ({·})")))
    cut := recordStop
    leave := do
      if let some h ← slot.get then h.unlock
      slot.set none
  }

/-- The registrations the statement whose public file is `publicFile` reads, with its known trees, on the record's
events and the verifier's list of them, under the salt key in `kf`: its hm96 trees read once, and each either an
entry of the list or read against the registrations on the record (`Salts.readsKnown`). -/
def readsOf (kf : String) (publicFile : ByteArray) :
    IO (Except String (Events → ByteArray → Except String (Array Nat × Array ByteArray))) := do
  match ← Flock.Firewall.Salts.loadKey kf false with
  | .error e => return .error e
  | .ok k =>
    match Flock.Firewall.Salts.treesOfFile publicFile with
    | .error e => return .error s!"--salt-key: {e}"
    | .ok trees => return .ok fun evs list => Flock.Firewall.Salts.readsKnown k (registrations evs) trees list

/-- A known tree's entry on the record is its entry in the verifier's list, whole. -/
example : KNOWN_TREE_BYTES = Flock.Firewall.Salts.KNOWN_BYTES := rfl

/-- `core-firewall session --state DIR --session ID --part NAME [--salt-key FILE]` and `flock-firewall session`'s flags:
the session loop (`Session.run`), each of its sessions an outer session of the recursive session `ID` at `NAME`, held to
the record in `DIR` (`hold`), and with `--salt-key` its statement's registrations admitted (`readsOf`). -/
def main (args : List String) : IO UInt32 := do
  let usage := "usage: core-firewall session --state DIR --session ID --part NAME [--salt-key FILE] --verifier HOST:PORT " ++
    s!"[--listen PORT] {Flock.Firewall.Hello.FLAGS}"
  let (some dir, some s, some part) := (Flock.Witness.argOf args "--state", Flock.Witness.argOf args "--session",
      Flock.Witness.argOf args "--part")
    | IO.eprintln usage; return 2
  IO.FS.createDirAll dir
  Flock.Firewall.Session.run usage args <| some fun S => do
    match Flock.Witness.argOf args "--salt-key" with
    | none => return .ok (hold dir s part none)
    | some kf => return (← readsOf kf S.I.publicFile).map fun f => hold dir s part (some f)

/-- `core-firewall values --state DIR --salt-key FILE [--listen PORT]`: the ZK server's value service
(`Flock.Firewall.Values.serve`), each registration numbered on the record in `DIR` and appended there (`register`).
Before it serves it makes `DIR` its user's alone to read, write and enter, and it serves nothing where it can't. -/
def values (args : List String) : IO UInt32 := do
  let usage := "usage: core-firewall values --state DIR --salt-key FILE [--listen PORT]"
  let some dir := Flock.Witness.argOf args "--state" | IO.eprintln usage; return 2
  IO.FS.createDirAll dir
  try IO.setAccessRights dir { user := { read := true, write := true, execution := true } }
  catch e =>
    IO.eprintln (s!"--state {dir}: {e}: the record holds interior sessions' keys, so the state directory is the ZK " ++
      "server's user's alone")
    return 2
  Flock.Firewall.Values.serve usage args <| some fun rec => do
    try
      return .ok (← withRecord dir fun r => r.op fun evs => let ((_, b), evs) := register evs rec; (b, evs))
    catch e => return .error s!"{e}"

end Core.Firewall.Loop
