import Lean.Data.Json
import FlockAudit.Frame
import FlockAudit.Prog
import FlockAudit.Transcript

/-!
# The host side of a run: workers, the content store, and the coins

A worker is a process `flock-audit` spawns, with piped stdin and stdout, that serves one or more channels: each request to
one of its channels is a frame on its stdin, and its reply is the next frame on its stdout. The channel `archive` is the
auditor's content store, answered in-process from `DIR/sha512/<key>`: `0x01` and the object held under the key, or `0x00`
when none is (the program holds every object to its key). Every request and draw is appended to the log, which is the
run's transcript.
-/

namespace FlockAudit

open Prog

abbrev Child := IO.Process.Child { stdin := .piped, stdout := .piped, stderr := .inherit }

structure Worker where
  name : String
  channels : Array String
  child : Child

/-- `{"workers": [{"name", "channels": [...], "argv": [...]}, ...]}`: each worker's channels and command line. -/
def Worker.parseSpecs (j : Lean.Json) : Except String (Array (String × Array String × Array String)) := do
  let ws ← j.getObjValAs? (Array Lean.Json) "workers"
  ws.mapM fun w => do
    let name ← w.getObjValAs? String "name"
    let channels ← w.getObjValAs? (Array String) "channels"
    let argv ← w.getObjValAs? (Array String) "argv"
    if argv.isEmpty then throw s!"worker {name}: an empty argv"
    if channels.contains "archive" then throw s!"worker {name}: the archive channel is the auditor's own"
    return (name, channels, argv)

def Worker.spawn (name : String) (channels argv : Array String) : IO Worker := do
  let child ← IO.Process.spawn { cmd := argv[0]!, args := argv.extract 1 argv.size, stdin := .piped, stdout := .piped,
                                 stderr := .inherit }
  return { name, channels, child }

/-- The archive's reply to `key`: `0x01` and the object, or `0x00`. -/
def archiveReply (dir : String) (key : ByteArray) (max : Nat) : IO ByteArray := do
  let some k := String.fromUTF8? key | return ByteArray.mk #[0]
  if k.length != 128 || !k.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) then return ByteArray.mk #[0]
  let p : System.FilePath := dir ++ s!"/sha512/{k}"
  if !(← p.pathExists) then return ByteArray.mk #[0]
  if (← p.metadata).byteSize.toNat + 1 > max then
    throw (IO.userError s!"archive: the object held under {k} is over the request's bound of {max}")
  return ByteArray.mk #[1] ++ (← IO.FS.readBinFile p)

/-- The interpreter's answers over `ws`, the content store `dir` and the coin source `random`, each event appended to
`log`. A reply over its request's bound, a channel no worker serves, or a draw of the wrong length throws. -/
def hostAsk (ws : Array Worker) (dir : String) (log : IO.Ref (Array Event)) (r : Req) : IO ByteArray := do
  let reply ← if r.chan == "archive" then archiveReply dir r.body r.max
    else
      let some w := ws.find? (·.channels.contains r.chan) | throw (IO.userError s!"no worker serves {r.chan}")
      Frame.write w.child.stdin r.body
      Frame.read w.child.stdout r.max
  log.modify (·.push (.asked r.chan r.body reply))
  return reply

def hostRandom (random : Nat → IO ByteArray) (log : IO.Ref (Array Event)) (n : Nat) : IO ByteArray := do
  let b ← random n
  if b.size != n then throw (IO.userError s!"random: {b.size} bytes, not {n}")
  log.modify (·.push (.drew b))
  return b

end FlockAudit
