import FlockAudit.Auditor
import FlockAudit.Worker
import FlockAudit.Transcript

/-!
# `flock-audit`'s commands

`run` is the auditor's program over its workers, with coins from a given source (`flock-audit`: the operating system's),
and `replay` the same program against a run's transcript. Both print one `VERDICT {"accepted", "why", "draw", "report"}`
line and exit 0 iff the verdict accepts. A run that cannot finish (a worker that exits, a reply over its bound, a channel no
worker serves) is refused.
-/

namespace FlockAudit.Cli

open Flock

def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

/-- `--zk --statement ID --circuit HEX --k K [--session-tables J]`: the auditor's own inputs. -/
def ownOf (args : List String) : Except String Own := do
  if !args.contains "--zk" then throw "flock-audit audits --zk sessions only (the keyed coin tree): pass --zk"
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuit.name)
    | throw s!"unknown statement; known: {Tags.all.map (·.name)}"
  if !tags.hm96Rows then throw s!"{tags.name}: flock-audit audits hm96-sha512 row statements"
  let some circuit := argOf args "--circuit" | throw "--circuit HEX (the circuit's SHA-512, the auditor's pin) is required"
  let some k := (argOf args "--k").bind String.toNat? | throw "--k K (a subset draw of K units) is required"
  let some count := ((argOf args "--session-tables").getD "1").toNat? | throw "--session-tables J"
  return { tags, circuit, k, count }

def printVerdict (v : Verdict) : IO UInt32 := do
  IO.println s!"VERDICT {v.toJson.compress}"
  return if v.accepted then 0 else 1

def usage : String :=
  "usage: flock-audit run --zk [--statement ID] --circuit HEX --k K [--session-tables J] --archive DIR --workers FILE " ++
  "[--transcript OUT]\n       flock-audit replay --zk [--statement ID] --circuit HEX --k K [--session-tables J] --transcript FILE"

/-- The auditor's program over the workers `--workers FILE` names and the content store `--archive DIR`, each coin node's
bytes from `random`; the transcript to `--transcript OUT`. -/
def run (args : List String) (random : Nat → IO ByteArray) : IO UInt32 := do
  let own ← match ownOf args with
    | .ok o => pure o
    | .error e => IO.eprintln e; IO.eprintln usage; return 2
  let (some archive, some wfile) := (argOf args "--archive", argOf args "--workers") | IO.eprintln usage; return 2
  let specs ← match Lean.Json.parse (← IO.FS.readFile wfile) >>= Worker.parseSpecs with
    | .ok s => pure s
    | .error e => IO.eprintln s!"--workers: {e}"; return 2
  let ws ← specs.mapM fun (n, c, a) => Worker.spawn n c a
  let log ← IO.mkRef #[]
  let v ← try Prog.runM (hostAsk ws archive log) (hostRandom random log) (auditor own)
    catch e => pure (refused none s!"run: {e}")
  if let some t := argOf args "--transcript" then Transcript.write t (← log.get)
  let drew := (← log.get).foldl (fun n e => match e with | .drew _ => n + 1 | _ => n) 0
  IO.eprintln s!"flock-audit: {(← log.get).size} events, {drew} coin requests"
  printVerdict v

/-- The auditor's program against the transcript `--transcript FILE`. -/
def replay (args : List String) : IO UInt32 := do
  let own ← match ownOf args with
    | .ok o => pure o
    | .error e => IO.eprintln e; IO.eprintln usage; return 2
  let some t := argOf args "--transcript" | IO.eprintln usage; return 2
  match Transcript.decode (← IO.FS.readBinFile t) with
  | .error e => printVerdict (refused none e)
  | .ok es => match Prog.replay (auditor own) es with
    | .ok v => printVerdict v
    | .error e => printVerdict (refused none e)

/-- `flock-audit run …`, its coins the operating system's (`IO.getRandomBytes`, as `Draw.drawOS`), and
`flock-audit replay …`. -/
def cmd (args : List String) : IO UInt32 :=
  match args with
  | "run" :: rest => run rest fun n => IO.getRandomBytes n.toUSize
  | "replay" :: rest => replay rest
  | _ => do IO.eprintln usage; return 2

end FlockAudit.Cli
