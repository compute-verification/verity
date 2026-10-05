import RunModel.Audit

/-!
`run-model`: the toy audit both ways.

* `auditor --depth D --units K -- CMD ARGS…`: Lean drives. The auditor's program is the main loop; it spawns the
  developer's worker `CMD ARGS…` and asks it for bytes. It prints the verdict, then replays its own transcript.
* `draw --depth D --units K` and `verify --depth D --root HEX --units U,… --openings FILE`: Python drives. Each is a
  stateless executable a driver runs as a subprocess, as `flock-verify draw` and `flock-verify verify` are today.
* `ping --n N --bytes B -- CMD ARGS…`: the cost of one request to a worker and its reply.
-/

open RunModel RunModel.Audit RunModel.Frame

def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

def natOf (args : List String) (k : String) (d : Nat) : Nat := ((argOf args k).bind String.toNat?).getD d

def workerCmd (args : List String) : Option (String × Array String) :=
  match args.dropWhile (· != "--") with
  | _ :: cmd :: rest => some (cmd, rest.toArray)
  | _ => none

def unhex (s : String) : Option ByteArray := do
  let digit (c : Char) : Option Nat :=
    if c.isDigit then some (c.toNat - 48) else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 87) else none
  let cs := s.toList
  if cs.length % 2 != 0 then none
  let mut out := ByteArray.empty
  for i in [0:cs.length / 2] do
    out := out.push ((← digit cs[2 * i]!) * 16 + (← digit cs[2 * i + 1]!)).toUInt8
  return out

def auditorCmd (args : List String) : IO UInt32 := do
  let some (cmd, wargs) := workerCmd args | IO.eprintln "auditor: the worker's command follows --"; return 2
  let (d, k) := (natOf args "--depth" 10, natOf args "--units" 16)
  let w ← Worker.spawn "developer" cmd wargs
  let log ← IO.mkRef #[]
  let v ← (auditor d k).run (Prog.Io.ofWorkers [w] log)
  IO.println s!"VERDICT {v.json}"
  let events := (← log.get).toList
  match (auditor d k).replay events with
  | .ok v' => IO.println s!"REPLAY {events.length} events, same verdict: {v'.json == v.json}"
  | .error e => IO.println s!"REPLAY refused: {e}"
  return (if v.accepted then 0 else 1)

def drawCmd (args : List String) : IO UInt32 := do
  let seed ← IO.getRandomBytes 32
  let units := draw seed (natOf args "--depth" 10) (natOf args "--units" 16)
  IO.println s!"DRAW \{\"seed\": \"{hex seed}\", \"units\": {units.toList}}"
  return 0

def verifyCmd (args : List String) : IO UInt32 := do
  let d := natOf args "--depth" 10
  let (some root, some us, some f) := ((argOf args "--root").bind unhex, argOf args "--units", argOf args "--openings")
    | IO.eprintln "verify: --root HEX --units U,… --openings FILE"; return 2
  let units := ((us.splitOn ",").filterMap String.toNat?).toArray
  let h ← IO.FS.Handle.mk f .read
  let mut v : Verdict := ⟨true, "", root, units⟩
  for u in units do
    let o ← Frame.read h (8 + 64 * d)
    if let .error e := checkOpening root d u o then
      v := ⟨false, s!"unit {u}: {e}", root, units⟩
      break
  IO.println s!"VERDICT {v.json}"
  return (if v.accepted then 0 else 1)

def pingCmd (args : List String) : IO UInt32 := do
  let some (cmd, wargs) := workerCmd args | IO.eprintln "ping: the worker's command follows --"; return 2
  let (n, size) := (natOf args "--n" 10000, natOf args "--bytes" 8)
  let w ← Worker.spawn "echo" cmd wargs
  let body := ByteArray.mk (Array.replicate size 0xA5)
  -- untimed: the worker's interpreter start
  for _ in [0:20] do
    Frame.write w.stdin body
    discard <| Frame.read w.stdout size
  let mut times : Array Nat := #[]
  for _ in [0:n] do
    let t0 ← IO.monoNanosNow
    Frame.write w.stdin body
    let reply ← Frame.read w.stdout size
    let t1 ← IO.monoNanosNow
    if reply != body then IO.eprintln "ping: the echo differs"; return 1
    times := times.push (t1 - t0)
  let sorted := times.qsort (· < ·)
  let at_ (q : Nat) := sorted[min (sorted.size - 1) (sorted.size * q / 1000)]!
  IO.println s!"PING n={n} bytes={size} p50={at_ 500} ns p99={at_ 990} ns p99.9={at_ 999} ns max={sorted.back!} ns"
  return 0

def main (args : List String) : IO UInt32 :=
  match args with
  | "auditor" :: rest => auditorCmd rest
  | "draw" :: rest => drawCmd rest
  | "verify" :: rest => verifyCmd rest
  | "ping" :: rest => pingCmd rest
  | _ => do IO.eprintln "usage: run-model auditor|draw|verify|ping …"; return 2
