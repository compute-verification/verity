import Flock

open Flock

def parseHex128 (s : String) : Option F128 := do
  if s.length != 32 then none
  let v ← (String.toNat? ("0x" ++ s)) <|> (s.foldl (fun acc c => acc.bind fun n =>
      let d := if c.isDigit then some (c.toNat - 48) else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 87) else none
      d.map (fun d => n * 16 + d)) (some 0))
  pure ⟨(v % 2^64).toUInt64, (v / 2^64).toUInt64⟩

def fieldTest (path : String) : IO UInt32 := do
  let lines ← IO.FS.lines path
  let mut bad := 0
  for l in lines do
    match (l.splitOn " ").map parseHex128 with
    | [some a, some b, some p, some i] =>
      if a * b != p || F128.inv a != i then bad := bad + 1
    | _ => bad := bad + 1
  IO.println s!"field-test: {lines.size} cases, {bad} failures; phi8(2) = {(phi8 2).toHex}"
  return (if bad == 0 then 0 else 1)

def bytesOfHex (s : String) : ByteArray := match unhex s with | .ok b => b | .error _ => ByteArray.empty

def hashTest (path : String) : IO UInt32 := do
  let lines ← IO.FS.lines path
  let mut bad := 0
  for l in lines do
    match l.splitOn " " with
    | [kind, key, msg, want] =>
      let m := bytesOfHex msg
      let got := match kind with
        | "sha256" => Sha256.hash m
        | "blake3" => Blake3.hash m
        | _ => Blake3.keyedHash (bytesOfHex key) m
      if hexOf got != want then
        bad := bad + 1
        IO.println s!"MISMATCH {kind} len={m.size}"
    | _ => bad := bad + 1
  IO.println s!"hash-test: {lines.size} cases, {bad} failures"
  return (if bad == 0 then 0 else 1)

def compCheck (path : String) : IO UInt32 := do
  let b ← IO.FS.readBinFile path
  let t0 ← IO.monoMsNow
  match Comp.load b with
  | .ok c => IO.println s!"comp-check: ok, D_b3 {hexOf c.digest} ({(← IO.monoMsNow) - t0} ms)"; return 0
  | .error e => IO.println s!"comp-check: {e}"; return 1

def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

/-- `verify --circuit F --comp F --public F [--coins seed|os] --session DIR...`: each DIR holds `session.json`,
`nl.rep0.bin` and `nl.rep1.bin`. The statement is set up once; one verdict line per session, in order. Exit 0 iff every
session is accepted. -/
def verifyCmd (args : List String) : IO UInt32 := do
  let (some circuit, some comp, some pub) := (argOf args "--circuit", argOf args "--comp", argOf args "--public")
    | IO.eprintln "usage: flock-verify verify --circuit F --comp F --public F [--coins seed|os] --session DIR..."; return 2
  let dirs := (args.dropWhile (· != "--session")).drop 1
  let coins := if argOf args "--coins" == some "os" then Coins.os else Coins.seed
  let t0 ← IO.monoMsNow
  let setup ← match Stmt.setup (← IO.FS.readBinFile circuit) (← IO.FS.readBinFile comp) (← IO.FS.readBinFile pub) coins with
    | .ok st => pure (Except.ok (Setup.ofCircuit st))
    | .error e => pure (Except.error s!"setup: {e}")
  IO.eprintln s!"setup: {(← IO.monoMsNow) - t0} ms"
  let mut all := true
  for dir in dirs do
    let t1 ← IO.monoMsNow
    let record ← IO.FS.readFile (dir ++ "/session.json")
    let proofs ← #[0, 1].mapM fun r => IO.FS.readBinFile s!"{dir}/{TABLE}.rep{r}.bin"
    let verdict := do verify (← setup) record proofs
    let (ok, why) := match verdict with | .ok () => (true, "") | .error e => (false, e)
    all := all && ok
    let line := Lean.Json.mkObj [("session", Lean.Json.str dir), ("accepted", Lean.Json.bool ok), ("why", Lean.Json.str why)]
    IO.println s!"VERDICT {line.compress}"
    IO.eprintln s!"{dir}: {(← IO.monoMsNow) - t1} ms"
  return (if all then 0 else 1)

def main (args : List String) : IO UInt32 := do
  match args with
  | "verify" :: rest => verifyCmd rest
  | ["comp-check", path] => compCheck path
  | ["field-test", path] => fieldTest path
  | ["hash-test", path] => hashTest path
  | _ => IO.eprintln "usage: flock-verify field-test FILE"; return 2
