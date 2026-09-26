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
  match Comp.load Blake3Row.spec b with
  | .ok c => IO.println s!"comp-check: ok, D_b3 {hexOf c.digest} ({(← IO.monoMsNow) - t0} ms)"; return 0
  | .error e => IO.println s!"comp-check: {e}"; return 1

def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

/-- `verify --statement ID --circuit F --comp F --public F [--coins seed|os] --session DIR...`: `ID` names the statement's
tags (`verity/flock-circuit` by default, or `verity/flock-netlist/v1`); `--comp` is the row leaf's pinned compression
circuit. Each DIR holds `session.json`, `<table>.rep0.bin` and `<table>.rep1.bin`. The statement is set up once; one
verdict line per session, in order. Exit 0 iff every session is accepted. -/
def verifyCmd (args : List String) : IO UInt32 := do
  let (some circuit, some comp, some pub) := (argOf args "--circuit", argOf args "--comp", argOf args "--public")
    | IO.eprintln "usage: flock-verify verify [--statement ID] --circuit F --comp F --public F [--coins seed|os] --session DIR..."; return 2
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuit.statement)
    | IO.eprintln s!"unknown statement; known: {Tags.all.map (·.statement)}"; return 2
  let dirs := (args.dropWhile (· != "--session")).drop 1
  let coins := if argOf args "--coins" == some "os" then Coins.os else Coins.seed
  let t0 ← IO.monoMsNow
  let (compB, circuitB, pubB) := (← IO.FS.readBinFile comp, ← IO.FS.readBinFile circuit, ← IO.FS.readBinFile pub)
  let setup ← match (do Stmt.setup tags (Blake3Row.leaf (← Comp.load Blake3Row.spec compB)) circuitB pubB coins : V Stmt) with
    | .ok st => pure (Except.ok (Setup.ofCircuit st))
    | .error e => pure (Except.error s!"setup: {e}")
  IO.eprintln s!"setup: {(← IO.monoMsNow) - t0} ms"
  let mut all := true
  for dir in dirs do
    let t1 ← IO.monoMsNow
    let record ← IO.FS.readFile (dir ++ "/session.json")
    let proofs ← #[0, 1].mapM fun r => IO.FS.readBinFile s!"{dir}/{tags.table}.rep{r}.bin"
    let verdict := do verify (← setup) record proofs
    let (ok, why) := match verdict with | .ok () => (true, "") | .error e => (false, e)
    all := all && ok
    let line := Lean.Json.mkObj [("session", Lean.Json.str dir), ("accepted", Lean.Json.bool ok), ("why", Lean.Json.str why)]
    IO.println s!"VERDICT {line.compress}"
    IO.eprintln s!"{dir}: {(← IO.monoMsNow) - t1} ms"
  return (if all then 0 else 1)

/-- `statement` with `verify`'s statement flags: the derived statement (digest, Σ, `Hello`, shape) as one JSON line. -/
def statementCmd (args : List String) : IO UInt32 := do
  let (some circuit, some comp, some pub) := (argOf args "--circuit", argOf args "--comp", argOf args "--public")
    | IO.eprintln "usage: flock-verify statement [--statement ID] --circuit F --comp F --public F [--coins seed|os]"; return 2
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuit.statement)
    | IO.eprintln s!"unknown statement; known: {Tags.all.map (·.statement)}"; return 2
  let coins := if argOf args "--coins" == some "os" then Coins.os else Coins.seed
  let (compB, circuitB, pubB) := (← IO.FS.readBinFile comp, ← IO.FS.readBinFile circuit, ← IO.FS.readBinFile pub)
  match (do Stmt.setup tags (Blake3Row.leaf (← Comp.load Blake3Row.spec compB)) circuitB pubB coins : V Stmt) with
  | .ok st =>
    let j := Lean.Json.mkObj [("statement", Lean.Json.str tags.statement), ("circuit_sha256", Lean.Json.str (hexOf st.c.sha)),
      ("statement_digest", Lean.Json.str (hexOf st.digest)), ("sigma", Lean.Json.str (hexOf st.sigma)),
      ("public_sha256", Lean.Json.str (hexOf st.pub.sha)), ("hello", Lean.Json.str st.hello),
      ("identity", Lean.Json.str ((canon tags.identity).toOption.getD "")),
      ("m", Lean.toJson st.m), ("regions", Lean.toJson st.regions.size), ("delta", Lean.toJson (st.da.size / 2))]
    IO.println j.compress
    return 0
  | .error e => IO.println s!"setup: {e}"; return 1

def main (args : List String) : IO UInt32 := do
  match args with
  | "statement" :: rest => statementCmd rest
  | "verify" :: rest => verifyCmd rest
  | ["comp-check", path] => compCheck path
  | ["field-test", path] => fieldTest path
  | ["hash-test", path] => hashTest path
  | _ => IO.eprintln "usage: flock-verify field-test FILE"; return 2
