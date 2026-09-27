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
        | "sha512" => Sha512.hash m
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

/-- The pinned MUFU tables found under `--tables DIR` (`<name>.u32`); the circuit checks each one it needs. -/
def readTables (args : List String) : IO (List (String × ByteArray)) := do
  let some dir := argOf args "--tables" | return []
  let mut out := []
  for t in MUFU_TABLES do
    let p : System.FilePath := dir ++ s!"/{t.name}.u32"
    if ← p.pathExists then out := out ++ [(t.name, ← IO.FS.readBinFile p)]
  return out

/-- `ARCHIVE/sha512/<hex>`: the verifier's own copy of an object, by content. The file must hash to its key. -/
def archiveGet (archive hex : String) : IO (Except String ByteArray) := do
  if hex.length != 128 || !hex.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) then
    return .error s!"archive: {hex} is not a SHA-512 digest"
  let p : System.FilePath := archive ++ s!"/sha512/{hex}"
  if !(← p.pathExists) then return .error s!"archive: {hex} is not held"
  let b ← IO.FS.readBinFile p
  if hexOf (Sha512.hash b) != hex then return .error s!"archive: the object held under {hex} does not hash to it"
  return .ok b

/-- The statement's inputs: with `--archive DIR`, the circuit, public and compression files are the verifier's own objects
named by `--circuit`, `--public`, `--comp` SHA-512 digests, and the lookup tables are those `MUFU_TABLES` pins (§16.8);
without it, the files at those paths. -/
def statementInputs (args : List String) : IO (Except String (ByteArray × ByteArray × ByteArray × List (String × ByteArray))) := do
  let (some circuit, some pub) := (argOf args "--circuit", argOf args "--public")
    | return .error "--circuit and --public are required"
  let comp := argOf args "--comp"
  match argOf args "--archive" with
  | none =>
    let cm ← match comp with | some f => IO.FS.readBinFile f | none => pure ByteArray.empty
    return .ok (← IO.FS.readBinFile circuit, ← IO.FS.readBinFile pub, cm, ← readTables args)
  | some dir =>
    let get := fun hex => archiveGet dir hex
    let c ← match ← get circuit with | .ok b => pure b | .error e => return .error s!"circuit: {e}"
    let pb ← match ← get pub with | .ok b => pure b | .error e => return .error s!"public file: {e}"
    let cm ← match comp with
      | none => pure ByteArray.empty
      | some h => match ← get h with | .ok b => pure b | .error e => return .error s!"compression circuit: {e}"
    -- `Circuit.parse` hashes each table a lookup uses against its pinned SHA-512, the key it is held under here
    let mut tables := []
    for t in MUFU_TABLES do
      let p : System.FilePath := dir ++ s!"/sha512/{t.sha512}"
      if ← p.pathExists then tables := tables ++ [(t.name, ← IO.FS.readBinFile p)]
    return .ok (c, pb, cm, tables)

/-- The statement of `tags` from its inputs; an `hm96-sha512/row/v1` statement needs no compression circuit and, for a drawn
session, is the drawn units' statement over the registered population (§7.3). -/
def buildStmt (tags : Tags) (circuitB pubB compB : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw : Option Lean.Json := none) (partition : Option Lean.Json := none) (program : Option ByteArray := none) : V Stmt :=
  if tags.hm96Rows then Stmt.setupH tags circuitB pubB coins tables draw partition program
  else do Stmt.setup tags (Blake3Row.leaf (← Comp.load Blake3Row.spec compB)) circuitB pubB coins tables

/-- `--program F`: the verifier's own copy of the program's descriptor bytes (a path, or with `--archive` its SHA-512), from which
it derives a template query's units. -/
def programInput (args : List String) : IO (Except String (Option ByteArray)) := do
  let some f := argOf args "--program" | return .ok none
  match argOf args "--archive" with
  | none => try return .ok (some (← IO.FS.readBinFile f)) catch e => return .error s!"program: {e}"
  | some dir => return (← archiveGet dir f).map some

/-- `--partition F`: the verifier's own copy of the `verity/partition/v1` object the circuit binds (a path, or with `--archive`
its SHA-512). -/
def partitionInput (args : List String) : IO (Except String (Option Lean.Json)) := do
  let some f := argOf args "--partition" | return .ok none
  let bytes ← match argOf args "--archive" with
    | none => (try pure (Except.ok (← IO.FS.readBinFile f)) catch e => pure (Except.error s!"partition: {e}"))
    | some dir => archiveGet dir f
  match bytes with
  | .error e => return .error e
  | .ok b => match String.fromUTF8? b with
    | none => return .error "partition: not UTF-8"
    | some t => match Lean.Json.parse t with
      | .ok j => return .ok (some j)
      | .error e => return .error s!"partition: not JSON: {e}"

/-- A record's `unit_draw`, when it is a draw (not absent, not null). -/
def recordDraw (record : String) : Option Lean.Json :=
  match Lean.Json.parse record with
  | .ok j => match j.getObjVal? "unit_draw" with
    | .ok Lean.Json.null => none
    | .ok d => some d
    | .error _ => none
  | .error _ => none

/-- `archive-put DIR FILE...`: store each file under `DIR/sha512/<its SHA-512>`, and print the digests. -/
def archivePutCmd (dir : String) (paths : List String) : IO UInt32 := do
  IO.FS.createDirAll (dir ++ "/sha512")
  for path in paths do
    let b ← IO.FS.readBinFile path
    let hex := hexOf (Sha512.hash b)
    IO.FS.writeBinFile (dir ++ s!"/sha512/{hex}") b
    IO.println s!"{hex}  {path}"
  return 0

/-- `verify --statement ID --circuit F --comp F --public F [--coins seed|os] --session DIR...`: `ID` names the statement's
tags (`verity/flock-circuit` by default, or `verity/flock-netlist/v1`); `--comp` is the row leaf's pinned compression
circuit. Each DIR holds `session.json`, `<table>.rep0.bin` and `<table>.rep1.bin`. The statement is set up once; one
verdict line per session, in order. Exit 0 iff every session is accepted. -/
def verifyCmd (args : List String) : IO UInt32 := do
  let usage := "usage: flock-verify verify [--statement ID] [--archive DIR] --circuit F [--comp F] --public F [--partition F] [--program F] [--coins seed|os] --session DIR..."
  let (some _, some _) := (argOf args "--circuit", argOf args "--public") | IO.eprintln usage; return 2
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuit.name)
    | IO.eprintln s!"unknown statement; known: {Tags.all.map (·.name)}"; return 2
  if !tags.hm96Rows && (argOf args "--comp").isNone then IO.eprintln s!"{tags.name}: --comp is required"; return 2
  let dirs := (args.dropWhile (· != "--session")).drop 1
  let coins := if argOf args "--coins" == some "os" then Coins.os else Coins.seed
  let t0 ← IO.monoMsNow
  let (circuitB, pubB, compB, tables) ← match ← statementInputs args with
    | .ok x => pure x
    | .error e => IO.eprintln s!"statement inputs: {e}"; return 2
  let partition ← match ← partitionInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let program ← match ← programInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let setupOf := fun (draw : Option Lean.Json) => match buildStmt tags circuitB pubB compB coins tables draw partition program with
    | .ok st => Except.ok (Setup.ofCircuit st, st.partition)
    | .error e => Except.error s!"setup: {e}"
  let setup := setupOf none
  IO.eprintln s!"setup: {(← IO.monoMsNow) - t0} ms"
  let mut drawn : Array (String × Except String (Setup × Lean.Json)) := #[]
  let mut all := true
  for dir in dirs do
    let t1 ← IO.monoMsNow
    let files ← try
        let record ← IO.FS.readFile (dir ++ "/session.json")
        let proofs ← #[0, 1].mapM fun r => IO.FS.readBinFile s!"{dir}/{tags.table}.rep{r}.bin"
        pure (Except.ok (record, proofs))
      catch e => pure (Except.error s!"session files: {e}")
    -- a drawn session is verified against the drawn units' statement (Σ covers the draw)
    let mut st := setup
    if tags.hm96Rows then
      if let .ok (record, _) := files then
        if let some d := recordDraw record then
          let key := (canon d).toOption.getD ""
          match drawn.find? (·.1 == key) with
          | some (_, s) => st := s
          | none =>
            st := setupOf (some d)
            drawn := drawn.push (key, st)
    let verdict := do let (record, proofs) ← files; verify (← st).1 record proofs
    let (ok, why) := match verdict with | .ok () => (true, "") | .error e => (false, e)
    all := all && ok
    -- the partition finding: the rule, the query, the digest, and whether the units were derived or taken as stated
    let part := match st with | .ok (_, p) => p | .error _ => Lean.Json.null
    let line := Lean.Json.mkObj ([("session", Lean.Json.str dir), ("accepted", Lean.Json.bool ok), ("why", Lean.Json.str why)] ++
      (if part.isNull then [] else [("partition", part)]))
    IO.println s!"VERDICT {line.compress}"
    IO.eprintln s!"{dir}: {(← IO.monoMsNow) - t1} ms"
  return (if all then 0 else 1)

/-- `statement` with `verify`'s statement flags: the derived statement (digest, Σ, `Hello`, shape) as one JSON line. -/
def statementCmd (args : List String) : IO UInt32 := do
  let (some _, some _) := (argOf args "--circuit", argOf args "--public")
    | IO.eprintln "usage: flock-verify statement [--statement ID] [--archive DIR] --circuit F [--comp F] --public F [--coins seed|os]"; return 2
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuit.name)
    | IO.eprintln s!"unknown statement; known: {Tags.all.map (·.name)}"; return 2
  if !tags.hm96Rows && (argOf args "--comp").isNone then IO.eprintln s!"{tags.name}: --comp is required"; return 2
  let coins := if argOf args "--coins" == some "os" then Coins.os else Coins.seed
  let (circuitB, pubB, compB, tables) ← match ← statementInputs args with
    | .ok x => pure x
    | .error e => IO.eprintln s!"statement inputs: {e}"; return 2
  let partition ← match ← partitionInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let program ← match ← programInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  match buildStmt tags circuitB pubB compB coins tables none partition program with
  | .ok st =>
    let dk := if tags.hm96Rows then "sha512" else "sha256"
    let j := Lean.Json.mkObj [("statement", Lean.Json.str tags.statement), (s!"circuit_{dk}", Lean.Json.str (hexOf st.c.sha)),
      ("statement_digest", Lean.Json.str (hexOf st.digest)), ("sigma", Lean.Json.str (hexOf st.sigma)),
      (s!"public_{dk}", Lean.Json.str (hexOf st.pub.sha)), ("hello", Lean.Json.str st.hello),
      ("identity", Lean.Json.str ((canon tags.identity).toOption.getD "")),
      ("m", Lean.toJson st.m), ("regions", Lean.toJson st.regions.size), ("delta", Lean.toJson (st.da.size / 2)),
      ("class_sha512", Lean.Json.str (hexOf st.c.classSha)),
      ("program_sha512", (st.c.metaJ.getObjVal? "program_sha512").toOption.getD Lean.Json.null),
      ("partition", (st.c.metaJ.getObjVal? "partition").toOption.getD Lean.Json.null), ("partition_check", st.partition)]
    IO.println j.compress
    return 0
  | .error e => IO.println s!"setup: {e}"; return 1

/-- `merkle-test FILE`: lines `scheme c d lanes positions rows siblings cap expect` (lists comma-separated, hex; `rows` as
32-hex F128 words joined by `.` within a row); `expect` 1 = opens, 0 = rejected. -/
def merkleTest (path : String) : IO UInt32 := do
  let lines ← IO.FS.lines path
  let mut bad := 0
  for l in lines do
    match l.splitOn " " with
    | [sid, c, d, lanes, pos, rows, sibs, cap, want] =>
      let some ms := MerkleScheme.find? sid | bad := bad + 1; continue
      let nat := fun (x : String) => x.toNat!
      let hexes := fun (x : String) => ((x.splitOn ",").map bytesOfHex).toArray
      let row := fun (x : String) => ((x.splitOn ".").filterMap parseHex128).toArray
      let got := merkleCheck ms (hexes cap) (nat c) (nat d) ((pos.splitOn ",").map nat).toArray
        ((rows.splitOn ",").map row).toArray (hexes sibs) #[] (nat lanes)
      let ok := match got with | .ok () => "1" | .error _ => "0"
      if ok != want then bad := bad + 1
    | _ => bad := bad + 1
  IO.println s!"merkle-test: {lines.size} cases, {bad} failures"
  return (if bad == 0 then 0 else 1)

/-- `hm96-test VECTORS [sha512]`: PR #88's `hm96/vectors.json` against `Hm96.sha256` (PR #93's `vectors_sha512.json` against
`Hm96.sha512`): the pinned key, the leaf prefix, every mask and every leaf. -/
def hm96Test (path : String) (h : Hm96 := Hm96.sha256) : IO UInt32 := do
  let some v := (Lean.Json.parse (← IO.FS.readFile path)).toOption | IO.println "hm96-test: not JSON"; return 1
  let str := fun (j : Lean.Json) (k : String) => (j.getObjValAs? String k).toOption.getD ""
  let key := h.keyFromLabel
  let other := bytesOfHex (str v "other_key")
  let mut bad := 0
  let mut n := 0
  if hexOf key != str v "default_key" then bad := bad + 1
  if hexOf (h.leafPrefix key) != str v "leaf_prefix_default_key" then bad := bad + 1
  for m in ((v.getObjValAs? (Array Lean.Json) "masks").toOption.getD #[]) do
    n := n + 1
    let k := if str m "key" == "default" then key else other
    if hexOf (h.mask k (bytesOfHex (str m "salt"))) != str m "mask" then bad := bad + 1
    if hexOf (h.saltDigest (bytesOfHex (str m "salt"))) != str m "salt_digest" then bad := bad + 1
  for x in ((v.getObjValAs? (Array Lean.Json) "leaves").toOption.getD #[]) do
    n := n + 1
    let bc := h.commitString key (bytesOfHex (str x "inner_digest")) (bytesOfHex (str x "salt"))
    if hexOf bc != str x "b" ++ str x "c" || hexOf (h.treeLeaf key bc) != str x "leaf" then bad := bad + 1
  IO.println s!"hm96-test: {n} cases, {bad} failures"
  return (if bad == 0 then 0 else 1)

/-- `unit-cut FILE...`: the partition invariant (`Flock.Partition.validate`) on each cut, one `CUT` line per file. -/
def unitCutCmd (paths : List String) : IO UInt32 := do
  let mut all := true
  for path in paths do
    let line ← match (Lean.Json.parse (← IO.FS.readFile path)) with
      | .error e => pure (Lean.Json.mkObj [("file", Lean.Json.str path), ("ok", Lean.Json.bool false),
          ("error", Lean.Json.str s!"not JSON: {e}")])
      | .ok j => match Partition.Cut.ofJson j with
        | .error e => pure (Lean.Json.mkObj [("file", Lean.Json.str path), ("ok", Lean.Json.bool false),
            ("error", Lean.Json.str e)])
        | .ok c =>
          let cs := Partition.validate c
          pure (Lean.Json.mkObj [("file", Lean.Json.str path), ("ok", Lean.Json.bool cs.isEmpty),
            ("codes", Lean.Json.arr (cs.toArray.map Lean.Json.str))])
    all := all && (line.getObjValAs? Bool "ok").toOption.getD false
    IO.println s!"CUT {line.compress}"
  return (if all then 0 else 1)

/-- `draw --population N (--k K | --p NUM/DEN) [--stream HEX]`: the unit draw (§7.3), from the operating system's randomness,
or from the given bytes (a test). Prints `DRAW {…}`, the object the verifier sends in the clear. -/
def drawCmd (args : List String) : IO UInt32 := do
  let some pop := (argOf args "--population").bind String.toNat? | IO.eprintln "draw: --population N"; return 2
  let law ← match argOf args "--k", argOf args "--p" with
    | some k, none => match k.toNat? with
      | some k => if k ≤ pop then pure (Draw.Law.subset k) else IO.eprintln "draw: k > population"; return 2
      | none => IO.eprintln "draw: --k K"; return 2
    | none, some p => match p.splitOn "/" with
      | [a, b] => match a.toNat?, b.toNat? with
        | some a, some b => if 0 < b && a ≤ b then pure (Draw.Law.bernoulli a b) else IO.eprintln "draw: p ∉ [0, 1]"; return 2
        | _, _ => IO.eprintln "draw: --p NUM/DEN"; return 2
      | _ => IO.eprintln "draw: --p NUM/DEN"; return 2
    | _, _ => IO.eprintln "draw: exactly one of --k, --p"; return 2
  let d ← match argOf args "--stream" with
    | none => Draw.drawOS law pop
    | some hex =>
      let src ← match unhex hex with | .ok b => pure b | .error e => IO.eprintln s!"draw: {e}"; return 2
      let r := match law with
        | .subset k => Draw.subset k pop src
        | .bernoulli a b => Draw.bernoulli a b pop src
      match r with
      | some units => pure { law, population := pop, units : Draw.UnitDraw }
      | none => IO.eprintln "draw: the stream ran out"; return 1
  IO.println s!"DRAW {d.toJson.compress}"
  return 0

/-- `draw-test FILE`: U1–U3 (§7.3) on each case `{"record", "statement", "instances"}` (a draw or null each), one
`DRAWCHECK {"ok", "why"}` line per case. -/
def drawTest (path : String) : IO UInt32 := do
  let .ok (.arr cases) := Lean.Json.parse (← IO.FS.readFile path) | IO.eprintln "draw-test: not a JSON list"; return 2
  for c in cases do
    let side := fun (k : String) => match c.getObjVal? k with
      | .ok .null => none
      | .ok v => some v
      | .error _ => none
    let n := (c.getObjValAs? Nat "instances").toOption.getD 0
    let (ok, why) := match Draw.check (side "record") (side "statement") n with
      | .ok () => (true, "")
      | .error e => (false, e)
    IO.println s!"DRAWCHECK {(Lean.Json.mkObj [("ok", Lean.Json.bool ok), ("why", Lean.Json.str why)]).compress}"
  return 0

/-- `template-query FILE...`: a template-instance query on a program (`Flock.TemplateQuery`) for each case `{"descriptor", "query"}`
(the descriptor's bytes as a string): the program's SHA-512, the population, the ranges and every unit's root node, member and
gates, or the refusal. One `TEMPLATEQUERY` line per file. -/
def templateQueryCmd (paths : List String) : IO UInt32 := do
  for path in paths do
    let text ← try IO.FS.readFile path catch _ => pure ""
    let res : V Lean.Json := do
      let j ← match Lean.Json.parse text with | .ok j => pure j | .error e => throw s!"not JSON: {e}"
      let q := (j.getObjVal? "query").toOption.getD Lean.Json.null
      let name := (q.getObjValAs? String "name").toOption.getD ""
      let version := (q.getObjValAs? Nat "version").toOption.getD 0
      let some ts := TemplateQuery.templatesOf name version ((q.getObjVal? "params").toOption.getD Lean.Json.null)
        | throw s!"partition: query-unknown: {name} v{version}"
      let ts ← ts
      let d ← TemplateQuery.Desc.parse ((j.getObjValAs? String "descriptor").toOption.getD "").toUTF8
      let rs ← TemplateQuery.evaluate d ts
      let pop := TemplateQuery.population rs
      let units := (Array.range pop).map fun i => match TemplateQuery.locate rs i, TemplateQuery.gates d rs i with
        | some (k, m, _), some (lo, hi) => Lean.toJson #[i, k, m, lo, hi]
        | _, _ => Lean.Json.null
      return Lean.Json.mkObj [("program_sha512", Lean.Json.str (hexOf d.sha512)), ("population", Lean.toJson pop),
        ("ranges", Lean.Json.arr (rs.map fun r => Lean.Json.arr #[Lean.toJson r.node, Lean.Json.str r.template, Lean.toJson r.first, Lean.toJson r.count])),
        ("units", Lean.Json.arr units)]
    let line := match res with
      | .ok r => (r.setObjVal! "file" (Lean.Json.str path)).setObjVal! "ok" (Lean.Json.bool true)
      | .error e => Lean.Json.mkObj [("file", Lean.Json.str path), ("ok", Lean.Json.bool false), ("error", Lean.Json.str e)]
    IO.println s!"TEMPLATEQUERY {line.compress}"
  return 0

/-- `cut-check FILE...`: `check_cut` (the invariant on one Call's cut against its graph, and the width rule) on each case, one
`CUTCHECK` line per file. -/
def cutCheckCmd (paths : List String) : IO UInt32 := do
  let mut all := true
  for path in paths do
    let res := match Lean.Json.parse (← IO.FS.readFile path) with
      | .error e => Except.error s!"not JSON: {e}"
      | .ok j => Partition.checkCutOfJson j
    let line := match res with
      | .error e => Lean.Json.mkObj [("file", Lean.Json.str path), ("ok", Lean.Json.bool false), ("error", Lean.Json.str e)]
      | .ok (g, owner, committed, limits) =>
        let (cs, wide) := Partition.checkCut g owner committed limits
        Lean.Json.mkObj [("file", Lean.Json.str path), ("ok", Lean.Json.bool cs.isEmpty),
          ("codes", Lean.Json.arr (cs.toArray.map Lean.Json.str)),
          ("wide", Lean.Json.arr (wide.map fun (x : Nat × Nat × Nat) => Lean.Json.arr #[Lean.toJson x.1, Lean.toJson x.2.1, Lean.toJson x.2.2]))]
    all := all && (line.getObjValAs? Bool "ok").toOption.getD false
    IO.println s!"CUTCHECK {line.compress}"
  return (if all then 0 else 1)

/-- `evaluate FILE...`: `Q_word` v1 on each case `{"X", "calls": [{"call", "def", "graph"?}]}` (`Flock.Qword`): per Call its
owners, units, committed set (from its graph) and, for a flat cut, its kinds and heads; the population and every unit's
`locate`. One `EVALUATE` line per file. -/
def evaluateCmd (paths : List String) : IO UInt32 := do
  let mut all := true
  for path in paths do
    let text ← try IO.FS.readFile path catch _ => pure ""
    let res : V Lean.Json := do
      let j ← match Lean.Json.parse text with
        | .ok j => pure j
        | .error e => throw s!"not JSON: {e}"
      let X ← match j.getObjValAs? Nat "X" with | .ok x => pure x | .error _ => throw "evaluate: X"
      let cs ← match j.getObjVal? "calls" with
        | .ok (.arr a) => pure a
        | _ => throw "evaluate: calls"
      let mut calls := #[]
      let mut outs := #[]
      for c in cs do
        let k := (c.getObjValAs? Nat "call").toOption.getD 0
        let d ← Qword.defOfJson ((c.getObjVal? "def").toOption.getD Lean.Json.null)
        let dc := Qword.evaluate X d
        calls := calls.push (k, dc)
        let own := dc.owner
        let mut o := [("call", Lean.toJson k), ("owner", Lean.toJson own), ("units", Lean.toJson dc.units)]
        if let .ok gj := c.getObjVal? "graph" then
          let (g, _, _, _) ← Partition.checkCutOfJson (Lean.Json.mkObj [("graph", gj), ("owner", Lean.Json.arr #[])])
          o := o ++ [("committed", Lean.toJson (if own.size == g.n then Partition.derivedCommitted g own else #[]))]
        if let .flat g := d then
          let (_, kinds, heads) := Qword.cutWord g X
          o := o ++ [("kinds", Lean.toJson kinds), ("head", Lean.toJson heads)]
        outs := outs.push (Lean.Json.mkObj o)
      let P : Qword.Units := ⟨calls⟩
      let loc := (Array.range P.population).map fun i => match P.locate i with
        | some (p, k, u, path) => Lean.Json.arr #[Lean.toJson i, Lean.toJson #[p, k, u],
            Lean.Json.arr (path.toArray.map fun (a, b, c) => Lean.toJson #[a, b, c])]
        | none => Lean.Json.null
      return Lean.Json.mkObj [("calls", Lean.Json.arr outs), ("population", Lean.toJson P.population), ("locate", Lean.Json.arr loc)]
    let line := match res with
      | .ok r => (r.setObjVal! "file" (Lean.Json.str path)).setObjVal! "ok" (Lean.Json.bool true)
      | .error e => Lean.Json.mkObj [("file", Lean.Json.str path), ("ok", Lean.Json.bool false), ("error", Lean.Json.str e)]
    all := all && (line.getObjValAs? Bool "ok").toOption.getD false
    IO.println s!"EVALUATE {line.compress}"
  return (if all then 0 else 1)

/-- A Call's graph as core's `CallGraph` exports it (`test_format_spec._graph` and `partition_vectors.json`'s fields). -/
def graphJson (gr : Extract.Graph) : Lean.Json :=
  let g := gr.g
  let pairs := fun (xs : Array (Nat × Nat)) => Lean.Json.arr (xs.map fun (a, b) => Lean.toJson #[a, b])
  Lean.Json.mkObj [("total", Lean.toJson g.total), ("structure", Lean.toJson g.free), ("n", Lean.toJson g.n),
    ("keep", Lean.toJson gr.keep), ("width", Lean.toJson g.width), ("prim", Lean.toJson gr.prim),
    ("external", Lean.toJson (gr.external.map fun b => if b then 1 else 0)), ("input_ids", Lean.toJson gr.inputIds),
    ("src", Lean.toJson (g.reads.map (·.1))), ("dst", Lean.toJson (g.reads.map (·.2))), ("input_reads", pairs g.inputReads),
    ("outputs", Lean.toJson g.outputs), ("returned_inputs", Lean.toJson g.returnedInputs), ("recomputed", pairs g.recomputed)]

private def sha512Of (j : Lean.Json) : V String := do return hexOf (Sha512.hash (← canon j).toUTF8)

/-- `qword-program DESCRIPTOR [X] [--digests]`: `Q_word` v1 on the program (`Flock.Program`, `Flock.Extract`, `Flock.Qword`):
the program's SHA-512, the population and per Call its units. In full: per Call its owners, derived committed set and graph,
and every unit's `locate`. With `--digests`: per Definition a Call names, its units and the SHA-512 of its owners, and of
its graph for a flat cut. One `QWORDPROGRAM` line. -/
def qwordProgramCmd (args : List String) : IO UInt32 := do
  let path := args.headD ""
  let X := ((args.drop 1).find? (·.all Char.isDigit)).bind String.toNat? |>.getD 16
  let digests := args.contains "--digests"
  let bytes ← try IO.FS.readBinFile path catch _ => pure ByteArray.empty
  let res : V Lean.Json := do
    let P ← Program.decode bytes
    let calls ← Extract.evaluate P X
    let units : Qword.Units := ⟨calls.map fun (k, _, c) => (k, c)⟩
    let base := [("program_sha512", Lean.Json.str (hexOf P.sha512)), ("population", Lean.toJson units.population),
      ("calls", Lean.Json.arr (calls.map fun (k, d, c) => Lean.Json.arr #[Lean.toJson k, Lean.Json.str (P.defs[d]!).key,
        Lean.toJson c.units]))]
    if digests then
      let mut defs : Array (String × Lean.Json) := #[]
      for (_, d, c) in calls do
        let key := (P.defs[d]!).key
        if defs.any (·.1 == key) then continue
        let mut o := [("units", Lean.toJson c.units), ("owner_sha512", Lean.Json.str (← sha512Of (Lean.toJson c.owner)))]
        if let .flat .. := c then
          if !(P.defs[d]!).prim then o := o ++ [("graph_sha512", Lean.Json.str (← sha512Of (graphJson (← Extract.graph P d))))]
        defs := defs.push (key, Lean.Json.mkObj o)
      return Lean.Json.mkObj (base ++ [("definitions", Lean.Json.mkObj defs.toList)])
    let mut detail := #[]
    for (k, d, c) in calls do
      let own := c.owner
      let mut o := [("call", Lean.toJson k), ("definition", Lean.Json.str (P.defs[d]!).key), ("units", Lean.toJson c.units),
        ("owner", Lean.toJson own)]
      if !(P.defs[d]!).prim then
        let gr ← Extract.graph P d
        o := o ++ [("graph", graphJson gr),
          ("committed", Lean.toJson (if own.size == gr.g.n then Partition.derivedCommitted gr.g own else #[]))]
      detail := detail.push (Lean.Json.mkObj o)
    let loc := (Array.range units.population).map fun i => match units.locate i with
      | some (p, k, u, path) => Lean.Json.arr #[Lean.toJson i, Lean.toJson #[p, k, u],
          Lean.Json.arr (path.toArray.map fun (a, b, c) => Lean.toJson #[a, b, c])]
      | none => Lean.Json.null
    return Lean.Json.mkObj (base ++ [("detail", Lean.Json.arr detail), ("locate", Lean.Json.arr loc)])
  let line := match res with
    | .ok r => r.setObjVal! "ok" (Lean.Json.bool true)
    | .error e => Lean.Json.mkObj [("ok", Lean.Json.bool false), ("error", Lean.Json.str e)]
  IO.println s!"QWORDPROGRAM {line.compress}"
  return (if (line.getObjValAs? Bool "ok").toOption.getD false then 0 else 1)

def main (args : List String) : IO UInt32 := do
  match args with
  | ["merkle-test", path] => merkleTest path
  | ["hm96-test", path] => hm96Test path
  | ["hm96-test", path, "sha512"] => hm96Test path Hm96.sha512
  | "unit-cut" :: paths => unitCutCmd paths
  | "cut-check" :: paths => cutCheckCmd paths
  | "evaluate" :: paths => evaluateCmd paths
  | "qword-program" :: rest => qwordProgramCmd rest
  | "archive-put" :: dir :: paths => archivePutCmd dir paths
  | "template-query" :: paths => templateQueryCmd paths
  | "draw" :: rest => drawCmd rest
  | ["draw-test", path] => drawTest path
  | "statement" :: rest => statementCmd rest
  | "verify" :: rest => verifyCmd rest
  | ["comp-check", path] => compCheck path
  | ["field-test", path] => fieldTest path
  | ["hash-test", path] => hashTest path
  | _ => IO.eprintln "usage: flock-verify field-test FILE"; return 2
