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

/-- `--program P --partition Q --stratified K`: the stratified law the verifier derives from its own program and partition
object (`HmRow.stratifiedLaw`), and the population; none without `--stratified`. -/
def stratifiedInput (args : List String) : IO (Except String (Option (Draw.Law × Nat))) := do
  let some k := argOf args "--stratified" | return .ok none
  let some k := k.toNat? | return .error "--stratified K: K is a natural number"
  let program ← match ← programInput args with
    | .ok (some p) => pure p
    | .ok none => return .error "--stratified needs --program (the verifier derives the strata itself)"
    | .error e => return .error e
  let obj ← match ← partitionInput args with
    | .ok (some o) => pure o
    | .ok none => return .error "--stratified needs --partition"
    | .error e => return .error e
  return (HmRow.stratifiedLaw obj program k).map some

/-- `draw (--population N (--k K | --p NUM/DEN) | --program P --partition Q --stratified K) [--stream HEX | --stream-file F]`:
the unit draw (§7.3), from the operating system's randomness (a stratified draw: one stream per stratum), or from the
given bytes (a test; a stratified draw's streams are its consecutive segments). The stratified law's strata are the verifier's own,
derived from its program and partition object. Prints `DRAW {…}`, the object the verifier sends in the clear. -/
def drawCmd (args : List String) : IO UInt32 := do
  let (law, pop) ← match ← stratifiedInput args with
    | .error e => IO.eprintln s!"draw: {e}"; return 2
    | .ok (some lp) => pure lp
    | .ok none =>
      let some pop := (argOf args "--population").bind String.toNat?
        | IO.eprintln "draw: --population N, or --program P --partition Q --stratified K"; return 2
      match argOf args "--k", argOf args "--p" with
      | some k, none => match k.toNat? with
        | some k => if k ≤ pop then pure (Draw.Law.subset k, pop) else IO.eprintln "draw: k > population"; return 2
        | none => IO.eprintln "draw: --k K"; return 2
      | none, some p => match p.splitOn "/" with
        | [a, b] => match a.toNat?, b.toNat? with
          | some a, some b =>
            if 0 < b && a ≤ b then pure (Draw.Law.bernoulli a b, pop) else IO.eprintln "draw: p ∉ [0, 1]"; return 2
          | _, _ => IO.eprintln "draw: --p NUM/DEN"; return 2
        | _ => IO.eprintln "draw: --p NUM/DEN"; return 2
      | _, _ => IO.eprintln "draw: exactly one of --k, --p"; return 2
  let stream ← match argOf args "--stream", argOf args "--stream-file" with
    | none, none => pure none
    | some hex, none => match unhex hex with | .ok b => pure (some b) | .error e => IO.eprintln s!"draw: {e}"; return 2
    | none, some f => pure (some (← IO.FS.readBinFile f))
    | some _, some _ => IO.eprintln "draw: at most one of --stream, --stream-file"; return 2
  let d ← match stream with
    | none => Draw.drawOS law pop
    | some src =>
      match Draw.drawOn law pop src with
      | some d => pure d
      | none => IO.eprintln "draw: the stream ran out"; return 1
  IO.println s!"DRAW {d.toJson.compress}"
  return 0

/-- `draw-test FILE [--program P --partition Q --stratified K]`: U1–U3 (§7.3) on each case `{"record", "statement",
"instances"}` (a draw or null each), one `DRAWCHECK {"ok", "why"}` line per case. With the stratified law's flags, the
statement's draw is also held to the verifier's own law and population (`draw.check_draw`'s `law` and `population`). -/
def drawTest (path : String) (args : List String) : IO UInt32 := do
  let .ok (.arr cases) := Lean.Json.parse (← IO.FS.readFile path) | IO.eprintln "draw-test: not a JSON list"; return 2
  let own ← match ← stratifiedInput args with
    | .ok o => pure o
    | .error e => IO.eprintln s!"draw-test: {e}"; return 2
  for c in cases do
    let side := fun (k : String) => match c.getObjVal? k with
      | .ok .null => none
      | .ok v => some v
      | .error _ => none
    let n := (c.getObjValAs? Nat "instances").toOption.getD 0
    let r : V Unit := do
      Draw.check (side "record") (side "statement") n
      if let (some (law, pop), some s) := (own, side "statement") then
        let d ← Draw.UnitDraw.ofJson s
        if d.population != pop then throw s!"U2: the draw's population {d.population} is not the verifier's {pop}"
        if d.law != law then throw "U2: the draw's law is not the verifier's own (its program and query)"
    let (ok, why) := match r with
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

def main (args : List String) : IO UInt32 := do
  match args with
  | ["merkle-test", path] => merkleTest path
  | ["hm96-test", path] => hm96Test path
  | ["hm96-test", path, "sha512"] => hm96Test path Hm96.sha512
  | "unit-cut" :: paths => unitCutCmd paths
  | "archive-put" :: dir :: paths => archivePutCmd dir paths
  | "template-query" :: paths => templateQueryCmd paths
  | "draw" :: rest => drawCmd rest
  | "draw-test" :: path :: rest => drawTest path rest
  | "statement" :: rest => statementCmd rest
  | "verify" :: rest => verifyCmd rest
  | ["comp-check", path] => compCheck path
  | ["field-test", path] => fieldTest path
  | ["hash-test", path] => hashTest path
  | _ => IO.eprintln "usage: flock-verify field-test FILE"; return 2
