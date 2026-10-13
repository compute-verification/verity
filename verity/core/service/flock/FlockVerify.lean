import Flock
import Flock.CircuitType
import Flock.Layout

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

/-- A recursive session's inner statement's registered reads (§16.16, a plain-leaves statement): with the verifier's own
registered reads, its description read from the developer's registration (`Registered.checkDescription`); without, a
statement that names registered reads is refused, as `Registered.check` refuses one. -/
def checkDescription (st : Stmt) (registered : Option (Array Registered.Port)) : V Unit :=
  match registered with
  | some own => Registered.checkDescription st.c st.pub own
  | none => Registered.check st.c st.pub none false

/-- A statement's identifier (§16.10): the public header's `identifier`, a JSON object C-Flock does not read, must be the
verifier's own `--identifier F` (`own`, its RFC 8785 form) in RFC 8785 form: `canon`'s, on the domain where it is RFC 8785
(`Program.domainError`). The public digest binds the header, so Σ binds the identifier; a header with one and no `F`, or
`F` and a header with none, refuses. -/
def checkIdentifier (st : Stmt) (own : Option String) : V Unit :=
  match st.pub.header.getObjVal? "identifier", own with
  | .error _, none => pure ()
  | .error _, some _ => throw "identifier: the verifier holds an identifier (--identifier F), and the public header names none"
  | .ok j, own => do
    let .obj _ := j | throw "identifier: the public header's identifier is not a JSON object"
    if let some e := Program.domainError j "identifier" then
      throw s!"identifier: the public header's identifier is outside canonical JSON's RFC 8785 domain: {e}"
    let some own := own | throw "identifier: the public header names an identifier; pass the verifier's own (--identifier F)"
    if (← canon j) != own then throw "identifier: the public header's identifier is not the verifier's own (--identifier F)"

/-- The statement of `tags` from its inputs; an `hm96-sha512/row/v1` statement needs no compression circuit and, for a drawn
session, is the drawn units' statement over the registered population (§7.3). -/
def buildStmt (tags : Tags) (circuitB pubB compB : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw : Option Lean.Json := none) (partition : Option Lean.Json := none) (program : Option ByteArray := none)
    (registered : Option (Array Registered.Port) := none) (publicInputs : Option ByteArray := none)
    (identifier : Option String := none) : V Stmt :=
  if publicInputs.isSome && !tags.hiddenOutputs then
    throw s!"--public-inputs: a {tags.name} statement has no public input ports (only a statement with hidden outputs has them)"
  else if tags.hm96Rows then do
    let st ← if tags.hiddenOutputs then Stmt.setupHidden tags circuitB pubB coins tables draw partition program publicInputs
      else Stmt.setupH tags circuitB pubB coins tables draw partition program
    if tags.plainLeaves then checkDescription st registered else checkRegistered st registered
    checkIdentifier st identifier
    return st
  else do
    if registered.isSome then throw s!"{tags.name}: registered reads are an hm96-sha512 row statement's"
    Stmt.setup tags (Blake3Row.leaf (← Comp.load Blake3Row.spec compB)) circuitB pubB coins tables

/-- `buildStmt` for a session of `count` tables (`Stmt.setupTables`): the session and each table's statement. `work` is the
verifier's own `K` and work table, which a work draw's law is held to, `stratified` its own `K` for a stratified draw,
`closure` its closure map, and `publicInputs` its own recomputation of the public input words. `registered` is the
verifier's own registered reads, which every table's statement is held to (`Registered.check`), and `identifier` its own
identifier (`checkIdentifier`). -/
def buildSession (tags : Tags) (circuitB pubB compB : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (count : Nat) (draw : Option Lean.Json := none) (partition : Option Lean.Json := none) (program : Option ByteArray := none)
    (work : Option (Nat × Array (String × Nat × Nat)) := none) (stratified : Option Nat := none)
    (closure : Option (Std.HashMap Nat (Array Nat)) := none) (registered : Option (Array Registered.Port) := none)
    (publicInputs : Option ByteArray := none) (identifier : Option String := none) : V (SessionSpec × Array Stmt) :=
  if tags.hm96Rows then do
    let (spec, sts) ← Stmt.setupTables tags circuitB pubB coins count tables draw partition program work stratified closure
      publicInputs
    for st in sts do
      checkRegistered st registered
      checkIdentifier st identifier
    Registered.checkFull registered draw
    return (spec, sts)
  else if count > 1 then throw s!"{tags.name}: one table per session"
  else do
    let st ← buildStmt tags circuitB pubB compB coins tables draw partition program registered publicInputs identifier
    return (st.spec, #[st])

/-- `--program F`: the verifier's own copy of the program's descriptor bytes (a path, or with `--archive` its SHA-512), from which
it derives a template query's units. -/
def programInput (args : List String) : IO (Except String (Option ByteArray)) := do
  let some f := argOf args "--program" | return .ok none
  match argOf args "--archive" with
  | none => try return .ok (some (← IO.FS.readBinFile f)) catch e => return .error s!"program: {e}"
  | some dir => return (← archiveGet dir f).map some

/-- `--public-inputs F`: the verifier's own recomputation of a statement's public input words (`HmIn.checkOwn`'s file), a
path, or with `--archive` its SHA-512. -/
def publicInputsInput (args : List String) : IO (Except String (Option ByteArray)) := do
  let some f := argOf args "--public-inputs" | return .ok none
  match argOf args "--archive" with
  | none => try return .ok (some (← IO.FS.readBinFile f)) catch e => return .error s!"public inputs: {e}"
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

/-- `--work-table F`: the verifier's own work table (`Draw.workTableOfJson`: template descriptor id ↦ the PoUW work of one
of its units, and its floor), a path, or with `--archive` its SHA-512. -/
def workTableInput (args : List String) : IO (Except String (Option (Array (String × Nat × Nat)))) := do
  let some f := argOf args "--work-table" | return .ok none
  let bytes ← match argOf args "--archive" with
    | none => (try pure (Except.ok (← IO.FS.readBinFile f)) catch e => pure (Except.error s!"work table: {e}"))
    | some dir => archiveGet dir f
  match bytes with
  | .error e => return .error e
  | .ok b => match String.fromUTF8? b with
    | none => return .error "work table: not UTF-8"
    | some t => match Lean.Json.parse t with
      | .ok j => return (Draw.workTableOfJson j).map some
      | .error e => return .error s!"work table: not JSON: {e}"

/-- `--closure F`: the verifier's own closure map (`Draw.closureMapOfJson`: a unit's index ↦ the units its credit depends
on), a path, or with `--archive` its SHA-512. -/
def closureInput (args : List String) : IO (Except String (Option (Std.HashMap Nat (Array Nat)))) := do
  let some f := argOf args "--closure" | return .ok none
  let bytes ← match argOf args "--archive" with
    | none => (try pure (Except.ok (← IO.FS.readBinFile f)) catch e => pure (Except.error s!"closure map: {e}"))
    | some dir => archiveGet dir f
  match bytes with
  | .error e => return .error e
  | .ok b => match String.fromUTF8? b with
    | none => return .error "closure map: not UTF-8"
    | some t => match Lean.Json.parse t with
      | .ok j => return (Draw.closureMapOfJson j).map some
      | .error e => return .error s!"closure map: not JSON: {e}"

/-- `--registered F`: the verifier's own registered reads (`Registered.ofJson`: per registered port, the value, its
registered root, leaf count, row width and domain, and the position each unit reads), a path, or with `--archive` its
SHA-512. With `flag` `--rec-rows`, the verifier's record of V*'s shared rows (`RecVStar.checkRows`), a file of the same
form. -/
def registeredInput (args : List String) (flag : String := "--registered") (ctx : String := "registered reads") :
    IO (Except String (Option (Array Registered.Port))) := do
  let some f := argOf args flag | return .ok none
  let bytes ← match argOf args "--archive" with
    | none => (try pure (Except.ok (← IO.FS.readBinFile f)) catch e => pure (Except.error s!"{ctx}: {e}"))
    | some dir => archiveGet dir f
  match bytes with
  | .error e => return .error e
  | .ok b => match String.fromUTF8? b with
    | none => return .error s!"{ctx}: not UTF-8"
    | some t => match Lean.Json.parse t with
      | .ok j => return ((Registered.ofJson j).mapError fun e => if flag == "--registered" then e else s!"{ctx}: {e}").map some
      | .error e => return .error s!"{ctx}: not JSON: {e}"

/-- `--identifier F`: the verifier's own identifier, a JSON object (a path, or with `--archive` its SHA-512), and its RFC 8785
form (`checkIdentifier`). -/
def identifierInput (args : List String) : IO (Except String (Option (Lean.Json × String))) := do
  let some f := argOf args "--identifier" | return .ok none
  let bytes ← match argOf args "--archive" with
    | none => (try pure (Except.ok (← IO.FS.readBinFile f)) catch e => pure (Except.error s!"identifier: {e}"))
    | some dir => archiveGet dir f
  match bytes with
  | .error e => return .error e
  | .ok b => match String.fromUTF8? b with
    | none => return .error "identifier: not UTF-8"
    | some t => match Lean.Json.parse t with
      | .ok j@(.obj _) =>
        if let some e := Program.domainError j "identifier" then
          return .error s!"identifier: --identifier F is outside canonical JSON's RFC 8785 domain: {e}"
        return (canon j).map fun s => some (j, s)
      | .ok _ => return .error "identifier: not a JSON object"
      | .error e => return .error s!"identifier: not JSON: {e}"

/-- `--rec-inner F`: the verifier's own shape of V*'s inner session (`RecVStar.Inner`, the line `rec-inner` printed), a path,
or with `--archive` its SHA-512. -/
def recInnerInput (args : List String) : IO (Except String (Option RecVStar.Inner)) := do
  let some f := argOf args "--rec-inner" | return .ok none
  let bytes ← match argOf args "--archive" with
    | none => (try pure (Except.ok (← IO.FS.readBinFile f)) catch e => pure (Except.error s!"rec-inner: {e}"))
    | some dir => archiveGet dir f
  match bytes with
  | .error e => return .error e
  | .ok b => match String.fromUTF8? b with
    | none => return .error "rec-inner: not UTF-8"
    | some t => match Lean.Json.parse t with
      | .ok j => return (RecVStar.Inner.ofJson j).map some
      | .error e => return .error s!"rec-inner: not JSON: {e}"

/-- `--rec-part J`: which of V*'s statements the verifier holds this session for, `J` indexing `RecVStar.statements` at the
verifier's own inner shape from 0 (the list `rec-statements` prints): the levels first, `L<level>` as `J = level`, then the
algebra's parts, `alg/p<j>` as `J = levels + j`, then candidate 3's: RecSparse (`sparse`), R's levels (`stacked/L<level>`)
and R's opening's parts (`stacked/open/p<j>`). -/
def recPartInput (args : List String) : Except String (Option Nat) :=
  match argOf args "--rec-part" with
  | none => .ok none
  | some j => match j.toNat? with
    | some j => .ok (some j)
    | none => .error "--rec-part J: J is a natural number"

/-- `--rec-bindings F`: the verifier's own bindings of V*'s statement (`RecVStar.ownBindingsOfJson`: `vstar`'s `BINDINGS`
object, which the auditor writes with `python -m verity_flock.rec_bindings PARTITION PROGRAM` from the verifier's own
partition object and program), a path, or with `--archive` its SHA-512. -/
def recBindingsInput (args : List String) : IO (Except String (Option VStar.Bindings)) := do
  let some f := argOf args "--rec-bindings" | return .ok none
  let bytes ← match argOf args "--archive" with
    | none => (try pure (Except.ok (← IO.FS.readBinFile f)) catch e => pure (Except.error s!"rec-bindings: {e}"))
    | some dir => archiveGet dir f
  match bytes with
  | .error e => return .error e
  | .ok b => match String.fromUTF8? b with
    | none => return .error "rec-bindings: not UTF-8"
    | some t => match Lean.Json.parse t with
      | .ok j => return (RecVStar.ownBindingsOfJson j).map some
      | .error e => return .error s!"rec-bindings: not JSON: {e}"

/-- The inner circuit `f`'s META, from its header alone, up to its `META` line. -/
def innerMeta (f : String) : IO (Except String Lean.Json) := do
  try
    let h ← IO.FS.Handle.mk f .read
    let mut line ← h.getLine
    let mut n := 0
    while !line.isEmpty && !line.startsWith "META " && n < 8 do
      line ← h.getLine
      n := n + 1
    if !line.startsWith "META " then return .error "rec inner circuit: no META line"
    match Lean.Json.parse (line.drop 5).toString with
    | .error e => return .error s!"rec inner circuit: META is not JSON: {e}"
    | .ok m => return .ok m
  catch e => return .error s!"rec inner circuit: {e}"

/-- META's `registration` (`inner_registered.REGISTRATION_KEY`) of the inner circuit (`--rec-inner-circuit`): the digest of
the circuit registration it binds (one-stage `registration.digest`), or none, when META names none or no inner circuit is
given; and META's `stacked` (`RecVStar.stackedShapeOf`), the shape staging committed R at. Only the circuit's header is
read, up to its `META` line. -/
def innerMetaRegistration (args : List String) :
    IO (Except String (Option String × Option RecVStar.StackedShape)) := do
  let some f := argOf args "--rec-inner-circuit" | return .ok (none, none)
  let mj ← innerMeta f
  return do
    let m ← mj
    let sh ← RecVStar.stackedShapeOf m
    match m.getObjVal? "registration" with
    | .error _ => pure (none, sh)
    | .ok (.str d) => pure (some d, sh)
    | .ok _ => throw "rec inner circuit: META's registration is not a digest"

/-- `--rec-inner-transcript F`: the coin server's record of V*'s inner session (`rec/inner-transcript/v1`), with the inner
session's record `record` (`--rec-inner-session`) that its coins must match: each rep's `c_k`, and with candidate 3's
streams every stream's `c_k` and tops (`RecVStar.innerCommits`). `--rec-inner-registration F`: the circuit registration
(`verity/registration/v2`, written at staging, `inner_registered.REGISTRATION`) that the inner circuit's META binds
(`--rec-inner-circuit`, `innerMetaRegistration`), whose entry `stacked` is R's registered top (`RecVStar.registration`),
which a session with R's stream needs. -/
def recTranscriptInput (args : List String) (record : Option String) :
    IO (Except String (Option RecVStar.InnerCommits)) := do
  let reg := argOf args "--rec-inner-registration"
  let some f := argOf args "--rec-inner-transcript"
    | return if reg.isSome then .error "--rec-inner-registration goes with --rec-inner-transcript" else .ok none
  let some record := record | return .error "--rec-inner-transcript goes with the inner session (--rec-inner-session and its three files)"
  if reg.isSome && (argOf args "--rec-inner-circuit").isNone then
    return .error "--rec-inner-registration goes with --rec-inner-circuit, whose META binds it"
  let bound ← innerMetaRegistration args
  let t ← try pure (Except.ok (← IO.FS.readFile f)) catch e => pure (Except.error s!"rec inner transcript: {e}")
  let r ← match reg with
    | none => pure (Except.ok none)
    | some g => try pure (Except.ok (some (← IO.FS.readFile g))) catch e => pure (Except.error s!"rec inner registration: {e}")
  return do
    let rj ← match ← r with
      | none => pure none
      | some s => match Lean.Json.parse s with
        | .ok j => pure (some j)
        | .error e => throw s!"rec inner registration: not JSON: {e}"
    let (bound, shape) ← bound
    match Lean.Json.parse (← t) with
    | .ok j => some <$> RecVStar.innerCommits j record rj bound shape
    | .error e => throw s!"rec inner transcript: not JSON: {e}"

/-- `--sparse NU,ELL,S_LOG,T_LOG`: the bucket's RecSparse shape, the verifier's own word for candidate 3's sparse phase. -/
def sparseInput (args : List String) : Except String (Option RecSparse.Shape) :=
  match argOf args "--sparse" with
  | none => .ok none
  | some v => match (v.splitOn ",").map String.toNat? with
    | [some nu, some ell, some sLog, some tLog] => .ok (some { nu, ell, sLog, tLog })
    | _ => .error s!"--sparse {v}: NU,ELL,S_LOG,T_LOG, four natural numbers"

/-- `rec-inner --circuit F --public F [--partition F] [--program F] [--hide] [--sparse NU,ELL,S_LOG,T_LOG]`: the shape of
V*'s inner session (`RecVStar.innerOf`) from the inner statement, as one `verity/rec-inner/v1` line, for `verify
--rec-inner`; `--hide`: the verifier's own word that the inner session's circuits are private (`RecVStar.Inner.holo`);
`--sparse`: its word that the session is candidate 3's, with the bucket's RecSparse shape (`RecVStar.Inner.sparse`). R's
shape (`RecVStar.Inner.stacked`) is the inner circuit's META's (`RecVStar.stackedShapeOf`), the shape staging committed R
at; a META naming one with no `--sparse` is refused, since R opens the sparse phase's claims. -/
def recInnerCmd (args : List String) : IO UInt32 := do
  let usage := "usage: flock-verify rec-inner --circuit F --public F [--partition F] [--program F] [--hide] [--sparse NU,ELL,S_LOG,T_LOG]"
  let (some circuit, some pub) := (argOf args "--circuit", argOf args "--public")
    | IO.eprintln usage; return 2
  let partition ← match ← partitionInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let program ← match ← programInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let sparse ← match sparseInput args with
    | .ok s => pure s
    | .error e => IO.eprintln e; return 2
  let mj ← innerMeta circuit
  let stacked ← match (do RecVStar.stackedShapeOf (← mj)) with
    | .ok r => pure r
    | .error e => IO.eprintln s!"rec-inner: {e}"; return 1
  if let (none, some r) := (sparse, stacked) then
    IO.eprintln (s!"rec-inner: the inner circuit's META names R's shape (m {r.m}, claims {r.claims}), and the verifier names " ++
      "no RecSparse shape (--sparse NU,ELL,S_LOG,T_LOG): R opens the sparse phase's claims, so V*'s list would miss RecSparse")
    return 1
  match RecVStar.innerOf (← IO.FS.readBinFile circuit) (← IO.FS.readBinFile pub) partition program (args.contains "--hide") with
  | .ok i => IO.println { i with sparse, stacked }.toJson.compress; return 0
  | .error e => IO.eprintln s!"rec-inner: {e}"; return 1

/-- `rec-coins --rec-inner-circuit F --rec-inner-public F --rec-inner-partition F --rec-inner-session DIR`: what check 5
reads of the inner session (`RecCoins.Inner.toJson`), set up once from the inner statement, as one `verity/rec-coins/v1`
line, for `verify --rec-inner-coins F --rec-inner-session DIR` in place of the inner statement's three files. -/
def recCoinsCmd (args : List String) : IO UInt32 := do
  match ← RecCoins.filesOnly args with
  | .ok (some f) => match RecCoins.Inner.load "the inner session" f with
    | .ok inn => IO.println (inn.toJson f.record).compress; return 0
    | .error e => IO.eprintln e; return 1
  | .ok none =>
    IO.eprintln "usage: flock-verify rec-coins --rec-inner-circuit F --rec-inner-public F --rec-inner-partition F --rec-inner-session DIR"
    return 2
  | .error e => IO.eprintln e; return 2

/-- `rec-transcript --rec-inner-transcript F --rec-inner-session DIR [--rec-inner F]`: what `verify` holds of the inner
session's transcript (`RecVStar.innerCommits`), as one `RECTRANSCRIPT` line: each rep's message rounds, and each stream's
message rounds and its tops' node counts by level; with `--rec-inner`, that the verifier's inner shape covers it
(`RecVStar.checkCovers`, as `verify` checks it); exit 1 when the transcript is refused. -/
def recTranscriptOfSession (args : List String) : IO (Except String (Option RecVStar.InnerCommits)) := do
  let some dir := argOf args "--rec-inner-session"
    | return if (argOf args "--rec-inner-transcript").isSome then .error "--rec-inner-transcript goes with --rec-inner-session" else .ok none
  let record ← try pure (← IO.FS.readFile (dir ++ "/session.json"))
    catch e => return .error s!"rec inner session: {e}"
  recTranscriptInput args (some record)

/-- Candidate 3's check 5 inputs: the verifier's RecSparse shape (`rec-inner --sparse`) and R's, its own inner shape's or
else the shape the inner transcript held R's stream to, and the coin server's record of the inner session. -/
def recC3 (inner : Option RecVStar.Inner) (commits : Option RecVStar.InnerCommits) : Option RecCoins.C3 := do
  let i ← inner
  let cs ← commits
  let sp ← i.sparse
  pure { sparse := sp, stacked := i.stacked.or cs.stacked, commits := cs }

def recTranscriptCmd (args : List String) : IO UInt32 := do
  let some _ := argOf args "--rec-inner-session"
    | IO.eprintln "usage: flock-verify rec-transcript --rec-inner-transcript F --rec-inner-session DIR [--rec-inner-circuit F --rec-inner-registration F] [--rec-inner F]"; return 2
  let inner ← match ← recInnerInput args with
    | .ok i => pure i
    | .error e => IO.eprintln e; return 2
  match ← recTranscriptOfSession args with
  | .ok (some ic) =>
    if let some i := inner then
      if let .error e := RecVStar.checkCovers i ic then IO.eprintln e; return 1
    let n := fun (xs : Array (Nat × ByteArray)) => Lean.toJson xs.size
    IO.println s!"RECTRANSCRIPT {(Lean.Json.mkObj [("reps", .arr (ic.reps.map n)),
      ("streams", .arr (ic.streams.map fun s => Lean.Json.mkObj [("stream", .str s.name), ("commits", n s.commits),
        ("tops", .arr (s.tops.map fun t => Lean.toJson t.size))]))]).compress}"
    return 0
  | .ok none => IO.eprintln "usage: flock-verify rec-transcript --rec-inner-transcript F --rec-inner-session DIR [--rec-inner-circuit F --rec-inner-registration F] [--rec-inner F]"; return 2
  | .error e => IO.eprintln e; return 1

/-- `rec-statements --rec-inner F [--unit NAME | --circuit F --rec-bindings F] [--instances N] [--rec-part J]
[--registered F]`: V*'s statements at the inner shape (`RecVStar.statements`), each unit name and instance count, as one
`RECVSTAR` line; with `--unit` and `--instances`, `RecVStar.check` on them as well (as statement `J` with `--rec-part`),
with `--circuit`, `--instances`, `--rec-part` and `--rec-bindings`, `RecVStar.checkCircuit` on that circuit file as
statement `J` under those bindings, and then with `--registered`, `RecVStar.checkOwn` on those reads, with the statement's
own message-round ports (`RecVStar.ownPorts`). -/
def recStatementsCmd (args : List String) : IO UInt32 := do
  let usage := "usage: flock-verify rec-statements --rec-inner F [--unit NAME | --circuit F --rec-bindings F] [--instances N] [--rec-part J] [--registered F]"
  let i ← match ← recInnerInput args with
    | .ok (some i) => pure i
    | .ok none => IO.eprintln usage; return 2
    | .error e => IO.eprintln e; return 2
  let registered ← match ← registeredInput args with
    | .ok r => pure r
    | .error e => IO.eprintln e; return 2
  let part ← match recPartInput args with
    | .ok j => pure j
    | .error e => IO.eprintln e; return 2
  let bindings ← match ← recBindingsInput args with
    | .ok b => pure b
    | .error e => IO.eprintln e; return 2
  let circuitB ← (argOf args "--circuit").mapM IO.FS.readBinFile
  let res : V Lean.Json := do
    let ss ← RecVStar.statements i
    let base := [("statements", Lean.Json.arr (ss.map fun s => Lean.Json.arr #[Lean.Json.str s.name, Lean.toJson s.instances]))]
    let checked := fun (s : VStar.Stmt) (n : Nat) => do
      if let some r := registered then RecVStar.checkOwn s r
      let own := (RecVStar.ownPorts s).map fun (nm, v, rd) => Lean.Json.arr #[Lean.Json.str nm, Lean.Json.str v, Lean.toJson rd]
      pure (Lean.Json.mkObj (base ++ [("checked", Lean.Json.arr #[Lean.Json.str s.name, Lean.toJson n]),
        ("own", Lean.Json.arr own.toArray)]))
    match argOf args "--unit", circuitB, (argOf args "--instances").bind String.toNat? with
    | some u, none, some n => do checked (← RecVStar.check i u n part) n
    | none, some b, some n => do
      let c ← HmRow.parse Tags.circuit b []
      let some j := part | throw s!"{c.unit.name}: pass which of V*'s statements this circuit is for (--rec-part J)"
      let some f := bindings | throw s!"{c.unit.name}: pass the verifier's own bindings (--rec-bindings F)"
      checked (← RecVStar.checkCircuit i j c n b f) n
    | none, none, _ => pure (Lean.Json.mkObj base)
    | _, _, _ => throw usage
  let line := match res with
    | .ok j => j.setObjVal! "ok" (Lean.Json.bool true)
    | .error e => Lean.Json.mkObj [("ok", Lean.Json.bool false), ("error", Lean.Json.str e)]
  IO.println s!"RECVSTAR {line.compress}"
  return (if res.toBool then 0 else 1)

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

/-- `coin-tree-check FILE`: R7 and the record's `coin_tree` for the keyed coin tree (`Flock.CoinTree`), on `{"hello": the
verifier's Hello object (its coins.nonce empty), "sessions": [{"hello": the prover's Hello, "coin_tree": ...}, ...]}`, in
order, each key fresh. One `COINTREE` line per session: the nonce and the canonical `coins` it checked, or the refusal. -/
def coinTreeCmd (path : String) : IO UInt32 := do
  let j ← match Lean.Json.parse (← IO.FS.readFile path) with
    | .ok j => pure j
    | .error e => IO.eprintln s!"coin-tree-check: {e}"; return 2
  let own := (j.getObjVal? "hello").toOption.getD Lean.Json.null
  let some spec := CoinTree.ofHello ((canon own).toOption.getD "")
    | IO.eprintln "coin-tree-check: the Hello names no keyed coin tree"; return 2
  let mut keys : Array String := #[]
  let mut all := true
  for s in (j.getObjValAs? (Array Lean.Json) "sessions").toOption.getD #[] do
    let res : V (String × String) := do
      let nonce ← CoinTree.checkHello own ((s.getObjValAs? String "hello").toOption.getD "")
      let key ← CoinTree.checkRecord spec nonce ((s.getObjVal? "coin_tree").toOption.getD Lean.Json.null)
      CoinTree.checkFresh keys key
      return (nonce, key)
    let line := match res with
      | .ok (nonce, _) => Lean.Json.mkObj [("ok", Lean.Json.bool true), ("nonce", Lean.Json.str nonce),
          ("coins", Lean.Json.str ((canon (spec.coins nonce)).toOption.getD ""))]
      | .error e => Lean.Json.mkObj [("ok", Lean.Json.bool false), ("error", Lean.Json.str e)]
    if let .ok (_, k) := res then keys := keys.push k
    all := all && res.toBool
    IO.println s!"COINTREE {line.compress}"
  return (if all then 0 else 1)

/-- `verify --statement ID --circuit F --comp F --public F [--coins seed|os] [--session-tables J] --session DIR...`: `ID`
names the statement's tags (`verity/flock-circuit` by default, or `verity/flock-netlist/v1`); `--comp` is the row leaf's
pinned compression circuit; `J` (default 1) is the tables of every session (§16.12). A session with a work draw needs the
verifier's own `--work K`, `--work-table`, `--program` and `--partition`, one with a stratified draw its own
`--stratified K`, `--program` and `--partition`, a closure draw its own `--closure F`, a statement that reads registered
values its own `--registered F` (`Registered.check`), a circuit with public input ports its own `--public-inputs F`, and
a public file whose header names an identifier its own `--identifier F` (§16.10), one of V*'s statements its own inner
shape `--rec-inner F` (under which `--statement` must be `verity/flock-circuit`, V*'s, and no lookup table is held), which of V*'s statements the session is `--rec-part J` (`J` indexing the list `rec-statements`
prints from 0: the levels, then the algebra's parts) and its own bindings `--rec-bindings F` (from
`verity_flock.rec_bindings`, which META's must equal; `RecVStar.checkStmt`), and a V* statement (a `rec-open/…` or `rec-residuals/…` unit) the inner
statement and session its coefficients are checked against (`RecCoins.check`; or, with the session, `rec-coins`'s record
of what that check reads of them, `--rec-inner-coins`), and a statement reading a `rec-c<k>` the
coin server's record of the inner session whose `c_k` its rows must be (`--rec-inner-transcript`,
`RecVStar.checkPubRows`), and one of V*'s parts the verifier's record of V*'s shared rows, which its row ports must be
(`--rec-rows`, `RecVStar.checkRows`). Each DIR holds `session.json`
and `<table>.rep<r>.bin` for each table and rep. The statement is set up once; one verdict line per session, in order.
A session whose draw is by a law no soundness claim draws by is refused at setup (`ProvedScope.drawLaw`: a bernoulli or
work draw, whatever flags it holds, or a subset draw without `--zk`). Every session of a statement whose form the
soundness proof does not cover (`ProvedScope.check`, §16.15) is refused, and so is a session without a draw, both after
every check has passed (so the reason is the refusal only when nothing else fails; the form's first). Exit 0 iff every
session is accepted. `--zk`: the sessions are `flock-circuit --zk`'s (`Flock.Zk`), on the keyed coin tree;
none is on `os` or `seed` coins (the live build reads no `FC_COINS` under `--zk`), so `--zk` takes no `--coins`. -/
def verifyCmd (args : List String) : IO UInt32 := do
  let usage := "usage: flock-verify verify [--statement ID] [--archive DIR] --circuit F [--comp F] --public F [--public-inputs F] [--partition F] [--program F] [--work K --work-table F] [--stratified K] [--closure F] [--registered F] [--identifier F] [--rec-inner F --rec-part J --rec-bindings F [--rec-rows F]] [--rec-inner-circuit F --rec-inner-public F --rec-inner-partition F | --rec-inner-coins F] [--rec-inner-session DIR [--rec-inner-transcript F [--rec-inner-registration F]]] [--coins seed|os | --zk] [--session-tables J] --session DIR..."
  let (some _, some _) := (argOf args "--circuit", argOf args "--public") | IO.eprintln usage; return 2
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuit.name)
    | IO.eprintln s!"unknown statement; known: {Tags.all.map (·.name)}"; return 2
  let stmtTags := tags
  if !tags.hm96Rows && (argOf args "--comp").isNone then IO.eprintln s!"{tags.name}: --comp is required"; return 2
  let some count := ((argOf args "--session-tables").getD "1").toNat? | IO.eprintln usage; return 2
  let dirs := (args.dropWhile (· != "--session")).drop 1
  let zk := args.contains "--zk"
  if zk && (argOf args "--coins").isSome then
    IO.eprintln "verify --zk: a --zk session's coins are its keyed coin tree, so --coins (a plain session's) does not apply"; return 2
  if tags.plainLeaves then
    IO.eprintln s!"verify: refused: {tags.name}, a statement with plain SHA-512 leaves (flock-leaf/sha512-unsalted): a form the soundness proof does not cover yet"
    return 2
  let coins := if zk then Coins.tree (Zk.coinSpec (tags.tableNames count))
    else if argOf args "--coins" == some "os" then Coins.os else Coins.seed
  let tags := tags.withCoins coins
  let t0 ← IO.monoMsNow
  let (circuitB, pubB, compB, tables) ← match ← statementInputs args with
    | .ok x => pure x
    | .error e => IO.eprintln s!"statement inputs: {e}"; return 2
  let tags := if zk then Zk.tags tags (Zk.maskLogOf circuitB) count else tags
  let partition ← match ← partitionInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let program ← match ← programInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let table ← match ← workTableInput args with
    | .ok w => pure w
    | .error e => IO.eprintln e; return 2
  let ownK ← match argOf args "--work" with
    | none => pure none
    | some k => match k.toNat? with
      | some k => pure (some k)
      | none => IO.eprintln "verify: --work K: K is a natural number"; return 2
  let closure ← match ← closureInput args with
    | .ok c => pure c
    | .error e => IO.eprintln e; return 2
  let registered ← match ← registeredInput args with
    | .ok r => pure r
    | .error e => IO.eprintln e; return 2
  let publicInputs ← match ← publicInputsInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let identifier : Option (Lean.Json × String) ← match ← identifierInput args with
    | .ok i => pure i
    | .error e => IO.eprintln e; return 2
  let recInner ← match ← recInnerInput args with
    | .ok i => pure i
    | .error e => IO.eprintln e; return 2
  let recPart ← match recPartInput args with
    | .ok j => pure j
    | .error e => IO.eprintln e; return 2
  let recBindings ← match ← recBindingsInput args with
    | .ok b => pure b
    | .error e => IO.eprintln e; return 2
  let recCoins ← match ← RecCoins.filesInput args with
    | .ok f => pure f
    | .error e => IO.eprintln e; return 2
  let recCommits ← match ← recTranscriptInput args (recCoins.map (·.record)) with
    | .ok c => pure c
    | .error e => IO.eprintln e; return 2
  let recRows ← match ← registeredInput args "--rec-rows" "rec rows" with
    | .ok r => pure r
    | .error e => IO.eprintln e; return 2
  let work := match ownK, table with
    | some k, some t => some (k, t)
    | _, _ => none
  let stratified ← match argOf args "--stratified" with
    | none => pure none
    | some k => match k.toNat? with
      | some k => pure (some k)
      | none => IO.eprintln "verify: --stratified K: K is a natural number"; return 2
  -- a drawn statement that sets up is still refused when no soundness claim draws by its draw's law
  -- (`ProvedScope.drawLaw`); a statement whose form the soundness proof does not cover (`ProvedScope.check`) is refused in
  -- each session's verdict
  let scope := ProvedScope.check tags zk circuitB pubB tables count publicInputs.isSome registered.isSome
  let setupOf := fun (draw : Option Lean.Json) =>
    match (do
        let r ← buildSession tags circuitB pubB compB coins tables count draw partition program work stratified closure
          registered publicInputs (identifier.map Prod.snd)
        r.2.forM (RecVStar.checkStmt recInner stmtTags tables · circuitB partition program registered recPart recBindings recRows)
        if let (some i, some cs) := (recInner, recCommits) then RecVStar.checkCovers i cs
        if zk then r.2.forM Zk.shapeOk
        r.2.forM (RecCoins.check · registered (RecCoins.Source.atC3 recCoins recInner recCommits) (recC3 recInner recCommits))
        r.2.forM (RecVStar.checkPubRows · registered recCommits (recInner.bind (·.sparse)))
        pure r) with
    | .ok (spec, sts) => (if draw.isSome then ProvedScope.drawLaw zk draw else .ok ()).map fun () =>
        (spec, sts.map Setup.ofCircuit, (sts[0]?.map (·.partition)).getD Lean.Json.null)
    | .error e => Except.error s!"setup: {e}"
  IO.eprintln s!"inputs: {(← IO.monoMsNow) - t0} ms"
  -- each statement is set up once, when a session first needs it: the population's, and each draw's
  let mut setups : Array (Option String × Except String (SessionSpec × Array Setup × Lean.Json)) := #[]
  let mut coinKeys : Array String := #[]
  let mut all := true
  for dir in dirs do
    let t1 ← IO.monoMsNow
    let record ← try pure (Except.ok (← IO.FS.readFile (dir ++ "/session.json")))
      catch e => pure (Except.error s!"session files: {e}")
    -- a drawn session is verified against the drawn units' statement (Σ covers the draw); several tables split the draw
    let draw := if tags.hm96Rows then record.toOption.bind recordDraw else none
    let key := draw.map fun d => (canon d).toOption.getD ""
    let mut st : Except String (SessionSpec × Array Setup × Lean.Json) := .error "setup: none"
    match setups.find? (·.1 == key) with
    | some (_, s) => st := s
    | none =>
      st := setupOf draw
      setups := setups.push (key, st)
      IO.eprintln s!"setup{if draw.isSome then " (drawn)" else ""}: {(← IO.monoMsNow) - t1} ms"
    let files ← match record, st with
      | .ok r, .ok (spec, _, _) =>
        try pure (Except.ok (r, ← spec.repStreams.mapM fun (_, _, w) => IO.FS.readBinFile s!"{dir}/{w.replace "/" "."}.bin"))
        catch e => pure (Except.error s!"session files: {e}")
      | .error e, _ => pure (.error e)
      | _, .error e => pure (.error e)
    -- a keyed coin tree's session key is fresh: refuse a key an earlier session of this run used
    let coinKey : Option String := match st, record with
      | .ok (spec, _, _), .ok r =>
        if spec.coinTree.isNone then none
        else (Lean.Json.parse r).toOption.bind fun j =>
          (j.getObjVal? "coin_tree").toOption.bind fun ct => (ct.getObjValAs? String "key").toOption
      | _, _ => none
    let verdict := do
      let (record, proofs) ← files
      if let some k := coinKey then CoinTree.checkFresh coinKeys k
      let (spec, tabs, _) ← st
      if zk then Zk.verify spec tabs record proofs else verify spec tabs record proofs
      -- last, so a session that fails a check keeps that check's reason: the form's refusal, then a draw-less session's
      scope
      if draw.isNone then ProvedScope.drawLaw zk draw
    if let some k := coinKey then coinKeys := coinKeys.push k
    let (ok, why) := match verdict with | .ok () => (true, "") | .error e => (false, e)
    all := all && ok
    -- the partition finding: the rule, the query, the digest, and whether the units were derived or taken as stated
    let part := match st with | .ok (_, _, p) => p | .error _ => Lean.Json.null
    let line := Lean.Json.mkObj ([("session", Lean.Json.str dir), ("accepted", Lean.Json.bool ok), ("why", Lean.Json.str why)] ++
      (if part.isNull then [] else [("partition", part)]) ++
      (match registered with | some r => [("registered", Registered.finding r)] | none => []) ++
      (match identifier with | some (j, _) => [("identifier", j)] | none => []) ++
      -- the verifier's own inner shape's word, never the prover's META
      (match recInner with | some i => [("private_circuit", Lean.Json.bool i.holo)] | none => []))
    IO.println s!"VERDICT {line.compress}"
    IO.eprintln s!"{dir}: {(← IO.monoMsNow) - t1} ms"
  return (if all then 0 else 1)

/-- `statement` with `verify`'s statement flags: the derived statement (digest, Σ, `Hello`, shape) as one JSON line. With
`--rec-inner F`, the statement must be one of V*'s at that inner shape, statement `--rec-part J` of them under the
verifier's own `--rec-bindings F` (`RecVStar.checkStmt`). With `--rec-inner-transcript F --rec-inner-session DIR`, its
rows at what the inner transcript fixes are the `b ‖ c` the verifier holds (`RecVStar.checkPubRows`, as `verify` sets up).
Its verifier-registered values are checked against the verifier's coins (`RecCoins.check`) when it holds what that check
reads: the inner statement and session (`--rec-inner-public` and its three, or `--rec-inner-coins`), or for R's levels and
opening the inner transcript and the verifier's inner shape alone. -/
def statementCmd (args : List String) : IO UInt32 := do
  let (some _, some _) := (argOf args "--circuit", argOf args "--public")
    | IO.eprintln "usage: flock-verify statement [--statement ID] [--archive DIR] --circuit F [--comp F] --public F [--public-inputs F] [--partition F] [--program F] [--registered F] [--identifier F] [--rec-inner F --rec-part J --rec-bindings F [--rec-rows F]] [--rec-inner-transcript F --rec-inner-session DIR [--rec-inner-circuit F --rec-inner-registration F] [--rec-inner-public F --rec-inner-partition F | --rec-inner-coins F]] [--coins seed|os]"; return 2
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuit.name)
    | IO.eprintln s!"unknown statement; known: {Tags.all.map (·.name)}"; return 2
  let stmtTags := tags
  if !tags.hm96Rows && (argOf args "--comp").isNone then IO.eprintln s!"{tags.name}: --comp is required"; return 2
  let coins := if argOf args "--coins" == some "os" then Coins.os else Coins.seed
  let tags := tags.withCoins coins
  let (circuitB, pubB, compB, tables) ← match ← statementInputs args with
    | .ok x => pure x
    | .error e => IO.eprintln s!"statement inputs: {e}"; return 2
  let partition ← match ← partitionInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let program ← match ← programInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let registered ← match ← registeredInput args with
    | .ok r => pure r
    | .error e => IO.eprintln e; return 2
  let publicInputs ← match ← publicInputsInput args with
    | .ok p => pure p
    | .error e => IO.eprintln e; return 2
  let identifier : Option (Lean.Json × String) ← match ← identifierInput args with
    | .ok i => pure i
    | .error e => IO.eprintln e; return 2
  let recInner ← match ← recInnerInput args with
    | .ok i => pure i
    | .error e => IO.eprintln e; return 2
  let recPart ← match recPartInput args with
    | .ok j => pure j
    | .error e => IO.eprintln e; return 2
  let recBindings ← match ← recBindingsInput args with
    | .ok b => pure b
    | .error e => IO.eprintln e; return 2
  let recRows ← match ← registeredInput args "--rec-rows" "rec rows" with
    | .ok r => pure r
    | .error e => IO.eprintln e; return 2
  let recCommits ← match ← recTranscriptOfSession args with
    | .ok c => pure c
    | .error e => IO.eprintln e; return 2
  let recCoins ← if (argOf args "--rec-inner-public").isSome || (argOf args "--rec-inner-coins").isSome then
      match ← RecCoins.filesInput args with
      | .ok f => pure f
      | .error e => IO.eprintln e; return 2
    else pure none
  let c3 := recC3 recInner recCommits
  match (do
      let st ← buildStmt tags circuitB pubB compB coins tables none partition program registered publicInputs
        (identifier.map Prod.snd)
      if recInner.isSome then RecVStar.checkStmt recInner stmtTags tables st circuitB partition program registered recPart recBindings recRows
      if let (some i, some cs) := (recInner, recCommits) then RecVStar.checkCovers i cs
      if recCommits.isSome then RecVStar.checkPubRows st registered recCommits (recInner.bind (·.sparse))
      let u := st.c.unit.name
      if recCoins.isSome || (c3.isSome && ((RecCoins.stackedParams u).isSome || (RecOpening.params u).isSome)) then
        RecCoins.check st registered (RecCoins.Source.atC3 recCoins recInner recCommits) c3
      pure st) with
  | .ok st =>
    let dk := if tags.hm96Rows then "sha512" else "sha256"
    let j := Lean.Json.mkObj [("statement", Lean.Json.str tags.statement), (s!"circuit_{dk}", Lean.Json.str (hexOf st.c.sha)),
      ("statement_digest", Lean.Json.str (hexOf st.digest)), ("sigma", Lean.Json.str (hexOf st.sigma)),
      (s!"public_{dk}", Lean.Json.str (hexOf st.pub.sha)), ("hello", Lean.Json.str st.hello),
      ("identity", Lean.Json.str ((canon tags.identity).toOption.getD "")),
      ("m", Lean.toJson st.m), ("regions", Lean.toJson st.regions.size), ("delta", Lean.toJson (st.da.size / 2)),
      ("class_sha512", Lean.Json.str (hexOf st.c.classSha)),
      ("program_sha512", (st.c.metaJ.getObjVal? "program_sha512").toOption.getD Lean.Json.null),
      ("partition", (st.c.metaJ.getObjVal? "partition").toOption.getD Lean.Json.null), ("partition_check", st.partition),
      ("registered", (registered.map Registered.finding).getD Lean.Json.null),
      ("identifier", (identifier.map Prod.fst).getD Lean.Json.null)]
    IO.println j.compress
    return 0
  | .error e => IO.println s!"setup: {e}"; return 1

/-- `region-words [--statement ID] --circuit F --public F [--delta Q,COL]...`: the region-word check
(`HmRow.checkRegionWords`) on an `hm96-sha512` statement, each `--delta` entry `(Q, COL)` added to both Δs. One
`REGIONWORDS` line: the partial regions' shared columns, and the refusal if any. -/
def regionWordsCmd (args : List String) : IO UInt32 := do
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuit.name)
    | IO.eprintln s!"unknown statement; known: {Tags.all.map (·.name)}"; return 2
  let (circuitB, pubB, _, tables) ← match ← statementInputs args with
    | .ok x => pure x
    | .error e => IO.eprintln s!"region-words inputs: {e}"; return 2
  let extra := (args.zip args.tail).filterMap fun (k, v) => if k != "--delta" then none else
    match (v.splitOn ",").map String.toNat? with
    | [some q, some col] => some (q, col)
    | _ => none
  let res : V Nat := do
    let hidden := tags.hiddenOutputs
    let c ← if hidden then HmOut.parse tags circuitB tables else HmRow.parse tags circuitB tables
    let pi ← if hidden then HmIn.parse c else pure {}
    let pub ← if hidden then HmIn.loadPublic tags c pi pubB else HmRow.loadPublic tags c pubB
    let pin ← HmRow.pin c
    let (da, db) := if hidden then HmIn.delta c pi pin else HmRow.delta c pin
    let regions ← if hidden then do pure ((← HmOut.regions c pub) ++ (← HmIn.regions c pi pub)) else HmRow.regions c pub
    HmRow.checkRegionWords c (da ++ extra.toArray) (db ++ extra.toArray) regions
    return (HmRow.sharedColumns regions).size
  let line := match res with
    | .ok n => Lean.Json.mkObj [("ok", Lean.Json.bool true), ("shared", Lean.toJson n)]
    | .error e => Lean.Json.mkObj [("ok", Lean.Json.bool false), ("error", Lean.Json.str e)]
  IO.println s!"REGIONWORDS {line.compress}"
  return (if (line.getObjValAs? Bool "ok").toOption.getD false then 0 else 1)

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

/-- `net-check FILE...`: each file as one `flock-ir-unit/v2` expanded circuit through `Net.parse` (its port groups, input
rows, constant row and row order); one `NETCHECK {"file", "ok", "error"}` line per file. -/
def netCheckCmd (paths : List String) : IO UInt32 := do
  let mut all := true
  for path in paths do
    let text ← try IO.FS.readFile path catch _ => pure ""
    let lines := (text.splitOn "\n").toArray
    let lines := if lines.back? == some "" then lines.pop else lines
    let (ok, err) := match Net.parse lines with
      | .ok _ => (true, "")
      | .error e => (false, e)
    all := all && ok
    IO.println s!"NETCHECK {(Lean.Json.mkObj [("file", Lean.Json.str path), ("ok", Lean.Json.bool ok), ("error", Lean.Json.str err)]).compress}"
  return (if all then 0 else 1)

/-- `unit-cut FILE...`: the partition invariant (`Flock.Partition.validate`) on each cut, one `CUT` line per file; a cut with
`"recompute": "report"` (`Q_word` v2's rule) also gives its `recomputed_across` count. -/
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
          pure (Lean.Json.mkObj ([("file", Lean.Json.str path), ("ok", Lean.Json.bool cs.isEmpty),
            ("codes", Lean.Json.arr (cs.toArray.map Lean.Json.str))] ++
            if c.reportRecompute then [("recomputed_across", Lean.toJson c.recomputedAcross.size)] else []))
    all := all && (line.getObjValAs? Bool "ok").toOption.getD false
    IO.println s!"CUT {line.compress}"
  return (if all then 0 else 1)

/-- `typed-expand FILE [--statement ID] [--out F]`: a typed file (item 1e; `ID` by default `verity/flock-circuit/types`, or
`verity/flock-circuit/types@210d32e1`) parsed as `verify` parses it, its unit's rows derived (`HmOut.parse`, or
`HmRow.parseTyped` for a statement with public outputs). One `TYPED` line: the circuits, the unit's rows, and the file's
SHA-512 and class. With `--out`, the file with the derived rows as its `CIRCUIT unit` section (`Typed.expand`). -/
def typedExpandCmd (args : List String) : IO UInt32 := do
  let some path := args.head? | IO.eprintln "typed-expand FILE [--statement ID] [--out F]"; return 2
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuitTypes.name) |>.filter (·.typed)
    | IO.eprintln "typed-expand: --statement names no typed statement"; return 2
  let bytes ← IO.FS.readBinFile path
  let line ← match (if tags.hiddenOutputs then HmOut.parse tags bytes else HmRow.parseTyped tags bytes) with
    | .error e => pure (Lean.Json.mkObj [("ok", Lean.Json.bool false), ("error", Lean.Json.str e)])
    | .ok c =>
      if let some out := argOf args "--out" then
        if let .ok ex := Typed.expand tags bytes then IO.FS.writeBinFile out ex
      let u := c.nets[c.unitNet]!.2
      pure (Lean.Json.mkObj [("ok", Lean.Json.bool true), ("name", Lean.Json.str c.name),
        ("nets", Lean.toJson (c.nets.map (·.1))), ("unit_useful", Lean.toJson u.useful),
        ("sha512", Lean.Json.str (hexOf c.sha)), ("class", Lean.Json.str (hexOf c.classSha))])
  IO.println s!"TYPED {line.compress}"
  return (if (line.getObjValAs? Bool "ok").toOption.getD false then 0 else 1)


/-- `typed-rows FILE [--tables DIR] --out DIR`: a typed template's rows as the verifier derives them, through
`deriveChecked` with its tables held (`Typed.rows`), written in the writer's `rows/` formats. -/
def typedRowsCmd (args : List String) : IO UInt32 := do
  let (some path, some out) := (args.head?, argOf args "--out") | IO.eprintln "typed-rows FILE [--tables DIR] --out DIR"; return 2
  match Typed.rows Tags.circuitTypes (← IO.FS.readBinFile path) (← readTables args) with
  | .error e => IO.eprintln s!"typed-rows: {e}"; return 1
  | .ok files =>
    IO.FS.createDirAll out
    for (n, text) in files do IO.FS.writeFile (out ++ "/" ++ n) text
    IO.println s!"TYPED-ROWS {files.size}"
    return 0

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

/-- `--program P --partition Q --work K --work-table F`: the work law the verifier derives from its own program, partition
object and work table of works and floors (`HmRow.workLaw`), and the population; none without `--work`. -/
def workInput (args : List String) : IO (Except String (Option (Draw.Law × Nat))) := do
  let some k := argOf args "--work" | return .ok none
  let some k := k.toNat? | return .error "--work K: K is a natural number"
  let program ← match ← programInput args with
    | .ok (some p) => pure p
    | .ok none => return .error "--work needs --program (the verifier derives the strata itself)"
    | .error e => return .error e
  let obj ← match ← partitionInput args with
    | .ok (some o) => pure o
    | .ok none => return .error "--work needs --partition"
    | .error e => return .error e
  let table ← match ← workTableInput args with
    | .ok (some t) => pure t
    | .ok none => return .error "--work needs --work-table (the verifier's own work per template)"
    | .error e => return .error e
  return (HmRow.workLaw obj program k table).map some

/-- The verifier's own law from `--stratified K` or `--work K` (at most one), and the population; none without either. -/
def ownLaw (args : List String) : IO (Except String (Option (Draw.Law × Nat))) :=
  match argOf args "--stratified", argOf args "--work" with
  | some _, some _ => pure (.error "at most one of --stratified, --work")
  | none, some _ => workInput args
  | _, none => stratifiedInput args

/-- `draw (--population N (--k K | --p NUM/DEN) | --program P --partition Q (--stratified K | --work K --work-table F))
[--closure F] [--stream HEX | --stream-file F]`: the unit draw (§7.3), from the operating system's randomness (a stratified or work draw:
one stream per stratum), or from the given bytes (a test; a stratified or work draw's streams are its consecutive segments).
The strata are the verifier's own, derived from its program and partition object, and a work law's works and floors are its
own work table's; with `--closure F`, a work draw is widened by the verifier's closure map. Prints `DRAW {…}`, the object
the verifier sends in the clear. -/
def drawCmd (args : List String) : IO UInt32 := do
  let (law, pop) ← match ← ownLaw args with
    | .error e => IO.eprintln s!"draw: {e}"; return 2
    | .ok (some lp) => pure lp
    | .ok none =>
      let some pop := (argOf args "--population").bind String.toNat?
        | IO.eprintln "draw: --population N, or --program P --partition Q (--stratified K | --work K --work-table F)"; return 2
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
  let d ← match ← closureInput args, d.law with
    | .error e, _ => IO.eprintln s!"draw: {e}"; return 2
    | .ok none, _ => pure d
    | .ok (some cm), .work .. => pure { d with closure := Draw.closureOf cm d.units }
    | .ok (some _), _ => IO.eprintln "draw: --closure widens a work draw only"; return 2
  IO.println s!"DRAW {d.toJson.compress}"
  return 0

/-- `draw-test FILE [--program P --partition Q (--stratified K | --work K --work-table F)] [--closure F]`: U1–U3 (§7.3) on each case
`{"record", "statement", "instances"}` (a draw or null each), one `DRAWCHECK {"ok", "why"}` line per case. With a law's
flags, the statement's draw is also held to the verifier's own law and population (`draw.check_draw`'s `law` and
`population`). -/
def drawTest (path : String) (args : List String) : IO UInt32 := do
  let .ok (.arr cases) := Lean.Json.parse (← IO.FS.readFile path) | IO.eprintln "draw-test: not a JSON list"; return 2
  let own ← match ← ownLaw args with
    | .ok o => pure o
    | .error e => IO.eprintln s!"draw-test: {e}"; return 2
  let closure ← match ← closureInput args with
    | .ok c => pure c
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
      if let (some cm, some s) := (closure, side "statement") then
        let d ← Draw.UnitDraw.ofJson s
        if d.closure != Draw.closureOf cm d.units then throw "U3: the draw's closure is not the verifier's own (its closure map)"
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

/-- `cut-check FILE...`: `check_cut` (the invariant on one Call's cut against its graph, and the width rule) on each case, one
`CUTCHECK` line per file; a case with `"recompute": "report"` (`Q_word` v2's rule) also gives its `recomputed_across` count. -/
def cutCheckCmd (paths : List String) : IO UInt32 := do
  let mut all := true
  for path in paths do
    let res := match Lean.Json.parse (← IO.FS.readFile path) with
      | .error e => Except.error s!"not JSON: {e}"
      | .ok j => do return (← Partition.checkCutOfJson j, ← Partition.reportOfJson j)
    let line := match res with
      | .error e => Lean.Json.mkObj [("file", Lean.Json.str path), ("ok", Lean.Json.bool false), ("error", Lean.Json.str e)]
      | .ok ((g, owner, committed, limits), report) =>
        let (cs, wide) := Partition.checkCut g owner committed limits report
        Lean.Json.mkObj ([("file", Lean.Json.str path), ("ok", Lean.Json.bool cs.isEmpty),
          ("codes", Lean.Json.arr (cs.toArray.map Lean.Json.str)),
          ("wide", Lean.Json.arr (wide.map fun (x : Nat × Nat × Nat) => Lean.Json.arr #[Lean.toJson x.1, Lean.toJson x.2.1, Lean.toJson x.2.2]))] ++
          if report && owner.size == g.n then [("recomputed_across", Lean.toJson (g.cut owner committed true).recomputedAcross.size)] else [])
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

/-- `qword-program DESCRIPTOR [X [W]] [--digests] [--cuts] [--query-version N]`: `Q_word` v1 (or vN, 1 or 2) on the program
(`Flock.Program`, `Flock.Extract`, `Flock.Qword`): the program's SHA-512, the population and per Call its units. In full: per
Call its owners, derived committed set and graph, and every unit's `locate`. With `--digests`: per Definition a Call names, its
units and the SHA-512 of its owners, and of its graph for a flat cut. v2's units are v1's; with `--cuts` it checks each cut by
v2's rule and gives each Definition's `recomputed_across`. One `QWORDPROGRAM` line. -/
def qwordProgramCmd (args : List String) : IO UInt32 := do
  let path := args.headD ""
  let nums := ((args.drop 1).takeWhile (·.all Char.isDigit)).filterMap String.toNat?
  let X := nums.headD 16
  let W := (nums.drop 1).headD 32
  let digests := args.contains "--digests"
  let cuts := args.contains "--cuts"
  let version := ((argOf args "--query-version").bind String.toNat?).getD 1
  let bytes ← try IO.FS.readBinFile path catch _ => pure ByteArray.empty
  let res : V Lean.Json := do
    if !HmRow.EVALUABLE.contains ("Q_word", version) then throw s!"query-unknown: Q_word v{version}"
    let P ← Program.decode bytes
    if let some v := P.orderViolation then
      return Lean.Json.mkObj [("inapplicable", v), ("error", Lean.Json.str (match Extract.evaluate P X with
        | .error e => e
        | .ok _ => ""))]
    let calls ← Extract.evaluate P X
    let units : Qword.Units := ⟨calls.map fun (k, _, c) => (k, c)⟩
    -- `--template T --units i,j,…`: the claimed units' membership, as a Flock statement's finding checks it
    if let some t := argOf args "--template" then
      let idx := ((argOf args "--units").getD "").splitOn "," |>.filterMap String.toNat? |>.toArray
      return match Extract.claimedUnits P calls t idx with
        | .ok note => Lean.Json.mkObj [("members", Lean.Json.bool true), ("note", Lean.Json.str note)]
        | .error e => Lean.Json.mkObj [("members", Lean.Json.bool false), ("note", Lean.Json.str e)]
    let base := [("program_sha512", Lean.Json.str (hexOf P.sha512)), ("population", Lean.toJson units.population),
      ("calls", Lean.Json.arr (calls.map fun (k, d, c) => Lean.Json.arr #[Lean.toJson k, Lean.Json.str (P.defs[d]!).key,
        Lean.toJson c.units]))]
    -- `--cuts`: every Definition's cut checked on its Call's whole graph (`partition_object.verify`'s `cut`)
    let cutJson : List (String × Lean.Json) ← if !cuts then pure [] else do
      let checked ← Extract.checkCalls P calls X W (report := version == 2)
      let entry := fun (r : Extract.CutCheck) =>
        if version == 2 then r.toJson.setObjVal! "recomputed_across" (Lean.toJson r.across) else r.toJson
      pure [("cuts", Lean.Json.mkObj (checked.toList.map fun (d, r) => ((P.defs[d]!).key, entry r))),
        ("cut", Lean.Json.arr ((Extract.cutFailures calls checked).map fun (k, r) =>
          Lean.Json.arr #[Lean.toJson k, Lean.toJson r.codes.toArray]))]
    let base := base ++ cutJson
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
      if let .flat _ kinds := c then o := o ++ [("kinds", Lean.toJson kinds)]
      let gr ← Extract.graph P d
      o := o ++ [("graph", graphJson gr),
        ("committed", Lean.toJson (if own.size == gr.g.n then Partition.derivedCommitted gr.g own else #[]))]
      detail := detail.push (Lean.Json.mkObj o)
    let loc := (Array.range units.population).map fun i => match units.locate i with
      | some (p, k, u, path) => Lean.Json.arr #[Lean.toJson i, Lean.toJson #[p, k, u],
          Lean.Json.arr (path.toArray.map fun (a, b, c) => Lean.toJson #[a, b, c])]
      | none => Lean.Json.null
    let owners := Lean.Json.arr (calls.map fun (k, _, c) => Lean.Json.arr #[Lean.toJson k, Lean.toJson c.owner])
    let od := hexOf (Sha512.hash ("verity/partition/v1/owners".toUTF8.push 0 ++ (← canon owners).toUTF8))
    return Lean.Json.mkObj (base ++ [("owners_digest", Lean.Json.str od), ("detail", Lean.Json.arr detail),
      ("locate", Lean.Json.arr loc)])
  let line := match res with
    | .ok r => r.setObjVal! "ok" (Lean.Json.bool !(r.getObjVal? "inapplicable").isOk)
    | .error e => Lean.Json.mkObj [("ok", Lean.Json.bool false), ("error", Lean.Json.str e)]
  IO.println s!"QWORDPROGRAM {line.compress}"
  return (if (line.getObjValAs? Bool "ok").toOption.getD false then 0 else 1)

def main (args : List String) : IO UInt32 := do
  match args with
  | ["merkle-test", path] => merkleTest path
  | ["coin-tree-check", path] => coinTreeCmd path
  | ["hm96-test", path] => hm96Test path
  | ["hm96-test", path, "sha512"] => hm96Test path Hm96.sha512
  | "unit-cut" :: paths => unitCutCmd paths
  | "cut-check" :: paths => cutCheckCmd paths
  | "evaluate" :: paths => evaluateCmd paths
  | "qword-program" :: rest => qwordProgramCmd rest
  | "qcall-program" :: rest => Qcall.cmd rest
  | "input-ids" :: ids => Qcall.inputIdsCmd ids
  | "archive-put" :: dir :: paths => archivePutCmd dir paths
  | "template-query" :: paths => templateQueryCmd paths
  | "net-check" :: paths => netCheckCmd paths
  | "draw" :: rest => drawCmd rest
  | "draw-test" :: path :: rest => drawTest path rest
  | "statement" :: rest => statementCmd rest
  | "verify" :: rest => verifyCmd rest
  | "rec-inner" :: rest => recInnerCmd rest
  | "rec-coins" :: rest => recCoinsCmd rest
  | "rec-statements" :: rest => recStatementsCmd rest
  | "rec-transcript" :: rest => recTranscriptCmd rest
  | ["comp-check", path] => compCheck path
  | ["field-test", path] => fieldTest path
  | "region-words" :: rest => regionWordsCmd rest
  | ["hash-test", path] => hashTest path
  | "typed-expand" :: rest => typedExpandCmd rest
  | "typed-rows" :: rest => typedRowsCmd rest
  | "circuit-type" :: paths => CircuitType.cmd paths
  | "layout" :: paths => Layout.cmd paths
  | _ => IO.eprintln "usage: flock-verify field-test FILE"; return 2
