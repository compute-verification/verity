import Flock.Zk
import Flock.ProvedScope
import Flock.RecVStar
import Flock.Registered
import Flock.Lookup

/-!
# `flock-witness`: the recursive audit's predicates, evaluated on recorded sessions

`RecursiveAudit`'s event (`Proofs.Flock.Recursive.Property`) and the facts its premises take of V*'s sessions are this
library's functions returning `.ok`, so they can be evaluated on recorded files, and a theorem whose event no honest session
meets is found by a run rather than by reading. Each check here is one of them, as a function of the files `flock-verify`
reads; `Proofs.Flock.Recursive.Witness` proves that each check's `.ok` is the model's predicate.

* `inner --circuit F --public F [--partition F] [--program F] [--registered F] [--record F]`: the inner statement.
  `acceptsStmt` (`AcceptsStmt`: `Stmt.setupTables` at `RecVStar.innerTags`, the coin server's coins and the statement's
  partition object and program returns one table, and `Zk.schedule` accepts its `m`), `acceptsDescription`
  (`AcceptsDescription`, at each of the verifier's own registered reads), and with `--record` the recorded inner
  session's Σ (its `hello`'s `link.sigma`) against the statement's. `accepts_stmt`'s line also prints the statement's
  partition finding, the verifier's own inner shape (`shapeOf`, `innerAt`), the number of V*'s outer statements over it
  (`outerPartsOf`, `outerParts`) and V*'s statements there (`RecVStar.statements`).
* `session [--statement ID] --circuit F --public F [--partition F] [--program F] [--registered F] [--tables DIR]
  [--session-tables J] [--rec-inner F --rec-part J --rec-bindings F [--rec-rows F]] --session DIR...`: one of V*'s outer
  sessions, with `flock-verify verify --zk`'s flags. Once: `tagsOk` (`DecodeZL.TagsOk`, public outputs) and `tagsOkZHJ`
  (`ZkHidden.TagsOkZHJ`, hidden outputs, `ZkOuterR.hTags`), which no session meets both of, the statement check without
  and with the registered reads (`ScopeZ`, `ScopeZR`), and the circuit's `scopeOk` (`VStmt.hscope`). Per session: `fullDraw`
  (the record's draw is the full draw `RecordCustody` reads, `subset` of every unit of the population), `acceptsZKRAt`
  (`ZkReg.AcceptsZKR`, with the statement's `m` and its schedule's level 0), and with `--rec-inner` `accepts_zkv`
  (`AcceptsZKV`: `acceptsZKRAt`, then `checkZKV`, V*'s statement check, whose refusal it prints apart as `check_stmt`).

One `WITNESS` line per check: its name, `ok`, and the refusal when it fails. Exit 0 iff every check passes.
-/

namespace Flock.Witness

open Lean (Json)

/-- What `flock-verify verify --zk` reads besides the session directory, for one of V*'s sessions: `Exec.Inputs`'s
fields but the work, stratified and closure inputs, which V*'s full draw reads none of, and the coins, which `--zk` sets. -/
structure Inputs where
  tags : Tags
  circuitFile : ByteArray
  publicFile : ByteArray
  count : Nat
  tables : List (String × ByteArray)
  partition : Option Json
  program : Option ByteArray

/-- A record's `unit_draw`, when it is a draw (not absent, not null): `Main.recordDraw`. -/
def recordDraw (record : String) : Option Json :=
  match Json.parse record with
  | .ok j => match j.getObjVal? "unit_draw" with
    | .ok Json.null => none
    | .ok d => some d
    | .error _ => none
  | .error _ => none

/-- The coins `verify --zk` sets a session up on: the keyed coin tree of its tables. -/
def zkCoins (I : Inputs) : Coins := .tree (Zk.coinSpec (I.tags.tableNames I.count))

/-- The tags `verify --zk` sets a session up under. -/
def zkTags (I : Inputs) : Tags := Zk.tags (I.tags.withCoins (zkCoins I)) (Zk.maskLogOf I.circuitFile) I.count

/-- The session's statement `verify --zk` sets up for a record. -/
def setupOf (I : Inputs) (record : String) : V (SessionSpec × Array Stmt) :=
  Stmt.setupTables (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.count I.tables (recordDraw record) I.partition
    I.program

/-- `verify --zk --registered F`'s verdict on a session set up as `s`: every table's registered reads against `own`,
every table's shape, and `Zk.verify`. -/
def acceptsZKRAt (s : V (SessionSpec × Array Stmt)) (own : Array Registered.Port) (record : String)
    (proofs : Array ByteArray) : V Unit := do
  let r ← s
  r.2.forM (checkRegistered · (some own))
  r.2.forM Zk.shapeOk
  Zk.verify r.1 (r.2.map Setup.ofCircuit) record proofs

/-- The statement check `--zk` makes, with or without the verifier's own registered reads (`reg`). -/
def scope (I : Inputs) (reg : Bool) : V Unit :=
  ProvedScope.check (zkTags I) true I.circuitFile I.publicFile I.tables I.count false reg

/-- The tags the soundness headlines read: public outputs, retained rounds, `hm96-sha512/v1` leaves, one table. -/
def tagsOk (I : Inputs) : V Unit :=
  if I.tags.hiddenOutputs = false ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1" ∧ I.count ≤ 1 then
    pure ()
  else
    throw s!"{I.tags.name}: hidden outputs {I.tags.hiddenOutputs}, retained rounds {I.tags.retainRounds}, leaves {I.tags.merkleLeaf}, {I.count} tables"

/-- The tags V*'s sessions take (`ZkOuterR.hTags`): hidden outputs, retained rounds, `hm96-sha512/v1` leaves, any number
of tables. -/
def tagsOkZHJ (I : Inputs) : V Unit :=
  if I.tags.hiddenOutputs = true ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1" then
    pure ()
  else
    throw s!"{I.tags.name}: hidden outputs {I.tags.hiddenOutputs}, retained rounds {I.tags.retainRounds}, leaves {I.tags.merkleLeaf}"

/-- The session's circuit, as `ZkOuterR` parses it, when `scopeOk` accepts it. -/
def scopeOk (I : Inputs) : V Circuit := do
  let c ← HmRow.parse (zkTags I) I.circuitFile I.tables
  if ProvedScope.scopeOk c then pure c else throw "scopeOk refuses the circuit"

/-- The record's draw is the full draw over `n` units: `subset n` of the population `[0, n)`, every unit, no closure. -/
def fullDraw (n : Nat) (record : String) : V Unit :=
  match recordDraw record with
  | none => throw "the record has no draw"
  | some d => do
    let D ← Draw.UnitDraw.ofJson d
    match D.law with
    | .subset k =>
      if k = n ∧ D.population = n ∧ D.units = Array.range n ∧ D.closure = #[] then pure ()
      else throw s!"the draw is subset:{k} of {D.units.size} units over {D.population}, not every one of {n}"
    | _ => throw "the draw's law is not subset"

/-- The one table of a session. -/
def one (sts : Array Stmt) : V Stmt :=
  match sts.toList with
  | [st] => .ok st
  | _ => .error s!"{sts.size} tables, not one"

/-- The inner statement's setup, as `RecVStar.innerOf` makes it. -/
def innerSetup (cf pf : ByteArray) (partition : Option Json) (program : Option ByteArray) : V (SessionSpec × Array Stmt) :=
  Stmt.setupTables RecVStar.innerTags cf pf Coins.os 1 (partition := partition) (program := program)

/-- `acceptsStmt` at a setup already made. -/
def acceptsStmtAt (r : V (SessionSpec × Array Stmt)) : V Stmt := do
  let r ← r
  let st ← one r.2
  let _ ← Zk.schedule st.m
  pure st

/-- **Lean's verifier accepts `st` as the inner statement** (`AcceptsStmt`), from the inner statement's files, partition
object and program, as `RecVStar.innerOf` sets it up. -/
def acceptsStmt (cf pf : ByteArray) (partition : Option Json) (program : Option ByteArray) : V Stmt :=
  acceptsStmtAt (innerSetup cf pf partition program)

/-- The verifier's own inner shape at the inner statement `st` set up with `spec` (`innerAt`, `RecVStar.innerOf`'s value). -/
def shapeOf (spec : SessionSpec) (st : Stmt) : RecVStar.Inner :=
  { m := st.m, kLog := st.c.kLog, claims := 2 + spec.points * st.regions.size, regions := st.regions.size }

/-- The number of V*'s outer statements over an inner session of `m` variables, `k_log` and `claims` claims: the body of
`Proofs.Flock.Recursive.outerParts`, which is this by `rfl`. -/
def outerPartsOf (m kLog claims : Nat) : Nat :=
  match RecResiduals.shape? m kLog claims with
  | none => 0
  | some sh =>
    match RecAlgebra.structureOf sh >>= fun S => RecResiduals.groups S (RecResiduals.portSizes sh) with
    | .ok gs => sh.sched.levels.size + gs.size
    | .error _ => 0

/-- **`verify --zk --registered F --rec-inner F --rec-part J --rec-bindings F --rec-rows F`'s V* check on a session set
up as `s`** (`AcceptsZKV`'s second conjunct): each statement passes `RecVStar.checkStmt` at the inner shape `i`, as V*'s
statement `j` under the verifier's own bindings `f`, against the inputs, `own` and the shared rows `rows`. -/
def checkZKV (I : Inputs) (s : V (SessionSpec × Array Stmt)) (i : RecVStar.Inner) (j : Nat) (f : VStar.Bindings)
    (rows : Option (Array Registered.Port)) (own : Array Registered.Port) : V Unit := do
  let r ← s
  r.2.forM (RecVStar.checkStmt (some i) I.tags I.tables · I.circuitFile I.partition I.program (some own) (some j) (some f)
    rows)

/-- **`st`'s description is accepted against the verifier's own registered reads** (`AcceptsDescription` at each). -/
def acceptsDescription (st : Stmt) (own : Array Registered.Port) : V Unit := do
  Registered.checkDescription st.c st.pub own
  if own.isEmpty then throw "the verifier holds no registered reads (--registered)"

def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

def readTables (args : List String) : IO (List (String × ByteArray)) := do
  let some dir := argOf args "--tables" | return []
  let mut out := []
  for t in MUFU_TABLES do
    let p : System.FilePath := dir ++ s!"/{t.name}.u32"
    if (← p.pathExists) then out := out ++ [(t.name, (← IO.FS.readBinFile p))]
  return out

def readJson (path : String) : IO (Except String Json) := do
  try
    let t ← IO.FS.readFile path
    return (Json.parse t).mapError fun e => s!"{path}: not JSON: {e}"
  catch e => return .error s!"{path}: {e}"

def registeredOf (args : List String) : IO (Except String (Array Registered.Port)) := do
  let some f := argOf args "--registered" | return .ok #[]
  return (← readJson f) >>= Registered.ofJson

/-- `--partition` and `--program`, each optional, as `flock-verify` reads them. -/
def bindingsOf (args : List String) : IO (Except String (Option Json × Option ByteArray)) := do
  let partition : Except String (Option Json) ← match argOf args "--partition" with
    | none => pure (.ok none)
    | some f => do pure ((← readJson f).map some)
  let program ← (argOf args "--program").mapM IO.FS.readBinFile
  return partition.map fun p => (p, program)

/-- `--rec-inner F`, `--rec-part J`, `--rec-bindings F` and `--rec-rows F`, as `flock-verify verify` reads them: the inner
shape (`RecVStar.Inner.ofJson`), which of V*'s statements the sessions are, the verifier's own bindings
(`RecVStar.ownBindingsOfJson`) and its shared rows (`Registered.ofJson`). -/
def vstarOf (args : List String) :
    IO (Except String (Option (RecVStar.Inner × Nat × VStar.Bindings × Option (Array Registered.Port)))) := do
  let some fi := argOf args "--rec-inner" | return .ok none
  let (some j, some fb) := ((argOf args "--rec-part").bind String.toNat?, argOf args "--rec-bindings")
    | return .error "--rec-inner goes with --rec-part J and --rec-bindings F"
  let ij ← readJson fi
  let bj ← readJson fb
  let rj ← (argOf args "--rec-rows").mapM readJson
  return do
    let i ← ij >>= RecVStar.Inner.ofJson
    let f ← bj >>= RecVStar.ownBindingsOfJson
    let rows ← rj.mapM (· >>= Registered.ofJson)
    pure (some (i, j, f, rows))

/-- A set-up statement's `m` and its `Zk.schedule`'s level 0. -/
def scheduleOf (s : V (SessionSpec × Array Stmt)) : List (String × Json) :=
  match s with
  | .ok (_, sts) => match sts[0]? with
    | some st => match Zk.schedule st.m with
      | .ok (sch, _) => match sch.levels[0]? with
        | some l => [("m", Lean.toJson st.m), ("level0", Json.mkObj [("n", Lean.toJson sch.n), ("rate", Lean.toJson l.rate),
            ("q", Lean.toJson l.q), ("cols", Lean.toJson l.cols), ("d", Lean.toJson l.d), ("lanes", Lean.toJson l.lanes)])]
        | none => [("m", Lean.toJson st.m)]
      | .error _ => [("m", Lean.toJson st.m)]
    | none => []
  | .error _ => []

def line (check : String) (r : V Unit) (extra : List (String × Json) := []) : IO Bool := do
  let j := Json.mkObj ([("check", Json.str check), ("ok", Json.bool r.toBool)] ++
    (match r with | .ok () => [] | .error e => [("why", Json.str e)]) ++ extra)
  IO.println s!"WITNESS {j.compress}"
  return r.toBool

/-- The Σ a session record's `hello` names (`link.sigma`). -/
def recordSigma (record : Json) : V String := do
  let h ← Json.parse (← record.getObjValAs? String "hello")
  (← h.getObjVal? "link").getObjValAs? String "sigma"

def innerCmd (args : List String) : IO UInt32 := do
  let usage := "usage: flock-witness inner --circuit F --public F [--partition F] [--program F] [--registered F] [--record F]"
  let (some cf, some pf) := (argOf args "--circuit", argOf args "--public") | IO.eprintln usage; return 2
  let (partition, program) ← match ← bindingsOf args with
    | .ok b => pure b
    | .error e => IO.eprintln e; return 2
  let own ← match ← registeredOf args with
    | .ok o => pure o
    | .error e => IO.eprintln s!"registered reads: {e}"; return 2
  let t0 ← IO.monoMsNow
  let r := innerSetup (← IO.FS.readBinFile cf) (← IO.FS.readBinFile pf) partition program
  let st := acceptsStmtAt r
  let extra := match r, st with
    | .ok (spec, _), .ok st =>
      let i := shapeOf spec st
      [("m", Lean.toJson st.m), ("sigma", Json.str (hexOf st.sigma)), ("partition_check", st.partition),
        ("shape", i.toJson), ("outer_parts", Lean.toJson (outerPartsOf i.m i.kLog i.claims)),
        ("statements", match RecVStar.statements i with
          | .ok ss => Json.arr (ss.map (Json.str ·.name))
          | .error e => Json.str s!"refused: {e}")]
    | _, _ => []
  let mut all ← line "accepts_stmt" (st.map fun _ => ()) extra
  IO.eprintln s!"inner setup: {(← IO.monoMsNow) - t0} ms"
  let .ok st := st | return 1
  all := (← line "accepts_description" (acceptsDescription st own) [("registered", Lean.toJson own.size)]) && all
  if let some rf := argOf args "--record" then
    let r := (← readJson rf) >>= recordSigma
    let ok : V Unit := r >>= fun s => if s == hexOf st.sigma then pure () else throw s!"the record's sigma is {s}"
    all := (← line "record_sigma" ok) && all
  return (if all then 0 else 1)

def sessionCmd (args : List String) : IO UInt32 := do
  let usage := "usage: flock-witness session [--statement ID] --circuit F --public F [--partition F] [--program F] [--registered F] [--tables DIR] [--session-tables J] [--rec-inner F --rec-part J --rec-bindings F [--rec-rows F]] --session DIR..."
  let (some cf, some pf) := (argOf args "--circuit", argOf args "--public") | IO.eprintln usage; return 2
  let some tags := Tags.find? ((argOf args "--statement").getD Tags.circuit.name)
    | IO.eprintln s!"unknown statement; known: {Tags.all.map (·.name)}"; return 2
  let some count := ((argOf args "--session-tables").getD "1").toNat? | IO.eprintln usage; return 2
  let (partition, program) ← match ← bindingsOf args with
    | .ok b => pure b
    | .error e => IO.eprintln e; return 2
  let own ← match ← registeredOf args with
    | .ok o => pure o
    | .error e => IO.eprintln s!"registered reads: {e}"; return 2
  let vstar ← match ← vstarOf args with
    | .ok v => pure v
    | .error e => IO.eprintln s!"V*'s inputs: {e}"; return 2
  let circuitFile ← IO.FS.readBinFile cf
  let publicFile ← IO.FS.readBinFile pf
  let tables ← readTables args
  let I : Inputs := { tags, circuitFile, publicFile, count, tables, partition, program }
  let mut all ← line "tags_ok" (tagsOk I)
  all := (← line "tags_ok_zhj" (tagsOkZHJ I)) && all
  all := (← line "scope_z" (scope I false)) && all
  all := (← line "scope_zr" (scope I true)) && all
  let c := scopeOk I
  all := (← line "scope_ok" (c.map fun _ => ())) && all
  let n : Option Nat := match c with
    | .ok c => (ProvedScope.publicOf (zkTags I) c I.publicFile).toOption.map (·.n)
    | .error _ => none
  for dir in (args.dropWhile (· != "--session")).drop 1 do
    let t1 ← IO.monoMsNow
    let record ← IO.FS.readFile (dir ++ "/session.json")
    let draw : V Unit := match n with
      | some n => fullDraw n record
      | none => throw "the public file does not load"
    all := (← line "full_draw" draw [("session", Json.str dir), ("population", Lean.toJson n)]) && all
    let s := setupOf I record
    let verdict ← match s with
      | .error e => pure (Except.error s!"setup: {e}" : V Unit)
      | .ok (spec, _) => do
        let proofs ← spec.repStreams.mapM fun (_, _, w) => IO.FS.readBinFile s!"{dir}/{w.replace "/" "."}.bin"
        pure (acceptsZKRAt s own record proofs)
    all := (← line "accepts_zkr" verdict ([("session", Json.str dir)] ++ scheduleOf s)) && all
    if let some (i, j, f, rows) := vstar then
      let c := checkZKV I s i j f rows own
      let why : List (String × Json) := match c with
        | .ok () => []
        | .error e => [("check_stmt", Json.str e)]
      all := (← line "accepts_zkv" (verdict >>= fun _ => c) ([("session", Json.str dir), ("part", Lean.toJson j)] ++ why)) && all
    IO.eprintln s!"{dir}: {(← IO.monoMsNow) - t1} ms"
  return (if all then 0 else 1)

def cmd : List String → IO UInt32
  | "inner" :: args => innerCmd args
  | "session" :: args => sessionCmd args
  | _ => do IO.eprintln "usage: flock-witness (inner | session) ..."; return 2

end Flock.Witness
