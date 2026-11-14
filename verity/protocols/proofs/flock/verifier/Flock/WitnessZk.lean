module

public import Flock.HmRow
public import Flock.Lookup
public import Flock.ProvedScope
public import Flock.Registered
public import Flock.Zk

/-!
# What `flock-witness` and the ZK server read of one `--zk` session

The inputs the Lean verifier's `verify --zk` reads for a session, the session's setup and verdict on them, the checks
of its tags and its circuit, and the readers of the command line's files. Nothing here names the recursion:
`Flock.Witness` holds the checks that do.
-/

@[expose] public section

namespace Flock.Witness

open Lean (Json)

/-- What `flock-verify verify --zk` reads besides the session directory, for one of V*'s sessions: `Exec.Inputs`'s
fields but the work, stratified and closure inputs, which V*'s full sample reads none of, and the coins, which `--zk` sets. -/
structure Inputs where
  tags : Tags
  circuitFile : ByteArray
  publicFile : ByteArray
  count : Nat
  tables : List (String × ByteArray)
  partition : Option Json
  program : Option ByteArray

/-- A record's `unit_draw`, when it is a sample (not absent, not null), rebuilt (`norm`): `Main.recordDraw`. -/
def recordDraw (record : String) : Option Json :=
  match JsonParse.parse record with
  | .ok j => match j.getObjVal? "unit_draw" with
    | .ok Json.null => none
    | .ok d => some (Flock.norm d)
    | .error _ => none
  | .error _ => none

/-- The coins `verify --zk` sets a session up on: the keyed coin tree of its tables. -/
def zkCoins (I : Inputs) : Coins := .tree (Zk.coinSpec (I.tags.tableNames I.count))

/-- The tags `verify --zk` sets a session up under. -/
def zkTags (I : Inputs) : Tags := Zk.tags I.tags (Zk.maskLogOf I.circuitFile) I.count

/-- The session's statement `verify --zk` sets up for a record. -/
def setupOf (I : Inputs) (record : String) : V (SessionSpec × Array Stmt) :=
  Stmt.setupTables (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.count I.tables (recordDraw record) I.partition
    I.program

/-- `verify --zk --registered F`'s verdict on a session set up as `s`: every table's registered reads against `own`,
every port of every table's circuit a port of `own` (`Registered.checkComplete`), every table's shape, and
`Zk.verify`. -/
def acceptsZKRAt (s : V (SessionSpec × Array Stmt)) (own : Array Registered.Port) (record : String)
    (proofs : Array ByteArray) : V Unit := do
  let r ← s
  r.2.forM (checkRegistered · (some own))
  r.2.forM fun st => Registered.checkComplete st.c own
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
    return (JsonParse.parse t).mapError fun e => s!"{path}: not JSON: {e}"
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

end Flock.Witness
