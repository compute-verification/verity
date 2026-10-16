import Definitions.Claims
import Flock.ProvedScope
import Proofs.Flock.Soundness.Discharge.ZkExec.Event

/-!
# The verifier's statement, read as `Claims.Flock.Stmt`

What `flock-verify verify` reads of a statement besides the session directory (`Read`), what its setup makes of it
(`Read.parse`: `Stmt.setupTables`, and under `--zk` every table's `Zk.shapeOk`, as `Main.verifyCmd`'s `setupOf` runs
them), and that statement in readable form (`toReadable`): `m` and `kLog` of the first table's statement, the
relation's fields, and the `Form` the scope check reads, each field the value `ProvedScope.check` tests.

* `Read`: the flags' values (`Exec.Inputs`, under `--zk` before `zkTags` and `zkCoins` replace `--statement`'s tags and
  `--coins`), whether the verifier is the masked one, the record's draw, the verifier's own public input words
  (`--public-inputs`) and whether it holds its own registered reads (`--registered`).
* `Read.check`: `ProvedScope.check` as `verifyCmd` calls it.
* `Read.form`: the `Form` of `ProvedScope.check`'s tests, on the circuit and public file as it reads them
  (`ProvedScope.circuitOf`, `ProvedScope.publicOf`).
* `Relation`: the readable relation's fields (`C`, `inp`, `out`, `pub`, `cs`, `grp`), which `toReadable` takes as a
  function of the parsed statement; the scope reads none of them.
-/

namespace Flock.Readable

open FlockSoundness.Discharge

/-- What `flock-verify verify` reads of a statement besides the session directory: its flags' values (`I`, as
`Exec.Inputs` holds them: `--statement`'s tags after `withCoins` and `--coins`, which `verify --zk` replaces by
`zkTags I` and `zkCoins I`), whether it is the masked verifier (`--zk`), the record's draw (`Main.recordDraw`, none
without one), its own public input words (`--public-inputs`) and whether it holds its own registered reads
(`--registered`). -/
structure Read where
  I : Exec.Inputs
  zk : Bool
  draw : Option Lean.Json
  publicInputs : Option ByteArray
  registered : Bool

/-- The session's tags: `verify --zk`'s (`zkTags`), else `--statement`'s. -/
def Read.tags (r : Read) : Tags := if r.zk then ZkExec.zkTags r.I else r.I.tags

/-- The session's coins: `verify --zk`'s keyed coin tree (`zkCoins`), else `--coins`. -/
def Read.coins (r : Read) : Coins := if r.zk then ZkExec.zkCoins r.I else r.I.coins

/-- The session's statement as `verifyCmd` sets it up (`buildSession` on an `hm96Rows` statement). -/
def Read.setup (r : Read) : V (SessionSpec × Array Stmt) :=
  Stmt.setupTables r.tags r.I.circuitFile r.I.publicFile r.coins r.I.count r.I.tables r.draw r.I.partition r.I.program
    r.I.work r.I.stratified r.I.closure r.publicInputs

/-- The tables the setup accepted: `Read.setup`'s, each passing `Zk.shapeOk` under `--zk`. -/
def Read.tablesOf (r : Read) : Option (Array Stmt) :=
  match r.setup with
  | .ok (_, sts) => if r.zk then (match sts.forM Zk.shapeOk with | .ok () => some sts | .error _ => none) else some sts
  | .error _ => none

/-- **`verify`'s scope check** (`ProvedScope.check`), as `verifyCmd` calls it: the session's tags, the masked verifier
or not, the circuit, public and lookup-table files, the table count, and whether it holds its own public input words and
registered reads. -/
def Read.check (r : Read) : Except String Unit :=
  ProvedScope.check r.tags r.zk r.I.circuitFile r.I.publicFile r.I.tables r.I.count r.publicInputs.isSome r.registered

/-- The circuit as the scope check parses it (`ProvedScope.circuitOf`; `default` where it doesn't parse). -/
def Read.circ (r : Read) : Circuit :=
  match ProvedScope.circuitOf r.tags r.I.circuitFile r.I.tables with
  | .ok c => c
  | .error _ => default

/-- The public file as the scope check loads it (`ProvedScope.publicOf`; `default` where it doesn't load). -/
def Read.pub (r : Read) : Public :=
  match ProvedScope.publicOf r.tags r.circ r.I.publicFile with
  | .ok p => p
  | .error _ => default

/-- **The statement's form**, each field the value `ProvedScope.check` tests: the tags' (`statementRules`), the table
count, the verifier's own public input words and registered reads, a shared-row public file (`Public.tables`), and the
unit's and the rows' (`scopeOk`, `scopeOkH`). -/
def Read.form (r : Read) : Claims.Flock.Form where
  hm96Rows := r.tags.hm96Rows
  typed := r.tags.typed
  retainedRounds := r.tags.retainRounds
  leaf := r.tags.merkleLeaf
  tables := r.I.count
  publicInputs := r.publicInputs.isSome
  sharedRows := r.pub.tables.isSome
  registeredReads := r.registered
  hiddenOutputs := r.tags.hiddenOutputs
  unitInputGroups := r.circ.unit.inGroups.size
  tail := r.circ.outNet != r.circ.unitNet
  unitLog := r.circ.unit.unitLog
  readsOnlyPortBits := ProvedScope.portReadsB r.circ.unit
  rowsV1 := ProvedScope.v1Ports r.circ
  rowsV2OrSeg := ProvedScope.bitRows r.circ

/-- A statement the verifier parsed: what it read, and the tables its setup made of it. -/
structure Parsed extends Read where
  sts : Array Stmt

/-- **The verifier's parser**: the tables `Read.tablesOf` accepted, none where the setup refuses. -/
def Read.parse (r : Read) : Option Parsed := r.tablesOf.map fun sts => { r with sts }

/-- The parsed statement is the parser's own. -/
def Parses (p : Parsed) : Prop := p.toRead.parse = some p

/-- The readable relation's fields: the block circuit, each block's inputs and outputs, the public values, and the
commit strings with the hidden values each commits to. -/
structure Relation where
  C : Claims.Circuit
  inp : ℕ → List Claims.Flock.Src
  out : ℕ → List Claims.Flock.Src
  pub : List Bool
  cs : List Claims.Msg
  grp : ℕ → List ℕ

/-- **The verifier's statement, readable**: `m` and `kLog` of the first table's statement (`Stmt.m`, the circuit's
`k_log`), the relation's fields from `rel`, and the form the scope check reads (`Read.form`). -/
def toReadable (rel : Parsed → Relation) (p : Parsed) : Claims.Flock.Stmt where
  m := (p.sts[0]?.map (·.m)).getD 0
  kLog := (p.sts[0]?.map (·.c.kLog)).getD 0
  C := (rel p).C
  inp := (rel p).inp
  out := (rel p).out
  pub := (rel p).pub
  cs := (rel p).cs
  grp := (rel p).grp
  form := p.form

end Flock.Readable
