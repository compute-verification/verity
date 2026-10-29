import Proofs.Flock.Claims.Scope.Matches
import Flock.Registered
import Flock.RecSetup
import Proofs.Flock.Claims.JsonNorm

/-!
# The deployed verifier's statement: its bytes, its parser and its setup

A statement `x` of Flock's deployed verifier is the canonical byte encoding (`Raw.encode`) of exactly what
`flock-verify verify` reads besides the session directory: the `--statement` id, the circuit and public files, the lookup
tables, `--zk` and `--coins os`, `--session-tables`, the drawn session's draw (the record's `unit_draw`), and the bytes of
`--partition`, `--program`, `--work K` with `--work-table`, `--stratified`, `--closure`, `--registered` and
`--public-inputs`. Fields are length-prefixed (8 bytes, little-endian), options and flags one tag byte, after a format
string. `decode` reads that layout back and keeps only a reading whose encoding is `x` (`decode_encode`), so it is
injective (`decode_injective`).

The verifier's own parsers then run on those bytes, as `Main.verifyCmd` runs them (`Raw.ins`): `Tags.byName?` on the
statement id (a statement of `Tags.all`, else a templated one), the refusals of a plain-leaf statement under `--zk` and of `--zk` with `--coins`, the coins (`--zk`'s keyed coin tree,
else `--coins`), `Flock.JsonParse.parse` then, for the draw, `norm` (as `Main.recordDraw` rebuilds the record's), and
`Draw.workTableOfJson`, `Draw.closureMapOfJson` and `Registered.ofJson` on the files.
The result is C's `Readable.Read` (`readOf`), so the claims' parser is C's `parseOf` (`parse`: `Read.parse`,
`Stmt.setupTables` and under `--zk` every table's `Zk.shapeOk`, read as `Claims.Flock.Stmt` by `toReadable`), and the
scope check C's `scopeCheckOf`, the verifier's own scope refusal (`scopeCheck`: `ProvedScope.check`, then
`ProvedScope.sizesOk` on the tables), and each table's own statement C's `tableOf` (`table`); `readOf_zk` is the
agreement on `--zk` that C's `scope_iff` and `scopeMatches` take.

`Ins.setup` is `verifyCmd`'s `setupOf` at the statement's draw, with no recursive input and no `--identifier`: the
session's statement (`Stmt.setupTables`), each table's registered reads (`checkRegistered`) and the identifier's absence
(`checkIdentifier` without `--identifier`), `Registered.checkFull`, `RecSetup.checks`, `Zk.shapeOk` under `--zk`,
`RecCoins.check`, `RecVStar.checkPubRows`, and the draw's law (`ProvedScope.drawLaw`). It refuses a statement without
hm96-sha512 rows, which `verifyCmd` sets up on its compression circuit and then always refuses (such a session has no
draw, `drawLaw zk tags false none`); the deployed form does not carry `--comp`.
-/

namespace FlockDeploy

open Flock
open Lean (Json)

/-! ## The statement's bytes -/

/-- **What `flock-verify verify` reads of a statement**, as bytes and flags: the `--statement` id, the circuit and public
files, the lookup tables the circuit pins, `--zk`, `--coins os`, `--session-tables`, the drawn session's draw (JSON text),
and the files and numbers of `--partition`, `--program`, `--work`, `--work-table`, `--stratified`, `--closure`,
`--registered` and `--public-inputs`. -/
structure Raw where
  statement : String
  circuit : ByteArray
  publicFile : ByteArray
  tables : List (String × ByteArray)
  zk : Bool
  osCoins : Bool
  count : ℕ
  draw : Option ByteArray
  partition : Option ByteArray
  program : Option ByteArray
  work : Option ℕ
  workTable : Option ByteArray
  stratified : Option ℕ
  closure : Option ByteArray
  registered : Option ByteArray
  publicInputs : Option ByteArray

/-- A number in 8 bytes, little-endian. -/
def encNat (n : ℕ) : Claims.Msg := (List.range 8).map fun i => (n / 256 ^ i).toUInt8

/-- Bytes, after their length. -/
def encBytes (b : ByteArray) : Claims.Msg := encNat b.size ++ b.data.toList

def encStr (s : String) : Claims.Msg := encBytes s.toUTF8

def encBool (b : Bool) : Claims.Msg := [if b then 1 else 0]

def encOpt {α : Type} (e : α → Claims.Msg) : Option α → Claims.Msg
  | none => [0]
  | some a => 1 :: e a

def encList {α : Type} (e : α → Claims.Msg) (l : List α) : Claims.Msg := encNat l.length ++ l.flatMap e

/-- The statement encoding's format string, its first field. -/
def FORMAT : String := "flock-verify-statement/v1"

/-- **The statement's canonical bytes.** -/
def Raw.encode (w : Raw) : Claims.Msg :=
  encStr FORMAT ++ encStr w.statement ++ encBytes w.circuit ++ encBytes w.publicFile ++
    encList (fun p => encStr p.1 ++ encBytes p.2) w.tables ++ encBool w.zk ++ encBool w.osCoins ++ encNat w.count ++
    encOpt encBytes w.draw ++ encOpt encBytes w.partition ++ encOpt encBytes w.program ++ encOpt encNat w.work ++
    encOpt encBytes w.workTable ++ encOpt encNat w.stratified ++ encOpt encBytes w.closure ++
    encOpt encBytes w.registered ++ encOpt encBytes w.publicInputs

/-- A reader of bytes. -/
abbrev Rd := StateT Claims.Msg Option

def rdTake (n : ℕ) : Rd (List UInt8) := fun s => if n ≤ s.length then some (s.take n, s.drop n) else none

def rdNat : Rd ℕ := do
  let bs ← rdTake 8
  pure (bs.foldr (fun b acc => b.toNat + 256 * acc) 0)

def rdBytes : Rd ByteArray := do
  let n ← rdNat
  pure ⟨(← rdTake n).toArray⟩

def rdStr : Rd String := do
  let some s := String.fromUTF8? (← rdBytes) | failure
  pure s

def rdBool : Rd Bool := do
  match ← rdTake 1 with
  | [0] => pure false
  | [1] => pure true
  | _ => failure

def rdOpt {α : Type} (r : Rd α) : Rd (Option α) := do
  if ← rdBool then some <$> r else pure none

def rdListN {α : Type} (r : Rd α) : ℕ → Rd (List α)
  | 0 => pure []
  | n + 1 => do
    let a ← r
    let as ← rdListN r n
    pure (a :: as)

def rdList {α : Type} (r : Rd α) : Rd (List α) := do rdListN r (← rdNat)

/-- `Raw.encode`'s layout, read back. -/
def Raw.read : Rd Raw := do
  if (← rdStr) ≠ FORMAT then failure
  let statement ← rdStr
  let circuit ← rdBytes
  let publicFile ← rdBytes
  let tables ← rdList do
    let n ← rdStr
    let b ← rdBytes
    pure (n, b)
  let zk ← rdBool
  let osCoins ← rdBool
  let count ← rdNat
  let draw ← rdOpt rdBytes
  let partition ← rdOpt rdBytes
  let program ← rdOpt rdBytes
  let work ← rdOpt rdNat
  let workTable ← rdOpt rdBytes
  let stratified ← rdOpt rdNat
  let closure ← rdOpt rdBytes
  let registered ← rdOpt rdBytes
  let publicInputs ← rdOpt rdBytes
  pure ⟨statement, circuit, publicFile, tables, zk, osCoins, count, draw, partition, program, work, workTable,
    stratified, closure, registered, publicInputs⟩

/-- **The statement's decoder**: `Raw.read` on all of `x`, kept only when its encoding is `x`. -/
def decode (x : Claims.Msg) : Option Raw :=
  match Raw.read.run x with
  | some (w, []) => if w.encode = x then some w else none
  | _ => none

/-- A decoded statement encodes back to its bytes. -/
theorem decode_encode {x : Claims.Msg} {w : Raw} (h : decode x = some w) : w.encode = x := by
  unfold decode at h
  split at h
  · split_ifs at h with he
    cases h
    exact he
  · cases h

/-- **The decoder is injective.** -/
theorem decode_injective {x y : Claims.Msg} {w : Raw} (hx : decode x = some w) (hy : decode y = some w) : x = y :=
  (decode_encode hx).symm.trans (decode_encode hy)

/-! ## The verifier's parsers -/

/-- A file read as JSON, as `verifyCmd`'s inputs read one: UTF-8, then `Flock.JsonParse.parse`. -/
def jsonOf (what : String) (b : ByteArray) : Except String Json :=
  match String.fromUTF8? b with
  | none => .error s!"{what}: not UTF-8"
  | some t => (Flock.JsonParse.parse t).mapError fun e => s!"{what}: not JSON: {e}"

/-- What the setup reads of a parsed statement: the tags `--statement` names (`stmtTags`, before `--coins`), the
verifier's own registered reads, and C's `Readable.Read`. -/
structure Ins where
  stmtTags : Tags
  registered : Option (Array Registered.Port)
  read : Readable.Read

/-- **The verifier's parsers on the statement's bytes**, as `verifyCmd` runs them. -/
def Raw.ins (w : Raw) : Except String Ins := do
  let some tags := Tags.byName? w.statement | throw "unknown statement"
  if w.zk && tags.plainLeaves then throw s!"verify --zk: refused: {tags.name}, a statement with plain SHA-512 leaves"
  if w.zk && w.osCoins then throw "verify --zk: a --zk session's coins are its keyed coin tree, so --coins does not apply"
  let coins : Coins := if w.zk then .tree (Zk.coinSpec (tags.tableNames w.count)) else if w.osCoins then .os else .seed
  let draw := (← w.draw.mapM (jsonOf "draw")).map Flock.norm
  let partition ← w.partition.mapM (jsonOf "partition")
  let table ← w.workTable.mapM fun b => do Draw.workTableOfJson (← jsonOf "work table" b)
  let closure ← w.closure.mapM fun b => do Draw.closureMapOfJson (← jsonOf "closure map" b)
  let registered ← w.registered.mapM fun b => do Registered.ofJson (← jsonOf "registered reads" b)
  let work := match w.work, table with
    | some k, some t => some (k, t)
    | _, _ => none
  pure { stmtTags := tags, registered, read := {
    I := { tags := tags.withCoins coins, circuitFile := w.circuit, publicFile := w.publicFile, coins, count := w.count,
           tables := w.tables, partition, program := w.program, work, stratified := w.stratified, closure }
    zk := w.zk, draw, publicInputs := w.publicInputs, registered := registered.isSome } }

/-- `checkIdentifier` without `--identifier`: a public header that names an identifier is refused. -/
def noIdentifier (st : Stmt) : V Unit :=
  match st.pub.header.getObjVal? "identifier" with
  | .error _ => pure ()
  | .ok _ => throw "identifier: the public header names an identifier; pass the verifier's own (--identifier F)"

/-- **`verifyCmd`'s setup** at the statement's draw, with no recursive input and no `--identifier`. -/
def Ins.setup (s : Ins) : V (SessionSpec × Array Stmt) := do
  let r := s.read
  if !r.tags.hm96Rows then
    throw s!"{r.tags.name}: refused: a statement without hm96-sha512 rows has no draw, and every claim is about a drawn session"
  RecSetup.checkInner none none (pure (none, none))
  let (spec, sts) ← r.setup
  for st in sts do
    checkRegistered st s.registered
    noIdentifier st
  Registered.checkFull s.registered r.draw
  RecSetup.checks none none s.stmtTags r.I.tables sts r.I.circuitFile r.I.partition r.I.program s.registered none none
    none
  if r.zk then sts.forM Zk.shapeOk
  sts.forM (RecCoins.check · s.registered (RecCoins.Source.atC3 none none none) (RecSetup.c3 none none))
  sts.forM (RecVStar.checkPubRows · s.registered none none)
  if r.draw.isSome then ProvedScope.drawLaw r.zk r.tags false r.draw
  pure (spec, sts)

/-- The statement's inputs, for the verifier that is masked exactly when `zk`: a statement of the other verifier does not
parse. -/
def insOf (zk : Bool) (x : Claims.Msg) : Option Ins := do
  let w ← decode x
  if w.zk ≠ zk then none
  w.ins.toOption

/-- The statement's inputs and its setup. -/
def setupOf (zk : Bool) (x : Claims.Msg) : Option (Ins × SessionSpec × Array Stmt) := do
  let s ← insOf zk x
  let (spec, sts) ← s.setup.toOption
  pure (s, spec, sts)

/-- The setup's session is the verifier's statement's (`Read.setup`, `Stmt.setupTables`). -/
theorem Ins.setup_read {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    s.read.setup = .ok (spec, sts) := by
  unfold Ins.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

theorem setupOf_some {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupOf zk x = some (s, spec, sts)) : insOf zk x = some s ∧ s.setup = .ok (spec, sts) := by
  unfold setupOf at h
  cases hi : insOf zk x with
  | none => simp [hi] at h
  | some s' =>
    cases hsu : s'.setup with
    | error e => simp [hi, hsu, Except.toOption] at h
    | ok p =>
      simp [hi, hsu, Except.toOption] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact ⟨rfl, hsu⟩

/-- What the verifier reads of the statement's bytes (`Readable.Read`). -/
def readOf (zk : Bool) (x : Claims.Msg) : Option Readable.Read := (insOf zk x).map Ins.read

theorem Raw.ins_zk {w : Raw} {s : Ins} (h : w.ins = .ok s) : s.read.zk = w.zk := by
  unfold Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; rfl)

/-- **The statement's draw is rebuilt** (`norm`) from the JSON its bytes parse to. -/
theorem Raw.ins_draw {w : Raw} {s : Ins} (h : w.ins = .ok s) :
    ∃ draw, w.draw.mapM (jsonOf "draw") = .ok draw ∧ s.read.draw = draw.map Flock.norm := by
  unfold Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨_, by assumption, rfl⟩)

/-- A statement of the verifier that is masked exactly when `zk` is read as masked exactly when `zk`. -/
theorem readOf_zk {zk : Bool} {x : Claims.Msg} {r : Readable.Read} (h : readOf zk x = some r) : r.zk = zk := by
  unfold readOf insOf at h
  obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.1 h
  obtain ⟨w, -, hw⟩ := Option.bind_eq_some_iff.1 hs
  split_ifs at hw with hz
  · cases hw
  change w.ins.toOption = some s at hw
  rw [← not_not.1 hz]
  cases hi : w.ins with
  | error e => rw [hi] at hw; cases hw
  | ok s' =>
    rw [hi] at hw
    cases hw
    exact Raw.ins_zk hi

/-- **A statement's draw is rebuilt** (`Flock.norm`, `Raw.ins_draw`). -/
theorem insOf_draw_norm {zk : Bool} {x : Claims.Msg} {s : Ins} (h : insOf zk x = some s) {d : Json}
    (hd : s.read.draw = some d) : Flock.norm d = d ∧ Claims.Json.Wf d := by
  unfold insOf at h
  obtain ⟨w, -, hw⟩ := Option.bind_eq_some_iff.1 h
  split_ifs at hw with hz
  · cases hw
  change w.ins.toOption = some s at hw
  cases hi : w.ins with
  | error e => rw [hi] at hw; cases hw
  | ok s' =>
    rw [hi] at hw
    cases hw
    obtain ⟨draw, -, hdr⟩ := Raw.ins_draw hi
    rw [hdr] at hd
    obtain ⟨d₀, -, rfl⟩ := Option.map_eq_some_iff.1 hd
    exact ⟨norm_norm d₀, wf_norm d₀⟩

/-- **The claims' parser**: C's `parseOf` at the verifier's reading of the bytes: `Read.parse` (the verifier's setup of
the session's statement), read as `Claims.Flock.Stmt` (`toReadable`, with the relation's fields `rel`). -/
def parse (zk : Bool) (rel : Readable.Parsed → Readable.Relation) : Claims.Msg → Option Claims.Flock.Stmt :=
  Readable.parseOf rel (readOf zk)

/-- **Each table's own statement**: C's `tableOf` at the verifier's reading of the bytes, table `j` of the parsed
statement alone (`Readable.Parsed.only`), read as `Claims.Flock.Stmt`. -/
def table (zk : Bool) (rel : Readable.Parsed → Readable.Relation) : Claims.Msg → ℕ → Option Claims.Flock.Stmt :=
  Readable.tableOf rel (readOf zk)

/-- **The claims' scope check**: C's `scopeCheckOf`, the statement parses and passes the verifier's own scope refusal
(`Readable.Parsed.scope`: `ProvedScope.check`, then `ProvedScope.sizesOk`). -/
def scopeCheck (zk : Bool) : Claims.Msg → Bool := Readable.scopeCheckOf (readOf zk)

end FlockDeploy
