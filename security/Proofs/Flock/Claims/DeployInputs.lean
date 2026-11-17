module

public import Flock.RecSetup
public import Proofs.Flock.Claims.Complete.Json.Value
public import Proofs.Flock.Claims.JsonNorm
public import Proofs.Flock.Claims.Scope.Matches
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import VCVio.Prelude.Core

/-! The deployed verifier's statement: its canonical bytes, its parser and its setup. -/

@[expose] public section

namespace FlockDeploy

open Flock
open Lean (Json)

/-- **What `verity-verify-proof verify` reads of a statement**, as bytes and flags: the `--statement` id, the circuit and public
files, the lookup tables the circuit pins, `--zk`, whether it names `--coins os` (which it refuses), `--session-tables`, the sampled session's sample (JSON text),
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

theorem decode_encode {x : Claims.Msg} {w : Raw} (h : decode x = some w) : w.encode = x := by
  unfold decode at h
  split at h
  · split_ifs at h with he
    cases h
    exact he
  · cases h

theorem decode_injective {x y : Claims.Msg} {w : Raw} (hx : decode x = some w) (hy : decode y = some w) : x = y :=
  (decode_encode hx).symm.trans (decode_encode hy)

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
  if w.osCoins then throw "verify: --coins os: the verifier reads coins only from a challenger's log (--coins is tree)"
  let coins : Coins := if w.zk then .tree (Zk.coinSpec (tags.tableNames w.count)) else .seed
  let draw := (← w.draw.mapM (jsonOf "draw")).map Flock.norm
  let partition ← w.partition.mapM (jsonOf "partition")
  let table ← w.workTable.mapM fun b => do Draw.workTableOfJson (← jsonOf "work table" b)
  let closure ← w.closure.mapM fun b => do Draw.closureMapOfJson (← jsonOf "closure map" b)
  let registered ← w.registered.mapM fun b => do Registered.ofJson (← jsonOf "registered reads" b)
  let work := match w.work, table with
    | some k, some t => some (k, t)
    | _, _ => none
  pure { stmtTags := tags, registered, read := {
    I := { tags, circuitFile := w.circuit, publicFile := w.publicFile, coins, count := w.count,
           tables := w.tables, partition, program := w.program, work, stratified := w.stratified, closure }
    zk := w.zk, draw, publicInputs := w.publicInputs, registered := registered.isSome } }

/-- `checkIdentifier` without `--identifier`: a public header that names an identifier is refused. -/
def noIdentifier (st : Stmt) : V Unit :=
  match st.pub.header.getObjVal? "identifier" with
  | .error _ => pure ()
  | .ok _ => throw "identifier: the public header names an identifier; pass the verifier's own (--identifier F)"

/-- **`verifyCmd`'s setup** at the statement's sample, with no recursive input and no `--identifier`. -/
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
  if r.draw.isSome then ProvedScope.sampleDesign r.zk r.tags false r.draw
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

theorem Raw.ins_sample {w : Raw} {s : Ins} (h : w.ins = .ok s) :
    ∃ draw, w.draw.mapM (jsonOf "draw") = .ok draw ∧ s.read.draw = draw.map Flock.norm := by
  unfold Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨_, by assumption, rfl⟩)

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

theorem insOf_sample_norm {zk : Bool} {x : Claims.Msg} {s : Ins} (h : insOf zk x = some s) {d : Json}
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
    obtain ⟨draw, -, hdr⟩ := Raw.ins_sample hi
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
