import PouwWindow.OwnPorts
import PouwWindow.FullDraw
import Flock.KnownRead
import SampledProofs.Produced
import SampledProofs.Record

/-!
# A drawn H1 leaf's registered files, from the registration

The verifier's `--registered` file of each of a drawn leaf's statements, from the accepted registration record and the
accepted produced record alone:

* `roots?`: each value's registered port (`leaf_lanes.verifier_entries`). A registered value's (`a`, `b`, `zy`,
  `receipt`) is its entry in the registration's `registered` (the first of its value) at that entry's root. The
  partials' is the verifier's own entry (`produced.entry`: `lanes` rows of `words` 16-bit words, under the domain
  `SampledProofs.Produced.domain` binds to the registration's program and digest) at the root of the produced record's
  `partials`. Every value's rows are `schema`'s.
* `want?`: statement `k`'s data ports at those roots and the unit's placement `place` (`ownOf`: each value's table and
  base row), where the verifier holds every table they read.
* `fileOf?`: statement `k`'s file (`leaf_lanes.registered_file` at `leaf_lanes.reads`): those ports, each port
  `readFull` names (A, B and partials rows, by layout value) marked `read_in_full`, then the known ports.
* `leafOwn?`: the leaf verifier's check of a statement's file against the two records (`pouw-leaf-own`): the file parsed
  as flock-verify parses it (`Flock.Registered.ofJson`) has one read at each port (`checkOneRead`) and holds `want?`'s
  ports (`checkOwn`). `leafOwnCmd` runs it on files with each value its own table (`selfPlace`), then holds each of the
  statement's session records to the full draw (`checkFull`); `PouwCheck.LeafOwn` runs it at a window unit's placement,
  from the window's descriptor and round log.

`tests/test_reg_file.py` holds `fileOf?` to the Python's file at the Python's entries and roots, and holds `leafOwn?` to
accept it.
-/

namespace PouwWindow.Lanes

open Lean (Json ToJson toJson)

/-- The value whose rows are the lanes' partials (`leaf_lanes.PARTIALS`). -/
def PARTIALS : String := "partials"

/-- The 64 bytes of a lowercase hex string. -/
def bytes64 (s what : String) : Except String ByteArray := do
  let b ← Flock.unhex s
  if b.size != 64 then throw s!"{what} is 64 bytes"
  pure b

/-- One value's registered port from its entry (`{"leaves", "words", "word_bits"?, "domain"}`, a registration's or
`produced.entry`'s) and its root, its rows `schema`'s: `produced.port` before its reads. -/
def valuePort (value : String) (e : Json) (root schema : String) : Except String Flock.Registered.Port := do
  let ctx := s!"the entry of {value}"
  let leaves ← Flock.J.nat e "leaves" ctx
  let words ← Flock.J.nat e "words" ctx
  let wordBits ← if Flock.J.has e "word_bits" then Flock.J.nat e "word_bits" ctx else pure 16
  let domain ← bytes64 (← Flock.J.str e "domain" ctx) s!"{value}'s domain"
  pure { name := value, value, root := ← bytes64 root s!"{value}'s root", leaves, words := some words, wordBits, domain,
         readsRel := #[], schema }

/-- **The verifier's roots** (`leaf_lanes.verifier_entries`): the partials' port, from the verifier's own entry
(`lanes` rows of `words` 16-bit words under the registration's digest) at the produced record's root of `partials`, then
each registered value's, from its entry in the registration's `registered` at that entry's root. A lookup finds the
partials first, so a registered value of that name is not read. -/
def roots? (registration produced : Json) (lanes words : Nat) (schema : String) :
    Except String (List (String × Flock.Registered.Port)) := do
  let program ← Flock.J.str registration "program" "the registration"
  let digest ← Flock.WindowRecord.digestHex registration
  let dom ← SampledProofs.Produced.domain program PARTIALS words lanes 16 digest
  let some p := (← Flock.J.arr produced "produced" "the produced record").find? fun e =>
      (e.getObjValAs? String "value").toOption == some PARTIALS
    | throw s!"the produced record has no {PARTIALS}"
  let mine := Json.mkObj [("leaves", toJson lanes), ("words", toJson words), ("word_bits", toJson (16 : Nat)),
    ("domain", Json.str (Flock.hexOf dom))]
  let part ← valuePort PARTIALS mine (← Flock.J.str p "root" "the produced partials") schema
  let regs ← match registration.getObjVal? "registered" with
    | .ok (.arr es) => es.toList.mapM fun e => do
        let v ← Flock.J.str e "value" "a registered entry"
        pure (v, ← valuePort v e (← Flock.J.str e "root" s!"the entry of {v}") schema)
    | .ok _ => throw "the registration's registered is not a list"
    | .error _ => pure []
  pure ((PARTIALS, part) :: regs)

/-- The roots as a function: each value's first port, the default where none is held. -/
def rootsFn (rs : List (String × Flock.Registered.Port)) (v : String) : Flock.Registered.Port := (rs.lookup v).getD default

/-- **Statement `k`'s data ports at the verifier's roots** (`ownOf`, at the unit's placement `place`), where the
verifier holds every table they read. -/
def want? (rs : List (String × Flock.Registered.Port)) (place : String → String × Nat) (T : Tile) (k : Option Nat) :
    Except String (List Flock.Registered.Port) := do
  for (nm, v, _) in regReads T k do
    if (rs.lookup (place v).1).isNone then
      throw s!"port {nm} reads {(place v).1}, whose entry and root the verifier does not hold"
  pure (ownOf (rootsFn rs) place T k)

/-- A registered port's `--registered` entry (`produced.port`): its value, root, leaves, row words and leaf width,
domain, schema (named where it is not `hm96-sha512/row/v1`) and reads, and `read_in_full` where `full`. -/
def portJson (p : Flock.Registered.Port) (full : Bool) : Json :=
  Json.mkObj ([("value", Json.str p.value), ("root", Json.str (Flock.hexOf p.root)), ("leaves", toJson p.leaves)] ++
    (match p.words with | some w => [("words", toJson w)] | none => []) ++
    [("word_bits", toJson p.wordBits), ("domain", Json.str (Flock.hexOf p.domain))] ++
    (if p.schema == Flock.HmRow.ROW_SCHEMA then [] else [(Flock.Registered.SCHEMA_KEY, Json.str p.schema)]) ++
    [("reads", Json.arr (p.reads.map fun | some n => toJson n | none => Json.null))] ++
    (if full then [(Flock.KnownRead.READ_KEY, Json.bool true)] else []))

/-- **Statement `k`'s `--registered` file** (`leaf_lanes.registered_file` at `leaf_lanes.reads`): its data ports at the
verifier's roots and the unit's placement (`want?`), each port `readFull` names (a read of a `fullValue`, by its
layout value, whatever table it is placed in) marked `read_in_full`, then the known ports `known`: every port of the
statement's circuit, as flock-verify's `Registered.checkComplete` holds it to. A known port that is also a read is an error. It is a `Flock.Registered.FORMAT_V0` file, as
`leaf_lanes.registered_file` writes it (`registered.READS`): every port reads from unit 0 (`readsFrom := 0`). -/
def fileOf? (rs : List (String × Flock.Registered.Port)) (place : String → String × Nat) (T : Tile) (k : Option Nat)
    (known : List (String × Json)) : Except String Json := do
  let ps ← want? rs place T k
  if let some (nm, _) := known.find? fun (nm, _) => ps.any (·.name == nm) then
    throw s!"port {nm} is both a read and known"
  let full := readFull T k
  pure <| Json.mkObj [("format", Json.str Flock.Registered.FORMAT_V0),
    ("ports", Json.mkObj (ps.map (fun p => (p.name, portJson p (full.contains p.name))) ++ known))]

/-- **The leaf verifier's check of statement `k`'s file `file`** against the accepted registration and produced record:
the tile's statement exists (`lane?`, `combine?`), and the file, parsed as flock-verify parses it, has one read at each
port (`checkOneRead`) and holds the statement's data ports at the records' roots and the unit's placement `place`
(`checkOwn`). -/
def leafOwn? (registration produced file : Json) (place : String → String × Nat) (T : Tile) (k : Option Nat)
    (schema : String) : Except String Unit := do
  match k with
  | some l => discard (lane? T l)
  | none => discard (combine? T)
  let rs ← roots? registration produced T.lanes (2 * T.nw) schema
  let own ← Flock.Registered.ofJson file
  checkOneRead own
  checkOwn (← want? rs place T k) own

def readJson (path : String) : IO Json := do
  match Json.parse (← IO.FS.readFile path) with
  | .ok j => pure j
  | .error e => throw (IO.userError s!"{path}: {e}")

def usage (why : String) : IO UInt32 := do
  IO.eprintln why
  pure 2

/-- A statement named on the command line: a lane's index, or `combine`. -/
def stmtArg? (stmt : String) : Option (Option Nat) :=
  if stmt == "combine" then some none else stmt.toNat?.map some

/-- Each session directory's record, `SESSION/session.json`. -/
def sessionRecords (sess : List String) : IO (List String) :=
  sess.mapM fun (d : String) => IO.FS.readFile (System.FilePath.mk d / "session.json")

/-- `pouw-leaf-own`'s verdict, printed as `OWN {"accepted", "why"?}`: exit 0 where it accepts and 1 where it refuses. -/
def printOwn (verdict : Except String Unit) : IO UInt32 := do
  match verdict with
  | .ok () =>
    IO.println s!"OWN {(Json.mkObj [("accepted", Json.bool true)]).compress}"
    return 0
  | .error e =>
    IO.println s!"OWN {(Json.mkObj [("accepted", Json.bool false), ("why", Json.str e)]).compress}"
    return 1

/-- **`pouw-leaf-own REGISTRATION PRODUCED REGISTERED R C K (LANE | combine) SESSION...`**: `leafOwn?` on the three
files at the `R × C` tile of contraction `K`, each value its own table (`selfPlace`), every value's rows
`hm96-sha512/row/v2`, then `checkFull` on each session directory's record (`SESSION/session.json`, at least one). It
prints `OWN {"accepted", "why"?}` and exits 0 where it accepts, 1 where it refuses and 2 on a usage error. A unit of a
window Program, its tables placed by the window's descriptor, is `pouw-leaf-own-window`'s (`PouwCheck.LeafOwn`). -/
def leafOwnCmd (args : List String) : IO UInt32 := do
  let reg :: prod :: file :: r :: c :: k :: stmt :: sess@(_ :: _) := args
    | usage "usage: pouw-leaf-own REGISTRATION PRODUCED REGISTERED R C K (LANE | combine) SESSION..."
  let (some r, some c, some k) := (r.toNat?, c.toNat?, k.toNat?)
    | usage "R, C and K are natural numbers"
  let some lane := stmtArg? stmt
    | usage s!"the statement is a lane's index or combine, not {stmt}"
  let (reg, prod, file) := (← readJson reg, ← readJson prod, ← readJson file)
  let records ← sessionRecords sess
  printOwn do
    leafOwn? reg prod file selfPlace ⟨r, c, k⟩ lane Flock.HmRow.ROW_SCHEMA_V2
    records.forM checkFull

end PouwWindow.Lanes
