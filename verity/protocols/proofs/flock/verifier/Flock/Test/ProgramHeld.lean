module

public import Flock.ProvedScope
public import Flock.Registered
meta import Flock.ProvedScope
meta import Flock.Registered

@[expose] public section

namespace Flock.Test.ProgramHeld

/-!
The verifier's own program (`--program`) against the circuit, in every mode.

What the setup compares at a `verity/partition/v1` binding: the partition object's `program` with META's
`program_sha512` (`HmRow.checkPartition`), and the handed program with the object's `program` only where it evaluates the
query on it (`HmRow.deriveTemplateUnits` here); for a query it does not evaluate, `HmRow.qwordFinding` returns the
finding as it stands, whatever program it was handed. `ProvedScope.programBound` refuses a handed program whose digest
is not META's for every query, in `ProvedScope.programWhyP`'s words, and refuses nothing under the provisional unit
cover. It takes no mode: `verify` runs it on every session, `--zk` or plain. It and the verdict's fields read the
program's digest (`handed`), which `verify` computes once a call from its one read of the program.

What the verdict says (`ProvedScope.programFields`): `program_sha512` only for a program whose digest is META's; for any
other, its digest under `program_handed_sha512` and a `program_note`, neither of which a reader looking for
`program_sha512` finds.
-/

open Lean (Json)

/-- A descriptor `TemplateQuery.Desc.parse` reads, with no callee: one primitive Definition, its root. -/
def desc (root : String) : ByteArray :=
  ("{\"schema\":\"verity-ir/descriptor/v1\",\"callees\":[],\"definitions\":{\"" ++ root ++
    "\":{\"form\":\"primitive\"}},\"root\":\"" ++ root ++ "\"}").toUTF8

/-- The circuit's program. -/
def mine : ByteArray := desc "Mine_v1"

/-- Another computation's program. -/
def other : ByteArray := desc "Other_v1"

def sha (b : ByteArray) : String := (ProvedScope.programDigest b).getD ""

/-- The digest `verify` hands `ProvedScope.programBound` and `ProvedScope.programFields`: none without a program,
`some none` for bytes that are not a JSON descriptor. -/
def handed (program : Option ByteArray) : Option (Option String) := program.map ProvedScope.programDigest

/-- A `verity/partition/v1` object over the program of digest `program`, asking the query `name` v`version`. -/
def object (program name : String) (version : Nat) (params : Json := Json.mkObj []) : Json :=
  Json.mkObj [("format", Json.str HmRow.PARTITION_V1), ("program", Json.str program),
    ("query", Json.mkObj [("name", Json.str name), ("version", Lean.toJson version), ("params", params)])]

/-- A circuit whose META binds `program` and a partition of `rule`: `obj`'s digest, or the provisional unit cover's. -/
def circuit (program : Option String) (rule : String) (obj : Option Json := none) : Circuit :=
  let digest := match obj.map HmRow.partitionDigest with
    | some (.ok d) => hexOf d
    | _ => ""
  { (default : Circuit) with metaJ := Json.mkObj ((program.toList.map fun h => ("program_sha512", Json.str h)) ++
      [("partition", Json.mkObj [("rule", Json.str rule), ("digest", Json.str digest)])]) }

/-- A partition object over `mine` whose query the verifier does not evaluate. -/
def unread : Json := object (sha mine) "Q_other" 1

/-- A partition object over `mine` whose query is a template query. -/
def template : Json := object (sha mine) "Q_template_instance" 0 (Json.mkObj [("template", Json.str "Mine_v1")])

/-- The circuit that binds `mine` and `obj` under `verity/partition/v1`. -/
def bound (obj : Json) : Circuit := circuit (some (sha mine)) HmRow.PARTITION_V1 (some obj)

/-- A circuit under the provisional unit cover whose META names `program`. -/
def cover (program : Option String) : Circuit := circuit program HmRow.PARTITION_PROVISIONAL

def NOT_THE_OBJECTS : String := "partition: the verifier's program is not the one the partition object names"

-- two descriptors of two digests, each read as `TemplateQuery.Desc.parse` reads it and naming no registered input
#guard (sha mine).length == 128 && sha mine != sha other
#guard (TemplateQuery.Desc.parse other).toOption.map (hexOf ·.sha512) == some (sha other)
#guard (Registered.checkProgram (some other) false).toBool
#guard ProvedScope.metaRule (bound unread) == HmRow.PARTITION_V1
#guard ProvedScope.metaRule (cover none) == HmRow.PARTITION_PROVISIONAL
#guard ProvedScope.metaRule default == ""

/-! The setup: the object against META, and the handed program against the object where the query is evaluated. -/

def partition (c : Circuit) (obj : Json) : String :=
  match HmRow.checkPartition c (some obj) with
  | .error e => e
  | .ok f => f.note

#guard partition (bound unread) unread == "units taken as stated, not derived: the verifier does not evaluate Q_other v1"
#guard partition (circuit (some (sha other)) HmRow.PARTITION_V1 (some unread)) unread ==
  "partition: the partition object is over another program than META's program_sha512"

def derived (program : ByteArray) : String :=
  match HmRow.deriveTemplateUnits (bound template) template program #[] with
  | .error e => e
  | .ok note => note

#guard derived other == NOT_THE_OBJECTS
#guard derived mine != NOT_THE_OBJECTS

def stated : HmRow.PartitionFinding := { rule := HmRow.PARTITION_V1, note := "as stated" }

-- a query the verifier does not evaluate: the finding as it stands, with the circuit's program and with another
#guard [mine, other].all fun p => match HmRow.qwordFinding (bound unread) default unread p stated with
  | .ok (some f) => f.note == "as stated" && !f.derived
  | _ => false

/-! `verify`'s refusals on the program, on every session and in its order: `programHeld`, then `programBound`. -/

def why : Except String Unit → String
  | .error e => e
  | .ok () => ""

def refused (parsed : V Circuit) (program : Option ByteArray) : String :=
  why (do
    ProvedScope.programHeld program
    ProvedScope.programBound parsed (handed program))

def NOT_THE_CIRCUITS : String :=
  s!"refused: the verifier's program (--program, program_sha512 {sha other}) is not the circuit's (META program_sha512 {sha mine})"

-- under verity/partition/v1 a program that is not META's is refused whatever the query, in `programWhyP`'s words
#guard refused (.ok (bound unread)) (some other) == NOT_THE_CIRCUITS
#guard refused (.ok (bound template)) (some other) == NOT_THE_CIRCUITS
#guard ProvedScope.programWhyP (bound unread) (some other) == some NOT_THE_CIRCUITS
#guard refused (.ok (bound unread)) (some "not json".toUTF8) ==
  "refused: the verifier's program (--program) is not a JSON descriptor"
-- the circuit's program is accepted, under either query
#guard refused (.ok (bound unread)) (some mine) == ""
#guard refused (.ok (bound template)) (some mine) == ""
-- a session without the program is `programHeld`'s, and `programBound` alone passes it
#guard refused (.ok (bound unread)) none == ProvedScope.NO_PROGRAM
#guard ProvedScope.programWhyBound (bound unread) none == none
-- under the provisional unit cover, or with no program bound, nothing is compared
#guard refused (.ok (cover (some (sha mine)))) (some other) == ""
#guard refused (.ok (cover none)) (some other) == ""
#guard refused (.ok (circuit none HmRow.PARTITION_V1 (some unread))) (some other) == ""
#guard refused (.error "e") (some mine) ==
  "refused: a circuit that does not parse (e): a statement form the soundness proof does not cover"

/-! A plain session beside a `--plain-tables` one, whose own refusal (`programP`) `verify` runs first. -/

-- under verity/partition/v1 a session of any mode is held as a `--plain-tables` one is: the same programs refused, in
-- the same words
#guard [mine, other, "not json".toUTF8].all fun p =>
  why (ProvedScope.programBound (.ok (bound unread)) (handed (some p))) ==
    why (ProvedScope.programP (.ok (bound unread)) (some p))
-- under the provisional unit cover only `--plain-tables` refuses another program; any other mode's verdict says it was
-- not held
#guard why (ProvedScope.programP (.ok (cover (some (sha mine)))) (some other)) == NOT_THE_CIRCUITS
#guard why (ProvedScope.programBound (.ok (cover (some (sha mine)))) (handed (some other))) == ""

/-! The verdict's program fields. -/

def fields (parsed : V Circuit) (program : Option ByteArray) : List (String × String) :=
  (ProvedScope.programFields parsed (handed program)).map fun (k, v) => (k, (v.getStr?.toOption).getD "")

def note (why : String) : String :=
  s!"the Program was handed and not held to the circuit: {why}; the registered-read refusals (Registered.checkProgram, Registered.checkCover) read the handed Program only"

-- the circuit's program is held, under verity/partition/v1 and under the provisional unit cover
#guard fields (.ok (bound unread)) (some mine) == [("program_sha512", sha mine)]
#guard fields (.ok (cover (some (sha mine)))) (some mine) == [("program_sha512", sha mine)]
-- under the provisional unit cover another program is not refused, and the verdict names no `program_sha512`
#guard fields (.ok (cover (some (sha mine)))) (some other) == [("program_handed_sha512", sha other),
  ("program_note", note "the provisional unit cover binds no Program descriptor, and the handed Program's digest is not the one META names")]
-- so is a refused session's under verity/partition/v1
#guard fields (.ok (bound unread)) (some other) == [("program_handed_sha512", sha other),
  ("program_note", note "the handed Program's digest is not the one META names")]
#guard fields (.ok (circuit none HmRow.PARTITION_V1)) (some mine) == [("program_handed_sha512", sha mine),
  ("program_note", note "the circuit's META names no Program digest")]
#guard fields (.error "e") (some mine) == [("program_handed_sha512", sha mine),
  ("program_note", note "the circuit does not parse")]
#guard fields (.ok (bound unread)) (some "not json".toUTF8) ==
  [("program_note", note "the handed bytes (--program) are not a JSON descriptor")]
#guard fields (.ok (bound unread)) none == []
-- a reader that looks for `program_sha512` finds it in no field of a program that was not held
#guard (fields (.ok (cover (some (sha mine)))) (some other) ++ fields (.ok (bound unread)) (some other)).all
  fun (k, v) => ((k ++ " " ++ v).splitOn "program_sha512").length == 1

end Flock.Test.ProgramHeld
