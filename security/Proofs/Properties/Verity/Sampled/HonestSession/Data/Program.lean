module

public import Spot.Partition
public import Spot.Program
public import Flock.HmRow

/-!
# The honest session's program, query, law and partition

The registered program: a root that reads one 16-bit input and calls one composite, `Unit_v1`, on it, whose one gate
is the primitive `Not16_v1`: the session's one proof block (`nUnits`). Its query names that composite's one instance (`Q_template_instance` v0), its law samples one unit
(`subset`, `k = 1`), and its partition object is the one `Spot.Partition.build` writes at the program's SHA-512.
Every digest is a term (`programSha`, `partitionDigestB`), never a literal, so a proof about these values compares hashes
of terms and evaluates none.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)

/-- A natural number as a JSON number. -/
def jn (n : Nat) : Json := .num ⟨n, 0⟩

/-- The population: the public file's proof blocks. -/
def nUnits : Nat := 1

/-- The template the query names: the composite each proof block lowers. -/
def templateId : String := "Unit_v1"

/-- A 16-bit value's type, `["Value",16]`. -/
def v16 : Json := .arr #[.str "Value", jn 16]

/-- The root's id. -/
def rootId : String := "Root[HsProg_v0]_v1"

/-- The descriptor's members other than `annotations`, in ascending key order. -/
def bareMembers : List (String × Json) := [
  ("callees", .arr #[.str "Not16_v1", .str "Input16_v1", .str templateId]),
  ("definitions", Json.mkObj [
    ("Input16_v1", Json.mkObj [("form", .str "primitive"), ("id", .str "Input16_v1"), ("params", .arr #[]),
      ("ret", v16)]),
    ("Not16_v1", Json.mkObj [("form", .str "primitive"), ("id", .str "Not16_v1"),
      ("params", .arr #[.arr #[.str "a", v16]]), ("ret", v16)]),
    (rootId, Json.mkObj [("bindings", Json.mkObj []),
      ("body", Json.mkObj [
        ("nodes", .arr #[
          Json.mkObj [("args", .arr #[]), ("axes", .arr #[]), ("fn", jn 1), ("form", .str "batch"), ("n", jn 1),
            ("out", jn 1)],
          Json.mkObj [("args", .arr #[Json.mkObj [("refs", Json.mkObj [("affine", .arr #[.str "n", jn 0, jn 0, jn 1])]),
              ("type", jn 0)]]), ("fn", jn 2), ("form", .str "call"), ("out", jn 0)]]),
        ("params", .arr #[]), ("ret", Json.mkObj [("affine", .arr #[.str "n", jn 1, jn 0, jn 1])]),
        ("ret_type", v16)]),
      ("form", .str "composite"), ("id", .str rootId)]),
    (templateId, Json.mkObj [("bindings", Json.mkObj []),
      ("body", Json.mkObj [
        ("nodes", .arr #[Json.mkObj [("args", .arr #[Json.mkObj [
            ("refs", Json.mkObj [("affine", .arr #[.str "p", jn 0, jn 0, jn 1])]), ("type", jn 0)]]),
          ("fn", jn 0), ("form", .str "primitive"), ("out", jn 0)]]),
        ("params", .arr #[.arr #[.str "x", v16]]),
        ("ret", Json.mkObj [("affine", .arr #[.str "n", jn 0, jn 0, jn 1])]), ("ret_type", v16)]),
      ("form", .str "composite"), ("id", .str templateId)])]),
  ("program_fn", .str "HsProg_v0"),
  ("root", .str rootId),
  ("schema", .str "verity-ir/descriptor/v1"),
  ("types", .arr #[v16, .arr #[.str "Array", jn 1, v16]])]

/-- The registered descriptor without `annotations`, the object its SHA-512 is taken of. -/
def programBare : Json := Json.mkObj bareMembers

/-- The registered descriptor: `programBare` with empty `annotations`. -/
def programJ : Json := Json.mkObj (("annotations", Json.mkObj []) :: bareMembers)

/-- The registered program file, its descriptor's canonical text. -/
def programFile : ByteArray := ((Flock.canon programJ).toOption.getD "").toUTF8

/-- The statement's descriptor bytes (`Live.desc`). -/
def desc : List UInt8 := programFile.data.toList

/-- `program_sha512`: SHA-512 of the canonical descriptor without `annotations`, as `TemplateQuery.Desc.parse` takes it. -/
def programSha : ByteArray := Flock.Sha512.hash ((Flock.canon programBare).toOption.getD "").toUTF8

/-- The program the descriptor parses to. -/
def program : Spot.Program := (Spot.Program.parse programFile).toOption.getD default

/-- The registration's query: the template's one instance. -/
def query : Json := Json.mkObj [("name", .str "Q_template_instance"),
  ("params", Json.mkObj [("template", .str templateId)]), ("version", jn 0)]

/-- The registration's sampling-design object: one unit, sampled uniformly. -/
def law : Json := Json.mkObj [("k", jn 1), ("law", .str "subset")]

/-- The partition object the verifier builds from the program and the query. -/
def partitionObj : Json := Spot.Partition.build (Flock.hexOf programSha) query

/-- The partition object's canonical text. -/
def partitionText : String := (Flock.canon partitionObj).toOption.getD ""

/-- `HmRow.partitionDigest partitionObj`, as a term. -/
def partitionDigestB : ByteArray :=
  Flock.Sha512.hash (Flock.HmRow.PARTITION_V1.toUTF8.push 0 ++ partitionText.toUTF8)

/-- The template's digest as the circuit's META `program_digests` names it: SHA-512 of its Definition's canonical entry.
`HmRow.deriveTemplateUnits` reads only the key. -/
def templateDigest : ByteArray := Flock.Sha512.hash ((Flock.canon (programBare.getObjValD "definitions" |>.getObjValD templateId)).toOption.getD "").toUTF8

end Verity.Sampled.Instance.HonestSession
