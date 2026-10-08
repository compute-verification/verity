/-
The partition as a named query over the verifier's own program (`one_stage/partition.py`): core's `verity/partition/v1`
object and its digest (C-Flock's `HmRow.checkPartitionObject` and `HmRow.partitionDigest`), and the template queries
applied to the program by C-Flock's `TemplateQuery`. The verifier holds its own copy of the program and applies the query
itself; it never takes a population or a unit index on the prover's word.
-/
import Flock.HmRow
import Flock.TemplateQuery
import Flock.Draw
import SampledProofs.Json

namespace SampledProofs.Partition

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (Desc TRange)

/-- `partition_object.QUERIES`, the queries a partition may name. C-Flock's form check leaves the names open. -/
def QUERIES : List (String × Nat) :=
  [("Q_word", 1), ("Q_word", 2), Flock.HmRow.Q_CALL, ("Q_template_instance", 0), ("Q_template_instances", 0)]

/-- `partition_object.validate`: C-Flock's form check, narrowed to core's queries, with `Q_word`'s `X` and `W` at most
`PARAM_MAX` as core's are. -/
def valid (obj : Json) : Bool :=
  match Flock.HmRow.checkPartitionObject obj with
  | .error _ => false
  | .ok (name, version) =>
    let params := getD (getD obj "query") "params"
    QUERIES.contains (name, version) &&
      (name != "Q_word" || ["X", "W"].all fun k => (nat? (getD params k)).any (· ≤ Flock.Qcall.PARAM_MAX))

/-- `partition_object.query()`, the query `build` falls back to for an empty one (Python's `q or query()`). -/
def defaultQuery : Json :=
  Json.mkObj [("name", "Q_word"), ("version", Lean.toJson (1 : Nat)),
    ("params", Json.mkObj [("X", Lean.toJson (16 : Nat)), ("W", Lean.toJson (32 : Nat))])]

/-- `partition_object.build`: the object naming the partition of the program (its SHA-512, hex) as the query. -/
def build (program : String) (query : Json) : Json :=
  Json.mkObj [("format", Flock.HmRow.PARTITION_V1), ("program", program),
    ("query", if keys? query == some #[] then defaultQuery else query)]

/-- `partition.digest`: core's digest of a valid object, or of a pre-canonical `{program, query}` (the envelope's rule);
`none` for anything else (Python's `ValueError`). -/
def digest (obj : Json) : Option String :=
  if keys? obj == some #["program", "query"] || valid obj then
    (Flock.HmRow.partitionDigest obj).toOption.map Flock.hexOf
  else none

/-- `partition.evaluate`: the query applied to the program, once core accepts its form; `query-refused` when it is
malformed, is not a template query, or does not apply. -/
def evaluate (d : Desc) (query : Json) : V (Array TRange) := do
  let obj := build (Flock.hexOf d.sha512) query
  if !valid obj then throw "query-refused: not a verity/partition/v1 query"
  let q := getD obj "query"
  let name := (q.getObjValAs? String "name").toOption.getD ""
  let version := (nat? (getD q "version")).getD 0
  match Flock.TemplateQuery.templatesOf name version (getD q "params") with
  | none => throw s!"query-refused: one-stage applies the template queries, not {name} v{version}"
  | some ts => Flock.TemplateQuery.evaluate d (← ts)

/-- `partition.units`: the population. -/
def units (d : Desc) (query : Json) : V Nat := return Flock.TemplateQuery.population (← evaluate d query)

/-- `partition.strata`: one stratum per template, its units as ascending, disjoint ranges, in the order of their first unit. -/
def strata (d : Desc) (query : Json) : V (Array (String × Array (Nat × Nat))) := do
  return Flock.Draw.strataOf ((← evaluate d query).map fun r => (r.template, r.first, r.count))

end SampledProofs.Partition
