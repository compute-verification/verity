module

public import Proofs.Properties.Nci.RecordServed
import Batteries.Tactic.OpenPrivate
import Std.Data.HashMap.Lemmas
import Std.Data.HashSet.Lemmas
import all Flock.Program
import all Flock.Canon
import all Flock.JsonParse
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.Array.Attach
import all Init.WF

/-!
# Where a closed evaluation of the Program's decoder stops

A scratch file, never merged. Each `stop_*` asks the kernel to run one piece of `Flock.Program.decode` on the text of
the Program of no node (`programEmpty`), in the order `decode` reaches them; the first that the kernel refuses is where
a closed evaluation stops. `stop_hashNat` against `stop_hash`, and `stop_mapNat` against `stop_mapEmpty`, separate the
hash of a `String` from the bucket's index.
-/

namespace Flock.Program.Closed

open Lean (Json JsonNumber)
open NciCheck.Properties.Record.Attack (programEmpty)
open private getDef composite staticFns DecState DecState.defs DecState.index DecState.busy Desc Desc.types Desc.callees
  Desc.ddefs liftV at_ paramsOf buildNode natJ? from Flock.Program

/-- The parsed text of `programEmpty`, null where the parse refuses. -/
def docProbe : Json := (JsonParse.parse programEmpty).toOption.getD .null

/-- Its root's definition. -/
def rootProbe : Json :=
  ((docProbe.getObjVal? "definitions").toOption.bind fun d => (d.getObjVal? "Root_v1").toOption).getD .null

/-- The descriptor `decode` hands its walk, its definitions under their hash map. -/
def descProbe : Desc :=
  { ddefs := (∅ : Std.HashMap String Json).insert "Root_v1" rootProbe, callees := #[], types := #[] }

theorem stop_utf8 : (String.fromUTF8? programEmpty.toUTF8).isSome = true := by decide +kernel

theorem stop_parse : (JsonParse.parse programEmpty).toOption.isSome = true := by decide +kernel

theorem stop_schema : ((docProbe.getObjValAs? String "schema").toOption == some SCHEMA) = true := by decide +kernel

theorem stop_form : ((rootProbe.getObjValAs? String "form").toOption == some "composite") = true := by decide +kernel

theorem stop_domain : (domainError docProbe).isNone = true := by decide +kernel

theorem stop_domainLeaf : (domainError (.str "a")).isNone = true := by decide +kernel

theorem stop_ty : (Ty.ofJson (.arr #[.str "Tuple", .arr #[]])).toOption.isSome = true := by decide +kernel

theorem stop_tyValue : (Ty.ofJson (.arr #[.str "Value", .num ⟨16, 0⟩])).toOption.isSome = true := by decide +kernel

theorem stop_record : (recordFunctionIds "Root_v1").isEmpty = true := by decide +kernel

theorem stop_canon : (canon docProbe).toOption.isSome = true := by decide +kernel

theorem stop_hashNat : (hash (7 : Nat) == hash (7 : Nat)) = true := by decide +kernel

theorem stop_hash : (hash "Root_v1" == hash "Root_v1") = true := by decide +kernel

theorem stop_mapNat : ((∅ : Std.HashMap Nat Nat)[7]?).isNone = true := by decide +kernel

theorem stop_mapEmpty : ((∅ : Std.HashMap String Nat)["Root_v1"]?).isNone = true := by decide +kernel

theorem stop_setEmpty : (!(∅ : Std.HashSet String).contains "Root_v1") = true := by decide +kernel

theorem stop_composite :
    (composite 3 descProbe "Root_v1" rootProbe "composite" "Root_v1" {}).toOption.isSome = true := by decide +kernel

theorem stop_getDef : (getDef 3 descProbe "Root_v1" {}).toOption.isSome = true := by decide +kernel

theorem stop_decode : (decode programEmpty.toUTF8).toOption.isSome = true := by decide +kernel

end Flock.Program.Closed
