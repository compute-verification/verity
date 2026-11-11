module

public import Proofs.Properties.Nci.DecodeText
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

/-! ### Controls: a recursion by well-founded order that reads no string, and the domain's leaves -/

/-- A definition by well-founded recursion on a number. -/
def halfProbe (n : Nat) : Nat := if h : n < 2 then 0 else 1 + halfProbe (n - 2)
termination_by n
decreasing_by omega

theorem stop_wf : halfProbe 4 = 2 := by decide +kernel

theorem stop_domainNull : (domainError .null).isNone = true := by decide +kernel

theorem stop_domainNum : (domainError (.num ⟨1, 0⟩)).isNone = true := by decide +kernel

theorem stop_recordPairs : (recordPairs "Root_v1").isEmpty = true := by decide +kernel

/-! ### The same answers by the definitions' equations -/

theorem eq_domainNull : domainError .null = none := by simp [domainError]

theorem eq_domainLeaf : domainError (.str "a") = none := by simp [domainError]

theorem eq_domainLeafKernel : domainError (.str "a") = none := by
  rw [domainError]
  decide +kernel

theorem eq_domainArr : domainError (.arr #[.str "a", .null]) = none := by simp [domainError]

theorem eq_ty : (Ty.ofJson (.arr #[.str "Tuple", .arr #[]])).toOption.isSome = true := by simp [Ty.ofJson]

/-! ### Controls: a recursion by well-founded order on a parsed text, written here -/

/-- The depth of a parsed text's arrays, by well-founded recursion on the text. -/
def depthProbe (j : Json) : Nat :=
  match j with
  | .arr a => (Array.range a.size).attach.foldl
    (fun n i => max n (1 + depthProbe (a[i.1]'(Array.mem_range.1 i.2)))) 0
  | _ => 0
termination_by j
decreasing_by exact sizeOf_lt_of_mem_arr (Array.getElem_mem _)

/-- The same with one more argument, as `domainError` has its path. -/
def depthProbe2 (j : Json) (k : Nat := 0) : Nat :=
  match j with
  | .arr a => (Array.range a.size).attach.foldl
    (fun n i => max n (depthProbe2 (a[i.1]'(Array.mem_range.1 i.2)) (k + 1))) k
  | _ => k
termination_by j
decreasing_by exact sizeOf_lt_of_mem_arr (Array.getElem_mem _)

theorem stop_sizeOf : sizeOf Json.null = 1 := by decide +kernel

theorem stop_wfJson : depthProbe .null = 0 := by decide +kernel

theorem stop_wfJsonArr : depthProbe (.arr #[.null]) = 1 := by decide +kernel

theorem stop_wfJson2 : depthProbe2 .null = 0 := by decide +kernel

theorem eq_tyPure : (Ty.ofJson (.arr #[.str "Tuple", .arr #[]])).toOption.isSome = true := by
  simp [Ty.ofJson, pure, Except.pure, Except.toOption]

/-! ### Control: the same recursion, its order stated as the text's size, a number -/

/-- `depthProbe`, by well-founded recursion on the text's size. -/
def depthProbe3 (j : Json) : Nat :=
  match j with
  | .arr a => (Array.range a.size).attach.foldl
    (fun n i => max n (1 + depthProbe3 (a[i.1]'(Array.mem_range.1 i.2)))) 0
  | _ => 0
termination_by sizeOf j
decreasing_by exact sizeOf_lt_of_mem_arr (Array.getElem_mem _)

theorem stop_wfSize : depthProbe3 .null = 0 := by decide +kernel

theorem stop_wfSizeArr : depthProbe3 (.arr #[.null]) = 1 := by decide +kernel

theorem stop_wfSizeStr : depthProbe3 (.arr #[.str "a", .arr #[.null]]) = 2 := by decide +kernel

/-! ### The decoder's two bodies, their order stated as the text's size -/

/-- `domainError`'s lines, by well-founded recursion on the text's size. -/
def domainErrorS (j : Json) (path : String := "$") : Option String :=
  match j with
  | .null | .bool _ => none
  | .num n =>
    if n.exponent != 0 then some s!"{path}: a float is not a JSON integer, string, boolean, null, array or object"
    else if n.mantissa.natAbs > MAX_SAFE_INT then some s!"{path}: integer {n.mantissa} exceeds 2^53 - 1 in magnitude"
    else none
  | .str s => if s.toList.all (fun c : Char => c.toNat < 0x7f) then none
    else some s!"{path}: string is not ASCII without DEL"
  | .arr a => (Array.range a.size).attach.findSome? fun i =>
    domainErrorS (a[i.1]'(Array.mem_range.1 i.2)) s!"{path}[{i.1}]"
  | .obj kvs => kvs.toArray.attach.findSome? fun x =>
    if x.1.1.toList.all (fun c : Char => c.toNat < 0x7f) then domainErrorS x.1.2 s!"{path}.{x.1.1}"
    else some s!"{path}.<key>: string is not ASCII without DEL"
termination_by sizeOf j
decreasing_by
  · exact sizeOf_lt_of_mem_arr (Array.getElem_mem _)
  · exact sizeOf_lt_of_mem_obj x.2

/-- `Ty.ofJson`'s lines, by well-founded recursion on the text's size. -/
def tyOfJsonS (j : Json) : V Ty := do
  let bad := fun (why : String) => (throw s!"type {j.compress}: {why}" : V Ty)
  match j with
  | .arr a =>
    match (a[0]? : Option Json) with
    | some (.str "Value") =>
      match (a[1]? : Option Json).bind natJ?, a.size with
      | some w, 2 => if w > 0 then pure (.value w) else bad "Value<w> needs a positive int width"
      | _, _ => bad "Value<w> needs a positive int width"
    | some (.str "Array") =>
      match (a[1]? : Option Json).bind natJ?, h : (a[2]? : Option Json), a.size with
      | some n, some e, 3 => pure (.array n (← tyOfJsonS e))
      | _, _, _ => bad "Array<n,T> needs a non-negative int length and an element type"
    | some (.str "Tuple") =>
      match h : (a[1]? : Option Json), a.size with
      | some (.arr fs), 2 => pure (.tuple (← fs.attach.mapM fun f => tyOfJsonS f.1))
      | _, _ => bad "a Tuple's fields are a list"
    | some (.str "Record") =>
      match h : (a[1]? : Option Json), a.size with
      | some (.arr fs), 2 => do
        let fs ← fs.attach.mapM fun f => match hf : f.1 with
          | .arr ys =>
            if hy : ys.size = 2 then
              match ys[0] with
              | .str k => do pure (k, ← tyOfJsonS ys[1])
              | _ => throw s!"type {j.compress}: a Record field is [name, type]"
            else throw s!"type {j.compress}: a Record field is [name, type]"
          | _ => throw s!"type {j.compress}: a Record field is [name, type]"
        if (List.range fs.size).any (fun i => (fs.extract 0 i).any (·.1 == fs[i]!.1)) then
          throw s!"type {j.compress}: duplicate record field names"
        pure (.record fs)
      | _, _ => bad "a Record's fields are a list"
    | _ => bad "unknown type tag"
  | _ => bad "not a list"
termination_by sizeOf j
decreasing_by
  · exact sizeOf_lt_of_getElem?_arr h
  · exact Nat.lt_trans (sizeOf_lt_of_mem_arr f.2) (sizeOf_lt_of_getElem?_arr h)
  · have h1 := sizeOf_lt_of_mem_arr (Array.getElem_mem (xs := ys) (i := 1) (by omega))
    rw [← hf] at h1
    exact Nat.lt_trans h1 (Nat.lt_trans (sizeOf_lt_of_mem_arr f.2) (sizeOf_lt_of_getElem?_arr h))

theorem stop_domainS : (domainErrorS docProbe).isNone = true := by decide +kernel

theorem stop_domainSLeaf : (domainErrorS (.str "a")).isNone = true := by decide +kernel

theorem stop_tyS : (tyOfJsonS (.arr #[.str "Tuple", .arr #[]])).toOption.isSome = true := by decide +kernel

theorem stop_tySValue : (tyOfJsonS (.arr #[.str "Value", .num ⟨16, 0⟩])).toOption.isSome = true := by
  decide +kernel

/-! ### What `recordPairs` and `composite` read of an id -/

theorem stop_contains : ("Root_v1".contains '{') = false := by decide +kernel

theorem stop_endsWith : ("Root_v1".endsWith "}") = false := by decide +kernel

theorem stop_toList : "Root_v1".toList.length = 7 := by decide +kernel

theorem stop_splitOn : ("Root_v1".splitOn "_v").length = 2 := by decide +kernel

theorem stop_toNat : "1".toNat! = 1 := by decide +kernel

end Flock.Program.Closed
