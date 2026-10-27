/-
Canonical JSON: compact, object keys sorted bytewise, serde_json's string escaping (the form `compact-sorted-JSON` of
PROTOCOL.md §16 hashes). Only integers are accepted as numbers, and only strings and keys of ASCII without DEL (every
character below `0x7f`, the domain on which this form is RFC 8785's and Python's `codec.canonical_json` writes the same
bytes); anything else is refused rather than guessed.
-/
import Lean.Data.Json
import Std.Data.TreeMap.Raw.Lemmas
import Flock.Bytes

namespace Flock

open Lean (Json JsonNumber)

private def hex4 (n : Nat) : String :=
  let d := fun (k : Nat) => let x := (n >>> (4 * k)) % 16; if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)
  String.ofList [d 3, d 2, d 1, d 0]

def canonStr (s : String) : String := Id.run do
  let mut out := "\""
  for c in s.toList do
    out := match c with
      | '"' => out ++ "\\\""
      | '\\' => out ++ "\\\\"
      | '\n' => out ++ "\\n"
      | '\r' => out ++ "\\r"
      | '\t' => out ++ "\\t"
      | c => if c.toNat == 8 then out ++ "\\b" else if c.toNat == 12 then out ++ "\\f"
             else if c.toNat < 0x20 then out ++ "\\u" ++ hex4 c.toNat else out.push c
  return out ++ "\""

/-- ASCII without DEL: every character below `0x7f`, the strings and keys `canon` prints. `canonStr` itself prints any
string, as a message's free text needs. -/
def ascii (s : String) : Bool := s.toList.all fun c => c.toNat < 0x7f

def notAscii : String := "canonical JSON: a string outside ASCII without DEL (a character at or above 0x7f)"

/-- An object's values are smaller than it: `canon`'s termination on objects. -/
theorem sizeOf_lt_of_mem_toArray {kvs : Std.TreeMap.Raw String Json} {p : String × Json}
    (h : p ∈ kvs.toArray) : sizeOf p.2 < sizeOf (Json.obj kvs) := by
  rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray] at h
  obtain ⟨⟨t⟩⟩ := kvs
  simp only [Std.TreeMap.Raw.toList, Std.DTreeMap.Raw.Const.toList,
    Std.DTreeMap.Internal.Impl.Const.toList_eq_toListModel_map, List.mem_map] at h
  obtain ⟨q, hm, he⟩ := h
  have hv : q.2 = p.2 := congrArg Prod.snd he
  suffices hs : ∀ t : Std.DTreeMap.Internal.Impl String (fun _ => Json), q ∈ t.toListModel → sizeOf q.2 < sizeOf t by
    have := hs t hm
    rw [hv] at this
    simp only [Json.obj.sizeOf_spec, Std.TreeMap.Raw.mk.sizeOf_spec, Std.DTreeMap.Raw.mk.sizeOf_spec]
    omega
  intro t ht
  induction t with
  | leaf => simp at ht
  | inner sz k₀ v₀ l r ihl ihr =>
    simp only [Std.DTreeMap.Internal.Impl.toListModel_inner, List.mem_append, List.mem_cons] at ht
    simp only [Std.DTreeMap.Internal.Impl.inner.sizeOf_spec]
    rcases ht with ht | ht | ht
    · have := ihl ht; omega
    · subst ht; simp only [sizeOf]; omega
    · have := ihr ht; omega

/-- Total by `sizeOf`: an array's elements and an object's values are smaller than it. An object's keys are checked and
its values printed in the tree's order, and the pairs then sorted by key: `qsort` compares keys only, so it puts the
keys in the order it always did. An error is the first in the tree's order, so it depends on the value alone. -/
def canon : Json → V String
  | Json.null => pure "null"
  | Json.bool b => pure (if b then "true" else "false")
  | Json.num n => if n.exponent == 0 then pure (toString n.mantissa) else throw "canonical JSON: a non-integer number"
  | Json.str s => if ascii s then pure (canonStr s) else throw notAscii
  | Json.arr a => do
    let parts ← a.attach.mapM fun x => canon x.1
    pure ("[" ++ ",".intercalate parts.toList ++ "]")
  | Json.obj kvs => do
    let vals ← kvs.toArray.attach.mapM fun p =>
      if ascii p.1.1 then do pure (p.1.1, ← canon p.1.2) else throw notAscii
    let items := vals.qsort (fun a b => a.1 < b.1)
    pure ("{" ++ ",".intercalate (items.toList.map fun kv => canonStr kv.1 ++ ":" ++ kv.2) ++ "}")
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem x.2
    simp only [Json.arr.sizeOf_spec]
    omega
  · exact sizeOf_lt_of_mem_toArray p.2

/-- `canon`'s form at any string: the line a difftest server or a command writes, whose free text (a refusal's reason)
may be any string. No record is digested in it; it refuses only a non-integer number. -/
def canonLine : Json → V String
  | Json.null => pure "null"
  | Json.bool b => pure (if b then "true" else "false")
  | Json.num n => if n.exponent == 0 then pure (toString n.mantissa) else throw "canonical JSON: a non-integer number"
  | Json.str s => pure (canonStr s)
  | Json.arr a => do
    let parts ← a.attach.mapM fun x => canonLine x.1
    pure ("[" ++ ",".intercalate parts.toList ++ "]")
  | Json.obj kvs => do
    let vals ← kvs.toArray.attach.mapM fun p => do pure (p.1.1, ← canonLine p.1.2)
    let items := vals.qsort (fun a b => a.1 < b.1)
    pure ("{" ++ ",".intercalate (items.toList.map fun kv => canonStr kv.1 ++ ":" ++ kv.2) ++ "}")
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem x.2
    simp only [Json.arr.sizeOf_spec]
    omega
  · exact sizeOf_lt_of_mem_toArray p.2

/- Total accessors over parsed JSON; every failure names the field. -/
namespace J

def get (j : Json) (k : String) (ctx : String) : V Json :=
  match j.getObjVal? k with
  | .ok v => .ok v
  | .error _ => .error s!"{ctx}: {k} missing"

def has (j : Json) (k : String) : Bool :=
  match j.getObjVal? k with
  | .ok _ => true
  | .error _ => false

def asStr (j : Json) (ctx : String) : V String :=
  match j.getStr? with
  | .ok s => .ok s
  | .error _ => .error s!"{ctx}: not a string"

def asNat (j : Json) (ctx : String) : V Nat :=
  match j.getNat? with
  | .ok s => .ok s
  | .error _ => .error s!"{ctx}: not a natural number"

def asInt (j : Json) (ctx : String) : V Int :=
  match j.getInt? with
  | .ok s => .ok s
  | .error _ => .error s!"{ctx}: not an integer"

def asArr (j : Json) (ctx : String) : V (Array Json) :=
  match j.getArr? with
  | .ok s => .ok s
  | .error _ => .error s!"{ctx}: not a list"

def asBool (j : Json) (ctx : String) : V Bool :=
  match j.getBool? with
  | .ok s => .ok s
  | .error _ => .error s!"{ctx}: not a boolean"

def str (j : Json) (k ctx : String) : V String := do asStr (← get j k ctx) s!"{ctx}.{k}"
def nat (j : Json) (k ctx : String) : V Nat := do asNat (← get j k ctx) s!"{ctx}.{k}"
def arr (j : Json) (k ctx : String) : V (Array Json) := do asArr (← get j k ctx) s!"{ctx}.{k}"
def bool (j : Json) (k ctx : String) : V Bool := do asBool (← get j k ctx) s!"{ctx}.{k}"

end J

end Flock
