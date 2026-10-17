/-
Canonical JSON: compact, object keys sorted bytewise, serde_json's string escaping (the form `compact-sorted-JSON` of
PROTOCOL.md §16 hashes). Only integers are accepted as numbers; anything else is refused rather than guessed.
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

/-- An object's values are smaller than it: `canon`'s termination on objects. -/
theorem sizeOf_lt_of_mem_toArray {kvs : Std.TreeMap.Raw String Json} {k : String} {v : Json}
    (h : (k, v) ∈ kvs.toArray) : sizeOf v < sizeOf (Json.obj kvs) := by
  rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray] at h
  obtain ⟨⟨t⟩⟩ := kvs
  simp only [Std.TreeMap.Raw.toList, Std.DTreeMap.Raw.Const.toList,
    Std.DTreeMap.Internal.Impl.Const.toList_eq_toListModel_map, List.mem_map] at h
  obtain ⟨p, hm, he⟩ := h
  have hv : p.2 = v := congrArg Prod.snd he
  suffices hs : ∀ t : Std.DTreeMap.Internal.Impl String (fun _ => Json), p ∈ t.toListModel → sizeOf p.2 < sizeOf t by
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

/-- Total by `sizeOf`: an array's elements and an object's values are smaller than it, and `attach` carries the
membership through the sort; the sort still compares keys only, so the output is the order `qsort` gives the keys. -/
def canon : Json → V String
  | Json.null => pure "null"
  | Json.bool b => pure (if b then "true" else "false")
  | Json.num n => if n.exponent == 0 then pure (toString n.mantissa) else throw "canonical JSON: a non-integer number"
  | Json.str s => pure (canonStr s)
  | Json.arr a => do
    let parts ← a.attach.mapM fun ⟨x, _⟩ => canon x
    pure ("[" ++ ",".intercalate parts.toList ++ "]")
  | Json.obj kvs => do
    let items := kvs.toArray.attach.qsort (fun a b => a.1.1 < b.1.1)
    let parts ← items.mapM fun ⟨(k, v), _⟩ => do pure (canonStr k ++ ":" ++ (← canon v))
    pure ("{" ++ ",".intercalate parts.toList ++ "}")
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem ‹_›
    simp only [Json.arr.sizeOf_spec]
    omega
  · exact sizeOf_lt_of_mem_toArray ‹_›

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
