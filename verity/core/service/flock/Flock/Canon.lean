/-
Canonical JSON: compact, object keys sorted bytewise, serde_json's string escaping (the form `compact-sorted-JSON` of
PROTOCOL.md §16 hashes), with every character outside printable ASCII escaped: a control character, DEL and every
character from `0x80` on are `\uXXXX` in lowercase hex (a UTF-16 surrogate pair above `0xFFFF`), but for the short
escapes `\b \f \n \r \t`. So a canonical text has no byte at or above `0x7f`, and each value has one. Only integers are
accepted as numbers; anything else is refused rather than guessed.
-/
import Lean.Data.Json
import Std.Data.TreeMap.Raw.Lemmas
import Flock.Bytes

namespace Flock

open Lean (Json JsonNumber)

private def hex4 (n : Nat) : String :=
  let d := fun (k : Nat) => let x := (n >>> (4 * k)) % 16; if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)
  String.ofList [d 3, d 2, d 1, d 0]

/-- A character's `\u` escape: one below `0x10000`, else its UTF-16 surrogate pair (a `Char` is never a surrogate). -/
def escU (n : Nat) : String :=
  if n < 0x10000 then "\\u" ++ hex4 n
  else "\\u" ++ hex4 (0xD800 + (n - 0x10000) / 0x400) ++ "\\u" ++ hex4 (0xDC00 + (n - 0x10000) % 0x400)

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
             else if c.toNat < 0x20 || 0x7f ≤ c.toNat then out ++ escU c.toNat else out.push c
  return out ++ "\""

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

/-- Total by `sizeOf`: an array's elements and an object's values are smaller than it. An object's values are printed
in the tree's order and the pairs then sorted by key: `qsort` compares keys only, so it puts the keys in the order it
always did, and its one error is the same whichever value fails first. -/
def canon : Json → V String
  | Json.null => pure "null"
  | Json.bool b => pure (if b then "true" else "false")
  | Json.num n => if n.exponent == 0 then pure (toString n.mantissa) else throw "canonical JSON: a non-integer number"
  | Json.str s => pure (canonStr s)
  | Json.arr a => do
    let parts ← a.attach.mapM fun x => canon x.1
    pure ("[" ++ ",".intercalate parts.toList ++ "]")
  | Json.obj kvs => do
    let vals ← kvs.toArray.attach.mapM fun p => do pure (p.1.1, ← canon p.1.2)
    let items := vals.qsort (fun a b => a.1 < b.1)
    pure ("{" ++ ",".intercalate (items.toList.map fun kv => canonStr kv.1 ++ ":" ++ kv.2) ++ "}")
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem x.2
    simp only [Json.arr.sizeOf_spec]
    omega
  · exact sizeOf_lt_of_mem_toArray p.2

/-- `canonStr` with DEL and every character from `0x80` on as itself, in UTF-8: a string of `canonUtf8`. -/
def canonStrUtf8 (s : String) : String := Id.run do
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

/-- An identity manifest's bytes, which a tagged identity hash digests (`identity.canonical_json_bytes`,
`json.dumps(..., ensure_ascii=False)`): `canon`'s form with strings in UTF-8 (`canonStrUtf8`). It is not canonical JSON
(`canon`), which escapes them; it stays the form already-signed records were digested in, the auditor's ledger among
them. -/
def canonUtf8 : Json → V String
  | Json.null => pure "null"
  | Json.bool b => pure (if b then "true" else "false")
  | Json.num n => if n.exponent == 0 then pure (toString n.mantissa) else throw "canonical JSON: a non-integer number"
  | Json.str s => pure (canonStrUtf8 s)
  | Json.arr a => do
    let parts ← a.attach.mapM fun x => canonUtf8 x.1
    pure ("[" ++ ",".intercalate parts.toList ++ "]")
  | Json.obj kvs => do
    let vals ← kvs.toArray.attach.mapM fun p => do pure (p.1.1, ← canonUtf8 p.1.2)
    let items := vals.qsort (fun a b => a.1 < b.1)
    pure ("{" ++ ",".intercalate (items.toList.map fun kv => canonStrUtf8 kv.1 ++ ":" ++ kv.2) ++ "}")
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
