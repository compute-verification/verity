/-
PROTOCOL.md §5.1: the session record's canonical text, the text `Json.compress` prints of what it parses to, which every
writer emits (`verity/ml/flock/live/src/record_text.rs`, `verity_flock.record_text`) and S0 holds a record to. So the
record's text is a function of what the verifier reads of it.

`printT` prints what `Json.compress` prints, by structural recursion, so proofs read its bytes (the proofs' honest printer
`FlockComplete.printJ` is the same function): `null`, `true` and `false`, a number's `toString`, a string between quotes
with `"`, `\`, newline and carriage return escaped and every other character below `0x20` as `\u` and four hex digits, and
an array's elements and an object's fields (in its tree's order, sorted by key) between commas. `Flock.Test.RecordText`
holds it to `Json.compress` on the writers' shared vectors.
-/
import Lean.Data.Json
import Flock.Bytes

namespace Flock.RecordText

open Lean (Json)
open Std.DTreeMap.Internal (Impl)

/-- A character inside a printed string. -/
def escC (c : Char) : String :=
  if c = '"' then "\\\""
  else if c = '\\' then "\\\\"
  else if c = '\n' then "\\n"
  else if c = '\x0d' then "\\r"
  else if 0x0020 ≤ c.val ∧ c.val ≤ 0x10ffff then String.singleton c
  else
    let n := c.toNat
    "\\u" ++ String.ofList [Nat.digitChar (n / 4096), Nat.digitChar ((n % 4096) / 256), Nat.digitChar ((n % 256) / 16),
      Nat.digitChar (n % 16)]

/-- A string, printed. -/
def strT (s : String) : String := "\"" ++ String.join (s.toList.map escC) ++ "\""

mutual
/-- **A JSON value, printed** as `Json.compress` prints it. -/
def printT : Json → String
  | .null => "null"
  | .bool b => toString b
  | .num n => toString n
  | .str s => strT s
  | .arr ⟨l⟩ => "[" ++ ",".intercalate (itemsT l) ++ "]"
  | .obj ⟨⟨t⟩⟩ => "{" ++ ",".intercalate (fieldsT t) ++ "}"

/-- An array's elements, printed. -/
def itemsT : List Json → List String
  | [] => []
  | j :: js => printT j :: itemsT js

/-- An object's fields, printed in its tree's order. -/
def fieldsT : Impl String (fun _ => Json) → List String
  | .leaf => []
  | .inner _ k v l r => fieldsT l ++ (strT k ++ ":" ++ printT v) :: fieldsT r
end

/-- S0's refusal. -/
def refusal : String :=
  "S0: the record is not in its canonical text (what Json.compress prints of what it parses to: PROTOCOL.md §5.1)"

/-- **S0: the record is in its canonical text**: it parses, and `printT` of what it parses to is the record. -/
def check (text : String) : V Unit :=
  match Json.parse text with
  | .ok j => if printT j == text then pure () else throw refusal
  | .error e => throw s!"record: not JSON: {e}"

/-- **S0's soundness**: a record S0 accepts is what `printT` prints of what it parses to, so two accepted records that
parse alike are one text. -/
theorem check_ok {text : String} (h : check text = .ok ()) : ∃ j, Json.parse text = .ok j ∧ printT j = text := by
  unfold check at h
  split at h
  · rename_i j hj
    by_cases he : (printT j == text) = true
    · exact ⟨j, hj, beq_iff_eq.1 he⟩
    · simp only [he] at h
      cases h
  · cases h

/-- S0 accepts a text that is `printT` of what it parses to. -/
theorem check_of {text : String} {j : Json} (hp : Json.parse text = .ok j) (he : printT j = text) :
    check text = .ok () := by
  unfold check
  rw [hp]
  simp [he]
  rfl

/-- **Two records S0 accepts that parse alike are one text.** -/
theorem check_inj {r r' : String} (h' : check r' = .ok ()) (h : check r = .ok ())
    (hp : Json.parse r' = Json.parse r) : r' = r := by
  obtain ⟨j', hj', e'⟩ := check_ok h'
  obtain ⟨j, hj, e⟩ := check_ok h
  rw [hj', hj] at hp
  injection hp with hp
  subst hp
  exact e'.symm.trans e

end Flock.RecordText
