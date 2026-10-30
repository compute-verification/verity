import PouwWindow.Json

/-!
# The fixed length (`window.py`, "the fixed length")

Correct tokens served past the honest run's end of sequence are not useful work. A row of
record serves its slot's public max_tokens whatever it samples (vLLM's `ignore_eos`), so W_use is public. `lengthItem` is
the `length` item of a window whose slots serve `maxTokens`, and `lengthProblem` is why a registered one is not, the first
rule it breaks, with `length_problem`'s reason.
-/

namespace PouwWindow

open Lean (Json)

/-- `STOP_FIXED`. -/
def stopFixed : String := "fixed"

/-- `LENGTH_KEYS`, sorted. -/
def lengthKeys : List String := ["max_tokens", "served", "stop"]

/-- `length_item`. -/
def lengthItem (maxTokens : List Nat) : Json :=
  Json.mkObj [("max_tokens", Json.arr (maxTokens.map fun (n : Nat) => Json.num n).toArray),
    ("served", Json.arr (maxTokens.map fun (n : Nat) => Json.num n).toArray), ("stop", stopFixed)]

/-- An object's keys, sorted. -/
def sortedKeys : Json → Option (List String)
  | .obj kvs => some ((kvs.toArray.map (·.1)).qsort (· < ·)).toList
  | _ => none

/-- A list of natural numbers, each written as an integer. -/
def nats? : Json → Option (List Nat)
  | .arr a => a.toList.mapM natOf?
  | _ => none

/-- The first slot `i`, counting from the first argument, whose served length `s` is not its max_tokens `n`. -/
def firstMismatch : Nat → List Nat → List Nat → Option (Nat × Nat × Nat)
  | i, s :: ss, n :: ns => if s ≠ n then some (i, s, n) else firstMismatch (i + 1) ss ns
  | _, _, _ => none

/-- `length_problem`. -/
def lengthProblem (item : Json) (maxTokens : List Nat) : Option String :=
  if sortedKeys item ≠ some lengthKeys then some "window-form: length is not {max_tokens, served, stop}"
  else match field? item "stop" with
  | some (.str stop) =>
    if stop ≠ stopFixed then some s!"window-form: eos-keyed-row: the stop mode is {stop}, not {stopFixed} (ignore_eos)"
    else match (field? item "max_tokens").bind nats?, (field? item "served").bind nats? with
    | some m, some served =>
      if m ≠ maxTokens then some s!"window-form: max-tokens: registered {m}, the verifier's {maxTokens}"
      else if served.length ≠ m.length then
        some s!"window-form: length-mismatch: {served.length} slots served, not {m.length}"
      else (firstMismatch 0 served m).map fun (i, s, n) =>
        s!"window-form: length-mismatch: slot {i} served {s} tokens, not its max_tokens {n}"
    | _, _ => some "window-form: length's max_tokens and served are not lists of natural numbers"
  | _ => some "window-form: length's stop is not a string"

end PouwWindow
