import Difftest
import PouwCheck.Opening

/-!
# The difftest's functions: the check's Lean behind `Difftest.serve`

`serve` cases go through `pouw-check serve` (`Core.Protocol.main`); what is not a `serve` case is here, for the shared
harness (`verity/core/difftest`):

* `opening` (`params` as JSON, `commitment` the registration's bytes as hex, `units` a list of integers or null, `unit`,
  `pub` the proof's public file as hex, `opening` as JSON): `core_check.opening`'s reason, or null. A refusal of the
  parameters or the registration is a reject, as `core_check.params` and `commitment` raise `Refused` (a ValueError).
-/

namespace PouwCheck.Difftest

open Lean (Json ToJson toJson)

def need {α : Type} (what : String) : Option α → Except String α
  | some v => .ok v
  | none => .error s!"the input's {what} is malformed"

def get (j : Json) (k : String) : Except String Json := need k (PouwWindow.pyGet? j k)

def hex (j : Json) (k : String) : Except String ByteArray := do
  let .str s ← get j k | throw s!"the input's {k} is not a string"
  PouwCommit.pyFromHex s

def openingOf (j : Json) : Except String Json := do
  let p ← paramsOf (← get j "params")
  let c ← registrationOf (← hex j "commitment")
  let units ← match ← get j "units" with
    | .null => pure Option.none
    | .arr a => some <$> a.toList.mapM fun u => need "units" (PouwWindow.strictInt? u)
    | _ => throw "the input's units is malformed"
  let unit ← need "unit" (PouwWindow.strictInt? (← get j "unit"))
  pure (toJson (← opening p c units unit (← hex j "pub") (← get j "opening")))

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("opening", openingOf)]

end PouwCheck.Difftest
