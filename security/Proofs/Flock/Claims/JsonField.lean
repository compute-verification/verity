module

public import Spot.Json
public import Proofs.Flock.Claims.Complete.Json.Value
import all Lean.Data.Json.Basic

/-!
# A field of a well-formed JSON value is well formed

`Spot.getD u k` is the field `k` of `u`, and `null` where `u` has none. On an object it is the tree's `get?`
(`field?_obj`), so where `u` is well formed (`Claims.Json.Wf`: every object a well-formed tree), the field it reads is
well formed too (`wf_getD`).

Both are declared in the namespace `Properties.Spot.Units`: `Proofs.Properties.Spot.Units` imports this file and holds
the lemmas on rebuilt values that stand on them.
-/

@[expose] public section

open Lean (Json)
open Spot (getD field?)

namespace Properties.Spot.Units

theorem field?_obj (m : Std.TreeMap.Raw String Json compare) (k : String) : field? (.obj m) k = m.get? k := by
  simp only [field?, Json.getObjVal?]
  cases m.get? k <;> rfl

/-- A field of a well-formed value is well formed. -/
theorem wf_getD {u : Json} (hu : Claims.Json.Wf u) (k : String) : Claims.Json.Wf (getD u k) := by
  cases hu with
  | obj hm h =>
    simp only [getD, field?_obj]
    cases hk : Std.TreeMap.Raw.get? _ k with
    | none => exact .null
    | some v => exact h k v hk
  | _ => exact .null

end Properties.Spot.Units
