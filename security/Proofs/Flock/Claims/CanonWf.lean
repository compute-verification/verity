module

public import Proofs.Flock.Claims.CanonRead
public import Proofs.Flock.Claims.JsonNorm
import Std.Data.TreeMap.Raw.Lemmas
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String

/-!
# A well-formed value is sorted

A well-formed value (`Claims.Json.Wf`: every object a well-formed tree) holds each object's keys in increasing order
(`sortedKeys_of_wf`), so what `CanonRead` and `CanonLine` prove of sorted values holds of every value a parser returns.
The file reads the canonical text's lemmas and the values' induction, and nothing of a session, a record or a device.
-/

@[expose] public section

namespace Properties.Challenger.CanonRead

open Lean (Json)

/-- **A well-formed value's objects hold their keys in order.** -/
theorem sortedKeys_of_wf : ∀ j : Json, Claims.Json.Wf j → _root_.Spot.sortedKeys j = true := by
  refine FlockDeploy.json_ind (fun _ => by simp [_root_.Spot.sortedKeys])
    (fun _ _ => by simp [_root_.Spot.sortedKeys]) (fun _ _ => by simp [_root_.Spot.sortedKeys])
    (fun _ _ => by simp [_root_.Spot.sortedKeys]) ?_ ?_
  · intro a ih hw
    cases hw with
    | arr h => exact sortedKeys_arr.2 fun x hx => ih x hx (h x hx)
  · intro m ih hw
    cases hw with
    | obj hm h =>
      refine sortedKeys_obj.2 ⟨?_, fun p hp => ?_⟩
      · rw [Std.TreeMap.Raw.toList_toArray, increasing_iff, List.pairwise_map]
        exact (Std.TreeMap.Raw.ordered_keys_toList hm).imp fun hlt => compareOfLessAndEq_eq_lt.1 hlt
      · have hp' : p ∈ m.toList := by
          rw [← Std.TreeMap.Raw.toList_toArray]
          exact Array.mem_toList_iff.2 hp
        refine ih p hp' (h p.1 p.2 ?_)
        rw [Std.TreeMap.Raw.get?_eq_getElem?]
        exact (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp'

end Properties.Challenger.CanonRead
