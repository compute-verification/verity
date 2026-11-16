module

public import Proofs.Properties.Verity.Sampled.HonestSession.Setup
import Proofs.Flock.Claims.Complete.Json.Record
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-!
# `SetupFacts`' fields that hold at the honest session

The sample file `draw₀` reads to the subset-of-one sampling design (`draw_ofJson`): `UnitSample.ofJson`'s reads of the four keys of
`UnitSample.toJson`'s object rewrite by `FlockComplete.JsonRT.getObjVal?_mkObj`, its key count by `TreeMap.Raw.size_ofList`,
so no string-keyed tree lookup is left to the kernel.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)
open Flock

private def drawPairs : List (String × Json) := [("law", Json.str "subset"), ("population", Lean.toJson nUnits),
  ("k", Lean.toJson (1 : Nat)), ("units", Json.arr (#[(0 : Nat)].map fun x => Lean.toJson x))]

private theorem draw₀_eq : draw₀ = Json.mkObj drawPairs := rfl

private theorem drawPairs_distinct : FlockComplete.JsonRT.Distinct drawPairs := by
  simp [FlockComplete.JsonRT.Distinct, drawPairs]

private theorem draw₀_obj : draw₀.getObj? = .ok (Std.TreeMap.Raw.ofList drawPairs compare) := rfl

private theorem draw₀_size : (Std.TreeMap.Raw.ofList drawPairs compare).toArray.size = 4 := by
  rw [Std.TreeMap.Raw.size_toArray Std.TreeMap.Raw.WF.ofList, Std.TreeMap.Raw.size_ofList drawPairs_distinct]
  rfl

private theorem draw₀_population : draw₀.getObjVal? "population" = .ok (Lean.toJson nUnits) :=
  draw₀_eq ▸ FlockComplete.JsonRT.getObjVal?_mkObj drawPairs_distinct (by simp [drawPairs])

private theorem draw₀_units : draw₀.getObjVal? "units" = .ok (Json.arr (#[(0 : Nat)].map fun x => Lean.toJson x)) :=
  draw₀_eq ▸ FlockComplete.JsonRT.getObjVal?_mkObj drawPairs_distinct (by simp [drawPairs])

private theorem draw₀_law : draw₀.getObjVal? "law" = .ok (Json.str "subset") :=
  draw₀_eq ▸ FlockComplete.JsonRT.getObjVal?_mkObj drawPairs_distinct (by simp [drawPairs])

private theorem draw₀_k : draw₀.getObjVal? "k" = .ok (Lean.toJson (1 : Nat)) :=
  draw₀_eq ▸ FlockComplete.JsonRT.getObjVal?_mkObj drawPairs_distinct (by simp [drawPairs])

private theorem draw₀_closure : ∃ e, draw₀.getObjVal? "closure" = .error e := by
  rw [draw₀_eq]
  simp only [Json.mkObj, Json.getObjVal?, FlockComplete.JsonRT.get?_mkObj_none
    (show (drawPairs.map Prod.fst).contains "closure" = false by simp [drawPairs])]
  exact ⟨_, rfl⟩

private theorem asNat_toJson (x : Nat) (c : String) : J.asNat (Lean.toJson x) c = .ok x := rfl

private theorem asArr_arr (a : Array Json) (c : String) : J.asArr (Json.arr a) c = .ok a := rfl

private theorem asStr_str (t c : String) : J.asStr (Json.str t) c = .ok t := rfl

private theorem ok_bind {α β : Type} (a : α) (f : α → Except String β) : (Except.ok a >>= f) = f a := rfl

/-- **`SetupFacts.draw_ofJson`**: the sample file reads to the subset-of-one sampling design over `nUnits` with unit `0`. -/
theorem draw_ofJson :
    Flock.Draw.UnitSample.ofJson draw₀ = .ok { samplingDesign := .subset 1, population := nUnits, units := #[0] } := by
  obtain ⟨e, hc⟩ := draw₀_closure
  unfold Flock.Draw.UnitSample.ofJson
  simp only [draw₀_obj, J.nat, J.arr, J.str, J.get, draw₀_population, draw₀_units, draw₀_law, draw₀_k, hc,
    draw₀_size]
  simp [asNat_toJson, asArr_arr, asStr_str, ok_bind, nUnits]
  rfl

end Verity.Sampled.Instance.HonestSession
