module

public import Proofs.Flock.Soundness.Discharge.Hidden.CanonOk
public import Proofs.Flock.Soundness.Discharge.Hidden.OkWith
public import Proofs.Flock.Soundness.Refine.Setup
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-!
# The sample that proves no unit

`emptySample N` is the subset sample of no unit over a population of `N`, as the verifier's executable prints a sample
(`Draw.UnitDraw.toJson`). `Draw.UnitDraw.ofJson` decodes it to that sample (`ofJson_emptySample`) and `canonUtf8` prints
it (`canonUtf8_emptySample`), so a setup can be made at it over any population.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Hidden

open Flock Refine
open Lean (Json)

/-- The sample that proves no unit of a population of `N`, as the verifier's executable prints a sample. -/
def emptySample (N : Nat) : Json := Draw.UnitDraw.toJson ⟨.subset 0, N, #[], #[]⟩

theorem emptySample_eq (N : Nat) : emptySample N =
    Json.mkObj [("law", "subset"), ("population", Lean.toJson N), ("k", Lean.toJson 0), ("units", Json.arr #[])] := by
  unfold emptySample Draw.UnitDraw.toJson
  simp only [Array.map_empty]

/-- `canonUtf8` prints the sample of no unit. -/
theorem canonUtf8_emptySample (N : Nat) : (canonUtf8 (emptySample N)).isOk = true := by
  rw [emptySample_eq]
  refine canonUtf8_mkObj fun v hv => ?_
  simp only [List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at hv
  rcases hv with rfl | rfl | rfl | rfl
  · exact canonUtf8_str _
  · exact canonUtf8_nat _
  · exact canonUtf8_nat _
  · exact canonUtf8_arr fun x hx => by cases hx; contradiction

/-- An object built from a list of distinct keys holds each of the list's values under its key. -/
theorem getObjVal?_mkObj {l : List (String × Json)} (hd : l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq))
    {k : String} {v : Json} (hm : (k, v) ∈ l) : (Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm]
  rfl

/-- An object built from a list holds nothing under a key the list does not name. -/
theorem getObjVal?_mkObj_not {l : List (String × Json)} {k : String} (hk : (l.map Prod.fst).contains k = false) :
    ∃ e, (Json.mkObj l).getObjVal? k = .error e := by
  simp only [Json.mkObj, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false hk]
  exact ⟨_, rfl⟩

/-- **The sample of no unit decodes**: to the subset sample of no unit over the population of `N`. -/
theorem ofJson_emptySample (N : Nat) : Draw.UnitDraw.ofJson (emptySample N) = .ok ⟨.subset 0, N, #[], #[]⟩ := by
  have hd : [("law", (Json.str "subset")), ("population", Lean.toJson N), ("k", Lean.toJson 0),
      ("units", Json.arr #[])].Pairwise (fun a b => ¬ compare a.1 b.1 = .eq) := by simp
  have hl := getObjVal?_mkObj hd (k := "law") (v := Json.str "subset") (by simp)
  have hp := getObjVal?_mkObj hd (k := "population") (v := Lean.toJson N) (by simp)
  have hk := getObjVal?_mkObj hd (k := "k") (v := Lean.toJson 0) (by simp)
  have hu := getObjVal?_mkObj hd (k := "units") (v := Json.arr #[]) (by simp)
  obtain ⟨e, he⟩ := getObjVal?_mkObj_not (l := [("law", (Json.str "subset")), ("population", Lean.toJson N),
    ("k", Lean.toJson 0), ("units", Json.arr #[])]) (k := "closure") (by simp)
  have hsz : (Std.TreeMap.Raw.ofList [("law", (Json.str "subset")), ("population", Lean.toJson N),
      ("k", Lean.toJson 0), ("units", Json.arr #[])] compare).toArray.size = 4 := by
    rw [Std.TreeMap.Raw.size_toArray Std.TreeMap.Raw.WF.ofList, Std.TreeMap.Raw.size_ofList hd]
    rfl
  have ho : (Json.mkObj [("law", (Json.str "subset")), ("population", Lean.toJson N), ("k", Lean.toJson 0),
      ("units", Json.arr #[])]).getObj? = .ok (Std.TreeMap.Raw.ofList [("law", (Json.str "subset")),
        ("population", Lean.toJson N), ("k", Lean.toJson 0), ("units", Json.arr #[])] compare) := rfl
  have hN : ∀ c, J.asNat (Lean.toJson N) c = .ok N := fun _ => rfl
  have h0 : ∀ c, J.asNat (Lean.toJson 0) c = .ok 0 := fun _ => rfl
  rw [emptySample_eq]
  unfold Draw.UnitDraw.ofJson
  simp only [J.nat, J.arr, J.str, J.get, J.asStr, J.asArr, hl, hp, hk, hu, he, ho, hN, h0, hsz, ok_bind,
    Json.getArr?, Json.getStr?, Array.mapM_empty, pure, Except.pure]
  simp only [throw_bind']
  rfl

end FlockSoundness.Discharge.Hidden
