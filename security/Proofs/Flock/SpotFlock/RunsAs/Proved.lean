module

public import Proofs.Flock.SpotFlock.RunsAs.Steps
public import Proofs.Flock.SpotFlock.SampleInts
public import Proofs.Properties.Spot
public import Proofs.Flock.Bridge.V5.Accepted
public import Proofs.Flock.Claims.Complete.Json.Stmt
public import Proofs.Flock.Soundness.ExecDraw
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas

/-!
# The units a first-stage sample proves are the ones its select drew (part 1)

`Select.select` under a sampling design, with no closure map, builds its sample object from the sampling design's entries, the population and the
units it drew (`select_shape`), and checks it (`Draw.checkDraw`). The sampling design `Draw.flockLaw` accepts has no `closure` key
(`flockLaw_keys`, `closure_not_key`), so the sample carries none, and the checked sample's units ascend; so the units the
sample proves (`Draw.proved`, read back as naturals) are the select's (`proved_select`). This is the step from the
run's sample, `(Draw.proved draw).map Int.toNat`, to the model's, `RegTwo.sample`, which is the select's units.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- `getD` of `Json.mkObj (l ++ [(k, v)])` at `k` is `v`: the last entry for a key is the one the object holds. -/
theorem getD_mkObj_last (l : List (String × Json)) (k : String) (v : Json) :
    Spot.getD (Json.mkObj (l ++ [(k, v)])) k = v := by
  rw [Json.mkObj, getD_obj, Std.TreeMap.Raw.ofList_eq_insertMany_empty, Std.TreeMap.Raw.insertMany_append,
    Std.TreeMap.Raw.insertMany_list_singleton,
    Std.TreeMap.Raw.getElem?_insert_self (Std.TreeMap.Raw.WF.insertMany Std.TreeMap.Raw.WF.emptyc)]
  rfl

/-- **A sampling design `flockLaw` accepts has its kind's keys**, none of them `closure`. -/
theorem flockLaw_keys {law : Json} {n : ℕ} {r : Flock.Draw.Law} (h : Spot.Draw.flockLaw law n = .ok r) :
    ∃ names : Array String, Spot.keys? law = some (Spot.sortStrings names) ∧ "closure" ∉ names := by
  unfold Spot.Draw.flockLaw at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  all_goals
    first
    | focus (cases h; done)
    | (split at h
       · cases h
       · rename_i hk
         have hk' := hk
         simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hk'
         exact ⟨_, hk', by simp⟩)

/-- **An object whose keys are `names`' has no key outside `names`.** -/
theorem closure_not_key {law : Json} {names : Array String} (hk : Spot.keys? law = some (Spot.sortStrings names))
    (hX : "closure" ∉ names) : ((Spot.entries law).toList.map Prod.fst).contains "closure" = false := by
  cases law with
  | obj kvs =>
    simp only [Spot.keys?, Option.some.injEq] at hk
    rw [Bool.eq_false_iff]
    intro hc
    apply hX
    have h1 : "closure" ∈ kvs.toArray.map Prod.fst := by
      rw [Array.mem_def, Array.toList_map]
      simpa [Spot.entries] using hc
    have h2 := (FlockSoundness.Audit.DrawExec.mem_qsort _ (· < ·) _).2 h1
    rw [show (kvs.toArray.map Prod.fst).qsort (· < ·) = Spot.sortStrings names from hk] at h2
    exact (FlockSoundness.Audit.DrawExec.mem_qsort _ _ _).1 h2
  | _ => simp [Spot.keys?] at hk

/-- **A select under a sampling design, with no closure map, accepted**: its sampling design `flockLaw` accepts, its sample object is the sampling design's
entries with the population and the units it drew, and `checkDraw` passes it. -/
private theorem select_shape {rule : Spot.Select.Rule} {p : Spot.Program} {query : Json} {table : Option Json}
    {src : ByteArray} {draw : Json} {units : Array ℕ} {read : ℕ} (hr : rule ≠ .all)
    (h : Spot.Select.select rule p query table none src = .ok (draw, units, read)) :
    ∃ (n : ℕ) (law : Json) (parsed : Flock.Draw.Law), Spot.Draw.flockLaw law n = .ok parsed ∧
      draw = Json.mkObj ((Spot.entries law).toList ++ [("population", Lean.toJson n), ("units", Spot.natsJson units)]) ∧
      ¬(!(Spot.Draw.checkDraw draw (Lean.toJson n) law).isEmpty) = true := by
  unfold Spot.Select.select at h
  cases rule with
  | all => exact absurd rfl hr
  | law t =>
    simp only [bind, Except.bind, pure, Except.pure, Array.isEmpty_empty, ↓reduceIte, Array.append_empty] at h
    repeat'
      first
      | split at h
      | focus (cases h; done)
      | focus (rename_i hthrow; cases hthrow; done)
      | (simp only [Except.ok.injEq, Prod.mk.injEq] at h
         obtain ⟨rfl, rfl, -⟩ := h
         exact ⟨_, _, _, by assumption, rfl, by assumption⟩)
  | twoStage g t k =>
    simp only [bind, Except.bind, pure, Except.pure, Array.isEmpty_empty, ↓reduceIte, Array.append_empty] at h
    repeat'
      first
      | split at h
      | focus (cases h; done)
      | focus (rename_i hthrow; cases hthrow; done)
      | (simp only [Except.ok.injEq, Prod.mk.injEq] at h
         obtain ⟨rfl, rfl, -⟩ := h
         exact ⟨_, _, _, by assumption, rfl, by assumption⟩)

/-- **Part 1: the units a first-stage sample proves are the ones its select drew**: under a sampling design, with no closure map,
`(Draw.proved draw).map Int.toNat` is the select's units. -/
theorem proved_select {rule : Spot.Select.Rule} {p : Spot.Program} {query : Json} {table : Option Json}
    {src : ByteArray} {draw : Json} {units : Array ℕ} {read : ℕ} (hr : rule ≠ .all)
    (h : Spot.Select.select rule p query table none src = .ok (draw, units, read)) :
    (Spot.Draw.proved draw).map Int.toNat = units := by
  obtain ⟨n, law, parsed, hfl, rfl, hcd⟩ := select_shape hr h
  have hcd' : Spot.Draw.checkDraw (Json.mkObj ((Spot.entries law).toList ++
      [("population", Lean.toJson n), ("units", Spot.natsJson units)])) (Lean.toJson n) law = #[] := by
    simpa using hcd
  obtain ⟨-, -, -, -, -, us, xs, hus, hxs, hasc, -, -⟩ := (Properties.Spot.checkDraw_nil _ _ _).1 hcd'
  have hu : Spot.getD (Json.mkObj ((Spot.entries law).toList ++
      [("population", Lean.toJson n), ("units", Spot.natsJson units)])) "units" = Spot.natsJson units := by
    rw [show (Spot.entries law).toList ++ [("population", Lean.toJson n), ("units", Spot.natsJson units)] =
      ((Spot.entries law).toList ++ [("population", Lean.toJson n)]) ++ [("units", Spot.natsJson units)] by simp]
    exact getD_mkObj_last _ _ _
  obtain ⟨names, hk, hX⟩ := flockLaw_keys hfl
  have hcl : Spot.getD (Json.mkObj ((Spot.entries law).toList ++
      [("population", Lean.toJson n), ("units", Spot.natsJson units)])) "closure" = .null :=
    getD_mkObj_none (by simpa using closure_not_key hk hX)
  rw [hu] at hus
  simp only [Spot.natsJson, Json.arr.injEq] at hus
  subst hus
  have hx : xs = units.map fun m : ℕ => (m : Int) := by
    have hf : (Spot.int? ∘ fun m : ℕ => Lean.toJson m) = some ∘ fun m : ℕ => (m : Int) := by
      funext m
      exact Properties.Spot.toJson_nat_int m
    rw [← Properties.Spot.Credited.filterMap_int_arr hxs]
    apply Array.toList_inj.1
    rw [Array.toList_filterMap, Array.toList_map, Array.toList_map, List.filterMap_map, hf,
      List.filterMap_eq_map]
  unfold Spot.Draw.proved
  dsimp only
  rw [hu, hcl]
  dsimp only [Spot.natsJson]
  rw [Array.append_empty, Properties.Spot.Credited.filterMap_int_arr hxs, Verity.Sampled.qsort_ascending hasc, hx]
  simp [Array.map_map, Function.comp_def]

end SpotFlock.RunsAs
