module

public import Proofs.Flock.SpotFlock.RunsAs.Steps
public import Proofs.Flock.SpotFlock.SampleInts
public import Proofs.Properties.Spot
public import Proofs.Flock.Bridge.V5.Accepted
public import Proofs.Flock.Claims.Complete.Json.Stmt
public import Proofs.Flock.Soundness.ExecSample
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas

/-! The units a first-stage sample proves are the ones its select drew (`proved_select`). -/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

theorem getD_mkObj_last (l : List (String × Json)) (k : String) (v : Json) :
    Spot.getD (Json.mkObj (l ++ [(k, v)])) k = v := by
  rw [Json.mkObj, getD_obj, Std.TreeMap.Raw.ofList_eq_insertMany_empty, Std.TreeMap.Raw.insertMany_append,
    Std.TreeMap.Raw.insertMany_list_singleton,
    Std.TreeMap.Raw.getElem?_insert_self (Std.TreeMap.Raw.WF.insertMany Std.TreeMap.Raw.WF.emptyc)]
  rfl

theorem flockSamplingDesign_keys {samplingDesign : Json} {n : ℕ} {r : Flock.Draw.SamplingDesign} (h : Spot.Draw.flockSamplingDesign samplingDesign n = .ok r) :
    ∃ names : Array String, Spot.keys? samplingDesign = some (Spot.sortStrings names) ∧ "closure" ∉ names := by
  unfold Spot.Draw.flockSamplingDesign at h
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

theorem closure_not_key {samplingDesign : Json} {names : Array String} (hk : Spot.keys? samplingDesign = some (Spot.sortStrings names))
    (hX : "closure" ∉ names) : ((Spot.entries samplingDesign).toList.map Prod.fst).contains "closure" = false := by
  cases samplingDesign with
  | obj kvs =>
    simp only [Spot.keys?, Option.some.injEq] at hk
    rw [Bool.eq_false_iff]
    intro hc
    apply hX
    have h1 : "closure" ∈ kvs.toArray.map Prod.fst := by
      rw [Array.mem_def, Array.toList_map]
      simpa [Spot.entries] using hc
    have h2 := (FlockSoundness.Audit.SampleExec.mem_qsort _ (· < ·) _).2 h1
    rw [show (kvs.toArray.map Prod.fst).qsort (· < ·) = Spot.sortStrings names from hk] at h2
    exact (FlockSoundness.Audit.SampleExec.mem_qsort _ _ _).1 h2
  | _ => simp [Spot.keys?] at hk

private theorem select_shape {rule : Spot.Select.Rule} {p : Spot.Program} {query : Json} {table : Option Json}
    {src : ByteArray} {draw : Json} {units : Array ℕ} {read : ℕ} (hr : rule ≠ .all)
    (h : Spot.Select.select rule p query table none src = .ok (draw, units, read)) :
    ∃ (n : ℕ) (samplingDesign : Json) (parsed : Flock.Draw.SamplingDesign), Spot.Draw.flockSamplingDesign samplingDesign n = .ok parsed ∧
      draw = Json.mkObj ((Spot.entries samplingDesign).toList ++ [("population", Lean.toJson n), ("units", Spot.natsJson units)]) ∧
      ¬(!(Spot.Draw.checkDraw draw (Lean.toJson n) samplingDesign).isEmpty) = true := by
  unfold Spot.Select.select at h
  cases rule with
  | all => exact absurd rfl hr
  | samplingDesign t =>
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

theorem proved_select {rule : Spot.Select.Rule} {p : Spot.Program} {query : Json} {table : Option Json}
    {src : ByteArray} {draw : Json} {units : Array ℕ} {read : ℕ} (hr : rule ≠ .all)
    (h : Spot.Select.select rule p query table none src = .ok (draw, units, read)) :
    (Spot.Draw.proved draw).map Int.toNat = units := by
  obtain ⟨n, samplingDesign, parsed, hfl, rfl, hcd⟩ := select_shape hr h
  have hcd' : Spot.Draw.checkDraw (Json.mkObj ((Spot.entries samplingDesign).toList ++
      [("population", Lean.toJson n), ("units", Spot.natsJson units)])) (Lean.toJson n) samplingDesign = #[] := by
    simpa using hcd
  obtain ⟨-, -, -, -, -, us, xs, hus, hxs, hasc, -, -⟩ := (Properties.Spot.checkSample_nil _ _ _).1 hcd'
  have hu : Spot.getD (Json.mkObj ((Spot.entries samplingDesign).toList ++
      [("population", Lean.toJson n), ("units", Spot.natsJson units)])) "units" = Spot.natsJson units := by
    rw [show (Spot.entries samplingDesign).toList ++ [("population", Lean.toJson n), ("units", Spot.natsJson units)] =
      ((Spot.entries samplingDesign).toList ++ [("population", Lean.toJson n)]) ++ [("units", Spot.natsJson units)] by simp]
    exact getD_mkObj_last _ _ _
  obtain ⟨names, hk, hX⟩ := flockSamplingDesign_keys hfl
  have hcl : Spot.getD (Json.mkObj ((Spot.entries samplingDesign).toList ++
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
