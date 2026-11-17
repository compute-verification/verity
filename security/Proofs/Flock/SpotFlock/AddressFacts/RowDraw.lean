module

public import Proofs.Flock.SpotFlock.AddressFacts.Row
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Flock.Claims.Complete.Json.RoundTrip
public import Proofs.Flock.Claims.Complete.Json.Stmt
public import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-! A printable sample held to a subset sample by `jeq` that Flock parses proves exactly its units (`subsetParses`). -/

@[expose] public section

namespace SpotFlock

open Lean (Json)
open Claims.Json (Good)
open FlockComplete.JsonRT (Distinct)

namespace AddressRow

def subsetList (pop : ℕ) (us : Array ℕ) : List (String × Json) :=
  [("law", Json.str "subset"), ("population", Lean.toJson pop), ("k", Lean.toJson us.size),
    ("units", Spot.natsJson us)]

def subsetJson (pop : ℕ) (us : Array ℕ) : Json := Json.mkObj (subsetList pop us)

theorem ownSample_eq (q : ℕ) (p : Spot.SecondStage.Part) (drawn : Array ℕ) :
    Spot.SecondStage.ownSample q p drawn = subsetJson (q * p.n) ((drawn.filter (p.holds q)).map (· - p.lo)) := rfl

def SubsetParses : Prop :=
  ∀ {j : Json} {pop : ℕ} {us : Array ℕ} {D : _root_.Flock.Draw.UnitSample}, Good j →
    Spot.jeq j (subsetJson pop us) = true → _root_.Flock.Draw.UnitSample.ofJson j = .ok D → D.proved = us

theorem subsetList_distinct (pop : ℕ) (us : Array ℕ) : Distinct (subsetList pop us) := by
  simp [subsetList]

theorem subsetJson_obj (pop : ℕ) (us : Array ℕ) :
    subsetJson pop us = .obj (Std.TreeMap.Raw.ofList (subsetList pop us) compare) := by
  simp only [subsetJson, Json.mkObj]

theorem good_subsetJson (pop : ℕ) (us : Array ℕ) : Good (subsetJson pop us) := by
  refine FlockComplete.JsonRT.good_mkObj (subsetList_distinct pop us) ?_
  intro p hp
  simp only [subsetList, List.mem_cons, List.mem_nil_iff, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl
  · exact .str _
  · exact FlockComplete.JsonRT.good_toJson _
  · exact FlockComplete.JsonRT.good_toJson _
  · refine FlockComplete.JsonRT.good_arr_of fun x hx => ?_
    obtain ⟨n, -, rfl⟩ := Array.mem_map.1 hx
    exact FlockComplete.JsonRT.good_toJson n

theorem jsonEquiv_of_jeq {a b : Json} (ha : Good a) (hb : Good b) (h : Spot.jeq a b = true) :
    FlockDeploy.JsonEquiv a b := by
  obtain ⟨s, hs⟩ := FlockComplete.JsonRT.canon_ok_some a ha
  obtain ⟨t, ht⟩ := FlockComplete.JsonRT.canon_ok_some b hb
  unfold Spot.jeq at h
  rw [hs, ht] at h
  have hst : s = t := by simpa using h
  obtain ⟨x, hx, hax⟩ := FlockComplete.JsonRT.parse_canon a s ha hs
  obtain ⟨y, hy, hby⟩ := FlockComplete.JsonRT.parse_canon b t hb ht
  rw [← hst, hx] at hy
  have hxy : x = y := Except.ok.inj hy
  rw [← hxy] at hby
  have e₁ : FlockDeploy.jsonForm a = FlockDeploy.jsonForm x := FlockComplete.JsonRT.same_jsonEquiv hax
  have e₂ : FlockDeploy.jsonForm b = FlockDeploy.jsonForm x := FlockComplete.JsonRT.same_jsonEquiv hby
  exact e₁.trans e₂.symm

theorem eq_toJson_of_jsonEquiv {x : Json} {u : ℕ} (h : FlockDeploy.JsonEquiv x (Lean.toJson u)) :
    x = Lean.toJson u := by
  have h' : FlockDeploy.jsonForm (Lean.toJson u) = FlockDeploy.jsonForm x := Eq.symm h
  cases x with
  | num n =>
    have : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .num n := h'
    cases this
    rfl
  | null => cases (h' : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .null)
  | bool b => cases (h' : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .bool b)
  | str s => cases (h' : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .str s)
  | arr a =>
    obtain ⟨a⟩ := a
    cases (h' : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .arr _)
  | obj m =>
    rw [FlockDeploy.jsonForm_obj] at h'
    cases (h' : FlockDeploy.JsonForm.num ⟨Int.ofNat u, 0⟩ = .obj _)

theorem getObjVal_closure_none {m : Std.TreeMap.Raw String Json compare} (h : m.get? "closure" = none) :
    (Json.obj m).getObjVal? "closure" = .error "property not found: closure" := by
  simp only [Json.getObjVal?]
  cases hg : m.get? "closure" with
  | none => rfl
  | some _ => rw [hg] at h; cases h

theorem subsetParses : SubsetParses := by
  intro j pop us D hg hj hD
  have he := jsonEquiv_of_jeq hg (good_subsetJson pop us) hj
  rw [subsetJson_obj] at he
  have hn : (Std.TreeMap.Raw.ofList (subsetList pop us) compare).WF := Std.TreeMap.Raw.WF.ofList
  cases hg with
  | null => cases FlockDeploy.jsonEquiv_null he
  | bool _ => cases FlockDeploy.jsonEquiv_bool he
  | num _ => cases FlockDeploy.jsonEquiv_num he
  | str _ => cases FlockDeploy.jsonEquiv_str he
  | arr _ =>
    obtain ⟨b', hb'⟩ := FlockDeploy.jsonEquiv_arr he
    cases hb'
  | @obj m hm _ =>
    obtain ⟨hk, hv⟩ := (FlockDeploy.jsonEquiv_obj_iff hm hn).1 he
    obtain ⟨⟨a, ha, hmap⟩, hcl, -⟩ := FlockSoundness.Discharge.Composed.ofJson_shape hD
    have hga : m.get? "units" = some (Json.arr a) :=
      (FlockComplete.JsonRT.mem_of_getObjVal hm (FlockComplete.JsonRT.arr_ok ha)).2
    have hgu : (Std.TreeMap.Raw.ofList (subsetList pop us) compare).get? "units" = some (Spot.natsJson us) :=
      FlockComplete.JsonRT.get?_mkObj (subsetList_distinct pop us) (by simp [subsetList])
    have hau := (FlockDeploy.jsonEquiv_arr_iff a (us.map fun x => Lean.toJson x)).1 (hv "units" _ _ hga hgu)
    have hae : a = us.map fun x => Lean.toJson x := by
      apply Array.ext hau.1
      intro i h₁ h₂
      have hi := hau.2 i h₁ h₂
      rw [Array.getElem_map] at hi ⊢
      exact eq_toJson_of_jsonEquiv hi
    rw [hae, FlockSoundness.Discharge.Composed.mapM_asNat_toJson] at hmap
    have hnc : (Std.TreeMap.Raw.ofList (subsetList pop us) compare).get? "closure" = none :=
      FlockComplete.JsonRT.get?_mkObj_none (by simp [subsetList])
    have hs := hk "closure"
    rw [hnc] at hs
    have hmc : m.get? "closure" = none := by
      cases hmg : m.get? "closure" with
      | none => rfl
      | some _ => rw [hmg] at hs; cases hs
    have hc : D.closure = #[] := hcl (getObjVal_closure_none hmc)
    rw [_root_.Flock.Draw.UnitSample.proved, hc, ← Except.ok.inj hmap]
    rfl

theorem proved_of_wf {j : Json} {pop : ℕ} {us : Array ℕ} {D : _root_.Flock.Draw.UnitSample} (hw : Claims.Json.Wf j)
    (hD : _root_.Flock.Draw.UnitSample.ofJson j = .ok D)
    (hl : (∃ K strata, D.samplingDesign = .stratified K strata) ∨ ∃ k, D.samplingDesign = .subset k)
    (hj : Spot.jeq j (subsetJson pop us) = true) : D.proved = us := by
  have hg : Good j := by
    rcases hl with ⟨K, strata, hl⟩ | ⟨k, hl⟩
    · exact FlockComplete.JsonRT.good_stratified hw hD hl
    · exact FlockComplete.JsonRT.good_subset hw hD hl
  exact subsetParses hg hj hD

theorem wholeJson_eq (m : ℕ) :
    Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson m), ("k", Lean.toJson m),
      ("units", Spot.natsJson (Array.range m))] = subsetJson m (Array.range m) := by
  rw [subsetJson, subsetList, Array.size_range]

end AddressRow

end SpotFlock
