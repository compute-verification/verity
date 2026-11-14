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

/-!
# A subset sample's JSON, as Flock parses it

The second stage holds a session statement's sample to its own (`SecondStage.ownDraw`, `second-sample`) by `jeq`, and
Flock's verifier finds the session's instances by parsing that sample (`Draw.UnitDraw.ofJson`). Both read a subset
sample: the four keys `law` (`subset`), `population`, `k` (the units' count) and `units` (`subsetJson`). `subsetParses`
proves that a printable sample `jeq` to `subsetJson pop us` that Flock parses proves exactly `us`. The second stage's
own sample is one (`ownDraw_eq`), and so is a sample of every unit of a population of `m` (units `0..m-1`, no closure:
`wholeJson_eq`). A sample the verifier rebuilt is well formed, and printable where its sampling design is a subset's or a
stratified one, so `proved_of_wf` asks only that.

The canonical text is read back by the verifier's parser (`FlockComplete.JsonRT.parse_canon`), so two printable values
of one text are equivalent (`jsonEquiv_of_jeq`), and the decoder's reads follow the equivalence.
-/

@[expose] public section

namespace SpotFlock

open Lean (Json)
open Claims.Json (Good)
open FlockComplete.JsonRT (Distinct)

namespace AddressRow

/-- A subset sample's fields. -/
def subsetList (pop : ℕ) (us : Array ℕ) : List (String × Json) :=
  [("law", Json.str "subset"), ("population", Lean.toJson pop), ("k", Lean.toJson us.size),
    ("units", Spot.natsJson us)]

/-- **A subset sample's JSON**: law `subset`, population `pop`, `k` the units' count, and the units `us`. -/
def subsetJson (pop : ℕ) (us : Array ℕ) : Json := Json.mkObj (subsetList pop us)

/-- **The second stage's own sample for part `p`'s session is a subset sample**: of its `q · n` slots, the second sample's
units among them, counted from the part's first. -/
theorem ownDraw_eq (q : ℕ) (p : Spot.SecondStage.Part) (drawn : Array ℕ) :
    Spot.SecondStage.ownDraw q p drawn = subsetJson (q * p.n) ((drawn.filter (p.holds q)).map (· - p.lo)) := rfl

/-- **A sample held to a subset sample parses to its units**: a printable sample that `jeq` holds to `subsetJson pop us`
and that Flock's verifier parses proves `us`, with no closure. -/
def SubsetParses : Prop :=
  ∀ {j : Json} {pop : ℕ} {us : Array ℕ} {D : _root_.Flock.Draw.UnitDraw}, Good j →
    Spot.jeq j (subsetJson pop us) = true → _root_.Flock.Draw.UnitDraw.ofJson j = .ok D → D.proved = us

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

/-- **One canonical text, one value up to key order**: two printable values `jeq` holds alike are equivalent. -/
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

/-- A value equivalent to a natural number's JSON is it. -/
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

/-- **A key an object lacks**: the verifier's read of `closure` there fails as `ofJson_shape` names it. -/
theorem getObjVal_closure_none {m : Std.TreeMap.Raw String Json compare} (h : m.get? "closure" = none) :
    (Json.obj m).getObjVal? "closure" = .error "property not found: closure" := by
  simp only [Json.getObjVal?]
  cases hg : m.get? "closure" with
  | none => rfl
  | some _ => rw [hg] at h; cases h

/-- **A sample held to a subset sample parses to its units** (`SubsetParses`). The sample is an object, since its text is
one's; its `units` are the subset sample's, entry by entry; and it has no `closure`, since the subset sample has none. -/
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
    rw [_root_.Flock.Draw.UnitDraw.proved, hc, ← Except.ok.inj hmap]
    rfl

/-- **A well-formed sample held to a subset sample parses to its units**: `subsetParses` at a sample the verifier rebuilt
(`Flock.norm`, well formed: `draw_wf`) whose sampling design is one a soundness claim samples by (`law_of_lawWhy`). Such a sample
that decodes is printable (`good_subset`, `good_stratified`). -/
theorem proved_of_wf {j : Json} {pop : ℕ} {us : Array ℕ} {D : _root_.Flock.Draw.UnitDraw} (hw : Claims.Json.Wf j)
    (hD : _root_.Flock.Draw.UnitDraw.ofJson j = .ok D)
    (hl : (∃ K strata, D.law = .stratified K strata) ∨ ∃ k, D.law = .subset k)
    (hj : Spot.jeq j (subsetJson pop us) = true) : D.proved = us := by
  have hg : Good j := by
    rcases hl with ⟨K, strata, hl⟩ | ⟨k, hl⟩
    · exact FlockComplete.JsonRT.good_stratified hw hD hl
    · exact FlockComplete.JsonRT.good_subset hw hD hl
  exact subsetParses hg hj hD

/-- **A sample of every unit of a population of `m`** (law `subset`, `k = m`, units `0..m-1`) is the subset sample of
those units, so `subsetParses` reads it as it reads the second stage's own. -/
theorem wholeJson_eq (m : ℕ) :
    Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson m), ("k", Lean.toJson m),
      ("units", Spot.natsJson (Array.range m))] = subsetJson m (Array.range m) := by
  rw [subsetJson, subsetList, Array.size_range]

end AddressRow

end SpotFlock
