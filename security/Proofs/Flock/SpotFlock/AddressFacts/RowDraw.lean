module

public import Proofs.Flock.SpotFlock.AddressFacts.Row
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Flock.Claims.Complete.Json.RoundTrip
public import Proofs.Flock.Claims.Complete.Json.Stmt
public import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ

/-!
# A subset draw's JSON, as Flock parses it

The second stage holds a session statement's draw to its own (`SecondStage.ownDraw`, `second-sample`) by `jeq`, and
Flock's verifier finds the session's instances by parsing that draw (`Draw.UnitDraw.ofJson`). Both read a subset
draw: the four keys `law` (`subset`), `population`, `k` (the units' count) and `units` (`subsetJson`). `subsetParses`
proves that a printable draw `jeq` to `subsetJson pop us` that Flock parses proves exactly `us`. The second stage's
own draw is one (`ownDraw_eq`), and so is a draw of every unit of a population of `m` (units `0..m-1`, no closure).

The canonical text is read back by the verifier's parser (`FlockComplete.JsonRT.parse_canon`), so two printable values
of one text are equivalent (`jsonEquiv_of_jeq`), and the decoder's reads follow the equivalence.
-/

@[expose] public section

namespace SpotFlock

open Lean (Json)
open Claims.Json (Good)
open FlockComplete.JsonRT (Distinct)

namespace AddressRow

/-- A subset draw's fields. -/
def subsetList (pop : ℕ) (us : Array ℕ) : List (String × Json) :=
  [("law", Json.str "subset"), ("population", Lean.toJson pop), ("k", Lean.toJson us.size),
    ("units", Spot.natsJson us)]

/-- **A subset draw's JSON**: law `subset`, population `pop`, `k` the units' count, and the units `us`. -/
def subsetJson (pop : ℕ) (us : Array ℕ) : Json := Json.mkObj (subsetList pop us)

/-- **The second stage's own draw for part `p`'s session is a subset draw**: of its `q · n` slots, the second draw's
units among them, counted from the part's first. -/
theorem ownDraw_eq (q : ℕ) (p : Spot.SecondStage.Part) (drawn : Array ℕ) :
    Spot.SecondStage.ownDraw q p drawn = subsetJson (q * p.n) ((drawn.filter (p.holds q)).map (· - p.lo)) := rfl

/-- **A draw held to a subset draw parses to its units**: a printable draw that `jeq` holds to `subsetJson pop us`
and that Flock's verifier parses proves `us`, with no closure. -/
def SubsetParses : Prop :=
  ∀ {j : Json} {pop : ℕ} {us : Array ℕ} {D : _root_.Flock.Draw.UnitDraw}, Good j →
    Spot.jeq j (subsetJson pop us) = true → _root_.Flock.Draw.UnitDraw.ofJson j = .ok D → D.proved = us

theorem subsetList_distinct (pop : ℕ) (us : Array ℕ) : Distinct (subsetList pop us) := by
  simp [subsetList]

theorem subsetJson_obj (pop : ℕ) (us : Array ℕ) :
    subsetJson pop us = .obj (Std.TreeMap.Raw.ofList (subsetList pop us) compare) := rfl

theorem good_subsetJson (pop : ℕ) (us : Array ℕ) : Good (subsetJson pop us) := by
  refine FlockComplete.JsonRT.good_mkObj (subsetList_distinct pop us) ?_
  simp only [subsetList, Spot.natsJson, List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨.str _, FlockComplete.JsonRT.good_toJson _, FlockComplete.JsonRT.good_toJson _,
    FlockComplete.JsonRT.good_arr_of fun x hx => ?_⟩
  obtain ⟨n, -, rfl⟩ := Array.mem_map.1 hx
  exact FlockComplete.JsonRT.good_toJson n

/-- **One canonical text, one value up to key order**: two printable values `jeq` holds alike are equivalent. -/
theorem jsonEquiv_of_jeq {a b : Json} (ha : Good a) (hb : Good b) (h : Spot.jeq a b = true) :
    FlockDeploy.JsonEquiv a b := by
  obtain ⟨s, hs⟩ := FlockComplete.JsonRT.canon_ok_some a ha
  obtain ⟨t, ht⟩ := FlockComplete.JsonRT.canon_ok_some b hb
  have hst : s = t := by
    unfold Spot.jeq at h
    rw [hs, ht] at h
    try dsimp only at h
    exact beq_iff_eq.1 h
  subst hst
  obtain ⟨x, hx, hax⟩ := FlockComplete.JsonRT.parse_canon a s ha hs
  obtain ⟨y, hy, hby⟩ := FlockComplete.JsonRT.parse_canon b s hb ht
  rw [hx] at hy
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

theorem subsetParses : SubsetParses := by
  intro j pop us D hg hj hD
  have he := jsonEquiv_of_jeq hg (good_subsetJson pop us) hj
  rw [subsetJson_obj] at he
  cases hg with
  | null => cases FlockDeploy.jsonEquiv_null he
  | bool b => cases FlockDeploy.jsonEquiv_bool he
  | num m => cases FlockDeploy.jsonEquiv_num he
  | str s => cases FlockDeploy.jsonEquiv_str he
  | arr _ =>
    obtain ⟨b', hb'⟩ := FlockDeploy.jsonEquiv_arr he
    cases hb'
  | @obj m hm _ =>
    obtain ⟨hk, hv⟩ := (FlockDeploy.jsonEquiv_obj_iff hm Std.TreeMap.Raw.WF.ofList).1 he
    obtain ⟨⟨a, ha, hmap⟩, hcl, -⟩ := FlockSoundness.Discharge.Composed.ofJson_shape hD
    have hga := (FlockComplete.JsonRT.mem_of_getObjVal hm (FlockComplete.JsonRT.arr_ok ha)).2
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
    have hmc : m.get? "closure" = none := by
      have := hk "closure"
      rw [hnc] at this
      cases hmg : m.get? "closure" with
      | none => rfl
      | some _ => rw [hmg] at this; cases this
    have hc : D.closure = #[] := hcl (by simp only [Json.getObjVal?, hmc] <;> rfl)
    rw [_root_.Flock.Draw.UnitDraw.proved, hc, ← Except.ok.inj hmap]
    rfl

end AddressRow

end SpotFlock
