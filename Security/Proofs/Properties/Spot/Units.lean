module

public import Spot.OneStageZk
public import Flock.With.SetupEq
public import Proofs.Flock.Claims.JsonNorm
public import Proofs.Flock.Claims.Complete.Json.Parse
public import Proofs.Flock.Claims.Recursion.SetupValues
import all Lean.Data.Json.Basic

/-!
# v2's units, read off the values the verifier keeps

One-stage v2 credits a session the public file's unit indices of the instances its record draws
(`Spot.OneStageZk.provedUnits`): it reads the record's `unit_draw` as `Flock.JsonParse.parse` built it, and the
public file's header as `Spot.FilePart.header` reads the file's first line. The session keeps both rebuilt (`Flock.norm`):
the draw as `Flock.Witness.recordDraw` keeps it, and the header as `Flock.HmIn.loadPublic` keeps it. The two readings
agree:

* `getD_norm`, `nat?_norm`: a field of a rebuilt well-formed value is the field rebuilt, and a number reads the same
  rebuilt; `wf_getD`: a field of a well-formed value is well formed;
* `headerOf_of`: v2's `FilePart.header` reads the header `RecDeploy.SetupValues.headerOf` reads, its first line up to the
  first newline (`headerLine_of_findByte`), so it reads the header the session's public file load keeps
  (`headerOf_loadPublic`, through `RecDeploy.SetupValues.loadPublic_counts`);
* `provedUnits_eq`: at a record whose draw the verifier keeps as `D` and a public file whose header v2 reads as `h`,
  v2's units are `unitsAt h D`, the units read off the rebuilt draw; `provedUnits_of_loadPublic`,
  `unitsAt_of_provedUnits`: at the header the load keeps.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open Lean (Json)
open Spot (getD field? nat?)


namespace Properties.Spot.Units

/-! ## Fields of rebuilt values -/

theorem field?_obj (m : Std.TreeMap.Raw String Json compare) (k : String) : field? (.obj m) k = m.get? k := by
  simp only [field?, Json.getObjVal?]
  cases m.get? k <;> rfl

theorem get?_of_getObjVal {m : Std.TreeMap.Raw String Json compare} {k : String} {v : Json}
    (h : (Json.obj m).getObjVal? k = .ok v) : m.get? k = some v := by
  simp only [Json.getObjVal?] at h
  cases hm : m.get? k with
  | none => rw [hm] at h; cases h
  | some a => rw [hm] at h; cases h; rfl

/-- A well-formed object rebuilt has its fields rebuilt at each key. -/
theorem get?_norm_obj {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (k : String) :
    (Flock.normObj (m.toList.map fun p => (p.1, Flock.norm p.2))).get? k = (m.get? k).map Flock.norm := by
  have hM : (m.map fun _ v => Flock.norm v).WF := ⟨⟨Std.DTreeMap.Internal.Impl.WF.map hm.out.out⟩⟩
  have hL : m.toList.map (fun p => (p.1, Flock.norm p.2)) = (m.map fun _ v => Flock.norm v).toList :=
    (Std.TreeMap.Raw.toList_map (f := fun _ v => Flock.norm v)).symm
  have he : (Flock.normObj (m.toList.map fun p => (p.1, Flock.norm p.2))).Equiv (m.map fun _ v => Flock.norm v) :=
    (Std.TreeMap.Raw.equiv_iff_toList_eq (FlockDeploy.wf_normObj _) hM).2
      (by rw [FlockDeploy.toList_normObj, hL, FlockDeploy.toList_ofList_toList hM])
  rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.get?_eq_getElem?,
    Std.TreeMap.Raw.Equiv.getElem?_eq (FlockDeploy.wf_normObj _) hM he, Std.TreeMap.Raw.getElem?_map hm]

/-- **A field of a rebuilt well-formed value is the field rebuilt.** -/
theorem getD_norm {u : Json} (hu : Claims.Json.Wf u) (k : String) :
    getD (Flock.norm u) k = Flock.norm (getD u k) := by
  cases hu with
  | obj hm _ =>
    rw [FlockDeploy.norm_obj]
    simp only [getD, field?_obj, get?_norm_obj hm]
    cases (Std.TreeMap.Raw.get? _ k) <;> rfl
  | arr _ => rw [FlockDeploy.norm_arr]; rfl
  | _ => rfl

/-- **A number reads the same rebuilt.** -/
theorem nat?_norm (e : Json) : nat? (Flock.norm e) = nat? e := by
  cases e with
  | arr a => rw [FlockDeploy.norm_arr]; rfl
  | obj m => rw [FlockDeploy.norm_obj]; rfl
  | _ => rfl

/-- A field of a well-formed value is well formed. -/
theorem wf_getD {u : Json} (hu : Claims.Json.Wf u) (k : String) : Claims.Json.Wf (getD u k) := by
  cases hu with
  | obj hm h =>
    simp only [getD, field?_obj]
    cases hk : Std.TreeMap.Raw.get? _ k with
    | none => exact .null
    | some v => exact h k v hk
  | _ => exact .null

/-! ## Reading in `V` and in `Option` -/

theorem list_mapM_toOption {α β ε : Type} (G : α → Except ε β) (F : α → Option β)
    (hG : ∀ x, (G x).toOption = F x) (l : List α) : (l.mapM G).toOption = l.mapM F := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.mapM_cons]
    rw [← hG a, ← ih]
    cases G a <;> cases List.mapM G l <;> rfl

theorem mapM_toOption {α β ε : Type} (G : α → Except ε β) (F : α → Option β) (hG : ∀ x, (G x).toOption = F x)
    (xs : Array α) : (xs.mapM G).toOption = xs.mapM F := by
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList, ← list_mapM_toOption G F hG]
  cases List.mapM G xs.toList <;> rfl

theorem ok_bind {α β ε : Type} (a : α) (f : α → Except ε β) : ((Except.ok a : Except ε α) >>= f) = f a := rfl

theorem toOption_bind_pure {α β ε : Type} (x : Except ε α) (f : α → β) :
    (x >>= fun a => pure (f a)).toOption = x.toOption.map f := by
  cases x <;> rfl

/-! ## The public file's header -/

theorem forIn_find {β : Type} (l : List Nat) (p : Nat → Bool) (g : Nat → β) :
    (forIn (m := Id) l ((none : Option β), ()) fun i (_ : Option β × Unit) =>
      if p i = true then pure (ForInStep.done (some (g i), ())) else pure (ForInStep.yield (none, ()))) =
    ((l.find? p).map g, ()) := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.forIn_cons, List.find?_cons]
    cases h : p a
    · simpa [h] using ih
    · simp only [↓reduceIte]; rfl

theorem getElem!_eq_get! (b : ByteArray) (i : Nat) : b[i]! = b.get! i := by
  obtain ⟨bs⟩ := b
  show getElem! _ _ = bs[i]!
  rw [getElem!_def, getElem!_def, getElem?_def, getElem?_def]
  rfl

theorem findByte_eq (b : ByteArray) (x : UInt8) :
    Flock.findByte b x = (List.range' 0 b.size).find? (fun i => b.get! i == x) := by
  unfold Flock.findByte
  simp only [Std.Legacy.Range.forIn_eq_forIn_range']
  rw [forIn_find]
  have hs : ([:b.size] : Std.Legacy.Range).size = b.size := by simp [Std.Legacy.Range.size]
  rw [hs]
  cases List.find? _ _ <;> rfl

theorem headerLine_eq (b : ByteArray) :
    Spot.FilePart.headerLine b =
      ((List.range' 0 b.size).find? (fun i => b[i]! == 10)).elim b (b.extract 0) := by
  unfold Spot.FilePart.headerLine
  simp only [Std.Legacy.Range.forIn_eq_forIn_range']
  rw [forIn_find]
  have hs : ([:b.size] : Std.Legacy.Range).size = b.size := by simp [Std.Legacy.Range.size]
  rw [hs]
  cases List.find? _ _ <;> rfl

/-- v2's header line is the file up to its first newline. -/
theorem headerLine_of_findByte {b : ByteArray} {nl : Nat} (h : Flock.findByte b 10 = some nl) :
    Spot.FilePart.headerLine b = b.extract 0 nl := by
  rw [headerLine_eq]
  rw [findByte_eq] at h
  simp only [getElem!_eq_get!]
  rw [h]
  rfl

/-- **v2 reads the public file's header as the bridge reads it.** -/
theorem headerOf_of {f : ByteArray} {hd : Json} (h : RecDeploy.SetupValues.headerOf f = some hd) :
    Spot.FilePart.header f = .ok hd := by
  unfold RecDeploy.SetupValues.headerOf at h
  cases hnl : Flock.findByte f 10 with
  | none => simp only [hnl] at h; cases h
  | some nl =>
    simp only [hnl] at h
    cases ht : String.fromUTF8? (f.extract 0 nl) with
    | none => simp only [ht] at h; cases h
    | some text =>
      simp only [ht] at h
      cases hj : Flock.JsonParse.parse text with
      | error e => simp only [hj] at h; cases h
      | ok j =>
        simp only [hj] at h
        cases h
        unfold Spot.FilePart.header
        rw [headerLine_of_findByte hnl, ht]
        simp only [hj]
        rfl

/-- **v2 reads the header the session's public file load keeps** (`HmIn.loadPublic`, as `Flock.Witness` loads it). -/
theorem headerOf_loadPublic {tags : Flock.Tags} {c : Flock.Circuit} {pi : Flock.HmIn.Ports} {f : ByteArray}
    {P : Flock.Public} (hp : Flock.HmIn.loadPublic tags c pi f = .ok P) :
    Spot.FilePart.header f = .ok P.header := by
  rw [← Flock.With.inLoadPublicWith_std] at hp
  obtain ⟨hd, he, -, hh⟩ :=
    RecDeploy.SetupValues.loadPublic_counts _ tags _ f P (RecDeploy.SetupValues.hmInLoad_row _ tags c pi f P hp)
  rw [hh]
  exact headerOf_of he

/-! ## The units -/

/-- **The units a draw proves, at a public header**: the header's `units.indices` (`FilePart.indicesOf`) at the
instances the draw `D` names, ascending, as v2 credits a session's (`Spot.OneStageZk.provedUnits`). -/
def unitsAt (h D : Json) : Option (Array Nat) := do
  let is ← (Spot.FilePart.indicesOf h).toOption
  let .arr ds := getD D "units" | none
  let units ← ds.mapM fun e => nat? e >>= (is[·]?)
  pure (units.qsort (· < ·))

/-- **v2's units are the units read off the draw the verifier keeps**: at a record whose draw the verifier keeps as `D`
(`Flock.Witness.recordDraw`) and a public file whose header v2 reads as `h`. -/
theorem provedUnits_eq {pubB : ByteArray} {record : String} {h D : Json}
    (hh : Spot.FilePart.header pubB = .ok h) (hD : Flock.Witness.recordDraw record = some D) :
    (Spot.OneStageZk.provedUnits pubB record).toOption = unitsAt h D := by
  unfold Flock.Witness.recordDraw at hD
  split at hD
  · rename_i r hr
    split at hD
    · cases hD
    · rename_i u _ hu
      cases hD
      cases FlockComplete.JsonRT.parse_wf _ _ hr with
      | @obj m hm hv =>
        have hg := get?_of_getObjVal hu
        have hwu := hv _ _ hg
        have hdraw : getD (Json.obj m) "unit_draw" = u := by simp only [getD, field?_obj, hg]; rfl
        unfold Spot.OneStageZk.provedUnits unitsAt
        simp only [hh, ok_bind]
        cases hI : Spot.FilePart.indicesOf h with
        | error e => rfl
        | ok is =>
          simp only [getD_norm hwu, hr, ok_bind, hdraw]
          generalize getD u "units" = X
          cases X with
          | arr ds =>
            simp only [FlockDeploy.norm_arr, Array.mapM_map, Function.comp_def, nat?_norm]
            rw [toOption_bind_pure, mapM_toOption _ (fun e => (nat? e).bind fun i => is[i]?)]
            · show _ = Array.mapM (fun e => (nat? e).bind fun i => is[i]?) ds >>= fun units =>
                pure (units.qsort (· < ·))
              cases Array.mapM (fun e => (nat? e).bind fun i => is[i]?) ds <;> rfl
            · intro x
              split <;> (rename_i heq; exact heq.symm)
          | obj m' => simp only [FlockDeploy.norm_obj]; rfl
          | _ => rfl
      | _ => simp [Json.getObjVal?] at hu
    · cases hD
  · cases hD

/-- **v2's units at the session's public file**: the units read off the kept draw at the kept header. -/
theorem provedUnits_of_loadPublic {tags : Flock.Tags} {c : Flock.Circuit} {pi : Flock.HmIn.Ports} {pubB : ByteArray}
    {P : Flock.Public} {record : String} {D : Json} (hp : Flock.HmIn.loadPublic tags c pi pubB = .ok P)
    (hD : Flock.Witness.recordDraw record = some D) :
    (Spot.OneStageZk.provedUnits pubB record).toOption = unitsAt P.header D :=
  provedUnits_eq (headerOf_loadPublic hp) hD

/-- Where v2 credits a session units `us`, they are the units read off the kept draw at the kept header. -/
theorem unitsAt_of_provedUnits {tags : Flock.Tags} {c : Flock.Circuit} {pi : Flock.HmIn.Ports} {pubB : ByteArray}
    {P : Flock.Public} {record : String} {D : Json} {us : Array Nat} (hp : Flock.HmIn.loadPublic tags c pi pubB = .ok P)
    (hD : Flock.Witness.recordDraw record = some D)
    (hu : Spot.OneStageZk.provedUnits pubB record = .ok us) : unitsAt P.header D = some us := by
  rw [← provedUnits_of_loadPublic hp hD, hu]
  rfl

end Properties.Spot.Units
