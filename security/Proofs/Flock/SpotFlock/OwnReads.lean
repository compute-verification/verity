module

public import Proofs.Flock.SpotFlock.WiringOk
public import Proofs.Flock.SpotFlock.EntriesAt
public import Proofs.Flock.SpotFlock.Sessions
public import Proofs.Flock.SpotFlock.Opens
public import Proofs.Flock.Claims.Complete.Json.Parse
public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Flock.Claims.CanonLine
public import Proofs.Flock.Claims.CanonWf
import all Lean.Data.Json.Basic

/-!
# A session's registered reads are its template's reference reads

The second stage's `second-values` (`SecondStage.valuesOk`) holds each session's registered reads to its template's reference
reads (`Held.refReads`) as canonical JSON text (`jeq`), each with the run's fields blanked (`blankReads`): an entry's
root, domain and leaves, and its rows where the run places them. An entry's `value` is never blanked. Two well-formed
values with one canonical text are equivalent field by field (`CanonLine.canon_inj`, `CanonLine.field?_equiv`), so:

* `reads_ref_value`: in an accepted run, each registered read of a checked unit's session has, at its port, an entry
  of the reference reads of a template of its template's name, whose `value` is the read's value. It reads no link or
  wire, so it holds at every port, an input port where a unit reads a registered value included.
* `ownReads_of`: where each reference read at a port no link or wire names is of a port of the interior registration
  (`refReadsOwn`), a registered read at such a port is of a port of the first stage's record (`OwnReads`), whose ports are the
  interior registration's (`recordOk`).

The session's registered reads are parsed (`FlockComplete.JsonRT.parse_wf`), so well formed. The reference reads are a
field of `Held`, and their well-formedness is the premise `RefReadsWf`.
-/

@[expose] public section

namespace SpotFlock.ReadsJson

open Lean (Json)
open Claims.Json (Wf)

/-! ## Well-formed values under `setKey` and `Json.mkObj` -/

theorem wf_obj {m : Std.TreeMap.Raw String Json compare} (h : Wf (.obj m)) :
    m.WF ∧ ∀ (k : String) (v : Json), m[k]? = some v → Wf v := by
  cases h with
  | obj hm hv => exact ⟨hm, fun k v hk => hv k v (by rw [Std.TreeMap.Raw.get?_eq_getElem?]; exact hk)⟩

theorem wf_mk {m : Std.TreeMap.Raw String Json compare} (hm : m.WF)
    (hv : ∀ (k : String) (v : Json), m[k]? = some v → Wf v) :
    Wf (.obj m) :=
  Wf.obj hm fun k v hk => hv k v (by rw [← Std.TreeMap.Raw.get?_eq_getElem?]; exact hk)

/-- An entry of a well-formed object is its value at its key. -/
theorem getElem?_of_mem {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {k : String} {v : Json}
    (hp : (k, v) ∈ m.toArray) : m[k]? = some v := by
  have hp' : (k, v) ∈ m.toList := by
    rw [← Std.TreeMap.Raw.toList_toArray]
    exact Array.mem_toList_iff.2 hp
  exact (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp'

theorem wf_setKey {j v : Json} (hj : Wf j) (hv : Wf v) (k : String) : Wf (Spot.SecondStage.setKey j k v) := by
  cases j with
  | obj m =>
    obtain ⟨hm, hvs⟩ := wf_obj hj
    show Wf (.obj (m.insert k v))
    refine wf_mk (Std.TreeMap.Raw.WF.insert hm) fun k' v' hk => ?_
    rw [Std.TreeMap.Raw.getElem?_insert hm] at hk
    split at hk
    · cases hk
      exact hv
    · exact hvs k' v' hk
  | _ => exact hj

theorem wf_mkObj {l : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct l) (hl : ∀ kv ∈ l, Wf kv.2) :
    Wf (Json.mkObj l) :=
  wf_mk Std.TreeMap.Raw.WF.ofList fun k v hk =>
    hl (k, v) (FlockComplete.JsonRT.mem_of_get?_mkObj hd (by rw [Std.TreeMap.Raw.get?_eq_getElem?]; exact hk))

theorem wf_blank : ∀ (ks : List String) (e : Json), Wf e →
    Wf (ks.foldl (fun e k => Spot.SecondStage.setKey e k .null) e)
  | [], _, he => he
  | k :: ks, _, he => wf_blank ks _ (wf_setKey he Wf.null k)

/-! ## Fields under `setKey` -/

theorem field?_setKey_ne {j v : Json} (hj : Wf j) {k k' : String} (hk : k ≠ k') :
    Spot.field? (Spot.SecondStage.setKey j k v) k' = Spot.field? j k' := by
  cases j with
  | obj m =>
    obtain ⟨hm, -⟩ := wf_obj hj
    show Spot.field? (.obj (m.insert k v)) k' = Spot.field? (.obj m) k'
    rw [Properties.Challenger.CanonLine.field?_obj, Properties.Challenger.CanonLine.field?_obj,
      Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert hm,
      ite_eq_right (fun he => hk (Std.LawfulEqOrd.compare_eq_iff_eq.1 he))]
  | _ => rfl

theorem field?_setKey_self {m : Std.TreeMap.Raw String Json compare} (hj : Wf (.obj m)) (k : String) (v : Json) :
    Spot.field? (Spot.SecondStage.setKey (.obj m) k v) k = some v := by
  obtain ⟨hm, -⟩ := wf_obj hj
  show Spot.field? (.obj (m.insert k v)) k = some v
  rw [Properties.Challenger.CanonLine.field?_obj, Std.TreeMap.Raw.get?_eq_getElem?,
    Std.TreeMap.Raw.getElem?_insert_self hm]

theorem field?_blank : ∀ (ks : List String) (e : Json), Wf e → ∀ (k' : String), k' ∉ ks →
    Spot.field? (ks.foldl (fun e k => Spot.SecondStage.setKey e k .null) e) k' = Spot.field? e k'
  | [], _, _, _, _ => rfl
  | k :: ks, e, he, k', hk => by
    have hne : k ≠ k' := by
      intro hkk
      subst hkk
      exact hk (by simp)
    have hks : k' ∉ ks := by
      intro hm
      exact hk (by simp [hm])
    exact (field?_blank ks _ (wf_setKey he Wf.null k) k' hks).trans (field?_setKey_ne he hne)

theorem field?_of_getD {j x : Json} {k : String} (h : Spot.getD j k = x) (hx : x ≠ .null) :
    Spot.field? j k = some x := by
  unfold Spot.getD at h
  cases hf : Spot.field? j k with
  | none =>
    rw [hf] at h
    exact absurd h.symm hx
  | some y =>
    rw [hf] at h
    exact congrArg some h

theorem getD_of_field? {j x : Json} {k : String} (h : Spot.field? j k = some x) : Spot.getD j k = x := by
  simp only [Spot.getD, h, Option.getD_some]

theorem obj_of_field? {j y : Json} {k : String} (h : Spot.field? j k = some y) : ∃ n, j = .obj n := by
  by_contra hn
  rw [Properties.Challenger.CanonLine.field?_of_not_obj hn] at h
  cases h

theorem field?_mkObj {l : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct l) {k : String} {v : Json}
    (hm : (k, v) ∈ l) : Spot.field? (Json.mkObj l) k = some v := by
  simp only [Spot.field?, FlockComplete.JsonRT.getObjVal?_mkObj hd hm, Except.toOption]

theorem mem_of_field?_mkObj {l : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct l) {k : String}
    {v : Json} (h : Spot.field? (Json.mkObj l) k = some v) : (k, v) ∈ l := by
  have h' : (Std.TreeMap.Raw.ofList l compare).get? k = some v := by
    rw [← Properties.Challenger.CanonLine.field?_obj]
    exact h
  exact FlockComplete.JsonRT.mem_of_get?_mkObj hd h'

/-! ## Equivalent values -/

/-- Values `jeq` holds alike have one canonical text. -/
theorem canon_of_jeq {a b : Json} (h : Spot.jeq a b = true) : ∃ s, Flock.canon a = .ok s ∧ Flock.canon b = .ok s := by
  unfold Spot.jeq at h
  cases ha : Flock.canon a with
  | error e =>
    rw [ha] at h
    simp at h
  | ok x =>
    cases hb : Flock.canon b with
    | error e =>
      rw [ha, hb] at h
      simp at h
    | ok y =>
      rw [ha, hb] at h
      refine ⟨x, rfl, ?_⟩
      rw [eq_of_beq h]

/-- **A field of one of two equivalent well-formed values** is, up to equivalence, the other's. -/
theorem field?_of_equiv {a b : Json} (ha : Wf a) (hb : Wf b) (h : FlockDeploy.JsonEquiv a b) {k : String} {x : Json}
    (hx : Spot.field? a k = some x) : ∃ y, Spot.field? b k = some y ∧ FlockDeploy.JsonEquiv x y := by
  have he := Properties.Challenger.CanonLine.field?_equiv (Properties.Challenger.CanonRead.sortedKeys_of_wf a ha)
    (Properties.Challenger.CanonRead.sortedKeys_of_wf b hb) h k
  rw [hx] at he
  cases hy : Spot.field? b k with
  | none =>
    rw [hy] at he
    simp at he
  | some y =>
    rw [hy] at he
    refine ⟨y, rfl, ?_⟩
    have he' : FlockDeploy.jsonForm x = FlockDeploy.jsonForm y := by simpa using he
    exact he'

/-! ## What `blankReads` does -/

/-- **Entry `e` of port `p` of a registered-reads file of template `t` with its run's fields null**, as
`SecondStage.blankReads` blanks it. -/
def blankEntry (own : Spot.SecondStage.Own) (t p : String) (e : Json) : Json :=
  (if Spot.SecondStage.runPlacesRows own t p e then
      Spot.SecondStage.READ_RUN_FIELDS ++ Spot.SecondStage.READ_ROW_FIELDS
    else Spot.SecondStage.READ_RUN_FIELDS).foldl (fun e k => Spot.SecondStage.setKey e k .null) e

/-- The rows blanked of an entry's map of `ports` (`blankReads`). -/
abbrev blankList (own : Spot.SecondStage.Own) (t : String) (kvs : Std.TreeMap.Raw String Json compare) :
    List (String × Json) :=
  kvs.toArray.toList.map fun pe => (pe.1, blankEntry own t pe.1 pe.2)

theorem blankReads_obj (own : Spot.SecondStage.Own) (t : String) {r : Json} {kvs : Std.TreeMap.Raw String Json compare}
    (h : Spot.getD r "ports" = .obj kvs) :
    Spot.SecondStage.blankReads own t r = Spot.SecondStage.setKey r "ports" (Json.mkObj (blankList own t kvs)) := by
  unfold Spot.SecondStage.blankReads
  rw [h]
  all_goals rfl

theorem blankReads_not_obj (own : Spot.SecondStage.Own) (t : String) {r : Json}
    (h : ∀ kvs, Spot.getD r "ports" ≠ .obj kvs) : Spot.SecondStage.blankReads own t r = r := by
  unfold Spot.SecondStage.blankReads
  cases hp : Spot.getD r "ports" with
  | obj kvs => exact absurd hp (h kvs)
  | _ => rfl

theorem value_not_blanked (own : Spot.SecondStage.Own) (t p : String) (e : Json) :
    "value" ∉ (if Spot.SecondStage.runPlacesRows own t p e then
      Spot.SecondStage.READ_RUN_FIELDS ++ Spot.SecondStage.READ_ROW_FIELDS
    else Spot.SecondStage.READ_RUN_FIELDS) := by
  split
  · decide
  · decide

theorem wf_blankEntry (own : Spot.SecondStage.Own) (t p : String) {e : Json} (he : Wf e) :
    Wf (blankEntry own t p e) :=
  wf_blank _ _ he

theorem value_blankEntry (own : Spot.SecondStage.Own) (t p : String) {e : Json} (he : Wf e) :
    Spot.field? (blankEntry own t p e) "value" = Spot.field? e "value" :=
  field?_blank _ _ he "value" (value_not_blanked own t p e)

theorem distinct_blankList (own : Spot.SecondStage.Own) (t : String) {kvs : Std.TreeMap.Raw String Json compare}
    (hm : kvs.WF) : FlockComplete.JsonRT.Distinct (blankList own t kvs) := by
  simp only [blankList, FlockComplete.JsonRT.Distinct, Std.TreeMap.Raw.toList_toArray, List.pairwise_map]
  exact Std.TreeMap.Raw.distinct_keys_toList hm

theorem wf_blankList (own : Spot.SecondStage.Own) (t : String) {kvs : Std.TreeMap.Raw String Json compare}
    (hw : Wf (.obj kvs)) : ∀ kv ∈ blankList own t kvs, Wf kv.2 := by
  obtain ⟨hm, hv⟩ := wf_obj hw
  intro kv hkv
  obtain ⟨⟨p, e⟩, hpe, rfl⟩ := List.mem_map.1 hkv
  exact wf_blankEntry own t p (hv p e (getElem?_of_mem hm (Array.mem_toList_iff.1 hpe)))

/-- **A blanked registered-reads file is well formed and has its blanked `ports`**, where its `ports` is an object. -/
theorem blankReads_spec (own : Spot.SecondStage.Own) (t : String) {r : Json} (hr : Wf r)
    {kvs : Std.TreeMap.Raw String Json compare} (h : Spot.getD r "ports" = .obj kvs) :
    Wf (.obj kvs) ∧ Wf (Spot.SecondStage.blankReads own t r) ∧
      Spot.field? (Spot.SecondStage.blankReads own t r) "ports" = some (Json.mkObj (blankList own t kvs)) := by
  have hf := field?_of_getD h (fun hn => by cases hn)
  obtain ⟨n, rfl⟩ := obj_of_field? hf
  obtain ⟨hn, hnv⟩ := wf_obj hr
  have hwk : Wf (.obj kvs) := by
    rw [Properties.Challenger.CanonLine.field?_obj, Std.TreeMap.Raw.get?_eq_getElem?] at hf
    exact hnv _ _ hf
  have hm := (wf_obj hwk).1
  have hL : Wf (Json.mkObj (blankList own t kvs)) :=
    wf_mkObj (distinct_blankList own t hm) (wf_blankList own t hwk)
  rw [blankReads_obj own t h]
  exact ⟨hwk, wf_setKey hr hL "ports", field?_setKey_self hr "ports" _⟩

/-! ## What `second-values` holds a registered read's value to -/

/-- **Two registered-reads files `jeq` holds alike once blanked have one value at each port**: each entry `e` of `S`'s
`ports` whose `value` is the string `v` has an entry of `R`'s `ports` at its port whose `value` is `v`, where both
files are well formed. -/
theorem value_of_jeq {own : Spot.SecondStage.Own} {t : String} {S R : Json} (hS : Wf S) (hR : Wf R)
    (hj : Spot.jeq (Spot.SecondStage.blankReads own t S) (Spot.SecondStage.blankReads own t R) = true)
    {p : String} {e : Json} (hm : (p, e) ∈ Spot.entries (Spot.getD S "ports")) {v : String}
    (hv : Spot.getD e "value" = .str v) :
    ∃ e', (p, e') ∈ Spot.entries (Spot.getD R "ports") ∧ Spot.getD e' "value" = .str v := by
  obtain ⟨kS, hkS⟩ : ∃ kS, Spot.getD S "ports" = .obj kS := by
    cases hp : Spot.getD S "ports" with
    | obj kS => exact ⟨kS, rfl⟩
    | _ =>
      rw [hp] at hm
      simp [Spot.entries] at hm
  rw [hkS] at hm
  have hm' : (p, e) ∈ kS.toArray := hm
  obtain ⟨hwS, hwA, hfA⟩ := blankReads_spec own t hS hkS
  obtain ⟨s, hsA, hsB⟩ := canon_of_jeq hj
  by_cases hRo : ∃ kR, Spot.getD R "ports" = .obj kR
  · obtain ⟨kR, hkR⟩ := hRo
    obtain ⟨hwR, hwB, hfB⟩ := blankReads_spec own t hR hkR
    have hE := Properties.Challenger.CanonLine.canon_inj (Properties.Challenger.CanonRead.sortedKeys_of_wf _ hwA)
      (Properties.Challenger.CanonRead.sortedKeys_of_wf _ hwB) hsA hsB
    obtain ⟨y, hy, hEy⟩ := field?_of_equiv hwA hwB hE hfA
    rw [hfB] at hy
    cases hy
    have hdS := distinct_blankList own t (wf_obj hwS).1
    have hdR := distinct_blankList own t (wf_obj hwR).1
    have hLS : Wf (Json.mkObj (blankList own t kS)) := wf_mkObj hdS (wf_blankList own t hwS)
    have hLR : Wf (Json.mkObj (blankList own t kR)) := wf_mkObj hdR (wf_blankList own t hwR)
    have hmemS : (p, blankEntry own t p e) ∈ blankList own t kS :=
      List.mem_map.2 ⟨(p, e), Array.mem_toList_iff.2 hm', rfl⟩
    obtain ⟨y', hy', hEy'⟩ := field?_of_equiv hLS hLR hEy (field?_mkObj hdS hmemS)
    obtain ⟨⟨p', e'⟩, hpe', hq⟩ := List.mem_map.1 (mem_of_field?_mkObj hdR hy')
    simp only [Prod.mk.injEq] at hq
    obtain ⟨hpp, hyy⟩ := hq
    subst hyy
    have hmR : (p', e') ∈ kR.toArray := Array.mem_toList_iff.1 hpe'
    have hwe : Wf e := (wf_obj hwS).2 _ _ (getElem?_of_mem (wf_obj hwS).1 hm')
    have hwe' : Wf e' := (wf_obj hwR).2 _ _ (getElem?_of_mem (wf_obj hwR).1 hmR)
    have hfe : Spot.field? (blankEntry own t p e) "value" = some (.str v) := by
      rw [value_blankEntry own t p hwe]
      exact field?_of_getD hv (fun hn => by cases hn)
    obtain ⟨z, hz, hEz⟩ := field?_of_equiv (wf_blankEntry own t p hwe) (wf_blankEntry own t p' hwe') hEy' hfe
    rw [value_blankEntry own t p' hwe'] at hz
    have hzv : z = .str v := Properties.Challenger.CanonRead.equiv_str hEz.symm
    subst hzv
    refine ⟨e', ?_, getD_of_field? hz⟩
    rw [hkR, ← hpp]
    exact hmR
  · exfalso
    have hB : Spot.SecondStage.blankReads own t R = R :=
      blankReads_not_obj own t fun kR hk => hRo ⟨kR, hk⟩
    rw [hB] at hsB
    have hE := Properties.Challenger.CanonLine.canon_inj (Properties.Challenger.CanonRead.sortedKeys_of_wf _ hwA)
      (Properties.Challenger.CanonRead.sortedKeys_of_wf _ hR) hsA hsB
    obtain ⟨y, hy, hEy⟩ := field?_of_equiv hwA hR hE hfA
    have hfo : FlockDeploy.jsonForm y = .obj _ :=
      (Eq.symm hEy).trans (FlockDeploy.jsonForm_obj (Std.TreeMap.Raw.ofList (blankList own t kS) compare))
    obtain ⟨kR, rfl⟩ := Properties.Challenger.CanonRead.form_obj hfo
    exact hRo ⟨kR, getD_of_field? hy⟩

end SpotFlock.ReadsJson

namespace SpotFlock

open Lean (Json)
open Claims.Json (Wf)
open SpotFlock.ReadsJson

/-! ## An accepted run's registered reads -/

/-- A file the verifier reads as JSON is well formed. -/
theorem wf_of_jsonOf {what : String} {b : ByteArray} {j : Json} (h : FlockDeploy.jsonOf what b = .ok j) : Wf j := by
  unfold FlockDeploy.jsonOf at h
  cases hu : String.fromUTF8? b with
  | none =>
    rw [hu] at h
    cases h
  | some s =>
    rw [hu] at h
    dsimp only at h
    cases hp : Flock.JsonParse.parse s with
    | error e =>
      rw [hp] at h
      cases h
    | ok j' =>
      rw [hp] at h
      have hj : j' = j := Except.ok.inj h
      subst hj
      exact FlockComplete.JsonRT.parse_wf s j' hp

variable {h : Held}

/-- The second stage's statement check passes `second-values`. -/
theorem values_of_codesOk {own : Spot.SecondStage.Own} {t : ℕ} {S : Spot.SecondStage.Stmt}
    (hc : codesOk h own t S = true) :
    Spot.SecondStage.valuesOk { own with verified := S.checked.toArray } S = true := by
  unfold codesOk at hc
  obtain ⟨-, -, -, -, -, -, -, -, -, -, hv⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  exact hv

/-- **The reference reads `second-values` holds a statement to** are a template's of the statement's template's name. -/
theorem refReads_of_values {own : Spot.SecondStage.Own} (hown : own.refReads = refRegisteredOf h)
    {S : Spot.SecondStage.Stmt} (hv : Spot.SecondStage.valuesOk own S = true) :
    ∃ t' < h.ms.length, h.names.getD t' "" = S.template ∧
      Spot.jeq (Spot.SecondStage.blankReads own S.template S.registered)
        (Spot.SecondStage.blankReads own S.template (h.refReads t')) = true := by
  unfold Spot.SecondStage.valuesOk at hv
  cases hr : Spot.SecondStage.refReadsOf own S.template with
  | none =>
    rw [hr] at hv
    cases hv
  | some R =>
    rw [hr] at hv
    unfold Spot.SecondStage.refReadsOf at hr
    rw [hown] at hr
    obtain ⟨⟨T, R'⟩, hf, hR⟩ := Option.map_eq_some_iff.1 hr
    have hT : T = S.template := by simpa using Array.find?_some hf
    have hmem := Array.mem_of_find?_eq_some hf
    simp only [refRegisteredOf, List.mem_toArray, List.mem_map, List.mem_range, Prod.mk.injEq] at hmem
    obtain ⟨t', ht', hn, hrr⟩ := hmem
    have hRR : R = h.refReads t' := by
      rw [hrr]
      exact hR.symm
    subst hRR
    exact ⟨t', ht', hn.trans hT, hv⟩

/-- **The reference reads are well formed**: each template's, as the auditor's params give them, parsed. -/
def RefReadsWf (h : Held) : Prop := ∀ t < h.ms.length, Wf (h.refReads t)

/-- **An accepted run's registered reads are its template's reference reads, value for value**: at a checked unit of an
accepted run, each registered read of its session has, at its port, an entry of the reference reads of a template of
its template's name, whose `value` is the read's value. The second stage's `second-values` holds the session's registered-reads
file to those reference reads with the run's fields blanked, and a `value` is not one of them. -/
theorem reads_ref_value (hW : RefReadsWf h) {r : CheckedRun h.stage} (hr : r.Ok h)
    {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port}
    (hown : r.ins.registered = some own) (hreg : reg ∈ own) :
    ∃ t' < h.ms.length, h.names.getD t' "" = h.names.getD (templateOf h.ms r.u) "" ∧
      ∃ e, (reg.name, e) ∈ Spot.entries (Spot.getD (h.refReads t') "ports") ∧
        Spot.getD e "value" = .str reg.value := by
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, hc⟩ := readsOk_true h hat.2
  obtain ⟨bR, jR, hbR, hjR, hoj⟩ := registered_of_ins (ins_of_setupOf hw hset) hown
  obtain ⟨hSt, -, hSr, -⟩ := stmtJsonOf_fields h hS
  have hSj : S.registered = jR := by
    have h4 := hSr bR hbR
    rw [hjR] at h4
    exact (Option.some.inj h4).symm
  obtain ⟨kvs, hports, hent, -⟩ := portsOf_entries (portsOf_of_ofJson hoj)
  obtain ⟨ej, hm, -, -, -, -, hv⟩ := hent reg hreg
  obtain ⟨t', ht', hnt, hj⟩ := refReads_of_values (h := h)
    (own := { ownOf h st rec draw _ with verified := S.checked.toArray }) rfl (values_of_codesOk hc)
  have hWS : Wf S.registered := by
    rw [hSj]
    exact wf_of_jsonOf hjR
  have hmS : (reg.name, ej) ∈ Spot.entries (Spot.getD S.registered "ports") := by
    rw [hSj, getD_of_ok hports]
    exact hm
  obtain ⟨e', he', hv'⟩ := value_of_jeq hWS (hW t' ht') hj hmS (getD_of_ok hv)
  exact ⟨t', ht', hnt.trans hSt, e', he', hv'⟩

/-! ## The first stage's record's ports -/

theorem mem_of_mapM (f : Json → Option String) :
    ∀ (l : List Json) (ps : List String), l.mapM f = some ps → ∀ p ∈ ps, ∃ e ∈ l, f e = some p
  | [], ps, h, p, hp => by
    have hps : ps = [] := by simpa using h.symm
    subst hps
    cases hp
  | a :: l, ps, h, p, hp => by
    cases hfa : f a with
    | none => simp [List.mapM_cons, hfa] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [List.mapM_cons, hfa, hl] at h
      | some bs =>
        have hps : ps = b :: bs := by simpa [List.mapM_cons, hfa, hl] using h.symm
        subst hps
        rcases List.mem_cons.1 hp with rfl | hp'
        · exact ⟨a, by simp, hfa⟩
        · obtain ⟨e, he, hfe⟩ := mem_of_mapM f l bs hl p hp'
          exact ⟨e, by simp [he], hfe⟩

/-- **A port a record's roots name** is named by one of its roots' entries. -/
theorem port_of_namesAt {rec : Json} {ps : List String} (h : namesAt rec "roots" "port" = some ps) {p : String}
    (hp : p ∈ ps) : ∃ rs, Spot.getD rec "roots" = .arr rs ∧ ∃ e ∈ rs.toList, Spot.getD e "port" = .str p := by
  unfold namesAt at h
  obtain ⟨e, he, hfe⟩ := mem_of_mapM _ _ _ h p hp
  try dsimp only at hfe
  cases hr : Spot.getD rec "roots" with
  | arr rs =>
    rw [hr] at he
    refine ⟨rs, rfl, e, he, ?_⟩
    cases hq : Spot.getD e "port" with
    | str s =>
      rw [hq] at hfe
      have hs : s = p := by simpa using hfe
      rw [hs]
    | _ =>
      rw [hq] at hfe
      simp at hfe
  | _ =>
    rw [hr] at he
    simp at he

theorem find?_isSome_of {acc : Array (String × Json)} {p : String} (h : ∃ q ∈ acc, q.1 = p) :
    (acc.find? (·.1 == p)).isSome = true := by
  obtain ⟨q, hq, rfl⟩ := h
  cases hf : acc.find? (·.1 == q.1) with
  | none => exact (Array.find?_eq_none.1 hf q hq (by simp)).elim
  | some _ => rfl

/-- A fold that keeps an entry of key `p` once it has one, and adds one at each element `P` holds of, ends with one
where it starts with one or `P` holds of an element. -/
theorem foldl_keeps {α : Type} (F : Array (String × Json) → α → Array (String × Json)) (P : α → Prop) (p : String)
    (hkeep : ∀ acc a, (∃ q ∈ acc, q.1 = p) → ∃ q ∈ F acc a, q.1 = p)
    (hadd : ∀ acc a, P a → ∃ q ∈ F acc a, q.1 = p) :
    ∀ (l : List α) (acc : Array (String × Json)), ((∃ q ∈ acc, q.1 = p) ∨ ∃ a ∈ l, P a) →
      ∃ q ∈ l.foldl F acc, q.1 = p
  | [], acc, hl => by
    rcases hl with hl | ⟨a, ha, -⟩
    · exact hl
    · cases ha
  | a :: l, acc, hl => by
    rw [List.foldl_cons]
    refine foldl_keeps F P p hkeep hadd l (F acc a) ?_
    rcases hl with hl | ⟨b, hb, hPb⟩
    · exact Or.inl (hkeep acc a hl)
    · rcases List.mem_cons.1 hb with rfl | hb'
      · exact Or.inl (hadd acc b hPb)
      · exact Or.inr ⟨b, hb', hPb⟩

/-- **A port a record's roots name is one of its ports** (`Record.byPort`). -/
theorem byPort_has {rec : Json} {rs : Array Json} (hr : Spot.getD rec "roots" = .arr rs) {e : Json}
    (he : e ∈ rs.toList) {p : String} (hp : Spot.getD e "port" = .str p) :
    ((Spot.Record.byPort rec).find? (·.1 == p)).isSome = true := by
  apply find?_isSome_of
  unfold Spot.Record.byPort
  rw [hr]
  try dsimp only
  rw [← Array.foldl_toList]
  refine foldl_keeps _ (fun a => Spot.getD a "port" = .str p) p ?_ ?_ rs.toList #[] (Or.inr ⟨e, he, hp⟩)
  · intro acc a hacc
    obtain ⟨q, hq, hqp⟩ := hacc
    try dsimp only
    cases hpa : Spot.getD a "port" with
    | str p' =>
      show ∃ q ∈ (acc.filter fun (q : String × Json) => q.1 != p').push (p', a), q.1 = p
      by_cases hpp : p' = p
      · exact ⟨(p', a), by simp, hpp⟩
      · refine ⟨q, ?_, hqp⟩
        simp only [Array.mem_push, Array.mem_filter]
        refine Or.inl ⟨hq, ?_⟩
        rw [hqp]
        simpa using Ne.symm hpp
    | _ => exact ⟨q, hq, hqp⟩
  · intro acc a ha
    try dsimp only at ha ⊢
    rw [ha]
    exact ⟨(p, a), by simp, rfl⟩

/-! ## `OwnReads` from the reference reads -/

/-- **`OwnReads` from `refReadsOwn`**: at a checked unit of an accepted run, a registered read of its session at a
port no link or wire names has its template's reference reads' value there (`reads_ref_value`), which `refReadsOwn`
holds to one of the interior registration's ports, and the first stage's record names exactly those ports among its roots
(`recordOk`, which the statement check holds it to), so it is one of the record's ports. -/
theorem ownReads_of (hc : refReadsOwn h = true) (hW : RefReadsWf h) : OwnReads h := by
  unfold refReadsOwn at hc
  intro r hr own reg hown hreg hl hw rec hrec
  obtain ⟨t', ht', hnt, e, he, hv⟩ := reads_ref_value hW hr hown hreg
  have h1 := List.all_eq_true.1 hc t' (List.mem_range.2 ht')
  have h2 := Array.all_eq_true_iff_forall_mem.1 h1 (reg.name, e) he
  try dsimp only at h2
  simp only [hnt, hl, hw, hv, Option.isSome_none, Bool.false_or] at h2
  have hpv : reg.value ∈ h.interiorNames.ports := by
    exact List.contains_iff_mem.1 h2
  obtain ⟨Pt, d, spec, hx, hP, h2', hin, hok, -⟩ := hr
  obtain ⟨st, rec', -, hrec', -, hrok⟩ := recordsOk_spec h (recordsOk_of_stmtOk h hok)
  rw [hrec] at hrec'
  cases hrec'
  obtain ⟨⟨ps, hps, -, hmem⟩, -⟩ := recordOk_spec hrok
  obtain ⟨rs, hrs, e', he', hpe⟩ := port_of_namesAt hps ((hmem _).2 hpv)
  exact byPort_has hrs he' hpe

end SpotFlock
