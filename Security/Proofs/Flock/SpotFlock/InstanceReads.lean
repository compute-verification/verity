module

public import Proofs.Flock.SpotFlock.InstanceAddresses
public import Proofs.Flock.SpotFlock.DrawnAt
public import Proofs.Flock.SpotFlock.HeaderRoots
public import Proofs.Flock.SpotFlock.EntriesAt
public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Flock.Claims.Complete.Json.Stmt
import all Lean.Data.Json.Basic

/-!
# The ports an accepted run's session registers

Round 2's values check (`SecondStage.valuesOk`) holds a statement's registered reads, each port's run fields blanked
(`SecondStage.blankReads`), to its template's reference reads blanked the same. Blanking sets fields of each port's
entry and keeps the ports, so a statement the check passes registers only ports the reference reads name
(`port_mem_of_jeq`).

At `SpotFlock.Instance.held R B desc q c₀`, every registered read of an accepted run's checked session is so named
after a port of `R.reads` (`regName_mem`), and a port `R.reads` doesn't name is no registered read's (`regNamed_none`):
the session reads it under the replay block's roots.
-/

@[expose] public section

namespace SpotFlock.Instance

open Claims Claims.Sampling SamplingDeploy Properties.Spot
open Lean (Json)

/-- Setting a field keeps a well-formed value well formed. -/
theorem wf_setKey {j : Json} (k : String) {v : Json} (hj : Claims.Json.Wf j) (hv : Claims.Json.Wf v) :
    Claims.Json.Wf (Spot.SecondStage.setKey j k v) := by
  cases hj with
  | obj hm hw =>
    refine .obj hm.insert fun k' v' h => ?_
    rw [Std.TreeMap.Raw.get?_eq_getElem?, Std.TreeMap.Raw.getElem?_insert hm] at h
    split at h
    · cases h
      exact hv
    · exact hw k' v' h
  | null => exact .null
  | bool b => exact .bool b
  | num n => exact .num n
  | str s => exact .str s
  | arr h => exact .arr h

theorem wf_foldl_setKey (ks : List String) {e : Json} (he : Claims.Json.Wf e) :
    Claims.Json.Wf (ks.foldl (fun e k => Spot.SecondStage.setKey e k .null) e) := by
  induction ks generalizing e with
  | nil => exact he
  | cons k ks ih => exact ih (wf_setKey k he .null)

theorem wf_mkObj {l : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct l)
    (hw : ∀ p ∈ l, Claims.Json.Wf p.2) : Claims.Json.Wf (Json.mkObj l) :=
  .obj Std.TreeMap.Raw.WF.ofList fun _ _ hv => hw _ (FlockComplete.JsonRT.mem_of_get?_mkObj hd hv)

theorem getObjVal_setKey {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (k : String) (v : Json) :
    (Spot.SecondStage.setKey (.obj m) k v).getObjVal? k = .ok v :=
  FlockComplete.JsonRT.getObjVal_of_get (by
    rw [Std.TreeMap.Raw.get?_eq_getElem?]
    exact Std.TreeMap.Raw.getElem?_insert_self hm)

/-- A field `Spot.getD` reads as anything but `null` is present. -/
theorem getObjVal_of_getD {j : Json} {k : String} (h : Spot.getD j k ≠ .null) :
    j.getObjVal? k = .ok (Spot.getD j k) := by
  unfold Spot.getD Spot.field? at *
  cases hj : j.getObjVal? k with
  | error e => simp [hj, Except.toOption] at h
  | ok v => simp [hj, Except.toOption]

/-- A key an object built from a list holds is one of the list's. -/
theorem mem_keys_of_getObjVal {l : List (String × Json)} {k : String} {v : Json}
    (h : (Json.mkObj l).getObjVal? k = .ok v) : k ∈ l.map Prod.fst := by
  cases hc : (l.map Prod.fst).contains k with
  | true => exact List.contains_iff_mem.1 hc
  | false =>
    exfalso
    have hn := FlockComplete.JsonRT.get?_mkObj_none hc
    simp only [Json.mkObj, Json.getObjVal?, hn] at h
    all_goals cases h

/-- Round 2's blanking of entry `e` of port `p` (`SecondStage.blankReads`): its run fields set to `null`. -/
def blankE (own : Spot.SecondStage.Own) (t p : String) (e : Json) : Json :=
  (if Spot.SecondStage.runPlacesRows own t p e then Spot.SecondStage.READ_RUN_FIELDS ++
    Spot.SecondStage.READ_ROW_FIELDS else Spot.SecondStage.READ_RUN_FIELDS).foldl
      (fun e k => Spot.SecondStage.setKey e k .null) e

/-- The `ports` object `SecondStage.blankReads` puts in place of `m`: each entry blanked, under its port. -/
def blankList (own : Spot.SecondStage.Own) (t : String) (m : Std.TreeMap.Raw String Json compare) :
    List (String × Json) :=
  m.toArray.toList.map fun pe => (pe.1, blankE own t pe.1 pe.2)

theorem blankReads_obj {own : Spot.SecondStage.Own} {t : String} {r : Json}
    {m : Std.TreeMap.Raw String Json compare} (hg : Spot.getD r "ports" = .obj m) :
    Spot.SecondStage.blankReads own t r =
      Spot.SecondStage.setKey r "ports" (Json.mkObj (blankList own t m)) := by
  unfold Spot.SecondStage.blankReads
  rw [hg]
  all_goals rfl

theorem fst_blankList (own : Spot.SecondStage.Own) (t : String) (m : Std.TreeMap.Raw String Json compare) :
    (blankList own t m).map Prod.fst = m.toArray.toList.map Prod.fst := by
  unfold blankList
  rw [List.map_map]
  all_goals rfl

theorem distinct_blankList (own : Spot.SecondStage.Own) (t : String) {m : Std.TreeMap.Raw String Json compare}
    (hm : m.WF) : FlockComplete.JsonRT.Distinct (blankList own t m) := by
  have hd := Std.TreeMap.Raw.distinct_keys_toList hm
  rw [← Std.TreeMap.Raw.toList_toArray] at hd
  unfold blankList
  exact List.Pairwise.map _ (fun _ _ h => h) hd

theorem wf_blankList (own : Spot.SecondStage.Own) (t : String) {m : Std.TreeMap.Raw String Json compare}
    (hw : Claims.Json.Wf (.obj m)) : ∀ p ∈ blankList own t m, Claims.Json.Wf p.2 := by
  cases hw with
  | obj hm hv =>
    intro p hp
    obtain ⟨pe, hpe, rfl⟩ := List.mem_map.1 hp
    rw [Std.TreeMap.Raw.toList_toArray] at hpe
    have hg := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hpe
    exact wf_foldl_setKey _ (hv pe.1 pe.2 (by rw [Std.TreeMap.Raw.get?_eq_getElem?]; exact hg))

/-- Round 2's blanking keeps a well-formed value well formed. -/
theorem wf_blankReads {own : Spot.SecondStage.Own} {t : String} {r : Json} (hr : Claims.Json.Wf r) :
    Claims.Json.Wf (Spot.SecondStage.blankReads own t r) := by
  cases hg : Spot.getD r "ports"
  case obj m =>
    rw [blankReads_obj hg]
    have hp : r.getObjVal? "ports" = .ok (.obj m) := by
      rw [← hg]
      exact getObjVal_of_getD (by rw [hg]; nofun)
    have hw := HeaderRoots.wf_of_getObjVal hr hp
    have hm : m.WF := by cases hw with | obj hm _ => exact hm
    exact wf_setKey _ hr (wf_mkObj (distinct_blankList own t hm) (wf_blankList own t hw))
  all_goals
    unfold Spot.SecondStage.blankReads
    rw [hg]
    exact hr

/-- **A statement round 2's values check passes registers only the reference's ports**: where the registered reads
`a`, blanked (`SecondStage.blankReads`), are the reference reads `b` blanked, each port `a` names is one `b` names. -/
theorem port_mem_of_jeq {own : Spot.SecondStage.Own} {t : String} {a b : Json} (hwa : Claims.Json.Wf a)
    (hwb : Claims.Json.Wf b)
    (hj : Spot.jeq (Spot.SecondStage.blankReads own t a) (Spot.SecondStage.blankReads own t b) = true)
    {kvs : Std.TreeMap.Raw String Json compare} (hp : a.getObjVal? "ports" = .ok (.obj kvs)) {port : String}
    {e : Json} (he : (port, e) ∈ kvs.toArray) :
    port ∈ (Spot.entries (Spot.getD b "ports")).map (·.1) := by
  have hA := wf_blankReads (own := own) (t := t) hwa
  have hB := wf_blankReads (own := own) (t := t) hwb
  obtain ⟨sa, hsa⟩ := DrawnAt.canon_of_jeq hj
  obtain ⟨sb, hsb⟩ := DrawnAt.canon_of_jeq (jeq_symm hj)
  have heq := DrawnAt.jsonEquiv_of_jeq (DrawnAt.good_of_canon _ hA sa hsa) (DrawnAt.good_of_canon _ hB sb hsb) hj
  obtain ⟨ma, rfl⟩ := FlockComplete.JsonRT.obj_of_getObjVal hp
  have hma : ma.WF := by cases hwa with | obj hm _ => exact hm
  have hwk := HeaderRoots.wf_of_getObjVal hwa hp
  have hkvs : kvs.WF := by cases hwk with | obj hm _ => exact hm
  have hdA := distinct_blankList own t hkvs
  have hAp : (Spot.SecondStage.blankReads own t (.obj ma)).getObjVal? "ports" =
      .ok (Json.mkObj (blankList own t kvs)) := by
    rw [blankReads_obj (HeaderRoots.getD_of_getObjVal hp)]
    exact getObjVal_setKey hma _ _
  obtain ⟨w, hBp, hAw, hww⟩ := HeaderRoots.getObjVal_equiv hA hB heq hAp
  have hmem : (port, blankE own t port e) ∈ blankList own t kvs :=
    List.mem_map.2 ⟨(port, e), Array.mem_toList_iff.2 he, rfl⟩
  obtain ⟨wv, hwv, -, -⟩ := HeaderRoots.getObjVal_equiv (wf_mkObj hdA (wf_blankList own t hwk)) hww hAw
    (FlockComplete.JsonRT.getObjVal?_mkObj hdA hmem)
  cases hgb : Spot.getD b "ports"
  case obj mb =>
    have hpb : b.getObjVal? "ports" = .ok (.obj mb) := by
      rw [← hgb]
      exact getObjVal_of_getD (by rw [hgb]; nofun)
    obtain ⟨mb₀, rfl⟩ := FlockComplete.JsonRT.obj_of_getObjVal hpb
    have hmb₀ : mb₀.WF := by cases hwb with | obj hm _ => exact hm
    rw [blankReads_obj hgb, getObjVal_setKey hmb₀] at hBp
    cases hBp
    have hk := mem_keys_of_getObjVal hwv
    rw [fst_blankList, Std.TreeMap.Raw.toList_toArray] at hk
    show port ∈ mb.toArray.map (·.1)
    rw [Array.mem_map]
    obtain ⟨pe, hpe, hpk⟩ := List.mem_map.1 hk
    exact ⟨pe, by rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray]; exact hpe, hpk⟩
  all_goals
    have hBr : Spot.SecondStage.blankReads own t b = b := by
      unfold Spot.SecondStage.blankReads
      rw [hgb]
    rw [hBr] at hBp
    have hw' := HeaderRoots.getD_of_getObjVal hBp
    rw [hgb] at hw'
    subst hw'
    cases hwv

variable {R : Ref} {B : Bnd} {desc : Msg} {q : Query} {c₀ : Code}

/-- **An accepted run's session registers only the reference's ports**: each registered read of the checked unit's
session is named after a port the reference reads `R.reads` name, where `R.reads` is well formed. -/
theorem regName_mem {r : CheckedRun} (hr : r.Ok (held R B desc q c₀)) (hwr : Claims.Json.Wf R.reads)
    {own : Array _root_.Flock.Registered.Port} (hown : r.ins.registered = some own)
    {reg : _root_.Flock.Registered.Port} (hreg : reg ∈ own) :
    reg.name ∈ (Spot.entries (Spot.getD R.reads "ports")).map (·.1) := by
  obtain ⟨_, _, _, hx, -, -, -, hok, hu, -, -, -, -, hset⟩ := hr
  have hat := stmtOkAt_unit _ hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, w, S, -, -, -, hw, hS, hc⟩ := readsOk_true _ hat.2
  obtain ⟨bR, jR, hbR, hjR, hoj⟩ := registered_of_ins (ins_of_setupOf hw hset) hown
  obtain ⟨-, -, hSr, -⟩ := stmtJsonOf_fields _ hS
  have hSj : S.registered = jR := by
    have h4 := hSr bR hbR
    rw [hjR] at h4
    exact (Option.some.inj h4).symm
  obtain ⟨kvs, hports, hrds, -⟩ := portsOf_entries (portsOf_of_ofJson hoj)
  obtain ⟨e, hm, -⟩ := hrds reg hreg
  obtain ⟨-, -, -, -, -, -, -, -, -, -, hv⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  obtain ⟨rr, hrr, hj⟩ := Spot.SecondStage.valuesOk_ref hv
  have hrr' : (S.template, rr) ∈ #[(R.name, R.reads)] := hrr
  simp only [Array.mem_singleton, Prod.mk.injEq] at hrr'
  obtain ⟨-, rfl⟩ := hrr'
  rw [hSj] at hj
  exact port_mem_of_jeq (DrawnAt.wf_of_jsonOf hjR) hwr hj hports hm

/-- **A port the reference reads don't name is read under the replay block's roots**: no registered read of an
accepted run's checked session is named like it. -/
theorem regNamed_none {r : CheckedRun} (hr : r.Ok (held R B desc q c₀)) (hwr : Claims.Json.Wf R.reads) {p : ℕ}
    (hp : (r.sts[0]!.c.ports[p]!).name ∉ (Spot.entries (Spot.getD R.reads "ports")).map (·.1)) :
    regNamed r.ins r.sts p = none := by
  unfold regNamed
  cases hown : r.ins.registered with
  | none => rfl
  | some own =>
    show own.find? _ = none
    rw [Array.find?_eq_none]
    intro reg hreg hn
    have hnm : reg.name = (r.sts[0]!.c.ports[p]!).name := by simpa using hn
    apply hp
    rw [← hnm]
    exact regName_mem hr hwr hown hreg

end SpotFlock.Instance
