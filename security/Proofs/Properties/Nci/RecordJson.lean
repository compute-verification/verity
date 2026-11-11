module

public import NciCheck.Record
public import Std.Data.TreeMap.Raw.Lemmas
public import Std.Data.TreeMap.Raw.WF
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-!
# What SPOT's reader reads back of the wiring the record check writes

The record check hands SPOT's reader (`Spot.Wires.Wiring.ofJson`) a JSON object of its own making (`Record.piece`):
the text's cuts and wires, with the templates it read off the Program, each with its units, its rows and its outputs.
This module shows the reader reads back what was written: `piece_templates`, each template of the wiring is one of the
check's templates, with its units, its rows by name with their words and outer lengths, and its outputs by name with
their words, and every template of the check is one of the wiring's.

The steps are about JSON objects (`Json`, below): a field of an object built from a list with distinct keys is the
list's pair (`getObjVal?_mkObj`), its fields are the list's pairs (`mem_fields_mkObj`), and a field of two objects
merged is the second's where it has the key (`getObjVal?_mergeObj`). The keys are distinct by the check's own tests:
no two templates of the text have one id, and no two ports of a template one name (`tmplOf_names`). What a
template of the check is, of its Definition's parameters and return type, is `tmplOf_spec`.
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs.Json

open Lean (Json toJson)
open Spot.Wires

theorem cmp_ne {a b : String} (h : a ≠ b) : ¬ compare a b = .eq := fun hc =>
  h (Std.LawfulEqCmp.eq_of_compare hc)

/-- A key of a list with distinct keys is found in the object `mkObj` builds from it. -/
theorem getElem?_ofList {kvs : List (String × Json)} (hd : kvs.Pairwise (fun a b => a.1 ≠ b.1)) {k : String}
    {v : Json} (hm : (k, v) ∈ kvs) : (Std.TreeMap.Raw.ofList kvs compare)[k]? = some v :=
  Std.TreeMap.Raw.getElem?_ofList_of_mem Std.ReflCmp.compare_self (hd.imp fun h => cmp_ne h) hm

theorem getElem?_ofList_none {kvs : List (String × Json)} {k : String} (hk : k ∉ kvs.map (·.1)) :
    (Std.TreeMap.Raw.ofList kvs compare)[k]? = none :=
  Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false (by simpa using hk)

/-- What an object built from a list with distinct keys holds at a key: the list's pair. -/
theorem getElem?_ofList_iff {kvs : List (String × Json)} (hd : kvs.Pairwise (fun a b => a.1 ≠ b.1)) (k : String)
    (v : Json) : (Std.TreeMap.Raw.ofList kvs compare)[k]? = some v ↔ (k, v) ∈ kvs := by
  refine ⟨fun h => ?_, getElem?_ofList hd⟩
  by_cases hk : k ∈ kvs.map (·.1)
  · obtain ⟨⟨k', v'⟩, hm, rfl⟩ := List.mem_map.1 hk
    rw [getElem?_ofList hd hm] at h
    cases h
    exact hm
  · rw [getElem?_ofList_none hk] at h
    cases h

theorem getObjVal?_obj (t : Std.TreeMap.Raw String Json) (k : String) (v : Json) :
    (Json.obj t).getObjVal? k = .ok v ↔ t[k]? = some v := by
  simp only [Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  cases h : t[k]? with
  | none => simp [throw, throwThe, MonadExceptOf.throw]
  | some w => simp [pure, Except.pure]

/-- **A field of an object built from a list** with distinct keys is the list's pair of that key. -/
theorem getObjVal?_mkObj {kvs : List (String × Json)} (hd : kvs.Pairwise (fun a b => a.1 ≠ b.1)) (k : String)
    (v : Json) : (Json.mkObj kvs).getObjVal? k = .ok v ↔ (k, v) ∈ kvs := by
  unfold Json.mkObj
  rw [getObjVal?_obj, getElem?_ofList_iff hd]

theorem field_ok (j : Json) (k : String) (v : Json) : Spot.Wires.field j k = .ok v ↔ j.getObjVal? k = .ok v := by
  unfold Spot.Wires.field
  cases h : j.getObjVal? k with
  | error e => simp
  | ok w => simp

/-- **The fields of an object built from a list** with distinct keys are the list's pairs. -/
theorem mem_fields_mkObj {kvs : List (String × Json)} (hd : kvs.Pairwise (fun a b => a.1 ≠ b.1))
    {l : List (String × Json)} (h : Spot.Wires.fields (Json.mkObj kvs) = .ok l) (k : String) (v : Json) :
    (k, v) ∈ l ↔ (k, v) ∈ kvs := by
  unfold Json.mkObj Spot.Wires.fields at h
  cases h
  have hf : (fun (x : String × Json) => match x with | (k, v) => (k, v)) = id := by
    funext ⟨a, b⟩; rfl
  rw [hf, Array.map_id, Array.mem_toList_iff,
    Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some Std.TreeMap.Raw.WF.ofList, getElem?_ofList_iff hd]

theorem nat?_toJson (n : Nat) : nat? (toJson n) = some n := by
  simp [nat?, int?, toJson, Lean.JsonNumber.fromNat]

/-- Pairs inserted one by one into a well-formed map: the map stays well formed, a key of none of them keeps its
value, and where their keys are distinct each pair is found. -/
theorem foldl_insert : ∀ (es : List (String × Json)) (t : Std.TreeMap.Raw String Json), t.WF →
    (es.foldl (fun a b => a.insert b.1 b.2) t).WF ∧
    (∀ k, k ∉ es.map (·.1) → (es.foldl (fun a b => a.insert b.1 b.2) t)[k]? = t[k]?) ∧
    (es.Pairwise (fun a b => a.1 ≠ b.1) → ∀ k v, (k, v) ∈ es →
      (es.foldl (fun a b => a.insert b.1 b.2) t)[k]? = some v)
  | [], t, ht => ⟨ht, fun _ _ => rfl, fun _ _ _ h => by cases h⟩
  | e :: es, t, ht => by
    obtain ⟨h1, h2, h3⟩ := foldl_insert es (t.insert e.1 e.2) ht.insert
    refine ⟨h1, fun k hk => ?_, fun hd k v hm => ?_⟩
    · have hk' : k ≠ e.1 ∧ k ∉ es.map (·.1) := by simpa using hk
      rw [List.foldl_cons, h2 k hk'.2, Std.TreeMap.Raw.getElem?_insert ht, ite_eq_right (cmp_ne (Ne.symm hk'.1))]
    · rw [List.pairwise_cons] at hd
      rw [List.foldl_cons]
      rcases List.mem_cons.1 hm with rfl | hm
      · have hk : k ∉ es.map (·.1) := fun hk => by
          obtain ⟨x, hx, hxk⟩ := List.mem_map.1 hk
          exact hd.1 x hx hxk.symm
        rw [h2 k hk, Std.TreeMap.Raw.getElem?_insert ht, ite_eq_left Std.ReflCmp.compare_self]
      · exact h3 hd.2 k v hm

/-- **A field of two objects merged**, the second built from a list with distinct keys: the list's pair where the
list has the key, else the first object's. -/
theorem getObjVal?_mergeObj {t : Std.TreeMap.Raw String Json} (ht : t.WF) {kvs : List (String × Json)}
    (hd : kvs.Pairwise (fun a b => a.1 ≠ b.1)) (k : String) (v : Json) :
    ((Json.obj t).mergeObj (Json.mkObj kvs)).getObjVal? k = .ok v ↔
      if k ∈ kvs.map (·.1) then (k, v) ∈ kvs else t[k]? = some v := by
  unfold Json.mergeObj Json.mkObj
  rw [getObjVal?_obj, Std.TreeMap.Raw.foldl_eq_foldl_toList]
  have hw : (Std.TreeMap.Raw.ofList kvs compare).WF := Std.TreeMap.Raw.WF.ofList
  have hmem : ∀ k v, (k, v) ∈ (Std.TreeMap.Raw.ofList kvs compare).toList ↔ (k, v) ∈ kvs := fun k v => by
    rw [Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hw, getElem?_ofList_iff hd]
  have hdist : (Std.TreeMap.Raw.ofList kvs compare).toList.Pairwise (fun a b => a.1 ≠ b.1) :=
    (Std.TreeMap.Raw.distinct_keys_toList hw).imp fun h he => h (by rw [he]; exact Std.ReflCmp.compare_self)
  have hkeys : k ∈ (Std.TreeMap.Raw.ofList kvs compare).toList.map (·.1) ↔ k ∈ kvs.map (·.1) := by
    simp only [List.mem_map]
    exact ⟨fun ⟨x, hx, hxk⟩ => ⟨x, (hmem x.1 x.2).1 hx, hxk⟩,
      fun ⟨x, hx, hxk⟩ => ⟨x, (hmem x.1 x.2).2 hx, hxk⟩⟩
  obtain ⟨-, h2, h3⟩ := foldl_insert (Std.TreeMap.Raw.ofList kvs compare).toList t ht
  by_cases hk : k ∈ kvs.map (·.1)
  · rw [ite_eq_left hk]
    obtain ⟨⟨k', v'⟩, hm, rfl⟩ := List.mem_map.1 hk
    rw [h3 hdist _ v' ((hmem _ v').2 hm)]
    refine ⟨fun h => ?_, fun h => ?_⟩
    · cases h; exact hm
    · have := (getElem?_ofList_iff hd _ v).2 h
      rw [getElem?_ofList hd hm] at this
      exact this
  · rw [ite_eq_right hk, h2 k (fun h => hk (hkeys.1 h))]

/-- What a `mapM` in `Except` gives: the results of its list's elements. -/
theorem mapM_ok {α β : Type} (f : α → Except String β) : ∀ (l : List α) (l' : List β), l.mapM f = .ok l' →
    ∀ y, y ∈ l' ↔ ∃ x ∈ l, f x = .ok y
  | [], l', h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    simp
  | x :: l, l', h => by
    rw [List.mapM_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    · rename_i y hy
      split at h
      · cases h
      · rename_i ys hys
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        intro z
        rw [List.mem_cons, mapM_ok f l ys hys z]
        constructor
        · rintro (rfl | ⟨x', hx', hz⟩)
          · exact ⟨x, List.mem_cons_self, hy⟩
          · exact ⟨x', List.mem_cons_of_mem _ hx', hz⟩
        · rintro ⟨x', hx', hz⟩
          rcases List.mem_cons.1 hx' with rfl | hx'
          · rw [hy] at hz; cases hz; exact Or.inl rfl
          · exact Or.inr ⟨x', hx', hz⟩

theorem bind_ok {α β : Type} {x : Except String α} {f : α → Except String β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- Where a `mapM` in `Except` succeeds, each of its list's elements has a result. -/
theorem mapM_ok_all {α β : Type} (f : α → Except String β) :
    ∀ (l : List α) (l' : List β), l.mapM f = .ok l' → ∀ x ∈ l, ∃ y, f x = .ok y
  | [], _, _ => fun _ h => by cases h
  | x :: l, l', h => by
    rw [List.mapM_cons] at h
    obtain ⟨y, hy, h⟩ := bind_ok h
    obtain ⟨ys, hys, -⟩ := bind_ok h
    intro z hz
    rcases List.mem_cons.1 hz with rfl | hz
    · exact ⟨y, hy⟩
    · exact mapM_ok_all f l ys hys z hz

/-- A template as the record check writes it for SPOT's reader: its units, its rows `[words, outer]` by name and its
outputs' words by name. -/
def tmplJson (units : Nat) (rows : List (String × Nat × Nat)) (outs : List (String × Nat)) : Json :=
  Json.mkObj [
    ("units", toJson units),
    ("rows", Json.mkObj (rows.map fun (p, words, outer) => (p, Json.arr #[toJson words, toJson outer]))),
    ("outs", Json.mkObj (outs.map fun (o, words) => (o, toJson words)))]

theorem keys3 (a b c : Json) :
    [("units", a), ("rows", b), ("outs", c)].Pairwise (fun (x y : String × Json) => x.1 ≠ y.1) := by
  simp

theorem field_mkObj {kvs : List (String × Json)} (hd : kvs.Pairwise (fun a b => a.1 ≠ b.1)) (k : String)
    (v : Json) : Spot.Wires.field (Json.mkObj kvs) k = .ok v ↔ (k, v) ∈ kvs := by
  rw [field_ok, getObjVal?_mkObj hd]

/-- **What SPOT's reader reads of a template the record check wrote**: its units, its rows and its outputs, where no
two rows and no two outputs have one name. -/
theorem template_ofJson {units : Nat} {rows : List (String × Nat × Nat)} {outs : List (String × Nat)}
    (hr : rows.Pairwise (fun a b => a.1 ≠ b.1)) (ho : outs.Pairwise (fun a b => a.1 ≠ b.1)) {t : Template}
    (h : Template.ofJson (tmplJson units rows outs) = .ok t) :
    t.units = units ∧
      (∀ p q, (p, q) ∈ t.rows ↔
        ∃ words outer, (p, words, outer) ∈ rows ∧ q.words = words ∧ q.outer = outer) ∧
      ∀ o n, (o, n) ∈ t.outs ↔ (o, n) ∈ outs := by
  unfold Template.ofJson at h
  replace h := bind_ok h
  obtain ⟨-, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨u, hu, h⟩ := h
  dsimp only at h
  split at h
  · obtain ⟨_, hx, -⟩ := bind_ok h
    cases hx
  replace h := bind_ok h
  obtain ⟨jr, hjr, h⟩ := h
  replace h := bind_ok h
  obtain ⟨lr, hlr, h⟩ := h
  replace h := bind_ok h
  obtain ⟨rows', hrows, h⟩ := h
  replace h := bind_ok h
  obtain ⟨jo, hjo, h⟩ := h
  replace h := bind_ok h
  obtain ⟨lo, hlo, h⟩ := h
  replace h := bind_ok h
  obtain ⟨outs', houts, h⟩ := h
  cases h
  have hk := keys3 (toJson units)
    (Json.mkObj (rows.map fun (p, words, outer) => (p, Json.arr #[toJson words, toJson outer])))
    (Json.mkObj (outs.map fun (o, words) => (o, toJson words)))
  have hr' : (rows.map fun (p, words, outer) => (p, Json.arr #[toJson words, toJson outer])).Pairwise
      (fun a b => a.1 ≠ b.1) := List.pairwise_map.2 hr
  have ho' : (outs.map fun (o, words) => (o, toJson words)).Pairwise (fun a b => a.1 ≠ b.1) :=
    List.pairwise_map.2 ho
  have hu' : u = units := by
    unfold natAt at hu
    obtain ⟨ju, hju, hu⟩ := bind_ok hu
    have := (field_mkObj hk "units" ju).1 hju
    simp at this
    subst this
    simp [nat?_toJson] at hu
    exact hu.symm
  have hjr' := (field_mkObj hk "rows" jr).1 hjr
  simp at hjr'
  subst hjr'
  have hjo' := (field_mkObj hk "outs" jo).1 hjo
  simp at hjo'
  subst hjo'
  have hmr := mem_fields_mkObj hr' hlr
  have hmo := mem_fields_mkObj ho' hlo
  refine ⟨hu', fun p q => ?_, fun o n => ?_⟩
  · show (p, q) ∈ rows' ↔ _
    rw [mapM_ok _ lr rows' hrows (p, q)]
    constructor
    · rintro ⟨⟨p', v⟩, hx, hF⟩
      obtain ⟨⟨p'', w, o⟩, hm, he⟩ := List.mem_map.1 ((hmr p' v).1 hx)
      cases he
      simp [arrOf, natOf, nat?_toJson, bind, Except.bind, pure, Except.pure] at hF
      split at hF
      · cases hF
      · cases hF
        exact ⟨w, o, hm, rfl, rfl⟩
    · rintro ⟨w, o, hm, hw, ho⟩
      have hx : (p, Json.arr #[toJson w, toJson o]) ∈ lr :=
        (hmr _ _).2 (List.mem_map.2 ⟨(p, w, o), hm, rfl⟩)
      obtain ⟨y, hy⟩ := mapM_ok_all _ lr rows' hrows _ hx
      refine ⟨_, hx, ?_⟩
      rw [hy]
      simp [arrOf, natOf, nat?_toJson, bind, Except.bind, pure, Except.pure] at hy
      split at hy
      · cases hy
      · cases hy
        cases q
        simp_all
  · show (o, n) ∈ outs' ↔ _
    rw [mapM_ok _ lo outs' houts (o, n)]
    constructor
    · rintro ⟨⟨o', v⟩, hx, hF⟩
      obtain ⟨⟨o'', w⟩, hm, he⟩ := List.mem_map.1 ((hmo o' v).1 hx)
      cases he
      simp [natOf, nat?_toJson, bind, Except.bind, pure, Except.pure] at hF
      split at hF
      · cases hF
      · cases hF
        exact hm
    · intro hm
      have hx : (o, toJson n) ∈ lo := (hmo _ _).2 (List.mem_map.2 ⟨(o, n), hm, rfl⟩)
      obtain ⟨y, hy⟩ := mapM_ok_all _ lo outs' houts _ hx
      refine ⟨_, hx, ?_⟩
      rw [hy]
      simp [natOf, nat?_toJson, bind, Except.bind, pure, Except.pure] at hy
      split at hy
      · cases hy
      · cases hy
        rfl

/-- **What SPOT's reader reads as the templates** of a wiring whose `templates` the record check wrote over a base
object: each template's text, read. -/
theorem wiring_templates {base : Std.TreeMap.Raw String Json} (hb : base.WF) {ts rest : List (String × Json)}
    (hd : ts.Pairwise (fun a b => a.1 ≠ b.1))
    (hrest : (("templates", Json.mkObj ts) :: rest).Pairwise (fun a b => a.1 ≠ b.1)) {W : Wiring}
    (h : Wiring.ofJson ((Json.obj base).mergeObj (Json.mkObj (("templates", Json.mkObj ts) :: rest))) = .ok W)
    :
    (∀ tid t, (tid, t) ∈ W.templates ↔ ∃ v, (tid, v) ∈ ts ∧ Template.ofJson v = .ok t) ∧
      ∀ tid v, (tid, v) ∈ ts → ∃ t, Template.ofJson v = .ok t := by
  unfold Wiring.ofJson at h
  replace h := bind_ok h
  obtain ⟨jt, hjt, h⟩ := h
  replace h := bind_ok h
  obtain ⟨lt, hlt, h⟩ := h
  replace h := bind_ok h
  obtain ⟨ts', hts, h⟩ := h
  replace h := bind_ok h
  obtain ⟨jc, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨lc, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨cs, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨jw, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨lw, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨wires, -, h⟩ := h
  have hW : W.templates = ts' := by
    split at h
    · replace h := bind_ok h
      obtain ⟨a1, -, h⟩ := h
      replace h := bind_ok h
      obtain ⟨a2, -, h⟩ := h
      replace h := bind_ok h
      obtain ⟨a3, -, h⟩ := h
      cases h
      rfl
    · replace h := bind_ok h
      obtain ⟨a3, -, h⟩ := h
      cases h
      rfl
  have hjt' : jt = Json.mkObj ts := by
    have h1 := (field_ok _ _ _).1 hjt
    have h2 := (getObjVal?_mergeObj hb hrest "templates" (Json.mkObj ts)).2 (by simp)
    rw [h1] at h2
    cases h2
    rfl
  subst hjt'
  have hm := mem_fields_mkObj hd hlt
  refine ⟨fun tid t => ?_, fun tid v hv => ?_⟩
  · rw [hW, mapM_ok _ lt ts' hts (tid, t)]
    constructor
    · rintro ⟨⟨k, v⟩, hx, hF⟩
      replace hF := bind_ok hF
      obtain ⟨t', ht', hF⟩ := hF
      cases hF
      exact ⟨v, (hm _ _).1 hx, ht'⟩
    · rintro ⟨v, hv, ht⟩
      refine ⟨(tid, v), (hm _ _).2 hv, ?_⟩
      show (Template.ofJson v >>= fun t => pure (tid, t)) = _
      rw [ht]
      rfl
  · obtain ⟨y, hy⟩ := mapM_ok_all _ lt ts' hts _ ((hm _ _).2 hv)
    replace hy := bind_ok hy
    obtain ⟨t, ht, -⟩ := hy
    exact ⟨t, ht⟩

end NciCheck.Record.SecurityProofs.Json

namespace NciCheck.Record.SecurityProofs

open Lean (Json toJson)
open Flock.Program (Prog)
open Flock.Sigma (bits)
open NciCheck.RecordText (Tmpl tmplOf wiresOf fieldsOf outerOf)

/-- `eraseDups` keeps a list's length only where the list has no duplicates. -/
theorem nodup_of_eraseDups : ∀ (n : Nat) (l : List String), l.length ≤ n →
    l.eraseDups.length ≤ l.length ∧ (l.eraseDups.length = l.length → l.Nodup)
  | _, [], _ => by simp
  | 0, _ :: _, h => by simp at h
  | n + 1, a :: as, h => by
    have hf : (as.filter fun b => !b == a).length ≤ as.length := List.length_filter_le ..
    obtain ⟨ih1, ih2⟩ := nodup_of_eraseDups n (as.filter fun b => !b == a) (by simp at h; omega)
    rw [List.eraseDups_cons, List.length_cons, List.length_cons]
    refine ⟨by omega, fun he => ?_⟩
    have hfl : (as.filter fun b => !b == a).length = as.length := by omega
    have hall := List.length_filter_eq_length_iff.1 hfl
    have hfe : (as.filter fun b => !b == a) = as := List.filter_eq_self.2 hall
    rw [hfe] at ih2 he
    refine List.nodup_cons.2 ⟨fun hm => ?_, ih2 (by omega)⟩
    have := hall a hm
    simp at this

/-- **No two ports of a template have one name**: its rows' names and its outputs', together, as the text's reader
tests them. -/
theorem tmplOf_names {P : Prog} {r : Flock.TemplateQuery.TRange} {t : Tmpl} (h : tmplOf P r = .ok t) :
    (t.rows.map (·.1) ++ t.outs.map (·.1)).Nodup := by
  unfold tmplOf at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rename_i hne
  exact (nodup_of_eraseDups _ _ (Nat.le_refl _)).2 (Classical.not_not.1 fun hc => hne (bne_iff_ne.2 hc))

/-- A `mapM` of a test and a value: where it succeeds, every element passes the test and the result is the values. -/
theorem mapM_test {α β : Type} (c : α → Bool) (g : α → β) (e : String) :
    ∀ (l : List α) (l' : List β),
      l.mapM (fun x => if c x = true then Except.error e else Except.ok (g x)) = .ok l' →
        l' = l.map g ∧ ∀ x ∈ l, c x = false
  | [], l', h => by
    rw [List.mapM_nil] at h
    cases h
    exact ⟨rfl, fun _ hx => by cases hx⟩
  | x :: l, l', h => by
    rw [List.mapM_cons] at h
    replace h := Json.bind_ok h
    obtain ⟨y, hy, h⟩ := h
    replace h := Json.bind_ok h
    obtain ⟨ys, hys, h⟩ := h
    cases h
    obtain ⟨ih1, ih2⟩ := mapM_test c g e l ys hys
    by_cases hc : c x = true
    · rw [ite_eq_left hc] at hy
      cases hy
    · rw [ite_eq_right hc] at hy
      cases hy
      refine ⟨by rw [ih1]; rfl, fun z hz => ?_⟩
      rcases List.mem_cons.1 hz with rfl | hz
      · simpa using hc
      · exact ih2 z hz

/-- **What the text's reader gives as a template**: of the range `r`, the Definition `d` its root node calls, the
range's template and count, a row a parameter of `d`, with its words and its outer length, each parameter whole
16-bit words and no fewer than one, and an output a field of `d`'s return type, named `o<j>`, with its bits rounded
up to words, each field of at least one bit. -/
theorem tmplOf_spec {P : Prog} {r : Flock.TemplateQuery.TRange} {t : Tmpl} (h : tmplOf P r = .ok t) :
    t.defn = (((P.defs[P.root]!).body.nodes[r.node]?).map (·.fn)).getD 0 ∧ t.tid = r.template ∧
      t.units = r.count ∧
      t.rows = (P.defs[t.defn]!).params.toList.map (fun p => (p.1, bits p.2 / 16, outerOf p.2)) ∧
      (∀ p ∈ (P.defs[t.defn]!).params.toList, bits p.2 ≠ 0 ∧ bits p.2 % 16 = 0) ∧
      t.outs = (fieldsOf (P.defs[t.defn]!).ret).zipIdx.map (fun x => ("o" ++ toString x.2, (bits x.1 + 15) / 16)) ∧
      ∀ ty ∈ fieldsOf (P.defs[t.defn]!).ret, bits ty ≠ 0 := by
  unfold tmplOf at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rename_i _ rows hrows _ outs houts _
  obtain ⟨hr1, hr2⟩ := mapM_test _ _ _ _ _ hrows
  obtain ⟨ho1, ho2⟩ := mapM_test _ _ _ _ _ houts
  refine ⟨rfl, rfl, rfl, hr1, fun p hp => ?_, ?_, fun ty hty => ?_⟩
  · have := hr2 p hp
    simp at this
    exact this
  · rw [ho1]
    rfl
  · obtain ⟨j, hj⟩ := List.mem_iff_getElem?.1 hty
    have := ho2 (ty, j) (List.mem_zipIdx_iff_getElem?.2 hj)
    simp at this
    exact this

/-- **SPOT's reader reads back the templates the record check writes.** Where no two of the check's templates have
one id and no two ports of one have one name, each template of the wiring read from `piece` is one of the check's:
it has its units, its rows by name, each with its words and its outer length, and its outputs by name, each with its
words. And each of the check's templates is one of the wiring's. -/
theorem piece_templates {f : Front} (hd : (f.ts.map (·.tid)).Nodup)
    (hn : ∀ t ∈ f.ts, (t.rows.map (·.1) ++ t.outs.map (·.1)).Nodup) {W : Spot.Wires.Wiring}
    (h : Spot.Wires.Wiring.ofJson (piece f) = .ok W) :
    (∀ tid t', (tid, t') ∈ W.templates → ∃ t ∈ f.ts, t.tid = tid ∧ t'.units = t.units ∧
        (∀ p q, (p, q) ∈ t'.rows ↔
          ∃ words outer, (p, words, outer) ∈ t.rows ∧ q.words = words ∧ q.outer = outer) ∧
        ∀ o n, (o, n) ∈ t'.outs ↔ (o, n) ∈ t.outs) ∧
      ∀ t ∈ f.ts, ∃ t', (t.tid, t') ∈ W.templates := by
  have hts : (f.ts.map fun t => (t.tid, Json.tmplJson t.units t.rows t.outs)).Pairwise (fun a b => a.1 ≠ b.1) :=
    List.pairwise_map.2 (List.pairwise_map.1 hd)
  have key : ∃ rest : List (String × Lean.Json),
      (("templates", Lean.Json.mkObj (f.ts.map fun t => (t.tid, Json.tmplJson t.units t.rows t.outs))) :: rest).Pairwise
        (fun a b => a.1 ≠ b.1) ∧
      piece f = (Lean.Json.obj (Std.TreeMap.Raw.ofList [("templates", Lean.Json.mkObj []),
          ("cuts", (f.doc.getObjVal? "cuts").toOption.getD (Lean.Json.mkObj [])),
          ("wires", RecordText.valueAt f.doc "wires")] compare)).mergeObj
        (Lean.Json.mkObj (("templates",
          Lean.Json.mkObj (f.ts.map fun t => (t.tid, Json.tmplJson t.units t.rows t.outs))) :: rest)) := by
    refine ⟨match (f.doc.getObjVal? "order").toOption, f.i.order with
      | some o, _ => [("order", o)]
      | none, some o => [("order", o)]
      | none, none => [], ?_, rfl⟩
    split <;> simp
  obtain ⟨rest, hrest, hp⟩ := key
  rw [hp] at h
  obtain ⟨h1, h2⟩ := Json.wiring_templates Std.TreeMap.Raw.WF.ofList hts hrest h
  refine ⟨fun tid t' hm => ?_, fun t ht => ?_⟩
  · obtain ⟨v, hv, ht'⟩ := (h1 tid t').1 hm
    obtain ⟨t, ht, he⟩ := List.mem_map.1 hv
    cases he
    have hnt := List.nodup_append.1 (hn t ht)
    obtain ⟨hu, hr, ho⟩ := Json.template_ofJson (List.pairwise_map.1 hnt.1) (List.pairwise_map.1 hnt.2.1) ht'
    exact ⟨t, ht, rfl, hu, hr, ho⟩
  · obtain ⟨t', ht'⟩ := h2 t.tid _ (List.mem_map.2 ⟨t, ht, rfl⟩)
    exact ⟨t', (h1 _ _).2 ⟨_, List.mem_map.2 ⟨t, ht, rfl⟩, ht'⟩⟩

end NciCheck.Record.SecurityProofs
