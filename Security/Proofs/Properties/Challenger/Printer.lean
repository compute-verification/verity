import Proofs.Properties.Challenger.CanonRead
import Proofs.Flock.Claims.Complete.Json.RoundTrip

/-!
# Canonical JSON reads back, whichever way it prints strings

`CanonRead.canon_read`'s argument holds of any printer of canonical JSON's form (`Printer P S`): `P` prints null,
booleans, integers, lists and objects as `Flock.canon` does, its strings and keys printed by `S`, whose texts start with
a quote (`Printer.head`) and end where they end (`Printer.split`). Two sorted values that `P` prints alike are then
equivalent (`Printer.inj`).

`Flock.canonUtf8` is one (`utf8`), and so is `Flock.canon` (`canon`), which writes every character outside printable
ASCII as a `\u` escape: its strings end where they end because the verifier's parser `Flock.JsonParse.str` reads each
back (`FlockComplete.JsonRT.ReadBack.esc_of`), escape by escape. So two sorted values with one `Flock.canon` text are
equivalent (`canon_inj`), and a release log line `ChallengerSession.event` accepts names the value its device printed.
-/

namespace Properties.Challenger.Printer

open Lean (Json JsonNumber)
open SampledProofs (sortedKeys)
open FlockDeploy (JsonEquiv jsonForm jsonForm_obj)
open Properties.Challenger.CanonRead (Stop Starts stop_cons inter_read mapM_forall₂ forall₂_mem_right forall₂_comb
  forall₂_map_eq forall₂_imp digits_ne nat_text int_text cls kind int_split Reads jsonForm_arr obj_lt arr_lt keyLt
  qsort_keys sortedKeys_arr sortedKeys_obj increasing_iff)

/-- **A printer of canonical JSON's form**, strings printed by `S`: `Flock.canon`'s equations, with a string's text
starting with a quote and ending where it ends. -/
structure _root_.Properties.Challenger.Printer (P : Json → Flock.V String) (S : String → String) : Prop where
  null : P .null = pure "null"
  bool : ∀ b, P (.bool b) = pure (if b then "true" else "false")
  num : ∀ {n : JsonNumber} {s : String}, P (.num n) = .ok s → n.exponent = 0 ∧ s = toString n.mantissa
  str : ∀ x, P (.str x) = pure (S x)
  arr : ∀ a : Array Json, P (.arr a) = (do let parts ← a.mapM P; pure ("[" ++ ",".intercalate parts.toList ++ "]"))
  obj : ∀ kvs : Std.TreeMap.Raw String Json, P (.obj kvs) =
    (do
      let vals ← kvs.toArray.mapM fun p => do pure (p.1, ← P p.2)
      let items := vals.qsort (fun a b => a.1 < b.1)
      pure ("{" ++ ",".intercalate (items.toList.map fun kv => S kv.1 ++ ":" ++ kv.2) ++ "}"))
  head : ∀ x, ∃ rest, (S x).toList = '"' :: rest
  split : ∀ {s s' : String} {r r' : List Char}, (S s).toList ++ r = (S s').toList ++ r' → s = s' ∧ r = r'

variable {P : Json → Flock.V String} {S : String → String}

/-! ## The texts a printer prints -/

theorem text_null (hP : Printer P S) {s : String} (h : P .null = .ok s) : s.toList = ['n', 'u', 'l', 'l'] := by
  rw [hP.null] at h
  cases h
  rfl

theorem text_bool (hP : Printer P S) {b : Bool} {s : String} (h : P (.bool b) = .ok s) :
    s.toList = if b then ['t', 'r', 'u', 'e'] else ['f', 'a', 'l', 's', 'e'] := by
  rw [hP.bool] at h
  cases h
  cases b <;> rfl

theorem text_str (hP : Printer P S) {x s : String} (h : P (.str x) = .ok s) : s = S x := by
  rw [hP.str] at h
  cases h
  rfl

theorem text_arr (hP : Printer P S) {a : Array Json} {s : String} (h : P (.arr a) = .ok s) :
    ∃ ps : List String, a.toList.mapM P = .ok ps ∧
      s.toList = '[' :: ([','].intercalate (ps.map String.toList) ++ [']']) := by
  rw [hP.arr] at h
  cases hm : a.mapM P with
  | error e => simp [hm, bind, Except.bind] at h
  | ok parts =>
    simp only [hm, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨parts.toList, ?_, ?_⟩
    · rw [← Array.toList_mapM, hm]; rfl
    · simp [String.toList_append, String.toList_intercalate]

/-- An object's item as `S` prints its key: the key, a colon and its value's text. -/
def item (S : String → String) (kv : String × String) : List Char := (S kv.1).toList ++ ':' :: kv.2.toList

theorem text_obj (hP : Printer P S) {kvs : Std.TreeMap.Raw String Json} {s : String} (h : P (.obj kvs) = .ok s) :
    ∃ vals : Array (String × String),
      kvs.toArray.toList.mapM (fun p => do pure (p.1, ← P p.2)) = .ok vals.toList ∧
      s.toList = '{' :: ([','].intercalate ((vals.qsort keyLt).toList.map (item S)) ++ ['}']) := by
  rw [hP.obj] at h
  cases hm : kvs.toArray.mapM (fun p => do pure (p.1, ← P p.2)) with
  | error e => rw [hm] at h; simp [bind, Except.bind] at h
  | ok vals =>
    rw [hm] at h
    simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨vals, ?_, ?_⟩
    · rw [← Array.toList_mapM, hm]; rfl
    · have hi : (String.toList ∘ fun kv : String × String => S kv.1 ++ ":" ++ kv.2) = item S := by
        funext kv
        simp [item, String.toList_append]
      simp only [String.toList_append, String.toList_intercalate, List.map_map, hi]
      rfl

theorem pair_ok {p : String × Json} {kv : String × String}
    (h : (do pure (p.1, ← P p.2) : Flock.V (String × String)) = .ok kv) :
    kv.1 = p.1 ∧ P p.2 = .ok kv.2 := by
  cases hc : P p.2 with
  | error e => simp [hc, bind, Except.bind] at h
  | ok t =>
    simp only [hc, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨rfl, rfl⟩

/-! ## Where a text starts -/

/-- **A value's text starts with a character of its class**, which no comma or closing bracket is. -/
theorem head_of (hP : Printer P S) {a : Json} {s : String} (h : P a = .ok s) :
    ∃ c rest, s.toList = c :: rest ∧ cls c = kind a ∧ c ≠ ',' ∧ c ≠ ']' ∧ c ≠ '}' := by
  cases a with
  | null => exact ⟨_, _, hP.text_null h, by decide, by decide, by decide, by decide⟩
  | bool b =>
    rw [hP.text_bool h]
    cases b
    · exact ⟨_, _, rfl, by decide, by decide, by decide, by decide⟩
    · exact ⟨_, _, rfl, by decide, by decide, by decide, by decide⟩
  | num n =>
    obtain ⟨-, rfl⟩ := hP.num h
    rcases int_text n.mantissa with ⟨k, -, hk⟩ | ⟨k, -, hk⟩
    · obtain ⟨c, rest, hc, hd, -⟩ := nat_text k
      obtain ⟨h1, h2, h3, -, h5, h6, h7, h8, h9, h10⟩ := digits_ne hd
      refine ⟨c, rest, hk.trans hc, ?_, h1, h2, h3⟩
      simp [cls, kind, h5, h6, h7, h8, h9, h10]
    · exact ⟨_, _, hk, by simp [cls, kind], by decide, by decide, by decide⟩
  | str x =>
    obtain ⟨rest, hr⟩ := hP.head x
    exact ⟨_, _, (hP.text_str h) ▸ hr, by simp [cls, kind], by decide, by decide, by decide⟩
  | arr a =>
    obtain ⟨ps, -, hs⟩ := hP.text_arr h
    exact ⟨_, _, hs, by simp [cls, kind], by decide, by decide, by decide⟩
  | obj kvs =>
    obtain ⟨vals, -, hs⟩ := hP.text_obj h
    exact ⟨_, _, hs, by simp [cls, kind], by decide, by decide, by decide⟩

theorem starts (hP : Printer P S) {a : Json} {s : String} (h : P a = .ok s) : Starts s.toList := by
  obtain ⟨c, rest, hs, -, h1, h2, h3⟩ := hP.head_of h
  exact ⟨c, rest, hs, h1, h2, h3⟩

theorem item_inj (hP : Printer P S) {kv kv' : String × String} {r r' : List Char}
    (h : item S kv ++ r = item S kv' ++ r') : kv.1 = kv'.1 ∧ kv.2.toList ++ r = kv'.2.toList ++ r' := by
  simp only [item, List.append_assoc, List.cons_append] at h
  obtain ⟨h1, h2⟩ := hP.split h
  exact ⟨h1, (List.cons.inj h2).2⟩

theorem item_starts (hP : Printer P S) (kv : String × String) : Starts (item S kv) := by
  obtain ⟨rest, hr⟩ := hP.head kv.1
  exact ⟨'"', rest ++ ':' :: kv.2.toList, by simp [item, hr], by decide, by decide, by decide⟩

/-! ## Unique readability -/

/-- The induction step for lists: items that read back, read back as a list. -/
theorem arr_read (hP : Printer P S) {a b : Array Json} {s t : String}
    (ih : ∀ x ∈ a.toList, ∀ (y : Json) (s t : String) (r r' : List Char), P x = .ok s →
      P y = .ok t → Stop r → Stop r' → s.toList ++ r = t.toList ++ r' → Reads x y s t)
    (hs : P (.arr a) = .ok s) (ht : P (.arr b) = .ok t) {r r' : List Char}
    (h : s.toList ++ r = t.toList ++ r') : Reads (.arr a) (.arr b) s t := by
  obtain ⟨ps, hps, hs⟩ := hP.text_arr hs
  obtain ⟨qs, hqs, ht⟩ := hP.text_arr ht
  have Fa := mapM_forall₂ hps
  have Fb := mapM_forall₂ hqs
  rw [hs, ht] at h
  simp only [List.cons_append, List.append_assoc, List.cons.injEq, true_and] at h
  have hmem : ∀ {l : List Json} {zs : List String}, List.Forall₂ (fun x y => P x = .ok y) l zs →
      ∀ z ∈ zs.map String.toList, ∃ x ∈ l, ∃ p, P x = .ok p ∧ z = p.toList := by
    intro l zs F z hz
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hz
    obtain ⟨x, hx, hxp⟩ := forall₂_mem_right F p hp
    exact ⟨x, hx, p, hxp, rfl⟩
  obtain ⟨he, -⟩ := inter_read (.inl rfl)
    (fun z hz => by obtain ⟨x, -, p, hp, rfl⟩ := hmem Fa z hz; exact hP.starts hp)
    (fun z hz => by obtain ⟨x, -, p, hp, rfl⟩ := hmem Fb z hz; exact hP.starts hp)
    (fun z hz z' hz' r1 r1' h1 h1' he => by
      obtain ⟨x, hx, p, hp, rfl⟩ := hmem Fa z hz
      obtain ⟨y, -, q, hq, rfl⟩ := hmem Fb z' hz'
      rw [(ih x hx y p q r1 r1' hp hq h1 h1' he).1])
    h
  have hpq : ps = qs := (List.map_inj_right fun _ _ e => String.toList_inj.1 e).1 he
  subst hpq
  refine ⟨String.toList_inj.1 (hs.trans ht.symm), fun sa sb => ?_⟩
  have F := forall₂_comb (Q := fun x y => jsonForm x = jsonForm y) Fa Fb fun x hx y hy p hp hq =>
    (ih x hx y p p [']'] [']'] hp hq (stop_cons (.inr (.inl rfl))) (stop_cons (.inr (.inl rfl))) rfl).2
      (sortedKeys_arr.1 sa x (Array.mem_toList_iff.1 hx)) (sortedKeys_arr.1 sb y (Array.mem_toList_iff.1 hy))
  show jsonForm (.arr a) = jsonForm (.arr b)
  rw [jsonForm_arr, jsonForm_arr, forall₂_map_eq F]

/-- The induction step for objects: values that read back, read back as an object. -/
theorem obj_read (hP : Printer P S) {m n : Std.TreeMap.Raw String Json} {s t : String}
    (ih : ∀ p ∈ m.toArray.toList, ∀ (y : Json) (s t : String) (r r' : List Char), P p.2 = .ok s →
      P y = .ok t → Stop r → Stop r' → s.toList ++ r = t.toList ++ r' → Reads p.2 y s t)
    (hs : P (.obj m) = .ok s) (ht : P (.obj n) = .ok t) {r r' : List Char}
    (h : s.toList ++ r = t.toList ++ r') : Reads (.obj m) (.obj n) s t := by
  obtain ⟨vs, hvs, hs⟩ := hP.text_obj hs
  obtain ⟨ws, hws, ht⟩ := hP.text_obj ht
  have Fm := mapM_forall₂ hvs
  have Fn := mapM_forall₂ hws
  rw [hs, ht] at h
  simp only [List.cons_append, List.append_assoc, List.cons.injEq, true_and] at h
  have hmem : ∀ {k : Std.TreeMap.Raw String Json} {zs : Array (String × String)},
      List.Forall₂ (fun p kv => (do pure (p.1, ← P p.2) : Flock.V (String × String)) = .ok kv)
        k.toArray.toList zs.toList →
      ∀ z ∈ (zs.qsort keyLt).toList.map (item S), ∃ p ∈ k.toArray.toList, ∃ kv, kv.1 = p.1 ∧
        P p.2 = .ok kv.2 ∧ z = item S kv := by
    intro k zs F z hz
    obtain ⟨kv, hkv, rfl⟩ := List.mem_map.1 hz
    have hkv' : kv ∈ zs.toList := (Properties.RemoteAudit.qsort_toList_perm zs keyLt).mem_iff.1 hkv
    obtain ⟨p, hp, hpk⟩ := forall₂_mem_right F kv hkv'
    obtain ⟨h1, h2⟩ := pair_ok hpk
    exact ⟨p, hp, kv, h1, h2, rfl⟩
  obtain ⟨he, -⟩ := inter_read (.inr rfl)
    (fun z hz => by obtain ⟨-, -, kv, -, -, rfl⟩ := hmem Fm z hz; exact hP.item_starts kv)
    (fun z hz => by obtain ⟨-, -, kv, -, -, rfl⟩ := hmem Fn z hz; exact hP.item_starts kv)
    (fun z hz z' hz' r1 r1' h1 h1' he => by
      obtain ⟨p, hp, kv, -, hkv, rfl⟩ := hmem Fm z hz
      obtain ⟨q, -, kv', -, hkv', rfl⟩ := hmem Fn z' hz'
      obtain ⟨hk, hv⟩ := hP.item_inj he
      have := (ih p hp q.2 kv.2 kv'.2 r1 r1' hkv hkv' h1 h1' hv).1
      simp only [item, hk, this])
    h
  have hinj : ∀ a b : String × String, item S a = item S b → a = b := by
    intro a b e
    obtain ⟨h1, h2⟩ := hP.item_inj (r := []) (r' := []) (by simpa using e)
    exact Prod.ext h1 (String.toList_inj.1 (by simpa using h2))
  have hq : (vs.qsort keyLt).toList = (ws.qsort keyLt).toList := (List.map_inj_right hinj).1 he
  refine ⟨String.toList_inj.1 (by rw [hs, ht, hq]), fun sa sb => ?_⟩
  have keys : ∀ {k : Std.TreeMap.Raw String Json} {zs : Array (String × String)}, sortedKeys (.obj k) = true →
      List.Forall₂ (fun p kv => (do pure (p.1, ← P p.2) : Flock.V (String × String)) = .ok kv)
        k.toArray.toList zs.toList → zs.qsort keyLt = zs := by
    intro k zs hk F
    apply qsort_keys
    have hk := increasing_iff.1 (sortedKeys_obj.1 hk).1
    rwa [← forall₂_map_eq (f := fun p : String × Json => p.1) (g := fun kv : String × String => kv.1)
      (forall₂_imp (fun p kv h => (pair_ok h).1.symm) F)]
  rw [keys sa Fm, keys sb Fn] at hq
  rw [hq] at Fm
  have F := forall₂_comb (Q := fun p q => (p.1, jsonForm p.2) = (q.1, jsonForm q.2)) Fm Fn fun p hp q hq' kv h1 h2 => by
    obtain ⟨k1, c1⟩ := pair_ok h1
    obtain ⟨k2, c2⟩ := pair_ok h2
    have e := (ih p hp q.2 kv.2 kv.2 ['}'] ['}'] c1 c2 (stop_cons (.inr (.inr rfl)))
      (stop_cons (.inr (.inr rfl))) rfl).2 ((sortedKeys_obj.1 sa).2 p (Array.mem_toList_iff.1 hp))
      ((sortedKeys_obj.1 sb).2 q (Array.mem_toList_iff.1 hq'))
    rw [← k1, ← k2]
    exact congrArg _ e
  show jsonForm (.obj m) = jsonForm (.obj n)
  rw [jsonForm_obj, jsonForm_obj, ← Std.TreeMap.Raw.toList_toArray, ← Std.TreeMap.Raw.toList_toArray (t := n),
    forall₂_map_eq F]

/-- **A printer's texts read back**: two values' texts, each followed by a comma or a closing bracket, agree only if the
texts are one text, and two sorted values print one text only if they are equivalent. -/
theorem read (hP : Printer P S) : ∀ (N : Nat) (a b : Json) (s t : String) (r r' : List Char), sizeOf a < N →
    P a = .ok s → P b = .ok t → Stop r → Stop r' → s.toList ++ r = t.toList ++ r' → Reads a b s t := by
  intro N
  induction N with
  | zero => intro _ _ _ _ _ _ h; omega
  | succ N ihN =>
    intro a b s t r r' hN hs ht hr hr' h
    have hk : kind a = kind b := by
      obtain ⟨c, rest, hc, hca, -⟩ := hP.head_of hs
      obtain ⟨c', rest', hc', hcb, -⟩ := hP.head_of ht
      rw [hc, hc'] at h
      rw [← hca, ← hcb, (List.cons.inj h).1]
    cases a <;> cases b <;> simp only [kind] at hk <;> try omega
    · exact ⟨String.toList_inj.1 ((hP.text_null hs).trans (hP.text_null ht).symm), fun _ _ => rfl⟩
    · rename_i x y
      have e1 := hP.text_bool hs
      have e2 := hP.text_bool ht
      rw [e1, e2] at h
      cases x <;> cases y <;> simp at h
      all_goals exact ⟨String.toList_inj.1 (e1.trans e2.symm), fun _ _ => rfl⟩
    · rename_i x y
      obtain ⟨hx, rfl⟩ := hP.num hs
      obtain ⟨hy, rfl⟩ := hP.num ht
      have e := int_split hr hr' h
      obtain ⟨xm, xe⟩ := x
      obtain ⟨ym, ye⟩ := y
      simp only at hx hy e
      subst hx hy e
      exact ⟨rfl, fun _ _ => rfl⟩
    · rename_i x y
      rw [hP.text_str hs, hP.text_str ht] at h ⊢
      obtain ⟨rfl, -⟩ := hP.split h
      exact ⟨rfl, fun _ _ => rfl⟩
    · rename_i a b
      exact hP.arr_read (fun x hx y s t r r' => ihN x y s t r r' (by have := arr_lt a hx; omega)) hs ht h
    · rename_i m n
      exact hP.obj_read (fun p hp y s t r r' => ihN p.2 y s t r r' (by have := obj_lt m hp; omega)) hs ht h

/-- **Two sorted values a printer prints alike are equivalent.** -/
theorem inj (hP : Printer P S) {a b : Json} {s : String} (ha : sortedKeys a = true) (hb : sortedKeys b = true)
    (hs : P a = .ok s) (ht : P b = .ok s) : JsonEquiv a b :=
  (hP.read _ a b s s ['}'] ['}'] (Nat.lt_succ_self _) hs ht (stop_cons (.inr (.inr rfl)))
    (stop_cons (.inr (.inr rfl))) rfl).2 ha hb

/-! ## The two printers -/

/-- `Flock.canonUtf8` prints canonical JSON's form, its strings in UTF-8. -/
theorem utf8 : Printer Flock.canonUtf8 Flock.canonStrUtf8 where
  null := by rw [Flock.canonUtf8]
  bool _ := by rw [Flock.canonUtf8]
  num h := CanonRead.text_num h
  str := Properties.Pouw.LiveSalt.Canon.canon_str
  arr := Properties.Pouw.LiveSalt.Canon.canon_arr
  obj := Properties.Pouw.LiveSalt.Canon.canon_obj
  head := CanonRead.canonStr_head
  split := Properties.Pouw.LiveSalt.Canon.canonStr_split

open FlockComplete.JsonRT.ReadBack (cesc cesc_esc esc_of) in
/-- **A `Flock.canonStr` text ends where it ends**: the parser's `str` reads the string back from it, so what follows it
doesn't change which string it is. -/
theorem canonStr_split {s s' : String} {r r' : List Char}
    (h : (Flock.canonStr s).toList ++ r = (Flock.canonStr s').toList ++ r') : s = s' ∧ r = r' := by
  rw [FlockComplete.JsonRT.ReadBack.canonStr_toList, FlockComplete.JsonRT.ReadBack.canonStr_toList] at h
  simp only [List.cons_append, List.append_assoc, List.nil_append, List.cons.injEq, true_and] at h
  have e := esc_of (fun c => (cesc c).toList) cesc_esc
  obtain ⟨o, ho⟩ := e s ((s.toList.flatMap fun c => (cesc c).toList).length +
    (s'.toList.flatMap fun c => (cesc c).toList).length + 1) "" r 0 (by omega)
  obtain ⟨o', ho'⟩ := e s' ((s.toList.flatMap fun c => (cesc c).toList).length +
    (s'.toList.flatMap fun c => (cesc c).toList).length + 1) "" r' 0 (by omega)
  rw [h, ho'] at ho
  injection ho with h1 h2 h3
  exact ⟨by simpa using h3.symm, h1.symm⟩

theorem canon_num {n : JsonNumber} {s : String} (h : Flock.canon (.num n) = .ok s) :
    n.exponent = 0 ∧ s = toString n.mantissa := by
  rw [Flock.canon] at h
  by_cases he : n.exponent = 0
  · simp only [he, beq_self_eq_true, ite_true, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨he, h.symm⟩
  · simp [he, throw, throwThe, MonadExceptOf.throw] at h

/-- `Flock.canon` prints canonical JSON's form, its strings escaped to printable ASCII. -/
theorem canon : Printer Flock.canon Flock.canonStr where
  null := by rw [Flock.canon]
  bool _ := by rw [Flock.canon]
  num h := canon_num h
  str _ := by rw [Flock.canon]
  arr a := by
    rw [Flock.canon]
    simp only [Array.mapM_subtype, Array.unattach_attach]
  obj kvs := by
    rw [Flock.canon]
    simp only [Array.mapM_subtype, Array.unattach_attach]
  head x := ⟨_, by rw [FlockComplete.JsonRT.ReadBack.canonStr_toList]; rfl⟩
  split := canonStr_split

/-- **Two sorted values with one `Flock.canon` text are equivalent.** -/
theorem canon_inj {a b : Json} {s : String} (ha : sortedKeys a = true) (hb : sortedKeys b = true)
    (hs : Flock.canon a = .ok s) (ht : Flock.canon b = .ok s) : JsonEquiv a b :=
  canon.inj ha hb hs ht

end Properties.Challenger.Printer
