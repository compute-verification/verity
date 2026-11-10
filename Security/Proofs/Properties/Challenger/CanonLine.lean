module

public import Proofs.Properties.Challenger.CanonRead
public import Proofs.Properties.Composition
public import Proofs.Flock.Claims.CanonEq
public import Spot.ChallengerSession
import all Lean.Data.Json.Basic

/-!
# A release log line reads back

`Flock.canon` prints a value as `Flock.canonUtf8` does, but for the characters from `0x7f` on, which it writes as `\u`
escapes (`Composition.Esc.esc`) where `canonUtf8` writes them as themselves (`LiveSalt.Canon.esc`). `utf8` reads each
escape of a text back (`Composition.Esc.unesc2`) and writes its character as `canonUtf8` does, and leaves every other
character as it is, so of a sorted value it turns `canon`'s text into `canonUtf8`'s (`utf8_canon`). Two sorted values
with one `canon` text then have one `canonUtf8` text, and are equivalent by `CanonRead.canon_inj` (`canon_inj`).

A line the release log's reader accepts (`Spot.Challenger.event`) is its parse's `canon` text, and the parse is
sorted (`event_ok`). So a line that is a sorted value's `canon` text is read as a value equivalent to it
(`event_equiv`), field by field (`CanonRead.getD_equiv`, `field?_equiv`): the same string, number, Boolean or null
wherever it has one (`equiv_leaf`, `getD_str`, `int?_equiv`, `nat?_equiv`).
-/

@[expose] public section

namespace Properties.Challenger.CanonLine

open Lean (Json JsonNumber)
open Spot (sortedKeys getD field?)
open FlockDeploy (JsonEquiv jsonForm)
open Properties.Challenger.CanonRead (sortedKeys_arr sortedKeys_obj increasing_iff mapM_forall₂ forall₂_map_eq
  forall₂_imp qsort_keys)

/-! ## Reading `canon`'s escapes back -/

theorem unescLow_lt {r r' : List Char} {n : Nat} (h : Composition.Esc.unescLow r = some (n, r')) :
    r'.length < r.length := by
  unfold Composition.Esc.unescLow at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    simp only [List.length_cons]
    omega
  · cases h

theorem unesc2_lt {r r' : List Char} {c : Char} (h : Composition.Esc.unesc2 r = some (c, r')) :
    r'.length < r.length := by
  unfold Composition.Esc.unesc2 at h
  split at h
  any_goals
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨-, rfl⟩ := h
    simp only [List.length_cons]
    omega
  · split at h
    · cases hl : Composition.Esc.unescLow _ with
      | none => rw [hl] at h; cases h
      | some p =>
        obtain ⟨l, r''⟩ := p
        rw [hl] at h
        simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨-, rfl⟩ := h
        have := unescLow_lt hl
        simp only [List.length_cons]
        omega
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨-, rfl⟩ := h
      simp only [List.length_cons]
      omega
  · cases h

/-- **`canon`'s text as `canonUtf8` writes it**: each escape read back (`Composition.Esc.unesc2`) and its character
written as `canonUtf8` writes it (`LiveSalt.Canon.esc`), and every other character as it is. -/
def utf8 : List Char → List Char
  | [] => []
  | c :: r =>
    if c = '\\' then
      match _h : Composition.Esc.unesc2 r with
      | some (d, r') => Pouw.LiveSalt.Canon.esc d ++ utf8 r'
      | none => c :: utf8 r
    else c :: utf8 r
termination_by l => l.length
decreasing_by
  all_goals simp only [List.length_cons]
  · have := unesc2_lt _h; omega
  · omega
  · omega

@[simp] theorem utf8_nil : utf8 [] = [] := by rw [utf8]

theorem utf8_cons {c : Char} (hc : c ≠ '\\') (r : List Char) : utf8 (c :: r) = c :: utf8 r := by
  rw [utf8]
  simp only [hc, ite_false]

theorem utf8_unesc {r r' : List Char} {d : Char} (h : Composition.Esc.unesc2 r = some (d, r')) :
    utf8 ('\\' :: r) = Pouw.LiveSalt.Canon.esc d ++ utf8 r' := by
  rw [utf8]
  simp only [ite_true]
  split
  · rename_i heq
    rw [h] at heq
    cases heq
    rfl
  · rename_i heq
    rw [h] at heq
    cases heq

theorem utf8_plain {l : List Char} (h : ∀ c ∈ l, c ≠ '\\') (r : List Char) : utf8 (l ++ r) = l ++ utf8 r := by
  induction l with
  | nil => rfl
  | cons c l ih =>
    rw [List.cons_append, utf8_cons (h c List.mem_cons_self), ih fun x hx => h x (List.mem_cons_of_mem _ hx)]
    rfl

theorem escU_head (n : Nat) : ∃ t, (Flock.escU n).toList = '\\' :: t := by
  unfold Flock.escU
  split
  · rw [String.toList_append, Composition.Esc.u_toList, List.cons_append]
    exact ⟨_, rfl⟩
  · rw [String.toList_append, String.toList_append, String.toList_append, Composition.Esc.u_toList,
      List.append_assoc, List.append_assoc, List.cons_append]
    exact ⟨_, rfl⟩

/-- A character `canonStr` writes as itself, `canonStrUtf8` writes as itself. -/
theorem plain_esc {c : Char} (h : Composition.Esc.esc c = [c]) : Pouw.LiveSalt.Canon.esc c = [c] := by
  unfold Composition.Esc.esc at h
  unfold Pouw.LiveSalt.Canon.esc
  split at h
  any_goals (simp at h; done)
  rename_i h1 h2 h3 h4 h5
  split
  · exact absurd rfl h1
  · exact absurd rfl h2
  · exact absurd rfl h3
  · exact absurd rfl h4
  · exact absurd rfl h5
  · by_cases h8 : c.toNat = 8
    · simp [h8] at h
    by_cases h12 : c.toNat = 12
    · simp [h12] at h
    by_cases h32 : c.toNat < 32
    · simp only [h8, h12, h32, beq_iff_eq, decide_true, Bool.true_or, ite_true, ite_false] at h
      obtain ⟨t, ht⟩ := escU_head c.toNat
      rw [ht] at h
      exact absurd (List.cons.inj h).1.symm h2
    · simp [h8, h12, h32]

/-- **`utf8` reads a character `canonStr` writes back as `canonStrUtf8` writes it**, whatever follows. -/
theorem utf8_esc (c : Char) (r : List Char) :
    utf8 (Composition.Esc.esc c ++ r) = Pouw.LiveSalt.Canon.esc c ++ utf8 r := by
  have hu := Composition.Esc.unesc_esc c r
  cases he : Composition.Esc.esc c with
  | nil =>
    have h0 := Composition.Esc.unesc_esc c []
    rw [he] at h0
    simp [Composition.Esc.unesc] at h0
  | cons d t =>
    rw [he] at hu
    by_cases hd : d = '\\'
    · subst hd
      simp only [List.cons_append, Composition.Esc.unesc, ite_true] at hu
      rw [List.cons_append]
      exact utf8_unesc hu
    · simp only [List.cons_append, Composition.Esc.unesc, hd, ite_false, Option.some.injEq, Prod.mk.injEq] at hu
      obtain ⟨rfl, ht⟩ := hu
      have ht : t = [] := by simpa using ht
      subst ht
      rw [plain_esc he, List.singleton_append, List.singleton_append]
      exact utf8_cons hd r

theorem utf8_escs (cs r : List Char) :
    utf8 (cs.flatMap Composition.Esc.esc ++ r) = cs.flatMap Pouw.LiveSalt.Canon.esc ++ utf8 r := by
  induction cs with
  | nil => rfl
  | cons c cs ih => rw [List.flatMap_cons, List.flatMap_cons, List.append_assoc, utf8_esc, ih, List.append_assoc]

/-- **`utf8` reads `canonStr`'s string back as `canonStrUtf8` writes it**, whatever follows. -/
theorem utf8_canonStr (s : String) (r : List Char) :
    utf8 ((Flock.canonStr s).toList ++ r) = (Flock.canonStrUtf8 s).toList ++ utf8 r := by
  rw [Composition.Esc.canonStr_toList, Pouw.LiveSalt.Canon.canonStr_toList]
  simp only [List.cons_append, List.append_assoc, List.nil_append]
  rw [utf8_cons (by decide), utf8_escs, utf8_cons (by decide)]

theorem int_plain (m : Int) : ∀ c ∈ (toString m).toList, c ≠ '\\' := by
  have hn : ∀ n : Nat, ∀ c ∈ (toString n).toList, c ≠ '\\' := fun n c hc => by
    obtain ⟨-, -, -, -, hd⟩ := CanonRead.nat_text n
    have := hd c hc
    rintro rfl
    simp at this
  rcases CanonRead.int_text m with ⟨n, -, hk⟩ | ⟨n, -, hk⟩
  · rw [hk]; exact hn n
  · rw [hk]
    intro c hc
    rcases List.mem_cons.1 hc with rfl | hc
    · decide
    · exact hn _ c hc

/-! ## `canon`'s text, as `canonUtf8`'s -/

theorem canon_arr (a : Array Json) : Flock.canon (.arr a) =
    (do let parts ← a.mapM Flock.canon; pure ("[" ++ ",".intercalate parts.toList ++ "]")) := by
  rw [Flock.canon]
  simp only [Array.mapM_subtype, Array.unattach_attach]

theorem canon_obj (kvs : Std.TreeMap.Raw String Json) : Flock.canon (.obj kvs) =
    (do
      let vals ← kvs.toArray.mapM fun p => do pure (p.1, ← Flock.canon p.2)
      let items := vals.qsort (fun a b => a.1 < b.1)
      pure ("{" ++ ",".intercalate (items.toList.map fun kv => Flock.canonStr kv.1 ++ ":" ++ kv.2) ++ "}")) := by
  rw [Flock.canon]
  simp only [Array.mapM_subtype, Array.unattach_attach]

/-- What `utf8` makes of text `s`: text `t`, whatever follows. -/
def ToUtf8 (s t : String) : Prop := ∀ r, utf8 (s.toList ++ r) = t.toList ++ utf8 r

theorem pair_ok {P : Json → Flock.V String} {p : String × Json} {kv : String × String}
    (h : (do pure (p.1, ← P p.2) : Flock.V (String × String)) = .ok kv) : kv.1 = p.1 ∧ P p.2 = .ok kv.2 := by
  cases hc : P p.2 with
  | error e => simp [hc, bind, Except.bind] at h
  | ok t =>
    simp only [hc, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨rfl, rfl⟩

/-- A second printer succeeds wherever the first did, position by position, with results related by `R`. -/
theorem lift_mapM {α β γ : Type} {f : α → Flock.V β} {g : α → Flock.V γ} {R : β → γ → Prop} :
    ∀ {l : List α} {ps : List β}, List.Forall₂ (fun x p => f x = .ok p) l ps →
      (∀ x ∈ l, ∀ p, f x = .ok p → ∃ t, g x = .ok t ∧ R p t) → ∃ qs, l.mapM g = .ok qs ∧ List.Forall₂ R ps qs
  | _, _, .nil, _ => ⟨[], rfl, .nil⟩
  | x :: _, p :: _, .cons hx F, h => by
    obtain ⟨t, ht, hp⟩ := h x List.mem_cons_self p hx
    obtain ⟨qs, hqs, G⟩ := lift_mapM F fun y hy => h y (List.mem_cons_of_mem _ hy)
    exact ⟨t :: qs, by rw [List.mapM_cons, ht, hqs]; rfl, .cons hp G⟩

theorem forall₂_map {α β γ δ : Type} {R : α → β → Prop} {S : γ → δ → Prop} {f : α → γ} {g : β → δ}
    (hRS : ∀ a b, R a b → S (f a) (g b)) : ∀ {l : List α} {m : List β}, List.Forall₂ R l m →
      List.Forall₂ S (l.map f) (m.map g)
  | _, _, .nil => .nil
  | _, _, .cons h hs => .cons (hRS _ _ h) (forall₂_map hRS hs)

theorem utf8_inter : ∀ {ps qs : List String}, List.Forall₂ ToUtf8 ps qs → ∀ r,
    utf8 ([','].intercalate (ps.map String.toList) ++ r) = [','].intercalate (qs.map String.toList) ++ utf8 r
  | _, _, .nil, _ => rfl
  | _, _, .cons h .nil, r => by simpa using h r
  | _, _, .cons h (.cons h' F), r => by
    simp only [List.map_cons, List.intercalate_cons_cons, List.append_assoc, List.cons_append, List.nil_append]
    rw [h, utf8_cons (by decide)]
    have := utf8_inter (.cons h' F) r
    simp only [List.map_cons] at this
    rw [this]

theorem toUtf8_list {o c : String} (ho : ∀ x ∈ o.toList, x ≠ '\\') (hc : ∀ x ∈ c.toList, x ≠ '\\')
    {ps qs : List String} (F : List.Forall₂ ToUtf8 ps qs) :
    ToUtf8 (o ++ ",".intercalate ps ++ c) (o ++ ",".intercalate qs ++ c) := by
  intro r
  simp only [String.toList_append, String.toList_intercalate, List.append_assoc,
    show ",".toList = [','] from rfl]
  rw [utf8_plain ho, utf8_inter F, utf8_plain hc]

theorem toUtf8_item {k s t : String} (h : ToUtf8 s t) :
    ToUtf8 (Flock.canonStr k ++ ":" ++ s) (Flock.canonStrUtf8 k ++ ":" ++ t) := by
  intro r
  simp only [String.toList_append, List.append_assoc, show ":".toList = [':'] from rfl, List.cons_append,
    List.nil_append]
  rw [utf8_canonStr, utf8_cons (by decide), h]

/-- **`utf8` turns a sorted value's `canon` text into its `canonUtf8` text**, whatever follows it. -/
theorem utf8_canon : ∀ (N : Nat) (a : Json) (s : String), sizeOf a < N → sortedKeys a = true →
    Flock.canon a = .ok s → ∃ t, Flock.canonUtf8 a = .ok t ∧ ToUtf8 s t := by
  intro N
  induction N with
  | zero => intro _ _ h; omega
  | succ N ih =>
    intro a s hN ha hs
    cases a with
    | null =>
      rw [Flock.canon] at hs
      cases hs
      exact ⟨"null", by rw [Flock.canonUtf8]; rfl, fun r => utf8_plain (by decide) r⟩
    | bool b =>
      rw [Flock.canon] at hs
      cases hs
      refine ⟨if b then "true" else "false", by rw [Flock.canonUtf8]; rfl, fun r => utf8_plain ?_ r⟩
      cases b <;> decide
    | num n =>
      rw [Flock.canon] at hs
      by_cases he : n.exponent = 0
      · simp only [he, beq_self_eq_true, ite_true, pure, Except.pure, Except.ok.injEq] at hs
        subst hs
        refine ⟨toString n.mantissa, ?_, fun r => utf8_plain (int_plain _) r⟩
        rw [Flock.canonUtf8]
        simp only [he, beq_self_eq_true, ite_true]
        rfl
      · simp [he, throw, throwThe, MonadExceptOf.throw] at hs
    | str x =>
      rw [Flock.canon] at hs
      cases hs
      exact ⟨Flock.canonStrUtf8 x, by rw [Flock.canonUtf8]; rfl, utf8_canonStr x⟩
    | arr a =>
      rw [canon_arr] at hs
      cases hm : a.mapM Flock.canon with
      | error e => simp [hm, bind, Except.bind] at hs
      | ok parts =>
        simp only [hm, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hs
        subst hs
        have F := mapM_forall₂ (show a.toList.mapM Flock.canon = .ok parts.toList by rw [← Array.toList_mapM, hm]; rfl)
        obtain ⟨qs, hqs, G⟩ := lift_mapM (g := Flock.canonUtf8) (R := ToUtf8) F fun x hx p hp =>
          ih x p (by have := CanonRead.arr_lt a hx; omega) (sortedKeys_arr.1 ha x (Array.mem_toList_iff.1 hx)) hp
        refine ⟨"[" ++ ",".intercalate qs ++ "]", ?_, toUtf8_list (by decide) (by decide) G⟩
        rw [Pouw.LiveSalt.Canon.canon_arr, Array.mapM_eq_mapM_toList, hqs]
        rfl
    | obj m =>
      rw [canon_obj] at hs
      cases hm : m.toArray.mapM (fun p => do pure (p.1, ← Flock.canon p.2)) with
      | error e => rw [hm] at hs; simp [bind, Except.bind] at hs
      | ok vals =>
        rw [hm] at hs
        simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hs
        subst hs
        have F := mapM_forall₂ (show m.toArray.toList.mapM (fun p => do pure (p.1, ← Flock.canon p.2)) =
          .ok vals.toList by rw [← Array.toList_mapM, hm]; rfl)
        obtain ⟨ws, hws, G⟩ := lift_mapM (g := fun p => do pure (p.1, ← Flock.canonUtf8 p.2))
          (R := fun kv kv' => kv.1 = kv'.1 ∧ ToUtf8 kv.2 kv'.2) F fun p hp kv h => by
            obtain ⟨k1, c1⟩ := pair_ok h
            obtain ⟨t, ht, hr⟩ := ih p.2 kv.2 (by have := CanonRead.obj_lt m hp; omega)
              ((sortedKeys_obj.1 ha).2 p (Array.mem_toList_iff.1 hp)) c1
            exact ⟨(p.1, t), by rw [ht]; rfl, k1, hr⟩
        have hk := increasing_iff.1 (sortedKeys_obj.1 ha).1
        have hv : m.toArray.toList.map (·.1) = vals.toList.map (·.1) :=
          forall₂_map_eq (forall₂_imp (fun p kv h => (pair_ok h).1.symm) F)
        have hw : vals.toList.map (·.1) = ws.map (·.1) := forall₂_map_eq (forall₂_imp (fun _ _ h => h.1) G)
        have q1 : vals.qsort (fun a b => a.1 < b.1) = vals := qsort_keys (by rw [← hv]; exact hk)
        have q2 : ws.toArray.qsort (fun a b => a.1 < b.1) = ws.toArray :=
          qsort_keys (by rw [List.toList_toArray, ← hw, ← hv]; exact hk)
        refine ⟨"{" ++ ",".intercalate (ws.map fun kv => Flock.canonStrUtf8 kv.1 ++ ":" ++ kv.2) ++ "}", ?_, ?_⟩
        · rw [Pouw.LiveSalt.Canon.canon_obj, Array.mapM_eq_mapM_toList, hws]
          simp only [Functor.map, Except.map, bind, Except.bind, pure, Except.pure]
          rw [q2]
        · rw [q1]
          refine toUtf8_list (by decide) (by decide) (forall₂_map (fun kv kv' h => ?_) G)
          rw [← h.1]
          exact toUtf8_item h.2

/-- **A sorted value's `canon` text, read by `utf8`, is its `canonUtf8` text.** -/
theorem canon_utf8 {a : Json} {s : String} (ha : sortedKeys a = true) (hs : Flock.canon a = .ok s) :
    ∃ t, Flock.canonUtf8 a = .ok t ∧ utf8 s.toList = t.toList := by
  obtain ⟨t, ht, hr⟩ := utf8_canon _ a s (Nat.lt_succ_self _) ha hs
  exact ⟨t, ht, by simpa using hr []⟩

/-- **Two sorted values with one `Flock.canon` text are equivalent**: they have one `canonUtf8` text (`canon_utf8`),
so `CanonRead.canon_inj`. -/
theorem canon_inj {a b : Json} {s : String} (ha : sortedKeys a = true) (hb : sortedKeys b = true)
    (hs : Flock.canon a = .ok s) (ht : Flock.canon b = .ok s) : JsonEquiv a b := by
  obtain ⟨t, h1, e1⟩ := canon_utf8 ha hs
  obtain ⟨t', h2, e2⟩ := canon_utf8 hb ht
  obtain rfl : t = t' := String.toList_inj.1 (e1.symm.trans e2)
  exact CanonRead.canon_inj ha hb h1 h2

/-! ## Field reads of equivalent values -/

theorem field?_obj (m : Std.TreeMap.Raw String Json) (k : String) : field? (.obj m) k = m.get? k := by
  unfold field? Json.getObjVal?
  dsimp only
  cases m.get? k <;> rfl

theorem field?_of_not_obj {a : Json} (h : ¬ ∃ m, a = .obj m) (k : String) : field? a k = none := by
  cases a with
  | obj m => exact absurd ⟨m, rfl⟩ h
  | _ => rfl

/-- **Equivalent sorted values have one field under each key, up to equivalence**, or none under it. -/
theorem field?_equiv {a b : Json} (ha : sortedKeys a = true) (hb : sortedKeys b = true) (h : JsonEquiv a b)
    (k : String) : (field? a k).map jsonForm = (field? b k).map jsonForm := by
  by_cases hA : ∃ m, a = .obj m
  · obtain ⟨m, rfl⟩ := hA
    obtain ⟨n, rfl⟩ := CanonRead.form_obj (Eq.trans (Eq.symm h) (FlockDeploy.jsonForm_obj m))
    have hf := h
    rw [JsonEquiv, FlockDeploy.jsonForm_obj, FlockDeploy.jsonForm_obj, FlockDeploy.JsonForm.obj.injEq,
      FlockDeploy.FieldForms.ofList_inj] at hf
    rw [field?_obj, field?_obj, CanonRead.get?_lookup (CanonRead.sorted_toList ha),
      CanonRead.get?_lookup (CanonRead.sorted_toList hb)]
    exact CanonRead.lookup_forms hf k
  · have hB : ¬ ∃ n, b = .obj n := fun ⟨n, hn⟩ =>
      hA (CanonRead.form_obj (Eq.trans h (hn ▸ FlockDeploy.jsonForm_obj n)))
    rw [field?_of_not_obj hA, field?_of_not_obj hB]

theorem equiv_num {x : Json} {n : JsonNumber} (h : JsonEquiv x (.num n)) : x = .num n := by
  cases x with
  | num m => cases h; rfl
  | arr a => rw [JsonEquiv, CanonRead.jsonForm_arr] at h; cases h
  | obj m => rw [JsonEquiv, FlockDeploy.jsonForm_obj] at h; cases h
  | _ => cases h

/-- **Equivalent values are the same string**, by the verifier's reducible match, or neither is it. -/
theorem jsonStrIs_equiv {a b : Json} (h : JsonEquiv a b) (s : String) :
    Spot.jsonStrIs a s = Spot.jsonStrIs b s := by
  by_cases hb : ∃ t, b = .str t
  · obtain ⟨t, rfl⟩ := hb
    rw [CanonRead.equiv_str h]
  · have ha : ¬ ∃ t, a = .str t := fun ⟨t, ht⟩ => hb ⟨t, CanonRead.equiv_str (ht ▸ h.symm)⟩
    cases a <;> cases b <;> first | rfl | exact absurd ⟨_, rfl⟩ ha | exact absurd ⟨_, rfl⟩ hb

/-- **Equivalent values are the same integer**, or neither is one. -/
theorem int?_equiv {a b : Json} (h : JsonEquiv a b) : Spot.int? a = Spot.int? b := by
  by_cases hb : ∃ n, b = .num n
  · obtain ⟨n, rfl⟩ := hb
    rw [equiv_num h]
  · have ha : ¬ ∃ n, a = .num n := fun ⟨n, hn⟩ => hb ⟨n, equiv_num (hn ▸ h.symm)⟩
    cases a <;> cases b <;> first | rfl | exact absurd ⟨_, rfl⟩ ha | exact absurd ⟨_, rfl⟩ hb

theorem nat?_equiv {a b : Json} (h : JsonEquiv a b) : Spot.nat? a = Spot.nat? b := by
  unfold Spot.nat?
  rw [int?_equiv h]

/-! ## A release log line -/

/-- **A line the release log's reader accepts is its value's `canon` text**, and the value is sorted. -/
theorem event_ok {line : String} {j : Json} (h : Spot.Challenger.event line = .ok j) :
    Flock.canon j = .ok line ∧ sortedKeys j = true := by
  unfold Spot.Challenger.event at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw, beq_iff_eq] at h
  split at h
  · split at h
    · rename_i j' _
      split at h
      · rename_i text htext
        split at h
        · rename_i heq
          split at h
          · rename_i hs
            cases h
            exact ⟨heq ▸ htext, hs⟩
          · cases h
        · cases h
      · cases h
    · cases h
  · cases h

/-- **A sorted value's `canon` line reads as a value equivalent to it.** -/
theorem event_equiv {line : String} {j d : Json} (h : Spot.Challenger.event line = .ok j)
    (hd : sortedKeys d = true) (hc : Flock.canon d = .ok line) : JsonEquiv j d :=
  canon_inj (event_ok h).2 hd (event_ok h).1 hc

/-- **A sorted value's `canon` line reads with its fields**, each up to equivalence. -/
theorem event_getD {line : String} {j d : Json} (h : Spot.Challenger.event line = .ok j)
    (hd : sortedKeys d = true) (hc : Flock.canon d = .ok line) (k : String) : JsonEquiv (getD j k) (getD d k) :=
  CanonRead.getD_equiv (event_ok h).2 hd (event_equiv h hd hc) k

end Properties.Challenger.CanonLine
