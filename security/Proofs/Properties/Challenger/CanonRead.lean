module

public import Batteries.Data.List.Basic
public import Std.Data.TreeMap.Raw.WF
public import Proofs.Properties.Pouw.LiveSalt.Canon
public import Proofs.Properties.Accounting.Remote.Sort
public import Proofs.Flock.Claims.JsonEq
public import Proofs.Core.QSort
import all Init.Data.Repr
import all Init.Data.Int.Repr
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# Canonical JSON reads back

`Flock.canonUtf8` prints a value so that it reads back. Each value's text starts with a character no comma or closing
bracket is (`canon_starts`), and ends where it ends when a comma or a closing bracket follows (`canon_read`): a numeral
at its first non-digit, a string at its first unescaped quote (`Canon.canonStr_split`), a list or an object at its
closing bracket, item by item (`inter_read`). So two values whose objects hold their keys in order
(`Spot.sortedKeys`, as every parse's do) and print the same text are equivalent (`canon_inj`): `canonUtf8`
sorts an object's fields by key, which leaves a sorted object's where they are (`qsort_keys`).

A field read of a sorted object is a lookup of the field in its fields' list (`get?_lookup`), so equivalent sorted values
have equivalent fields (`getD_equiv`), every field of a sorted value is sorted (`sorted_getD`), and an entry without its
`sig` (`Core.Record.unsigned`) is sorted and has the entry's other fields (`sorted_unsigned`, `getD_unsigned`).
-/

@[expose] public section

namespace Properties.Challenger.CanonRead

open Lean (Json JsonNumber)
open Spot (sortedKeys increasing getD field?)
open FlockDeploy (JsonEquiv jsonForm jsonForms jsonForms_eq jsonForm_obj FieldForms JsonForm JsonForms)
open Properties.Pouw.LiveSalt.Canon (canon_arr canon_obj canonStr_split canonStr_toList nat_split)
open Std.DTreeMap.Internal (Impl)

/-! ## Sorted values -/

theorem sortedKeys_arr {a : Array Json} : sortedKeys (.arr a) = true ↔ ∀ x ∈ a, sortedKeys x = true := by
  rw [sortedKeys, Array.all_eq_true']
  exact ⟨fun h x hx => h ⟨x, hx⟩ (Array.mem_attach _ _), fun h x _ => h x.1 x.2⟩

theorem sortedKeys_obj {kvs : Std.TreeMap.Raw String Json} :
    sortedKeys (.obj kvs) = true ↔
      increasing (kvs.toArray.toList.map (·.1)) = true ∧ ∀ p ∈ kvs.toArray, sortedKeys p.2 = true := by
  rw [sortedKeys, Bool.and_eq_true, Array.all_eq_true']
  exact and_congr_right fun _ => ⟨fun h p hp => h ⟨p, hp⟩ (Array.mem_attach _ _), fun h p _ => h p.1 p.2⟩

theorem increasing_iff : ∀ {l : List String}, increasing l = true ↔ l.Pairwise (· < ·)
  | [] => by simp [increasing]
  | [a] => by simp [increasing]
  | a :: b :: t => by
    rw [increasing, Bool.and_eq_true, decide_eq_true_eq, increasing_iff]
    constructor
    · rintro ⟨hab, hbt⟩
      refine List.Pairwise.cons (fun c hc => ?_) hbt
      rcases List.mem_cons.1 hc with rfl | hc
      · exact hab
      · exact String.lt_trans hab (List.rel_of_pairwise_cons hbt hc)
    · intro h
      exact ⟨List.rel_of_pairwise_cons h (List.mem_cons_self), h.of_cons⟩

/-! ## Lists of items -/

/-- What may follow a value's text in canonical JSON: a comma or a closing bracket. -/
def Stop (r : List Char) : Prop := ∃ c rest, r = c :: rest ∧ (c = ',' ∨ c = ']' ∨ c = '}')

/-- A text that starts with a character no comma or closing bracket is. -/
def Starts (s : List Char) : Prop := ∃ c rest, s = c :: rest ∧ c ≠ ',' ∧ c ≠ ']' ∧ c ≠ '}'

theorem stop_cons {c : Char} {r : List Char} (h : c = ',' ∨ c = ']' ∨ c = '}') : Stop (c :: r) := ⟨c, r, rfl, h⟩

theorem inter_cons (x : List Char) (t : List (List Char)) (r : List Char) :
    [','].intercalate (x :: t) ++ r = x ++ (match t with | [] => r | _ :: _ => ',' :: ([','].intercalate t ++ r)) := by
  cases t with
  | nil => simp
  | cons y t => simp [List.intercalate_cons_cons]

/-- **Items separated by commas, then a closing bracket, read back**, each item's text ending where it ends when a comma
or a closing bracket follows it (`hR`) and starting with neither. -/
theorem inter_read {c : Char} (hc : c = ']' ∨ c = '}') :
    ∀ {xs ys : List (List Char)} {r r' : List Char}, (∀ x ∈ xs, Starts x) → (∀ y ∈ ys, Starts y) →
      (∀ x ∈ xs, ∀ y ∈ ys, ∀ r r', Stop r → Stop r' → x ++ r = y ++ r' → x = y) →
      [','].intercalate xs ++ c :: r = [','].intercalate ys ++ c :: r' → xs = ys ∧ r = r' := by
  have hc' : c ≠ ',' := by rcases hc with rfl | rfl <;> decide
  have hS : ∀ t : List (List Char), ∀ r : List Char,
      Stop (match t with | [] => c :: r | _ :: _ => ',' :: ([','].intercalate t ++ c :: r)) := by
    intro t r
    cases t with
    | nil => exact stop_cons (.inr hc)
    | cons _ _ => exact stop_cons (.inl rfl)
  intro xs
  induction xs with
  | nil =>
    intro ys r r' _ hy _ h
    cases ys with
    | nil => simpa using h
    | cons y t =>
      exfalso
      obtain ⟨d, rest, rfl, h1, h2, h3⟩ := hy y List.mem_cons_self
      rw [inter_cons] at h
      simp only [List.intercalate_nil, List.nil_append, List.cons_append, List.cons.injEq] at h
      rcases hc with rfl | rfl
      · exact h2 h.1.symm
      · exact h3 h.1.symm
  | cons x s ih =>
    intro ys r r' hx hy hR h
    cases ys with
    | nil =>
      exfalso
      obtain ⟨d, rest, rfl, h1, h2, h3⟩ := hx x List.mem_cons_self
      rw [inter_cons] at h
      simp only [List.intercalate_nil, List.nil_append, List.cons_append, List.cons.injEq] at h
      rcases hc with rfl | rfl
      · exact h2 h.1
      · exact h3 h.1
    | cons y t =>
      rw [inter_cons, inter_cons] at h
      obtain rfl := hR x List.mem_cons_self y List.mem_cons_self _ _ (hS s r) (hS t r') h
      have h := List.append_cancel_left h
      cases s with
      | nil =>
        cases t with
        | nil => exact ⟨rfl, (List.cons.inj h).2⟩
        | cons _ _ => exact absurd (List.cons.inj h).1 hc'
      | cons x' s =>
        cases t with
        | nil => exact absurd (List.cons.inj h).1.symm hc'
        | cons y' t =>
          obtain ⟨he, hr⟩ := ih (fun z hz => hx z (List.mem_cons_of_mem _ hz))
            (fun z hz => hy z (List.mem_cons_of_mem _ hz))
            (fun a ha b hb => hR a (List.mem_cons_of_mem _ ha) b (List.mem_cons_of_mem _ hb)) (List.cons.inj h).2
          exact ⟨by rw [he], hr⟩

/-! ## `List.mapM` and `List.Forall₂` -/

theorem mapM_forall₂ {α β : Type} {f : α → Flock.V β} :
    ∀ {l : List α} {ys : List β}, l.mapM f = .ok ys → List.Forall₂ (fun x y => f x = .ok y) l ys
  | [], ys, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; exact .nil
  | x :: l, ys, h => by
    rw [List.mapM_cons] at h
    cases hx : f x with
    | error e => simp [hx, bind, Except.bind] at h
    | ok y =>
      cases hl : l.mapM f with
      | error e => simp [hx, hl, bind, Except.bind] at h
      | ok zs =>
        simp only [hx, hl, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact .cons hx (mapM_forall₂ hl)

theorem forall₂_mem_right {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {ys : List β}, List.Forall₂ R l ys → ∀ y ∈ ys, ∃ x ∈ l, R x y
  | _, _, .nil => fun _ h => absurd h List.not_mem_nil
  | _, _, .cons hxy h => fun y hy => by
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨_, List.mem_cons_self, hxy⟩
    · obtain ⟨x, hx, hr⟩ := forall₂_mem_right h y hy
      exact ⟨x, List.mem_cons_of_mem _ hx, hr⟩

theorem forall₂_comb {α β γ : Type} {R : α → γ → Prop} {S : β → γ → Prop} {Q : α → β → Prop} :
    ∀ {l : List α} {m : List β} {zs : List γ}, List.Forall₂ R l zs → List.Forall₂ S m zs →
      (∀ x ∈ l, ∀ y ∈ m, ∀ z, R x z → S y z → Q x y) → List.Forall₂ Q l m
  | _, _, _, .nil, .nil, _ => .nil
  | _, _, _, .cons hr h, .cons hs h', hQ =>
    .cons (hQ _ List.mem_cons_self _ List.mem_cons_self _ hr hs)
      (forall₂_comb h h' fun x hx y hy z => hQ x (List.mem_cons_of_mem _ hx) y (List.mem_cons_of_mem _ hy) z)

theorem forall₂_map_eq {α β γ : Type} {f : α → γ} {g : β → γ} :
    ∀ {l : List α} {m : List β}, List.Forall₂ (fun x y => f x = g y) l m → l.map f = m.map g
  | _, _, .nil => rfl
  | _, _, .cons h hs => by simp [h, forall₂_map_eq hs]

theorem forall₂_imp {α β : Type} {R S : α → β → Prop} (hRS : ∀ x y, R x y → S x y) :
    ∀ {l : List α} {m : List β}, List.Forall₂ R l m → List.Forall₂ S l m
  | _, _, .nil => .nil
  | _, _, .cons h hs => .cons (hRS _ _ h) (forall₂_imp hRS hs)

/-! ## The texts `canonUtf8` prints -/

theorem toList_lit (s : String) (h : s = "null" ∨ s = "true" ∨ s = "false") :
    s.toList = ['n', 'u', 'l', 'l'] ∨ s.toList = ['t', 'r', 'u', 'e'] ∨ s.toList = ['f', 'a', 'l', 's', 'e'] := by
  rcases h with rfl | rfl | rfl
  · exact .inl rfl
  · exact .inr (.inl rfl)
  · exact .inr (.inr rfl)

theorem text_null {s : String} (h : Flock.canonUtf8 .null = .ok s) : s.toList = ['n', 'u', 'l', 'l'] := by
  rw [Flock.canonUtf8] at h
  cases h
  rfl

theorem text_bool {b : Bool} {s : String} (h : Flock.canonUtf8 (.bool b) = .ok s) :
    s.toList = if b then ['t', 'r', 'u', 'e'] else ['f', 'a', 'l', 's', 'e'] := by
  rw [Flock.canonUtf8] at h
  cases h
  cases b <;> rfl

theorem text_num {n : JsonNumber} {s : String} (h : Flock.canonUtf8 (.num n) = .ok s) :
    n.exponent = 0 ∧ s = toString n.mantissa := by
  rw [Flock.canonUtf8] at h
  by_cases he : n.exponent = 0
  · simp only [he, beq_self_eq_true, ite_true, pure, Except.pure, Except.ok.injEq] at h
    exact ⟨he, h.symm⟩
  · simp [he, throw, throwThe, MonadExceptOf.throw] at h

theorem text_str {x s : String} (h : Flock.canonUtf8 (.str x) = .ok s) : s = Flock.canonStrUtf8 x := by
  rw [Flock.canonUtf8] at h
  cases h
  rfl

theorem text_arr {a : Array Json} {s : String} (h : Flock.canonUtf8 (.arr a) = .ok s) :
    ∃ ps : List String, a.toList.mapM Flock.canonUtf8 = .ok ps ∧
      s.toList = '[' :: ([','].intercalate (ps.map String.toList) ++ [']']) := by
  rw [canon_arr] at h
  cases hm : a.mapM Flock.canonUtf8 with
  | error e => simp [hm, bind, Except.bind] at h
  | ok parts =>
    simp only [hm, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨parts.toList, ?_, ?_⟩
    · rw [← Array.toList_mapM, hm]; rfl
    · simp [String.toList_append, String.toList_intercalate]

/-- An object's item as `canonUtf8` prints it: its key, a colon and its value's text. -/
def item (kv : String × String) : List Char := (Flock.canonStrUtf8 kv.1).toList ++ ':' :: kv.2.toList

/-- `canonUtf8`'s order on an object's printed fields: by key. -/
def keyLt (a b : String × String) : Bool := a.1 < b.1

theorem text_obj {kvs : Std.TreeMap.Raw String Json} {s : String} (h : Flock.canonUtf8 (.obj kvs) = .ok s) :
    ∃ vals : Array (String × String),
      kvs.toArray.toList.mapM (fun p => do pure (p.1, ← Flock.canonUtf8 p.2)) = .ok vals.toList ∧
      s.toList = '{' :: ([','].intercalate ((vals.qsort keyLt).toList.map item) ++ ['}']) := by
  rw [canon_obj] at h
  cases hm : kvs.toArray.mapM (fun p => do pure (p.1, ← Flock.canonUtf8 p.2)) with
  | error e => rw [hm] at h; simp [bind, Except.bind] at h
  | ok vals =>
    rw [hm] at h
    simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨vals, ?_, ?_⟩
    · rw [← Array.toList_mapM, hm]; rfl
    · have hi : (String.toList ∘ fun kv : String × String => Flock.canonStrUtf8 kv.1 ++ ":" ++ kv.2) = item := by
        funext kv
        simp [item, String.toList_append]
      simp only [String.toList_append, String.toList_intercalate, List.map_map, hi]
      rfl

theorem pair_ok {p : String × Json} {kv : String × String}
    (h : (do pure (p.1, ← Flock.canonUtf8 p.2) : Flock.V (String × String)) = .ok kv) :
    kv.1 = p.1 ∧ Flock.canonUtf8 p.2 = .ok kv.2 := by
  cases hc : Flock.canonUtf8 p.2 with
  | error e => simp [hc, bind, Except.bind] at h
  | ok t =>
    simp only [hc, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨rfl, rfl⟩

/-! ## Where a text starts -/

theorem digits_ne {c : Char} (h : c.isDigit = true) : c ≠ ',' ∧ c ≠ ']' ∧ c ≠ '}' ∧ c ≠ '-' ∧ c ≠ 'n' ∧ c ≠ 't' ∧
    c ≠ 'f' ∧ c ≠ '"' ∧ c ≠ '[' ∧ c ≠ '{' := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> (rintro rfl; simp at h)

theorem nat_text (n : Nat) : ∃ c rest, (toString n).toList = c :: rest ∧ c.isDigit = true ∧
    ∀ x ∈ (toString n).toList, x.isDigit = true := by
  have hd : ∀ x ∈ (toString n).toList, x.isDigit = true := fun x hx => by
    rw [Properties.Pouw.LiveSalt.Canon.toString_toList] at hx
    exact Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  cases hn : (toString n).toList with
  | nil =>
    rw [Properties.Pouw.LiveSalt.Canon.toString_toList] at hn
    exact absurd hn Nat.toDigits_ne_nil
  | cons c rest => exact ⟨c, rest, rfl, hd c (by rw [hn]; exact List.mem_cons_self), hn ▸ hd⟩

theorem int_text (m : Int) : (∃ n : Nat, m = n ∧ (toString m).toList = (toString n).toList) ∨
    ∃ n : Nat, m = Int.negSucc n ∧ (toString m).toList = '-' :: (toString (n + 1)).toList := by
  cases m with
  | ofNat n => exact .inl ⟨n, rfl, rfl⟩
  | negSucc n => exact .inr ⟨n, rfl, by simp [toString, Int.repr]⟩

/-- The class of value a text's first character starts. -/
def cls (c : Char) : Nat :=
  if c = 'n' then 0 else if c = 't' ∨ c = 'f' then 1 else if c = '"' then 3 else if c = '[' then 4
  else if c = '{' then 5 else 2

/-- A value's class. -/
def kind : Json → Nat
  | .null => 0
  | .bool _ => 1
  | .num _ => 2
  | .str _ => 3
  | .arr _ => 4
  | .obj _ => 5

theorem canonStr_head (x : String) : ∃ rest, (Flock.canonStrUtf8 x).toList = '"' :: rest :=
  ⟨_, by rw [canonStr_toList]; rfl⟩

/-- **A value's text starts with a character of its class**, which no comma or closing bracket is. -/
theorem canon_head {a : Json} {s : String} (h : Flock.canonUtf8 a = .ok s) :
    ∃ c rest, s.toList = c :: rest ∧ cls c = kind a ∧ c ≠ ',' ∧ c ≠ ']' ∧ c ≠ '}' := by
  cases a with
  | null => exact ⟨_, _, text_null h, by decide, by decide, by decide, by decide⟩
  | bool b =>
    rw [text_bool h]
    cases b
    · exact ⟨_, _, rfl, by decide, by decide, by decide, by decide⟩
    · exact ⟨_, _, rfl, by decide, by decide, by decide, by decide⟩
  | num n =>
    obtain ⟨-, rfl⟩ := text_num h
    rcases int_text n.mantissa with ⟨k, -, hk⟩ | ⟨k, -, hk⟩
    · obtain ⟨c, rest, hc, hd, -⟩ := nat_text k
      obtain ⟨h1, h2, h3, -, h5, h6, h7, h8, h9, h10⟩ := digits_ne hd
      refine ⟨c, rest, hk.trans hc, ?_, h1, h2, h3⟩
      simp [cls, kind, h5, h6, h7, h8, h9, h10]
    · exact ⟨_, _, hk, by simp [cls, kind], by decide, by decide, by decide⟩
  | str x =>
    obtain ⟨rest, hr⟩ := canonStr_head x
    exact ⟨_, _, (text_str h) ▸ hr, by simp [cls, kind], by decide, by decide, by decide⟩
  | arr a =>
    obtain ⟨ps, -, hs⟩ := text_arr h
    exact ⟨_, _, hs, by simp [cls, kind], by decide, by decide, by decide⟩
  | obj kvs =>
    obtain ⟨vals, -, hs⟩ := text_obj h
    exact ⟨_, _, hs, by simp [cls, kind], by decide, by decide, by decide⟩

theorem canon_starts {a : Json} {s : String} (h : Flock.canonUtf8 a = .ok s) : Starts s.toList := by
  obtain ⟨c, rest, hs, -, h1, h2, h3⟩ := canon_head h
  exact ⟨c, rest, hs, h1, h2, h3⟩

/-! ## Numerals -/

theorem stop_digit {r : List Char} (h : Stop r) : ∃ c rest, r = c :: rest ∧ c.isDigit = false := by
  obtain ⟨c, rest, rfl, hc⟩ := h
  refine ⟨c, rest, rfl, ?_⟩
  rcases hc with rfl | rfl | rfl <;> decide

theorem int_split {m m' : Int} {r r' : List Char} (hr : Stop r) (hr' : Stop r')
    (h : (toString m).toList ++ r = (toString m').toList ++ r') : m = m' := by
  obtain ⟨c, rest, rfl, hc⟩ := stop_digit hr
  obtain ⟨c', rest', rfl, hc'⟩ := stop_digit hr'
  rcases int_text m with ⟨k, rfl, hk⟩ | ⟨k, rfl, hk⟩ <;>
    rcases int_text m' with ⟨k', rfl, hk'⟩ | ⟨k', rfl, hk'⟩ <;> rw [hk, hk'] at h
  · rw [(nat_split hc hc' h).1]
  · obtain ⟨d, ds, hd, hdd, -⟩ := nat_text k
    rw [hd] at h
    exact absurd (List.cons.inj h).1 (digits_ne hdd).2.2.2.1
  · obtain ⟨d, ds, hd, hdd, -⟩ := nat_text k'
    rw [hd] at h
    exact absurd (List.cons.inj h).1.symm (digits_ne hdd).2.2.2.1
  · have := (nat_split hc hc' (List.cons.inj h).2).1
    rw [Nat.add_right_cancel this]

/-! ## Unique readability -/

/-- What `canon_read` concludes of two texts: they are one text, and of sorted values, of equivalent ones. -/
def Reads (a b : Json) (s t : String) : Prop := s = t ∧ (sortedKeys a = true → sortedKeys b = true → JsonEquiv a b)

theorem jsonForm_arr (a : Array Json) : jsonForm (.arr a) = .arr (.ofList (a.toList.map jsonForm)) := by
  obtain ⟨a⟩ := a
  show JsonForm.arr (.ofList (FlockDeploy.jsonForms a)) = _
  rw [jsonForms_eq]

theorem obj_lt (kvs : Std.TreeMap.Raw String Json) {p : String × Json} (h : p ∈ kvs.toArray.toList) :
    sizeOf p.2 < sizeOf (Json.obj kvs) :=
  Flock.sizeOf_lt_of_mem_toArray (Array.mem_toList_iff.1 h)

theorem arr_lt (a : Array Json) {x : Json} (h : x ∈ a.toList) : sizeOf x < sizeOf (Json.arr a) := by
  have := Array.sizeOf_lt_of_mem (Array.mem_toList_iff.1 h)
  simp only [Json.arr.sizeOf_spec]
  omega

theorem item_inj {kv kv' : String × String} {r r' : List Char}
    (h : item kv ++ r = item kv' ++ r') : kv.1 = kv'.1 ∧ kv.2.toList ++ r = kv'.2.toList ++ r' := by
  simp only [item, List.append_assoc, List.cons_append] at h
  obtain ⟨h1, h2⟩ := canonStr_split h
  exact ⟨h1, (List.cons.inj h2).2⟩

theorem item_starts (kv : String × String) : Starts (item kv) := by
  obtain ⟨rest, hr⟩ := canonStr_head kv.1
  exact ⟨'"', rest ++ ':' :: kv.2.toList, by simp [item, hr], by decide, by decide, by decide⟩

/-- `qsort` by key leaves an array whose keys strictly increase where it is. -/
theorem qsort_keys {as : Array (String × String)} (hp : (as.toList.map (·.1)).Pairwise (· < ·)) :
    as.qsort keyLt = as := by
  rw [List.pairwise_map, List.pairwise_iff_getElem] at hp
  apply Verity.QSort.qsort_sorted
  intro i hi j hj
  unfold keyLt
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · have := hp i j (by simpa using hi) (by simpa using hj) hij
    simp only [Array.getElem_toList] at this
    simp [hij, this]
  · simp [String.lt_irrefl]
  · have := hp j i (by simpa using hj) (by simpa using hi) hij
    simp only [Array.getElem_toList] at this
    simp [String.lt_asymm this, Nat.lt_asymm hij]

/-- The induction step for lists: items that read back, read back as a list. -/
theorem arr_read {a b : Array Json} {s t : String}
    (ih : ∀ x ∈ a.toList, ∀ (y : Json) (s t : String) (r r' : List Char), Flock.canonUtf8 x = .ok s →
      Flock.canonUtf8 y = .ok t → Stop r → Stop r' → s.toList ++ r = t.toList ++ r' → Reads x y s t)
    (hs : Flock.canonUtf8 (.arr a) = .ok s) (ht : Flock.canonUtf8 (.arr b) = .ok t) {r r' : List Char}
    (h : s.toList ++ r = t.toList ++ r') : Reads (.arr a) (.arr b) s t := by
  obtain ⟨ps, hps, hs⟩ := text_arr hs
  obtain ⟨qs, hqs, ht⟩ := text_arr ht
  have Fa := mapM_forall₂ hps
  have Fb := mapM_forall₂ hqs
  rw [hs, ht] at h
  simp only [List.cons_append, List.append_assoc, List.cons.injEq, true_and] at h
  have hmem : ∀ {l : List Json} {zs : List String}, List.Forall₂ (fun x y => Flock.canonUtf8 x = .ok y) l zs →
      ∀ z ∈ zs.map String.toList, ∃ x ∈ l, ∃ p, Flock.canonUtf8 x = .ok p ∧ z = p.toList := by
    intro l zs F z hz
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hz
    obtain ⟨x, hx, hxp⟩ := forall₂_mem_right F p hp
    exact ⟨x, hx, p, hxp, rfl⟩
  obtain ⟨he, -⟩ := inter_read (.inl rfl)
    (fun z hz => by obtain ⟨x, -, p, hp, rfl⟩ := hmem Fa z hz; exact canon_starts hp)
    (fun z hz => by obtain ⟨x, -, p, hp, rfl⟩ := hmem Fb z hz; exact canon_starts hp)
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
theorem obj_read {m n : Std.TreeMap.Raw String Json} {s t : String}
    (ih : ∀ p ∈ m.toArray.toList, ∀ (y : Json) (s t : String) (r r' : List Char), Flock.canonUtf8 p.2 = .ok s →
      Flock.canonUtf8 y = .ok t → Stop r → Stop r' → s.toList ++ r = t.toList ++ r' → Reads p.2 y s t)
    (hs : Flock.canonUtf8 (.obj m) = .ok s) (ht : Flock.canonUtf8 (.obj n) = .ok t) {r r' : List Char}
    (h : s.toList ++ r = t.toList ++ r') : Reads (.obj m) (.obj n) s t := by
  obtain ⟨vs, hvs, hs⟩ := text_obj hs
  obtain ⟨ws, hws, ht⟩ := text_obj ht
  have Fm := mapM_forall₂ hvs
  have Fn := mapM_forall₂ hws
  rw [hs, ht] at h
  simp only [List.cons_append, List.append_assoc, List.cons.injEq, true_and] at h
  have hmem : ∀ {k : Std.TreeMap.Raw String Json} {zs : Array (String × String)},
      List.Forall₂ (fun p kv => (do pure (p.1, ← Flock.canonUtf8 p.2) : Flock.V (String × String)) = .ok kv)
        k.toArray.toList zs.toList →
      ∀ z ∈ (zs.qsort keyLt).toList.map item, ∃ p ∈ k.toArray.toList, ∃ kv, kv.1 = p.1 ∧
        Flock.canonUtf8 p.2 = .ok kv.2 ∧ z = item kv := by
    intro k zs F z hz
    obtain ⟨kv, hkv, rfl⟩ := List.mem_map.1 hz
    have hkv' : kv ∈ zs.toList := (Properties.RemoteAudit.qsort_toList_perm zs keyLt).mem_iff.1 hkv
    obtain ⟨p, hp, hpk⟩ := forall₂_mem_right F kv hkv'
    obtain ⟨h1, h2⟩ := pair_ok hpk
    exact ⟨p, hp, kv, h1, h2, rfl⟩
  obtain ⟨he, -⟩ := inter_read (.inr rfl)
    (fun z hz => by obtain ⟨-, -, kv, -, -, rfl⟩ := hmem Fm z hz; exact item_starts kv)
    (fun z hz => by obtain ⟨-, -, kv, -, -, rfl⟩ := hmem Fn z hz; exact item_starts kv)
    (fun z hz z' hz' r1 r1' h1 h1' he => by
      obtain ⟨p, hp, kv, -, hkv, rfl⟩ := hmem Fm z hz
      obtain ⟨q, -, kv', -, hkv', rfl⟩ := hmem Fn z' hz'
      obtain ⟨hk, hv⟩ := item_inj he
      have := (ih p hp q.2 kv.2 kv'.2 r1 r1' hkv hkv' h1 h1' hv).1
      simp only [item, hk, this])
    h
  have hinj : ∀ a b : String × String, item a = item b → a = b := by
    intro a b e
    obtain ⟨h1, h2⟩ := item_inj (r := []) (r' := []) (by simpa using e)
    exact Prod.ext h1 (String.toList_inj.1 (by simpa using h2))
  have hq : (vs.qsort keyLt).toList = (ws.qsort keyLt).toList := (List.map_inj_right hinj).1 he
  refine ⟨String.toList_inj.1 (by rw [hs, ht, hq]), fun sa sb => ?_⟩
  have keys : ∀ {k : Std.TreeMap.Raw String Json} {zs : Array (String × String)}, sortedKeys (.obj k) = true →
      List.Forall₂ (fun p kv => (do pure (p.1, ← Flock.canonUtf8 p.2) : Flock.V (String × String)) = .ok kv)
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

/-- **Canonical JSON reads back**: two values' texts, each followed by a comma or a closing bracket, agree only if the
texts are one text, and two sorted values print one text only if they are equivalent. -/
theorem canon_read : ∀ (N : Nat) (a b : Json) (s t : String) (r r' : List Char), sizeOf a < N →
    Flock.canonUtf8 a = .ok s → Flock.canonUtf8 b = .ok t → Stop r → Stop r' → s.toList ++ r = t.toList ++ r' →
    Reads a b s t := by
  intro N
  induction N with
  | zero => intro _ _ _ _ _ _ h; omega
  | succ N ihN =>
    intro a b s t r r' hN hs ht hr hr' h
    have hk : kind a = kind b := by
      obtain ⟨c, rest, hc, hca, -⟩ := canon_head hs
      obtain ⟨c', rest', hc', hcb, -⟩ := canon_head ht
      rw [hc, hc'] at h
      rw [← hca, ← hcb, (List.cons.inj h).1]
    cases a <;> cases b <;> simp only [kind] at hk <;> try omega
    · exact ⟨String.toList_inj.1 ((text_null hs).trans (text_null ht).symm), fun _ _ => rfl⟩
    · rename_i x y
      have e1 := text_bool hs
      have e2 := text_bool ht
      rw [e1, e2] at h
      cases x <;> cases y <;> simp at h
      all_goals exact ⟨String.toList_inj.1 (e1.trans e2.symm), fun _ _ => rfl⟩
    · rename_i x y
      obtain ⟨hx, rfl⟩ := text_num hs
      obtain ⟨hy, rfl⟩ := text_num ht
      have e := int_split hr hr' h
      obtain ⟨xm, xe⟩ := x
      obtain ⟨ym, ye⟩ := y
      simp only at hx hy e
      subst hx hy e
      exact ⟨rfl, fun _ _ => rfl⟩
    · rename_i x y
      rw [text_str hs, text_str ht] at h ⊢
      obtain ⟨rfl, -⟩ := canonStr_split h
      exact ⟨rfl, fun _ _ => rfl⟩
    · rename_i a b
      exact arr_read (fun x hx y s t r r' => ihN x y s t r r' (by have := arr_lt a hx; omega)) hs ht h
    · rename_i m n
      exact obj_read (fun p hp y s t r r' => ihN p.2 y s t r r' (by have := obj_lt m hp; omega)) hs ht h

/-- **Two sorted values with one canonical text are equivalent.** -/
theorem canon_inj {a b : Json} {s : String} (ha : sortedKeys a = true) (hb : sortedKeys b = true)
    (hs : Flock.canonUtf8 a = .ok s) (ht : Flock.canonUtf8 b = .ok s) : JsonEquiv a b :=
  (canon_read _ a b s s ['}'] ['}'] (Nat.lt_succ_self _) hs ht (stop_cons (.inr (.inr rfl)))
    (stop_cons (.inr (.inr rfl))) rfl).2 ha hb

/-! ## Field reads -/

theorem sortedKeys_null : sortedKeys .null = true := by
  unfold sortedKeys
  rfl

theorem getD_obj (m : Std.TreeMap.Raw String Json) (k : String) : getD (.obj m) k = (m.get? k).getD .null := by
  unfold getD field? Json.getObjVal?
  dsimp only
  cases m.get? k <;> rfl

theorem getD_of_not_obj {a : Json} (h : ¬ ∃ m, a = .obj m) (k : String) : getD a k = .null := by
  cases a with
  | obj m => exact absurd ⟨m, rfl⟩ h
  | _ => rfl

theorem form_obj {a : Json} {fs : FieldForms} (h : jsonForm a = .obj fs) : ∃ m, a = .obj m := by
  cases a with
  | obj m => exact ⟨m, rfl⟩
  | arr x => rw [jsonForm_arr] at h; cases h
  | _ => cases h

theorem lookup_toList {l : List (String × Json)} {k : String} {v : Json} (h : l.lookup k = some v) : (k, v) ∈ l := by
  induction l with
  | nil => cases h
  | cons p l ih =>
    obtain ⟨k', v'⟩ := p
    by_cases hk : k = k'
    · subst hk
      simp only [List.lookup, beq_self_eq_true, Option.some.injEq] at h
      subst h
      exact List.mem_cons_self
    · simp only [List.lookup, beq_eq_false_iff_ne.2 hk] at h
      exact List.mem_cons_of_mem _ (ih h)

theorem lookup_forms {l l' : List (String × Json)}
    (h : l.map (fun p => (p.1, jsonForm p.2)) = l'.map (fun p => (p.1, jsonForm p.2))) (k : String) :
    (l.lookup k).map jsonForm = (l'.lookup k).map jsonForm := by
  induction l generalizing l' with
  | nil => cases l' with
    | nil => rfl
    | cons _ _ => cases h
  | cons p l ih =>
    cases l' with
    | nil => cases h
    | cons q l' =>
      obtain ⟨k1, v1⟩ := p
      obtain ⟨k2, v2⟩ := q
      simp only [List.map_cons, List.cons.injEq, Prod.mk.injEq] at h
      obtain ⟨⟨rfl, hv⟩, ht⟩ := h
      by_cases hk : k = k1
      · subst hk
        simp only [List.lookup, beq_self_eq_true, Option.map_some, hv]
      · simp only [List.lookup, beq_eq_false_iff_ne.2 hk]
        exact ih ht

/-- **A field read of an object whose keys increase is a lookup** in its fields' list. `Json.parse` builds objects
whose trees nothing here proves well-formed; increasing keys in the tree's order are all the search needs. -/
theorem get?_lookup {m : Std.TreeMap.Raw String Json} (h : (m.toList.map (·.1)).Pairwise (· < ·)) (k : String) :
    m.get? k = m.toList.lookup k := by
  obtain ⟨⟨t⟩⟩ := m
  change Impl.Const.get? t k = (Impl.Const.toList t).lookup k
  change ((Impl.Const.toList t).map (·.1)).Pairwise (· < ·) at h
  rw [Impl.Const.toList_eq_toListModel_map, List.map_map] at h
  rw [Impl.Const.toList_eq_toListModel_map]
  have hto : t.Ordered := by
    unfold Impl.Ordered
    rw [List.pairwise_map] at h
    exact h.imp fun hab => compareOfLessAndEq_eq_lt.2 hab
  rw [Impl.Const.get?_eq_getValue? hto]
  clear h hto
  induction t.toListModel with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k', v'⟩ := p
    by_cases hk : k = k'
    · subst hk
      simp [Std.Internal.List.getValue?]
    · simp [Std.Internal.List.getValue?, List.lookup, beq_eq_false_iff_ne.2 hk, Ne.symm hk, ih]

theorem sorted_toList {m : Std.TreeMap.Raw String Json} (h : sortedKeys (.obj m) = true) :
    (m.toList.map (·.1)).Pairwise (· < ·) := by
  rw [← Std.TreeMap.Raw.toList_toArray]
  exact increasing_iff.1 (sortedKeys_obj.1 h).1

theorem getD_lookup {m : Std.TreeMap.Raw String Json} (h : sortedKeys (.obj m) = true) (k : String) :
    getD (.obj m) k = (m.toList.lookup k).getD .null := by
  rw [getD_obj, get?_lookup (sorted_toList h)]

/-- **Equivalent sorted values have equivalent fields.** -/
theorem getD_equiv {a b : Json} (ha : sortedKeys a = true) (hb : sortedKeys b = true) (h : JsonEquiv a b)
    (k : String) : JsonEquiv (getD a k) (getD b k) := by
  by_cases hA : ∃ m, a = .obj m
  · obtain ⟨m, rfl⟩ := hA
    obtain ⟨n, rfl⟩ := form_obj (Eq.trans (Eq.symm h) (jsonForm_obj m))
    have hf := h
    rw [JsonEquiv, jsonForm_obj, jsonForm_obj, JsonForm.obj.injEq, FieldForms.ofList_inj] at hf
    have hl := lookup_forms hf k
    rw [getD_lookup ha, getD_lookup hb]
    revert hl
    cases m.toList.lookup k <;> cases n.toList.lookup k <;> intro hl <;> simp only [Option.map_none,
      Option.map_some, reduceCtorEq, Option.some.injEq] at hl
    · exact JsonEquiv.refl _
    · exact hl
  · have hB : ¬ ∃ n, b = .obj n := fun ⟨n, hn⟩ => hA (form_obj (Eq.trans h (hn ▸ jsonForm_obj n)))
    rw [getD_of_not_obj hA, getD_of_not_obj hB]
    exact JsonEquiv.refl _

/-- **Every field of a sorted value is sorted.** -/
theorem sorted_getD {a : Json} (ha : sortedKeys a = true) (k : String) : sortedKeys (getD a k) = true := by
  by_cases hA : ∃ m, a = .obj m
  · obtain ⟨m, rfl⟩ := hA
    rw [getD_lookup ha]
    cases hl : m.toList.lookup k with
    | none => exact sortedKeys_null
    | some v =>
      have hm := lookup_toList hl
      rw [← Std.TreeMap.Raw.toList_toArray] at hm
      exact (sortedKeys_obj.1 ha).2 _ (Array.mem_toList_iff.1 hm)
  · rw [getD_of_not_obj hA]
    exact sortedKeys_null

theorem equiv_str {x : Json} {s : String} (h : JsonEquiv x (.str s)) : x = .str s := by
  cases x with
  | str t => cases h; rfl
  | arr a => rw [JsonEquiv, jsonForm_arr] at h; cases h
  | obj m => rw [JsonEquiv, jsonForm_obj] at h; cases h
  | _ => cases h

theorem equiv_bool {x : Json} {b : Bool} (h : JsonEquiv x (.bool b)) : x = .bool b := by
  cases x with
  | bool c => cases h; rfl
  | arr a => rw [JsonEquiv, jsonForm_arr] at h; cases h
  | obj m => rw [JsonEquiv, jsonForm_obj] at h; cases h
  | _ => cases h

/-! ## An entry without its `sig` -/

theorem unsigned_obj (kvs : Std.TreeMap.Raw String Json) :
    Core.Record.unsigned (.obj kvs) = .obj (Std.TreeMap.Raw.ofList (kvs.toList.filter (·.1 != "sig")) compare) := by
  rw [← Std.TreeMap.Raw.toList_toArray]
  rfl

theorem filter_keys {l : List (String × Json)} (h : (l.map (·.1)).Pairwise (· < ·)) :
    ((l.filter (·.1 != "sig")).map (·.1)).Pairwise (· < ·) := by
  rw [List.pairwise_map] at h ⊢
  exact h.sublist List.filter_sublist

theorem distinct_of_keys {l : List (String × Json)} (h : (l.map (·.1)).Pairwise (· < ·)) :
    l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq) := by
  rw [List.pairwise_map] at h
  exact h.imp fun {a b} hab he => by
    rw [(Std.LawfulEqCmp.compare_eq_iff_eq (cmp := compare)).1 he] at hab
    exact String.lt_irrefl _ hab

theorem lookup_none {l : List (String × Json)} {k : String} (h : l.lookup k = none) :
    (l.map Prod.fst).contains k = false := by
  induction l with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k', v'⟩ := p
    by_cases hk : k = k'
    · subst hk
      simp [List.lookup] at h
    · simp only [List.lookup, beq_eq_false_iff_ne.2 hk] at h
      simp only [List.map_cons, List.contains_cons, ih h, Bool.or_false]
      exact beq_eq_false_iff_ne.2 hk

theorem get?_ofList {l : List (String × Json)} (h : (l.map (·.1)).Pairwise (· < ·)) (k : String) :
    (Std.TreeMap.Raw.ofList l compare).get? k = l.lookup k := by
  cases hl : l.lookup k with
  | some v =>
    exact Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.LawfulEqCmp.compare_eq_iff_eq.2 rfl) (distinct_of_keys h) (lookup_toList hl)
  | none => exact Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false (lookup_none hl)

theorem lookup_filter {l : List (String × Json)} {k : String} (hk : k ≠ "sig") :
    (l.filter (·.1 != "sig")).lookup k = l.lookup k := by
  induction l with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k', v'⟩ := p
    by_cases hs : k' = "sig"
    · subst hs
      simp only [List.filter, bne_self_eq_false, List.lookup, beq_eq_false_iff_ne.2 hk, ih]
    · have hs' : (k' != "sig") = true := bne_iff_ne.2 hs
      by_cases hk' : k = k'
      · subst hk'
        simp only [List.filter, hs', List.lookup, beq_self_eq_true]
      · simp only [List.filter, hs', List.lookup, beq_eq_false_iff_ne.2 hk', ih]

/-- **An entry without its `sig` has the entry's other fields.** -/
theorem getD_unsigned {kvs : Std.TreeMap.Raw String Json} (h : sortedKeys (.obj kvs) = true) {k : String}
    (hk : k ≠ "sig") : getD (Core.Record.unsigned (.obj kvs)) k = getD (.obj kvs) k := by
  rw [unsigned_obj, getD_obj, get?_ofList (filter_keys (sorted_toList h)), getD_lookup h, lookup_filter hk]

/-- **An entry without its `sig` is sorted** when the entry is. -/
theorem sorted_unsigned {kvs : Std.TreeMap.Raw String Json} (h : sortedKeys (.obj kvs) = true) :
    sortedKeys (Core.Record.unsigned (.obj kvs)) = true := by
  have hL := filter_keys (sorted_toList h)
  have hwf : (Std.TreeMap.Raw.ofList (kvs.toList.filter (·.1 != "sig")) compare).WF := Std.TreeMap.Raw.WF.ofList
  rw [unsigned_obj, sortedKeys_obj, increasing_iff, Std.TreeMap.Raw.toList_toArray]
  refine ⟨?_, fun p hp => ?_⟩
  · rw [List.pairwise_map]
    exact (Std.TreeMap.Raw.ordered_keys_toList hwf).imp fun hab => compareOfLessAndEq_eq_lt.1 hab
  · have hp' := Array.mem_toList_iff.2 hp
    rw [Std.TreeMap.Raw.toList_toArray] at hp'
    have hg := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hwf).1 hp'
    rw [← Std.TreeMap.Raw.get?_eq_getElem?, get?_ofList hL] at hg
    have hm := List.mem_filter.1 (lookup_toList hg)
    rw [← Std.TreeMap.Raw.toList_toArray] at hm
    exact (sortedKeys_obj.1 h).2 _ (Array.mem_toList_iff.1 hm.1)

end Properties.Challenger.CanonRead
