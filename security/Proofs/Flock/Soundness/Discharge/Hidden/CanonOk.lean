module

public import Flock.Statement
public import Std.Data.TreeMap.Raw.Lemmas
public import Std.Data.TreeMap.Raw.WF
import all Lean.Data.Json.Basic

/-!
# Canonical text that exists

`canon` and `canonUtf8` refuse only a number that is not an integer, so each returns on an object exactly when it
returns on each of the object's values (`canon_obj`, `canonUtf8_obj`). So a value put into an object keeps the
object's canonical text when the value has one (`canonUtf8_insert`), an object built from a list has one when each
value of the list has (`canonUtf8_mkObj`), and the hello has one at every statement (`helloOf_isOk`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Hidden

open Flock
open Lean (Json)

theorem isOk_bind_iff {ε α β : Type} {x : Except ε α} {f : α → Except ε β} :
    (x >>= f).isOk = true ↔ ∃ a, x = .ok a ∧ (f a).isOk = true := by
  cases x with
  | error e => exact ⟨fun h => (by cases h), fun ⟨a, h, _⟩ => (by cases h)⟩
  | ok a => exact ⟨fun h => ⟨a, rfl, h⟩, fun ⟨b, h, hb⟩ => (by cases h; exact hb)⟩

theorem mapM_list_isOk {α β ε : Type} (f : α → Except ε β) :
    ∀ l : List α, (l.mapM f).isOk = true ↔ ∀ x ∈ l, (f x).isOk = true
  | [] => ⟨fun _ _ hx => (by cases hx), fun _ => rfl⟩
  | x :: xs => by
    rw [List.mapM_cons, isOk_bind_iff]
    constructor
    · rintro ⟨a, ha, h⟩ y hy
      obtain ⟨b, hb, -⟩ := isOk_bind_iff.1 h
      rcases List.mem_cons.1 hy with rfl | hy
      · rw [ha]; rfl
      · exact (mapM_list_isOk f xs).1 (by rw [hb]; rfl) y hy
    · intro h
      have hx := h x List.mem_cons_self
      cases hfx : f x with
      | error e => rw [hfx] at hx; cases hx
      | ok a =>
        refine ⟨a, rfl, isOk_bind_iff.2 ?_⟩
        have hxs := (mapM_list_isOk f xs).2 fun y hy => h y (List.mem_cons_of_mem _ hy)
        cases hm : xs.mapM f with
        | error e => rw [hm] at hxs; cases hxs
        | ok bs => exact ⟨bs, rfl, rfl⟩

theorem mapM_array_isOk {α β ε : Type} (f : α → Except ε β) (a : Array α) :
    (a.mapM f).isOk = true ↔ ∀ x ∈ a, (f x).isOk = true := by
  rw [Array.mapM_eq_mapM_toList]
  have h := mapM_list_isOk f a.toList
  constructor
  · intro ha x hx
    refine h.1 ?_ x (Array.mem_toList_iff.2 hx)
    revert ha
    cases a.toList.mapM f <;> intro ha <;> first | rfl | cases ha
  · intro ha
    have := h.2 fun x hx => ha x (Array.mem_toList_iff.1 hx)
    revert this
    cases a.toList.mapM f <;> intro h <;> first | rfl | cases h

/-- **`canonUtf8` of an object returns exactly when it returns on each member.** -/
theorem canonUtf8_obj (kvs : Std.TreeMap.Raw String Json) :
    (canonUtf8 (.obj kvs)).isOk = true ↔ ∀ p ∈ kvs.toArray, (canonUtf8 p.2).isOk = true := by
  rw [canonUtf8, Array.mapM_subtype (g := fun p => do pure (p.1, ← canonUtf8 p.2)) (fun _ _ => rfl),
    Array.unattach_attach, isOk_bind_iff]
  constructor
  · rintro ⟨vals, hv, -⟩ p hp
    have := (mapM_array_isOk _ _).1 (isOk_of_eq hv) p hp
    obtain ⟨s, hs, -⟩ := isOk_bind_iff.1 this
    exact isOk_of_eq hs
  · intro h
    have : (kvs.toArray.mapM fun (p : String × Json) => do pure (p.1, ← canonUtf8 p.2)).isOk = true :=
      (mapM_array_isOk _ _).2 fun p hp => isOk_bind_iff.2 (by
        cases hc : canonUtf8 p.2 with
        | error e => have := h p hp; rw [hc] at this; cases this
        | ok s => exact ⟨s, rfl, rfl⟩)
    cases hm : kvs.toArray.mapM fun (p : String × Json) => do pure (p.1, ← canonUtf8 p.2) with
    | error e => rw [hm] at this; cases this
    | ok vals => exact ⟨vals, rfl, rfl⟩

/-- **`canon` of an object returns exactly when it returns on each member.** -/
theorem canon_obj (kvs : Std.TreeMap.Raw String Json) :
    (canon (.obj kvs)).isOk = true ↔ ∀ p ∈ kvs.toArray, (canon p.2).isOk = true := by
  rw [canon, Array.mapM_subtype (g := fun p => do pure (p.1, ← canon p.2)) (fun _ _ => rfl),
    Array.unattach_attach, isOk_bind_iff]
  constructor
  · rintro ⟨vals, hv, -⟩ p hp
    have := (mapM_array_isOk _ _).1 (isOk_of_eq hv) p hp
    obtain ⟨s, hs, -⟩ := isOk_bind_iff.1 this
    exact isOk_of_eq hs
  · intro h
    have : (kvs.toArray.mapM fun (p : String × Json) => do pure (p.1, ← canon p.2)).isOk = true :=
      (mapM_array_isOk _ _).2 fun p hp => isOk_bind_iff.2 (by
        cases hc : canon p.2 with
        | error e => have := h p hp; rw [hc] at this; cases this
        | ok s => exact ⟨s, rfl, rfl⟩)
    cases hm : kvs.toArray.mapM fun (p : String × Json) => do pure (p.1, ← canon p.2) with
    | error e => rw [hm] at this; cases this
    | ok vals => exact ⟨vals, rfl, rfl⟩

/-- A member of an object with one key set is the value set or a member of the object. -/
theorem mem_insert_toArray {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {k : String} {v : Json}
    {p : String × Json} (hp : p ∈ (m.insert k v).toArray) : p.2 = v ∨ p ∈ m.toArray := by
  have h := (Std.TreeMap.Raw.toArray_insert_perm hm (k := k) (v := v)).mem_iff.1 hp
  rcases Array.mem_push.1 h with h | h
  · exact .inr (Array.mem_filter.1 h).1
  · exact .inl (by rw [h])

/-- **`canonUtf8` returns on an object with one key set** when it returned on the object and returns on the value. -/
theorem canonUtf8_insert {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {k : String} {v : Json}
    (h : (canonUtf8 (.obj m)).isOk = true) (hv : (canonUtf8 v).isOk = true) :
    (canonUtf8 (.obj (m.insert k v))).isOk = true :=
  (canonUtf8_obj _).2 fun p hp => by
    rcases mem_insert_toArray hm hp with h' | h'
    · rw [h']; exact hv
    · exact (canonUtf8_obj _).1 h p h'

theorem canonUtf8_nat (n : Nat) : (canonUtf8 (Lean.toJson n)).isOk = true := by
  rw [show Lean.toJson n = Json.num (Lean.JsonNumber.fromNat n) from rfl, canonUtf8]
  rfl

theorem canonUtf8_str (s : String) : (canonUtf8 (Json.str s)).isOk = true := by
  rw [canonUtf8]
  rfl

theorem canonUtf8_arr {a : Array Json} (h : ∀ x ∈ a, (canonUtf8 x).isOk = true) :
    (canonUtf8 (.arr a)).isOk = true := by
  rw [canonUtf8, Array.mapM_subtype (g := fun x => canonUtf8 x) (fun _ _ => rfl), Array.unattach_attach]
  have := (mapM_array_isOk _ _).2 h
  cases hm : a.mapM fun x => canonUtf8 x with
  | error e => rw [hm] at this; cases this
  | ok parts => rfl

/-- A member of a list's object is a member of the list. -/
theorem mem_insertMany {l : List (String × Json)} : ∀ {m : Std.TreeMap.Raw String Json compare}, m.WF →
    ∀ {p : String × Json}, p ∈ (m.insertMany l).toArray → p ∈ m.toArray ∨ p.2 ∈ l.map Prod.snd := by
  induction l with
  | nil => intro m _ p hp; exact .inl (by simpa only [Std.TreeMap.Raw.insertMany_nil] using hp)
  | cons kv l ih =>
    intro m hm p hp
    rw [Std.TreeMap.Raw.insertMany_cons] at hp
    rcases ih hm.insert hp with h | h
    · rcases mem_insert_toArray hm h with h | h
      · exact .inr (by rw [h, List.map_cons]; exact List.mem_cons_self)
      · exact .inl h
    · exact .inr (by simp only [List.map_cons, List.mem_cons]; exact .inr h)

/-- **`canonUtf8` returns on a list's object** when it returns on each value of the list. -/
theorem canonUtf8_mkObj {l : List (String × Json)} (h : ∀ v ∈ l.map Prod.snd, (canonUtf8 v).isOk = true) :
    (canonUtf8 (Json.mkObj l)).isOk = true := by
  refine (canonUtf8_obj _).2 fun p hp => ?_
  rw [Std.TreeMap.Raw.ofList_eq_insertMany_empty] at hp
  rcases mem_insertMany Std.TreeMap.Raw.WF.emptyc hp with h' | h'
  · have he : (∅ : Std.TreeMap.Raw String Json compare).toArray = #[] :=
      Array.isEmpty_iff.1 (by rw [Std.TreeMap.Raw.isEmpty_toArray]; exact Std.TreeMap.Raw.isEmpty_emptyc)
    rw [he] at h'
    exact absurd h' (Array.not_mem_empty _)
  · exact h _ h'

/-! ## The hello always exists -/

theorem canon_nat (n : Nat) : (canon (Lean.toJson n)).isOk = true := by
  rw [show Lean.toJson n = Json.num (Lean.JsonNumber.fromNat n) from rfl, canon]
  rfl

theorem canon_str (s : String) : (canon (Json.str s)).isOk = true := by
  rw [canon]
  rfl

theorem canon_arr {a : Array Json} (h : ∀ x ∈ a, (canon x).isOk = true) : (canon (.arr a)).isOk = true := by
  rw [canon, Array.mapM_subtype (g := fun x => canon x) (fun _ _ => rfl), Array.unattach_attach]
  have := (mapM_array_isOk _ _).2 h
  cases hm : a.mapM fun x => canon x with
  | error e => rw [hm] at this; cases this
  | ok parts => rfl

theorem canon_insert {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {k : String} {v : Json}
    (h : (canon (.obj m)).isOk = true) (hv : (canon v).isOk = true) : (canon (.obj (m.insert k v))).isOk = true :=
  (canon_obj _).2 fun p hp => by
    rcases mem_insert_toArray hm hp with h' | h'
    · rw [h']; exact hv
    · exact (canon_obj _).1 h p h'

theorem canon_mkObj {l : List (String × Json)} (h : ∀ v ∈ l.map Prod.snd, (canon v).isOk = true) :
    (canon (Json.mkObj l)).isOk = true := by
  refine (canon_obj _).2 fun p hp => ?_
  rw [Std.TreeMap.Raw.ofList_eq_insertMany_empty] at hp
  rcases mem_insertMany Std.TreeMap.Raw.WF.emptyc hp with h' | h'
  · have he : (∅ : Std.TreeMap.Raw String Json compare).toArray = #[] :=
      Array.isEmpty_iff.1 (by rw [Std.TreeMap.Raw.isEmpty_toArray]; exact Std.TreeMap.Raw.isEmpty_emptyc)
    rw [he] at h'
    exact absurd h' (Array.not_mem_empty _)
  · exact h _ h'

theorem canon_strs (a : Array String) : (canon (Json.arr (a.map Json.str))).isOk = true :=
  canon_arr fun x hx => by
    obtain ⟨s, -, rfl⟩ := Array.mem_map.1 hx
    exact canon_str s

theorem canon_coins (s : CoinTree.Spec) (nonce : String) : (canon (s.coins nonce)).isOk = true := by
  have hj : (canon s.toJson).isOk = true := canon_mkObj fun v hv => by
    simp only [List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl | rfl
    · exact canon_nat _
    · exact canon_nat _
    · exact canon_str _
    · exact canon_strs _
  unfold CoinTree.Spec.coins
  unfold CoinTree.Spec.toJson at hj ⊢
  exact canon_insert Std.TreeMap.Raw.WF.ofList hj (canon_str _)

/-- **`helloOf` returns** at every tags, Σ, count of link points, coins and tables: its object holds strings, natural
numbers and the coin tree's spec, and `canon` refuses only a number that is not an integer. -/
theorem helloOf_isOk (tags : Tags) (sigma : ByteArray) (mPts : Nat) (coins : Coins) (tables : Array String) :
    (helloOf tags sigma mPts coins tables).isOk = true := by
  have hl : (canon (Json.mkObj [("sigma", Json.str (hexOf sigma)), ("m", Lean.toJson mPts),
      ("points", Lean.toJson (2 : Nat))])).isOk = true := canon_mkObj fun v hv => by
    simp only [List.map_cons, List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl
    · exact canon_str _
    · exact canon_nat _
    · exact canon_nat _
  unfold helloOf
  refine canon_mkObj fun v hv => ?_
  cases coins <;> cases hr : tags.hm96Rows <;> cases hq : tags.retainRounds <;>
    simp only [hr, hq, Bool.false_eq_true, ↓reduceIte, List.map_cons, List.map_nil, List.cons_append,
      List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hv <;>
    rcases hv with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    first
      | exact canon_str _
      | exact canon_nat _
      | exact canon_strs _
      | exact hl
      | exact canon_coins _ _

end FlockSoundness.Discharge.Hidden
