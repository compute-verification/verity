module

public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.Size
import Proofs.Core.QSortPerm
import Std.Data.TreeMap.Raw.Lemmas
import all Lean.Data.Json.Basic

/-!
# A canonical text's length

`canon`'s bytes are at most `jsz`, as `printJ`'s are (`canon_size_le`): it writes a character as at most twelve ASCII
characters (`esc_length`), and an object's fields, sorted, weigh what its tree's do (`Verity.QSort.qsort_perm`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace FlockComplete.JsonRT

open Lean (Json JsonNumber)
open Flock
open Std.DTreeMap.Internal (Impl)
open private hex4 from Base.Canon

theorem length_le_ofList_size : ∀ l : List Char, l.length ≤ (String.ofList l).utf8ByteSize
  | [] => by decide +kernel
  | c :: l => by
    rw [String.ofList_cons, String.utf8ByteSize_append, String.utf8ByteSize_singleton, List.length_cons]
    have h1 := Char.utf8Size_pos c
    have h2 := length_le_ofList_size l
    omega

/-- A string has at most as many characters as bytes. -/
theorem length_le_size (s : String) : s.length ≤ s.utf8ByteSize := by
  have h := length_le_ofList_size s.toList
  rwa [String.ofList_toList, String.length_toList] at h

theorem ofList_size_ascii : ∀ l : List Char, (∀ x ∈ l, x.toNat < 0x7f) → (String.ofList l).utf8ByteSize = l.length
  | [], _ => by decide +kernel
  | c :: l, h => by
    rw [String.ofList_cons, String.utf8ByteSize_append, String.utf8ByteSize_singleton, List.length_cons,
      ofList_size_ascii l fun x hx => h x (List.mem_cons_of_mem _ hx)]
    have hc : c.utf8Size = 1 := Char.utf8Size_eq_one_iff.2 (UInt32.le_iff_toNat_le.2
      (show c.val.toNat ≤ 127 from Nat.le_of_lt (h c (List.mem_cons_self ..))))
    omega

theorem escU_length (n : Nat) : (Flock.escU n).toList.length ≤ 12 := by
  have h2 : ("\\u" : String).toList.length = 2 := by decide +kernel
  have h4 : ∀ m, (hex4 m).toList.length = 4 := fun m => by simp [hex4]
  unfold Flock.escU
  split <;> simp only [String.toList_append, List.length_append, h2, h4] <;> omega

/-- `canonStr` writes a character as at most twelve characters. -/
theorem esc_length (c : Char) : (Ascii.esc c).length ≤ 12 := by
  unfold Ascii.esc
  split
  all_goals first
    | (simp; done)
    | ((repeat' split) <;> first | (simp; done) | exact escU_length _)

theorem flatMap_esc_length : ∀ l : List Char, (l.flatMap Ascii.esc).length ≤ 12 * l.length
  | [] => by simp
  | c :: l => by
    rw [List.flatMap_cons, List.length_append, List.length_cons]
    have h1 := esc_length c
    have h2 := flatMap_esc_length l
    omega

/-- **A canonical string's bytes**: at most twelve per character, all ASCII, and its two quotes. -/
theorem canonStr_size_le (t : String) : (Flock.canonStr t).utf8ByteSize ≤ 18 * t.length + 2 := by
  have h1 : (Flock.canonStr t).utf8ByteSize = (Flock.canonStr t).length := by
    have := ofList_size_ascii (Flock.canonStr t).toList (Ascii.canonStr_ascii t)
    rwa [String.ofList_toList, String.length_toList] at this
  have h2 := congrArg List.length (Ascii.canonStr_toList t)
  have h3 := flatMap_esc_length t.toList
  simp only [List.length_cons, List.length_append, List.length_nil, String.length_toList] at h2 h3
  omega

theorem sum_map_succ {β : Type} (w : β → ℕ) : ∀ l : List β, (l.map fun s => w s + 1).sum = (l.map w).sum + l.length
  | [] => rfl
  | _ :: l => by
    simp only [List.map_cons, List.sum_cons, List.length_cons, sum_map_succ w l]
    omega

theorem intercalate_le (l : List String) : (",".intercalate l).utf8ByteSize ≤ (l.map fun s => s.utf8ByteSize + 1).sum := by
  rw [sum_map_succ]
  exact intercalate_size_le l

theorem forall₂_sum_le {α β : Type} {P : α → β → Prop} (w : β → ℕ) (f : α → ℕ) :
    ∀ {xs : List α} {ys : List β}, List.Forall₂ P xs ys → (∀ x y, P x y → w y ≤ f x) →
      (ys.map w).sum ≤ (xs.map f).sum
  | [], [], .nil, _ => le_refl _
  | x :: _, y :: _, .cons h t, hP => by
    simp only [List.map_cons, List.sum_cons]
    exact Nat.add_le_add (hP x y h) (forall₂_sum_le w f t hP)

theorem mapM_forall₂ {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {r : List β}, l.mapM f = .ok r → List.Forall₂ (fun a b => f a = .ok b) l r
  | [], r, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact .nil
  | a :: l, r, h => by
    rw [List.mapM_cons] at h
    cases hb : f a with
    | error e => rw [hb] at h; cases h
    | ok b =>
      cases hl : l.mapM f with
      | error e => rw [hb] at h; simp only [bind, Except.bind] at h; rw [hl] at h; cases h
      | ok bs =>
        rw [hb] at h; simp only [bind, Except.bind] at h; rw [hl] at h
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        exact .cons hb (mapM_forall₂ hl)

theorem array_mapM_forall₂ {α β ε : Type} {f : α → Except ε β} {a : Array α} {r : Array β}
    (h : a.mapM f = .ok r) : List.Forall₂ (fun x y => f x = .ok y) a.toList r.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    simpa using mapM_forall₂ hl

theorem attach_sum {α : Type} (a : Array α) (F : α → ℕ) :
    (a.attach.toList.map fun p => F p.1).sum = (a.toList.map F).sum := by
  have e : a.attach.toList.map Subtype.val = a.toList := by simp
  rw [← e, List.map_map]
  rfl

theorem jsz_obj_sum (m : Std.TreeMap.Raw String Json) :
    jsz (.obj m) = (m.toList.map fun q => 18 * q.1.length + 4 + jsz q.2).sum + 2 := by
  have e0 : jsz (.obj m) = jszF m.inner.inner + 2 := by
    obtain ⟨⟨t⟩⟩ := m
    rw [jsz]
    all_goals rfl
  have e2 : m.toList = m.inner.inner.toListModel.map fun e => (e.1, e.2) := by
    show Impl.Const.toList m.inner.inner = _
    exact Impl.Const.toList_eq_toListModel_map
  rw [e0, jszF_model, e2, List.map_map]
  rfl

/-- `canon_size_le` below a size `k`. -/
theorem canon_size_lt : ∀ (k : Nat) (j : Json) {s : String}, sizeOf j < k → Flock.canon j = .ok s →
    s.utf8ByteSize ≤ jsz j
  | 0, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | k + 1, j, s, hk, h => by
    have hl : ("[" : String).utf8ByteSize = 1 := by decide +kernel
    have hr : ("]" : String).utf8ByteSize = 1 := by decide +kernel
    have hl' : ("{" : String).utf8ByteSize = 1 := by decide +kernel
    have hr' : ("}" : String).utf8ByteSize = 1 := by decide +kernel
    match j, h with
    | .null, h => rw [Flock.canon] at h; cases h; decide +kernel
    | .bool b, h => rw [Flock.canon] at h; cases h; cases b <;> decide +kernel
    | .num ⟨m, e⟩, h =>
      rw [Flock.canon] at h
      split at h
      · rename_i he
        simp only [beq_iff_eq] at he
        subst he
        cases h
        exact Nat.le_refl _
      · cases h
    | .str t, h => rw [Flock.canon] at h; cases h; exact canonStr_size_le t
    | .arr a, h =>
      rw [Flock.canon] at h
      cases hp : a.attach.mapM (fun x => Flock.canon x.1) with
      | error e => rw [hp] at h; cases h
      | ok parts =>
        rw [hp] at h
        simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
        subst h
        rw [String.utf8ByteSize_append, String.utf8ByteSize_append, hl, hr, jsz_arr]
        have hI := intercalate_le parts.toList
        have hF := forall₂_sum_le (fun s : String => s.utf8ByteSize + 1) (fun x : {x // x ∈ a} => jsz x.1 + 1)
          (array_mapM_forall₂ hp) fun x p hxp => by
            have hlt := Array.sizeOf_lt_of_mem x.2
            simp only [Json.arr.sizeOf_spec] at hk
            have := canon_size_lt k x.1 (by omega) hxp
            omega
        have hA := attach_sum a fun j => jsz j + 1
        omega
    | .obj m, h =>
      rw [Flock.canon] at h
      cases hp : m.toArray.attach.mapM
          (fun p => do pure (p.1.1, ← Flock.canon p.1.2) : _ → Flock.V (String × String)) with
      | error e => rw [hp] at h; cases h
      | ok vals =>
        rw [hp] at h
        simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
        subst h
        rw [String.utf8ByteSize_append, String.utf8ByteSize_append, hl', hr', jsz_obj_sum m]
        refine Nat.le_trans (Nat.add_le_add_right (Nat.add_le_add_left (intercalate_le _) 1) 1) ?_
        rw [List.map_map, ((Verity.QSort.qsort_perm vals _).toList.map _).sum_eq]
        have hF := forall₂_sum_le
          ((fun s : String => s.utf8ByteSize + 1) ∘ fun kv : String × String => Flock.canonStr kv.1 ++ ":" ++ kv.2)
          (fun p : {p // p ∈ m.toArray} => 18 * p.1.1.length + 4 + jsz p.1.2) (array_mapM_forall₂ hp)
          fun p kv hkv => by
            cases hc : Flock.canon p.1.2 with
            | error e => rw [hc] at hkv; cases hkv
            | ok v =>
              rw [hc] at hkv
              simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hkv
              subst hkv
              have hlt := Flock.sizeOf_lt_of_mem_toArray p.2
              have ih := canon_size_lt k p.1.2 (by omega) hc
              have hks := canonStr_size_le p.1.1
              have hcolon : (":" : String).utf8ByteSize = 1 := by decide +kernel
              simp only [Function.comp, String.utf8ByteSize_append, hcolon]
              omega
        have hA := attach_sum m.toArray fun q : String × Json => 18 * q.1.length + 4 + jsz q.2
        rw [Std.TreeMap.Raw.toList_toArray] at hA
        omega

/-- **`canon`'s bytes are at most `jsz`**, as `printJ`'s are (`printJ_size_le`). -/
theorem canon_size_le (j : Json) {s : String} (h : Flock.canon j = .ok s) : s.utf8ByteSize ≤ jsz j :=
  canon_size_lt _ j (Nat.lt_succ_self _) h

end FlockComplete.JsonRT
