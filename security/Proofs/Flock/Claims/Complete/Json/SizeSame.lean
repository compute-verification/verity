module

public import Proofs.Flock.Claims.Complete.Json.Size
public import Proofs.Flock.Claims.Complete.Json.Value
import Proofs.Flock.Claims.Complete.Json.CanonSize
import Std.Data.TreeMap.Raw.Lemmas
import all Lean.Data.Json.Basic

/-!
# Alike JSON values weigh alike

`jsz_same`: two alike values (`Same`) have one weight, since an object's weight is the sum over its keys of the key's
and its value's, and alike objects have one key set (`sum_eq_of_keys`). `jsz_insert_le`: a field inserted into an
object adds at most its own weight.
-/

@[expose] public section

namespace FlockComplete

open Lean (Json JsonNumber)
open Claims.Json (Same)

theorem sum_eq_of_keys (w : String → Json → ℕ) :
    ∀ (la lb : List (String × Json)), (la.map Prod.fst).Nodup → (lb.map Prod.fst).Nodup →
      (∀ k, k ∈ la.map Prod.fst ↔ k ∈ lb.map Prod.fst) →
      (∀ k a b, (k, a) ∈ la → (k, b) ∈ lb → w k a = w k b) →
      (la.map fun p => w p.1 p.2).sum = (lb.map fun p => w p.1 p.2).sum
  | [], lb, _, _, hk, _ => by
    have : lb = [] := by
      cases lb with
      | nil => rfl
      | cons p lb' =>
        exfalso
        have := (hk p.1).2 (by simp)
        simp at this
    subst this
    rfl
  | (k, a) :: la', lb, hna, hnb, hk, hw => by
    have hkl : k ∈ lb.map Prod.fst := (hk k).1 (by simp)
    obtain ⟨⟨k', b⟩, hmem, hk'⟩ := List.mem_map.1 hkl
    simp only at hk'
    subst hk'
    obtain ⟨l1, l2, hl⟩ := List.append_of_mem hmem
    have hperm : lb.Perm ((k', b) :: (l1 ++ l2)) := hl ▸ List.perm_middle
    have hperm2 := hperm.map Prod.fst
    have hnb' := hperm2.nodup_iff.1 hnb
    simp only [List.map_cons, List.nodup_cons] at hnb' hna
    have hrec := sum_eq_of_keys w la' (l1 ++ l2) hna.2 hnb'.2 (fun x => by
        have h1 := hk x
        have h2 := hperm2.mem_iff (a := x)
        simp only [List.map_cons, List.mem_cons] at h1 h2
        by_cases hx : x = k'
        · subst hx
          exact ⟨fun h => absurd h hna.1, fun h => absurd h hnb'.1⟩
        · constructor
          · intro h
            have := h1.1 (Or.inr h)
            rcases h2.1 this with h3 | h3
            · exact absurd h3 hx
            · exact h3
          · intro h
            have := h1.2 (h2.2 (Or.inr h))
            rcases this with h3 | h3
            · exact absurd h3 hx
            · exact h3) (fun x a' b' h1 h2 => hw x a' b' (List.mem_cons_of_mem _ h1) (by
          rw [hl]
          simp only [List.mem_append, List.mem_cons] at h2 ⊢
          rcases h2 with h | h
          · exact Or.inl h
          · exact Or.inr (Or.inr h)))
    have hs := (hperm.map fun p : String × Json => w p.1 p.2).sum_nat
    rw [hs]
    simp only [List.map_cons, List.sum_cons, List.map_append, List.sum_append] at hrec ⊢
    have := hw k' a b (by simp) hmem
    omega

/-- **Alike values weigh alike.** -/
theorem jsz_same : ∀ {a b : Json}, Same a b → jsz a = jsz b := by
  intro a b h
  induction h with
  | null => rfl
  | bool b => rfl
  | num n => rfl
  | str s => rfl
  | arr hs he ih =>
    rename_i a b
    rw [jsz_arr, jsz_arr]
    have : a.toList.map (fun j => jsz j + 1) = b.toList.map (fun j => jsz j + 1) := by
      apply List.ext_getElem
      · simp [hs]
      · intro i h1 h2
        simp only [List.getElem_map, Array.getElem_toList]
        rw [ih i (by simpa using h1) (by simpa using h2)]
    rw [this]
  | obj hm hn hk hv ih =>
    rename_i m n
    rw [JsonRT.jsz_obj_sum, JsonRT.jsz_obj_sum]
    congr 1
    refine sum_eq_of_keys (fun k v => 18 * k.length + 4 + jsz v) m.toList n.toList ?_ ?_ ?_ ?_
    · rw [Std.TreeMap.Raw.map_fst_toList_eq_keys]
      exact Std.TreeMap.Raw.nodup_keys hm
    · rw [Std.TreeMap.Raw.map_fst_toList_eq_keys]
      exact Std.TreeMap.Raw.nodup_keys hn
    · intro k
      rw [Std.TreeMap.Raw.map_fst_toList_eq_keys, Std.TreeMap.Raw.map_fst_toList_eq_keys,
        Std.TreeMap.Raw.mem_keys hm, Std.TreeMap.Raw.mem_keys hn, Std.TreeMap.Raw.mem_iff_isSome_getElem? hm,
        Std.TreeMap.Raw.mem_iff_isSome_getElem? hn]
      have := hk k
      simp only [Std.TreeMap.Raw.get?_eq_getElem?] at this
      simp [this]
    · intro k a b ha hb
      rw [Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm] at ha
      rw [Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hn] at hb
      simp only [← Std.TreeMap.Raw.get?_eq_getElem?] at ha hb
      have := ih k a b ha hb
      simp only [this]

theorem sum_filter_le {α : Type} (f : α → ℕ) (p : α → Bool) :
    ∀ l : List α, ((l.filter p).map f).sum ≤ (l.map f).sum
  | [] => Nat.le_refl _
  | a :: l => by
    have ih := sum_filter_le f p l
    rw [List.filter_cons]
    split <;> simp only [List.map_cons, List.sum_cons] <;> omega

/-- **A field inserted into an object adds at most its weight.** -/
theorem jsz_insert_le {m : Std.TreeMap.Raw String Json} (hm : m.WF) (k : String) (v : Json) :
    jsz (.obj (m.insert k v)) ≤ jsz (.obj m) + (18 * k.length + 4 + jsz v) := by
  rw [JsonRT.jsz_obj_sum, JsonRT.jsz_obj_sum]
  have hp := (Std.TreeMap.Raw.toList_insert_perm (t := m) hm (k := k) (v := v)).map
    (fun q : String × Json => 18 * q.1.length + 4 + jsz q.2)
  rw [hp.sum_nat]
  have hs := sum_filter_le (fun q : String × Json => 18 * q.1.length + 4 + jsz q.2)
    (fun x : String × Json => decide ¬(k == x.1) = true) m.toList
  simp only [List.map_cons, List.sum_cons]
  omega

end FlockComplete
