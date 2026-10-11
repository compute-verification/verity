import Mathlib.Data.List.Perm.Basic
import Batteries.Tactic.OpenPrivate

/-!
# `qsort` commutes with a map its order reads through, and keeps every element

`Array.qsort` only compares and swaps, so sorting `as.map g` by an order on the images is sorting `as` by the order
pulled back through `g`, then mapping (`qsort_map`); and each swap is a permutation, so the result holds every element
of the input (`mem_qsort`).
-/

open private Array.qpartition.loop Array.qsort.sort from Init.Data.Array.QSort.Basic

namespace MemoryChallenger.Properties.SecurityProofs.Sort

variable {α β : Type}

theorem map_swap {n : Nat} (g : α → β) (as : Vector α n) (i j : Nat) (hi : i < n) (hj : j < n) :
    (as.map g).swap i j hi hj = (as.swap i j hi hj).map g := by
  ext k hk
  simp only [Vector.getElem_swap, Vector.getElem_map]
  split_ifs <;> rfl

theorem map_ite_swap {n : Nat} (g : α → β) (as : Vector α n) (b : Bool) (i j : Nat) (hi : i < n) (hj : j < n) :
    (if b then (as.map g).swap i j hi hj else as.map g) = (if b then as.swap i j hi hj else as).map g := by
  split <;> simp [map_swap]

theorem loop_map {n : Nat} (g : α → β) (lt : β → β → Bool) {lo hi : Nat} (hhi : hi < n) (pivot : α) :
    ∀ (d : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      Array.qpartition.loop lt lo hi hhi (g pivot) (as.map g) i k ilo ik w =
        let r := Array.qpartition.loop (fun x y => lt (g x) (g y)) lo hi hhi pivot as i k ilo ik w
        (r.1, r.2.map g) := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as i k ilo ik w hd
    unfold Array.qpartition.loop
    split
    · rename_i hk
      rw [Vector.getElem_map]
      split
      · rw [map_swap]
        exact ih (hi - (k + 1)) (by omega) _ _ _ _ _ _ rfl
      · exact ih (hi - (k + 1)) (by omega) _ _ _ _ _ _ rfl
    · simp [map_swap]

theorem qpartition_map {n : Nat} (g : α → β) (lt : β → β → Bool) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi)
    (hlo : lo < n) (hhi : hi < n) :
    Array.qpartition (as.map g) lt lo hi w hlo hhi =
      let r := Array.qpartition as (fun x y => lt (g x) (g y)) lo hi w hlo hhi
      (r.1, r.2.map g) := by
  unfold Array.qpartition
  simp only [Vector.getElem_map, map_ite_swap]
  exact loop_map g lt hhi _ _ _ _ _ _ _ _ rfl

theorem sort_map {n : Nat} (g : α → β) (lt : β → β → Bool) :
    ∀ (d : Nat) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo = d →
      Array.qsort.sort lt (as.map g) lo hi w hlo hhi =
        (Array.qsort.sort (fun x y => lt (g x) (g y)) as lo hi w hlo hhi).map g := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as lo hi w hlo hhi hd
    rw [Array.qsort.sort, Array.qsort.sort]
    split
    · rw [qpartition_map]
      dsimp only
      generalize Array.qpartition as (fun x y => lt (g x) (g y)) lo hi w hlo hhi = q
      obtain ⟨⟨mid, hmid⟩, as'⟩ := q
      dsimp only
      split
      · rfl
      · rename_i h₁ h₂
        rw [ih (mid - lo) (by omega) _ _ _ _ _ _ rfl, ih (hi - (mid + 1)) (by omega) _ _ _ _ _ _ rfl]
    · rfl

theorem sort_congr {n n' : Nat} (h : n = n') (lt : β → β → Bool) (v : Vector β n) (v' : Vector β n')
    (hv : v.toArray = v'.toArray) {lo hi lo' hi' : Nat} (hl : lo = lo') (hh : hi = hi') (w : lo ≤ hi) (hlo : lo < n)
    (hhi : hi < n) (w' : lo' ≤ hi') (hlo' : lo' < n') (hhi' : hi' < n') :
    (Array.qsort.sort lt v lo hi w hlo hhi).toArray = (Array.qsort.sort lt v' lo' hi' w' hlo' hhi').toArray := by
  subst h hl hh
  obtain rfl : v = v' := Vector.toArray_inj.mp hv
  rfl

/-- **`qsort` commutes with a map its order reads through.** -/
theorem qsort_map (g : α → β) (lt : β → β → Bool) (as : Array α) :
    (as.map g).qsort lt = (as.qsort fun x y => lt (g x) (g y)).map g := by
  unfold Array.qsort
  have hs : (as.map g).size = as.size := Array.size_map
  split
  · rename_i h0
    have : as.size = 0 := hs ▸ h0
    simp [this]
  · rename_i h0
    have h0' : ¬ as.size = 0 := hs ▸ h0
    simp only [h0', ↓reduceDIte]
    rw [← Vector.toArray_map, ← sort_map g lt _ _ _ _ _ _ _ rfl]
    exact sort_congr hs lt _ _ (by simp) (by rw [hs]) (by rw [hs]) _ _ _ _ _ _

theorem loop_perm {n : Nat} (lt : α → α → Bool) {lo hi : Nat} (hhi : hi < n) (pivot : α) :
    ∀ (d : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w).2.Perm as := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as i k ilo ik w hd
    unfold Array.qpartition.loop
    split
    · split
      · exact (ih (hi - (k + 1)) (by omega) _ _ _ _ _ _ rfl).trans (Vector.swap_perm _ _)
      · exact ih (hi - (k + 1)) (by omega) _ _ _ _ _ _ rfl
    · exact Vector.swap_perm (by omega) hhi

theorem ite_swap_perm {n : Nat} (as : Vector α n) (b : Bool) (i j : Nat) (hi : i < n) (hj : j < n) :
    (if b then as.swap i j hi hj else as).Perm as := by
  split
  · exact Vector.swap_perm _ _
  · exact Vector.Perm.refl _

theorem qpartition_perm {n : Nat} (lt : α → α → Bool) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n)
    (hhi : hi < n) : (Array.qpartition as lt lo hi w hlo hhi).2.Perm as := by
  unfold Array.qpartition
  exact (loop_perm lt hhi _ _ _ _ _ _ _ _ rfl).trans
    (((ite_swap_perm _ _ _ _ _ _).trans (ite_swap_perm _ _ _ _ _ _)).trans (ite_swap_perm _ _ _ _ _ _))

theorem sort_perm {n : Nat} (lt : α → α → Bool) :
    ∀ (d : Nat) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo = d →
      (Array.qsort.sort lt as lo hi w hlo hhi).Perm as := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as lo hi w hlo hhi hd
    rw [Array.qsort.sort]
    split
    · have hp := qpartition_perm lt as lo hi w hlo hhi
      revert hp
      generalize Array.qpartition as lt lo hi w hlo hhi = q
      obtain ⟨⟨mid, hmid⟩, as'⟩ := q
      intro hp
      dsimp only at hp ⊢
      split
      · exact hp
      · exact ((ih (hi - (mid + 1)) (by omega) _ _ _ _ _ _ rfl).trans
          (ih (mid - lo) (by omega) _ _ _ _ _ _ rfl)).trans hp
    · exact Vector.Perm.refl _

/-- **`qsort` keeps every element.** -/
theorem mem_qsort (lt : α → α → Bool) {as : Array α} {x : α} (hx : x ∈ as) : x ∈ as.qsort lt := by
  unfold Array.qsort
  split
  · exact hx
  · exact (Vector.Perm.toArray (sort_perm lt _ as.toVector _ _ _ _ _ rfl)).mem_iff.2 hx

/-- **`qsort` adds no element.** -/
theorem qsort_mem (lt : α → α → Bool) {as : Array α} {x : α} (hx : x ∈ as.qsort lt) : x ∈ as := by
  revert hx
  unfold Array.qsort
  split
  · exact id
  · exact (Vector.Perm.toArray (sort_perm lt _ as.toVector _ _ _ _ _ rfl)).mem_iff.1

end MemoryChallenger.Properties.SecurityProofs.Sort
