import Batteries.Tactic.OpenPrivate
import Proofs.Properties.PoubCheck.Sort

/-!
# `Array.qsort` sorts strings

PoUW's check compares a registration's item names, sorted with `Array.qsort (· < ·)` (`PouwCheck.keys`,
`PouwWindow.sortedKeys`), with `registeredSorted` and `lengthKeys`. `qsort_sorted` is that the sort's output is sorted,
the proof of `Proofs.Properties.Nci.ReadsExact.qsort_sorted` (a sort by a `Nat` key) at strings' own order; with
`qsort_perm` it gives `qsort_eq_of_perm`, that the sort of a list depends only on its elements, and
`qsort_of_sorted`, that it leaves a sorted list as it is.
-/

open private Array.qpartition.loop Array.qsort.sort from Init.Data.Array.QSort.Basic

namespace Properties.Pouw.Period.SecurityProofs.Sort

theorem le_of_lt' {a b : String} (h : a < b) : a ≤ b := String.not_lt.1 (String.lt_asymm h)

/-- `bs` agrees with `as` outside `[lo, hi]`, and each of its elements in `[lo, hi]` is one of `as`'s there. -/
def Within {n : Nat} (as bs : Vector String n) (lo hi : Nat) : Prop :=
  (∀ t (h : t < n), (t < lo ∨ hi < t) → bs[t] = as[t]) ∧
  (∀ t (h : t < n), lo ≤ t → t ≤ hi → ∃ s, ∃ hs : s < n, lo ≤ s ∧ s ≤ hi ∧ bs[t] = as[s])

theorem within_refl {n : Nat} (as : Vector String n) (lo hi : Nat) : Within as as lo hi :=
  ⟨fun _ _ _ => rfl, fun t h h1 h2 => ⟨t, h, h1, h2, rfl⟩⟩

theorem within_trans {n : Nat} {as bs cs : Vector String n} {lo hi : Nat} (h1 : Within as bs lo hi)
    (h2 : Within bs cs lo hi) : Within as cs lo hi := by
  refine ⟨fun t h ht => (h2.1 t h ht).trans (h1.1 t h ht), fun t h a b => ?_⟩
  obtain ⟨s, hs, s1, s2, e⟩ := h2.2 t h a b
  obtain ⟨u, hu, u1, u2, e'⟩ := h1.2 s hs s1 s2
  exact ⟨u, hu, u1, u2, e.trans e'⟩

theorem within_mono {n : Nat} {as bs : Vector String n} {lo hi lo' hi' : Nat} (h : Within as bs lo hi)
    (h1 : lo' ≤ lo) (h2 : hi ≤ hi') : Within as bs lo' hi' := by
  refine ⟨fun t ht h' => h.1 t ht (by omega), fun t ht a b => ?_⟩
  by_cases hin : lo ≤ t ∧ t ≤ hi
  · obtain ⟨s, hs, s1, s2, e⟩ := h.2 t ht hin.1 hin.2
    exact ⟨s, hs, by omega, by omega, e⟩
  · exact ⟨t, ht, a, b, h.1 t ht (by omega)⟩

theorem within_swap {n : Nat} (as : Vector String n) {lo hi i j : Nat} (hi' : i < n) (hj : j < n) (h1 : lo ≤ i)
    (h2 : i ≤ hi) (h3 : lo ≤ j) (h4 : j ≤ hi) : Within as (as.swap i j hi' hj) lo hi := by
  refine ⟨fun t h ht => ?_, fun t h a b => ?_⟩
  · rw [Vector.getElem_swap_of_ne (by omega) (by omega)]
  · rw [Vector.getElem_swap]
    split
    · exact ⟨j, hj, h3, h4, rfl⟩
    · split
      · exact ⟨i, hi', h1, h2, rfl⟩
      · exact ⟨t, h, a, b, rfl⟩

theorem within_ite_swap {n : Nat} (as : Vector String n) (b : Bool) {lo hi i j : Nat} (hi' : i < n) (hj : j < n)
    (h1 : lo ≤ i) (h2 : i ≤ hi) (h3 : lo ≤ j) (h4 : j ≤ hi) :
    Within as (if b then as.swap i j hi' hj else as) lo hi := by
  split
  · exact within_swap as hi' hj h1 h2 h3 h4
  · exact within_refl as lo hi

/-- `qpartition`'s loop: below `i` the elements are below the pivot, from `i` to `k` not, the pivot at `hi`; it ends
with the pivot at its result, below `hi` when some element from `i` is not below it. -/
theorem qpartition_loop_spec {n : Nat} {lo hi : Nat} (hhi : hi < n) (pivot : String) :
    ∀ (d : Nat) (as : Vector String n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (∀ t (h : t < n), lo ≤ t → t < i → as[t] < pivot) →
      (∀ t (h : t < n), i ≤ t → t < k → pivot ≤ as[t]) →
      as[hi] = pivot →
      (∃ j, ∃ hj : j < n, i ≤ j ∧ j < hi ∧ pivot ≤ as[j]) →
      let r := Array.qpartition.loop (fun x y => decide (x < y)) lo hi hhi pivot as i k ilo ik w
      ∃ hm : r.1.1 < hi, Within as r.2 lo hi ∧
        (∀ t (h : t < n), lo ≤ t → t < r.1.1 → r.2[t] < pivot) ∧
        r.2[r.1.1] = pivot ∧
        (∀ t (h : t < n), r.1.1 < t → t ≤ hi → pivot ≤ r.2[t]) := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as i k ilo ik w hd hlt hge hp hex
    dsimp only
    unfold Array.qpartition.loop
    split
    · rename_i hk
      split
      · rename_i hc
        have hc' : as[k] < pivot := of_decide_eq_true hc
        obtain ⟨hm, hw, h1, h2, h3⟩ := ih (hi - (k + 1)) (by omega) (as.swap i k (by omega) (by omega)) (i + 1)
          (k + 1) (by omega) (by omega) (by omega) rfl
          (fun t ht a b => by
            rw [Vector.getElem_swap]
            split
            · rename_i e; subst e; exact hc'
            · split
              · omega
              · exact hlt t ht a (by omega))
          (fun t ht a b => by
            rw [Vector.getElem_swap]
            split
            · omega
            · split
              · exact hge i (by omega) (Nat.le_refl _) (by omega)
              · exact hge t ht (by omega) (by omega))
          (by rw [Vector.getElem_swap_of_ne (by omega) (by omega)]; exact hp)
          (by
            obtain ⟨j, hj, j1, j2, hj'⟩ := hex
            by_cases hji : j = i
            · subst hji
              have hik : j ≠ k := fun e => by subst e; exact String.not_lt.2 hj' hc'
              refine ⟨k, by omega, by omega, hk, ?_⟩
              rw [Vector.getElem_swap_right]; exact hj'
            · have hjk : j ≠ k := fun e => by subst e; exact String.not_lt.2 hj' hc'
              refine ⟨j, hj, by omega, j2, ?_⟩
              rw [Vector.getElem_swap_of_ne hji hjk]; exact hj')
        exact ⟨hm, within_trans (within_swap as _ _ ilo (by omega) (by omega) (by omega)) hw, h1, h2, h3⟩
      · rename_i hc
        have hc' : ¬ as[k] < pivot := fun h => hc (decide_eq_true h)
        exact ih (hi - (k + 1)) (by omega) as i (k + 1) ilo (by omega) (by omega) rfl hlt
          (fun t ht a b => by
            by_cases e : t = k
            · subst e; exact String.not_lt.1 hc'
            · exact hge t ht a (by omega)) hp hex
    · rename_i hk
      have hkh : k = hi := by omega
      subst hkh
      obtain ⟨j, hj, j1, j2, hj'⟩ := hex
      refine ⟨by dsimp only; omega, within_swap as (by omega) hhi ilo (by omega) (by omega) (Nat.le_refl _), ?_, ?_, ?_⟩
      · intro t ht a b
        dsimp only at b ⊢
        rw [Vector.getElem_swap_of_ne (by omega) (by omega)]
        exact hlt t ht a b
      · dsimp only
        rw [Vector.getElem_swap_left]; exact hp
      · intro t ht a b
        dsimp only at a ⊢
        rw [Vector.getElem_swap]
        split
        · omega
        · split
          · exact hge i (by omega) (Nat.le_refl _) (by omega)
          · exact hge t ht (by omega) (by omega)

/-- `qpartition`'s third swap leaves its pivot at most the middle element. -/
theorem median_le {n : Nat} (as : Vector String n) (i j : Nat) (hi : i < n) (hj : j < n) :
    (if decide (as[i] < as[j]) = true then as.swap i j hi hj else as)[j] ≤
      (if decide (as[i] < as[j]) = true then as.swap i j hi hj else as)[i] := by
  split
  · rename_i hc
    rw [Vector.getElem_swap_left, Vector.getElem_swap_right]
    exact le_of_lt' (of_decide_eq_true hc)
  · rename_i hc
    exact String.not_lt.1 fun h => hc (decide_eq_true h)

theorem qpartition_core {n : Nat} (as as3 : Vector String n) (lo hi : Nat) (w : lo ≤ hi) (hhi : hi < n)
    (hlh : lo < hi) (hmid : (lo + hi) / 2 < n) (hW : Within as as3 lo hi) (hmed : as3[hi] ≤ as3[(lo + hi) / 2]) :
    let r := Array.qpartition.loop (fun x y => decide (x < y)) lo hi hhi as3[hi] as3 lo lo (Nat.le_refl _)
      (Nat.le_refl _) w
    r.1.1 < hi ∧ Within as r.2 lo hi ∧ ∃ v : String,
      (∀ t (h : t < n), lo ≤ t → t ≤ r.1.1 → r.2[t] ≤ v) ∧ (∀ t (h : t < n), r.1.1 < t → t ≤ hi → v ≤ r.2[t]) := by
  obtain ⟨hm, hw, h1, h2, h3⟩ := qpartition_loop_spec hhi as3[hi] (hi - lo) as3 lo lo (Nat.le_refl _)
    (Nat.le_refl _) w rfl (fun t _ a b => by omega) (fun t _ a b => by omega) rfl
    ⟨(lo + hi) / 2, hmid, by omega, by omega, hmed⟩
  refine ⟨hm, within_trans hW hw, as3[hi], fun t ht a b => ?_, h3⟩
  rcases Nat.lt_or_ge t (Array.qpartition.loop (fun x y => decide (x < y)) lo hi hhi as3[hi] as3 lo lo
    (Nat.le_refl _) (Nat.le_refl _) w).1.1 with b' | b'
  · exact le_of_lt' (h1 t ht a b')
  · have e : t = (Array.qpartition.loop (fun x y => decide (x < y)) lo hi hhi as3[hi] as3 lo lo
      (Nat.le_refl _) (Nat.le_refl _) w).1.1 := by omega
    simp only [e, h2, String.le_refl]

/-- `qpartition` of `[lo, hi]`, `lo < hi`: its result `m` is below `hi`, the elements to `m` are at most `m`'s, after it
at least, and only `[lo, hi]` changes, to elements it held. -/
theorem qpartition_spec {n : Nat} (as : Vector String n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n)
    (hlh : lo < hi) :
    let r := Array.qpartition as (fun x y => decide (x < y)) lo hi w hlo hhi
    r.1.1 < hi ∧ Within as r.2 lo hi ∧ ∃ v : String,
      (∀ t (h : t < n), lo ≤ t → t ≤ r.1.1 → r.2[t] ≤ v) ∧ (∀ t (h : t < n), r.1.1 < t → t ≤ hi → v ≤ r.2[t]) := by
  dsimp only
  unfold Array.qpartition
  dsimp only
  refine qpartition_core as _ lo hi w hhi hlh (by omega) ?_ ?_
  · refine within_trans (within_trans (within_ite_swap as _ _ _ ?_ ?_ ?_ ?_) (within_ite_swap _ _ _ _ ?_ ?_ ?_ ?_))
      (within_ite_swap _ _ _ _ ?_ ?_ ?_ ?_) <;> omega
  · exact median_le _ _ _ _ _

/-- `qsort`'s `sort` of `[lo, hi]` sorts it, changing only it, to elements it held. -/
theorem sort_spec {n : Nat} :
    ∀ (d : Nat) (as : Vector String n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo = d →
      Within as (Array.qsort.sort (fun x y => decide (x < y)) as lo hi w hlo hhi) lo hi ∧
      ∀ i j (hi' : i < n) (hj : j < n), lo ≤ i → i ≤ j → j ≤ hi →
        (Array.qsort.sort (fun x y => decide (x < y)) as lo hi w hlo hhi)[i] ≤
          (Array.qsort.sort (fun x y => decide (x < y)) as lo hi w hlo hhi)[j] := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as lo hi w hlo hhi hd
    rw [Array.qsort.sort]
    split
    · rename_i h₁
      have hp := qpartition_spec as lo hi w hlo hhi h₁
      revert hp
      match Array.qpartition as (fun x y => decide (x < y)) lo hi w hlo hhi with
      | (⟨mid, hmid⟩, as') =>
        intro hp
        dsimp only at hp ⊢
        obtain ⟨hm, hW, v, hle, hge⟩ := hp
        split
        · omega
        · obtain ⟨hW1, hS1⟩ := ih (mid - lo) (by omega) as' lo mid hmid.1 hlo (by omega) rfl
          obtain ⟨hW2, hS2⟩ := ih (hi - (mid + 1)) (by omega)
            (Array.qsort.sort (fun x y => decide (x < y)) as' lo mid hmid.1 hlo (by omega)) (mid + 1) hi
            (by omega) (by omega) hhi rfl
          refine ⟨within_trans hW (within_trans (within_mono hW1 (Nat.le_refl _) (by omega))
            (within_mono hW2 (by omega) (Nat.le_refl _))), fun i j hi' hj a b c => ?_⟩
          by_cases hjm : j ≤ mid
          · rw [hW2.1 i hi' (by omega), hW2.1 j hj (by omega)]
            exact hS1 i j hi' hj a b hjm
          · by_cases him : mid + 1 ≤ i
            · exact hS2 i j hi' hj him b c
            · rw [hW2.1 i hi' (by omega)]
              obtain ⟨s, hs, s1, s2, es⟩ := hW1.2 i hi' a (by omega)
              obtain ⟨u, hu, u1, u2, eu⟩ := hW2.2 j hj (by omega) c
              rw [es, eu, hW1.1 u hu (by omega)]
              exact String.le_trans (hle s hs s1 s2) (hge u hu (by omega) u2)
    · refine ⟨within_refl as lo hi, fun i j hi' hj a b c => ?_⟩
      have e : i = j := by omega
      subst e; exact String.le_refl _

/-- **`Array.qsort` sorts strings.** -/
theorem qsort_sorted (as : Array String) : (as.qsort (· < ·)).toList.Pairwise (· ≤ ·) := by
  unfold Array.qsort
  split
  · rename_i h
    rw [Array.size_eq_zero_iff.1 h]
    exact List.Pairwise.nil
  · rename_i h
    dsimp only
    obtain ⟨-, hS⟩ := sort_spec _ as.toVector (min 0 (as.size - 1)) (max (min 0 (as.size - 1))
      (min (as.size - 1) (as.size - 1))) (by omega) (by omega) (by omega) rfl
    rw [List.pairwise_iff_getElem]
    intro i j hi hj hij
    simp only [Array.length_toList, Vector.size_toArray] at hi hj
    simp only [Array.getElem_toList, Vector.getElem_toArray]
    exact hS i j hi hj (by omega) (by omega) (by omega)

/-- **The sort of strings depends only on the elements**: arrays that are permutations of each other sort to one
list. -/
theorem qsort_eq_of_perm {as bs : Array String} (h : as.toList.Perm bs.toList) :
    (as.qsort (· < ·)).toList = (bs.qsort (· < ·)).toList :=
  List.Perm.eq_of_pairwise (fun _ _ _ _ h1 h2 => String.le_antisymm h1 h2) (qsort_sorted as) (qsort_sorted bs)
    (((Properties.PoubCheck.qsort_toList_perm as _).trans h).trans
      (Properties.PoubCheck.qsort_toList_perm bs _).symm)

/-- **The sort leaves a sorted array as it is.** -/
theorem qsort_of_sorted {as : Array String} (h : as.toList.Pairwise (· < ·)) :
    (as.qsort (· < ·)).toList = as.toList :=
  List.Perm.eq_of_pairwise (fun _ _ _ _ h1 h2 => String.le_antisymm h1 h2) (qsort_sorted as)
    (h.imp le_of_lt') (Properties.PoubCheck.qsort_toList_perm as _)

end Properties.Pouw.Period.SecurityProofs.Sort
