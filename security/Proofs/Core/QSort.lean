module

public import Batteries.Tactic.OpenPrivate
import all Init.Data.Array.QSort.Basic

/-!
# `Array.qsort` leaves a sorted array as it is, and sorts by a key

`qsort_sorted`: an array strictly sorted by `lt` is `qsort lt`'s own output. Each partition's pivot is the middle element,
which splits a sorted range in place (`qpartition_sorted`), and both halves stay sorted (`sort_sorted`).

`ByKey.qsort_sorted`: `qsort` by a `Nat` key sorts by it, and keeps only elements it was given (`ByKey.qsort_mem`). Each
partition leaves the elements up to its result at most a value and those after at least it (`ByKey.qpartition_spec`),
changing only its range, to elements it held (`ByKey.Within`).
-/

@[expose] public section

set_option Elab.async false

open private Array.qsort.sort Array.qpartition.loop from Init.Data.Array.QSort.Basic

namespace Verity.QSort

variable {α : Type} {n : Nat}

theorem swap_self (xs : Vector α n) (i : Nat) (h : i < n) : xs.swap i i h h = xs := by
  ext k hk
  rw [Vector.getElem_swap]
  split
  · subst_vars; rfl
  · rfl

section Loop

variable (lt : α → α → Bool) {lo hi : Nat} (hhi : hi < n) (pivot : α) (as : Vector α n)

private theorem loop_skip (i : Nat) (ilo : lo ≤ i) :
    ∀ (d k : Nat) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (∀ j (hj : j < n), k ≤ j → j < hi → lt as[j] pivot = false) →
      Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w =
        (⟨i, ilo, by omega⟩, as.swap i hi (by omega) hhi) := by
  intro d
  induction d with
  | zero =>
    intro k ik w hd _
    rw [Array.qpartition.loop.eq_def]
    have : ¬ k < hi := by omega
    simp only [this, dite_false]
  | succ d ih =>
    intro k ik w hd hb
    rw [Array.qpartition.loop.eq_def]
    have hk : k < hi := by omega
    simp only [hk, dite_true, hb k (by omega) (Nat.le_refl _) hk, Bool.false_eq_true, ite_false]
    exact ih (k + 1) (by omega) (by omega) (by omega) (fun j hj h1 h2 => hb j hj (by omega) h2)

private theorem loop_take (m : Nat) (hm : m ≤ hi) :
    ∀ (d k : Nat) (ilo : lo ≤ k) (w : k ≤ m), m - k = d →
      (∀ j (hj : j < n), k ≤ j → j < m → lt as[j] pivot = true) →
      (∀ j (hj : j < n), m ≤ j → j < hi → lt as[j] pivot = false) →
      Array.qpartition.loop lt lo hi hhi pivot as k k ilo (Nat.le_refl _) (by omega) =
        (⟨m, by omega, hm⟩, as.swap m hi (by omega) hhi) := by
  intro d
  induction d with
  | zero =>
    intro k ilo w hd _ hb
    obtain rfl : k = m := by omega
    exact loop_skip lt hhi pivot as k ilo (hi - k) k (Nat.le_refl _) _ rfl hb
  | succ d ih =>
    intro k ilo w hd ht hb
    rw [Array.qpartition.loop.eq_def]
    have hk : k < hi := by omega
    simp only [hk, dite_true, ht k (by omega) (Nat.le_refl _) (by omega), ite_true, swap_self]
    exact ih (k + 1) (by omega) (by omega) (by omega) (fun j hj h1 h2 => ht j hj (by omega) h2) hb

end Loop

/-- `as` is strictly sorted by `lt` on `[lo, hi]`: `lt` holds of two of its elements there exactly in their order. -/
def SortedOn (lt : α → α → Bool) (lo hi : Nat) (as : Vector α n) : Prop :=
  ∀ i j (hi' : i < n) (hj : j < n), lo ≤ i → i ≤ hi → lo ≤ j → j ≤ hi → lt as[i] as[j] = decide (i < j)

theorem qpartition_sorted (lt : α → α → Bool) (as : Vector α n) {lo hi : Nat} (w : lo ≤ hi)
    (hlo : lo < n) (hhi : hi < n) (hlt : lo < hi) (hs : SortedOn lt lo hi as) :
    Array.qpartition as lt lo hi w hlo hhi = (⟨(lo + hi) / 2, by omega, by omega⟩, as) := by
  have hm1 : lo ≤ (lo + hi) / 2 := by omega
  have hm2 : (lo + hi) / 2 < hi := by omega
  unfold Array.qpartition
  have h1 : lt (as[(lo + hi) / 2]'(by omega)) as[lo] = false := by
    rw [hs ((lo + hi) / 2) lo _ _ hm1 (by omega) (Nat.le_refl _) w]; simp; omega
  have h2 : lt as[hi] as[lo] = false := by
    rw [hs hi lo _ _ w (Nat.le_refl _) (Nat.le_refl _) w]; simp; omega
  have h3 : lt (as[(lo + hi) / 2]'(by omega)) as[hi] = true := by
    rw [hs ((lo + hi) / 2) hi _ _ hm1 (by omega) w (Nat.le_refl _)]; simp; omega
  simp only [h1, h2, h3, Bool.false_eq_true, ite_false, ite_true, Vector.getElem_swap_right]
  rw [loop_take lt hhi _ _ ((lo + hi) / 2) (Nat.le_of_lt hm2) _ lo (Nat.le_refl _) hm1 rfl]
  · simp only [Vector.swap_swap]
  · intro j hj h4 h5
    rw [Vector.getElem_swap_of_ne (by omega) (by omega), hs j ((lo + hi) / 2) _ _ h4 (by omega) hm1 (by omega)]
    simp; omega
  · intro j hj h4 h5
    rw [Vector.getElem_swap]
    split
    · rename_i e; subst e
      rw [hs hi ((lo + hi) / 2) _ _ w (Nat.le_refl _) hm1 (by omega)]; simp; omega
    · split
      · omega
      · rw [hs j ((lo + hi) / 2) _ _ (by omega) (by omega) hm1 (by omega)]; simp; omega

private theorem sort_sorted (lt : α → α → Bool) :
    ∀ (d : Nat) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo < d →
      SortedOn lt lo hi as → Array.qsort.sort lt as lo hi w hlo hhi = as := by
  intro d
  induction d with
  | zero => intro _ _ _ _ _ _ hd; omega
  | succ d ih =>
    intro as lo hi w hlo hhi hd hs
    rw [Array.qsort.sort.eq_def]
    split
    · rename_i hlt
      rw [qpartition_sorted lt as w hlo hhi hlt hs]
      simp only
      split
      · omega
      · rw [ih as lo ((lo + hi) / 2) _ hlo _ (by omega)
          (fun i j _ _ h1 h2 h3 h4 => hs i j _ _ h1 (by omega) h3 (by omega))]
        exact ih as ((lo + hi) / 2 + 1) hi _ _ hhi (by omega)
          (fun i j _ _ h1 h2 h3 h4 => hs i j _ _ (by omega) h2 (by omega) h4)
    · rfl

/-- `Array.qsort` leaves a strictly sorted array as it is: its pivot, the middle element, splits it in place. -/
theorem qsort_sorted (as : Array α) (lt : α → α → Bool)
    (hs : ∀ i (hi : i < as.size) j (hj : j < as.size), lt as[i] as[j] = decide (i < j)) : as.qsort lt = as := by
  unfold Array.qsort
  split
  · rfl
  · simp only
    rw [sort_sorted lt _ as.toVector _ _ _ _ _ (Nat.lt_succ_self _)
      (fun i j hi hj _ _ _ _ => by simpa using hs i hi j hj)]

/-- `qsort` keeps its array's size. -/
theorem size_qsort {α : Type} (xs : Array α) (lt : α → α → Bool) : (xs.qsort lt).size = xs.size := by
  unfold Array.qsort
  split
  · rfl
  · simp

end Verity.QSort

/-! ## `Array.qsort` sorts by a key -/

namespace Verity.QSort.ByKey

variable {α : Type} (f : α → Nat)

/-- `bs` agrees with `as` outside `[lo, hi]`, and each of its elements in `[lo, hi]` is one of `as`'s there. -/
def Within {n : Nat} (as bs : Vector α n) (lo hi : Nat) : Prop :=
  (∀ t (h : t < n), (t < lo ∨ hi < t) → bs[t] = as[t]) ∧
  (∀ t (h : t < n), lo ≤ t → t ≤ hi → ∃ s, ∃ hs : s < n, lo ≤ s ∧ s ≤ hi ∧ bs[t] = as[s])

theorem within_refl {n : Nat} (as : Vector α n) (lo hi : Nat) : Within as as lo hi :=
  ⟨fun _ _ _ => rfl, fun t h h1 h2 => ⟨t, h, h1, h2, rfl⟩⟩

theorem within_trans {n : Nat} {as bs cs : Vector α n} {lo hi : Nat} (h1 : Within as bs lo hi)
    (h2 : Within bs cs lo hi) : Within as cs lo hi := by
  refine ⟨fun t h ht => (h2.1 t h ht).trans (h1.1 t h ht), fun t h a b => ?_⟩
  obtain ⟨s, hs, s1, s2, e⟩ := h2.2 t h a b
  obtain ⟨u, hu, u1, u2, e'⟩ := h1.2 s hs s1 s2
  exact ⟨u, hu, u1, u2, e.trans e'⟩

theorem within_mono {n : Nat} {as bs : Vector α n} {lo hi lo' hi' : Nat} (h : Within as bs lo hi) (h1 : lo' ≤ lo)
    (h2 : hi ≤ hi') : Within as bs lo' hi' := by
  refine ⟨fun t ht h' => h.1 t ht (by omega), fun t ht a b => ?_⟩
  by_cases hin : lo ≤ t ∧ t ≤ hi
  · obtain ⟨s, hs, s1, s2, e⟩ := h.2 t ht hin.1 hin.2
    exact ⟨s, hs, by omega, by omega, e⟩
  · exact ⟨t, ht, a, b, h.1 t ht (by omega)⟩

theorem within_swap {n : Nat} (as : Vector α n) {lo hi i j : Nat} (hi' : i < n) (hj : j < n) (h1 : lo ≤ i)
    (h2 : i ≤ hi) (h3 : lo ≤ j) (h4 : j ≤ hi) : Within as (as.swap i j hi' hj) lo hi := by
  refine ⟨fun t h ht => ?_, fun t h a b => ?_⟩
  · rw [Vector.getElem_swap_of_ne (by omega) (by omega)]
  · rw [Vector.getElem_swap]
    split
    · exact ⟨j, hj, h3, h4, rfl⟩
    · split
      · exact ⟨i, hi', h1, h2, rfl⟩
      · exact ⟨t, h, a, b, rfl⟩

theorem within_ite_swap {n : Nat} (as : Vector α n) (b : Bool) {lo hi i j : Nat} (hi' : i < n) (hj : j < n)
    (h1 : lo ≤ i) (h2 : i ≤ hi) (h3 : lo ≤ j) (h4 : j ≤ hi) :
    Within as (if b then as.swap i j hi' hj else as) lo hi := by
  split
  · exact within_swap as hi' hj h1 h2 h3 h4
  · exact within_refl as lo hi

/-- `qpartition`'s loop: below `i` the elements are below the pivot, from `i` to `k` not, the pivot at `hi`; it ends
with the pivot at its result, below `hi` when some element from `i` is not below it. -/
private theorem qpartition_loop_spec {n : Nat} {lo hi : Nat} (hhi : hi < n) (pivot : α) :
    ∀ (d : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k = d →
      (∀ t (h : t < n), lo ≤ t → t < i → f as[t] < f pivot) →
      (∀ t (h : t < n), i ≤ t → t < k → f pivot ≤ f as[t]) →
      as[hi] = pivot →
      (∃ j, ∃ hj : j < n, i ≤ j ∧ j < hi ∧ f pivot ≤ f as[j]) →
      let r := Array.qpartition.loop (fun x y => decide (f x < f y)) lo hi hhi pivot as i k ilo ik w
      ∃ hm : r.1.1 < hi, Within as r.2 lo hi ∧
        (∀ t (h : t < n), lo ≤ t → t < r.1.1 → f r.2[t] < f pivot) ∧
        r.2[r.1.1] = pivot ∧
        (∀ t (h : t < n), r.1.1 < t → t ≤ hi → f pivot ≤ f r.2[t]) := by
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
        have hc' : f as[k] < f pivot := of_decide_eq_true hc
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
              have hik : j ≠ k := fun e => by subst e; omega
              refine ⟨k, by omega, by omega, hk, ?_⟩
              rw [Vector.getElem_swap_right]; exact hj'
            · have hjk : j ≠ k := fun e => by subst e; omega
              refine ⟨j, hj, by omega, j2, ?_⟩
              rw [Vector.getElem_swap_of_ne hji hjk]; exact hj')
        exact ⟨hm, within_trans (within_swap as _ _ ilo (by omega) (by omega) (by omega)) hw, h1, h2, h3⟩
      · rename_i hc
        have hc' : ¬ f as[k] < f pivot := fun h => hc (decide_eq_true h)
        exact ih (hi - (k + 1)) (by omega) as i (k + 1) ilo (by omega) (by omega) rfl hlt
          (fun t ht a b => by
            by_cases e : t = k
            · subst e; omega
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
theorem median_le {n : Nat} (as : Vector α n) (i j : Nat) (hi : i < n) (hj : j < n) :
    f (if decide (f as[i] < f as[j]) = true then as.swap i j hi hj else as)[j] ≤
      f (if decide (f as[i] < f as[j]) = true then as.swap i j hi hj else as)[i] := by
  split
  · rename_i hc
    rw [Vector.getElem_swap_left, Vector.getElem_swap_right]
    exact Nat.le_of_lt (of_decide_eq_true hc)
  · rename_i hc
    exact Nat.le_of_not_lt fun h => hc (decide_eq_true h)

private theorem qpartition_core {n : Nat} (as as3 : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hhi : hi < n) (hlh : lo < hi)
    (hmid : (lo + hi) / 2 < n) (hW : Within as as3 lo hi) (hmed : f as3[hi] ≤ f as3[(lo + hi) / 2]) :
    let r := Array.qpartition.loop (fun x y => decide (f x < f y)) lo hi hhi as3[hi] as3 lo lo (Nat.le_refl _)
      (Nat.le_refl _) w
    r.1.1 < hi ∧ Within as r.2 lo hi ∧ ∃ v : Nat,
      (∀ t (h : t < n), lo ≤ t → t ≤ r.1.1 → f r.2[t] ≤ v) ∧ (∀ t (h : t < n), r.1.1 < t → t ≤ hi → v ≤ f r.2[t]) := by
  obtain ⟨hm, hw, h1, h2, h3⟩ := qpartition_loop_spec f hhi as3[hi] (hi - lo) as3 lo lo (Nat.le_refl _)
    (Nat.le_refl _) w rfl (fun t _ a b => by omega) (fun t _ a b => by omega) rfl
    ⟨(lo + hi) / 2, hmid, by omega, by omega, hmed⟩
  refine ⟨hm, within_trans hW hw, f as3[hi], fun t ht a b => ?_, h3⟩
  rcases Nat.lt_or_ge t (Array.qpartition.loop (fun x y => decide (f x < f y)) lo hi hhi as3[hi] as3 lo lo
    (Nat.le_refl _) (Nat.le_refl _) w).1.1 with b' | b'
  · exact Nat.le_of_lt (h1 t ht a b')
  · have e : t = (Array.qpartition.loop (fun x y => decide (f x < f y)) lo hi hhi as3[hi] as3 lo lo
      (Nat.le_refl _) (Nat.le_refl _) w).1.1 := by omega
    simp only [e, h2, Nat.le_refl]

/-- `qpartition` of `[lo, hi]`, `lo < hi`: its result `m` is below `hi`, the elements to `m` are at most `m`'s, after it
at least, and only `[lo, hi]` changes, to elements it held. -/
theorem qpartition_spec {n : Nat} (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n)
    (hlh : lo < hi) :
    let r := Array.qpartition as (fun x y => decide (f x < f y)) lo hi w hlo hhi
    r.1.1 < hi ∧ Within as r.2 lo hi ∧ ∃ v : Nat,
      (∀ t (h : t < n), lo ≤ t → t ≤ r.1.1 → f r.2[t] ≤ v) ∧ (∀ t (h : t < n), r.1.1 < t → t ≤ hi → v ≤ f r.2[t]) := by
  dsimp only
  unfold Array.qpartition
  dsimp only
  refine qpartition_core f as _ lo hi w hhi hlh (by omega) ?_ ?_
  · refine within_trans (within_trans (within_ite_swap as _ _ _ ?_ ?_ ?_ ?_) (within_ite_swap _ _ _ _ ?_ ?_ ?_ ?_))
      (within_ite_swap _ _ _ _ ?_ ?_ ?_ ?_) <;> omega
  · exact median_le f _ _ _ _ _

/-- `qsort`'s `sort` of `[lo, hi]` sorts it by `f`, changing only it, to elements it held. -/
private theorem sort_spec {n : Nat} :
    ∀ (d : Nat) (as : Vector α n) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo = d →
      Within as (Array.qsort.sort (fun x y => decide (f x < f y)) as lo hi w hlo hhi) lo hi ∧
      ∀ i j (hi' : i < n) (hj : j < n), lo ≤ i → i ≤ j → j ≤ hi →
        f (Array.qsort.sort (fun x y => decide (f x < f y)) as lo hi w hlo hhi)[i] ≤
          f (Array.qsort.sort (fun x y => decide (f x < f y)) as lo hi w hlo hhi)[j] := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro as lo hi w hlo hhi hd
    rw [Array.qsort.sort]
    split
    · rename_i h₁
      have hp := qpartition_spec f as lo hi w hlo hhi h₁
      revert hp
      match Array.qpartition as (fun x y => decide (f x < f y)) lo hi w hlo hhi with
      | (⟨mid, hmid⟩, as') =>
        intro hp
        dsimp only at hp ⊢
        obtain ⟨hm, hW, v, hle, hge⟩ := hp
        split
        · omega
        · obtain ⟨hW1, hS1⟩ := ih (mid - lo) (by omega) as' lo mid hmid.1 hlo (by omega) rfl
          obtain ⟨hW2, hS2⟩ := ih (hi - (mid + 1)) (by omega)
            (Array.qsort.sort (fun x y => decide (f x < f y)) as' lo mid hmid.1 hlo (by omega)) (mid + 1) hi
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
              exact Nat.le_trans (hle s hs s1 s2) (hge u hu (by omega) u2)
    · refine ⟨within_refl as lo hi, fun i j hi' hj a b c => ?_⟩
      have e : i = j := by omega
      subst e; exact Nat.le_refl _

/-- **`Array.qsort` by a key sorts by it.** -/
theorem qsort_sorted (as : Array α) :
    (as.qsort (fun x y => decide (f x < f y))).toList.Pairwise (fun x y => f x ≤ f y) := by
  unfold Array.qsort
  split
  · rename_i h
    rw [Array.size_eq_zero_iff.1 h]
    exact List.Pairwise.nil
  · rename_i h
    dsimp only
    obtain ⟨-, hS⟩ := sort_spec f _ as.toVector (min 0 (as.size - 1)) (max (min 0 (as.size - 1))
      (min (as.size - 1) (as.size - 1))) (by omega) (by omega) (by omega) rfl
    rw [List.pairwise_iff_getElem]
    intro i j hi hj hij
    simp only [Array.length_toList, Vector.size_toArray] at hi hj
    simp only [Array.getElem_toList, Vector.getElem_toArray]
    exact hS i j hi hj (by omega) (by omega) (by omega)

/-- `Array.qsort` by a key keeps only elements it was given. -/
theorem qsort_mem (as : Array α) {x : α} (hx : x ∈ (as.qsort (fun x y => decide (f x < f y))).toList) :
    x ∈ as.toList := by
  unfold Array.qsort at hx
  split at hx
  · exact hx
  · rename_i h
    dsimp only at hx
    obtain ⟨hW, -⟩ := sort_spec f _ as.toVector (min 0 (as.size - 1)) (max (min 0 (as.size - 1))
      (min (as.size - 1) (as.size - 1))) (by omega) (by omega) (by omega) rfl
    obtain ⟨t, ht, e⟩ := List.getElem_of_mem hx
    simp only [Array.length_toList, Vector.size_toArray] at ht
    obtain ⟨s, hs, -, -, es⟩ := hW.2 t ht (by omega) (by omega)
    rw [← e]
    simp only [Array.getElem_toList, Vector.getElem_toArray]
    rw [es]
    exact Array.mem_toList_iff.2 (Array.getElem_mem hs)

end Verity.QSort.ByKey
