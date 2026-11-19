module

public import Flock.Program
public import Mathlib

/-!
# The decoder's parsing and layout, as lemmas

What `Flock.Program.decode` parses and lays out, stated for its proofs: walks that lower a potential (`run_of_potential`),
sequences and their parts (`seq_get`, `bisect_spec`, `parsePartsIn_valid`), grids (`parseGrid_ok`) and a composite's
body laid out as the decoder lays it (`LaidOut`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.SecurityProofs.Walk

open Flock.Program (Prog Seq Part parsePartsIn Lane gridChecks parseGrid)

/-- **A walk that lowers a potential finishes.** If every step from a state satisfying `Inv` either ends with a result
satisfying `Good` or moves to a state satisfying `Inv` of lower potential, a walk from such a state with more fuel than
its potential ends with such a result. -/
theorem run_of_potential {α β : Type} {what : String} {step : α → Flock.V (β ⊕ α)} (Inv : α → Prop) (Φ : α → ℕ)
    (Good : β → Prop)
    (h : ∀ a, Inv a → (∃ b, step a = .ok (.inl b) ∧ Good b) ∨ ∃ a', step a = .ok (.inr a') ∧ Inv a' ∧ Φ a' < Φ a) :
    ∀ n a, Inv a → Φ a < n → ∃ b, Flock.Steps.run what step n a = .ok b ∧ Good b := by
  intro n
  induction n with
  | zero => intro a _ h0; omega
  | succ n ih =>
    intro a ha hlt
    rcases h a ha with ⟨b, hs, hb⟩ | ⟨a', hs, ha', hlt'⟩
    · refine ⟨b, ?_, hb⟩
      rw [Flock.Steps.run.eq_2, hs]
      rfl
    · obtain ⟨b, hr, hb⟩ := ih a' ha' (by omega)
      refine ⟨b, ?_, hb⟩
      rw [Flock.Steps.run.eq_2, hs]
      exact hr

/-- A fold pushing one entry per index, each computed from the entries before it. -/
theorem pushFold_spec {α : Type} [Inhabited α] (f : Array α → ℕ → α) :
    ∀ N, ((List.range N).foldl (fun a d => a.push (f a d)) #[]).size = N ∧
      ∀ d < N, ((List.range N).foldl (fun a d => a.push (f a d)) #[])[d]! =
        f (((List.range N).foldl (fun a d => a.push (f a d)) #[]).extract 0 d) d := by
  intro N
  induction N with
  | zero => exact ⟨rfl, fun d hd => absurd hd (Nat.not_lt_zero d)⟩
  | succ N ih =>
    obtain ⟨hs, hg⟩ := ih
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    set a := (List.range N).foldl (fun a d => a.push (f a d)) #[]
    refine ⟨by simp [hs], fun d hd => ?_⟩
    have hext : (a.push (f a N)).extract 0 d = a.extract 0 d := by
      apply Array.ext
      · simp; omega
      · intro i h1 h2
        simp only [Array.getElem_extract, Nat.zero_add]
        simp only [Array.size_extract, Array.size_push] at h1
        rw [Array.getElem_push_lt]
    rw [hext]
    rcases Nat.lt_or_ge d N with hdN | hdN
    · rw [← hg d hdN, getElem!_pos (a.push (f a N)) d (by simp; omega), getElem!_pos a d (by omega),
        Array.getElem_push_lt]
    · have hdN' : d = N := by omega
      subst hdN'
      have hfull : a.extract 0 d = a := by
        apply Array.ext
        · simp; omega
        · intro i h1 h2; simp
      rw [hfull, getElem!_pos (a.push (f a d)) d (by simp; omega), Array.getElem_push]
      simp [hs]

/-- Entry `i < d` of the first `d` entries is the array's. -/
theorem getElem!_extract {α : Type} [Inhabited α] (a : Array α) {i d : ℕ} (hi : i < d) (hd : d ≤ a.size) :
    (a.extract 0 d)[i]! = a[i]! := by
  rw [getElem!_pos (a.extract 0 d) i (by simp; omega), getElem!_pos a i (by omega), Array.getElem_extract]
  simp

theorem foldl_list_congr {β γ : Type} (l : List γ) (f g : β → γ → β) (b : β)
    (h : ∀ x ∈ l, ∀ s, f s x = g s x) : l.foldl f b = l.foldl g b := by
  induction l generalizing b with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldl_cons]
    rw [h x (by simp)]
    exact ih _ (fun y hy => h y (by simp [hy]))

/-- A fold over the nodes depends only on what it does at each node. -/
theorem foldl_nodes_congr {β : Type} (ns : Array Flock.Program.Node) (f g : β → Flock.Program.Node → β) (b : β)
    (h : ∀ x ∈ ns, ∀ s, f s x = g s x) : ns.foldl f b = ns.foldl g b := by
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  exact foldl_list_congr _ _ _ _ (fun x hx => h x (by simpa using hx))

/-- **The bisection over `starts`** (`Seq.get`'s, `Desc.nodeOf`'s): from `[lo, hi)` with `starts[lo] ≤ k` and
`k < starts[hi]` (or `hi = n`), as many rounds as the width needs end at one index `lo` with the same bounds. -/
theorem bisect_spec (starts : Array ℕ) (k n : ℕ) (f : ℕ → ℕ × ℕ → Id (ForInStep (ℕ × ℕ)))
    (hf : ∀ x lo hi, f x (lo, hi) = if hi - lo ≤ 1 then pure (.done (lo, hi))
      else if starts[(lo + hi) / 2]! ≤ k then pure (.yield ((lo + hi) / 2, hi)) else pure (.yield (lo, (lo + hi) / 2))) :
    ∀ (l : List ℕ) (lo hi : ℕ), lo < hi → hi ≤ n → starts[lo]! ≤ k → (hi < n → k < starts[hi]!) →
      hi - lo ≤ l.length + 1 →
      (forIn l (lo, hi) f : Id (ℕ × ℕ)).1 < (forIn l (lo, hi) f : Id (ℕ × ℕ)).2 ∧
      (forIn l (lo, hi) f : Id (ℕ × ℕ)).2 ≤ n ∧ starts[(forIn l (lo, hi) f : Id (ℕ × ℕ)).1]! ≤ k ∧
      ((forIn l (lo, hi) f : Id (ℕ × ℕ)).2 < n → k < starts[(forIn l (lo, hi) f : Id (ℕ × ℕ)).2]!) ∧
      (forIn l (lo, hi) f : Id (ℕ × ℕ)).2 - (forIn l (lo, hi) f : Id (ℕ × ℕ)).1 ≤ 1 := by
  intro l
  induction l with
  | nil =>
    intro lo hi h1 h2 h3 h4 h5
    simp only [List.forIn_nil, List.length_nil] at h5 ⊢
    exact ⟨h1, h2, h3, h4, by omega⟩
  | cons x xs ih =>
    intro lo hi h1 h2 h3 h4 h5
    simp only [List.forIn_cons, hf]
    by_cases hw : hi - lo ≤ 1
    · simp only [hw, ite_true]
      exact ⟨h1, h2, h3, h4, hw⟩
    · simp only [hw, ite_false]
      simp only [List.length_cons] at h5
      by_cases hm : starts[(lo + hi) / 2]! ≤ k
      · simp only [hm, ite_true]
        exact ih _ _ (by omega) h2 hm h4 (by omega)
      · simp only [hm, ite_false]
        exact ih _ _ (by omega) (by omega) h3 (fun _ => by omega) (by omega)

/-- `Seq.ofParts`'s starts: part `i` starts after the counts of the parts before it, and the count is all of them. -/
theorem ofParts_spec (ps : Array Part) :
    (Seq.ofParts ps).parts = ps ∧ (Seq.ofParts ps).starts.size = ps.size ∧
      (∀ i < ps.size, (Seq.ofParts ps).starts[i]! = ((ps.toList.take i).map Part.count).sum) ∧
      (Seq.ofParts ps).count = (ps.toList.map Part.count).sum := by
  have key : ∀ (l : List Part) (acc : Array ℕ × ℕ),
      (l.foldl (fun (acc : Array ℕ × ℕ) q => (acc.1.push acc.2, acc.2 + q.count)) acc).1.size = acc.1.size + l.length ∧
      (∀ i < acc.1.size, (l.foldl (fun (acc : Array ℕ × ℕ) q => (acc.1.push acc.2, acc.2 + q.count)) acc).1[i]! =
        acc.1[i]!) ∧
      (∀ i < l.length, (l.foldl (fun (acc : Array ℕ × ℕ) q => (acc.1.push acc.2, acc.2 + q.count)) acc).1[acc.1.size + i]!
        = acc.2 + ((l.take i).map Part.count).sum) ∧
      (l.foldl (fun (acc : Array ℕ × ℕ) q => (acc.1.push acc.2, acc.2 + q.count)) acc).2 = acc.2 + (l.map Part.count).sum := by
    intro l
    induction l with
    | nil => intro acc; simp
    | cons q qs ih =>
      intro acc
      obtain ⟨h1, h2, h3, h4⟩ := ih (acc.1.push acc.2, acc.2 + q.count)
      simp only [List.foldl_cons, Array.size_push, List.length_cons] at h1 h2 h3 h4 ⊢
      refine ⟨by omega, fun i hi => ?_, fun i hi => ?_, by rw [h4]; simp; omega⟩
      · rw [h2 i (by omega), getElem!_pos (acc.1.push acc.2) i (by simp; omega), getElem!_pos acc.1 i hi,
          Array.getElem_push_lt]
      · rcases i with _ | i
        · rw [← Nat.add_zero (acc.1.size), ← Nat.add_zero (acc.1.size + 0)] at *
          have := h2 acc.1.size (by simp)
          simp only [Nat.add_zero] at this ⊢
          rw [this, getElem!_pos (acc.1.push acc.2) acc.1.size (by simp)]
          simp
        · have := h3 i (by omega)
          rw [show acc.1.size + (i + 1) = acc.1.size + 1 + i by omega, this]
          simp [List.take_succ_cons]
          omega
  obtain ⟨h1, -, h3, h4⟩ := key ps.toList (#[], 0)
  refine ⟨rfl, ?_, fun i hi => ?_, ?_⟩
  · simp only [Seq.ofParts, ← Array.foldl_toList]; simpa using h1
  · simp only [Seq.ofParts, ← Array.foldl_toList]
    have := h3 i (by simpa using hi)
    simpa using this
  · simp only [Seq.ofParts, ← Array.foldl_toList]; simpa using h4

theorem sum_take_succ_map (ps : Array Part) {i : ℕ} (hi : i < ps.size) :
    ((ps.toList.take (i + 1)).map Part.count).sum = ((ps.toList.take i).map Part.count).sum + ps[i]!.count := by
  rw [List.take_add_one, List.map_append, List.sum_append, getElem!_pos ps i hi]
  simp [hi]

/-- **Reference `k` of a sequence** is part `i`'s element `k - starts[i]`, for the part `i` holding it. -/
theorem seq_get (ps : Array Part) {k : ℕ} (hk : k < (Seq.ofParts ps).count) :
    ∃ i < ps.size, (Seq.ofParts ps).starts[i]! ≤ k ∧ k < (Seq.ofParts ps).starts[i]! + ps[i]!.count ∧
      (Seq.ofParts ps).get k = ps[i]!.get (k - (Seq.ofParts ps).starts[i]!) := by
  obtain ⟨hp, hsz, hst, hcnt⟩ := ofParts_spec ps
  have hst' : ∀ i < ps.size, (Seq.ofParts ps).starts[i]! + ps[i]!.count =
      ((ps.toList.take (i + 1)).map Part.count).sum := fun i hi => by rw [hst i hi, sum_take_succ_map ps hi]
  have hpos : 0 < ps.size := by
    rcases Nat.eq_zero_or_pos ps.size with h | h
    · rw [hcnt] at hk
      have : ps.toList = [] := by simpa [← Array.length_toList] using h
      simp [this] at hk
    · exact h
  have h0 : (Seq.ofParts ps).starts[0]! = 0 := by rw [hst 0 hpos]; simp
  have hlast : (Seq.ofParts ps).count = (Seq.ofParts ps).starts[ps.size - 1]! + ps[ps.size - 1]!.count := by
    rw [hst' _ (by omega), hcnt, show ps.size - 1 + 1 = ps.toList.length by simp; omega, List.take_length]
  by_cases h1 : ps.size = 1
  · refine ⟨0, hpos, by rw [h0]; omega, ?_, ?_⟩
    · rw [h0]; rw [hlast, h1] at hk; simpa [h0] using hk
    · unfold Flock.Program.Seq.get
      rw [if_pos (by simp [hp, h1])]
      simp [h0, hp]
  · let f : ℕ → ℕ × ℕ → Id (ForInStep (ℕ × ℕ)) := fun _ (st : ℕ × ℕ) =>
      have lo := st.1; have hi := st.2
      if hi - lo ≤ 1 then pure (ForInStep.done (lo, hi))
      else
        have mid := (lo + hi) / 2
        if (Seq.ofParts ps).starts[mid]! ≤ k then pure (ForInStep.yield (mid, hi)) else pure (ForInStep.yield (lo, mid))
    obtain ⟨b1, b2, b3, b4, b5⟩ := bisect_spec (Seq.ofParts ps).starts k ps.size f (fun _ _ _ => rfl)
      (List.range' 0 ps.size 1) 0 ps.size hpos (le_refl _) (by rw [h0]; omega) (fun h => absurd h (lt_irrefl _))
      (by simp)
    set r := (forIn (List.range' 0 ps.size 1) (0, ps.size) f : Id (ℕ × ℕ))
    have hr2 : r.2 = r.1 + 1 := by omega
    refine ⟨r.1, by omega, b3, ?_, ?_⟩
    · rw [hst' r.1 (by omega)]
      rcases Nat.lt_or_ge r.2 ps.size with hlt | hge
      · have := b4 hlt
        rw [hst r.2 hlt, hr2] at this
        exact this
      · have hre : r.1 = ps.size - 1 := by omega
        rw [← hst' r.1 (by omega), hre, ← hlast]
        exact hk
    · unfold Flock.Program.Seq.get
      rw [if_neg (by simp [hp, h1])]
      have hfor : ∀ (g : ℕ → ℕ × ℕ → Id (ForInStep (ℕ × ℕ))) (init : ℕ × ℕ),
          (forIn [:ps.size] init g : Id (ℕ × ℕ)) = forIn (List.range' 0 ps.size) init g := by
        intro g init
        rw [Std.Legacy.Range.forIn_eq_forIn_range']
        simp [Std.Legacy.Range.size]
      simp only [hp, hfor]
      rfl

/-- An affine map on `[0, Z]` lies between its values at the ends. -/
theorem affine_between (A B : ℤ) {z Z : ℕ} (hz : z ≤ Z) :
    min A (A + (Z : ℤ) * B) ≤ A + (z : ℤ) * B ∧ A + (z : ℤ) * B ≤ max A (A + (Z : ℤ) * B) := by
  have hzZ : (z : ℤ) ≤ Z := by exact_mod_cast hz
  have hz0 : (0 : ℤ) ≤ z := by positivity
  rcases le_total 0 B with hB | hB
  · have h1 : (z : ℤ) * B ≤ Z * B := mul_le_mul_of_nonneg_right hzZ hB
    have h2 : 0 ≤ (z : ℤ) * B := mul_nonneg hz0 hB
    constructor
    · exact le_trans (min_le_left _ _) (by linarith)
    · exact le_trans (by linarith) (le_max_right _ _)
  · have h1 : (Z : ℤ) * B ≤ z * B := mul_le_mul_of_nonpos_right hzZ hB
    have h2 : (z : ℤ) * B ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hz0 hB
    constructor
    · exact le_trans (min_le_right _ _) (by linarith)
    · exact le_trans (by linarith) (le_max_left _ _)

theorem foldl_min_le : ∀ (l : List ℤ) (init x : ℤ), (x ∈ l ∨ x = init) → l.foldl min init ≤ x
  | [], init, x, h => by simp at h; subst h; simp
  | y :: ys, init, x, h => by
    simp only [List.foldl_cons]
    have hm : min init y ≤ x ∨ x ∈ ys := by
      rcases h with h | h
      · rcases List.mem_cons.1 h with h | h
        · left; rw [h]; exact min_le_right _ _
        · right; exact h
      · left; rw [h]; exact min_le_left _ _
    rcases hm with hm | hm
    · exact le_trans (foldl_min_le ys (min init y) (min init y) (Or.inr rfl)) hm
    · exact foldl_min_le ys _ x (Or.inl hm)

theorem le_foldl_max : ∀ (l : List ℤ) (init x : ℤ), (x ∈ l ∨ x = init) → x ≤ l.foldl max init
  | [], init, x, h => by simp at h; subst h; simp
  | y :: ys, init, x, h => by
    simp only [List.foldl_cons]
    have hm : x ≤ max init y ∨ x ∈ ys := by
      rcases h with h | h
      · rcases List.mem_cons.1 h with h | h
        · left; rw [h]; exact le_max_right _ _
        · right; exact h
      · left; rw [h]; exact le_max_left _ _
    rcases hm with hm | hm
    · exact le_trans hm (le_foldl_max ys (max init y) (max init y) (Or.inr rfl))
    · exact le_foldl_max ys _ x (Or.inl hm)

/-- **A strided part's elements lie between its corners' values** (`Flock.Program.parsePartsIn` checks only those):
element `t < c` is `b + (t / inner) · os + (t % inner) · is`, and some corner of the pattern is at most it, some at
least. -/
theorem strided_corners (b os is : ℤ) {inner c t : ℕ} (hin : 0 < inner) (ht : t < c) :
    let q := (c - 1) / inner
    let last := (c - 1) % inner
    let corners : List (ℕ × ℕ) := [(0, 0), (0, min inner c - 1), (q, 0), (q, last)] ++
      (if q > 0 then [(q - 1, inner - 1)] else [])
    (∃ p ∈ corners, b + (p.1 : ℤ) * os + (p.2 : ℤ) * is ≤
        b + ((t / inner : ℕ) : ℤ) * os + ((t % inner : ℕ) : ℤ) * is) ∧
      ∃ p ∈ corners, b + ((t / inner : ℕ) : ℤ) * os + ((t % inner : ℕ) : ℤ) * is ≤
        b + (p.1 : ℤ) * os + (p.2 : ℤ) * is := by
  intro q last corners
  have hq : q = (c - 1) / inner := rfl
  have hl : last = (c - 1) % inner := rfl
  clear_value q last
  let vals := corners.map fun (x, y) => b + (x : ℤ) * os + (y : ℤ) * is
  generalize hx : t / inner = x
  generalize hyy : t % inner = y
  have hxq : x ≤ q := by rw [← hx, hq]; exact Nat.div_le_div_right (by omega)
  have hy : y < inner := by rw [← hyy]; exact Nat.mod_lt _ hin
  have hdecomp : t = x * inner + y := by
    rw [← hx, ← hyy, Nat.mul_comm]; exact (Nat.div_add_mod t inner).symm
  have hc1 : c - 1 = q * inner + last := by
    rw [hq, hl, Nat.mul_comm]; exact (Nat.div_add_mod (c - 1) inner).symm
  have hlast : last < inner := by rw [hl]; exact Nat.mod_lt _ hin
  have hyq : x = q → y ≤ last := by
    intro hxq'; subst hxq'; omega
  -- v lies between v(x, 0) and v(x, ymax), and each of those between two corners
  let v := fun (x y : ℕ) => b + (x : ℤ) * os + (y : ℤ) * is
  have hmem : ∀ p ∈ corners, v p.1 p.2 ∈ vals := fun p hp => List.mem_map.2 ⟨p, hp, rfl⟩
  have key : (∃ u ∈ vals, u ≤ v x y) ∧ ∃ u ∈ vals, v x y ≤ u := by
    have hy1 := affine_between (b + (x : ℤ) * os) is (show y ≤ (if x = q then last else inner - 1) by
      split_ifs with h <;> [exact hyq h; omega])
    have hx1 := affine_between b os hxq
    have hrow : ∀ Y, (Y = last ∧ x = q) ∨ (Y = inner - 1 ∧ x < q) →
        (∃ u ∈ vals, u ≤ v x Y) ∧ ∃ u ∈ vals, v x Y ≤ u := by
      rintro Y (⟨hY, hx⟩ | ⟨hY, hxlt⟩) <;> rw [hY]
      · rw [hx]
        exact ⟨⟨_, hmem (q, last) (by simp [corners]), le_refl _⟩, _, hmem (q, last) (by simp [corners]), le_refl _⟩
      · have hq0 : q > 0 := by omega
        have hcq : min inner c - 1 = inner - 1 := by
          have : inner ≤ c - 1 := by
            have : 1 * inner ≤ q * inner := Nat.mul_le_mul_right _ hq0
            omega
          rw [Nat.min_eq_left (by omega)]
        have hm1 : v 0 (inner - 1) ∈ vals := by
          have := hmem (0, min inner c - 1) (by simp [corners]); rw [hcq] at this; exact this
        have hm2 : v (q - 1) (inner - 1) ∈ vals := hmem (q - 1, inner - 1) (by simp [corners, hq0])
        have hb := affine_between (b + ((inner - 1 : ℕ) : ℤ) * is) os (show x ≤ q - 1 by omega)
        have e : ∀ z : ℕ, v z (inner - 1) = b + ((inner - 1 : ℕ) : ℤ) * is + (z : ℤ) * os := fun z => by
          simp only [v]; ring
        rw [e x]
        refine ⟨⟨min (v 0 (inner - 1)) (v (q - 1) (inner - 1)), ?_, ?_⟩, max (v 0 (inner - 1)) (v (q - 1) (inner - 1)), ?_, ?_⟩
        · rcases min_choice (v 0 (inner - 1)) (v (q - 1) (inner - 1)) with h | h <;> rw [h] <;> assumption
        · rw [e 0, e (q - 1)]; simpa using hb.1
        · rcases max_choice (v 0 (inner - 1)) (v (q - 1) (inner - 1)) with h | h <;> rw [h] <;> assumption
        · rw [e 0, e (q - 1)]; simpa using hb.2
    have hcol : (∃ u ∈ vals, u ≤ v x 0) ∧ ∃ u ∈ vals, v x 0 ≤ u := by
      have hm1 : v 0 0 ∈ vals := hmem (0, 0) (by simp [corners])
      have hm2 : v q 0 ∈ vals := hmem (q, 0) (by simp [corners])
      refine ⟨⟨min (v 0 0) (v q 0), ?_, ?_⟩, max (v 0 0) (v q 0), ?_, ?_⟩
      · rcases min_choice (v 0 0) (v q 0) with h | h <;> rw [h] <;> assumption
      · simp only [v]; simpa using hx1.1
      · rcases max_choice (v 0 0) (v q 0) with h | h <;> rw [h] <;> assumption
      · simp only [v]; simpa using hx1.2
    obtain ⟨⟨u1, hu1, hu1'⟩, ⟨u2, hu2, hu2'⟩⟩ := hcol
    obtain ⟨⟨u3, hu3, hu3'⟩, ⟨u4, hu4, hu4'⟩⟩ := hrow (if x = q then last else inner - 1) (by
      by_cases h : x = q
      · exact Or.inl ⟨by simp [h], h⟩
      · exact Or.inr ⟨by simp [h], by omega⟩)
    have e0 : v x 0 = b + (x : ℤ) * os + ((0 : ℕ) : ℤ) * is := rfl
    have eY : ∀ Y : ℕ, v x Y = b + (x : ℤ) * os + (Y : ℤ) * is := fun _ => rfl
    constructor
    · rcases min_choice (b + (x : ℤ) * os) (b + (x : ℤ) * os + ((if x = q then last else inner - 1 : ℕ) : ℤ) * is)
        with h | h
      · exact ⟨u1, hu1, le_trans hu1' (by simp only [v]; simp; rw [h] at hy1; simpa using hy1.1)⟩
      · exact ⟨u3, hu3, le_trans hu3' (by rw [eY]; rw [h] at hy1; exact hy1.1)⟩
    · rcases max_choice (b + (x : ℤ) * os) (b + (x : ℤ) * os + ((if x = q then last else inner - 1 : ℕ) : ℤ) * is)
        with h | h
      · exact ⟨u2, hu2, le_trans (by simp only [v]; rw [h] at hy1; simpa using hy1.2) hu2'⟩
      · exact ⟨u4, hu4, le_trans (by rw [eY]; rw [h] at hy1; exact hy1.2) hu4'⟩
  obtain ⟨⟨u, hu, hu'⟩, ⟨u', hu2, hu2'⟩⟩ := key
  obtain ⟨⟨x1, y1⟩, hp1, rfl⟩ := List.mem_map.1 hu
  obtain ⟨⟨x2, y2⟩, hp2, rfl⟩ := List.mem_map.1 hu2
  exact ⟨⟨(x1, y1), hp1, hu'⟩, ⟨(x2, y2), hp2, hu2'⟩⟩

/-- Every reference of part `q` names a leaf of its collection (`leaves p i`, as `Flock.Program.parsePartsIn` takes
it). -/
def PartValid (leaves : Bool → ℕ → Option ℕ) (q : Part) : Prop :=
  ∀ k < q.count, ∃ n, leaves (q.get k).p (q.get k).i = some n ∧ (q.get k).j < n

theorem list_mapM_ok {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α) (ys : List β), l.mapM f = .ok ys → ∀ y ∈ ys, ∃ x ∈ l, f x = .ok y
  | [], ys, h, y, hy => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h; subst h; simp at hy
  | a :: l, ys, h, y, hy => by
    rw [List.mapM_cons] at h
    rcases hfa : f a with e | c <;> simp only [hfa, bind, Except.bind] at h
    · exact absurd h (by intro h; injection h)
    rcases hl : l.mapM f with e | cs <;> simp only [hl, pure, Except.pure] at h
    · exact absurd h (by intro h; injection h)
    injection h with h; subst h
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨a, List.mem_cons_self, hfa⟩
    · obtain ⟨x, hx, hfx⟩ := list_mapM_ok f l cs hl y hy
      exact ⟨x, List.mem_cons_of_mem _ hx, hfx⟩

theorem array_mapM_ok {α β ε : Type} (f : α → Except ε β) (xs : Array α) (ys : Array β)
    (h : xs.mapM f = .ok ys) : ∀ y ∈ ys, ∃ x ∈ xs, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  rcases hl : xs.toList.mapM f with e | l <;> simp only [hl, Functor.map, Except.map] at h
  · exact absurd h (by intro h; injection h)
  injection h with h; subst h
  intro y hy
  obtain ⟨x, hx, hfx⟩ := list_mapM_ok f _ l hl y (by simpa using hy)
  exact ⟨x, by simpa using hx, hfx⟩

theorem mem_foldl_append {α : Type} : ∀ (l : List (Array α)) (acc : Array α) (q : α),
    q ∈ l.foldl (· ++ ·) acc → q ∈ acc ∨ ∃ y ∈ l, q ∈ y
  | [], _, _, h => Or.inl h
  | y :: l, acc, q, h => by
    rcases mem_foldl_append l (acc ++ y) q h with h | ⟨z, hz, hq⟩
    · rcases Array.mem_append.1 h with h | h
      · exact Or.inl h
      · exact Or.inr ⟨y, List.mem_cons_self, h⟩
    · exact Or.inr ⟨z, List.mem_cons_of_mem _ hz, hq⟩

theorem except_bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {y : β}
    (h : x >>= f = .ok y) : ∃ a, x = .ok a ∧ f a = .ok y := by
  rcases x with e | a
  · exact absurd h (by intro h; injection h)
  · exact ⟨a, rfl, h⟩

private def loS : List (Nat × Int) → Int
  | [] => 0
  | (r, d) :: rest => ((r : Int) - 1) * min d 0 + loS rest

private def hiS : List (Nat × Int) → Int
  | [] => 0
  | (r, d) :: rest => ((r : Int) - 1) * max d 0 + hiS rest

private def prodR : List (Nat × Int) → Nat
  | [] => 1
  | (r, _) :: rest => r * prodR rest

private theorem bounds_foldl (xs : List (Nat × Int)) (a b : Int) :
    xs.foldl (fun (lo, hi) (r, d) => (lo + ((r : Int) - 1) * min d 0, hi + ((r : Int) - 1) * max d 0)) (a, b) =
      (a + loS xs, b + hiS xs) := by
  induction xs generalizing a b with
  | nil => simp [loS, hiS]
  | cons x xs ih =>
    obtain ⟨r, d⟩ := x
    simp only [List.foldl_cons, ih, loS, hiS, Prod.mk.injEq]
    constructor <;> omega

private theorem periods_foldl (xs : List (Nat × Int)) (c : Nat) :
    xs.foldl (fun a (r, _) => a * r) c = c * prodR xs := by
  induction xs generalizing c with
  | nil => simp [prodR]
  | cons x xs ih =>
    obtain ⟨r, d⟩ := x
    simp only [List.foldl_cons, ih, prodR, Nat.mul_assoc]

private theorem one_level {t r : Nat} (d : Int) (ht : t < r) :
    ((r : Int) - 1) * min d 0 ≤ (t : Int) * d ∧ (t : Int) * d ≤ ((r : Int) - 1) * max d 0 := by
  have h0 : (0 : Int) ≤ t := Int.natCast_nonneg t
  have h1 : (t : Int) ≤ (r : Int) - 1 := by omega
  by_cases hd : 0 ≤ d
  · rw [Int.min_eq_right hd, Int.max_eq_left hd, Int.mul_zero]
    exact ⟨Int.mul_nonneg h0 hd, Int.mul_le_mul_of_nonneg_right h1 hd⟩
  · have hd' : d ≤ 0 := by omega
    rw [Int.min_eq_left hd', Int.max_eq_right hd', Int.mul_zero]
    exact ⟨Int.mul_le_mul_of_nonpos_right h1 hd', Int.mul_nonpos_of_nonneg_of_nonpos h0 hd'⟩

private theorem go_mem : ∀ (lv : List (Nat × Int)) (t : Nat), t < prodR lv →
    loS lv ≤ Lane.leafAt.go lv t ∧ Lane.leafAt.go lv t ≤ hiS lv
  | [], t, _ => by simp [loS, hiS, Lane.leafAt.go]
  | [(r, d)], t, ht => by
    simp only [prodR, Nat.mul_one] at ht
    have := one_level d ht
    simp only [loS, hiS, Lane.leafAt.go, Int.add_zero]
    exact this
  | (r, d) :: (r2, d2) :: rest, t, ht => by
    simp only [prodR] at ht
    have hr : 0 < r := Nat.pos_of_ne_zero (by rintro rfl; simp at ht)
    have h1 := one_level d (Nat.mod_lt t hr)
    have h2 := go_mem ((r2, d2) :: rest) (t / r) (by
      simp only [prodR]; exact Nat.div_lt_of_lt_mul ht)
    have e : Lane.leafAt.go ((r, d) :: (r2, d2) :: rest) t
        = ((t % r : Nat) : Int) * d + Lane.leafAt.go ((r2, d2) :: rest) (t / r) := rfl
    rw [e]
    simp only [loS, hiS] at h2 ⊢
    exact ⟨Int.add_le_add h1.1 h2.1, Int.add_le_add h1.2 h2.2⟩

theorem leafAt_mem (l : Lane) {run t o : Nat} (ho : o < run) (ht : t < l.periods) :
    (l.bounds run).1 ≤ l.leafAt t o ∧ l.leafAt t o ≤ (l.bounds run).2 := by
  have hp : l.periods = prodR l.lv := by
    simp only [Lane.periods, periods_foldl, Nat.one_mul]
  rw [hp] at ht
  have hb : l.bounds run = (l.b + loS ((run, l.e) :: l.lv), l.b + hiS ((run, l.e) :: l.lv)) := by
    simp only [Lane.bounds]; exact bounds_foldl _ _ _
  rw [hb]
  have h1 := one_level l.e ho
  have h2 := go_mem l.lv t ht
  show l.b + (((run : Int) - 1) * min l.e 0 + loS l.lv) ≤ l.b + (o : Int) * l.e + Lane.leafAt.go l.lv t ∧
    l.b + (o : Int) * l.e + Lane.leafAt.go l.lv t ≤ l.b + (((run : Int) - 1) * max l.e 0 + hiS l.lv)
  constructor
  · rw [Int.add_assoc]; exact Int.add_le_add_left (Int.add_le_add h1.1 h2.1) _
  · rw [Int.add_assoc]; exact Int.add_le_add_left (Int.add_le_add h1.2 h2.2) _

private theorem one_top {r : Nat} (d : Int) (hr : 0 < r) : ∃ t < r, (t : Int) * d = ((r : Int) - 1) * max d 0 := by
  by_cases hd : 0 ≤ d
  · refine ⟨r - 1, by omega, ?_⟩
    rw [Int.max_eq_left hd, Nat.cast_sub hr]
    rfl
  · refine ⟨0, hr, ?_⟩
    rw [Int.max_eq_right (by omega)]
    simp

private theorem go_top : ∀ (lv : List (Nat × Int)), 0 < prodR lv → ∃ t < prodR lv, Lane.leafAt.go lv t = hiS lv
  | [], _ => ⟨0, by simp [prodR], by simp [Lane.leafAt.go, hiS]⟩
  | [(r, d)], hp => by
    simp only [prodR, Nat.mul_one] at hp ⊢
    obtain ⟨t, ht, he⟩ := one_top d hp
    exact ⟨t, ht, by simp only [Lane.leafAt.go, hiS, Int.add_zero]; exact he⟩
  | (r, d) :: (r2, d2) :: rest, hp => by
    simp only [prodR] at hp ⊢
    have hr : 0 < r := Nat.pos_of_mul_pos_right hp
    have hq : 0 < r2 * prodR rest := Nat.pos_of_mul_pos_left hp
    obtain ⟨t1, ht1, he1⟩ := one_top d hr
    obtain ⟨t2, ht2, he2⟩ := go_top ((r2, d2) :: rest) (by simpa only [prodR] using hq)
    simp only [prodR] at ht2
    refine ⟨t1 + r * t2, ?_, ?_⟩
    · calc t1 + r * t2 < r + r * t2 := by omega
        _ = r * (t2 + 1) := by rw [Nat.mul_add, Nat.mul_one, Nat.add_comm]
        _ ≤ r * (r2 * prodR rest) := Nat.mul_le_mul_left _ ht2
    · have e1 : (t1 + r * t2) % r = t1 := by rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ht1]
      have e2 : (t1 + r * t2) / r = t2 := by
        rw [Nat.add_mul_div_left _ _ hr, Nat.div_eq_of_lt ht1, Nat.zero_add]
      have e : Lane.leafAt.go ((r, d) :: (r2, d2) :: rest) (t1 + r * t2)
          = (((t1 + r * t2) % r : Nat) : Int) * d + Lane.leafAt.go ((r2, d2) :: rest) ((t1 + r * t2) / r) := rfl
      rw [e, e1, e2, he1, he2]
      rfl

/-- **A lane reaches its greatest bound**: at some period and offset its leaf is `(l.bounds run).2`. -/
theorem leafAt_top (l : Lane) {run : Nat} (hrun : 0 < run) (hp : 0 < l.periods) :
    ∃ t < l.periods, ∃ o < run, l.leafAt t o = (l.bounds run).2 := by
  have hpr : l.periods = prodR l.lv := by
    simp only [Lane.periods, periods_foldl, Nat.one_mul]
  rw [hpr] at hp ⊢
  obtain ⟨t, ht, he⟩ := go_top l.lv hp
  obtain ⟨o, ho, heo⟩ := one_top l.e hrun
  refine ⟨t, ht, o, ho, ?_⟩
  have hb : l.bounds run = (l.b + loS ((run, l.e) :: l.lv), l.b + hiS ((run, l.e) :: l.lv)) := by
    simp only [Lane.bounds]; exact bounds_foldl _ _ _
  rw [hb]
  show l.b + (o : Int) * l.e + Lane.leafAt.go l.lv t = l.b + (((run : Int) - 1) * max l.e 0 + hiS l.lv)
  rw [he, heo, Int.add_assoc]


/-- What a grid the decoder accepts holds: one part, lanes of one period count, each inside its collection. -/
theorem gridChecks_ok {j g : Lean.Json} {run : Nat} {lanes : Array Lane} {leaves : Bool → Nat → Option Nat}
    {ps : Array Part} (h : gridChecks j g run lanes leaves = .ok ps) :
    ps = #[.grid run lanes] ∧ (∀ l ∈ lanes, l.periods = lanes[0]!.periods) ∧ ∀ l ∈ lanes, l.fits leaves run = true := by
  unfold gridChecks at h
  split at h
  · exact absurd h (by intro h; injection h)
  rename_i h1
  split at h
  · exact absurd h (by intro h; injection h)
  split at h
  · exact absurd h (by intro h; injection h)
  rename_i h3
  injection h with h
  refine ⟨h.symm, fun l hl => ?_, fun l hl => ?_⟩
  · simp only [Array.any_eq_true, bne_iff_ne, ne_eq, not_exists, Decidable.not_not] at h1
    obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hl
    exact h1 i hi
  · have h3' : lanes.all (fun x => x.fits leaves run) = true := by
      cases hc : lanes.all (fun x => x.fits leaves run) <;> simp_all
    rw [Array.all_eq_true] at h3'
    obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hl
    exact h3' i hi

theorem parseGrid_ok {j g : Lean.Json} {leaves : Bool → Nat → Option Nat} {ps : Array Part}
    (h : parseGrid j g leaves = .ok ps) : ∃ run lanes, ps = #[.grid run lanes] ∧
      (∀ l ∈ lanes, l.periods = lanes[0]!.periods) ∧ ∀ l ∈ lanes, l.fits leaves run = true := by
  rcases g with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
  all_goals try (delta parseGrid at h; injection h; done)
  rcases l with _ | ⟨run, _ | ⟨ls, _ | ⟨y, l⟩⟩⟩
  all_goals try (delta parseGrid at h; injection h; done)
  rcases ls with _ | _ | _ | _ | ⟨⟨ls⟩⟩ | _
  all_goals try (delta parseGrid at h; injection h; done)
  delta parseGrid at h
  conv at h => lhs; whnf
  dsimp only [Array.getLit, List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ] at h
  split at h
  · rename_i n hn
    split at h
    · exact absurd h (by intro h; injection h)
    obtain ⟨lanes, -, h⟩ := except_bind_ok h
    exact ⟨n, lanes, gridChecks_ok h⟩
  · exact absurd h (by intro h; injection h)

theorem grid_valid {run : Nat} {lanes : Array Lane} {leaves : Bool → Nat → Option Nat}
    (hp : ∀ l ∈ lanes, l.periods = lanes[0]!.periods) (hf : ∀ l ∈ lanes, l.fits leaves run = true) :
    PartValid leaves (.grid run lanes) := by
  intro k hk
  simp only [Part.count] at hk
  have hg : 0 < lanes.size := Nat.pos_of_ne_zero (by intro h0; simp [h0] at hk)
  have hrun : 0 < run := Nat.pos_of_ne_zero (by intro h0; simp [h0] at hk)
  have hgr : 0 < lanes.size * run := Nat.mul_pos hg hrun
  have hidx : k % (lanes.size * run) / run < lanes.size :=
    Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm run]; exact Nat.mod_lt _ hgr)
  have h0 : lanes[0]? = some lanes[0] := Array.getElem?_eq_getElem hg
  rw [h0] at hk
  simp only [Option.map_some, Option.getD_some] at hk
  generalize hl : lanes[k % (lanes.size * run) / run]! = l
  have hmem : l ∈ lanes := by
    rw [← hl, getElem!_pos lanes _ hidx]; exact Array.getElem_mem hidx
  have hper : l.periods = lanes[0].periods := by
    rw [hp l hmem, getElem!_pos lanes 0 hg]
  have ht : k / (lanes.size * run) < l.periods := by
    rw [hper]; exact Nat.div_lt_of_lt_mul hk
  have hb := leafAt_mem l (Nat.mod_lt k hrun) ht
  have hfl := hf l hmem
  show ∃ n, leaves (lanes[k % (lanes.size * run) / run]!).p (lanes[k % (lanes.size * run) / run]!).i = some n ∧
    ((lanes[k % (lanes.size * run) / run]!).leafAt (k / (lanes.size * run)) (k % run)).toNat < n
  rw [hl]
  unfold Lane.fits at hfl
  split at hfl
  · rename_i n hn
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hfl
    exact ⟨n, hn, by omega⟩
  · cases hfl

/-- One unfolding of `parsePartsIn`, with the nested call `b.1` assumed to return valid parts. Lean generates no
equation lemmas for `parsePartsIn` (its `.arr #[…]` patterns), so this unfolds by `delta` and reduces each array
pattern by `whnf` once its list has been cased to a literal. -/
theorem parsePartsIn_step (leaves : Bool → ℕ → Option ℕ) (args : ℕ → ℕ → Option Seq)
    (hargs : ∀ k x s, args k x = some s → ∀ q ∈ s.parts, PartValid leaves q) (n : ℕ)
    (b : Nat.below (motive := fun _ => Lean.Json → Flock.V (Array Part)) (n + 1))
    (hb : ∀ x ps, b.1 x = .ok ps → ∀ q ∈ ps, PartValid leaves q) (j : Lean.Json) (ps : Array Part)
    (h : parsePartsIn._f leaves args (n + 1) b j = .ok ps) : ∀ q ∈ ps, PartValid leaves q := by
  delta parsePartsIn._f at h
  dsimp only at h
  cases j with
  | null => exact nomatch h
  | bool _ => exact nomatch h
  | num _ => exact nomatch h
  | str _ => exact nomatch h
  | arr _ => exact nomatch h
  | obj kvs =>
    dsimp only at h
    split at h
    · -- alias
      rename_i a _
      rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
      all_goals try (injection h; done)
      rcases l with _ | ⟨k, _ | ⟨x, _ | ⟨y, l⟩⟩⟩
      all_goals try (injection h; done)
      conv at h => lhs; whnf
      split at h
      all_goals try (injection h; done)
      split at h
      all_goals try (injection h; done)
      rename_i s hs
      injection h with h
      subst h
      exact hargs _ _ s hs
    · split at h
      · rename_i c _
        rcases c with _ | _ | _ | _ | cs | _
        all_goals try (injection h; done)
        dsimp only at h
        rcases hm : cs.mapM (fun x => b.1 x) with e | r <;> simp only [hm, bind, Except.bind] at h
        · exact absurd h (by intro h; injection h)
        injection h with h; subst h
        intro q hq
        rw [← Array.foldl_toList] at hq
        rcases mem_foldl_append r.toList #[] q hq with hq | ⟨y, hy, hq⟩
        · simp at hq
        · obtain ⟨x, -, hx⟩ := array_mapM_ok _ cs r hm y (by simpa using hy)
          exact hb x y hx q hq
      · split at h
        · rename_i a _
          rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
          all_goals try (injection h; done)
          rcases l with _ | ⟨s0, _ | ⟨i0, _ | ⟨b0, _ | ⟨c0, _ | ⟨y, l⟩⟩⟩⟩⟩
          all_goals try (injection h; done)
          conv at h => lhs; whnf
          split at h
          all_goals try (injection h; done)
          rename_i bb cc _ _
          split at h
          · injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            intro k hk
            simp [Part.count] at hk
          · split at h
            all_goals try (injection h; done)
            rename_i p i hw
            injection h with h; subst h
            split at hw
            all_goals try (injection hw; done)
            split at hw
            all_goals try (injection hw; done)
            rename_i nn hl
            split at hw
            all_goals try (injection hw; done)
            rename_i hc
            injection hw with hw
            injection hw with hp hi
            subst hp hi
            simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            intro k hk
            simp only [Part.count] at hk
            refine ⟨nn, hl, ?_⟩
            show cc.toNat + k < nn
            omega
        · split at h
          · rename_i a _
            rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
            all_goals try (injection h; done)
            rcases l with _ | ⟨s0, _ | ⟨i0, _ | ⟨b0, _ | ⟨n0, _ | ⟨o0, _ | ⟨j0, _ | ⟨c0, _ | ⟨y, l⟩⟩⟩⟩⟩⟩⟩⟩
            all_goals try (injection h; done)
            conv at h => lhs; whnf
            split at h
            all_goals try (injection h; done)
            rename_i _ cc inn ost ist c _
            split at h
            all_goals try (injection h; done)
            rename_i hinn
            try dsimp only at h
            split at h
            · injection h with h; subst h
              intro q hq
              simp only [Array.mem_singleton] at hq
              subst hq
              intro k hk
              simp [Part.count] at hk
            try dsimp only at h
            split at h
            all_goals try (injection h; done)
            rename_i p i hw
            injection h with h; subst h
            intro q hq
            simp only [Array.mem_singleton] at hq
            subst hq
            intro k hk
            simp only [Part.count] at hk
            have hin : 0 < inn := by simp at hinn; omega
            obtain ⟨⟨⟨x1, y1⟩, hp1, h1⟩, ⟨⟨x2, y2⟩, hp2, h2⟩⟩ := strided_corners cc ost ist hin hk
            split at hw
            all_goals try (injection hw; done)
            split at hw
            all_goals try (injection hw; done)
            rename_i nn hl
            split at hw
            all_goals rename_i hq0
            · rw [if_pos hq0] at hp1 hp2
              split at hw
              all_goals try (injection hw; done)
              rename_i hc
              injection hw with hw
              injection hw with hp hi
              subst hp hi
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
              have e1 := le_trans hc.1 (foldl_min_le _ _ _ (Or.inl (List.mem_map_of_mem hp1)))
              have e2 := lt_of_le_of_lt (le_foldl_max _ _ _ (Or.inl (List.mem_map_of_mem hp2))) hc.2
              dsimp only at h1 h2 e1 e2
              refine ⟨nn, hl, ?_⟩
              show (cc + ((k / inn : ℕ) : ℤ) * ost + ((k % inn : ℕ) : ℤ) * ist).toNat < nn
              generalize cc + ((k / inn : ℕ) : ℤ) * ost + ((k % inn : ℕ) : ℤ) * ist = E at h1 h2 ⊢
              have hE0 : 0 ≤ E := by linarith
              have hE1 : E < nn := by linarith
              omega
            · rw [if_neg hq0] at hp1 hp2
              split at hw
              all_goals try (injection hw; done)
              rename_i hc
              injection hw with hw
              injection hw with hp hi
              subst hp hi
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
              have e1 := le_trans hc.1 (foldl_min_le _ _ _ (Or.inl (List.mem_map_of_mem hp1)))
              have e2 := lt_of_le_of_lt (le_foldl_max _ _ _ (Or.inl (List.mem_map_of_mem hp2))) hc.2
              dsimp only at h1 h2 e1 e2
              refine ⟨nn, hl, ?_⟩
              show (cc + ((k / inn : ℕ) : ℤ) * ost + ((k % inn : ℕ) : ℤ) * ist).toNat < nn
              generalize cc + ((k / inn : ℕ) : ℤ) * ost + ((k % inn : ℕ) : ℤ) * ist = E at h1 h2 ⊢
              have hE0 : 0 ≤ E := by linarith
              have hE1 : E < nn := by linarith
              omega
          · split at h
            · rename_i e _
              rcases e with _ | _ | _ | _ | items | _
              all_goals try (injection h; done)
              dsimp only at h
              obtain ⟨rs, hm, h⟩ := except_bind_ok h
              injection h with h; subst h
              intro q hq
              simp only [Array.mem_singleton] at hq
              subst hq
              intro k hk
              simp only [Part.count] at hk
              have hr : (Part.explicit rs).get k ∈ rs := by
                show rs[k]! ∈ rs
                rw [getElem!_pos rs k hk]; exact Array.getElem_mem hk
              obtain ⟨x, -, hx⟩ := array_mapM_ok _ items rs hm _ hr
              generalize (Part.explicit rs).get k = r at hx ⊢
              rcases x with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
              all_goals try (injection hx; done)
              rcases l with _ | ⟨s0, _ | ⟨i0, _ | ⟨l0, _ | ⟨y, l⟩⟩⟩⟩
              all_goals try (injection hx; done)
              conv at hx => lhs; whnf
              split at hx
              all_goals try (injection hx; done)
              rename_i ll _
              split at hx
              all_goals try (injection hx; done)
              rename_i p i hw
              injection hx with hx; subst hx
              split at hw
              all_goals try (injection hw; done)
              split at hw
              all_goals try (injection hw; done)
              rename_i nn hl
              split at hw
              all_goals try (injection hw; done)
              rename_i hc
              injection hw with hw
              injection hw with hp hi
              subst hp hi
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
              exact ⟨nn, hl, by show ll.toNat < nn; omega⟩
            · split at h
              · obtain ⟨run, lanes, rfl, hp, hf⟩ := parseGrid_ok h
                intro q hq
                simp only [Array.mem_singleton] at hq
                subst hq
                exact grid_valid hp hf
              · exact absurd h (by intro h; injection h)

/-- **Every reference a parsed sequence holds names a leaf**, given that every argument an alias may name holds only
such references: affine and explicit parts by `within`'s check, strided ones by their corners' (`strided_corners`), and
concatenations by induction on the fuel. -/
theorem parsePartsIn_valid (leaves : Bool → ℕ → Option ℕ) (args : ℕ → ℕ → Option Seq)
    (hargs : ∀ k x s, args k x = some s → ∀ q ∈ s.parts, PartValid leaves q) :
    ∀ fuel j ps, parsePartsIn fuel j leaves args = .ok ps → ∀ q ∈ ps, PartValid leaves q := by
  intro fuel
  induction fuel with
  | zero =>
    intro j ps h
    delta parsePartsIn at h
    exact nomatch h
  | succ n ih =>
    intro j ps h
    delta parsePartsIn at h
    exact parsePartsIn_step leaves args hargs n _ (fun x ps hx => ih x ps (by delta parsePartsIn; exact hx)) j ps h

theorem foldl_add_ge {α : Type} (f : α → ℕ) : ∀ (l : List α) (s : ℕ), s ≤ l.foldl (fun s n => s + f n) s
  | [], _ => le_refl _
  | x :: xs, s => le_trans (Nat.le_add_right s (f x)) (foldl_add_ge f xs (s + f x))

theorem le_foldl_add {α : Type} (f : α → ℕ) :
    ∀ (l : List α) (s : ℕ) (x : α), x ∈ l → f x ≤ l.foldl (fun s n => s + f n) s
  | [], _, _, h => by simp at h
  | y :: ys, s, x, h => by
    rcases List.mem_cons.1 h with h | h
    · subst h; exact le_trans (Nat.le_add_left _ _) (foldl_add_ge f ys (s + f x))
    · exact le_foldl_add f ys _ x h

theorem foldl_max_ge {α : Type} (f : α → ℕ) : ∀ (l : List α) (s : ℕ), s ≤ l.foldl (fun s n => max s (f n)) s
  | [], _ => le_refl _
  | x :: xs, s => le_trans (le_max_left s (f x)) (foldl_max_ge f xs _)

theorem le_foldl_max_of_mem {α : Type} (f : α → ℕ) :
    ∀ (l : List α) (s : ℕ) (x : α), x ∈ l → f x ≤ l.foldl (fun s n => max s (f n)) s
  | [], _, _, h => by simp at h
  | y :: ys, s, x, h => by
    rcases List.mem_cons.1 h with h | h
    · subst h; exact le_trans (le_max_right s (f x)) (foldl_max_ge f ys _)
    · exact le_foldl_max_of_mem f ys _ x h

/-- **A composite's body laid out as the decoder lays it** (`Flock.Program.getDef`): its offsets from 0 to its gates, each
node's gates its members' callee's after the ones before (a primitive node's one: one member, of one gate). -/
def LaidOut (P : Prog) (d : ℕ) : Prop :=
  (P.defs[d]!).prim = false →
    (P.defs[d]!).body.offsets[0]! = 0 ∧ (P.defs[d]!).body.offsets[(P.defs[d]!).body.nodes.size]! = (P.defs[d]!).gates ∧
    ∀ k < (P.defs[d]!).body.nodes.size, (P.defs[d]!).body.offsets[k + 1]! =
      (P.defs[d]!).body.offsets[k]! + ((P.defs[d]!).body.nodes[k]!).members * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates

end InferenceOnlyVerifier.Properties.SecurityProofs.Walk
