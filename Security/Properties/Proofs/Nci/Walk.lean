import Properties.Proofs.Nci

/-!
# `Prog.gate`'s walk, within `Desc.walkable`

Toward `DescAgree.GatesLocated`: what `Desc.walkable` gives a walk of `Flock.Steps.run`. A walk whose steps each lower a
potential finishes within any fuel above the potential it starts at (`run_of_potential`), and `Desc.activations` and
`Desc.depths`, folded callees first, are their recurrences (`activations_get`, `depths_get`). A reference of a sequence
is its part's element, found by bisection over the parts' starts (`seq_get`, `bisect_spec`), and a strided part's
elements lie between the corners `Flock.Program.parsePartsIn` checks (`strided_corners`).
-/

namespace NciCheck.Properties.SecurityProofs.Walk

open Flock.Program (Prog Seq Part)

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

theorem calleesFirst_spec (P : Prog) (h : Desc.calleesFirst P = true) :
    ∀ d < P.defs.size, ∀ n ∈ (P.defs[d]!).body.nodes, n.fn < d := by
  intro d hd n hn
  unfold Desc.calleesFirst at h
  have := List.all_eq_true.1 h d (List.mem_range.2 hd)
  exact of_decide_eq_true (Array.all_eq_true_iff_forall_mem.1 this n hn)

theorem activations_size (P : Prog) : (Desc.activations P).size = P.defs.size :=
  (pushFold_spec _ _).1

/-- **The activations of one activation of Definition `d`**: none for a primitive, else its own and its composite
nodes' members'. -/
theorem activations_get (P : Prog) (hc : Desc.calleesFirst P = true) {d : ℕ} (hd : d < P.defs.size) :
    (Desc.activations P)[d]! = if (P.defs[d]!).prim then 0 else
      1 + (P.defs[d]!).body.nodes.foldl (fun s n => s + n.members * (Desc.activations P)[n.fn]!) 0 := by
  unfold Desc.activations
  rw [(pushFold_spec _ _).2 d hd]
  simp only
  split
  · rfl
  · congr 1
    apply foldl_nodes_congr
    intro n hn s
    rw [getElem!_extract _ (calleesFirst_spec P hc d hd n hn) (by rw [(pushFold_spec _ _).1]; omega)]

theorem depths_size (P : Prog) : (Desc.depths P).size = P.defs.size :=
  (pushFold_spec _ _).1

/-- **How deep Definition `d`'s activations nest**: none for a primitive, else one more than its callees'. -/
theorem depths_get (P : Prog) (hc : Desc.calleesFirst P = true) {d : ℕ} (hd : d < P.defs.size) :
    (Desc.depths P)[d]! = if (P.defs[d]!).prim then 0 else
      1 + (P.defs[d]!).body.nodes.foldl (fun s n => max s (Desc.depths P)[n.fn]!) 0 := by
  unfold Desc.depths
  rw [(pushFold_spec _ _).2 d hd]
  simp only
  split
  · rfl
  · congr 1
    apply foldl_nodes_congr
    intro n hn s
    rw [getElem!_extract _ (calleesFirst_spec P hc d hd n hn) (by rw [(pushFold_spec _ _).1]; omega)]


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
    let vals := corners.map fun (x, y) => b + (x : ℤ) * os + (y : ℤ) * is
    vals.foldl min vals.head! ≤ b + ((t / inner : ℕ) : ℤ) * os + ((t % inner : ℕ) : ℤ) * is ∧
      b + ((t / inner : ℕ) : ℤ) * os + ((t % inner : ℕ) : ℤ) * is ≤ vals.foldl max vals.head! := by
  intro q last corners vals
  have hq : q = (c - 1) / inner := rfl
  have hl : last = (c - 1) % inner := rfl
  clear_value q last
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
  exact ⟨le_trans (foldl_min_le _ _ _ (Or.inl hu)) hu', le_trans hu2' (le_foldl_max _ _ _ (Or.inl hu2))⟩

end NciCheck.Properties.SecurityProofs.Walk
