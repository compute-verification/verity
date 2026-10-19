import Properties.Proofs.Nci

/-!
# `Prog.gate`'s walk, within `Desc.walkable`

Toward `DescAgree.GatesLocated`: what `Desc.walkable` gives a walk of `Flock.Steps.run`. A walk whose steps each lower a
potential finishes within any fuel above the potential it starts at (`run_of_potential`), and `Desc.activations` and
`Desc.depths`, folded callees first, are their recurrences (`activations_get`, `depths_get`). A reference of a sequence
is its part's element, found by bisection over the parts' starts (`seq_get`, `bisect_spec`).
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

end NciCheck.Properties.SecurityProofs.Walk
