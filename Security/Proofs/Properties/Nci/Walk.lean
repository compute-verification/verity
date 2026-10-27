import Proofs.Properties.Nci.DescAgree

/-!
# `Prog.gate`'s walk, within `Desc.walkable`

Toward `DescAgree.GatesLocated`: what `Desc.walkable` gives a walk of `Flock.Steps.run`. A walk whose steps each lower a
potential finishes within any fuel above the potential it starts at (`run_of_potential`), and `Desc.activations` and
`Desc.depths`, folded callees first, are their recurrences (`activations_get`, `depths_get`). A reference of a sequence
is its part's element, found by bisection over the parts' starts (`seq_get`, `bisect_spec`), and a strided part's
elements lie between the corners `Flock.Program.parsePartsIn` checks (`strided_corners`), so every reference a parsed
sequence holds names a leaf (`parsePartsIn_valid`). A composite's nodes' members hold fewer activations than it and nest
less deep (`activations_child`, `depths_child`), which is what the walk's potential lowers. On bodies laid out as the
decoder lays them (`LaidOut`), `Prog.gate`'s descent goes one level down a step to the activation holding a primitive
(`gateStep_holds`), so it locates every gate of the root's within `Desc.walkable`'s fuel wherever the operands it reaches
resolve in the activations it visits (`Reach`; `gate_ok_of`, `depth_lt_fuel`).

`Desc.walkable` checks every body laid out so, the root a composite, every primitive returning one value and every
input node called with no arguments (`Desc.bodiesLaidOut`, `bodiesLaidOut_spec`), so the primitive the walk locates
returns one value and an input's gate reads nothing (`gate_reads_of`).

So `GatesLocated` holds wherever two smaller statements do, each a named `Prop` here (`gatesLocated_of`):
`OperandsResolve` (operands resolve in every activation the walk visits; `OperandsResolve.operandsResolve_of` proves it
from `Resolve.BodiesOk`) and `OperandsEarlier` (every operand the walk locates is an earlier gate;
`Resolve.operandsEarlier_of` proves it from `Resolve.BodiesOk`), and `Resolve.bodiesOk` proves `Resolve.BodiesOk`.
`descAgrees_of_gaps` gives `DescAgrees` from those two and `NovelExact`.
-/

namespace NciCheck.Properties.SecurityProofs.Walk

open Flock.Program (Prog Seq Part parsePartsIn)

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

/-- **A composite's node's members are fewer activations than the composite's own**: `Desc.activations` counts one for
the activation itself, so a node's `members` copies of its callee's activations stay strictly below. -/
theorem activations_child (P : Prog) (hc : Desc.calleesFirst P = true) {d : ℕ} (hd : d < P.defs.size)
    (hp : (P.defs[d]!).prim = false) {n : Flock.Program.Node} (hn : n ∈ (P.defs[d]!).body.nodes) :
    n.members * (Desc.activations P)[n.fn]! < (Desc.activations P)[d]! := by
  rw [activations_get P hc hd, hp, if_neg (by simp)]
  have := le_foldl_add (fun n : Flock.Program.Node => n.members * (Desc.activations P)[n.fn]!)
    (P.defs[d]!).body.nodes.toList 0 n (Array.mem_toList_iff.2 hn)
  rw [Array.foldl_toList] at this
  omega

/-- **A composite's callees nest less deep than it**. -/
theorem depths_child (P : Prog) (hc : Desc.calleesFirst P = true) {d : ℕ} (hd : d < P.defs.size)
    (hp : (P.defs[d]!).prim = false) {n : Flock.Program.Node} (hn : n ∈ (P.defs[d]!).body.nodes) :
    (Desc.depths P)[n.fn]! < (Desc.depths P)[d]! := by
  rw [depths_get P hc hd, hp, if_neg (by simp)]
  have := le_foldl_max_of_mem (fun n : Flock.Program.Node => (Desc.depths P)[n.fn]!)
    (P.defs[d]!).body.nodes.toList 0 n (Array.mem_toList_iff.2 hn)
  rw [Array.foldl_toList] at this
  omega

/-- **A composite's body laid out as the decoder lays it** (`Flock.Program.getDef`): its offsets from 0 to its gates, each
node's gates its members' callee's after the ones before (a primitive node's one: one member, of one gate). -/
def LaidOut (P : Prog) (d : ℕ) : Prop :=
  (P.defs[d]!).prim = false →
    (P.defs[d]!).body.offsets[0]! = 0 ∧ (P.defs[d]!).body.offsets[(P.defs[d]!).body.nodes.size]! = (P.defs[d]!).gates ∧
    ∀ k < (P.defs[d]!).body.nodes.size, (P.defs[d]!).body.offsets[k + 1]! =
      (P.defs[d]!).body.offsets[k]! + ((P.defs[d]!).body.nodes[k]!).members * (P.defs[((P.defs[d]!).body.nodes[k]!).fn]!).gates

/-- What `Prog.gate P g`'s walk keeps: an activation of a composite Definition, `g` within its gates. -/
def Holds (P : Prog) (g : ℕ) (sc : Flock.Program.Scope) : Prop :=
  sc.d < P.defs.size ∧ (P.defs[sc.d]!).prim = false ∧ sc.off ≤ g ∧ g - sc.off < (P.defs[sc.d]!).gates

/-- **The member `gateStep` goes into**: `x` gates into a node, member `m` holds it, `x - m · gates` into that member. -/
theorem member_of (nd : Flock.Program.Node) (cg x : ℕ) (hx : x < nd.members * cg) :
    (if (nd.form == .prim || nd.form == .call) = true then 0 else x / cg) < nd.members ∧
      (if (nd.form == .prim || nd.form == .call) = true then 0 else x / cg) * cg ≤ x ∧
      x - (if (nd.form == .prim || nd.form == .call) = true then 0 else x / cg) * cg < cg := by
  have hpos : 0 < cg := Nat.pos_of_ne_zero (by rintro rfl; simp at hx)
  unfold Flock.Program.Node.members at *
  rcases nd with ⟨f, fn, args, out, axes, n, nxs⟩
  cases f
  · change x < 1 * cg at hx
    change 0 < 1 ∧ 0 * cg ≤ x ∧ x - 0 * cg < cg
    omega
  · change x < 1 * cg at hx
    change 0 < 1 ∧ 0 * cg ≤ x ∧ x - 0 * cg < cg
    omega
  · change x < n * cg at hx
    change x / cg < n ∧ x / cg * cg ≤ x ∧ x - x / cg * cg < cg
    refine ⟨(Nat.div_lt_iff_lt_mul hpos).2 hx, Nat.div_mul_le_self x cg, ?_⟩
    have h1 := Nat.div_add_mod x cg
    rw [Nat.mul_comm] at h1
    have h2 := Nat.mod_lt x hpos
    omega
  · change x < n * cg at hx
    change x / cg < n ∧ x / cg * cg ≤ x ∧ x - x / cg * cg < cg
    refine ⟨(Nat.div_lt_iff_lt_mul hpos).2 hx, Nat.div_mul_le_self x cg, ?_⟩
    have h1 := Nat.div_add_mod x cg
    rw [Nat.mul_comm] at h1
    have h2 := Nat.mod_lt x hpos
    omega

/-- **One step of `Prog.gate`'s walk goes down**: from an activation that `Holds`, where the operands of a primitive node's
member resolve, `gateStep` ends at a primitive or moves to an activation that `Holds`, of a Definition nesting less deep. -/
theorem gateStep_holds (P : Prog) (hc : Desc.calleesFirst P = true) (hlo : ∀ d < P.defs.size, LaidOut P d) {g : ℕ}
    {sc : Flock.Program.Scope} (hs : Holds P g sc)
    (hops : ∀ k < (P.defs[sc.d]!).body.nodes.size, (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true →
      ∀ m < ((P.defs[sc.d]!).body.nodes[k]!).members, ∀ e,
        Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m ≠ .error e) :
    (∃ x, Flock.Program.gateStep P g sc = .ok (.inl x)) ∨
      ∃ sc', Flock.Program.gateStep P g sc = .ok (.inr sc') ∧ Holds P g sc' ∧
        (Desc.depths P)[sc'.d]! < (Desc.depths P)[sc.d]! := by
  suffices H : ∀ r, Flock.Program.gateStep P g sc = r → (∀ e, r ≠ .error e) ∧ ∀ sc', r = .ok (.inr sc') →
      Holds P g sc' ∧ (Desc.depths P)[sc'.d]! < (Desc.depths P)[sc.d]! by
    rcases h : Flock.Program.gateStep P g sc with e | (x | sc')
    · exact absurd rfl ((H _ h).1 e)
    · exact Or.inl ⟨x, rfl⟩
    · exact Or.inr ⟨sc', rfl, (H _ h).2 sc' rfl⟩
  intro r h
  obtain ⟨hd, hp, hoff, hlt⟩ := hs
  obtain ⟨h0, hK, hnodes⟩ := hlo sc.d hd hp
  obtain ⟨k0, hk0, hb0⟩ := DescAgree.node_exists _ (P.defs[sc.d]!).body.nodes.size h0 (g - sc.off)
    (by rw [hK]; exact hlt)
  unfold Flock.Program.gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k hk
    obtain ⟨hkK, hpk⟩ := DescAgree.find?_range_some hk
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hpk
    have hx : g - sc.off - (P.defs[sc.d]!).body.offsets[k]! <
        ((P.defs[sc.d]!).body.nodes[k]!).members * (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates := by
      have := hnodes k hkK
      omega
    obtain ⟨hm1, hm2, hm3⟩ := member_of ((P.defs[sc.d]!).body.nodes[k]!) _ _ hx
    by_cases hpc : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
    · simp only [hpc, ite_true] at h
      split at h
      · rename_i e he
        exact absurd he (hops k hkK hpc _ hm1 e)
      · simp only [pure, Except.pure] at h
        subst h
        exact ⟨fun e he => (by cases he), fun sc' he => (by cases he)⟩
    · simp only [hpc, Bool.false_eq_true, ite_false, pure, Except.pure] at h
      subst h
      refine ⟨fun e he => (by cases he), fun sc' he => ?_⟩
      simp only [Except.ok.injEq, Sum.inr.injEq] at he
      subst he
      have hmem : (P.defs[sc.d]!).body.nodes[k]! ∈ (P.defs[sc.d]!).body.nodes := by
        rw [getElem!_pos _ k hkK]
        exact Array.getElem_mem hkK
      have hfn := calleesFirst_spec P hc sc.d hd _ hmem
      simp only [Holds, Flock.Program.Prog.child, Flock.Program.Scope.d.eq_2, Flock.Program.Scope.off.eq_2]
      refine ⟨⟨by omega, by simpa using hpc, by omega, by omega⟩, depths_child P hc hd hp hmem⟩
  · exfalso
    first
    | (rename_i hne
       have hnone := Option.eq_none_iff_forall_ne_some.2 hne
       rw [Array.find?_eq_none] at hnone
       exact hnone k0 (Array.mem_range.2 hk0) (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hb0))
    | (rename_i hne _
       have hnone := Option.eq_none_iff_forall_ne_some.2 hne
       rw [Array.find?_eq_none] at hnone
       exact hnone k0 (Array.mem_range.2 hk0) (by simp only [Bool.and_eq_true, decide_eq_true_eq]; exact hb0))

/-- The activations `Prog.gate P g`'s walk visits: the root's, and each one a step of it moves to. -/
inductive Reach (P : Prog) (g : ℕ) : Flock.Program.Scope → Prop
  | root : Reach P g P.rootScope
  | step {sc sc' : Flock.Program.Scope} : Reach P g sc → Flock.Program.gateStep P g sc = .ok (.inr sc') → Reach P g sc'

/-- **`Prog.gate`'s walk goes down to a primitive**: on a Program its decoder laid out, every gate of the root's is located
within the root's depth of steps, where the operands of each primitive node's member resolve in the activations the walk
visits (`Reach`). -/
theorem gate_ok_of (P : Prog) (hc : Desc.calleesFirst P = true) (hlo : ∀ d < P.defs.size, LaidOut P d)
    (hr : P.root < P.defs.size) (hrp : (P.defs[P.root]!).prim = false)
    (hfuel : (Desc.depths P)[P.root]! < Flock.Steps.FUEL) {g : ℕ} (hg : g < (P.defs[P.root]!).gates)
    (hops : ∀ sc, Reach P g sc → Holds P g sc → ∀ k < (P.defs[sc.d]!).body.nodes.size,
      (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true → ∀ m < ((P.defs[sc.d]!).body.nodes[k]!).members, ∀ e,
        Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m ≠ .error e) :
    ∃ x, P.gate g = .ok x := by
  obtain ⟨b, hb, -⟩ := run_of_potential (what := "gate location") (step := Flock.Program.gateStep P g)
    (fun sc => Reach P g sc ∧ Holds P g sc) (fun sc => (Desc.depths P)[sc.d]!) (fun _ => True)
    (fun sc hsc => by
      obtain ⟨hR, hs⟩ := hsc
      rcases gateStep_holds P hc hlo hs (hops sc hR hs) with ⟨x, hx⟩ | ⟨sc', h1, h2, h3⟩
      · exact Or.inl ⟨x, hx, trivial⟩
      · exact Or.inr ⟨sc', h1, ⟨Reach.step hR h1, h2⟩, h3⟩)
    Flock.Steps.FUEL P.rootScope
    ⟨Reach.root, ⟨hr, hrp, Nat.zero_le _, (by show g - 0 < (P.defs[P.root]!).gates; omega)⟩⟩ hfuel
  have hng : ¬ (g ≥ (P.defs[P.root]!).gates) := by omega
  refine ⟨b, ?_⟩
  simp only [Flock.Program.Prog.gate, hng, ite_false, bind, Except.bind]
  exact hb

/-- **`Desc.walkable` leaves the descent its fuel.** -/
theorem depth_lt_fuel (P : Prog) (h : Desc.walkable P = true) : (Desc.depths P)[P.root]! < Flock.Steps.FUEL := by
  unfold Desc.walkable Desc.walkBound at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  have h2 := h.2.2
  have : (Desc.depths P)[P.root]! + 2 ≤
      ((Desc.activations P)[P.root]! + 1) * ((Desc.depths P)[P.root]! + 2) := by
    exact Nat.le_mul_of_pos_left _ (by omega)
  omega

/-- What `Desc.bodiesLaidOut` checks: the root a composite, every composite's body laid out (`LaidOut`), every primitive
returning one `Value<w>`, and every input node of a composite called with no arguments. -/
theorem bodiesLaidOut_spec (P : Prog) (h : Desc.bodiesLaidOut P = true) :
    (P.defs[P.root]!).prim = false ∧ (∀ d < P.defs.size, LaidOut P d) ∧
      (∀ d < P.defs.size, (P.defs[d]!).prim = true → ∃ w, (P.defs[d]!).ret = .value w) ∧
      ∀ d < P.defs.size, (P.defs[d]!).prim = false → ∀ k < (P.defs[d]!).body.nodes.size,
        Desc.isInputNode P ((P.defs[d]!).body.nodes[k]!) = true → ((P.defs[d]!).body.nodes[k]!).args = #[] := by
  unfold Desc.bodiesLaidOut at h
  simp only [Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true, List.mem_range] at h
  obtain ⟨hr, hall⟩ := h
  refine ⟨hr, fun d hd hp => ?_, fun d hd hp => ?_, fun d hd hp k hk hin => ?_⟩
  · have := hall d hd
    simp only [hp, Bool.false_eq_true, ite_false, Bool.and_eq_true, beq_iff_eq, List.all_eq_true,
      List.mem_range] at this
    exact ⟨this.1.1, this.1.2, fun k hk => (this.2 k hk).1⟩
  · have := hall d hd
    simp only [hp, ite_true] at this
    revert this
    cases (P.defs[d]!).ret <;> simp
  · have := hall d hd
    simp only [hp, Bool.false_eq_true, ite_false, Bool.and_eq_true, beq_iff_eq, List.all_eq_true,
      List.mem_range] at this
    have := (this.2 k hk).2
    simp only [hin, Bool.not_true, Bool.false_or, Array.isEmpty_iff] at this
    exact this

/-- A step of the walk that ends: at a primitive node `k` of the activation, with its member's operands. -/
theorem gateStep_inl (P : Prog) (g : ℕ) (sc : Flock.Program.Scope) (x)
    (h : Flock.Program.gateStep P g sc = .ok (.inl x)) :
    ∃ k < (P.defs[sc.d]!).body.nodes.size, ∃ m, (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true ∧
      x.1 = ((P.defs[sc.d]!).body.nodes[k]!).fn ∧ x.2.2 = sc ∧
      Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m = .ok x.2.1 := by
  unfold Flock.Program.gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k hk
    obtain ⟨hkK, -⟩ := DescAgree.find?_range_some hk
    by_cases hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
    · simp only [hp, ite_true] at h
      split at h
      · cases h
      · rename_i ops hops
        simp only [pure, Except.pure, Except.ok.injEq, Sum.inl.injEq] at h
        subst h
        exact ⟨k, hkK, _, hp, rfl, rfl, hops⟩
    · simp only [hp, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq] at h
      cases h
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- A step of the walk that goes on: into a member of a composite node `k` of the activation. -/
theorem gateStep_inr (P : Prog) (g : ℕ) (sc sc' : Flock.Program.Scope)
    (h : Flock.Program.gateStep P g sc = .ok (.inr sc')) :
    ∃ k < (P.defs[sc.d]!).body.nodes.size, (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false ∧
      sc'.d = ((P.defs[sc.d]!).body.nodes[k]!).fn := by
  unfold Flock.Program.gateStep at h
  simp only [bind, Except.bind] at h
  split at h
  · rename_i k hk
    obtain ⟨hkK, -⟩ := DescAgree.find?_range_some hk
    by_cases hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true
    · simp only [hp, ite_true] at h
      split at h
      · cases h
      · simp only [pure, Except.pure, Except.ok.injEq] at h
        cases h
    · simp only [hp, Bool.false_eq_true, ite_false, pure, Except.pure, Except.ok.injEq, Sum.inr.injEq] at h
      subst h
      exact ⟨k, hkK, by simpa using hp, rfl⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem operands_nil (P : Prog) (sc : Flock.Program.Scope) (node : Flock.Program.Node) (m : ℕ) (ops : Array ℕ)
    (hargs : node.args = #[]) (h : Flock.Program.operands P sc node m = .ok ops) : ops = #[] := by
  unfold Flock.Program.operands at h
  simp [hargs, pure, Except.pure, bind, Except.bind] at h
  first | exact h | exact h.symm

/-- **What the walk locates, on bodies the decoder's way**: the primitive returns one `Value<w>`, and an input primitive's
gate reads nothing (it is an input node, so `Desc.bodiesLaidOut` gives it no arguments). -/
theorem gate_reads_of (P : Prog) (hb : Desc.bodiesLaidOut P = true) {g p : ℕ} {ops : Array ℕ}
    {sc : Flock.Program.Scope} (h : P.gate g = .ok (p, ops, sc)) :
    (∃ w, (P.defs[p]!).ret = .value w) ∧ (Flock.Program.isInputId (P.defs[p]!).key = true → ops = #[]) := by
  obtain ⟨hr, -, hret, hinp⟩ := bodiesLaidOut_spec P hb
  unfold Flock.Program.Prog.gate at h
  by_cases hge : g ≥ (P.defs[P.root]!).gates
  · simp [hge, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  simp only [hge, ite_false, bind, Except.bind] at h
  refine DescAgree.run_inv "gate location" (Flock.Program.gateStep P g) (fun sc => (P.defs[sc.d]!).prim = false)
    (fun b => (∃ w, (P.defs[b.1]!).ret = .value w) ∧ (Flock.Program.isInputId (P.defs[b.1]!).key = true → b.2.1 = #[]))
    ?_ ?_ Flock.Steps.FUEL P.rootScope (p, ops, sc) hr h
  · intro a b ha hb'
    obtain ⟨k, hk, m, hp, h1, -, hops⟩ := gateStep_inl P g a b hb'
    have hd : a.d < P.defs.size := by
      refine (Nat.lt_or_ge a.d P.defs.size).elim id fun hn => ?_
      rw [getElem!_neg P.defs a.d (by omega)] at hk
      have h0 : (default : Flock.Program.Defn).body.nodes.size = 0 := rfl
      omega
    have hfn : ((P.defs[a.d]!).body.nodes[k]!).fn < P.defs.size := by
      refine (Nat.lt_or_ge _ P.defs.size).elim id fun hn => ?_
      rw [getElem!_neg P.defs _ (by omega)] at hp
      cases hp
    rw [h1]
    refine ⟨hret _ hfn hp, fun hkey => operands_nil P a _ m _ (hinp a.d hd ha k hk ?_) hops⟩
    simp [Desc.isInputNode, hp, hkey]
  · intro a a' _ ha'
    obtain ⟨k, -, hp, hd'⟩ := gateStep_inr P g a a' ha'
    show (P.defs[a'.d]!).prim = false
    rw [hd']; exact hp

/-- **Operands resolve on the walk.** On a Program `Desc.evaluate` reads, in every activation `Prog.gate`'s walk
visits, every primitive node's members' operands resolve (`Flock.Program.resolve` within its fuel, past no scan of no
iterations). Only a primitive node's: `Prog.gate` resolves no other node's operands, and a composite node's arguments
may name a scan of no iterations that no gate reads. -/
def OperandsResolve : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), Flock.Program.decode bytes = .ok P → P.orderViolation = none →
    Desc.rootLaidOut P = true → Desc.inputsAtRoot P = true → Desc.walkable P = true →
    (∃ reads, Desc.analyze P = .ok reads) → ∀ g < gates P, ∀ sc, Reach P g sc → Holds P g sc →
      ∀ k < (P.defs[sc.d]!).body.nodes.size, (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true →
        ∀ m < ((P.defs[sc.d]!).body.nodes[k]!).members, ∀ e,
          Flock.Program.operands P sc ((P.defs[sc.d]!).body.nodes[k]!) m ≠ .error e

/-- **Every operand is an earlier gate.** On such a Program, each operand `Prog.gate` locates for a gate is a gate
before it. -/
def OperandsEarlier : Prop :=
  ∀ (bytes : ByteArray) (P : Prog), Flock.Program.decode bytes = .ok P → P.orderViolation = none →
    Desc.rootLaidOut P = true → Desc.inputsAtRoot P = true → Desc.walkable P = true →
    (∃ reads, Desc.analyze P = .ok reads) → ∀ g < gates P, ∀ p ops sc, P.gate g = .ok (p, ops, sc) → ∀ a ∈ ops, a < g

/-- **`GatesLocated`, from resolution on the walk and the order of its operands**: `Desc.walkable` gives the walk its fuel
(`depth_lt_fuel`) and bodies laid out (`bodiesLaidOut_spec`), so with operands resolving it ends at a primitive
(`gate_ok_of`), which returns one value and, for an input, reads nothing (`gate_reads_of`). -/
theorem gatesLocated_of (hO : OperandsResolve) (hE : OperandsEarlier) : GatesLocated := by
  intro bytes P hdec hov hlo hin hwk han g hg
  have hwk' := hwk
  unfold Desc.walkable at hwk'
  simp only [Bool.and_eq_true] at hwk'
  have hc : Desc.calleesFirst P = true := hwk'.1.1
  have hb : Desc.bodiesLaidOut P = true := hwk'.1.2
  obtain ⟨hrp, hlay, -, -⟩ := bodiesLaidOut_spec P hb
  have hr : P.root < P.defs.size := by
    refine (Nat.lt_or_ge P.root P.defs.size).elim id fun hn => ?_
    have hg' : g < (P.defs[P.root]!).gates := hg
    rw [getElem!_neg P.defs P.root (by omega)] at hg'
    have h0 : (default : Flock.Program.Defn).gates = 0 := rfl
    omega
  obtain ⟨⟨p, ops, sc⟩, hx⟩ := gate_ok_of P hc hlay hr hrp (depth_lt_fuel P hwk) hg
    (hO bytes P hdec hov hlo hin hwk han g hg)
  obtain ⟨⟨w, hw⟩, hinp⟩ := gate_reads_of P hb hx
  exact ⟨p, ops, w, ⟨⟨sc, hx⟩, hw, hinp⟩, hE bytes P hdec hov hlo hin hwk han g hg p ops sc hx⟩

/-- **`DescAgrees`, from three statements**: `GatesLocated` through `gatesLocated_of`, and `NovelExact`. -/
theorem descAgrees_of_gaps (hO : OperandsResolve) (hE : OperandsEarlier) (hN : NovelExact) :
    Properties.DescAgrees :=
  descAgrees_of (gatesLocated_of hO hE) hN

end NciCheck.Properties.SecurityProofs.Walk
