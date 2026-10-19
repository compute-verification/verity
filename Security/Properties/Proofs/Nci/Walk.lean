import Properties.Proofs.Nci

/-!
# `Prog.gate`'s walk, within `Desc.walkable`

Toward `DescAgree.GatesLocated`: what `Desc.walkable` gives a walk of `Flock.Steps.run`. A walk whose steps each lower a
potential finishes within any fuel above the potential it starts at (`run_of_potential`), and `Desc.activations` and
`Desc.depths`, folded callees first, are their recurrences (`activations_get`, `depths_get`).
-/

namespace NciCheck.Properties.SecurityProofs.Walk

open Flock.Program (Prog)

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

end NciCheck.Properties.SecurityProofs.Walk
