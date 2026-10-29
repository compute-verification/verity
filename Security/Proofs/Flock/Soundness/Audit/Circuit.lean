import Definitions.Flock.Circuit

/-!
# Circuits, partitions, and the correctness of a unit: the lemmas

About the circuits and partitions of `Definitions.Flock.Circuit`:

* `compose`: if every unit is correct and the committed inputs are the anchors' values, every
  committed wire carries its value in the circuit's evaluation;
* `compose_cone`: the same for the wires `T` a consumer reads, given only that the units of `T`'s
  cone (the units holding an ancestor of `T`) are correct;
* `correct_congr`: a unit's correctness depends only on its committed inputs and outputs
  (`Partition.io`), which is what an extracted opening of the unit determines;
* `Refines.exists_wrong_fine`: when a fine partition refines a coarse one, a coarse unit that is
  wrong has a wrong fine unit inside it, whatever the values on the coarse unit's interior.
-/

namespace FlockSoundness.Audit

theorem xorSum_map {ι κ : Type} (l : List ι) (f : ι → κ) (v : κ → Bool) :
    xorSum (l.map f) v = xorSum l (v ∘ f) := by
  unfold xorSum
  rw [List.map_map]

namespace Circuit

variable (C : Circuit)

theorem run_eq (fix : Fin C.N → Option Bool) (g : Fin C.N) :
    C.run fix g = match fix g with
      | some b => b
      | none => (C.op g).apply (C.run fix) := by
  rw [run]
  cases fix g with
  | some b => rfl
  | none =>
      exact Op.apply_congr _ fun x hx => by simp [C.topo g x hx]

theorem run_fixed {fix : Fin C.N → Option Bool} {g : Fin C.N} {b : Bool} (h : fix g = some b) :
    C.run fix g = b := by
  rw [run_eq, h]

theorem run_free {fix : Fin C.N → Option Bool} {g : Fin C.N} (h : fix g = none) :
    C.run fix g = (C.op g).apply (C.run fix) := by
  rw [run_eq, h]

theorem eval_input (a : Fin C.N → Bool) {g : Fin C.N} (h : (C.op g).isInput = true) :
    C.eval a g = a g :=
  C.run_fixed (by simp [h])

theorem eval_op (a : Fin C.N → Bool) {g : Fin C.N} (h : (C.op g).isInput = false) :
    C.eval a g = (C.op g).apply (C.eval a) :=
  C.run_free (by simp [h])

end Circuit

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

theorem mem_committed_of_read {g h : Fin C.N} (hg : g ∈ (C.op h).args) (hne : P.unit h ≠ P.unit g) :
    g ∈ P.committed := by
  classical
  simp only [committed, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Or.inr (Or.inr ⟨h, hg, hne⟩)

theorem mem_committed_of_output {g : Fin C.N} (hg : g ∈ C.outputs) : g ∈ P.committed := by
  classical
  simp only [committed, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Or.inr (Or.inl hg)

theorem mem_committed_of_input {g : Fin C.N} (hg : (C.op g).isInput = true) : g ∈ P.committed := by
  classical
  simp only [committed, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Or.inl hg

theorem localEval_of_mem (X : Fin C.N → Bool) {u : Fin n} {g : Fin C.N} (h : P.unit g = some u) :
    P.localEval X u g = (C.op g).apply (P.localEval X u) :=
  C.run_free (by simp [h])

theorem localEval_of_not_mem (X : Fin C.N → Bool) {u : Fin n} {g : Fin C.N}
    (h : P.unit g ≠ some u) : P.localEval X u g = X g :=
  C.run_fixed (by simp [h])

theorem mem_wrong {X : Fin C.N → Bool} {u : Fin n} : u ∈ P.wrong X ↔ ¬ P.Correct X u := by
  classical
  simp [wrong]

theorem mem_cone {T : Finset (Fin C.N)} {u : Fin n} {t g : Fin C.N} (ht : t ∈ T) (hg : C.Feeds g t)
    (hu : P.unit g = some u) : u ∈ P.cone T := by
  classical
  simp only [cone, Finset.mem_filter, Finset.mem_univ, true_and]
  exact ⟨t, ht, g, hg, hu⟩

/-- The induction behind `compose_cone`: on the ancestors of `T`, committed wires carry their
evaluation, and so does each unit's local evaluation of its own gates. -/
theorem cone_induction (X a : Fin C.N → Bool) (T : Finset (Fin C.N)) (hin : C.InputsAgree X a)
    (hc : ∀ u ∈ P.cone T, P.Correct X u) :
    ∀ m (g : Fin C.N), g.val = m → (∃ t ∈ T, C.Feeds g t) →
      (∀ u, P.unit g = some u → P.localEval X u g = C.eval a g) ∧
      (g ∈ P.committed → X g = C.eval a g) := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g hgm ⟨t, ht, hgt⟩
  have hB : ∀ u, P.unit g = some u → P.localEval X u g = C.eval a g := by
    intro u hu
    have hni : (C.op g).isInput = false := by
      cases h : (C.op g).isInput
      · rfl
      · exact absurd ((P.input_iff g).2 h) (by simp [hu])
    rw [P.localEval_of_mem X hu, C.eval_op a hni]
    refine Op.apply_congr _ fun x hx => ?_
    have hxg : x.val < m := hgm ▸ C.topo g x hx
    have hxt : ∃ t ∈ T, C.Feeds x t := ⟨t, ht, .step hx hgt⟩
    obtain ⟨ihB, ihA⟩ := ih x.val hxg x rfl hxt
    by_cases hxu : P.unit x = some u
    · exact ihB u hxu
    · rw [P.localEval_of_not_mem X hxu]
      exact ihA (P.mem_committed_of_read hx (by rw [hu]; exact Ne.symm hxu))
  refine ⟨hB, fun hg => ?_⟩
  cases hi : (C.op g).isInput
  · obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 fun h => by
      simp [(P.input_iff g).1 h] at hi
    rw [hc u (P.mem_cone ht hgt hu) g hg hu]
    exact hB u hu
  · rw [hin g hi, C.eval_input a hi]

/-- **A consumer's value, through its cone.** If the committed inputs are the anchors' values and
no unit of `T`'s cone is wrong, every committed wire of `T` carries its evaluation. -/
theorem compose_cone (X a : Fin C.N → Bool) (T : Finset (Fin C.N)) (hin : C.InputsAgree X a)
    (h : Disjoint (P.wrong X) (P.cone T)) : ∀ t ∈ T, t ∈ P.committed → X t = C.eval a t := by
  intro t ht htc
  have hc : ∀ u ∈ P.cone T, P.Correct X u := fun u hu => by
    by_contra hw
    exact Finset.disjoint_left.1 h (P.mem_wrong.2 hw) hu
  exact (P.cone_induction X a T hin hc t.val t rfl ⟨t, ht, .refl t⟩).2 htc

/-- **Composition.** If every unit is correct and the committed inputs are the anchors' values,
every committed wire carries its value in the circuit's evaluation. -/
theorem compose (X a : Fin C.N → Bool) (hin : C.InputsAgree X a) (h : P.wrong X = ∅) :
    ∀ g ∈ P.committed, X g = C.eval a g :=
  fun g hg => P.compose_cone X a P.committed hin (by simp [h]) g hg hg

theorem localEval_congr {X Y : Fin C.N → Bool} {u : Fin n} (hXY : ∀ g ∈ P.io u, X g = Y g) :
    ∀ m (g : Fin C.N), g.val = m → P.unit g = some u → P.localEval X u g = P.localEval Y u g := by
  classical
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g hgm hu
  rw [P.localEval_of_mem X hu, P.localEval_of_mem Y hu]
  refine Op.apply_congr _ fun x hx => ?_
  by_cases hxu : P.unit x = some u
  · exact ih x.val (hgm ▸ C.topo g x hx) x rfl hxu
  · rw [P.localEval_of_not_mem X hxu, P.localEval_of_not_mem Y hxu]
    refine hXY x ?_
    simp only [io, Finset.mem_filter]
    exact ⟨P.mem_committed_of_read hx (by rw [hu]; exact Ne.symm hxu), Or.inr ⟨g, hu, hx⟩⟩

/-- A unit's correctness depends only on its committed inputs and outputs. -/
theorem correct_congr {X Y : Fin C.N → Bool} {u : Fin n} (hXY : ∀ g ∈ P.io u, X g = Y g)
    (hY : P.Correct Y u) : P.Correct X u := by
  classical
  intro g hg hu
  have hio : g ∈ P.io u := by simp only [io, Finset.mem_filter]; exact ⟨hg, Or.inl hu⟩
  rw [hXY g hio, P.localEval_congr hXY g.val g rfl hu]
  exact hY g hg hu

/-! ## Refinement -/

namespace Refines

variable {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc} (r : Refines Pf Pc)
include r

/-- A coarse committed wire is a fine committed wire. -/
theorem committed_subset : Pc.committed ⊆ Pf.committed := by
  classical
  intro g hg
  simp only [committed, Finset.mem_filter, Finset.mem_univ, true_and] at hg ⊢
  rcases hg with hg | hg | ⟨h, hgh, hne⟩
  · exact Or.inl hg
  · exact Or.inr (Or.inl hg)
  · refine Or.inr (Or.inr ⟨h, hgh, fun heq => hne ?_⟩)
    rw [r.sub h, r.sub g, heq]

/-- The induction behind `exists_wrong_fine`: when every fine unit inside `u` is correct in `Y`,
and `Y` agrees with `X` on the coarse committed wires, `u`'s local evaluation in `X` is each fine
unit's local evaluation in `Y`. -/
theorem localEval_eq {X Y : Fin C.N → Bool} {u : Fin nc}
    (hY : ∀ g ∈ Pc.committed, Y g = X g) (hf : ∀ v, r.parent v = u → Pf.Correct Y v) :
    ∀ m (g : Fin C.N) (v : Fin nf), g.val = m → Pf.unit g = some v → r.parent v = u →
      Pc.localEval X u g = Pf.localEval Y v g := by
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g v hgm hv hvu
  have hu : Pc.unit g = some u := by rw [r.sub g, hv]; simp [hvu]
  rw [Pc.localEval_of_mem X hu, Pf.localEval_of_mem Y hv]
  refine Op.apply_congr _ fun x hx => ?_
  have hxg : x.val < m := hgm ▸ C.topo g x hx
  by_cases hxu : Pc.unit x = some u
  · obtain ⟨v', hv', hv'u⟩ : ∃ v', Pf.unit x = some v' ∧ r.parent v' = u := by
      rw [r.sub x] at hxu
      simpa using hxu
    rw [ih x.val hxg x v' rfl hv' hv'u]
    by_cases hvv : v' = v
    · subst hvv; rfl
    · have hxv : Pf.unit x ≠ some v := by rw [hv']; simpa using hvv
      rw [Pf.localEval_of_not_mem Y hxv]
      have hxc : x ∈ Pf.committed := Pf.mem_committed_of_read hx (by rw [hv, hv']; simpa using Ne.symm hvv)
      exact (hf v' hv'u x hxc hv').symm
  · have hxv : Pf.unit x ≠ some v := fun h => hxu (by rw [r.sub x, h]; simp [hvu])
    rw [Pc.localEval_of_not_mem X hxu, Pf.localEval_of_not_mem Y hxv]
    exact (hY x (Pc.mem_committed_of_read hx (by rw [hu]; exact Ne.symm hxu))).symm

/-- **A wrong coarse unit has a wrong fine unit inside it**, whatever `Y` holds on the coarse unit's
interior, as long as `Y` agrees with `X` on the coarse committed wires. -/
theorem exists_wrong_fine {X Y : Fin C.N → Bool} (hY : ∀ g ∈ Pc.committed, Y g = X g)
    {u : Fin nc} (hu : u ∈ Pc.wrong X) : ∃ v, r.parent v = u ∧ v ∈ Pf.wrong Y := by
  by_contra hno
  push Not at hno
  have hf : ∀ v, r.parent v = u → Pf.Correct Y v := fun v hv => by
    have := hno v hv
    rwa [Pf.mem_wrong, not_not] at this
  refine Pc.mem_wrong.1 hu fun g hg hgu => ?_
  obtain ⟨v, hv, hvu⟩ : ∃ v, Pf.unit g = some v ∧ r.parent v = u := by
    rw [r.sub g] at hgu
    simpa using hgu
  rw [← hY g hg, hf v hvu g (r.committed_subset hg) hv, r.localEval_eq hY hf g.val g v rfl hv hvu]

/-- A fine transcript that extends a coarse one has at least as many wrong fine units as the coarse
one has wrong coarse units. -/
theorem card_wrong_le {X Y : Fin C.N → Bool} (hY : ∀ g ∈ Pc.committed, Y g = X g) :
    (Pc.wrong X).card ≤ (Pf.wrong Y).card := by
  classical
  calc (Pc.wrong X).card ≤ ((Pf.wrong Y).image r.parent).card := by
        refine Finset.card_le_card fun u hu => ?_
        obtain ⟨v, hv, hvw⟩ := r.exists_wrong_fine hY hu
        exact Finset.mem_image.2 ⟨v, hvw, hv⟩
    _ ≤ (Pf.wrong Y).card := Finset.card_image_le

end Refines

end Partition

end FlockSoundness.Audit
