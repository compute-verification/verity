import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Definitions.Accounting.Guarantees.RequirementPouw

/-!
# PoUW's credits and help when a node runs a declared circuit as placed

The actual run of a placed execution has `W`'s regions (`Placement.region_actual`): its computed gates are the values the
execution produces but the registered ones and the run's inputs, each at its device and its step's tick. Its wires into
a region are operands its gates use (`Placement.mem_into_of_actual`), all of them in a topological wiring the execution
follows (`Placement.ingress_actual`). PoUW's credits and help are `Placed.work` and `Placed.ingress` of the run at each
coin, so `RequirementCredits`, `RequirementHelp` and `RequirementHelpBound` follow. `RequirementStepsInner` and
`RequirementStepsOuter` order a paced device's steps as their ticks.
-/

namespace Accounting.Requirement

open Finset

variable {n : ℕ} {L : Type} (W : Wiring (Fin n)) (p : Placement L (Fin n))

/-- A gate is computed in the actual run when the execution produces it and it is neither registered nor a run input. -/
theorem Placement.computed_actual (g : Fin n) :
    (p.actual W).A.isInput g = false ↔ g ∈ p.ran ∧ g ∉ W.registered ∧ g ∉ W.inputs := by
  simp only [decide_eq_false_iff_not, not_or, not_not]

theorem Placement.mem_region_actual {X : Finset L} {I : Set ℕ} (g : Fin n) :
    g ∈ (p.actual W).A.region X I ↔ g ∈ region W p X I := by
  simp only [Placed.region, Verity.Circuit.computed, region, mem_filter, mem_univ, true_and, p.computed_actual W,
    and_assoc]

/-- The actual run's region is `W`'s. -/
theorem Placement.region_actual (X : Finset L) (I : Set ℕ) : (p.actual W).A.region X I = region W p X I :=
  Finset.ext fun g => p.mem_region_actual W g

/-- A gate of a region reads the earlier operands it uses. -/
theorem Placement.reads_of_region {X : Finset L} {I : Set ℕ} {g : Fin n} (hg : g ∈ region W p X I) (a : Fin n) :
    a ∈ (p.actual W).A.reads g ↔ a ∈ W.uses p.value g ∧ a < g := by
  simp only [region, mem_filter] at hg
  simp only [hg.1, hg.2.1, hg.2.2.1, not_true_eq_false, or_self, ite_false, mem_filter]

/-- A gate of a region reads exactly the operands it uses, in a topological wiring the execution follows. -/
theorem Placement.reads_actual (hW : W.Topological) (hp : p.Follows W) {X : Finset L} {I : Set ℕ} {g : Fin n}
    (hg : g ∈ region W p X I) (a : Fin n) : a ∈ (p.actual W).A.reads g ↔ a ∈ W.uses p.value g := by
  have hran : g ∈ p.ran := by
    simp only [region, mem_filter] at hg
    exact hg.1
  rw [p.reads_of_region W hg a]
  exact ⟨fun h => h.1, fun h => ⟨h, hW g a ((hp.2 g hran).1 h)⟩⟩

/-- The actual run's wires into a region are among the operands `W`'s gates use, with no hypothesis. -/
theorem Placement.mem_into_of_actual {X : Finset L} {I : Set ℕ} (a : Fin n) :
    a ∈ (p.actual W).A.into ((p.actual W).A.region X I) → a ∈ into W p (region W p X I) := by
  rw [p.region_actual W X I]
  simp only [Verity.Circuit.into, into, mem_filter, mem_univ, true_and, mem_sdiff, mem_biUnion]
  rintro ⟨haR, g, hg, ha⟩
  exact ⟨⟨g, hg, ((p.reads_of_region W hg a).mp ha).1⟩, haR⟩

theorem Placement.mem_into_actual (hW : W.Topological) (hp : p.Follows W) {X : Finset L} {I : Set ℕ} (a : Fin n) :
    a ∈ (p.actual W).A.into ((p.actual W).A.region X I) ↔ a ∈ into W p (region W p X I) := by
  rw [p.region_actual W X I]
  simp only [Verity.Circuit.into, into, mem_filter, mem_univ, true_and, mem_sdiff, mem_biUnion]
  constructor
  · rintro ⟨haR, g, hg, ha⟩
    exact ⟨⟨g, hg, (p.reads_actual W hW hp hg a).mp ha⟩, haR⟩
  · rintro ⟨⟨g, hg, ha⟩, haR⟩
    exact ⟨haR, g, hg, (p.reads_actual W hW hp hg a).mpr ha⟩

/-- The actual run's ingress is `W`'s, in a topological wiring the execution follows. -/
theorem Placement.ingress_actual (hW : W.Topological) (hp : p.Follows W) (X : Finset L) (I : Set ℕ) :
    (p.actual W).A.ingress X I = ingress W p X I := by
  unfold Placed.ingress ingress Verity.Circuit.width bits
  refine sum_congr (Finset.ext fun a => ?_) fun _ _ => rfl
  simp only [mem_filter, p.mem_into_actual W hW hp a]

open Classical in
/-- The actual run's ingress is at most `W`'s, with no hypothesis. -/
theorem Placement.ingress_le_actual (X : Finset L) (I : Set ℕ) : (p.actual W).A.ingress X I ≤ ingress W p X I := by
  unfold Placed.ingress ingress Verity.Circuit.width bits
  refine Finset.sum_le_sum_of_subset ?_
  rw [Finset.subset_iff]
  intro a ha
  simp only [mem_filter] at ha ⊢
  exact ⟨p.mem_into_of_actual W a ha.1, ha.2⟩

end Accounting.Requirement

namespace Accounting.SecurityProofs

open Accounting.Requirement

theorem RequirementCredits : Accounting.Guarantees.RequirementCredits := by
  intro Ω L n W p P X π m r
  simp only [PouwWork.credits, Placed.work, work]
  rw [Placement.region_actual]

theorem RequirementHelp : Accounting.Guarantees.RequirementHelp := by
  intro Ω L n W p P X r hW hp
  exact Placement.ingress_actual W (p r) hW hp _ _

theorem RequirementHelpBound : Accounting.Guarantees.RequirementHelpBound := by
  intro Ω L n W p P X r
  exact Placement.ingress_le_actual W (p r) _ _

theorem RequirementStepsInner : Accounting.Guarantees.RequirementStepsInner := by
  intro G _ L W p X w ts td hp hX
  rw [Finset.subset_iff]
  intro g hg
  simp only [stepRegion, region, Finset.mem_filter] at hg ⊢
  simp only [Window.steps, Finset.mem_Ico, Set.mem_Ico, Placement.tick] at hg ⊢
  obtain ⟨hran, hreg, hin, hx, h1, h2⟩ := hg
  exact ⟨hran, hreg, hin, hx, (hX _ hx).1.trans ((hp _ hx).monotone h1),
    lt_of_lt_of_le ((hp _ hx).lt_iff_lt.mpr h2) (hX _ hx).2⟩

theorem RequirementStepsOuter : Accounting.Guarantees.RequirementStepsOuter := by
  intro G _ L W p X w ts td hp hX
  rw [Finset.subset_iff]
  intro g hg
  simp only [stepRegion, region, Finset.mem_filter] at hg ⊢
  simp only [Window.steps, Finset.mem_Ico, Set.mem_Ico, Placement.tick] at hg ⊢
  obtain ⟨hran, hreg, hin, hx, h1, h2⟩ := hg
  exact ⟨hran, hreg, hin, hx, (hp _ hx).le_iff_le.mp ((hX _ hx).1.trans h1),
    (hp _ hx).lt_iff_lt.mp (lt_of_lt_of_le h2 (hX _ hx).2)⟩

theorem RequirementCreditsSteps : Accounting.Guarantees.RequirementCreditsSteps := by
  intro Ω L n W p P X π m r w hp hX
  rw [RequirementCredits Ω L n W p P X π m r]
  unfold work
  exact Finset.sum_le_sum_of_subset
    (Finset.sdiff_subset_sdiff (RequirementStepsInner W (p r) X w _ _ hp hX) (subset_refl _))

end Accounting.SecurityProofs
