import Models.Accounting.Guarantees.RequirementPouw

/-!
# PoUW's credits and help over a declared circuit's placed execution

The actual run of a placed execution has `W`'s regions (`Placement.region_actual`): its computed gates are the values the
execution produces but the registered ones, each at its device and its step's tick. In a topological wiring the
execution follows, a gate reads exactly the operands it uses, so the actual run's wires into a region are `W`'s and its
ingress is `W`'s (`Placement.ingress_actual`). PoUW's credits and help are `Placed.work` and `Placed.ingress` of the run
at each coin, so `RequirementCredits` and `RequirementHelp` follow. `RequirementSteps` orders a device's steps as their
ticks.
-/

namespace Accounting.Requirement

open Finset

variable {n : ℕ} {L : Type} (W : Wiring (Fin n)) (p : Placement L (Fin n))

/-- A gate is computed in the actual run when the execution produces it and it isn't registered. -/
theorem Placement.computed_actual (g : Fin n) :
    (p.actual W).A.isInput g = false ↔ g ∈ p.ran ∧ g ∉ W.registered := by
  simp only [decide_eq_false_iff_not, not_or, not_not]

theorem Placement.mem_region_actual {X : Finset L} {I : Set ℕ} (g : Fin n) :
    g ∈ (p.actual W).A.region X I ↔ g ∈ region W p X I := by
  simp only [Placed.region, Verity.Circuit.computed, region, mem_filter, mem_univ, true_and, p.computed_actual W,
    and_assoc]

/-- The actual run's region is `W`'s. -/
theorem Placement.region_actual (X : Finset L) (I : Set ℕ) : (p.actual W).A.region X I = region W p X I :=
  Finset.ext fun g => p.mem_region_actual W g

/-- A gate of a region reads exactly the operands it uses, in a topological wiring the execution follows. -/
theorem Placement.reads_actual (hW : W.Topological) (hp : p.Follows W) {X : Finset L} {I : Set ℕ} {g : Fin n}
    (hg : g ∈ region W p X I) (a : Fin n) : a ∈ (p.actual W).A.reads g ↔ a ∈ W.uses p.value g := by
  simp only [region, mem_filter] at hg
  simp only [hg.1, hg.2.1, not_true_eq_false, or_self, ite_false, mem_filter]
  exact ⟨fun h => h.1, fun h => ⟨h, hW g a ((hp.2 g hg.1).1 h)⟩⟩

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

theorem RequirementSteps : Accounting.Guarantees.RequirementSteps := by
  intro G _ L W p X w ts td hp hX
  ext g
  simp only [region, stepRegion, Finset.mem_filter]
  simp only [Placement.tick, Set.mem_Ico, Window.ticks, Finset.mem_Ico]
  refine and_congr_right fun _ => and_congr_right fun _ => ?_
  refine ⟨fun ⟨hx, h1, h2⟩ => ⟨hx, ?_, ?_⟩, fun ⟨hx, h1, h2⟩ => ⟨hx, ?_, ?_⟩⟩
  · rw [← (hX _ hx).1] at h1
    exact (hp _).le_iff_le.mp h1
  · rw [← (hX _ hx).2] at h2
    exact (hp _).lt_iff_lt.mp h2
  · rw [← (hX _ hx).1]
    exact (hp _).le_iff_le.mpr h1
  · rw [← (hX _ hx).2]
    exact (hp _).lt_iff_lt.mpr h2

end Accounting.SecurityProofs
