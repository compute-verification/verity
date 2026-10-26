import Properties.Verity.Soundness
import Proofs.Properties.Composition
import Proofs.Core.Game.Prob
import Proofs.Properties.Verity.Soundness.Legs

/-!
# Soundness at the verifier of record, from its subprotocols' targets

Onsite's acceptance is every component's (`accepts_components`, from `Properties.Composition`'s `answerAccepts` at
the product and `acceptMeans`). Outside the break no component read a release missing from the device's log, so each
subprotocol's verdict accepts, and the event `Verity.Soundness` bounds lies inside the union of the subprotocols'
events, each at most its target's error (`prob_exists_univ_le`).
-/

namespace Verity

open Accounting Core

theorem component_mem (a : Audit) : component a ∈ Onsite.components := by
  cases a <;> simp [component, Onsite.components]

/-- **What the verifier of record accepts, every component accepts on its own case.** -/
theorem accepts_components {s : Sent} (h : Accepts s) (a : Audit) : ComponentAccepts a s := by
  obtain ⟨ps, c, ev, hp, hc, he, pub, prof, hch⟩ := Properties.Composition.answerAccepts _ _ _ _ h
  obtain ⟨⟨_, _, hfst, _⟩, hcases, _⟩ :=
    Properties.Composition.acceptMeans "verity/onsite/v0" Onsite.components ps c ev pub prof hch
  have hm : component a ∈ (Properties.Composition.triples Onsite.components ps ev).map (·.1) := by
    rw [hfst]
    exact component_mem a
  obtain ⟨x, hx, hx1⟩ := List.mem_map.mp hm
  exact ⟨ps, c, ev, hp, hc, he, x, hx, hx1, hcases x hx⟩

/-- The union bound over a finite family of events, for one strategy. -/
theorem prob_exists_finset_le {β κ : Type} (S : Finset κ) (p : κ → β → Prop) (g : Game β)
    (σ : Game.Strategy g) : Game.prob (fun b => ∃ i ∈ S, p i b) g σ ≤ ∑ i ∈ S, Game.prob (p i) g σ := by
  classical
  induction S using Finset.induction_on with
  | empty => simpa using (Game.prob_false g σ).le
  | insert j S hj ih =>
      rw [Finset.sum_insert hj]
      calc Game.prob (fun b => ∃ i ∈ insert j S, p i b) g σ
          ≤ Game.prob (fun b => p j b ∨ ∃ i ∈ S, p i b) g σ :=
            Game.prob_mono (fun b ⟨i, hi, hpi⟩ => by
              rcases Finset.mem_insert.1 hi with rfl | hi
              · exact Or.inl hpi
              · exact Or.inr ⟨i, hi, hpi⟩) g σ
        _ ≤ Game.prob (p j) g σ + Game.prob (fun b => ∃ i ∈ S, p i b) g σ := Game.prob_or_le _ _ g σ
        _ ≤ Game.prob (p j) g σ + ∑ i ∈ S, Game.prob (p i) g σ := add_le_add le_rfl ih

theorem soundness_of_subprotocols : Soundness := by
  intro ι α _ d V hS hW hP hB hN θ F c σ
  have hT : ∀ a, V.Target a := by
    intro a
    cases a with
    | sampled => exact hS
    | pouw => exact hW
    | pous => exact hP
    | poub => exact hB
    | network => exact hN
  calc Game.prob (fun o => Accepts o.sent ∧ V.Registered o.params o.file ∧ (∃ a, ¬ V.Holds θ F c σ a o) ∧
          ¬ V.Break θ F c o) (V.game θ F c) σ
      ≤ Game.prob (fun o => ∃ a ∈ Finset.univ, V.Verdict a o ∧ ¬ V.Holds θ F c σ a o ∧ ¬ V.Break θ F c o)
          (V.game θ F c) σ := by
        refine Game.prob_mono
          (fun o ⟨hacc, hreg, ⟨a, hfail⟩, hbrk⟩ => ⟨a, Finset.mem_univ a, ?_, hfail, hbrk⟩) _ σ
        refine ⟨hreg, accepts_components hacc a, fun r hr => ?_⟩
        by_contra hne
        exact hbrk (Or.inl ⟨a, accepts_components hacc a, r, hr, hne⟩)
    _ ≤ ∑ a, Game.prob (fun o => V.Verdict a o ∧ ¬ V.Holds θ F c σ a o ∧ ¬ V.Break θ F c o) (V.game θ F c) σ :=
        prob_exists_finset_le _ _ _ σ
    _ ≤ ∑ a, V.delta θ a := Finset.sum_le_sum fun a _ => hT a θ F c σ

end Verity
