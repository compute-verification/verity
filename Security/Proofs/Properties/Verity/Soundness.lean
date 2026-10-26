import Properties.Verity.Soundness
import Proofs.Properties.Composition
import Proofs.Accounting.JointPeriod
import Proofs.Accounting.PouwWork

/-!
# Soundness at the verifier of record, from its legs

Onsite's acceptance is every component's (`accepts_components`, from `Properties.Composition`'s `answerAccepts` at
the product and `acceptMeans`), so the event `Verity.Soundness` bounds lies inside `JointHorizon`'s for the compiled
runs, whose claims are the same.
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

theorem soundness_of_legs : Soundness := by
  intro ι α I _ _ _ Ω _ _ d C m r s c p δ h
  refine le_trans (Accounting.SecurityProofs.PouwWork.pr_mono ?_)
    (Accounting.SecurityProofs.JointHorizon ι α I Ω d C m (fun j => compiled (r j) (s j)) c p δ h)
  rintro ξ ⟨j, hacc, a, hfail⟩
  exact ⟨j, a, accepts_components hacc a, hfail⟩

end Verity
