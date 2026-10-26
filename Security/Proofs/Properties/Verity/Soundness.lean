import Properties.Verity.Soundness
import Proofs.Properties.Composition
import Proofs.Accounting.JointPeriod
import Proofs.Accounting.PouwWork

/-!
# Soundness at the verifier of record, from its subprotocols' targets

Onsite's acceptance is every component's (`accepts_components`, from `Properties.Composition`'s `answerAccepts` at
the product and `acceptMeans`). Outside the break, each component accepted on the coins the device logged for it, so
the event `Verity.Soundness` bounds lies inside `JointPeriod`'s for the run at the committed transcript, whose claims
are the same.
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

theorem soundness_of_subprotocols : Soundness := by
  intro ι α I _ _ _ Ω _ _ d V hS hW hP hB hN τ
  obtain ⟨X, hX⟩ := hS τ
  refine ⟨X, le_trans (Accounting.SecurityProofs.PouwWork.pr_mono ?_)
    (Accounting.SecurityProofs.JointPeriod ι α I Ω d (V.circuitOf τ.params) (V.run τ (X τ.commitment))
      (V.certOf τ.params) (V.priceOf τ.params) (V.delta τ.params) ?_)⟩
  · rintro ξ ⟨hacc, ⟨a, hfail⟩, hbrk⟩
    refine ⟨a, ?_, hfail⟩
    show ComponentAccepts a (τ.sent ξ) ∧ V.coinsRead a (τ.sent ξ) = some (ξ (V.own a))
    refine ⟨accepts_components hacc a, ?_⟩
    by_contra hne
    exact hbrk ⟨a, accepts_components hacc a, hne⟩
  · intro a
    cases a with
    | sampled => exact hX
    | pouw => exact hW τ _
    | pous => exact hP τ _
    | poub => exact hB τ _
    | network => exact hN τ _

end Verity
