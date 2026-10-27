import Properties.Verity.ServingRoute
import Proofs.Accounting.ServingRoute

/-!
# The serving route at the verifier of record: proof

`Verity.Soundness` at the five targets bounds the probability that onsite's compiled verifier accepts registered files
while a claim fails; that is `ServingEndToEnd`'s first hypothesis at the acceptance `Accepts ∧ Registered` and the claims
`Accepts ∧ Registered ∧ ∀ a, Holds a`, whose difference is exactly `Verity.Soundness`'s event.
-/

namespace Verity.SecurityProofs

open Accounting Accounting.ServingRoute
open scoped ENNReal

theorem OnsiteServingEndToEnd : Verity.OnsiteServingEndToEnd := by
  intro ι α _ d V L σ E _ _ P θ F c play success N δb η hS h1 h2 h3 h4 h5 hcS hqS hbr hcov s
  have hV := hS ι α d V h1 h2 h3 h4 h5 θ F c
  have hL2 : ∀ s, Game.prob (fun o => (Accepts o.sent ∧ V.Registered o.params o.file) ∧
      ¬ (Accepts o.sent ∧ V.Registered o.params o.file ∧ ∀ a, V.Holds θ F c s a o)) (V.game θ F c) s ≤
      ∑ a, V.error a θ F c s := fun s =>
    (Game.prob_mono (fun o ⟨⟨hA, hR⟩, hn⟩ => ⟨hA, hR, by
      by_contra hall
      push_neg at hall
      exact hn ⟨hA, hR, hall⟩⟩) _ _).trans (hV s)
  refine (Game.prob_mono (fun o ⟨hA, hR, hs, hw⟩ => ⟨⟨hA, hR⟩, hs, hw⟩) _ _).trans ?_
  exact Accounting.SecurityProofs.ServingRoute.ServingEndToEnd (V.game θ F c) P play
    (fun _ o => Accepts o.sent ∧ V.Registered o.params o.file)
    (fun s o => Accepts o.sent ∧ V.Registered o.params o.file ∧ ∀ a, V.Holds θ F c s a o) success N
    (fun s => ∑ a, V.error a θ F c s) δb η hcS hqS hL2 hbr hcov s

end Verity.SecurityProofs
