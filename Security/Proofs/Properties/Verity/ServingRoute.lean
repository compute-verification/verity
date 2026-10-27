import Properties.Verity.ServingRoute
import Proofs.Accounting.ServingRoute
import Proofs.Properties.Verity.Soundness

/-!
# The serving route at the verifier of record: proof

`Verity.Soundness` at the five targets bounds the probability that onsite's compiled verifier accepts registered files
while a claim fails; that is `ServingEndToEnd`'s first hypothesis at the acceptance `Accepts ∧ Registered` and the claims
`Accepts ∧ Registered ∧ ∀ a, Holds a`, whose difference is exactly `Verity.Soundness`'s event.

Beside it, the two checks in `soundness_of_subprotocols`' form, over the vocabulary as given:
`OnsiteServingEndToEnd_honest`, at the honest deployment (registered files, each subprotocol's honest lemma
`Vocab.HonestHolds`) `Verity.Soundness` holds and the subprotocols' term is `0`, so the bound at the honest prover is the
bridge's and coverage's alone; and `OnsiteServingEndToEnd_nontrivial`, the conclusion fails for a verifier that accepts
everything, at registered files and a strategy whose every play succeeds on fewer server-windows than `windowsMin N`
with the bound below one.
-/

namespace Verity.SecurityProofs

open Accounting Accounting.ServingRoute
open scoped ENNReal

theorem OnsiteServingEndToEnd : Verity.OnsiteServingEndToEnd := by
  intro ι α _ d V L σ E _ P θ F c play success N δb η hS h1 h2 h3 h4 h5 hcS hqS hbr hcov s
  have hV := hS ι α d V h1 h2 h3 h4 h5 θ F c
  have hL2 : ∀ s, Game.prob (fun o => (Accepts o.sent ∧ V.Registered o.params o.file) ∧
      ¬ (Accepts o.sent ∧ V.Registered o.params o.file ∧ ∀ a, V.Holds θ F c s a o)) (V.game θ F c) s ≤
      ∑ a, V.error a θ F c s := fun s =>
    (Game.prob_mono (fun o ⟨⟨hA, hR⟩, hn⟩ => ⟨hA, hR, not_forall.mp fun hall => hn ⟨hA, hR, hall⟩⟩) _ _).trans
      (hV s)
  exact (Game.prob_mono (fun o ⟨hA, hR, hs, hw⟩ => ⟨⟨hA, hR⟩, hs, hw⟩) _ _).trans
    (Accounting.SecurityProofs.ServingRoute.ServingEndToEnd (V.game θ F c) P play
      (fun _ o => Accepts o.sent ∧ V.Registered o.params o.file)
      (fun s o => Accepts o.sent ∧ V.Registered o.params o.file ∧ ∀ a, V.Holds θ F c s a o) success N
      (fun s => ∑ a, V.error a θ F c s) δb η hcS hqS hL2 hbr hcov s)

/-- **Its hypotheses hold at the honest deployment, and its error term vanishes there**: at registered files, where each
subprotocol's honest lemma holds (`Vocab.HonestHolds`), `Verity.Soundness` holds (`soundness_of_subprotocols`), and the
deployment's honest prover accepts with a failed claim with probability `0`, so under the serving bridge and coverage
the probability that it is accepted and succeeds on fewer server-windows than `windowsMin N` is at most `δ_b + η`. -/
theorem OnsiteServingEndToEnd_honest {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α) {L σ E : Type}
    [Fintype E] (h : ∀ a, V.HonestHolds a) (P : Params L σ E) (θ F c : ByteArray) (hreg : V.Registered θ F)
    (play : Game.Strategy (V.game θ F c) → Outcome → Play L)
    (success : Game.Strategy (V.game θ F c) → Outcome → Prop) (N : ℝ) (δb η : Game.Strategy (V.game θ F c) → ℝ≥0∞)
    (hcS : ∀ s, P.hw.compute (P.X s) * P.ω ≤ P.cS) (hqS : 0 < P.qS)
    (hbr : Assumptions.ServingBridge (V.game θ F c) P play
      (fun s o => Accepts o.sent ∧ V.Registered o.params o.file ∧ ∀ a, V.Holds θ F c s a o) δb)
    (hcov : Assumptions.ServingCoverage (V.game θ F c) P play success N η) :
    Soundness ∧ Game.prob (fun o => Accepts o.sent ∧ V.Registered o.params o.file ∧ success (V.honest θ F c) o ∧
        ((play (V.honest θ F c) o).windows : ℝ) < P.windowsMin N) (V.game θ F c) (V.honest θ F c) ≤
      δb (V.honest θ F c) + η (V.honest θ F c) := by
  refine ⟨soundness_of_subprotocols, ?_⟩
  have key := Accounting.SecurityProofs.ServingRoute.ServingEndToEnd (V.game θ F c) P play
    (fun _ o => Accepts o.sent ∧ V.Registered o.params o.file)
    (fun s o => Accepts o.sent ∧ V.Registered o.params o.file ∧ ∀ a, V.Holds θ F c s a o) success N _
    δb η hcS hqS (fun _ => le_rfl) hbr hcov (V.honest θ F c)
  have h0 : Game.prob (fun o => (Accepts o.sent ∧ V.Registered o.params o.file) ∧
      ¬ (Accepts o.sent ∧ V.Registered o.params o.file ∧ ∀ a, V.Holds θ F c (V.honest θ F c) a o))
      (V.game θ F c) (V.honest θ F c) = 0 :=
    le_antisymm ((Game.prob_mono (fun o ⟨⟨hA, hR⟩, hn⟩ =>
      ⟨hA, hR, not_forall.mp fun hall => hn ⟨hA, hR, hall⟩⟩) _ _).trans
      (soundness_of_subprotocols_honest V h θ F c hreg).2.le) zero_le
  beta_reduce at key
  rw [h0, zero_add] at key
  exact (Game.prob_mono (fun o ⟨hA, hR, hs, hw⟩ => ⟨⟨hA, hR⟩, hs, hw⟩) _ _).trans key

/-- **Its conclusion fails for a verifier that accepts everything**: at files that pass the commit phase's check, a
strategy on whose every play the computation succeeds on fewer server-windows than `windowsMin N`, and the
subprotocols' errors, `δ_b` and `η` summing below one there. -/
theorem OnsiteServingEndToEnd_nontrivial {ι α : Type} [Fintype α] {d : Device ι} (V : Vocab d α) {L σ E : Type}
    [Fintype E] (P : Params L σ E) (θ F c : ByteArray)
    (play : Game.Strategy (V.game θ F c) → Outcome → Play L)
    (success : Game.Strategy (V.game θ F c) → Outcome → Prop) (N : ℝ) (δb η : Game.Strategy (V.game θ F c) → ℝ≥0∞)
    (s : Game.Strategy (V.game θ F c)) (hreg : V.Registered θ F)
    (hwin : ∀ o, success s o ∧ ((play s o).windows : ℝ) < P.windowsMin N)
    (hlt : ∑ a, V.error a θ F c s + δb s + η s < 1) :
    ¬ ∀ s, Game.prob (fun o => (fun _ => True) o.sent ∧ V.Registered o.params o.file ∧ success s o ∧
        ((play s o).windows : ℝ) < P.windowsMin N) (V.game θ F c) s ≤ ∑ a, V.error a θ F c s + δb s + η s := by
  intro hb
  have h1 := prob_play_eq_one (V.rounds θ) θ F c
    (fun o => (fun _ => True) o.sent ∧ V.Registered o.params o.file ∧ success s o ∧
      ((play s o).windows : ℝ) < P.windowsMin N) (fun _ _ => ⟨trivial, hreg, hwin _⟩) s
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le (hb s)).trans_lt hlt)

end Verity.SecurityProofs
