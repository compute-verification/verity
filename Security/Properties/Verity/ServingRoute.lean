import Properties.Verity.Soundness
import Definitions.Accounting.Assumptions.ServingRoute

/-!
# The serving route at the verifier of record, end to end

The end-to-end overhead guarantee's serving-block route (note:training-20261027T0134Z-draft-e2e-overhead-proof §9) in
the one experiment `Verity.Soundness` is stated in: the causal game after the parameters `θ`, the committed files `F`
and the commitment `c` (`Vocab.game`), onsite's compiled verifier's acceptance of files that passed the commit check
(`Accepts`, `Vocab.Registered`), and the claims `Vocab.Holds`, read as they are (one vocabulary). The play's actual
circuit (`Accounting.ServingRoute.Play`) is read off each strategy and run.

`OnsiteServingEndToEnd` composes `Verity.Soundness` (Layer 2, a hypothesis until the five targets are proved, pinned at
#4613's f76306ccd4) with `Accounting.Guarantees.ServingRoute.ServingEndToEnd` (Layer 3 on the serving route, the serving
bridge and coverage). The bound is the subprotocols' errors at the strategy (each its published `δ` plus its collision
term at its finder's SHA-512 collisions, `Vocab.error`), the bridge's `δ_b` and coverage's `η`.
-/

namespace Verity

open Accounting Accounting.ServingRoute
open scoped ENNReal

/-- **The serving route at the verifier of record.** On any device, for any names `V` the subprotocols define, with
`Verity.Soundness` and the five subprotocol targets: at all parameters `θ`, committed files `F` and commitments `c`,
for the serving route's parameters `P` (each server's compute over a window within the serving budget, `q_S > 0`) and
the play `play s o` read off each strategy and run, if the claims (`Vocab.Holds` at every subprotocol, with onsite's
acceptance and the commit check) put the play in the serving class except with probability `δ_b`
(`Accounting.Assumptions.ServingBridge`) and success in the class with counted weight below `N` has probability at most
`η` (`Accounting.Assumptions.ServingCoverage`), then for every strategy the probability that onsite's compiled verifier
accepts files that passed the commit check and the computation succeeds on fewer server-windows than
`windowsMin N = (N − n_dark)/(q(c_S, k_S) + μ k_S)` is at most the subprotocols' errors summed plus `δ_b` plus `η`. -/
def OnsiteServingEndToEnd : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) {L σ E : Type} [Fintype σ] [Fintype E]
    (P : Params L σ E) (θ F c : ByteArray) (play : Game.Strategy (V.game θ F c) → Outcome → Play L σ)
    (success : Game.Strategy (V.game θ F c) → Outcome → Prop) (N : ℝ) (δb η : Game.Strategy (V.game θ F c) → ℝ≥0∞),
    Soundness →
    V.SampledProofsSound → V.PouwSound → V.PousSound → V.BandwidthAuditSound → V.NetworkCertifierSound →
    (∀ s, P.hw.compute (P.X s) * P.ω ≤ P.cS) → 0 < P.qS →
    Assumptions.ServingBridge (V.game θ F c) P play
      (fun s o => Accepts o.sent ∧ V.Registered o.params o.file ∧ ∀ a, V.Holds θ F c s a o) δb →
    Assumptions.ServingCoverage (V.game θ F c) P play success N η →
    ∀ s, Game.prob (fun o => Accepts o.sent ∧ V.Registered o.params o.file ∧ success s o ∧
        ((play s o).windows : ℝ) < P.windowsMin N) (V.game θ F c) s ≤
      ∑ a, V.error a θ F c s + δb s + η s

end Verity
