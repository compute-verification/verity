import Definitions.Pous.Game.Prob

/-!
# A PoUS verdict proved through the proof service (trusted)

Under the proof service, a PoUS audit publishes its verdict and nothing about a single round (`PROTOCOL.md`, "Public
outputs"). The memory challenger signs what it saw, which is witness, and the prover proves the verdict of it through
the proof service. So the auditor does not run the clear verdict (`contAccepts` on the challenger's records); it runs
the proof service's verifier on a claim of that verdict, over the verifier's own coins, after the audit's.

`ProofSound` is what the auditor needs of that verifier, in the shape the joint audit takes each audit's guarantee
(`Accounting.Run.ComputeSound`): on every outcome of the audit, the verifier accepts a claim of the verdict that the
clear verdict refutes with probability at most `ε` over its coins. Sampling's knowledge soundness at the deployment
whose statements are these claims (`Claims.Sampling.KS`, `Definitions/Claims/Sampling.lean`) is to give it; that bridge
is open.
-/

namespace Pous

open scoped ENNReal

/-- **The proof layer is sound at `ε`** for the clear verdict `clear` on the audit's outcomes `X`: on every outcome
`x`, the proof service's verifier, over its coins `Ω`, accepts (`acc x`) while `clear x` fails with probability at
most `ε`. -/
def ProofSound {X Ω : Type} [Fintype Ω] (acc : X → Ω → Prop) (clear : X → Prop) (ε : ℝ≥0∞) : Prop :=
  ∀ x, pr (fun c => acc x c ∧ ¬ clear x) ≤ ε

end Pous
