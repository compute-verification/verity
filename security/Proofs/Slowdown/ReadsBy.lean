module

public import Proofs.Slowdown.Threads

/-! Theorem 1 over a read measure (`threaded_reads_by`); Theorem 1 is its instance at `ρ = freshInput`. -/

@[expose] public section

open InferenceOnly

namespace Slowdown.SecurityProofs

open Verity Finset

theorem threaded_reads_by (c : RegisteredCircuit) (T : Threading c) (ρ : Finset (Fin c.D.N) → ℝ) (Γ : ℕ) (S b_w b_a : ℝ)
    (d e : ℕ) (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d)
    (hT : ∀ R, R ⊆ c.D.computed ∧ c.D.size R ≤ Γ ∧ ρ R ≤ S → ∀ t, (T.threadReads R t).card < d →
      (T.threadMults R t).card ≤ e * (T.threadReads R t).card)
    (hW : ∀ R, R ⊆ c.D.computed ∧ c.D.size R ≤ Γ ∧ ρ R ≤ S →
      b_w * (T.weightsUsed R).card + b_a * ∑ t, (T.threadReads R t).card ≤ ρ R)
    (R : Finset (Fin c.D.N)) (hR : R ⊆ c.D.computed ∧ c.D.size R ≤ Γ ∧ ρ R ≤ S) :
    b_a * (T.mults R).card ≤ ρ R * T.bound S b_w d e :=
  threaded_count T R hbw hba hd (hT R hR) (hW R hR) hR.2.2

example : Guarantees.ThreadedReads := by
  intro c T Γ S b_w b_a d e hbw hba hd hT hW
  exact threaded_reads_by c T (fun R => (freshInput c R : ℝ)) Γ S b_w b_a d e hbw hba hd hT hW

end Slowdown.SecurityProofs
