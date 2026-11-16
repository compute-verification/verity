module

public import Proofs.Slowdown.Threads

/-!
# Theorem 1 over a read measure

`threaded_reads_by`: Theorem 1 (`threaded_reads`) with reads counted by any measure `ρ`. Its proof reads the count
twice, through `ρ R ≤ S` and through `ρ R` paying for the weights used and the thread values read, so it goes through for
any `ρ`; `threaded_count` (`Proofs.Slowdown.Threads`) is the counting. Theorem 1 is the instance `ρ = freshInput`.
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.SecurityProofs

open Verity Finset

/-- **Theorem 1 over a read measure** `ρ`: for a `(d, e)`-threaded `c` whose small subcircuits pay for the weights they
use and the thread values they read, both read by `ρ` (small: at most `Γ` computed gates and `ρ R ≤ S`), a small
subcircuit `R` holds at most `ρ R Λ / b_a` counted multiplications, `Λ = T.bound S b_w d e`. -/
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
