import Definitions.Accounting.CostModel

/-! The proof-only part of `Definitions.Accounting.CostModel`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting

/-- **An adversary's timed run** in `CM_A` on the workload `U`: the salt is released at `tₛ`, the transcript is
received at `t_d` (it is what the verifier receives before then), and by each time `t` the prover has spent the cost
window's work placed before `t`, its work at `π` in `[tₛ + ℓ, t_d − ℓ)` before `t`. -/
noncomputable def Actual.timedRun {Q R S : Type} {L : Type} [LinearOrder L] [Fintype L] {𝓕 : Set (Actual.Run L)}
    {V : Finset L} {tₛ t_d ℓ : ℕ} {read : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) → Pouw.Transcript}
    {hash : (ρ : Actual.Run L) → Finset (Fin ρ.A.N)} {π : (ρ : Actual.Run L) → Fin ρ.A.N → ℕ}
    (A : Pouw.Adversary (Actual.costModel Q R S 𝓕 V tₛ t_d ℓ read hash π)) (U : Pouw.Workload) :
    Pouw.TimedRun Q R S where
  ts := tₛ
  tr _ _ := t_d
  spent H s t :=
    let P : Actual.Prover Q R S 𝓕 V tₛ t_d ℓ read hash := A.online
    ((P.strategy (H, U, A.act, A.pre.eval H, s)).workIn (π _) V
      ({u | (u : ℝ) < t} ∩ Set.Ico (tₛ + ℓ) (t_d - ℓ)) : ℝ)

end Accounting
