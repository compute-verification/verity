module

public import Definitions.Pouw.Guarantees.Game

/-!
# The game's guarantees: lemmas

The statements of `Definitions.Pouw.Guarantees.Game` that are steps of its guarantees' proofs rather than guarantees.
Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same name, and
audited with the proofs: no record pins it.
-/

@[expose] public section

namespace Pouw.Guarantees

open Pouw.GameStatements Finset

/-- **(b) Theorem 1** (§3.5). If `G_g` holds with error `δ` and, for an adversary within `(T, q)`, the audit (any
coins `C` sampled independently of `H` and `s`, any acceptance predicate) satisfies
`Pr[accept ∧ ∑_{u ∉ K} W_ref(u) > ε_s·W] ≤ δ_s`, then
`Pr[accept ∧ T < (1 − g)(1 − ε_s)·W ∧ the run doesn't break SHA-512] ≤ δ_s + δ`. -/
def Theorem1 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (g : ℝ) (δ : ℕ → ℕ → ℝ),
    g < 1 → GGap CM P D g δ →
    ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q →
    ∀ (C : Type) [Fintype C] (accept : (Q → R) → S → C → Prop) (εs δs : ℝ),
      pr (fun x : (Q → R) × S × C => accept x.1 x.2.1 x.2.2 ∧
          εs * totalWref P U.layout <
            ∑ u ∈ range U.layout.N \ correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1), (P.Wref U.layout u : ℝ))
        ≤ δs →
      pr (fun x : (Q → R) × S × C => accept x.1 x.2.1 x.2.2 ∧ T < (1 - g) * (1 - εs) * totalWref P U.layout ∧
          ¬ A.Breaks U x.1 x.2.1)
        ≤ δs + δ q U.layout.N

/-- **(c) Work-weighted sampling.** `t` independent samples, each unit sampled with probability `∝ w(u)`; the audit
accepts iff every sample is in `K` (which may depend on the rest of the randomness `α`). Then
`Pr[accept ∧ ∑_{u ∉ K} w(u) > ε_s·∑_u w(u)] ≤ (1 − ε_s)^t` for `ε_s ≤ 1`. -/
def WorkWeightedSampling : Prop :=
  ∀ (α : Type) [Fintype α] (w : ℕ → ℕ) (N t : ℕ) (K : α → Finset ℕ) (εs : ℝ), εs ≤ 1 →
    pr (fun x : α × (Fin t → Ticket w N) => AuditAccepts (K x.1) x.2 ∧
        εs * ∑ u ∈ range N, (w u : ℝ) < ∑ u ∈ range N \ K x.1, (w u : ℝ)) ≤ (1 - εs) ^ t

end Pouw.Guarantees
