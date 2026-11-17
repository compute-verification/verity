module

public import Definitions.ResourceAccounting.Actual
public import Definitions.Registration.Physical
public import Definitions.Slowdown.Assumptions.Fit

/-! The fit bound on the windows a physical run's cuts make, its reads counted by width. -/

@[expose] public section

namespace Slowdown.Guarantees

open Verity ResourceAccounting Finset

/-- **The fit bound on a run**. For an actual circuit `A` meeting causality and links at the developer's pairwise
disjoint servers `X i`, cuts `κ` of `[0, t)` into windows of at most `ω` ticks, and a reference training algorithm `c` contained
(`M`) in the developer's declaration over `[0, t)` and threaded by `T`: when each server's state at each cut, in the
image of `c`'s novel gates, is at most `R` bits wide, the capacity of the links into each server is at most `r`, and
`S = R + r ω < N* b_w`, then under F3 (`Assumptions.Reach`) and F1 with F4 (`Assumptions.Fits`) over the windows pulled
into `c` (any division of `c` whose unit `(i, j)` is the gates `M` sends into the window `A[X i, [τ j, τ (j + 1))]`),
the counted multiplications number at most `n J · reuse · N* S² / (4 w_e (N* b_w − S))`; and when every window is at
least `ω₀` ticks long, at most `reuse · N* S² / (4 w_e (N* b_w − S) ω₀)` per server-tick, both written multiplied out.

The state is counted by width, and at the first cut, `0`, it is what each server holds when the run starts. A wipe
bounds information, so a wipe's residual is an `R` here only through the read bound's determination form. -/
def FitRate : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t : ℕ) (κ : Cuts t) (ω : ℕ) (R r : ℝ)
    (c : RegisteredCircuit) (M : Containment c.D (A.declaration (univ.biUnion X) t θ hθ).D) (T : Threading c)
    (start : Fin T.m → Fin (n * κ.J)) (Nstar : ℕ) (b_w w_e : ℝ),
    (∀ i j, i ≠ j → Disjoint (X i) (X j)) → A.Causal ℓ → A.Links ends β link →
    (∀ j : Fin κ.J, κ.τ j.succ - κ.τ j.castSucc ≤ ω) → (∀ i, (linkCapacity ends β (X i) : ℝ) ≤ r) →
    (∀ (i : Fin n) (j : Fin κ.J),
      (A.stateIn (X i) (κ.τ j.castSucc) ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) : ℝ) ≤ R) →
    R + r * ω < Nstar * b_w → Assumptions.Reach c T Nstar →
    (∀ Q : Partition c.D (n * κ.J), (∀ (i : Fin n) (j : Fin κ.J) (g : Fin c.D.N),
        g ∈ Q.gates (finProdFinEquiv (i, j)) ↔ M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))) →
      Assumptions.Fits c T Q start b_w w_e) →
      4 * w_e * (Nstar * b_w - (R + r * ω)) * T.M.card ≤ (n * κ.J : ℕ) * T.reuse * Nstar * (R + r * ω) ^ 2 ∧
      ∀ ω₀ : ℕ, (∀ j : Fin κ.J, ω₀ ≤ κ.τ j.succ - κ.τ j.castSucc) →
        ω₀ * (4 * w_e * (Nstar * b_w - (R + r * ω)) * T.M.card) ≤ n * t * T.reuse * Nstar * (R + r * ω) ^ 2

end Slowdown.Guarantees
