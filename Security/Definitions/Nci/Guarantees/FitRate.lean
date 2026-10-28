import Definitions.Accounting.Actual
import Definitions.Accounting.Physical
import Definitions.Nci.Assumptions.Fit

/-!
# NCI over windows: the fit bound on a physical run

`FitBound` on the windows a run's cuts make. The servers `X i` of an actual circuit `A` (`Accounting.Placed`), cut at
`0 = τ 0 ≤ … ≤ τ J = t` (`Accounting.Cuts`), divide the developer's region over `[0, t)` into `n J` windows
`A[X i, [τ j, τ (j + 1))]`; through a containment of the reference circuit `c` in the developer's declaration, those
windows divide `c`. Each window reads at most its server's state at the cut that opens it plus what its links carry, so
when that state, in the image of `c`'s novel gates, is at most `R` bits wide, the capacity of the links into each server
at most `r` and each window at most `ω` ticks long, every window reads at most `S = R + r ω`, and `FitBound` applies
with `k = n J`. Divided by the run's `n t` server-ticks, it bounds the counted multiplications a server does a tick.

Reads are counted by width (`Placed.stateIn`, `novelReads`), as in `Accounting.Guarantees.Actual.ThreadedRuntime`: a wipe
bounds information, so plugging a wipe's residual in as `R` needs the read bound in its determination form. `r` bounds
the links' capacity (`linkCapacity`), not the intake a run measures.
-/

namespace Nci.Guarantees

open Verity Accounting Finset

/-- **The fit bound on a run**. For an actual circuit `A` meeting causality and links at the developer's pairwise
disjoint servers `X i`, cuts `κ` of `[0, t)` into windows of at most `ω` ticks, and a reference circuit `c` contained
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

end Nci.Guarantees
