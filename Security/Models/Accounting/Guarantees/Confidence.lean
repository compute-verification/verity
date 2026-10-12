import Definitions.Accounting.Confidence
import Mathlib.Data.Fintype.Pi

/-!
# Composition and fit (trusted)

Sections 4 and 5 of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.2).

* `Composition`: `k` protocols in one run, with independent coins (protocol `i`'s coins reach nothing but the input
  gates at its verifier's leaves `V i`) and `E i` sound at `δ i` alone, accept together with the run outside the
  confidence set with probability at most `∑ δ i`. The feasible set is a set of runs, so it doesn't depend on which
  party runs a gate.
* `Fit`: when `μ` adds over the confidence set and some run of it implements `P`, `req_μ(P)` is at most the residual
  `min(cap, obs) − cert`, given that every run of the set uses at most `cap` and at most `obs`.

The proofs are in `Security/Proofs/Accounting/Actual/Confidence.lean`.
-/

namespace Accounting.Guarantees

open Finset

open Classical in
/-- **Composition.** For `k` protocols with coins `Ω i`, verifiers at `V i` accepting on `acc i` and guarantees `E i`
sound at `δ i` against `𝓕`, and every admissible joint strategy `S` with runs in `𝓕` in which the input gates off
`V i` depend on no coin of protocol `i`: all accept and `S r ∉ 𝓒(v)` (`v` the view at `W`) for at most a `∑ δ i`
fraction of the coins. -/
def Composition : Prop :=
  ∀ {L : Type*} {k : ℕ} (Ω : Fin k → Type) [∀ i, Fintype (Ω i)] [∀ i, DecidableEq (Ω i)] (𝓕 : Set (Actual.Run L))
    (V : Fin k → Finset L) (W : Finset L) (acc : Fin k → Set (ℕ × L × ℕ × ℕ) → Prop) (E : Fin k → Actual.Run L → Prop)
    (δ : Fin k → ℝ) (S : ((i : Fin k) → Ω i) → Actual.Run L),
    (∀ i, Sound (Ω i) 𝓕 (V i) (acc i) (E i) (δ i)) → (∀ r, S r ∈ 𝓕) → Admissible S →
    (∀ i r r', (∀ j, j ≠ i → r j = r' j) → (S r).inputs (↑(V i))ᶜ = (S r').inputs (↑(V i))ᶜ) →
    ((univ.filter fun r => (∀ i, acc i ((S r).view (V i))) ∧ S r ∉ confidence 𝓕 W E ((S r).view W)).card : ℝ) ≤
      (∑ i, δ i) * Fintype.card ((i : Fin k) → Ω i)

/-- **Fit.** When `μ` adds over `𝓒` for `P` above `cert`, some run of `𝓒` implements `P`, and every run of `𝓒` has
`μ ≤ cap` and `μ ≤ obs`: `req_μ(P) ≤ min(cap, obs) − cert`. -/
def Fit : Prop :=
  ∀ {L : Type*} (𝓕 𝓒 : Set (Actual.Run L)) (μ : Actual.Run L → ℕ) (impl : Actual.Run L → Prop) (cert cap obs : ℕ),
    Additive 𝓒 μ cert (req μ 𝓕 impl) impl → (∃ R ∈ 𝓒, impl R) → (∀ R ∈ 𝓒, μ R ≤ cap ∧ μ R ≤ obs) →
    req μ 𝓕 impl ≤ ((min cap obs - cert : ℕ) : ℕ∞)

end Accounting.Guarantees
