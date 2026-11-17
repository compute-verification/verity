module

public import Definitions.Pouw.Confidence

/-! The actual-circuit model's composition and fit. -/

@[expose] public section

namespace ResourceAccounting.Guarantees

open Finset

open Classical in
/-- **Composition.** For `k` protocols with coins `Ω i`, verifiers at `V i` accepting on `acc i` and guarantees `E i`
sound at `δ i` against `𝓕`, and every joint strategy `S` causal from every tick with runs in `𝓕` in which the input
gates off `V i` depend on no coin of protocol `i`: all accept and some `E i` fails, read on protocol `i`'s slice at `r`
(`x ↦ S (update r i x)` at `r i`), for at most a `∑ δ i` fraction of the coins. -/
def Composition : Prop :=
  ∀ {L : Type*} [LinearOrder L] {k : ℕ} (Ω : Fin k → Type) [∀ i, Fintype (Ω i)] [∀ i, DecidableEq (Ω i)]
    (𝓕 : Set (Actual.Run L)) (V : Fin k → Finset L) (acc : Fin k → Set (ℕ × L × ℕ × ℕ × ℕ) → Prop)
    (E : (i : Fin k) → (Ω i → Actual.Run L) → Ω i → Prop) (δ : Fin k → ℝ) (S : ((i : Fin k) → Ω i) → Actual.Run L),
    (∀ i, Sound (Ω i) 𝓕 (V i) (acc i) (E i) (δ i)) → (∀ r, S r ∈ 𝓕) → CausalFrom S →
    (∀ i r r', (∀ j, j ≠ i → r j = r' j) → (S r).inputs (↑(V i))ᶜ = (S r').inputs (↑(V i))ᶜ) →
    ((univ.filter fun r => (∀ i, acc i ((S r).view (V i))) ∧
        ¬∀ i, E i (fun x => S (Function.update r i x)) (r i)).card : ℝ) ≤
      (∑ i, δ i) * Fintype.card ((i : Fin k) → Ω i)

/-- **Fit.** When `μ` adds over `𝓒` for `P` above `cert`, some run of `𝓒` implements `P`, and every run of `𝓒` has
`μ ≤ cap` and `μ ≤ obs`: `req_μ(P) ≤ min(cap, obs) − cert`. -/
def Fit : Prop :=
  ∀ {L : Type*} (𝓕 𝓒 : Set (Actual.Run L)) (μ : Actual.Run L → ℕ) (impl : Actual.Run L → Prop) (cert cap obs : ℕ),
    Additive 𝓒 μ cert (req μ 𝓕 impl) impl → (∃ R ∈ 𝓒, impl R) → (∀ R ∈ 𝓒, μ R ≤ cap ∧ μ R ≤ obs) →
    req μ 𝓕 impl ≤ ((min cap obs - cert : ℕ) : ℕ∞)

end ResourceAccounting.Guarantees
