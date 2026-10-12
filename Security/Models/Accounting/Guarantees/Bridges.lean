import Definitions.Accounting.Confidence
import Definitions.Accounting.Localization
import Definitions.Accounting.Physical

/-!
# Obligations B1 and B4, stated (trusted)

Section 8 of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.1). Neither is proved here.

* `B1` (conjecture, TT_OUT on arbitrary circuits): PoUW's audit is sound for a lower bound on the work of `F_s(A)`. It
  is a property of an audit (`acc`, read on the verifier's view), and the conjecture is `B1` at PoUW's.
* `B4` (egress factoring, reduced to the assumptions): under causality, links and the trusted devices, with `X`'s
  certified links running the certifier, what gates outside `X` and the certifiers read during `I` depends on `X`
  during `I` only through the certifier's record and a set `Z` of at most `log₂ χ` bits a bucket plus `β°_X |I|`.
-/

namespace Accounting.Guarantees

open Finset

/-- **B1** at an audit: the verifier at `V`, accepting on `acc`, with seed gates `seed` released at `tₛ` and response
gates `resp` due at `t_d`, is sound at `(1 − ε)^K + η + ε_ks + ε_bind` against `𝓕` for the work at `π` of `F_s(A)`: the
computed gates downstream of the seed and upstream of the response placed in `[tₛ + ℓ, t_d − ℓ)`, at least
`(1 − γ)(1 − ε) W`. -/
def B1 {L : Type*} (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L) (acc : Set (ℕ × L × ℕ × ℕ) → Prop)
    (seed resp : (R : Actual.Run L) → Finset (Fin R.A.N)) (π : (R : Actual.Run L) → Fin R.A.N → ℕ) (tₛ t_d ℓ K : ℕ)
    (γ ε W η εks εbind : ℝ) : Prop :=
  open Classical in
  Sound Ω 𝓕 V acc (fun R => (1 - γ) * (1 - ε) * W ≤ R.A.work (π R) (R.A.computed.filter fun g =>
      g ∈ R.A.downstream (seed R) ∧ (∃ d ∈ resp R, Relation.ReflTransGen (fun a g => a ∈ R.A.reads g) g d) ∧
      R.A.t g ∈ Set.Ico (tₛ + ℓ) (t_d - ℓ)))
    ((1 - ε) ^ K + η + εks + εbind)

/-- **B4.** For every actual circuit under causality, links and the trusted devices (certifiers at `cert`, record
`record`), and every location `X` apart from the certifiers whose certified links run the certifier (a wire out of `X`
on a link with an end at a certifier is read there), and `I = [a, b)`: some `Z` of at most `jitter` bits a bucket of
`I` plus `β°_X (b − a)` has every gate that a gate outside `X` and `cert` reads during `I` in `record ∪ Z` or not
downstream of `A[X, I]` avoiding `record ∪ Z`. -/
def B4 : Prop :=
  ∀ {L E : Type*} [Fintype E] (A : Placed L) (ℓ : L → L → ℕ) (ends : E → L × L) (β : E → ℕ)
    (link : Fin A.N → Fin A.N → E) (𝒱 cert X : Finset L) (spec : A.Spec) (record : Finset (Fin A.N))
    (bucket : ℕ → ℕ) (jitter a b : ℕ),
    A.Causal ℓ → A.Links ends β link → A.TrustedUnits 𝒱 cert spec record bucket jitter → Disjoint X cert →
    (∀ g, ∀ s ∈ A.reads g, A.x s ∈ X → A.x g ∉ X → ((ends (link g s)).1 ∈ cert ∨ (ends (link g s)).2 ∈ cert) →
      A.x g ∈ cert) →
    ∃ Z : Finset (Fin A.N),
      A.width Z ≤ jitter * ((Finset.Ico a b).image bucket).card + uncertifiedCapacity ends β cert X * (b - a) ∧
      ∀ g, A.x g ∉ X → A.x g ∉ cert → A.t g ∈ Set.Ico a b → ∀ s ∈ A.reads g,
        s ∈ record ∪ Z ∨ s ∉ A.downstreamAvoiding (record ∪ Z) (A.region X (Set.Ico a b))

end Accounting.Guarantees
