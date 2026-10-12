import Definitions.Accounting.Confidence
import Definitions.Accounting.Physical

/-!
# Obligations B1 and B4, stated (trusted)

Section 8 of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.4). Neither is proved here.

* `B1` (conjecture, TT_OUT on arbitrary circuits): PoUW's audit is sound for a lower bound on the work of the prover's
  computed gates placed between the seed's release and the response's deadline. It is a property of an audit (`acc`,
  read on the verifier's view), and the conjecture is `B1` at PoUW's. It counts the window's work, not dependence on
  the seed: a prover that branches on the seed carries that dependence by control flow, not by wires.
* `B4` (egress factoring, reduced to the assumptions), across `X`'s secret inputs: over strategies that vary only them,
  under causality, links as alphabets and the trusted devices across the strategy (v2.4: the certifiers' sends a
  function of their record and a jitter word), what the outside does during `I` takes at most (the number of records) ×
  `2^{res_out}` values, `res_out` being `log₂ χ` bits a bucket plus `β°_X |I|`. Per run, a width bound misses when a
  send leaves, which signals by its tick.
-/

namespace Accounting.Guarantees

open Finset

/-- **B1** at an audit: the verifier at `V`, accepting on `acc`, with the seed released at `tₛ` and the response due at
`t_d`, is sound at `(1 − ε)^K + η + ε_ks + ε_bind` against `𝓕` for the work at `π` of the prover's (off `V`) computed
gates placed in `[tₛ + ℓ, t_d − ℓ)`, at least `(1 − γ)(1 − ε) W`. -/
def B1 {L : Type*} [LinearOrder L] (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L) (acc : Set (ℕ × L × ℕ × ℕ) → Prop)
    (π : (R : Actual.Run L) → Fin R.A.N → ℕ) (tₛ t_d ℓ K : ℕ) (γ ε W η εks εbind : ℝ) : Prop :=
  open Classical in
  Sound Ω 𝓕 V acc (fun S r => (1 - γ) * (1 - ε) * W ≤ (S r).A.work (π (S r)) ((S r).A.computed.filter fun g =>
      (S r).A.x g ∉ V ∧ (S r).A.t g ∈ Set.Ico (tₛ + ℓ) (t_d - ℓ)))
    ((1 - ε) ^ K + η + εks + εbind)

/-- **B4.** For a strategy `S` causal from every tick that varies only the secret inputs of a location `X` apart
from the certifiers `cert` (what the certifiers and the outside `O = (X ∪ cert)ᶜ` can tell at `a` doesn't depend on
the coins), whose runs meet causality and links, whose links are alphabets and whose trusted devices are trusted across
it (record `record r`, `TrustedUnits`): what `O` does during `I = [a, b)` takes at most (the number of records) ×
`2^{res_out}` values, with `res_out = jitter · (buckets of I) + β°_X (b − a)`. A wire from `X` to a certifier runs on
a certified link and one from `X` to `O` on an uncertified one, by `Links`. -/
def B4 : Prop :=
  ∀ {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L) (ℓ : L → L → ℕ) (ends : E → L × L)
    (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (𝒱 cert X : Finset L) (spec : ∀ r, (S r).A.Spec)
    (record : ∀ r, Finset (Fin (S r).A.N)) (bucket : ℕ → ℕ) (jitter a b : ℕ),
    CausalFrom S → Alphabet S β link → TrustedUnits S 𝒱 cert spec record bucket jitter → Disjoint X cert →
    (∀ r r', (S r).history cert a = (S r').history cert a ∧
      (S r).history (X ∪ cert)ᶜ a = (S r').history (X ∪ cert)ᶜ a) →
    (∀ r, (S r).A.Causal ℓ ∧ (S r).A.Links ends β (link r)) →
    (Set.range fun r => (S r).acts ↑((X ∪ cert)ᶜ) (Set.Ico a b)).encard ≤
      (Set.range fun r => (S r).seen '' ↑(record r)).encard *
        ((2 ^ (jitter * ((Finset.Ico a b).image bucket).card + uncertifiedCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞)

end Accounting.Guarantees
