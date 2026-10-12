import Definitions.Accounting.Confidence
import Definitions.Accounting.Physical
import Definitions.Accounting.CostModel
import Models.Pouw.Assumptions.Game
import Definitions.Pouw.Game.Params

/-!
# Obligations B1 and B4, and PoUW's row, stated (trusted)

Section 8 of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.4). None is proved here.

* `B1` (conjecture, TT on arbitrary circuits): PoUW's named assumption `Pouw.Assumptions.TT` at the cost model of `A`'s
  provers, `CM_A`, whose cost is the prover's work between the salt's arrival and the last tick from which a value
  reaches the verifier by the deadline. It counts the window's work, not dependence on the salt: a prover that branches
  on the salt carries that dependence by control flow, not by wires. Nothing else about PoUW is conjectured: the
  deadline theorem holds at every cost model, and the honest clock holds at `CM_A`.
* `pouwRow` (PoUW's row, from B1): the deadline theorem at `CM_A`. A prover whose work at `π` in the cost window is at
  most `T` at every oracle and salt is accepted while `T < (1 − γ)(1 − ε_s)·W` with probability at most
  `(1 − ε_s)^t + ε(q, N)`.
* `B4` (egress factoring, reduced to the assumptions), across `X`'s secret inputs: over strategies that vary only them,
  under causality, links as alphabets and the trusted devices across the strategy (v2.5: the certifiers' sends during
  `[a, b)` a function of their record before `b` and a jitter word), what the outside does during `I` takes at most
  (the number of records placed in `I`) × `2^{res_out}` values, `res_out` being `log₂ χ` bits a bucket plus
  `β°_X |I|`. Per run, a width bound misses when a send leaves, which signals by its tick.
-/

namespace Accounting.Guarantees

open Finset

/-- **B1** (v2.4): PoUW's assumption TT(γ₀) at `CM_A`, the cost model of `A`'s provers (`Actual.costModel`: the
verifier at `V` releases the salt at `tₛ` and decodes the transcript with `read` from what it receives before `t_d`;
a program is a prover, a strategy with runs in `𝓕` causal from every tick whose oracle calls are the hash's gates
`hash`; its cost is the prover's work at `π` in `[tₛ + ℓ, t_d − ℓ)`). With `HonestClock` at `CM_A`
(`Accounting.SecurityProofs.Actual.honestClock`), `Pouw.SecurityProofs.WorkBetweenSaltAndDeadline` then bounds the
work between the salt and the deadline. -/
def B1 (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {L : Type} [LinearOrder L] [Fintype L]
    (𝓕 : Set (Actual.Run L)) (V : Finset L) (tₛ t_d ℓ : ℕ) (read : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) → Pouw.Transcript)
    (hash : (ρ : Actual.Run L) → Finset (Fin ρ.A.N)) (π : (ρ : Actual.Run L) → Fin ρ.A.N → ℕ)
    (P : Pouw.Protocol Q R S) (D : Pouw.Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  Pouw.Assumptions.TT (Actual.costModel Q R S 𝓕 V tₛ t_d ℓ read hash π) P D γ₀ ε

/-- **PoUW's row, from B1** (v2.4): `Pouw.Guarantees.WorkBetweenSaltAndDeadline` at `CM_A`, its terms and names as
there. Under B1 and the `Ω_max` bound, for every in-domain workload and adversary of `CM_A` whose prover's work at `π`
in `[tₛ + ℓ, t_d − ℓ)` is at most `T` and whose oracle calls are at most `q` at every oracle and salt
(`Adversary.Bounded` at `CM_A`), the audit with `t` work-weighted samples accepts the transcript the verifier receives
before `t_d` while `T < (1 − γ)(1 − ε_s)·W`, `γ = 1 − (1 − γ₀)/Ω_max`, with probability at most
`(1 − ε_s)^t + ε(q, N)`. -/
def pouwRow : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {L : Type} [LinearOrder L] [Fintype L]
    (𝓕 : Set (Actual.Run L)) (V : Finset L) (tₛ t_d ℓ : ℕ) (read : Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) → Pouw.Transcript)
    (hash : (ρ : Actual.Run L) → Finset (Fin ρ.A.N)) (π : (ρ : Actual.Run L) → Fin ρ.A.N → ℕ)
    (P : Pouw.Protocol Q R S) (D : Pouw.Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Pouw.Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    B1 Q R S 𝓕 V tₛ t_d ℓ read hash π P D γ₀ ε →
    ∀ U : Pouw.Workload, U.InDomain D → ∀ A : Pouw.Adversary (Actual.costModel Q R S 𝓕 V tₛ t_d ℓ read hash π),
    ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → A.Bounded U T q → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      Pouw.pr (fun x : (Q → R) × S × (Fin t → Pouw.Ticket (P.Wref U.layout) U.layout.N) =>
          Pouw.AuditAccepts (Pouw.correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          T < (1 - Pouw.reductionGamma γ₀ Ωmax) * (1 - εs) * Pouw.totalWref P U.layout)
        ≤ (1 - εs) ^ t + ε q U.layout.N

/-- **B4.** For a strategy `S` causal from every tick that varies only the secret inputs of a location `X` apart
from the certifiers `cert` (what the certifiers and the outside `O = (X ∪ cert)ᶜ` can tell at `a`, and the record
placed before `a`, don't depend on the coins), whose runs meet causality and links, whose links are alphabets and whose
trusted devices are trusted across it (record `record r`, `TrustedUnits`): what `O` does during `I = [a, b)` takes at
most (the number of records placed in `I`) × `2^{res_out}` values, with `res_out = jitter · (buckets of I) + β°_X
(b − a)`. A wire from `X` to a certifier runs on a certified link and one from `X` to `O` on an uncertified one, by
`Links`. -/
def B4 : Prop :=
  ∀ {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L) (ℓ : L → L → ℕ) (ends : E → L × L)
    (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (𝒱 cert X : Finset L) (spec : ∀ r, (S r).A.Spec)
    (record : ∀ r, Finset (Fin (S r).A.N)) (bucket : ℕ → ℕ) (jitter a b : ℕ),
    CausalFrom S → Alphabet S β link → TrustedUnits S 𝒱 cert spec record bucket jitter → Disjoint X cert →
    (∀ r r', (S r).history cert a = (S r').history cert a ∧
      (S r).history (X ∪ cert)ᶜ a = (S r').history (X ∪ cert)ᶜ a ∧
      (S r).seen '' ↑((record r).filter fun g => (S r).A.t g < a) =
        (S r').seen '' ↑((record r').filter fun g => (S r').A.t g < a)) →
    (∀ r, (S r).A.Causal ℓ ∧ (S r).A.Links ends β (link r)) →
    (Set.range fun r => (S r).acts ↑((X ∪ cert)ᶜ) (Set.Ico a b)).encard ≤
      (Set.range fun r => (S r).seen '' ↑((record r).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b)).encard *
        ((2 ^ (jitter * ((Finset.Ico a b).image bucket).card + uncertifiedCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞)

end Accounting.Guarantees
