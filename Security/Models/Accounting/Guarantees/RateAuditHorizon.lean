import Definitions.Accounting.RateAuditRemote
import Definitions.Accounting.RateAuditHorizon

/-!
# The causal horizon bound (trusted)

A session's openings certify its average, not each round's: over rounds whose good sets precede their samples
(`Definitions/Accounting/RateAuditHorizon.lean`), every opening passes while the rounds hold at most a `1 − d̄` share of
good items, weighted by their sample counts, with probability at most `(1 − d̄)^(∑ s_i)`. One budget of openings, about
`ln(1/ε)/d̄`, serves the whole session, however its rounds are cut. Proofs are in
`Security/Proofs/Accounting/RateAudit/Horizon.lean`. No named assumption: the samples are the verifier's uniform
coins (or, under Fiat–Shamir, the random oracle's, as in `Rate.Chain`).

* `HorizonSampling`: the bound for any causal family of rounds.
* `RateHorizon`: the bound for a chain's passes, with their correct receipts as the good sets.
* `RateChainSampledRO`, `RateChainSampled`: the route `Sampled`: with `RateChainPaidRO`'s conservation, a window whose
  passes are all accepted read the words of more than `K` correct receipts, except with the two bounds' sum.
-/

namespace Accounting.Guarantees

open Accounting Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal

/-- **The causal horizon bound.** For every family of `n` rounds, round `j` with its own coins, uniform over a finite
`X j`, and a uniform sample of `s j` of `M` items, and every causal choice of good sets (`Rate.Causal`: round `i`'s good
set any function of the earlier rounds' coins, their samples among them, and of round `i`'s own coins other than its
sample): the probability that every sample lands in its round's good set while the good sets, weighted by the sample
counts, hold at most `K` items,

  `∑ᵢ sᵢ·|Gᵢ| ≤ K`,

is at most `(K / (M·∑ᵢ sᵢ))^(∑ᵢ sᵢ)`.

**The reading.** At `K = (1 − d̄)·M·∑ᵢ sᵢ`, rounds that fall short of the items by `d̄` on average, weighted by their
sample counts, all pass with probability at most `(1 − d̄)^(∑ᵢ sᵢ)`: the session pays `ln(1/ε)/d̄` openings in all, as
one round would, not in each round. The prover may adapt every round to the earlier rounds' samples; it cannot shift
its shortfall into rounds it already knows were lightly sampled, because each round's sample is drawn after its good
set is fixed.

Proof: `Accounting.SecurityProofs.HorizonSampling`, by induction on the rounds: round 0's sample lands in its good set
with probability `(g₀/M)^s₀`, the later rounds are a causal family at every value of round 0's coins, and weighted
AM–GM joins `(g₀/M)^s₀` to the rest's `((K − s₀·g₀)/(M·∑ᵢ₌₁ sᵢ))^(∑ᵢ₌₁ sᵢ)`.

Frozen (no-function): a bound on uniform samples over an abstract causal family, not on a deployed sampler. -/
def HorizonSampling : Prop :=
  ∀ (n M K : ℕ) (X : Fin n → Type) [∀ i, Fintype (X i)] (s : Fin n → ℕ)
    (G : (i : Fin n) → ((j : Fin n) → X j × (Fin (s j) → Fin M)) → Finset (Fin M)),
    Causal G →
    pr (fun ω : (j : Fin n) → X j × (Fin (s j) → Fin M) =>
        (∀ i t, (ω i).2 t ∈ G i ω) ∧ ∑ i, s i * (G i ω).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (M * (∑ i, s i : ℕ))) ^ (∑ i, s i)

/-- **A chain's passes are sampled as a whole.** For every chained prover over `P` passes with `r`-bit roots
(`Rate.Chain`: pass `j`'s interactive device and the root its seed is drawn at, each any function of the earlier
passes' coins, the root also of the store), over a uniform store and each pass's uniform stream, oracle family and
sample of `s` words: the probability that every pass is accepted at its seed's oracle while the passes' correct
receipts number at most `K` in all is at most `(K / (P·N))^(s·P)`.

**The reading.** At `K = (1 − d̄)·P·N`, a prover whose passes fall short of the store by `d̄` on average passes all
`s·P` openings with probability at most `(1 − d̄)^(s·P)`. The passes may be a window's chained passes, or a session's
windows (with `r = 0`, a fresh oracle a window), each device adapting to every earlier sample. Each pass draws its own
sample here; a window's single pooled draw over its passes' receipts is `HorizonSampling` with `M = P·N`.

Proof: `Accounting.SecurityProofs.RateHorizon`, `HorizonSampling` at each store, with pass `j`'s correct receipts as
its good set, which the store, the earlier passes' coins and pass `j`'s stream and oracle family fix.

Frozen (no-function, hash): as `RateAudit`. -/
def RateHorizon : Prop :=
  ∀ (P N b h ρ S L q r s K : ℕ) (A : Rate.Chain P N b h ρ S L q r s),
    pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        (∀ j, (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2) ∧
          ∑ j, ((A.dev j x.2).correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (P * N)) ^ (s * P)

/-- **The route `Sampled`, with side sources that ignore the oracle.** For every chained prover of `RateChainPaidRO`'s
class, every split into tiers and every `K`: the probability that every pass is accepted while the passes' reads of
every tier and rebuilt words, plus each pass's on-chip state, free link bits and `λ`, fall short of `K + 1` words,

  `∑ⱼ (∑ᵢ readsⱼᵢ + rebuiltⱼ·b) + P·(S + (L − ρ) + λ) < (K + 1)·b`,

is at most `(K / (P·N))^(s·P) + P·(N/2^h + 2^q/2^λ)`.

**The reading.** The route `Sampled` of the remote audit: a window whose `s·P` openings all pass read, rebuilt or held
the words of more than `K = (1 − d̄)·P·N` correct receipts, except with `(1 − d̄)^(s·P)` plus `RateChainPaidRO`'s error.

Proof: `Accounting.SecurityProofs.RateChainSampledRO`: at most `K` correct receipts is `RateHorizon`'s event, and more
is `RateChainPaidRO`'s.

Frozen (no-function, hash): as `RateAudit`. -/
def RateChainSampledRO : Prop :=
  ∀ (P N b h ρ S L q r s K lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.Chain P N b h ρ S L q r s),
    (∀ j η E R O O', (A.device j η).side E R O = (A.device j η).side E R O') →
    pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        (∀ j, (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2) ∧
        ∑ j, (∑ t, (A.dev j x.2).tierReads tier t x.1 (x.2 j).1 (A.oracle j x.1 x.2) +
            (A.dev j x.2).rebuilt x.1 (x.2 j).1 (A.oracle j x.1 x.2) * b) + P * (S + (L - ρ) + lam) <
          (K + 1) * b)
      ≤ ((K : ℝ≥0∞) / (P * N)) ^ (s * P) + (P : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h + (2 ^ q : ℝ≥0∞) / 2 ^ lam)

/-- **The route `Sampled`, with any side sources.** `RateChainSampledRO` for devices whose side sources may relay
receipts: with `m` words a pass added to the slack, the probability is at most
`(K / (P·N))^(s·P) + P·(C(N, m)·2^L/2^(h·m) + 2^q/2^λ)`. Proof: `Accounting.SecurityProofs.RateChainSampled`, from
`RateHorizon` and `RateChainPaid`.

Frozen (no-function, hash): as `RateAudit`. -/
def RateChainSampled : Prop :=
  ∀ (P N b h ρ S L q r s K m lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.Chain P N b h ρ S L q r s),
    pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        (∀ j, (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2) ∧
        ∑ j, (∑ t, (A.dev j x.2).tierReads tier t x.1 (x.2 j).1 (A.oracle j x.1 x.2) +
            (A.dev j x.2).rebuilt x.1 (x.2 j).1 (A.oracle j x.1 x.2) * b) + P * (S + (L - ρ) + lam + m * b) <
          (K + 1) * b)
      ≤ ((K : ℝ≥0∞) / (P * N)) ^ (s * P) +
        (P : ℝ≥0∞) * ((N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam)

end Accounting.Guarantees
