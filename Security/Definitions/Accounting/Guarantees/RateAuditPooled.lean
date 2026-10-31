import Definitions.Accounting.RateAuditRemote
import Definitions.Accounting.RateAuditPooled

/-!
# The causal bound over the monitoring period (trusted)

A session's openings certify its average, not each round's: over rounds whose good sets precede their samples
(`Definitions/Accounting/RateAuditPooled.lean`), every opening passes while the rounds hold at most a `1 − d̄` share of
good items, weighted by their sample counts, with probability at most `(1 − d̄)^(∑ s_i)`. One budget of openings, about
`ln(1/ε)/d̄`, serves the whole session, however its rounds are cut. Proofs are in
`Security/Proofs/Accounting/RateAudit/Pooled.lean`. No named assumption: the samples are the verifier's uniform
coins (or, under Fiat–Shamir, the random oracle's, as in `Rate.Chain`).

* `PooledSampling`: the bound for any causal family of rounds.
* `RatePooled`: the bound for a chain's passes, with their correct receipts as the good sets.
* `RateChainSampledRO`, `RateChainSampled`: the route `Sampled`: with `RateChainPaidRO`'s conservation, a window whose
  passes are all accepted read the words of more than `K` correct receipts, except with the two bounds' sum.
* `RateWindowDraw`: the deployed window's draw, `d` openings pooled over its passes' `P·N` receipts after the last root.
* `RateSessionDraw`: a session of such windows, each adapting to the earlier windows' draws: the monitoring
  period's shortfall and every window's, at once.

The prover is one undifferentiated resource (`Definitions/Accounting/RateAuditPooled.lean`): these bound the correct
receipts, and with conservation the words read, rebuilt or held, whichever devices or parties hold them.
-/

namespace Accounting.Guarantees

open Accounting Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal

/-- **The causal bound over the monitoring period.** For every family of `n` rounds, round `j` with its own coins,
uniform over a finite `X j`, and a uniform sample of `s j` of `M` items, and every causal choice of good sets
(`Rate.Causal`: round `i`'s good set any function of the earlier rounds' coins, their samples among them, and of round
`i`'s own coins other than its sample): the probability that every sample lands in its round's good set while the good
sets, weighted by the sample counts, hold at most `K` items,

  `∑ᵢ sᵢ·|Gᵢ| ≤ K`,

is at most `(K / (M·∑ᵢ sᵢ))^(∑ᵢ sᵢ)`.

**The reading.** At `K = (1 − d̄)·M·∑ᵢ sᵢ`, rounds that fall short of the items by `d̄` on average, weighted by their
sample counts, all pass with probability at most `(1 − d̄)^(∑ᵢ sᵢ)`: the session pays `ln(1/ε)/d̄` openings in all, as
one round would, not in each round. The prover may adapt every round to the earlier rounds' samples; it cannot shift
its shortfall into rounds it already knows were lightly sampled, because each round's sample is drawn after its good
set is fixed.

Proof: `Accounting.SecurityProofs.PooledSampling`, by induction on the rounds: round 0's sample lands in its good set
with probability `(g₀/M)^s₀`, the later rounds are a causal family at every value of round 0's coins, and weighted
AM–GM joins `(g₀/M)^s₀` to the rest's `((K − s₀·g₀)/(M·∑ᵢ₌₁ sᵢ))^(∑ᵢ₌₁ sᵢ)`.

Frozen (no-function): a bound on uniform samples over an abstract causal family, not on a deployed sampler. -/
def PooledSampling : Prop :=
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
sample here; a window's single pooled draw over its passes' receipts is `PooledSampling` with `M = P·N`.

Proof: `Accounting.SecurityProofs.RatePooled`, `PooledSampling` at each store, with pass `j`'s correct receipts as
its good set, which the store, the earlier passes' coins and pass `j`'s stream and oracle family fix.

Frozen (no-function, hash): as `RateAudit`. -/
def RatePooled : Prop :=
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

Proof: `Accounting.SecurityProofs.RateChainSampledRO`: at most `K` correct receipts is `RatePooled`'s event, and more
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
`RatePooled` and `RateChainPaid`.

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

/-- **A window's pooled draw.** For every chained prover over `P` passes (`Rate.Chain`), over a uniform store, each
pass's uniform coins and one draw of `d` openings, each a pass and a word, uniform over the window's `P·N` receipts
(`Rate.WindowCoins`): the probability that every opening lands on a correct receipt (`Chain.correctAll`) while the
window holds at most `K` correct receipts is at most `(K / (P·N))^d`.

**The reading.** The deployed window: its passes draw nothing of their own, and one draw from the openings coin, after
the last root, opens `d` of all its passes' receipts. At `K = (1 − d̄)·P·N`, a window short of its receipts by `d̄`
passes with probability at most `(1 − d̄)^d`, however its shortfall is spread over its passes.

Proof: `Accounting.SecurityProofs.RateWindowDraw`: `PooledSampling`'s one round at `M = P·N`: the draw is uniform and
independent of the window's other coins, so at each value of them it lands in the correct receipts with probability
`(#correct/(P·N))^d`.

Frozen (no-function, hash): as `RateAudit`. -/
def RateWindowDraw : Prop :=
  ∀ (P N b h ρ S L q r s d K : ℕ) (A : Rate.Chain P N b h ρ S L q r s),
    pr (fun x : Store N b × WindowCoins P N b h ρ r s d =>
        (∀ t, x.2.2 t ∈ A.correctAll x.1 x.2.1) ∧ (A.correctAll x.1 x.2.1).card ≤ K)
      ≤ ((K : ℝ≥0∞) / (P * N)) ^ d

/-- **A session's pooled draws: the monitoring period and every window.** For every session of `n` chained windows on
one store (`Rate.Session`: window `w`'s chained prover any function of the earlier windows' coins, their draws among
them), each closed by its pooled draw of `d` openings: the probability that every window's openings land on correct
receipts while either the session holds at most `K` correct receipts in all or some window holds at most `K_w` is at
most

  `(K / (n·P·N))^(d·n) + (((n − 1)·P·N + K_w) / (n·P·N))^(d·n)`.

**The reading.** The hybrid claim: at `K = (1 − d̄)·n·P·N` and `K_w = (1 − d_w)·P·N`, a session whose openings all pass
was short by less than `d̄` over its monitoring period and by less than `d_w` in every window, except with
`(1 − d̄)^(d·n) + (1 − d_w/n)^(d·n) ≤ (1 − d̄)^(d·n) + e^(−d·d_w)`. The floor needs no union over the windows: a prover
can't make a short window pass more often by shorting several. At `2^-40` for each, `d̄ = 0.15%` needs `d·n ≥ 18,470`
openings over the monitoring period and `d_w = 1%` needs `d ≥ 2,773` a window.

Proof: `Accounting.SecurityProofs.RateSessionDraw`, `PooledSampling` with the windows as rounds and `M = P·N` at each
store, twice: with each window's correct receipts as its good set, and with every window's receipts good but the first
window holding at most `K_w`, whose correct receipts are its good set.

Frozen (no-function, hash): as `RateAudit`. -/
def RateSessionDraw : Prop :=
  ∀ (n P N b h ρ S L q r s d K Kw : ℕ) (A : Rate.Session n P N b h ρ S L q r s d),
    pr (fun x : Store N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.chain w x.2).correctAll x.1 (x.2 w).1) ∧
          (∑ w, ((A.chain w x.2).correctAll x.1 (x.2 w).1).card ≤ K ∨
            ∃ w, ((A.chain w x.2).correctAll x.1 (x.2 w).1).card ≤ Kw))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n)

end Accounting.Guarantees
