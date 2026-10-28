import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditInteractive
import Definitions.Accounting.RateAuditRemote
import Definitions.Accounting.RateAuditPooled

/-!
# Conservation over tiers with named words (trusted)

`RateConservationRO`'s compression names which `k` of the run's `q` oracle queries hold the correct receipts' words,
`log₂ C(q, k)` bits, which `λ` must cover. Naming the `k` words themselves costs `log₂ C(N, k)`: a decoder that replays
the chip recomputes its queries and the receipts it committed, and finds each named word's contents at the query whose
oracle value is that word's receipt. Two of the run's queries with one oracle value could mislead it, at
`C(q, 2)/2^h`. Proofs are in `Security/Proofs/Accounting/RateAudit/Named.lean`. No named assumption, as for
`RateConservationRO`.

* `RateConservationNamedRO`: at a count `k`, `C(N, k)/2^λ` in place of `C(q, k)/2^λ`.
* `RatePaidNamedRO`: at every count at once, `2^N/2^λ` in place of `RatePaidRO`'s `2^q/2^λ`.
* `RateChainSampledNamedRO`: the route `Sampled` (`RateChainSampledRO`) with that error at each pass.
* `RateSessionSampledNamedRO`: the route `Sampled` for a session of windows, each closed by its pooled draw
  (`RateSessionDraw`): over the monitoring period and in every window at once.

The prover is one undifferentiated resource: these conserve the words read, rebuilt or held, whichever devices or
parties read, rebuild or hold them, with no assumption on who they are.
-/

namespace Accounting.Guarantees

open Accounting Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal

/-- **Conservation over tiers with named words, with side sources that ignore the oracle.** For every device of
`RateConservationRO`'s class and every split of its device memory into `τ` tiers: the probability that at least `k` of
the committed receipts are correct while the run's reads of every tier, its rebuilt words, its on-chip state, the link's
free bits and `λ` fall short of the `k` words,

  `∑ᵢ readsᵢ + rebuilt·b + S + (L − ρ) + λ < k·b`,

is at most `N/2^h + C(q, 2)/2^h + C(N, k)/2^λ`.

**The reading.** `RateConservationRO` with `C(N, k)` in place of `C(q, k)`: the slack `λ` pays for naming `k` of the
store's `N` words, not `k` of the run's `q` queries. At `k = (1 − μ)·N`, `log₂ C(N, k) ≈ N·H(μ)`, so `λ ≈ N·H(μ) + 40`
bits, a fraction `H(μ)/b` of the store: about `0.011/b` at `μ = 0.1%`. The crude form needs `log₂ C(q, k)`, about `2N`
bits at `q ≈ 2N`. `C(q, 2)/2^h` is two queries the run made with one oracle value, about `2^-180` at `q = 2^39` and
`h = 256`.

Proof: `Accounting.SecurityProofs.RateConservationNamedRO`: a guessed receipt (`N/2^h`), a collision among the run's
queries (`C(q, 2)/2^h`, an adaptive `q`-query search for a collision), or, otherwise, compression by a code of the
chip's state, the run's trace and the `k` named words (`C(N, k)/2^λ`).

Frozen (no-function, hash): as `RateAudit`, a bound in the random-oracle model over an abstract chip program. -/
def RateConservationNamedRO : Prop :=
  ∀ (N b h ρ S L q k lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.InteractiveDevice N b h ρ S L q),
    (∀ E R O O', A.side E R O = A.side E R O') →
    pr (fun x : Payloads N b × Bits ρ × Oracle N b h ρ =>
        k ≤ (A.correct x.1 x.2.1 x.2.2).card ∧
          ∑ t, A.tierReads tier t x.1 x.2.1 x.2.2 + A.rebuilt x.1 x.2.1 x.2.2 * b + S + (L - ρ) + lam < k * b)
      ≤ (N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (N.choose k : ℝ≥0∞) / 2 ^ lam

/-- **Correct receipts are paid for, at every count, with named words.** For every device of `RatePaidRO`'s class and
every split into tiers: the probability that the correct receipts' words exceed the reads of every tier, the rebuilt
words, the on-chip state, the link's free bits and `λ`,

  `∑ᵢ readsᵢ + rebuilt·b + S + (L − ρ) + λ < #correct·b`,

is at most `N/2^h + C(q, 2)/2^h + 2^N/2^λ`.

**The reading.** `RatePaidRO` with `2^N` in place of `2^q`: `λ = N + 40` bits a pass makes the term `2^-40`, a
fraction `1/b` of the store's bits (about `0.003%` at 4 KiB words), whatever the run's query count `q`. `RatePaidRO`
needs `λ = q + 40`, which grows with every hash the prover makes in a pass.

Proof: `Accounting.SecurityProofs.RatePaidNamedRO`: `RateConservationNamedRO`'s three events, the compression taken at
each count `k ≤ N`, and `∑ₖ C(N, k) = 2^N`.

Frozen (no-function, hash): as `RateAudit`. -/
def RatePaidNamedRO : Prop :=
  ∀ (N b h ρ S L q lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.InteractiveDevice N b h ρ S L q),
    (∀ E R O O', A.side E R O = A.side E R O') →
    pr (fun x : Payloads N b × Bits ρ × Oracle N b h ρ =>
        ∑ t, A.tierReads tier t x.1 x.2.1 x.2.2 + A.rebuilt x.1 x.2.1 x.2.2 * b + S + (L - ρ) + lam <
          (A.correct x.1 x.2.1 x.2.2).card * b)
      ≤ (N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam

/-- **The route `Sampled` with named words, with side sources that ignore the oracle.** `RateChainSampledRO` with
`RatePaidNamedRO`'s error at each pass: the probability that every pass is accepted while the passes' reads of every
tier and rebuilt words, plus each pass's on-chip state, free link bits and `λ`, fall short of `K + 1` words,

  `∑ⱼ (∑ᵢ readsⱼᵢ + rebuiltⱼ·b) + P·(S + (L − ρ) + λ) < (K + 1)·b`,

is at most `(K / (P·N))^(s·P) + P·(N/2^h + C(q, 2)/2^h + 2^N/2^λ)`.

**The reading.** At `K = (1 − d̄)·P·N`, a window whose `s·P` openings all pass read, rebuilt or held the words of more
than `K` correct receipts, at a slack of `N + 40` bits a pass for `λ`.

Proof: `Accounting.SecurityProofs.RateChainSampledNamedRO`: `RatePooled`, and `RatePaidNamedRO` at each pass through
`RateChainPaidRO`'s union over passes.

Frozen (no-function, hash): as `RateAudit`. -/
def RateChainSampledNamedRO : Prop :=
  ∀ (P N b h ρ S L q r s K lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.Chain P N b h ρ S L q r s),
    (∀ j η E R O O', (A.device j η).side E R O = (A.device j η).side E R O') →
    pr (fun x : Payloads N b × (Fin P → PassCoins N b h ρ r s) =>
        (∀ j, (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2) ∧
        ∑ j, (∑ t, (A.dev j x.2).tierReads tier t x.1 (x.2 j).1 (A.oracle j x.1 x.2) +
            (A.dev j x.2).rebuilt x.1 (x.2 j).1 (A.oracle j x.1 x.2) * b) + P * (S + (L - ρ) + lam) <
          (K + 1) * b)
      ≤ ((K : ℝ≥0∞) / (P * N)) ^ (s * P) +
        (P : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam)

/-- **The route `Sampled` for a session, with named words, with side sources that ignore the oracle.** For every
session of `n` chained windows on one store, each closed by its pooled draw of `d` openings (`Rate.Session`, as in
`RateSessionDraw`), with every pass's side sources ignoring the oracle, and every split into tiers: the probability
that every window's openings land on correct receipts while either the session's reads of every tier and rebuilt
words (`Chain.spent`), plus each pass's on-chip state, free link bits and `λ`, fall short of `K + 1` words,

  `∑_w spent_w + n·P·(S + (L − ρ) + λ) < (K + 1)·b`,

or some window's fall short of `K_w + 1` words,

  `spent_w + P·(S + (L − ρ) + λ) < (K_w + 1)·b`,

is at most `RateSessionDraw`'s bound plus `n·P·(N/2^h + C(q, 2)/2^h + 2^N/2^λ)`.

**The reading.** The hybrid claim on the words: at `K = (1 − d̄)·n·P·N` and `K_w = (1 − d_w)·P·N`, a session whose
openings all pass read, rebuilt or held more than `(1 − d̄)` of its passes' words over its monitoring period and more
than `(1 − d_w)` of them in every window, at a slack of `N + 40` bits a pass for `λ`. The words are conserved whoever
reads, rebuilds or holds them: the prover is one resource, and nothing here assumes which device or party it is.

Proof: `Accounting.SecurityProofs.RateSessionSampledNamedRO`: at most `K` correct receipts in all or `K_w` in some
window is `RateSessionDraw`'s event; otherwise some window's reads fall short of its correct receipts' words, so some
pass's do, which `RatePaidNamedRO` bounds at each pass, with a union over the windows and their passes.

Frozen (no-function, hash): as `RateAudit`. -/
def RateSessionSampledNamedRO : Prop :=
  ∀ (n P N b h ρ S L q r s d K Kw lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.Session n P N b h ρ S L q r s d),
    (∀ w η j ι E R O O', ((A.window w η).device j ι).side E R O = ((A.window w η).device j ι).side E R O') →
    pr (fun x : Payloads N b × (Fin n → WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.chain w x.2).correctAll x.1 (x.2 w).1) ∧
          (∑ w, (A.chain w x.2).spent tier x.1 (x.2 w).1 + n * (P * (S + (L - ρ) + lam)) < (K + 1) * b ∨
            ∃ w, (A.chain w x.2).spent tier x.1 (x.2 w).1 + P * (S + (L - ρ) + lam) < (Kw + 1) * b))
      ≤ ((K : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        ((((n - 1) * (P * N) + Kw : ℕ) : ℝ≥0∞) / (n * P * N)) ^ (d * n) +
        (n * P : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam)

end Accounting.Guarantees
