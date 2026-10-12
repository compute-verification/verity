import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditInteractive
import Definitions.Accounting.RateAuditRemote

/-!
# Conservation over tiers with named words (trusted)

`RateConservationRO`'s compression names which `k` of the run's `q` oracle queries hold the correct receipts' words,
`log₂ C(q, k)` bits, which `λ` must cover. Naming the `k` words themselves costs `log₂ C(N, k)`: a decoder that replays
the chip recomputes its queries and the receipts it committed, and finds each named word's contents at the query whose
oracle value is that word's receipt. Two of the run's queries with one oracle value could mislead it, at
`C(q, 2)/2^h`. Proofs are in `Security/Proofs/Accounting/RateAudit/Named.lean`. No named assumption, as for
`RateConservationRO`.
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
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        k ≤ (A.correct x.1 x.2.1 x.2.2).card ∧
          ∑ t, A.tierReads tier t x.1 x.2.1 x.2.2 + A.rebuilt x.1 x.2.1 x.2.2 * b + S + (L - ρ) + lam < k * b)
      ≤ (N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (N.choose k : ℝ≥0∞) / 2 ^ lam

end Accounting.Guarantees
