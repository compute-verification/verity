module

public import Definitions.Poub.RateAudit
public import Definitions.Poub.RateAuditInteractive
public import Definitions.Poub.RateAuditRemote

/-! The remote rate audit: conservation over tiers and chained passes, with no named assumption, the codec entering
only as `c_enc`. -/

@[expose] public section

namespace ResourceAccounting.Guarantees

open ResourceAccounting ResourceAccounting.Rate
open Pous (Bits pr)
open scoped ENNReal

/-- **Conservation over tiers, with side sources that ignore the oracle.** For every interactive device of `RateAudit`'s
class whose side sources answer as any function of the store, the certified stream and what the chip sent, but not of
the oracle, for every split of its device memory into `τ` tiers, over a uniform store of `N` words of `b` bits, a
uniform certified stream of `ρ` bits and a fresh oracle with `h`-bit receipts: the probability that at least `k` of the
committed receipts are correct while the run's reads of every tier, its rebuilt words, its on-chip state, the link's
free bits and `λ` fall short of the `k` words,

  `∑ᵢ readsᵢ + rebuilt·b + S + (L − ρ) + λ < k·b`,

is at most `N/2^h + C(q, k)/2^λ`.

**The reading.** No sample: this counts the correct receipts themselves. Except with that probability, `k` correct
receipts mean `k·b ≤ ∑ᵢ readsᵢ + rebuilt·b + S + (L − ρ) + λ`: every receipt the node computed needed its word in hand,
and the node had it from a read of some tier, from on chip, from over its NIC, or by rebuilding it from the codec at
`c_enc` a word (`rate-codec-work`). The tiers are any partition of the node's memory (each GPU's HBM, host DRAM), and
the chip is all of the node's processors, so it holds wherever on the node a receipt was computed; a host CPU reading
host DRAM reads a counted tier. `N/2^h` is a receipt guessed for a word never queried, over all `N` words.

Proof: `ResourceAccounting.SecurityProofs.RateConservationRO`, `RateAudit`'s guessing and compression on the run's queried
words, with the footprint at most the reads summed over the tiers.

Frozen (no-function, hash): as `RateAudit`, a bound in the random-oracle model over an abstract chip program. -/
def RateConservationRO : Prop :=
  ∀ (N b h ρ S L q k lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.InteractiveProver N b h ρ S L q),
    (∀ E R O O', A.side E R O = A.side E R O') →
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        k ≤ (A.correct x.1 x.2.1 x.2.2).card ∧
          ∑ t, A.tierReads tier t x.1 x.2.1 x.2.2 + A.rebuilt x.1 x.2.1 x.2.2 * b + S + (L - ρ) + lam < k * b)
      ≤ (N : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam

/-- **Conservation over tiers, with any side sources.** `RateConservationRO` for side sources that may answer as any
function of the oracle too, so that they may compute receipts and relay them in: for every `m`, the probability that at
least `k` committed receipts are correct while the reads, rebuilt words, state, free link bits and `λ` fall short of
`k − m` words,

  `∑ᵢ readsᵢ + rebuilt·b + S + (L − ρ) + λ < (k − m)·b`,

is at most `C(N, m)·2^L/2^(h·m) + C(q, k − m)/2^λ`.

**The reading.** A receipt computed beyond the node's boundary (a peer holding a copy of the store, once it has the
seed) comes in over the NIC: `m` relayed receipts fix `h·m` bits of the oracle from at most `L` bits read, so about
`L/h` of them can be relayed, each at `h` bits of the link instead of `b` bits of a tier. Every other correct receipt
costs its word, as in `RateConservationRO`.

Proof: `ResourceAccounting.SecurityProofs.RateConservation`, `RateAuditInteractive`'s relay count and compression.

Frozen (no-function, hash): as `RateAudit`. -/
def RateConservation : Prop :=
  ∀ (N b h ρ S L q k m lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.InteractiveProver N b h ρ S L q),
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        k ≤ (A.correct x.1 x.2.1 x.2.2).card ∧
          ∑ t, A.tierReads tier t x.1 x.2.1 x.2.2 + A.rebuilt x.1 x.2.1 x.2.2 * b + S + (L - ρ) + lam < (k - m) * b)
      ≤ (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + ((q.choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **Correct receipts are bounded by the tiers' capacity, with side sources that ignore the oracle.** For every device
of `RateConservationRO`'s class whose reads of each tier `i` are at most `cap i` bits in every world (`TierCapped`, with
`cap i = R_i·T` for a tier of read capacity `R_i` over a window of length `T`), and for every compute budget `F` at
`c_tag` per receipt and `c_enc` per rebuilt word: if the capacities fall short of `RateAudit`'s floor at `k` words,

  `(∑ᵢ cap i)·c_enc < k·b·c_enc − (S + (L − ρ) + λ)·c_enc − (F − k·c_tag)·b`,

then the probability that at least `k` committed receipts are correct and the run's compute is within `F` is at most
`N/2^h + C(q, k)/2^λ`.

**The reading.** Dividing by `b·c_enc`: except with that probability, the window's correct receipts number at most

  `∑ᵢ R_i·T/b + (S + (L − ρ) + λ)/b + (F − k·c_tag)/c_enc`,

the tiers' capacity in words plus the slack terms of `RateAudit`: on-chip state, the link's free bits, `λ`, and the
words the compute left after receipting could rebuild. The HBM bandwidth a cheater frees is not lost to the count: it
must be spent on another counted tier, or bought back at the codec's price (`RateDisplacement`).

Proof: `ResourceAccounting.SecurityProofs.RateConservationCapRO`.

Frozen (no-function, hash): as `RateAudit`. -/
def RateConservationCapRO : Prop :=
  ∀ (N b h ρ S L q k F cTag cEnc lam τ : ℕ) (tier : ℕ → Fin τ) (cap : Fin τ → ℕ)
    (A : Rate.InteractiveProver N b h ρ S L q),
    (∀ E R O O', A.side E R O = A.side E R O') → A.TierCapped tier cap →
    (∑ t, cap t) * cEnc < floor b ρ S L F cTag cEnc k lam →
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        k ≤ (A.correct x.1 x.2.1 x.2.2).card ∧ A.compute cTag cEnc x.1 x.2.1 x.2.2 ≤ F)
      ≤ (N : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam

/-- **Correct receipts are bounded by the tiers' capacity, with any side sources.** `RateConservationCapRO` for side
sources that may relay receipts: for every `m`, if the capacities fall short of `RateAudit`'s floor at `k − m` words,
the probability that at least `k` committed receipts are correct with the compute within `F` is at most
`C(N, m)·2^L/2^(h·m) + C(q, k − m)/2^λ`. The count is `RateConservationCapRO`'s plus the `m` relayed receipts, about
`L/h`. Proof: `ResourceAccounting.SecurityProofs.RateConservationCap`.

Frozen (no-function, hash): as `RateAudit`. -/
def RateConservationCap : Prop :=
  ∀ (N b h ρ S L q k m F cTag cEnc lam τ : ℕ) (tier : ℕ → Fin τ) (cap : Fin τ → ℕ)
    (A : Rate.InteractiveProver N b h ρ S L q),
    A.TierCapped tier cap →
    (∑ t, cap t) * cEnc < floor b ρ S L F cTag cEnc (k - m) lam →
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        k ≤ (A.correct x.1 x.2.1 x.2.2).card ∧ A.compute cTag cEnc x.1 x.2.1 x.2.2 ≤ F)
      ≤ (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + ((q.choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **Freed bandwidth is spent on another tier or rebuilt.** For every device of `RateConservationRO`'s class, every
tier `t₀` and every `D`: the probability that at least `k` committed receipts are correct, the reads of `t₀` fall
short of the `k` words by `D` bits beyond the slack,

  `reads_{t₀} + D + S + (L − ρ) + λ ≤ k·b`,

and the reads of the other tiers and the rebuilt words make up less than `D`,

  `∑_{i ≠ t₀} readsᵢ + rebuilt·b < D`,

is at most `N/2^h + C(q, k)/2^λ`.

**The reading.** A cheater who reads `D` bits less of HBM than its receipts need reads them from another counted tier
(host DRAM over PCIe, at that tier's capacity) or rebuilds `D/b` words at `c_enc` each, `D·c_enc/b` of compute. Proof:
`ResourceAccounting.SecurityProofs.RateDisplacement`, from `RateConservationRO`.

Frozen (no-function, hash): as `RateAudit`. -/
def RateDisplacement : Prop :=
  ∀ (N b h ρ S L q k D lam τ : ℕ) (tier : ℕ → Fin τ) (t₀ : Fin τ) (A : Rate.InteractiveProver N b h ρ S L q),
    (∀ E R O O', A.side E R O = A.side E R O') →
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        k ≤ (A.correct x.1 x.2.1 x.2.2).card ∧
          A.tierReads tier t₀ x.1 x.2.1 x.2.2 + D + S + (L - ρ) + lam ≤ k * b ∧
          ∑ t ∈ Finset.univ.erase t₀, A.tierReads tier t x.1 x.2.1 x.2.2 + A.rebuilt x.1 x.2.1 x.2.2 * b < D)
      ≤ (N : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam

/-- **A chained pass is a fresh audit, with side sources that ignore the oracle.** For every chained prover over `P`
passes with `r`-bit roots (`Rate.Chain`: pass `j`'s interactive device of `RateAudit`'s class and the root its seed is
sampled at, each any function of the earlier passes' coins, the root also of the store) whose devices' side sources
ignore the oracle, over a uniform store and each pass's uniform stream, oracle family and sample, and for every pass
`j`: the probability that pass `j` is accepted at its seed's oracle with its compute within `F` and its footprint short
of `RateAudit`'s floor is at most `RateAudit`'s error, `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ`.

**The reading.** Sequencing: pass `j`'s receipts are as hard as a fresh window's, because nothing the prover holds when
`root_{j−1}` is fixed depends on pass `j`'s oracle, and the oracle at a root the prover chose from everything before
is still uniform. The prover may pick the root (grind pass `j − 1`'s commitments), lay out device memory anew and keep
`S` bits on chip at the boundary; none of it helps. The sides may depend on earlier passes' oracles: those are other
seeds.

Proof: `ResourceAccounting.SecurityProofs.RateChainPassRO`, `RateAuditInteractiveRO` at each value of the other passes'
coins, after the oracle family is split at the root.

Frozen (no-function, hash): as `RateAudit`. -/
def RateChainPassRO : Prop :=
  ∀ (P N b h ρ S L q r s k F cTag cEnc lam : ℕ) (A : Rate.Chain P N b h ρ S L q r s),
    (∀ j η E R O O', (A.device j η).side E R O = (A.device j η).side E R O') →
    ∀ j : Fin P,
      pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
          (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2 ∧
            (A.dev j x.2).compute cTag cEnc x.1 (x.2 j).1 (A.oracle j x.1 x.2) ≤ F ∧
            (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc < floor b ρ S L F cTag cEnc k lam)
        ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam

/-- **A chained pass is a fresh audit, with any side sources.** `RateChainPassRO` for devices whose side sources may
answer as any function of pass `j`'s oracle too, at `RateAuditInteractive`'s floor at `k − m` words and its error,
`((k − 1)/N)^s + C(N, m)·2^L/2^(h·m) + C(q, k − m)/2^λ`. Proof: `ResourceAccounting.SecurityProofs.RateChainPass`, from
`RateAuditInteractive`.

Frozen (no-function, hash): as `RateAudit`. -/
def RateChainPass : Prop :=
  ∀ (P N b h ρ S L q r s k m F cTag cEnc lam : ℕ) (A : Rate.Chain P N b h ρ S L q r s) (j : Fin P),
    pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2 ∧
          (A.dev j x.2).compute cTag cEnc x.1 (x.2 j).1 (A.oracle j x.1 x.2) ≤ F ∧
          (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc <
            floor b ρ S L F cTag cEnc (k - m) lam)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m)
        + ((q.choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **A window of chained passes certifies the sum over its passes, with side sources that ignore the oracle.** For
every chained prover of `RateChainPassRO`'s class and every `c`: `c` times the probability that every pass is accepted
with its compute within `F` while the footprints, summed over the passes, fall short of `P + 1 − c` floors,

  `∑ⱼ footprintⱼ · c_enc < (P + 1 − c) · floor`,

is at most `P` times `RateAudit`'s error.

**The reading.** As `RateAuditWindows`', with one challenge for the whole window: at `c = δ·P` the passes' footprints
average at least `(1 − δ)·(1 − μ)·|E|` except with probability `ε/δ`. A footprint is at most the reads summed over the
tiers, so the window's reads of all tiers, at most `∑ᵢ R_i·T`, certify the sum: `P` passes in a window cost `P` passes'
reads, less the slack. Proof: `ResourceAccounting.SecurityProofs.RateChainRO`, from `RateChainPassRO` at each pass.

Frozen (no-function, hash): as `RateAudit`. -/
def RateChainRO : Prop :=
  ∀ (P N b h ρ S L q r s k F cTag cEnc lam c : ℕ) (A : Rate.Chain P N b h ρ S L q r s),
    (∀ j η E R O O', (A.device j η).side E R O = (A.device j η).side E R O') →
    (c : ℝ≥0∞) * pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        (∀ j, (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2 ∧
          (A.dev j x.2).compute cTag cEnc x.1 (x.2 j).1 (A.oracle j x.1 x.2) ≤ F) ∧
        ∑ j, (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc <
          (P + 1 - c) * floor b ρ S L F cTag cEnc k lam)
      ≤ (P : ℝ≥0∞) * ((((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam)

/-- **A window of chained passes certifies the sum over its passes, with any side sources.** `RateChainRO` at
`RateChainPass`'s floor at `k − m` words and its error. Proof: `ResourceAccounting.SecurityProofs.RateChain`.

Frozen (no-function, hash): as `RateAudit`. -/
def RateChain : Prop :=
  ∀ (P N b h ρ S L q r s k m F cTag cEnc lam c : ℕ) (A : Rate.Chain P N b h ρ S L q r s),
    (c : ℝ≥0∞) * pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        (∀ j, (A.dev j x.2).Accepts x.1 (x.2 j).1 (A.oracle j x.1 x.2) (x.2 j).2.2 ∧
          (A.dev j x.2).compute cTag cEnc x.1 (x.2 j).1 (A.oracle j x.1 x.2) ≤ F) ∧
        ∑ j, (A.dev j x.2).footprint x.1 (x.2 j).1 (A.oracle j x.1 x.2) * cEnc <
          (P + 1 - c) * floor b ρ S L F cTag cEnc (k - m) lam)
      ≤ (P : ℝ≥0∞) * ((((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m)
        + ((q.choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam)

/-- **Correct receipts are paid for, at every count, with side sources that ignore the oracle.** For every device of
`RateConservationRO`'s class and every split of its device memory into `τ` tiers: the probability that the correct
receipts' words exceed the run's reads of every tier, its rebuilt words, its on-chip state, the link's free bits and
`λ`,

  `∑ᵢ readsᵢ + rebuilt·b + S + (L − ρ) + λ < #correct·b`,

is at most `N/2^h + 2^q/2^λ`.

**The reading.** `RateConservationRO` with no threshold: whatever number of correct receipts the run made, except with
that probability `#correct·b ≤ ∑ᵢ readsᵢ + rebuilt·b + S + (L − ρ) + λ`. This is the route `Paid` of PoUB
for one pass, `correct·b ≤ paid + λ`. `2^q` counts every set of the run's `q` oracle queries the correct words could
be; `λ = q + 40` makes the term `2^-40` for about `q/b` words of slack, negligible beside the store.

Proof: `ResourceAccounting.SecurityProofs.RatePaidRO`, `RateConservationRO`'s guessing and its compression at each count
`k ≤ q`.

Frozen (no-function, hash): as `RateAudit`. -/
def RatePaidRO : Prop :=
  ∀ (N b h ρ S L q lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.InteractiveProver N b h ρ S L q),
    (∀ E R O O', A.side E R O = A.side E R O') →
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        ∑ t, A.tierReads tier t x.1 x.2.1 x.2.2 + A.rebuilt x.1 x.2.1 x.2.2 * b + S + (L - ρ) + lam <
          (A.correct x.1 x.2.1 x.2.2).card * b)
      ≤ (N : ℝ≥0∞) / 2 ^ h + (2 ^ q : ℝ≥0∞) / 2 ^ lam

/-- **Correct receipts are paid for, at every count, with any side sources.** `RatePaidRO` for side sources that may
relay receipts: for every `m`, the probability that the correct receipts' words exceed the reads, rebuilt words, state,
free link bits, `λ` and `m` words,

  `∑ᵢ readsᵢ + rebuilt·b + S + (L − ρ) + λ + m·b < #correct·b`,

is at most `C(N, m)·2^L/2^(h·m) + 2^q/2^λ`: `m` relayed receipts, about `L/h`, are free. Proof:
`ResourceAccounting.SecurityProofs.RatePaid`.

Frozen (no-function, hash): as `RateAudit`. -/
def RatePaid : Prop :=
  ∀ (N b h ρ S L q m lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.InteractiveProver N b h ρ S L q),
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        ∑ t, A.tierReads tier t x.1 x.2.1 x.2.2 + A.rebuilt x.1 x.2.1 x.2.2 * b + S + (L - ρ) + lam + m * b <
          (A.correct x.1 x.2.1 x.2.2).card * b)
      ≤ (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam

/-- **A window's correct receipts are paid for pass by pass, with side sources that ignore the oracle.** For every
chained prover of `RateChainPassRO`'s class and every split into tiers: the probability that the correct receipts'
words, summed over the passes, exceed the passes' reads of every tier and rebuilt words plus each pass's on-chip state,
free link bits and `λ`,

  `∑ⱼ (∑ᵢ readsⱼᵢ + rebuiltⱼ·b) + P·(S + (L − ρ) + λ) < ∑ⱼ #correctⱼ·b`,

is at most `P·(N/2^h + 2^q/2^λ)`.

**The reading.** The route `Paid` of PoUB for a window, `correct·b ≤ paid + P·λ`: the reads summed over
passes and tiers, the rebuilt words, state `S` and link bits `L − ρ` at each pass boundary, and `λ` a pass; nothing
outside the node, whose side sources cannot query the oracle. No sample is sampled, so it holds with the window-wide
sample, which the route `Sampled` prices. A word read in pass `j − 1` serves pass `j` only through the `S` bits on chip
when `root_{j−1}` is fixed, because pass `j`'s oracle is fresh at that root.

Proof: `ResourceAccounting.SecurityProofs.RateChainPaidRO`, `RatePaidRO` at each pass at every value of the other passes'
coins, and a union bound over the passes.

Frozen (no-function, hash): as `RateAudit`. -/
def RateChainPaidRO : Prop :=
  ∀ (P N b h ρ S L q r s lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.Chain P N b h ρ S L q r s),
    (∀ j η E R O O', (A.device j η).side E R O = (A.device j η).side E R O') →
    pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        ∑ j, (∑ t, (A.dev j x.2).tierReads tier t x.1 (x.2 j).1 (A.oracle j x.1 x.2) +
            (A.dev j x.2).rebuilt x.1 (x.2 j).1 (A.oracle j x.1 x.2) * b) + P * (S + (L - ρ) + lam) <
          ∑ j, ((A.dev j x.2).correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card * b)
      ≤ (P : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h + (2 ^ q : ℝ≥0∞) / 2 ^ lam)

/-- **A window's correct receipts are paid for pass by pass, with any side sources.** `RateChainPaidRO` for devices
whose side sources may relay receipts: for every `m`, with `m` words a pass added to the slack, the probability is at
most `P·(C(N, m)·2^L/2^(h·m) + 2^q/2^λ)`. Proof: `ResourceAccounting.SecurityProofs.RateChainPaid`, from `RatePaid`.

Frozen (no-function, hash): as `RateAudit`. -/
def RateChainPaid : Prop :=
  ∀ (P N b h ρ S L q r s m lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.Chain P N b h ρ S L q r s),
    pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        ∑ j, (∑ t, (A.dev j x.2).tierReads tier t x.1 (x.2 j).1 (A.oracle j x.1 x.2) +
            (A.dev j x.2).rebuilt x.1 (x.2 j).1 (A.oracle j x.1 x.2) * b) + P * (S + (L - ρ) + lam + m * b) <
          ∑ j, ((A.dev j x.2).correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card * b)
      ≤ (P : ℝ≥0∞) * ((N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam)

/-- **Grinding roots costs a factor `G`, with side sources that ignore the oracle.** For every chained prover that
grinds its roots (`Rate.GrindChain`: pass `j`'s root is one of `G` candidates the earlier passes' coins and the store
fix, picked as any function of pass `j`'s whole oracle family, and pass `j`'s device may depend on the pick) whose
devices' side sources ignore the oracle: the probability of `RateChainPaidRO`'s event is at most
`P·G·(N/2^h + 2^q/2^λ)`.

**The reading.** The pass seed is Fiat–Shamir (`fiat-shamir`), so a prover may hash `G` candidate roots for pass `j − 1`
with pass `j`'s seed and keep the root whose oracle suits it: the window's correct receipts are still paid for pass by
pass, at `G` times the error. The pick here sees the store and the whole family for free, so it can keep a candidate
whose device guessed right, and the factor reaches the guessing term too. A pick that is part of the pass's device,
seeing the store only through counted reads and the family only through counted queries, would leave `N/2^h` alone and
add its queries to `q` (not stated).

Proof: `ResourceAccounting.SecurityProofs.RateGrindPaidRO`, `RatePaidRO` at each pass of the chain that keeps each candidate,
and a union bound over the passes and the candidates.

Frozen (no-function, hash): as `RateAudit`. -/
def RateGrindPaidRO : Prop :=
  ∀ (P N b h ρ S L q r s G lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.GrindChain P N b h ρ S L q r s G),
    (∀ j η c E R O O', (A.device j η c).side E R O = (A.device j η c).side E R O') →
    pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        ∑ j, (∑ t, (A.dev j x.1 x.2).tierReads tier t x.1 (x.2 j).1 (A.oracle j x.1 x.2) +
            (A.dev j x.1 x.2).rebuilt x.1 (x.2 j).1 (A.oracle j x.1 x.2) * b) + P * (S + (L - ρ) + lam) <
          ∑ j, ((A.dev j x.1 x.2).correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card * b)
      ≤ (P : ℝ≥0∞) * (G * ((N : ℝ≥0∞) / 2 ^ h + (2 ^ q : ℝ≥0∞) / 2 ^ lam))

/-- **Grinding roots costs a factor `G`, with any side sources.** `RateGrindPaidRO` for devices whose side sources may
relay receipts, at the same Fiat–Shamir seed (`fiat-shamir`): with `m` words a pass added to the slack, the probability
is at most `P·G·(C(N, m)·2^L/2^(h·m) + 2^q/2^λ)`. Proof: `ResourceAccounting.SecurityProofs.RateGrindPaid`, from `RatePaid`.

Frozen (no-function, hash): as `RateAudit`. -/
def RateGrindPaid : Prop :=
  ∀ (P N b h ρ S L q r s G m lam τ : ℕ) (tier : ℕ → Fin τ) (A : Rate.GrindChain P N b h ρ S L q r s G),
    pr (fun x : Store N b × (Fin P → PassCoins N b h ρ r s) =>
        ∑ j, (∑ t, (A.dev j x.1 x.2).tierReads tier t x.1 (x.2 j).1 (A.oracle j x.1 x.2) +
            (A.dev j x.1 x.2).rebuilt x.1 (x.2 j).1 (A.oracle j x.1 x.2) * b) + P * (S + (L - ρ) + lam + m * b) <
          ∑ j, ((A.dev j x.1 x.2).correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card * b)
      ≤ (P : ℝ≥0∞) * (G * ((N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam))

end ResourceAccounting.Guarantees
