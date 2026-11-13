module

public import Mathlib.LinearAlgebra.Matrix.Rank
public import Definitions.Accounting.RateAuditSketch
public import Definitions.Accounting.RateAuditSketchInteractive
public import Definitions.Accounting.RateAuditSketchLayout

/-!
# The rate audit with a sketch-then-hash tag (trusted)

The game is `Security/Definitions/Accounting/RateAuditSketch.lean`; proofs are in
`Security/Proofs/Accounting/RateAudit/Sketch.lean`, `SketchHonest.lean`, `SketchRank.lean` and `SketchTable.lean`. No
named assumption: the device's bounds (on-chip state `S`, side link `L`, oracle queries `q`, the compute budget `F` at
`c_tag` per tag and `c_enc` per rebuilt sector) and its class, a device that obtains sectors only whole and verbatim, define what the guarantees quantify over;
the claims registry carries the class to a deployed device (`rate-sketch-verbatim`) beside `RateAudit`'s.

* `RateAuditSketch`: in one window, for any digest, an accepted device's footprint reaches the floor except with
  probability `((k − 1)/N)^s + s/2^h + Pr_σ[g < fib_σ] + q·g/|X|`.
* `RateAuditSketchInteractiveRO`: `RateAuditSketch`'s bound when the chip may send and side sources answer whole
  sectors at addresses picked from what it sent.
* `SketchBinds`: signs that bind sectors make the tensor-core sketch's fiber `1`.
* `SketchRankFiber`: signs of rank at least `r` on every sector make it at most `256^(32 − r)`.
* `RateAuditTensorSketch`: the two at the deployed tag, 32-byte sectors and INT8 sketches into INT32 accumulators.
* `RateAuditTensorSketchRank`: the same priced by the signs' rank, `q/2^(8·r) + Pr_σ[∃ u, rank < r]`.

The honest device (`RateSketchHonest`) and the table outside the verbatim class (`RateSketchTable`,
`RateTensorSketchTable`) are lemmas: `Security/Proofs/Accounting/Lemmas/RateAuditSketch.lean`.
-/

@[expose] public section

namespace Accounting.Guarantees

open Accounting Accounting.RateSketch
open Pous (Bits pr)
open scoped ENNReal

/-- **The rate audit with a sketch-then-hash tag, for a device that reads sectors verbatim.** For every digest
`dig σ` of a tile of `m` sectors of a finite type `X` (`w` bits each in the accounting), and every device of
`RateSketch.Prover`'s class (on-chip state of at most `S` bits of held sectors, a side link that carries at most `L`
bits per window, at most `q` oracle queries), over uniform data (a store of `N` tiles and a certified stream of `ρ`
sectors), uniform signs `σ`, a fresh oracle with `h`-bit tags and `s` sampled tiles, and for every compute budget `F`,
at `c_tag` per tag and `c_enc` per rebuilt sector, and every `g`: the probability that the verifier accepts, the run's
compute is within `F`, and its footprint falls short of the floor (`RateSketch.floor`),

  `footprint · c_enc < k·m·w·c_enc − (S + (L − ρ·w))·c_enc − (F − k·c_tag)·w`,

is at most

  `((k − 1)/N)^s + s/2^h + Pr_σ[g < fib_σ] + q·g/|X|`.

**The reading.** Divide by `c_enc` and by `|E| = N·m·w`, with `k = ⌈(1 − ε_s)·N⌉`: an accepted device's footprint is
at least `(1 − μ)·|E|` bits except with probability `ε = (1 − ε_s)^s + s·2^-h + Pr_σ[g < fib_σ] + q·g/|X|`, where
`μ = ε_s + (S + c_free)/|E| + (F − k·c_tag)/(N·m·c_enc)` and `c_free = L − ρ·w` is the side capacity the certified
stream leaves free. Unlike `RateAudit`'s, the floor has no compression slack `λ`: in this class a sector the device did
not obtain is one it does not know at all. The digest enters only through `fib_σ`, the min-entropy that one unread
sector leaves in it (`2^-w·fib_σ` per query is the guess): binding signs give `fib_σ = 1` (`SketchBinds`), and the
signs' chance of not binding is the per-window `Pr_σ[g < fib_σ]`, once, not per query.

**The proof** (`Accounting.SecurityProofs.RateAuditSketch`) follows `RateAudit`'s. A sampled tile passes because the
device queried the oracle at its digest under the stream, or by a guessed tag (`2^-h` per sampled tile); the sample
misses the tiles never queried with chance `((k − 1)/N)^s`. A correct query for a tile one of whose sectors, or one of
the stream's, the run never obtained (held, read, taken in or rebuilt) is a guess: resampling that sector leaves the run
as it is, so at most `g` of its `|X|` values make the query right (`q·g/|X|` over the queries). Otherwise the `k`
queried tiles and the stream are `k·m + ρ` sectors the run obtained: at most `S/w` held, `footprint/w` read, `L/w` over
the link, and the rest rebuilt at `c_enc` each from the `F − k·c_tag` that tagging leaves. As in `RateAudit`, a rebuild
is all-or-nothing, a whole sector at `c_enc`: partial completion is outside the game, and `rate-codec-work`, stated for
codecs with `k ≥ 2` rounds, carries it.

Frozen (no-function, hash): a bound in the random-oracle model over an abstract chip program, which no Lean function of
the rate audit's code computes. -/
def RateAuditSketch : Prop :=
  ∀ (N m ρ h S L q w s k F cTag cEnc g : ℕ) (X Sig Dg : Type) [Fintype X] [DecidableEq X] [Fintype Sig]
    [Fintype Dg] [DecidableEq Dg] (dig : Sig → (Fin m → X) → Dg) (A : RateSketch.Prover N m ρ h S L q w X Sig Dg),
    pr (fun x : Data N m ρ X × Coins N ρ h s X Sig Dg =>
        A.Accepts dig x.2.1.1 x.2.1.2 x.1 x.2.2 ∧ A.compute cTag cEnc x.2.1.1 x.2.1.2 x.1 ≤ F ∧
          A.footprint x.2.1.1 x.2.1.2 x.1 * cEnc < RateSketch.floor m w ρ S L F cTag cEnc k)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + pr (fun σ : Sig => g < fib dig σ) +
        ((q * g : ℕ) : ℝ≥0∞) / Fintype.card X

/-- **The sketch rate audit with interactive side sources that ignore the oracle.** `RateAuditSketch`'s bound, at its
floor and its error, for an `InteractiveProver` (`Security/Definitions/Accounting/RateAuditSketchInteractive.lean`):
its chip may send any number of bits, however they depend on the signs and on what it read, to side sources that answer
each read with a whole sector of the store or of the certified stream, as it is, at an address picked from every bit
sent so far. `L` bounds the sectors that came in over the side link (side sectors and stream sectors together, `w` bits
each); sends are free.

**The reading.** It is `RateAuditInteractiveRO`'s (`RateAudit.lean`) reading for the sketch tag: the rest of the prover
may hold a copy of `E` and of the stream and serve the chip whatever sectors it asks for, keyed by what the chip sent,
with no `2^e` factor. What it serves is still verbatim (`rate-sketch-verbatim`): a source that sends a sketch, or any
function of several sectors, is outside the class, and that is the amplification the verbatim class excludes. The
oracle-free form needs the oracle's seed to reach the chip on a path no side source reads: a source that holds the seed
can compute tags and relay them, which this statement does not price (`RateAuditInteractive`'s relay term at tile
granularity is not stated here).

Proof: `Accounting.SecurityProofs.RateAuditSketchInteractiveRO`: an interactive device is a `RateSketch.Prover` whose
chip keeps what it sent and takes in, over the link, the sector at the address the sources would pick.

Frozen (no-function, hash): as `RateAuditSketch`, a bound in the random-oracle model over an abstract chip program. -/
def RateAuditSketchInteractiveRO : Prop :=
  ∀ (N m ρ h S L q w s k F cTag cEnc g : ℕ) (X Sig Dg : Type) [Fintype X] [DecidableEq X] [Fintype Sig]
    [Fintype Dg] [DecidableEq Dg] (dig : Sig → (Fin m → X) → Dg)
    (A : RateSketch.InteractiveProver N m ρ h S L q w X Sig Dg),
    pr (fun x : Data N m ρ X × Coins N ρ h s X Sig Dg =>
        A.Accepts dig x.2.1.1 x.2.1.2 x.1 x.2.2 ∧ A.compute cTag cEnc x.2.1.1 x.2.1.2 x.1 ≤ F ∧
          A.footprint x.2.1.1 x.2.1.2 x.1 * cEnc < RateSketch.floor m w ρ S L F cTag cEnc k)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + pr (fun σ : Sig => g < fib dig σ) +
        ((q * g : ℕ) : ℝ≥0∞) / Fintype.card X

/-- **Binding signs make the tensor-core sketch's fiber `1`.** For signs whose block on every sector is injective on
INT8 sectors (`SectorsBind`), two contents of one sector that give one sketch, with the rest of the tile fixed, are
equal: their difference changes each coordinate by at most `2·32·128 < 2^32`, so the INT32 accumulators see it
exactly. Proof: `Accounting.SecurityProofs.SketchBinds`.

Frozen (premise): a property of the sketch's definition under a hypothesis on the window's signs, which
`RateAuditTensorSketch` prices as `Pr_σ[¬ SectorsBind σ]`. -/
def SketchBinds : Prop :=
  ∀ (k m : ℕ) (σ : SketchSigns k m), SectorsBind σ → fib (sketch (k := k) (m := m)) σ ≤ 1

/-- **Signs of rank `r` leave the tensor-core sketch's fiber at most `256^(32 − r)`.** For signs whose `k × 32` block
on every sector has rank at least `r` over `ℚ`, at most `256^(32 − r)` contents of one sector give one sketch with the
rest of the tile fixed: two such contents differ by an INT8 difference that the block maps to `0` exactly (the INT32
accumulators see it, as in `SketchBinds`), and `r` independent columns of the block fix the bytes they index once the
other `32 − r` are fixed. Proof: `Accounting.SecurityProofs.SketchRankFiber`.

Frozen (premise): like `SketchBinds`, a property of the sketch under a hypothesis on the window's signs, which
`RateAuditTensorSketchRank` prices as `Pr_σ[∃ u, rank < r]`. -/
def SketchRankFiber : Prop :=
  ∀ (k m r : ℕ) (σ : SketchSigns k m), (∀ u, r ≤ ((Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank) →
    fib (sketch (k := k) (m := m)) σ ≤ 256 ^ (32 - r)

/-- **The rate audit at the deployed sketch-then-hash tag.** `RateAuditSketch` for 32-byte sectors (`w = 256`), uniform
`k_s × n` signs and the tensor-core sketch into INT32 accumulators, at `g = 1` with `SketchBinds`: an accepted device's
footprint reaches the floor except with probability

  `((k − 1)/N)^s + s/2^h + Pr_σ[¬ SectorsBind σ] + q/2^256`.

**The reading.** At `k_s = 64` the signs bind but for the chance that some sector's `64 × 32` block of uniform signs
has a column dependency, a number of the signs' law that is evaluated outside Lean. At `k_s = 16` no block binds, and
the bound to read is `RateAuditSketch`'s at the sketch's largest fiber `g`. Proof:
`Accounting.SecurityProofs.RateAuditTensorSketch`, from the two.

Frozen (no-function, hash): like `RateAuditSketch`, a bound in the random-oracle model over abstract chip programs. -/
def RateAuditTensorSketch : Prop :=
  ∀ (N m ρ h S L q s k F cTag cEnc ks : ℕ)
    (A : RateSketch.Prover N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32))),
    pr (fun x : Data N m ρ Sector × Coins N ρ h s Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)) =>
        A.Accepts sketch x.2.1.1 x.2.1.2 x.1 x.2.2 ∧ A.compute cTag cEnc x.2.1.1 x.2.1.2 x.1 ≤ F ∧
          A.footprint x.2.1.1 x.2.1.2 x.1 * cEnc < RateSketch.floor m 256 ρ S L F cTag cEnc k)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + pr (fun σ : SketchSigns ks m => ¬ SectorsBind σ) +
        (q : ℝ≥0∞) / 2 ^ 256

/-- **The rate audit at the deployed tag, priced by the signs' rank.** `RateAuditSketch` at the tensor-core sketch
with `g = 256^(32 − r)` (`SketchRankFiber`), for every `r ≤ 32`: an accepted device's footprint reaches the floor except
with probability

  `((k − 1)/N)^s + s/2^h + Pr_σ[∃ u, rank_ℚ(block_u σ) < r] + q/2^(8·r)`.

**The reading.** It is the per-pass reading, `q·2^-8r + Pr_σ[∃ u, rank < r]`, as a theorem for the verbatim class.
At `k_s = 16` no block binds (its rank is at most 16), and `r` trades the guess, `2^-8r` per query, against the chance
that some sector's `16 × 32` block of uniform signs has rank below `r`, a number of the signs' law evaluated outside
Lean (`r* = 12`). Outside the verbatim class no bound of this form holds (`RateTensorSketchTable`). A
session's passes and the grinding of their keys are not modeled here. Proof:
`Accounting.SecurityProofs.RateAuditTensorSketchRank`, from `RateAuditSketch` and `SketchRankFiber`.

Frozen (no-function, hash): like `RateAuditSketch`, a bound in the random-oracle model over abstract chip programs. -/
def RateAuditTensorSketchRank : Prop :=
  ∀ (N m ρ h S L q s k F cTag cEnc ks r : ℕ)
    (A : RateSketch.Prover N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32))), r ≤ 32 →
    pr (fun x : Data N m ρ Sector × Coins N ρ h s Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)) =>
        A.Accepts sketch x.2.1.1 x.2.1.2 x.1 x.2.2 ∧ A.compute cTag cEnc x.2.1.1 x.2.1.2 x.1 ≤ F ∧
          A.footprint x.2.1.1 x.2.1.2 x.1 * cEnc < RateSketch.floor m 256 ρ S L F cTag cEnc k)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h +
        pr (fun σ : SketchSigns ks m => ∃ u, ((Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank < r) +
        (q : ℝ≥0∞) / 2 ^ (8 * r)

end Accounting.Guarantees
