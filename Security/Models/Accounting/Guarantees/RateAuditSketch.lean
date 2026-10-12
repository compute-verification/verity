import Definitions.Accounting.RateAuditSketch
import Definitions.Accounting.RateAuditSketchInteractive

/-!
# The rate audit with a sketch-then-hash tag (trusted)

The game is `Security/Definitions/Accounting/RateAuditSketch.lean`; proofs are in
`Security/Proofs/Accounting/RateAudit/Sketch.lean` and `SketchHonest.lean`. No named assumption: the device's bounds
(on-chip state `S`, side link `L`, oracle queries `q`, the compute budget `F` at `c_tag` per tag and `c_enc` per rebuilt
sector) and its class, a device that obtains sectors only whole and verbatim, define what the guarantees quantify over;
the claims registry carries the class to a deployed device (`rate-sketch-verbatim`) beside `RateAudit`'s.

* `RateAuditSketch`: in one window, for any digest, an accepted device's footprint reaches the floor except with
  probability `((k − 1)/N)^s + s/2^h + Pr_σ[g < fib_σ] + q·g/|X|`.
* `RateSketchHonest`: an honest device that reads every sector once is accepted on every coin, with footprint `N·m·w`.
* `RateAuditSketchInteractiveRO`: `RateAuditSketch`'s bound when the chip may send and side sources answer whole
  sectors at addresses picked from what it sent.
* `SketchBinds`: signs that bind sectors make the tensor-core sketch's fiber `1`.
* `RateAuditTensorSketch`: the two at the deployed tag, 32-byte sectors and INT8 sketches into INT32 accumulators.
-/

namespace Accounting.Guarantees

open Accounting Accounting.RateSketch
open Pous (Bits pr)
open scoped ENNReal

/-- **The rate audit with a sketch-then-hash tag, for a device that reads sectors verbatim.** For every digest
`dig σ` of a tile of `m` sectors of a finite type `X` (`w` bits each in the accounting), and every device of
`RateSketch.Device`'s class (on-chip state of at most `S` bits of held sectors, a side link that carries at most `L`
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
the stream's, the run never obtained (held, read, taken in or rebuilt) is a guess: resampling that sector leaves the
run as it is, so at most `g` of its `|X|` values make the query right (`q·g/|X|` over the queries). Otherwise the `k`
queried tiles and the stream are `k·m + ρ` sectors the run obtained: at most `S/w` held, `footprint/w` read, `L/w` over
the link, and the rest rebuilt at `c_enc` each from the `F − k·c_tag` that tagging leaves. As in `RateAudit`, a
rebuild is all-or-nothing, a whole sector at `c_enc`: partial completion (the red team's F27) is outside the game, and
`rate-codec-work`, stated for codecs with `k ≥ 2` rounds, carries it.

Frozen (no-function, hash): a bound in the random-oracle model over an abstract chip program, which no Lean function of
the rate audit's code computes. -/
def RateAuditSketch : Prop :=
  ∀ (N m ρ h S L q w s k F cTag cEnc g : ℕ) (X Sig Dg : Type) [Fintype X] [DecidableEq X] [Fintype Sig]
    [Fintype Dg] [DecidableEq Dg] (dig : Sig → (Fin m → X) → Dg) (A : RateSketch.Device N m ρ h S L q w X Sig Dg),
    pr (fun x : Data N m ρ X × Coins N ρ h s X Sig Dg =>
        A.Accepts dig x.2.1 x.2.2.1 x.1 x.2.2.2 ∧ A.compute cTag cEnc x.2.1 x.2.2.1 x.1 ≤ F ∧
          A.footprint x.2.1 x.2.2.1 x.1 * cEnc < RateSketch.floor m w ρ S L F cTag cEnc k)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + pr (fun σ : Sig => g < fib dig σ) +
        ((q * g : ℕ) : ℝ≥0∞) / Fintype.card X

/-- **The sketch rate audit with interactive side sources that ignore the oracle.** `RateAuditSketch`'s bound, at its
floor and its error, for an `InteractiveDevice` (`Security/Definitions/Accounting/RateAuditSketchInteractive.lean`):
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

Proof: `Accounting.SecurityProofs.RateAuditSketchInteractiveRO`: an interactive device is a `RateSketch.Device` whose
chip keeps what it sent and takes in, over the link, the sector at the address the sources would pick.

Frozen (no-function, hash): as `RateAuditSketch`, a bound in the random-oracle model over an abstract chip program. -/
def RateAuditSketchInteractiveRO : Prop :=
  ∀ (N m ρ h S L q w s k F cTag cEnc g : ℕ) (X Sig Dg : Type) [Fintype X] [DecidableEq X] [Fintype Sig]
    [Fintype Dg] [DecidableEq Dg] (dig : Sig → (Fin m → X) → Dg)
    (A : RateSketch.InteractiveDevice N m ρ h S L q w X Sig Dg),
    pr (fun x : Data N m ρ X × Coins N ρ h s X Sig Dg =>
        A.Accepts dig x.2.1 x.2.2.1 x.1 x.2.2.2 ∧ A.compute cTag cEnc x.2.1 x.2.2.1 x.1 ≤ F ∧
          A.footprint x.2.1 x.2.2.1 x.1 * cEnc < RateSketch.floor m w ρ S L F cTag cEnc k)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + pr (fun σ : Sig => g < fib dig σ) +
        ((q * g : ℕ) : ℝ≥0∞) / Fintype.card X

/-- **Completeness of `RateAuditSketch`: an honest device that reads every sector once is accepted on every coin.**
For every digest and every `N`, `m`, `ρ`, `h` and `w` there is a device that holds nothing on chip, whose link carries
exactly the certified stream's `ρ` sectors (`L = ρ·w`) and which makes `N` oracle queries (a
`Device N m ρ h 0 (ρ·w) N w`), that is accepted at every signs, oracle, data and sample, so with probability 1; its
footprint is the store, `N·m·w` bits, and its compute is `N·c_tag`, with nothing rebuilt. It takes in the stream, reads
every sector once and queries the oracle at each tile's digest under the stream. Against `RateAuditSketch` at `S = 0`,
`L = ρ·w`, `q = N` and `F = N·c_tag`, the honest `N·m·w` sits above the floor `k·m·w − (N − k)·c_tag·w/c_enc` by the
sample's slack `(N − k)·m·w` and the tagging it leaves unspent. Proof: `Accounting.SecurityProofs.RateSketchHonest`.

Frozen (completeness): the honest device's acceptance, which shows that `RateAuditSketch`'s event is not vacuous. -/
def RateSketchHonest : Prop :=
  ∀ (N m ρ h w : ℕ) (X Sig Dg : Type) [Inhabited X] (dig : Sig → (Fin m → X) → Dg),
    ∃ A : RateSketch.Device N m ρ h 0 (ρ * w) N w X Sig Dg,
      ∀ (cTag cEnc : ℕ) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) (s : ℕ) (I : Fin s → Fin N),
        A.Accepts dig σ O D I ∧ A.footprint σ O D = N * m * w ∧ A.compute cTag cEnc σ O D = N * cTag

/-- **Binding signs make the tensor-core sketch's fiber `1`.** For signs whose block on every sector is injective on
INT8 sectors (`SectorsBind`), two contents of one sector that give one sketch, with the rest of the tile fixed, are
equal: their difference changes each coordinate by at most `2·32·128 < 2^32`, so the INT32 accumulators see it
exactly. Proof: `Accounting.SecurityProofs.SketchBinds`.

Frozen (premise): a property of the sketch's definition under a hypothesis on the window's signs, which
`RateAuditTensorSketch` prices as `Pr_σ[¬ SectorsBind σ]`. -/
def SketchBinds : Prop :=
  ∀ (k m : ℕ) (σ : SketchSigns k m), SectorsBind σ → fib (sketch (k := k) (m := m)) σ ≤ 1

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
    (A : RateSketch.Device N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32))),
    pr (fun x : Data N m ρ Sector × Coins N ρ h s Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)) =>
        A.Accepts sketch x.2.1 x.2.2.1 x.1 x.2.2.2 ∧ A.compute cTag cEnc x.2.1 x.2.2.1 x.1 ≤ F ∧
          A.footprint x.2.1 x.2.2.1 x.1 * cEnc < RateSketch.floor m 256 ρ S L F cTag cEnc k)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + pr (fun σ : SketchSigns ks m => ¬ SectorsBind σ) +
        (q : ℝ≥0∞) / 2 ^ 256

end Accounting.Guarantees
