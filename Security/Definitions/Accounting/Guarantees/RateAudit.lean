import Definitions.Accounting.RateAudit
import Definitions.Accounting.RateAuditJoint
import Definitions.Accounting.RateAuditInteractive

/-!
# The rate audit: what an accepted device held and read (trusted)

The game is `Security/Definitions/Accounting/RateAudit.lean`, its embedding in the device model
`Security/Definitions/Accounting/RateAuditJoint.lean`; proofs are in `Security/Proofs/Accounting/RateAudit/`. No named
assumption: the device's bounds (on-chip state `S`, side link `L`, oracle queries `q`, the compute budget `F` at `c_tag`
per tag and `c_enc` per rebuilt word) define the class of devices or runs the guarantees quantify over, and the claims
registry names, for a deployed device, the measurement or conjecture that carries each to it (`bandwidth-s-chip`,
`bandwidth-side-cap`, `rate-codec-work` for `c_enc`, `rate-device` for the model, `random-oracle` for the tag,
`bandwidth-instantiation` for the stored words).

* `RateAudit`: in one window, an accepted device's footprint (residency) and ingress (bandwidth) reach the floor except
  with probability `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ`.
* `RateHonest`: an honest device that reads every word once is accepted on every coin, with footprint and ingress
  exactly `N·b`.
* `RateAuditWindows`: over `n` windows on one store, the time-averaged footprint.
* `RateOfAudit`: in a joint run, the bandwidth account's reads, with the rebuilds charged as compute, so a compute
  account that did `W` of work leaves `F·T − W` for them.
* `RateAuditInteractiveRO`: `RateAudit`'s bound for a chip that sends to side sources answering any function of the
  store, the certified stream and what it sent (`Definitions/Accounting/RateAuditInteractive.lean`), with no `2^e`.
* `RateAuditInteractive`: the same with answers that may depend on the oracle (relayed tags), `m·b` less.
* `RateOfAuditInteractive` and `RateOfAuditInteractiveRO`: `RateOfAudit` for a joint run whose bandwidth audit an
  interactive device plays.
-/

namespace Accounting.Guarantees

open Accounting Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal

/-- **The rate audit in the random-oracle model.** For every device with on-chip state of `S` bits whose side link
carries at most `L` bits per window and which makes at most `q` oracle queries, over a uniform store of `N` words of
`b` bits, a uniform certified stream of `ρ` bits, a fresh oracle with `h`-bit tags and `s` sampled words, and for every
compute budget `F`, at `c_tag` per tag and `c_enc` per rebuilt word: the probability that the verifier accepts, the
run's compute is within `F`, and its footprint falls short of the floor (`Rate.floor`),

  `footprint · c_enc < k·b·c_enc − (S + (L − ρ) + λ)·c_enc − (F − k·c_tag)·b`,

is at most

  `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ`.

**The reading.** Divide by `c_enc` and by `|E| = N·b`, with `k = ⌈(1 − ε_s)·N⌉`: an accepted device's footprint is at
least `(1 − μ)·|E|` bits, except with probability `ε = (1 − ε_s)^s + s·2^-h + C(q, k)·2^-λ`, where

  `μ = ε_s + (S + c_free + λ)/|E| + (F − tagging)/(N·c_enc)`,

`c_free = L − ρ` is the side capacity the certified stream leaves free and `tagging = k·c_tag`. Against the
actual-circuit model's B3 (at `e = 0`), whose chip reads at least `k·b − M_chip − c − λ` bits from HBM with
`c = β_side·(T + RTT)`: `S` is `M_chip`, `L` is `c` with no `2^e` factor, the certified stream (certified ingress) takes
`ρ` of it, and DMA into HBM during the window is side traffic. The footprint is the distinct device-memory bits the
window read, so device memory held at least that much of the store's information (residency; the on-chip `S` is on the
device too and counts against reads, not residency); the ingress counts every read, so it is at least the footprint
(bandwidth). `(F − tagging)/(N·c_enc)` is `μ_rebuild`: with `x = N·c_enc/F`, the encode time of one pass's data in units
of the window's compute, it is `(1 − τ)/x` when tagging takes the fraction `τ` of `F`, and in a joint run `F` is the
compute residual (`RateOfAudit`). The bound holds at every `F`; a device whose compute is within `F` on every run has
the conjunct always.

**The proof** (`Accounting.SecurityProofs.RateAudit`) generalizes `Bandwidth.Guarantees.AuditRO`'s. A sampled word
passes either because the device queried the oracle at its contents under the stream, or by a guessed tag (`2^-h` per
sampled word); the sample misses the words never queried with chance `((k − 1)/N)^s`. A device that queried `k` words
under the stream knows those `k·b` bits and all `ρ` bits of the stream at its queries, and it had them from its state,
the device memory at its footprint, the link (free side bits and stream bits together, at most `L`) and its rebuilt
words, plus which `k` of its queries named them: compressing the store and the stream together (`C(q, k)·2^-λ`) leaves
it at least `k·b + ρ − S − L − λ` bits of footprint and rebuilt words, and a rebuilt word costs `c_enc` of the compute
`F − k·c_tag` that tagging leaves. The stream is what makes `L − ρ`, not `L`, the link's share: its bits are fresh,
so the link's bits that carried them carried nothing about the store. Its completeness is `RateHonest`.

The game's `rebuild j` is all-or-nothing: it returns the whole word at `c_enc`. Completing part of a word from the rest
(at one codec round with the payload known, a lattice completes a word's bottom third far below its per-bit price) is
outside this game; `RateAuditPartial` (`RateAuditPartial.lean`) puts it inside, at this floor and error, under the named
assumption `Accounting.Assumptions.WorkHardness` (`rate-codec-work` in partial-completion form, for codecs with `k ≥ 2`
rounds).

Frozen (no-function, hash): a bound in the random-oracle model over an abstract chip program, which no Lean function of
the rate audit's code computes. -/
def RateAudit : Prop :=
  ∀ (N b h ρ S L q s k F cTag cEnc lam : ℕ) (A : Rate.Device N b h ρ S L q),
    pr (fun x : Store N b × Coins N b h ρ s =>
        A.Accepts x.1 x.2.1.1 x.2.1.2 x.2.2 ∧ A.compute cTag cEnc x.1 x.2.1.1 x.2.1.2 ≤ F ∧
          A.footprint x.1 x.2.1.1 x.2.1.2 * cEnc < floor b ρ S L F cTag cEnc k lam)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam

/-- **Completeness of `RateAudit`: an honest device that reads every word once is accepted on every coin.** For every
`N`, `b`, `h` and `ρ` there is a device with no on-chip state and no free side sources, whose link carries exactly the
certified stream (`L = ρ`) and which makes `N` oracle queries (a `Device N b h ρ 0 ρ N`), that is accepted on every
store, stream, oracle and sample, so with probability 1; its footprint and its ingress are exactly the store, `N·b`
bits, and its compute is `N·c_tag`, with nothing rebuilt. It reads the stream, then reads every word once and queries
the oracle at it under the stream. Against `RateAudit` at `S = 0`, `L = ρ` (so `c_free = 0`), `q = N` and
`F = N·c_tag`, the honest `N·b` sits above the floor `k·b − λ − (N − k)·c_tag·b/c_enc` by the sample's slack
`(N − k)·b`, about `ε_s·N·b`, plus `λ` and the tagging the sample's slack leaves unspent. Proof:
`Accounting.SecurityProofs.RateHonest`.

Frozen (completeness): the honest device's acceptance, which shows that `RateAudit`'s event is not vacuous. -/
def RateHonest : Prop :=
  ∀ N b h ρ : ℕ, ∃ A : Rate.Device N b h ρ 0 ρ N,
    ∀ (cTag cEnc : ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) (s : ℕ) (I : Fin s → Fin N),
      A.Accepts E R O I ∧ A.footprint E R O = N * b ∧ A.ingress E R O = N * b ∧ A.compute cTag cEnc E R O = N * cTag

/-- **The rate audit over `n` windows: time-averaged residency.** Over a uniform store and `n` windows, each with its
own uniform stream, oracle and sample, played by devices of `RateAudit`'s class (`Rate.Windows`: window `i`'s device
and its layout any function of the other windows' coins, earlier or later), and for every `m`: `m` times the
probability that every window is accepted with its compute within `F` while the footprints, summed over the windows,
fall short of `n + 1 − m` floors,

  `∑ᵢ footprintᵢ · c_enc < (n + 1 − m) · floor`,

is at most `n` times `RateAudit`'s error. A sum that short leaves at least `m` windows below the floor, and the expected
number of accepted windows below it is at most `n·ε`.

**The reading.** At `m = 1`: the average footprint over the windows is at least `RateAudit`'s `(1 − μ)·|E|` except with
probability `n·ε`. At `m = δ·n`: it is at least `(1 − δ)·(1 − μ)·|E|` except with probability `ε/δ`, whatever `n` is,
so a time average at a slack of `δ` costs no union over the windows. Proof:
`Accounting.SecurityProofs.RateAuditWindows`, from `RateAudit` at each window, at each value of the other windows'
coins.

Frozen (no-function, hash): like `RateAudit`, a bound in the random-oracle model over abstract chip programs. -/
def RateAuditWindows : Prop :=
  ∀ (n N b h ρ S L q s k F cTag cEnc lam m : ℕ) (A : Windows n N b h ρ S L q s),
    (m : ℝ≥0∞) * pr (fun x : Store N b × (Fin n → Coins N b h ρ s) =>
        (∀ i, (A.dev i x.2).Accepts x.1 (x.2 i).1.1 (x.2 i).1.2 (x.2 i).2 ∧
          (A.dev i x.2).compute cTag cEnc x.1 (x.2 i).1.1 (x.2 i).1.2 ≤ F) ∧
        ∑ i, (A.dev i x.2).footprint x.1 (x.2 i).1.1 (x.2 i).1.2 * cEnc < (n + 1 - m) * floor b ρ S L F cTag cEnc k lam)
      ≤ (n : ℝ≥0∞) * ((((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam)

/-- **The rate audit as a joint run's bandwidth guarantee, with the rebuilds charged as compute.** For any run on any
device whose bandwidth audit realizes the rate audit's game at price `p`, `c_tag` per tag and `c_enc` per rebuilt word
(`Accounting.RealizesRate`: the game's device within the device model's limits over the bandwidth window, its ingress
charged to the bandwidth account and its tags and rebuilds to accounts other than the compute account), at any value
of the compute and capacity audits' coins, and for any `W`: the probability over the bandwidth coins that the
bandwidth audit accepts, the compute account did at least `W` of work over the compute window, and the bandwidth
account read fewer bits of device memory than the floor at the compute budget `F·T − W`,

  `read · c_enc < floor` at `S = S_chip`, `L = ∑ cap_i · T_bw`, `F = F·T − W`,

is at most `RateAudit`'s error at `q = rate_oracle · T_bw`, where `F·T` is the device's priced work over the compute
window and `T_bw` the bandwidth window's length. A bandwidth window that spans the deadline and a round trip,
`T_bw = T + RTT`, makes `L` the actual-circuit model's `c = β_side·(T + RTT)`.

**The reading.** Every op is charged once, to one account (`Step.Fits`), so the compute account's `W` leaves at most
`F·T − W` for every other account, the rebuilds and tags among them: `μ_rebuild = (F·T − W − tagging)/(N·c_enc)`, which
is `(1 − τ)/x` with `τ = (W + tagging)/(F·T)` and `x = N·c_enc/(F·T)`. The compute audit's certificate is what makes `W`
hold: a per-run compute guarantee bounding the probability that the compute audit accepts with the compute account's
work below `W` by `ε_W` (lane `ra-per-run`'s `ComputeSoundPerRun`) gives, by a union bound, the probability that both
audits accept and the bandwidth account read short of the floor at most `ε + ε_W`. Unlike `BandwidthOfAudit`, this is
a per-run reading: the event is about the reads at the coins drawn. A `RealizesRate` for a deployed run must put the
deadline and a round trip, `T + RTT`, inside `T_bw`, and every rebuild in the bandwidth window must fall in the compute
window's work of the other accounts (`compute_le`). Proof: `Accounting.SecurityProofs.RateOfAudit`, from `RateAudit`.

Frozen (no-function, hash): it carries `RateAudit`, a bound in the random-oracle model over an abstract chip program, to
the bandwidth audit of an abstract run of the device model whose verdicts are a field of the run, not to a Lean
function. -/
def RateOfAudit : Prop :=
  ∀ (N b h ρ s k cTag cEnc lam W : ℕ) (p : Price) (ι α ΩW ΩS : Type) [Fintype ι] [Fintype α] [DecidableEq α]
    (d : Device ι) (r : Run d α ΩW (RateCoins N b h ρ s) ΩS), RealizesRate p cTag cEnc r →
    ∀ (x : ΩW) (s' : ΩS),
      pr (fun y : RateCoins N b h ρ s =>
          r.accepts .bandwidth (x, y, s') ∧ W ≤ r.work p (x, y, s') ∧
            r.read (x, y, s') * cEnc < floor b ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len)
              (d.priced p * (r.win .compute).len - W) cTag cEnc k lam)
        ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h +
          (((d.rate .oracle * (r.win .bandwidth).len).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **The rate audit with interactive side sources that ignore the oracle.** `RateAudit`'s bound, at its floor and its
error, for an `InteractiveDevice` (`Security/Definitions/Accounting/RateAuditInteractive.lean`): its chip may send any
number of bits, however they depend on the seed, to side sources that answer each read as any function of the store,
the certified stream and every bit sent so far, but not of the oracle. `L` bounds what came in over the side link (side
bits and stream bits together); sends are free. The rebuild charge `(F − k·c_tag)·b` and the stream's `ρ` are
`RateAudit`'s.

**The reading.** In the actual-circuit model, the rest of the prover answers the chip as such a source: localization's
table of replies, keyed by what the chip sent, is a side source addressed by what was sent, and the chip reads only its
own entry. When nothing the chip sends depends on the seed (`e = 0`), the rest can't evaluate the tag at the fresh seed,
so its answers ignore the oracle. The certified ingress of the HBM link (and the residency it implies) is then
`RateAudit`'s `(1 − μ)·|E|`, with `L` the side links' signals in the window, `β_side·(T + RTT)`, and no `2^e` factor:
the compression behind `RateAudit` encodes the bits the chip read, not what its sources hold, so a table of any size
costs only what the chip read of it. A source may see the certified stream as well (the host that sends it, a tap on the
link): the stream is fresh, so whatever the chip learns of it still came in over the link.

Proof: `Accounting.SecurityProofs.RateAuditInteractiveRO`: an interactive device is a device whose side source,
addressed by what was sent and where, may depend on the store and the stream, and `RateAudit`'s argument holds for such
a device.

Frozen (no-function, hash): as `RateAudit`, a bound in the random-oracle model over an abstract chip program. -/
def RateAuditInteractiveRO : Prop :=
  ∀ (N b h ρ S L q s k F cTag cEnc lam : ℕ) (A : Rate.InteractiveDevice N b h ρ S L q),
    (∀ E R O O', A.side E R O = A.side E R O') →
    pr (fun x : Store N b × Coins N b h ρ s =>
        A.Accepts x.1 x.2.1.1 x.2.1.2 x.2.2 ∧ A.compute cTag cEnc x.1 x.2.1.1 x.2.1.2 ≤ F ∧
          A.footprint x.1 x.2.1.1 x.2.1.2 * cEnc < floor b ρ S L F cTag cEnc k lam)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam

/-- **The rate audit with interactive side sources, in the random-oracle model.** For every `InteractiveDevice` of
`RateAudit`'s class, whose side sources may answer as any function of the store, the certified stream, the oracle and
every bit sent so far, and for every `m`: the probability that the verifier accepts, the run's compute is within `F`,
and its footprint falls short of the floor at `k − m` words,

  `footprint · c_enc < (k − m)·b·c_enc − (S + (L − ρ) + λ)·c_enc − (F − (k − m)·c_tag)·b`,

is at most

  `((k − 1)/N)^s + C(N, m)·2^L/2^(h·m) + C(q, k − m)/2^λ`.

**The reading.** A source that sees the seed (a peer holding a copy of `E`, or the rest of the prover once the chip has
sent the seed, `e > 0`) can compute tags and relay them, so `m` relayed words cost `m·b` of the floor, and the `m` tags
the chip didn't compute leave `m·c_tag` more of `F` for rebuilds: the floor is `RateAudit`'s at `k − m` words. The
tag-guessing term `s/2^h` becomes `C(N, m)·2^L/2^(h·m)`, the chance that `m` words' tags are determined by the bits
that came in over the link. That term has `L`, not `L − ρ`: a relayed tag absorbs the stream at its source, so the chip
needn't read the stream for it. As for `RateAudit`, `L` is the side links' signals in the window, with no `2^e`, and
the rebuild charge and `ρ` are `RateAudit`'s at the floor's `k − m` words; at `m = 0` the bound is weaker than
`RateAuditInteractiveRO`'s, which is the one to use when the answers ignore the oracle.

Proof: `Accounting.SecurityProofs.RateAuditInteractive`, `Bandwidth.Guarantees.AuditRelay`'s argument on `RateAudit`'s
game: a sampled word passes because the chip queried it, or its tag was relayed; `m` relayed tags fix `h·m` bits of the
oracle from at most `L` bits read.

Frozen (no-function, hash): as `RateAudit`, a bound in the random-oracle model over an abstract chip program. -/
def RateAuditInteractive : Prop :=
  ∀ (N b h ρ S L q s k m F cTag cEnc lam : ℕ) (A : Rate.InteractiveDevice N b h ρ S L q),
    pr (fun x : Store N b × Coins N b h ρ s =>
        A.Accepts x.1 x.2.1.1 x.2.1.2 x.2.2 ∧ A.compute cTag cEnc x.1 x.2.1.1 x.2.1.2 ≤ F ∧
          A.footprint x.1 x.2.1.1 x.2.1.2 * cEnc < floor b ρ S L F cTag cEnc (k - m) lam)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (N.choose m : ℝ≥0∞) * 2 ^ L / 2 ^ (h * m)
        + ((q.choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **The rate audit with interactive side sources as a joint run's bandwidth guarantee.** `RateOfAudit` for a run
whose bandwidth audit an interactive device plays (`Accounting.RealizesRateInteractive`), at `RateAuditInteractive`'s
floor and error: for any `m`, the probability over the bandwidth coins that the bandwidth audit accepts, the compute
account did at least `W` of work over the compute window, and the bandwidth account read fewer bits of device memory
than the floor at `k − m` words and the compute budget `F·T − W`,

  `read · c_enc < floor` at `S = S_chip`, `L = ∑ cap_i · T_bw`, `F = F·T − W`, `k − m` words,

is at most `((k − 1)/N)^s + C(N, m)·2^L/2^(h·m) + C(q, k − m)/2^λ` at `q = rate_oracle · T_bw`. When the device's side
sources ignore the oracle at every value of the other audits' coins, `RateOfAudit`'s bound holds as it is
(`RateOfAuditInteractiveRO`).

**The reading.** As `RateOfAudit`'s: the compute account's `W` leaves at most `F·T − W` for the rebuilds and the tags,
and the compute audit's per-run certificate makes `W` hold. Proof: `Accounting.SecurityProofs.RateOfAuditInteractive`,
from `RateAuditInteractive`.

Frozen (no-function, hash): as `RateOfAudit`. -/
def RateOfAuditInteractive : Prop :=
  ∀ (N b h ρ s k m cTag cEnc lam W : ℕ) (p : Price) (ι α ΩW ΩS : Type) [Fintype ι] [Fintype α] [DecidableEq α]
    (d : Device ι) (r : Run d α ΩW (RateCoins N b h ρ s) ΩS), RealizesRateInteractive p cTag cEnc r →
    ∀ (x : ΩW) (s' : ΩS),
      pr (fun y : RateCoins N b h ρ s =>
          r.accepts .bandwidth (x, y, s') ∧ W ≤ r.work p (x, y, s') ∧
            r.read (x, y, s') * cEnc < floor b ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len)
              (d.priced p * (r.win .compute).len - W) cTag cEnc (k - m) lam)
        ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s +
            (N.choose m : ℝ≥0∞) * 2 ^ (∑ i, d.cap i * (r.win .bandwidth).len) / 2 ^ (h * m) +
          (((d.rate .oracle * (r.win .bandwidth).len).choose (k - m) : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **The rate audit with interactive side sources that ignore the oracle, as a joint run's bandwidth guarantee.**
`RateOfAudit`'s bound, as it is, for a run whose bandwidth audit an interactive device plays
(`Accounting.RealizesRateInteractive`) whose side sources' answers, at every value of the other audits' coins, don't
depend on the oracle. Proof: `Accounting.SecurityProofs.RateOfAuditInteractiveRO`, from `RateAuditInteractiveRO`.

Frozen (no-function, hash): as `RateOfAudit`. -/
def RateOfAuditInteractiveRO : Prop :=
  ∀ (N b h ρ s k cTag cEnc lam W : ℕ) (p : Price) (ι α ΩW ΩS : Type) [Fintype ι] [Fintype α] [DecidableEq α]
    (d : Device ι) (r : Run d α ΩW (RateCoins N b h ρ s) ΩS) (real : RealizesRateInteractive p cTag cEnc r),
    (∀ (x : ΩW) (s' : ΩS) E R O O', (real.dev (x, s')).side E R O = (real.dev (x, s')).side E R O') →
    ∀ (x : ΩW) (s' : ΩS),
      pr (fun y : RateCoins N b h ρ s =>
          r.accepts .bandwidth (x, y, s') ∧ W ≤ r.work p (x, y, s') ∧
            r.read (x, y, s') * cEnc < floor b ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len)
              (d.priced p * (r.win .compute).len - W) cTag cEnc k lam)
        ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h +
          (((d.rate .oracle * (r.win .bandwidth).len).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam

end Accounting.Guarantees
