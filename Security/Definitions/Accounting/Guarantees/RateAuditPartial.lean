import Definitions.Accounting.RateAuditPartial
import Definitions.Accounting.Assumptions.RateAudit

/-!
# The rate audit with partial completion: what an accepted device held and read (trusted)

The game is `Security/Definitions/Accounting/RateAuditPartial.lean`: `RateAudit`'s, with `rebuild` replaced by
`complete j x g`, which obtains any `x` of word `j`'s `b` bits at `cost x`. Proofs are in
`Security/Proofs/Accounting/RateAudit/Partial*.lean`. The rebuild term rests on the named assumption
`Accounting.Assumptions.WorkHardness b rounds c_root cost` (`rate-codec-work` in partial-completion form: `x` bits
cost at least `x/b` of a word's `rounds · c_root`, `Definitions/Accounting/Assumptions/RateAudit.lean`), and in a joint run
on `Accounting.Assumptions.RateComputeAdditive` (W: the compute account's work and the game's completions share no
work).

* `RateAuditPartial`: `RateAudit`'s bound, at its floor with `c_enc = rounds · c_root` and its error, for a device that
  may keep part of a word and complete the rest.
* `RatePartialHonest`: the honest device of `RateHonest`, accepted on every coin, completing nothing.
* `RatePartialRebuild`: on every run, the bits completed cost `c_word / b` each (the remote audit's route
  `Accounting.Remote.Run.Rebuild` at `δ = 0`).
* `RateOfAuditPartial`: in a joint run, the bandwidth account's reads, with the completions charged as compute through
  W, so a compute account that did `W` of work leaves `F·T − W` for them.
* `RateOfAuditPartialPasses`: the same with the rebuild term in passes,
  `μ_rebuild ≤ (1 − τ)·d/(rounds·x₁) ≤ d/(rounds·x₁)`.

`rounds` is the codec's round count, the `k` of `WorkHardness(p, k)` and of `μ_rebuild ≤ d/(k·x₁)` in the claims; the
statements' `k` is `RateAudit`'s, the `⌈(1 − ε_s)·N⌉` words the sample needs.

**At one round (`rounds = 1`) these say nothing about a deployed codec.** They hold at every `rounds`, but their
hypothesis `WorkHardness b 1 c_root cost` is false for Sloth at one round with the payload known, at every width (a
2-dimensional lattice completes a word's bottom third at 0.003–0.025 of a root against the per-bit 0.32–0.33). They
carry a deployed codec at `k ≥ 2` rounds, where `WorkHardness` is the residual conjecture.

**The price is a parameter, and `k = 3` is the stated instance** (`WorkHardness b 3 c_root cost`), the deployed codec's:
Sloth over `p = 2^1024 − 105`, masked with a public XOR `σ_K` between rounds (as
ruled), its price justified by no completion skipping the inner root. The mask changes no statement:
`WorkHardness` reads the codec only through `b`, `rounds` and `c_root`. Unmasked at `k = 2`, 512 bits rest on the
reduction's operation-count floor, an estimate, not a measurement.
-/

namespace Accounting.Guarantees

open Accounting Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal

/-- **The rate audit with partial completion, in the random-oracle model.** For every partial device with on-chip
state of `S` bits whose side link carries at most `L` bits per window and which makes at most `q` oracle queries, over
a uniform store of `N` words of `b` bits, a uniform certified stream of `ρ` bits, a fresh oracle with `h`-bit tags and
`s` sampled words, for every compute budget `F` at `c_tag` per tag and `cost x` per completion of `x` bits, if
completions are as hard as `WorkHardness b rounds c_root cost` says (`x` bits cost at least `x/b` of
`c_word = rounds · c_root`): the probability that the verifier accepts, the run's compute is within `F`, and its
footprint falls short of `RateAudit`'s floor at `c_enc = c_word`,

  `footprint · c_word < k·b·c_word − (S + (L − ρ) + λ)·c_word − (F − k·c_tag)·b`,

is at most `RateAudit`'s error

  `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ`.

**The reading.** `RateAudit`'s, with `μ_rebuild = (F − tagging)/(N·c_word)`: a device that keeps the top `b − x` bits of
a word and completes the bottom `x` pays `cost x ≥ (x/b)·c_word` for them, so per bit completed it pays what a whole
rebuild pays per bit, and partial completion buys it nothing over rebuilding. At the stated three rounds of Sloth over
`p = 2^512 − 569`, `c_word = 3·c_root` and a completed bit costs `3·c_root/512` (at two rounds `2·c_root/512`); at one
round the hypothesis is false and the bound is about no deployed codec.

**The proof** (`Accounting.SecurityProofs.RateAuditPartial`) is `RateAudit`'s with completions in place of rebuilds. The
compression encodes, for a completion of `x` bits, those `x` bits (`RateAudit`'s encodes `b` per rebuilt word), so a
run that queried `k` words had its state, footprint, link bits and completed bits, at least `k·b + ρ − λ` of them
except on a `C(q, k)·2^-λ` fraction of stores and streams; and `WorkHardness` turns the completions' compute, at most
`F − k·c_tag`, into at most `(F − k·c_tag)·b/c_word` completed bits.

Frozen (no-function, hash): a bound in the random-oracle model over an abstract chip program, which no Lean function of
the rate audit's code computes. -/
def RateAuditPartial : Prop :=
  ∀ (N b h ρ S L q s k F cTag rounds cRoot lam : ℕ) (cost : ℕ → ℕ) (A : Rate.PartialDevice N b h ρ S L q),
    Assumptions.WorkHardness b rounds cRoot cost →
    pr (fun x : Store N b × Coins N b h ρ s =>
        A.Accepts x.1 x.2.1.1 x.2.1.2 x.2.2 ∧ A.compute cTag cost x.1 x.2.1.1 x.2.1.2 ≤ F ∧
          A.footprint x.1 x.2.1.1 x.2.1.2 * (rounds * cRoot) < floor b ρ S L F cTag (rounds * cRoot) k lam)
      ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h + (q.choose k : ℝ≥0∞) / 2 ^ lam

/-- **Completeness of `RateAuditPartial`: an honest device that reads every word once is accepted on every coin.** For
every `N`, `b`, `h` and `ρ` there is a partial device with no on-chip state and no free side sources, whose link
carries exactly the certified stream (`L = ρ`) and which makes `N` oracle queries, that is accepted on every store,
stream, oracle and sample; its footprint and its ingress are exactly the store, `N·b` bits, it completes nothing, and
its compute is `N·c_tag` at every price of a completion. It is `RateHonest`'s device. Proof:
`Accounting.SecurityProofs.RatePartialHonest`.

Frozen (completeness): the honest device's acceptance, which shows that `RateAuditPartial`'s event is not vacuous. -/
def RatePartialHonest : Prop :=
  ∀ N b h ρ : ℕ, ∃ A : Rate.PartialDevice N b h ρ 0 ρ N,
    ∀ (cTag : ℕ) (cost : ℕ → ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) (s : ℕ) (I : Fin s → Fin N),
      A.Accepts E R O I ∧ A.footprint E R O = N * b ∧ A.ingress E R O = N * b ∧ A.completed E R O = 0 ∧
        A.compute cTag cost E R O = N * cTag

/-- **The rebuild route: a partial device pays `c_word / b` per bit it completes.** For every partial device, if
completions are as hard as `WorkHardness b rounds c_root cost` says, then on every store, stream and oracle the bits
the device completed, at `c_word = rounds · c_root` per `b` of them, cost at most its completions' compute:

  `completed · c_word ≤ b · work`.

**The reading.** It is the remote audit's route `Accounting.Remote.Run.Rebuild` at `δ = 0`
(`rebuilt · c_word ≤ b · rebuildWork`, the node-level guarantee) for a run whose node completes
words as a partial device does: its rebuilt bits (the correct receipts' share) are at most the device's completed bits,
however it splits them into partial words, and its rebuild work covers the completions' compute. `RateAuditPartial` uses it to turn the compute
`F − k·c_tag` into at most `(F − k·c_tag)·b/c_word` completed bits. Proof: `Accounting.SecurityProofs.RatePartialRebuild`,
by induction on the chip program, one completion at a time.

Frozen (no-function): a bound on every run of an abstract chip program, which no Lean function of the rate audit's
code computes. -/
def RatePartialRebuild : Prop :=
  ∀ (N b h ρ S L q rounds cRoot : ℕ) (cost : ℕ → ℕ) (A : Rate.PartialDevice N b h ρ S L q),
    Assumptions.WorkHardness b rounds cRoot cost →
    ∀ (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ),
      A.completed E R O * (rounds * cRoot) ≤ b * A.work cost E R O

/-- **The rate audit with partial completion as a joint run's bandwidth guarantee, the adversary's idle compute
charged through W.** For any run on any device whose bandwidth audit a partial device plays
(`Accounting.RealizesRatePartial`: the game's device within the device model's limits over the bandwidth window, its
ingress charged to the bandwidth account), at any value of the compute and capacity audits' coins, if completions are
as hard as `WorkHardness b rounds c_root cost` says and the compute account's work and the game's compute add
(`RateComputeAdditive`, W), then for any `W`: the probability over the bandwidth coins that the bandwidth audit accepts, the
compute account did at least `W` of work over the compute window, and the bandwidth account read fewer bits of device
memory than the floor at the compute budget `F·T − W`,

  `read · c_word < floor` at `S = S_chip`, `L = ∑ cap_i · T_bw`, `F = F·T − W`, `c_enc = c_word = rounds · c_root`,

is at most `RateAudit`'s error at `q = rate_oracle · T_bw`, where `F·T` is the device's priced work over the compute
window.

**The reading.** W says the compute audit's certified work is not available to complete words: whatever the device
completes in the window, it completes with the idle compute `F·T − W`. So `μ_rebuild = (F·T − W − tagging)/(N·c_word)`,
`(1 − τ)/(rounds·x)` with `τ = (W + tagging)/(F·T)` and `x = N·c_root/(F·T)`, the one-round encode time of the store in
units of the window's compute. The compute audit's per-run certificate makes `W` hold, as in `RateOfAudit`.
`RateOfAuditPartialPasses` restates the term in passes. Proof: `Accounting.SecurityProofs.RateOfAuditPartial`, from
`RateAuditPartial`.

Frozen (no-function, hash): it carries `RateAuditPartial`, a bound in the random-oracle model over an abstract chip
program, to the bandwidth audit of an abstract run of the device model whose verdicts are a field of the run, not to a
Lean function. -/
def RateOfAuditPartial : Prop :=
  ∀ (N b h ρ s k cTag rounds cRoot lam W : ℕ) (cost : ℕ → ℕ) (p : Price) (ι α ΩW ΩS : Type) [Fintype ι] [Fintype α]
    (d : Device ι) (r : Run d α ΩW (RateCoins N b h ρ s) ΩS) (real : RealizesRatePartial r),
    Assumptions.WorkHardness b rounds cRoot cost → Assumptions.RateComputeAdditive p cTag cost real →
    ∀ (x : ΩW) (s' : ΩS),
      pr (fun y : RateCoins N b h ρ s =>
          r.accepts .bandwidth (x, y, s') ∧ W ≤ r.work p (x, y, s') ∧
            r.read (x, y, s') * (rounds * cRoot) < floor b ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len)
              (d.priced p * (r.win .compute).len - W) cTag (rounds * cRoot) k lam)
        ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h +
          (((d.rate .oracle * (r.win .bandwidth).len).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam

/-- **The rebuild term in passes: `μ_rebuild ≤ (1 − τ)·d/(rounds·x₁) ≤ d/(rounds·x₁)`.** `RateOfAuditPartial`, with the codec's
one-round encode of a word at least `x₁` times its read (`x₁ · b/BW ≤ c_root/F` per word, written
`x₁ · b · F ≤ c_root · BW` with `F` the device's priced work and `BW` its device-memory bandwidth, per tick): the
probability over the bandwidth coins that the bandwidth audit accepts, the compute account did at least `W`, and

  `read · (rounds·x₁·F) < (k·b − (S + (L − ρ) + λ))·(rounds·x₁·F) − BW·(F·T − W − k·c_tag)`

is at most `RateOfAuditPartial`'s error, at `S = S_chip`, `L = ∑ cap_i · T_bw` and `T` the compute window.

**The reading.** Divide by `rounds·x₁·F` and by `|E| = N·b`, with `k = ⌈(1 − ε_s)·N⌉` as in `RateAudit` and
`d = BW·T/(N·b)` the compute window in passes over the store: an accepted device read at least `(1 − μ)·|E|` bits, with
`RateAudit`'s `μ` and

  `μ_rebuild = (1 − τ)·d/(rounds·x₁) ≤ d/(rounds·x₁)`,  `τ = (W + k·c_tag)/(F·T)`,

so idle compute `F·T − W` completes at most `d/(rounds·x₁)` of the store per window, however the device splits it into
partial words. At the deployed `x₁ ≈ 175` and the stated three rounds that is `d/525` (`d/350` at two). A natural `x₁` at most the true ratio
suffices: rounding it down only weakens the bound. Proof: `Accounting.SecurityProofs.RateOfAuditPartialPasses`, from
`RateOfAuditPartial`: `(F·T − W − k·c_tag)·b/(rounds·c_root) ≤ BW·(F·T − W − k·c_tag)/(rounds·x₁·F)`.

Frozen (no-function, hash): as `RateOfAuditPartial`. -/
def RateOfAuditPartialPasses : Prop :=
  ∀ (N b h ρ s k cTag rounds cRoot x₁ lam W : ℕ) (cost : ℕ → ℕ) (p : Price) (ι α ΩW ΩS : Type) [Fintype ι]
    [Fintype α] (d : Device ι) (r : Run d α ΩW (RateCoins N b h ρ s) ΩS) (real : RealizesRatePartial r),
    Assumptions.WorkHardness b rounds cRoot cost → Assumptions.RateComputeAdditive p cTag cost real →
    x₁ * b * d.priced p ≤ cRoot * d.bw →
    ∀ (x : ΩW) (s' : ΩS),
      pr (fun y : RateCoins N b h ρ s =>
          r.accepts .bandwidth (x, y, s') ∧ W ≤ r.work p (x, y, s') ∧
            r.read (x, y, s') * (rounds * x₁ * d.priced p) <
              (k * b - (d.chip + (∑ i, d.cap i * (r.win .bandwidth).len - ρ) + lam)) * (rounds * x₁ * d.priced p) -
                d.bw * (d.priced p * (r.win .compute).len - W - k * cTag))
        ≤ (((k - 1 : ℕ) : ℝ≥0∞) / N) ^ s + (s : ℝ≥0∞) / 2 ^ h +
          (((d.rate .oracle * (r.win .bandwidth).len).choose k : ℕ) : ℝ≥0∞) / 2 ^ lam

end Accounting.Guarantees
