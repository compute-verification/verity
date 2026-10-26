import Definitions.Accounting.JointBlocks

/-!
# The joint guarantee over a horizon of server-time blocks and the wipes between them (trusted)

A server's life is cut into `B` server-time blocks by certified wipes (`Accounting.BlocksRun`). Each block is a period
of the node's run (`Accounting.PeriodRun`) decided by the five audits, sampled proofs, PoUW, PoUS, PoUB and the network
certifier, each on its own coordinate of one coin space, so a block may adapt to the coins of the blocks before it.
Each block is opened by a wipe (`Accounting.BlockWipe`), and each block's PoUS, over the weights alone, bounds its
undeclared memory at its challenge times (`BlocksRun.MemoryTerm`).

`JointHorizonBlocks` takes three kinds of guarantee as hypotheses: each audit's per-period guarantee in each block
(`PeriodRun.Sound`), each wipe's survivor bound at its error `ε_wipe`, and each block's memory term at its error
`δ_PoUS`. Then the probability that some block's accepted audit's claim fails, some accepted wipe's survivor bound
fails, or some block's memory term fails while its PoUS accepted is at most every one of those errors summed. The proof
is `Accounting.SecurityProofs.JointHorizonBlocks`, a union bound over the product of the coins. The wipes' and the
memory terms' bounds are taken over the whole coin space, the weaker form: one over the challenger's own coordinate at
every value of the others, as the audits' are, gives it (`Pouw.pr_le_of_update`).

Each hypothesis is to come from its protocol's own statement:

* the five audits: as in `Accounting.Guarantees.JointPeriod`. For integrity, proofs' per-block bound is
  `Flock.Guarantees.IntegrityBlock` (and `IntegrityBlocks` over the horizon): block `b` accepted with at least
  `ln(1/δ)/q` wrong units with probability at most `δ + Σ_{t ∈ b} (ε_ks,t + δ_link,t)`. It is stated in C-Flock's
  analysis (the proofs tree) over its game of sessions played by one prover, not over this coin space, so sampled
  proofs' term here is `PeriodRun.Sound` for `.sampled` in each block until a bridge carries it over;
* the wipe: the streamed wipe, `Pous.Guarantees.StreamedWipe`, at what `verity/wipe/v0` publishes with an accepted
  verdict: `foreign_bits = ⌈fill/19⌉ + E` with 2152 reads (`StreamedWipeMeetsFamily`, #4490), or the fill's share at
  `k` reads plus `E` (`StreamedWipeMeetsReads`, #4707), at `ε_wipe = 2^-128 + Adv_PRF`, through DEVICE-WIPE's
  `Verity.Wipe.SurvivorBound` (that bridge is not stated yet). `Adv_PRF` is the labels' PRF advantage against the
  distinguisher the streamed wipe's game builds from the prover (`Pous.StreamedWipe.wipeDistinguisher`), and `δ_net`
  joins `ε_wipe` once the network certifier's count of the egress `E` enters (NET-78);
* the memory term: weights-only PoUS at a share of `0.07` (ASK-132), at `δ_PoUS`.
Demo A (Daniel's ruling of 25 Oct) is PoUW, sampled proofs and network accounting only: `JointHorizonBlocksA` is its
composed theorem, the blocks' per-period guarantees for those three audits with no wipe or memory term, the instance of
`JointHorizonBlocksLegs` (any set of the audits). Memory accounting (PoUS, PoUB, the wipe and the memory and wipe terms)
joins at M4 through `JointHorizonBlocks`.
-/

namespace Accounting.Guarantees

open Accounting

/-- **The joint guarantee over a horizon of `B` server-time blocks and the wipes that open them.** On any device and
any horizon of `B` blocks of a node's run over a circuit on one coin space (each block a period decided by sampled
proofs, PoUW, PoUS, PoUB and the network certifier on its own coordinates, and a function of every coin, so a block may
adapt to the earlier blocks' coins), suppose:

* each audit `a` of block `b` has its per-period guarantee at certificates `c b`, price `p` and error `δ b a`;
* the wipe that opens block `b` is accepted while more than the foreign bits it publishes plus the unfilled state
  survive beyond the committed carried state with probability at most `ε_wipe b`;
* block `b`'s PoUS accepts while, at some of its challenge times, its undeclared memory exceeds `0.07 · U + S_chip + c`
  (`c` the side channels' bits over the read deadline) with probability at most `δ_PoUS b`.

Then the probability that some block's accepted audit's claim fails, some accepted wipe's survivor bound fails, or
some block's memory term fails while its PoUS accepted is at most
$$\sum_b \sum_a \delta_{b,a} + \sum_b \varepsilon_{\mathrm{wipe},b} + \sum_b \delta_{\mathrm{PoUS},b}.$$

Sampled proofs' term is `PeriodRun.Sound` for `.sampled` in each block, not proofs' `Flock.Guarantees.IntegrityBlock`,
which is stated over C-Flock's game rather than this coin space. The memory term holds at PoUS's challenge times only,
until the continuous audit is tied in. **Gap (PRF-167):** the state a block carries across a wipe (the KV cache,
committed before the wipe and re-read after it) is outside the integrity bound: nothing here says the value a block
re-reads is the one the block before committed. -/
def JointHorizonBlocks : Prop :=
  ∀ (ι α I : Type) [Fintype ι] [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (C : Verity.Circuit) (B : ℕ) (s : BlocksRun d α C Ω B)
    (c : Fin B → Certificates) (p : Price) (δ : Fin B → Audit → ℝ) (εWipe δPous : Fin B → ℝ),
    (∀ b a, (s.block b).Sound (c b) p a (δ b a)) →
    (∀ b, Pouw.pr (fun ξ => (s.wipe b).accepts ξ ∧ ¬ (s.wipe b).Survives ξ) ≤ εWipe b) →
    (∀ b, Pouw.pr (fun ξ => (s.block b).accepts .pous ξ ∧ ¬ s.MemoryTerm b ξ) ≤ δPous b) →
    Pouw.pr (fun ξ => (∃ b a, (s.block b).accepts a ξ ∧ ¬ (s.block b).Holds (c b) p a ξ) ∨
        (∃ b, (s.wipe b).accepts ξ ∧ ¬ (s.wipe b).Survives ξ) ∨
        (∃ b, (s.block b).accepts .pous ξ ∧ ¬ s.MemoryTerm b ξ)) ≤
      ∑ b, ∑ a, δ b a + ∑ b, εWipe b + ∑ b, δPous b

/-- **The joint guarantee over a horizon of `B` server-time blocks, for a set of the audits.** On any device and any
`B` blocks of a node's run over a circuit on one coin space (each block a period decided on its own coordinates, and a
function of every coin, so a block may adapt to the earlier blocks' coins), if each audit `a ∈ A` of block `b` has its
per-period guarantee at certificates `c b`, price `p` and error `δ b a`, the probability that some audit of `A` in some
block accepts while its claim fails is at most `∑_b ∑_{a ∈ A} δ b a`. The audits outside `A` take no hypothesis and
are bounded by nothing. -/
def JointHorizonBlocksLegs : Prop :=
  ∀ (ι α I : Type) [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (C : Verity.Circuit) (B : ℕ) (r : Fin B → PeriodRun d α C Ω)
    (c : Fin B → Certificates) (p : Price) (A : Finset Audit) (δ : Fin B → Audit → ℝ),
    (∀ b, ∀ a ∈ A, (r b).Sound (c b) p a (δ b a)) →
    Pouw.pr (fun ξ => ∃ b, ∃ a ∈ A, (r b).accepts a ξ ∧ ¬ (r b).Holds (c b) p a ξ) ≤ ∑ b, ∑ a ∈ A, δ b a

/-- **Demo A's composed theorem: sampled proofs, PoUW and the network certifier over a horizon of `B` server-time
blocks.** On any device and any `B` blocks of a node's run over a circuit on one coin space (each block a period
decided on its own coordinates, and a function of every coin, so a block may adapt to the earlier blocks' coins), if
block `b`'s sampled proofs, PoUW and network certifier have their per-period guarantees at certificates `c b`, price
`p` and errors `δ b a`, the probability that one of those three audits in some block accepts while its claim fails
(more than an `ε` share of the circuit's units wrong, less than `W` of work, or more than `K` bits through the network
certifiers) is at most `∑_b (δ b sampled + δ b pouw + δ b network)`.

The weights are pinned at their transfer in, like any ingress (ASK-130 A), not by PoUS. The memory and wipe terms
join at M4 through `JointHorizonBlocks`. As there, sampled proofs' term is `PeriodRun.Sound` for `.sampled` in each
block, not proofs' `Flock.Guarantees.IntegrityBlock`, which is stated over C-Flock's game rather than this coin space;
and the state a block carries across a wipe is outside the integrity bound (PRF-167). -/
def JointHorizonBlocksA : Prop :=
  ∀ (ι α I : Type) [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (C : Verity.Circuit) (B : ℕ) (r : Fin B → PeriodRun d α C Ω)
    (c : Fin B → Certificates) (p : Price) (δ : Fin B → Audit → ℝ),
    (∀ b, ∀ a ∈ ({.sampled, .pouw, .network} : Finset Audit), (r b).Sound (c b) p a (δ b a)) →
    Pouw.pr (fun ξ => ∃ b, ∃ a ∈ ({.sampled, .pouw, .network} : Finset Audit),
        (r b).accepts a ξ ∧ ¬ (r b).Holds (c b) p a ξ) ≤
      ∑ b, ∑ a ∈ ({.sampled, .pouw, .network} : Finset Audit), δ b a

end Accounting.Guarantees
