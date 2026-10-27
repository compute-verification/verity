import Definitions.Accounting.Guarantees.JointLive
import Definitions.Accounting.Guarantees.JointBridge.BandwidthOfRemoteLineLive
import Definitions.Accounting.JointLive.Honest

/-!
# The honest and nontrivial lemmas of `JointHorizonBlocksLive` and `Bridge.BandwidthOfRemoteLineLive` (trusted)

Every security guarantee has an honest lemma (its hypotheses hold at the honest deployed instance) and a nontrivial one
(its conclusion fails for a verifier that accepts everything); failing either, it says nothing (#4825). These are
`JointHorizonBlocksLive`'s and `Bridge.BandwidthOfRemoteLineLive`'s, at `Accounting.Honest`'s instances: the honest
deployed horizon on one H200 of `vy-dc-h200-2` and the honest remote bandwidth session, and an idle node under verdicts
that accept everything. Each instance is one in any causal game, since neither node's world reads the strategy or the
run.

Each per-period hypothesis is stated as its event having probability 0 at every strategy, which is
`PeriodLive.Sound` at every error. Each conclusion that fails is stated as its event's probability exceeding every
bound below 1, so the conclusion fails at every error below 1.

Proved by `Accounting.SecurityProofs.G : Accounting.Guarantees.G` (`Proofs.Accounting.JointLive.Honest`).
-/

namespace Accounting.Guarantees

open Accounting
open scoped ENNReal

/-- **`JointHorizonBlocksLive`'s honest lemma: its hypotheses hold at the honest deployed horizon, at error 0.** At
`Accounting.Honest.honestBlocks` (one H200 of `vy-dc-h200-2`, the honest wipe of `r20261026-003454-bfd2`, P2's deployed
store and read deadline, in any causal game; any op and link rates, circuit, partition, inputs, number of blocks, block
length and bits moved), every verdict accepts on every run, and at every strategy each subprotocol of each block
accepts while its claim fails at the honest node's certificates (`Accounting.Honest.certs`), each wipe is accepted
while more than the foreign bits it publishes plus the unfilled state survive, and each block's PoUS accepts while its
memory term fails, each with probability 0. Proof: `Accounting.SecurityProofs.JointHorizonBlocksLive_honest`. -/
def JointHorizonBlocksLive_honest : Prop :=
  ∀ (rate : Op → ℕ) (cap : Unit → ℕ) (C : Verity.Circuit) (n : ℕ) (P : Verity.Partition C n) (v : C.Transcript)
    (O : Type) (G : Verity.Game O) (B L E : ℕ),
    (∀ o, (∀ b a, ((Honest.honestBlocks rate cap P v G B L E).block b).accepts a o) ∧
      ∀ b, ((Honest.honestBlocks rate cap P v G B L E).wipe b).accepts o) ∧
    (∀ b a (s : Verity.Game.Strategy G), Verity.Game.prob (fun o =>
      ((Honest.honestBlocks rate cap P v G B L E).block b).accepts a o ∧
        ¬ ((Honest.honestBlocks rate cap P v G B L E).block b).Holds (Honest.certs C L E) s a o) G s = 0) ∧
    (∀ b (s : Verity.Game.Strategy G), Verity.Game.prob (fun o =>
      ((Honest.honestBlocks rate cap P v G B L E).wipe b).accepts o ∧
        ¬ ((Honest.honestBlocks rate cap P v G B L E).wipe b).Survives o) G s = 0) ∧
    (∀ b (s : Verity.Game.Strategy G), Verity.Game.prob (fun o =>
      ((Honest.honestBlocks rate cap P v G B L E).block b).accepts .pous o ∧
        ¬ (Honest.honestBlocks rate cap P v G B L E).MemoryTerm s b o) G s = 0)

/-- **`JointHorizonBlocksLive`'s nontrivial lemma: its conclusion fails for verdicts that accept everything.** At
`Accounting.Honest.idleBlocks`, a node that does nothing on the deployed device under verdicts that accept everything,
in any causal game, claimed at the honest node's certificates, PoUB's claim fails on every run of every block of
positive length, so at every strategy the probability that some accepted claim fails exceeds every bound below 1: the
conclusion fails whenever its errors sum to less than 1. Proof:
`Accounting.SecurityProofs.JointHorizonBlocksLive_nontrivial`. -/
def JointHorizonBlocksLive_nontrivial : Prop :=
  ∀ (rate : Op → ℕ) (cap : Unit → ℕ) (C : Verity.Circuit) (n : ℕ) (P : Verity.Partition C n) (v : C.Transcript)
    (O : Type) (G : Verity.Game O) (B L E : ℕ) (s : Verity.Game.Strategy G) (e : ℝ≥0∞),
    0 < B → 0 < L → e < 1 →
    ¬ Verity.Game.prob (fun o =>
        (∃ b a, ((Honest.idleBlocks rate cap P v G B L E).block b).accepts a o ∧
          ¬ ((Honest.idleBlocks rate cap P v G B L E).block b).Holds (Honest.certs C L E) s a o) ∨
        (∃ b, ((Honest.idleBlocks rate cap P v G B L E).wipe b).accepts o ∧
          ¬ ((Honest.idleBlocks rate cap P v G B L E).wipe b).Survives o) ∨
        (∃ b, ((Honest.idleBlocks rate cap P v G B L E).block b).accepts .pous o ∧
          ¬ (Honest.idleBlocks rate cap P v G B L E).MemoryTerm s b o)) G s ≤ e

namespace Bridge

/-- **`BandwidthOfRemoteLineLive`'s honest lemma: its hypotheses hold at the honest deployed session, at error 0.** At
each block of the honest node (`Accounting.Honest.honestPeriod`) as long as the honest remote bandwidth session of
`r20261026-180902-bc3d` (`Accounting.Honest.sessionSpan` ticks), in any causal game, with the bandwidth certificate the
bits that session credited as read (`Accounting.Honest.sessionCerts`), the period realizes a session that accepts and
whose line credits as read exactly the bits the node read from its registered store, and at every strategy the
session's bound holds at error 0. Proof: `Accounting.SecurityProofs.Bridge.BandwidthOfRemoteLineLive_honest`. -/
def BandwidthOfRemoteLineLive_honest : Prop :=
  ∀ (rate : Op → ℕ) (cap : Unit → ℕ) (C : Verity.Circuit) (n : ℕ) (P : Verity.Partition C n) (v : C.Transcript)
    (O : Type) (G : Verity.Game O) (E b : ℕ),
    RealizesBandwidthOfRemoteLineLive (Honest.honestPeriod rate cap P v G Honest.sessionSpan E b)
      (Honest.sessionCerts C E) (fun _ => True) (fun _ => (Honest.sessionBits : ℝ))
      (fun _ _ => (Honest.sessionBits : ℝ)) ∧
    ∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun _ => True ∧ (Honest.sessionBits : ℝ) < Honest.sessionBits) G s = 0

/-- **`BandwidthOfRemoteLineLive`'s nontrivial lemma: its conclusion fails for a PoUB verdict that accepts
everything.** At each block of positive length of the idle node (`Accounting.Honest.idlePeriod`), which reads nothing
under verdicts that accept everything, in any causal game, claimed at the honest node's certificates, the probability
that PoUB accepts while its claim fails exceeds every bound below 1 at every strategy: the conclusion fails at every
error below 1. Proof: `Accounting.SecurityProofs.Bridge.BandwidthOfRemoteLineLive_nontrivial`. -/
def BandwidthOfRemoteLineLive_nontrivial : Prop :=
  ∀ (rate : Op → ℕ) (cap : Unit → ℕ) (C : Verity.Circuit) (n : ℕ) (P : Verity.Partition C n) (v : C.Transcript)
    (O : Type) (G : Verity.Game O) (L E b : ℕ) (s : Verity.Game.Strategy G) (e : ℝ≥0∞),
    0 < L → e < 1 →
    ¬ Verity.Game.prob (fun o => (Honest.idlePeriod rate cap P v G L E b).accepts .poub o ∧
        ¬ (Honest.idlePeriod rate cap P v G L E b).Holds (Honest.certs C L E) s .poub o) G s ≤ e

end Bridge

end Accounting.Guarantees
