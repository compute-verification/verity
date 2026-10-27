import Definitions.Accounting.Guarantees.JointBridge.BandwidthOfRemoteLineLive
import Definitions.Accounting.JointLive.Honest

/-!
# The honest and nontrivial lemmas of `Bridge.BandwidthOfRemoteLineLive` (trusted)

Every security guarantee has an honest lemma (its hypotheses hold at the honest deployed instance) and a nontrivial one
(its conclusion fails for a verifier that accepts everything); failing either, it says nothing. These are
`Bridge.BandwidthOfRemoteLineLive`'s, at `Accounting.Honest`'s instances: the honest deployed block on one H200 of
`vy-dc-h200-2` with the honest remote bandwidth session, and an idle node under verdicts that accept everything. Each
instance is one in any causal game, since neither node's world reads the strategy or the run.

Each hypothesis is stated as written in its guarantee, at error 0. The conclusion that fails is stated as its event's
probability exceeding every bound below 1, so the conclusion fails at every error below 1.

Proved by `Accounting.SecurityProofs.G : Accounting.Guarantees.G` (`Proofs.Accounting.JointLive.Honest`).
-/

namespace Accounting.Guarantees

open Accounting
open scoped ENNReal

namespace Bridge

/-- **`BandwidthOfRemoteLineLive`'s honest lemma: its hypotheses hold at the honest deployed session, at error 0.** At
each block of the honest node (`Accounting.Honest.honestPeriod`) as long as the honest remote bandwidth session
(`Accounting.Honest.sessionSpan` ticks), in any causal game, with the bandwidth certificate the bits that session
credited as read (`Accounting.Honest.sessionCerts`), the period realizes a session that accepts and whose line credits
as read exactly the bits the node read from its registered store, and at every strategy the session's bound holds at
error 0. Proof: `Accounting.SecurityProofs.Bridge.BandwidthOfRemoteLineLive_honest`. -/
def BandwidthOfRemoteLineLive_honest : Prop :=
  ∀ (rate : Op → ℕ) (cap : Unit → ℕ) (C : Verity.Circuit) (n : ℕ) (P : Verity.Partition C n) (v : C.Transcript)
    (O : Type) (G : Verity.Game O) (E b : ℕ),
    RealizesBandwidthOfRemoteLineLive (Honest.honestPeriod rate cap P v G Honest.sessionSpan E b)
      (Honest.sessionCerts C E) (fun _ => True) (fun _ => (Honest.sessionBits : ℝ))
      (fun _ _ => (Honest.sessionBits : ℝ)) ∧
    ∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun _ => True ∧ (Honest.sessionBits : ℝ) < Honest.sessionBits) G s ≤ 0

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
