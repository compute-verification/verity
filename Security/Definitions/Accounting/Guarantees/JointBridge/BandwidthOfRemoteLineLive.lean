import Definitions.Accounting.JointBridge.BandwidthOfRemoteLineLive

/-!
# PoUB's per-period guarantee over the causal game, from the remote bandwidth audit's (trusted)

The bridge from the remote bandwidth audit's per-session bound to PoUB's per-period guarantee over the causal game
(`Accounting.PeriodLive.Sound` at `.poub`, the form `Accounting.Guarantees.JointPeriodLive` and its siblings take): in a
period whose PoUB is the session (`Accounting.RealizesBandwidthOfRemoteLineLive`), if at every strategy `s` the session
accepts while its line credits more bits read than the node read with probability at most `err s`, then at every
strategy `s` PoUB accepts while the bandwidth account read fewer than `Q` bits with probability at most `err s`. The
conclusion is `PeriodLive.Sound`'s event at `.poub`, at any error that is a function of the strategy, so
`PeriodLive.error`'s `δ + loss(Pr[B^s finds a collision])` among them.

The session's bound is to be `Properties.RemoteAudit.Line.LineHeld`'s bits-read form (MEM-97 item 4):
`δs + δ + Σ_c s/2^64 + bind` at `LineHeldAtComponent`'s premises (#4786), the binding term a SHA-512 collision on the
session's receipts, charged at the finder built from the strategy.

It replaces the bridge from the bandwidth audit v0's `Bandwidth.Guarantees.AuditRO`, which is in the random-oracle
model and which the joint guarantee no longer takes.

Proved by `Accounting.SecurityProofs.Bridge.BandwidthOfRemoteLineLive :
Accounting.Guarantees.Bridge.BandwidthOfRemoteLineLive` (`Proofs.Accounting.JointBridge.BandwidthOfRemoteLineLive`).
Its honest and nontrivial lemmas are `BandwidthOfRemoteLineLive_honest` and `BandwidthOfRemoteLineLive_nontrivial`
(`Definitions.Accounting.Guarantees.JointLive.Honest`).
-/

namespace Accounting.Guarantees.Bridge

open Accounting
open scoped ENNReal

/-- **PoUB's per-period guarantee over the causal game is the remote bandwidth audit's.** On any device and any period
read off the run of a causal game `G`, over a circuit, whose PoUB is the remote bandwidth audit's session
(`Accounting.RealizesBandwidthOfRemoteLineLive`: PoUB's verdict is the session's acceptance, the bandwidth certificate
is within the line's credit on an accepted run, and the bits the node read from its registered store are within the
bandwidth account's reads), if at every strategy `s` the session accepts while its line credits more bits read than
the node read with probability at most `err s`, then at every strategy `s` the probability that PoUB accepts while the
bandwidth account read fewer than `Q` bits over the period's window is at most `err s`. -/
def BandwidthOfRemoteLineLive : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (C : Verity.Circuit) (O : Type) (G : Verity.Game O)
    (r : PeriodLive d α C G) (c : Certificates) (acc : O → Prop) (credit : O → ℝ)
    (readBits : Verity.Game.Strategy G → O → ℝ) (err : Verity.Game.Strategy G → ℝ≥0∞),
    RealizesBandwidthOfRemoteLineLive r c acc credit readBits →
    (∀ s, Verity.Game.prob (fun o => acc o ∧ readBits s o < credit o) G s ≤ err s) →
    ∀ s, Verity.Game.prob (fun o => r.accepts .poub o ∧ ¬ r.Holds c s .poub o) G s ≤ err s

end Accounting.Guarantees.Bridge
