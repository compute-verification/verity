import Definitions.Accounting.JointPeriod
import Definitions.Accounting.BandwidthAudit

/-!
# A period whose PoUB audit is the bandwidth audit's game (trusted)

A period of a node's run (`Accounting.PeriodRun`) realizes the bandwidth audit's game with at most `G` regenerations
when PoUB's coordinate of the period's coins is the game's coins (the store, the oracle at the fresh seed and the
sample), and at each value of the other audits' coins the device's bandwidth interaction over the period's window is
a device of the game (`Bandwidth.Device`) within the device model's limits over that window: on-chip state of `chip`
bits, side sources of `∑ i, cap i * len` bits, `rate .oracle * len` oracle queries and `G` regenerations. PoUB's
verdict is the game's, and the game's ingress is covered by the bandwidth account's reads from device memory over the
window. It is `Accounting.RealizesBandwidth` restated for a period's coins, `(i : I) → Ω i`, in place of a joint run's
three coordinates.

As in `RealizesBandwidth`, the game's side sources are a function of the store laid out before the seed, so no
realization carries relayed tags: a device that relays tags in over a side channel is the relay variant's
(`Bandwidth.AuditRelay`), which this structure does not cover.
-/

namespace Accounting

/-- Period `r`'s PoUB audit is the bandwidth audit's game with a store of `N` blocks of `b` bits, `h`-bit tags, `s`
sampled blocks and at most `G` regenerations. Every field is a property of the deployed period: its PoUB coins, the
device it ran on, PoUB's verdict and the bandwidth account's reads. -/
structure RealizesBandwidthOfAuditPerRunPeriod (G : ℕ) {N b h s : ℕ} {ι : Type} [Fintype ι] {d : Device ι}
    {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) where
  /-- PoUB's coordinate of the period's coins is the game's coins: the store, the oracle at the fresh seed and the
  sample, each uniform -/
  coins : Ω (r.own .poub) ≃ BandwidthCoins N b h s
  /-- the device of the game the node's device plays over the period's window, at the coins; its side sources, on-chip
  state, oracle queries and regenerations are within the device model's limits over the window -/
  dev : ((i : I) → Ω i) → Bandwidth.Device N b h d.chip (∑ i, d.cap i * r.win.len) G (d.rate .oracle * r.win.len)
  /-- the device of the game is fixed by the other audits' coins: PoUB's own coins are drawn after it is -/
  dev_update : ∀ ξ x, dev (Function.update ξ (r.own .poub) x) = dev ξ
  /-- PoUB's verdict is the game's, on PoUB's coins -/
  accepts_iff : ∀ ξ, r.accepts .poub ξ ↔
    Bandwidth.Accepts (dev ξ) (coins (ξ (r.own .poub))).1 (coins (ξ (r.own .poub))).2.1 (coins (ξ (r.own .poub))).2.2
  /-- the bandwidth account's reads from device memory over the period's window cover the game's ingress -/
  ingress_le : ∀ ξ, (dev ξ).ingress (coins (ξ (r.own .poub))).1 (coins (ξ (r.own .poub))).2.1 ≤ r.read ξ

end Accounting
