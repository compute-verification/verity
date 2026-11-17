module

public import Definitions.Compose.JointPeriod
public import Definitions.Poub.BandwidthAudit
import Mathlib.Basic.ENNReal.Real

/-! A period whose PoUB audit is the bandwidth audit's game; a device that relays tags in over a side channel is the relay
variant's (`Bandwidth.AuditRelay`), which this does not cover. -/

@[expose] public section

namespace ResourceAccounting

/-- Period `r`'s PoUB audit is the bandwidth audit's game with a store of `N` blocks of `b` bits, `h`-bit tags, `s`
sampled blocks and at most `G` regenerations. Every field is a property of the deployed period: its PoUB coins, the
device it ran on, PoUB's verdict and the bandwidth account's reads. -/
structure RealizesBandwidthOfAuditPerRunPeriod (G : ℕ) {N b h s : ℕ} {ι : Type} [Fintype ι] {d : Device ι}
    {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} [DecidableEq I] {Ω : I → Type} (r : PeriodRun d α C Ω) where
  coins : Ω (r.own .poub) ≃ BandwidthCoins N b h s
  dev : ((i : I) → Ω i) → Bandwidth.Device N b h d.chip (∑ i, d.cap i * r.win.len) G (d.rate .oracle * r.win.len)
  /-- the device of the game is fixed by the other audits' coins: PoUB's own coins are flipped after it is -/
  dev_update : ∀ ξ x, dev (Function.update ξ (r.own .poub) x) = dev ξ
  accepts_iff : ∀ ξ, r.accepts .poub ξ ↔
    Bandwidth.Accepts (dev ξ) (coins (ξ (r.own .poub))).1 (coins (ξ (r.own .poub))).2.1 (coins (ξ (r.own .poub))).2.2
  ingress_le : ∀ ξ, (dev ξ).ingress (coins (ξ (r.own .poub))).1 (coins (ξ (r.own .poub))).2.1 ≤ r.read ξ

end ResourceAccounting
