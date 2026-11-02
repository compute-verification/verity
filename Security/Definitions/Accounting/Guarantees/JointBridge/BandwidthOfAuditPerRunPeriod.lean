import Definitions.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod

/-!
# PoUB's per-period guarantee, from the bandwidth audit's (trusted)

The bridge from the bandwidth audit v0's guarantee (`Accounting.Guarantees.Remote.AuditRO`, through its per-run reading
`Accounting.Guarantees.BandwidthOfAuditPerRun`) to PoUB's per-period guarantee (`Accounting.PeriodRun.Sound` at audit
`.poub`), the coins model's form: in a period that realizes the bandwidth audit's game
(`Accounting.RealizesBandwidthOfAuditPerRunPeriod`), at any bandwidth certificate `Q` at most
`k·b − S_chip − c − G·b − λ`, PoUB accepts while the bandwidth account read fewer than `Q` bits over the period's window
with probability at most `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ` over PoUB's own coins, at any value of the others.

Proved by `Accounting.SecurityProofs.Bridge.BandwidthOfAuditPerRunPeriod :
Accounting.Guarantees.Bridge.BandwidthOfAuditPerRunPeriod` (`Proofs.Accounting.JointBridge.BandwidthOfAuditPerRunPeriod`),
from `Accounting.SecurityProofs.BandwidthOfAuditPerRun`.
-/

namespace Accounting.Guarantees.Bridge

open Accounting

/-- **PoUB's per-period guarantee is the bandwidth audit's.** On any device and any period of a node's run over a
circuit whose PoUB audit realizes the bandwidth audit's game with a store of `N > 0` blocks of `b` bits, `h`-bit tags,
`s` sampled blocks and at most `G` regenerations, for any `k`, slack `λ`, price `p` and certificates `c` whose bandwidth
certificate is at most `k·b − S_chip − c − G·b − λ` bits, PoUB has its per-period guarantee at error
`((k − 1)/N)^s + s/2^h + C(q, k)/2^λ`, where `c = ∑ cap_i · T` is every side channel's capacity over the period's
window, `q = rate_oracle · T` the device's oracle queries over it, and `T` its length. -/
def BandwidthOfAuditPerRunPeriod : Prop :=
  ∀ (N b h s G k lam : ℕ) (ι α : Type) [Fintype ι] [Fintype α] (d : Device ι) (C : Verity.Circuit) (I : Type)
    [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)] (r : PeriodRun d α C Ω) (c : Certificates) (p : Price),
    0 < N → RealizesBandwidthOfAuditPerRunPeriod G (N := N) (b := b) (h := h) (s := s) r →
    c.Q ≤ ((k * b : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * r.win.len : ℕ) : ℝ) - ((G * b : ℕ) : ℝ) - lam →
      r.Sound c p .poub
        ((((k - 1 : ℕ) : ℝ) / N) ^ s + (s : ℝ) / 2 ^ h + (((d.rate .oracle * r.win.len).choose k : ℕ) : ℝ) / 2 ^ lam)

end Accounting.Guarantees.Bridge
