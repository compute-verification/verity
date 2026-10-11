import Definitions.Accounting.BandwidthAudit

/-!
# The bandwidth audit's guarantee as a joint run's bandwidth guarantee (trusted)

The bandwidth audit v0's guarantee (`Bandwidth.Guarantees.AuditRO`) embeds into the device model unweakened: in a run
that realizes its game (`Accounting.RealizesBandwidth`), the bandwidth audit has its guarantee
(`Accounting.Run.BandwidthSound`) at AuditRO's certificate and error, with the game's bounds the device's limits over the
bandwidth window.

Proved by `Accounting.SecurityProofs.BandwidthOfAudit : Accounting.Guarantees.BandwidthOfAudit`
(`Proofs.Accounting.BandwidthAudit`), from `Bandwidth.SecurityProofs.AuditRO`.
-/

namespace Accounting.Guarantees

open Accounting

/-- **The bandwidth audit's guarantee is the bandwidth guarantee.** For any run on any device whose bandwidth audit
realizes the bandwidth audit's game with a store of `N > 0` blocks of `b` bits, `h`-bit tags, `s` sampled blocks and at
most `G` regenerations, and for any `k` and slack `λ`, the bandwidth audit has its guarantee at certificate
`k·b − S_chip − c − G·b − λ` bits and error `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ`, where `c = ∑ cap_i · T` is every side
channel's capacity over the bandwidth window, `q = rate_oracle · T` the device's oracle queries over it, and `T` its
length. -/
def BandwidthOfAudit : Prop :=
  ∀ (N b h s G k lam : ℕ) (ι α ΩW ΩS : Type) [Fintype ι] [Fintype α] [Fintype ΩW] [Fintype ΩS] (d : Device ι)
    (r : Run d α ΩW (BandwidthCoins N b h s) ΩS),
    0 < N → RealizesBandwidth G r →
      r.BandwidthSound
        (((k * b : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * (r.win .bandwidth).len : ℕ) : ℝ) - ((G * b : ℕ) : ℝ) - lam)
        ((((k - 1 : ℕ) : ℝ) / N) ^ s + (s : ℝ) / 2 ^ h +
          (((d.rate .oracle * (r.win .bandwidth).len).choose k : ℕ) : ℝ) / 2 ^ lam)

end Accounting.Guarantees
