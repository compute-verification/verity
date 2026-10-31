import Definitions.Accounting.BandwidthAudit
import Definitions.Accounting.PerRun

/-!
# The bandwidth audit's guarantee as a joint run's bandwidth guarantee (trusted)

The bandwidth audit v0's guarantee, PoUB's per-pass lemma (`Accounting.Guarantees.Remote.AuditRO`), embeds into the
device model at its own certificate and error: in a run that realizes its game (`Accounting.RealizesBandwidth`), the
bandwidth audit has its guarantee (`Accounting.Run.BandwidthSound`) at AuditRO's certificate and error, with the game's
bounds the device's limits over the bandwidth window.

`BandwidthOfAudit` keeps AuditRO's numbers but not its per-run reading: `BandwidthSound`, the form `Joint` takes, is about
a bound on the bandwidth account's reads on every one of the bandwidth audit's own coins, its worst case over them.
AuditRO bounds the ingress at the coins drawn, so it also gives the per-run form, `BandwidthSoundPerRun`, the one
`JointPerRun` takes: `BandwidthOfAuditPerRun`, at the same certificate and error.

Proved by `Accounting.SecurityProofs.BandwidthOfAudit : Accounting.Guarantees.BandwidthOfAudit` and
`Accounting.SecurityProofs.BandwidthOfAuditPerRun : Accounting.Guarantees.BandwidthOfAuditPerRun`
(`Proofs.Accounting.BandwidthAudit`), from `Accounting.SecurityProofs.Remote.AuditRO`.
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

/-- **The bandwidth audit's guarantee is the per-run bandwidth guarantee.** `BandwidthOfAudit` at the coins drawn: for
any run on any device whose bandwidth audit realizes the bandwidth audit's game, at any value of the other audits' coins,
the bandwidth audit accepts while the bandwidth account read fewer than `k·b − S_chip − c − G·b − λ` bits from device
memory over the bandwidth window with probability at most `((k − 1)/N)^s + s/2^h + C(q, k)/2^λ` over the bandwidth
coins. -/
def BandwidthOfAuditPerRun : Prop :=
  ∀ (N b h s G k lam : ℕ) (ι α ΩW ΩS : Type) [Fintype ι] [Fintype α] [Fintype ΩW] [Fintype ΩS] (d : Device ι)
    (r : Run d α ΩW (BandwidthCoins N b h s) ΩS),
    0 < N → RealizesBandwidth G r →
      r.BandwidthSoundPerRun
        (((k * b : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * (r.win .bandwidth).len : ℕ) : ℝ) - ((G * b : ℕ) : ℝ) - lam)
        ((((k - 1 : ℕ) : ℝ) / N) ^ s + (s : ℝ) / 2 ^ h +
          (((d.rate .oracle * (r.win .bandwidth).len).choose k : ℕ) : ℝ) / 2 ^ lam)

end Accounting.Guarantees
