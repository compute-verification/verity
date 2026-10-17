import Definitions.Accounting.Joint
import Definitions.Accounting.Bandwidth.Game

/-!
# The bandwidth audit's game as the bandwidth audit of a joint run (trusted)

A run realizes the bandwidth audit's game (`Bandwidth/Game.lean`) with at most `G` regenerations when its bandwidth
coins are the game's (the store, the oracle at the fresh seed and the sample), and at each value of the other audits'
coins the device's bandwidth interaction is a device of the game (`Bandwidth.Device`) within the device model's limits
over the bandwidth window: on-chip state of `chip` bits, side sources of `∑ i, cap i * len` bits (every side channel's
capacity over the window), `rate .oracle * len` oracle queries, and `G` regenerations. The bandwidth audit's verdict is
the game's, and the game's ingress is covered by the bandwidth account's reads from device memory.

The ingress field is where double counting is ruled out. Every bit the game counts as read from device memory is
charged to the bandwidth account, so a bit another account read (a PoUS decode's, a PoUW GEMM's) is not credited to the
bandwidth audit, and the device's limits (`Step.Fits`) bound the accounts' reads together.
-/

namespace Accounting

/-- The bandwidth audit's coins in its game with a store of `N` blocks of `b` bits, `h`-bit tags and `s` sampled blocks:
the store, the oracle at the fresh seed and the sample. -/
abbrev BandwidthCoins (N b h s : ℕ) : Type := Bandwidth.Store N b × Bandwidth.Oracle N b h × (Fin s → Fin N)

/-- Run `r`'s bandwidth audit is the bandwidth audit's game, played at each value of the other audits' coins by a device
of the game within the device model's limits over the bandwidth window and at most `G` regenerations, whose ingress the
bandwidth account's reads from device memory cover. The game's side sources are a function of the store
(`Bandwidth.Device.side`), laid out before the seed, so no realization carries relayed tags: the claim
`bandwidth-device`'s exclusion holds here intact, and a device that relays tags in over a side channel is the relay
variant's (`AuditRelay`). -/
structure RealizesBandwidth (G : ℕ) {N b h s : ℕ} {ι : Type} [Fintype ι] {d : Device ι} {α ΩW ΩS : Type} [Fintype α]
    (r : Run d α ΩW (BandwidthCoins N b h s) ΩS) where
  /-- the device of the game at the other audits' coins -/
  dev : ΩW × ΩS → Bandwidth.Device N b h d.chip (∑ i, d.cap i * (r.win .bandwidth).len) G
    (d.rate .oracle * (r.win .bandwidth).len)
  /-- the bandwidth audit's verdict is the game's -/
  accepts_iff : ∀ x y s', r.accepts .bandwidth (x, y, s') ↔ Bandwidth.Accepts (dev (x, s')) y.1 y.2.1 y.2.2
  /-- the bandwidth account's reads from device memory cover the game's ingress -/
  ingress_le : ∀ x y s', (dev (x, s')).ingress y.1 y.2.1 ≤ r.read (x, y, s')

end Accounting
