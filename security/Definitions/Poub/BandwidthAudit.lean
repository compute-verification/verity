module

public import Definitions.Compose.Joint
public import Definitions.Poub.Bandwidth.Game
import Mathlib.Basic.ENNReal.Real

/-! The bandwidth audit's game as the bandwidth audit of a joint run, its ingress charged to the bandwidth account so
that no bit another account read is credited to it. -/

@[expose] public section

namespace ResourceAccounting

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
  dev : ΩW × ΩS → Bandwidth.Device N b h d.chip (∑ i, d.cap i * (r.win .bandwidth).len) G
    (d.rate .oracle * (r.win .bandwidth).len)
  accepts_iff : ∀ x y s', r.accepts .bandwidth (x, y, s') ↔ Bandwidth.Accepts (dev (x, s')) y.1 y.2.1 y.2.2
  ingress_le : ∀ x y s', (dev (x, s')).ingress y.1 y.2.1 ≤ r.read (x, y, s')

end ResourceAccounting
