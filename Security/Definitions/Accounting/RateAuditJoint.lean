import Definitions.Accounting.Joint
import Definitions.Accounting.RateAudit

/-!
# The rate audit's game as the bandwidth audit of a joint run (trusted)

A run realizes the rate audit's game (`RateAudit.lean`) at price `p`, `c_tag` per tag and `c_enc` per rebuilt word when
its bandwidth coins are the game's (the store, and the window's stream, oracle and sample), and at each value of the
other audits' coins the device's bandwidth interaction is a device of the game within the device model's limits over
the bandwidth window: on-chip state of `chip` bits, a side link of `∑ i, cap i * len` bits (every side channel's
capacity over the window, the certified stream's share included) and `rate .oracle * len` oracle queries. The bandwidth
audit's verdict is the game's; the game's ingress is covered by the bandwidth account's reads from device memory; and
the game's compute, its tags at `c_tag` and its rebuilt words at `c_enc`, is covered by the work, at price `p` over the
compute window, of the accounts other than the compute account.

The compute field is where the rebuild is charged as compute. A word the device neither holds nor reads it can
rebuild, and every op of the rebuild is charged to one account (`Step.Fits`). The compute account is the one whose
work the compute audit certifies, so the rebuild's ops are another account's, and the compute account's work and
theirs together are within the device's priced work over the window: what the compute audit certifies is not
available to rebuild.
-/

namespace Accounting

/-- The rate audit's coins in its game with a store of `N` words of `b` bits, a `ρ`-bit certified stream, `h`-bit tags
and `s` sampled words: the store, and the window's stream, oracle and sample. -/
abbrev RateCoins (N b h ρ s : ℕ) : Type := Rate.Payloads N b × Rate.Coins N b h ρ s

/-- Run `r`'s bandwidth audit is the rate audit's game at price `p`, `c_tag` per tag and `c_enc` per rebuilt word,
played at each value of the other audits' coins by a device of the game within the device model's limits over the
bandwidth window, whose ingress the bandwidth account's reads from device memory cover and whose compute the other
accounts' work over the compute window covers. -/
structure RealizesRate (p : Price) (cTag cEnc : ℕ) {N b h ρ s : ℕ} {ι : Type} [Fintype ι] {d : Device ι}
    {α ΩW ΩS : Type} [Fintype α] [DecidableEq α] (r : Run d α ΩW (RateCoins N b h ρ s) ΩS) where
  /-- the device of the game at the other audits' coins -/
  dev : ΩW × ΩS → Rate.Device N b h ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len)
    (d.rate .oracle * (r.win .bandwidth).len)
  /-- the bandwidth audit's verdict is the game's -/
  accepts_iff : ∀ x y s', r.accepts .bandwidth (x, y, s') ↔ (dev (x, s')).Accepts y.1 y.2.1.1 y.2.1.2 y.2.2
  /-- the bandwidth account's reads from device memory cover the game's ingress -/
  ingress_le : ∀ x y s', (dev (x, s')).ingress y.1 y.2.1.1 y.2.1.2 ≤ r.read (x, y, s')
  /-- the work of the accounts other than the compute account covers the game's tags and rebuilds -/
  compute_le : ∀ x y s', (dev (x, s')).compute cTag cEnc y.1 y.2.1.1 y.2.1.2 ≤ r.othersWork p (x, y, s')

end Accounting
