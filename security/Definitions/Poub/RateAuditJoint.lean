module

public import Definitions.Compose.Joint
public import Definitions.Poub.RateAudit

/-! The rate audit's game as the bandwidth audit of a joint run, its rebuilds charged to the accounts other than the
compute account. -/

@[expose] public section

namespace ResourceAccounting

/-- The rate audit's coins in its game with a store of `N` words of `b` bits, a `ρ`-bit certified stream, `h`-bit tags
and `s` sampled words: the store, and the window's stream, oracle and sample. -/
abbrev RateCoins (N b h ρ s : ℕ) : Type := Rate.Store N b × Rate.Coins N b h ρ s

/-- Run `r`'s bandwidth audit is the rate audit's game at price `p`, `c_tag` per tag and `c_enc` per rebuilt word,
played at each value of the other audits' coins by a device of the game within the device model's limits over the
bandwidth window, whose ingress the bandwidth account's reads from device memory cover and whose compute the other
accounts' work over the compute window covers. -/
structure RealizesRate (p : Price) (cTag cEnc : ℕ) {N b h ρ s : ℕ} {ι : Type} [Fintype ι] {d : Device ι}
    {α ΩW ΩS : Type} [Fintype α] [DecidableEq α] (r : Run d α ΩW (RateCoins N b h ρ s) ΩS) where
  dev : ΩW × ΩS → Rate.Prover N b h ρ d.chip (∑ i, d.cap i * (r.win .bandwidth).len)
    (d.rate .oracle * (r.win .bandwidth).len)
  accepts_iff : ∀ x y s', r.accepts .bandwidth (x, y, s') ↔ (dev (x, s')).Accepts y.1 y.2.1.1 y.2.1.2 y.2.2
  ingress_le : ∀ x y s', (dev (x, s')).ingress y.1 y.2.1.1 y.2.1.2 ≤ r.read (x, y, s')
  compute_le : ∀ x y s', (dev (x, s')).compute cTag cEnc y.1 y.2.1.1 y.2.1.2 ≤ r.othersWork p (x, y, s')

end ResourceAccounting
