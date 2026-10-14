import Definitions.Accounting.Joint

/-!
# A state-bounded game as the capacity audit of a joint run (trusted)

A capacity audit is a game whose servers are graded by the state they keep between rounds (`StateGame`): PoUS's ring
game (`Pous.EndToEnd.Server S D Q L`) is one, with the state a server carries across a round boundary in `Bits S`. A run
realizes such a game, with rounds of `round` ticks, when at each value of the other audits' coins, if the capacity
account holds at most `B` bits of device memory at every tick of the capacity window on every capacity coin, the device's
capacity interaction is a server of the game with state `B + chip + ∑ i, cap i * round`: what the capacity account
holds, the whole chip, and every side channel's capacity over a round. The capacity audit's verdict is the game's.

The state bound is where double counting is ruled out. Device memory the capacity audit's answers depend on is charged
to the capacity account, so memory another account holds at the same tick (a PoUW weight, a bandwidth store) is not
credited to the capacity audit, and the device's limits (`Step.Fits`) bound the accounts' memory together. A server of
the game is also handed the game's public inputs (PoUS's `W`), so state it computes from them alone, without the
operations the game charges (PoUS's ring operations and key queries), is not part of its bound. The weight-derived state
a PoUW scheme keeps (Pearl-C's `B̃`) is such state: a device may keep it in another account, uncounted by the capacity
certificate, and it then counts against the capacity residual (`Accounting.Guarantees.DerivedStateResidual`). Like
every residual event here, the one that guarantee bounds (`Run.CapacityCovertBut`) holds only when the other accounts are
beyond the residual on every one of the capacity audit's own coins.

The game bounds the state at every round boundary at once, while `B` bounds the capacity account's memory only at the
capacity window's ticks, so a run realizes the game only when the window contains every round boundary: for PoUS, the
whole session. A window of one tick does so only for a one-round game. The capacity guarantee (`Run.CapacitySound`) is
in bit-ticks over the window, a time average, and the account's bit-ticks bound its memory at one tick only by their
whole total, so over a window of `T` ticks the certificate this realization gives is about `S₀/T` bits on average,
where `S₀` is the state the game is sound against (`Accounting.Guarantees.CapacityOfStateBound`). That is inherent to
games graded by their peak state: a useful certificate needs a game graded at each round boundary (the continuous
audit, for P2), and that choice is Daniel's. No table should cite `CapacityOfStateBound` at a session window.
-/

namespace Accounting

/-- A game whose servers are graded by their state in bits, played on coins `Ω`: the servers with state `n`, and the
coins on which a server passes. -/
structure StateGame (Ω : Type) : Type 1 where
  /-- the servers with state `n` bits -/
  Server : ℕ → Type
  /-- a server passes the game on coins `s` -/
  Passes : {n : ℕ} → Server n → Ω → Prop

/-- Run `r`'s capacity audit is game `G`, with rounds of `round` ticks: at each value of the other audits' coins, if the
capacity account holds at most `B` bits of device memory at every tick of the capacity window on every capacity coin, the
device is a server with state `B + chip + ∑ i, cap i * round`, which passes exactly where the capacity audit accepts.
Since `B` bounds the account's memory only at the window's ticks, a device is such a server only when the window
contains every round boundary. -/
structure RealizesState {ι : Type} [Fintype ι] {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (G : StateGame ΩS)
    (r : Run d α ΩW ΩQ ΩS) (round : ℕ) where
  /-- the server the device is, at the other audits' coins and a bound on the capacity account's memory -/
  server : ∀ (o : ΩW × ΩQ) (B : ℕ),
    (∀ s, ∀ t ∈ (r.win .capacity).ticks, (r.schedule (o.1, o.2, s) t).mem (r.acct .capacity) ≤ B) →
      G.Server (B + d.chip + ∑ i, d.cap i * round)
  /-- the capacity audit's verdict is the game's -/
  accepts_iff : ∀ o B hB s, r.accepts .capacity (o.1, o.2, s) ↔ G.Passes (server o B hB) s

namespace Run

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] [DecidableEq α] (r : Run d α ΩW ΩQ ΩS)

/-- The device memory the accounts other than the capacity account and account `w` held over the capacity window, in
bit-ticks. -/
def othersHeldBut (w : α) (ω : ΩW × ΩQ × ΩS) : ℕ :=
  ∑ a ∈ (Finset.univ.erase (r.acct .capacity)).erase w, (r.schedule ω).held (r.win .capacity) a

/-- The accounts other than the capacity account and `w` are beyond the capacity residual `M - S`, less what `w` holds,
at the other audits' coins: on every one of the capacity audit's own coins, they hold more than `M - S` bits on average
over the capacity window, less what `w` holds. -/
def CapacityCovertBut (w : α) (S : ℝ) (ω : ΩW × ΩQ × ΩS) : Prop :=
  ∀ s : ΩS, ((d.mem : ℝ) - S) * (r.win .capacity).len - (r.schedule (ω.1, ω.2.1, s)).held (r.win .capacity) w <
    r.othersHeldBut w (ω.1, ω.2.1, s)

end Run

end Accounting
