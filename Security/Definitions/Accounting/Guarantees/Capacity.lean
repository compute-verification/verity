import Definitions.Accounting.Capacity

/-!
# A state-bounded game's soundness as a joint run's capacity guarantee, and derived state's share of the residual
(trusted)

A capacity audit whose game is sound against every server with state at most `S₀` bits embeds into the device model: in
a run that realizes the game (`Accounting.RealizesState`), the capacity audit has its guarantee
(`Accounting.Run.CapacitySound`) at the game's error, at any certificate whose bit-ticks over the capacity window, with
the chip and a round's side ingress, fit in `S₀` (`CapacityOfStateBound`). Its conclusion is `CapacitySound`, about a
bound on the capacity account's bit-ticks on every one of the capacity audit's own coins: the account's worst case over
them, not its charge at the coin drawn.

The window must contain every round boundary: the game bounds the state at each one, and the capacity account's memory
is bounded only at the window's ticks, so a run realizes the game only over such a window (for PoUS, the whole session).
Over a window of `T` ticks the certificate is `(S₀ − S_chip − ∑ cap_i · round) / T` bits on average, about `S₀/T`:
the game bounds the peak, and the account's bit-ticks bound its peak only by their whole total. A window of one tick
gives the full `S₀ − S_chip − ∑ cap_i · round` only for a one-round game. That is inherent to games graded by their
peak state; a useful certificate needs a game graded at each round boundary (the continuous audit, for P2), and that
choice is still open. No table should cite `CapacityOfStateBound` at a session window.

State derived from the game's public inputs (Pearl-C's weight state `B̃`, a function of PoUW's public weights) needs no
room in a server's state, so a device may keep it in an account other than the capacity account, and the capacity
certificate does not count it. It is not free: it shares the capacity residual `M − S` with every other account's state
(`DerivedStateResidual`). Like `CovertResidual`, that bounds the other accounts being beyond the residual on every one
of the capacity audit's own coins, not at the coin drawn: outside the error, some capacity coin leaves them within it. A
per-run reading is not claimed yet; a proof by truncation is in progress.

Each `def G : Prop` is proved by `Accounting.SecurityProofs.G : Accounting.Guarantees.G` (`Proofs.Accounting.Capacity`).
-/

namespace Accounting.Guarantees

open Accounting

/-- **A state-bounded game's soundness is the capacity guarantee.** For any run on any device whose capacity audit
realizes a game `G` with rounds of `round` ticks, if every server of `G` with state at most `S₀` bits passes with
probability at most `ε ≥ 0`, the capacity audit has its guarantee at error `ε` and at any certificate `S` with
`S·T + S_chip + ∑ cap_i · round ≤ S₀`, where `T` is the capacity window's length. A run realizes `G` only when the
window contains every round boundary, so `S` is then about `S₀/T` bits on average. -/
def CapacityOfStateBound : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype ι] [Fintype α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] (d : Device ι)
    (r : Run d α ΩW ΩQ ΩS) (G : StateGame ΩS) (round S₀ : ℕ) (S ε : ℝ),
    0 ≤ ε → (∀ n ≤ S₀, ∀ A : G.Server n, Pouw.pr (G.Passes A) ≤ ε) → RealizesState G r round →
    S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ S₀ →
      r.CapacitySound S ε

/-- **Derived state shares the capacity residual.** On any device and any run whose capacity audit has its guarantee at
certificate `S` and error `εS`, and for any account `w` other than the capacity account (one holding weight-derived
state such as Pearl-C's `B̃`, which the capacity certificate does not count), the probability over the three audits'
coins that the capacity audit accepts and, on every one of the capacity audit's own coins (at the other audits' coins
drawn), the accounts other than the capacity account and `w` hold more than `M − S` bits on average over the capacity
window, less what `w` holds, is at most `εS`. So `B̃` is not free: what `w` holds counts against `M − S`. Outside
that probability, some capacity coin leaves them within the residual, not necessarily the coin drawn; a per-run reading
is not claimed yet. -/
def DerivedStateResidual : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW]
    [Nonempty ΩQ] [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (w : α) (S εS : ℝ),
    w ≠ r.acct .capacity → r.CapacitySound S εS →
      Pouw.pr (fun ω => r.accepts .capacity ω ∧ r.CapacityCovertBut w S ω) ≤ εS

end Accounting.Guarantees
