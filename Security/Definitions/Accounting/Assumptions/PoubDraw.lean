import Definitions.Accounting.Poub.Draw
import Definitions.Accounting.Poub.Held

/-!
# What PoUB's opening draw assumes (trusted)

The premises of `Accounting.Guarantees.Poub.PooledDraw`, which derives the memory line's `LiveCoins` from the
verifier's opening draw (`Accounting.Poub.Draw`), each a named `Prop` on a session and the trace over its coins:

* `Live`: window `t`'s size and correct receipts are fixed before its openings coin is released, by the other coins and
  the earlier windows' openings coins, never by its own or a later one. The verifier releases the coin after the
  window's roots arrived (`PoubCheck.verifyWindow`), and a receipt is correct when the row its root holds
  (`Receipts.held`, the plurality of what the receipt shows over the window's continuations, read only off what
  precedes the coin) is: the roots bind the receipts only up to a collision, which is `Refused`'s and the guarantees'
  binding term (`Receipts.binding`), not this premise. The guarantees take it of the
  session the prover plays at each fixed coin commitment (`Committed.view c`), the world after the junk-commitment
  hybrid: the deployed prover holds a commitment to every coin from the start, and a window sized off that
  commitment's bits is not `Live` (red team RD-12). What the commitment reveals is the guarantees' hiding term
  (`Committed.hide`), not this premise.
* `Refused`: the verifier refuses a window one of whose opened receipts is wrong as presented
  (`PoubCheck.checkOpening`): on an accepted session each receipt a window's draw opens is correct, or moved
  (`Receipts.Moved`: accepted at its root and rank with a row other than the one the root holds). The guarantees
  charge a move as `Receipts.binding`, which `ReceiptsBindingOfFork` bounds by two runs from the window's cut that
  hash a SHA-512 collision (`Hash.Guarantees.RowBinding`) or split the receipt's roots.
* `Counted`: the trace's wrong receipts are among the windows': each credited pass's wrong registered blocks are wrong
  receipts of the window it ran in.

The draw is the deployed one, not a premise: each window's openings are `Key.subset` on the first blocks its coin gives
(`Session.opened`), the sampler exact on each draw's first word and the rest free. The openings coins are uniform and
independent of the other coins, as the challenger draws them (the trace's uniform measure on `Coins`).
-/

namespace Accounting.Poub.Draw.Assumptions

open Accounting.Poub.Held

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type}

/-- **Live coins, as the verifier releases them**: two outcomes that agree on the other coins and on the earlier
windows' openings coins give window `t` the same size and the same correct receipts, whatever its own openings coin and
the later ones. The guarantees read it of `Committed.view c` at every commitment `c`, the prover's session once the
commitment it holds is fixed apart from the coins. -/
def Live (D : Session n m s κ X) : Prop :=
  ∀ t (ω ω' : Coins n κ X), ω.1 = ω'.1 → (∀ j, j < t → ω.2 j = ω'.2 j) →
    D.size t ω = D.size t ω' ∧ D.correct t ω = D.correct t ω'

/-- **Refused**: on every accepting outcome, every receipt a window opens, on the first blocks `prf` and the
fallbacks `rest`, is among its correct ones or moved: presented, accepted at its root and rank, with a row other than
the one its root holds. -/
def Refused [Fintype X] (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (T : Trace (Coins n κ X)) (R : Receipts n κ X) : Prop :=
  ∀ ω, T.accepts ω → ∀ t, ∀ p ∈ D.opened prf rest t ω, p ∈ D.correct t ω ∨ R.Moved D prf rest t ω p

/-- **Counted**: on every accepting outcome, the credited passes' wrong receipts of registered blocks are at most the
windows' wrong receipts. -/
def Counted (D : Session n m s κ X) (T : Trace (Coins n κ X)) : Prop :=
  ∀ ω, T.accepts ω → T.wrong ω ≤ D.wrong ω

end Accounting.Poub.Draw.Assumptions
