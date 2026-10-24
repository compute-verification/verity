import Definitions.Accounting.Remote.Draw
import Definitions.Accounting.Remote.Held

/-!
# What the remote audit's opening draw assumes (trusted)

The premises of `Accounting.Guarantees.Remote.PooledDraw`, which derives the memory line's `LiveCoins` from the
verifier's opening draw (`Accounting.Remote.Draw`), each a named `Prop` on a session and the trace over its coins:

* `Live`: window `t`'s size and correct receipts are fixed before its openings coin is released, by the earlier
  windows' coins (their openings among them) and window `t`'s other coins, never by its own openings. The verifier
  releases the coin after the window's roots arrived (`RemoteAudit.verifyWindow`), and the roots bind the receipts.
* `Refused`: the verifier refuses a window one of whose opened receipts is wrong: on an accepted session every window's
  draw lands in its correct receipts (`RemoteAudit.checkOpening`).
* `Counted`: the trace's wrong receipts are among the windows': each credited pass's wrong registered blocks are wrong
  receipts of the window it ran in.

The draw's law is the model's, not a premise: each window's openings coin is a uniform subset
(`Accounting.Remote.Draw.Openings`), where the verifier derives it from the window's live coin with SHA-512
(`Key.subset`; `Properties.RemoteAudit.DrawIsSubset` states the code's draws are its distinct values).
-/

namespace Accounting.Remote.Draw.Assumptions

open Accounting.Remote.Held

variable {n m : ℕ} {s : Fin n → ℕ} {X : Fin n → Type}

/-- **Live coins, as the verifier releases them**: two outcomes that agree on the earlier windows' coins and on window
`t`'s other coins give window `t` the same size and the same correct receipts, whatever its openings coin. -/
def Live (D : Session n m s X) : Prop :=
  ∀ t (ω ω' : Coins n m s X), (∀ j, j < t → ω j = ω' j) → (ω t).1 = (ω' t).1 →
    D.size t ω = D.size t ω' ∧ D.correct t ω = D.correct t ω'

/-- **Refused**: on every accepting outcome, every window's opened receipts are among its correct ones. -/
def Refused (D : Session n m s X) (T : Trace (Coins n m s X)) : Prop :=
  ∀ ω, T.accepts ω → ∀ t, D.drawn t ω ⊆ D.correct t ω

/-- **Counted**: on every accepting outcome, the credited passes' wrong receipts of registered blocks are at most the
windows' wrong receipts. -/
def Counted (D : Session n m s X) (T : Trace (Coins n m s X)) : Prop :=
  ∀ ω, T.accepts ω → T.wrong ω ≤ D.wrong ω

end Accounting.Remote.Draw.Assumptions
