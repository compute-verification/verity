import Definitions.Accounting.Remote.Fork

/-!
# What the receipts' binding bound assumes (trusted)

The premise of `Accounting.Guarantees.Remote.ReceiptsBindingOfFork` and `ReceiptsBindingOfCR`, a named `Prop` on the
receipts and a session:

* `HeldLikeliest`: the opening each receipt's root holds (`Receipts.held`) is fixed by its window's cut, and its row is
  the likeliest of the rows the prover presents for the receipt, opened and accepted, over the window's continuations
  (`cont t`): no row is shown alone on more of them (`Accounting.TS.tally`). That is how `Receipts.held` is stated,
  with ties broken in any fixed order; `Refused` and `Receipts.Moved` read the held opening, and this premise is what
  ties it to the prover's runs.
-/

namespace Accounting.Remote.Draw.Assumptions

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type}

/-- **The held row is the likeliest**: for every window `t`, coins `ω` and receipt `p`, the held opening's row is the
same on every continuation of `ω`'s cut, and no row is shown alone on more of those continuations than it. -/
def HeldLikeliest [Fintype X] (R : Receipts n κ X) (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) : Prop :=
  ∀ t ω p, (∀ ζ, (R.held t (cont t ω ζ) p).row = (R.held t ω p).row) ∧
    ∀ v, Accounting.TS.tally (cont t) (R.Shows D prf rest t p) ω v ≤
      Accounting.TS.tally (cont t) (R.Shows D prf rest t p) ω (R.held t ω p).row

end Accounting.Remote.Draw.Assumptions
