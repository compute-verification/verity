import Mathlib

/-!
# The causal horizon: rounds whose good sets precede their samples (trusted)

A session of the rate audit is a sequence of rounds (windows, or a window's chained passes), each closed by a uniform
sample of items (words, or a window's receipts) that the verifier opens. A round's good set is the items whose
opening it would pass: for the rate audit, its correct receipts. The prover fixes it before the round's sample is
drawn, as any function of everything it has seen: the earlier rounds' coins, their samples among them, and the round's
own coins other than its sample. `Causal` says exactly that. `Rate.Chain` is such a family (the correct receipts of
pass `j` read the store, the earlier passes' coins, and pass `j`'s stream and oracle family), and so is a session of
windows whose devices see only the earlier windows' coins.
-/

namespace Accounting.Rate

/-- Rounds `0, …, n − 1`, round `j` with its own coins in `X j` and a sample of `s j` of `M` items, and round `i`'s
good set `G i`, a function of all the rounds' coins: causal when `G i` reads only the rounds before `i` and round
`i`'s own coins other than its sample. -/
def Causal {M n : ℕ} {X : Fin n → Type} {s : Fin n → ℕ}
    (G : (i : Fin n) → ((j : Fin n) → X j × (Fin (s j) → Fin M)) → Finset (Fin M)) : Prop :=
  ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → (ω i).1 = (ω' i).1 → G i ω = G i ω'

end Accounting.Rate
