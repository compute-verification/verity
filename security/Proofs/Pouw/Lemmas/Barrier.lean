module

public import Definitions.Pouw.Barrier.Pinned

/-!
# The degenerate-input barrier: a lemma

These are steps of proofs, not guarantees: they sit with the proofs, and no record of the lock pins them.

Each guarantee is a `def G : Prop`, stated over `Definitions.Pouw`'s definitions with the named assumptions it rests on as
hypotheses (`Pouw.Assumptions`). Its proof is the theorem `Pouw.SecurityProofs.Barrier.G : Pouw.Guarantees.Barrier.G`.
The generic barrier (`Definitions.Pouw.Barrier.Pinned`). Each keeps the text, the docstring and the `open`s of the
statement it was in its source module.
-/

@[expose] public section

namespace Pouw.Guarantees.Barrier

open Pouw.Barrier Pouw.Barrier.Pinned Finset

/-- **The generic barrier.** In every cost model, for every protocol and domain: if `(U, 𝒜)` is a cheap point at
cost `c₀` with `q` oracle calls and `W = ∑_u W_ref(u) > 0`, then `G_g` fails for every `g < 1 − c₀/W` and every
error `δ` with `δ(q, N) < 1`: the adversary wins with probability 1. -/
def Generic : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] [Fintype S] [Nonempty S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (U : Workload) (A : Adversary CM) (c₀ : ℝ)
    (q : ℕ),
    CheapPoint CM P D U A c₀ q → 0 < totalWref P U.layout →
    ∀ g : ℝ, g < 1 - c₀ / totalWref P U.layout → ∀ δ : ℕ → ℕ → ℝ, GGap CM P D g δ → 1 ≤ δ q U.layout.N

end Pouw.Guarantees.Barrier
