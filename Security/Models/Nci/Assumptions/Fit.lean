import Definitions.Nci.Fit

/-!
# NCI over windows: the fit bound's named assumptions

The fit bound rests on two empirical premises about the reference training circuit `c` and its division into windows,
measured rather than proved ("The fit bound", Notion, sections 1, 6 and 7): `Reach`, the loss premise F3, and `Fits`,
the capacity premise F1 with the boundary premise F4, at window scale. `Security/lean-audit.json` lists this module under
`shape.premises`: each is an assumption about the world, not a conjecture.
-/

namespace Nci.Assumptions

open Verity

/-- **F3, the loss premise (empirical)**: every thread that does a counted multiplication uses at least `Nstar` distinct
weights in its counted multiplications. In a model that reaches the target loss `L*`, every token's computation depends
on at least `N*` updated weights, `N*` being the fewest parameters that reach `L*` on any amount of data under the
recipe's loss law: 87.6B for a Llama 3.1 405B-class target under Besiroglu et al.'s refit of Chinchilla's law, and
17–218B with its exponents at ±1 SE.

It fails when a model reaches `L*` with fewer weights per thread: sequence-routed path composition (the main open
attack), retrieval, looped models, a better recipe than the one the law is fitted to, or a registered base in `θ` that an
update stacks on. -/
def Reach (c : Declaration) (T : Threading c) (Nstar : ℕ) : Prop :=
  ∀ t, (T.threadMults Finset.univ t).Nonempty → Nstar ≤ (T.threadWeights Finset.univ t).card

/-- **F1 and F4 at window scale (empirical)**: every window `j` of `P` reads at least `b_w` bits of novel information
for each weight it uses and `w_e` bits for each thread it works on but didn't start:
`b_w · #(weights j uses) + w_e · #(threads j entered) ≤ width(in(P j) \ θ)`.

F1 is the weight half of `WeightsIncompressible` at window scale: `b_w` is a rate, in bits of novel information per
updated weight used (2.5 at the C1 floor). F4: one window, `start t`, starts each thread from its token id, and any
other window that works on it reads at least `w_e` bits of its values, its hidden state coming in and its gradient going
back, or its context's keys and values, which cost more (1,147 / 35,840 / 114,688 bits, low / central / high, at
`d_model` 7168). A window may read a thread's state from what it carries across the wipe that opens it as well as from
the network: both are reads.

It fails when a window rebuilds the weights it uses from fewer bits (registered bases, low-rank updates, very low-bit
codes, or updates it computes itself from fewer bits of activations and gradients; a weight in `θ` is a free read, so
`T.W` must hold updated weights only), or when a window other than `start t` works on a thread from fewer bits: an
ensemble, whose members each start the token from its id, or a low-bit boundary code. -/
def Fits (c : Declaration) (T : Threading c) {k : ℕ} (P : Partition c.D k) (start : Fin T.m → Fin k)
    (b_w w_e : ℝ) : Prop :=
  ∀ j, b_w * (T.weightsUsed (P.gates j)).card + w_e * (T.entered P start j).card ≤ novelReads c (P.gates j)

end Nci.Assumptions
