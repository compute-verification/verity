import Definitions.Nci.Fit

/-!
# NCI over windows: the fit bound's named assumptions

The fit bound rests on two empirical premises about the reference training circuit `c` and its division into windows,
measured rather than proved: `Reach`, the loss premise F3, and `Fits`, the capacity premise F1 with the boundary premise
F4, at window scale. `ReachSlots` and `FitsSlots` are the same premises counted in parameter slots (`Threading.Slots`),
so that a window spanning several optimizer steps is charged once per parameter, not once per version.
`Security/lean-audit.json` lists this module under `shape.premises`: each is an assumption about the world. `Reach` and
`ReachSlots` are predicates on the threading `T`, so its `shape.conjectures` lists them as open: false at token threads
for a token-routed mixture of experts, the per-sequence form at sequence threads.
-/

namespace Nci.Assumptions

open Verity

/-- **F3, the loss premise (empirical)**: every thread that does a counted multiplication uses at least `Nstar` distinct
weights in its counted multiplications. In a model that reaches the target loss `L*`, every token's computation depends
on at least `N*` updated weights, `N*` being the fewest parameters that reach `L*` on any amount of data under the
recipe's loss law: 87.6B for a Llama 3.1 405B-class target under Besiroglu et al.'s refit of Chinchilla's law, and
17–218B with its exponents at ±1 SE.

It fails when a model reaches `L*` with fewer weights per thread: a token-routed mixture of experts at token threads,
whose tokens each use fewer than `N*` weights (about 32B a token for Kimi K2), sequence-routed path composition (the
main open attack), retrieval, looped models, a better recipe than the one the law is fitted to, or a registered base in
`θ` that an update stacks on. -/
def Reach (c : RegisteredCircuit) (T : Threading c) (Nstar : ℕ) : Prop :=
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
def Fits (c : RegisteredCircuit) (T : Threading c) {k : ℕ} (P : Partition c.D k) (start : Fin T.m → Fin k)
    (b_w w_e : ℝ) : Prop :=
  ∀ j, b_w * (T.weightsUsed (P.gates j)).card + w_e * (T.entered P start j).card ≤ novelReads c (P.gates j)

/-- **F3 per parameter (empirical)**: every thread that does a counted multiplication uses at least `Nstar` distinct
parameter slots of `σ` in its counted multiplications. It is `Reach` counted in parameters rather than weight gates: in
a model that reaches the target loss `L*`, every token's computation depends on at least `N*` updated parameters, at
whatever optimizer step it runs. A thread that uses one version of each parameter uses as many slots as weights, so
there it is `Reach`.

It fails where `Reach` does: a token-routed mixture of experts whose tokens each use fewer than `N*` parameters (about
32B a token for Kimi K2), sequence-routed path composition, retrieval, looped models, a better recipe than the one the
law is fitted to, or a registered base in `θ` that an update stacks on. -/
def ReachSlots (c : RegisteredCircuit) (T : Threading c) (σ : T.Slots) (Nstar : ℕ) : Prop :=
  ∀ t, (T.threadMults Finset.univ t).Nonempty → Nstar ≤ (σ.threadSlots Finset.univ t).card

/-- **F1 per parameter slot and F4 at window scale (empirical)**: every window `j` of `P` reads at least `b_w` bits of
novel information for each parameter slot of `σ` it uses and `w_e` bits for each thread it works on but didn't start:
`b_w · #(slots j uses) + w_e · #(threads j entered) ≤ width(in(P j) \ θ)`.

It is `Fits` with each parameter charged once per window, however many of its versions (optimizer steps) the window
uses: a window that carries a parameter across steps reads it once, in the version it starts from, and computes the
later versions itself. F4 is as in `Fits`.

It fails when a window rebuilds a parameter's first version it uses from fewer bits (a registered base or
initialization in `θ` that the window updates from, low-rank updates, very low-bit codes, or a version it computes from
fewer bits of activations and gradients, which a window does that holds a slot's every version from its
initialization), or when a window other than `start t` works on a thread from fewer bits: an ensemble, whose members
each start the token from its id, or a low-bit boundary code. -/
def FitsSlots (c : RegisteredCircuit) (T : Threading c) (σ : T.Slots) {k : ℕ} (P : Partition c.D k)
    (start : Fin T.m → Fin k) (b_w w_e : ℝ) : Prop :=
  ∀ j, b_w * (σ.slotsUsed (P.gates j)).card + w_e * (T.entered P start j).card ≤ novelReads c (P.gates j)

end Nci.Assumptions
