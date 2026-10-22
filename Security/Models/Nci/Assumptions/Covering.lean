import Definitions.Nci.Covering

/-!
# The covering bound's named assumptions

The covering bound (`Nci.Guarantees.CoveringBound`) rests on two premises about the circuit `C` and its multiplication
blocks (`Nci.Cover.Mults`), each at one width `b` for a weight and for an activation, in bits, and one compression rate
`r ∈ (0, 1]` for both, so that a weight and an activation each cost `c = r·b` bits: `ParameterHard` (Assumption 1,
parameter-hardness at `(s, r, b)`) and `ActivationChained` (Assumption 2, activation-chaining at `(D, r, b)`), which picks
out the multiplication blocks `𝓜*` the bound counts.
-/

namespace Nci.Assumptions

open Verity

/-- **Parameter-hardness at `(s, r, b)`** (Assumption 1 of the covering bound): every subcircuit `S` of at most `s` gates
reads at least `c = r·b` bits outside the registered parameters for each weight of the updated parameters `θ'` whose
multiplication with some token it completes: `c · |θ'_S| ≤ |in(S) \ θ|`.

An updated weight is computed after the commitment from more training than a block of `s` gates can do, so a block that
multiplies by it reads it, `b` bits at full width and at least `r·b` however it is encoded. A registered parameter in
`θ` is free, so `θ'` holds the updated weights only. It fails when a block rebuilds the weights it uses from fewer bits:
a registered base in `θ` that an update stacks on, a low-rank update (LoRA) whose factors are narrower than the weights
they update, codes below `r·b` bits, or an update small enough that a block of `s` gates computes it itself. -/
def ParameterHard (d : Declaration) {ι τ : Type} [DecidableEq ι] (M : Cover.Mults d ι τ) (s r b : ℝ) : Prop :=
  ∀ S : Cover.Subcircuit d.D, (S.gates.card : ℝ) ≤ s → r * b * (M.completed S).card ≤ ((S.ins \ d.θ).card : ℝ)

/-- **Activation-chaining at `(D, r, b)`** (Assumption 2 of the covering bound): the multiplication block `m_{i,j}` is
chained when there are a set `A_{i,j} ⊆ T_j` of gates feeding it (each read by a gate of `m_{i,j}`: its activation) and a
set `W_{i,j}` of at least `D` weights, each with a multiplication block `m_{i',j} ⊆ T_j` whose output gate reaches
`A_{i,j}` inside `T_j`, such that for every set `Q ⊆ θ'` of fewer than `D` weights, every `X ⊆ T_j` that cuts `A_{i,j}`
off from every `m_{i',j}` with `i' ∈ W_{i,j} \ Q` (`Cover.Mults.Cuts`: `X` meets `m_{i',j}`, or meets every path inside
`T_j` from `o_{i',j}` to `A_{i,j}`) has at least `c = r·b` gates. The bound counts the chained blocks, `𝓜*`.

A block that completes `m_{i,j}` holds its activation, and it has completed multiplications by fewer than `D` weights
(`Q`); what it reads of token `j` cuts the activation off from the multiplications by the other weights of `W_{i,j}`, so
it reads at least `c` bits of the token. An activation's own `b` bits are such a cut, and everything upstream of it (the
residual stream, `d · 16` bits a layer at bf16) is wider, so `c ≈ b` for activations as for weights, 16 at bf16.

In a transformer, with a read cap `k ≈ 10⁷` bits and `c ≈ 16`, the bound needs `D > k/c ≈ 6·10⁵`. A token's activations
cross that inside layer 1's attention, where one head's score already depends on about `2·10⁶` weights, so of a 70B
model's multiplications only layer 1's Q/K/V products, about 0.1%, are not chained. Without it the bound fails: on a
one-layer model trained on data generated from a 64-bit seed, one block does `s` multiplications for `c + 64` bits. -/
def ActivationChained (d : Declaration) {ι τ : Type} [DecidableEq ι] (M : Cover.Mults d ι τ) (D r b : ℝ) (i : ι)
    (j : τ) : Prop :=
  ∃ A ⊆ M.T j, (∀ a ∈ A, ∃ g ∈ M.m i j, a ∈ d.D.reads g) ∧
    ∃ W : Finset ι, D ≤ W.card ∧
      (∀ i' ∈ W, (i', j) ∈ M.P ∧ M.m i' j ⊆ M.T j ∧ ∃ a ∈ A, Cover.ReachesIn d.D (M.T j) (M.o i' j) a) ∧
      ∀ Q ⊆ M.θ', (Q.card : ℝ) < D → ∀ X ⊆ M.T j, (∀ i' ∈ W \ Q, M.Cuts X A i' j) → r * b ≤ X.card

end Nci.Assumptions
