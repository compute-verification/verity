import Definitions.Nci.Covering
import Mathlib.Basic.Real.Basic

/-!
# The covering bound's two assumptions, at one width `b` and one compression rate `r`

Daniel's covering bound (22 Oct 2026) charges weights and activations at one rate: `b` bits per weight and per
activation, compressible at most to a fraction `r ∈ (0, 1]` of it, so `c = r · b` bits each. Separate widths `b_w`, `b_a`
(`Nci.F`) are the special case the one-width form simplifies away.

* **Parameter-hardness** (`ParameterHard`): a small subcircuit reads `c` bits outside the registered parameters for
  every updated weight whose multiplications it completes.
* **Activation-chaining** (`ActivationChained`): a token's activation feeding a multiplication depends, through cuts of
  at least `c` gates, on the multiplications of many weights at that token. `𝓜*` (`chainedBlocks`) is the set of chained
  multiplication blocks; the guarantee counts it.

In a transformer each (weight, token) pair is exactly one multiplication of the forward pass, which is what
`MulBlocks` indexes by `P ⊆ θ' × J`.
-/

namespace Nci.Assumptions

open Verity

/-- **Parameter-hardness** `(s, r, b)` ([PARAMETER-HARDNESS]): every subcircuit `S` of `d` with `|S| ≤ s` reads at
least `c = r · b` bits outside the registered parameters for each weight whose multiplications it completes,
`c · |θ'_S| ≤ |in(S) \ θ|`.

Why it's realistic: the weights `θ'` are updated after the commitment, so none of them is in `θ`, and a block of at most
`s` gates cannot rebuild an updated weight from fewer bits than the weight compresses to, `r · b` of its `b` bits. -/
def ParameterHard (d : Declaration) (B : MulBlocks d) (s : ℕ) (r b : ℝ) : Prop :=
  ∀ S : Subcircuit d.D, S.gates.card ≤ s → r * b * (B.weightsCompleted S).card ≤ (S.ins \ d.θ).card

/-- **Activation-chaining** `(D, r, b)` of the multiplication block `m_{i,j}`, `p = (i, j)` ([ACTIVATION-CHAINING]):
`p ∈ P`, and there are an activation `A ⊆ T_j`, gates each read by a gate of `m_{i,j}`, and a set `W` of at least `D`
weights `i'` with multiplication blocks `m_{i',j} ⊆ T_j` whose output gates reach `A` inside `T_j`, such that for every
`Q ⊆ θ'` with `|Q| < D`, every `X ⊆ T_j` that cuts `A` off from every `m_{i',j}`, `i' ∈ W \ Q` (`MulBlocks.Cuts`), has
`|X| ≥ c = r · b`. The talk's earlier form had `c = 1` here, one bit per token; the width condition is what lets
activations cost `c` bits as weights do.

Why it's realistic: in a transformer with `k ≈ 10⁷` bits and `c ≈ 16`, the bound needs `D > k / c ≈ 6 · 10⁵`, and a
token's activations cross that inside layer 1's attention: one head's score already depends on about `2 · 10⁶` weights.
Only layer 1's Q/K/V products (about 0.1% of a 70B model's) are not chained. `c ≈ b` for activations (16 at bf16): an
activation's own `b` bits are a cut, and everything upstream of it (the residual stream, `d · 16` bits a layer) is wider.

Without it the bound fails: a one-layer model on data generated from a 64-bit seed lets one block do `s`
multiplications for `c + 64` bits of reads. -/
def ActivationChained (d : Declaration) (B : MulBlocks d) (D : ℕ) (r b : ℝ) (p : Fin B.nw × Fin B.nt) : Prop :=
  p ∈ B.P ∧ ∃ (A : Finset (Fin d.D.N)) (W : Finset (Fin B.nw)),
    A ⊆ B.T p.2 ∧ (∀ a ∈ A, ∃ g ∈ B.m p, a ∈ d.D.reads g) ∧ D ≤ W.card ∧
    (∀ i' ∈ W, (i', p.2) ∈ B.P ∧ B.m (i', p.2) ⊆ B.T p.2 ∧
      ∃ a ∈ A, FeedsWithin d.D (B.T p.2) (B.o (i', p.2)) a) ∧
    ∀ Q : Finset (Fin B.nw), Q.card < D → ∀ X ⊆ B.T p.2, (∀ i' ∈ W \ Q, B.Cuts p.2 A X i') → r * b ≤ X.card

open Classical in
/-- `𝓜*`: the chained multiplication blocks. -/
noncomputable def chainedBlocks (d : Declaration) (B : MulBlocks d) (D : ℕ) (r b : ℝ) : Finset (Fin B.nw × Fin B.nt) :=
  B.P.filter (ActivationChained d B D r b)

end Nci.Assumptions
