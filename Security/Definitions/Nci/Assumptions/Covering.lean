import Definitions.Nci.Covering
import Mathlib.Basic.Real.Basic

/-!
# The covering bound's two assumptions, at one width `b` and one compression rate `r`

The covering bound charges weights and activations at one rate: `b` bits per weight and per
activation, compressible at most to a fraction `r ∈ (0, 1]` of it, so `c = r · b` bits each, where `Nci.F` keeps separate
widths `b_w`, `b_a`. The guarantee needs no bound on `r` or `b`: `c D > k` alone makes `c` positive.

* **Parameter-hardness** (`ParameterHard`): a small subcircuit reads `c` bits outside the registered parameters for
  every updated weight whose multiplications it completes.
* **Activation-chaining** (`ActivationChained`): a token's activation feeding a multiplication depends, through cuts of
  at least `c` gates, on the multiplications of many weights at that token. `𝓜*` (`chainedBlocks`) is the set of chained
  multiplication blocks; the guarantee counts it.

In a transformer each (weight, token) pair is exactly one multiplication of the forward pass, which is what
`MulBlocks` indexes by `P ⊆ θ' × J`.

**What an updated weight is.** The weight index `i : Fin nw` is abstract, so both readings below are instances of the
guarantee and it holds under either; only whether `ParameterHard` holds with a useful `c` depends on the reading.
* (a) `i` is a weight *gate*, each training step's version its own `i`. A block that completes products for consecutive
  versions is handed version `t` plus small increments, so `ParameterHard` holds only with `c` at most the one-step
  increment's entropy, a small fraction of the weight's width.
* (b) `i` is the parameter *slot*: versions share an index, and the products counted are those from a step `t0` after the
  commitment on. A block pays once per slot, and `c = H(θ'_{t0} | θ)`, which grows with `t0`. These docstrings take (b),
  as it is what "θ plus what training added since the commitment" means.

**One shared `c`.** With one rate, `c` is at most both a weight's entropy and an activation's width
(`c_a ≤ 16` at bf16, 8 at FP8, 4 at FP4), so the weights' low entropy caps what an activation is charged. The proof goes through unchanged with `c_w` in
`ParameterHard` and `c_a` in `ActivationChained`, giving `(k / c_w)(k / c_a) + k` a block; that two-rate form is
`Nci.Guarantees.CoveringBoundTwoCosts`.
-/

namespace Nci.Assumptions

open Verity Nci.Covering

/-- **Parameter-hardness** `(s, r, b)` ([PARAMETER-HARDNESS]): every subcircuit `S` of `d` with `|S| ≤ s` reads at
least `c = r · b` bits outside the registered parameters for each weight whose multiplications it completes,
`c · |θ'_S| ≤ |in(S) \ θ|`.

Why it's realistic: the weights `θ'` are updated after the commitment, so none of them is in `θ`, and a block of at most
`s` gates cannot rebuild an updated weight from fewer bits than the weight compresses to, `r · b` of its `b` bits. That
holds with a useful `c` when `i` is a parameter slot counted from a step `t0` after the commitment (`c = H(θ'_{t0} | θ)`);
if each step's version were its own `i`, `c` would be at most one step's increment entropy (the module doc). -/
def ParameterHard (d : RegisteredCircuit) (B : MulBlocks d) (s : ℕ) (r b : ℝ) : Prop :=
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
def ActivationChained (d : RegisteredCircuit) (B : MulBlocks d) (D : ℕ) (r b : ℝ) (p : Fin B.nw × Fin B.nt) : Prop :=
  p ∈ B.P ∧ ∃ (A : Finset (Fin d.D.N)) (W : Finset (Fin B.nw)),
    A ⊆ B.T p.2 ∧ (∀ a ∈ A, ∃ g ∈ B.m p, a ∈ d.D.reads g) ∧ D ≤ W.card ∧
    (∀ i' ∈ W, (i', p.2) ∈ B.P ∧ B.m (i', p.2) ⊆ B.T p.2 ∧
      ∃ a ∈ A, FeedsWithin d.D (B.T p.2) (B.o (i', p.2)) a) ∧
    ∀ Q : Finset (Fin B.nw), Q.card < D → ∀ X ⊆ B.T p.2, (∀ i' ∈ W \ Q, B.Cuts p.2 A X i') → r * b ≤ X.card

open Classical in
/-- `𝓜*`: the chained multiplication blocks. -/
noncomputable def chainedBlocks (d : RegisteredCircuit) (B : MulBlocks d) (D : ℕ) (r b : ℝ) : Finset (Fin B.nw × Fin B.nt) :=
  B.P.filter (ActivationChained d B D r b)

end Nci.Assumptions
