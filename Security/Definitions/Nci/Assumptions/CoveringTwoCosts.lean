import Definitions.Nci.Assumptions.Covering

/-!
# The covering bound's two assumptions at two costs: `c_w` a weight, `c_a` an activation

`ParameterHard` and `ActivationChained` (`Definitions.Nci.Assumptions.Covering`) charge an updated weight and an activation the
one cost `c = r · b`. Here each has its own: parameter-hardness at a weight cost `c_w` (`ParameterHardAt`) and
activation-chaining at an activation cost `c_a` (`ActivationChainedAt`), over the same vocabulary (`Nci.Covering`). At
`c_w = c_a = r · b` they are `ParameterHard` and `ActivationChained`.

**Why two costs.** One shared `c` is at most both what an
updated weight costs to rebuild and an activation's width, `c_a ≤ 16` at bf16, 8 at FP8, 4 at FP4. An updated weight
compresses far below that, so one `c` charges every activation the weights' entropy. At fine-tuning's measured costs the
slowdown bound is `ρ` about 45 with two costs against 1.1 with one.

**What an updated weight is.** The weight index is abstract, so each reading is an instance; it only decides which `c_w`
holds. Read as a parameter *slot*, counting the products from a step `t0` after the commitment on, a block pays once per
slot and `c_w = H(θ'_{t0} | θ)`, what training added since the commitment, which grows with `t0`. Read as one gate per
training step's version, `c_w` is at most one step's increment entropy: on an exact Pythia resume about 1.1-1.7 bits a
weight at bf16, 0.2-0.3 at FP8, 0.03-0.05 at FP4.
-/

namespace Nci.Assumptions

open Verity Nci.Covering

/-- **Parameter-hardness at weight cost `c_w`** `(s, c_w)`: every subcircuit `S` of `d` with `|S| ≤ s` reads at least
`c_w` bits outside the registered parameters for each updated weight whose multiplications it completes,
`c_w · |θ'_S| ≤ |in(S) \ θ|`. `ParameterHard d B s r b` is this at `c_w = r · b`.

Why it's realistic: an updated weight is not in `θ`, and a block of at most `s` gates cannot rebuild it from fewer bits
than it carries beyond `θ`. Read as a parameter slot from a step `t0` after the commitment on, that is
`c_w = H(θ'_{t0} | θ)` (the module doc). It fails when a block rebuilds the weights it uses from fewer bits: a registered
base that an update stacks on, a low-rank update narrower than the weights it updates, or an update small enough for a
block of `s` gates to compute itself. -/
def ParameterHardAt (d : RegisteredCircuit) (B : MulBlocks d) (s : ℕ) (cw : ℝ) : Prop :=
  ∀ S : Subcircuit d.D, S.gates.card ≤ s → cw * (B.weightsCompleted S).card ≤ (S.ins \ d.θ).card

/-- **Activation-chaining at activation cost `c_a`** `(D, c_a)` of the multiplication block `m_{i,j}`, `p = (i, j)`:
`p ∈ P`, and there are an activation `A ⊆ T_j`, gates each read by a gate of `m_{i,j}`, and a set `W` of at least `D`
weights `i'` with multiplication blocks `m_{i',j} ⊆ T_j` whose output gates reach `A` inside `T_j`, such that for every
`Q` with `|Q| < D`, every `X ⊆ T_j` that cuts `A` off from every `m_{i',j}`, `i' ∈ W \ Q` (`MulBlocks.Cuts`), has
`|X| ≥ c_a`. `ActivationChained d B D r b` is this at `c_a = r · b`.

Why it's realistic: an activation's own bits are such a cut, and everything upstream of it (the residual stream) is wider,
so `c_a` is about an activation's width, 16 at bf16, 8 at FP8, 4 at FP4, whatever the weights' entropy. The depth the
guarantee asks is `c_w · D > k`: at `k ≈ 10⁷` bits and `c_w` about a bit, `D ≈ 10⁷`, which a token's activation crosses at
layer 1's attention output, whose Q, K and V weights number about `8 · 10⁷` in a 70B model (`d = 8192`, eight KV heads).
So, as at one cost, only layer 1's Q/K/V products are not chained. -/
def ActivationChainedAt (d : RegisteredCircuit) (B : MulBlocks d) (D : ℕ) (ca : ℝ) (p : Fin B.nw × Fin B.nt) : Prop :=
  p ∈ B.P ∧ ∃ (A : Finset (Fin d.D.N)) (W : Finset (Fin B.nw)),
    A ⊆ B.T p.2 ∧ (∀ a ∈ A, ∃ g ∈ B.m p, a ∈ d.D.reads g) ∧ D ≤ W.card ∧
    (∀ i' ∈ W, (i', p.2) ∈ B.P ∧ B.m (i', p.2) ⊆ B.T p.2 ∧
      ∃ a ∈ A, FeedsWithin d.D (B.T p.2) (B.o (i', p.2)) a) ∧
    ∀ Q : Finset (Fin B.nw), Q.card < D → ∀ X ⊆ B.T p.2, (∀ i' ∈ W \ Q, B.Cuts p.2 A X i') → ca ≤ X.card

open Classical in
/-- `𝓜*` at activation cost `c_a`: the multiplication blocks chained at `(D, c_a)`. -/
noncomputable def chainedBlocksAt (d : RegisteredCircuit) (B : MulBlocks d) (D : ℕ) (ca : ℝ) :
    Finset (Fin B.nw × Fin B.nt) :=
  B.P.filter (ActivationChainedAt d B D ca)

end Nci.Assumptions
