module

public import Definitions.ResourceAccounting.BlockProperties
public import Definitions.Slowdown.ServingBlocks

/-!
# A policy over the blocks' properties, the inference-only policy, and the chain

A policy takes the blocks accounting establishes as given (`Statement`, `FeasibleSet`). It is a check, a function of
the statement alone with every limit a public parameter, and a theorem: every actual circuit in the feasible set of a
statement that passes the check is policy-compliant (`Policy.Theorem`).

The inference-only policy's check (`inferenceOnly`): at most `n` inference blocks, each claimed of at most `c` gates
(work at `π`) with at most `k` bits of input from its own past, another block and the outside world; and each free
block claimed of at most `freeSize` gates, the share of capacity the utilization target leaves. Policy-compliant:
training efficiency at most `share + allowance`, the free blocks' share plus what the theorem allows inside inference
blocks. Training efficiency is a parameter here (`Slowdown.Slowdown.efficiency` over counted multiplications is the
existing one): its definition on the actual circuit is open. What the existing statements give and what is open:

* `Slowdown.Guarantees.SlowdownRemote` takes the inference blocks as `InferenceOnly.CompliesAtMost c k n` of a
  declaration and the residual through `Slowdown.Assumptions.ResidualAtMost`; `InferenceBlocksComply` (open) is the
  check's inference clauses read as the first, `FreeBlocksResidual` (open) its free clause read as the second.
* `ResourceAccounting.Guarantees.ServingBlocks.ServingBlockBound` takes server-day blocks as serving blocks
  (`ServingSizes`); `ServerDayBlocksServing` (open) is the server-day blocks' size and input read as those.
* `InferenceOnlyTheorem` (open) is the policy's theorem.

`Chain` is the composition: accounting statements that between them establish every claim of a statement that passes
the check, and the policy's theorem, give that an accepted run's actual circuit is policy-compliant except with the
statements' probabilities summed. Proved by `ResourceAccounting.SecurityProofs.BlockProperties.chain`.
-/

@[expose] public section

namespace ResourceAccounting

open Verity Finset
open scoped ENNReal

/-- **A policy**: its check on a statement, and when an actual circuit is policy-compliant. -/
structure Policy (L : Type*) where
  check : Statement → Bool
  compliant : Placed L → Prop

/-- **The policy's theorem**: every actual circuit in the feasible set of a statement that passes the check is
policy-compliant. -/
def Policy.Theorem {L : Type*} (pol : Policy L) : Prop :=
  ∀ (St : Statement) (A : Placed L) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)),
    pol.check St = true → FeasibleSet St A π θ → pol.compliant A

/-- **The chain**: over a monitoring period whose statement `St` passes the policy's check, the accounting statements
`s` (statement `x` establishing `p x i` for every block `i` of the class `K x`, except with probability at most
`ε x`) that between them establish every claim of `St`, and the policy's theorem, give that an accepted run's actual
circuit is policy-compliant except with probability at most `∑ x ∈ s, ε x`. -/
def Chain {L : Type*} {St : Statement} (P : Period L St) (pol : Policy L) {κ : Type} (s : Finset κ)
    (K : κ → Finset St.ι) (p : κ → St.ι → Property) (ε : κ → ℝ≥0∞) : Prop :=
  pol.check St = true → pol.Theorem → (∀ i, ∀ q ∈ St.claims i, ∃ x ∈ s, i ∈ K x ∧ p x i = q) →
    (∀ x ∈ s, P.Established (K x) (p x) (ε x)) →
    ExceptWith P.pr P.accepts (fun r => pol.compliant (P.actual r)) (∑ x ∈ s, ε x)

namespace InferenceOnly

/-- The claim is of size at most `c`. -/
def sizeAtMost (c : ℕ) : Property → Bool
  | .sizeAtMost s => decide (s ≤ c)
  | _ => false

/-- The claim is of input at most `k` bits from the block's own past, another block and the outside world. -/
def freshAtMost (k : ℕ) : Property → Bool
  | .inputAtMost [.past, .block, .world] s => decide (s ≤ k)
  | _ => false

/-- **The inference-only policy's limits**, public parameters: the inference blocks' count `n`, size `c` and input
`k`; the free blocks' size `freeSize`; and in the theorem's bound, the free blocks' share `share` and the allowance
inside inference blocks `allowance`. -/
structure Limits where
  n : ℕ
  c : ℕ
  k : ℕ
  freeSize : ℕ
  share : ℝ
  allowance : ℝ

/-- **The inference-only check**: at most `n` inference blocks, each claimed of size at most `c` and input at most
`k` bits; each free block claimed of size at most `freeSize`. -/
def check (lim : Limits) (St : Statement) : Bool :=
  decide (St.inference.card ≤ lim.n) &&
    decide (∀ i ∈ St.inference, (St.claims i).any (sizeAtMost lim.c) = true ∧
      (St.claims i).any (freshAtMost lim.k) = true) &&
    decide (∀ i ∈ St.free, (St.claims i).any (sizeAtMost lim.freeSize) = true)

end InferenceOnly

/-- **The inference-only policy** at the limits `lim`, with training efficiency `efficiency` (open on the actual
circuit): policy-compliant when training efficiency is at most `share + allowance`. -/
def inferenceOnly {L : Type*} (lim : InferenceOnly.Limits) (efficiency : Placed L → ℝ) : Policy L where
  check := InferenceOnly.check lim
  compliant A := efficiency A ≤ lim.share + lim.allowance

/-- **The inference-only theorem** (open, training): every actual circuit in the feasible set of a statement that
passes the inference-only check has training efficiency at most the free blocks' share plus the allowance. -/
def InferenceOnlyTheorem {L : Type*} (lim : InferenceOnly.Limits) (efficiency : Placed L → ℝ) : Prop :=
  (inferenceOnly lim efficiency).Theorem

open Classical in
/-- **The check's inference clauses give `CompliesAtMost`** (open, lean): in the feasible set of a statement that
passes the check at unit pricing, the inference blocks' gates, cut as a circuit with the registered input gates,
divide into at most `n` blocks of a `(c, k)`-partition: `InferenceOnly.CompliesAtMost c k n`, unfolded since this layer
does not import it, as `Slowdown.Guarantees.SlowdownRemote` takes it. -/
def InferenceBlocksComply (lim : InferenceOnly.Limits) : Prop :=
  ∀ (St : Statement) {L : Type*} (A : Placed L) (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true)
    (B : St.Gates A), InferenceOnly.check lim St = true → (∀ i, ∀ p ∈ St.claims i, B.Holds (fun _ => 1) θ i p) →
    let d : RegisteredCircuit :=
      .ofCircuit (A.cut (St.inference.biUnion B.gates)) θ fun a ha => by simp [Placed.cut, hθ a ha]
    ∃ nB ≤ lim.n, ∃ Pd : Partition d.D nB, Pd.IsCK d lim.c lim.k

/-- **The free blocks' size gives the residual** (open, training): in the feasible set, the residual's counted
multiplications `Mr A` are at most `κ/2` per gate of the free blocks, the form `Slowdown.Assumptions.ResidualAtMost`
bounds them in. -/
def FreeBlocksResidual {L : Type*} (Mr : Placed L → ℕ) (κ : ℝ) : Prop :=
  ∀ (St : Statement) (A : Placed L) (π : Fin A.N → ℕ) (θ : Finset (Fin A.N)) (B : St.Gates A),
    (∀ i, ∀ p ∈ St.claims i, B.Holds π θ i p) → (Mr A : ℝ) ≤ κ * (∑ i ∈ St.free, B.size π i : ℕ) / 2

/-- **The server-day blocks are serving blocks** (open, training): in the feasible set at unit pricing, some
partition of a declaration has serving blocks whose sizes and inputs (`ServingSizes`) are the server-day blocks' sizes
and their inputs from their own past and the outside world. -/
def ServerDayBlocksServing : Prop :=
  ∀ (St : Statement) {L : Type*} (A : Placed L) (θ : Finset (Fin A.N)) (B : St.Gates A),
    (∀ i, ∀ p ∈ St.claims i, B.Holds (fun _ => 1) θ i p) →
    ∃ (d : RegisteredCircuit) (nB : ℕ) (Pd : Partition d.D nB) (S : Finset (Fin nB)) (cS kS : Fin nB → ℕ),
      ServingBlocks.ServingSizes d Pd S cS kS ∧
      ∀ u ∈ S, ∃ i ∈ St.serverDay, cS u = B.size (fun _ => 1) i ∧ kS u = B.input θ i [.past, .world]

end ResourceAccounting
