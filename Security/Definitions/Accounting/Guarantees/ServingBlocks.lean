import Definitions.Accounting.ServingBlocks
import Mathlib.Data.Fintype.Sigma

/-!
# The serving-block bound

Layer 3 of `note:training/20261026T1840Z-draft-formal-properties` (TOP-95) and the working draft's §8: for an actual
circuit `A` with counted operations `J`, serving blocks `S` (`Accounting.ServingBlocks.block`) and dark leaves,

  `Σ_v ω_v ≤ Σ_S min {q(c_S, k_S) + ω_max μ k_S, n_S q(c, k) + q_U(S)} + q_U(L_dark)`.

Each argument of the minimum is Theorem 3 (counting) on one family of blocks inside `S`, with what that family leaves:
the one block `S` itself at `(c_S, k_S)`, or the `n_S` inference blocks counted for `S` at `(c, k)`. The statements are
about one actual circuit; Layer 2's envelopes supply the hypotheses, except with their probability.

Two terms differ from the note's statement:
* `ω_max μ k_S` bounds the weight assigned to `S` but not computed wholly inside it. `Ω_S` counts an operation by its
  representative, and the per-block cap counts only operations computed wholly inside a block, so an operation with its
  representative in `S` and a gate before `S`'s wipe or on another server escapes the cap. Such an operation has a gate
  outside `S` that a gate of `S` reads (`OperationBodies`), an unregistered input gate of `S`, and each such gate is in
  at most `μ` operations (`Overlap`): as every gate holds at least one bit, at most `μ |in(S) \ θ| ≤ μ k_S` operations,
  each of weight at most `ω_max`.
* The bound takes no alignment: `q_U(S)` caps whatever the inference blocks counted for `S` leave, wherever they lie.
  Alignment is what lets `q_U(S)` be the uncertified count, `ω_max |S \ W| + split(S)` (`UncertifiedElementary`).

Each guarantee has an honest and a nontrivial lemma beside its proof
(`Security/Proofs/Accounting/ServingBlocks/Witnesses.lean`): its premises hold at a small instance of a serving-block
layout, and its conclusion fails at an explicit circuit once one named premise is dropped. The boundary term is needed:
without it the bound fails at a circuit where every premise holds (`ServingBlockBound_nontrivial_overlap`).

`ServingBlockBoundRemote` is the remote case: no wipe or certifier, so `k_S` is unbounded and only the inference cap
applies; the serving blocks need only cover the counted operations.

The proofs are in `Security/Proofs/Accounting/ServingBlocks.lean`.
-/

namespace Accounting.Guarantees.ServingBlocks

open Verity Finset Accounting.ServingBlocks

/-- **The serving-block bound**. For an actual circuit `A` with registered inputs `θ` and counted operations `J`, servers
`X s` wiped at the cuts `κ s` of `[0, t)` and dark leaves `dark`: when the serving blocks are a `(c_S, k_S)`-partition
covering the counted operations off the dark leaves, `q` is a per-block cap at each `(c_S, k_S)` and at `(c, k)`, the
`n_S` inference blocks counted for each serving block `S` are `(c, k)`-blocks, the operations' bodies lead to their
representatives with each gate in at most `μ` of them, every gate holds at least one bit, the weights are at most
`ω_max`, `q_U(S)` caps what `S`'s inference blocks leave, and `q_U(L_dark)` caps the weight at the dark leaves:
`Σ_v ω_v ≤ Σ_S min {q(c_S, k_S) + ω_max μ k_S, n_S q(c, k) + q_U(S)} + q_U(L_dark)`. -/
def ServingBlockBound : Prop :=
  ∀ {L σ : Type*} [Fintype σ] (A : Placed L) (θ : Finset (Fin A.N)) (J : Counted A.toCircuit)
    (q : ℕ → ℕ → ℝ) (c k : ℕ) (dark : Finset L) (X : σ → Finset L) (t : ℕ) (κ : σ → Cuts t)
    (cS kS n : Block κ → ℕ) (inf : (S : Block κ) → Fin (n S) → Finset (Fin A.N))
    (ωmax : ℝ) (μ : ℕ) (qU : Block κ → ℝ) (qD : ℝ),
    ServingBlocksPartition A θ J dark X κ cS kS →
    (∀ S, PerBlockCap A.toCircuit θ J q (cS S) (kS S)) → PerBlockCap A.toCircuit θ J q c k →
    InferenceBlocks A.toCircuit θ n inf c k →
    OperationBodies A.toCircuit θ J → Overlap J μ → (∀ g, 1 ≤ A.w g) →
    0 ≤ ωmax → (∀ v ∈ J.V, J.ω v ≤ ωmax) →
    (∀ S, UncertifiedCap J (block A X κ S) (inf S) (qU S)) →
    UncertifiedCap J (A.region dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin A.N)) qD →
      J.total ≤ ∑ S, min (q (cS S) (kS S) + ωmax * μ * kS S) ((n S : ℝ) * q c k + qU S) + qD

/-- **The serving-block bound, remote**: with no wipe or certifier, only the inference cap applies, and the serving
blocks need only cover the counted operations off the dark leaves:
`Σ_v ω_v ≤ Σ_S (n_S q(c, k) + q_U(S)) + q_U(L_dark)`. -/
def ServingBlockBoundRemote : Prop :=
  ∀ {L σ : Type*} [Fintype σ] (A : Placed L) (θ : Finset (Fin A.N)) (J : Counted A.toCircuit)
    (q : ℕ → ℕ → ℝ) (c k : ℕ) (dark : Finset L) (X : σ → Finset L) (t : ℕ) (κ : σ → Cuts t)
    (n : Block κ → ℕ) (inf : (S : Block κ) → Fin (n S) → Finset (Fin A.N)) (qU : Block κ → ℝ) (qD : ℝ),
    ServingBlocksCover A J dark X t → PerBlockCap A.toCircuit θ J q c k →
    InferenceBlocks A.toCircuit θ n inf c k →
    (∀ S, UncertifiedCap J (block A X κ S) (inf S) (qU S)) →
    UncertifiedCap J (A.region dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin A.N)) qD →
      J.total ≤ ∑ S, ((n S : ℝ) * q c k + qU S) + qD

/-- **The elementary uncertified cap**: with weights at most `ω_max` and the blocks `B` counted for `R` holding every
certified gate (`W`) of `R` (`Alignment`), the weight of `R`'s counted operations computed wholly inside none of the
blocks is at most `ω_max |R \ W|`, the uncertified representatives, plus the operations split between a block and gates
outside it (`Counted.split`). At the dark leaves (no blocks, `W = ∅`) it is `ω_max |A[L_dark]|`. -/
def UncertifiedElementary : Prop :=
  ∀ (C : Circuit) (J : Counted C) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin C.N))
    (W : Finset (Fin C.N)) (ωmax : ℝ),
    0 ≤ ωmax → (∀ v ∈ J.V, J.ω v ≤ ωmax) → Alignment R B W →
      UncertifiedCap J R B (ωmax * (R \ W).card + J.split R B)

end Accounting.Guarantees.ServingBlocks
