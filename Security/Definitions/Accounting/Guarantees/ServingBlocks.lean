import Definitions.Accounting.ServingBlocks

/-!
# The serving-block bound

For a declaration `d` (the actual circuit with its registered input gates) with counted operations `J`, a partition `P`
whose units `S` are the serving blocks and whose other units are the dark remainder, and inference blocks `Pf`, a
`(c, k)`-partition refining `P` (`r`):

  `Σ_v ω_v ≤ Σ_{u ∈ S} min {q(c_S, k_S) + ω_max μ k_S, n_u q(c, k) + q_U(u)} + q_U(L_dark)`,

where `n_u` is the number of inference blocks inside serving block `u`. Each argument of the minimum is Theorem 3
(counting) on one family of blocks inside `u`, with what that family leaves: the one block `u` itself at `(c_S, k_S)`,
or its `n_u` inference blocks at `(c, k)` (`Verity.Guarantees.InferenceCap`). `Verity.Guarantees.ServingBlockSum` sums
them. The statements are about one circuit; Layer 2's envelopes supply the premises, each with its probability.

A serving block has at most `c_S` gates and reads at most `k_S` unregistered input bits, `Verity.Guarantees.Padding`'s
premise: padded to exactly `c_S` gates, the serving blocks read as many unregistered input bits as before and are proof
blocks of exactly `c_S` gates of the padded circuit. The per-block cap reads blocks of at most `c` gates, so the bound is
stated on the circuit at hand.

Two terms differ from Theorem 3 at the serving blocks:
* `ω_max μ k_S` bounds the weight assigned to `u` but not computed wholly inside it. `Ω_u` counts an operation by its
  representative, and the per-block cap counts only operations computed wholly inside a block, so an operation with its
  representative in `u` and a gate before `u`'s wipe or on another server escapes the cap. Such an operation has a gate
  outside `u` that a gate of `u` reads (`OperationBodies`), an unregistered input gate of `u`, and each such gate is in
  at most `μ` operations (`Overlap`): at most `μ |in(u) \ θ| ≤ μ k_S` operations, each of weight at most `ω_max`.
* `q_U(u)` caps the weight assigned to `u` that its inference blocks leave. Every operation gate is in an inference
  block, so these are the operations split between inference blocks, or between one and gates outside `u`
  (`UncertifiedElementary`).

Each guarantee has an honest and a nontrivial lemma
(`Security/Definitions/Accounting/Guarantees/ServingBlocks/Witnesses.lean`): its premises hold together at a small
serving-block layout, and its conclusion fails at an explicit circuit once one named premise is dropped. The boundary
term is needed: without it the bound fails at a circuit where every premise holds
(`ServingBlockBound_nontrivial_overlap`).

`ServingBlockBoundRemote` is the remote case: no wipe or certifier, so `k_S` is unbounded and only the inference cap
applies.

The proofs are in `Security/Proofs/Accounting/ServingBlocks.lean`.
-/

namespace Accounting.Guarantees.ServingBlocks

open Verity Finset Accounting.ServingBlocks

/-- **The serving-block bound**. For a declaration `d` with counted operations `J`, a partition `P` whose units `S` are
the serving blocks (the others the dark remainder) and a `(c, k)`-partition `Pf` of inference blocks refining it: when
each serving block `u` has at most `c_S u` gates and reads at most `k_S u` unregistered input bits, `q` is a per-block cap
at each `(c_S u, k_S u)` and at `(c, k)`, the operations' bodies lead to their representatives with each gate in at most
`μ` of them, the weights are at most `ω_max`, `q_U(u)` caps what `u`'s inference blocks leave and `q_U(L_dark)` caps the
weight in the dark remainder:
`Σ_v ω_v ≤ Σ_{u ∈ S} min {q(c_S, k_S) + ω_max μ k_S, n_u q(c, k) + q_U(u)} + q_U(L_dark)`. -/
def ServingBlockBound : Prop :=
  ∀ (d : Registered) (J : Counted d.D) (q : ℕ → ℕ → ℝ) (c k B nf : ℕ) (P : Partition d.D B) (S : Finset (Fin B))
    (cS kS : Fin B → ℕ) (Pf : Partition d.D nf) (r : Pf.Refines P) (ωmax : ℝ) (μ : ℕ) (qU : Fin B → ℝ) (qD : ℝ),
    (∀ u ∈ S, (P.gates u).card ≤ cS u ∧ (d.unregistered (P.gates u)).card ≤ kS u) →
    (∀ u ∈ S, PerBlockCap d J q (cS u) (kS u)) → Pf.IsCK d c k → PerBlockCap d J q c k →
    OperationBodies d J → Overlap J μ → 0 ≤ ωmax → (∀ v ∈ J.V, J.ω v ≤ ωmax) →
    (∀ u ∈ S, UncertifiedCap J (P.gates u) (fun i : {i // r.parent i = u} => Pf.gates i) (qU u)) →
    UncertifiedCap J ((univ \ S).biUnion P.gates) (Fin.elim0 : Fin 0 → Finset (Fin d.D.N)) qD →
      J.total ≤ ∑ u ∈ S, min (q (cS u) (kS u) + ωmax * μ * kS u)
        (((univ.filter fun i => r.parent i = u).card : ℝ) * q c k + qU u) + qD

/-- **The serving-block bound, remote**: with no wipe or certifier, only the inference cap applies:
`Σ_v ω_v ≤ Σ_{u ∈ S} (n_u q(c, k) + q_U(u)) + q_U(L_dark)`. -/
def ServingBlockBoundRemote : Prop :=
  ∀ (d : Registered) (J : Counted d.D) (q : ℕ → ℕ → ℝ) (c k B nf : ℕ) (P : Partition d.D B) (S : Finset (Fin B))
    (Pf : Partition d.D nf) (r : Pf.Refines P) (qU : Fin B → ℝ) (qD : ℝ),
    Pf.IsCK d c k → PerBlockCap d J q c k →
    (∀ u ∈ S, UncertifiedCap J (P.gates u) (fun i : {i // r.parent i = u} => Pf.gates i) (qU u)) →
    UncertifiedCap J ((univ \ S).biUnion P.gates) (Fin.elim0 : Fin 0 → Finset (Fin d.D.N)) qD →
      J.total ≤ ∑ u ∈ S, (((univ.filter fun i => r.parent i = u).card : ℝ) * q c k + qU u) + qD

/-- **The elementary uncertified cap**: when the inference blocks `Pf` refine the partition `Pc`, the weight of unit
`s`'s counted operations computed wholly inside none of the inference blocks inside `s` is at most the weight of the
operations split between one of them and gates outside it (`Counted.split`). Every representative is an operation gate,
so it is in an inference block, and that block is inside `s`. -/
def UncertifiedElementary : Prop :=
  ∀ (C : Circuit) (J : Counted C) (nf nc : ℕ) (Pf : Partition C nf) (Pc : Partition C nc) (r : Pf.Refines Pc)
    (s : Fin nc),
    UncertifiedCap J (Pc.gates s) (fun i : {i // r.parent i = s} => Pf.gates i)
      (J.split (Pc.gates s) (fun i : {i // r.parent i = s} => Pf.gates i))

end Accounting.Guarantees.ServingBlocks
