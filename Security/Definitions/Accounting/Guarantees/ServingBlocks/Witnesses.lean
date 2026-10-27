import Definitions.Accounting.Guarantees.ServingBlocks
import Definitions.Accounting.ServingBlocks.Instances

/-!
# The serving-block bound: honest and nontrivial lemmas

Each guarantee of `Accounting.Guarantees.ServingBlocks` has two statements here, at the instances of
`Security/Definitions/Accounting/ServingBlocks/Instances.lean`. `<Guarantee>_honest`: its premises hold together at
`Honest`, a small instance of a serving-block layout. `<Guarantee>_nontrivial`: at an explicit circuit every premise but
one named premise holds, that one fails, and the counted weight is above the bound's right-hand side, so the guarantee is
false without it. `ServingBlockBound` has two more, one for each premise that bounds its boundary term:
`ServingBlockBound_nontrivial_bodies` (`OperationBodies`) and `ServingBlockBound_nontrivial_overlap` (`Overlap`, and so
the boundary term itself).

The proofs are in `Security/Proofs/Accounting/ServingBlocks/Witnesses.lean`.
-/

namespace Accounting.Guarantees.ServingBlocks

open Verity Finset Accounting.ServingBlocks

/-- **The serving-block bound's premises hold** at `Honest`, at `(c_S, k_S) = (c, k) = (1, 1)`, one inference block per
serving block (the block itself), `ω_max = μ = 1`, `q_U(S) = 1` and `q_U(L_dark) = 1`. -/
def ServingBlockBound_honest : Prop :=
  ServingBlocksPartition Honest.A Honest.θ Honest.J Honest.dark Honest.X Honest.κ (fun _ => 1) (fun _ => 1) ∧
  (∀ _ : Block Honest.κ, PerBlockCap Honest.A.toCircuit Honest.θ Honest.J Honest.q 1 1) ∧
  PerBlockCap Honest.A.toCircuit Honest.θ Honest.J Honest.q 1 1 ∧
  InferenceBlocks Honest.A.toCircuit Honest.θ (fun _ : Block Honest.κ => 1) Honest.inf 1 1 ∧
  OperationBodies Honest.A.toCircuit Honest.θ Honest.J ∧ Overlap Honest.J 1 ∧ (∀ g, 1 ≤ Honest.A.w g) ∧
  (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Honest.J.V, Honest.J.ω v ≤ 1) ∧
  (∀ S, UncertifiedCap Honest.J (block Honest.A Honest.X Honest.κ S) (Honest.inf S) 1) ∧
  UncertifiedCap Honest.J (Honest.A.region Honest.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin Honest.A.N)) 1

/-- **The serving-block bound is false without the per-block cap at the serving blocks' size**: at `Counter`, with
`(c_S, k_S) = (1, 0)`, `(c, k) = (1, 1)`, one inference block, `ω_max = μ = 1` and `q_U = 0`, every other premise holds,
the cap at `(1, 0)` fails, and the counted weight `1` is above the right-hand side `0`. -/
def ServingBlockBound_nontrivial : Prop :=
  ServingBlocksPartition Counter.A Counter.θ Counter.J Counter.dark Counter.X Counter.κ (fun _ => 1) (fun _ => 0) ∧
  ¬ (∀ _ : Block Counter.κ, PerBlockCap Counter.A.toCircuit Counter.θ Counter.J Counter.q 1 0) ∧
  PerBlockCap Counter.A.toCircuit Counter.θ Counter.J Counter.q 1 1 ∧
  InferenceBlocks Counter.A.toCircuit Counter.θ (fun _ : Block Counter.κ => 1) Counter.inf 1 1 ∧
  OperationBodies Counter.A.toCircuit Counter.θ Counter.J ∧ Overlap Counter.J 1 ∧ (∀ g, 1 ≤ Counter.A.w g) ∧
  (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Counter.J.V, Counter.J.ω v ≤ 1) ∧
  (∀ S, UncertifiedCap Counter.J (block Counter.A Counter.X Counter.κ S) (Counter.inf S) 0) ∧
  UncertifiedCap Counter.J (Counter.A.region Counter.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin Counter.A.N)) 0 ∧
  ∑ _S : Block Counter.κ, min (Counter.q 1 0 + 1 * (1 : ℕ) * (0 : ℕ)) ((1 : ℕ) * Counter.q 1 1 + 0) + 0 <
    Counter.J.total

/-- **The serving-block bound is false without operation bodies**: at `Detached`, whose operation takes in gate 0 though
no gate of it reads gate 0, with `(c_S, k_S) = (c, k) = (1, 0)`, one inference block, `ω_max = μ = 1`, `q_U(S) = 1` and
`q_U(L_dark) = 0`, every other premise holds and the counted weight `1` is above the right-hand side `0`: the serving
block after the wipe reads nothing, yet the operation is assigned to it. -/
def ServingBlockBound_nontrivial_bodies : Prop :=
  ServingBlocksPartition Detached.A Detached.θ Detached.J Detached.dark Detached.X Detached.κ (fun _ => 1)
    (fun _ => 0) ∧
  (∀ _ : Block Detached.κ, PerBlockCap Detached.A.toCircuit Detached.θ Detached.J Detached.q 1 0) ∧
  PerBlockCap Detached.A.toCircuit Detached.θ Detached.J Detached.q 1 0 ∧
  InferenceBlocks Detached.A.toCircuit Detached.θ (fun _ : Block Detached.κ => 1) Detached.inf 1 0 ∧
  ¬ OperationBodies Detached.A.toCircuit Detached.θ Detached.J ∧ Overlap Detached.J 1 ∧
  (∀ g, 1 ≤ Detached.A.w g) ∧ (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Detached.J.V, Detached.J.ω v ≤ 1) ∧
  (∀ S, UncertifiedCap Detached.J (block Detached.A Detached.X Detached.κ S) (Detached.inf S) 1) ∧
  UncertifiedCap Detached.J (Detached.A.region Detached.dark Set.univ)
    (Fin.elim0 : Fin 0 → Finset (Fin Detached.A.N)) 0 ∧
  ∑ _S : Block Detached.κ, min (Detached.q 1 0 + 1 * (1 : ℕ) * (0 : ℕ)) ((1 : ℕ) * Detached.q 1 0 + 1) + 0 <
    Detached.J.total

/-- **The serving-block bound is false without overlap, and so without its boundary term**: at `Straddle`, whose
operation is split across the wipe, with `(c_S, k_S) = (c, k) = (1, 1)`, one inference block, `ω_max = 1`, `μ = 0`,
`q_U(S) = 1` and `q_U(L_dark) = 0`, every other premise holds, no gate is in at most `0` bodies, and the counted weight
`1` is above the right-hand side `0`. At `μ = 0` the boundary term `ω_max μ k_S` vanishes: the bound without it is
false where every premise holds at `μ = 1`. -/
def ServingBlockBound_nontrivial_overlap : Prop :=
  ServingBlocksPartition Straddle.A Straddle.θ Straddle.J Straddle.dark Straddle.X Straddle.κ (fun _ => 1)
    (fun _ => 1) ∧
  (∀ _ : Block Straddle.κ, PerBlockCap Straddle.A.toCircuit Straddle.θ Straddle.J Straddle.q 1 1) ∧
  PerBlockCap Straddle.A.toCircuit Straddle.θ Straddle.J Straddle.q 1 1 ∧
  InferenceBlocks Straddle.A.toCircuit Straddle.θ (fun _ : Block Straddle.κ => 1) Straddle.inf 1 1 ∧
  OperationBodies Straddle.A.toCircuit Straddle.θ Straddle.J ∧ Overlap Straddle.J 1 ∧ ¬ Overlap Straddle.J 0 ∧
  (∀ g, 1 ≤ Straddle.A.w g) ∧ (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Straddle.J.V, Straddle.J.ω v ≤ 1) ∧
  (∀ S, UncertifiedCap Straddle.J (block Straddle.A Straddle.X Straddle.κ S) (Straddle.inf S) 1) ∧
  UncertifiedCap Straddle.J (Straddle.A.region Straddle.dark Set.univ)
    (Fin.elim0 : Fin 0 → Finset (Fin Straddle.A.N)) 0 ∧
  ∑ _S : Block Straddle.κ, min (Straddle.q 1 1 + 1 * (0 : ℕ) * (1 : ℕ)) ((1 : ℕ) * Straddle.q 1 1 + 1) + 0 <
    Straddle.J.total

/-- **The remote bound's premises hold** at `Honest`, at `(c, k) = (1, 1)`, one inference block per serving block,
`q_U(S) = 1` and `q_U(L_dark) = 1`. -/
def ServingBlockBoundRemote_honest : Prop :=
  ServingBlocksCover Honest.A Honest.J Honest.dark Honest.X 2 ∧
  PerBlockCap Honest.A.toCircuit Honest.θ Honest.J Honest.q 1 1 ∧
  InferenceBlocks Honest.A.toCircuit Honest.θ (fun _ : Block Honest.κ => 1) Honest.inf 1 1 ∧
  (∀ S, UncertifiedCap Honest.J (block Honest.A Honest.X Honest.κ S) (Honest.inf S) 1) ∧
  UncertifiedCap Honest.J (Honest.A.region Honest.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin Honest.A.N)) 1

/-- **The remote bound is false without the per-block cap at the inference size**: at `Counter`, with `(c, k) = (1, 0)`,
one inference block and `q_U = 0`, every other premise holds, the cap at `(1, 0)` fails, and the counted weight `1` is
above the right-hand side `0`. -/
def ServingBlockBoundRemote_nontrivial : Prop :=
  ServingBlocksCover Counter.A Counter.J Counter.dark Counter.X 1 ∧
  ¬ PerBlockCap Counter.A.toCircuit Counter.θ Counter.J Counter.q 1 0 ∧
  InferenceBlocks Counter.A.toCircuit Counter.θ (fun _ : Block Counter.κ => 1) Counter.inf 1 0 ∧
  (∀ S, UncertifiedCap Counter.J (block Counter.A Counter.X Counter.κ S) (Counter.inf S) 0) ∧
  UncertifiedCap Counter.J (Counter.A.region Counter.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin Counter.A.N)) 0 ∧
  ∑ _S : Block Counter.κ, ((1 : ℕ) * Counter.q 1 0 + 0) + 0 < Counter.J.total

/-- **The elementary uncertified cap's premises hold** at `Honest`: at each serving block with its inference block and
every gate certified, and at the dark leaves with no blocks and none certified, at `ω_max = 1`. -/
def UncertifiedElementary_honest : Prop :=
  (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Honest.J.V, Honest.J.ω v ≤ 1) ∧
  (∀ S, Alignment (block Honest.A Honest.X Honest.κ S) (Honest.inf S) (block Honest.A Honest.X Honest.κ S)) ∧
  Alignment (Honest.A.region Honest.dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin Honest.A.N)) ∅

/-- **The elementary uncertified cap is false without alignment**: at `Counter`'s gate, with no blocks and the gate
certified, at `ω_max = 1`, the cap `ω_max |R \ W| + split(R)` is `0` and the operation's weight `1` is above it. -/
def UncertifiedElementary_nontrivial : Prop :=
  (0 : ℝ) ≤ 1 ∧ (∀ v ∈ Counter.J.V, Counter.J.ω v ≤ 1) ∧
  ¬ Alignment ({0} : Finset (Fin Counter.A.N)) (Fin.elim0 : Fin 0 → Finset (Fin Counter.A.N)) {0} ∧
  ¬ UncertifiedCap Counter.J {0} (Fin.elim0 : Fin 0 → Finset (Fin Counter.A.N))
    (1 * (({0} : Finset (Fin Counter.A.N)) \ {0}).card + Counter.J.split {0} Fin.elim0)

end Accounting.Guarantees.ServingBlocks
