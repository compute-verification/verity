module

public import Definitions.Slowdown.Guarantees.ServingBlocks
public import Definitions.Slowdown.ServingBlocks.Instances

/-! The serving-block bound's honest and nontrivial lemmas, at small serving-block layouts. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.ServingBlocks

open Verity Finset ResourceAccounting.ServingBlocks

/-- **The serving-block bound's premises hold** at `Honest`, at `(c_S, k_S) = (c, k) = (1, 1)`, the serving blocks as
their own inference blocks, `ω_max = μ = 1`, `q_U(u) = 1` and `q_U(L_dark) = 1`. -/
def ServingBlockBound_honest : Prop :=
  ServingSizes Honest.d Honest.P Honest.S (fun _ => 1) (fun _ => 1) ∧
  (∀ u ∈ Honest.S, PerBlockCap Honest.d Honest.J Honest.q 1 1) ∧ Honest.P.IsCK Honest.d 1 1 ∧
  PerBlockCap Honest.d Honest.J Honest.q 1 1 ∧ OperationBodies Honest.d Honest.J ∧ Overlap Honest.J 1 ∧
  (0 : ℝ) ≤ 1 ∧ WeightsAtMost Honest.J 1 ∧
  (∀ u ∈ Honest.S, UncertifiedCap Honest.J (Honest.P.gates u)
    (fun i : {i // Honest.r.parent i = u} => Honest.P.gates i) 1) ∧
  UncertifiedCap Honest.J ((univ \ Honest.S).biUnion Honest.P.gates) (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N)) 1

/-- **The serving-block bound is false without the per-block cap at the serving blocks' size**: at `Counter`, with
`(c_S, k_S) = (1, 0)`, `(c, k) = (1, 1)`, the serving block as its own inference block, `ω_max = μ = 1` and `q_U = 0`,
every other premise holds, the cap at `(1, 0)` fails, and the counted weight `1` is above the right-hand side `0`. -/
def ServingBlockBound_nontrivial : Prop :=
  ServingSizes Counter.d Counter.P Counter.S (fun _ => 1) (fun _ => 0) ∧
  ¬ (∀ u ∈ Counter.S, PerBlockCap Counter.d Counter.J Counter.q 1 0) ∧ Counter.P.IsCK Counter.d 1 1 ∧
  PerBlockCap Counter.d Counter.J Counter.q 1 1 ∧ OperationBodies Counter.d Counter.J ∧ Overlap Counter.J 1 ∧
  (0 : ℝ) ≤ 1 ∧ WeightsAtMost Counter.J 1 ∧
  (∀ u ∈ Counter.S, UncertifiedCap Counter.J (Counter.P.gates u)
    (fun i : {i // Counter.r.parent i = u} => Counter.P.gates i) 0) ∧
  UncertifiedCap Counter.J ((univ \ Counter.S).biUnion Counter.P.gates)
    (Fin.elim0 : Fin 0 → Finset (Fin Counter.d.D.N)) 0 ∧
  ¬ Counter.J.total ≤ ∑ u ∈ Counter.S, min (Counter.q 1 0 + 1 * (1 : ℕ) * (0 : ℕ))
    (((univ.filter fun i => Counter.r.parent i = u).card : ℝ) * Counter.q 1 1 + 0) + 0

/-- **The remote bound's premises hold** at `Honest`, at `(c, k) = (1, 1)`, the serving blocks as their own inference
blocks, `q_U(u) = 1` and `q_U(L_dark) = 1`. -/
def ServingBlockBoundRemote_honest : Prop :=
  Honest.P.IsCK Honest.d 1 1 ∧ PerBlockCap Honest.d Honest.J Honest.q 1 1 ∧
  (∀ u ∈ Honest.S, UncertifiedCap Honest.J (Honest.P.gates u)
    (fun i : {i // Honest.r.parent i = u} => Honest.P.gates i) 1) ∧
  UncertifiedCap Honest.J ((univ \ Honest.S).biUnion Honest.P.gates) (Fin.elim0 : Fin 0 → Finset (Fin Honest.d.D.N)) 1

/-- **The remote bound is false without the per-block cap at the inference size**: at `Counter`, with `(c, k) = (1, 0)`,
the serving block as its own inference block and `q_U = 0`, every other premise holds, the cap at `(1, 0)` fails, and
the counted weight `1` is above the right-hand side `0`. -/
def ServingBlockBoundRemote_nontrivial : Prop :=
  Counter.P.IsCK Counter.d 1 0 ∧ ¬ PerBlockCap Counter.d Counter.J Counter.q 1 0 ∧
  (∀ u ∈ Counter.S, UncertifiedCap Counter.J (Counter.P.gates u)
    (fun i : {i // Counter.r.parent i = u} => Counter.P.gates i) 0) ∧
  UncertifiedCap Counter.J ((univ \ Counter.S).biUnion Counter.P.gates)
    (Fin.elim0 : Fin 0 → Finset (Fin Counter.d.D.N)) 0 ∧
  ¬ Counter.J.total ≤
    ∑ u ∈ Counter.S, (((univ.filter fun i => Counter.r.parent i = u).card : ℝ) * Counter.q 1 0 + 0) + 0

/-- **The elementary uncertified cap is attained** at `Honest`, at its serving block after the wipe with that block as
its one inference block: the weight the inference block leaves is the split weight, which is `1`, gate 2's operation
split across the wipe. -/
def UncertifiedElementary_honest : Prop :=
  Honest.J.outside (Honest.P.gates 1) (fun i : {i // Honest.r.parent i = 1} => Honest.P.gates i) =
    Honest.J.split (Honest.P.gates 1) (fun i : {i // Honest.r.parent i = 1} => Honest.P.gates i) ∧
  Honest.J.split (Honest.P.gates 1) (fun i : {i // Honest.r.parent i = 1} => Honest.P.gates i) = 1

/-- **A tighter elementary uncertified cap is false** at `Straddle`, at its serving block after the wipe with that block
as its one inference block: the operation split across the wipe is computed wholly inside no inference block, so the
weight left is the split weight, and the cap fails at every right-hand side below it, `0` among them. -/
def UncertifiedElementary_nontrivial : Prop :=
  Straddle.J.outside (Straddle.P.gates 1) (fun i : {i // Straddle.r.parent i = 1} => Straddle.P.gates i) =
    Straddle.J.split (Straddle.P.gates 1) (fun i : {i // Straddle.r.parent i = 1} => Straddle.P.gates i) ∧
  ¬ UncertifiedCap Straddle.J (Straddle.P.gates 1) (fun i : {i // Straddle.r.parent i = 1} => Straddle.P.gates i) 0

end ResourceAccounting.Guarantees.ServingBlocks
