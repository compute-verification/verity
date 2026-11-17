module

public import Definitions.Assumptions.Flock
public import Definitions.Hash
public import Definitions.Core
public import Definitions.ResourceAccounting
public import Definitions.Assumptions.ResourceAccounting
public import Definitions.Assumptions.Hash
public import Definitions.Assumptions.Pous
public import Definitions.Slowdown.Cut
public import Definitions.Assumptions.Pouw
public import Definitions.Assumptions.WeightSecurity

/-! The named assumptions: the `Prop`s a guarantee takes as hypotheses without proof, never as an `axiom`. -/

@[expose] public section

/-! ## The min-cut bound's cut conjectures (H) and (E): no check of the circuit decides either, and one set of gates that beats it refutes it. -/

namespace Slowdown.Cut.Assumptions

open Verity Finset

/-- `training-cut-hardness` (conjectured; refuted by a set `S` of words with `κ(S) ≤ k` whose values cost more than
`κ(S)`). **(H), hardness** at arrays `A`, costs `β` a value and read cap `k`: every set `S` of words of `d` with
`κ(S) ≤ k` has `∑_r β_r |val(S ∩ A r)| ≤ κ(S)`. -/
def Hardness {ρ V : Type} [Fintype ρ] [DecidableEq V] (d : RegisteredCircuit) (A : ρ → Finset (Finset (Fin d.D.N)))
    (val : Finset (Fin d.D.N) → V) (β : ρ → ℝ) (k : ℕ) : Prop :=
  ∀ S, kappaWords d S ≤ k → ∑ r, β r * ((S ∩ A r).image val).card ≤ kappaWords d S

/-- `training-cut-exports` (conjectured; refuted by a set `B` of computed gates with `κ(B → E(B)) < b |E(B)|`).
**(E), exports** at `b`: every set `B` of computed gates has `κ(B → E(B)) ≥ b |E(B)|`. -/
def Exports {C : Circuit} {ι : Type} [Fintype ι] [DecidableEq ι] (W : Work C ι) (b : ℝ) : Prop :=
  ∀ B ⊆ C.computed, b * (W.clear B).card ≤ kappa C B ((W.clear B).biUnion W.y)

end Slowdown.Cut.Assumptions
