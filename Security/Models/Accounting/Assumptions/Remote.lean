import Definitions.Accounting.Remote

/-!
# The remote audit's named assumptions (trusted)

What the remote audit's node-level guarantee assumes beyond the model's rows, each a named `Prop` on a run:

* `ComputeAdditive` is W at the rebuild pricing: the node's work at π_r outside PoUW's account and PoUW's account's add
  within the Compute row's `Φ_r · T`. PoUW certifies `W_r` of the latter (`Run.Certified`); W is known and may be zero,
  and at `W_r = 0` the guarantee needs only the Compute row.
* `Identity` bounds the prover's work outside the node at PoUW's pricing by `Φ_out` per tick: no undeclared
  tensor-class hardware is `Φ_out = 0`. Conservation (`Accounting.Guarantees.Remote.Conservation`) assumes no location.
* `ZCost` is PoUW's conjecture as the receipt binds it: `z_b` bits of a prior tile's noised output `z` cost `W_tile` at
  PoUW's pricing, wherever and for whatever weights they are computed, except with probability `δ`.

WorkHardness, the codec's cost of a partial completion, is lane ra-partial-lean's (`Accounting.Assumptions.WorkHardness`
on its branch); the route it gives here is `Run.Rebuild`.
-/

namespace Accounting.Remote.Assumptions

open Accounting.Remote

variable {τ : Type} {N : Node τ} {a : Audit} {Ω : Type}

/-- **W at the rebuild pricing**: on every outcome, the node's work at π_r outside PoUW's account and PoUW's account's
add within `Φ_r · T`. -/
def ComputeAdditive (r : Run N a Ω) : Prop := ∀ ω, r.rebuildWork ω + r.pouwWork ω ≤ N.rebuild * a.T

/-- **Identity, at `Φ_out`**: on every outcome, the outside compute's work at PoUW's pricing is at most `Φ_out · T`; no
undeclared tensor-class hardware is `Identity r 0`. -/
def Identity (r : Run N a Ω) (Φout : ℝ) : Prop := ∀ ω, (r.workOut ω : ℝ) ≤ Φout * a.T

/-- **PoUW's conjecture for the bound `z`**: outside the node, each `z_b` bits of bound `z` cost `W_tile` at PoUW's
pricing, except with probability `δ` when the audit accepts. -/
def ZCost [Fintype Ω] (r : Run N a Ω) (Wtile δ : ℝ) : Prop :=
  r.Holds δ fun ω => (r.zOut ω : ℝ) * Wtile ≤ a.zb * r.workOut ω

end Accounting.Remote.Assumptions
