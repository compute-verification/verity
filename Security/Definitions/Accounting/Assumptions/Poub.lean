import Definitions.Accounting.Poub

/-!
# PoUB's named assumptions (trusted)

What PoUB's node-level bounds (`Accounting.SecurityProofs.Poub`) assume beyond the model's rows, each a
named `Prop` on a run:

* `Identity` is statement C's: no undeclared device of any kind. Every location that could read, hold or rebuild a
  word is a declared device of the node (a fleet audited at once is one node, `Node.fleet`), so outside the node
  nothing reads, holds or rebuilds one. Conservation (`Accounting.SecurityProofs.Poub.Conservation`) assumes no
  location.
* `ComputeAdditive` is W at the rebuild pricing: the node's work at π_r outside PoUW's account and PoUW's account's add
  within the Compute row's `Φ_r · T`. PoUW certifies `W_r` of the latter (`Run.Certified`), and the two give the
  rebuild's fit beside `W_r` (`Run.Fit`); at `W_r = 0` the fit needs only the Compute row.
* `OutsideWork` bounds the prover's work outside the node at PoUW's pricing by `Φ_out` per tick, for the statement
  without Identity, which allows memory-only helpers and charges the links.
* `ZCost` is PoUW's conjecture as the receipt binds it: `z_b` bits of a prior tile's noised output `z` cost `W_tile` at
  PoUW's pricing, wherever and for whatever weights they are computed, except with probability `δ`.

WorkHardness, the codec's cost of a partial completion, is `Accounting.Assumptions.WorkHardness`; the route it gives
here is `Run.Rebuild`.
-/

namespace Accounting.Poub.Assumptions

open Accounting.Poub

variable {τ : Type} {N : Node τ} {a : Audit} {Ω : Type}

/-- **Identity** (statement C: no undeclared device of any kind): on every outcome, no location outside the node reads a
tier, holds state or rebuilds a word's bits for the receipts. -/
def Identity (r : Run N a Ω) : Prop := ∀ ω, r.readOut ω = 0 ∧ r.stateOut ω = 0 ∧ r.rebuiltOut ω = 0

/-- **W at the rebuild pricing**: on every outcome, the node's work at π_r outside PoUW's account and PoUW's account's
add within `Φ_r · T`. -/
def ComputeAdditive (r : Run N a Ω) : Prop := ∀ ω, r.rebuildWork ω + r.pouwWork ω ≤ N.rebuild * a.T

/-- **Outside work, at `Φ_out`**: on every outcome, the outside compute's work at PoUW's pricing is at most
`Φ_out · T`. -/
def OutsideWork (r : Run N a Ω) (Φout : ℝ) : Prop := ∀ ω, (r.workOut ω : ℝ) ≤ Φout * a.T

/-- **PoUW's conjecture for the bound `z`**: outside the node, each `z_b` bits of bound `z` cost `W_tile` at PoUW's
pricing, except with probability `δ` when the audit accepts. -/
def ZCost [Fintype Ω] (r : Run N a Ω) (Wtile δ : ℝ) : Prop :=
  r.Holds δ fun ω => (r.zOut ω : ℝ) * Wtile ≤ a.zb * r.workOut ω

end Accounting.Poub.Assumptions
