import Definitions.Nci.Hardness

/-!
# Priced reads: the one hardness schema

`Priced` is the one named premise the hardness arguments instantiate (Daniel, 26 Oct): the operations a block computes
whole cost it at least their operands' price in bits it is handed. Each argument picks its counted operations, the
values it prices and the price (`Priced`'s instances). It reads no decoder: the local-description-length route is one
way to establish it (`Guarantees.PricedOfActivationRate`).
-/

namespace Nci.Hardness.Assumptions

open Verity

/-- NEW PREMISE. **Priced reads** ([PRICED]) at price `π`, deficit `Λ` (Daniel's `λ`; Lean reserves `λ`) and block
size `c`, with free side information `A`: every set `S` of at most `c` operation gates is handed, at inputs outside `θ`
and outside `A S`, at least the price of the operand values of the counted operations it computes whole, less `Λ`:
`π(Ops(S)) − Λ ≤ |in(S) \ (θ ∪ A S)|`, where `π(X) = Σ_{v ∈ X} π v` and `|·|` counts bits.

The price is additive over values, and `Ops(S)` holds each value once, so an updated parameter that many products in `S`
use is priced once (reuse is free), and an activation occurrence, each its own value, is priced per occurrence. `A` is
side information that `S` gets for free, `fun _ => ∅` when there is none. A set holds operation gates only, so an input
gate it uses is one it reads.

Its instances, each a choice of counted operations, the values priced and the price:

* **cover**: products, pricing each distinct updated parameter at `p` (`Nci.Cover.Assumptions.UpdateHardness` at
  `b = p`, `Λ = 0`, with `in(S)` counted as the cover theorem counts it);
* **fan-out**: products, pricing each activation occurrence at `β` (`Operations.ofActs`;
  `Guarantees.FanOutBoundPriced`);
* **partial sums**: accumulation steps, pricing each partial sum that enters `S`, an operand computed outside it;
* **recovery** (§5.3): output occurrences, pricing the coordinates of the update that their outputs determine, with
  `A S` the other inputs of `S` (the activations they multiply), which are free.

Rationale: a set that computes an operation whole holds its operands. One it does not compute it is handed, or computes
from what it is handed, and the price is what each operand costs in handed bits at block size `c`. It fails where the
price overstates that: a registered value, an operand a short code describes (a low-rank update, a run that has not
drifted from `θ`), or several operands computed from one. -/
def Priced (d : Declaration) {ι V : Type*} [DecidableEq V] (O : Operations d.D ι V) (π : V → ℝ) (Λ : ℝ) (c : ℕ)
    (A : Finset (Fin d.D.N) → Finset (Fin d.D.N)) : Prop :=
  ∀ S ⊆ d.D.computed, d.D.size S ≤ c → ∑ v ∈ O.ops S, π v - Λ ≤ d.D.width (d.D.into S \ (d.θ ∪ A S))

end Nci.Hardness.Assumptions
