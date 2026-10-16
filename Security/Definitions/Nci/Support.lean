import Definitions.Nci
import Mathlib.Data.FinEnum

/-!
# NCI's incompressibility in support form: the vocabulary

The draft's incompressibility charges an IU `b_w` bits for each weight entry and `b_a` for each input entry of which it
holds a product. Stated over wiring alone, that is false: a declaration can hold a weight in a constant gate, or rebuild
it from a narrower wire, and its product gates then read less than they are charged. What an IU must read is what its
values determine. Every value an IU `R` holds is a function of the values on `in(R)`, so over the assignments that keep
the baseline at its committed values, the values of any gates of `R` take at most `2^novelReads(R)` distinct values,
whatever `R`'s gates compute. The support form charges a set of gates by how many distinct values it takes over those
assignments, which wiring cannot change.

* `admissible d θv`: the assignments to `d`'s gates that give the baseline `θ` its committed values `θv`, every novel
  input free;
* `values T a`: the values of the gates in `T` on the evaluation of `a`;
* `support d θv T`: the number of distinct `values T` over the admissible assignments;
* `Chain.charge R b_w b_a T`: what the draft charges for the products in `T`, `b_w` per weight entry and `b_a` per
  input entry of which `T` holds a product, other than an input entry linked to an entry whole in `R`.
-/

namespace Nci

open Verity Finset

open Classical in
/-- The assignments to `d`'s gates that give each baseline gate its committed value in `θv`; the novel inputs range
freely. Evaluation reads only the input gates, so these are the runs `d` can have once `θ` is committed. -/
noncomputable def admissible (d : Declaration) (θv : d.D.Transcript) : Finset d.D.Transcript :=
  univ.filter fun a => ∀ g ∈ d.θ, a g = θv g

/-- The values of the gates in `T` on the evaluation of `a`. -/
def values {D : Circuit} (T : Finset (Fin D.N)) (a : D.Transcript) : (g : T) → BitVec (D.w g) :=
  fun g => D.eval a g

open Classical in
/-- **The support of `T`**: the number of distinct values the gates in `T` take over the runs of `d` with the baseline
at `θv`. -/
noncomputable def support (d : Declaration) (θv : d.D.Transcript) (T : Finset (Fin d.D.N)) : ℕ :=
  ((admissible d θv).image (values T)).card

namespace Chain

variable {D : Circuit} (c : Chain D)

open Classical in
/-- **The draft's charge for the products in `T`**, relative to an IU `R`: `b_w` for each weight entry and `b_a` for
each input entry of which `T` holds a product, other than an input entry linked to an entry whole in `R`, which `R` can
compute from its own products. At `T = R` it is the left side of the draft's incompressibility. -/
noncomputable def charge (R : Finset (Fin D.N)) (b_w b_a : ℝ) (T : Finset (Fin D.N)) : ℝ :=
  b_w * (c.weightsUsed T).card + b_a * ((c.inputsUsed T).filter fun x => ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).card

end Chain

end Nci
