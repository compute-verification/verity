import Properties.Flock.Level3.Seam
import Flock.Piop

/-!
The executable's univariate-skip weights in `GF128`: `lagS`, `lagΛ` (`lagrange` on the skip nodes and on the output
coset) and `combW` (`combWeights`). That they interpolate (`Arith.Correct.nodes`) is `Proofs.Flock.Level3.Nodes`.
-/

open Polynomial Flock

set_option maxRecDepth 20000

namespace FlockLevel3

/-! ### `S` and `Λ` -/

/-- `L_s(ζ)`, the executable's skip weights. -/
noncomputable def lagS (ζ : GF128) (s : Fin 64) : GF128 := GF128.of ((skipWeights ζ.val)[s.val]!)

/-- `L^Λ_i(ζ)`, the executable's output-coset weights (the ones `Pc` uses). -/
noncomputable def lagΛ (ζ : GF128) (i : Fin 64) : GF128 := GF128.of ((lagrange nodesL den6Inv ζ.val)[i.val]!)

/-! ### `Pcomb`'s weights -/

/-- The weights of `Pcomb(ζ)`, the executable's `combWeights`. -/
noncomputable def combW (ζ : GF128) (i : Fin 64) : GF128 := GF128.of ((combWeights ζ.val)[i.val]!)

end FlockLevel3
