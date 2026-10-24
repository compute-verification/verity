import Mathlib.Data.Finite.Prod
import Mathlib.Data.Fintype.Perm
import Definitions.Pous.Model.Labelling
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.List.Chain

/-!
# P3's invertible two-layer labels, their ideal model, and white/green hardness (N10)

The definitions of the N10 draft (`lean/submissions/pebbling/InvertibleExPostFactoDraft.lean`), verbatim; its header
lists the modelling choices. The model `p3Model` has an `H`-oracle and one ideal permutation `P` of labels, with `P`
and `P⁻¹` both queryable; the labels are `c¹ = P(W ⊕ H(0, ·))` and `c² = P(transpose(c¹) ⊕ H(1, ·))`.
-/

namespace Pous.Invertible

variable {n wc : ℕ}

/-- A label: `n` components of `wc` bits (width `m = n·wc`). -/
abbrev Lab (n wc : ℕ) : Type := Fin n → Bits wc

/-- Component-wise XOR of labels. -/
def labXor (a b : Lab n wc) : Lab n wc := fun v => Labelling.bxor (a v) (b v)

/-- The `K_{n,n}` / transpose connector: `transpose C v` collects component `v` of every label, so
`transpose C v u = C u v`. -/
def transpose (C : Fin n → Lab n wc) : Fin n → Lab n wc := fun v u => C u v

/-! ### The model, and the two-layer invertible labels -/

/-- Key queries: a layer, a node, and a partial assignment of labels (its parents'). -/
abbrev KQ (n wc : ℕ) : Type := Fin 2 × Fin n × (Fin n → Option (Lab n wc))

/-- The primitive of the P3 model: an `H`-oracle and an ideal permutation of labels. -/
abbrev P3Omega (n wc : ℕ) : Type := (KQ n wc → Lab n wc) × Equiv.Perm (Lab n wc)

noncomputable instance instFintypeP3Omega (n wc : ℕ) : Fintype (P3Omega n wc) := Fintype.ofFinite _
instance instNonemptyP3Omega (n wc : ℕ) : Nonempty (P3Omega n wc) := ⟨⟨fun _ _ _ => false, 1⟩⟩

/-! ### White/green hardness of the two-layer graph

The stacked graph on `V = Fin 2 × Fin n`: white edges are `G` inside a layer, green edges the complete bipartite
`layer 0 → layer 1`. The top layer is `layer 1`. Hardness is the white/green analogue of `Pous.Labelling.PebblingHard`,
using the trusted game `Pous.DRG.IsPebbling`; with a trusted encoder there are no red pebbles. -/

end Pous.Invertible
