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
def decodeLower (P : Equiv.Perm (Lab n wc)) (c2 k2 : Fin n → Lab n wc) : Fin n → Lab n wc :=
  transpose (fun v => labXor (P.symm (c2 v)) (k2 v))

/-! ### The model, and the two-layer invertible labels -/

/-- Key queries: a layer, a node, and a partial assignment of labels (its parents'). -/
abbrev KQ (n wc : ℕ) : Type := Fin 2 × Fin n × (Fin n → Option (Lab n wc))

/-- The primitive of the P3 model: an `H`-oracle and an ideal permutation of labels. -/
abbrev P3Omega (n wc : ℕ) : Type := (KQ n wc → Lab n wc) × Equiv.Perm (Lab n wc)

noncomputable instance instFintypeP3Omega (n wc : ℕ) : Fintype (P3Omega n wc) := Fintype.ofFinite _
    | .inr (.inl y) => ω.2 y
  termination_by v => v.val
  termination_by v => v.val
def topLabel (G : Labelling.TopoDAG n) (W : Fin n → Lab n wc) (ω : P3Omega n wc) (v : Fin n) : Lab n wc :=
  c2 G ω.1 ω.2 W v

/-! ### White/green hardness of the two-layer graph

The stacked graph on `V = Fin 2 × Fin n`: white edges are `G` inside a layer, green edges the complete bipartite
`layer 0 → layer 1`. The top layer is `layer 1`. Hardness is the white/green analogue of `Pous.Labelling.PebblingHard`,
/-- Vertices, and the trusted-graph data for the two-layer stack. -/
/-- White (intra-layer DRG) edges from `G`: `a → b` inside layer `i` when `a ∈ G.par b`. -/
/-- Green (connector) edges: complete bipartite from layer 0 to layer 1. -/
  ∀ C : ℕ → Finset (Fin 2 × Fin n), DRG.IsPebbling node2 (drg2 G) green2 C → (C 0).card ≤ s →
    (Finset.univ.filter fun v => node2 1 v ∈ C D).card < k

end Pous.Invertible
