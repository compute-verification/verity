import Definitions.Pous.Model.Invertible
import Proofs.Pous.Model.Pebbling

/-! The proof-only part of `Definitions.Pous.Model.Invertible`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pous.Invertible

variable {n wc : ℕ}

/-- Recover the lower-layer labels from the top labels `c²` and top keys `k²`:
`lower v u = component u of (P⁻¹(c² v) ⊕ k² v)`, so `lower` as a family is `transpose` of the decoded inputs. -/
def decodeLower (P : Equiv.Perm (Lab n wc)) (c2 k2 : Fin n → Lab n wc) : Fin n → Lab n wc :=
  transpose (fun v => labXor (P.symm (c2 v)) (k2 v))

/-- The P3 ideal model: a random oracle `H : KQ → Lab` and an ideal permutation `P` of labels, with `P` and `P⁻¹`
both queryable (`Qry := KQ ⊕ (Lab ⊕ Lab)` is `H | P | P⁻¹`). -/
noncomputable def p3Model (n wc : ℕ) : IdealModel where
  Ω := P3Omega n wc
  Qry := KQ n wc ⊕ (Lab n wc ⊕ Lab n wc)
  Ans := Lab n wc
  answer ω q := match q with
    | .inl x => ω.1 x
    | .inr (.inl y) => ω.2 y
    | .inr (.inr z) => ω.2.symm z

/-- Layer-1 labels `c¹_v = P(W_v ⊕ H(0, v, parents))`, over the intra-layer DRG `G`. -/
def c1 (G : Labelling.TopoDAG n) (Hf : KQ n wc → Lab n wc) (P : Equiv.Perm (Lab n wc)) (W : Fin n → Lab n wc) :
    Fin n → Lab n wc
  | v => P (labXor (W v) (Hf (0, v, fun u => if _h : u ∈ G.par v then some (c1 G Hf P W u) else none)))
  termination_by v => v.val
  decreasing_by exact G.par_lt v u _h

/-- Layer-2 (top) labels `c²_v = P(x²_v ⊕ H(1, v, parents))`, where `x²_v = transpose(c¹)_v`. -/
def c2 (G : Labelling.TopoDAG n) (Hf : KQ n wc → Lab n wc) (P : Equiv.Perm (Lab n wc)) (W : Fin n → Lab n wc) :
    Fin n → Lab n wc
  | v => P (labXor (transpose (c1 G Hf P W) v)
      (Hf (1, v, fun u => if _h : u ∈ G.par v then some (c2 G Hf P W u) else none)))
  termination_by v => v.val
  decreasing_by exact G.par_lt v u _h

/-- The top label of node `v`, as a function of the drawn primitive `ω = (H, P)`. -/
def topLabel (G : Labelling.TopoDAG n) (W : Fin n → Lab n wc) (ω : P3Omega n wc) (v : Fin n) : Lab n wc :=
  c2 G ω.1 ω.2 W v

/-- Vertices, and the trusted-graph data for the two-layer stack. -/
def node2 : Fin 2 → Fin n → Fin 2 × Fin n := fun i a => (i, a)

/-- White (intra-layer DRG) edges from `G`: `a → b` inside layer `i` when `a ∈ G.par b`. -/
def drg2 (G : Labelling.TopoDAG n) : Fin 2 → Fin n → Fin n → Prop := fun _ a b => a ∈ G.par b

/-- Green (connector) edges: complete bipartite from layer 0 to layer 1. -/
def green2 : (Fin 2 × Fin n) → (Fin 2 × Fin n) → Prop := fun u w => u.1 = 0 ∧ w.1 = 1

/-- White/green hardness: from any initial configuration `C` of the trusted game with at most `s` pebbles, fewer
than `k` top nodes (`layer 1`) carry a pebble within `D` rounds. -/
def WhiteGreenHard (G : Labelling.TopoDAG n) (s D k : ℕ) : Prop :=
  ∀ C : ℕ → Finset (Fin 2 × Fin n), DRG.IsPebbling node2 (drg2 G) green2 C → (C 0).card ≤ s →
    (Finset.univ.filter fun v => node2 1 v ∈ C D).card < k

end Pous.Invertible
