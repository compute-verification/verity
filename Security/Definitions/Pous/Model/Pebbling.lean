import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.List.Chain

/-!
# Stacked depth-robust graphs, pebbles and the white/green game (B2, Fisch Claims 13–14)

The definitions of B2 (Fisch 2018/702 §2.4, §2.6 and App. B), proved in
`Security/Proofs/Pous/Pebbling/B2Stacking.lean`, and the white/green game of Claims 13–14, proved in
`Security/Proofs/Pous/Pebbling/FischClaims1314.lean`.
-/

namespace Pous.DRG

/-- Fisch §2.6: a DAG on `n` nodes is `(n, α, β)`-depth-robust if every subgraph on (at least) `αn` nodes contains a
path of at least `βn` nodes. -/
def DepthRobust (n : ℕ) (E : Fin n → Fin n → Prop) (α β : ℝ) : Prop :=
  ∀ S : Finset (Fin n), α * n ≤ S.card →
    ∃ p : List (Fin n), p.IsChain E ∧ p.Nodup ∧ (∀ v ∈ p, v ∈ S) ∧ β * n ≤ p.length

variable {ℓ n : ℕ} {V : Type}

/-- The edges of the stacked graph: the white (DRG) edges inside each layer and the green (connector) edges. -/
def StackedEdge (node : Fin ℓ → Fin n → V) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (u v : V) : Prop :=
  green u v ∨ ∃ i a b, drg i a b ∧ u = node i a ∧ v = node i b

/-- The connector from layer `i` to layer `j`, as the disjoint-path property of an `n`-superconcentrator with
sources `V_i` and sinks `V_j`: any `k` nodes `A` of `V_i` and `k` nodes `B` of `V_j` are joined by `k` pairwise
vertex-disjoint green paths, the one from `a ∈ A` running inside the level `V_i ∪ G_i` until it ends in `B`. -/
def DisjointPaths (node : Fin ℓ → Fin n → V) (level : V → Fin ℓ) (green : V → V → Prop) (i j : Fin ℓ) : Prop :=
  ∀ A B : Finset (Fin n), A.card = B.card →
    ∃ path : Fin n → List V, ∀ a ∈ A,
      (path a).head? = some (node i a) ∧
      (∃ b ∈ B, (path a).getLast? = some (node j b)) ∧
      (path a).IsChain green ∧
      (∀ v ∈ (path a).dropLast, level v = i) ∧
      ∀ a' ∈ A, a' ≠ a → ∀ v ∈ path a, v ∉ path a'

/-- A vertex carries no pebble. -/
def Unpebbled (black red : Finset V) (v : V) : Prop := v ∉ black ∧ v ∉ red

/-- Fisch App. B, the white/green rule: `v` may receive a pebble if all its parents carry pebbles, or if `v` has a
green child and every green child `u` of `v` carries a pebble, as do all white parents of `u`. -/
def Available (node : Fin ℓ → Fin n → V) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (C : Finset V) (v : V) : Prop :=
  (∀ u, StackedEdge node drg green u v → u ∈ C) ∨
    ((∃ u, green v u) ∧
      ∀ u, green v u → u ∈ C ∧ ∀ w, (∃ i a b, drg i a b ∧ w = node i a ∧ u = node i b) → w ∈ C)

/-- A parallel pebbling: round `r + 1` places pebbles only on vertices available in `C r`, and may remove any. -/
def IsPebbling (node : Fin ℓ → Fin n → V) (drg : Fin ℓ → Fin n → Fin n → Prop) (green : V → V → Prop)
    (C : ℕ → Finset V) : Prop :=
  ∀ r v, v ∈ C (r + 1) → v ∉ C r → Available node drg green (C r) v

end Pous.DRG
