import Definitions.Pous.Model.Labelling

/-! The proof-only part of `Definitions.Pous.Model.Labelling`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pous.Labelling

variable {N w : ℕ}

/-- Greedy parallel black pebbling (Pietrzak §6.1): the nodes carrying a pebble after `r` rounds from `P₀`, when
every node whose parents all carry pebbles receives one. -/
def reach (G : TopoDAG N) (P₀ : Finset (Fin N)) : ℕ → Finset (Fin N)
  | 0 => P₀
  | r + 1 => reach G P₀ r ∪ Finset.univ.filter fun v => G.par v ⊆ reach G P₀ r

/-- `Φ(G, V_C)` is `(s, D, k)`-hard: from any `s` initial pebbles, fewer than `k` challenge nodes carry a pebble
within `D` rounds. -/
def PebblingHard (G : TopoDAG N) (VC : Finset (Fin N)) (s D k : ℕ) : Prop :=
  ∀ P₀ : Finset (Fin N), P₀.card ≤ s → (VC.filter fun c => c ∈ reach G P₀ D).card < k

end Pous.Labelling
