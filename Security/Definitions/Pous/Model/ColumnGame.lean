import Mathlib.Data.Fintype.Prod
import Definitions.Pous.Model.Labelling

/-!
# The two-directional column game over P3's two-layer labels

The game in which the band certificate (`Pous.SecurityProofs.P3.Column`) is stated. It is the column game of the N10′
draft (`lean/submissions/pebbling/ColumnAwareDraft.lean`): its holdings, one round, reachability, `ColumnHard` and
base-path graphs, with the definitions verbatim and the docstrings shortened. The draft's model, bound and N10′
statement are not part of the trusted layer yet.

N10 (`Pous.Guarantees.InvExPostFactoTwoLayer`) is false because top label `v` reads only column `v` of the lower layer,
so the adversary can hold a lower label one column at a time. Here a holding has whole lower labels (cost `n`),
lower columns `(w, u)` (cost `1`) and whole top labels (cost `n`), and an inverse query is a move of the game, not a
charge in the bound.
-/

namespace Pous.ColumnGame

variable {n : ℕ}

/-- A column-game holding: whole lower labels, lower columns `(w, u)`, whole top labels. -/
structure ColState (n : ℕ) where
  lowerWhole : Finset (Fin n)
  lowerCol : Finset (Fin n × Fin n)
  topWhole : Finset (Fin n)

/-- Size in **column units**: a whole label costs `n`, a single column costs `1`. -/
def ColState.size {n : ℕ} (c : ColState n) : ℕ :=
  n * c.lowerWhole.card + c.lowerCol.card + n * c.topWhole.card

/-- One round of the **two-directional** column game:
* **forward-lower / promotion:** a lower label becomes whole if its lower-parents are whole, or all its columns are held;
* **reverse (key-gated):** a held (whole) top `u` *whose top-parents are also whole* yields column `u` of *every* lower
  label. The gate `G.par u ⊆ topWhole` is faithful to `Pous.Invertible.c2`/`decodeLower`: recovering `x²_u = P⁻¹(c²_u) ⊕
  k²_u` needs the key `k²_u = H(1, u, {c²_w : w ∈ par u})`, which hashes the **full top labels** of `u`'s parents.
* **forward-top:** a top becomes whole if every column `(w,u)` is available (`w` whole or held) and its top-parents are
  whole. -/
def colRound (G : Labelling.TopoDAG n) (c : ColState n) : ColState n where
  lowerWhole := c.lowerWhole ∪ Finset.univ.filter (fun w => G.par w ⊆ c.lowerWhole ∨ ∀ u, (w, u) ∈ c.lowerCol)
  lowerCol := c.lowerCol ∪ (Finset.univ ×ˢ c.topWhole.filter (fun u => G.par u ⊆ c.topWhole))
  topWhole := c.topWhole ∪ Finset.univ.filter
    (fun u => (∀ w, w ∈ c.lowerWhole ∨ (w, u) ∈ c.lowerCol) ∧ G.par u ⊆ c.topWhole)

/-- The holding reachable from `c0` after `r` rounds. -/
def colReach (G : Labelling.TopoDAG n) (c0 : ColState n) : ℕ → ColState n
  | 0 => c0
  | r + 1 => colRound G (colReach G c0 r)

/-- **`ColumnHard G D k X`**: every holding that recovers `k` top labels within `D` two-directional rounds (forward +
reverse) stores at least `(k − X)·n` column units, i.e. saves at most `X` labels. -/
def ColumnHard (G : Labelling.TopoDAG n) (D k X : ℕ) : Prop :=
  ∀ c0 : ColState n, k ≤ (colReach G c0 D).topWhole.card → (k - X) * n ≤ c0.size

/-- A base-path DRG: node `v` has its spine parent `v − 1` plus given extra parents `extra v ⊆ {0,…,v−1}`. -/
def basePath (n : ℕ) (extra : Fin n → Finset (Fin n)) (hextra : ∀ v, ∀ u ∈ extra v, u < v) : Labelling.TopoDAG n where
  par v := Finset.univ.filter (fun u => (u : ℕ) + 1 = (v : ℕ)) ∪ extra v
  par_lt v u hu := by
    simp only [Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and] at hu
    rcases hu with hu | hu
    · show (u : ℕ) < (v : ℕ); omega
    · exact hextra v u hu

end Pous.ColumnGame
