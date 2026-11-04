import Definitions.Core.Partition
import Definitions.Nci.Covering
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Combinatorics.SimpleGraph.Clique

/-!
# The min-cut statement of the training slowdown

A lower bound on the number of blocks any partition of a circuit needs, from the matrix multiplications it charges,
stated with cuts: what a block reads is bounded by how few gates separate it from the unregistered inputs, and no width
premise enters. The circuit is read as a DAG of wires: an edge runs from a gate to each gate that reads it, so a path
runs from the inputs towards the outputs. The input gates outside `θ` are the unregistered inputs (`unregistered`).

* `κ(X → Y)` (`kappa`) is the least number of gates meeting every path from `X` to `Y`, its ends included
  (`Separates`). A word is a set of gates, and for a set `S` of words `κ(S)` (`kappaWords`) is `κ` from the unregistered
  inputs to the gates of `S`'s words.
* A **matmul** `y = wx` (`Matmul`) is only its communication pattern: words `w_il`, `x_lj` and `y_ij`, distinct indices
  giving distinct `w`- and `x`-words, and for each triple a **MAC** `g_ijl`, a connected set of computed gates
  (`Connected`) reading every wire of `w_il` and `x_lj`, one of whose gates has a path to a wire of `y_ij`. No
  arithmetic enters, and the summation order is free.
* The **charged work** (`Work`) is a family of matmuls whose MACs are pairwise disjoint: `N` MACs (`Work.N`), `O` output
  wires `∑ |y|` (`Work.O`), and `W_M`, `X_M` the `w`- and `x`-words of matmul `M`.
* `π(S) = ∑_M |S ∩ W_M| |S ∩ X_M|` (`Work.pi`) and `Π(k) = max {π(S) : κ(S) ≤ k}` (`piMax`). For a set `A` of computed
  gates, `E(A)` (`Work.clear`) is the outputs fed by a MAC inside `A` with no wire in `A`,
  `b_y = min_{E(A) ≠ ∅} κ(A → E(A)) / |E(A)|` (`Work.bY`, `0` when every `E(A)` is empty, where `b_y = ∞` would give the
  same `α`), and `α = √(k Π(k) / b_y)` (`alpha`).

`Bound` is the theorem: every partition into `m` blocks, each with at most `k` inputs outside `θ`, has
`N ≤ m (k + α) + √(Π(k) m O)`. `Slowdown` is its form for the slowdown `ρ = m c / |C|` of blocks of at most `c` gates;
nothing ties `c` to the blocks, and it holds for every `c`. `OnePremise` bounds `Π(k)` and `α` under (H), one premise
on cuts, and `SharedArrays` and `SharedArraysClique` bound `Π(k)` when the matmuls share arrays, each array at its own
cost a word: a layer's forward, `dX` and `dW` products, and in general. `HE` is the theorem of record: `Bound` and its
slowdown under (H) and (E), a floor on what a set of gates exports; `HEShared` is its form with shared arrays.

A word is a set of gates, so a value and each of its casts are distinct words, and (H) counts each. The statements
carry no hypothesis their proofs don't read: no fan-in bound, `W` and `X` may share words, and outputs may repeat. None
is listed in `lean-audit.json`; `Proofs.Nci.Cut` proves each.
-/

namespace Nci.Cut

open Verity Finset

/-- `Z` meets every path from `X` to `Y`, its ends included: no path from a gate of `X` to a gate of `Y` avoids `Z`. -/
def Separates (C : Circuit) (Z X Y : Finset (Fin C.N)) : Prop :=
  ∀ x ∈ X, ∀ y ∈ Y, ¬ Covering.FeedsWithin C (univ \ Z) x y

/-- `κ(X → Y)`: the least number of gates meeting every path from `X` to `Y`. -/
noncomputable def kappa (C : Circuit) (X Y : Finset (Fin C.N)) : ℕ :=
  sInf {n | ∃ Z : Finset (Fin C.N), Z.card = n ∧ Separates C Z X Y}

/-- `A` induces a connected subgraph: it is nonempty, and every nonempty proper part `T` of it has a wire inside `A`
between `T` and the rest of `A`. -/
def Connected (C : Circuit) (A : Finset (Fin C.N)) : Prop :=
  A.Nonempty ∧ ∀ T ⊆ A, T.Nonempty → T ≠ A → ∃ a ∈ A, ∃ h ∈ A, a ∈ C.reads h ∧ (a ∈ T ↔ h ∉ T)

/-- The unregistered inputs: the input gates outside `θ`. -/
def unregistered (d : Declaration) : Finset (Fin d.D.N) :=
  univ.filter fun g => d.D.isInput g = true ∧ g ∉ d.θ

/-- `κ(S)` for a set `S` of words: `κ` from the unregistered inputs to the gates of `S`'s words. -/
noncomputable def kappaWords (d : Declaration) (S : Finset (Finset (Fin d.D.N))) : ℕ :=
  kappa d.D (unregistered d) (S.biUnion id)

/-- A matmul `y = wx`, as its communication pattern: words `w_il`, `x_lj`, `y_ij`, and for each triple `(i, j, l)` a
MAC `g_ijl`. -/
structure Matmul (C : Circuit) where
  ni : ℕ
  nj : ℕ
  nl : ℕ
  w : Fin ni × Fin nl → Finset (Fin C.N)
  x : Fin nl × Fin nj → Finset (Fin C.N)
  y : Fin ni × Fin nj → Finset (Fin C.N)
  w_inj : Function.Injective w
  x_inj : Function.Injective x
  /-- The MAC `g_ijl` of the triple `(i, j, l)`. -/
  mac : Fin ni × Fin nj × Fin nl → Finset (Fin C.N)
  mac_computed : ∀ t : Fin ni × Fin nj × Fin nl, ∀ a ∈ mac t, C.isInput a = false
  mac_connected : ∀ t : Fin ni × Fin nj × Fin nl, Connected C (mac t)
  /-- Every wire of `w_il` is read by a gate of `g_ijl`. -/
  reads_w : ∀ t : Fin ni × Fin nj × Fin nl, ∀ a ∈ w (t.1, t.2.2), ∃ h ∈ mac t, a ∈ C.reads h
  /-- Every wire of `x_lj` is read by a gate of `g_ijl`. -/
  reads_x : ∀ t : Fin ni × Fin nj × Fin nl, ∀ a ∈ x (t.2.2, t.2.1), ∃ h ∈ mac t, a ∈ C.reads h
  /-- Some gate of `g_ijl` has a path to a wire of `y_ij`. -/
  feeds_y : ∀ t : Fin ni × Fin nj × Fin nl, ∃ h ∈ mac t, ∃ a ∈ y (t.1, t.2.1), C.Feeds h a

/-- The charged work: matmuls `M a`, for `a : ι`, whose MACs are pairwise disjoint. -/
structure Work (C : Circuit) (ι : Type) where
  M : ι → Matmul C
  mac_disjoint : ∀ s t : (Σ a, Fin (M a).ni × Fin (M a).nj × Fin (M a).nl), s ≠ t →
    Disjoint ((M s.1).mac s.2) ((M t.1).mac t.2)

namespace Work

variable {C : Circuit} {ι : Type} (W : Work C ι)

/-- The MACs: a matmul and a triple. -/
abbrev Mac := Σ a, Fin (W.M a).ni × Fin (W.M a).nj × Fin (W.M a).nl

/-- The outputs: a matmul and a pair `(i, j)`. -/
abbrev Out := Σ a, Fin (W.M a).ni × Fin (W.M a).nj

/-- A MAC's gates. -/
def mac (s : W.Mac) : Finset (Fin C.N) := (W.M s.1).mac s.2

/-- The output `y_ij` a MAC `g_ijl` feeds. -/
def out (s : W.Mac) : W.Out := ⟨s.1, (s.2.1, s.2.2.1)⟩

/-- An output's wires. -/
def y (o : W.Out) : Finset (Fin C.N) := (W.M o.1).y o.2

/-- `W_M`: matmul `a`'s `w`-words. -/
def wWords (a : ι) : Finset (Finset (Fin C.N)) := univ.image (W.M a).w

/-- `X_M`: matmul `a`'s `x`-words. -/
def xWords (a : ι) : Finset (Finset (Fin C.N)) := univ.image (W.M a).x

variable [Fintype ι]

/-- `N`: the number of MACs. -/
def N : ℕ := Fintype.card W.Mac

/-- `O = ∑ |y|`: the number of output wires, an output counted once for each of its pairs `(i, j)`. -/
def O : ℕ := ∑ o : W.Out, (W.y o).card

/-- `π(S) = ∑_M |S ∩ W_M| |S ∩ X_M|`. -/
def pi (S : Finset (Finset (Fin C.N))) : ℕ := ∑ a, (S ∩ W.wWords a).card * (S ∩ W.xWords a).card

/-- `W`: every matmul's `w`-words. -/
def allW : Finset (Finset (Fin C.N)) := univ.biUnion W.wWords

/-- `X`: every matmul's `x`-words. -/
def allX : Finset (Finset (Fin C.N)) := univ.biUnion W.xWords

variable [DecidableEq ι]

/-- `E(A)`: the outputs fed by a MAC inside `A` with no wire in `A`. -/
def clear (A : Finset (Fin C.N)) : Finset W.Out :=
  univ.filter fun o => (∃ s, W.out s = o ∧ W.mac s ⊆ A) ∧ Disjoint (W.y o) A

/-- The sets `A` of computed gates with `E(A) ≠ ∅`. -/
def fed : Finset (Finset (Fin C.N)) := univ.filter fun A => A ⊆ C.computed ∧ (W.clear A).Nonempty

/-- `b_y = min_{E(A) ≠ ∅} κ(A → E(A)) / |E(A)|`, over sets `A` of computed gates; `0` when every `E(A)` is empty. -/
noncomputable def bY : ℝ :=
  if h : W.fed.Nonempty then
    W.fed.inf' h fun A => (kappa C A ((W.clear A).biUnion W.y) : ℝ) / (W.clear A).card
  else 0

end Work

/-- `Π(k) = max {π(S) : κ(S) ≤ k}`, over sets `S` of words. -/
noncomputable def piMax {ι : Type} [Fintype ι] (d : Declaration) (W : Work d.D ι) (k : ℕ) : ℕ :=
  ((univ : Finset (Finset (Finset (Fin d.D.N)))).filter fun S => kappaWords d S ≤ k).sup W.pi

/-- `α = √(k Π(k) / b_y)`. -/
noncomputable def alpha {ι : Type} [Fintype ι] [DecidableEq ι] (d : Declaration) (W : Work d.D ι) (k : ℕ) : ℝ :=
  √(k * piMax d W k / W.bY)

/-- **The min-cut bound.** Every partition of the gates into `m` blocks with `|in(I) \ θ| ≤ k` for each block `I` has
`N ≤ m (k + α) + √(Π(k) m O)`. -/
def Bound : Prop :=
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι] (d : Declaration) (W : Work d.D ι) (k m : ℕ) (P : Partition d.D m),
    (∀ u, (d.D.into (P.gates u) \ d.θ).card ≤ k) →
      (W.N : ℝ) ≤ m * (k + alpha d W k) + √(piMax d W k * m * W.O)

/-- **The slowdown.** With blocks of at most `c` gates, `ρ = m c / |C|` is at least
`(c / |C|) (N / (k + α) - √(Π(k) O N) / (k + α)^{3/2})`; the inequality holds for every `c`. -/
def Slowdown : Prop :=
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι] (d : Declaration) (W : Work d.D ι) (k m c : ℕ) (P : Partition d.D m),
    (∀ u, (d.D.into (P.gates u) \ d.θ).card ≤ k) →
      (c / d.D.size d.D.computed : ℝ) *
          (W.N / (k + alpha d W k) - √(piMax d W k * W.O * W.N) / (k + alpha d W k) ^ (3 / 2 : ℝ)) ≤
        m * c / d.D.size d.D.computed

/-- **The one premise.** If no two matmuls share both a `w`- and an `x`-word, and every `S` with `κ(S) ≤ k` has
`β_w |S ∩ W| + β_x |S ∩ X| ≤ κ(S)`, then `Π(k) ≤ k² / (4 β_w β_x)` and `α ≤ ½ √(k³ / (β_w β_x b_y))`. -/
def OnePremise : Prop :=
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι] (d : Declaration) (W : Work d.D ι) (k : ℕ) (βw βx : ℝ), 0 < βw → 0 < βx →
    (∀ a b, a ≠ b → Disjoint (W.wWords a) (W.wWords b) ∨ Disjoint (W.xWords a) (W.xWords b)) →
    (∀ S, kappaWords d S ≤ k → βw * (S ∩ W.allW).card + βx * (S ∩ W.allX).card ≤ kappaWords d S) →
      (piMax d W k : ℝ) ≤ k ^ 2 / (4 * βw * βx) ∧ alpha d W k ≤ √(k ^ 3 / (βw * βx * W.bY)) / 2

/-- **Shared arrays, one layer.** The forward product `Y = W X` (matmul `0`), `dX = Wᵀ dY` (matmul `1`) and
`dW = dY Xᵀ` (matmul `2`) read three arrays `A 0`, `A 1`, `A 2` (the weights, the activations, the output gradients),
array `r` at cost `β r` a word. If every `S` with `κ(S) ≤ k` has `∑_r β_r |S ∩ A r| ≤ κ(S)`, then `Π(k)` is at most
`4/3` times `k² / (4 q)`, with `q` the least `β_r β_s` over the pairs of arrays a matmul reads. -/
def SharedArrays : Prop :=
  ∀ (d : Declaration) (W : Work d.D (Fin 3)) (k : ℕ) (A : Fin 3 → Finset (Finset (Fin d.D.N))) (β : Fin 3 → ℝ),
    (∀ r, 0 < β r) →
    W.wWords 0 ⊆ A 0 → W.xWords 0 ⊆ A 1 → W.wWords 1 ⊆ A 0 → W.xWords 1 ⊆ A 2 → W.wWords 2 ⊆ A 2 →
    W.xWords 2 ⊆ A 1 →
    (∀ S, kappaWords d S ≤ k → β 0 * (S ∩ A 0).card + β 1 * (S ∩ A 1).card + β 2 * (S ∩ A 2).card ≤ kappaWords d S) →
      (piMax d W k : ℝ) ≤ 4 / 3 * (k ^ 2 / (4 * min (β 0 * β 1) (min (β 0 * β 2) (β 2 * β 1))))

/-- **Shared arrays, in general.** Matmul `a` reads arrays `(e a).1` (its `w`-words) and `(e a).2` (its `x`-words),
an edge of a graph `G` on the arrays with no clique of `ω + 1`, distinct matmuls on distinct edges; array `r` costs
`β r` a word, and `q ≤ β_r β_s` for each matmul's pair of arrays `(r, s)` (at best, the least such product). If every
`S` with `κ(S) ≤ k` has `∑_r β_r |S ∩ A r| ≤ κ(S)`, then `Π(k)` is at most `2 (1 - 1/ω)` times `k² / (4 q)`. -/
def SharedArraysClique : Prop :=
  ∀ (ι ρ : Type) [Fintype ι] [DecidableEq ι] [Fintype ρ] (d : Declaration) (W : Work d.D ι) (k ω : ℕ)
    (G : SimpleGraph ρ) (A : ρ → Finset (Finset (Fin d.D.N))) (e : ι → ρ × ρ) (β : ρ → ℝ) (q : ℝ), 0 < ω →
    G.CliqueFree (ω + 1) → (∀ a, G.Adj (e a).1 (e a).2) → (∀ a b, s((e a).1, (e a).2) = s((e b).1, (e b).2) → a = b) →
    (∀ a, W.wWords a ⊆ A (e a).1 ∧ W.xWords a ⊆ A (e a).2) →
    (∀ r, 0 ≤ β r) → 0 < q → (∀ a, q ≤ β (e a).1 * β (e a).2) →
    (∀ S, kappaWords d S ≤ k → ∑ r, β r * (S ∩ A r).card ≤ kappaWords d S) →
      (piMax d W k : ℝ) ≤ 2 * (1 - 1 / ω) * (k ^ 2 / (4 * q))

/-- The theorem of record's `α`: `½ √(k³ / (q b))`, at `q = β_w β_x` and (E)'s floor `b`. -/
noncomputable def alphaHE (k : ℕ) (q b : ℝ) : ℝ := √(k ^ 3 / (q * b)) / 2

/-- **The theorem of record.** (H): no two matmuls share both a `w`- and an `x`-word, and every `S` with `κ(S) ≤ k`
has `β_w |S ∩ W| + β_x |S ∩ X| ≤ κ(S)`. (E) at `b > 0`: every set `B` of computed gates has `κ(B → E(B)) ≥ b |E(B)|`,
as at `b = b_y` (`Work.bY`). Then every partition into `m` blocks with `|in(I) \ θ| ≤ k` has
`N ≤ m (k + α) + (k/2) √(m O / (β_w β_x))`, with `α = ½ √(k³ / (β_w β_x b))` (`alphaHE`), and for every `c`,
`m c / |C| ≥ (c / |C|) (N / (k + α) - (k/2) √(O N / (β_w β_x)) / (k + α)^{3/2})`. -/
def HE : Prop :=
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι] (d : Declaration) (W : Work d.D ι) (k m c : ℕ) (P : Partition d.D m)
    (βw βx b : ℝ), 0 < βw → 0 < βx → 0 < b →
    (∀ a a', a ≠ a' → Disjoint (W.wWords a) (W.wWords a') ∨ Disjoint (W.xWords a) (W.xWords a')) →
    (∀ S, kappaWords d S ≤ k → βw * (S ∩ W.allW).card + βx * (S ∩ W.allX).card ≤ kappaWords d S) →
    (∀ B ⊆ d.D.computed, b * (W.clear B).card ≤ kappa d.D B ((W.clear B).biUnion W.y)) →
    (∀ u, (d.D.into (P.gates u) \ d.θ).card ≤ k) →
      (W.N : ℝ) ≤ m * (k + alphaHE k (βw * βx) b) + k / 2 * √(m * W.O / (βw * βx)) ∧
      (c / d.D.size d.D.computed : ℝ) * (W.N / (k + alphaHE k (βw * βx) b) -
          k / 2 * √(W.O * W.N / (βw * βx)) / (k + alphaHE k (βw * βx) b) ^ (3 / 2 : ℝ)) ≤
        m * c / d.D.size d.D.computed

/-- **The theorem of record with shared arrays.** The arrays and costs of `SharedArraysClique` at `ω = 3`: no four
arrays pairwise read together by a matmul, as with every layer's forward, `dX` and `dW` products on its weights,
activations and output gradients, however many layers share an array. With (E) at `b`, both terms of `HE` grow by
`F = 2/√3`, with `q` for `β_w β_x`: `N ≤ m (k + F α) + F (k/2) √(m O / q)` with `α = ½ √(k³ / (q b))`, and the
slowdown likewise. -/
def HEShared : Prop :=
  ∀ (ι ρ : Type) [Fintype ι] [DecidableEq ι] [Fintype ρ] (d : Declaration) (W : Work d.D ι) (k m c : ℕ)
    (P : Partition d.D m) (G : SimpleGraph ρ) (A : ρ → Finset (Finset (Fin d.D.N))) (e : ι → ρ × ρ) (β : ρ → ℝ)
    (q b : ℝ), G.CliqueFree 4 → (∀ a, G.Adj (e a).1 (e a).2) →
    (∀ a a', s((e a).1, (e a).2) = s((e a').1, (e a').2) → a = a') →
    (∀ a, W.wWords a ⊆ A (e a).1 ∧ W.xWords a ⊆ A (e a).2) →
    (∀ r, 0 ≤ β r) → 0 < q → (∀ a, q ≤ β (e a).1 * β (e a).2) → 0 < b →
    (∀ S, kappaWords d S ≤ k → ∑ r, β r * (S ∩ A r).card ≤ kappaWords d S) →
    (∀ B ⊆ d.D.computed, b * (W.clear B).card ≤ kappa d.D B ((W.clear B).biUnion W.y)) →
    (∀ u, (d.D.into (P.gates u) \ d.θ).card ≤ k) →
      (W.N : ℝ) ≤ m * (k + 2 / √3 * alphaHE k q b) + 2 / √3 * (k / 2 * √(m * W.O / q)) ∧
      (c / d.D.size d.D.computed : ℝ) * (W.N / (k + 2 / √3 * alphaHE k q b) -
          2 / √3 * (k / 2 * √(W.O * W.N / q)) / (k + 2 / √3 * alphaHE k q b) ^ (3 / 2 : ℝ)) ≤
        m * c / d.D.size d.D.computed

end Nci.Cut
