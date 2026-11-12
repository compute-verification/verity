module

public import Definitions.Core.Formalism
public import Definitions.Nci.Covering
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Combinatorics.SimpleGraph.Clique

/-!
# The min-cut statement of the training slowdown: its vocabulary

A lower bound on the number of blocks any partition of a circuit needs, from the matrix multiplications it charges,
stated with cuts: what a block reads is bounded by how few gates separate it from the unregistered inputs, and no width
premise enters. The circuit is read as a DAG of wires: an edge runs from a gate to each gate that reads it, so a path
runs from the inputs towards the outputs. The input gates outside `θ` are the unregistered inputs
(`unregisteredInputs`).

* `κ(X → Y)` (`kappa`) is the least number of gates meeting every path from `X` to `Y`, its ends included
  (`Separates`). A word is a set of gates, and for a set `S` of words `κ(S)` (`kappaWords`) is `κ` from the unregistered
  inputs to the gates of `S`'s words.
* A **matmul** `y = wx` (`Matmul`) is only its communication pattern: words `w_il`, `x_lj` and `y_ij`, distinct indices
  giving distinct `w`- and `x`-words, and for each triple a **MAC** `g_ijl`, a connected set of computed gates
  (`Connected`) reading every wire of `w_il` and `x_lj`, one of whose gates has a path to a wire of `y_ij`. No
  arithmetic enters, and the summation order is free.
* The **charged work** (`Work`) is a family of matmuls whose MACs are pairwise disjoint: `N` MACs (`Work.N`), `O` output
  wires `∑ |y|` (`Work.O`), and `W_M`, `X_M` the `w`- and `x`-words of matmul `M`.
* `π(S) = ∑_M |S ∩ W_M| |S ∩ X_M|` (`Work.pi`) and `Π(k) = max {π(S) : κ(S) ≤ k}` (`piMax`). A set `A` of gates
  holds a MAC whole when every gate of the MAC is in `A` (`Work.inside`). For a set `A` of computed gates, `E(A)`
  (`Work.clear`) is the outputs fed by a MAC inside `A` with no wire in `A`,
  `b_y = min_{E(A) ≠ ∅} κ(A → E(A)) / |E(A)|` (`Work.bY`, `0` when every `E(A)` is empty, where `b_y = ∞` would give the
  same `α`), and `α = √(k Π(k) / b_y)` (`alpha`).

A block reads at most `k` gates outside `θ` when `|in(I) \ θ| ≤ k` (`ReadsAtMost`, through
`RegisteredCircuit.unregistered`); over a set of the blocks, with what all the blocks read as a total and what the set's
blocks read squared and summed, it is `ReadsOn`. The slowdown `ρ = m c / |C|` prices each of a partition's `m` blocks at
`c`, so it is read with blocks of at most `c` gates, none padded. A word is a set of gates, so a value and each of its
casts are distinct words; `val` names the value each word carries, the casts of one value carrying one, one-to-one on
each matmul's `w`-words and on its `x`-words (`ValuesInjective`), and `val = id` counts words. When no two matmuls share
both a `w`- and an `x`-value (`NoSharedPair`), or when the matmuls read shared arrays, matmul `a` the pair `e a`, an
edge of a graph with no clique of `ω + 1` (`Arrays`), at least `q` a pair of values (`CostsAtLeast`), two cut
conjectures bound the work: (H) and (E) (`Definitions.Assumptions`). The guarantees are
`Definitions.Nci.Guarantees.Cut`'s.
-/

@[expose] public section

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
def unregisteredInputs (d : RegisteredCircuit) : Finset (Fin d.D.N) :=
  univ.filter fun g => d.D.isInput g = true ∧ g ∉ d.θ

/-- `κ(S)` for a set `S` of words: `κ` from the unregistered inputs to the gates of `S`'s words. -/
noncomputable def kappaWords (d : RegisteredCircuit) (S : Finset (Finset (Fin d.D.N))) : ℕ :=
  kappa d.D (unregisteredInputs d) (S.biUnion id)

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

/-- The MACs a set `A` of gates holds whole: every gate of the MAC is in `A`. -/
def inside (A : Finset (Fin C.N)) : Finset W.Mac := univ.filter fun s => W.mac s ⊆ A

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
noncomputable def piMax {ι : Type} [Fintype ι] (d : RegisteredCircuit) (W : Work d.D ι) (k : ℕ) : ℕ :=
  ((univ : Finset (Finset (Finset (Fin d.D.N)))).filter fun S => kappaWords d S ≤ k).sup W.pi

/-- `α = √(k Π(k) / b_y)`. -/
noncomputable def alpha {ι : Type} [Fintype ι] [DecidableEq ι] (d : RegisteredCircuit) (W : Work d.D ι) (k : ℕ) : ℝ :=
  √(k * piMax d W k / W.bY)

/-- `α` under (H) and (E): `½ √(k³ / (q b))`, at `q = β_w β_x` and (E)'s floor `b`. -/
noncomputable def alphaHE (k : ℕ) (q b : ℝ) : ℝ := √(k ^ 3 / (q * b)) / 2

/-- The work bound under (H) and (E) at `m` blocks, each reading at most `k` gates outside `θ`:
`m (k + α) + (k/2) √(m O / q)` at `α = alphaHE k q b`, for work of `O` output wires at `q = β_w β_x`. -/
noncomputable def workHE (m k : ℕ) (q b O : ℝ) : ℝ := m * (k + alphaHE k q b) + k / 2 * √(m * O / q)

/-- The work bound under (H) and (E) over a set of blocks: `T + ½ √(S O / q) + ½ √(S T / (q b))`, for blocks that read
`T` gates outside `θ` between them, the squares of what each block of the set reads summing to `S`, and work of `O`
output wires at `q = β_w β_x`. At `T = m k` and `S = m k²` it is `workHE m k q b O`. -/
noncomputable def workHEOn (T S : ℕ) (q b O : ℝ) : ℝ := T + √(S * O / q) / 2 + √(S * T / (q * b)) / 2

/-- `F = √(2 (1 - 1/ω))`, what shared arrays with no `ω + 1` of them pairwise read together put on both terms of the
bound: `1` at `ω = 2`, `2/√3` at `ω = 3`. -/
noncomputable def shareF (ω : ℕ) : ℝ := √(2 * (1 - 1 / ω))

/-- Every block of `P` reads at most `k` gates outside `θ`: `|in(I) \ θ| ≤ k` (`RegisteredCircuit.unregistered`). -/
def ReadsAtMost {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) (k : ℕ) : Prop :=
  ∀ u, (d.unregistered (P.gates u)).card ≤ k

/-- What the blocks of `P` read outside `θ`, with `G` the blocks a count is taken over and the blocks outside `G`
charged whole: `k` limits the gates each block of `G` reads, `T` limits the gates all the blocks read between them,
those outside `G` too, and `S` limits the sum over `G` of the squares of what each of its blocks reads. -/
def ReadsOn {m : ℕ} (d : RegisteredCircuit) (P : Partition d.D m) (G : Finset (Fin m)) (k T S : ℕ) : Prop :=
  (∀ u ∈ G, (d.unregistered (P.gates u)).card ≤ k) ∧ ∑ u, (d.unregistered (P.gates u)).card ≤ T ∧
    ∑ u ∈ G, (d.unregistered (P.gates u)).card ^ 2 ≤ S

/-- `val` names the value each word carries, one-to-one on each matmul's `w`-words and on its `x`-words: the casts of
one value carry one, and a matmul's distinct words carry distinct values. -/
def ValuesInjective {C : Circuit} {ι V : Type} (W : Work C ι) (val : Finset (Fin C.N) → V) : Prop :=
  ∀ a, Set.InjOn val (W.wWords a) ∧ Set.InjOn val (W.xWords a)

/-- No two matmuls share both a `w`- and an `x`-value. -/
def NoSharedPair {C : Circuit} {ι V : Type} [DecidableEq V] (W : Work C ι) (val : Finset (Fin C.N) → V) : Prop :=
  ∀ a a', a ≠ a' → Disjoint ((W.wWords a).image val) ((W.wWords a').image val) ∨
    Disjoint ((W.xWords a).image val) ((W.xWords a').image val)

/-- The matmuls read shared arrays: matmul `a` reads arrays `(e a).1` (its `w`-words) and `(e a).2` (its `x`-words), an
edge of the graph `G` on the arrays, which has no clique of `ω + 1`, distinct matmuls on distinct edges. -/
def Arrays {C : Circuit} {ι ρ : Type} (W : Work C ι) (G : SimpleGraph ρ) (A : ρ → Finset (Finset (Fin C.N)))
    (e : ι → ρ × ρ) (ω : ℕ) : Prop :=
  G.CliqueFree (ω + 1) ∧ (∀ a, G.Adj (e a).1 (e a).2) ∧
    (∀ a a', s((e a).1, (e a).2) = s((e a').1, (e a').2) → a = a') ∧
    ∀ a, W.wWords a ⊆ A (e a).1 ∧ W.xWords a ⊆ A (e a).2

/-- Each matmul's pair of arrays costs at least `q` a pair of values: `q ≤ β_r β_s` for matmul `a` on `(r, s) = e a`. -/
def CostsAtLeast {ι ρ : Type} (e : ι → ρ × ρ) (β : ρ → ℝ) (q : ℝ) : Prop :=
  ∀ a, q ≤ β (e a).1 * β (e a).2

end Nci.Cut
