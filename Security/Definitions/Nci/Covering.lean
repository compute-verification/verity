import Definitions.Core.Declaration

/-!
# Coverings by inference blocks, and multiplication blocks

The vocabulary of the covering bound, with one width `b` and one compression rate `r` for weights
and activations alike. `C` is a declaration's circuit, read as a finite DAG: its
sources `in(C)` are the gates with no predecessors (`sources`), and `θ ⊆ in(C)` are its registered parameters (the
baseline). The statement counts gates; in the Boolean profile (`w = 1`) a gate is a bit.

* A **subcircuit** `S` (`Subcircuit`) is a set of gates with inputs `in(S) ⊆ S`, every gate of `S \ in(S)` having all its
  predecessors in `S`: `S` computes `S \ in(S)` (`inner`) from `in(S)`. A gate of `in(C)` in `S` is one of its inputs,
  since a gate with no predecessors is given, not computed: otherwise a block could hold a prompt's bits, or an
  unregistered weight, as gates it computes and read them for nothing. An IU `R` of a division is the subcircuit
  `R ∪ into R` with inputs `into R` and the input gates of `C` in `R`.
* An **`(s, k)`-inference block** (`InferenceBlock`) is a subcircuit with `|S| ≤ s` and `|in(S) \ θ| ≤ k`. An
  **`(s, k)`-covering** (`Covers`) is a family of them that together compute `C \ in(C)`: every gate with a predecessor
  is computed by some block. Blocks may overlap and recompute, so a covering need not be a `Partition`. Its slowdown is
  `ρ = s · #blocks / |C|`.
* **Multiplication blocks** (`MulBlocks`): for `(i, j) ∈ P ⊆ θ' × J` (weights `i` of the updated parameters `θ'`, tokens
  `j`), pairwise disjoint `m_{i,j} ⊆ C \ in(C)`, each with an output gate `o_{i,j} ∈ m_{i,j}` that every gate of
  `m_{i,j}` reaches inside `m_{i,j}` (so `m_{i,j}` is nonempty), and pairwise disjoint token computations `T_j ⊆ C \ θ`.
  `S` **completes** `m` when `m ⊆ S \ in(S)`, and `θ'_S` (`weightsCompleted`) is the set of weights `i` for which `S`
  completes some `m_{i,j}`.
-/

namespace Nci.Covering

open Verity Finset

/-- `in(C)`: the gates with no predecessors. A registered parameter is one of them (`Declaration.θ_input`). -/
def sources (C : Circuit) : Finset (Fin C.N) := univ.filter fun g => C.reads g = ∅

/-- `g` reaches `t` inside `R`: a path of wires from `g` to `t` all of whose gates, its ends included, lie in `R`. -/
inductive FeedsWithin (C : Circuit) (R : Finset (Fin C.N)) : Fin C.N → Fin C.N → Prop
  | refl {t : Fin C.N} : t ∈ R → FeedsWithin C R t t
  | step {g h t : Fin C.N} : g ∈ R → g ∈ C.reads h → FeedsWithin C R h t → FeedsWithin C R g t

/-- A subcircuit `S` of `C`: its gates, and its inputs `in(S) ⊆ S`; every gate of `S \ in(S)` has all its predecessors
in `S`, and every gate of `in(C)` in `S` is an input of `S`. -/
structure Subcircuit (C : Circuit) where
  gates : Finset (Fin C.N)
  ins : Finset (Fin C.N)
  ins_sub : ins ⊆ gates
  closed : ∀ g ∈ gates, g ∉ ins → C.reads g ⊆ gates
  sources_ins : ∀ g ∈ gates, g ∈ sources C → g ∈ ins

/-- `S \ in(S)`: the gates `S` computes. -/
def Subcircuit.inner {C : Circuit} (S : Subcircuit C) : Finset (Fin C.N) := S.gates \ S.ins

/-- An **`(s, k)`-inference block** of `d`: a subcircuit of at most `s` gates with at most `k` inputs outside the
registered parameters, `|S| ≤ s` and `|in(S) \ θ| ≤ k`. -/
def InferenceBlock (d : Declaration) (s k : ℕ) (S : Subcircuit d.D) : Prop :=
  S.gates.card ≤ s ∧ (S.ins \ d.θ).card ≤ k

/-- An **`(s, k)`-covering** of `d` by `n` blocks: each is an `(s, k)`-inference block, and together they compute
`C \ in(C)`, every gate with a predecessor lying in `S \ in(S)` for some block `S`. -/
def Covers (d : Declaration) (s k : ℕ) {n : ℕ} (I : Fin n → Subcircuit d.D) : Prop :=
  (∀ u, InferenceBlock d s k (I u)) ∧ ∀ g, g ∉ sources d.D → ∃ u, g ∈ (I u).inner

/-- **Multiplication blocks and token computations** of `d`: weights `Fin nw` (the updated parameters `θ'`) and tokens
`Fin nt` (`J`); for `(i, j) ∈ P` a multiplication block `m (i, j)` with output gate `o (i, j)`; for each token `j` its
computation `T j`. -/
structure MulBlocks (d : Declaration) where
  nw : ℕ
  nt : ℕ
  P : Finset (Fin nw × Fin nt)
  m : Fin nw × Fin nt → Finset (Fin d.D.N)
  o : Fin nw × Fin nt → Fin d.D.N
  T : Fin nt → Finset (Fin d.D.N)
  m_disjoint : ∀ p ∈ P, ∀ q ∈ P, p ≠ q → Disjoint (m p) (m q)
  /-- `m_{i,j} ⊆ C \ in(C)`. -/
  m_computed : ∀ p ∈ P, Disjoint (m p) (sources d.D)
  o_mem : ∀ p ∈ P, o p ∈ m p
  reach : ∀ p ∈ P, ∀ g ∈ m p, FeedsWithin d.D (m p) g (o p)
  T_disjoint : ∀ j j', j ≠ j' → Disjoint (T j) (T j')
  /-- `T_j ⊆ C \ θ`. -/
  T_θ : ∀ j, Disjoint (T j) d.θ

namespace MulBlocks

variable {d : Declaration} (B : MulBlocks d)

/-- `S` **completes** the multiplication block `m p`: `m p ⊆ S \ in(S)`. -/
def Completes (S : Subcircuit d.D) (p : Fin B.nw × Fin B.nt) : Prop := B.m p ⊆ S.inner

open Classical in
/-- `θ'_S`: the weights `i` for which `S` completes some `m_{i,j}`. -/
noncomputable def weightsCompleted (S : Subcircuit d.D) : Finset (Fin B.nw) :=
  (B.P.filter fun p => B.Completes S p).image Prod.fst

/-- `X ⊆ T_j` **cuts** `A` off from `m_{i',j}`: `X` meets `m_{i',j}`, or meets every path inside `T_j` from `o_{i',j}`
to a gate of `A` (no such path runs inside `T_j \ X`). -/
def Cuts (j : Fin B.nt) (A X : Finset (Fin d.D.N)) (i' : Fin B.nw) : Prop :=
  (X ∩ B.m (i', j)).Nonempty ∨ ∀ a ∈ A, ¬ FeedsWithin d.D (B.T j \ X) (B.o (i', j)) a

end MulBlocks

end Nci.Covering
