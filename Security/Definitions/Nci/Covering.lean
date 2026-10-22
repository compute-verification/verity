import Definitions.Nci

/-!
# The covering bound: its vocabulary

The covering bound (Daniel's standard form, 22 Oct 2026) counts the inference blocks it takes to compute a circuit `C`
that runs updated multiplications, with one width `b` for a weight and an activation and one compression rate `r` for
both. It is stated over a declaration `(C, θ)` (`Verity.Declaration`): `C`'s gates are `Fin C.N` in topological order, a
gate's predecessors are the gates it reads (`C.reads`), `in(C)` is its input gates, `C \ in(C)` its computed gates
(`C.computed`), and the registered parameters `θ ⊆ in(C)` are the declaration's baseline. Counts are numbers of gates:
on the Boolean profile, where every gate holds one bit, they are bits.

* `ReachesIn C U g t`: `g` reaches `t` inside `U`, along a path of wires whose gates all lie in `U`.
* `Subcircuit C`: a set `S` of gates with inputs `in(S) ⊆ S`, every gate of `S \ in(S)` (the gates `S` computes) having
  all its predecessors in `S`.
* `Block d s k I`, `Covers d s k 𝓘` and `slowdown d s 𝓘`: an `(s,k)`-inference block (`|I| ≤ s`,
  `|in(I) \ θ| ≤ k`), an `(s,k)`-covering (blocks that together compute `C \ in(C)`), and its slowdown
  `ρ(𝓘) = s|𝓘|/|C|`.
* `Mults d ι τ`: the multiplication blocks `m_{i,j}` for `(i,j) ∈ P ⊆ θ' × J`, a weight `i : ι` of the updated
  parameters `θ'` and a token `j : τ`, their output gates `o_{i,j}`, and the token computations `T_j`.
* `Mults.Completes`, `Mults.completed` and `Mults.Cuts`: `S` completes `m_{i,j}` (`m_{i,j} ⊆ S \ in(S)`), the weights
  `θ'_S` of the blocks `S` completes, and a set of gates that cuts an activation off from a multiplication block. The two
  assumptions (`Nci.Assumptions.ParameterHard`, `Nci.Assumptions.ActivationChained`) are stated in them.
-/

namespace Nci.Cover

open Verity Finset

/-- `g` **reaches `t` inside `U`**: a path of wires from `g` to `t` whose gates, `g` and `t` included, all lie in `U`. -/
inductive ReachesIn (C : Circuit) (U : Finset (Fin C.N)) : Fin C.N → Fin C.N → Prop
  | refl {t : Fin C.N} : t ∈ U → ReachesIn C U t t
  | step {g h t : Fin C.N} : g ∈ U → g ∈ C.reads h → ReachesIn C U h t → ReachesIn C U g t

/-- **A subcircuit** `S ⊆ C` with its inputs `in(S) ⊆ S`: every gate of `S \ in(S)`, the gates `S` computes, has all its
predecessors in `S`. A gate of `in(C)` has no predecessors and is computed by nothing, so it is an input of every
subcircuit that holds it: a subcircuit that uses a gate of `in(C) \ θ` reads it, as an IU does (`Nci.novelReads`). -/
@[ext]
structure Subcircuit (C : Circuit) where
  /-- `S`: its gates, its inputs among them. -/
  gates : Finset (Fin C.N)
  /-- `in(S)`: the gates of `S` that it reads rather than computes. -/
  ins : Finset (Fin C.N)
  ins_subset : ins ⊆ gates
  reads_subset : ∀ g ∈ gates, g ∉ ins → C.reads g ⊆ gates
  input_mem : ∀ g ∈ gates, C.isInput g = true → g ∈ ins

/-- `S \ in(S)`: the gates `S` computes. -/
def Subcircuit.computes {C : Circuit} (S : Subcircuit C) : Finset (Fin C.N) := S.gates \ S.ins

/-- **An `(s,k)`-inference block** of `d`: a subcircuit `I` of at most `s` gates, its inputs included, with at most `k`
inputs outside the registered parameters, `|I| ≤ s` and `|in(I) \ θ| ≤ k`. -/
def Block (d : Declaration) (s k : ℝ) (I : Subcircuit d.D) : Prop :=
  (I.gates.card : ℝ) ≤ s ∧ ((I.ins \ d.θ).card : ℝ) ≤ k

/-- **An `(s,k)`-covering** of `d`: a set of `(s,k)`-inference blocks that together compute `C \ in(C)`, each computed
gate of `C` lying in `I \ in(I)` for some block `I`. -/
def Covers (d : Declaration) (s k : ℝ) (𝓘 : Finset (Subcircuit d.D)) : Prop :=
  (∀ I ∈ 𝓘, Block d s k I) ∧ ∀ g ∈ d.D.computed, ∃ I ∈ 𝓘, g ∈ I.computes

/-- **The slowdown of a covering** `ρ(𝓘) = s|𝓘|/|C|`: the gates its blocks may hold, `s` each, per gate of `C`. `|C|`
counts every gate, inputs included, as `|I|` does for a block, so `C` as the one block of itself has slowdown 1. -/
noncomputable def slowdown (d : Declaration) (s : ℝ) (𝓘 : Finset (Subcircuit d.D)) : ℝ :=
  s * 𝓘.card / d.D.N

/-- **Multiplication blocks** of `d`: for each pair `(i, j) ∈ P ⊆ θ' × J` of a weight `i` of the updated parameters `θ'`
and a token `j`, a block `m_{i,j}` of computed gates with an output gate `o_{i,j} ∈ m_{i,j}` (so `m_{i,j}` is nonempty)
that every gate of `m_{i,j}` reaches inside `m_{i,j}`, the blocks of distinct pairs being disjoint; and for each token
`j ∈ J` its computation `T_j`, a set of gates outside `θ`, the computations of distinct tokens being disjoint.

In a transformer each (weight, token) pair is one multiplication of a forward pass: `m_{i,j}` computes weight `i` times
the entry of token `j`'s activation it multiplies, and `T_j` is the rest of token `j`'s forward pass. -/
structure Mults (d : Declaration) (ι τ : Type) where
  /-- `θ'`: the weights of the updated parameters. -/
  θ' : Finset ι
  /-- `J`: the tokens. -/
  J : Finset τ
  /-- `P ⊆ θ' × J`: the pairs that have a multiplication block. -/
  P : Finset (ι × τ)
  P_subset : P ⊆ θ' ×ˢ J
  /-- `m_{i,j}`: the multiplication block of the pair `(i, j)`. -/
  m : ι → τ → Finset (Fin d.D.N)
  m_computed : ∀ p ∈ P, m p.1 p.2 ⊆ d.D.computed
  /-- `o_{i,j}`: the output gate of `m_{i,j}`. -/
  o : ι → τ → Fin d.D.N
  o_mem : ∀ p ∈ P, o p.1 p.2 ∈ m p.1 p.2
  reachesIn_o : ∀ p ∈ P, ∀ g ∈ m p.1 p.2, ReachesIn d.D (m p.1 p.2) g (o p.1 p.2)
  m_disjoint : ∀ p ∈ P, ∀ q ∈ P, p ≠ q → Disjoint (m p.1 p.2) (m q.1 q.2)
  /-- `T_j`: token `j`'s computation. -/
  T : τ → Finset (Fin d.D.N)
  T_disjoint_θ : ∀ j ∈ J, Disjoint (T j) d.θ
  T_disjoint : ∀ j ∈ J, ∀ j' ∈ J, j ≠ j' → Disjoint (T j) (T j')

namespace Mults

variable {d : Declaration} {ι τ : Type} (M : Mults d ι τ)

/-- `S` **completes** `m_{i,j}`: it computes every gate of it, `m_{i,j} ⊆ S \ in(S)`. -/
def Completes (S : Subcircuit d.D) (i : ι) (j : τ) : Prop := M.m i j ⊆ S.computes

open Classical in
/-- `θ'_S`: the weights `i` for which `S` completes some `m_{i,j}`, `(i, j) ∈ P`. -/
noncomputable def completed [DecidableEq ι] (S : Subcircuit d.D) : Finset ι :=
  (M.P.filter fun p => M.Completes S p.1 p.2).image Prod.fst

/-- `X` **cuts** the gates `A` off from `m_{i,j}` inside token `j`'s computation: `X` meets `m_{i,j}`, or meets every
path inside `T_j` from `o_{i,j}` to `A`, so that no gate of `A` is reached from `o_{i,j}` inside `T_j \ X`. -/
def Cuts (X A : Finset (Fin d.D.N)) (i : ι) (j : τ) : Prop :=
  (X ∩ M.m i j).Nonempty ∨ ¬ ∃ a ∈ A, ReachesIn d.D (M.T j \ X) (M.o i j) a

end Mults

end Nci.Cover
