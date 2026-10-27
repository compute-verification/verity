import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Image
import Mathlib.Data.Set.Pairwise.Basic

/-!
# The cover theorem: the vocabulary

The training dampener, stated over one Boolean circuit `C`. A run declares a cover of `C` by inference blocks: sets of
at most `s` gates, each reading at most `k` bits besides the registered parameters `θ`, with declared compute `s · #𝓘`.
The cover theorem bounds how many of training's counted products one block can finish, so that a cover of a
training circuit needs many blocks. Nothing here reads what a gate computes, only the wiring.

The covering bound (`Nci.Guarantees.CoveringBoundTwoCosts`) is stated in `Nci.Covering`'s vocabulary, over a
declaration's gates. This one is the bare wiring, so that it also reads a declaration bit by bit (`Circuit.ofDecl`), and
its products carry a use and an updated parameter, so that one parameter's many uses (fan-out) and a token's keyed
lookups (`Keyed`) can be counted. On a declaration's gates the two agree: its inference blocks and covers are
`Nci.Covering`'s, and `UpdateHardness` is `Nci.Assumptions.ParameterHardAt` (`Proofs.Nci.Cover.Gates`).

* `Circuit`: gates `Fin N`, each reading gates of smaller index (so the circuit is acyclic), and the registered
  parameters `θ`, which are input gates.
* `Circuit.inputs`, `Circuit.computed`, `Circuit.outputs`: for any set of gates, the gates it is handed (one bit each),
  the gates it computes from them, and the gates it produces.
* `Circuit.IsPath`, `Circuit.ReachesIn`: a gate's value depends on another's along a path, inside a region.
* `Circuit.InferenceBlock`, `Circuit.Covers`: the declared blocks (at most `s` gates each), and the condition that they
  compute all of `C`.
* `MulBlocks`: training's products, each a use `j` of an updated parameter `l`, and the counted ones `P`.
* `MulBlocks.Upstream`: the upstream certificate, a structural property of the training computation: each use's
  activation is computed, inside a region of its own, from products over at least `D` distinct updated parameters.
-/

namespace Nci.Cover

/-- **A Boolean circuit with registered parameters.** Its gates are `Fin N`, numbered so that every predecessor of a
gate has a smaller index, which makes the circuit acyclic. A gate with no predecessors is an input gate. The registered
parameters `θ` are input gates: the values committed before the run (the checkpoint's weights), which every block may
read for free. -/
structure Circuit where
  /-- The number of gates. -/
  N : ℕ
  /-- The gates each gate reads. -/
  pred : Fin N → Finset (Fin N)
  pred_lt : ∀ g, ∀ h ∈ pred g, h < g
  /-- The registered parameters. -/
  θ : Finset (Fin N)
  θ_input : ∀ g ∈ θ, pred g = ∅

namespace Circuit

/-- **The inputs of `I`**, `in(I)`: the gates of `I` with no predecessors, or with a predecessor outside `I`. These are
the gates `I` is handed rather than computes, one bit each. -/
def inputs (C : Circuit) (I : Finset (Fin C.N)) : Finset (Fin C.N) :=
  I.filter fun g => C.pred g = ∅ ∨ ¬ C.pred g ⊆ I

/-- **The gates `I` computes**: its gates that are not its inputs, so each has predecessors, all of them in `I`. `I`
computes all of a set `X` of gates when `X ⊆ computed I`. -/
def computed (C : Circuit) (I : Finset (Fin C.N)) : Finset (Fin C.N) :=
  I \ C.inputs I

/-- **The outputs of `I`**: its gates that no gate of `I` reads. -/
def outputs (C : Circuit) (I : Finset (Fin C.N)) : Finset (Fin C.N) :=
  I.filter fun g => ∀ h ∈ I, g ∉ C.pred h

/-- **A path from `x` to `y`**: a list of gates that starts at `x` and ends at `y`, each gate read by the next, so that
`y`'s value depends on `x`'s along it. -/
def IsPath (C : Circuit) (π : List (Fin C.N)) (x y : Fin C.N) : Prop :=
  π.head? = some x ∧ π.getLast? = some y ∧ π.IsChain fun u v => u ∈ C.pred v

/-- **`x` reaches `y` inside `A`**: some path from `x` to `y` has all of its gates in `A`. -/
def ReachesIn (C : Circuit) (A : Finset (Fin C.N)) (x y : Fin C.N) : Prop :=
  ∃ π, C.IsPath π x y ∧ ∀ v ∈ π, v ∈ A

/-- **An inference block** of size `s` and read cap `k`: a set of at most `s` gates that is handed at most `k` bits
besides the registered parameters, `#(inputs I \ θ) ≤ k`, as `Nci.Covering.InferenceBlock` is on a declaration's gates.
Inference fits in such blocks, since its weights are registered, in `θ`, and a block is handed only tokens and
activations. -/
def InferenceBlock (C : Circuit) (s k : ℕ) (I : Finset (Fin C.N)) : Prop :=
  I.card ≤ s ∧ (C.inputs I \ C.θ).card ≤ k

/-- **`𝓘` covers `C`**: every gate that is not an input gate is computed by some block of `𝓘`. The blocks may overlap,
and a gate may be computed in several of them. The declared compute is `s · #𝓘`; tying it to hardware time is part (1)
of the dampener's argument, not the cover theorem's. -/
def Covers (C : Circuit) (𝓘 : Finset (Finset (Fin C.N))) : Prop :=
  ∀ g, (C.pred g).Nonempty → ∃ I ∈ 𝓘, g ∈ C.computed I

end Circuit

/-- **Training's multiplication blocks.** A product `p = (j, l)` multiplies an activation by an updated parameter
`l : L` (a weight computed during the run, after the commitment to `θ`); its use `j : J` names the activation, such as
one token's input to one layer at one step. `Pall` holds all products and `P ⊆ Pall` the counted ones. Product `p`'s
gates `m p` are a nonempty set of non-input gates with exactly one output gate `o p`, its value, and distinct products
have disjoint gates. -/
structure MulBlocks (C : Circuit) (J L : Type*) where
  /-- All products. -/
  Pall : Finset (J × L)
  /-- The counted products. -/
  P : Finset (J × L)
  P_subset : P ⊆ Pall
  /-- The gates of each product. -/
  m : J × L → Finset (Fin C.N)
  m_nonempty : ∀ p ∈ Pall, (m p).Nonempty
  m_computed : ∀ p ∈ Pall, ∀ g ∈ m p, (C.pred g).Nonempty
  m_disjoint : (Pall : Set (J × L)).PairwiseDisjoint m
  /-- The output gate of each product. -/
  o : J × L → Fin C.N
  m_outputs : ∀ p ∈ Pall, C.outputs (m p) = {o p}

namespace MulBlocks

variable {C : Circuit} {J L : Type*} [DecidableEq J]

/-- **The uses of the counted products.** -/
def uses (M : MulBlocks C J L) : Finset J :=
  M.P.image Prod.fst

/-- **The counted products `I` computes whole**: all of their gates. -/
def whole (M : MulBlocks C J L) (I : Finset (Fin C.N)) : Finset (J × L) :=
  M.P.filter fun p => M.m p ⊆ C.computed I

/-- **The counted products `I` finishes**: those whose output gate `I` computes, whether or not it computes their other
gates. -/
def finished (M : MulBlocks C J L) (I : Finset (Fin C.N)) : Finset (J × L) :=
  M.P.filter fun p => M.o p ∈ C.computed I

/-- **The upstream certificate** at depth `D`, with activation regions `A j` for the uses `j` of the counted products: a
structural property of the training computation, checkable on a named circuit.

The regions are pairwise disjoint and hold no registered parameter. Each use `j` has a set `U ⊆ Pall` of upstream
products, all of whose gates lie in `A j`, over at least `D` distinct updated parameters, and each counted product
`(j, l)` has a gate `g` that reads an activation gate `a ∈ A j` that the output gate of every product in `U` reaches
inside `A j` (and so does each of its gates, through the product's output). In a transformer, `j` is one token's input
to one layer, `A j` the gates computing it from that token's previous layer, and `U` that layer's products for the
token, over its active weights: in Kimi K2 about `4·10⁸` of them.

First-layer products have no certificate, since their activation is a function of the token id: they are not in `P`. -/
structure Upstream [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N)) (D : ℕ) : Prop where
  /-- The regions of distinct uses are disjoint. -/
  disjoint : (M.uses : Set J).PairwiseDisjoint A
  /-- No region holds a registered parameter. -/
  disjoint_θ : ∀ j ∈ M.uses, Disjoint (A j) C.θ
  /-- Each use's upstream products, and the activation each of its counted products reads. -/
  upstream : ∀ j ∈ M.uses, ∃ U ⊆ M.Pall,
    (∀ q ∈ U, M.m q ⊆ A j) ∧
    D ≤ (U.image Prod.snd).card ∧
    ∀ p ∈ M.P, p.1 = j → ∃ g ∈ M.m p, ∃ a ∈ C.pred g ∩ A j, ∀ q ∈ U, C.ReachesIn (A j) (M.o q) a

end MulBlocks

end Nci.Cover
