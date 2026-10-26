import Definitions.Nci.CoverFanOut
import Mathlib.Data.Finset.Union

/-!
# The cover theorem's partial-sum tightening: the vocabulary

The fan-out bounds (`Definitions.Nci.Guarantees.CoverFanOut`) charge a block per value it reads, but they count its products
as a rectangle of tokens and parameters. A matrix multiplication's products are a cube: each is a triple of an output
row `o`, an input column `i` and a token `t`, and its three pair-shadows are its weight `(o, i)`, its activation (the
source `(i, t)`) and its output entry `(o, t)`. By Loomis–Whitney a block's products number at most
`√(#weights · #sources · #entries)`. Weights and sources cost reads. Entries cost reads across a cover: a dot product
whose products are split between blocks passes a partial sum from one block to another, which the receiving block is
handed (B2's Lemma 1). The structure that makes this precise is `Cube`, a structural property of the training
computation, checkable on a named circuit, as `Keyed` is for the fan-out bounds.

* `Keyed.tokReads`: how many of a token's gates (its values' gates) a block is handed.
* `Cube`: each counted product's matrix, row and column, and each output entry's accumulation.
* `Cube.entry`, `Cube.accGates`, `Cube.crossings`, `Cube.Wide`: a product's entry, the accumulations' gates, the
  entries a block contributes whole products to at tokens it reads little of, and the entries' width.
-/

namespace Nci.Cover

namespace Keyed

variable {C : Circuit} {J L Tok Val : Type*} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V w : ℕ}

/-- **`I`'s reads of token `t`**: how many of `t`'s gates, the gates of its values, are among `I`'s inputs. -/
def tokReads [DecidableEq Tok] (K : Keyed M Tok Val d aOut c V w) (I : Finset (Fin C.N)) (t : Tok) : ℕ :=
  (C.inputs I ∩ (K.vals.filter fun v => K.tok v = t).biUnion K.gates).card

end Keyed

/-- **A matrix-multiplication cube** for the counted products of a keyed token structure `K`: each counted product
`p = (j, l)` is a cell `(ℓ, o, i, t)` of a matrix `ℓ`, with row `o`, column `i` and token `t = tok (act p)`. The weight
`l` determines the matrix, the row and the column. Its *source* is its activation `act p` and its *entry* is
`(ℓ, o, t)`, one output coordinate of one token. Like `Keyed`, it is a structural property of the training computation,
checkable on a named circuit.

* Each cell holds at most one counted product (`cell_inj`).
* *Stacking*: products with one activation share their matrix and column (`stack`). All matrices that read a common
  activation are one stacked matrix, and the activation is one of its columns at one token.
* *Accumulations* (B2's Definition 5): each entry `E` has a set `acc E` of non-input gates, holding the gates of each
  of `E`'s products, and the entries' sets are disjoint. Every gate of `acc E` reaches its top `top E` inside it. In an
  honest circuit `acc E` is `E`'s products and the adder tree that sums them, in any order, and `top E` is the sum.

Kimi K2 (forward products): stacking makes one matrix of q_a and kv_a over the attention input, and one of the router and
the experts' gate and up over the MoE input. Each counted product is one routed or dense weight at one token. A fused
dot product (Dot16/32/64) is one accumulation per output coordinate. Fast matrix-multiplication circuits (Strassen,
Winograd) have no per-cell product and no cube. -/
structure Cube {C : Circuit} {J L Tok Val : Type*} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V w : ℕ}
    (K : Keyed M Tok Val d aOut c V w) (Mat Row Col : Type*) where
  /-- The matrix of each weight. -/
  mat : L → Mat
  /-- The row of each weight. -/
  row : L → Row
  /-- The column of each weight. -/
  col : L → Col
  /-- At most one counted product per cell. -/
  cell_inj : ∀ p ∈ M.P, ∀ p' ∈ M.P, mat p.2 = mat p'.2 → row p.2 = row p'.2 → col p.2 = col p'.2 →
    K.tok (K.act p) = K.tok (K.act p') → p = p'
  /-- Stacking: one activation is one column of one matrix. -/
  stack : ∀ p ∈ M.P, ∀ p' ∈ M.P, K.act p = K.act p' → mat p.2 = mat p'.2 ∧ col p.2 = col p'.2
  /-- The accumulation of each entry. -/
  acc : Mat × Row × Tok → Finset (Fin C.N)
  /-- The top of each accumulation. -/
  top : Mat × Row × Tok → Fin C.N
  acc_prod : ∀ p ∈ M.P, M.m p ⊆ acc (mat p.2, row p.2, K.tok (K.act p))
  acc_disjoint : ∀ p ∈ M.P, ∀ p' ∈ M.P,
    (mat p.2, row p.2, K.tok (K.act p)) ≠ (mat p'.2, row p'.2, K.tok (K.act p')) →
    Disjoint (acc (mat p.2, row p.2, K.tok (K.act p))) (acc (mat p'.2, row p'.2, K.tok (K.act p')))
  acc_computed : ∀ p ∈ M.P, ∀ g ∈ acc (mat p.2, row p.2, K.tok (K.act p)), (C.pred g).Nonempty
  acc_top : ∀ p ∈ M.P, ∀ g ∈ acc (mat p.2, row p.2, K.tok (K.act p)),
    C.ReachesIn (acc (mat p.2, row p.2, K.tok (K.act p))) g (top (mat p.2, row p.2, K.tok (K.act p)))

namespace Cube

variable {C : Circuit} {J L Tok Val Mat Row Col : Type*} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V w : ℕ}
  {K : Keyed M Tok Val d aOut c V w}

/-- **The entry of a product**: its matrix, its row and its token. -/
def entry (Q : Cube K Mat Row Col) (p : J × L) : Mat × Row × Tok :=
  (Q.mat p.2, Q.row p.2, K.tok (K.act p))

/-- **The accumulations' gates**: the gates of the counted products' entries' accumulations. -/
def accGates [DecidableEq Mat] [DecidableEq Row] [DecidableEq Tok] (Q : Cube K Mat Row Col) : Finset (Fin C.N) :=
  (M.P.image Q.entry).biUnion Q.acc

/-- **The crossings of `I`** at read rate `f · a` (B2's `Y_I`): the entries of the counted products `I` computes whole
at tokens of which `I` is handed fewer than `f · a` gates. Such an entry is not whole in `I` (`ActivationReads`), so
the rest of its sum is computed elsewhere, and a partial sum crosses between blocks. -/
def crossings [DecidableEq J] [DecidableEq Mat] [DecidableEq Row] [DecidableEq Tok] (Q : Cube K Mat Row Col)
    (I : Finset (Fin C.N)) (f a : ℕ) : Finset (Mat × Row × Tok) :=
  ((M.whole I).filter fun p => K.tokReads I (K.tok (K.act p)) < f * a).image Q.entry

/-- **Entry width `n`**: every entry of a counted product has at least `n` counted products, which have distinct
activations (`cell_inj`, `stack`). Kimi K2: `n = 512`, kv_b's inner dimension, the narrowest of its matrices. -/
def Wide [DecidableEq Mat] [DecidableEq Row] [DecidableEq Tok] (Q : Cube K Mat Row Col) (n : ℕ) : Prop :=
  ∀ p ∈ M.P, n ≤ (M.P.filter fun p' => Q.entry p' = Q.entry p).card

end Cube

end Nci.Cover
