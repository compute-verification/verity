import Definitions.Nci

/-!
# Updated matrix multiplication chains with several products to a gate

`Chain` gives each product `W_ℓ[i, j] x_t^{(ℓ)}[j]` a gate of its own (`prod_inj`). A circuit that computes in `Dot16`
steps does not: one step gate reads the previous accumulator, 16 activations and 16 weights, and holds 16 products. So a
declaration written in `Dot16` contains no per-product reference circuit, and every product of one is dark in it.
`PackedChain D p` lets `prod` send up to `p` products to one gate (`prod_mult`), `p = 16` for `Dot16`, and at `p = 1` it
is `Chain` (`Chain.toPacked`). Its vocabulary is `Chain`'s, word for word, where "`R` holds a product" now means that
`R` holds the product's gate.
-/

namespace Nci

open Verity

/-- **An updated matrix multiplication chain that `D` runs, at most `p` products to a gate**: `Chain`, with its first
condition relaxed from one gate per product to at most `p` products per gate (`prod_mult`). A `Dot16` step gate holds 16
products `W_ℓ[i, j] x_t^{(ℓ)}[j]` of one entry `y_t^{(ℓ)}[i]`, at 16 consecutive `j`, and feeds the next step of that
entry, so `prod` maps those 16 products to it, and each feeds the gate `y ℓ t i` (`Circuit.Feeds` is reachability). -/
structure PackedChain (D : Circuit) (p : ℕ) where
  /-- The depth: the number of multiplications. -/
  L : ℕ
  /-- The number of token positions. -/
  N : ℕ
  /-- The inner dimension of multiplication `ℓ`: the length of its input vectors, the number of columns of `W_ℓ`. -/
  d : Fin L → ℕ
  /-- The output dimension of multiplication `ℓ`: the length of its output vectors, the number of rows of `W_ℓ`. -/
  e : Fin L → ℕ
  /-- The gate that computes the product `W_ℓ[i, j] x_t^{(ℓ)}[j]`, possibly with other products. -/
  prod : (ℓ : Fin L) → Fin N → Fin (e ℓ) → Fin (d ℓ) → Fin D.N
  prod_mult : ∀ g, (Finset.univ.filter fun x : (ℓ : Fin L) × Fin N × Fin (e ℓ) × Fin (d ℓ) =>
    prod x.1 x.2.1 x.2.2.1 x.2.2.2 = g).card ≤ p
  prod_computed : ∀ ℓ t i j, D.isInput (prod ℓ t i j) = false
  /-- The gate that holds the complete entry `y_t^{(ℓ)}[i]`. -/
  y : (ℓ : Fin L) → Fin N → Fin (e ℓ) → Fin D.N
  sums : ∀ ℓ t i j, D.Feeds (prod ℓ t i j) (y ℓ t i)
  /-- The link into multiplication `ℓ + 1`: its input entry `j` is computed using the output entry `π ℓ h j` of
  multiplication `ℓ`. -/
  π : (ℓ : Fin L) → (h : ℓ.val + 1 < L) → Fin (d ⟨ℓ.val + 1, h⟩) → Fin (e ℓ)
  π_inj : ∀ ℓ h, Function.Injective (π ℓ h)
  links : ∀ ℓ h t i j, D.Feeds (y ℓ t (π ℓ h j)) (prod ⟨ℓ.val + 1, h⟩ t i j)

namespace PackedChain

open Finset

variable {D : Circuit} {p : ℕ} (c : PackedChain D p)

/-- `G = ∑_ℓ G_ℓ`, the number of products of the chain, `G_ℓ = e_ℓ d_ℓ N`: products, not the gates that hold them. -/
def G : ℕ := ∑ ℓ, c.e ℓ * c.d ℓ * c.N

/-- The weight entries `W_ℓ[i, j]` that `R` uses: those of which `R` holds a product's gate. -/
def weightsUsed (R : Finset (Fin D.N)) : Finset ((ℓ : Fin c.L) × Fin (c.e ℓ) × Fin (c.d ℓ)) :=
  univ.filter fun w => ∃ t, c.prod w.1 t w.2.1 w.2.2 ∈ R

/-- The input entries `x_t^{(ℓ)}[j]` that `R` uses: those of which `R` holds a product's gate. -/
def inputsUsed (R : Finset (Fin D.N)) : Finset ((ℓ : Fin c.L) × Fin c.N × Fin (c.d ℓ)) :=
  univ.filter fun x => ∃ i, c.prod x.1 x.2.1 i x.2.2 ∈ R

/-- The entry `y_t^{(ℓ)}[i]` is **whole** in `R` when `R` holds the gates of all of its products. -/
def Whole (R : Finset (Fin D.N)) (ℓ : Fin c.L) (t : Fin c.N) (i : Fin (c.e ℓ)) : Prop :=
  ∀ j, c.prod ℓ t i j ∈ R

/-- The entry `y_t^{(ℓ)}[i]` is **split** in a division `P` into inference blocks: no inference block holds the gates of
all of its products. -/
def Split {n : ℕ} (P : Partition D n) (ℓ : Fin c.L) (t : Fin c.N) (i : Fin (c.e ℓ)) : Prop :=
  ¬ ∃ u, c.Whole (P.gates u) ℓ t i

/-- Position `t` is a **full position** of `R` if `R` computes some whole entry at position `t`. -/
def Full (R : Finset (Fin D.N)) (t : Fin c.N) : Prop :=
  ∃ ℓ i, c.Whole R ℓ t i

/-- The input entry `x_t^{(ℓ)}[j]` is **linked to an entry whole in `R`**: `ℓ > 0`, and the entry
`y_t^{(ℓ-1)}[π j]` it is computed from is whole in `R`, so that `R` can compute it from its own products. -/
def LinkedWhole (R : Finset (Fin D.N)) : (ℓ : Fin c.L) → Fin c.N → Fin (c.d ℓ) → Prop
  | ⟨0, _⟩, _, _ => False
  | ⟨ℓ + 1, h⟩, t, j => c.Whole R ⟨ℓ, by omega⟩ t (c.π ⟨ℓ, by omega⟩ h j)

open Classical in
/-- The number of inference blocks of a division `P` that hold gates of products of the entry `y_t^{(ℓ)}[i]`. -/
noncomputable def spread {n : ℕ} (P : Partition D n) (ℓ : Fin c.L) (t : Fin c.N) (i : Fin (c.e ℓ)) : ℕ :=
  (univ.filter fun u => ∃ j, c.prod ℓ t i j ∈ P.gates u).card

end PackedChain

/-- **A chain is a packed chain with one product to a gate**: `prod_inj` gives `prod_mult` at `p = 1`. -/
def Chain.toPacked {D : Circuit} (c : Chain D) : PackedChain D 1 where
  L := c.L
  N := c.N
  d := c.d
  e := c.e
  prod := c.prod
  prod_mult _ := Finset.card_le_one.mpr fun _ hp _ hq =>
    c.prod_inj ((Finset.mem_filter.mp hp).2.trans (Finset.mem_filter.mp hq).2.symm)
  prod_computed := c.prod_computed
  y := c.y
  sums := c.sums
  π := c.π
  π_inj := c.π_inj
  links := c.links

end Nci
