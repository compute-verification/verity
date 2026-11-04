import Definitions.Nci.Threads

/-!
# Links: the counted multiplications an activation is computed from

A `Threading` names the reference circuit's counted multiplications and the activation each reads, but not where an
activation comes from. `Threading.Links` adds that: for each value `a`, the counted multiplications `src a` whose sum it
is computed from (an output entry's products in a forward multiplication, or a weight's products with an activation
gradient in the backward one), possibly together with other values; `src a` is empty for a value computed from no
counted multiplication (an embedding, an attention output). It generalizes a `Chain`'s links: there the input entry
`x_t^{(ℓ+1)}[j]` is computed from the products of the entry `y_t^{(ℓ)}[π j]`.

* `Links.Whole L R a`: `a` is computed from counted multiplications, all of them in `R`, so that `R` can compute it from
  its own products: a chain's `LinkedWhole`.
* `Threading.actsUsed T R`: the activations of which `R` holds a counted multiplication: a chain's `inputsUsed`.
* `ChainIn ch T L`: where a chain and a threading overlap: the chain's products are counted multiplications, its
  entries are distinct weight and activation gates, and its links are among `L`'s.
-/

namespace Nci

open Verity Finset

namespace Threading

variable {c : RegisteredCircuit} (T : Threading c)

/-- The activations `R` uses: those of which `R` holds a counted multiplication. -/
def actsUsed (R : Finset (Fin c.D.N)) : Finset (Fin c.D.N) := (T.mults R).image T.act

/-- **Links of a threading**: for each value `a`, the counted multiplications `src a` whose sum `a` is computed from,
each of which feeds `a`. -/
structure Links where
  /-- The counted multiplications `a` is computed from; empty when it is computed from none. -/
  src : Fin c.D.N → Finset (Fin c.D.N)
  src_M : ∀ a, src a ⊆ T.M
  src_feeds : ∀ a, ∀ g ∈ src a, c.D.Feeds g a

variable {T}

/-- The value `a` is **whole** in `R`: it is computed from counted multiplications and all of them lie in `R`, so that
`R` can compute it from its own products. A value computed from none is whole in no `R`. -/
def Links.Whole (L : T.Links) (R : Finset (Fin c.D.N)) (a : Fin c.D.N) : Prop :=
  (L.src a).Nonempty ∧ L.src a ⊆ R

instance instDecidablePredFinNDWhole (L : T.Links) (R : Finset (Fin c.D.N)) : DecidablePred (L.Whole R) :=
  fun a => inferInstanceAs (Decidable ((L.src a).Nonempty ∧ L.src a ⊆ R))

end Threading

/-- **A chain in a threading's terms**: the chain `ch` that `c` runs is part of the threading `T` with links `L` when
1. its products are counted multiplications, the product `prod ℓ t i j` reading the weight gate `wgate ⟨ℓ, i, j⟩` and
   the activation gate `xgate ⟨ℓ, t, j⟩`, distinct entries going to distinct gates;
2. the first multiplication's input entries are computed from no counted multiplication; and
3. each later input entry `x_t^{(ℓ+1)}[j]` is computed from every product of the output entry `y_t^{(ℓ)}[π j]` it is
   linked to, and possibly from more. -/
structure ChainIn {c : RegisteredCircuit} (ch : Chain c.D) (T : Threading c) (L : T.Links) where
  wgate : (ℓ : Fin ch.L) × Fin (ch.e ℓ) × Fin (ch.d ℓ) → Fin c.D.N
  wgate_inj : Function.Injective wgate
  xgate : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.d ℓ) → Fin c.D.N
  xgate_inj : Function.Injective xgate
  prod_M : ∀ ℓ t i j, ch.prod ℓ t i j ∈ T.M
  weight_prod : ∀ ℓ t i j, T.weight (ch.prod ℓ t i j) = wgate ⟨ℓ, i, j⟩
  act_prod : ∀ ℓ t i j, T.act (ch.prod ℓ t i j) = xgate ⟨ℓ, t, j⟩
  src_first : ∀ (h : 0 < ch.L) t j, L.src (xgate ⟨⟨0, h⟩, t, j⟩) = ∅
  src_link : ∀ (ℓ : Fin ch.L) (h : ℓ.val + 1 < ch.L) t j j',
    ch.prod ℓ t (ch.π ℓ h j) j' ∈ L.src (xgate ⟨⟨ℓ.val + 1, h⟩, t, j⟩)

end Nci
