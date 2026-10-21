import Definitions.Nci.Cover

/-!
# The cover theorem: the activation's cut width

The structural premise that replaces the width assumption (`Assumptions.Width`): a statement about the wiring of the
training circuit inside each activation region, checkable on a named circuit, that with `UpdateHardness` implies that a
block computing a counted product whole is handed at least `f` gates of its activation region.

* `MulBlocks.CutWidth`: every set of fewer than `f` gates leaves, for each counted product, upstream products over at
  least `D` distinct updated parameters connected to the product's activation gates inside the region. A cut of the
  activation's upstream computation, a set of gates that leaves fewer than `D` parameters' worth of it connected, has at
  least `f` gates.
* `MulBlocks.DisjointUpstreams`: `f` pairwise disjoint sub-regions of the activation region, each holding an upstream
  certificate of its own at depth `D`. It implies `CutWidth`, and it is a certificate a checker verifies in time linear
  in the regions: disjointness, and one path per gate of each upstream product.

A block that is handed the gates `X` of the region, and no others, computes everything that reaches the product's
activation inside the region minus `X` (`BackwardClosure`), so it is not enough that `f` paths enter the activation
(Menger): the block may compute their starting gates itself. The cut width counts what lies behind the cut instead, so
the `f` bits of one activation value may share their upstream (one dot product), as long as no `f - 1` gates separate
that upstream from all of them.
-/

namespace Nci.Cover

namespace MulBlocks

variable {C : Circuit} {J L : Type*}

/-- **The cut width** `f` at depth `D`, for the activation regions `A`: for every counted product `p` of use `j` and
every set `X` of fewer than `f` gates, some upstream products `U ⊆ Pall` over at least `D` distinct updated parameters
have every gate reaching, inside `A j \ X`, an activation gate that a gate of `p` reads.

At `X = ∅` it is the reach clause of the upstream certificate (`Upstream`). With `k < b D`, a block handed fewer than
`f` gates of `A j` would compute all of `U` (`BackwardClosure` on `A j \ X`), which `UpdateHardness` prices above `k`.

It caps `f` at the number of activation gates a product reads, since `X` may be all of them: at FP4, `f ≤ 4` when the
product's output reads all four bits of its activation, and `f ≤ 2` for an output that reads only two. It holds at
`f = 4` when the four bits are free (paper argument, not in Lean, which has no gate semantics): if, for every set `R` of
upstream products over fewer than `D` parameters, fixing every value entering `A j` except the parameters of the products
outside `R` still leaves the activation taking more than `8` of its `16` values, no three gates `X` can cut it, since the
activation would then be a function of `X`'s three bits. Kimi K2: `D = k / b + 1 ≈ 8·10⁶`, two percent of one layer's
`4·10⁸` active weights. -/
def CutWidth [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N)) (D f : ℕ) : Prop :=
  ∀ p ∈ M.P, ∀ X : Finset (Fin C.N), X.card < f → ∃ U ⊆ M.Pall, D ≤ (U.image Prod.snd).card ∧
    ∀ q ∈ U, ∀ x ∈ M.m q, ∃ g ∈ M.m p, ∃ a ∈ C.pred g, C.ReachesIn (A p.1 \ X) x a

/-- **Disjoint upstreams** `f` at depth `D`, for the activation regions `A`: each counted product `p` of use `j` has `f`
pairwise disjoint sub-regions `B i ⊆ A j`, each with upstream products `U ⊆ Pall` over at least `D` distinct updated
parameters, every gate of which reaches, inside `B i`, one activation gate that a gate of `p` reads.

It is the upstream certificate (`Upstream`) `f` times over, in disjoint parts of the region, and implies `CutWidth`: fewer
than `f` gates miss one of the parts. In a transformer the parts are groups of a layer's hidden units with their partial
sums, each routed to its own bit of the activation along paths disjoint from the others; whether the routing exists
through the final reduction, the normalization and the quantizer is a flow question on the circuit's last stages. -/
def DisjointUpstreams [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N)) (D f : ℕ) : Prop :=
  ∀ p ∈ M.P, ∃ B : Fin f → Finset (Fin C.N), (∀ i i', i ≠ i' → Disjoint (B i) (B i')) ∧
    ∀ i, B i ⊆ A p.1 ∧ ∃ U ⊆ M.Pall, D ≤ (U.image Prod.snd).card ∧
      ∃ g ∈ M.m p, ∃ a ∈ C.pred g, ∀ q ∈ U, ∀ x ∈ M.m q, C.ReachesIn (B i) x a

end MulBlocks

end Nci.Cover
