import Models.Nci.Assumptions.Cover
import Definitions.Nci.CoverWidth

/-!
# The cover theorem: the width bound from the wiring

`WidthOfCut` derives the width a block pays from a structural property of the training circuit, the activation's cut
width (`MulBlocks.CutWidth`), and `UpdateHardness`: an inference block that computes a counted product whole is handed at
least `f` gates of its activation region. `CutOfDisjointUpstreams` gives the cut width from a certificate a checker
verifies in linear time, `f` disjoint upstream certificates (`MulBlocks.DisjointUpstreams`). `CutWidthIntensity` and
`CutWidthCoverBound` are the width bounds (`WidthIntensity`, `WidthCoverBound`) with the named premise `Width` replaced
by the cut width.

What is derived is narrower than `Width`, and has to be: `Width` asks it of every set of at most `s` gates, whatever
it is handed, and a block handed many updated parameters can compute most of an activation's upstream itself and
be handed one gate of the region. The derivation covers blocks handed at most `k < b D` bits outside `θ`, the inference
blocks, which is all that the width bounds read.
-/

namespace Nci.Cover.Guarantees

/-- **Width from the cut width**: under `UpdateHardness` and the cut width `f` at depth `D` with `k < b D`, an inference
block that computes a counted product of use `j` whole is handed at least `f` gates of `A j`.

Were it handed fewer, the set `X` of them is a set of fewer than `f` gates; the cut width leaves upstream products over
`D` parameters connected to the product's activation inside `A j \ X`, which the block, handed no gate of `A j \ X`,
computes whole (`BackwardClosure`), and `UpdateHardness` prices them at `b D > k` bits. In ML terms: a block that computes
a token's product at a layer whole, handed fewer than four bits of that token's FP4 activation region, rebuilds all but
two percent of a layer's updated weights (Kimi K2). -/
def WidthOfCut : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N)) (s k D f : ℕ) (b : ℝ),
    0 < b → Assumptions.UpdateHardness M s b → M.CutWidth A D f → (k : ℝ) < b * D →
    ∀ I, C.InferenceBlock s k I → ∀ p ∈ M.P, M.m p ⊆ C.computed I → f ≤ (C.inputs I ∩ A p.1).card

/-- **Cut width from disjoint upstreams**: `f` pairwise disjoint upstream certificates at depth `D` inside each
activation region give the cut width `f` at depth `D`. A set of fewer than `f` gates misses one of the `f` parts, whose
upstream products still reach the activation inside the region minus the set. -/
def CutOfDisjointUpstreams : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N)) (D f : ℕ),
    M.DisjointUpstreams A D f → M.CutWidth A D f

/-- **Intensity, cut width `f`**: `WidthIntensity` with the cut width in place of `Width`. Under
`ConditionalUpdateHardness`, the cut width `f` at depth `D` with `k < b D`, and the upstream certificate's disjoint
regions (its depth `D₀` plays no part), an inference block finishes at most `k² / (4 b f) + k` counted products:
`4 b f · #(finished) ≤ k² + 4 b f k`. -/
def CutWidthIntensity : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N))
    (s k D D₀ f : ℕ) (b : ℝ),
    0 < b → Assumptions.ConditionalUpdateHardness M A s b → M.CutWidth A D f → M.Upstream A D₀ → (k : ℝ) < b * D →
    ∀ I, C.InferenceBlock s k I → 4 * b * f * (M.finished I).card ≤ k * k + 4 * b * f * k

/-- **Cover bound, cut width `f`**: `WidthCoverBound` with the cut width in place of `Width`. Under
`ConditionalUpdateHardness`, the cut width `f` at depth `D` with `k < b D`, and the upstream certificate's disjoint
regions, a cover of `C` by inference blocks has `4 b f · #P ≤ #𝓘 · (k² + 4 b f k)`.

Kimi K2 at `f = 4` (the product's output reads all four bits of its FP4 activation, and no three gates cut more than 98%
of a layer's updated weights off from them, `D ≈ 8·10⁶`): about 660, from the wiring and the two hardness premises, with
no width premise. -/
def CutWidthCoverBound : Prop :=
  ∀ (C : Circuit) {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L) (A : J → Finset (Fin C.N))
    (s k D D₀ f : ℕ) (b : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
    0 < b → Assumptions.ConditionalUpdateHardness M A s b → M.CutWidth A D f → M.Upstream A D₀ → (k : ℝ) < b * D →
    (∀ I ∈ 𝓘, C.InferenceBlock s k I) → C.Covers 𝓘 →
    4 * b * f * M.P.card ≤ 𝓘.card * (k * k + 4 * b * f * k)

end Nci.Cover.Guarantees
