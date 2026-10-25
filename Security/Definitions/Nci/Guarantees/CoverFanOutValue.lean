import Definitions.Nci.Assumptions.Cover
import Definitions.Nci.Guarantees.CoverFanOut
import Definitions.Nci.CoverFanOutValue

/-!
# The cover theorem's fan-out tightening: the read tier from wiring

`FanOutReadIntensity` reaches `k · aOut / f` counted products per block (Kimi K2 at `f = 4`: a slowdown of about
3.6·10⁴) under the named premise `ValueReadHardness`. These guarantees reach it from wiring instead: `Keyed`, the
per-value cut (`Keyed.PerValueCut`) and `ConditionalUpdateHardness` with the values' gates as its region, the additive
form of `UpdateHardness`.

A block that computes a product whole without being handed `f` gates of its activation's key path holds an exit of all
but `g − 1` of the legs of the root that path ends at (`PerValueCut`), and pays for them as a block that holds the root
does: the token is *heavy*, at `d − E − (g − 1)` reads. Otherwise the first value it reads on the key path is spent at
`f` gates of that value's own key path, and no other spent value's key path meets it (`Keyed`'s (I) and (F)). Heavy
tokens' reads are paid at one gate each, not `f`, so the `c k` term carries no `1 / f`. Kimi K2's forward bound is set
by fan-out, where the two agree; with Dgrad counted (`c = 2`) the `c k` term sets it, a little below
`FanOutReadIntensity`'s.

`ValueReadHardness` itself doesn't follow from the wiring: a block that holds a root and cuts the word of a value keyed
to it, reading one of its gates, spends that value at one gate.
-/

namespace Nci.Cover.Guarantees

/-- **Fan-out intensity, `f` bits per value, from wiring**: under `ConditionalUpdateHardness` over the values' gates,
`ColumnsIncompressible`, `Keyed` and its per-value cut at `g` legs (`Keyed.PerValueCut`), with
`k < bcol · V · (E + 1)`, an inference block finishes at most `k · max(aOut / f, c k / (4 δ b)) + k` counted products,
`δ = d − E − (g − 1)`: `4 δ f b · #(finished) ≤ k · max(4 δ b aOut, c f k) + 4 δ f b k`.

The block pays `f` gates for each value it spends on a token it doesn't hold a root of, one per read of a heavy token,
and `b` per parameter outside the values' gates, `k` in all. With `X` spent on light tokens' values, `Y` on heavy
tokens' and `P` on parameters, it computes at most `aOut X / f + c (P / b) (Y / δ)` products whole, and AM–GM bounds
this by `k` times the larger endpoint.

Kimi K2 (`FanOutReadIntensity`'s numbers, `f = 4`, `E = 122`) meets `PerValueCut` at `g = 1` (its docstring), so
`δ = d − E = 390`: `k · aOut / f`, a slowdown of about 3.6·10⁴, which holds as long as `δ ≥ c f k / (4 b aOut) ≈ 215`,
for any `g` up to 176. With Dgrad on the forward's tokens (`c = 2`, `α₀ = 0.301`, position 0's dq_rope cut), the
threshold is 430, above `d − E = 390`, so the `c k` term sets the bound for every `g`: about 6.6·10⁴ at `g = 1`,
against `FanOutReadIntensity`'s 7.3·10⁴; at `aOut = 49152` instead of the cut, 5.5·10⁴ for `g ≤ 65`. Scope:
`FanOutReadIntensity`'s.

It assumes `ConditionalUpdateHardness` with every value's gates as the region, roots' partial sums included: the values
a block is handed, a partial sum `Σ x_i w_i` of updated parameters among them, don't help it rebuild those parameters.
That is stronger than `FanOutIntensity`'s `UpdateHardness`, and than the additive covering bound's premise
(`Nci.Cover.Assumptions.ConditionalParameterHardAt`), whose regions are the token computations. Under plain `UpdateHardness` the reads and the parameters' bits needn't be disjoint inputs, so the `c k` term
loses AM–GM's 4 and becomes `c k / (δ b)`: Kimi K2 gets about 1.6·10⁴, forward only or with Dgrad on shared tokens. -/
def FanOutReadIntensityStructural : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Val] (M : MulBlocks C J L)
    (d aOut c V f : ℕ) (K : Keyed M Tok Val d aOut c V f) (s k E g : ℕ) (b bcol : ℝ),
    0 < b → Assumptions.ConditionalUpdateHardness M (fun _ => K.vals.biUnion K.gates) s b →
    Assumptions.ColumnsIncompressible K s bcol → K.PerValueCut g → (k : ℝ) < bcol * V * (E + 1) →
    ∀ I, C.InferenceBlock s k I →
      4 * (d - E - (g - 1) : ℕ) * f * b * (M.finished I).card ≤
        k * max (4 * (d - E - (g - 1) : ℕ) * b * aOut) ((c : ℝ) * f * k) + 4 * (d - E - (g - 1) : ℕ) * f * b * k

/-- **Fan-out cover bound, `f` bits per value, from wiring**: under `FanOutReadIntensityStructural`'s premises, a
cover of `C` by inference blocks has `#𝓘 ≥ #P / (k · max(aOut / f, c k / (4 δ b)) + k)` blocks, `δ = d − E − (g − 1)`:
`4 δ f b · #P ≤ #𝓘 · (k · max(4 δ b aOut, c f k) + 4 δ f b k)`. Kimi K2 (`g = 1`): a slowdown of about 3.6·10⁴, as
for any `g ≤ 176` (6.6·10⁴ with Dgrad on the forward's tokens), with `FanOutReadIntensity`'s scope. -/
def FanOutReadCoverBoundStructural : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Val] (M : MulBlocks C J L)
    (d aOut c V f : ℕ) (K : Keyed M Tok Val d aOut c V f) (s k E g : ℕ) (b bcol : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
    0 < b → Assumptions.ConditionalUpdateHardness M (fun _ => K.vals.biUnion K.gates) s b →
    Assumptions.ColumnsIncompressible K s bcol → K.PerValueCut g → (k : ℝ) < bcol * V * (E + 1) →
    (∀ I ∈ 𝓘, C.InferenceBlock s k I) → C.Covers 𝓘 →
    4 * (d - E - (g - 1) : ℕ) * f * b * M.P.card ≤
      𝓘.card * (k * max (4 * (d - E - (g - 1) : ℕ) * b * aOut) ((c : ℝ) * f * k) +
        4 * (d - E - (g - 1) : ℕ) * f * b * k)

end Nci.Cover.Guarantees
