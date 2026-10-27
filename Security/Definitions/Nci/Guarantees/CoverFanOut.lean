import Definitions.Nci.Assumptions.Cover
import Definitions.Nci.Assumptions.CoverFanOut

/-!
# The cover theorem's fan-in/fan-out tightening: the guarantees

The covering bound (`Nci.Guarantees.CoveringBound`) charges a block for the weights and the tokens of the
multiplications it completes. Fan-in and fan-out charge it per value it reads. A block that reads a value of a token finishes, through it, only
the counted products whose activations key back to that value: at most `aOut` (Kimi K2: 37,248, the MoE input). To do
more of a token's work without reading more of it, the block must hold one of the token's accumulation chains whole,
which costs `d − E` reads: the chain's `d` legs, less the `E` embedding lookups it can afford.

* `FanOutIntensity`, `FanOutCoverBound`: under `UpdateHardness`, `ColumnsIncompressible` and `Keyed`, a block finishes
  at most `k · max(aOut, c k / ((d − E) b)) + k` counted products (Kimi K2: a slowdown of about 8,900).
* `FanOutReadIntensity`, `FanOutReadCoverBound`: under `ValueReadHardness` (each spent value costs `f` bits, from the
  same bits as the parameters) in place of `UpdateHardness`, at most `k · max(aOut / f, c k / (4 (d − E) f b)) + k`
  (Kimi K2 at `f = 4`: about 3.6·10⁴).

Kimi K2's numbers are for its forward products (`c = 1`). With Dgrad counted, on the forward's tokens (`c = 2`,
`α₀ = 0.301`, and `Keyed`'s position-0 dq_rope cut), they are about 1.66·10⁴ and 7.3·10⁴; keeping those dq_rope
products at `aOut = 49152` instead gives 1.39·10⁴ and 5.5·10⁴.

The statements are division-free. The comparison with the covering bound at two costs
(`Nci.Guarantees.CoveringBoundTwoCosts`, about 166 for Kimi K2 at `c_w = 2.5`, `c_a = 4`) and its additive form
(`Nci.Cover.Guarantees.CoveringBoundAdditive`, about 660) is of numbers, not theorems: the premises differ (`Keyed` in
place of activation-chaining, and `ColumnsIncompressible` added; `FanOutIntensity` beats the additive bound at
`c_w = b`, `c_a = f` when `4 b f aOut ≤ k` and `4 f c ≤ d − E`, and `FanOutReadIntensity` beats it when `4 b aOut ≤ k`
and `c ≤ d − E`, which Kimi K2 meets with margin), and `Keyed`'s counted products leave out the LM head's forward
products, which the covering bound's may count but `α₀` leaves out.

Scope, `Keyed`'s for Kimi K2: the counted set excludes the LM head's forward products and layer 0's q_a and kv_a
products, whose activations reach more than `aOut` products or key to an embedding gather. It doesn't count norm-γ or
NVFP4 block-scale products, which would raise the MoE input's reach past `aOut`. Copies, capacity slots, and
router-scaled or re-quantized versions of a value are gates of that value (no two values share a key). A lookup is
exactly its gather's cone from the column's entries, updated at that step, so a frozen embedding has no `Keyed`. Each
counted product's output gate reads all `f` bits of its activation.
-/

namespace Nci.Cover.Guarantees

/-- **Fan-out intensity**: under `UpdateHardness`, `ColumnsIncompressible` and a keyed token structure (`Keyed`) whose
roots have `d` legs, with `k < bcol · V · (E + 1)` and `E < d`, an inference block finishes at most
`k · max(aOut, c k / ((d − E) b)) + k` counted products: `(d − E) b · #(finished) ≤ k · max((d − E) b aOut, c k) +
(d − E) b k`.

A block reads at most `k` values, at least one input gate each, and holds at most `E` lookups of a token. A token with
fewer than `d − E` spent values has no held root, so each of its products the block computes whole is charged to the
first value it reads on its activation's key path: at most `aOut` per value. Each token whose root the block holds
spends at least `d − E` values, so there are at most `k / (d − E)` of them, each with at most `c` products per
parameter, over at most `k / b` parameters. Products the block finishes without computing them whole are at most `k`,
as in `Nci.Guarantees.CoveringBound`'s proof.

Kimi K2 (`d = 512`, `aOut = 37248`, `c = 1`, `V = 163840`, `b = 2.5`, `bcol = 1`, so `E = 122`; `k = 2·10⁷`,
`s = 4.52·10¹⁶`, `α₀ = 0.147`): `7.45·10¹¹` products per block, a slowdown of about 8,900. Scope (`Keyed`): the
counted set excludes the LM head's forward products and layer 0's q_a and kv_a products, and norm-γ and NVFP4
block-scale products are not counted. With Dgrad on the forward's tokens (`c = 2`, `α₀ = 0.301`, position 0's
dq_rope cut): about 1.66·10⁴, set by the `c k` term (41,026 per block, above `aOut`). `UpdateHardness` needs drift and
fails under LoRA. -/
def FanOutIntensity : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Val] (M : MulBlocks C J L)
    (d aOut c V f : ℕ) (K : Keyed M Tok Val d aOut c V f) (s k E : ℕ) (b bcol : ℝ),
    0 < b → Assumptions.UpdateHardness M s b → Assumptions.ColumnsIncompressible K s bcol →
    (k : ℝ) < bcol * V * (E + 1) → E < d →
    ∀ I, C.InferenceBlock s k I →
      (d - E : ℕ) * b * (M.finished I).card ≤ k * max ((d - E : ℕ) * b * aOut) ((c : ℝ) * k) + (d - E : ℕ) * b * k

/-- **Fan-out cover bound**: under `FanOutIntensity`'s premises, a cover of `C` by inference blocks has
`#𝓘 ≥ #P / (k · max(aOut, c k / ((d − E) b)) + k)` blocks:
`(d − E) b · #P ≤ #𝓘 · (k · max((d − E) b aOut, c k) + (d − E) b k)`. Kimi K2: a slowdown of about 8,900 (1.66·10⁴
with Dgrad on the forward's tokens), with `FanOutIntensity`'s scope: `#P` excludes the LM head's forward products and
layer 0's q_a and kv_a products, and no norm-γ or NVFP4 block-scale product. -/
def FanOutCoverBound : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Val] (M : MulBlocks C J L)
    (d aOut c V f : ℕ) (K : Keyed M Tok Val d aOut c V f) (s k E : ℕ) (b bcol : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
    0 < b → Assumptions.UpdateHardness M s b → Assumptions.ColumnsIncompressible K s bcol →
    (k : ℝ) < bcol * V * (E + 1) → E < d →
    (∀ I ∈ 𝓘, C.InferenceBlock s k I) → C.Covers 𝓘 →
    (d - E : ℕ) * b * M.P.card ≤ 𝓘.card * (k * max ((d - E : ℕ) * b * aOut) ((c : ℝ) * k) + (d - E : ℕ) * b * k)

/-- **Fan-out intensity, `f` bits per value**: under `ValueReadHardness` (which implies `UpdateHardness` for counted
products), `ColumnsIncompressible` and `Keyed`, with `k < bcol · V · (E + 1)` and `E < d`, an inference block finishes
at most `k · max(aOut / f, c k / (4 (d − E) f b)) + k` counted products:
`4 (d − E) f b · #(finished) ≤ k · max(4 (d − E) b aOut, c k) + 4 (d − E) f b k`.

The argument is `FanOutIntensity`'s, with the values paid `f` bits each from the same `k` as the parameters. Spending
`X` bits on low tokens' values and `Y` on high tokens', with `P` on parameters and `X + Y + P ≤ k`, the block computes
at most `(aOut / f) X + c (P / b) (Y / (f (d − E)))` products whole. AM–GM bounds this by `k` times the larger of its
endpoints.

Kimi K2 (`FanOutIntensity`'s numbers, `f = 4`): `1.86·10¹¹` products per block, a slowdown of about 3.6·10⁴. It
reaches `k · aOut / f` once `d − E ≥ c k / (4 aOut b) ≈ 54`. Kimi K2 has 390, so the bound is set by fan-out alone, and
A2 built a block reaching 98.5% of it. With Dgrad on the forward's tokens (`c = 2`, `α₀ = 0.301`, position 0's
dq_rope cut): about 7.3·10⁴, still `k · aOut / f` (the threshold is 107); at `aOut = 49152` instead of the cut,
5.5·10⁴. Scope (`Keyed`): `FanOutIntensity`'s, and each counted product's output gate reads all `f` bits of its
activation. -/
def FanOutReadIntensity : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Val] (M : MulBlocks C J L)
    (d aOut c V f : ℕ) (K : Keyed M Tok Val d aOut c V f) (s k E : ℕ) (b bcol : ℝ),
    0 < b → Assumptions.ValueReadHardness K s b → Assumptions.ColumnsIncompressible K s bcol →
    (k : ℝ) < bcol * V * (E + 1) → E < d →
    ∀ I, C.InferenceBlock s k I →
      4 * (d - E : ℕ) * f * b * (M.finished I).card ≤
        k * max (4 * (d - E : ℕ) * b * aOut) ((c : ℝ) * k) + 4 * (d - E : ℕ) * f * b * k

/-- **Fan-out cover bound, `f` bits per value**: under `FanOutReadIntensity`'s premises, a cover of `C` by inference
blocks has `#𝓘 ≥ #P / (k · max(aOut / f, c k / (4 (d − E) f b)) + k)` blocks:
`4 (d − E) f b · #P ≤ #𝓘 · (k · max(4 (d − E) b aOut, c k) + 4 (d − E) f b k)`. Kimi K2: a slowdown of about
3.6·10⁴ (7.3·10⁴ with Dgrad on the forward's tokens), with `FanOutReadIntensity`'s scope. -/
def FanOutReadCoverBound : Prop :=
  ∀ (C : Circuit) {J L Tok Val : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Val] (M : MulBlocks C J L)
    (d aOut c V f : ℕ) (K : Keyed M Tok Val d aOut c V f) (s k E : ℕ) (b bcol : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
    0 < b → Assumptions.ValueReadHardness K s b → Assumptions.ColumnsIncompressible K s bcol →
    (k : ℝ) < bcol * V * (E + 1) → E < d →
    (∀ I ∈ 𝓘, C.InferenceBlock s k I) → C.Covers 𝓘 →
    4 * (d - E : ℕ) * f * b * M.P.card ≤
      𝓘.card * (k * max (4 * (d - E : ℕ) * b * aOut) ((c : ℝ) * k) + 4 * (d - E : ℕ) * f * b * k)

end Nci.Cover.Guarantees
