import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.Real.Sqrt
import Definitions.Nci.Assumptions.Cover
import Definitions.Nci.Assumptions.CoverFanOut
import Definitions.Nci.Assumptions.CoverPartialSum

/-!
# The cover theorem's partial-sum tightening: the guarantees

The covering bound (`Nci.Guarantees.CoveringBound`) and the fan-out bounds count a block's products as a
rectangle, tokens (or values) times parameters, which is `k²`-sized. A matrix multiplication's products are a cube (`Cube`): output row, input column and token, whose shadows
are weights, activations and output entries. By Loomis–Whitney a block that uses `p` weights and `Q` activations at
tokens it reads little of, and contributes to `Y` entries there, computes at most `√(p Q Y)` products there, so a block
handed `k` bits does about `k^1.5` products, provided its entries cost reads too. They do, across a cover:

1. `PartialSumsRead` (B2's Lemma 1, no premise): an entry whose products' outputs are finished in `m` blocks of a cover
   costs `m − 1` gates of its accumulation among the blocks' inputs. A partial sum computed in one block and added in
   another is a gate the second block is handed.
2. `PartialSumIntensity`: under `UpdateHardness` and `ActivationReads` (B2's (X)), a block finishes at most
   `k² / (b f a) + k √(Y / (b f)) + k` counted products, with `Y` its crossings (`Cube.crossings`): the `k^1.5` form.
3. `PartialSumCoverBound`: with the crossings charged by `PartialSumsRead` (`Σ Y ≤ 3 Σ` accumulation reads, so
   `b_p = 2/3` bit per crossing), the reads weighted by `μ` and `1 − μ`, and the fan-out face (`aOut` products per
   activation), `r(μ) · (#P − k #𝓘) ≤ 2 k #𝓘` for every `μ ∈ [0, 1]`, with `r = psRate`.
4. `KeyedActivationReads`: at `f = 1`, `ActivationReads` is a theorem of `Keyed` with entries of width `n`, at
   `a = min(d − E, n)` under `ColumnsIncompressible`; `PartialSumKeyedCoverBound` is the cover bound resting on
   structure, `UpdateHardness` and `ColumnsIncompressible` only.

Kimi K2's forward products (`k = 2·10⁷`, `s = 4.52·10¹⁶`, `α₀ = 0.147`, `b = 2.5`, `aOut = 37248`, `d = n = 512`), with
`ρ ≥ α₀ (s / k) · r / (r + 2)`: at `f = 1` and `a = 390` (`bcol = 1`, `E = 122`), `r = 9.6·10⁻⁵` (at `μ ≈ 0.97`) and
`ρ ≈ 1.6·10⁴`, 1.8 times `FanOutCoverBound`'s 8,900; at `a = 464` (`E = 48`), `1.9·10⁴`. Under `ActivationReads` at
`f = 4`, `6.1·10⁴` at `a = 390`. The gain grows with `a`: `r` saturates at `1.84 √(b f b_p / (2 k))` once
`a ≳ 0.82 √(2 k b_p / (b f))` (about 2,700 at `f = 1`), which needs entries wider than kv_b's 512 (B2, blue1).

The figures for this tier assume `ConditionalUpdateHardness` over `Keyed`'s value gates in place of
`UpdateHardness`, which no guarantee here states. The weights' bits are then paid outside the values' gates, the read
pool is `k`, and `ρ ≥ α₀ (s / k) · r / (r + 1)` with `r = psRate b f a aOut k μ`. At `f = 1` that is about 6.3·10⁴ at
`a = 390`, 7.4·10⁴ at `a = 464` (`E = 48`, at `bcol = 2.5`) and 8.1·10⁴ at `a = 512` (`E` deducted only where root walks
reach lookups). Under `ActivationReads` at `f = 4` it is 2.3·10⁵, 2.6·10⁵ and 2.8·10⁵ at those `a`. Each figure needs
every partial sum on the books: dark reads of about `3·10⁻⁴` of the baseline halve the gain over fan-out, and at `10⁻³`
only the fan-out faces remain, 8,900 at `f = 1` and 3.6·10⁴ at `f = 4` (red5).

Scope. These are counts in the cover model's units: Boolean gates, with `s` and `α₀` counted in them too (red2: the
brief's FLOP-based `s` and `α₀` scale every tier by `R = g_serve / g_train`). The accumulations must be in the covered
circuit: adders a cover leaves out (dark gates) lose the crossings' charge, and a dark fraction of about `10⁻⁴` to
`10⁻³` erases the gain over fan-out (B3). Pools overlap under `UpdateHardness`, so weights, activations and partial sums
are each paid from the block's whole `k`, which costs the factor 2 in `2 k`.
-/

namespace Nci.Cover.Guarantees

/-- **The partial-sum rate** at weight `μ ∈ [0, 1]`, for `b` bits per updated parameter, `f` per activation, entry
width `a`, `aOut` products per activation and read pool `K` (`2k` here): `max {min {c₁, c₂}, μ f / Λ}` with
`c₁ = (1 + √μ)² b f a / K`, `c₂ = 2 (1 + √μ) √(1 − μ) √(b f / (3 K))` and `Λ = max {aOut, K / (4 b a)}`. It is B2's
`r(μ)` at `b_p = 2/3`: `c₂ = √2 (1 + √μ) √(1 − μ) √(b f b_p / K)`. `c₁` prices the tokens whose entries a block can
finish, `c₂` the Loomis–Whitney products at the others, and `μ f / Λ` the fan-out face. -/
noncomputable def psRate (b f a aOut K μ : ℝ) : ℝ :=
  max (min ((1 + √μ) ^ 2 * (b * f * a / K)) (2 * (1 + √μ) * √(1 - μ) * √(b * f / (3 * K))))
    (μ * f / max aOut (K / (4 * b * a)))

/-- **Partial sums are read** (B2's Lemma 1): for any cover `𝓘` of `C` there is an assignment `φ` of the counted
products to blocks of `𝓘`, each finishing its product (computing its output gate), such that the entries' spreads, less
one each, add up to at most the accumulation gates among the blocks' inputs:
`Σ_E (#φ(products of E) − 1) ≤ Σ_{I ∈ 𝓘} #(inputs I ∩ accGates)`.

No premise: it is the wiring. Label each accumulation from its top down: a block computing a gate has all of the gate's
predecessors, so a predecessor in the accumulation either is computed by that block and keeps its label, or is one of
its inputs and takes a fresh one. Each fresh label is a gate of the accumulation that a block is handed, and the
accumulations are disjoint. Overlapping blocks are allowed, and nothing bounds a block's reads. -/
def PartialSumsRead : Prop :=
  ∀ (C : Circuit) {J L Tok Val Mat Row Col : Type} [DecidableEq J] [DecidableEq Tok] [DecidableEq Val]
    [DecidableEq Mat] [DecidableEq Row] (M : MulBlocks C J L) (d aOut c V w : ℕ) (K : Keyed M Tok Val d aOut c V w)
    (Q : Cube K Mat Row Col) (𝓘 : Finset (Finset (Fin C.N))),
    C.Covers 𝓘 → ∃ φ : J × L → Finset (Fin C.N), (∀ p ∈ M.P, φ p ∈ 𝓘 ∧ M.o p ∈ C.computed (φ p)) ∧
      ∑ E ∈ M.P.image Q.entry, (((M.P.filter fun p => Q.entry p = E).image φ).card - 1) ≤
        ∑ I ∈ 𝓘, (C.inputs I ∩ Q.accGates).card

/-- **Partial-sum intensity** (the `k^1.5` form): under `UpdateHardness` and `ActivationReads` at `f` bits per
activation and entry width `a`, an inference block finishes at most `k² / (b f a) + k √(Y / (b f)) + k` counted
products, with `Y` its crossings at rate `f a` (`Cube.crossings`): `b f a · #(finished) ≤ k² + a k √(b f Y) + b f a k`.

The products it computes whole at tokens it reads at least `f a` gates of number at most one per token and weight
(`cell_inj`): at most `(k / b) (k / (f a))`. The others, by Loomis–Whitney on their cells (matrix and row, column,
token), at most `√(p Q Y)` with `b p ≤ k` (`UpdateHardness`) and `f Q ≤ k` (`ActivationReads`). Products it
finishes without computing them whole are at most `k`, as in `Nci.Guarantees.CoveringBound`'s proof.

A block's crossings are paid by the blocks that finish their entries (`PartialSumsRead`), so this bounds a block on its
own only together with its crossings; `PartialSumCoverBound` sums it over a cover. -/
def PartialSumIntensity : Prop :=
  ∀ (C : Circuit) {J L Tok Val Mat Row Col : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Tok]
    [DecidableEq Val] [DecidableEq Mat] [DecidableEq Row] (M : MulBlocks C J L) (d aOut c V w : ℕ)
    (K : Keyed M Tok Val d aOut c V w) (Q : Cube K Mat Row Col) (s k f a : ℕ) (b : ℝ),
    0 < b → Assumptions.UpdateHardness M s b → Assumptions.ActivationReads Q s k f a →
    ∀ I, C.InferenceBlock s k I →
      b * f * a * (M.finished I).card ≤ k * k + a * k * √(b * f * (Q.crossings I f a).card) + b * f * a * k

/-- **Partial-sum cover bound** (B2's Theorem 2, blue1's T0): under `UpdateHardness` and `ActivationReads` at `f` bits
per activation and entry width `a ≥ 1`, a cover of `C` by inference blocks has, for every `μ ∈ [0, 1]`,
`r(μ) · (#P − k #𝓘) ≤ 2 k #𝓘` with `r = psRate b f a aOut (2k) μ`: at least `#P r / (k (r + 2))` blocks.

Assign each counted product to a block that finishes it (`PartialSumsRead`). Those it doesn't compute whole cost an
accumulation gate each, at most `k` per block. Per block, the whole ones obey `PartialSumIntensity`'s Loomis–Whitney
count and the fan-out count (`aOut` per activation, `Keyed.reach`). Two read counts bound the blocks' totals: weights
and activations (`2k` per block), and weights and crossings, at `1/3` gate each (`2k` per block: `Σ Y ≤ 3 Σ`
accumulation reads, by `PartialSumsRead` and the crossing entries not being whole). They share the activations' reads,
so they are combined with weights `μ` and `1 − μ`, and AM–GM per block gives `r(μ)` (the draft's `iu_reads_ge`,
`column_le`).

Kimi K2 at `f = 4` (`ActivationReads` as a premise) and `a = 390`: `ρ ≈ 6.1·10⁴` (module docstring). -/
def PartialSumCoverBound : Prop :=
  ∀ (C : Circuit) {J L Tok Val Mat Row Col : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Tok]
    [DecidableEq Val] [DecidableEq Mat] [DecidableEq Row] (M : MulBlocks C J L) (d aOut c V w : ℕ)
    (K : Keyed M Tok Val d aOut c V w) (Q : Cube K Mat Row Col) (s k f a : ℕ) (b μ : ℝ)
    (𝓘 : Finset (Finset (Fin C.N))),
    0 < b → 0 < a → 0 ≤ μ → μ ≤ 1 → Assumptions.UpdateHardness M s b → Assumptions.ActivationReads Q s k f a →
    (∀ I ∈ 𝓘, C.InferenceBlock s k I) → C.Covers 𝓘 →
    psRate b f a aOut (2 * k) μ * (M.P.card - k * 𝓘.card) ≤ 2 * k * 𝓘.card

/-- **Activation reads from the keyed structure** (blue3's per-value charge): under `ColumnsIncompressible` with
`k < bcol · V · (E + 1)` and entries of width `n` (`Cube.Wide`), `ActivationReads` holds at one gate per activation
with `a = min(d − E, n)`.

A block handed fewer than `a` gates of token `t` holds no root of `t`: a held root costs `d` spent values and held
lookups of `t` (`rootCost`), at most `E` of them lookups (`lookupCost`). So from each activation of `t` it uses, the
walk along keys reaches a value of `t` it reads (`walk`; a held stop would be a lookup, excluded by `reach_lookups`, or
a root). Distinct activations reach distinct values, since their key paths are disjoint (`Keyed.KeyDisjoint`, which
`Keyed`'s (I) and (F) imply), each with a gate among the block's inputs. An entry of `t` whole in the block would have
`n ≥ a` distinct activations (`cell_inj`, `stack`). -/
def KeyedActivationReads : Prop :=
  ∀ (C : Circuit) {J L Tok Val Mat Row Col : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Tok]
    [DecidableEq Val] [DecidableEq Mat] [DecidableEq Row] (M : MulBlocks C J L) (d aOut c V w : ℕ)
    (K : Keyed M Tok Val d aOut c V w) (Q : Cube K Mat Row Col) (s k E n : ℕ) (bcol : ℝ),
    Assumptions.ColumnsIncompressible K s bcol → (k : ℝ) < bcol * V * (E + 1) → Q.Wide n →
    Assumptions.ActivationReads Q s k 1 (min (d - E) n)

/-- **Partial-sum cover bound from the keyed structure**: under `UpdateHardness` and `ColumnsIncompressible` (the
premises of `FanOutCoverBound`), with `k < bcol · V · (E + 1)`, `E < d` and entries of width `n ≥ 1`, a cover of `C`
by inference blocks has `r(μ) · (#P − k #𝓘) ≤ 2 k #𝓘` for every `μ ∈ [0, 1]`, with
`r = psRate b 1 (min(d − E, n)) aOut (2k) μ`. It is `PartialSumCoverBound` at `f = 1` with `KeyedActivationReads`.

Kimi K2 (`a = 390` at `bcol = 1`): `ρ ≈ 1.6·10⁴`, 1.8 times `FanOutCoverBound`'s 8,900, on the same premises plus
the structure `Wide`. -/
def PartialSumKeyedCoverBound : Prop :=
  ∀ (C : Circuit) {J L Tok Val Mat Row Col : Type} [DecidableEq J] [DecidableEq L] [DecidableEq Tok]
    [DecidableEq Val] [DecidableEq Mat] [DecidableEq Row] (M : MulBlocks C J L) (d aOut c V w : ℕ)
    (K : Keyed M Tok Val d aOut c V w) (Q : Cube K Mat Row Col) (s k E n : ℕ) (b bcol μ : ℝ)
    (𝓘 : Finset (Finset (Fin C.N))),
    0 < b → 0 ≤ μ → μ ≤ 1 → Assumptions.UpdateHardness M s b → Assumptions.ColumnsIncompressible K s bcol →
    (k : ℝ) < bcol * V * (E + 1) → E < d → 0 < n → Q.Wide n →
    (∀ I ∈ 𝓘, C.InferenceBlock s k I) → C.Covers 𝓘 →
    psRate b 1 (min (d - E) n : ℕ) aOut (2 * k) μ * (M.P.card - k * 𝓘.card) ≤ 2 * k * 𝓘.card

end Nci.Cover.Guarantees
