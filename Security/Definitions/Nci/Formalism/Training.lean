import Mathlib.Basic.NNReal.Basic
import Definitions.Core.Annotation
import Definitions.Nci.Formalism.Inference

/-!
# The training formalism v2, §3: the annotation, and the token kind A1 caps

An **annotation** (`Annotation`) of a declaration's training is an existential witness: the new slow weights `θ'`,
computed values, and the token blocks `𝒯`, an annotation of the circuit (`Verity.Annotation`) by token labels.
`Annotation.Valid` is what a check establishes of it: `θ'` is computed by gates, not inputs, and the token blocks are
pairwise disjoint sets of gates. That a token block is one token's computation in one step is what the annotation means,
and no check reads it. `⋃ 𝒯` is `tokenGates`, and `Δ`, the bits a decoder of at most `c` gates needs to describe `θ'`
given `θ`, is `totalUpdate` (a decoder `M` allowed `c` steps, `Verity.LDL`). `Δ_grad`, the gradient and optimizer state
passed between blocks, is what a reduction block's children hold (`ReductionBlock`), and `d` is the decoder's gates, the
room a block leaves it beside at least `c - d` gates of the computation (`SessionBlocks`).

A1 (`Nci.Formalism.Assumptions.CutCost`) prices an inference block `I` by where its work on each token begins. What it
reads is not defined from the circuit: its values are set by checked witness blocks, and a theorem reads upper bounds of
them, never best-known values.

* `e_T(I)` (`e I T`, in a type `E` of entry points): where `I`'s work on token `T` begins;
* `a_T(x)` (`a T x`, bits): the bits of `T`'s computed state at the entry `x`, the earlier tokens' keys and values that
  the work reads and doesn't compute among them;
* `u_I` (`u I`, bits): the update bits `I`'s work reads, counted once.

The token kind's work is a set `X` of token gates (te_core's: the token work no block computing its token from its
inputs reaches). Defined from it and A1's values:

* `W_T(x, ν)` (`entryWork`): the most gates of `X` in `T` that an inference block holds whose work on `T` begins at `x`
  and that reads at most `ν` update bits, the token work reachable from `x`;
* `cap` **bounds the token work within A1's budget** (`TokenCapAt`): at every entry profile `f` and update reads `ν`
  with `ν + Σ_T a_T(f_T) ≤ k`, `Σ_T W_T(f_T, ν) ≤ cap`;
* `r(ν)` **bounds the work per bit of state** (`WorkPerBit`): `W_T(x, ν) ≤ r(ν) a_T(x)`, and `M` **bounds the peak**
  of `(k - ν) r(ν)` (`PeakAtMost`), which together give the cap `M`, `(k - u) max (W / a)`.
-/

namespace Nci.Formalism

open Finset
open Verity (Circuit Declaration)
open scoped NNReal ENNReal

/-- **An annotation** of `d`'s training, tokens labelled by `L`: the new slow weights `θ'` and the token blocks `𝒯`. -/
structure Annotation (d : Declaration) (L : Type) where
  /-- `θ'`, the new slow weights: values computed by gates of `d`, the trained weights. -/
  θ' : Finset (Fin d.D.N)
  /-- The token blocks `𝒯`: token `T` holds the gates of one token's computation in one step. -/
  tokens : Verity.Annotation d.D L

namespace Annotation

variable {d : Declaration} {L : Type} (A : Annotation d L)

/-- **A valid annotation**: `θ'` is computed by gates, not inputs, and the token blocks are pairwise disjoint sets of
gates. -/
structure Valid : Prop where
  θ'_computed : A.θ' ⊆ d.D.computed
  tokens_computed : ∀ T, A.tokens.holds T ⊆ d.D.computed
  disjoint : Pairwise fun T T' => Disjoint (A.tokens.holds T) (A.tokens.holds T')

/-- `⋃ 𝒯`: the gates of the token blocks. -/
def tokenGates [Fintype L] : Finset (Fin d.D.N) := univ.biUnion A.tokens.holds

/-- **The total update** `Δ` on the run of `d` on the inputs `a`: the bits the decoder `M` needs to describe `θ'`'s
values given `θ`'s within `c` steps, the decoder model's form of a decoder of at most `c` gates. -/
noncomputable def totalUpdate (M : Verity.Decoder (List Bool)) (c : ℕ) (a : d.D.Transcript) : ℕ∞ :=
  Verity.LDL M c (d.D.bits (d.D.eval a) (d.θ.sort (· ≤ ·))) (d.D.bits (d.D.eval a) (A.θ'.sort (· ≤ ·)))

open Classical in
/-- **The token work from an entry** `W_T(x, ν)`: the most gates of `X` in token `T` that an inference block holds whose
work on `T` begins at `x` (`e I T = x`) and that reads at most `ν` update bits (`u I ≤ ν`). -/
noncomputable def entryWork (c k : ℕ) {E : Type} (e : Finset (Fin d.D.N) → L → E) (u : Finset (Fin d.D.N) → ℝ≥0)
    (X : Finset (Fin d.D.N)) (T : L) (x : E) (ν : ℝ≥0) : ℕ :=
  (univ.filter fun I => InferenceBlock d c k I ∧ e I T = x ∧ u I ≤ ν).sup fun I => (I ∩ A.tokens.holds T ∩ X).card

/-- **`cap` bounds the token work within A1's budget**: `X` is a set of token gates, and at every entry profile `f` and
update reads `ν` with `ν + Σ_T a_T(f_T) ≤ k`, the token work from those entries is at most `cap`,
`Σ_T W_T(f_T, ν) ≤ cap`. -/
def TokenCapAt [Fintype L] (c k : ℕ) {E : Type} (e : Finset (Fin d.D.N) → L → E) (a : L → E → ℝ≥0)
    (u : Finset (Fin d.D.N) → ℝ≥0) (X : Finset (Fin d.D.N)) (cap : ℝ≥0) : Prop :=
  X ⊆ A.tokenGates ∧
    ∀ (ν : ℝ≥0) (f : L → E), ν + ∑ T, a T (f T) ≤ k → ∑ T, (A.entryWork c k e u X T (f T) ν : ℝ≥0) ≤ cap

/-- **`r(ν)` bounds the token work per bit of state**: `X` is a set of token gates, and the token work from every entry
`x` of every token `T`, reading at most `ν` update bits, is at most `r(ν)` a bit of `T`'s state there,
`W_T(x, ν) ≤ r(ν) a_T(x)`. -/
def WorkPerBit [Fintype L] (c k : ℕ) {E : Type} (e : Finset (Fin d.D.N) → L → E) (a : L → E → ℝ≥0)
    (u : Finset (Fin d.D.N) → ℝ≥0) (X : Finset (Fin d.D.N)) (r : ℝ≥0 → ℝ≥0) : Prop :=
  X ⊆ A.tokenGates ∧ ∀ T x ν, (A.entryWork c k e u X T x ν : ℝ≥0) ≤ r ν * a T x

end Annotation

/-- **`M` bounds the peak** of `(k - ν) r(ν)` over the update reads `ν`: the token work a block reaches with the reads
`k - ν` its update bits leave, at `r(ν)` a bit. -/
def PeakAtMost (k : ℕ) (r : ℝ≥0 → ℝ≥0) (M : ℝ≥0) : Prop := ∀ ν : ℝ≥0, ((k : ℝ≥0) - ν) * r ν ≤ M

end Nci.Formalism
