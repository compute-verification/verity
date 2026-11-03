import Mathlib.Basic.NNReal.Basic
import Definitions.Core.Annotation
import Definitions.Nci.Formalism.Inference

/-!
# The training formalism v1, §3: the annotation and what it defines

An **annotation** (`Annotation`) of a declaration's training is an existential witness: the new slow weights `θ'` and
the token blocks `𝒯`, an annotation of the circuit (`Verity.Annotation`) by token labels. `Annotation.Valid` is what a
check establishes of it: `θ'` is computed by gates, not inputs, and the token blocks are pairwise disjoint sets of
gates. That a token block is one token's computation in one step is what the annotation means, and no check reads it.

Defined from the annotation, and computed from the circuit and it:

* `Dep` (`dep`), the gates that depend on some value of `θ'`; `T ∩ Dep` (`tokenDep`);
* the share `x_T(I) = |I ∩ T ∩ Dep| / |T ∩ Dep|` (`share`, `0` when `T` has no dependent gate), and its largest value
  over the tokens, `max_T x_T(I)` (`maxShare`);
* `I` enters `T` mid-path (`EntersMidPath`) if `x_T(I) > 0` and `I` reads a value computed by a gate of `T` outside
  `I`; `m(I)` (`midEntries`) is the number of tokens it enters so;
* the symbols of the table: `w`, the gates per token block (`work`, the largest); `s = c / w` (`tokensPerBlock`); `τ`,
  the unregistered inputs a token block reads (`dataPerToken`, the largest); `φ = |Dep ∩ ⋃ 𝒯| / |C|` (`depShare`);
  `Δ`, the bits a decoder of at most `c` gates needs to describe `θ'` given `θ` (`totalUpdate`, with a decoder `M`
  allowed `c` steps, `Verity.LDL`). `Δ_grad`, the gradient and optimizer state passed between blocks, is what a tree
  block's children's states hold (`TreeBlocks`).

A1's parameters, the update profile `u` and the cut cost `a`, are not defined from the annotation: they are what A1
assumes (`Nci.Formalism.Assumptions.CutCost`), reported as the best values known from witnesses. Shares are in `ℝ≥0`,
`u` maps a share to bits in `ℝ`.

The Slowdown theorem reads three properties of `u` and the tokens: `PeakBound u k M`, that `M` bounds
`max_x x (k - u(x))` over shares `x ∈ [0, 1]`; `BottomShare u k x₀`, that every share with `u(x) ≤ k` is at most `x₀`
(the bottom share is the largest such); and `TokensLinked A ε`, that in each token all but a share `ε` of its dependent
gates have, two by two, a common ancestor inside the token. Without some such link the Slowdown bound fails: a token
whose dependent work is two independent halves can be done by two blocks that enter it nowhere mid-path. Its special
cases read `u(x) = x u(1)` and `FrozenTail u k t`, that `u` is above `k` at every share beyond `t`.
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

open Classical in
/-- `Dep`, the dependent gates: the gates that depend on some value of `θ'`. -/
noncomputable def dep : Finset (Fin d.D.N) := d.D.computed.filter fun g => ∃ v ∈ A.θ', DependsOn d.D g v

/-- `T ∩ Dep`: token `T`'s dependent gates. -/
noncomputable def tokenDep (T : L) : Finset (Fin d.D.N) := A.tokens.holds T ∩ A.dep

/-- **The share** `x_T(I) = |I ∩ T ∩ Dep| / |T ∩ Dep|` of token `T`'s dependent work the gates `I` do. -/
noncomputable def share (T : L) (I : Finset (Fin d.D.N)) : ℝ≥0 :=
  ((I ∩ A.tokenDep T).card : ℝ≥0) / ((A.tokenDep T).card : ℝ≥0)

/-- `max_T x_T(I)`: the largest share of one token's dependent work the gates `I` do, `0` with no tokens. -/
noncomputable def maxShare [Fintype L] (I : Finset (Fin d.D.N)) : ℝ≥0 := univ.sup fun T => A.share T I

/-- **`I` enters `T` mid-path**: `x_T(I) > 0`, and `I` reads a value computed by a gate of `T` outside `I`. -/
def EntersMidPath (I : Finset (Fin d.D.N)) (T : L) : Prop :=
  0 < A.share T I ∧ ∃ v ∈ d.D.into I, v ∈ A.tokens.holds T

open Classical in
/-- `m(I)`: the number of tokens `I` enters mid-path. -/
noncomputable def midEntries [Fintype L] (I : Finset (Fin d.D.N)) : ℕ :=
  (univ.filter fun T => A.EntersMidPath I T).card

/-- `⋃ 𝒯`: the gates of the token blocks. -/
def tokenGates [Fintype L] : Finset (Fin d.D.N) := univ.biUnion A.tokens.holds

/-- **The dependent share** `φ = |Dep ∩ ⋃ 𝒯| / |C|`. -/
noncomputable def depShare [Fintype L] : ℝ := ((A.dep ∩ A.tokenGates).card : ℝ) / (d.D.computed.card : ℝ)

/-- **The work per token** `w`: the most gates a token block holds. -/
def work [Fintype L] : ℕ := univ.sup fun T => (A.tokens.holds T).card

/-- **The tokens per block** `s = c / w`. -/
noncomputable def tokensPerBlock [Fintype L] (c : ℕ) : ℝ := (c : ℝ) / (A.work : ℝ)

/-- **The data per token** `τ`: the most unregistered inputs a token block reads. -/
def dataPerToken [Fintype L] : ℕ := univ.sup fun T => (dataReads d (A.tokens.holds T)).card

/-- **The total update** `Δ` on the run of `d` on the inputs `a`: the bits the decoder `M` needs to describe `θ'`'s
values given `θ`'s within `c` steps, the decoder model's form of a decoder of at most `c` gates. -/
noncomputable def totalUpdate (M : Verity.Decoder (List Bool)) (c : ℕ) (a : d.D.Transcript) : ℕ∞ :=
  Verity.LDL M c (d.D.bits (d.D.eval a) (d.θ.sort (· ≤ ·))) (d.D.bits (d.D.eval a) (A.θ'.sort (· ≤ ·)))

/-- **The tokens are linked** up to a share `ε`: in each token `T`, all but at most `ε |T ∩ Dep|` of its dependent gates
have, two by two, a common ancestor inside `T`, a gate of `T` that feeds both through gates of `T`. -/
def TokensLinked (ε : ℝ) : Prop :=
  ∀ T, ∃ R ⊆ A.tokenDep T, (R.card : ℝ) ≤ ε * (A.tokenDep T).card ∧
    ∀ g₁ ∈ A.tokenDep T \ R, ∀ g₂ ∈ A.tokenDep T \ R, ∃ h,
      FeedsWithin d.D (A.tokens.holds T) h g₁ ∧ FeedsWithin d.D (A.tokens.holds T) h g₂

end Annotation

/-- **`M` bounds the peak** of `x (k - u(x))` over the shares `x ∈ [0, 1]`: Slowdown's `max_x x (k - u(x))`. -/
def PeakBound (u : ℝ≥0 → ℝ) (k : ℕ) (M : ℝ) : Prop := ∀ x : ℝ≥0, x ≤ 1 → (x : ℝ) * (k - u x) ≤ M

/-- **`x₀` bounds the bottom share**: every share `x ∈ [0, 1]` with `u(x) ≤ k` is at most `x₀`. -/
def BottomShare (u : ℝ≥0 → ℝ) (k : ℕ) (x₀ : ℝ) : Prop := ∀ x : ℝ≥0, x ≤ 1 → u x ≤ k → (x : ℝ) ≤ x₀

/-- **A frozen tail** of share `t`: `u` is nonnegative, and above `k` at every share beyond `t`: a block that does more
of a token's dependent work than its frozen tail must read the update. -/
def FrozenTail (u : ℝ≥0 → ℝ) (k : ℕ) (t : ℝ) : Prop :=
  (∀ x : ℝ≥0, 0 ≤ u x) ∧ ∀ x : ℝ≥0, t < x → x ≤ 1 → (k : ℝ) < u x

end Nci.Formalism
