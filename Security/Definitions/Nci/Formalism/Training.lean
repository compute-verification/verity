import Mathlib.Basic.NNReal.Basic
import Definitions.Core.Annotation
import Definitions.Nci.Formalism.Inference

/-!
# The training formalism v2, §3: the annotation and what it defines

An **annotation** (`Annotation`) of a declaration's training is an existential witness: the new slow weights `θ'`,
computed values, and the token blocks `𝒯`, an annotation of the circuit (`Verity.Annotation`) by token labels.
`Annotation.Valid` is what a check establishes of it: `θ'` is computed by gates, not inputs, and the token blocks are
pairwise disjoint sets of gates. That a token block is one token's computation in one step is what the annotation means,
and no check reads it.

Defined from the annotation, and computed from the circuit and it:

* `A`, the affected gates (`affected`): the gates of `C` that depend on a value in `θ'`; `T ∩ A` (`tokenAffected`);
* `I` **computes `T` from its inputs** (`FromInputs`) when it reads no value computed by a gate of `T` outside `I`; the
  reachable gates `R ⊆ A` (`reachable`) are the affected gates of a token block `T` that lie in some inference block
  computing `T` from its inputs;
* the share `x_T(I) = |I ∩ T ∩ A| / |T ∩ A|` (`share`, `0` when `T` has no affected gate), and its largest value over the
  tokens, `max_T x_T(I)` (`maxShare`);
* `m(I)` (`cuts`), the number of token blocks `T` for which `I` contains a gate of `T ∩ A` and reads a computed value from
  `T`'s gates outside `I`;
* the tokens' worth of unreachable work done by `I`, `Σ_T |I ∩ T ∩ (A \ R)| / |T ∩ A|` (`unreachableWork`), and `s*`,
  its most over the inference blocks (`unreachableCapacity`);
* the symbols of the table: `w`, the gates per token block (`work`, the largest); `s = c / w` (`tokensPerBlock`); `σ`,
  the share of `C`'s gates that lie in token blocks (`tokenShare`); `τ`, the inputs of `C` outside `θ` that a token block
  reads (`dataPerToken`, the largest); `x₀`, the reachable share `|T ∩ R| / |T ∩ A|` (`reachableShare`); `Δ`, the bits a
  decoder of at most `c` gates needs to describe `θ'` given `θ` (`totalUpdate`, with a decoder `M` allowed `c` steps,
  `Verity.LDL`). `Δ_grad`, the gradient and optimizer state passed between blocks, is what a reduction block's children
  hold (`ReductionBlock`), and `d` is the decoder's gates, the room a block leaves it beside at least `c - d` gates of
  the computation (`SessionBlocks`).

The update profiles `u` and `u_in`, the cut cost `a` and the frozen share `t` are not defined from the annotation: they
are what A1 and A1′ assume (`Nci.Formalism.Assumptions.CutCost`, `Nci.Formalism.Assumptions.InputCost`), and a theorem
reads upper bounds of them, never best-known values. Shares and profiles are in `ℝ≥0`, `u` and `u_in` map a share to
bits.

The theorems read these properties: `Identical`, that every token block has the same `α > 0` affected gates;
`ReachableAtMost x₀`, that every reachable share is at most `x₀`; `PeakBound u k M`, that `M` bounds
`max_x x (k - u(x))` over shares `x ∈ [0, 1]`; `ReachLimit u_in k ξ`, that `ξ` bounds `max {x : u_in(x) ≤ k}`;
`ProfileAtMost ξ x₀`, that `x₀` bounds every token's dependency profile at `ξ` (`dependencyProfile`, the share of its
affected gates whose dependencies within it are at most a share `ξ` of its affected work); `FrozenShare u k t`, that `u`
is above `k` at every share beyond `t`.
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
/-- `A`, the affected gates: the gates of `C` that depend on a value in `θ'`. -/
noncomputable def affected : Finset (Fin d.D.N) := d.D.computed.filter fun g => ∃ v ∈ A.θ', DependsOn d.D g v

/-- `T ∩ A`: token `T`'s affected gates. -/
noncomputable def tokenAffected (T : L) : Finset (Fin d.D.N) := A.tokens.holds T ∩ A.affected

/-- **`I` computes `T` from its inputs**: it reads no value computed by a gate of `T` outside `I`. -/
def FromInputs (I : Finset (Fin d.D.N)) (T : L) : Prop := ∀ v ∈ d.D.into I, v ∉ A.tokens.holds T

open Classical in
/-- `R`, the reachable gates: the affected gates of a token block `T` that lie in some inference block computing `T`
from its inputs, reading no computed value from `T`'s gates outside it. -/
noncomputable def reachable (c k : ℕ) : Finset (Fin d.D.N) :=
  A.affected.filter fun g => ∃ T, g ∈ A.tokens.holds T ∧ ∃ I, InferenceBlock d c k I ∧ g ∈ I ∧ A.FromInputs I T

/-- **The share** `x_T(I) = |I ∩ T ∩ A| / |T ∩ A|` of token `T`'s affected work the gates `I` do. -/
noncomputable def share (T : L) (I : Finset (Fin d.D.N)) : ℝ≥0 :=
  ((I ∩ A.tokenAffected T).card : ℝ≥0) / ((A.tokenAffected T).card : ℝ≥0)

/-- `max_T x_T(I)`: the largest share of one token's affected work the gates `I` do, `0` with no tokens. -/
noncomputable def maxShare [Fintype L] (I : Finset (Fin d.D.N)) : ℝ≥0 := univ.sup fun T => A.share T I

open Classical in
/-- `m(I)`: the number of token blocks `T` for which `I` contains a gate of `T ∩ A` and reads a computed value from
`T`'s gates outside `I`. -/
noncomputable def cuts [Fintype L] (I : Finset (Fin d.D.N)) : ℕ :=
  (univ.filter fun T => (I ∩ A.tokenAffected T).Nonempty ∧ ¬ A.FromInputs I T).card

/-- **The unreachable work** `I` does, in tokens' worth: `Σ_T |I ∩ T ∩ (A \ R)| / |T ∩ A|`. -/
noncomputable def unreachableWork [Fintype L] (c k : ℕ) (I : Finset (Fin d.D.N)) : ℝ≥0 :=
  ∑ T, ((I ∩ (A.tokenAffected T \ A.reachable c k)).card : ℝ≥0) / ((A.tokenAffected T).card : ℝ≥0)

/-- `s*`: the most tokens' worth of unreachable work any inference block does. -/
noncomputable def unreachableCapacity [Fintype L] (c k : ℕ) : ℝ≥0 :=
  (univ.filter (InferenceBlock d c k)).sup (A.unreachableWork c k)

/-- **The reachable share** `|T ∩ R| / |T ∩ A|` of token `T`. -/
noncomputable def reachableShare (c k : ℕ) (T : L) : ℝ≥0 :=
  ((A.tokenAffected T ∩ A.reachable c k).card : ℝ≥0) / ((A.tokenAffected T).card : ℝ≥0)

/-- `⋃ 𝒯`: the gates of the token blocks. -/
def tokenGates [Fintype L] : Finset (Fin d.D.N) := univ.biUnion A.tokens.holds

/-- **The work per token** `w`: the most gates a token block holds. -/
def work [Fintype L] : ℕ := univ.sup fun T => (A.tokens.holds T).card

/-- **The tokens per block** `s = c / w`. -/
noncomputable def tokensPerBlock [Fintype L] (c : ℕ) : ℝ≥0 := (c : ℝ≥0) / (A.work : ℝ≥0)

/-- **The token share** `σ = |⋃ 𝒯| / |C|`: the share of `C`'s gates that lie in token blocks. -/
noncomputable def tokenShare [Fintype L] : ℝ≥0 := (A.tokenGates.card : ℝ≥0) / (d.D.computed.card : ℝ≥0)

/-- **The data per token** `τ`: the most inputs of `C` outside `θ` a token block reads. -/
def dataPerToken [Fintype L] : ℕ := univ.sup fun T => (dataReads d (A.tokens.holds T)).card

/-- **The largest affected count** `max_T |T ∩ A|`. -/
noncomputable def maxAffected [Fintype L] : ℕ := univ.sup fun T => (A.tokenAffected T).card

/-- **The total update** `Δ` on the run of `d` on the inputs `a`: the bits the decoder `M` needs to describe `θ'`'s
values given `θ`'s within `c` steps, the decoder model's form of a decoder of at most `c` gates. -/
noncomputable def totalUpdate (M : Verity.Decoder (List Bool)) (c : ℕ) (a : d.D.Transcript) : ℕ∞ :=
  Verity.LDL M c (d.D.bits (d.D.eval a) (d.θ.sort (· ≤ ·))) (d.D.bits (d.D.eval a) (A.θ'.sort (· ≤ ·)))

/-- **The token blocks are identical** in what the token form reads: each has the same `α > 0` affected gates. -/
def Identical (α : ℕ) : Prop := 0 < α ∧ ∀ T, (A.tokenAffected T).card = α

open Classical in
/-- **The dependencies of `g` in `T`**: the affected gates of `T` that feed `g` through gates of `T`, `g` itself when it
is affected. -/
noncomputable def dependencies (T : L) (g : Fin d.D.N) : Finset (Fin d.D.N) :=
  (A.tokenAffected T).filter fun h => FeedsWithin d.D (A.tokens.holds T) h g

/-- **The dependency share** of `g` in `T`: the share of `T`'s affected work a block that computes `T` from its inputs
does to compute `g`, at least. -/
noncomputable def dependencyShare (T : L) (g : Fin d.D.N) : ℝ≥0 :=
  ((A.dependencies T g).card : ℝ≥0) / ((A.tokenAffected T).card : ℝ≥0)

open Classical in
/-- **The dependency profile** `D_T(x)`: the share of `T`'s affected gates whose dependency share in `T` is at most `x`.
It is at most `x` when `T`'s affected gates are a chain, each depending on the one before. -/
noncomputable def dependencyProfile (T : L) (x : ℝ≥0) : ℝ≥0 :=
  (((A.tokenAffected T).filter fun g => A.dependencyShare T g ≤ x).card : ℝ≥0) / ((A.tokenAffected T).card : ℝ≥0)

/-- **`x₀` bounds the reachable share** of every token. -/
def ReachableAtMost (c k : ℕ) (x₀ : ℝ≥0) : Prop := ∀ T, A.reachableShare c k T ≤ x₀

/-- **`x₀` bounds the dependency profile** of every token at `ξ`. -/
def ProfileAtMost (ξ x₀ : ℝ≥0) : Prop := ∀ T, A.dependencyProfile T ξ ≤ x₀

end Annotation

/-- **`M` bounds the peak** of `x (k - u(x))` over the shares `x ∈ [0, 1]`: `max_x x (k - u(x)) ≤ M`. -/
def PeakBound (u : ℝ≥0 → ℝ≥0) (k : ℕ) (M : ℝ≥0) : Prop := ∀ x : ℝ≥0, x ≤ 1 → x * ((k : ℝ≥0) - u x) ≤ M

/-- **`ξ` bounds the reach** of `u_in` within `k`: every share `x ∈ [0, 1]` with `u_in(x) ≤ k` is at most `ξ`, so
`max {x : u_in(x) ≤ k} ≤ ξ`. -/
def ReachLimit (u_in : ℝ≥0 → ℝ≥0) (k : ℕ) (ξ : ℝ≥0) : Prop := ∀ x : ℝ≥0, x ≤ 1 → u_in x ≤ k → x ≤ ξ

/-- **A frozen share** `t`: `u` is above `k` at every share beyond `t`, so a block that does more of a token's affected
work than the frozen layers' must read the changed layers' update. -/
def FrozenShare (u : ℝ≥0 → ℝ≥0) (k : ℕ) (t : ℝ≥0) : Prop := ∀ x : ℝ≥0, t < x → x ≤ 1 → (k : ℝ≥0) < u x

end Nci.Formalism
