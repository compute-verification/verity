module

public import Mathlib.Basic.NNReal.Basic
public import Definitions.Core.Annotation
public import Definitions.Nci.Formalism.Inference

/-!
# The training formalism v2, §3: the annotation, A1's values, and the token kind's work

An **annotation** (`Annotation`) of a declaration's training is an existential witness: the new slow weights `θ'`,
computed values, and the token blocks `𝒯`, an annotation of the circuit (`Verity.Annotation`) by token labels.
`Annotation.Valid` is what a check establishes of it: `θ'` is computed by gates, not inputs, and the token blocks are
pairwise disjoint sets of gates. That a token block is one token's computation in one step is what the annotation means,
and no check reads it. `⋃ 𝒯` is `tokenGates`, and `Δ`, the bits a decoder of at most `c` gates needs to describe `θ'`
given `θ`, is `totalUpdate` (a decoder `M` allowed `c` steps, `Verity.LDL`). `Δ_grad`, the gradient and optimizer state
passed between blocks, is what a reduction block's children hold (`ReductionBlock`); `Δ_t`, the weights in the form the
update rule reads and writes (a straight-through run's latents; `Δ_t = Δ` for any other run), is what an update
application block reads beside the reduced state (`UpdateBlock`); and `d` is the decoder's gates, the room a block
leaves it beside at least `c - d` gates of the computation (`SessionBlocks`).

A1 (`Nci.Formalism.Assumptions.CutCost`) prices an inference block `I` by where its work on each token begins, at values
a finite family `𝒲` of witness blocks of the circuit sets. What a set of gates `S` reads beyond `θ` is `in(S) \ θ`, as
the policy counts it. A witness block **enters** token `T` at `x` (`Enters`) when its gates are of `𝒲`, its work is on
`T` alone, and that work begins at `x`. What it reads is `T`'s own state, the gates of `T`'s block it reads and doesn't
compute (`stateReads`), and the rest, which the tokens of a block share and the block pays for once:

* `e_T(I)` (`e I T`, in a type `E` of entry points): where `I`'s work on token `T` begins, a labelling of blocks given
  beside `𝒲`;
* `a_T(x)` (`stateAt`, bits): the least state bits of `T` any witness block entering `T` at `x` reads, and `0` where
  none enters;
* `R_T(x)` (`readsAt`): the reads every witness block entering `T` at `x` makes, and none where none enters; `u_I`
  (`blockShared`, bits) is the shared bits `I`'s work reads, counted once:
  `|⋃_T R_T(e_T(I)) \ I \ ⋃_T (R_T(e_T(I)) ∩ T)|` over the tokens `I` works on (`worksOn`), what their witness blocks
  read beyond the tokens' own state, net of what `I` computes. They are `θ'`, or the encoding it is decoded from (LoRA
  factors, a seed log, a code), the update's reads, and the shared context and the earlier tokens' keys and values.

So the values are reads of blocks of the circuit, not free parameters, and a checker has only to produce `𝒲` and `e`. A
witness works on one token, so its reads price one token's entry, and a block pays once for what its tokens share and
not at all for what it computes.

The token kind's work is a set `X` of token gates (training's slowdown code's: the token work no block computing its
token from its inputs reaches). Defined from it and A1's values:

* `W_T(x, ν)` (`entryWork`): the most gates of `X` in `T` that an inference block holds whose work on `T` begins at `x`
  and whose shared bits `u_I` are at most `ν`, the token work reachable from `x`;
* `r(ν)` **bounds the work per bit of state** (`WorkPerBit`): `W_T(x, ν) ≤ r(ν) a_T(x)`, and `M` **bounds the peak**
  of `(k - ν) r(ν)` (`PeakAtMost`), which together give the cap `M`, `(k - u) max (W / a)`.

That a cap bounds the token work within A1's budget, `Σ_{T ∈ S} W_T(f_T, ν) ≤ cap` wherever
`ν + Σ_{T ∈ S} a_T(f_T) ≤ k`, is the named assumption `training-token-work`
(`Nci.Formalism.Assumptions.TokenWork`): A1 prices a block's reads and says nothing of how much token work they reach.
-/

@[expose] public section

namespace Nci.Formalism

open Finset
open Verity (Circuit RegisteredCircuit)
open scoped NNReal ENNReal

/-- **An annotation** of `d`'s training, tokens labelled by `L`: the new slow weights `θ'` and the token blocks `𝒯`. -/
structure Annotation (d : RegisteredCircuit) (L : Type) where
  /-- `θ'`, the new slow weights: values computed by gates of `d`, the trained weights. -/
  θ' : Finset (Fin d.D.N)
  /-- The token blocks `𝒯`: token `T` holds the gates of one token's computation in one step. -/
  tokens : Verity.Annotation d.D L

namespace Annotation

variable {d : RegisteredCircuit} {L : Type} (A : Annotation d L)

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
noncomputable def totalUpdate (M : Verity.Decoder (List Bool)) (c : ℕ) (a : d.D.Trace) : ℕ∞ :=
  Verity.LDL M c (d.D.bits (d.D.eval a) (d.θ.sort (· ≤ ·))) (d.D.bits (d.D.eval a) (A.θ'.sort (· ≤ ·)))

/-- **The state bits** of `T` that `S` reads, `|in(S) \ θ ∩ T|`: the gates of `T`'s token block it reads and doesn't
compute, `T`'s state where `S`'s work on it begins. -/
def stateReads (S : Finset (Fin d.D.N)) (T : L) : ℕ := (d.unregistered S ∩ A.tokens.holds T).card

/-- **The tokens `I` works on**: those whose token block holds a gate of `I`. -/
def worksOn [Fintype L] (I : Finset (Fin d.D.N)) : Finset L := univ.filter fun T => (I ∩ A.tokens.holds T).Nonempty

/-- **A witness block entering `T` at `x`**: a set of gates `W` of the family `𝒲` whose work is on token `T` alone and
begins at `x`, `e W T = x`. -/
def Enters {E : Type} (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) (T : L) (x : E)
    (W : Finset (Fin d.D.N)) : Prop :=
  W ∈ 𝒲 ∧ GateSet d.D W ∧ (W ∩ A.tokens.holds T).Nonempty ∧ (∀ T', T' ≠ T → Disjoint W (A.tokens.holds T')) ∧
    e W T = x

/-- **`a_T(x)`**: the least state bits of `T` any witness block of `𝒲` entering `T` at `x` reads, and `0` when none
does. -/
noncomputable def stateAt {E : Type} (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) (T : L)
    (x : E) : ℕ :=
  sInf ((A.stateReads · T) '' {W | A.Enters e 𝒲 T x W})

open Classical in
/-- **`R_T(x)`**: the reads beyond `θ` every witness block of `𝒲` entering `T` at `x` makes, `⋂_W in(W) \ θ`, and none
when none does. -/
noncomputable def readsAt {E : Type} (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) (T : L)
    (x : E) : Finset (Fin d.D.N) :=
  univ.filter fun g => (∃ W, A.Enters e 𝒲 T x W) ∧ ∀ W, A.Enters e 𝒲 T x W → g ∈ d.unregistered W

/-- **`u_I`**: the shared bits `I`'s work reads, counted once, `|⋃_T R_T(e_T(I)) \ I \ ⋃_T (R_T(e_T(I)) ∩ T)|` over the
tokens `I` works on: what their witness blocks read beyond the tokens' own state, net of what `I` computes. -/
noncomputable def blockShared [Fintype L] {E : Type} (e : Finset (Fin d.D.N) → L → E)
    (𝒲 : Finset (Finset (Fin d.D.N))) (I : Finset (Fin d.D.N)) : ℕ :=
  ((((A.worksOn I).biUnion fun T => A.readsAt e 𝒲 T (e I T)) \ I) \
    (A.worksOn I).biUnion fun T => A.readsAt e 𝒲 T (e I T) ∩ A.tokens.holds T).card

open Classical in
/-- **The token work from an entry** `W_T(x, ν)`: the most gates of `X` in token `T` that an inference block holds whose
work on `T` begins at `x` (`e I T = x`) and whose shared bits are at most `ν` (`u_I ≤ ν`). -/
noncomputable def entryWork [Fintype L] (c k : ℕ) {E : Type} (e : Finset (Fin d.D.N) → L → E)
    (𝒲 : Finset (Finset (Fin d.D.N))) (X : Finset (Fin d.D.N)) (T : L) (x : E) (ν : ℕ) : ℕ :=
  (univ.filter fun I => InferenceBlock d c k I ∧ e I T = x ∧ A.blockShared e 𝒲 I ≤ ν).sup
    fun I => (I ∩ A.tokens.holds T ∩ X).card

/-- **`r(ν)` bounds the token work per bit of state**: `X` is a set of token gates, and the token work from every entry
`x` of every token `T`, at shared bits at most `ν`, is at most `r(ν)` a bit of `T`'s state there,
`W_T(x, ν) ≤ r(ν) a_T(x)`. -/
def WorkPerBit [Fintype L] (c k : ℕ) {E : Type} (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N)))
    (X : Finset (Fin d.D.N)) (r : ℕ → ℝ≥0) : Prop :=
  X ⊆ A.tokenGates ∧ ∀ T x ν, (A.entryWork c k e 𝒲 X T x ν : ℝ≥0) ≤ r ν * A.stateAt e 𝒲 T x

end Annotation

/-- **`M` bounds the peak** of `(k - ν) r(ν)` over the shared bits `ν`: the token work a block reaches with the reads
`k - ν` its shared bits leave, at `r(ν)` a bit. -/
def PeakAtMost (k : ℕ) (r : ℕ → ℝ≥0) (M : ℝ≥0) : Prop := ∀ ν : ℕ, ((k - ν : ℕ) : ℝ≥0) * r ν ≤ M

end Nci.Formalism
