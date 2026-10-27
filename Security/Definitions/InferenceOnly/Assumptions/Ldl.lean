import Definitions.InferenceOnly.Ldl
import Mathlib.Basic.ENNReal.Real
import Mathlib.Basic.Real.ENatENNReal

/-!
# Local description length: the named premises

Three premises, each its own named `Prop` so that what each costs is visible. `EvaluatesBlocks` is the one capability of
the fixed decoder that Lemma 1 uses; `ActivationRate` is Corollary 4b's lower bound on the description length of each
block's activations; `HeldRate` is the pair bound's (`Guarantees.PairBound`), charging only what a block must receive:
the weights its counted products read and the entries they descend from, each once.
-/

namespace InferenceOnly.Ldl.Assumptions

open Verity
open scoped ENNReal

/-- NEW PREMISE. **The decoder evaluates blocks** ([LDL-EVALUATES-BLOCKS]) for the extraction rule `ext` within `t`
steps at program overhead `aBlk`: there is an encoding `enc u X` of block `u`'s index and its unregistered input bits
in run `X` (it reads `X` only at `in(I) \ θ`), at most `⌈log₂ B⌉ + |in(I) \ θ| + aBlk` bits long, from which `D`
outputs, within `t`, what `ext` extracts from the block evaluated from those bits (`Partition.localEval`: the block's
gates recomputed in order, every other gate read from `X`).

This is the draft's "a fixed evaluator performs that extraction within time `t` for every block; let `a_blk` be its
program-encoding overhead", and its proof's program: the block index in `⌈log₂ B⌉` bits, then the values of
`in(I) \ θ` in registered order. The decoder knows `C`, `𝓘` and the registered values from `R`, so it evaluates the
block and applies `ext`. Its time is the evaluator's: the premise is that `t` covers it.

It is stated as a capability, not as a bound on `LDL`: the program depends on the run only through the block's
unregistered input bits, and the decoder must output the extraction for every run consistent with `R`. So `ext` must
read only `I ∪ in(I)` and the registered description of `I`, as the draft's extraction rule does; for an `ext` that
reads anything else no decoder meets it. -/
def EvaluatesBlocks {d : Declaration} {n : ℕ} (D : Decoder) (t : ℕ) (R : Registration d n)
    (ext : Fin n → d.D.Transcript → List Bool) (aBlk : ℕ) : Prop :=
  ∃ enc : Fin n → d.D.Transcript → List Bool, ∀ u X, R.Consistent X →
    (∀ X', R.Consistent X' → (∀ g ∈ d.D.into (R.P.gates u) \ d.θ, X' g = X g) → enc u X' = enc u X) ∧
    (enc u X).length ≤ Nat.clog 2 n + novelReads d (R.P.gates u) + aBlk ∧
    D (enc u X) t = some (ext u (R.P.localEval X u))

/-- NEW PREMISE. **Activation rate** ([LDL-ACTIVATION-RATE]) `β` with deficit `Λ` (the draft's `β_x` and `λ_x`; Lean
reserves `λ`), within `t` steps: in the run `X`, every block's complete activation tuple `a_I = a u X`, the activation
values indexed by `V_I` with their labels, has `LDL_{t,R}(a_I) ≥ β |V_I| − Λ`.

Rationale: a block that computes many counted products whole must hold their activation operands; if those can't be
described in fewer than `β` bits each, given the registration, the block must receive them, or compute them from what
it receives, which its `k` unregistered input bits limit (Lemma 1).

It is a premise on the fixed decoder `D` and the run, not on the data alone: for some decoder (one that prints `a_I` on
the empty program) it is false. It is a joint bound over the collections a block can select, internally generated
activations and mixtures of layers and steps included; a measured code rate for one tensor is not such a bound.

It fails once a block can hold a layer's update: the block then computes the layer's internal activations from the
entries it receives, so `|V_I|` outgrows `(k' + Λ) / β` while `LDL_{t,R}(a_I) ≤ k'` (Lemma 1). At the onsite reference's
serving block (`k_S ≈ 3.2·10¹¹` bits), one holding two layers of DeepSeek-V4-Pro's update and the entries of `3·10⁷`
tokens at `β_x` has 18 times the activations it allows (the held-layers attack; `activationRate_recompute` is it in
miniature). `HeldRate` is what survives: it charges the held weights and the entries, not the activations computed from them. -/
def ActivationRate {d : Declaration} {n : ℕ} {ι A : Type} [DecidableEq A] (β Λ : ℝ) (t : ℕ) (D : Decoder)
    (R : Registration d n) (M : Products d.D ι A) (a : Fin n → d.D.Transcript → List Bool) (X : d.D.Transcript) :
    Prop :=
  ∀ u, ENNReal.ofReal (β * (M.acts R.P u).card - Λ) ≤ (LDL D t (a u X) : ℝ≥0∞)

/-- NEW PREMISE. **Held-weight and entry rate** ([LDL-HELD-RATE]) `βw` per held weight and `βe` per entry, with deficit
`ℓ`, within `t` steps: in the run `X`, every block's tuple `a_I = a u X`, the values of the weights its counted
occurrences read and of the entries they descend from, has `LDL_{t,R}(a_I) ≥ βw |W_I| + βe |S_I| − ℓ`. Here
`W_I = wt(𝓙_I)` is the set of weights block `u`'s counted occurrences read and `S_I = src(𝓙_I)` the set of entries
(a token's activation where it enters the block) they descend from, each counted once however many occurrences read
it.

Rationale: a block that computes a counted product must know its weight and its token's activation. An activation it
computes inside itself, from an entry and weights it holds, costs it nothing beyond them, so the premise prices only
those two: an updated weight at `βw`, what it carries beyond the registration (`UpdateHardness`'s `b`), and an entry at
`βe`, what one token's crossing carries beyond the registered baseline's activations (E1's `β_x d ≤ h_tok + F`). It is
`ActivationRate` with the internally computed activations free, which a block that holds whole layers gets
(`ActivationRate`'s docstring).

Its counted set must leave out the products whose entries the block itself computes from registered values: a block
holding the first layers' updates computes their inputs from token ids, so for those products the entries cost
nothing and the premise is false (benchmarks/rho's `expensive_share`). It is a premise on the fixed decoder `D` and the
run: for a decoder that prints `a_I` on the empty program it is false. -/
def HeldRate {d : Declaration} {n : ℕ} {ι A W S : Type} [DecidableEq W] [DecidableEq S] (βw βe ℓ : ℝ) (t : ℕ)
    (D : Decoder) (R : Registration d n) (M : Products d.D ι A) (wt : ι → W) (src : ι → S)
    (a : Fin n → d.D.Transcript → List Bool) (X : d.D.Transcript) : Prop :=
  ∀ u, ENNReal.ofReal (βw * ((M.inBlock R.P u).image wt).card + βe * ((M.inBlock R.P u).image src).card - ℓ) ≤
    (LDL D t (a u X) : ℝ≥0∞)

end InferenceOnly.Ldl.Assumptions
