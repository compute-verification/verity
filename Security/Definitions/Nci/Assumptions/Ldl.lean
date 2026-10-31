import Definitions.Nci.Ldl
import Mathlib.Basic.ENNReal.Real
import Mathlib.Basic.Real.ENatENNReal

/-!
# Local description length: the named premises

Two premises, each its own named `Prop` so that what each costs is visible. `EvaluatesBlocks` is the one capability of
the fixed decoder that Lemma 1 uses; `ActivationRate` is Corollary 4b's lower bound on the description length of each
block's activations.
-/

namespace Nci.Ldl.Assumptions

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
unregistered input bits, and the decoder must output the extraction for every run that takes `R`'s registered values
(`R.Honest`). So `ext` must read only `I ∪ in(I)` and the registered description of `I`, as the draft's extraction rule
does; for an `ext` that reads anything else no decoder meets it. -/
def EvaluatesBlocks (D : Decoder) (t : ℕ) (R : Committed) (ext : Fin R.d.n → R.d.D.Transcript → List Bool)
    (aBlk : ℕ) : Prop :=
  ∃ enc : Fin R.d.n → R.d.D.Transcript → List Bool, ∀ u X, R.Honest X →
    (∀ X', R.Honest X' → (∀ g ∈ R.d.D.into (R.d.𝓘.gates u) \ R.d.θ, X' g = X g) → enc u X' = enc u X) ∧
    (enc u X).length ≤ Nat.clog 2 R.d.n + novelReads R.d (R.d.𝓘.gates u) + aBlk ∧
    D (enc u X) t = some (ext u (R.d.𝓘.localEval X u))

/-- NEW PREMISE. **Activation rate** ([LDL-ACTIVATION-RATE]) `β` with deficit `Λ` (the draft's `β_x` and `λ_x`; Lean
reserves `λ`), within `t` steps: in the run `X`, every block's complete activation tuple `a_I = a u X`, the activation
values indexed by `V_I` with their labels, has `LDL_{t,R}(a_I) ≥ β |V_I| − Λ`.

Rationale: a block that computes many counted products whole must hold their activation operands; if those can't be
described in fewer than `β` bits each, given the registration, the block must receive them, or compute them from what
it receives, which its `k` unregistered input bits limit (Lemma 1).

It is a premise on the fixed decoder `D` and the run, not on the data alone: for some decoder (one that prints `a_I` on
the empty program) it is false. It is a joint bound over the collections a block can select, internally generated
activations and mixtures of layers and steps included; a measured code rate for one tensor is not such a bound. -/
def ActivationRate {ι A : Type} [DecidableEq A] (β Λ : ℝ) (t : ℕ) (D : Decoder) (R : Committed)
    (M : Products R.d.D ι A) (a : Fin R.d.n → R.d.D.Transcript → List Bool) (X : R.d.D.Transcript) : Prop :=
  ∀ u, ENNReal.ofReal (β * (M.acts R.d.𝓘 u).card - Λ) ≤ (LDL D t (a u X) : ℝ≥0∞)

end Nci.Ldl.Assumptions
