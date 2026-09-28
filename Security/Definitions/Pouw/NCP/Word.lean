import Definitions.Pouw.Basic.Prob
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# R_word-NCP: straight-line word programs over a unit's checked positions (trusted)

The analogue of R_copy (`Pouw.Copy`, tt-attacks §6) for TT_NCP, priced by tt-attacks §2's per-word floor. Unit `u < N`
has `n u` checked positions in block-major order, and the true value at position `p` is `X ω u p`, a function of the
noise `ω` (uniform over a finite type). Before the salt (after seeing the workload) the adversary fixes, for every
position, one rule:

* `fresh`: computed, 16 units (one k16 mma or INT32 word);
* `copy s a`: `±x_a` for an earlier position `a`, 0 units;
* `sum s a t b`: `±x_a ± x_b` for earlier positions `a`, `b`, 8 units (the FP32 route);
* `const c`: the constant `c`, 0 units.

The unit is correct iff every produced value equals the true one. With rules that read earlier positions (`Valid`),
that is, by induction on `p`, the local condition `Correct`: every non-fresh rule holds at the true values.

`NzNCP` is the noise hypothesis: rules that are not identities (do not hold for every noise value) hold with
probability at most `1/2` each, independently. `identCount` is `F_u`, the number of positions whose rule is an
identity. `FinalOnly` bounds `F_u`; for NCP with an admissible `R` the expected bound is one identity per output entry,
which is not proved here: it is a named hypothesis, and `WordFinalWitness` only shows it is satisfiable.

The model is non-adaptive (the rules are fixed before the salt) and excludes shared tables. It is a witness that a
TT_NCP-shaped bound can hold because of the noise, not evidence for TT_NCP.
-/

namespace Pouw.NCP.Word

open Finset

/-- `sgn true = 1`, `sgn false = −1`. -/
def sgn : Bool → ℤ
  | true => 1
  | false => -1

/-- How one checked position is produced. -/
inductive Rule
  | fresh
  | copy (s : Bool) (a : ℕ)
  | sum (s : Bool) (a : ℕ) (t : Bool) (b : ℕ)
  | const (c : ℤ)
  deriving DecidableEq

/-- tt-attacks §2's per-word prices. -/
def Rule.cost : Rule → ℕ
  | .fresh => 16
  | .copy _ _ => 0
  | .sum _ _ _ _ => 8
  | .const _ => 0

/-- The value a non-fresh rule produces from the values `x` at its operand positions. -/
def Rule.target (x : ℕ → ℤ) : Rule → ℤ
  | .fresh => 0
  | .copy s a => sgn s * x a
  | .sum s a t b => sgn s * x a + sgn t * x b
  | .const c => c

/-- The rule reads only positions before `p`. -/
def Rule.Reads (p : ℕ) : Rule → Prop
  | .fresh => True
  | .copy _ a => a < p
  | .sum _ a _ b => a < p ∧ b < p
  | .const _ => True

/-- A word-program adversary: position `p` of unit `u` is produced by `rule u p`. -/
structure Adversary where
  rule : ℕ → ℕ → Rule

/-- Unit `u`'s cost. -/
def unitCost (n : ℕ → ℕ) (A : Adversary) (u : ℕ) : ℕ := ∑ p ∈ range (n u), (A.rule u p).cost

/-- The adversary's cost over units `u < N`. -/
def cost (n : ℕ → ℕ) (N : ℕ) (A : Adversary) : ℕ := ∑ u ∈ range N, unitCost n A u

/-- Straight-line: every rule reads only earlier positions. -/
def Valid (n : ℕ → ℕ) (A : Adversary) : Prop := ∀ u, ∀ p < n u, (A.rule u p).Reads p

variable {Ω : Type}

/-- Rule `r` holds at position `p` of the values `x`. -/
def Holds (r : Rule) (x : ℕ → ℤ) (p : ℕ) : Prop := r = .fresh ∨ x p = r.target x

/-- Correct on `u`: every rule holds at the true values. -/
def Correct (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (A : Adversary) (ω : Ω) (u : ℕ) : Prop :=
  ∀ p < n u, Holds (A.rule u p) (X ω u) p

open Classical in
/-- The units `u < N` it gets right. -/
noncomputable def correctSet (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (N : ℕ) (A : Adversary) (ω : Ω) : Finset ℕ :=
  (range N).filter (Correct n X A ω)

/-- Position `p` of unit `u` uses an identity: its rule is not fresh and holds for every noise value. -/
def Identity (X : Ω → ℕ → ℕ → ℤ) (A : Adversary) (u p : ℕ) : Prop :=
  A.rule u p ≠ .fresh ∧ ∀ ω, X ω u p = (A.rule u p).target (X ω u)

open Classical in
/-- `F_u`: the number of unit `u`'s positions whose rule is an identity. -/
noncomputable def identCount (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (A : Adversary) (u : ℕ) : ℕ :=
  ((range (n u)).filter (Identity X A u)).card

/-- **NzNCP**: for every set `P` of a unit's positions and rules `r p` that are not fresh, read earlier positions and
are not identities, all of `P`'s rules hold with probability at most `2^−|P|`. -/
def NzNCP [Fintype Ω] (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) : Prop :=
  ∀ (u : ℕ) (P : Finset ℕ), P ⊆ range (n u) → ∀ r : ℕ → Rule,
    (∀ p ∈ P, r p ≠ .fresh ∧ (r p).Reads p ∧ ¬ ∀ ω, X ω u p = (r p).target (X ω u)) →
    pr (fun ω => ∀ p ∈ P, X ω u p = (r p).target (X ω u)) ≤ (1 / 2 : ℝ) ^ P.card

/-- **FinalOnly** (named hypothesis): unit `u` has at most `f u` identity positions. -/
def FinalOnly (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (A : Adversary) (f : ℕ → ℕ) : Prop :=
  ∀ u, identCount n X A u ≤ f u

/-- Values uniform and independent on `{0, 1}`: position `p < M` reads bit `p` of `ω`. -/
def bits {M : ℕ} (ω : Fin M → Bool) (p : ℕ) : ℤ := if h : p < M then ((ω ⟨p, h⟩).toNat : ℤ) else 0

/-- The all-constant adversary for values `x` that do not depend on the noise. -/
def constAdv (x : ℕ → ℕ → ℤ) : Adversary := ⟨fun u p => .const (x u p)⟩

/-! ## Pinned statements -/

/-- **R_word-NCP's bound**: `Pr[16·Σ_{u ∈ K'} n_u > cost/(1 − γ₀)] ≤ Σ_u 2^−(γ₀·n_u − F_u)`. -/
def WordBound : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (N : ℕ) (A : Adversary) (γ₀ : ℝ),
    0 ≤ γ₀ → γ₀ < 1 → Valid n A → NzNCP n X →
    pr (fun ω => (cost n N A : ℝ) / (1 - γ₀) < 16 * ∑ u ∈ correctSet n X N A ω, (n u : ℝ)) ≤
      ∑ u ∈ range N, (2 : ℝ) ^ (-(γ₀ * n u - identCount n X A u))

/-- **The floor**: `c` output entries per unit, `w` checked words per entry (`w = 3k/16` for NCP), and at most one
identity per entry. The bound is `N·2^−(c·(γ₀·w − 1))`, below 1 per unit iff `γ₀ > 1/w = 16/(3k)`. -/
def WordFloor : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (N : ℕ) (A : Adversary) (γ₀ : ℝ) (w c : ℕ),
    0 ≤ γ₀ → γ₀ < 1 → Valid n A → NzNCP n X → (∀ u, n u = w * c) → FinalOnly n X A (fun _ => c) →
    pr (fun ω => (cost n N A : ℝ) / (1 - γ₀) < 16 * ∑ u ∈ correctSet n X N A ω, (n u : ℝ)) ≤
      N * (2 : ℝ) ^ (-(c * (γ₀ * w - 1)))

/-- **NzNCP is satisfiable**: independent uniform bits, `M` positions per unit. -/
def WordNzWitness : Prop :=
  ∀ M : ℕ, NzNCP (fun _ => M) (fun (ω : Fin M → Bool) (_ : ℕ) => bits ω)

/-- **FinalOnly is satisfiable**: on independent uniform bits no straight-line rule is an identity. -/
def WordFinalWitness : Prop :=
  ∀ (M : ℕ) (A : Adversary), Valid (fun _ => M) A →
    FinalOnly (fun _ => M) (fun (ω : Fin M → Bool) (_ : ℕ) => bits ω) A (fun _ => 0)

/-- **Zero noise**: values that do not depend on the noise satisfy NzNCP vacuously, but every rule of the
all-constant adversary is an identity (`F_u = n_u`), it costs 0, and it wins with probability 1. -/
def WordZeroNoise : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (n : ℕ → ℕ) (x : ℕ → ℕ → ℤ) (N : ℕ) (γ₀ : ℝ),
    γ₀ < 1 → (∃ u < N, 0 < n u) →
    NzNCP n (fun (_ : Ω) => x) ∧ cost n N (constAdv x) = 0 ∧
      (∀ u, identCount n (fun (_ : Ω) => x) (constAdv x) u = n u) ∧
      pr (fun ω : Ω => (cost n N (constAdv x) : ℝ) / (1 - γ₀) <
        16 * ∑ u ∈ correctSet n (fun (_ : Ω) => x) N (constAdv x) ω, (n u : ℝ)) = 1

end Pouw.NCP.Word
