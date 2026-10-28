import Definitions.Pouw.Assumptions.Dimension.Pipe

/-!
# A2 on the checked words: extra outputs do not matter (trusted statements)

A2 matches only the outputs in its target set, the checked words
(`wordSet`). A real program may also output functions the model prices above their real cost (triple products at
8 each, FP32 sums of a tensor-core word); the unrestricted clause made those refute A2 below `γ₂ ≈ 1/2`.

* `A2ExtraOutputs`: adding outputs outside the target set to a machine keeps A2, at the same `γ₂`.
* `A2WitnessExtra`: the quadratic closed-pipe machine whose every program also outputs every function that is not a
  checked word satisfies `A2(0)` into `QuadPipe` for the checked words, while no program, of any class and at any
  cost, has all of its outputs among its slots: the unrestricted clause fails for it.
* `A2Separation`: the same in A2's own terms. With `EmptyQuadPipe` the machine has a program, so it fails the
  unrestricted A2 (`Set.univ`) at every `γ₂` and into every class, and it meets the restricted one.
-/

namespace Pouw.Dimension.A2Words

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions

/-- A machine with extra outputs: each program also outputs the functions in `X`, at the same cost. -/
def withExtra {G : Type} (M : WordCostModel G) (X : Set (G → ℚ)) : WordCostModel G where
  Prog := M.Prog
  cost := M.cost
  outputs q := M.outputs q ∪ X

/-- The quadratic closed-pipe machine at FP32 price `cs` whose every program also outputs every function of the noise
outside `W`. -/
def extraMachine (N m n k cs : ℕ) (W : Set (Noise N m n k → ℚ)) : WordCostModel (Noise N m n k) :=
  withExtra (quadMachine N m n k cs) Wᶜ

/-! ## Pinned statements -/

/-- **Extra outputs do not matter**: if `M` satisfies `A2(γ₂)` into `Alg` for the target set `W`, then so does
`M` with any extra outputs outside `W`. -/
def A2ExtraOutputs : Prop :=
  ∀ (N m n k : ℕ) (M : WordCostModel (Noise N m n k)) (Alg : List (Instr (Noise N m n k)) → Prop)
    (W X : Set (Noise N m n k → ℚ)) (γ₂ : ℝ),
    Disjoint X W → A2_algebraicSufficiency γ₂ M Alg W → A2_algebraicSufficiency γ₂ (withExtra M X) Alg W

/-- **A2's witness with extra outputs**: for every instance, `extraMachine` for the non-final checked words satisfies
`A2(0)` into `QuadPipe cs` for those words, and for each of its programs no program, of any class and at any cost, has
every output of it among its slots. -/
def A2WitnessExtra : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) (cs : ℕ),
    A2_algebraicSufficiency 0 (extraMachine N m n k cs (wordSet (word k m n N R A B) U)) (QuadPipe cs)
      (wordSet (word k m n N R A B) U) ∧
    ∀ q : (extraMachine N m n k cs (wordSet (word k m n N R A B) U)).Prog,
      ¬ ∃ (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
        ∀ φ ∈ (extraMachine N m n k cs (wordSet (word k m n N R A B) U)).outputs q, φ ∈ slots P L

/-- **The empty program is a quadratic closed-pipe program**, at every FP32 price `cs`: it has no free instruction
and its pipe is empty. So `quadMachine` and `extraMachine` have a program. -/
def EmptyQuadPipe : Prop := ∀ (N m n k cs : ℕ), QuadPipe cs ([] : List (Instr (Noise N m n k)))

/-- **The two forms of A2 separate**: for every instance, `extraMachine` for the non-final checked
words satisfies the restricted `A2(0)` into `QuadPipe cs` for those words, and it fails the unrestricted A2 (target
set `Set.univ`: every output must be a slot) at every `γ₂` and into every class. -/
def A2Separation : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) (cs : ℕ),
    A2_algebraicSufficiency 0 (extraMachine N m n k cs (wordSet (word k m n N R A B) U)) (QuadPipe cs)
      (wordSet (word k m n N R A B) U) ∧
    ∀ (γ₂ : ℝ) (Alg : List (Instr (Noise N m n k)) → Prop),
      ¬ A2_algebraicSufficiency γ₂ (extraMachine N m n k cs (wordSet (word k m n N R A B) U)) Alg Set.univ

end Pouw.Dimension.A2Words
