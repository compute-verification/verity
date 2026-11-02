import Definitions.Pouw.Assumptions.Dimension.Forest

/-!
# The open quadratic pipe (trusted statements; staged)

`ClosedPipe` forbids an FP32 step to read an instruction priced at 16 or more: a real program that adds FP32 terms
to a tensor-core word had to be rebuilt as a closed pipe, which is the tensor-core-junk case. The **open**
pipe allows it. An FP32 step `a·b + c` may take any earlier priced value (a tensor-core word, an INT result, a
priced load or an FP32 output) as its addend `c`, or as its factor `b` when `a` is a constant. It stays quadratic: a
product of two non-constant factors multiplies two free values (`QuadStep` over all earlier priced values).

* **Prices.** Every priced instruction is **big** (price at least `pb`, any function of the noise) or an **FP32
  step** (price at least `cs`, a quadratic step over the earlier priced values). W1 is `pb = 16`, `cs = 8`. There
  are no free instructions.
* **The count.** `forestBound` with the big values as the steps outside `F`: a big value counts as two FP32 steps,
  whatever reads it. So a program that outputs the `D` non-final checked words costs at least `min(pb, 2·cs)·D`,
  which is `16·(1 − γ)·D` with `γ = 1 − min(pb, 2·cs)/16`. At W1's prices `γ = 0`: Theorem D's `16·D` with no A1
  (`AlgBoundOpenQuad8`).
* **The chain.** A2 into `OpenQuadPipe 16 8` follows from A2 into `QuadPipe 8` (`A2QuadToOpen`), and it gives
  TT_NCP (`TTNCPOpenQuadOfA2`), with a consistency witness at the tight `γ₀ = 16/(3k)`.
* **On the checked words the class does not matter** (`A2WordsIff`): under checkpoint independence and MinRank ≥ 5,
  A2 into either pipe is the per-word cost `(1 − γ₂)·16` itself. The open pipe widens the class the model bound
  covers (`OpenPipeWitness`: a program that feeds a tensor-core word into an FFMA, which no closed pipe is); it does
  not weaken A2 on the words.
-/

namespace Pouw.Dimension.OpenPipe

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.Dimension.Forest

/-- The open pipe's rule, given the earlier priced values `Vs`: no free instruction, and each priced instruction is
big (price at least `pb`) or an FP32 step (price at least `cs`) that is a `QuadStep` over `Vs`. Its value then joins
`Vs`. -/
def OpenOK {N m n k : ℕ} (pb cs : ℕ) : List (Noise N m n k → ℚ) → List (Instr (Noise N m n k)) → Prop
  | _, [] => True
  | _, .free _ _ :: _ => False
  | Vs, .priced p f :: P => (pb ≤ p ∨ (cs ≤ p ∧ QuadStep Vs f)) ∧ OpenOK pb cs (Vs ++ [f]) P

/-- **The open quadratic pipe** at big price `pb` and FP32 price `cs`. -/
def OpenQuadPipe {N m n k : ℕ} (pb cs : ℕ) (P : List (Instr (Noise N m n k))) : Prop := OpenOK pb cs [] P

/-- The open-pipe AlgM_4090 as a machine: its programs are the `OpenQuadPipe pb cs` programs on the noise
coordinates, at their AlgM cost, outputting their slots. -/
def openMachine (N m n k pb cs : ℕ) : Machine (Noise N m n k) where
  Prog := {P : List (Instr (Noise N m n k)) // OpenQuadPipe pb cs P}
  cost P := (cost P.1 : ℝ)
  outputs P := {φ | φ ∈ slots P.1 (coords N m n k)}

/-! ## Pinned statements -/

/-- **The open pipe contains the quadratic closed pipe**: every `QuadPipe cs` program is an `OpenQuadPipe 16 cs`
one. -/
def QuadToOpen : Prop :=
  ∀ (N m n k cs : ℕ) (P : List (Instr (Noise N m n k))), QuadPipe cs P → OpenQuadPipe 16 cs P

/-- **A2 into the open pipe follows from A2 into the closed one**: A2 into `QuadPipe cs` gives A2 into
`OpenQuadPipe 16 cs`, at the same `γ₂` and target set. On the checked words, under checkpoint independence and MinRank
≥ 5, the converse holds as well at `cs = 8` (`A2WordsIff`). -/
def A2QuadToOpen : Prop :=
  ∀ (N m n k cs : ℕ) (M : Machine (Noise N m n k)) (W : Set (Noise N m n k → ℚ)) (γ₂ : ℝ),
    A2_algebraicSufficiency γ₂ M (QuadPipe cs) W → A2_algebraicSufficiency γ₂ M (OpenQuadPipe 16 cs) W

/-- The open-pipe count for a word family: every `OpenQuadPipe pb cs` program on inputs in `V₀` whose slots include
the family's values at `S ⊆ U × [0, nonFinal)` costs at least `min(pb, 2·cs)·|S|`. -/
def OpenCount {N m n k : ℕ} (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ) : Prop :=
  ∀ (pb cs : ℕ) (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))), (∀ f ∈ L, f ∈ V0 N m n k) →
    OpenQuadPipe pb cs P → ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, w x.1 x.2 ∈ slots P L) →
    min pb (2 * cs) * S.card ≤ cost P

/-- **Theorem 1 in the open quadratic pipe, for any word family**: `Theorem1Family`'s hypotheses (the two-atom
property and independence modulo `V₀`) give the open-pipe count. -/
def Theorem1Open : Prop :=
  ∀ (N m n k : ℕ) (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ),
    TwoAtomFree (familyQ w U) (atoms N m n k) → FamilyIndepQ w U → OpenCount w U

/-- **Theorem D in the open quadratic pipe** (no A1): under checkpoint independence and MinRank ≥ 5, every
`OpenQuadPipe pb cs` program on inputs in `V₀` that outputs the non-final checked words of the units in `U` costs at
least `min(pb, 2·cs)·D`, with `D = |U|·m·n·(3k/16 − 1)`: `16·(1 − γ)·D` at `γ = 1 − min(pb, 2·cs)/16`. -/
def AlgBoundOpenQuad : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (pb cs : ℕ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    OpenQuadPipe pb cs P → Outputs k m n N R A B U P L →
    min pb (2 * cs) * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **Theorem D in the open quadratic pipe at W1's prices** (`pb = 16`, `cs = 8`): `16·D`, with `γ = 0`. -/
def AlgBoundOpenQuad8 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    OpenQuadPipe 16 8 P → Outputs k m n N R A B U P L →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **TT_NCP from A2 into the open quadratic pipe** (no A1): if a machine satisfies `A2(γ₂)` into
`OpenQuadPipe 16 8` for the non-final checked words, `R` is checkpoint independent with MinRank ≥ 5, one of the
machine's programs outputs those words, and `1 − γ₀ ≤ (1 − γ₂)·(1 − 16/(3k))`, then TT_NCP's event fails for that
program. -/
def TTNCPOpenQuadOfA2 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (M : Machine (Noise N m n k)) (q : M.Prog) (γ₀ γ₂ : ℝ), 16 ∣ k → 0 < k →
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    A2_algebraicSufficiency γ₂ M (OpenQuadPipe 16 8) (wordSet (word k m n N R A B) U) →
    (∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ M.outputs q) →
    γ₂ < 1 → γ₀ < 1 → 1 - γ₀ ≤ (1 - γ₂) * (1 - 16 / (3 * k : ℝ)) →
    ¬ (M.cost q / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- **`TTNCPOpenQuadOfA2`'s hypotheses hold together**, at `γ₂ = 0` and the tight `γ₀ = 16/(3k)`: for every
`k ≥ 1024` with `16 ∣ k`, every shape, `A = g(F₁)`, `B` and unit set, one `R` and one program of `openMachine` at W1's
prices satisfy all of them. -/
def OpenQuadChainConsistent : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N → ∃ R : Mat, CheckpointIndependent k 16 R ∧ MinRankGe k R 5 ∧
      A2_algebraicSufficiency 0 (openMachine N m n k 16 8) (OpenQuadPipe 16 8) (wordSet (word k m n N R A B) U) ∧
      (∃ q : (openMachine N m n k 16 8).Prog,
        ∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ (openMachine N m n k 16 8).outputs q) ∧
      (0 : ℝ) < 1 ∧ 16 / (3 * k : ℝ) < 1 ∧
      1 - 16 / (3 * k : ℝ) = (1 - 0) * (1 - 16 / (3 * k : ℝ))

/-- **The per-word cost**: every program of `M` that outputs the values of a word family `w` at `S ⊆ U × [0,
nonFinal)` costs at least `(1 − γ₂)·16·|S|`. -/
def PerWordCost (γ₂ : ℝ) {N m n k : ℕ} (M : Machine (Noise N m n k)) (w : ℕ → ℕ → Noise N m n k → ℚ)
    (U : Finset ℕ) : Prop :=
  ∀ q : M.Prog, ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, w x.1 x.2 ∈ M.outputs q) →
    (1 - γ₂) * (16 * S.card : ℝ) ≤ M.cost q

/-- **An open-pipe program that no closed pipe is** (the tensor-core-junk shape): for every `k ≥ 1024` with
`16 ∣ k`, `m, n ≥ 1` and a unit `u₀ ∈ U`, some `R` passes `Admissible`, is checkpoint independent and has MinRank ≥
16, and some `OpenQuadPipe 16 8` program on the noise coordinates outputs the non-final checked words. One of its
FP32 steps is an FFMA `a·b + w` of two free values whose addend `w` is the output of a tensor-core instruction (`u₀`'s
word at position 0, priced 16). It is not a `ClosedPipe 8` program, and it costs `16·|U|·m·n·(3k/16 − 1) + 8`. -/
def OpenPipeWitness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (u₀ : ℕ), U ⊆ range N → u₀ ∈ U → 0 < m → 0 < n →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))) (a b : Noise N m n k → ℚ), Admissible k 16 R ∧
      CheckpointIndependent k 16 R ∧ MinRankGe k R 16 ∧ OpenQuadPipe 16 8 P ∧ ¬ ClosedPipe 8 P ∧
      Outputs k m n N R A B U P (coords N m n k) ∧ a ∈ V0 N m n k ∧ b ∈ V0 N m n k ∧
      Instr.priced 16 (word k m n N R A B u₀ 0) ∈ P ∧ Instr.priced 8 (a * b + word k m n N R A B u₀ 0) ∈ P ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 8

end Pouw.Dimension.OpenPipe
