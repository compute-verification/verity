import Models.Pouw.Assumptions.Dimension.PipeAssumptions

/-!
# `A1_cost` in the closed scalar pipe (trusted statements)

The coordinator's model decision of 06:01Z (`barrier.md`, "`A1_cost`"; the red team's P8-1 and P8-2 in
`red-team.md`, Phase 8). AlgM_4090 with a **closed scalar pipe**:

* **No free instructions.** W1 prices every addition, so a program has no free linear combinations.
* **Priced instructions** cost at least 16 and may output any function of the noise (tensor-core words, INT32
  results, priced loads), or they are **FP32 steps** (FFMA, FMUL, FADD) and cost at least `c_s` (8 in W1).
* **An FP32 step** outputs `a·b + c`, each operand a free value (an element of `V₀`: constants, affine functions of
  the noise and every function of `F₁`) or a scalar multiple of an earlier FP32 output (`PipeStep`). It never reads the
  output of an instruction priced at 16 or more: such a product is priced at 16, as before. FMUL is `c = 0`; FADD
  `α·x + β·y` is `a = α` constant, `b = x`, `c = β·y`.
* **Quadratic** (`QuadStep`, `a1-cost.md` §3.1 with P8-2): each product multiplies two free values, or one factor is
  a constant. Every known algorithm for the checked words is quadratic.

`A1_cost(c, γ)` (additive form, `c` a parameter, under checkpoint independence and MinRank ≥ 5): a closed FP32
pipe whose outputs include `|S|` non-final checked words has at least `c·|S| − c·γ·D` steps, with
`D = |U|·m·n·(3k/16 − 1)`. At `c = 2` and `c_s = 8` that is `8·#steps ≥ 16·|S| − 16·γ·D`. `a1-cost`'s Theorem 1
(proved on paper, granted by the red team with P8-1 and P8-2) is `A1_cost_quad 2 0`; the open part is
`A1_nonquad`, the pipes that are not quadratic.

The model's definitions are in `Pouw.Dimension.PipeModel`; `A1_cost`, `A1_cost_quad`, `A1_nonquad` and the lifting's
`LiftMinRank` and `LiftMinRankL5e` are in the assumptions module `Pouw.Dimension.PipeAssumptions`.
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

variable {N m n k : ℕ}

/-- The cyclic shift by 24 (`minrank.md`): column `j < k` has its `1` in row `j + 24 mod k`. -/
def shift24 (k : ℕ) : Mat := fun i j => if j < k ∧ i = (if j + 24 < k then j + 24 else j + 24 - k) then 1 else 0

/-- The closed-pipe AlgM_4090 as a machine: its programs are the `ClosedPipe cs` programs on the noise coordinates, at
their AlgM cost, outputting their slots. -/
def pipeMachine (N m n k cs : ℕ) : Machine (Noise N m n k) where
  Prog := {P : List (Instr (Noise N m n k)) // ClosedPipe cs P}
  cost P := (cost P.1 : ℝ)
  outputs P := {φ | φ ∈ slots P.1 (coords N m n k)}

/-- The quadratic closed-pipe AlgM_4090 as a machine: its programs are the `QuadPipe cs` programs on the noise
coordinates, at their AlgM cost, outputting their slots. -/
def quadMachine (N m n k cs : ℕ) : Machine (Noise N m n k) where
  Prog := {P : List (Instr (Noise N m n k)) // QuadPipe cs P}
  cost P := (cost P.1 : ℝ)
  outputs P := {φ | φ ∈ slots P.1 (coords N m n k)}

/-! ## Pinned statements -/

/-- **`A1_cost` splits**: its quadratic and non-quadratic parts give it. -/
def A1CostOfSplit : Prop :=
  ∀ (c γ : ℝ) (k : ℕ) (R : Mat), A1_cost_quad c γ k R → A1_nonquad c γ k R → A1_cost c γ k R

/-- **Theorem D with a closed scalar pipe**: under checkpoint independence, MinRank ≥ 5 and `A1_cost(c, γ)` with
`c·c_s ≥ 16`, every `ClosedPipe c_s` program on inputs in `V₀` that outputs the non-final checked words of the units in
`U` costs at least `16·(1 − γ)·D`. -/
def AlgBoundPipe : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ) (c γ : ℝ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    ClosedPipe cs P → Outputs k m n N R A B U P L → A1_cost c γ k R → 16 ≤ c * cs →
    16 * (1 - γ) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P

/-- **Theorem D with a quadratic closed pipe**: the same with `QuadPipe c_s` and `A1_cost_quad(c, γ)`. With
Theorem 1 (`A1_cost_quad 2 0`) and `c_s = 8` the bound is `16·D`, Theorem D's own. -/
def AlgBoundQuad : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ) (c γ : ℝ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    QuadPipe cs P → Outputs k m n N R A B U P L → A1_cost_quad c γ k R → 16 ≤ c * cs →
    16 * (1 - γ) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P

/-- **`TT_NCP_of_A1cost_A2`** at W1's constants (`c = 2`, `c_s = 8`): if a machine satisfies `A2(γ₂)` into the
closed pipe, `A1_cost(2, γ₁)` holds, `R` is checkpoint independent with MinRank ≥ 5, one of the machine's programs
outputs the non-final checked words of the units in `U` for some `A = g(F₁)`, and
`1 − γ₀ ≤ (1 − γ₂)·(1 − γ₁)·(1 − 16/(3k))`, then TT_NCP's event fails for that program. -/
def TTNCPOfA1costA2 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (M : Machine (Noise N m n k)) (q : M.Prog) (γ₀ γ₁ γ₂ : ℝ), 16 ∣ k → 0 < k →
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    A2_algebraicSufficiency γ₂ M (ClosedPipe 8) (wordSet (word k m n N R A B) U) → A1_cost 2 γ₁ k R →
    (∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ M.outputs q) →
    γ₂ < 1 → γ₀ < 1 → 1 - γ₀ ≤ (1 - γ₂) * (1 - γ₁) * (1 - 16 / (3 * k : ℝ)) →
    ¬ (M.cost q / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- **`TT_NCP_of_A1quad_A2`**: the same with `A2(γ₂)` into the quadratic closed pipe and `A1_cost_quad(2, γ₁)`
(Theorem 1 gives it at `γ₁ = 0`). -/
def TTNCPOfQuadA2 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (M : Machine (Noise N m n k)) (q : M.Prog) (γ₀ γ₁ γ₂ : ℝ), 16 ∣ k → 0 < k →
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    A2_algebraicSufficiency γ₂ M (QuadPipe 8) (wordSet (word k m n N R A B) U) → A1_cost_quad 2 γ₁ k R →
    (∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ M.outputs q) →
    γ₂ < 1 → γ₀ < 1 → 1 - γ₀ ≤ (1 - γ₂) * (1 - γ₁) * (1 - 16 / (3 * k : ℝ)) →
    ¬ (M.cost q / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- **The closed pipe's witness, with an FP32 step**: for every `k ≥ 1024` with `16 ∣ k`, every shape, `A = g(F₁)`,
`B` and unit set, some `R` passes `Admissible`, is checkpoint independent and has MinRank ≥ 16, and some quadratic
closed-pipe program at `c_s = 8` on the noise coordinates outputs the non-final checked words, has one FP32 step
(`1·1 + 0`, at 8) and costs `16·|U|·m·n·(3k/16 − 1) + 8`. -/
def PipeWitness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      MinRankGe k R 16 ∧ QuadPipe 8 P ∧ Outputs k m n N R A B U P (coords N m n k) ∧ (prodOuts P).length = 1 ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 8

/-- **The closed pipe's witness, with a checked word in the pipe** (red team P9B-3): as `PipeWitness`, for `m, n ≥ 1`
and a unit `u₀ ∈ U`, the program's FP32 pipe is the 16-FFMA chain from `c₀ = 0`, `c_{l+1} = X[0, l]·Y[l, 0] + c_l`
with `X`, `Y` unit `u₀`'s stacked operands. It has 16 quadratic steps, it outputs `u₀`'s non-final checked word at
position 0 (depth 16), which is not a free value, and the program costs `16·|U|·m·n·(3k/16 − 1) + 16·8`. -/
def PipeWordWitness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (u₀ : ℕ), U ⊆ range N → u₀ ∈ U → 0 < m → 0 < n →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      MinRankGe k R 16 ∧ QuadPipe 8 P ∧ Outputs k m n N R A B U P (coords N m n k) ∧ (prodOuts P).length = 16 ∧
      word k m n N R A B u₀ 0 ∈ prodOuts P ∧ word k m n N R A B u₀ 0 ∉ V0 N m n k ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 16 * 8

/-- **A2's witness into the closed pipe** (a trivial model): the closed-pipe AlgM_4090, as a machine, satisfies
`A2(0)` into `ClosedPipe cs`, for every target set. -/
def A2WitnessPipe : Prop :=
  ∀ (N m n k cs : ℕ) (W : Set (Noise N m n k → ℚ)), A2_algebraicSufficiency 0 (pipeMachine N m n k cs) (ClosedPipe cs) W

/-- **A2's witness into the quadratic closed pipe** (a trivial model): the quadratic closed-pipe AlgM_4090, as a
machine, satisfies `A2(0)` into `QuadPipe cs`, for every target set. -/
def A2WitnessQuad : Prop :=
  ∀ (N m n k cs : ℕ) (W : Set (Noise N m n k → ℚ)), A2_algebraicSufficiency 0 (quadMachine N m n k cs) (QuadPipe cs) W

/-- **The `A1_cost` family's witness**: `A1_cost`, `A1_cost_quad` and `A1_nonquad` hold at `c = 2`, `γ = 1/2` for
every `k` and `R`, by the count of one step per output word. Their content is `γ` near 0. -/
def A1CostWitness : Prop :=
  ∀ (k : ℕ) (R : Mat), A1_cost 2 (1 / 2) k R ∧ A1_cost_quad 2 (1 / 2) k R ∧ A1_nonquad 2 (1 / 2) k R

/-- **The two chains' hypotheses hold together**: for every `k ≥ 1024` with `16 ∣ k`, every shape, `A = g(F₁)`, `B`
and unit set, one `R` and one program satisfy every hypothesis of `TTNCPOfA1costA2` (machine `pipeMachine`) and of
`TTNCPOfQuadA2` (machine `quadMachine`) at `γ₂ = 0`, `γ₁ = 1/2` and `γ₀ = 3/4`. -/
def PipeChainConsistent : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N → ∃ R : Mat, CheckpointIndependent k 16 R ∧ MinRankGe k R 5 ∧
      A2_algebraicSufficiency 0 (pipeMachine N m n k 8) (ClosedPipe 8) (wordSet (word k m n N R A B) U) ∧
      A2_algebraicSufficiency 0 (quadMachine N m n k 8) (QuadPipe 8) (wordSet (word k m n N R A B) U) ∧
      A1_cost 2 (1 / 2) k R ∧ A1_cost_quad 2 (1 / 2) k R ∧
      (∃ q : (pipeMachine N m n k 8).Prog,
        ∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ (pipeMachine N m n k 8).outputs q) ∧
      (∃ q : (quadMachine N m n k 8).Prog,
        ∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ (quadMachine N m n k 8).outputs q) ∧
      1 - (3 / 4 : ℝ) ≤ (1 - 0) * (1 - 1 / 2) * (1 - 16 / (3 * k : ℝ))

/-- **MinRank ≥ 16 for the shift by 24** (`minrank.md` §3), at every `k ≥ 1024` with `16 ∣ k`: every nonzero
combination has a diagonal 16 × 16 minor, on the step block of a nonzero `α`, on the block moved by 24 of a nonzero
`β`, or on 8 columns at each of two change points of `γ` around the cycle of steps. 16 is the maximum (one step
increment has rank 16). -/
def MinRankGeShift16 : Prop := ∀ k, 1024 ≤ k → 16 ∣ k → MinRankGe k (shift24 k) 16

/-- **MinRank is monotone in `ρ`**, and **MinRank ≥ 2 in the minor form gives `MinRank2`** (P8-6). -/
def MinRankGeFacts : Prop :=
  (∀ (k : ℕ) (R : Mat) (ρ ρ' : ℕ), ρ' ≤ ρ → MinRankGe k R ρ → MinRankGe k R ρ') ∧
  (∀ (k : ℕ) (R : Mat), MinRankGe k R 2 → MinRank2 k R)

/-- **`Lift_MinRank`'s witness** (trivial): it holds at `ε = 1` for every `ρ`, `k` and `R`. The content is `ε` small. -/
def LiftMinRankWitness : Prop := ∀ (ρ k : ℕ) (R : Mat), LiftMinRank ρ (fun _ => 1) k R

end Pouw.Dimension
