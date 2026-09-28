import Definitions.Pouw.Dimension.Defs
import Models.Pouw.Assumptions.Dimension.Assumptions

/-!
# The mixing lemma: Theorem D in AlgM_4090, and `TT_NCP_of_A1_A2` (trusted statements)

`docs/pouw/milder-assumptions.md` §2.4 and §2.7. AlgM_4090 runs the programs of AlgM_16 (`Pouw.Dimension.Instr`, on
inputs in `V₀`, with A = g(F₁)) under W1's full price table: a priced instruction costs at least 16, except a
scalar-pipe product (FFMA/FMUL), which costs at least `c_s` (8 in W1; `H4090`).

A scalar-pipe product is an exact product of two slot-linear forms over the free values: its output is `ℓ·ℓ′` with `ℓ`
and `ℓ′` in `V₀` (`ProdFree`). The model has one level of scalar products: a product with a factor that depends on a
priced or product output is priced at 16 here. In the formal-polynomial model this is no loss, since the degree-2 part
of any product is the product of its factors' linear parts plus earlier degree-2 parts (Strassen). On the grid the
difference is a representation effect (products that alias through `x² = x` on the probes, or high-degree identities),
which A2 absorbs.

`B` is the set of priced instructions at price 16 or more (`bigOuts`), `R` the scalar-pipe products below 16
(`prodOuts`), and `L(V)` the least number of products of two free values whose span with `V₀` contains `V` (`Lmul`).
The mixing lemma is `cost ≥ 16·|B| + c_s·|R|`, `|B| ≥ D − dim W` and `|R| ≥ L(W)`, for `W = T ∩ (span R ⊔ V₀)` and
`D = dim T` (`MixingLemma`). With A1(c, γ₁) and `c·c_s ≥ 16` it gives `cost ≥ 16·(1 − γ₁)·D` (`AlgBound4090`).
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

variable {G : Type}

/-- The outputs of the priced instructions at price 16 or more (the set `B` of §2.4). -/
def bigOuts : List (Instr G) → List (G → ℚ)
  | [] => []
  | .free _ _ :: P => bigOuts P
  | .priced p f :: P => if 16 ≤ p then f :: bigOuts P else bigOuts P

/-- The outputs of the priced instructions below price 16, the scalar-pipe products (the set `R` of §2.4). -/
def prodOuts : List (Instr G) → List (G → ℚ)
  | [] => []
  | .free _ _ :: P => prodOuts P
  | .priced p f :: P => if 16 ≤ p then prodOuts P else f :: prodOuts P

/-- **AlgM_4090's price rule** at scalar-product price `cs` (one register written per priced instruction, as in
`H16`): every priced instruction costs at least 16, or it is a scalar-pipe product, an exact product of two free values,
and costs at least `cs`. -/
def H4090 {N m n k : ℕ} (cs : ℕ) (P : List (Instr (Noise N m n k))) : Prop :=
  ∀ p f, Instr.priced p f ∈ P → 16 ≤ p ∨ (cs ≤ p ∧ ProdFree f)

/-- **`L(V)`, multiplicative complexity over `V₀`**: the least number of products of two free values whose span,
together with `V₀`, contains `V`. A1 reads `L(V) ≥ c·(1 − γ₁)·dim V`. -/
noncomputable def Lmul {N m n k : ℕ} (V : Submodule ℚ (Noise N m n k → ℚ)) : ℕ :=
  sInf {r | ∃ p : Fin r → Noise N m n k → ℚ, (∀ i, ProdFree (p i)) ∧
    V ≤ Submodule.span ℚ (Set.range p) ⊔ V0 N m n k}

/-- AlgM_4090 as a machine: its programs are the `H4090 cs` programs on the noise coordinates, at their AlgM cost,
outputting their slots. -/
def alg4090Machine (N m n k cs : ℕ) : Machine (Noise N m n k) where
  Prog := {P : List (Instr (Noise N m n k)) // H4090 cs P}
  cost P := (cost P.1 : ℝ)
  outputs P := {φ | φ ∈ slots P.1 (coords N m n k)}

/-! ## Pinned statements -/

/-- **The mixing lemma** (survey §2.4): under checkpoint independence, for every AlgM_4090 program on inputs in `V₀`
that outputs the non-final checked words of the units in `U` (A = g(F₁)), with `T` their span,
`D = |U|·m·n·(3k/16 − 1) = dim T` and `W = T ∩ (span R ⊔ V₀)`:
`cost ≥ 16·|B| + c_s·|R|`, `|B| ≥ D − dim W` and `|R| ≥ L(W)`. -/
def MixingLemma : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H4090 cs P →
    Outputs k m n N R A B U P L →
    Module.finrank ℚ (targetSpan k m n N R A B U) = U.card * (m * n) * (3 * k / 16 - 1) ∧
    16 * (bigOuts P).length + cs * (prodOuts P).length ≤ cost P ∧
    U.card * (m * n) * (3 * k / 16 - 1) ≤ (bigOuts P).length +
      Module.finrank ℚ ↥(targetSpan k m n N R A B U ⊓
        (Submodule.span ℚ {f | f ∈ prodOuts P} ⊔ V0 N m n k)) ∧
    Lmul (targetSpan k m n N R A B U ⊓ (Submodule.span ℚ {f | f ∈ prodOuts P} ⊔ V0 N m n k)) ≤
      (prodOuts P).length

/-- **Theorem D in AlgM_4090** (the mixing lemma with A1): under checkpoint independence and `A1(c, γ₁)` with
`c·c_s ≥ 16` and `0 ≤ γ₁ ≤ 1`, every `H4090 c_s` program on inputs in `V₀` that outputs the non-final checked words
of the units in `U` costs at least `16·(1 − γ₁)·|U|·m·n·(3k/16 − 1)`. -/
def AlgBound4090 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ) (c γ₁ : ℝ),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H4090 cs P →
    Outputs k m n N R A B U P L → A1_multiplicative c γ₁ k R → 16 ≤ c * cs → 0 ≤ γ₁ → γ₁ ≤ 1 →
    16 * (1 - γ₁) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P

/-- **`TT_NCP_of_A1_A2`** (survey §2.6 and §2.7, whole-grid form), `c` a parameter: if a machine satisfies `A2(γ₂)`
into AlgM_4090 at scalar price `c_s`, `A1(c, γ₁)` holds with `c·c_s ≥ 16`, one of the machine's programs outputs the
non-final checked words of the units in `U` for some `A = g(F₁)`, and `1 − γ₀ ≤ (1 − γ₂)·(1 − γ₁)·(1 − 16/(3k))`,
then TT_NCP's event fails for that program: not `cost/(1 − γ₀) < Σ_{u ∈ U} 3·m·k·n`. -/
def TTNCPOfA1A2 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (M : Machine (Noise N m n k)) (q : M.Prog) (cs : ℕ) (c γ₀ γ₁ γ₂ : ℝ), 16 ∣ k → 0 < k →
    CheckpointIndependent k 16 R → U ⊆ range N →
    A2_algebraicSufficiency γ₂ M (H4090 cs) (wordSet (word k m n N R A B) U) →
    A1_multiplicative c γ₁ k R → 16 ≤ c * cs →
    (∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ M.outputs q) →
    0 ≤ γ₁ → γ₁ ≤ 1 → γ₂ < 1 → γ₀ < 1 → 1 - γ₀ ≤ (1 - γ₂) * (1 - γ₁) * (1 - 16 / (3 * k : ℝ)) →
    ¬ (M.cost q / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- **`TT_NCP_of_A1_A2` at W1's constants** (`c = 2`, `c_s = 8`), with A1 as `A1_W1` under MinRank ≥ 2. `A1_W1` is
stronger, probably false asymptotically; not used by the recommended chain (`TTNCPOfA1costA2`). -/
def TTNCPOfA1A2W1 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (M : Machine (Noise N m n k)) (q : M.Prog) (γ₀ γ₁ γ₂ : ℝ), 16 ∣ k → 0 < k →
    CheckpointIndependent k 16 R → MinRank2 k R → U ⊆ range N →
    A2_algebraicSufficiency γ₂ M (H4090 8) (wordSet (word k m n N R A B) U) → A1_W1 γ₁ k R → (∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ M.outputs q) →
    0 ≤ γ₁ → γ₁ ≤ 1 → γ₂ < 1 → γ₀ < 1 → 1 - γ₀ ≤ (1 - γ₂) * (1 - γ₁) * (1 - 16 / (3 * k : ℝ)) →
    ¬ (M.cost q / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- **H4090's witness, with a scalar product**: for every `k ≥ 1024` with `16 ∣ k`, every shape, `A = g(F₁)`, `B` and
unit set, some `R` passes `Admissible` and is checkpoint independent, and some `H4090 8` program on the noise
coordinates outputs the non-final checked words, has exactly one scalar-pipe product (`1·1`, at 8), and costs
`16·|U|·m·n·(3k/16 − 1) + 8`. It shows the product branch of `H4090` is satisfiable jointly with the rest. -/
def Alg4090Witness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      H4090 8 P ∧ Outputs k m n N R A B U P (coords N m n k) ∧ (prodOuts P).length = 1 ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 8

/-- **A2's witness into AlgM_4090** (a trivial model): AlgM_4090, as a machine, satisfies `A2(0)` into `H4090 cs`,
for every target set. -/
def A2Witness4090 : Prop :=
  ∀ (N m n k cs : ℕ) (W : Set (Noise N m n k → ℚ)), A2_algebraicSufficiency 0 (alg4090Machine N m n k cs) (H4090 cs) W

/-- **`A1_W1`'s witness**: `A1_W1 γ₁ k R` holds for every `k` and `R` once `γ₁ ≥ 1/2`, by Theorem D's own count
(`A1Witness`, `2·(1 − γ₁) ≤ 1`). The assumption's content is `γ₁` near 0. `A1_W1` is stronger, probably false
asymptotically; not used by the recommended chain. -/
def A1W1Witness : Prop := ∀ (k : ℕ) (R : Mat) (γ₁ : ℝ), 1 / 2 ≤ γ₁ → A1_W1 γ₁ k R

/-- **MinRank ≥ 2's witness, jointly with `Admissible` and checkpoint independence** (`minrank.md` §3, the shift by
24): for every `k ≥ 1024` with `16 ∣ k`, some `R` passes `Admissible`, is checkpoint independent and has MinRank ≥ 2,
so `A1_W1`'s premises on `R` hold together at every production `k`. -/
def MinRank2Witness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∃ R, Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧ MinRank2 k R

end Pouw.Dimension
