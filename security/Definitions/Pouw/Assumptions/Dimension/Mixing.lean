module

public import Definitions.Pouw.Assumptions.Dimension.Assumptions

/-! The mixing lemma: Theorem D in AlgM_4090, AlgM_16's programs under W1's price table with scalar-pipe products at `c_s`. -/

@[expose] public section

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

variable {G : Type}

def bigOuts : List (Instr G) → List (G → ℚ)
  | [] => []
  | .free _ _ :: P => bigOuts P
  | .priced p f :: P => if 16 ≤ p then f :: bigOuts P else bigOuts P

def prodOuts : List (Instr G) → List (G → ℚ)
  | [] => []
  | .free _ _ :: P => prodOuts P
  | .priced p f :: P => if 16 ≤ p then prodOuts P else f :: prodOuts P

/-- **AlgM_4090's price rule** at scalar-product price `cs`, one register written per priced instruction as in `H16`. -/
def H4090 {N m n k : ℕ} (cs : ℕ) (P : List (Instr (Noise N m n k))) : Prop :=
  ∀ p f, Instr.priced p f ∈ P → 16 ≤ p ∨ (cs ≤ p ∧ ProdFree f)

/-- **`L(V)`, multiplicative complexity over `V₀`**; A1 reads `L(V) ≥ c·(1 − γ₁)·dim V`. -/
noncomputable def Lmul {N m n k : ℕ} (V : Submodule ℚ (Noise N m n k → ℚ)) : ℕ :=
  sInf {r | ∃ p : Fin r → Noise N m n k → ℚ, (∀ i, ProdFree (p i)) ∧
    V ≤ Submodule.span ℚ (Set.range p) ⊔ V0 N m n k}

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

/-- **Theorem D in AlgM_4090**: the mixing lemma with A1. -/
def AlgBound4090 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ) (c γ₁ : ℝ),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H4090 cs P →
    Outputs k m n N R A B U P L → A1_multiplicative c γ₁ k R → 16 ≤ c * cs → 0 ≤ γ₁ → γ₁ ≤ 1 →
    16 * (1 - γ₁) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P

/-- **H4090's witness, with a scalar product**: the product branch of `H4090` is satisfiable jointly with the rest. -/
def Alg4090Witness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      H4090 8 P ∧ Outputs k m n N R A B U P (coords N m n k) ∧ (prodOuts P).length = 1 ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 8

/-- **`A1_W1`'s witness**: it holds once `γ₁ ≥ 1/2`, by Theorem D's own count; the assumption's content is `γ₁` near 0. -/
def A1W1Witness : Prop := ∀ (k : ℕ) (R : Mat) (γ₁ : ℝ), 1 / 2 ≤ γ₁ → A1_W1 γ₁ k R

/-- **MinRank ≥ 2's witness, jointly with `Admissible` and checkpoint independence**: `A1_W1`'s premises on `R` hold
together at every served `k`. -/
def MinRank2Witness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∃ R, Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧ MinRank2 k R

end Pouw.Dimension
