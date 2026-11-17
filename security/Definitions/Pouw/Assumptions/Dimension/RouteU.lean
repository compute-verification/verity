module

public import Definitions.Pouw.Assumptions.Dimension.Mixing
public import Definitions.Pouw.Dimension.Causal
public import Definitions.Pouw.Assumptions.NCP.RouteUWord

/-! Theorem D and its companions on route U's checked words, which differ from the unbiased words by an element of `V₀` at every bias `β`. -/

@[expose] public section

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

def wordU (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ) :
    Noise N m n k → ℚ :=
  fun ω => (checkedUPos β k m n N R (A ω.2) B ω u p : ℚ)

def wordCU (β : ℤ) (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ) :
    Noise N m n k → ℚ :=
  fun ω => (checkedUPos β k m n N R (A ω) B ω u p : ℚ)

def OutputsU (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (P : List (Instr (Noise N m n k))) (L : List (Noise N m n k → ℚ)) : Prop :=
  ∀ u ∈ U, ∀ p < nonFinal k m n, wordU β k m n N R A B u p ∈ slots P L

def targetSpanU (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) :
    Submodule ℚ (Noise N m n k → ℚ) :=
  Submodule.span ℚ (Set.range fun x : ↥(U ×ˢ range (nonFinal k m n)) => wordU β k m n N R A B x.1.1 x.1.2)

/-- **The shift is free**: at every position, route U's word minus the unbiased word lies in `V₀`. -/
def WordUSubV0 : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ),
    wordU β k m n N R A B u p - word k m n N R A B u p ∈ V0 N m n k

def WordCUSubV0 : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ),
    wordCU β k m n N R A B u p - wordC k m n N R A B u p ∈ V0 N m n k

def CombUV0Iff : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (c : ℕ → ℕ → ℚ),
    (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • wordU β k m n N R A B u p) ∈ V0 N m n k ↔
      (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • word k m n N R A B u p) ∈ V0 N m n k

def IndepModShift : Prop :=
  ∀ (G ι : Type) [Fintype ι] (V : Submodule ℚ (G → ℚ)) (t t' : ι → G → ℚ), (∀ i, t' i - t i ∈ V) →
    (∀ c : ι → ℚ, ∑ i, c i • t i ∈ V → ∀ i, c i = 0) → ∀ c : ι → ℚ, ∑ i, c i • t' i ∈ V → ∀ i, c i = 0

def IndepModU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (c : ℕ → ℕ → ℚ), CheckpointIndependent k 16 R → U ⊆ range N →
    (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • wordU β k m n N R A B u p) ∈ V0 N m n k →
    ∀ u ∈ U, ∀ p < nonFinal k m n, c u p = 0

def SecondDifferenceU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (c : ℕ → ℕ → ℚ), U ⊆ range N →
    (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • wordU β k m n N R A B u p) ∈ V0 N m n k →
    ∀ u ∈ U, ∀ e < m * n, ∀ a < k, ∀ b < k,
      ∑ t ∈ range (3 * k / 16 - 1), c u (t * (m * n) + e) * (ckMat k R (16 * (t + 1)) a b : ℚ) = 0

def PricedCountU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → OutputsU β k m n N R A B U P L →
    U.card * (m * n) * (3 * k / 16 - 1) ≤ nPriced P

def AlgBoundU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P → OutputsU β k m n N R A B U P L →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

def AlgSharpU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) (g₀ : ℝ),
    16 ∣ k → 0 < k → U ⊆ range N → U.Nonempty → 0 < m * n → g₀ < 16 / (3 * k : ℝ) →
    ∃ P : List (Instr (Noise N m n k)), H16 P ∧ OutputsU β k m n N R A B U P (coords N m n k) ∧
      (cost P : ℝ) / (1 - g₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ)

def TTNCPAlgU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (g₀ : ℝ),
    16 ∣ k → 0 < k → CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    OutputsU β k m n N R A B U P L → 16 / (3 * k : ℝ) ≤ g₀ → g₀ < 1 →
    ¬ ((cost P : ℝ) / (1 - g₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

def AlgBoundCausalU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → Causal A → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    (∀ u ∈ U, ∀ p < nonFinal k m n, wordCU β k m n N R A B u p ∈ slots P L) →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

def MixingLemmaU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H4090 cs P →
    OutputsU β k m n N R A B U P L →
    Module.finrank ℚ (targetSpanU β k m n N R A B U) = U.card * (m * n) * (3 * k / 16 - 1) ∧
    16 * (bigOuts P).length + cs * (prodOuts P).length ≤ cost P ∧
    U.card * (m * n) * (3 * k / 16 - 1) ≤ (bigOuts P).length +
      Module.finrank ℚ ↥(targetSpanU β k m n N R A B U ⊓
        (Submodule.span ℚ {f | f ∈ prodOuts P} ⊔ V0 N m n k)) ∧
    Lmul (targetSpanU β k m n N R A B U ⊓ (Submodule.span ℚ {f | f ∈ prodOuts P} ⊔ V0 N m n k)) ≤
      (prodOuts P).length

/-- **Theorem D for route U's protocol words**, mod `2^32` (`Params.YU`'s by `RouteU.YUChecked`), where the wrap is
the identity (`RouteU.Int7Int32U`). -/
def AlgBoundWrappedU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))), 127 ≤ β → β ≤ 129 →
    CheckpointIndependent k 16 R → SignedPerm k R → k ≤ 2 ^ 15 → U ⊆ range N →
    (∀ F, ∀ u ∈ U, ∀ i, ∀ l < k, InInt7 (A F u i l)) → (∀ u ∈ U, ∀ l < k, ∀ j, InInt7 (B u l j)) →
    (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    (∀ u ∈ U, ∀ p < nonFinal k m n,
      (fun ω => (wrap32 (checkedUPos β k m n N R (A ω.2) B ω u p) : ℚ)) ∈ slots P L) →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- `AlgBoundWrappedU` for a permutation-matrix `R` (the served `P`), where the wrap is the identity by
`RouteU.Int7Int32UUnsigned`. -/
def AlgBoundWrappedUUnsigned : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))), 127 ≤ β → β ≤ 129 →
    CheckpointIndependent k 16 R → Unsigned k R → k ≤ 2 ^ 16 → U ⊆ range N →
    (∀ F, ∀ u ∈ U, ∀ i, ∀ l < k, InInt7 (A F u i l)) → (∀ u ∈ U, ∀ l < k, ∀ j, InInt7 (B u l j)) →
    (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    (∀ u ∈ U, ∀ p < nonFinal k m n,
      (fun ω => (wrap32 (checkedUPos β k m n N R (A ω.2) B ω u p) : ℚ)) ∈ slots P L) →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **A1 on route U's span**: A1 is stated on the unbiased span and unchanged, because both spans are complements of `V₀`
in the same space and `L(V)` reads `V` only modulo `V₀`. -/
def A1U : Prop :=
  ∀ (β : ℤ) (c γ₁ : ℝ) (k : ℕ) (R : Mat), CheckpointIndependent k 16 R → A1_multiplicative c γ₁ k R →
    ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ), U ⊆ range N →
    ∀ V : Submodule ℚ (Noise N m n k → ℚ), V ≤ targetSpanU β k m n N R A B U →
    ∀ (r : ℕ) (p : Fin r → Noise N m n k → ℚ), (∀ i, ProdFree (p i)) →
      V ≤ Submodule.span ℚ (Set.range p) ⊔ V0 N m n k →
      c * (1 - γ₁) * (Module.finrank ℚ V : ℝ) ≤ r

def AlgBound4090U : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ) (c γ₁ : ℝ),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H4090 cs P →
    OutputsU β k m n N R A B U P L → A1_multiplicative c γ₁ k R → 16 ≤ c * cs → 0 ≤ γ₁ → γ₁ ≤ 1 →
    16 * (1 - γ₁) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P

/-- **Theorem D on route U**: its hypotheses hold jointly and its bound is attained. -/
def AlgWitnessU : Prop :=
  ∀ (β : ℤ) (k : ℕ), 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ), U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      (∀ f ∈ coords N m n k, f ∈ V0 N m n k) ∧ H16 P ∧ OutputsU β k m n N R A B U P (coords N m n k) ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1))

/-- **AlgM_4090's hypotheses on route U hold jointly.** -/
def Alg4090WitnessU : Prop :=
  ∀ (β : ℤ) (k : ℕ), 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ), U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      H4090 8 P ∧ OutputsU β k m n N R A B U P (coords N m n k) ∧ (prodOuts P).length = 1 ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 8

/-- **`AlgBoundWrappedU`'s hypotheses hold jointly, and its bound is attained**, with a permutation-matrix (so
signed-permutation) `R`. -/
def AlgWrappedWitnessU : Prop :=
  ∀ (β : ℤ) (k : ℕ), 127 ≤ β → β ≤ 129 → 1024 ≤ k → 16 ∣ k → k ≤ 2 ^ 16 →
    ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ), U ⊆ range N →
    (∀ F, ∀ u ∈ U, ∀ i, ∀ l < k, InInt7 (A F u i l)) → (∀ u ∈ U, ∀ l < k, ∀ j, InInt7 (B u l j)) →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ Unsigned k R ∧ SignedPerm k R ∧
      CheckpointIndependent k 16 R ∧ (∀ f ∈ coords N m n k, f ∈ V0 N m n k) ∧ H16 P ∧
      (∀ u ∈ U, ∀ p < nonFinal k m n,
        (fun ω => (wrap32 (checkedUPos β k m n N R (A ω.2) B ω u p) : ℚ)) ∈ slots P (coords N m n k)) ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1))

end Pouw.Dimension
