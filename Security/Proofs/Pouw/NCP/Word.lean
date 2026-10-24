import Definitions.Pouw.NCP.Word

/-! The proof-only part of `Definitions.Pouw.NCP.Word`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.NCP.Word

open Finset

variable {Ω : Type}

/-- **FinalOnly** (named hypothesis): unit `u` has at most `f u` identity positions. -/
def FinalOnly (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → ℤ) (A : Adversary) (f : ℕ → ℕ) : Prop :=
  ∀ u, identCount n X A u ≤ f u

/-- Values uniform and independent on `{0, 1}`: position `p < M` reads bit `p` of `ω`. -/
def bits {M : ℕ} (ω : Fin M → Bool) (p : ℕ) : ℤ := if h : p < M then ((ω ⟨p, h⟩).toNat : ℤ) else 0

/-- The all-constant adversary for values `x` that do not depend on the noise. -/
def constAdv (x : ℕ → ℕ → ℤ) : Adversary := ⟨fun u p => .const (x u p)⟩

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
