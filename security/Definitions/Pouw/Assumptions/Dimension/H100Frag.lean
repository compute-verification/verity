module

public import Definitions.Pouw.Assumptions.Dimension.H100Dense

/-! The k8 lemma with computed fragments, `AccFragIn`, and why it needs a lane bound on the multiplicands (draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.H100

variable {G : Type}

def DetBy (S : Set G) (ws : List (G → ℚ)) (d : G → ℚ) : Prop :=
  ∀ g ∈ S, ∀ g' ∈ S, (∀ w ∈ ws, w g = w g') → d g = d g'

def Op100.FragOK (a K : ℕ) (β : ℚ) (S : Set G) (F W : Set (G → ℚ)) (Rs : List (G → ℚ)) : Op100 G → Prop
  | .mma outs p => p < a * outs.length →
      ∀ f ∈ outs, ∃ C : G → ℚ, (C ∈ Rs ∨ C ∈ F) ∧ ∃ ws : List (G → ℚ), ws.length ≤ K ∧
        (∀ w ∈ ws, (w ∈ Rs ∨ w ∈ F) ∧ w ∈ W) ∧
        ∃ d₀ : G → ℚ, DetBy S ws d₀ ∧ ∀ g ∈ S, |f g - (C g + d₀ g)| ≤ β
  | _ => True

def AccFragIn (a K : ℕ) (β : ℚ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G) (F W : Set (G → ℚ)) :
    List (Op100 G) → List (G → ℚ) → Prop
  | [], _ => True
  | o :: P, Rs => o.FragOK a K β S F W Rs ∧ AccFragIn a K β fpSem S F W P (Rs ++ o.outs fpSem Rs)

/-! ## Guarantees (draft semantics, pending review) -/

/-- **Computed fragments reduce to lane increments** (draft semantics, pending review). -/
def AccInOfFrag : Prop :=
  ∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ) (F W : Set (G → ℚ)) (a K q : ℕ) (β : ℚ)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (P : List (Op100 G)) (L : List (G → ℚ)),
    (∀ w ∈ W, ∃ J : Finset Λ, J.card ≤ q ∧ DependsOn S p J w) →
    AccFragIn a K β fpSem S F W P L → AccIn a fpSem S F (LaneFns (K * q) β S p) P L

/-- **Dense words cost `a` with computed fragments** (draft semantics, pending review): `DenseWritesLegacy` with
`AccFragIn` in place of `AccIn`. -/
def DenseWritesFrag : Prop :=
  ∀ (a c : ℕ) (pr : Prices100), FloorTc a c pr → c ≤ a → a ≤ 2 * c →
    ∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ) (F W : Set (G → ℚ)) (K q : ℕ) (β : ℚ),
    (∀ x : ℚ, (fun _ : G => x) ∈ F) → 0 ≤ β → (∀ w ∈ W, ∃ J : Finset Λ, J.card ≤ q ∧ DependsOn S p J w) →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T₁ T₂ : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → AccFragIn a K β fpSem S F W P L →
      DenseOn S F (LaneFns (K * q) β S p) T₁ → SeparatedOn S (LaneFns (K * q) β S p) T₁ → DistinctOn S F T₂ →
      (∀ t ∈ T₁, ∀ t' ∈ T₂, ApartOn S (LaneFns (K * q) β S p) t t') →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      a * T₁.card + c * T₂.card ≤ cost100 pr P

/-- **A shared fragment amortizes its price** (draft semantics, pending review): without a lane bound on the
multiplicands, a fragment's price does not save the accounting. -/
def FragShareGain : Prop :=
  ∀ (m n : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ),
    ∃ (P : List (Op100 (Fin (2 * m + 1) → ℚ))) (T : Finset ((Fin (2 * m + 1) → ℚ) → ℚ)),
      T.card = n ∧ (∀ o ∈ P, o.WFw h100Legacy {f | ∃ x : ℚ, f = fun _ => x}) ∧
      AccFragIn 32 1 0 fpSem Set.univ {f | ∃ x : ℚ, f = fun _ => x} Set.univ P [] ∧
      DenseOn Set.univ {f | ∃ x : ℚ, f = fun _ => x}
        (LaneFns m 0 Set.univ fun j (g : Fin (2 * m + 1) → ℚ) => g j) T ∧
      SeparatedOn Set.univ (LaneFns m 0 Set.univ fun j (g : Fin (2 * m + 1) → ℚ) => g j) T ∧
      OutputsOn fpSem Set.univ P [] T ∧ cost100 h100Legacy P = 32 + 31 * n

end Pouw.Dimension.H100
